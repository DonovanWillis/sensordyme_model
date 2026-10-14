import Foundation
import Network
import SwiftUI

/// Keeps a working address for the device, without anyone ever typing one.
///
/// The device already advertises itself over Bonjour and the app already knows
/// how to browse for it. What was missing is that the app *resolved once and
/// remembered the answer* — storing `http://192.168.1.20:8899`, which is true
/// on exactly one network. Move either machine and that address points at
/// nothing, while discovery would have found the device immediately if anyone
/// had asked it to.
///
/// So the durable thing stored here is the Bonjour **service name**, which is
/// the device's identity and survives every address change. The IP is treated
/// as a cache: used while it works, re-resolved the moment it doesn't.
///
/// Deliberately not built: a cloud relay. Live video between a phone and a
/// camera should not travel through a server — it costs money, adds latency,
/// and sends frames somewhere they currently never go. When both are on one
/// network this is strictly better; when they are not, the honest answer is to
/// say so, which `state` does.
@MainActor
final class DeviceLink: ObservableObject {
    enum State: Equatable {
        case idle
        case searching
        case connected(url: String)
        case notFound
        case needsProvisioning

        var isConnected: Bool { if case .connected = self { return true }; return false }

        /// Non-technical, because the person reading it did not ask for a
        /// networking lesson — they asked whether the camera is working.
        var summary: String {
            switch self {
            case .idle: "Not connected"
            case .searching: "Looking for your device…"
            case .connected: "Connected"
            case .notFound: "Can't find your device on this network"
            case .needsProvisioning: "Your device needs Wi-Fi setup"
            }
        }
    }

    @Published private(set) var state: State = .idle
    @Published private(set) var lastResolvedAt: Date?

    /// The device's own stable ID (`SD-XXXXXXXX`), published in its mDNS TXT
    /// record. This is what survives a move; the address never does.
    ///
    /// The storage key still says "service name" because that is what earlier
    /// versions kept here, and changing the key would strand every device
    /// already linked. The value's meaning changed; the migration in
    /// `resolve(force:)` handles the old contents.
    @AppStorage("deviceServiceName") private var serviceName: String = ""
    /// Last known address. A cache, never the source of truth.
    @AppStorage("deviceLocalURL") private var cachedURL: String = ""

    private let discovery = DeviceDiscovery()
    private var monitor: NWPathMonitor?
    private var lastPathID: String?
    /// `resolve(force:)` has two independent callers — `ensureConnection()`
    /// (from DevicesView's `.task`) and `handlePathChange` (fired whenever the
    /// network itself changes) — and `force: true` deliberately bypasses the
    /// "already searching" guard just below, so two discovery scans can be in
    /// flight together with no ordering guarantee on which Bonjour answer
    /// comes back first. Without this, a stale scan finishing after a fresh
    /// one could silently rebind `state`/`cachedURL` back to a worse or
    /// outdated address. Found and fixed while sweeping the app for the same
    /// race class as DeviceDetailView/DevicesView/ShopView.
    private let reloadGuard = LatestRequestGuard()

    // MARK: Lifecycle

    /// Start watching for network changes. Safe to call repeatedly.
    func start() {
        guard monitor == nil else { return }
        let monitor = NWPathMonitor()
        monitor.pathUpdateHandler = { [weak self] path in
            Task { @MainActor in self?.handlePathChange(path) }
        }
        monitor.start(queue: DispatchQueue(label: "sensordyme.path"))
        self.monitor = monitor
    }

    func stop() {
        monitor?.cancel()
        monitor = nil
    }

    /// A fingerprint of the current network. Not an identifier of *which*
    /// network — iOS does not hand that out without location permission, and
    /// asking for location to detect a Wi-Fi change would be a poor trade.
    /// Knowing that it changed is enough to justify re-resolving.
    private func fingerprint(_ path: NWPath) -> String {
        let interfaces = path.availableInterfaces.map(\.name).sorted().joined(separator: ",")
        return "\(path.status)-\(interfaces)"
    }

    private func handlePathChange(_ path: NWPath) {
        let current = fingerprint(path)
        defer { lastPathID = current }
        guard path.status == .satisfied else {
            state = .idle
            return
        }
        // First observation is not a change; re-resolving on launch is handled
        // by `ensureConnection()` so this does not fire twice.
        guard let previous = lastPathID, previous != current else { return }
        Task { await resolve(force: true) }
    }

    // MARK: Resolution

    /// Make sure there is a usable address, doing the least work that achieves it.
    func ensureConnection() async {
        if !cachedURL.isEmpty {
            let token = reloadGuard.begin()
            let ok = await reachable(cachedURL)
            guard reloadGuard.isLatest(token) else { return }  // a resolve() has since started/finished
            if ok {
                state = .connected(url: cachedURL)
                return
            }
        }
        await resolve(force: false)
    }

    /// Re-discover the device and adopt its current address.
    func resolve(force: Bool) async {
        if !force, case .searching = state { return }
        let token = reloadGuard.begin()
        state = .searching

        let found = await discovery.discoverOnce(timeout: 4.0)
        guard reloadGuard.isLatest(token) else { return }  // a newer scan has since started

        // Prefer the device we were previously linked to. On a network with
        // several SensorDymes, silently adopting a different one would be worse
        // than failing — the user would be watching someone else's camera.
        var match = found.first { $0.id == serviceName }

        if match == nil, serviceName.isEmpty {
            match = found.first
        }

        // Devices linked before identity moved to the device's own ID stored a
        // Bonjour service name instead. That no longer matches anything, so the
        // app would report a perfectly healthy device as missing and make the
        // user re-pair it. Adopt it — but only when exactly one device answers,
        // so this can never silently rebind someone to a neighbour's camera.
        if match == nil, !serviceName.hasPrefix("SD-"), found.count == 1 {
            match = found.first
        }

        guard let match else {
            // Silence from Bonjour has several causes and the app cannot tell
            // them apart: the device may be powered off, on another network, out
            // of range, or on a network that blocks multicast.
            //
            // An earlier version reported "needs Wi-Fi setup" here, which named
            // the one cause the user can act on — and was wrong most of the
            // time. Worse, it pointed at Bluetooth provisioning, which only
            // works within a few metres of the device, so someone whose camera
            // was simply in another building went looking for it on Bluetooth
            // and found nothing. Listing the real possibilities is less tidy and
            // far more useful.
            state = .notFound
            return
        }

        serviceName = match.id
        cachedURL = match.url
        lastResolvedAt = Date()
        state = .connected(url: match.url)
    }

    /// Forget the device entirely — for unpairing, or a genuinely new unit.
    func forget() {
        serviceName = ""
        cachedURL = ""
        state = .idle
    }

    var currentURL: String? { cachedURL.isEmpty ? nil : cachedURL }

    /// A knowable reason discovery cannot work — a denied Local Network
    /// permission, most usefully. Nil when the only honest answer is
    /// "nothing answered".
    var discoveryProblem: String? { discovery.problem }

    // MARK: Reachability

    /// A short probe against the device's own health endpoint. Deliberately
    /// brief: this runs on app foreground, and a long timeout would make a
    /// moved device feel like a hung app.
    private func reachable(_ urlString: String) async -> Bool {
        guard let url = URL(string: urlString)?.appendingPathComponent("api/health") else {
            return false
        }
        var request = URLRequest(url: url)
        request.timeoutInterval = 2.5
        request.cachePolicy = .reloadIgnoringLocalCacheData
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            // 200, or 401/403 from a device that wants its token: that is a
            // SensorDyme answering. Anything else (a 404 from whatever now holds
            // this DHCP address) is not, and accepting it reported a stranger's
            // box as the camera.
            return (response as? HTTPURLResponse).map { [200, 401, 403].contains($0.statusCode) } ?? false
        } catch {
            return false
        }
    }
}

// MARK: - One-shot discovery

extension DeviceDiscovery {
    /// Browse for a bounded time and return what was found.
    ///
    /// The existing API is a long-lived publisher suited to a settings screen
    /// that shows results as they arrive. Automatic re-linking needs the
    /// opposite shape: ask once, wait briefly, act on the answer.
    @MainActor
    func discoverOnce(timeout: TimeInterval) async -> [Found] {
        start()
        try? await Task.sleep(for: .seconds(timeout))
        let results = found
        stop()
        return results
    }
}
