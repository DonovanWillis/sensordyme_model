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

    /// The device's stable Bonjour identity. This is what survives a move.
    @AppStorage("deviceServiceName") private var serviceName: String = ""
    /// Last known address. A cache, never the source of truth.
    @AppStorage("deviceLocalURL") private var cachedURL: String = ""

    private let discovery = DeviceDiscovery()
    private var monitor: NWPathMonitor?
    private var lastPathID: String?

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
        if !cachedURL.isEmpty, await reachable(cachedURL) {
            state = .connected(url: cachedURL)
            return
        }
        await resolve(force: false)
    }

    /// Re-discover the device and adopt its current address.
    func resolve(force: Bool) async {
        if !force, case .searching = state { return }
        state = .searching

        let found = await discovery.discoverOnce(timeout: 4.0)

        // Prefer the device we were previously linked to. On a network with
        // several SensorDymes, silently adopting a different one would be worse
        // than failing — the user would be watching someone else's camera.
        let match = found.first { $0.id == serviceName } ?? (serviceName.isEmpty ? found.first : nil)

        guard let match else {
            // Nothing answered. Either the device is on another network, or
            // this network blocks multicast, or it needs provisioning. The app
            // cannot tell those apart from here, so it says the actionable one.
            state = found.isEmpty ? .needsProvisioning : .notFound
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
            return (response as? HTTPURLResponse).map { (200..<500).contains($0.statusCode) } ?? false
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
