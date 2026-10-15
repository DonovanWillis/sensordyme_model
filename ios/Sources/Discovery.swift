import Foundation
import Network

/// Finds SensorDyme devices on the local network via Bonjour (_sensordyme._tcp,
/// advertised by avahi on the device) and resolves them to an http base URL —
/// so nobody ever types an IP address into Settings.
///
/// This browses for the DEVICE (a Pi running the camera and its own dashboard),
/// which is a different service from the backend: the backend publishes
/// `_sensordyme-api._tcp` from cloud/mdns.py and is found by `BackendLocator`.
/// Pointing this type at `-api` makes the Mac backend show up in the device list
/// — it has no `id` TXT record, so it lands there under its service name — while
/// the actual camera never appears at all.
@MainActor
final class DeviceDiscovery: ObservableObject {
    /// One device on the network.
    ///
    /// `id` is the device's own stable ID (`SD-XXXXXXXX`) from its mDNS TXT
    /// record, not its service name and not its address. Both of those change —
    /// the name if the hostname is edited, the address every time the device
    /// moves network — and keying on either turns one device into two entries.
    struct Found: Identifiable, Equatable {
        let id: String       // SD-XXXXXXXX, or the service name on an older device
        let name: String     // what to show, e.g. "SensorDyme on sensordyme"
        let url: String      // resolved http://ip:port — a cache, never identity
    }

    @Published var found: [Found] = []
    @Published var isSearching = false
    /// Why discovery cannot work, when the reason is knowable.
    ///
    /// Distinct from "found nothing": a denied permission and an absent device
    /// look identical in the results, and only one of them can be fixed by
    /// looking harder.
    @Published private(set) var problem: String?

    /// The device's own Bonjour service — NOT the backend's. Named here rather
    /// than written inline so the two can't quietly become the same string.
    static let serviceType = "_sensordyme._tcp"

    private var browser: NWBrowser?
    private var resolvers: [NWConnection] = []
    /// Held because NetService stops resolving once released.
    private var srvResolvers: [SRVResolver] = []
    /// The pending auto-stop. Held so a second `start()` cancels the first one's
    /// timer: without this, each start queues its own sleeping task, and the one
    /// from an abandoned search stops a newer search 10 seconds into it.
    private var autoStop: Task<Void, Never>?
    /// Bumped by every `start()`, so a bounded search can tell whether the
    /// browse it started is still the one running. See `discoverOnce`.
    private(set) var scanGeneration = 0

    func start() {
        stop()
        scanGeneration += 1
        isSearching = true
        found = []
        let browser = NWBrowser(
            for: .bonjourWithTXTRecord(type: DeviceDiscovery.serviceType, domain: nil),
            using: NWParameters()
        )
        self.browser = browser
        browser.browseResultsChangedHandler = { [weak self] results, _ in
            Task { @MainActor in
                guard let self else { return }
                for result in results {
                    if case let .service(name, _, _, _) = result.endpoint {
                        let id = Self.deviceID(from: result, fallback: name)
                        // Name first, address as the fallback — `record` keeps
                        // whichever is more durable. See `resolveServiceName`.
                        self.resolveServiceName(
                            endpoint: result.endpoint, name: name, deviceID: id
                        )
                        self.resolve(endpoint: result.endpoint, name: name, deviceID: id)
                    }
                }
            }
        }
        // NWBrowser reports its own failures only through this handler, and
        // nowhere else. Without it a denied Local Network permission — which
        // iOS prompts for, and which people do decline — makes the browser fail
        // while `browseResultsChangedHandler` simply never fires. The app then
        // says it cannot find the device on this network and suggests checking
        // the Wi-Fi, which can never fix it. The user re-joins networks for ever.
        browser.stateUpdateHandler = { [weak self] state in
            Task { @MainActor in
                guard let self else { return }
                switch state {
                case .failed(let error):
                    self.problem = Self.explain(error)
                    self.isSearching = false
                case .waiting(let error):
                    self.problem = Self.explain(error)
                case .ready:
                    self.problem = nil
                case .cancelled:
                    self.isSearching = false
                default:
                    break
                }
            }
        }
        browser.start(queue: .main)
        // Auto-stop after a while; discovery is a burst, not a background drain.
        autoStop = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 10_000_000_000)
            guard !Task.isCancelled else { return }
            self?.stop()
        }
    }

    func stop() {
        autoStop?.cancel()
        autoStop = nil
        browser?.cancel()
        browser = nil
        resolvers.forEach { $0.cancel() }
        resolvers.removeAll()
        srvResolvers.forEach { $0.cancel() }
        srvResolvers.removeAll()
        isSearching = false
    }

    /// Turn a Network framework error into something actionable.
    ///
    /// Only the cases we can genuinely identify. A confident wrong explanation
    /// is worse than the raw error, because it sends someone to fix the wrong
    /// thing — and the permission case is the one that is otherwise invisible.
    private static func explain(_ error: NWError) -> String {
        if case let .dns(code) = error {
            // The code iOS returns when Local Network access is denied.
            if code == -65570 || code == kDNSServiceErr_PolicyDenied {
                return "SensorDyme needs Local Network permission to find your device. "
                     + "Settings → SensorDyme → Local Network."
            }
            return "Couldn't search this network (DNS error \(code))."
        }
        return "Couldn't search this network: \(error.localizedDescription)"
    }

    /// The device's own ID from its mDNS TXT record.
    ///
    /// Falls back to the service name for a device running firmware that predates
    /// the `id` record. That is weaker — a renamed host looks like a new device —
    /// but it is the only handle such a device offers, and refusing to list it
    /// would be worse than listing it imperfectly.
    private static func deviceID(from result: NWBrowser.Result, fallback: String) -> String {
        guard case let .bonjour(record) = result.metadata,
              let identifier = record["id"],
              identifier.hasPrefix("SD-")
        else { return fallback }
        return identifier
    }

    /// Resolve a device service to a URL, preferring its STABLE HOSTNAME.
    ///
    /// Same fix as `BackendLocator.resolveServiceName`, and for the same reason:
    /// reading `currentPath.remoteEndpoint` yields the ADDRESS the service name
    /// resolved to, so the cached `deviceLocalURL` was only true on the network
    /// it was read on. When the Pi's DHCP lease changes the live feed points at
    /// nothing. The SRV target survives that; the address does not.
    ///
    /// The address resolution below is kept as the fallback, because a device
    /// running firmware whose hostname does not resolve must still be reachable.
    private func resolveServiceName(endpoint: NWEndpoint, name: String, deviceID: String) {
        guard case let .service(serviceName, type, domain, _) = endpoint else { return }
        let resolver = SRVResolver(name: serviceName, type: type, domain: domain) { [weak self] host, port in
            Task { @MainActor in
                guard let self, let host, let port else { return }
                var cleaned = host.lowercased()
                while cleaned.hasSuffix(".") { cleaned.removeLast() }
                guard !cleaned.isEmpty,
                      !BackendResolution.isIPLiteral(cleaned),
                      !BackendResolution.isLoopbackName(cleaned)
                else { return }
                self.record(Found(id: deviceID, name: name, url: "http://\(cleaned):\(port)"),
                            preferStable: true)
            }
        }
        srvResolvers.append(resolver)
        resolver.start()
    }

    /// Store a resolution, keeping a stable hostname in preference to an address.
    ///
    /// Replace in place. Appending would move a device down the list every time
    /// it re-resolved, and re-resolution is routine — Bonjour reports the same
    /// service repeatedly, and a device with both IPv4 and IPv6 resolves twice.
    private func record(_ item: Found, preferStable: Bool) {
        if let index = found.firstIndex(where: { $0.id == item.id }) {
            if !preferStable, let existing = BackendResolution.host(of: found[index].url),
               !BackendResolution.isIPLiteral(existing) {
                return  // already have a name for this device; an address is no improvement
            }
            found[index] = item
        } else {
            found.append(item)
        }
    }

    /// Bonjour gives a service endpoint; opening a throwaway connection to it is
    /// the supported way to learn the concrete ip:port for building a URL.
    private func resolve(endpoint: NWEndpoint, name: String, deviceID: String) {
        let connection = NWConnection(to: endpoint, using: .tcp)
        resolvers.append(connection)
        connection.stateUpdateHandler = { [weak self, weak connection] state in
            guard case .ready = state,
                  let remote = connection?.currentPath?.remoteEndpoint,
                  case let .hostPort(host, port) = remote else { return }
            // Scope ("%en0") off BEFORE brackets on -- the other order produced
            // "[fe80::1", which is not a URL. See BackendResolution.urlHostLiteral.
            let cleaned: String
            switch host {
            case .ipv4(let addr): cleaned = BackendResolution.urlHostLiteral("\(addr)", isIPv6: false)
            case .ipv6(let addr): cleaned = BackendResolution.urlHostLiteral("\(addr)", isIPv6: true)
            case .name(let hostname, _): cleaned = BackendResolution.urlHostLiteral(hostname, isIPv6: false)
            @unknown default: return
            }
            let url = "http://\(cleaned):\(port.rawValue)"
            Task { @MainActor in
                guard let self else { return }
                self.record(Found(id: deviceID, name: name, url: url), preferStable: false)
                connection?.cancel()
            }
        }
        connection.start(queue: .main)
    }
}

/// Reads the device's /api/health (served by the dashboard from the on-device
/// health monitor) for the status card in Settings.
struct DeviceHealth: Decodable {
    struct Component: Decodable {
        let state: String
        let detail: String
        let critical: Bool
    }

    /// The `device` block the runtime adds (core/device_health.py). Only the
    /// identity is read: it is what says WHICH camera answered at an address.
    struct Identity: Decodable {
        let device_id: String?
    }

    let state: String
    let uptime_s: Double
    let components: [String: Component]
    var device: Identity? = nil

    /// The camera's own SD- id, when it reported one.
    var deviceID: String? { device?.device_id }

    static func fetch(baseURL: String, token: String, timeout: TimeInterval = 8) async throws -> DeviceHealth {
        guard let url = URL(string: baseURL)?.appendingPathComponent("api/health") else {
            throw APIError(message: "invalid device address")
        }
        var request = URLRequest(url: url)
        request.timeoutInterval = timeout
        request.cachePolicy = .reloadIgnoringLocalCacheData
        let cleaned = token.trimmingCharacters(in: .whitespacesAndNewlines)
        if !cleaned.isEmpty {
            request.setValue(cleaned, forHTTPHeaderField: MJPEGStreamRequest.tokenHeader)
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        if let problem = Self.problem(status: status, hasToken: !cleaned.isEmpty) {
            throw APIError(message: problem, status: status)
        }
        return try JSONDecoder().decode(DeviceHealth.self, from: data)
    }

    /// What a non-200 from the device means. A 401/403 is the device ANSWERING
    /// and refusing the token -- reporting it as "did not answer, is it on your
    /// network?" sent people to fix Wi-Fi over a settings mistake.
    static func problem(status: Int, hasToken: Bool) -> String? {
        switch status {
        case 200: return nil
        case 401, 403:
            return MJPEGStreamState.unauthorized(hasToken: hasToken).message
        default:
            return "device did not answer (is it on and on your network?)"
        }
    }
}
