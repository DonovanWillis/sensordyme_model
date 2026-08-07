import Foundation
import Network

/// Finds SensorDyme devices on the local network via Bonjour (_sensordyme._tcp,
/// advertised by avahi on the device) and resolves them to an http base URL —
/// so nobody ever types an IP address into Settings.
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

    private var browser: NWBrowser?
    private var resolvers: [NWConnection] = []

    func start() {
        stop()
        isSearching = true
        found = []
        let browser = NWBrowser(
            for: .bonjour(type: "_sensordyme._tcp", domain: nil),
            using: NWParameters()
        )
        self.browser = browser
        browser.browseResultsChangedHandler = { [weak self] results, _ in
            Task { @MainActor in
                guard let self else { return }
                for result in results {
                    if case let .service(name, _, _, _) = result.endpoint {
                        self.resolve(
                            endpoint: result.endpoint,
                            name: name,
                            deviceID: Self.deviceID(from: result, fallback: name)
                        )
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
        Task { [weak self] in
            try? await Task.sleep(nanoseconds: 10_000_000_000)
            self?.stop()
        }
    }

    func stop() {
        browser?.cancel()
        browser = nil
        resolvers.forEach { $0.cancel() }
        resolvers.removeAll()
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

    /// Bonjour gives a service endpoint; opening a throwaway connection to it is
    /// the supported way to learn the concrete ip:port for building a URL.
    private func resolve(endpoint: NWEndpoint, name: String, deviceID: String) {
        let connection = NWConnection(to: endpoint, using: .tcp)
        resolvers.append(connection)
        connection.stateUpdateHandler = { [weak self, weak connection] state in
            guard case .ready = state,
                  let remote = connection?.currentPath?.remoteEndpoint,
                  case let .hostPort(host, port) = remote else { return }
            let ip: String
            switch host {
            case .ipv4(let addr): ip = "\(addr)"
            case .ipv6(let addr): ip = "[\(addr)]"
            case .name(let hostname, _): ip = hostname
            @unknown default: return
            }
            // Strip any interface scope suffix ("%en0") NWEndpoint may append.
            let cleaned = ip.split(separator: "%").first.map(String.init) ?? ip
            let url = "http://\(cleaned):\(port.rawValue)"
            Task { @MainActor in
                guard let self else { return }
                let item = Found(id: deviceID, name: name, url: url)
                // Replace in place. Appending would move a device down the list
                // every time it re-resolved, and re-resolution is routine —
                // Bonjour reports the same service repeatedly, and a device with
                // both IPv4 and IPv6 resolves twice in a row.
                if let index = self.found.firstIndex(where: { $0.id == deviceID }) {
                    self.found[index] = item
                } else {
                    self.found.append(item)
                }
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

    let state: String
    let uptime_s: Double
    let components: [String: Component]

    static func fetch(baseURL: String, token: String) async throws -> DeviceHealth {
        guard let url = URL(string: baseURL)?.appendingPathComponent("api/health") else {
            throw APIError(message: "invalid device address")
        }
        var request = URLRequest(url: url)
        request.timeoutInterval = 8
        if !token.isEmpty {
            request.setValue(token, forHTTPHeaderField: "X-Device-Token")
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else {
            throw APIError(message: "device did not answer (is it on and on your network?)")
        }
        return try JSONDecoder().decode(DeviceHealth.self, from: data)
    }
}
