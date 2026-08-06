import Foundation
import Network

/// Finds SensorDyme devices on the local network via Bonjour (_sensordyme._tcp,
/// advertised by avahi on the device) and resolves them to an http base URL —
/// so nobody ever types an IP address into Settings.
@MainActor
final class DeviceDiscovery: ObservableObject {
    struct Found: Identifiable, Equatable {
        let id: String   // service name, e.g. "SensorDyme on sensordyme"
        let url: String  // resolved http://ip:port
    }

    @Published var found: [Found] = []
    @Published var isSearching = false

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
                        self.resolve(endpoint: result.endpoint, name: name)
                    }
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

    /// Bonjour gives a service endpoint; opening a throwaway connection to it is
    /// the supported way to learn the concrete ip:port for building a URL.
    private func resolve(endpoint: NWEndpoint, name: String) {
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
                let item = Found(id: name, url: url)
                if !self.found.contains(item) {
                    self.found.removeAll { $0.id == name }
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
