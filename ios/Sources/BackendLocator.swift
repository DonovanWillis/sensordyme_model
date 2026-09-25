import Foundation
import Network

/// Finds the SensorDyme backend automatically on the local network.
@MainActor
final class BackendLocator: ObservableObject {

    static let shared = BackendLocator()

    /// Must match the service published by cloud/mdns.py
    static let serviceType = "_sensordyme-api._tcp"

    @Published private(set) var isSearching = false
    @Published private(set) var problem: String?

    private static let retryInterval: TimeInterval = 10

    private var browser: NWBrowser?
    private var resolvers: [NWConnection] = []
    private var candidates: [String] = []

    private var inFlight: Task<String?, Never>?
    private var lastFailure: Date?

    // MARK: - Public discovery

    func search() async -> String? {

        print("🔎 SensorDyme: search() started")

        if let existing = inFlight {
            print("🔎 SensorDyme: joining existing search")
            return await existing.value
        }

        if let lastFailure,
           Date().timeIntervalSince(lastFailure) < Self.retryInterval {

            print("🔎 SensorDyme: search skipped because of retry backoff")
            return nil
        }

        let task = Task { @MainActor [weak self] () -> String? in
            guard let self else {
                print("❌ SensorDyme: BackendLocator disappeared")
                return nil
            }

            print("🔎 SensorDyme: beginning Bonjour discovery")

            let discovered = await self.discover()

            print("🔎 SensorDyme: Bonjour returned \(discovered.count) candidate(s)")

            for candidate in discovered {

                print("🔎 SensorDyme: testing candidate:")
                print("   \(candidate)")

                let works = await Self.answers(candidate)

                print("🔎 SensorDyme: health check result = \(works)")

                if works {
                    print("✅ SensorDyme: FOUND BACKEND")
                    print("✅ \(candidate)")
                    return candidate
                }
            }

            print("❌ SensorDyme: no working backend found")
            return nil
        }

        inFlight = task

        let found = await task.value

        inFlight = nil
        lastFailure = found == nil ? Date() : nil

        if let found {
            print("✅ SensorDyme: search completed with \(found)")
        } else {
            print("❌ SensorDyme: search completed with nil")
        }

        return found
    }

    // MARK: - Health check

    static func answers(_ base: String) async -> Bool {

        print("🩺 SensorDyme: checking \(base)/v1/health")

        guard let url = URL(string: base)?
            .appendingPathComponent("v1/health") else {

            print("❌ SensorDyme: invalid URL \(base)")
            return false
        }

        var request = URLRequest(url: url)
        request.timeoutInterval = 5
        request.cachePolicy = .reloadIgnoringLocalCacheData

        do {

            let (data, response) = try await URLSession.shared.data(
                for: request
            )

            guard let http = response as? HTTPURLResponse else {
                print("❌ SensorDyme: response was not HTTP")
                return false
            }

            print("🩺 SensorDyme: HTTP status \(http.statusCode)")

            guard http.statusCode == 200 else {
                print("❌ SensorDyme: health endpoint returned \(http.statusCode)")
                return false
            }

            guard let payload =
                    (try? JSONSerialization.jsonObject(with: data))
                    as? [String: Any] else {

                print("❌ SensorDyme: health response was not valid JSON")
                return false
            }

            let service = payload["service"] as? String

            print("🩺 SensorDyme: service = \(service ?? "nil")")

            let valid = service == "sensordyme-cloud"

            if valid {
                print("✅ SensorDyme: health check PASSED")
            } else {
                print("❌ SensorDyme: wrong service identifier")
            }

            return valid

        } catch {

            print("❌ SensorDyme: health request failed")
            print("   \(error)")
            print("   \(error.localizedDescription)")

            return false
        }
    }

    // MARK: - Bonjour discovery

    func discover(timeout: TimeInterval = 5) async -> [String] {

        print("📡 SensorDyme: starting Bonjour browser")
        print("📡 Service type: \(Self.serviceType)")

        startBrowsing()

        defer {
            stopBrowsing()
            print("📡 SensorDyme: Bonjour browser stopped")
        }

        let deadline = Date().addingTimeInterval(timeout)

        while Date() < deadline {

            try? await Task.sleep(
                nanoseconds: 200_000_000
            )

            if !candidates.isEmpty {
                print(
                    "📡 SensorDyme: candidate(s) found early: \(candidates)"
                )
                break
            }
        }

        if candidates.isEmpty {
            print("❌ SensorDyme: Bonjour found ZERO candidates")
        } else {
            print("📡 SensorDyme: candidates = \(candidates)")
        }

        return candidates.sorted { first, second in
            !first.contains("[") && second.contains("[")
        }
    }

    // MARK: - Start Bonjour

    private func startBrowsing() {

        stopBrowsing()

        candidates.removeAll()
        problem = nil
        isSearching = true

        print("📡 SensorDyme: creating NWBrowser")

        let browser = NWBrowser(
            for: .bonjour(
                type: Self.serviceType,
                domain: nil
            ),
            using: NWParameters()
        )

        self.browser = browser

        browser.browseResultsChangedHandler = {
            [weak self] results, changes in

            Task { @MainActor in

                guard let self else { return }

                print(
                    "📡 SensorDyme: browse results changed. " +
                    "results=\(results.count)"
                )

                for change in changes {
                    print("📡 SensorDyme: change = \(change)")
                }

                for result in results {

                    print(
                        "📡 SensorDyme: discovered endpoint = " +
                        "\(result.endpoint)"
                    )

                    self.resolve(result.endpoint)
                }
            }
        }

        browser.stateUpdateHandler = {
            [weak self] state in

            Task { @MainActor in

                guard let self else { return }

                print(
                    "📡 SensorDyme: browser state = \(state)"
                )

                switch state {

                case .failed(let error):

                    print(
                        "❌ SensorDyme: Bonjour browser FAILED"
                    )

                    print("❌ \(error)")
                    print("❌ \(error.localizedDescription)")

                    self.problem = Self.explain(error)
                    self.isSearching = false

                case .waiting(let error):

                    print(
                        "⏳ SensorDyme: Bonjour browser WAITING"
                    )

                    print("⏳ \(error)")
                    print("⏳ \(error.localizedDescription)")

                    self.problem = Self.explain(error)

                case .ready:

                    print(
                        "✅ SensorDyme: Bonjour browser READY"
                    )

                    self.problem = nil

                case .setup:

                    print(
                        "📡 SensorDyme: Bonjour browser SETUP"
                    )

                case .cancelled:

                    print(
                        "📡 SensorDyme: Bonjour browser CANCELLED"
                    )

                @unknown default:

                    print(
                        "📡 SensorDyme: unknown Bonjour state"
                    )
                }
            }
        }

        print("📡 SensorDyme: starting browser")

        browser.start(queue: .main)
    }

    // MARK: - Stop Bonjour

    private func stopBrowsing() {

        browser?.cancel()
        browser = nil

        resolvers.forEach {
            $0.cancel()
        }

        resolvers.removeAll()

        isSearching = false
    }

    // MARK: - Resolve Bonjour endpoint

    private func resolve(_ endpoint: NWEndpoint) {

        print(
            "🔗 SensorDyme: resolving endpoint \(endpoint)"
        )

        let connection = NWConnection(
            to: endpoint,
            using: .tcp
        )

        resolvers.append(connection)

        connection.stateUpdateHandler = {
            [weak self, weak connection] state in

            guard let self else { return }

            print(
                "🔗 SensorDyme: resolver state = \(state)"
            )

            guard case .ready = state else {

                if case .failed(let error) = state {

                    print(
                        "❌ SensorDyme: resolver FAILED"
                    )

                    print("❌ \(error)")
                    print("❌ \(error.localizedDescription)")
                }

                return
            }

            guard let remote =
                    connection?.currentPath?.remoteEndpoint else {

                print(
                    "❌ SensorDyme: resolver ready but no remote endpoint"
                )

                return
            }

            print(
                "🔗 SensorDyme: remote endpoint = \(remote)"
            )

            guard case let .hostPort(host, port) = remote else {

                print(
                    "❌ SensorDyme: remote endpoint was not host/port"
                )

                return
            }

            guard let address = Self.urlHost(from: host) else {

                print(
                    "❌ SensorDyme: could not convert host \(host)"
                )

                return
            }

            let url =
                "http://\(address):\(port.rawValue)"

            print(
                "✅ SensorDyme: resolved backend = \(url)"
            )

            Task { @MainActor in

                guard !self.candidates.contains(url) else {
                    connection?.cancel()
                    return
                }

                self.candidates.append(url)

                print(
                    "✅ SensorDyme: added candidate \(url)"
                )

                connection?.cancel()
            }
        }

        connection.start(queue: .main)
    }

    // MARK: - URL host conversion

    nonisolated private static func urlHost(
        from host: NWEndpoint.Host
    ) -> String? {

        func unscoped(_ value: String) -> String {

            value
                .split(separator: "%")
                .first
                .map(String.init) ?? value
        }

        switch host {

        case .ipv4(let address):

            return unscoped("\(address)")

        case .ipv6(let address):

            return "[\(unscoped("\(address)"))]"

        case .name(let hostname, _):

            return unscoped(hostname)

        @unknown default:

            return nil
        }
    }

    // MARK: - Bonjour error explanation

    private static func explain(
        _ error: NWError
    ) -> String {

        if case let .dns(code) = error {

            if code == -65570 ||
                code == kDNSServiceErr_PolicyDenied {

                return """
                SensorDyme needs Local Network permission to \
                find your server. \
                Settings → SensorDyme → Local Network.
                """
            }

            return """
            Couldn't search this network \
            (DNS error \(code)).
            """
        }

        return """
        Couldn't search this network: \
        \(error.localizedDescription)
        """
    }
}
