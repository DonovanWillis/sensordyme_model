import Foundation
import Network
import os

/// Finds the SensorDyme backend on the local network and confirms it is really
/// the backend before saying so.
///
/// Two mechanisms, deliberately separate: Bonjour tells us where a service
/// *claims* to be, and `answers` tells us whether anything is actually there.
/// Treating the first as the second is the mistake this type exists to avoid —
/// an mDNS record outlives the process that published it, and a laptop that has
/// changed networks leaves a record pointing at an address that no longer exists.
///
/// Policy lives in `BackendResolution` (what to try, in what order, how long to
/// wait between attempts) and sequencing lives in `AppState`. This type only
/// does the networking, which is what keeps the decisions testable without a
/// simulator.
@MainActor
final class BackendLocator: ObservableObject {

    static let shared = BackendLocator()

    /// Must match the service published by cloud/mdns.py, and must be listed in
    /// NSBonjourServices (ios/Info.plist and ios/project.yml). iOS returns
    /// nothing at all for an undeclared type and says nothing about why.
    static let serviceType = "_sensordyme-api._tcp"

    /// The `service` field the real backend reports from `/v1/health`. Checked so
    /// the app cannot latch onto some other server that happens to answer on the
    /// same port on the same Wi-Fi.
    static let expectedService = "sensordyme-cloud"

    private static let log = Logger(subsystem: "com.sensordyme.app", category: "backend-discovery")

    @Published private(set) var isSearching = false
    /// Why discovery cannot work, when the reason is knowable — as distinct from
    /// "found nothing". A denied Local Network permission and an absent server
    /// look identical in the results, and only one of them can be fixed by
    /// looking harder.
    @Published private(set) var problem: String?
    /// Set when several backends answered and none is the remembered one.
    ///
    /// Kept apart from `problem`, which AppState reads as "discovery itself
    /// cannot work" (a denied Local Network permission) and which therefore
    /// skips the reconnecting state. Stored there, this message outlived the
    /// search that produced it and was shown as the cause of later, unrelated
    /// outages -- a stopped server read as "Found 2 SensorDyme servers".
    @Published private(set) var ambiguity: String?

    private var browser: NWBrowser?
    private var resolvers: [NWConnection] = []
    /// SRV resolvers, held because NetService does nothing once released.
    private var srvResolvers: [SRVResolver] = []
    private var candidates: [BackendResolution.FoundBackend] = []
    /// Endpoints a resolver has already been started for. Bonjour reports the
    /// full result set on every change, so without this the same service gets a
    /// fresh NWConnection on every callback.
    private var resolving: Set<String> = []

    /// The search currently running, so concurrent callers join it instead of
    /// starting a second browse. Several views ask at once on launch.
    private var inFlight: Task<BackendResolution.FoundBackend?, Never>?

    // MARK: - Public discovery

    /// The first discovered address that answers as the SensorDyme backend, or
    /// nil if none does.
    ///
    /// No retry policy here on purpose. This used to keep its own 10-second
    /// "don't ask again" window, which was a second, invisible backoff competing
    /// with the caller's — so a deliberate "find my server" tap could silently do
    /// nothing. Retrying is the caller's decision; see `BackendResolution.retryDelay`.
    ///
    /// Returns the backend to use, or nil if none is usable.
    ///
    /// `remembering` is the id of the backend this app last connected to. Passed
    /// in rather than read here so this type stays free of storage: the policy
    /// lives in `BackendResolution.chooseBackend`, which is testable without a
    /// network.
    func search(timeout: TimeInterval = 5, remembering remembered: String? = nil)
        async -> BackendResolution.FoundBackend? {
        if let existing = inFlight {
            return await existing.value
        }

        let task = Task { @MainActor [weak self] () -> BackendResolution.FoundBackend? in
            guard let self else { return nil }
            let discovered = await self.discover(timeout: timeout)
            Self.log.info("Bonjour returned \(discovered.count, privacy: .public) candidate(s)")

            // Health-check every candidate before choosing between them. A stale
            // record from a network this machine has left still resolves, so
            // "answered" is what makes a candidate real — and choosing first and
            // validating afterwards would let one dead record hide a live backend.
            var healthy: [BackendResolution.FoundBackend] = []
            for candidate in discovered {
                if Task.isCancelled { return nil }
                if await Self.answers(candidate.url) {
                    healthy.append(candidate)
                } else {
                    Self.log.info(
                        "candidate did not answer as the backend: \(candidate.url, privacy: .public)")
                }
            }

            switch BackendResolution.chooseBackend(from: healthy, remembering: remembered) {
            case .use(let backend):
                Self.log.info(
                    "backend confirmed healthy at \(backend.url, privacy: .public) (id \(backend.id ?? "none", privacy: .public))")
                self.problem = nil
                return backend
            case .ambiguous(let all):
                // Deliberately not a guess. On a shared network the wrong choice
                // means someone watching a stranger's camera.
                Self.log.error(
                    "\(all.count, privacy: .public) SensorDyme backends answered and none is the one this app knows; not choosing between them")
                let found = all.count == 1 ? "Found a SensorDyme server" : "Found \(all.count) SensorDyme servers"
                self.ambiguity = "\(found) on this Wi-Fi, but not the one this app uses. Your server may be asleep or on another network; to use a different one, open Settings → Manual server address."
                return nil
            case .none:
                return nil
            }
        }

        inFlight = task
        let found = await task.value
        // Only clear the slot if it is still ours: after cancelSearch() a newer
        // search may already occupy it.
        if inFlight == task { inFlight = nil }
        return found
    }

    /// Cancel an in-flight search. Used when the network changes underneath one,
    /// so a browse against the old interface cannot deliver a stale answer.
    func cancelSearch() {
        inFlight?.cancel()
        inFlight = nil
        stopBrowsing()
    }

    // MARK: - Health check

    /// Does a real SensorDyme backend answer here?
    ///
    /// Three separate facts are checked, because passing only the first two is
    /// how the app used to report a connection to something that was not the
    /// backend: the request completes, the status is 200, and the body identifies
    /// the service as ours.
    static func answers(_ base: String) async -> Bool {
        guard let url = URL(string: base)?.appendingPathComponent("v1/health") else {
            log.error("not a usable base URL: \(base, privacy: .public)")
            return false
        }
        var request = URLRequest(url: url)
        request.timeoutInterval = 5
        // A cached 200 from a server that has since gone away would make an
        // unreachable backend look healthy.
        request.cachePolicy = .reloadIgnoringLocalCacheData

        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                return false
            }
            guard
                let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                let service = payload["service"] as? String
            else { return false }
            return service == expectedService
        } catch {
            return false
        }
    }

    // MARK: - Bonjour discovery

    /// Every backend address advertised on this network, resolved to a URL.
    ///
    /// Collects the whole set rather than returning the first one to arrive. The
    /// earlier version broke out of its wait as soon as one candidate appeared,
    /// which meant a single stale record was the entire result: it failed its
    /// health check, there was nothing behind it, and discovery reported no
    /// backend on a network where the backend was present and healthy.
    func discover(timeout: TimeInterval = 5) async -> [BackendResolution.FoundBackend] {
        startBrowsing()
        defer { stopBrowsing() }

        let deadline = Date().addingTimeInterval(timeout)
        /// Once something has been found, wait this much longer for the rest
        /// rather than the full timeout — long enough for a second interface or a
        /// second server to report, short enough not to stall sign-in.
        let grace: TimeInterval = 1.0
        var firstFoundAt: Date?

        while Date() < deadline {
            if Task.isCancelled { return [] }
            try? await Task.sleep(nanoseconds: 200_000_000)
            if !candidates.isEmpty {
                let now = Date()
                if let firstFoundAt {
                    if now.timeIntervalSince(firstFoundAt) >= grace { break }
                } else {
                    firstFoundAt = now
                }
            }
        }

        if candidates.isEmpty {
            Self.log.info("Bonjour found no backend on this network")
        }

        // A NAME first, then IPv4, then IPv6 — see
        // BackendResolution.preferringStableHost for why the name is not merely
        // one option among several.
        let endpoints = candidates.compactMap { found in
            BackendResolution.BackendEndpoint(url: found.url, backendID: found.id)
        }
        return BackendResolution.preferringStableHost(endpoints).map { endpoint in
            BackendResolution.FoundBackend(
                url: endpoint.baseURL, id: endpoint.backendID,
                service: candidates.first { $0.url == endpoint.baseURL }?.service
            )
        }
    }

    // MARK: - Browser lifecycle

    private func startBrowsing() {
        stopBrowsing()
        candidates.removeAll()
        resolving.removeAll()
        problem = nil
        ambiguity = nil
        isSearching = true

        let browser = NWBrowser(
            for: .bonjourWithTXTRecord(type: Self.serviceType, domain: nil),
            using: NWParameters()
        )
        self.browser = browser

        browser.browseResultsChangedHandler = { [weak self] results, _ in
            Task { @MainActor in
                guard let self else { return }
                for result in results {
                    // The id travels with the browse result, not the resolved
                    // connection, so it has to be read here and carried through.
                    let id = Self.backendID(from: result)
                    // Two resolutions per service, and the order matters:
                    // the SRV target is the backend's STABLE name and is what we
                    // want to keep, while the connection only yields today's
                    // address. Both are offered as candidates and
                    // `BackendResolution.preferringStableHost` puts the name
                    // first, so the address is a fallback for a service that
                    // publishes no usable hostname rather than the normal answer.
                    self.resolveServiceName(result.endpoint, id: id)
                    self.resolve(result.endpoint, id: id)
                }
            }
        }

        // NWBrowser reports its own failures only through this handler and
        // nowhere else. Without it, a denied Local Network permission makes the
        // browser fail while the results handler simply never fires — so the app
        // says it cannot find the server and suggests checking the Wi-Fi, which
        // can never fix it, and the user re-joins networks for ever.
        browser.stateUpdateHandler = { [weak self] state in
            Task { @MainActor in
                guard let self else { return }
                switch state {
                case .failed(let error):
                    Self.log.error("Bonjour browser failed: \(error.localizedDescription, privacy: .public)")
                    self.problem = Self.explain(error)
                    self.isSearching = false
                case .waiting(let error):
                    Self.log.info("Bonjour browser waiting: \(error.localizedDescription, privacy: .public)")
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
    }

    private func stopBrowsing() {
        browser?.cancel()
        browser = nil
        resolvers.forEach { $0.cancel() }
        resolvers.removeAll()
        srvResolvers.forEach { $0.cancel() }
        srvResolvers.removeAll()
        resolving.removeAll()
        isSearching = false
    }

    // MARK: - Resolving an endpoint to a URL

    /// Bonjour hands back a service endpoint; opening a throwaway connection to
    /// it is the supported way to learn the concrete ip:port behind it.
    /// This backend's id out of the service's TXT record, or nil for a backend
    /// published by a version that predates it. Same shape as
    /// `DeviceDiscovery.deviceID`, which reads `id=SD-…` for the camera.
    nonisolated private static func backendID(from result: NWBrowser.Result) -> String? {
        guard case let .bonjour(record) = result.metadata,
              let identifier = record["id"],
              !identifier.isEmpty
        else { return nil }
        return identifier
    }

    /// Recover the backend's STABLE HOSTNAME from the Bonjour service record.
    ///
    /// This is the half that was missing, and its absence was the whole bug.
    /// `resolve` below opens a connection and reads `currentPath.remoteEndpoint`,
    /// which is the address the name resolved to — so the app cached
    /// `http://10.97.118.172:8787` and never learned that the backend calls
    /// itself `sensordyme-backend.local`. cloud/mdns.py publishes that name in
    /// the SRV record precisely so the address can change without the app caring
    /// (see STABLE_HOSTNAME there, and the ten-day outage that motivated it).
    ///
    /// NWBrowser does not expose the SRV target, so this uses NetService, which
    /// does: `hostName` after a successful resolve is exactly the SRV target.
    /// Failure is not fatal — `resolve` still contributes the address.
    private func resolveServiceName(_ endpoint: NWEndpoint, id: String?) {
        guard case let .service(name, type, domain, _) = endpoint else { return }
        let key = "srv:\(name).\(type).\(domain)"
        guard !resolving.contains(key) else { return }
        resolving.insert(key)

        let resolver = SRVResolver(name: name, type: type, domain: domain) { [weak self] host, port in
            Task { @MainActor in
                guard let self, let host, let port else { return }
                let endpoint = BackendResolution.BackendEndpoint(
                    scheme: "http", host: host, port: port, backendID: id
                )
                // A service may publish a loopback or numeric target; neither is
                // better than what `resolve` already provides.
                guard endpoint.hostIsStable,
                      BackendResolution.isUsableFromPhone(endpoint.baseURL)
                else {
                    Self.log.info(
                        "service target is not a usable stable name: \(endpoint.baseURL, privacy: .public)")
                    return
                }
                guard !self.candidates.contains(where: { $0.url == endpoint.baseURL }) else { return }
                self.candidates.append(.init(url: endpoint.baseURL, id: id, service: key))
                Self.log.info(
                    "resolved the backend's stable name \(endpoint.baseURL, privacy: .public) (id \(id ?? "none", privacy: .public))")
            }
        }
        srvResolvers.append(resolver)
        resolver.start()
    }

    /// Identifies the Bonjour service an endpoint belongs to, so the two
    /// resolutions of one service can be recognised as one backend.
    nonisolated private static func serviceKey(_ endpoint: NWEndpoint) -> String? {
        guard case let .service(name, type, domain, _) = endpoint else { return nil }
        return "srv:\(name).\(type).\(domain)"
    }

    private func resolve(_ endpoint: NWEndpoint, id: String?) {
        let key = "\(endpoint)"
        guard !resolving.contains(key) else { return }
        resolving.insert(key)

        let connection = NWConnection(to: endpoint, using: .tcp)
        resolvers.append(connection)

        connection.stateUpdateHandler = { [weak self, weak connection] state in
            switch state {
            case .ready:
                guard
                    let remote = connection?.currentPath?.remoteEndpoint,
                    case let .hostPort(host, port) = remote,
                    let address = Self.urlHost(from: host)
                else {
                    connection?.cancel()
                    return
                }
                let url = "http://\(address):\(port.rawValue)"
                Task { @MainActor in
                    guard let self else { return }
                    defer { connection?.cancel() }
                    // macOS publishes 127.0.0.1 in the A record for its own
                    // .local hostname alongside the real address, so resolution
                    // genuinely can hand back loopback — which on a phone means
                    // the phone itself.
                    guard BackendResolution.isUsableFromPhone(url) else {
                        Self.log.info("ignoring unusable address \(url, privacy: .public)")
                        return
                    }
                    guard !self.candidates.contains(where: { $0.url == url }) else { return }
                    self.candidates.append(.init(url: url, id: id, service: Self.serviceKey(endpoint)))
                    Self.log.info(
                        "resolved a backend candidate at \(url, privacy: .public) (id \(id ?? "none", privacy: .public))")
                }
            case .failed, .cancelled:
                connection?.cancel()
            default:
                break
            }
        }

        connection.start(queue: .main)
    }

    nonisolated private static func urlHost(from host: NWEndpoint.Host) -> String? {
        /// Strip any interface scope suffix ("%en0") NWEndpoint may append.
        func unscoped(_ value: String) -> String {
            value.split(separator: "%").first.map(String.init) ?? value
        }
        switch host {
        case .ipv4(let address): return unscoped("\(address)")
        case .ipv6(let address): return "[\(unscoped("\(address)"))]"
        case .name(let hostname, _): return unscoped(hostname)
        @unknown default: return nil
        }
    }

    // MARK: - Explaining a failure

    private static func explain(_ error: NWError) -> String {
        if case let .dns(code) = error {
            // The code iOS returns when Local Network access is denied.
            if code == -65570 || code == kDNSServiceErr_PolicyDenied {
                return "SensorDyme needs Local Network permission to find your server. "
                     + "Settings → SensorDyme → Local Network."
            }
            return "Couldn't search this network (DNS error \(code))."
        }
        return "Couldn't search this network: \(error.localizedDescription)"
    }
}

/// Resolves one Bonjour service to its SRV target hostname and port.
///
/// A small NSObject wrapper because NetService is delegate-based and predates
/// Network.framework — but it is the only API that hands back the SRV target,
/// which is the backend's stable identity. NWBrowser/NWConnection only ever
/// reveal the resolved address.
final class SRVResolver: NSObject, NetServiceDelegate {
    private let service: NetService
    private var finish: ((String?, Int?) -> Void)?

    init(name: String, type: String, domain: String,
         onResolved: @escaping (String?, Int?) -> Void) {
        // NetService wants a dotted, trailing-dot domain; Bonjour reports "local".
        let resolvedDomain = domain.isEmpty ? "local." : domain
        self.service = NetService(domain: resolvedDomain, type: type, name: name)
        self.finish = onResolved
        super.init()
        self.service.delegate = self
    }

    func start() {
        // Short on purpose: this races the address resolution in `resolve`, and a
        // service whose name will not resolve must not delay discovery.
        service.resolve(withTimeout: 3)
    }

    func cancel() {
        finish = nil
        service.delegate = nil
        service.stop()
    }

    private func deliver(_ host: String?, _ port: Int?) {
        let callback = finish
        finish = nil  // exactly once, whichever delegate method fires
        callback?(host, port)
    }

    func netServiceDidResolveAddress(_ sender: NetService) {
        deliver(sender.hostName, sender.port > 0 ? sender.port : nil)
    }

    func netService(_ sender: NetService, didNotResolve errorDict: [String: NSNumber]) {
        deliver(nil, nil)
    }

    func netServiceDidStop(_ sender: NetService) {
        deliver(nil, nil)
    }
}
