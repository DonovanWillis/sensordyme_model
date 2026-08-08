import Foundation
import Network

/// Finds the SensorDyme backend on the local network, so signing in never
/// requires anyone to know an address.
///
/// The address used to be something a person typed into Settings, which failed
/// in two ways that produce the identical symptom — "nothing is answering":
///
///   * a new phone has nothing saved, and cannot reach the server to sign in at
///     all, so the one screen that needs the address is the one screen that
///     cannot get past it; and
///   * a saved address goes stale on its own, because DHCP moves machines. The
///     server keeps running perfectly while the app points at where it used to be.
///
/// The server now announces itself over mDNS (`_sensordyme-api._tcp`, published
/// by cloud/mdns.py) and this finds it. Note what this does and does not fix:
/// credentials authenticate, they cannot *locate*. Something still has to say
/// where the server is, and this is that something — it just asks the network
/// instead of asking the user.
///
/// Deliberately link-local. This makes the address irrelevant on the network the
/// server is on and does nothing off it; reaching a backend from anywhere needs a
/// real public address. That case is still supported and still wins: a saved
/// address that answers is always preferred over searching, so a public URL typed
/// in once is never overridden by whatever happens to be on the local Wi-Fi.
@MainActor
final class BackendLocator: ObservableObject {
    static let shared = BackendLocator()

    /// Must match SERVICE_TYPE in cloud/mdns.py and the NSBonjourServices entry
    /// in ios/Info.plist. iOS silently returns nothing for an undeclared service
    /// type, so a mismatch here looks exactly like "no server on this network".
    static let serviceType = "_sensordyme-api._tcp"

    @Published private(set) var isSearching = false
    /// Why searching cannot work, when that is knowable. A denied Local Network
    /// permission and an absent server are indistinguishable in the results, and
    /// only one of them can be fixed by looking harder.
    @Published private(set) var problem: String?

    /// How long to leave the network alone after a search that found nothing.
    private static let retryInterval: TimeInterval = 10

    private var browser: NWBrowser?
    private var resolvers: [NWConnection] = []
    private var candidates: [String] = []
    /// The search everyone currently waiting is sharing.
    private var inFlight: Task<String?, Never>?
    /// When the last search came back empty, or nil if the last one succeeded.
    private var lastFailure: Date?

    // MARK: - What callers use

    /// An address that is answering right now, or nil.
    ///
    /// A saved address gets first refusal, and only a *failed* health check sends
    /// us to the network. That ordering is what keeps a deployed public backend
    /// stable: it is checked first and, while it answers, never replaced.
    func locate(saved: String) async -> String? {
        if !saved.isEmpty, await Self.answers(saved) { return saved }
        return await search()
    }

    /// Ignore whatever is saved and ask the network.
    ///
    /// Used when the saved address has just failed, which is precisely the moment
    /// it must not be trusted again.
    func search() async -> String? {
        // Callers arrive in bursts: several polling screens fail within the same
        // second, and each one starting its own browse would cancel the others'
        // and reset the shared results. One search, shared by everyone waiting.
        if let existing = inFlight { return await existing.value }
        // And when there is genuinely no server here, a failed search must not
        // cost a five-second sweep every few seconds for ever. Back off instead.
        if let lastFailure, Date().timeIntervalSince(lastFailure) < Self.retryInterval {
            return nil
        }
        let task = Task { [weak self] () -> String? in
            guard let self else { return nil }
            for candidate in await self.discover() {
                if await Self.answers(candidate) { return candidate }
            }
            return nil
        }
        inFlight = task
        let found = await task.value
        inFlight = nil
        lastFailure = found == nil ? Date() : nil
        return found
    }

    /// Is a SensorDyme backend answering at this address?
    ///
    /// Asks /v1/health rather than merely opening a socket. Something else on the
    /// network can be listening on 8787, and "a TCP connection succeeded" is a
    /// much weaker claim than "this is the server" — checking the identifying
    /// field is what stops the app from latching onto a stranger's web server.
    static func answers(_ base: String) async -> Bool {
        guard let url = URL(string: base)?.appendingPathComponent("v1/health") else { return false }
        var request = URLRequest(url: url)
        request.timeoutInterval = 5
        request.cachePolicy = .reloadIgnoringLocalCacheData
        guard let (data, response) = try? await URLSession.shared.data(for: request),
              (response as? HTTPURLResponse)?.statusCode == 200
        else { return false }
        let payload = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        return payload?["service"] as? String == "sensordyme-cloud"
    }

    // MARK: - Bonjour

    /// Every address currently advertising itself as a SensorDyme backend.
    func discover(timeout: TimeInterval = 5) async -> [String] {
        startBrowsing()
        defer { stopBrowsing() }
        let deadline = Date().addingTimeInterval(timeout)
        // Poll rather than wait out the whole timeout: on a quiet home network a
        // Bonjour answer arrives in well under a second, and sitting on the
        // sign-in screen for five seconds after the server already replied is
        // dead time the user reads as the app being broken.
        while Date() < deadline {
            try? await Task.sleep(nanoseconds: 200_000_000)
            if !candidates.isEmpty { break }
        }
        // IPv4 first. A link-local IPv6 address is only usable with the interface
        // scope attached, which a URL cannot carry, so those are last-resort.
        return candidates.sorted { first, second in
            !first.contains("[") && second.contains("[")
        }
    }

    private func startBrowsing() {
        stopBrowsing()
        candidates = []
        problem = nil
        isSearching = true
        let browser = NWBrowser(
            for: .bonjour(type: Self.serviceType, domain: nil),
            using: NWParameters()
        )
        self.browser = browser
        browser.browseResultsChangedHandler = { [weak self] results, _ in
            Task { @MainActor in
                guard let self else { return }
                for result in results {
                    self.resolve(result.endpoint)
                }
            }
        }
        // NWBrowser reports its own failures only here. Without this handler a
        // denied Local Network permission — which iOS prompts for, and which
        // people do decline — makes the browser fail while the results handler
        // simply never fires, and the app blames the network for ever.
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
        isSearching = false
    }

    /// Bonjour hands back a service endpoint; opening a throwaway connection to
    /// it is the supported way to learn the concrete ip:port behind it.
    private func resolve(_ endpoint: NWEndpoint) {
        let connection = NWConnection(to: endpoint, using: .tcp)
        resolvers.append(connection)
        connection.stateUpdateHandler = { [weak self, weak connection] state in
            guard case .ready = state,
                  let remote = connection?.currentPath?.remoteEndpoint,
                  case let .hostPort(host, port) = remote
            else { return }
            guard let address = Self.urlHost(from: host) else { return }
            let url = "http://\(address):\(port.rawValue)"
            Task { @MainActor in
                guard let self else { return }
                if !self.candidates.contains(url) { self.candidates.append(url) }
                connection?.cancel()
            }
        }
        connection.start(queue: .main)
    }

    /// The host as it must appear in a URL.
    ///
    /// The interface scope ("%en0") that NWEndpoint appends has to come off, and
    /// for IPv6 it has to come off *before* the brackets go on — stripping at the
    /// "%" after bracketing leaves "[fe80::1", which is not a URL at all.
    /// `nonisolated` because it is called from the connection's state handler,
    /// which is not on the main actor. It is pure string work and touches nothing
    /// this class owns, so it has no business hopping actors to run.
    nonisolated private static func urlHost(from host: NWEndpoint.Host) -> String? {
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

    /// Only the cases we can genuinely identify: a confident wrong explanation is
    /// worse than the raw error, because it sends someone to fix the wrong thing.
    private static func explain(_ error: NWError) -> String {
        if case let .dns(code) = error {
            if code == -65570 || code == kDNSServiceErr_PolicyDenied {
                return "SensorDyme needs Local Network permission to find your server. "
                     + "Settings → SensorDyme → Local Network."
            }
            return "Couldn't search this network (DNS error \(code))."
        }
        return "Couldn't search this network: \(error.localizedDescription)"
    }
}
