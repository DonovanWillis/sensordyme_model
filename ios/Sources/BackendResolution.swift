import Foundation

/// Decides *which* backend address the app should use, and keeps a permanent
/// one from ever being replaced by something found on the local Wi-Fi.
///
/// The bug this exists to remove: the app kept one stored address, and both a
/// person typing a URL in Settings and mDNS discovery wrote to it. Discovery
/// only ever produces `http://<ip>:<port>` — an address that is true on exactly
/// one network — so the moment it ran, that LAN address became indistinguishable
/// from a deliberate configuration. Take the phone to a different network and
/// sign-in tries a numeric address nobody chose, times out, and blames Wi-Fi.
///
/// So there are now three sources, in strict precedence:
///
///   1. `configured`        — typed by a person in Settings. Never overwritten.
///   2. `productionDefault` — built in from `SENSORDYME_API_BASE_URL`, read out
///                            of Info.plist. Never overwritten.
///   3. `discovered`        — the mDNS cache. The ONLY slot discovery may write.
///
/// Nothing here writes anything; the storage split is what makes rule 5
/// ("discovery can never overwrite the configured backend") structural rather
/// than a matter of remembering to check. This type only answers questions
/// about the strings, which is what makes it testable without a simulator.
///
/// Foundation only, on purpose: `ios/Tests/run.sh` compiles the subject file
/// together with its test as a plain script, so anything importing SwiftUI or
/// Network could not be covered.
enum BackendResolution {

    // MARK: - Precedence

    /// The address to use right now, or "" when nothing is known yet.
    static func effective(configured: String, productionDefault: String, discovered: String) -> String {
        for candidate in [configured, productionDefault, discovered] {
            let trimmed = candidate.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { return trimmed }
        }
        return ""
    }

    /// The address a person or a build chose, ignoring anything discovered.
    /// This is the value discovery is forbidden to touch.
    static func pinned(configured: String, productionDefault: String) -> String {
        let typed = configured.trimmingCharacters(in: .whitespacesAndNewlines)
        if !typed.isEmpty { return typed }
        return productionDefault.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// May the app go looking on the local network?
    ///
    /// Only when nothing is pinned, or what is pinned is itself LAN-scoped (a
    /// developer pointing at a Mac). A pinned *public* backend is never
    /// abandoned for whatever happens to be on the Wi-Fi: a brief internet
    /// outage would otherwise silently sign someone in against a different
    /// server, which is a far worse outcome than an error message.
    static func mayFallBackToDiscovery(configured: String, productionDefault: String) -> Bool {
        let target = pinned(configured: configured, productionDefault: productionDefault)
        if target.isEmpty { return true }
        return isLANScoped(target)
    }

    // MARK: - Which backend, when several answer

    /// A backend found on the network: where it is, and which one it is.
    ///
    /// `id` comes from the service's TXT record (`cloud/mdns.py`'s `backend_id`)
    /// and is nil for a backend published by a build that predates it.
    struct FoundBackend: Equatable {
        let url: String
        let id: String?
        /// The Bonjour service this address was resolved from. One service is
        /// resolved twice (its stable name and today's address), so this is what
        /// says two URLs are the same backend when it publishes no `id`.
        var service: String? = nil
    }

    /// What to do about the backends currently on the network.
    enum BackendChoice: Equatable {
        /// Use this one.
        case use(FoundBackend)
        /// Several answered, none of them the one we know. Picking for the user
        /// would mean possibly connecting them to somebody else's Mac.
        case ambiguous([FoundBackend])
        /// Nothing usable found.
        case none
    }

    /// Choose among the healthy backends on this network.
    ///
    /// The app used to take whichever answered its health check first. The health
    /// check proves a server is *a* SensorDyme backend, not that it is *yours* —
    /// so on a shared flat, an office or a campus network that is a coin flip
    /// between your Mac and a colleague's. The device side solved this a while
    /// ago (`id=SD-…`, and `DeviceLink` deliberately refuses to adopt a different
    /// camera); this is the same guarantee for the backend.
    ///
    /// The rule, and the reasoning for each branch:
    ///
    ///   * **A remembered id that is present wins.** That is the whole point of
    ///     having an identity.
    ///   * **Exactly one backend, none remembered → adopt it.** A single
    ///     SensorDyme backend on the network is overwhelmingly yours, and this is
    ///     what makes first launch need no setup at all. There is no ambiguity to
    ///     get wrong.
    ///   * **Several, none remembered → refuse.** This is the only genuinely
    ///     dangerous case, and guessing is what must not happen. Reported so the
    ///     app can say which it found rather than silently binding to one.
    ///
    /// Note what is NOT a rule: "remembered one is missing, so take another".
    /// A remembered backend that has gone quiet is usually a Mac that is asleep,
    /// not a Mac that has been replaced — and silently moving someone onto a
    /// different server is a worse failure than telling them theirs is not here.
    static func chooseBackend(
        from found: [FoundBackend],
        remembering remembered: String?
    ) -> BackendChoice {
        guard !found.isEmpty else { return .none }

        // One backend is discovered as several URLs -- its stable name and the
        // address it resolves to today. Counted as two, the "exactly one" rule
        // below never applied, and a fresh install found its only backend
        // "ambiguous" and refused to connect, forever. The first spelling is
        // kept: discovery orders the stable name first.
        var seen: Set<String> = []
        let found = found.filter { backend in
            seen.insert(backend.id ?? backend.service ?? backend.url).inserted
        }

        if let remembered, !remembered.isEmpty {
            if let mine = found.first(where: { $0.id == remembered }) {
                return .use(mine)
            }
            // Ours is not here. Whatever IS here is somebody else's: adopting it
            // would send this account's bearer token to a stranger's server (a
            // token they could replay against ours) and overwrite the id that
            // lets the app find its own backend again when the Mac wakes. A
            // backend reachable at a typed or deployed address never gets here --
            // those are tried, by address, before discovery is consulted.
            return .ambiguous(found)
        }

        if found.count == 1 { return .use(found[0]) }

        // Several. If none carries an id at all we cannot distinguish them even
        // in principle, so this is still ambiguous rather than a free choice.
        return .ambiguous(found)
    }

    // MARK: - What to try, in what order

    /// Every address worth trying right now, best first, with the ones that
    /// cannot work from a phone removed.
    ///
    /// `effective` answers "which address is in force"; this answers "what should
    /// the app actually attempt". They differ, and collapsing them is what made
    /// the app unusable: precedence alone means a pinned address that has stopped
    /// existing is tried forever, because being first in the list is a fact about
    /// configuration and says nothing about whether anything answers there. A
    /// build-time `http://11.22.7.64:8787` outranked discovery, so the app kept
    /// dialling a network it had left, and the healthy Mac that discovery had
    /// already found — and written to the discovered slot — was never tried.
    ///
    /// Returning an ordered list instead lets the caller health-check down it and
    /// use the first address that actually answers. That resolves the tension
    /// between the two things this app needs at once: a deployed HTTPS backend is
    /// never abandoned for whatever is on the local Wi-Fi (it answers, so it wins
    /// on the first probe), while a stale LAN address yields immediately to a
    /// server that is really there (it does not answer, so the next candidate is
    /// tried). Neither behaviour has to be configured.
    static func candidates(
        configured: String,
        productionDefault: String,
        discovered: String,
        allowingLoopback: Bool = loopbackIsUsable
    ) -> [String] {
        var ordered: [String] = []
        for candidate in [configured, productionDefault, discovered] {
            let trimmed = candidate.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty,
                  isUsableFromPhone(trimmed, allowingLoopback: allowingLoopback),
                  !ordered.contains(trimmed)
            else { continue }
            ordered.append(trimmed)
        }
        return ordered
    }

    /// Whether loopback counts as a real backend here.
    ///
    /// On a phone it does not: `localhost` is the phone. On the simulator it does
    /// — the simulator shares the Mac's network stack, so loopback genuinely is
    /// the development backend, and filtering it out would break the one setup
    /// where it is correct.
    static let loopbackIsUsable: Bool = {
        #if targetEnvironment(simulator)
        return true
        #else
        return false
        #endif
    }()

    /// Can this app reach this address at all, whatever is configured?
    ///
    /// Loopback is the case that matters and it is not hypothetical. macOS
    /// publishes `127.0.0.1` alongside the real address in the A record for its
    /// own `.local` hostname, so resolving the backend's Bonjour record on a
    /// phone can genuinely yield loopback. On a device that means the phone
    /// itself: trying it wastes a full connection timeout to learn nothing, and
    /// if it is the only candidate the app reports the server as down while the
    /// server is fine.
    ///
    /// `allowingLoopback` is a parameter rather than a compile-time branch inside
    /// the function so both behaviours are testable from one build.
    static func isUsableFromPhone(_ urlString: String, allowingLoopback: Bool = loopbackIsUsable) -> Bool {
        guard let host = host(of: urlString) else { return false }
        if allowingLoopback { return true }
        return !isLoopbackName(host)
    }

    // MARK: - How hard to keep trying

    /// The smallest and largest gap between reconnection attempts.
    ///
    /// A floor because a phone that retries a missing server as fast as it can
    /// is a phone with a flat battery; a ceiling because a backend that comes
    /// back should be noticed in under a minute, not in an hour.
    static let minimumRetryDelay: TimeInterval = 2
    static let maximumRetryDelay: TimeInterval = 60

    /// How long to wait before attempt number `attempt` (0-based).
    ///
    /// Exponential so a server that is genuinely gone is not hammered, capped so
    /// a server that comes back is found again promptly, and jittered so several
    /// phones — or several views in one app — that lost the backend at the same
    /// instant do not retry in lockstep for ever.
    ///
    /// `jitter` is injected rather than drawn here so the schedule is testable;
    /// pass a fresh `Double.random(in: 0...1)` in real use.
    static func retryDelay(attempt: Int, jitter: Double) -> TimeInterval {
        let exponent = max(0, attempt)
        // Cap the exponent before shifting: `1 << 64` is undefined, and an app
        // left disconnected overnight reaches large attempt counts honestly.
        let scale = pow(2.0, Double(min(exponent, 16)))
        let uncapped = minimumRetryDelay * scale
        let base = min(uncapped, maximumRetryDelay)
        // ±25%, clamped so a jitter argument outside 0...1 cannot produce a
        // negative delay (which would busy-loop).
        let spread = 0.25
        let factor = 1 + spread * (2 * min(max(jitter, 0), 1) - 1)
        return max(minimumRetryDelay * (1 - spread), base * factor)
    }

    // MARK: - The build-time production URL

    /// The production backend URL from Info.plist (`SDBackendBaseURL`, fed by
    /// the `SENSORDYME_API_BASE_URL` build setting), or "" when it is not set.
    ///
    /// Three things all mean "not configured" and all have to be caught here,
    /// because each of them would otherwise become a base URL the app earnestly
    /// tries to reach: absent, empty, and — when the build setting is undefined
    /// and Xcode is not doing the substitution — the literal `$(...)` template.
    static func productionDefault(fromInfoPlist raw: String?) -> String {
        let value = (raw ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty, !value.hasPrefix("$(") else { return "" }
        guard let url = URL(string: value),
              let scheme = url.scheme?.lowercased(),
              scheme == "https" || scheme == "http",
              !(url.host ?? "").isEmpty
        else { return "" }
        return value
    }

    // MARK: - Migrating the single old slot

    /// What to do with a value stored under the old single `baseURL` key.
    enum LegacyDisposition: Equatable {
        /// A discovery artifact. Dropping it is what stops an upgraded app from
        /// going on trying to reach an address from a network it has left.
        case discard
        /// Something only a person could have entered. Keep it, as configuration.
        case promoteToConfigured(String)
    }

    /// Sort one legacy `baseURL` into "a decision" or "a stale cache".
    ///
    /// The test is whether the host is a bare IP literal, and it is decisive
    /// rather than a guess: `BackendLocator` builds its result from a resolved
    /// `NWEndpoint.hostPort`, so every address discovery has ever written is
    /// numeric. A name in this slot was typed by a human.
    ///
    /// Deliberately not an RFC 1918 check. The address that prompted this,
    /// `11.33.6.240`, is a *public* range being handed out by a local router —
    /// so "is it a private address" would have looked at the one value this has
    /// to catch and waved it through. "Is it an address at all" cannot.
    ///
    /// `localhost` and `127.0.0.1` are discarded too. They are names, but on a
    /// phone they mean the phone itself, so they are never a usable backend;
    /// the simulator gets its default back from the build configuration.
    static func migrate(legacy: String) -> LegacyDisposition {
        let value = legacy.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty, let host = host(of: value) else { return .discard }
        if isIPLiteral(host) || isLoopbackName(host) { return .discard }
        return .promoteToConfigured(value)
    }

    // MARK: - Host classification

    /// The host of a URL string, lowercased, or nil if it has none.
    /// IPv6 arrives without its brackets, which is what the literal check wants.
    static func host(of urlString: String) -> String? {
        guard let host = URL(string: urlString.trimmingCharacters(in: .whitespacesAndNewlines))?.host,
              !host.isEmpty
        else { return nil }
        return host.lowercased()
    }

    /// Is this address only meaningful on one local network?
    ///
    /// A bare IP, a `.local` mDNS name, or loopback. Used both to decide whether
    /// discovery is allowed to step in, and to decide whether an error message
    /// should say anything about Wi-Fi.
    static func isLANScoped(_ urlStringOrHost: String) -> Bool {
        let host = self.host(of: urlStringOrHost)
            ?? urlStringOrHost.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !host.isEmpty else { return false }
        if isLoopbackName(host) || isIPLiteral(host) { return true }
        return host == "local" || host.hasSuffix(".local") || host.hasSuffix(".local.")
    }

    static func isLoopbackName(_ host: String) -> Bool {
        let name = host.lowercased()
        return name == "localhost" || name.hasSuffix(".localhost") || name.hasPrefix("127.")
    }

    static func isIPLiteral(_ host: String) -> Bool {
        isIPv4Literal(host) || isIPv6Literal(host)
    }

    static func isIPv4Literal(_ host: String) -> Bool {
        let parts = host.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 4 else { return false }
        return parts.allSatisfy { part in
            guard !part.isEmpty, part.count <= 3, part.allSatisfy(\.isNumber) else { return false }
            return (UInt(part) ?? 256) <= 255
        }
    }

    static func isIPv6Literal(_ host: String) -> Bool {
        // URL.host hands back the inside of the brackets, and an interface scope
        // ("%en0") may still be attached.
        let bare = host.hasPrefix("[") && host.hasSuffix("]")
            ? String(host.dropFirst().dropLast())
            : host
        let unscoped = bare.split(separator: "%", maxSplits: 1).first.map(String.init) ?? bare
        guard unscoped.contains(":") else { return false }
        return unscoped.allSatisfy { $0.isHexDigit || $0 == ":" || $0 == "." }
    }

    // MARK: - What to tell the user

    /// The sentence that follows "X didn't answer in time."
    ///
    /// Telling someone to check they are on the same Wi-Fi as the server is good
    /// advice about a Mac on the LAN and actively misleading about a hosted
    /// backend — it sends them to fix a network that is working, and implies the
    /// product requires something it does not.
    static func unreachableAdvice(host: String) -> String {
        if isLANScoped(host) {
            return "That's a local address, so the phone has to be on the same Wi-Fi as the server."
        }
        return "Check the phone's internet connection and try again."
    }

    // MARK: - Where a backend is, versus which backend it is

    /// One backend: which one it is, and where to reach it.
    ///
    /// This type exists because the app kept conflating the two, and the bug that
    /// cost was structural rather than careless. `BackendLocator` resolved the
    /// Bonjour service by opening a connection and reading
    /// `currentPath.remoteEndpoint` — the address the name had *resolved to* —
    /// and cached that. So the backend published a stable identity
    /// (`sensordyme-backend.local`, plus `id=bk_…` in the TXT record,
    /// deliberately, after a ten-day outage caused by a name that could go
    /// stale), and the phone threw the name away and kept the IP. The moment the
    /// Mac got a new address the cache pointed at nothing, and because the cache
    /// was never cleared on failure it stayed the first thing tried for ever.
    ///
    /// `host` is therefore the ADVERTISED host — a name when Bonjour gave one.
    /// An IP belongs in `resolvedAddress`, which is diagnostics and a
    /// short-lived fallback, never the thing that gets written to storage.
    struct BackendEndpoint: Equatable {
        let scheme: String
        let host: String
        let port: Int
        /// From the service TXT record. nil for a backend predating `backend_id`.
        let backendID: String?
        /// What the name resolved to when it was last reached. Never persisted.
        let resolvedAddress: String?

        init(scheme: String = "http", host: String, port: Int,
             backendID: String? = nil, resolvedAddress: String? = nil) {
            self.scheme = scheme.lowercased()
            // A trailing dot is correct in DNS and breaks URL(string:).
            var cleaned = host.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            while cleaned.hasSuffix(".") { cleaned.removeLast() }
            self.host = cleaned
            self.port = port
            self.backendID = backendID
            self.resolvedAddress = resolvedAddress
        }

        /// Parse a stored or typed URL back into an endpoint.
        init?(url: String, backendID: String? = nil) {
            let trimmed = url.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let parsed = URL(string: trimmed),
                  let scheme = parsed.scheme?.lowercased(),
                  scheme == "http" || scheme == "https",
                  let host = parsed.host, !host.isEmpty
            else { return nil }
            self.init(
                scheme: scheme,
                host: host,
                port: parsed.port ?? (scheme == "https" ? 443 : 8787),
                backendID: backendID
            )
        }

        /// The base URL every consumer should build its requests from.
        var baseURL: String {
            let bracketed = host.contains(":") && !host.hasPrefix("[") ? "[\(host)]" : host
            return "\(scheme)://\(bracketed):\(port)"
        }

        /// Whether `host` is a name that can be resolved again later.
        ///
        /// This is the whole distinction the type exists for. A name survives the
        /// machine moving; an address is only true on the network it was read on.
        var hostIsStable: Bool {
            !BackendResolution.isIPLiteral(host) && !BackendResolution.isLoopbackName(host)
        }
    }

    /// May this endpoint be written to storage as where the backend lives?
    ///
    /// Only a name. An address may be used immediately — it is the fastest way to
    /// reach a backend that is right there — but persisting one is what turned a
    /// transient location into a permanent claim about the world.
    static func mayPersistAsLocation(_ endpoint: BackendEndpoint) -> Bool {
        endpoint.hostIsStable
    }

    /// What the discovery cache should hold after an endpoint failed.
    ///
    /// `reportUnreachable` used to clear only the confirmed address, so a dead
    /// numeric candidate stayed in the discovery slot — first in `candidates`,
    /// and the host named in every error the user read. A name is kept: it is
    /// the identity, and the reason it did not answer is usually that the Mac is
    /// asleep, not that the name is wrong. An address is dropped: there is
    /// nothing to re-resolve and no reason to try it again.
    static func discoveryCache(after failed: String, keeping cached: String) -> String {
        let cachedValue = cached.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cachedValue.isEmpty else { return "" }
        guard let endpoint = BackendEndpoint(url: cachedValue) else { return "" }
        if endpoint.hostIsStable { return cachedValue }
        // An address: drop it if it is the thing that just failed, or if the
        // failure was at the same host by another spelling.
        let failedHost = host(of: failed)
        if failedHost == nil || failedHost == endpoint.host { return "" }
        return cachedValue
    }

    /// Choose between endpoints that are the same backend in different spellings.
    ///
    /// Bonjour can report one service on several interfaces, and resolving it
    /// yields both a name and one or more addresses. They are not alternatives to
    /// weigh — the name is strictly better, because it is the only one that still
    /// means something after the address changes.
    static func preferringStableHost(_ endpoints: [BackendEndpoint]) -> [BackendEndpoint] {
        endpoints.enumerated().sorted { left, right in
            if left.element.hostIsStable != right.element.hostIsStable {
                return left.element.hostIsStable
            }
            // IPv4 before IPv6: the dev backend binds 0.0.0.0, so a v6 candidate
            // resolves and then refuses, costing a timeout.
            let leftV6 = isIPv6Literal(left.element.host)
            let rightV6 = isIPv6Literal(right.element.host)
            if leftV6 != rightV6 { return rightV6 }
            return left.offset < right.offset  // otherwise keep discovery order
        }.map(\.element)
    }

    /// Whether a verification that has just finished is still worth acting on.
    ///
    /// The race this closes: candidate A is found, verification is slow, B is
    /// found and verified and becomes active, then A's verification finally
    /// returns and overwrites it. Generations make "newest verified wins" a
    /// check rather than a hope.
    static func verificationIsCurrent(generation: Int, current: Int) -> Bool {
        generation >= current
    }

    // MARK: - Where the endpoint came from, and how hard we have tried

    /// Which mechanism produced the endpoint in force.
    ///
    /// Worth carrying rather than inferring: "the cached address answered" and
    /// "Bonjour found it just now" are the same URL and very different facts
    /// about the network, and the logs were unable to tell them apart.
    enum ResolutionSource: String, Equatable {
        /// The last-known-good endpoint, reused for a fast start.
        case cached
        /// Found on the network this moment.
        case bonjour
        /// Typed by a person in Settings.
        case manual
    }

    /// How many automatic attempts must fail before the app admits defeat and
    /// points the user at Settings.
    ///
    /// The reason this constant exists: the app told the user to go and type an
    /// IP address after the FIRST failed resolve, while exponential retry was
    /// still running happily in the background and would usually succeed
    /// seconds later. So the advice arrived before the recovery it was
    /// pre-empting, and taught people that SensorDyme needs manual setup when
    /// it does not. Three attempts is ~2s + ~4s + ~8s of jittered backoff,
    /// which covers a backend still starting up or a Wi-Fi association that has
    /// not settled.
    static let attemptsBeforeSuggestingManualEntry = 3

    /// What to say about a failed attempt, given how many have failed.
    ///
    /// `permissionProblem` short-circuits everything: a denied Local Network
    /// permission is the one cause no amount of retrying can fix, so it is said
    /// at once rather than after the backoff has run its course.
    static func failureMessage(
        attempts: Int,
        triedHost: String?,
        permissionProblem: String? = nil
    ) -> String {
        if let permissionProblem, !permissionProblem.isEmpty { return permissionProblem }

        let stillTrying = attempts < attemptsBeforeSuggestingManualEntry
        if stillTrying {
            // Deliberately no mention of Settings. Recovery is in progress, and
            // the honest thing to report is that it is in progress.
            if let triedHost, !triedHost.isEmpty {
                return "Reconnecting to SensorDyme — \(triedHost) didn't answer."
            }
            return "Looking for SensorDyme on this Wi-Fi…"
        }

        guard let triedHost, !triedHost.isEmpty else {
            return "No SensorDyme server answered on this Wi-Fi. Check that the backend "
                + "is running on your Mac and that this phone is on the same network. "
                + "You can also enter the address yourself in Settings → Manual server address."
        }
        return "Nothing answered at \(triedHost), and no server was found on this Wi-Fi. "
            + unreachableAdvice(host: triedHost)
            + " You can also enter the address yourself in Settings → Manual server address."
    }

    /// Whether automatic recovery is still worth waiting for, which is what
    /// decides between a "reconnecting" state and a terminal one.
    static func isStillRecovering(attempts: Int) -> Bool {
        attempts < attemptsBeforeSuggestingManualEntry
    }
}
