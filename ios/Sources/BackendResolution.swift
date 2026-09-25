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
}
