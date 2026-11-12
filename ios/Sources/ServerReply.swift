import Foundation

/// A server reply that is not the answer the app asked for, put into words a
/// person can act on — never the raw body, never a bare status number, never
/// silence.
///
/// Before this, a non-2xx reply showed its `detail` only when it was a plain
/// string. Everything else read "request failed (502)": a reverse proxy's HTML
/// error page, FastAPI's 422 (whose `detail` is a list), a plain-text
/// "Internal Server Error", an empty body. A 200 carrying a web page — a Wi-Fi
/// sign-in page, another app on the same port — read as "it may be running a
/// different version", which sends someone to update a server that is not the
/// one answering.
///
/// The body is sniffed rather than trusting Content-Type: proxies and captive
/// portals label HTML as anything. Foundation only (ios/Tests/run.sh).
enum ServerReply {
    struct Failure: Equatable {
        /// The banner's headline.
        let title: String
        /// What happened and what to do.
        let message: String
        /// The backend's machine-readable code, when it sent one.
        let code: String?
    }

    enum Title {
        static let unreadable = "The server's reply couldn't be read"
        static let serverError = "The server reported a problem"
        static let refused = "The server refused the request"
        static let notSupported = "Not supported by this server"
        static let wrongServer = "Something else answered at this address"
    }

    /// What the body looks like, from its first bytes.
    enum BodyKind: Equatable { case empty, json, html, text, binary }

    static func kind(of data: Data) -> BodyKind {
        guard let text = String(data: data.prefix(512), encoding: .utf8) else { return .binary }
        let head = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if head.isEmpty { return .empty }
        if head.hasPrefix("{") || head.hasPrefix("[") { return .json }
        if head.hasPrefix("<") { return .html }
        return .text
    }

    /// A 2xx reply this build could not decode.
    static func undecodable(_ data: Data) -> Failure {
        switch kind(of: data) {
        case .empty:
            return Failure(
                title: Title.unreadable,
                message: "The server answered with an empty reply, so there was nothing to read. "
                    + "Pull down to try again; if it keeps happening, check Settings → Server.",
                code: nil)
        case .html:
            return Failure(title: Title.wrongServer, message: wrongServerMessage, code: nil)
        case .json, .text, .binary:
            return Failure(title: Title.unreadable, message: ReplyError.unreadable, code: nil)
        }
    }

    static let wrongServerMessage = "It sent a web page instead of SensorDyme data — usually a Wi-Fi sign-in "
        + "page or another app using the same address. Nothing was changed. Check Settings → Server."

    /// A non-2xx reply (the caller handles a 401 on a signed-in request).
    static func httpFailure(status: Int, data: Data) -> Failure {
        let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        let code = object?["code"] as? String
        let said = object.flatMap(explanation(in:))
        let title: String
        switch status {
        case 500...: title = Title.serverError
        case 404 where code == nil: title = Title.notSupported
        default: title = Title.refused
        }
        if let said { return Failure(title: title, message: said, code: code) }
        return Failure(title: title, message: fallback(status: status, body: kind(of: data)), code: code)
    }

    /// The server's own sentence: FastAPI's `detail` (a string, a list of
    /// validation errors, or an object), or the `error`/`message` other
    /// servers and proxies use. nil when there is nothing readable.
    static func explanation(in object: [String: Any]) -> String? {
        if let text = readable(object["detail"] as? String) { return text }
        if let items = object["detail"] as? [[String: Any]], !items.isEmpty {
            let parts = items.prefix(2).compactMap { item -> String? in
                guard let msg = readable(item["msg"] as? String) else { return nil }
                let field = (item["loc"] as? [Any])?.last.map { "\($0)" }
                return field.map { "\($0): \(msg)" } ?? msg
            }
            if !parts.isEmpty { return "The server couldn't accept that — " + parts.joined(separator: "; ") + "." }
        }
        if let nested = object["detail"] as? [String: Any], let text = explanation(in: nested) { return text }
        for key in ["message", "error", "msg"] {
            if let text = readable(object[key] as? String) { return text }
        }
        return nil
    }

    /// A sentence fit to show: trimmed, bounded, and not itself markup or JSON.
    static func readable(_ text: String?) -> String? {
        guard let trimmed = text?.trimmingCharacters(in: .whitespacesAndNewlines), !trimmed.isEmpty else { return nil }
        if trimmed.hasPrefix("<") || trimmed.hasPrefix("{") || trimmed.hasPrefix("[") { return nil }
        let limit = 280
        return trimmed.count > limit ? String(trimmed.prefix(limit - 1)) + "…" : trimmed
    }

    private static func fallback(status: Int, body: BodyKind) -> String {
        switch status {
        case 502, 503, 504:
            return "The server's gateway couldn't get an answer from SensorDyme (error \(status)). It may be "
                + "restarting — try again in a moment."
        case 500...:
            return "The server hit an internal error (\(status)) and couldn't finish. Try again; if it keeps "
                + "happening, its log has the details."
        case 404:
            return "This server doesn't have that feature (404). It may be running an older version."
        case 403:
            return "The server refused access (403). Sign out and in again, or check this account's permissions."
        case 408:
            return "The server gave up waiting for the request (408). Try again."
        case 413:
            return "That was too large for the server to accept (413)."
        case 429:
            return "The server is busy (too many requests). Wait a moment and try again."
        default:
            return body == .html
                ? "The server refused the request (error \(status)) and sent a web page instead of an explanation."
                : "The server refused the request (error \(status)) without saying why."
        }
    }
}
