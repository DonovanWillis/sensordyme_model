import Foundation

/// Which HTTP redirects a request to the backend may follow.
///
/// `BackendResolution.cleartextPermitted` checks the URL a request is made to,
/// but URLSession follows a redirect by itself, carrying the request's
/// Authorization header and body with it. A server (or anything in the path)
/// answering 307 could send a sign-in to another host, or downgrade it from
/// https to http. So a redirect is followed only when it stays on the same host,
/// never goes from https to http, and its target would itself be allowed
/// (`cleartextPermitted`). Anything else is refused, and the 3xx reply is
/// handled as the answer (an error the app explains), never followed.
///
/// Foundation-only and unit-tested (ios/Tests/RedirectPolicyTests.swift).
enum RedirectPolicy {
    static func allows(from original: URL?, to target: URL?) -> Bool {
        guard let original, let target,
              let fromHost = original.host?.lowercased(), !fromHost.isEmpty,
              let toHost = target.host?.lowercased(), !toHost.isEmpty,
              let fromScheme = original.scheme?.lowercased(),
              let toScheme = target.scheme?.lowercased()
        else { return false }
        guard fromHost == toHost else { return false }
        if fromScheme == "https" && toScheme != "https" { return false }
        return BackendResolution.cleartextPermitted(target.absoluteString)
    }
}

/// Applies `RedirectPolicy` to one request (`URLSession.data(for:delegate:)`).
final class RedirectGuard: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    static let shared = RedirectGuard()

    func urlSession(_ session: URLSession, task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest) async -> URLRequest? {
        RedirectPolicy.allows(from: task.originalRequest?.url, to: request.url) ? request : nil
    }
}
