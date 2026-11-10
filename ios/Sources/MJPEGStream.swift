import Foundation

// The pure, testable half of the live feed: what state the stream is in, how to
// frame JPEGs out of a byte stream, how long to wait before retrying, and what
// counts as a usable device address.
//
// Deliberately Foundation-only and free of SwiftUI/UIKit so ios/Tests/run.sh can
// compile and run it directly. The half that cannot be tested that way -- the
// URLSession plumbing -- lives in FeedViews.swift and is kept as thin as
// possible precisely so that this file holds the parts worth pinning down.

// MARK: - What the viewer is actually looking at

/// The live feed's real state, at the granularity the person watching cares about.
///
/// This exists because the feed previously had exactly two observable states:
/// "showing a picture" and "reconnecting…". Everything that can actually go
/// wrong -- a rejected device token, an address that is not an address, a device
/// that answers but whose camera is not producing frames -- looked identical to
/// a Wi-Fi blip, and all of them were presented as something that would heal on
/// its own. Two of them never heal on their own.
enum MJPEGStreamState: Equatable {
    /// Not started, or deliberately stopped.
    case idle
    /// Opening the connection for the first time.
    case connecting
    /// Frames are arriving and decoding.
    case streaming
    /// The device answered 200 but no frame has decoded yet. Distinct from
    /// `connecting` on purpose: the network is fine and the device is up, so
    /// pointing the user at their Wi-Fi would send them to fix the wrong thing.
    case waitingForFrames
    /// A drop we expect to recover from, on its own, without anyone doing anything.
    ///
    /// `reason` carries what actually went wrong. Without it the badge said
    /// "reconnecting…" for every possible transport failure -- a device that has
    /// moved to a new DHCP address, a phone with Local Network access switched
    /// off, and a genuine Wi-Fi blip all looked identical, and only one of those
    /// heals by waiting.
    case reconnecting(attempt: Int, reason: String?)
    /// The device refused the token. Retrying cannot fix this, so we do not.
    ///
    /// `hasToken` separates the two cases, which need opposite actions: with no
    /// token stored the user must add one, and with a token stored the one they
    /// have is wrong. Reporting "rejected the token" to someone who never
    /// entered a token sends them to check a setting that is empty.
    case unauthorized(hasToken: Bool)
    /// The configured address is not usable as a URL. Also unfixable by retrying.
    case unusableAddress

    /// Whether the app should keep retrying by itself.
    ///
    /// The two terminal cases need a human to change a setting; looping on them
    /// would spin the radio, hammer the device, and tell the user "reconnecting…"
    /// forever about something that is never going to connect.
    var isRecoverableAutomatically: Bool {
        switch self {
        case .unauthorized, .unusableAddress: false
        default: true
        }
    }

    /// True while a picture may still be on screen and correct.
    var isShowingLiveVideo: Bool { self == .streaming }

    /// What to show while a connection attempt is actually in flight.
    ///
    /// This exists because "connecting…" was shown for every attempt, including
    /// the twentieth, which made a fully diagnosed failure look like an
    /// indefinite connect. The arithmetic is the whole problem: a stale device
    /// address on a network that blackholes it takes the full request timeout
    /// (~16s) to fail, while the `reconnecting` state that carries the REASON is
    /// only on screen for the retry delay (0.5s at first, 8s at the cap). So the
    /// badge read "connecting…" for 70-97% of the time and the one state that
    /// said *why* flashed past. The app knew the answer -- "the device didn't
    /// answer in time" -- and threw it away every 0.5 seconds.
    ///
    /// An attempt after a failure is a *reconnection* attempt, and the reason the
    /// last one failed is still the best information available, so it stays on
    /// screen for the duration rather than being replaced by a word that implies
    /// nothing is known yet.
    ///
    /// `hasImage` keeps the previous behaviour where a picture already on screen
    /// is never covered by a badge: a brief drop mid-stream should not blank the
    /// last good frame or shout about it.
    static func inFlight(
        hasImage: Bool,
        current: MJPEGStreamState,
        attempt: Int,
        reason: String?
    ) -> MJPEGStreamState {
        if hasImage { return current }
        // A genuine first attempt, with nothing yet known, is the only case where
        // "connecting…" is the honest word.
        if attempt <= 0, reason == nil { return .connecting }
        return .reconnecting(attempt: max(attempt, 1), reason: reason)
    }

    /// What to put in front of the user. Nil means "say nothing" -- a healthy
    /// stream must not carry a status badge over the video.
    var message: String? {
        // Explicit `return` in every branch: one branch needs a statement, and a
        // switch that mixes a `return` with implicit-return expressions silently
        // stops returning anything from the others.
        switch self {
        case .idle: return nil
        case .streaming: return nil
        case .connecting: return "connecting…"
        case .waitingForFrames: return "connected — waiting for the camera"
        case .reconnecting(let attempt, let reason):
            let head = attempt <= 1 ? "reconnecting…" : "reconnecting… (attempt \(attempt))"
            return reason.map { "\(head) — \($0)" } ?? head
        case .unauthorized(let hasToken):
            return hasToken
                ? "device rejected the token — check Settings → Device token"
                : "this device requires a token — add it in Settings → Device token"
        case .unusableAddress:
            return "device address isn't a valid URL — check Settings → Live feed"
        }
    }
}

// MARK: - Framing JPEGs out of a multipart byte stream

/// Extracts complete JPEGs from an `multipart/x-mixed-replace` byte stream.
///
/// Kept as a value type with no I/O so the framing rules can be tested against
/// awkward inputs -- split markers, leading boundary headers, garbage, a frame
/// larger than the cap -- which is where MJPEG parsers actually go wrong.
struct MJPEGFrameParser {
    /// Start of Image / End of Image markers.
    private static let soi = Data([0xFF, 0xD8])
    private static let eoi = Data([0xFF, 0xD9])

    /// Hard ceiling on retained bytes. A stream that never yields a complete
    /// frame -- a wrong endpoint, a truncated encoder, a corrupted link -- must
    /// not grow a buffer until the app is killed for memory.
    let maxBufferBytes: Int

    private(set) var buffer = Data()
    /// Bytes thrown away because they could never become a frame. Surfaced so a
    /// caller can tell "nothing is arriving" from "nonsense is arriving".
    private(set) var discardedBytes = 0

    init(maxBufferBytes: Int = 4_000_000) {
        self.maxBufferBytes = max(1, maxBufferBytes)
    }

    /// Feed one chunk; get back every complete JPEG it completed, in order.
    mutating func append(_ data: Data) -> [Data] {
        buffer.append(data)
        var frames: [Data] = []

        while true {
            guard let start = buffer.range(of: Self.soi) else {
                // No frame has begun. Everything held is multipart boundary text
                // or noise, so keep only a marker's worth in case an SOI is split
                // across this chunk boundary -- dropping it whole would lose the
                // first byte of the next frame roughly once per stream.
                trimToTail(keeping: Self.soi.count - 1)
                break
            }
            // Anything before the SOI is boundary/header text, never image data.
            if start.lowerBound > buffer.startIndex {
                discardedBytes += buffer.distance(from: buffer.startIndex, to: start.lowerBound)
                buffer.removeSubrange(..<start.lowerBound)
                continue
            }
            // Search for EOI strictly AFTER the SOI, so a marker can never match itself.
            guard let end = buffer.range(of: Self.eoi, in: start.upperBound..<buffer.endIndex) else {
                break  // frame still in flight
            }
            frames.append(buffer.subdata(in: start.lowerBound..<end.upperBound))
            buffer.removeSubrange(..<end.upperBound)
        }

        enforceBufferCap()
        return frames
    }

    /// Drop everything held. Used when a connection is replaced, so bytes from a
    /// previous socket can never be spliced onto bytes from the new one -- which
    /// would produce a blob that starts in one frame and ends in another, decodes
    /// to nothing, and looks exactly like "the camera is broken".
    mutating func reset() {
        buffer.removeAll(keepingCapacity: true)
    }

    private mutating func trimToTail(keeping count: Int) {
        guard buffer.count > count else { return }
        let dropped = buffer.count - count
        discardedBytes += dropped
        buffer.removeSubrange(..<buffer.index(buffer.startIndex, offsetBy: dropped))
    }

    private mutating func enforceBufferCap() {
        guard buffer.count > maxBufferBytes else { return }
        // Prefer keeping the most recent frame start over keeping the oldest
        // bytes: the newest frame is the only one that can still complete.
        if let lastStart = buffer.range(of: Self.soi, options: .backwards) {
            let dropped = buffer.distance(from: buffer.startIndex, to: lastStart.lowerBound)
            if dropped > 0 {
                discardedBytes += dropped
                buffer.removeSubrange(..<lastStart.lowerBound)
            }
        }
        if buffer.count > maxBufferBytes {
            discardedBytes += buffer.count
            buffer.removeAll(keepingCapacity: true)
        }
    }
}

// MARK: - How long to wait before trying again

/// Capped exponential backoff for reconnect attempts.
///
/// A value type with no clock of its own so the schedule can be asserted rather
/// than observed by waiting.
struct MJPEGRetryPolicy: Equatable {
    var initialDelay: TimeInterval = 0.5
    var maximumDelay: TimeInterval = 8.0
    var multiplier: Double = 2.0

    /// Delay before the attempt numbered `attempt` (0 = the first retry).
    func delay(forAttempt attempt: Int) -> TimeInterval {
        guard attempt > 0 else { return initialDelay }
        // pow() on a large attempt overflows to .infinity; min() would then
        // return the cap, but only by luck. Clamp the exponent instead so the
        // arithmetic stays finite and the intent is visible.
        let exponent = Double(min(attempt, 32))
        let scaled = initialDelay * pow(multiplier, exponent)
        return scaled.isFinite ? min(maximumDelay, scaled) : maximumDelay
    }
}

// MARK: - Turning a configured address into a stream URL

/// Builds the authenticated `/stream` request.
///
/// Exists as one pure function, in one place, so that "does the request carry
/// the device token" is a property of the code rather than of whichever call
/// site happened to construct a URLRequest. Every connection the reader makes --
/// the first one and every reconnect after it -- goes through here, so a retry
/// cannot silently become an unauthenticated request.
enum MJPEGStreamRequest {
    /// The header the device's dashboard checks (core/dashboard.py `_authorized`).
    /// Named once here rather than spelled out at each call site: a typo in this
    /// string is a 401 that looks exactly like a wrong token.
    static let tokenHeader = "X-Device-Token"

    /// Nil when `base` cannot be a stream URL -- see MJPEGStreamURL.
    static func make(base: String, token: String, idleTimeout: TimeInterval = 15) -> URLRequest? {
        guard let url = MJPEGStreamURL.streamURL(forBase: base) else { return nil }
        var request = URLRequest(url: url)
        // An IDLE timeout, not a total one: the stream is unbounded by design,
        // so this is the "the device stopped sending" detector.
        request.timeoutInterval = idleTimeout
        request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        // Trimmed here as well as at the point of storage. The device compares
        // byte for byte (hmac.compare_digest), so one trailing newline from a
        // paste makes the CORRECT secret return 401 -- and a value written by an
        // older build never passed through the trimming setter.
        let cleaned = token.trimmingCharacters(in: .whitespacesAndNewlines)
        if !cleaned.isEmpty {
            request.setValue(cleaned, forHTTPHeaderField: tokenHeader)
        }
        return request
    }
}

/// Turns a transport failure into something the person watching can act on.
///
/// Every one of these used to be rendered as "reconnecting…", which is a promise
/// that waiting will fix it. For a device that has taken a new DHCP address, or
/// a phone whose Local Network permission is off, waiting never fixes it.
enum MJPEGStreamDiagnosis {
    static func describe(_ error: Error?) -> String? {
        guard let error else { return nil }
        let ns = error as NSError
        guard ns.domain == NSURLErrorDomain else { return ns.localizedDescription }
        switch ns.code {
        case NSURLErrorCancelled:
            return nil  // we cancelled it ourselves; not a fault worth reporting
        case NSURLErrorCannotConnectToHost:
            // Deliberately naming both plausible causes rather than picking one.
            // iOS reports a refused connection and a blocked-by-Local-Network
            // connection with the same code, and a confident wrong explanation
            // sends someone to fix the wrong thing.
            return "can't reach the device — it may have a new address on this network, or Local Network permission may be off for ORYN EDGE in iOS Settings"
        case NSURLErrorCannotFindHost:
            return "that address doesn't resolve on this network"
        case NSURLErrorTimedOut:
            return "the device didn't answer in time"
        case NSURLErrorNetworkConnectionLost:
            return "the connection dropped"
        case NSURLErrorNotConnectedToInternet:
            return "this phone isn't on a network"
        case NSURLErrorAppTransportSecurityRequiresSecureConnection:
            return "the build blocks plain HTTP to the device"
        default:
            return ns.localizedDescription
        }
    }
}

enum MJPEGStreamURL {
    /// The `/stream` URL for a device base address, or nil if the address cannot
    /// work as one.
    ///
    /// `URL(string:)` alone is not a validity check: it happily accepts
    /// "192.168.1.20:8899" as a *relative* URL with no scheme and no host, which
    /// then fails at connection time with an opaque error and an endless
    /// "reconnecting…". Someone typing an address into Settings without the
    /// scheme is the ordinary case, not an exotic one, so it is checked here and
    /// reported as the settings problem it is.
    static func streamURL(forBase base: String) -> URL? {
        let trimmed = base.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              let url = URL(string: trimmed),
              let scheme = url.scheme?.lowercased(),
              scheme == "http" || scheme == "https",
              let host = url.host, !host.isEmpty
        else { return nil }
        return url.appendingPathComponent("stream")
    }
}

// MARK: - The reader's decisions

/// What `MJPEGReader` (FeedViews.swift) decides, as pure functions.
///
/// The reader itself needs URLSession and UIKit, so ios/Tests/run.sh cannot
/// compile it. Everything it DECIDES lives here instead — when a start() must
/// reconnect, what a finished connection means, when to re-resolve the address
/// — so MJPEGStreamTests exercises the same code the reader runs rather than a
/// copy of it.
enum MJPEGReaderPolicy {
    /// How long without a single byte before the connection is considered dead.
    /// The device pushes at `dashboard.stream_fps` (12/s by default), so silence
    /// this long means the stream is not coming back on its own.
    static let idleTimeout: TimeInterval = 15
    /// Transport failures in a row before the address is worth re-resolving.
    static let failuresBeforeReresolving = 3
    /// The least time between two re-resolves, so an absent device cannot turn
    /// the retry loop into a discovery loop.
    static let addressRecheckInterval: TimeInterval = 20

    /// Whether `start()` has to tear the connection down and open a new one.
    ///
    /// An unchanged target must not restart a healthy stream: `onChange` and
    /// `onAppear` can both fire for the same values, and tearing the connection
    /// down to rebuild it identically is a visible stutter.
    static func needsRestart(
        stopped: Bool, state: MJPEGStreamState,
        currentBase: String, currentToken: String,
        base: String, token: String
    ) -> Bool {
        stopped || state == .idle || base != currentBase || token != currentToken
    }

    /// What a finished connection means.
    enum Completion: Equatable {
        /// The device refused the token. Not retried: no amount of waiting
        /// supplies a token.
        case unauthorized(hasToken: Bool)
        /// Try again. `countsAsTransportFailure` is false when the device DID
        /// answer: re-resolving its address cannot help with a camera that
        /// stopped producing frames.
        case reconnect(reason: String?, countsAsTransportFailure: Bool)
    }

    static let cameraStalledReason = "connected, but the camera stopped sending frames"

    /// - Parameters:
    ///   - status: the HTTP status of the finished task's response, nil when
    ///     nothing answered.
    static func completion(status: Int?, error: Error?, hasToken: Bool) -> Completion {
        if status == 401 || status == 403 { return .unauthorized(hasToken: hasToken) }
        if status == 200 {
            // The device answered and then went quiet. Calling that "the device
            // didn't answer in time" sent people to their Wi-Fi over a camera
            // that had stopped; it is also no reason to look for a new address.
            let ns = error.map { $0 as NSError }
            if ns?.domain == NSURLErrorDomain, ns?.code == NSURLErrorTimedOut {
                return .reconnect(reason: cameraStalledReason, countsAsTransportFailure: false)
            }
            return .reconnect(reason: MJPEGStreamDiagnosis.describe(error), countsAsTransportFailure: false)
        }
        return .reconnect(reason: MJPEGStreamDiagnosis.describe(error), countsAsTransportFailure: true)
    }

    /// Whether repeated failures justify asking discovery for a fresh address.
    static func shouldReresolve(consecutiveFailures: Int, sinceLastRecheck: TimeInterval) -> Bool {
        consecutiveFailures >= failuresBeforeReresolving && sinceLastRecheck >= addressRecheckInterval
    }
}

/// The one connection whose callbacks may act.
///
/// URLSession delivers `didCompleteWithError` for a task the reader cancelled
/// itself, asynchronously, after a replacement has opened. Read as "the stream
/// dropped", that stale callback tore down the healthy replacement, whose own
/// cancellation then did the same to the next — sessions multiplied and the
/// "reconnecting…" badge never cleared. Every callback is checked here first.
final class CurrentConnection<Connection: AnyObject> {
    private(set) var current: Connection?

    func replace(with connection: Connection) { current = connection }
    func clear() { current = nil }
    func isCurrent(_ connection: Connection) -> Bool { current === connection }
}
