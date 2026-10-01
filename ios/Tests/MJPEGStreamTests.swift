import Foundation

// Standalone, run by ios/Tests/run.sh (which concatenates the subject file with
// this one and runs it as main.swift).
//
// What these pin down is the live feed showing a black screen with a permanent
// "reconnecting…" badge while the device was healthy and streaming. The reader
// could not tell a callback from a URLSession it had deliberately cancelled from
// a genuine stream drop, so every teardown scheduled a reconnect that tore down
// its own replacement. Sessions multiplied, at least one became orphaned and
// impossible to cancel, and the badge -- which was only ever cleared inside the
// "a frame decoded" branch -- stayed on.

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

func jpeg(_ payload: [UInt8]) -> Data { Data([0xFF, 0xD8] + payload + [0xFF, 0xD9]) }
func part(_ frame: Data) -> Data {
    var out = Data("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: \(frame.count)\r\n\r\n".utf8)
    out.append(frame)
    out.append(Data("\r\n".utf8))
    return out
}

// ------------------------------------------------------------------ framing
print("framing JPEGs out of the device's multipart stream:")
do {
    var parser = MJPEGFrameParser()
    let a = jpeg([1, 2, 3]), b = jpeg([4, 5, 6, 7])
    var stream = Data("HTTP-ish boundary preamble".utf8)
    stream.append(part(a)); stream.append(part(b))
    let frames = parser.append(stream)
    check(frames.count == 2, "two complete frames are recovered from one chunk")
    check(frames.first == a && frames.last == b, "frames come back byte-for-byte, in order")
    check(parser.discardedBytes > 0, "the multipart boundary text is discarded, not treated as image data")
}

do {
    // The real delivery shape: URLSession hands over arbitrary slices.
    var parser = MJPEGFrameParser()
    let a = jpeg(Array(repeating: 0x41, count: 500))
    let whole = part(a)
    var recovered: [Data] = []
    for i in stride(from: 0, to: whole.count, by: 7) {
        recovered += parser.append(whole.subdata(in: i..<min(i + 7, whole.count)))
    }
    check(recovered == [a], "a frame split across many small chunks is reassembled exactly")
}

do {
    var parser = MJPEGFrameParser()
    // A marker split precisely across a chunk boundary is the classic MJPEG bug.
    let a = jpeg([9, 9, 9])
    let whole = part(a)
    // The part is header + frame + "\r\n", so the EOI sits at count-4/count-3.
    // Cut between its two bytes -- the split that actually matters.
    let cut = whole.count - 3
    var frames = parser.append(whole.subdata(in: 0..<cut))
    check(frames.isEmpty, "an incomplete frame yields nothing yet")
    frames += parser.append(whole.subdata(in: cut..<whole.count))
    check(frames == [a], "a frame completed by the next chunk is still recovered")
}

do {
    var parser = MJPEGFrameParser()
    _ = parser.append(Data([0xFF]))  // an SOI split across the chunk boundary
    let frames = parser.append(Data([0xD8, 0x11, 0xFF, 0xD9]))
    check(frames == [Data([0xFF, 0xD8, 0x11, 0xFF, 0xD9])],
          "an SOI split across two chunks is not lost")
}

do {
    var parser = MJPEGFrameParser(maxBufferBytes: 1000)
    // A stream that never completes a frame must not grow without bound.
    for _ in 0..<50 { _ = parser.append(Data(repeating: 0x00, count: 100)) }
    check(parser.buffer.count <= 1000, "a stream that never yields a frame cannot grow past the cap")
    check(parser.discardedBytes > 0, "and it says how much it threw away")
}

do {
    var parser = MJPEGFrameParser(maxBufferBytes: 200)
    _ = parser.append(Data([0xFF, 0xD8]) + Data(repeating: 0x55, count: 400))
    let tail = jpeg([7])
    let frames = parser.append(tail)
    check(frames == [tail],
          "after an oversized partial frame is dropped, the NEXT frame still decodes")
}

do {
    // The failure mode two concurrent connections produce: bytes from socket A
    // spliced onto bytes from socket B. reset() on reconnect is what prevents it.
    var parser = MJPEGFrameParser()
    _ = parser.append(Data([0xFF, 0xD8]) + Data(repeating: 0xAA, count: 50))  // half of A's frame
    parser.reset()
    let b = jpeg([3, 3])
    check(parser.append(b) == [b], "reset() drops a half-frame so a new connection cannot splice onto it")
}

// ------------------------------------------------------------------ backoff
print("\nreconnect backoff — quick at first, capped, never runaway:")
do {
    let policy = MJPEGRetryPolicy()
    check(policy.delay(forAttempt: 0) == 0.5, "the first retry is immediate enough to hide a Wi-Fi blip")
    check(policy.delay(forAttempt: 1) == 1.0, "then it backs off")
    check(policy.delay(forAttempt: 2) == 2.0, "exponentially")
    check(policy.delay(forAttempt: 8) == 8.0, "and is capped so a long outage does not become a long silence")
    check(policy.delay(forAttempt: 100_000) == 8.0,
          "a huge attempt count stays finite (pow() overflows to infinity if unclamped)")
    check(policy.delay(forAttempt: 100_000).isFinite, "the delay is always a real number")
}

// ------------------------------------------------------------------ states
print("\nconnection states say what is actually wrong:")
do {
    check(MJPEGStreamState.streaming.message == nil, "a healthy stream shows no badge over the video")
    check(MJPEGStreamState.idle.message == nil, "a stopped reader says nothing")
    check(MJPEGStreamState.waitingForFrames.message?.contains("camera") == true,
          "a device that answers but sends no frames blames the camera, not the network")
    check(MJPEGStreamState.reconnecting(attempt: 1, reason: nil).message == "reconnecting…",
          "the first retry is not made to sound alarming")
    check(MJPEGStreamState.reconnecting(attempt: 4, reason: nil).message?.contains("attempt 4") == true,
          "a repeated retry admits how many times it has tried")
    check(MJPEGStreamState.unauthorized(hasToken: true).message?.contains("Settings") == true,
          "a rejected token names the setting that fixes it")
    check(MJPEGStreamState.unusableAddress.message?.contains("Settings") == true,
          "an unusable address names the setting that fixes it")
}

print("\nonly states that can heal on their own are retried:")
do {
    check(MJPEGStreamState.reconnecting(attempt: 3, reason: nil).isRecoverableAutomatically,
          "a dropped connection keeps retrying")
    check(MJPEGStreamState.waitingForFrames.isRecoverableAutomatically,
          "a stalled camera keeps retrying — it may come back")
    check(!MJPEGStreamState.unauthorized(hasToken: true).isRecoverableAutomatically,
          "a rejected token does NOT retry: no amount of waiting supplies a token")
    check(!MJPEGStreamState.unusableAddress.isRecoverableAutomatically,
          "an invalid address does NOT retry: it needs a human to correct it")
    check(MJPEGStreamState.streaming.isShowingLiveVideo, "only streaming counts as live video")
    check(!MJPEGStreamState.reconnecting(attempt: 1, reason: nil).isShowingLiveVideo,
          "reconnecting is not live video, even while the last frame is still on screen")
}

// ------------------------------------------------------------------ address
print("\nthe device address is validated before it can become an endless retry:")
do {
    check(MJPEGStreamURL.streamURL(forBase: "http://192.168.1.20:8899")?.absoluteString
          == "http://192.168.1.20:8899/stream",
          "a normal LAN address gets /stream appended")
    check(MJPEGStreamURL.streamURL(forBase: "  http://192.168.1.20:8899  ")?.absoluteString
          == "http://192.168.1.20:8899/stream",
          "surrounding whitespace from a text field is tolerated")
    check(MJPEGStreamURL.streamURL(forBase: "https://device.local:8899")?.absoluteString
          == "https://device.local:8899/stream",
          "https is accepted too")
    check(MJPEGStreamURL.streamURL(forBase: "192.168.1.20:8899") == nil,
          "an address with no scheme is rejected — URL(string:) accepts it as a relative URL, "
          + "which then fails at connect time and looks like a broken camera")
    check(MJPEGStreamURL.streamURL(forBase: "") == nil, "an empty address is not an address")
    check(MJPEGStreamURL.streamURL(forBase: "   ") == nil, "whitespace is not an address")
    check(MJPEGStreamURL.streamURL(forBase: "ftp://192.168.1.20") == nil,
          "a scheme the stream cannot use is rejected")
    check(MJPEGStreamURL.streamURL(forBase: "http://") == nil, "a scheme with no host is rejected")
}


// ------------------------------------------------------- authenticated request
// The device gates /stream on X-Device-Token (core/dashboard.py `_authorized`).
// These pin down that the header is on the request the reader actually sends --
// the first one and, because every connect goes through this same builder, every
// reconnect after it.
print("\nthe /stream request carries the device token:")
do {
    let request = MJPEGStreamRequest.make(base: "http://11.33.0.129:8899", token: "pi-secret")
    check(request?.url?.absoluteString == "http://11.33.0.129:8899/stream", "it targets /stream on the device")
    check(request?.value(forHTTPHeaderField: "X-Device-Token") == "pi-secret",
          "X-Device-Token is present, with the token verbatim")
    check(MJPEGStreamRequest.tokenHeader == "X-Device-Token",
          "the header name matches the one the device checks, spelled once")
}

do {
    // A reconnect is not a different code path -- it rebuilds through the same
    // function with the same stored token. Building twice must be identical.
    let first = MJPEGStreamRequest.make(base: "http://11.33.0.129:8899", token: "pi-secret")
    let retry = MJPEGStreamRequest.make(base: "http://11.33.0.129:8899", token: "pi-secret")
    check(retry?.value(forHTTPHeaderField: "X-Device-Token") == "pi-secret",
          "a reconnect request carries the token too — it cannot silently become anonymous")
    check(first?.value(forHTTPHeaderField: "X-Device-Token")
          == retry?.value(forHTTPHeaderField: "X-Device-Token"),
          "the retry's credentials are identical to the first attempt's")
}

do {
    let pasted = MJPEGStreamRequest.make(base: "http://11.33.0.129:8899", token: "  pi-secret\n")
    check(pasted?.value(forHTTPHeaderField: "X-Device-Token") == "pi-secret",
          "whitespace from a paste is stripped — the device compares byte for byte, "
          + "so a trailing newline makes the CORRECT token return 401")

    let none = MJPEGStreamRequest.make(base: "http://11.33.0.129:8899", token: "")
    check(none != nil, "a device with no token configured still gets a request")
    check(none?.value(forHTTPHeaderField: "X-Device-Token") == nil,
          "and no empty header is sent, which the device would reject")

    let blank = MJPEGStreamRequest.make(base: "http://11.33.0.129:8899", token: "   ")
    check(blank?.value(forHTTPHeaderField: "X-Device-Token") == nil,
          "a whitespace-only token is treated as no token, not as a token of spaces")

    check(MJPEGStreamRequest.make(base: "11.33.0.129:8899", token: "pi-secret") == nil,
          "an address with no scheme yields no request at all, rather than an unauthenticated one")
}

do {
    let request = MJPEGStreamRequest.make(base: "http://11.33.0.129:8899", token: "t", idleTimeout: 15)
    check(request?.timeoutInterval == 15, "the idle timeout is what detects a device that went quiet")
    check(request?.cachePolicy == .reloadIgnoringLocalAndRemoteCacheData,
          "a live video stream is never served from cache")
}

// ------------------------------------------------------- 401 vs 403 vs a drop
print("\nauthentication failure is not treated as a stream drop:")
do {
    check(!MJPEGStreamState.unauthorized(hasToken: true).isRecoverableAutomatically,
          "a rejected token stops retrying — retrying cannot supply a token")
    check(!MJPEGStreamState.unauthorized(hasToken: false).isRecoverableAutomatically,
          "a device that requires a token the app doesn't have also stops retrying")
    check(MJPEGStreamState.unauthorized(hasToken: false).message?.contains("requires a token") == true,
          "with NO token stored, it asks for one instead of claiming the token was wrong")
    check(MJPEGStreamState.unauthorized(hasToken: true).message?.contains("rejected") == true,
          "with a token stored, it says the device rejected it")
    check(MJPEGStreamState.unauthorized(hasToken: false).message
          != MJPEGStreamState.unauthorized(hasToken: true).message,
          "the two cases need different actions, so they read differently")
}

// ------------------------------------------------------- why it is reconnecting
print("\na reconnect says what went wrong, so waiting is not the only advice:")
do {
    let unreachable = NSError(domain: NSURLErrorDomain, code: NSURLErrorCannotConnectToHost)
    let reason = MJPEGStreamDiagnosis.describe(unreachable)
    check(reason?.contains("new address") == true, "an unreachable host mentions a changed address")
    check(reason?.contains("Local Network") == true,
          "and Local Network permission — iOS reports both with the same code, so both are named")
    check(MJPEGStreamState.reconnecting(attempt: 2, reason: reason).message?.contains("Local Network") == true,
          "the reason reaches the badge instead of a bare 'reconnecting…'")

    check(MJPEGStreamDiagnosis.describe(NSError(domain: NSURLErrorDomain, code: NSURLErrorTimedOut))?
            .contains("didn't answer") == true, "a timeout is reported as the device not answering")
    check(MJPEGStreamDiagnosis.describe(nil) == nil, "no error means nothing to explain")
    check(MJPEGStreamDiagnosis.describe(
            NSError(domain: NSURLErrorDomain, code: NSURLErrorCancelled)) == nil,
          "a cancellation we caused ourselves is not reported as a fault")
}

// ------------------------------- what the badge says while an attempt is open
print("\nan attempt in flight keeps the reason it already knows:")
do {
    // The bug: `connect()` set a bare `.connecting` for EVERY attempt. A stale
    // device address on a network that blackholes it takes the full ~16s request
    // timeout to fail, while the `reconnecting` state carrying the reason is only
    // up for the retry delay (0.5s at first). So the badge read "connecting…" for
    // most of every cycle and the explanation flashed past — indistinguishable,
    // to the person watching, from a connect that never finishes.
    let reason = "the device didn't answer in time"

    check(MJPEGStreamState.inFlight(hasImage: false, current: .idle, attempt: 0, reason: nil)
          == .connecting,
          "a genuine first attempt with nothing known says 'connecting…'")

    check(MJPEGStreamState.inFlight(hasImage: false, current: .connecting, attempt: 1, reason: reason)
          == .reconnecting(attempt: 1, reason: reason),
          "the second attempt keeps the reason instead of reverting to 'connecting…'")
    check(MJPEGStreamState.inFlight(hasImage: false, current: .connecting, attempt: 7, reason: reason)
          == .reconnecting(attempt: 7, reason: reason),
          "and so does the eighth")

    // The symptom, stated as a property: after one failure the badge must never
    // again be the bare word that implies nothing is known.
    check(MJPEGStreamState.inFlight(hasImage: false, current: .connecting, attempt: 1, reason: reason)
          != .connecting,
          "after any failure the badge is never a bare 'connecting…' again")
    check(MJPEGStreamState.inFlight(hasImage: false, current: .connecting, attempt: 1, reason: reason)
          .message?.contains(reason) == true,
          "the reason is actually in the text the user reads")

    check(MJPEGStreamState.inFlight(hasImage: false, current: .idle, attempt: 3, reason: nil)
          == .reconnecting(attempt: 3, reason: nil),
          "a retry whose cause is genuinely unknown still reports itself as a retry")
    check(MJPEGStreamState.inFlight(hasImage: false, current: .idle, attempt: 0, reason: reason)
          == .reconnecting(attempt: 1, reason: reason),
          "a known reason with attempt 0 is a retry, not a first connect")
}

print("\na picture already on screen is never covered by a badge:")
do {
    let reason = "the device didn't answer in time"
    check(MJPEGStreamState.inFlight(hasImage: true, current: .streaming, attempt: 0, reason: nil)
          == .streaming,
          "a healthy stream reconnecting in the background keeps streaming")
    check(MJPEGStreamState.inFlight(hasImage: true, current: .streaming, attempt: 4, reason: reason)
          == .streaming,
          "even on a late attempt, the last good frame is not blanked")
}

print("\nthe terminal states are never turned back into a retry:")
do {
    // These two need a human to change a setting. Re-entering a retry state
    // would turn a one-line settings mistake back into a permanent spinner.
    check(!MJPEGStreamState.unauthorized(hasToken: true).isRecoverableAutomatically,
          "a rejected token is not retried")
    check(!MJPEGStreamState.unusableAddress.isRecoverableAutomatically,
          "nor is an address that cannot be a URL")
    check(MJPEGStreamState.connecting.isRecoverableAutomatically, "but connecting is")
    check(MJPEGStreamState.reconnecting(attempt: 9, reason: "x").isRecoverableAutomatically,
          "and so is reconnecting, however many attempts in")
}

print("\nthe timing that made this invisible, as arithmetic:")
do {
    // Not a tautology: it pins the relationship that produced the symptom, so if
    // the retry cap ever grows past the request timeout, this is the place that
    // says the ratio has changed.
    let requestTimeout = 15.0          // MJPEGReader.idleTimeout
    let policy = MJPEGRetryPolicy()
    check(policy.delay(forAttempt: 0) < requestTimeout,
          "the first retry delay is far shorter than a connection timeout")
    check(policy.maximumDelay <= requestTimeout,
          "even the capped delay does not exceed it — so a bare 'connecting…' would always dominate")
}

print(failures == 0 ? "\nPASSED" : "\n\(failures) FAILURE(S)")
exit(failures == 0 ? 0 : 1)
