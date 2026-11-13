// run.sh deps: MJPEGStream
import Foundation

// Standalone, run by ios/Tests/run.sh (StreamLiveness.swift + MJPEGStream.swift
// + this file, concatenated into one main.swift).
//
// What these pin down (M8.3): "a visible frame is not proof the stream is
// live". The badge said LIVE whenever the last frame had decoded, so a stream
// whose device stopped sending -- monitoring stopped, a frozen camera -- kept
// LIVE over a frozen picture, and a reconnect lit it again on the device's
// cached last frame. The tracker decides from the device's evidence (seq,
// capture age, heartbeats) and the coordinator keeps exactly one connection.

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

func jpeg(_ payload: [UInt8]) -> Data { Data([0xFF, 0xD8] + payload + [0xFF, 0xD9]) }

/// A part exactly as core/dashboard.py `_part_header` writes it.
func devicePart(_ frame: Data, seq: Int, ageMs: Int, source: String = "annotated", id: String = "a1b2c3d4") -> Data {
    var out = Data(("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: \(frame.count)\r\n"
        + "X-Frame-Seq: \(seq)\r\nX-Frame-Age-Ms: \(ageMs)\r\nX-Frame-Source: \(source)\r\n"
        + "X-Stream-Id: \(id)\r\n\r\n").utf8)
    out.append(frame)
    out.append(Data("\r\n".utf8))
    return out
}

/// A heartbeat exactly as core/dashboard.py `_heartbeat_part` writes it.
func heartbeat(status: String, cameraAgeMs: Int? = nil, id: String = "a1b2c3d4", seq: Int = 0) -> Data {
    var lines = ["--frame", "Content-Type: text/plain", "Content-Length: 0", "X-Stream-Heartbeat: 1",
                 "X-Stream-Id: \(id)", "X-Stream-Status: \(status)", "X-Frame-Seq: \(seq)"]
    if let cameraAgeMs { lines.append("X-Camera-Age-Ms: \(cameraAgeMs)") }
    return Data((lines.joined(separator: "\r\n") + "\r\n\r\n\r\n").utf8)
}

/// A part as firmware before M8.3 wrote it.
func legacyPart(_ frame: Data) -> Data {
    var out = Data("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: \(frame.count)\r\n\r\n".utf8)
    out.append(frame)
    out.append(Data("\r\n".utf8))
    return out
}

func meta(seq: Int?, ageMs: Int? = 50, id: String? = "a1b2c3d4", source: String? = "annotated") -> MJPEGPartMeta {
    var m = MJPEGPartMeta()
    m.seq = seq; m.ageMs = ageMs; m.streamID = id; m.source = source
    return m
}

// ------------------------------------------------------------------ parsing
print("the parser reads the device's evidence and stays compatible:")
do {
    var parser = MJPEGFrameParser()
    let a = jpeg([1, 2]), b = jpeg([3])
    var stream = devicePart(a, seq: 7, ageMs: 85)
    stream.append(heartbeat(status: "camera_stale", cameraAgeMs: 4200, seq: 7))
    stream.append(devicePart(b, seq: 9, ageMs: 40, source: "camera"))
    let parts = parser.appendParts(stream)
    check(parts.count == 3, "frame, heartbeat, frame are all recovered (got \(parts.count))")
    if parts.count == 3 {
        if case let .frame(data, m) = parts[0] {
            check(data == a && m.seq == 7 && m.ageMs == 85 && m.source == "annotated" && m.streamID == "a1b2c3d4",
                  "a frame arrives with its seq, capture age, source and stream id")
        } else { check(false, "the first part is a frame") }
        if case let .heartbeat(m) = parts[1] {
            check(m.status == "camera_stale" && m.cameraAgeMs == 4200, "a heartbeat says why no frame is coming")
        } else { check(false, "the second part is a heartbeat") }
        if case let .frame(data, m) = parts[2] {
            check(data == b && m.seq == 9 && m.source == "camera", "metadata belongs to ITS frame, not the previous one")
        } else { check(false, "the third part is a frame") }
    }
    check(parser.append(Data()) == [], "the heartbeat is not mistaken for image data")
}

do {
    // URLSession hands over arbitrary slices: headers split mid-line, heartbeats split.
    var parser = MJPEGFrameParser()
    let a = jpeg(Array(repeating: 0x41, count: 300))
    var whole = heartbeat(status: "no_fresh_frames")
    whole.append(devicePart(a, seq: 12, ageMs: 33))
    var parts: [MJPEGPart] = []
    for i in stride(from: 0, to: whole.count, by: 5) {
        parts += parser.appendParts(whole.subdata(in: i..<min(i + 5, whole.count)))
    }
    check(parts.count == 2, "a heartbeat and a frame split into 5-byte chunks are both recovered")
    if parts.count == 2, case let .frame(data, m) = parts[1] {
        check(data == a && m.seq == 12 && m.ageMs == 33, "the frame and its header survive the splitting exactly")
    } else { check(false, "the frame arrives") }
}

do {
    var parser = MJPEGFrameParser()
    let a = jpeg([5])
    let parts = parser.appendParts(legacyPart(a))
    if case let .frame(data, m) = parts.first {
        check(data == a && m.seq == nil && m.ageMs == nil, "firmware before M8.3: frames still decode, with no evidence")
    } else { check(false, "a legacy frame is recovered") }
}

do {
    var parser = MJPEGFrameParser(maxBufferBytes: 1000)
    for _ in 0..<50 { _ = parser.appendParts(Data(repeating: 0x00, count: 100)) }
    check(parser.buffer.count <= 1000, "noise with no header terminator still cannot grow past the cap")
    check(parser.highWaterBytes <= 1100, "the high-water mark is bounded too (cap + one chunk)")
}

print("\nURLSession hands each multipart part over as its own response (observed on the virtual device):")
do {
    // The shapes recorded in m83-evidence/stream/urlsession_multipart_behaviour.txt:
    // the HTTP response, then one NSHTTPURLResponse per part whose headers are the
    // part's own merged with the top-level ones.
    let top = ["Content-Type": "multipart/x-mixed-replace; boundary=frame", "X-Stream-Id": "674d9eec",
               "X-Stream-Heartbeat-Interval-Ms": "1000", "Cache-Control": "no-store"]
    check(MJPEGResponse.classify(status: 200, mimeType: "multipart/x-mixed-replace", headers: top)
          == .stream(status: 200, heartbeatCapable: true, streamID: "674d9eec"),
          "the HTTP response opens the stream, heartbeat-capable, with its id")
    var part = top
    part["Content-Type"] = "image/jpeg"; part["Content-Length"] = "19112"
    part["X-Frame-Seq"] = "4"; part["X-Frame-Age-Ms"] = "94"; part["X-Frame-Source"] = "annotated"
    if case let .frameHeader(m) = MJPEGResponse.classify(status: 200, mimeType: "image/jpeg", headers: part) {
        check(m.seq == 4 && m.ageMs == 94 && m.source == "annotated" && m.streamID == "674d9eec",
              "a JPEG part's response carries the frame's evidence")
    } else { check(false, "a JPEG part is a frame header, NOT a new connection (the probe caught this)") }
    var beat = top
    beat["Content-Type"] = "text/plain"; beat["Content-Length"] = "0"; beat["X-Stream-Heartbeat"] = "1"
    beat["X-Stream-Status"] = "camera_stale"; beat["X-Camera-Age-Ms"] = "6150"
    if case let .heartbeat(m) = MJPEGResponse.classify(status: 200, mimeType: "text/plain", headers: beat) {
        check(m.status == "camera_stale" && m.cameraAgeMs == 6150, "a heartbeat part is a heartbeat")
    } else { check(false, "a heartbeat part is recognised") }
    check(MJPEGResponse.classify(status: 401, mimeType: "application/json", headers: [:])
          == .stream(status: 401, heartbeatCapable: false, streamID: nil), "a refusal is a stream-level answer")
    check(MJPEGResponse.classify(status: 200, mimeType: "multipart/x-mixed-replace",
                                 headers: ["Content-Type": "multipart/x-mixed-replace; boundary=frame"])
          == .stream(status: 200, heartbeatCapable: false, streamID: nil),
          "firmware before M8.3: not heartbeat-capable, no id")

    // The part's headers handed to the parser, then the body alone (no header block in the bytes).
    var parser = MJPEGFrameParser()
    var m = MJPEGPartMeta(); m.seq = 4; m.ageMs = 94
    parser.expectNextFrame(m)
    let parts = parser.appendParts(jpeg([7, 7]))
    if case let .frame(_, got) = parts.first { check(got.seq == 4 && got.ageMs == 94, "the body gets its part's evidence") }
    else { check(false, "the body is a frame") }
    if case let .frame(_, got) = parser.appendParts(jpeg([8])).first {
        check(got.seq == nil, "and the evidence is used once, never carried to the next body")
    } else { check(false, "the next body is a frame") }
}

// ------------------------------------------------------------------ tracker
print("\nLIVE only while fresh frames advance:")
do {
    var t = FeedLivenessTracker()
    check(t.liveness(transport: .idle, now: 0) == .idle, "not watching is idle")
    check(t.liveness(transport: .connecting, now: 0) == .connecting, "opening is connecting")
    t.connectionOpened(now: 10, heartbeatCapable: true)
    check(t.liveness(transport: .waitingForFrames, now: 10.5) == .connecting,
          "connected with no frame yet is not live")
    check(t.frameReceived(meta(seq: 1, ageMs: 80), now: 10.6) == .advanced, "a fresh frame advances")
    check(t.liveness(transport: .streaming, now: 10.7) == .live, "connected and advancing is LIVE")
    t.frameReceived(meta(seq: 2), now: 11.0)
    check(t.liveness(transport: .streaming, now: 12.9) == .live, "still live within the stale window")
    check(t.liveness(transport: .streaming, now: 13.1) == .stalled(seconds: 2),
          "no part for > 2 s: stalled, while the connection is still open")
    check(t.liveness(transport: .streaming, now: 30) == .stalled(seconds: 19), "and it says how stale")
}

print("\na frozen seq is not live, however often it arrives:")
do {
    var t = FeedLivenessTracker()
    t.connectionOpened(now: 0, heartbeatCapable: true)
    t.frameReceived(meta(seq: 5), now: 0.1)
    var verdicts: [FrameVerdict] = []
    for i in 1...30 { verdicts.append(t.frameReceived(meta(seq: 5), now: 0.1 + Double(i) * 0.1)) }
    check(verdicts.allSatisfy { $0 == .duplicate }, "the same seq again is a duplicate")
    check(t.liveness(transport: .streaming, now: 3.1) == .stalled(seconds: 3),
          "parts keep arriving with seq 5, yet it is stalled: nothing new was captured")
    check(t.metrics.duplicateFrames == 30, "duplicates are counted")
}

print("\na frame captured too long ago is not live, however recently it arrived:")
do {
    var t = FeedLivenessTracker()
    t.connectionOpened(now: 0, heartbeatCapable: true)
    check(t.frameReceived(meta(seq: 1, ageMs: 30_000), now: 0.1) == .stale(ageMs: 30_000),
          "a 30 s old picture (the old device's cached frame) is stale")
    check(t.liveness(transport: .streaming, now: 0.2) != .live, "and is never shown as LIVE")
    // The pre-M8.3 rule (MJPEGLiveIndicator: `reader.state == .streaming`) lit
    // LIVE here: the frame decoded, so the transport state was .streaming.
    check(MJPEGStreamState.streaming.isShowingLiveVideo,
          "(the transport alone still says .streaming -- which is exactly why it is not the badge's input)")
    check(t.metrics.staleFrames == 1 && t.metrics.lastFrameAgeMs == 30_000, "and its age is recorded")
    t.frameReceived(meta(seq: 2, ageMs: 2500), now: 0.3)
    check(t.liveness(transport: .streaming, now: 0.4) != .live, "2.5 s old is still not live")
    t.frameReceived(meta(seq: 3, ageMs: 120), now: 0.5)
    check(t.liveness(transport: .streaming, now: 0.6) == .live, "a fresh frame makes it live")
}

print("\nevidence from a previous connection proves nothing about this one:")
do {
    var t = FeedLivenessTracker()
    t.connectionOpened(now: 0, heartbeatCapable: true)
    t.frameReceived(meta(seq: 1), now: 0.1)
    t.connectionClosed(error: true)
    check(t.liveness(transport: .reconnecting(attempt: 1, reason: "the connection dropped"), now: 0.2)
          == .unreachable(reason: "the connection dropped"), "a drop is unreachable, with its reason")
    t.noteReconnect()
    t.connectionOpened(now: 1.0, heartbeatCapable: true)
    check(t.liveness(transport: .waitingForFrames, now: 1.1) == .connecting,
          "after a reconnect it is not live until a new fresh frame arrives")
    check(t.metrics.transportErrors == 1 && t.metrics.reconnects == 1 && t.metrics.connectionOpens == 2,
          "reconnects, errors and opens are counted")
}

print("\na device reboot resets the seq; the tracker follows it:")
do {
    var t = FeedLivenessTracker()
    t.connectionOpened(now: 0, heartbeatCapable: true)
    t.frameReceived(meta(seq: 900, id: "aaaa"), now: 0.1)
    t.connectionClosed(error: true)
    t.connectionOpened(now: 5, heartbeatCapable: true, streamID: "bbbb")
    check(t.frameReceived(meta(seq: 1, id: "bbbb"), now: 5.1) == .advanced,
          "seq 1 from a new stream id advances (it is not 'older than 900')")
    check(t.metrics.streamResets == 1, "the restart is counted once")
    check(t.liveness(transport: .streaming, now: 5.2) == .live, "and the new stream is live")
    // firmware without a stream id that restarts: seq going backwards is a reset, not a stall
    var u = FeedLivenessTracker()
    u.connectionOpened(now: 0, heartbeatCapable: true)
    u.frameReceived(meta(seq: 50, id: nil), now: 0.1)
    check(u.frameReceived(meta(seq: 2, id: nil), now: 0.2) == .advanced, "a backwards seq without an id is a reset")
    check(u.metrics.streamResets == 1, "and counted")
}

print("\nframes the device skipped for a slow phone are counted, not mistaken for a stall:")
do {
    var t = FeedLivenessTracker()
    t.connectionOpened(now: 0, heartbeatCapable: true)
    t.frameReceived(meta(seq: 1), now: 0.1)
    check(t.frameReceived(meta(seq: 6), now: 0.2) == .advanced, "a jump forward advances")
    check(t.metrics.framesSkippedByDevice == 4, "the 4 frames dropped on the device are counted")
}

print("\nthe device says why: camera unavailable is not a network problem:")
do {
    var t = FeedLivenessTracker()
    t.connectionOpened(now: 0, heartbeatCapable: true)
    var hb = MJPEGPartMeta(); hb.isHeartbeat = true; hb.status = "camera_stale"; hb.cameraAgeMs = 6000
    t.heartbeatReceived(hb, now: 1.0)
    check(t.liveness(transport: .waitingForFrames, now: 1.1) == .cameraUnavailable,
          "a camera_stale heartbeat: device reachable, camera unavailable")
    check(!t.shouldForceReconnect(now: 5.9), "heartbeats arriving: the connection is fine, no reconnect")
    hb.status = "no_fresh_frames"
    t.heartbeatReceived(hb, now: 2.0)
    check(t.liveness(transport: .waitingForFrames, now: 2.1) == .stalled(seconds: 2),
          "no_fresh_frames with a working camera is a stall, not a camera fault")
    check(t.metrics.heartbeats == 2, "heartbeats are counted")
}

print("\nsilence from a heartbeat-capable device means a dead connection:")
do {
    var t = FeedLivenessTracker()
    t.connectionOpened(now: 0, heartbeatCapable: true)
    t.frameReceived(meta(seq: 1), now: 0.5)
    check(!t.shouldForceReconnect(now: 5.4), "within the silence window: keep it")
    check(t.shouldForceReconnect(now: 5.6), "nothing at all for > 5 s: replace the connection (network change, reboot)")
    var legacy = FeedLivenessTracker()
    legacy.connectionOpened(now: 0, heartbeatCapable: false)
    check(!legacy.shouldForceReconnect(now: 60),
          "old firmware sends no heartbeats, so its silence is left to the idle timeout")
    t.connectionClosed(error: false)
    check(!t.shouldForceReconnect(now: 100), "a closed connection is never 'reconnected' by the watchdog")
}

print("\nold firmware: one frame proves nothing, two advancing frames do:")
do {
    var t = FeedLivenessTracker()
    t.connectionOpened(now: 0, heartbeatCapable: false)
    check(t.frameReceived(meta(seq: nil, ageMs: nil, id: nil, source: nil), now: 0.1) == .unverified,
          "a frame without metadata is unverified")
    check(t.liveness(transport: .streaming, now: 0.2) != .live,
          "the old device's cached frame alone is NOT live (it was, before M8.3)")
    t.frameReceived(meta(seq: nil, ageMs: nil, id: nil, source: nil), now: 0.3)
    check(t.liveness(transport: .streaming, now: 0.4) == .live, "a second frame within the window is")
    check(t.liveness(transport: .streaming, now: 2.5) == .stalled(seconds: 2), "and it stalls like any other")
}

print("\nterminal transport states pass through:")
do {
    let t = FeedLivenessTracker()
    check(t.liveness(transport: .unauthorized(hasToken: true), now: 0) == .refused, "a refused token")
    check(t.liveness(transport: .unusableAddress, now: 0) == .refused, "an unusable address")
}

// ------------------------------------------------------------------ the badge
print("\nthe badge never says LIVE over stale video, and keeps video and monitoring apart:")
do {
    check(FeedIndicator.resolve(.live, monitoringStopped: false, frameSource: "annotated")
          == FeedIndicator(label: "LIVE", detail: nil, tone: .live, isLive: true), "live and analysed: LIVE")
    let paused = FeedIndicator.resolve(.live, monitoringStopped: true, frameSource: "camera")
    check(paused?.label == "LIVE" && paused?.detail == "Monitoring paused" && paused?.isLive == true,
          "video live while monitoring is stopped: LIVE, and it says monitoring is paused")
    check(FeedIndicator.resolve(.live, monitoringStopped: false, frameSource: "camera")?.detail == "Not analysed",
          "camera frames while monitoring runs (degraded): says the picture is not analysed")
    let stalled = FeedIndicator.resolve(.stalled(seconds: 4), monitoringStopped: false, frameSource: "annotated")
    check(stalled?.label != "LIVE" && stalled?.isLive == false && stalled?.detail == "4 s old",
          "a stalled stream says how old the picture is, never LIVE")
    check(FeedIndicator.resolve(.cameraUnavailable, monitoringStopped: false, frameSource: nil)?.label
          == "CAMERA NOT SENDING", "camera unavailable is named")
    for state: FeedLiveness in [.idle, .connecting, .unreachable(reason: nil), .refused, .disabled,
                                .stalled(seconds: 3), .cameraUnavailable] {
        check(FeedIndicator.resolve(state, monitoringStopped: false, frameSource: "annotated")?.isLive != true,
              "\(state) is never presented as live")
    }
}

// ------------------------------------------------------------------ metrics
print("\nrolling instrumentation:")
do {
    var m = StreamMetrics()
    for i in 0..<100 { m.noteAdvance(at: Double(i) * 0.1, ageMs: 50 + i) }
    check(m.recentAdvances.count == StreamMetrics.window && m.recentAgesMs.count == StreamMetrics.window,
          "the windows are bounded (no growth over a long session)")
    let fps = m.deliveredFPS(now: 9.95)
    check(abs(fps - 10) < 0.5, "delivered fps is measured over the window (got \(fps))")
    check(m.deliveredFPS(now: 60) == 0, "and reads 0 once frames stop")
    check(m.medianFrameAgeMs == 50 + 84, "the median capture age covers the recent window")
    check(m.summary(now: 9.95).contains("fps 10.0"), "and fits on one line")
}

// ------------------------------------------------------------------ one connection
final class FakeTransport {
    var opens = 0
    var closes = 0
    var isOpen = false
    var target: String?
    /// Mirrors MJPEGReader.start(): a no-op for an unchanged target on an open stream.
    func open(_ base: String, _ token: String) {
        let key = base + "|" + token
        if isOpen && target == key { return }
        if isOpen { closes += 1 }
        isOpen = true
        target = key
        opens += 1
    }
    func close() {
        guard isOpen else { return }
        isOpen = false
        closes += 1
    }
}

final class ManualScheduler {
    var pending: [(TimeInterval, @MainActor () -> Void)] = []
    func after(_ delay: TimeInterval, _ work: @escaping @MainActor () -> Void) { pending.append((delay, work)) }
    @MainActor func runAll() { let jobs = pending; pending = []; for job in jobs { job.1() } }
}

@MainActor
func coordinatorTests() {
    print("\nexactly ONE connection, however many views show the feed:")
    do {
        let fake = FakeTransport(), clock = ManualScheduler()
        let c = SharedFeedCoordinator(open: fake.open, close: fake.close, after: clock.after)
        let inline = c.attach(base: "http://d:8899", token: "t")
        let full = c.attach(base: "http://d:8899", token: "t")  // full screen over the inline camera
        check(fake.opens == 1 && c.viewers == 2, "inline + full screen share one open connection")
        c.detach(full)
        clock.runAll()
        check(fake.isOpen && fake.closes == 0, "closing full screen leaves the inline viewer's connection")
        c.detach(inline)
        check(fake.isOpen, "the last viewer leaving does not close at once (grace for a view swap)")
        clock.runAll()
        check(!fake.isOpen && fake.closes == 1, "after the grace period it closes, exactly once")
    }

    do {
        // SwiftUI does not promise the order of appear / disappear across a presentation.
        let fake = FakeTransport(), clock = ManualScheduler()
        let c = SharedFeedCoordinator(open: fake.open, close: fake.close, after: clock.after)
        var inline = c.attach(base: "http://d:8899", token: "t")
        for _ in 0..<100 {
            c.detach(inline)                                         // inline disappears first...
            let full = c.attach(base: "http://d:8899", token: "t")   // ...then full screen appears
            clock.runAll()                                           // the pending stop fires late
            inline = c.attach(base: "http://d:8899", token: "t")     // back: inline appears first...
            c.detach(full)                                           // ...then full screen goes
            clock.runAll()
        }
        check(fake.opens == 1 && fake.closes == 0,
              "200 rapid view swaps in either order: one connection, never dropped (opens \(fake.opens), closes \(fake.closes))")
        check(c.viewers == 1, "and the count is exact")
    }

    do {
        let fake = FakeTransport(), clock = ManualScheduler()
        let c = SharedFeedCoordinator(open: fake.open, close: fake.close, after: clock.after)
        let a = c.attach(base: "http://d:8899", token: "t")
        c.detach(a)
        c.detach(a)  // a stray second detach
        let b = c.attach(base: "http://d:8899", token: "t")
        c.detach(999)  // a lease that never existed
        clock.runAll()
        check(fake.isOpen && c.viewers == 1, "a stray or unknown detach cannot release someone else's viewer")
        c.detach(b)
        clock.runAll()
        check(!fake.isOpen && fake.opens == 1, "and the real last detach still closes")
    }

    print("\nbackground and foreground (scenePhase):")
    do {
        let fake = FakeTransport(), clock = ManualScheduler()
        let c = SharedFeedCoordinator(open: fake.open, close: fake.close, after: clock.after)
        let inline = c.attach(base: "http://d:8899", token: "t")
        _ = c.attach(base: "http://d:8899", token: "t")
        c.suspend(); c.suspend()  // both views see the scene phase change
        check(!fake.isOpen && fake.closes == 1, "backgrounding closes the connection once, whatever the view count")
        c.resume(); c.resume()
        check(fake.isOpen && fake.opens == 2, "foregrounding reopens exactly one connection")
        c.suspend()
        c.detach(inline)
        _ = c.attach(base: "http://d:8899", token: "t")
        check(!fake.isOpen, "a view appearing while backgrounded does not open anything")
        c.resume()
        check(fake.isOpen && fake.opens == 3, "it opens on return")
    }

    print("\nretargeting (new address from discovery, new token):")
    do {
        let fake = FakeTransport(), clock = ManualScheduler()
        let c = SharedFeedCoordinator(open: fake.open, close: fake.close, after: clock.after)
        _ = c.attach(base: "http://d:8899", token: "t")
        c.retarget(base: "http://e:8899", token: "t")
        check(fake.opens == 2 && fake.closes == 1 && fake.target == "http://e:8899|t",
              "a new address replaces the connection (one closed, one opened)")
        c.retarget(base: "http://e:8899", token: "t")
        check(fake.opens == 2, "the same target again is not a reconnect")
    }
}

await MainActor.run { coordinatorTests() }

print(failures == 0 ? "\nPASSED" : "\n\(failures) FAILURE(S)")
exit(failures == 0 ? 0 : 1)
