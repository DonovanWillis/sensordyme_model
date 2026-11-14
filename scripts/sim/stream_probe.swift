// The app's live-feed LOGIC against the virtual device, on macOS (M8.3).
//
// Concatenated after ios/Sources/StreamLiveness.swift and MJPEGStream.swift by
// scripts/sim/stream_probe.sh and run with `swift`: the exact parser, liveness
// tracker, request builder and shared-connection coordinator the iOS app runs,
// over a real URLSession to the real dashboard (SIMULATED camera). What it
// cannot cover is UIKit decoding and SwiftUI -- see docs/VIDEO_TRANSPORT.md.
//
// Usage: stream_probe.sh BASE_URL TOKEN OUT_JSON

final class ProbeTransport: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    var onEvent: (@Sendable (Event) -> Void)?
    enum Event { case response(MJPEGResponse), data(Data), done(Error?) }
    private var session: URLSession?
    private(set) var opens = 0
    var isOpen: Bool { session != nil }

    func open(base: String, token: String) {
        guard session == nil, let request = MJPEGStreamRequest.make(base: base, token: token) else { return }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.httpMaximumConnectionsPerHost = 1
        configuration.timeoutIntervalForRequest = MJPEGReaderPolicy.idleTimeout
        let s = URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
        session = s
        opens += 1
        s.dataTask(with: request).resume()
    }

    func close() {
        session?.invalidateAndCancel()
        session = nil
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                    completionHandler: @escaping @Sendable (URLSession.ResponseDisposition) -> Void) {
        let http = response as? HTTPURLResponse
        completionHandler(.allow)
        // Exactly what MJPEGReader does: URLSession calls this per multipart part.
        var headers: [String: String] = [:]
        for (name, value) in http?.allHeaderFields ?? [:] {
            if let name = name as? String { headers[name] = "\(value)" }
        }
        onEvent?(.response(MJPEGResponse.classify(status: http?.statusCode ?? 0, mimeType: response.mimeType,
                                                  headers: headers)))
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        onEvent?(.data(data))
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        onEvent?(.done(error))
    }
}

func now() -> TimeInterval { ProcessInfo.processInfo.systemUptime }

func api(_ base: String, _ token: String, _ path: String, body: [String: Any]? = nil) async -> [String: Any] {
    var request = URLRequest(url: URL(string: base + path)!)
    request.setValue(token, forHTTPHeaderField: "X-Device-Token")
    if let body {
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)
    }
    guard let (data, _) = try? await URLSession.shared.data(for: request),
          let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return [:] }
    return object
}

@MainActor
final class Probe {
    let base: String, token: String
    let transport = ProbeTransport()
    var parser = MJPEGFrameParser()
    var tracker = FeedLivenessTracker()
    var transportState: MJPEGStreamState = .idle
    var timeline: [[String: Any]] = []
    var lastLiveness: FeedLiveness = .idle
    let t0 = now()
    lazy var coordinator = SharedFeedCoordinator(
        open: { [unowned self] b, t in
            // Like MJPEGReader.start(): an open stream to the same target is left alone.
            if !self.transport.isOpen { self.transportState = .connecting }
            self.transport.open(base: b, token: t)
        },
        close: { [unowned self] in
            self.transport.close()
            self.tracker.connectionClosed(error: false)
            self.transportState = .idle
        },
        after: { delay, work in Task { @MainActor in try? await Task.sleep(for: .seconds(delay)); work() } }
    )

    init(base: String, token: String) {
        self.base = base
        self.token = token
        transport.onEvent = { [weak self] event in Task { @MainActor in self?.handle(event) } }
    }

    func handle(_ event: ProbeTransport.Event) {
        switch event {
        case let .response(.stream(status, capable, streamID)):
            if status == 200 {
                transportState = .waitingForFrames
                tracker.connectionOpened(now: now(), heartbeatCapable: capable, streamID: streamID)
            }
        case let .response(.frameHeader(meta)):
            parser.expectNextFrame(meta)
        case let .response(.heartbeat(meta)):
            tracker.heartbeatReceived(meta, now: now())
        case let .data(data):
            let parts = parser.appendParts(data)
            tracker.noteBytes(data.count, parserHighWater: parser.highWaterBytes)
            for part in parts {
                switch part {
                case let .heartbeat(meta): tracker.heartbeatReceived(meta, now: now())
                case let .frame(_, meta):
                    if tracker.frameReceived(meta, now: now()) != .duplicate { transportState = .streaming }
                }
            }
        case let .done(error):
            tracker.connectionClosed(error: MJPEGStreamDiagnosis.describe(error) != nil)
        }
        observe()
    }

    func observe(note: String? = nil) {
        let l = tracker.liveness(transport: transportState, now: now())
        if l != lastLiveness || note != nil {
            lastLiveness = l
            var row: [String: Any] = ["t": (now() - t0).rounded(toPlaces: 2),
                                      "liveness": "\(l)",
                                      "indicator": FeedIndicator.resolve(l, monitoringStopped: note == "stopped",
                                                                         frameSource: tracker.lastSource)?.label ?? "-",
                                      "source": tracker.lastSource ?? "-"]
            if let note { row["note"] = note }
            timeline.append(row)
            print(row)
        }
    }

    func run(seconds: TimeInterval) async {
        let end = now() + seconds
        while now() < end {
            try? await Task.sleep(for: .seconds(MJPEGReaderPolicy.livenessTick))
            observe()
        }
    }
}

extension Double {
    func rounded(toPlaces p: Int) -> Double { let m = pow(10, Double(p)); return (self * m).rounded() / m }
}

@MainActor
func main() async {
    let args = CommandLine.arguments
    guard args.count >= 4 else { print("usage: BASE TOKEN OUT_JSON"); exit(2) }
    let base = args[1], token = args[2], out = args[3]
    let probe = Probe(base: base, token: token)
    var report: [String: Any] = ["label": "SIMULATOR / virtual device; the app's Foundation-only feed logic on macOS"]

    // `watch N`: hold one view for N seconds and record every liveness change
    // (e.g. against the camera_freeze scenario).
    if args.count >= 6, args[4] == "watch", let seconds = Double(args[5]) {
        _ = probe.coordinator.attach(base: base, token: token)
        await probe.run(seconds: seconds)
        report["timeline"] = probe.timeline
        report["metrics"] = probe.tracker.metrics.summary(now: now())
        let data = try! JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        FileManager.default.createFile(atPath: out, contents: data)
        print(String(decoding: data, as: UTF8.self))
        return
    }

    // 1. inline + full screen share one connection: count it on the DEVICE.
    let inline = probe.coordinator.attach(base: base, token: token)
    let full = probe.coordinator.attach(base: base, token: token)
    await probe.run(seconds: 3)
    let viewersBoth = (await api(base, token, "/api/stream/stats"))["viewers"] as? Int ?? -1
    for _ in 0..<20 {  // rapid view swaps
        probe.coordinator.detach(full)
        let again = probe.coordinator.attach(base: base, token: token)
        probe.coordinator.detach(again)
        _ = probe.coordinator.attach(base: base, token: token)
    }
    await probe.run(seconds: 1)
    let viewersAfterSwaps = (await api(base, token, "/api/stream/stats"))["viewers"] as? Int ?? -1
    report["device_viewers_with_two_views"] = viewersBoth
    report["device_viewers_after_40_view_swaps"] = viewersAfterSwaps
    report["transport_opens"] = probe.transport.opens
    print("device viewers: two views \(viewersBoth), after swaps \(viewersAfterSwaps), opens \(probe.transport.opens)")

    // 2. live, then monitoring stopped (video must stay live), then resumed.
    await probe.run(seconds: 4)
    let live = probe.tracker.metrics
    report["live_fps"] = live.deliveredFPS(now: now())
    report["live_median_age_ms"] = live.medianFrameAgeMs ?? -1
    _ = await api(base, token, "/api/protection", body: ["action": "stop", "confirm": true])
    probe.observe(note: "stopped")
    await probe.run(seconds: 6)
    report["liveness_while_stopped"] = "\(probe.lastLiveness)"
    report["source_while_stopped"] = probe.tracker.lastSource ?? "-"
    _ = await api(base, token, "/api/protection", body: ["action": "resume"])
    probe.observe(note: "resumed")
    await probe.run(seconds: 5)
    report["source_after_resume"] = probe.tracker.lastSource ?? "-"

    // 3. background: suspend closes, resume reopens one connection.
    probe.coordinator.suspend()
    await probe.run(seconds: 2)
    report["device_viewers_backgrounded"] = (await api(base, token, "/api/stream/stats"))["viewers"] as? Int ?? -1
    probe.coordinator.resume()
    await probe.run(seconds: 3)
    report["device_viewers_foreground"] = (await api(base, token, "/api/stream/stats"))["viewers"] as? Int ?? -1

    // 4. everything detached: closed after the grace period.
    probe.coordinator.detach(inline)
    for lease in 1...200 { probe.coordinator.detach(lease) }
    await probe.run(seconds: 2)
    report["device_viewers_all_detached"] = (await api(base, token, "/api/stream/stats"))["viewers"] as? Int ?? -1

    let m = probe.tracker.metrics
    report["metrics"] = ["summary": m.summary(now: now()), "framesAdvanced": m.framesAdvanced,
                         "duplicates": m.duplicateFrames, "stale": m.staleFrames, "heartbeats": m.heartbeats,
                         "streamResets": m.streamResets, "parserHighWaterBytes": m.parserHighWaterBytes,
                         "connectionOpens": m.connectionOpens, "bytes": m.bytesReceived]
    report["timeline"] = probe.timeline
    let data = try! JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    FileManager.default.createFile(atPath: out, contents: data)
    print(String(decoding: data, as: UTF8.self))
}

await main()
