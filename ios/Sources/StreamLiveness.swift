import Foundation

// "A visible frame is not proof the stream is live." (M8.3)
//
// The live feed used to say LIVE whenever the last thing that happened was a
// decoded frame: a connection that stayed open while the device stopped
// sending -- monitoring stopped, a frozen camera, a module that could not draw
// -- kept the badge lit over a frozen picture for up to the 15 s idle timeout,
// and a reconnect was handed the device's CACHED last frame, however old, and
// lit it again. This file decides, from evidence, what the person is actually
// looking at, and it is the only place that may say "live".
//
// Foundation-only (ios/Tests/run.sh runs StreamLivenessTests against it).
// The reader (FeedViews.swift MJPEGReader) feeds it; it holds no UI state.

// MARK: - What the viewer is looking at

/// The feed's liveness, decided from evidence (docs/VIDEO_TRANSPORT.md).
enum FeedLiveness: Equatable {
    /// Not watching.
    case idle
    /// Opening a connection, or connected and waiting for the first fresh frame.
    case connecting
    /// Connected, and fresh frames are advancing: the only state that is LIVE.
    case live
    /// Connected, but no fresh frame for `seconds`: the seq stopped advancing,
    /// parts stopped arriving, or what arrives is too old. The picture on
    /// screen (if any) is the past.
    case stalled(seconds: Int)
    /// The device answered and says its camera is not producing frames
    /// (heartbeat `camera_stale` / `camera_not_started`).
    case cameraUnavailable
    /// M8.4 (owner Decision C): monitoring is stopped and the owner has not
    /// turned on camera-only, so the device sends no picture at all (heartbeat
    /// `camera_off_while_stopped`). Deliberate privacy, not a fault: never
    /// "stalled", and the last picture is not kept on screen.
    case cameraOffWhileStopped
    /// The device cannot be reached right now; `reason` says why if known.
    case unreachable(reason: String?)
    /// Needs a person: a refused token, or an address that is not one.
    case refused
    /// There is no live source for this device on this phone (no device on
    /// this network, demo mode, another device's camera).
    case disabled

    var isLive: Bool { self == .live }

    /// Whether the last decoded picture must be removed rather than faded:
    /// the device withheld the camera on purpose, so a picture from before
    /// the stop is neither live nor something to keep showing.
    var hidesLastPicture: Bool { self == .cameraOffWhileStopped }
}

/// Values the device sends about camera access (core/camera_privacy.py).
enum CameraAccess {
    /// Heartbeat `X-Stream-Status` while frames are withheld for privacy.
    static let offWhileStoppedStatus = "camera_off_while_stopped"
    /// `X-Frame-Source` values.
    static let sourceAnnotated = "annotated"
    static let sourceCamera = "camera"
    static let sourceCameraOnly = "camera_only"
    static let sourceCameraLocal = "camera_local"
}

// MARK: - The decision

/// Thresholds. One place, so the device's and the app's numbers can be read
/// side by side (core/dashboard.py DashboardServer constants).
struct FeedLivenessConfig: Equatable {
    /// No fresh frame for this long while connected: stalled. The device
    /// pushes at up to 12/s; the Pi on CPU inference at ~3/s. 2 s is several
    /// missed frames at the slowest healthy cadence.
    var staleAfter: TimeInterval = 2.0
    /// A frame captured longer ago than this is not live, however recently it
    /// arrived. (The device never sends one older than 5 s at all.)
    var maxLiveFrameAgeMs: Int = 2000
    /// Nothing at all -- no frame, no heartbeat -- for this long from a device
    /// that sends heartbeats: the connection is dead (network change, device
    /// rebooted without closing the socket). Reconnect instead of waiting for
    /// the 15 s idle timeout.
    var silenceBeforeReconnect: TimeInterval = 5.0
    /// A heartbeat this old no longer says anything about the camera.
    var heartbeatValidFor: TimeInterval = 3.0
}

/// What one received frame did to the evidence.
enum FrameVerdict: Equatable {
    /// A newer frame, fresh enough: the stream advanced.
    case advanced
    /// The same seq again: nothing new was captured (a frozen source).
    case duplicate
    /// Newer, but captured too long ago to call live.
    case stale(ageMs: Int)
    /// From firmware without metadata: counted by arrival only.
    case unverified
}

/// The liveness state machine. A value type driven entirely by its inputs and
/// an explicit `now`, so every transition is testable without a clock.
struct FeedLivenessTracker {
    var config = FeedLivenessConfig()
    private(set) var metrics = StreamMetrics()

    // Connection
    private(set) var connectedAt: TimeInterval?
    private(set) var heartbeatCapable = false
    private(set) var lastPartAt: TimeInterval?
    // Frames
    private(set) var lastAdvanceAt: TimeInterval?
    private(set) var lastSeq: Int?
    private(set) var streamID: String?
    private(set) var lastSource: String?
    private var previousUnverifiedAt: TimeInterval?
    // Heartbeats
    private(set) var heartbeatStatus: String?
    private var heartbeatAt: TimeInterval?

    init(config: FeedLivenessConfig = FeedLivenessConfig()) {
        self.config = config
    }

    // MARK: Inputs

    /// The device answered 200. `heartbeatCapable`: it advertised heartbeats
    /// (`X-Stream-Heartbeat-Interval-Ms`), so silence from it means something.
    mutating func connectionOpened(now: TimeInterval, heartbeatCapable: Bool, streamID: String? = nil) {
        connectedAt = now
        lastPartAt = now
        self.heartbeatCapable = heartbeatCapable
        lastAdvanceAt = nil  // evidence from a previous connection proves nothing about this one
        previousUnverifiedAt = nil
        heartbeatStatus = nil
        heartbeatAt = nil
        metrics.connectionOpens += 1
        if let streamID { noteStreamID(streamID) }
    }

    mutating func connectionClosed(error: Bool) {
        connectedAt = nil
        lastAdvanceAt = nil
        previousUnverifiedAt = nil
        heartbeatStatus = nil
        heartbeatAt = nil
        if error { metrics.transportErrors += 1 }
    }

    mutating func noteReconnect() { metrics.reconnects += 1 }
    mutating func noteDecodeFailure() { metrics.decodeFailures += 1 }
    mutating func noteBytes(_ count: Int, parserHighWater: Int) {
        metrics.bytesReceived += count
        metrics.parserHighWaterBytes = max(metrics.parserHighWaterBytes, parserHighWater)
    }

    @discardableResult
    mutating func frameReceived(_ meta: MJPEGPartMeta, now: TimeInterval) -> FrameVerdict {
        lastPartAt = now
        metrics.framesReceived += 1
        if let id = meta.streamID { noteStreamID(id) }
        guard let seq = meta.seq else {
            // Firmware before M8.3: no seq, no age. One frame proves nothing (the
            // old device sends its cached last frame to every new viewer); two
            // arriving within the window show something is advancing.
            defer { previousUnverifiedAt = now }
            if let previous = previousUnverifiedAt, now - previous <= config.staleAfter {
                lastAdvanceAt = now
                metrics.noteAdvance(at: now, ageMs: nil)
            }
            return .unverified
        }
        if let last = lastSeq {
            if seq == last {
                metrics.duplicateFrames += 1
                return .duplicate
            }
            if seq < last {
                // Backwards with no new stream id: the device restarted (or is
                // firmware that does not send one). A new baseline, not a stall.
                metrics.streamResets += 1
            } else if seq > last + 1 {
                metrics.framesSkippedByDevice += seq - last - 1
            }
        }
        lastSeq = seq
        lastSource = meta.source ?? lastSource
        if let age = meta.ageMs {
            metrics.lastFrameAgeMs = age
            if age > config.maxLiveFrameAgeMs {
                metrics.staleFrames += 1
                return .stale(ageMs: age)
            }
        }
        lastAdvanceAt = now
        metrics.noteAdvance(at: now, ageMs: meta.ageMs)
        return .advanced
    }

    mutating func heartbeatReceived(_ meta: MJPEGPartMeta, now: TimeInterval) {
        lastPartAt = now
        metrics.heartbeats += 1
        if let id = meta.streamID { noteStreamID(id) }
        heartbeatStatus = meta.status
        heartbeatAt = now
    }

    private mutating func noteStreamID(_ id: String) {
        if let current = streamID, current != id {
            // The device process restarted: its seq starts again from 1.
            metrics.streamResets += 1
            lastSeq = nil
        }
        streamID = id
    }

    // MARK: Decisions

    /// The liveness to show, given the transport's own state.
    func liveness(transport: MJPEGStreamState, now: TimeInterval) -> FeedLiveness {
        switch transport {
        case .idle:
            return .idle
        case .unauthorized, .unusableAddress:
            return .refused
        case .connecting:
            return .connecting
        case .reconnecting(_, let reason):
            return .unreachable(reason: reason)
        case .waitingForFrames, .streaming:
            break
        }
        guard let connectedAt else { return .connecting }
        // M8.4: a fresh "camera off" heartbeat newer than the last frame wins
        // over that frame -- the device withheld the camera at the stop.
        if let status = heartbeatStatus, status == CameraAccess.offWhileStoppedStatus,
           let at = heartbeatAt, now - at <= config.heartbeatValidFor, at >= (lastAdvanceAt ?? -.infinity) {
            return .cameraOffWhileStopped
        }
        if let advanced = lastAdvanceAt, now - advanced <= config.staleAfter {
            return .live
        }
        if let status = heartbeatStatus, let at = heartbeatAt, now - at <= config.heartbeatValidFor,
           status == "camera_stale" || status == "camera_not_started" {
            return .cameraUnavailable
        }
        let since = lastAdvanceAt ?? connectedAt
        let quiet = now - since
        if lastAdvanceAt == nil, quiet < config.staleAfter { return .connecting }
        return .stalled(seconds: max(Int(quiet.rounded(.down)), Int(config.staleAfter)))
    }

    /// Whether the connection should be replaced: open, from a device that
    /// sends heartbeats, and nothing at all has arrived for too long.
    func shouldForceReconnect(now: TimeInterval) -> Bool {
        guard connectedAt != nil, heartbeatCapable, let last = lastPartAt else { return false }
        return now - last > config.silenceBeforeReconnect
    }
}

// MARK: - Instrumentation

/// Rolling counters for the live feed. Plain stored values in a struct the
/// reader owns -- never @Published, so nothing redraws per frame because of
/// them. Read on demand (diagnostics, logs).
struct StreamMetrics: Equatable {
    var connectionOpens = 0
    var reconnects = 0
    var transportErrors = 0
    var framesReceived = 0
    var framesAdvanced = 0
    var duplicateFrames = 0
    var staleFrames = 0
    var framesSkippedByDevice = 0
    var streamResets = 0
    var heartbeats = 0
    var decodeFailures = 0
    var bytesReceived = 0
    var parserHighWaterBytes = 0
    var lastFrameAgeMs: Int?

    /// Arrival times of the last advancing frames (bounded ring).
    private(set) var recentAdvances: [TimeInterval] = []
    /// Capture ages of the last advancing frames that carried one (bounded ring).
    private(set) var recentAgesMs: [Int] = []
    static let window = 32

    mutating func noteAdvance(at now: TimeInterval, ageMs: Int?) {
        framesAdvanced += 1
        recentAdvances.append(now)
        if recentAdvances.count > Self.window { recentAdvances.removeFirst(recentAdvances.count - Self.window) }
        if let ageMs {
            recentAgesMs.append(ageMs)
            if recentAgesMs.count > Self.window { recentAgesMs.removeFirst(recentAgesMs.count - Self.window) }
        }
    }

    /// Delivered (advancing) frames per second over the recent window, as of
    /// `now`: 0 once the newest is older than the window it spans.
    func deliveredFPS(now: TimeInterval) -> Double {
        guard recentAdvances.count >= 2, let first = recentAdvances.first, let last = recentAdvances.last,
              last > first else { return 0 }
        let span = last - first
        if now - last > max(span, 2) { return 0 }
        return Double(recentAdvances.count - 1) / span
    }

    /// Median capture age of recent frames, nil when the device does not say.
    var medianFrameAgeMs: Int? {
        guard !recentAgesMs.isEmpty else { return nil }
        return recentAgesMs.sorted()[recentAgesMs.count / 2]
    }

    /// One line for logs and diagnostics.
    func summary(now: TimeInterval) -> String {
        let fps = String(format: "%.1f", deliveredFPS(now: now))
        let age = medianFrameAgeMs.map { "\($0) ms" } ?? "n/a"
        return "fps \(fps) · age p50 \(age) · frames \(framesAdvanced)/\(framesReceived)"
            + " · dup \(duplicateFrames) · stale \(staleFrames) · skipped \(framesSkippedByDevice)"
            + " · resets \(streamResets) · reconnects \(reconnects) · errors \(transportErrors)"
            + " · heartbeats \(heartbeats) · decode fails \(decodeFailures)"
    }
}

// MARK: - What the badge says

/// The words over the picture. Only `.live` may say LIVE; stale video says how
/// stale. Monitoring and video are separate facts: video can be live while
/// monitoring is stopped, and the badge says so.
struct FeedIndicator: Equatable {
    enum Tone: Equatable { case live, warning, neutral }
    let label: String
    let detail: String?
    let tone: Tone
    let isLive: Bool

    /// - Parameters:
    ///   - monitoringStopped: the device reports an explicit stop (the app's
    ///     monitoring state, not the stream).
    ///   - frameSource: the newest frame's `X-Frame-Source`, nil from old firmware.
    static func resolve(_ liveness: FeedLiveness, monitoringStopped: Bool, frameSource: String?) -> FeedIndicator? {
        switch liveness {
        case .live:
            let detail: String?
            // The device's label for the frame wins over the app's own idea of
            // the monitoring state (M8.4): it says what the device actually did.
            if frameSource == CameraAccess.sourceCameraOnly {
                detail = "Camera only — not monitoring"
            } else if frameSource == CameraAccess.sourceCameraLocal {
                detail = "Local view — not monitoring"
            } else if monitoringStopped {
                detail = "Monitoring paused"
            } else if frameSource == CameraAccess.sourceCamera {
                detail = "Not analysed"
            } else {
                detail = nil
            }
            return FeedIndicator(label: "LIVE", detail: detail, tone: .live, isLive: true)
        case .cameraOffWhileStopped:
            return FeedIndicator(label: "CAMERA OFF", detail: "Monitoring stopped", tone: .neutral, isLive: false)
        case .stalled(let seconds):
            return FeedIndicator(label: "NO NEW FRAMES", detail: "\(DevicePresenceDuration.short(seconds)) old",
                                 tone: .warning, isLive: false)
        case .cameraUnavailable:
            return FeedIndicator(label: "CAMERA NOT SENDING", detail: nil, tone: .warning, isLive: false)
        case .idle, .connecting, .unreachable, .refused, .disabled:
            // The picture area already says these in words (MJPEGStreamState
            // message, CameraPlaceholder); a second badge would repeat it.
            return nil
        }
    }

    var accessibilityText: String {
        [isLive ? "Live video" : label.capitalized, detail].compactMap { $0 }.joined(separator: ", ")
    }
}

// MARK: - Text inside the camera viewport (M8.4, AX sizes)

/// Text drawn inside the camera's frame has a fixed box to live in: at the
/// accessibility sizes the full sentences ran past the viewport and under the
/// tab bar, and the chips over the picture truncated to "SIMU…" (IOS audit,
/// m84-evidence/ios/before/home_ax5.png). The rule, for every camera state
/// (placeholder, camera off while stopped, camera only):
///   * at an accessibility size, the short wording replaces the long one --
///     shorter words at the reader's size beat the long words shrunk away;
///   * whatever is left may scale down to `minimumScale`, never further, and
///     is bounded to `maxLines`, so it can never spill out of the viewport;
///   * chips over footage are capped at xxxLarge (CameraViews.swift) and offer
///     the Large Content Viewer instead.
enum CameraTextPolicy {
    static let minimumScale: Double = 0.7
    static let maxLines = 4

    struct Copy: Equatable {
        let headline: String
        let detail: String
    }

    static func copy(_ full: Copy, short: Copy, accessibilitySize: Bool) -> Copy {
        accessibilitySize ? short : full
    }

    /// The full-screen camera's status line ("All clear · Online"): one line
    /// joined by " · " at standard sizes; at AX sizes the presence word goes on
    /// its own line under the readout (joined, it broke mid-word: "On-/line").
    static func stacksStatusLine(accessibilitySize: Bool) -> Bool { accessibilitySize }

    /// No picture: the device is not reachable from this phone.
    static let placeholder = Copy(
        headline: "Live picture on your Wi-Fi",
        detail: "The picture comes straight from your SensorDyme, so it needs to be switched on and on this Wi-Fi. ORYN EDGE finds it by itself.")
    static let placeholderShort = Copy(headline: "Live picture on your Wi-Fi",
                                       detail: "Switch the device on and join this Wi-Fi.")
    /// Decision C: monitoring stopped, camera-only off.
    static let cameraOff = Copy(
        headline: "Camera off — monitoring stopped",
        detail: "Resume monitoring to see the picture again. The owner can allow the camera while stopped in Settings.")
    static let cameraOffShort = Copy(headline: "Camera off — monitoring stopped",
                                     detail: "Resume monitoring to see it.")
}

// MARK: - Camera access the device reports (M8.4, owner Decision C)

/// `status.camera_mode` from GET /v1/devices (cloud/app.py `_camera_mode_status`).
/// Every field optional: a backend before M8.4 sends none of it.
struct DeviceCameraMode: Codable, Equatable {
    /// The device's software understands the setting (reports a camera block).
    var supported: Bool?
    /// What the DEVICE reports: camera-only on. Nil when unknown/unsupported.
    var cameraOnly: Bool?
    /// What a remote viewer gets right now: "monitoring" | "camera_only" |
    /// "off_while_stopped" | "unavailable"; nil when stale or unknown.
    var remoteStream: String?
    /// What the owner asked for (the backend's record), shown beside the truth.
    var desiredCameraOnly: Bool?
    /// The owner's request has not been reported back by the device yet.
    var pending: Bool?
    var commandID: String?
    var stale: Bool?

    enum CodingKeys: String, CodingKey {
        case supported, stale, pending
        case cameraOnly = "camera_only"
        case remoteStream = "remote_stream"
        case desiredCameraOnly = "desired_camera_only"
        case commandID = "command_id"
    }
}

/// What the app says about camera access while monitoring is stopped -- from
/// what the DEVICE reports, never from the request alone.
struct CameraModePresentation: Equatable {
    enum Kind: Equatable {
        /// Camera off while stopped (the default).
        case offWhileStopped
        /// The owner turned camera-only on; the device confirmed it.
        case cameraOnly
        /// The device's software predates the setting (M8.3 or older).
        case unsupported
        /// Nothing reported (backend before M8.4, device never seen).
        case unknown
    }
    let kind: Kind
    let title: String
    let detail: String
    /// The owner asked for a change the device has not reported yet.
    let pending: Bool
    /// The toggle's position: the request while pending, else the DEVICE's value.
    let toggleOn: Bool
    /// Whether the app may offer the toggle at all.
    let canChange: Bool

    static func resolve(_ mode: DeviceCameraMode?) -> CameraModePresentation {
        guard let mode, let supported = mode.supported else {
            return CameraModePresentation(kind: .unknown, title: "Camera while stopped",
                                          detail: "Not reported by this device yet.",
                                          pending: false, toggleOn: false, canChange: false)
        }
        guard supported else {
            return CameraModePresentation(
                kind: .unsupported, title: "Camera while stopped",
                detail: "This device's software can't turn the camera off while monitoring is stopped. Update it.",
                pending: false, toggleOn: false, canChange: false)
        }
        let pending = mode.pending == true
        let reportedOn = mode.cameraOnly == true
        let wanted = mode.desiredCameraOnly ?? reportedOn
        let staleNote = mode.stale == true ? " (last reported)" : ""
        if reportedOn {
            return CameraModePresentation(
                kind: .cameraOnly, title: "Camera only — not monitoring",
                detail: (pending ? "Turning off — waiting for the device to confirm."
                         : "While monitoring is stopped, you can still watch the live camera.") + staleNote,
                pending: pending, toggleOn: pending ? wanted : true, canChange: true)
        }
        return CameraModePresentation(
            kind: .offWhileStopped, title: "Camera off while stopped",
            detail: (pending ? "Turning on — waiting for the device to confirm."
                     : "While monitoring is stopped, the camera is not shown in the app.") + staleNote,
            pending: pending, toggleOn: pending ? wanted : false, canChange: true)
    }
}

/// "4 s", "2 min": short enough for a badge over video.
enum DevicePresenceDuration {
    static func short(_ seconds: Int) -> String {
        let s = max(0, seconds)
        if s < 60 { return "\(s) s" }
        if s < 3600 { return "\(s / 60) min" }
        return "\(s / 3600) h"
    }
}

// MARK: - One connection, however many views

/// Reference-counts the views showing the live feed so the app holds exactly
/// ONE connection to the device: inline and full screen (and the zone editor)
/// share it, and a presentation swapping one view for another -- whose
/// appear/disappear order SwiftUI does not promise -- never reconnects.
///
/// Leases, not a bare counter: a view that appears twice without disappearing
/// (it happens) cannot inflate the count and pin the stream open for ever, and
/// a stray detach cannot release someone else's viewer.
///
/// The transport is two closures so this is testable without URLSession
/// (StreamLivenessTests uses a fake that counts opens).
@MainActor
final class SharedFeedCoordinator {
    typealias Lease = Int

    private let open: @MainActor (_ base: String, _ token: String) -> Void
    private let close: @MainActor () -> Void
    private let after: (_ delay: TimeInterval, _ work: @escaping @MainActor () -> Void) -> Void
    /// Covers inline -> full screen and back without a reconnect.
    let grace: TimeInterval

    private var leases: Set<Lease> = []
    private var nextLease = 0
    private var stopGeneration = 0
    private(set) var target: (base: String, token: String)?
    /// The app is in the background: nothing is open, whatever the leases.
    private(set) var suspended = false
    /// Whether the transport is believed open (last call was open, not close).
    private(set) var isOpen = false

    init(grace: TimeInterval = 0.6,
         open: @escaping @MainActor (_ base: String, _ token: String) -> Void,
         close: @escaping @MainActor () -> Void,
         after: @escaping (_ delay: TimeInterval, _ work: @escaping @MainActor () -> Void) -> Void) {
        self.grace = grace
        self.open = open
        self.close = close
        self.after = after
    }

    var viewers: Int { leases.count }

    /// A view starts showing the feed. Returns its lease.
    func attach(base: String, token: String) -> Lease {
        nextLease += 1
        leases.insert(nextLease)
        stopGeneration += 1  // cancels a pending stop
        target = (base, token)
        if !suspended { openTransport() }
        return nextLease
    }

    /// A view stops showing the feed. Unknown or already-released leases do nothing.
    func detach(_ lease: Lease) {
        guard leases.remove(lease) != nil, leases.isEmpty else { return }
        stopGeneration += 1
        let generation = stopGeneration
        after(grace) { [weak self] in
            guard let self, self.stopGeneration == generation, self.leases.isEmpty else { return }
            self.closeTransport()
        }
    }

    /// The device's address or token changed while the feed is shown.
    func retarget(base: String, token: String) {
        target = (base, token)
        if !leases.isEmpty, !suspended { openTransport() }
    }

    /// The app went to the background: release the connection now (the
    /// device's inference loop pays for every viewer, core/dashboard.py).
    func suspend() {
        guard !suspended else { return }
        suspended = true
        stopGeneration += 1
        closeTransport()
    }

    /// Back in the foreground: reopen if anyone is still showing the feed.
    func resume() {
        guard suspended else { return }
        suspended = false
        if !leases.isEmpty { openTransport() }
    }

    private func openTransport() {
        guard let target else { return }
        isOpen = true
        open(target.base, target.token)  // the reader's start() is a no-op for an unchanged healthy target
    }

    private func closeTransport() {
        guard isOpen else { return }
        isOpen = false
        close()
    }
}
