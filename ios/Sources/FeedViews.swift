import SwiftUI

// MARK: - Live feed area: real MJPEG on the LAN, simulated viewport in demo mode.

struct FeedCard: View {
    @EnvironmentObject private var state: AppState
    let device: Device
    let alertActive: Bool

    private var isLiveFeed: Bool {
        !state.isDemo && !state.deviceLocalURL.isEmpty && URL(string: state.deviceLocalURL) != nil
            && FeedLink.showsLiveFeed(
                deviceHardwareID: device.hardwareId,
                linkedID: state.linkedDeviceID,
                accountDeviceCount: state.accountDeviceCount
            )
    }

    var body: some View {
        ZStack {
            if state.isDemo {
                SimulatedFeedView(preset: device.activePreset ?? "driver_safety", alert: alertActive)
            } else if isLiveFeed, let url = URL(string: state.deviceLocalURL) {
                MJPEGView(url: url)  // MJPEGView appends /stream (and the device token) itself
            } else {
                feedPlaceholder
            }
        }
        // The live feed sizes itself to the camera's real aspect ratio (never
        // cropped — see MJPEGView); the simulated feed and placeholder keep a
        // fixed card height.
        .frame(height: isLiveFeed ? nil : 230)
        .frame(maxWidth: .infinity)
        .clipShape(RoundedRectangle(cornerRadius: SD.Radius.lg, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: SD.Radius.lg, style: .continuous)
                .strokeBorder(alertActive ? SD.Colors.danger : SD.Colors.borderStrong,
                              lineWidth: alertActive ? 2 : 1)
        )
        .animation(.easeInOut(duration: 0.4), value: alertActive)
    }

    private var feedPlaceholder: some View {
        CameraPlaceholder()
    }
}

// MARK: - Simulated demo feed: stylized viewport with scan motion and detection chips.

struct SimulatedFeedView: View {
    let preset: String
    let alert: Bool

    private var sceneIcon: String {
        switch Preset.split(preset).base {
        case "baby_monitoring": "figure.child"
        case "elderly_monitoring": "figure.walk"
        case "security_monitoring": "door.left.hand.open"
        default: "car.side.fill"
        }
    }

    /// The placeholder breathes and scans; under Reduce Motion it holds still.
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        TimelineView(.animation(minimumInterval: 1 / 15,
                                paused: !MotionPolicy.animates(true, reduceMotion: reduceMotion))) { context in
            let t = context.date.timeIntervalSinceReferenceDate
            ZStack {
                SD.Colors.canvas

                // subject silhouette with gentle "breathing" motion
                Image(systemName: sceneIcon)
                    .font(.system(size: 90))
                    .foregroundStyle(.white.opacity(0.16))
                    .scaleEffect(1 + 0.02 * sin(t * 1.4))
                    .offset(x: 12 * sin(t * 0.35), y: 8)

                // tracking box drifting like a live detector
                RoundedRectangle(cornerRadius: 4)
                    .strokeBorder(alert ? SD.Colors.danger : SD.Colors.textPrimary.opacity(0.7), lineWidth: 1.5)
                    .frame(width: 130, height: 150)
                    .offset(x: 12 * sin(t * 0.35), y: 4 + 4 * sin(t * 0.9))

                // sweeping scan line
                LinearGradient(colors: [.clear, SD.Colors.textPrimary.opacity(0.12), .clear],
                               startPoint: .top, endPoint: .bottom)
                    .frame(height: 46)
                    .offset(y: CGFloat(((t * 40).truncatingRemainder(dividingBy: 300)) - 150))
                    .blendMode(.screen)

                VStack {
                    HStack {
                        // Not "LIVE": nothing here comes from a camera (M8.3 --
                        // the word is reserved for verified, advancing video).
                        FeedStatusBadge(text: "SIMULATED")
                        Spacer()
                        FeedTag(text: Preset.title(for: preset))
                    }
                    Spacer()
                    HStack {
                        FeedTag(text: "Demo · simulated feed")
                        Spacer()
                    }
                }
                .padding(12)
            }
        }
    }
}

/// ORYN's live indicator: a signal-orange dot and "LIVE" on a dark plate.
/// Orange rather than red, so "live" can never be mistaken for "alert".
struct FeedLiveBadge: View {
    var body: some View {
        HStack(spacing: 6) {
            Circle().fill(SD.Colors.signal).frame(width: 6, height: 6)
            Text("LIVE")
                .sdEyebrowFont()
                .kerning(1.2)
                .foregroundStyle(SD.Colors.textPrimary)
        }
        .padding(.horizontal, 9)
        .padding(.vertical, 5)
        .background(Capsule().fill(Color.black.opacity(0.6)))
        .overlay(Capsule().strokeBorder(Color.white.opacity(0.08), lineWidth: 1))
    }
}

/// The not-live counterpart of FeedLiveBadge: same plate, a steel dot, the
/// state in words ("NO NEW FRAMES"). Never orange: orange means live.
struct FeedStatusBadge: View {
    let text: String

    var body: some View {
        HStack(spacing: 6) {
            Circle().fill(SD.Colors.textTertiary).frame(width: 6, height: 6)
            Text(text)
                .sdEyebrowFont()
                .kerning(1.2)
                .foregroundStyle(SD.Colors.textPrimary)
                .lineLimit(1)
        }
        .padding(.horizontal, 9)
        .padding(.vertical, 5)
        .background(Capsule().fill(Color.black.opacity(0.6)))
        .overlay(Capsule().strokeBorder(Color.white.opacity(0.08), lineWidth: 1))
    }
}

/// A small label laid over footage.
struct FeedTag: View {
    let text: String

    var body: some View {
        Text(text.uppercased())
            .sdEyebrowFont()
            .tracking(1)
            .foregroundStyle(SD.Colors.textSecondary)
            .lineLimit(1)
            .padding(.horizontal, 9)
            .padding(.vertical, 5)
            .background(Capsule().fill(Color.black.opacity(0.6)))
            .overlay(Capsule().strokeBorder(Color.white.opacity(0.08), lineWidth: 1))
    }
}

// MARK: - Real MJPEG stream reader (the device dashboard's /stream endpoint on the LAN).

/// The live picture's width / height, once a frame has decoded.
///
/// Published so a view drawn OVER the feed (the zone editor) can size itself to
/// the picture the camera really sends. The editor used a fixed 4:3 canvas while
/// the feed letterboxes to the camera's own ratio, so on any other capture size
/// the drawn polygon was normalised against the letterbox, not the frame.
struct FeedAspectRatioKey: PreferenceKey {
    static var defaultValue: CGFloat? { nil }
    static func reduce(value: inout CGFloat?, nextValue: () -> CGFloat?) {
        value = value ?? nextValue()
    }
}

/// The app's live feed: ONE connection to the device, shared by every view
/// that shows it -- the inline camera, full screen, and the zone editor. The
/// phone links one camera (FeedLink; a single stored device address), so one
/// reader serves the whole app; opening full screen, or the zone editor over
/// the device screen, never opens a second stream to the device (whose
/// inference loop pays for every viewer, see core/dashboard.py).
///
/// Every `LiveFeedSession` value is a handle on the same reader and the same
/// reference count (`SharedFeedCoordinator`, StreamLiveness.swift -- leases,
/// a grace period for view swaps, suspend/resume for the scene phase). M8.3:
/// it used to be per screen, so the zone editor's own `MJPEGView` opened a
/// second connection while the device screen's stayed open underneath it.
@MainActor
final class LiveFeedSession {
    static let shared = LiveFeedSession()

    private static let reader = MJPEGReader()
    private static let coordinator = SharedFeedCoordinator(
        open: { base, token in LiveFeedSession.reader.start(base: base, token: token) },
        close: { LiveFeedSession.reader.stop() },
        after: { delay, work in
            Task { @MainActor in
                try? await Task.sleep(for: .seconds(delay))
                work()
            }
        }
    )

    var reader: MJPEGReader { Self.reader }
    var coordinator: SharedFeedCoordinator { Self.coordinator }

    func attach(base: String, token: String) -> SharedFeedCoordinator.Lease {
        Self.coordinator.attach(base: base, token: token)
    }

    func detach(_ lease: SharedFeedCoordinator.Lease) { Self.coordinator.detach(lease) }
    func retarget(base: String, token: String) { Self.coordinator.retarget(base: base, token: token) }
    func suspend() { Self.coordinator.suspend() }
    func resume() { Self.coordinator.resume() }
}

struct MJPEGView: View {
    let url: URL
    /// The stream this view shows. Nil: the app's shared stream all the same
    /// (every session is a handle on the one connection).
    var session: LiveFeedSession? = nil
    @EnvironmentObject private var state: AppState
    private var feed: LiveFeedSession { session ?? LiveFeedSession.shared }
    private var reader: MJPEGReader { feed.reader }
    /// This view's hold on the shared connection, while it is on screen.
    @State private var lease: SharedFeedCoordinator.Lease?
    /// Backgrounding does not remove this view, so `onDisappear` never fires for
    /// it — the stream has to be released here instead. See the handler below.
    @Environment(\.scenePhase) private var scenePhase
    /// Used only to re-discover the device's current address when the stored one
    /// stops answering. A second instance is safe and deliberate: DeviceLink's
    /// state lives entirely in @AppStorage (the device's identity and its cached
    /// address), so this shares that state with the one DevicesView owns rather
    /// than competing with it — and `start()` is never called here, so this one
    /// runs no network monitor of its own.
    ///
    /// Reusing DeviceLink rather than browsing Bonjour directly is the point: it
    /// already refuses to adopt a different device when several answer, which is
    /// what stops the feed silently binding to a neighbour's camera.
    @StateObject private var link = DeviceLink()

    var body: some View {
        MJPEGFrame(reader: reader, liveness: reader.livenessModel)
        .onAppear {
            // Wired before start(), so the very first run of failures can trigger
            // a re-resolve. Adopting the answer needs no code here: DeviceLink
            // writes the device's current address to the same stored key the feed
            // reads, so `onChange(of: url)` below picks it up and reconnects.
            reader.onAddressLooksStale = { Task { await link.resolve(force: true) } }
            // One lease per view: an appear without a disappear in between
            // (it happens) must not count this view twice and pin the stream.
            if lease == nil {
                lease = feed.attach(base: url.absoluteString, token: state.deviceToken)
            } else {
                feed.retarget(base: url.absoluteString, token: state.deviceToken)
            }
            // An address is not an identity: confirm the camera answering there
            // is still the linked one (a DHCP swap, or two default-named Pis
            // trading `sensordyme.local`). If not, DeviceLink re-resolves and the
            // corrected address arrives through `onChange(of: url)`.
            Task { await link.ensureConnection() }
        }
        .onDisappear {
            if let lease { feed.detach(lease) }
            lease = nil
        }
        // Backgrounding is not `onDisappear`. The view stays in the hierarchy when
        // the app leaves the foreground, so the MJPEG connection stayed open for a
        // feed nobody could see — and on the device that is not merely untidy:
        // `has_viewers()` is what makes the inference loop annotate and JPEG-encode
        // at all (see core/dashboard.py), so a held viewer slot spends Pi CPU that
        // the detector needs, for as long as iOS leaves the socket alive. It also
        // spends the phone's battery and data pulling frames at 12fps.
        //
        // Suspend / resume go through the shared coordinator, so two views on
        // screen (inline under full screen) close one connection and reopen one.
        // Only `.background`, deliberately not `.inactive`: inactive is also the
        // app switcher, Control Centre and an incoming-call banner, and tearing
        // the stream down for a swipe the user immediately reverses would be a
        // visible stutter for no saving.
        .onChange(of: scenePhase) { _, phase in
            switch phase {
            case .background:
                feed.suspend()
            case .active:
                feed.resume()
            case .inactive:
                break
            @unknown default:
                break
            }
        }
        // The address and the token are both editable in Settings while this view
        // is on screen, and discovery can re-point the device at a new address at
        // any moment (a network change). Without these the reader kept streaming
        // from — or endlessly retrying — whatever it was given when it first
        // appeared.
        .onChange(of: url) { _, newURL in
            feed.retarget(base: newURL.absoluteString, token: state.deviceToken)
        }
        .onChange(of: state.deviceToken) { _, newToken in
            feed.retarget(base: url.absoluteString, token: newToken)
        }
    }
}

/// The feed's liveness, published only when it CHANGES (a few times a minute
/// at most), so views that only need "is this live?" never redraw per frame.
@MainActor
final class FeedLivenessModel: ObservableObject {
    @Published fileprivate(set) var liveness: FeedLiveness = .idle
    /// The newest frame's X-Frame-Source ("annotated" / "camera"), nil from old firmware.
    @Published fileprivate(set) var frameSource: String?
}

/// The picture and the stream's own state, observed from the reader. The only
/// view that redraws per frame.
struct MJPEGFrame: View {
    @ObservedObject var reader: MJPEGReader
    @ObservedObject var liveness: FeedLivenessModel

    var body: some View {
        ZStack {
            SD.Colors.canvas
            if let image = reader.image {
                // FULL frame, always: fit (letterbox) at the camera's real aspect
                // ratio, never fill+crop — cropping a safety camera hides exactly
                // the edges where events happen. The container adopts the stream's
                // aspect ratio so letterboxing only appears if the width forces it.
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: .infinity)
                    .aspectRatio(image.size.height > 0 ? image.size.width / image.size.height : 4 / 3,
                                 contentMode: .fit)
                    // A picture that is not live must not LOOK live (M8.3): the
                    // last frame stays for context, visibly faded.
                    .saturation(liveness.liveness.isLive ? 1 : 0.2)
                    .opacity(liveness.liveness.isLive ? 1 : 0.6)
                    .accessibilityLabel(liveness.liveness.isLive ? "Live picture" : "Last picture, not live")
            } else if reader.state.isRecoverableAutomatically {
                ProgressView().tint(SD.Colors.textSecondary)
                    .frame(height: 230)
                    .accessibilityLabel("Connecting to the camera")
            } else {
                // A state that will never heal on its own gets an icon rather than
                // a spinner. A spinner is a promise that something is still
                // happening, and for a rejected token or a malformed address
                // nothing is.
                Image(systemName: "exclamationmark.triangle")
                    .font(.system(size: 24, weight: .light))
                    .foregroundStyle(SD.Colors.textSecondary)
                    .frame(height: 230)
                    .accessibilityHidden(true)
            }
            if let message = reader.state.message {
                VStack {
                    Spacer()
                    Label(message, systemImage: reader.state.isRecoverableAutomatically
                          ? "wifi.exclamationmark" : "exclamationmark.triangle.fill")
                        .font(SD.Typography.caption)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(Color.black.opacity(0.7), in: Capsule())
                        .foregroundStyle(SD.Colors.textPrimary)
                        .padding(SD.Space.sm)
                }
            }
        }
        .preference(
            key: FeedAspectRatioKey.self,
            value: reader.image.flatMap { $0.size.height > 0 ? $0.size.width / $0.size.height : nil }
        )
    }
}

/// "LIVE" only while fresh frames are advancing (FeedLivenessTracker); a
/// stalled stream says how old its picture is, and live video over stopped
/// monitoring says that monitoring is paused. Observes the liveness model,
/// never the per-frame reader.
struct MJPEGLiveIndicator: View {
    @ObservedObject private var model: FeedLivenessModel
    private let monitoringStopped: Bool

    init(reader: MJPEGReader, monitoringStopped: Bool = false) {
        _model = ObservedObject(wrappedValue: reader.livenessModel)
        self.monitoringStopped = monitoringStopped
    }

    var body: some View {
        if let indicator = FeedIndicator.resolve(model.liveness, monitoringStopped: monitoringStopped,
                                                 frameSource: model.frameSource) {
            HStack(spacing: 6) {
                if indicator.isLive {
                    FeedLiveBadge()
                } else {
                    FeedStatusBadge(text: indicator.label)
                }
                if let detail = indicator.detail {
                    FeedTag(text: detail)
                }
            }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(indicator.accessibilityText)
            .accessibilityIdentifier("camera.liveness")
            .transition(.opacity)
        }
    }
}

/// Reads the device's MJPEG stream, and keeps reading it across drops.
///
/// The hard part is not parsing MJPEG, it is knowing which connection a callback
/// belongs to. `URLSession` delivers `didCompleteWithError` for a task we
/// deliberately cancelled just as it does for a stream that genuinely dropped,
/// and it delivers it asynchronously — after `start()` has already opened a
/// replacement. Treating those as the same event is what produced a black screen
/// with a permanent "reconnecting…" badge in front of a perfectly healthy device:
///
///   1. `stop()` cancelled session A; its completion callback was queued.
///   2. `start()` cleared the stopped flag and opened session B, which streamed.
///   3. A's stale completion arrived, was read as "the stream dropped", and
///      cancelled B — the healthy connection — to schedule a reconnect.
///   4. B's cancellation then arrived and did the same to C, and so on.
///
/// Each turn of that cycle left a session behind that nothing held a reference to
/// any more, so it could never be cancelled: measured, one appear/disappear pair
/// turned one connection into four, left two permanently live, and doubled the
/// bandwidth pulled from the device. Meanwhile the badge was only ever cleared
/// inside the "a frame decoded" branch, so it stayed lit through the whole storm.
///
/// The fix is identity: every callback is checked against the session that is
/// current *now*, and anything from a superseded session is dropped on the floor.
@MainActor
final class MJPEGReader: NSObject, ObservableObject, URLSessionDataDelegate {
    @Published private(set) var image: UIImage?
    @Published private(set) var state: MJPEGStreamState = .idle {
        didSet { refreshLiveness() }
    }
    /// What the person is looking at, decided from evidence (M8.3). Separate
    /// object so its observers do not redraw per frame.
    let livenessModel = FeedLivenessModel()
    /// The liveness state machine and the rolling counters (StreamLiveness.swift).
    private var tracker = FeedLivenessTracker()
    private var watchdog: Task<Void, Never>?
    private var connectTimeoutTask: Task<Void, Never>?
    /// Whether the current connection has had its response yet.
    private var responded = false

    /// Rolling instrumentation: fps, frame age, reconnects, errors. Not
    /// published; read on demand (diagnostics, logs).
    var metrics: StreamMetrics { tracker.metrics }
    func metricsSummary() -> String { tracker.metrics.summary(now: Self.now()) }

    private static func now() -> TimeInterval { ProcessInfo.processInfo.systemUptime }

    private let retryPolicy = MJPEGRetryPolicy()
    private var parser = MJPEGFrameParser()
    /// The session whose callbacks may act. See `CurrentConnection`.
    private let connection = CurrentConnection<URLSession>()
    private var base = ""
    private var token = ""
    private var stopped = true
    private var retryAttempt = 0
    private var retryTask: Task<Void, Never>?
    private var lastFailureReason: String?
    private var consecutiveTransportFailures = 0
    private var lastAddressRecheck = Date.distantPast

    /// Called when the address has failed enough times to be worth re-discovering.
    /// A closure rather than a dependency so this class keeps knowing nothing
    /// about Bonjour, and so the behaviour is testable without a network.
    var onAddressLooksStale: (() -> Void)?

    // MARK: Lifecycle

    /// Point the reader at a device and (re)start it. Safe to call repeatedly —
    /// it is also how an address or token change is applied.
    func start(base: String, token: String = "") {
        // An unchanged target must not restart a healthy stream: `onChange` and
        // `onAppear` can both fire for the same values, and tearing the
        // connection down to rebuild it identically is a visible stutter.
        guard MJPEGReaderPolicy.needsRestart(
            stopped: stopped, state: state,
            currentBase: self.base, currentToken: self.token,
            base: base, token: token
        ) else { return }
        teardown()
        self.base = base
        self.token = token
        stopped = false
        retryAttempt = 0
        consecutiveTransportFailures = 0
        lastFailureReason = nil
        startWatchdog()
        connect()
    }

    func stop() {
        stopped = true
        watchdog?.cancel()
        watchdog = nil
        teardown()
        state = .idle
    }

    /// Re-evaluate liveness twice a second while running: a stream that stops
    /// delivering has no callback to say so, only the passing of time.
    private func startWatchdog() {
        guard watchdog == nil else { return }
        watchdog = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(MJPEGReaderPolicy.livenessTick))
                guard let self, !Task.isCancelled else { return }
                self.tick()
            }
        }
    }

    private func tick() {
        guard !stopped else { return }
        if tracker.shouldForceReconnect(now: Self.now()) {
            // A device that sends heartbeats has gone silent: the socket is dead
            // (a network change, a reboot without a FIN). Do not wait out the
            // 15 s idle timeout with a frozen picture on screen.
            scheduleReconnect(reason: MJPEGReaderPolicy.silenceReason, countsAsTransportFailure: true)
            return
        }
        refreshLiveness()
    }

    /// Publish liveness only when it changed.
    private func refreshLiveness() {
        let now = Self.now()
        let liveness = tracker.liveness(transport: state, now: now)
        if livenessModel.liveness != liveness { livenessModel.liveness = liveness }
        if livenessModel.frameSource != tracker.lastSource { livenessModel.frameSource = tracker.lastSource }
    }

    /// Release the session and the pending retry, without deciding what happens next.
    private func teardown() {
        retryTask?.cancel()
        retryTask = nil
        connectTimeoutTask?.cancel()
        connectTimeoutTask = nil
        if connection.current != nil { tracker.connectionClosed(error: false) }
        // invalidateAndCancel(), not finishTasksAndInvalidate(): a live stream has
        // no natural end, so "finish" would never come. This also releases the
        // session's strong reference to this delegate.
        connection.current?.invalidateAndCancel()
        connection.clear()
        // Bytes from a connection that is going away must never be spliced onto
        // bytes from its replacement: the result would start inside one frame and
        // end inside another, decode to nothing, and look like a broken camera.
        parser.reset()
    }

    // MARK: Connecting

    private func connect() {
        guard !stopped else { return }
        // EVERY connection -- the first and every reconnect -- is built by the
        // same function, so a retry cannot quietly become an unauthenticated
        // request. The token is read from `self.token`, which start() set and
        // nothing else mutates, so it survives session recreation.
        //
        // Header, not a query parameter: a token in a URL lands in proxy logs,
        // browser history and Referer. The device accepts both (an <img> on its
        // own web page cannot set a header); the app has no such excuse.
        guard let request = MJPEGStreamRequest.make(
            base: base, token: token, idleTimeout: Self.idleTimeout
        ) else {
            // Retrying cannot turn "192.168.1.20:8899" into a URL. Say so once.
            image = nil
            state = .unusableAddress
            return
        }
        teardown()

        let configuration = URLSessionConfiguration.ephemeral
        configuration.urlCache = nil                       // never cache a live video stream
        configuration.requestCachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        configuration.timeoutIntervalForRequest = Self.idleTimeout
        // Belt and braces against the failure this class exists to prevent: even
        // if a session were ever leaked again, the app cannot hold more than one
        // connection to the device at a time.
        configuration.httpMaximumConnectionsPerHost = 1
        configuration.networkServiceType = .video

        let session = URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
        connection.replace(with: session)
        // Not a bare `.connecting`: this attempt may take the full request
        // timeout to fail, and on attempt two onwards we already know why the
        // last one did. See MJPEGStreamState.inFlight.
        state = MJPEGStreamState.inFlight(
            hasImage: image != nil,
            current: state,
            attempt: retryAttempt,
            reason: lastFailureReason
        )
        responded = false
        session.dataTask(with: request).resume()
        // A bounded connect: URLSession's only timeout is the idle one (15 s),
        // so an address that blackholes packets held every attempt that long.
        connectTimeoutTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(MJPEGReaderPolicy.connectTimeout))
            guard let self, !Task.isCancelled, self.isCurrent(session), !self.responded else { return }
            self.scheduleReconnect(reason: MJPEGReaderPolicy.connectTimeoutReason, countsAsTransportFailure: true)
        }
    }

    /// See `MJPEGReaderPolicy.idleTimeout`.
    private static let idleTimeout = MJPEGReaderPolicy.idleTimeout

    /// Any drop (device rebooted, Wi-Fi blip, app came back from background)
    /// retries with capped exponential backoff instead of spinning forever.
    private func scheduleReconnect(reason: String? = nil, countsAsTransportFailure: Bool = true) {
        guard !stopped else { return }
        teardown()  // cancels any retry already pending, so retries cannot pile up
        tracker.noteReconnect()
        // Keep the last explanation while retrying rather than blanking it: the
        // reason a reconnect is happening does not stop being true between
        // attempts, and a badge that alternates between explaining itself and
        // saying nothing is worse than one that keeps saying it.
        lastFailureReason = reason ?? lastFailureReason
        state = .reconnecting(attempt: retryAttempt + 1, reason: lastFailureReason)
        // Only failures to REACH the device count toward re-resolving its
        // address; a device that answered and then went quiet has not moved.
        if countsAsTransportFailure { consecutiveTransportFailures += 1 }
        // A device that answered yesterday and refuses every connection today has
        // usually taken a new DHCP address -- the stored one is a cache, not
        // configuration (see DeviceLink). Ask for a fresh address rather than
        // retrying a dead one until someone opens Settings. Rate-limited so a
        // genuinely absent device cannot turn the retry loop into a discovery loop.
        let now = Date()
        if MJPEGReaderPolicy.shouldReresolve(
            consecutiveFailures: consecutiveTransportFailures,
            sinceLastRecheck: now.timeIntervalSince(lastAddressRecheck)
        ) {
            lastAddressRecheck = now
            onAddressLooksStale?()
        }
        let delay = retryPolicy.delay(forAttempt: retryAttempt)
        retryAttempt += 1
        retryTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            guard !Task.isCancelled else { return }
            await MainActor.run { self?.connect() }
        }
    }

    /// Whether a callback belongs to the connection that is current *now*.
    /// Everything in this class turns on this question — see the type's docs.
    private func isCurrent(_ session: URLSession) -> Bool {
        connection.isCurrent(session)
    }

    // MARK: URLSessionDataDelegate

    nonisolated func urlSession(
        _ session: URLSession,
        dataTask: URLSessionDataTask,
        didReceive response: URLResponse,
        completionHandler: @escaping @Sendable (URLSession.ResponseDisposition) -> Void
    ) {
        let http = response as? HTTPURLResponse
        let status = http?.statusCode ?? 0
        // M8.3: a device that sends heartbeats says so; its silence then means a
        // dead connection (FeedLivenessTracker.shouldForceReconnect).
        let heartbeatCapable = http?.value(forHTTPHeaderField: MJPEGStreamRequest.heartbeatIntervalHeader) != nil
        let streamID = http?.value(forHTTPHeaderField: MJPEGStreamRequest.streamIDHeader)
        // Reading the body of a response we cannot use only wastes the radio; the
        // status is recovered from `task.response` in didCompleteWithError below,
        // which is where the decision about retrying is made.
        completionHandler(status == 200 ? .allow : .cancel)
        guard status == 200 else { return }
        Task { @MainActor [weak self] in
            guard let self, self.isCurrent(session) else { return }
            self.responded = true
            self.connectTimeoutTask?.cancel()
            self.tracker.connectionOpened(now: Self.now(), heartbeatCapable: heartbeatCapable, streamID: streamID)
            // The device answered. Anything still wrong from here is the camera's
            // end, not the network's, and the badge should stop blaming Wi-Fi.
            if self.state != .streaming { self.state = .waitingForFrames }
        }
    }

    nonisolated func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        Task { @MainActor [weak self] in
            guard let self, self.isCurrent(session) else { return }  // bytes from a superseded connection
            let now = Self.now()
            let parts = self.parser.appendParts(data)
            self.tracker.noteBytes(data.count, parserHighWater: self.parser.highWaterBytes)
            for part in parts {
                switch part {
                case .heartbeat(let meta):
                    self.tracker.heartbeatReceived(meta, now: now)
                case .frame(let jpeg, let meta):
                    // The evidence first: a repeat of the same seq is not news,
                    // and is not even decoded.
                    if self.tracker.frameReceived(meta, now: now) == .duplicate { continue }
                    guard let decoded = UIImage(data: jpeg) else {
                        self.tracker.noteDecodeFailure()
                        continue
                    }
                    self.image = decoded
                    self.state = .streaming
                    self.retryAttempt = 0  // healthy again: future drops start with a short retry
                    self.consecutiveTransportFailures = 0
                    self.lastFailureReason = nil
                }
            }
            self.refreshLiveness()
        }
    }

    nonisolated func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        let status = (task.response as? HTTPURLResponse)?.statusCode
        Task { @MainActor [weak self] in
            guard let self, self.isCurrent(session) else { return }
            // THE fix. Without this guard, the completion for a session this class
            // cancelled itself was read as "the live stream dropped" and tore down
            // the healthy session that had already replaced it.
            self.tracker.connectionClosed(error: MJPEGStreamDiagnosis.describe(error) != nil)

            switch MJPEGReaderPolicy.completion(status: status, error: error, hasToken: !self.token.isEmpty) {
            case .unauthorized(let hasToken):
                // No amount of retrying supplies a token. Looping here is what
                // turned a one-line settings mistake into a permanent spinner.
                // The device DID answer, so this is not a transport failure and
                // must not count toward re-resolving the address.
                self.consecutiveTransportFailures = 0
                self.image = nil
                self.state = .unauthorized(hasToken: hasToken)
                self.teardown()
            case .reconnect(let reason, let countsAsTransportFailure):
                self.scheduleReconnect(reason: reason, countsAsTransportFailure: countsAsTransportFailure)
            }
        }
    }
}
