import SwiftUI

// MARK: - Live feed area: real MJPEG on the LAN, simulated viewport in demo mode.

struct FeedCard: View {
    @EnvironmentObject private var state: AppState
    let device: Device
    let alertActive: Bool

    private var isLiveFeed: Bool {
        !state.isDemo && !state.deviceLocalURL.isEmpty && URL(string: state.deviceLocalURL) != nil
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
        .clipShape(RoundedRectangle(cornerRadius: 22))
        .overlay(
            RoundedRectangle(cornerRadius: 22)
                .strokeBorder(alertActive ? AnyShapeStyle(SD.Colors.danger) : AnyShapeStyle(.quaternary),
                              lineWidth: alertActive ? 3 : 1)
        )
        .animation(.easeInOut(duration: 0.4), value: alertActive)
    }

    private var feedPlaceholder: some View {
        ZStack {
            SD.Colors.canvas
            VStack(spacing: 10) {
                Image(systemName: "video.slash.fill").font(.title).foregroundStyle(SD.Colors.textSecondary)
                Text("Live feed streams on your home/car network.\nSet the device address in Settings.")
                    .font(.caption)
                    .multilineTextAlignment(.center)
                    .foregroundStyle(SD.Colors.textSecondary)
            }
        }
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

    var body: some View {
        TimelineView(.animation(minimumInterval: 1 / 30)) { context in
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
                RoundedRectangle(cornerRadius: 10)
                    .strokeBorder(alert ? AnyShapeStyle(SD.Colors.danger) : AnyShapeStyle(SD.Colors.accent), lineWidth: 2)
                    .frame(width: 130, height: 150)
                    .offset(x: 12 * sin(t * 0.35), y: 4 + 4 * sin(t * 0.9))

                // sweeping scan line
                LinearGradient(colors: [.clear, SD.Colors.accent.opacity(0.35), .clear],
                               startPoint: .top, endPoint: .bottom)
                    .frame(height: 46)
                    .offset(y: CGFloat(((t * 40).truncatingRemainder(dividingBy: 300)) - 150))
                    .blendMode(.screen)

                VStack {
                    HStack {
                        Label("LIVE", systemImage: "dot.radiowaves.left.and.right")
                            .font(.caption2.bold())
                            .padding(.horizontal, 10).padding(.vertical, 5)
                            .background(Capsule().fill(.red.opacity(0.85)))
                            .foregroundStyle(.white)
                        Spacer()
                        Text(Preset.title(for: preset))
                            .font(.caption2.bold())
                            .padding(.horizontal, 10).padding(.vertical, 5)
                            .background(Capsule().fill(.ultraThinMaterial))
                    }
                    Spacer()
                    HStack {
                        Label("demo device — simulated feed", systemImage: "sparkles")
                            .font(.caption2)
                            .foregroundStyle(.white.opacity(0.65))
                        Spacer()
                    }
                }
                .padding(12)
            }
        }
    }
}

// MARK: - Real MJPEG stream reader (the device dashboard's /stream endpoint on the LAN).

struct MJPEGView: View {
    let url: URL
    @EnvironmentObject private var state: AppState
    @StateObject private var reader = MJPEGReader()
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
            } else if reader.state.isRecoverableAutomatically {
                ProgressView().tint(.white)
                    .frame(height: 230)
            } else {
                // A state that will never heal on its own gets an icon rather than
                // a spinner. A spinner is a promise that something is still
                // happening, and for a rejected token or a malformed address
                // nothing is.
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.title)
                    .foregroundStyle(SD.Colors.textSecondary)
                    .frame(height: 230)
            }
            if let message = reader.state.message {
                VStack {
                    Spacer()
                    Label(message, systemImage: reader.state.isRecoverableAutomatically
                          ? "wifi.exclamationmark" : "exclamationmark.triangle.fill")
                        .font(.caption2)
                        .multilineTextAlignment(.center)
                        .padding(6)
                        .background(.black.opacity(0.6), in: Capsule())
                        .foregroundStyle(.white)
                        .padding(8)
                }
            }
        }
        .onAppear {
            // Wired before start(), so the very first run of failures can trigger
            // a re-resolve. Adopting the answer needs no code here: DeviceLink
            // writes the device's current address to the same stored key the feed
            // reads, so `onChange(of: url)` below picks it up and reconnects.
            reader.onAddressLooksStale = { Task { await link.resolve(force: true) } }
            reader.start(base: url.absoluteString, token: state.deviceToken)
        }
        .onDisappear { reader.stop() }
        // Backgrounding is not `onDisappear`. The view stays in the hierarchy when
        // the app leaves the foreground, so the MJPEG connection stayed open for a
        // feed nobody could see — and on the device that is not merely untidy:
        // `has_viewers()` is what makes the inference loop annotate and JPEG-encode
        // at all (see core/dashboard.py), so a held viewer slot spends Pi CPU that
        // the detector needs, for as long as iOS leaves the socket alive. It also
        // spends the phone's battery and data pulling frames at 12fps.
        //
        // Stopping on background and starting again on return costs one reconnect,
        // which the reader already does routinely and shows as a brief spinner.
        // Only `.background`, deliberately not `.inactive`: inactive is also the
        // app switcher, Control Centre and an incoming-call banner, and tearing
        // the stream down for a swipe the user immediately reverses would be a
        // visible stutter for no saving.
        .onChange(of: scenePhase) { _, phase in
            switch phase {
            case .background:
                reader.stop()
            case .active:
                // start() is a no-op when the target is unchanged and the stream is
                // already healthy, so returning from a brief `.inactive` does not
                // restart anything.
                reader.start(base: url.absoluteString, token: state.deviceToken)
            case .inactive:
                break
            @unknown default:
                break
            }
        }
        // The address and the token are both editable in Settings while this view
        // is on screen, and discovery can re-point the device at a new address at
        // any moment. Without these the reader kept streaming from — or endlessly
        // retrying — whatever it was given when it first appeared, and the only
        // way to pick up a correction was to leave the screen and come back.
        .onChange(of: url) { _, newURL in
            reader.start(base: newURL.absoluteString, token: state.deviceToken)
        }
        .onChange(of: state.deviceToken) { _, newToken in
            reader.start(base: url.absoluteString, token: newToken)
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
    @Published private(set) var state: MJPEGStreamState = .idle

    private let retryPolicy = MJPEGRetryPolicy()
    private var parser = MJPEGFrameParser()
    private var session: URLSession?
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

    private static let failuresBeforeReresolving = 3
    private static let addressRecheckInterval: TimeInterval = 20

    // MARK: Lifecycle

    /// Point the reader at a device and (re)start it. Safe to call repeatedly —
    /// it is also how an address or token change is applied.
    func start(base: String, token: String = "") {
        // An unchanged target must not restart a healthy stream: `onChange` and
        // `onAppear` can both fire for the same values, and tearing the
        // connection down to rebuild it identically is a visible stutter.
        if !stopped, base == self.base, token == self.token, state != .idle { return }
        teardown()
        self.base = base
        self.token = token
        stopped = false
        retryAttempt = 0
        consecutiveTransportFailures = 0
        lastFailureReason = nil
        connect()
    }

    func stop() {
        stopped = true
        teardown()
        state = .idle
    }

    /// Release the session and the pending retry, without deciding what happens next.
    private func teardown() {
        retryTask?.cancel()
        retryTask = nil
        // invalidateAndCancel(), not finishTasksAndInvalidate(): a live stream has
        // no natural end, so "finish" would never come. This also releases the
        // session's strong reference to this delegate.
        session?.invalidateAndCancel()
        session = nil
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
        self.session = session
        state = image == nil ? .connecting : state
        session.dataTask(with: request).resume()
    }

    /// How long without a single byte before the connection is considered dead.
    /// The device pushes at `dashboard.stream_fps` (12/s by default), so silence
    /// this long means the stream is not coming back on its own.
    private static let idleTimeout: TimeInterval = 15

    /// Any drop (device rebooted, Wi-Fi blip, app came back from background)
    /// retries with capped exponential backoff instead of spinning forever.
    private func scheduleReconnect(reason: String? = nil) {
        guard !stopped else { return }
        teardown()  // cancels any retry already pending, so retries cannot pile up
        // Keep the last explanation while retrying rather than blanking it: the
        // reason a reconnect is happening does not stop being true between
        // attempts, and a badge that alternates between explaining itself and
        // saying nothing is worse than one that keeps saying it.
        lastFailureReason = reason ?? lastFailureReason
        state = .reconnecting(attempt: retryAttempt + 1, reason: lastFailureReason)
        consecutiveTransportFailures += 1
        // A device that answered yesterday and refuses every connection today has
        // usually taken a new DHCP address -- the stored one is a cache, not
        // configuration (see DeviceLink). Ask for a fresh address rather than
        // retrying a dead one until someone opens Settings. Rate-limited so a
        // genuinely absent device cannot turn the retry loop into a discovery loop.
        let now = Date()
        if consecutiveTransportFailures >= Self.failuresBeforeReresolving,
           now.timeIntervalSince(lastAddressRecheck) >= Self.addressRecheckInterval {
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
        self.session === session
    }

    // MARK: URLSessionDataDelegate

    nonisolated func urlSession(
        _ session: URLSession,
        dataTask: URLSessionDataTask,
        didReceive response: URLResponse,
        completionHandler: @escaping @Sendable (URLSession.ResponseDisposition) -> Void
    ) {
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        // Reading the body of a response we cannot use only wastes the radio; the
        // status is recovered from `task.response` in didCompleteWithError below,
        // which is where the decision about retrying is made.
        completionHandler(status == 200 ? .allow : .cancel)
        guard status == 200 else { return }
        Task { @MainActor [weak self] in
            guard let self, self.isCurrent(session) else { return }
            // The device answered. Anything still wrong from here is the camera's
            // end, not the network's, and the badge should stop blaming Wi-Fi.
            if self.state != .streaming { self.state = .waitingForFrames }
        }
    }

    nonisolated func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        Task { @MainActor [weak self] in
            guard let self, self.isCurrent(session) else { return }  // bytes from a superseded connection
            for frame in self.parser.append(data) {
                guard let decoded = UIImage(data: frame) else { continue }
                self.image = decoded
                self.state = .streaming
                self.retryAttempt = 0  // healthy again: future drops start with a short retry
                self.consecutiveTransportFailures = 0
                self.lastFailureReason = nil
            }
        }
    }

    nonisolated func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        let status = (task.response as? HTTPURLResponse)?.statusCode
        Task { @MainActor [weak self] in
            guard let self, self.isCurrent(session) else { return }
            // THE fix. Without this guard, the completion for a session this class
            // cancelled itself was read as "the live stream dropped" and tore down
            // the healthy session that had already replaced it.

            if status == 401 || status == 403 {
                // No amount of retrying supplies a token. Looping here is what
                // turned a one-line settings mistake into a permanent spinner.
                // The device DID answer, so this is not a transport failure and
                // must not count toward re-resolving the address.
                self.consecutiveTransportFailures = 0
                self.image = nil
                self.state = .unauthorized(hasToken: !self.token.isEmpty)
                self.teardown()
                return
            }
            self.scheduleReconnect(reason: MJPEGStreamDiagnosis.describe(error))
        }
    }
}
