import Combine
import SwiftUI

/// The screen a SensorDyme owner actually lives in: the camera, what it's
/// watching for, and what it has seen. Technical detail belongs in Device
/// Health, not here — nobody buying a baby monitor should meet the word "YOLO".
struct DeviceDetailView: View {
    @EnvironmentObject private var state: AppState
    @State var device: Device
    @State private var events: [EventItem] = []
    /// The mode switch this app instance most recently asked for and is still
    /// waiting to see confirmed — reconciled against the server's own
    /// `desiredPreset` on every reload (see `syncPendingSwitch`), so it never
    /// drifts from what's actually outstanding.
    @State private var pendingSwitch: ModeSwitchState.PendingSwitch?
    /// A switch that failed — either the request itself failed, or it never
    /// confirmed before ModeSwitchState.switchTimeout. Cleared the moment the
    /// user retries that same mode; sticky until then, so "tap to try again"
    /// has something to mean.
    @State private var failedSwitch: ModeSwitchState.PendingSwitch?
    /// Three independent triggers call reload() — on appear, a 4s timer, and
    /// pull-to-refresh — with no guarantee their network responses arrive in
    /// the order they were sent. Without this, a slow, stale response can
    /// complete after a fresh one and silently overwrite it: a mode switch
    /// that already confirmed could revert to "switching", or a just-fetched
    /// alert could vanish again. See LatestRequestGuard's own docs.
    @State private var reloadGuard = LatestRequestGuard()
    /// The same problem, for rapid mode-switch taps: an older tap's outcome
    /// (success or failure) must never land after a newer tap has already
    /// taken over the UI.
    @State private var setModeGuard = LatestRequestGuard()
    @State private var errorMessage: String?
    /// The banner's headline when the server answered with something unusable
    /// (ServerReply); nil = it couldn't be reached.
    @State private var errorTitle: String?
    /// A tapped notification whose alert couldn't be found. Its own banner: it
    /// used to go into `errorMessage`, which headed it "Can't reach the server"
    /// and turned the presence pill to Unknown for a server that was fine.
    @State private var alertOpenProblem: String?
    /// Advanced by the refresh timer whether or not a reload runs, so a reply
    /// that stops being refreshed stops claiming ONLINE (ConnectionTruth).
    @State private var now = Date()
    @Environment(\.dynamicTypeSize) private var typeSize
    @State private var showZones = false
    @State private var selectedEvent: EventItem?
    @State private var showDemoTrigger = false
    @StateObject private var notifier = AlertNotifier.shared
    @State private var showRename = false
    @State private var newName = ""
    @State private var hasLoaded = false
    /// Continuity: the camera, the modes and the Jupiter dock zoom into what
    /// they open (Continuity.swift).
    @Namespace private var zoomNS
    /// The device's live stream, shared by the inline and full-screen camera.
    /// `@State`, not `@StateObject`: held, never observed, so frames redraw the
    /// camera alone and never this screen.
    @State private var feedSession = LiveFeedSession()
    @State private var showCamera = false
    /// Device health opens itself when there is something to see (the device
    /// isn't reporting, or a component isn't working) and stays folded to one
    /// line while everything works. A tap overrides either way.
    @State private var healthOverride: Bool?
    @State private var expandedEventID: String?
    /// Counts transitions INTO ready, for one quiet acknowledgement.
    @State private var readyAck = 0
    @State private var protectionTaps = 0
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    /// The open Ask Jupiter screen and the request it starts with (sent straight
    /// away when non-empty). One value, so the sheet is built with the request it
    /// was opened for: a separate request string read only inside
    /// `.sheet(isPresented:)` could be captured stale, opening the screen with an
    /// empty prompt that never got an answer (XCUITest journey 10).
    @State private var jupiterSheet: JupiterSheet?
    @State private var showHistory = false
    @State private var renameError: String?

    // Preset library (PresetLibrary.swift). The catalog is the server's when it
    // publishes one, else the built-in list, labelled.
    @State private var catalog: PresetCatalog?
    @State private var devicePresets: DevicePresets?
    /// This server has no per-device presets endpoint; availability then
    /// comes from `status.allowed_presets` as before.
    @State private var devicePresetsMissing = false
    /// This server has no catalog endpoint; the built-in list is shown, labelled.
    @State private var catalogMissing = false
    @State private var presetSheet: CatalogPreset?

    // Custom rules (DeviceRules.swift).
    @State private var rulesLoad: CustomRulesCard.Load = .loading
    @State private var rulesBusy: Set<String> = []
    @State private var ruleToggleError: String?

    // Companion monitoring nodes (PerceptionStatus.swift): what was asked, and
    // when; the toggle shows the device's report, never the tap.
    @State private var nodeRequest: (id: String, wanted: Bool, at: Date)?
    @State private var nodeError: String?

    // Stop / resume (ProtectionControl.swift).
    @State private var protectionRequest: ProtectionRequest?
    @State private var protectionSending = false
    @State private var protectionError: String?
    /// The device's software or this server can't stop/resume; said instead
    /// of offering a control that cannot work.
    @State private var protectionUnsupported: String?
    /// What the device confirmed, shown for a while after it converged.
    @State private var protectionNotice: (text: String, at: Date)?
    @State private var confirmStop = false

    private let refreshTimer = Timer.publish(every: 4, on: .main, in: .common).autoconnect()
    @State private var timedReloadInFlight = false

    /// A loud alert in the last minute lights up the feed border + status.
    /// Whether to show the red alert state.
    ///
    /// Matches what the notification layer decides, so the screen and the phone
    /// never disagree about the same moment. Testing severity alone excluded
    /// configured rules — which default to "warning" -> "soft" — so someone who
    /// said "alert me when you see a bottle" got a notification about it and a
    /// perfectly calm screen.
    private var alertActive: Bool {
        let latest = events.first { $0.isCritical || $0.isConfigured }
        guard let latest, let date = latest.date else { return false }
        return Date().timeIntervalSince(date) < 60
    }

    var body: some View {
        ScrollViewReader { proxy in
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                banners

                // The camera is the centrepiece and always comes first.
                CameraViewport(device: device, instrument: instrument, session: feedSession) { openCamera() }
                    .sdZoomSource(.camera, in: zoomNS, cornerRadius: SD.Radius.camera, enabled: zoomEnabled)
                    .padding(.top, SD.Space.xs)

                instrumentPanel
                    .padding(.top, SD.Space.lg)

                actionKeys
                    .padding(.top, SD.Space.lg)

                recentActivity
                    .padding(.top, SD.Space.xl)

                modeLibrary
                    .padding(.top, SD.Space.xl)

                CustomRulesCard(
                    load: rulesLoad, deviceLive: deviceLive, busy: rulesBusy, toggleError: ruleToggleError,
                    onToggle: { rule, enabled in toggleRule(rule, enabled: enabled) },
                    onAskJupiter: { jupiterSheet = JupiterSheet(prompt: "") })
                    .padding(.top, SD.Space.xl)

                // Last on purpose: detail for someone who went looking for it,
                // not something competing with the picture.
                healthSection
                    .id("health")
                    .padding(.top, SD.Space.xl)
            }
            .padding(.horizontal, SD.Space.md)
            .padding(.bottom, SD.Space.xl)
        }
        .task {
            guard DebugLaunch.scrollToBottom else { return }
            try? await Task.sleep(for: .seconds(1))
            withAnimation { proxy.scrollTo("health", anchor: .bottom) }
        }
        }
        .scrollDismissesKeyboard(.interactively)
        // Ask Jupiter floats above the tab bar and rises with the keyboard.
        .safeAreaInset(edge: .bottom, spacing: 0) {
            JupiterDock(
                deviceName: device.name,
                onSubmit: { jupiterSheet = JupiterSheet(prompt: $0) },
                onOpenFull: { jupiterSheet = JupiterSheet(prompt: "") })
                .sdZoomSource(.jupiter, in: zoomNS, cornerRadius: 26, enabled: zoomEnabled)
                .padding(.horizontal, SD.Space.md)
                .padding(.top, SD.Space.xs)
                .padding(.bottom, SD.Space.xs)
        }
        .orynScreen()
        .navigationTitle(device.name)
        .navigationBarTitleDisplayMode(.inline)
        .modifier(HidesNavigationTitle())
        .navigationDestination(isPresented: $showHistory) {
            EventHistoryView(device: device, initialEvents: events)
        }
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Menu {
                    Button {
                        newName = device.name
                        showRename = true
                    } label: {
                        Label("Rename device", systemImage: "pencil")
                    }
                    Button {
                        showZones = true
                    } label: {
                        Label("Detection areas", systemImage: "square.dashed")
                    }
                    if state.isDemo {
                        Button {
                            showDemoTrigger = true
                        } label: {
                            Label("Run a demo alert", systemImage: "play.circle")
                        }
                    }
                } label: {
                    Image(systemName: "ellipsis")
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(SD.Colors.textPrimary)
                        .frame(width: 44, height: 44)  // the minimum touch target
                        .contentShape(Rectangle())
                }
                .accessibilityLabel("Device options")
            }
        }
        .alert("Rename device", isPresented: $showRename) {
            TextField("Name", text: $newName)
            Button("Save") {
                Task {
                    // Was `try?`: a refused or failed rename closed the alert
                    // and left the old name with no word of why.
                    do {
                        try await state.backend.renameDevice(deviceID: device.id, name: newName)
                        renameError = nil
                    } catch let error as APIError where error.isAuthFailure {
                        state.expireSession(error.message)
                    } catch {
                        renameError = "Couldn't rename the device: \(error.localizedDescription)"
                    }
                    await reload()
                }
            }
            Button("Cancel", role: .cancel) {}
        }
        .alert("Rename failed", isPresented: Binding(get: { renameError != nil }, set: { if !$0 { renameError = nil } })) {
            Button("OK", role: .cancel) { renameError = nil }
        } message: {
            Text(renameError ?? "")
        }
        .confirmationDialog(ProtectionControl.stopConfirmationTitle, isPresented: $confirmStop, titleVisibility: .visible) {
            Button("Stop monitoring", role: .destructive) { sendProtection(.stop) }
            Button("Keep monitoring", role: .cancel) {}
        } message: {
            Text(ProtectionControl.stopConfirmationMessage)
        }
        .sheet(item: $presetSheet) { preset in
            PresetDetailView(
                device: device, preset: preset,
                catalogSource: catalog?.source ?? .builtIn(reason: PresetCatalog.builtInBecauseUnreachable),
                deviceState: devicePresets?.state(for: preset.id),
                legacyUnavailableReason: devicePresets == nil
                    ? MonitoringStatusPresentation.presetUnavailableReason(preset.id, status: device.status)
                    : nil
            ) { await reload() }
            .sdZoomDestination(.mode(preset.id), in: zoomNS, enabled: zoomEnabled)
        }
        .sheet(item: $jupiterSheet) { sheet in
            AIConfigureView(device: device, initialPrompt: sheet.prompt) { await reload() }
                .sdZoomDestination(.jupiter, in: zoomNS, enabled: zoomEnabled)
        }
        .fullScreenCover(isPresented: $showCamera) {
            LiveCameraView(device: device, instrument: instrument,
                           presence: ConnectionTruth.label(presence), session: feedSession)
                .sdZoomDestination(.camera, in: zoomNS, enabled: zoomEnabled)
        }
        .sheet(isPresented: $showZones) {
            ZoneEditorView(device: device)
        }
        .sheet(item: $selectedEvent) { event in
            EventDetailView(device: device, event: event, allEvents: events)
        }
        .sheet(isPresented: $showDemoTrigger) {
            DemoTriggerView(device: device) { await reload() }
        }
        // Acknowledged only when the DEVICE confirms a stop or resume.
        .sensoryFeedback(.success, trigger: protectionNotice?.at) { _, new in new != nil }
        .onChange(of: instrument.kind) { old, new in
            // One quiet acknowledgement when monitoring becomes ready again —
            // never on first load, never for anything short of ready.
            if new == .ready, old != .ready, old != .connecting, !reduceMotion { readyAck += 1 }
        }
        // A tapped notification opens its alert, which is the last beat of the
        // demo: buzz, tap, "here is why".
        // A notification can be tapped before this screen has ever loaded the
        // event it's about — AlertWatcher polls independently, every 15s,
        // from wherever the user happens to be in the app, so its fetch
        // routinely gets there before this view's own events array does.
        // Looking up `id` in whatever's currently loaded and giving up on a
        // miss meant tapping the notification did nothing at all: no sheet,
        // no error, no retry — the exact same "silently does nothing" shape
        // as the mode-switching bugs elsewhere in this screen. One reload
        // and a second look covers the ordinary case; if it's still not
        // found the event is outside the 200-item window this screen loads,
        // which is worth noting rather than swallowing silently.
        // `initial: true`: this view can be pushed FRESH by a notification tap
        // (DevicesView's own handler navigates here via .navigationDestination
        // after matching the device), which means pendingEventID is already
        // non-nil the moment this onChange modifier first attaches — and
        // onChange only fires on a value actually CHANGING after that point,
        // never for the value it already holds. Without `initial: true` this
        // closure silently never ran for that id: no sheet, no error, no
        // notifier.pendingEventID reset — the exact "last beat of the demo"
        // this comment above already worried about, just from one screen
        // earlier than where the retry logic was looking.
        .onChange(of: notifier.pendingEventID, initial: true) { _, id in
            guard let id else { return }
            // Only this device's alerts. With device X open, a tap on an alert
            // for device Y used to be consumed here: X searched its own events,
            // failed, showed "Couldn't open that alert" and cleared the id —
            // racing the screen DevicesView was pushing for Y.
            if let target = notifier.pendingDeviceID, target != device.id { return }
            alertOpenProblem = nil
            if let match = events.first(where: { $0.eventID == id }) {
                selectedEvent = match
                notifier.pendingEventID = nil
                return
            }
            Task {
                await reload()
                if let match = events.first(where: { $0.eventID == id }) {
                    selectedEvent = match
                } else {
                    alertOpenProblem = "It may be older than the alerts this screen loads. Open Activity to find it."
                }
                notifier.pendingEventID = nil
            }
        }
        .task {
            await notifier.refreshAuthorization()
            if !notifier.authorized { await notifier.requestPermission() }
        }
        .refreshable { await reload() }
        .task {
            await reload()
            if DebugLaunch.openJupiter {
                jupiterSheet = JupiterSheet(prompt: DebugLaunch.jupiterPrompt ?? "")
            } else if DebugLaunch.openHistory {
                showHistory = true
            } else if DebugLaunch.openCamera || DebugLaunch.openMode != nil {
                // A beat on the device screen first, so a recording shows the
                // zoom out of its source.
                try? await Task.sleep(for: .seconds(1.5))
                if DebugLaunch.openCamera {
                    openCamera()
                } else if let id = DebugLaunch.openMode, let preset = libraryCatalog.preset(id) {
                    presetSheet = preset
                }
            }
        }
        .onReceive(refreshTimer) { tick in
            now = tick
            // One timed reload at a time. Each is two sequential requests; on a
            // link slower than the 4s tick every reload was superseded by the
            // next before it landed, so LatestRequestGuard discarded them all
            // and the screen never updated while requests piled up.
            guard !timedReloadInFlight, state.periodicRefreshMayRun else { return }
            timedReloadInFlight = true
            Task {
                await reload()
                timedReloadInFlight = false
            }
        }
    }

    // MARK: Banners

    /// Above the camera, not below the fold: this view polls every 4s and keeps
    /// showing the last frame and event list when a refresh fails.
    @ViewBuilder private var banners: some View {
        let notReporting = hasLoaded ? ConnectionTruth.notReportingMessage(presence) : nil
        if errorMessage != nil || notReporting != nil || alertOpenProblem != nil {
            VStack(spacing: SD.Space.sm) {
                if let errorMessage {
                    ConnectionBanner(message: errorMessage, title: errorTitle) { await reload() }
                } else if let message = notReporting {
                    // The server answers but the device's telemetry is not live:
                    // everything below is the last thing it said, and changes
                    // wait for it.
                    ConnectionBanner(
                        message: message,
                        title: isNotRefreshed
                            ? "This screen isn't up to date"
                            : "\(device.name) isn't reporting"
                    ) { await reload() }
                }
                if let alertOpenProblem {
                    ConnectionBanner(message: alertOpenProblem, title: "Couldn't open that alert")
                }
            }
            .padding(.top, SD.Space.xs)
            .padding(.bottom, SD.Space.xs)
            .transition(.opacity)
        }
    }

    // MARK: Status

    /// Presence, monitoring state and a recent alert, as one readout.
    private var instrument: InstrumentState {
        InstrumentState.make(presence: presence, monitoring: monitoringPresentation,
                             alertActive: alertActive, loaded: hasLoaded)
    }

    /// iOS 18 zoom transitions, unless the reader asked for less motion.
    private var zoomEnabled: Bool { !reduceMotion }

    private func openCamera() {
        // Under Reduce Motion the camera appears at once rather than sliding
        // up: a spatial transition becomes an instant change.
        var transaction = Transaction()
        transaction.disablesAnimations = reduceMotion
        withTransaction(transaction) { showCamera = true }
    }

    /// The colour of the readout's symbol. By kind, not tone alone: a stopped
    /// device is not "caution amber" on the instrument, it is simply off.
    private func readoutTint(_ kind: InstrumentState.Kind) -> Color {
        switch kind {
        case .ready: SD.Colors.success
        case .alert, .failed: SD.Colors.danger
        case .starting, .degraded, .stale: SD.Colors.warning
        case .stopped: SD.Colors.textSecondary
        case .offline, .unknown, .connecting: SD.Colors.textTertiary
        }
    }

    /// The sentence under the readout: the device's own words for its state.
    private var readoutDetail: String? {
        if alertActive, let latest = events.first(where: { $0.isCritical || $0.isConfigured }) {
            return "\(latest.title) — in the last minute. Open Activity for why."
        }
        if instrument.kind == .connecting { return "Connecting to \(device.name)…" }
        return monitoringPresentation?.detail
    }

    /// The mode the device is running, as the line above the name says it.
    private var modeLine: String {
        // Rules running ALONGSIDE the mode leave the mode in charge, so the
        // mode is what this line names.
        customReplacesModes ? (device.monitoring?.title ?? "AI Custom Monitoring") : presetTitle(device.activePreset)
    }

    /// The status word and its symbol: one element whose label is the word
    /// (what VoiceOver and the journeys read) and whose value is the device's
    /// sentence.
    private func readout(_ state: InstrumentState, wraps: Bool) -> some View {
        HStack(spacing: SD.Space.xs) {
            Image(systemName: state.symbol)
                .font(.system(size: 19, weight: .medium))
                .foregroundStyle(readoutTint(state.kind))
                .contentTransition(.symbolEffect(.replace))
                .symbolEffect(.bounce, value: readyAck)
            Text(state.word)
                .font(SD.Typography.readout)
                .foregroundStyle(SD.Colors.textPrimary)
                .contentTransition(.opacity)
                .fixedSize(horizontal: !wraps, vertical: true)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(state.word)
        .accessibilityValue(readoutDetail ?? "")
        .accessibilityIdentifier("detail.monitoring")
    }

    /// The device's identity and its one status readout, with the action the
    /// state calls for beside it (Stop while monitoring, Resume when stopped).
    private var instrumentPanel: some View {
        let state = instrument
        let accessible = typeSize.isAccessibilitySize
        return VStack(alignment: .leading, spacing: SD.Space.sm) {
            // Presence and mode on one quiet line. Stacked at accessibility
            // sizes, where one line truncated both.
            (accessible
                ? AnyLayout(VStackLayout(alignment: .leading, spacing: SD.Space.xs))
                : AnyLayout(HStackLayout(spacing: SD.Space.xs))) {
                SDStatusPill(status: presenceStatus, text: ConnectionTruth.label(presence))
                    .accessibilityLabel(ConnectionTruth.spoken(presence))
                    .accessibilityIdentifier("detail.online")
                HStack(spacing: 5) {
                    if customMonitoringActive {
                        Image(systemName: "sparkles")
                            .font(.system(size: 10, weight: .semibold))
                            .accessibilityHidden(true)
                    }
                    Text(modeLine)
                        .font(SD.Typography.caption.weight(.medium))
                        .fixedSize(horizontal: false, vertical: true)
                }
                .foregroundStyle(customMonitoringActive ? SD.Colors.textPrimary : SD.Colors.textSecondary)
                if !accessible { Spacer(minLength: 0) }
            }

            Text(device.name)
                .font(SD.Typography.title)
                .foregroundStyle(SD.Colors.textPrimary)
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityAddTraits(.isHeader)

            // Side by side when the word and the control fit on one line;
            // stacked otherwise ("Status unknown" beside "Stop monitoring"
            // wrapped the word) and always at accessibility sizes.
            if accessible {
                VStack(alignment: .leading, spacing: SD.Space.sm) {
                    readout(state, wraps: true)
                    protectionControl
                }
            } else {
                ViewThatFits(in: .horizontal) {
                    HStack(alignment: .center, spacing: SD.Space.sm) {
                        readout(state, wraps: false)
                        Spacer(minLength: SD.Space.xs)
                        protectionControl.fixedSize()
                    }
                    VStack(alignment: .leading, spacing: SD.Space.sm) {
                        readout(state, wraps: true)
                        protectionControl
                    }
                }
            }

            if let sentence = readoutDetail {
                Text(sentence)
                    .font(SD.Typography.callout)
                    .foregroundStyle(SD.Colors.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityHidden(true)  // the readout's value says it
            }

            // A component the modes rely on isn't working: said up here, in
            // the device's words, not only in Device health below.
            if let impaired = PerceptionPresentation.degradedSummary(device.status?.perception, live: deviceLive) {
                Label(impaired, systemImage: "exclamationmark.triangle")
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.warning)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .accessibilityIdentifier("perception.degraded")
            }

            if let note = protectionNote {
                Text(note.text)
                    .font(SD.Typography.caption)
                    .foregroundStyle(note.color)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            if state.showsProgress {
                ConnectingHairline()
                    .transition(.opacity)
            }
        }
        .animation(SD.Motion.resolved(SD.Motion.status, reduceMotion: reduceMotion), value: state.kind)
    }

    /// ONLINE only with fresh telemetry in a reply from a refresh that
    /// succeeded and is current (ConnectionTruth). Everything on this screen
    /// that says "live" reads this, so no two parts of it can disagree.
    private var presence: ConnectionTruth.Presence {
        device.presence(now: now, lastRefreshFailed: errorMessage != nil)
    }

    private var isNotRefreshed: Bool {
        if case .unknown(.notRefreshed) = presence { return true }
        return false
    }

    private var presenceStatus: SDStatus { SDStatus(presence) }

    /// The monitoring state, from `status` — "Status unknown" whenever the
    /// telemetry is not live, so "All clear" can't stand on an old reply.
    private var monitoringPresentation: MonitoringStatusPresentation? {
        device.monitoringPresentation(live: deviceLive)
    }

    /// Device health, folded to one line until someone opens it: the status
    /// readout above already says whether it is monitoring, so the component
    /// detail is for whoever goes looking.
    private var healthSection: some View {
        VStack(alignment: .leading, spacing: SD.Space.sm) {
            Button {
                withAnimation(SD.Motion.resolved(SD.Motion.spatial, reduceMotion: reduceMotion)) {
                    healthOverride = !showHealth
                }
            } label: {
                HStack(alignment: .center, spacing: SD.Space.sm) {
                    VStack(alignment: .leading, spacing: 3) {
                        SDEyebrow("Device health")
                        Text(healthSummary)
                            .font(SD.Typography.callout)
                            .foregroundStyle(SD.Colors.textSecondary)
                            .multilineTextAlignment(.leading)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    Spacer(minLength: SD.Space.xs)
                    Image(systemName: "chevron.down")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(SD.Colors.textTertiary)
                        .rotationEffect(.degrees(showHealth ? 180 : 0))
                }
                .frame(minHeight: 44)
                .contentShape(Rectangle())
            }
            .buttonStyle(SDUndimmedButtonStyle())
            .accessibilityLabel("Device health")
            .accessibilityValue(healthSummary)
            .accessibilityHint(showHealth ? "Hides the details" : "Shows camera, AI, network and detection capabilities")
            .accessibilityAddTraits(.isHeader)
            .accessibilityIdentifier("detail.health")

            if showHealth {
                healthCard
                    .transition(.opacity)
            }
        }
    }

    private var showHealth: Bool { healthOverride ?? healthNeedsAttention }

    /// Anything short of "reporting, and every component working".
    private var healthNeedsAttention: Bool {
        guard deviceLive, let health = device.health else { return true }
        let parts = [health.camera, health.accelerator, monitoringComponent(health), health.network].compactMap { $0 }
        if parts.contains(where: { $0.state != "ok" }) { return true }
        return PerceptionPresentation.degradedSummary(device.status?.perception, live: deviceLive) != nil
    }

    /// One line from what the device reported this cycle: how many of its
    /// components say they are working. Counted, never estimated.
    private var healthSummary: String {
        guard deviceLive else { return "Appears while the device is reporting." }
        guard let health = device.health else { return "This device isn't reporting its health yet." }
        let parts = [health.camera, health.accelerator, monitoringComponent(health), health.network].compactMap { $0 }
        guard !parts.isEmpty else { return "No components reported." }
        let notOK = parts.filter { $0.state != "ok" }.count
        let fps = health.camera?.metric("fps").map { " · camera \($0) fps" } ?? ""
        return notOK == 0
            ? "All \(parts.count) components working\(fps)"
            : "\(notOK) of \(parts.count) components need attention\(fps)"
    }

    // MARK: Health
    //
    // Every row here is something the Pi observed and sent up this cycle. When
    // the device stops reporting the backend withholds the whole report, so
    // these disappear rather than freezing on their last happy values.

    @ViewBuilder private var healthCard: some View {
        VStack(alignment: .leading, spacing: SD.Space.sm) {
            if let health = device.health, deviceLive {
                healthRow("Camera", systemImage: "camera", health.camera, metric: "fps", suffix: " fps")
                OrynRowDivider(inset: 22 + SD.Space.sm)
                healthRow("AI", systemImage: "cpu", health.accelerator)
                OrynRowDivider(inset: 22 + SD.Space.sm)
                healthRow("Monitoring", systemImage: "eye", monitoringComponent(health))
                OrynRowDivider(inset: 22 + SD.Space.sm)
                healthRow("Network", systemImage: "wifi", health.network)
            } else {
                // Online but silent about itself: an older device build that
                // predates health reporting. Offline: the backend withholds a
                // stale report. Either way, say so instead of leaving a gap.
                HStack(alignment: .top, spacing: SD.Space.sm) {
                    Image(systemName: "info.circle")
                        .foregroundStyle(SD.Colors.textTertiary)
                    Text(deviceLive
                         ? "This device isn't reporting its status yet. Update it to see camera and AI health here."
                         : "Health appears here while the device is reporting.")
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textTertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            capabilitiesCard
        }
    }

    /// What the device can detect right now, from `status.capabilities`:
    /// available, starting, or impaired with the device's reason, plus how far
    /// each has been evaluated when the backend says. Empty, with a sentence,
    /// whenever the device isn't reporting: an old list is never shown green.
    @ViewBuilder private var capabilitiesCard: some View {
        let health = CapabilityHealth.rows(status: device.status, online: deviceLive)
        SDEyebrow("Detection capabilities")
            .accessibilityAddTraits(.isHeader)
            .padding(.top, SD.Space.md)
        VStack(alignment: .leading, spacing: SD.Space.sm) {
            if let notice = health.notice {
                HStack(alignment: .top, spacing: SD.Space.sm) {
                    Image(systemName: "questionmark.circle")
                        .foregroundStyle(SD.Colors.textTertiary)
                    Text(notice)
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textTertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            // The shared perception engine's components, as the device
            // reports them: one model run feeds every mode that needs it.
            let components = PerceptionPresentation.componentRows(device.status?.perception, live: deviceLive)
            ForEach(components) { row in
                healthRow(row.title, systemImage: "cpu",
                          HealthComponent(state: row.tone.healthState, detail: row.detail, metrics: nil))
                    .accessibilityIdentifier("perception.\(row.name)")
                OrynRowDivider(inset: 22 + SD.Space.sm)
            }
            ForEach(Array(health.rows.enumerated()), id: \.element.id) { index, row in
                healthRow(row.title, systemImage: capabilityIcon(row.state),
                          HealthComponent(state: row.tone.healthState, detail: row.detail, metrics: nil))
                if index < health.rows.count - 1 {
                    OrynRowDivider(inset: 22 + SD.Space.sm)
                }
            }
            if !health.rows.isEmpty {
                Text("“Producing results” means the capability is running, not that its results are accurate.")
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textTertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func capabilityIcon(_ state: CapabilityHealthRow.State) -> String {
        switch state {
        case .available: "waveform.path.ecg"
        case .starting: "hourglass"
        case .impaired: "exclamationmark.triangle"
        }
    }

    /// The "Monitoring" row: the device's monitoring state (DeviceStatus), in
    /// the health card's own row format. It used to show the inference loop's
    /// health, which reads "ok" on a device that is stopped, or failed because
    /// the face model is gone. The inference component remains the fallback
    /// for a backend that predates the status contract.
    private func monitoringComponent(_ health: DeviceHealthReport) -> HealthComponent? {
        guard let presentation = monitoringPresentation else { return health.inference }
        return HealthComponent(state: presentation.tone.healthState, detail: presentation.detail, metrics: nil)
    }

    @ViewBuilder
    private func healthRow(
        _ title: String,
        systemImage: String,
        _ component: HealthComponent?,
        metric: String? = nil,
        suffix: String = ""
    ) -> some View {
        if let component {
            HStack(spacing: SD.Space.sm) {
                Image(systemName: systemImage)
                    .font(.system(size: 14))
                    .frame(width: 22)
                    .foregroundStyle(SD.Colors.textTertiary)
                VStack(alignment: .leading, spacing: 1) {
                    Text(title)
                        .font(SD.Typography.body)
                        .foregroundStyle(SD.Colors.textPrimary)
                    // The device's own sentence — "running on the Hailo-8
                    // accelerator", "no new frame for 12.3s" — rather than a
                    // word the app invented on its behalf.
                    Text(component.detail)
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textSecondary)
                        .lineLimit(2)
                }
                Spacer(minLength: SD.Space.xxs)
                if let metric, let value = component.metric(metric) {
                    Text(value + suffix)
                        .font(SD.Typography.caption.monospacedDigit())
                        .foregroundStyle(SD.Colors.textSecondary)
                }
                if component.isKnown {
                    Circle().fill(component.status.tint).frame(width: 7, height: 7)
                } else {
                    Image(systemName: "questionmark.circle")
                        .font(.caption)
                        .foregroundStyle(SD.Colors.textTertiary)
                }
            }
            .padding(.vertical, 2)
            .accessibilityElement(children: .combine)
            .accessibilityLabel("\(title): \(component.detail)")
        }
    }

    // MARK: Actions
    //
    // Ask Jupiter is the floating dock (JupiterDock): the same flow underneath.
    // The request goes to `/v1/ai/configure`, comes back as a draft, and nothing
    // changes on the device until the person reviews it and taps Activate in
    // AIConfigureView.

    /// The places you go FROM a device, as keys on one row: what it has seen,
    /// where rules apply, and the report. A row of three, stacked at
    /// accessibility sizes.
    private var actionKeys: some View {
        let layout = typeSize.isAccessibilitySize
            ? AnyLayout(VStackLayout(spacing: SD.Space.xs))
            : AnyLayout(HStackLayout(spacing: SD.Space.xs))
        return layout {
            Button { alertOpenProblem = nil; showHistory = true } label: {
                InstrumentKey(icon: "clock.arrow.circlepath", title: "Activity",
                              badge: alertActive ? "Alert" : nil, badgeColor: SD.Colors.danger)
            }
            .buttonStyle(SDPressStyle())
            .accessibilityLabel(alertActive ? "Activity, alert in the last minute" : "Activity")
            .accessibilityValue(historyDetail)
            .accessibilityIdentifier("detail.history")

            Button { showZones = true } label: {
                InstrumentKey(icon: "square.dashed", title: "Areas")
            }
            .buttonStyle(SDPressStyle())
            .accessibilityLabel("Detection areas")
            .accessibilityHint("Draw where rules apply")
            .accessibilityIdentifier("detail.zones")

            NavigationLink {
                ReportsView(device: device)
            } label: {
                InstrumentKey(icon: "chart.bar", title: "Report")
            }
            .buttonStyle(SDPressStyle())
            .accessibilityLabel("Report")
            .accessibilityHint("Estimates from recent events")

            if state.isDemo {
                Button { showDemoTrigger = true } label: {
                    InstrumentKey(icon: "play.circle", title: "Demo alert")
                }
                .buttonStyle(SDPressStyle())
                .accessibilityLabel("Run a demo alert")
                .accessibilityHint("Simulated: see an alert end to end")
            }
        }
    }

    /// The three latest events on the timeline spine; all of them one tap away.
    private var recentActivity: some View {
        let recent = Array(events.prefix(3))
        return VStack(alignment: .leading, spacing: SD.Space.sm) {
            SDSectionHeader(title: "Recent activity",
                            action: events.isEmpty ? nil : ("See all", { alertOpenProblem = nil; showHistory = true }))
            if recent.isEmpty {
                Text(hasLoaded
                     ? "Nothing to report yet. Events appear here as soon as \(device.name) sees something worth telling you about."
                     : "Loading…")
                    .font(SD.Typography.callout)
                    .foregroundStyle(SD.Colors.textTertiary)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                VStack(alignment: .leading, spacing: 0) {
                    ForEach(Array(recent.enumerated()), id: \.element.id) { index, event in
                        TimelineRow(event: event, isLast: index == recent.count - 1,
                                    expanded: expandedEventID == event.id, relativeTime: true,
                                    onToggle: { expandedEventID = expandedEventID == event.id ? nil : event.id },
                                    onExplain: { selectedEvent = event })
                    }
                }
                .animation(SD.Motion.resolved(SD.Motion.standard, reduceMotion: reduceMotion), value: expandedEventID)
            }
        }
    }

    /// One honest line about the history: the latest event, or that there is none.
    private var historyDetail: String {
        if !hasLoaded && events.isEmpty { return "Loading…" }
        guard let latest = events.first else { return "Nothing to report yet" }
        let title = latest.type.replacingOccurrences(of: "_", with: " ").capitalized
        if let date = latest.date {
            return "\(title) · \(Self.relative.localizedString(for: date, relativeTo: .now))"
        }
        return title
    }

    private static let relative: RelativeDateTimeFormatter = {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .short
        return formatter
    }()

    // MARK: Mode

    /// Whether custom rules run on the device at all (replacing the mode, or
    /// beside it on an additive role). From the mode the device reports.
    private var customMonitoringActive: Bool { device.customRules != .none }

    /// Whether custom rules have REPLACED the base mode (AI Custom). False on an
    /// additive role, where they run alongside the mode and the mode stays on.
    private var customReplacesModes: Bool { device.customRules == .replacing }

    private var activeBase: String { Preset.split(device.activePreset).base }

    /// What the running mode's row says when the device is NOT monitoring
    /// (failed or stopped); nil when it is (or when there is no status).
    private var notMonitoringReason: String? {
        guard let presentation = monitoringPresentation else { return nil }
        switch presentation.phase {
        case .failed: return presentation.detail
        case .stopped: return "Stopped — nothing is monitored until it's resumed"
        default: return nil
        }
    }
    /// The status can't be confirmed (stale health, unknown): the row the
    /// device runs must not read green (M8.1 frozen-health finding).
    private var unconfirmedReason: String? {
        guard let presentation = monitoringPresentation, presentation.phase == .unknown else { return nil }
        return "Status unknown — \(presentation.detail)"
    }
    private var activeVariant: String? { Preset.split(device.activePreset).variant }

    /// The mode leading the library: the one the device runs, else the
    /// flagship (Driver Safety, the Distracted Driving product), else the first.
    private var featuredModeID: String? {
        let presets = libraryCatalog.presets
        if !customReplacesModes, presets.contains(where: { $0.id == activeBase }) { return activeBase }
        if presets.contains(where: { $0.id == "driver_safety" }) { return "driver_safety" }
        return presets.first?.id
    }

    /// The library, edited rather than listed: the featured mode first and in
    /// full, the modes this device can run next, and the ones its role rules
    /// out last and quieter, each still openable to read why.
    private var modeLibrary: some View {
        let presets = libraryCatalog.presets
        let featured = presets.first { $0.id == featuredModeID }
        let rest = presets.filter { $0.id != featuredModeID }
        let available = rest.filter { unavailableReason($0.id) == nil }
        let unavailable = rest.filter { unavailableReason($0.id) != nil }
        return VStack(alignment: .leading, spacing: SD.Space.md) {
            SDSectionHeader(title: "Monitoring modes")

            // What is ACTUALLY running, when it is not one of the presets
            // below. The device reports this itself, so it cannot disagree
            // with what the device is doing.
            if customMonitoringActive, let custom = device.monitoring {
                VStack(alignment: .leading, spacing: SD.Space.xs) {
                    HStack(alignment: .top, spacing: SD.Space.sm) {
                        OrynMark(height: 16, color: SD.Colors.signal)
                            .frame(width: 28, height: 22)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(custom.title)
                                .font(SD.Typography.headline)
                                .foregroundStyle(SD.Colors.textPrimary)
                            Text(customReplacesModes
                                 ? "Set by Jupiter · replaces the modes below"
                                 : "Set by Jupiter · runs alongside the mode below")
                                .font(SD.Typography.caption)
                                .foregroundStyle(SD.Colors.textSecondary)
                            Text(custom.summary)
                                .font(SD.Typography.callout)
                                .foregroundStyle(SD.Colors.textSecondary)
                                .fixedSize(horizontal: false, vertical: true)
                                .padding(.top, SD.Space.xxs)
                            // Alongside a mode, picking the mode again leaves the
                            // rules on (measured against the backend: rules_active
                            // stays 1); their own switches under Custom rules are
                            // what turn them off.
                            Text(customReplacesModes
                                 ? "Pick a mode below to go back to standard monitoring."
                                 : "Turn these rules off with their switches under Custom rules.")
                                .font(SD.Typography.caption)
                                .foregroundStyle(SD.Colors.textTertiary)
                                .fixedSize(horizontal: false, vertical: true)
                                .padding(.top, SD.Space.xxs)
                        }
                        Spacer(minLength: 0)
                    }
                }
                .padding(SD.Space.md)
                .background(
                    RoundedRectangle(cornerRadius: SD.Radius.camera, style: .continuous)
                        .fill(SD.Colors.surface)
                )
            }

            // The device refused the latest rules (all or nothing) and kept
            // what it was running. They are not shown as running anywhere.
            if let notice = MonitoringStatusPresentation.rejectedRulesNotice(device.status) {
                Label(notice, systemImage: "exclamationmark.triangle")
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.danger)
                    .fixedSize(horizontal: false, vertical: true)
            }

            // The built-in list says so: it knows names, not what this
            // device can run or how far anything has been validated.
            if case .builtIn(let reason) = libraryCatalog.source {
                Label(reason, systemImage: "info.circle")
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textTertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if let featured {
                modeRow(featured, featured: true)
            }

            if !available.isEmpty {
                VStack(alignment: .leading, spacing: 0) {
                    ForEach(available) { preset in
                        modeRow(preset, featured: false)
                        if preset.id != available.last?.id { OrynRowDivider(inset: 28 + SD.Space.sm) }
                    }
                }
            }

            if !unavailable.isEmpty {
                VStack(alignment: .leading, spacing: SD.Space.xxs) {
                    SDEyebrow("Not available on this device")
                        .padding(.top, SD.Space.xs)
                    ForEach(unavailable) { preset in
                        modeRow(preset, featured: false)
                        if preset.id != unavailable.last?.id { OrynRowDivider(inset: 28 + SD.Space.sm) }
                    }
                }
            }

            nodesSection
        }
    }

    // MARK: Monitoring nodes

    private var perceptionReport: PerceptionReport? { deviceLive ? device.status?.perception : nil }

    /// What the device reports running on its shared perception engine (the
    /// mode, companion nodes, custom rules), and the occupancy companion's
    /// switch. Only shown when the device reports nodes at all.
    @ViewBuilder private var nodesSection: some View {
        if let report = perceptionReport {
            VStack(alignment: .leading, spacing: SD.Space.xs) {
                SDEyebrow(report.simulated == true ? "Running on this device · simulated" : "Running on this device")
                ForEach(report.nodes.filter { $0.isRunning || $0.state == "failed" }) { node in
                    nodeRow(node)
                }
                occupancyToggle(report)
            }
            // Its own element, so the id names the section without replacing the
            // rows' and the switch's own ids (journey 24 found no switch).
            .accessibilityElement(children: .contain)
            .accessibilityIdentifier("nodes.section")
        }
    }

    private func nodeRow(_ node: MonitoringNode) -> some View {
        let shown = NodePresentation.make(node)
        return HStack(alignment: .firstTextBaseline, spacing: SD.Space.sm) {
            Circle().fill(shown.tone.sdStatus.tint).frame(width: 7, height: 7)
            VStack(alignment: .leading, spacing: 1) {
                Text(shown.title)
                    .font(SD.Typography.body)
                    .foregroundStyle(SD.Colors.textPrimary)
                Text(shown.stateText)
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                if let experimental = NodePresentation.experimentalNote(node) {
                    Text(experimental)
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.warning)
                        .fixedSize(horizontal: false, vertical: true)
                }
                // How far it is validated, as the device reports it: said once,
                // quietly, under the name (the mode library's treatment).
                Label(shown.badge, systemImage: "seal")
                    .font(SD.Typography.caption.weight(.medium))
                    .foregroundStyle(SD.Colors.warning)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.top, 2)
            }
            Spacer(minLength: 0)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(shown.title): \(shown.stateText). \(shown.badge)"
            + (NodePresentation.experimentalNote(node).map { ". \($0)" } ?? ""))
        .accessibilityIdentifier("node.\(node.id)")
    }

    /// The occupancy companion runs beside the mode on the same detections
    /// (no extra model). The switch shows what the device reports.
    @ViewBuilder private func occupancyToggle(_ report: PerceptionReport) -> some View {
        let reported = CompanionNodeToggle.isOn("people_counter", report: report, live: deviceLive) ?? false
        let pending = nodeRequest.map { $0.id == "people_counter"
            && !CompanionNodeToggle.confirmed("people_counter", wanted: $0.wanted, report: report, live: deviceLive)
            && Date().timeIntervalSince($0.at) < ModeSwitchState.switchTimeout } ?? false
        VStack(alignment: .leading, spacing: SD.Space.xxs) {
            Toggle(isOn: Binding(get: { pending ? (nodeRequest?.wanted ?? reported) : reported },
                                 set: { setNode("people_counter", enabled: $0) })) {
                VStack(alignment: .leading, spacing: 1) {
                    Text("Also count people")
                        .font(SD.Typography.body)
                        .foregroundStyle(SD.Colors.textPrimary)
                    // How far it is validated comes from the device's report when it
                    // is running (it said "Experimental" here while the device called
                    // it "Not yet validated · simulated tests"); never "validated".
                    Text(pending ? "Waiting for \(device.name) to confirm…"
                         : "Occupancy, from the same detections. "
                            + (report.node("people_counter").map { NodePresentation.badge($0) + "." }
                               ?? "Not yet validated."))
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .tint(SD.Colors.accent)
            .disabled(pending || !deviceLive)
            .accessibilityIdentifier("node.toggle.people_counter")
            if let message = nodeError ?? DeviceRefusal.forNode("people_counter", status: device.status,
                                                                since: nodeRequest?.at) {
                Text(message)
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.danger)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("node.error.people_counter")
            }
        }
    }

    private func setNode(_ id: String, enabled: Bool) {
        nodeError = nil
        nodeRequest = (id, enabled, Date())
        Task {
            do {
                try await state.backend.setNode(deviceID: device.id, node: id, enabled: enabled)
                await reload()
            } catch let error as APIError where error.isAuthFailure {
                state.expireSession(error.message)
            } catch let error as APIError {
                nodeRequest = nil
                nodeError = error.endpointMissing ? ServerSupport.notSupported
                    : RefusalText.message(code: error.code, reason: error.message, fallback: error.message)
            } catch {
                nodeRequest = nil
                nodeError = error.localizedDescription
            }
        }
    }

    /// The single source of truth for a mode row's state — see
    /// ModeSwitchState.swift for why "active"/"switching" alone weren't
    /// enough: no distinct failure state, no offline state, no timeout.
    /// `now: .now` is what lets a stuck switch age past switchTimeout and
    /// flip to failed on the app's own 4s refresh cycle, with no separate
    /// timer needed.
    /// Only custom rules that REPLACE the modes stop a mode row reading active;
    /// rules running alongside (an additive role) leave the mode running.
    private func rowState(_ preset: CatalogPreset) -> ModeRowState {
        ModeSwitchState.state(
            for: preset.id, activeBase: activeBase, customMonitoringActive: customReplacesModes,
            pending: pendingSwitch, deviceOnline: deviceLive, failure: failedSwitch, now: .now,
            unavailable: unavailableReason(preset.id), notMonitoring: notMonitoringReason,
            unconfirmed: unconfirmedReason
        )
    }

    private func validation(_ preset: CatalogPreset) -> ValidationPresentation? {
        libraryCatalog.isBuiltIn ? nil
            : ValidationPresentation.make(devicePresets?.state(for: preset.id)?.validationStatus ?? preset.validation.status)
    }

    private func modeRow(_ preset: CatalogPreset, featured: Bool) -> some View {
        let state = rowState(preset)
        return VStack(alignment: .leading, spacing: SD.Space.sm) {
            HStack(alignment: featured ? .top : .center, spacing: SD.Space.xxs) {
                modeRowButton(preset, state: state, featured: featured)
                modeInfoButton(preset)
            }
            // Tuning variants, shown only for the mode currently running.
            if state.isSelected && !preset.variants.isEmpty {
                variantChips(preset, featured: featured)
            }
        }
        .padding(featured ? SD.Space.md : 0)
        .background {
            if featured {
                RoundedRectangle(cornerRadius: SD.Radius.camera, style: .continuous)
                    .fill(SD.Colors.surface)
            }
        }
        .sdZoomSource(.mode(preset.id), in: zoomNS,
                      cornerRadius: featured ? SD.Radius.camera : SD.Radius.lg, enabled: zoomEnabled)
    }

    /// One activation path: the row opens the preset's details, where its
    /// capabilities (from the device), validation and settings are shown and
    /// Activate waits for the device's own confirmation. A bare tap used to
    /// switch at once, sending the base name and so dropping a running variant.
    private func modeRowButton(_ preset: CatalogPreset, state: ModeRowState, featured: Bool) -> some View {
        Button { presetSheet = preset } label: {
            Group {
                if featured {
                    // The featured mode reads as a title block: its symbol and
                    // state on one line, then its name and words at full width.
                    VStack(alignment: .leading, spacing: SD.Space.sm) {
                        HStack(alignment: .center, spacing: SD.Space.sm) {
                            modeGlyph(preset.id, featured: true, active: state.isSelected,
                                      unavailable: state.isUnavailable)
                            modeStateGlyph(state)
                            Spacer(minLength: 0)
                        }
                        modeRowText(preset, state: state, featured: true)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                } else {
                    HStack(alignment: .center, spacing: SD.Space.sm) {
                        modeGlyph(preset.id, featured: false, active: state.isSelected,
                                  unavailable: state.isUnavailable)
                        modeRowText(preset, state: state, featured: false)
                        Spacer(minLength: SD.Space.xxs)
                        modeStateGlyph(state)
                    }
                    .padding(.vertical, SD.Space.sm)
                }
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(SDUndimmedButtonStyle())
        // Opens even when the preset can't be picked, so the reason can be
        // read in full; the sheet's Activate is what is disabled.
        // The row's state was a checkmark or spinner alone: said here too.
        .accessibilityValue(state.spokenValue)
        .accessibilityHint("Opens what it detects, its settings and Activate")
        .accessibilityAddTraits(state.isSelected ? .isSelected : [])
        .accessibilityIdentifier("preset.row.\(preset.id)")
    }

    private func modeRowText(_ preset: CatalogPreset, state: ModeRowState, featured: Bool) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            if featured, let context = ModeEditorial.context(preset.id) {
                SDEyebrow(context, color: SD.Colors.textTertiary)
                    .padding(.bottom, 2)
            }
            // Unavailable reads quieter than the others, at a contrast that can
            // still be read (textDisabled is 3.9:1).
            Text(preset.title)
                .font(featured ? SD.Typography.title : SD.Typography.body)
                .foregroundStyle(state.isUnavailable ? SD.Colors.textTertiary : SD.Colors.textPrimary)
                .fixedSize(horizontal: false, vertical: true)
            modeRowDetail(preset, state: state, featured: featured)
            // How far it has been validated, from the catalog. Only "validated
            // on device" is ever green; said quietly, once.
            // Not for a mode this device can't run: its reason is the news.
            if let validation = validation(preset), !state.isUnavailable {
                Label(validation.chip, systemImage: validation.isValidated ? "checkmark.seal" : "seal")
                    .font(SD.Typography.caption.weight(.medium))
                    .foregroundStyle(validation.isValidated ? SD.Colors.success : SD.Colors.warning)
                    .padding(.top, 2)
            }
        }
    }

    /// The line under a mode's name: its state when that is the news, else what
    /// it watches for.
    @ViewBuilder
    private func modeRowDetail(_ preset: CatalogPreset, state: ModeRowState, featured: Bool) -> some View {
        switch state {
        case .failed(let message):
            detailLine(message, SD.Colors.danger)
        case .unavailable(let reason):
            // textSecondary: the reason is the point of the row, and
            // textTertiary at caption size failed the audit's contrast.
            detailLine(reason, SD.Colors.textSecondary)
        case .queuedOffline:
            detailLine("Device offline — will apply when it reconnects", SD.Colors.warning)
        case .activeNotMonitoring(let message):
            detailLine(message, SD.Colors.danger)
        case .activeUnconfirmed(let message):
            detailLine(message, SD.Colors.warning)
        case .switching:
            detailLine("Switching — waiting for \(device.name) to confirm", SD.Colors.textSecondary)
        case .inactive, .active:
            if !preset.description.isEmpty {
                Text(preset.description)
                    .font(featured ? SD.Typography.callout : SD.Typography.caption)
                    .foregroundStyle(featured ? SD.Colors.textSecondary : SD.Colors.textTertiary)
                    .multilineTextAlignment(.leading)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func detailLine(_ text: String, _ color: Color) -> some View {
        Text(text)
            .font(SD.Typography.caption)
            .foregroundStyle(color)
            .multilineTextAlignment(.leading)
            .fixedSize(horizontal: false, vertical: true)
    }

    /// Details, settings and validation. Enabled even when the preset can't be
    /// picked, so the reason can be read in full.
    private func modeInfoButton(_ preset: CatalogPreset) -> some View {
        Button { presetSheet = preset } label: {
            Image(systemName: "info.circle")
                .font(.system(size: 15))
                .foregroundStyle(SD.Colors.textSecondary)
                .frame(width: 44, height: 44)  // the minimum touch target
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(preset.title) details and settings")
        .accessibilityIdentifier("preset.info.\(preset.id)")
    }

    /// The running mode's tuning variants. Selected from what the device
    /// reports, never from the tap.
    private func variantChips(_ preset: CatalogPreset, featured: Bool) -> some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: SD.Space.xs) {
                ForEach(preset.variants) { variant in
                    variantChip(preset, variant)
                }
            }
            .padding(.leading, featured ? 0 : 28 + SD.Space.sm)
        }
    }

    private func variantChip(_ preset: CatalogPreset, _ variant: PresetVariantInfo) -> some View {
        let selected = (activeVariant ?? preset.variants.first?.id) == variant.id
        return Button {
            setMode("\(preset.id):\(variant.id)")
        } label: {
            Text(variant.title)
                .font(SD.Typography.caption.weight(.semibold))
                .foregroundStyle(selected ? SD.Colors.textOnAccent : SD.Colors.textSecondary)
                .padding(.horizontal, SD.Space.sm + 2)
                .frame(minHeight: 34)
                .background(Capsule().fill(selected ? SD.Colors.accent : SD.Colors.surfaceRaised))
                .frame(minHeight: 44)
                .contentShape(Rectangle())
        }
        .buttonStyle(SDPressStyle(scale: 0.95))
        // Selection was shown by colour only.
        .accessibilityAddTraits(selected ? .isSelected : [])
        .accessibilityLabel("\(preset.title): \(variant.title)")
        .accessibilityIdentifier("preset.variant.\(variant.id)")
    }

    /// The mode's symbol: on a raised tile for the featured mode, bare in the
    /// list. Bone when it runs, steel otherwise.
    @ViewBuilder
    private func modeGlyph(_ id: String, featured: Bool, active: Bool, unavailable: Bool) -> some View {
        let color = active ? SD.Colors.textPrimary : (unavailable ? SD.Colors.textTertiary : SD.Colors.textSecondary)
        if featured {
            Image(systemName: Preset.icon(for: id))
                .font(.system(size: 19, weight: .regular))
                .foregroundStyle(color)
                .frame(width: 44, height: 44)
                .background(
                    RoundedRectangle(cornerRadius: SD.Radius.lg, style: .continuous)
                        .fill(SD.Colors.surfaceRaised)
                )
                .accessibilityHidden(true)
        } else {
            Image(systemName: Preset.icon(for: id))
                .font(.system(size: 15, weight: .regular))
                .foregroundStyle(color)
                .frame(width: 28)
                .accessibilityHidden(true)
        }
    }

    /// The row's state as a symbol (its words are in the row and its value).
    @ViewBuilder
    private func modeStateGlyph(_ rowState: ModeRowState) -> some View {
        switch rowState {
        case .active:
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(SD.Colors.success)
        case .switching:
            ProgressView().controlSize(.small).tint(SD.Colors.textSecondary)
        case .queuedOffline:
            Image(systemName: "wifi.slash")
                .foregroundStyle(SD.Colors.warning)
        case .failed:
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(SD.Colors.danger)
        case .activeNotMonitoring:
            // Selected, not protecting: never the green check.
            Image(systemName: "exclamationmark.circle.fill")
                .foregroundStyle(SD.Colors.danger)
        case .activeUnconfirmed:
            // Selected, but not confirmed: neither green nor a failure.
            Image(systemName: "questionmark.circle.fill")
                .foregroundStyle(SD.Colors.warning)
        case .inactive, .unavailable:
            EmptyView()
        }
    }

    // MARK: Preset library

    /// The server's catalog once loaded; until then (and when it has none) the
    /// built-in list, labelled as such.
    private var libraryCatalog: PresetCatalog {
        catalog ?? Preset.builtInCatalog(reason: PresetCatalog.builtInBecauseUnreachable)
    }

    private func presetTitle(_ name: String?) -> String {
        catalog?.title(for: name) ?? Preset.title(for: name)
    }

    /// Why a preset can't be picked here: the device's own answer when the
    /// server has GET /v1/devices/{id}/presets, else the role in `status`.
    private func unavailableReason(_ id: String) -> String? {
        if let devicePresets {
            let availability = PresetAvailability.make(devicePresets.state(for: id), roleTitle: device.status?.roleTitle)
            return availability.blocksSelection ? (availability.note ?? "Not available on this device") : nil
        }
        return MonitoringStatusPresentation.presetUnavailableReason(id, status: device.status)
    }

    private func loadPresetLibrary() async {
        if catalog == nil || (catalog?.isBuiltIn == true && !catalogMissing) {
            do {
                catalog = try await state.backend.presetCatalog()
            } catch let error as APIError where error.endpointMissing {
                catalogMissing = true
                catalog = Preset.builtInCatalog(reason: PresetCatalog.builtInBecauseOldServer)
            } catch {
                if catalog == nil { catalog = Preset.builtInCatalog(reason: PresetCatalog.builtInBecauseUnreachable) }
            }
        }
        guard !devicePresetsMissing else { return }
        do {
            devicePresets = try await state.backend.devicePresets(deviceID: device.id)
        } catch let error as APIError where error.endpointMissing {
            devicePresetsMissing = true
            devicePresets = nil
        } catch {
            // Not kept: an old "available" must not outlive the answer. The
            // role in `status` stands in until the next load succeeds.
            devicePresets = nil
        }
    }

    // MARK: Custom rules

    /// Online and reporting. Nothing is shown as running otherwise.
    private var deviceLive: Bool { presence.isOnline }

    private func loadRules() async {
        if rulesLoad == .unsupported { return }
        do {
            rulesLoad = .loaded(try await state.backend.deviceRules(deviceID: device.id))
        } catch let error as APIError where error.endpointMissing {
            rulesLoad = .unsupported
        } catch {
            if case .loaded = rulesLoad { return }  // keep the list; the banner says the server is unreachable
            rulesLoad = .failed(error.localizedDescription)
        }
    }

    /// PATCH the rule, then re-read the list: the toggle shows what the
    /// server answered, never an optimistic flip.
    private func toggleRule(_ rule: DeviceRule, enabled: Bool) {
        rulesBusy.insert(rule.id)
        ruleToggleError = nil
        Task {
            defer { rulesBusy.remove(rule.id) }
            do {
                try await state.backend.setRuleEnabled(deviceID: device.id, ruleID: rule.id, enabled: enabled)
            } catch let error as APIError where error.isAuthFailure {
                state.expireSession(error.message)
                return
            } catch let error as APIError {
                ruleToggleError = RuleToggleError.message(code: error.code, detail: error.message,
                                                          endpointMissing: error.endpointMissing)
            } catch {
                ruleToggleError = error.localizedDescription
            }
            await loadRules()
        }
    }

    // MARK: Stop / resume

    private var protectionReport: ProtectionReport {
        ProtectionReport(protection: device.status?.protection, monitoringState: device.status?.monitoringState,
                         lastCommandID: device.status?.lastCommandID, live: deviceLive)
    }

    /// The control the device's state calls for, beside the readout: Stop
    /// while it monitors, Resume (the primary action) once it has stopped.
    /// Never "Stopped" before the device says so: what is in flight is said
    /// under the readout (`protectionNote`) until the device confirms.
    @ViewBuilder private var protectionControl: some View {
        let report = protectionReport
        let phase = ProtectionControl.phase(protectionRequest, report: report, now: .now)
        let action = protectionRequest?.action ?? ProtectionControl.offeredAction(report)
        let pending: Bool = { if case .pending = phase { return true }; return false }()
        let busy = protectionSending || pending
        let disabled = busy || protectionUnsupported != nil || device.status == nil
        let resume = action == .resume
        let title = busy ? "\(action.progressive)…" : action.verb
        let detail: String = {
            switch phase {
            case .pending: return "Waiting for \(device.name) to confirm"
            case .overdue: return "Not confirmed yet — tap to send again"
            case .confirmed(let text): return text
            case .idle: break
            }
            if let notice = protectionNotice, Date().timeIntervalSince(notice.at) < 20 { return notice.text }
            if device.status == nil { return "Not supported by this server" }
            switch ProtectionControl.isStopped(report) {
            case true?: return "Stopped — nothing is being watched"
            case false?: return "Pause all monitoring and alerts"
            case nil: return "Status unknown — the device isn't reporting"
            }
        }()

        Button {
            protectionTaps += 1
            if action == .stop { confirmStop = true } else { sendProtection(.resume) }
        } label: {
            HStack(spacing: 6) {
                if busy {
                    ProgressView().controlSize(.small)
                        .tint(resume ? SD.Colors.textOnAccent : SD.Colors.textPrimary)
                } else {
                    Image(systemName: resume ? "play.fill" : "pause.fill")
                        .font(.system(size: 11, weight: .bold))
                }
                Text(title)
                    .font(SD.Typography.label.weight(.semibold))
                    .fixedSize(horizontal: false, vertical: true)
            }
            .foregroundStyle(resume ? SD.Colors.textOnAccent : SD.Colors.textPrimary)
            .padding(.horizontal, SD.Space.md)
            .frame(minHeight: 44)
            .background(Capsule().fill(resume ? SD.Colors.accent : SD.Colors.surfaceRaised))
            .contentShape(Capsule())
            .opacity(disabled ? 0.45 : 1)
        }
        .buttonStyle(SDPressStyle(scale: 0.96))
        .disabled(disabled)
        .sensoryFeedback(.impact(weight: .light), trigger: protectionTaps)
        .accessibilityLabel("\(title), \(detail)")
        .accessibilityIdentifier("detail.protection")
    }

    /// What a stop or resume is doing, or why it can't be offered: said under
    /// the readout, in the device's terms.
    private var protectionNote: (text: String, color: Color)? {
        let report = protectionReport
        let phase = ProtectionControl.phase(protectionRequest, report: report, now: .now)
        if let protectionUnsupported { return (protectionUnsupported, SD.Colors.textTertiary) }
        if let protectionError { return (protectionError, SD.Colors.danger) }
        switch phase {
        case .pending(let text): return (text, SD.Colors.textSecondary)
        case .overdue(let text): return (text, SD.Colors.warning)
        case .confirmed(let text): return (text, SD.Colors.textSecondary)
        case .idle: break
        }
        if let notice = protectionNotice, Date().timeIntervalSince(notice.at) < 20 {
            return (notice.text, SD.Colors.textSecondary)
        }
        if device.status == nil {
            return ("This server doesn't report monitoring state, so a stop or resume couldn't be confirmed.",
                    SD.Colors.textTertiary)
        }
        return nil
    }

    private func sendProtection(_ action: ProtectionAction) {
        protectionSending = true
        protectionError = nil
        protectionNotice = nil
        Task {
            defer { protectionSending = false }
            do {
                // `confirm` is true only for a stop, and only after the dialog.
                let command = try await state.backend.setProtection(
                    deviceID: device.id, action: action, confirm: action == .stop)
                protectionRequest = ProtectionRequest(action: action, commandID: command.commandID, requestedAt: Date())
                await reload()
            } catch let error as APIError where error.isAuthFailure {
                state.expireSession(error.message)
            } catch let error as APIError {
                let message = ProtectionControl.refusal(code: error.code, detail: error.message,
                                                        endpointMissing: error.endpointMissing)
                if error.code == "firmware_unsupported" || error.endpointMissing {
                    protectionUnsupported = message
                } else {
                    protectionError = message
                }
            } catch {
                protectionError = error.localizedDescription
            }
        }
    }

    /// Once the device applied the request, say so for a while and stop
    /// tracking it: a later stop or resume from elsewhere must not be read
    /// against this one.
    private func settleProtection() {
        guard let request = protectionRequest,
              case .confirmed(let text) = ProtectionControl.phase(request, report: protectionReport, now: Date())
        else { return }
        protectionRequest = nil
        protectionNotice = (text, Date())
    }

    // Recent activity moved to EventHistoryView (Manage → Event history). The
    // events are still loaded here: they drive `alertActive`, the
    // notification-tap → alert sheet, and the history row's summary.

    private func setMode(_ name: String) {
        // Only refuse a tap that would change nothing.
        //
        // This also required `switchingTo` to be empty, which latched: it was
        // cleared only when the device echoed back the EXACT string, so on an
        // offline or slow device — or whenever the device normalised the name,
        // which "driver_safety" -> "driver_safety:teen_driver" does — it never
        // cleared, and every mode row and variant chip silently returned for the
        // rest of the session. The card simply stopped responding, with no
        // press feedback and no message. That is the "I have to tap twice".
        //
        // A second tap re-targets instead. The server is last-write-wins on
        // desired_preset, so an overlapping request is safe.
        //
        // "Change nothing" is NOT the same as "name equals device.activePreset".
        // `activePreset` holds the last real preset the device ran even while AI
        // Custom Monitoring is layered on top of it (custom rules replace the
        // module, not the field) — so tapping that exact same mode row to exit
        // custom monitoring and go back to it hit this guard and did literally
        // nothing: no request sent, no error, no press feedback. That read as
        // "the button doesn't work" rather than "already selected", and only for
        // whichever preset happened to precede custom monitoring. The tap
        // changes something real whenever custom monitoring is active, even if
        // the target name matches — the device still has work to do (exit
        // custom monitoring), so only skip the request when neither is true.
        guard name != device.activePreset || customMonitoringActive else { return }

        let base = Preset.split(name).base
        // A retry supersedes its own prior failure — otherwise ModeSwitchState
        // would keep showing the OLD failure (it's checked before `pending`)
        // right on top of the new attempt.
        if failedSwitch?.targetBase == base { failedSwitch = nil }

        let requestedAt = Date()
        // Set BEFORE the request is even sent: the row shows "switching"
        // immediately, not after a round trip confirms the tap registered.
        pendingSwitch = ModeSwitchState.PendingSwitch(targetBase: base, requestedAt: requestedAt)

        let token = setModeGuard.begin()
        Task {
            do {
                try await state.backend.setPreset(deviceID: device.id, preset: name)
                guard setModeGuard.isLatest(token) else { return }
                // Don't wait for the next 4s timer tick to find out whether it
                // stuck — the entire point of this fix is that switching modes
                // should feel fast, and a reload right after the command is
                // accepted is most of that: see the latency notes in
                // docs/ (mode-switch path) for where the remaining time goes.
                await reload()
            } catch {
                guard setModeGuard.isLatest(token) else { return }
                if pendingSwitch?.targetBase == base { pendingSwitch = nil }
                // A refusal the backend explained (a coded error: the role
                // forbids this mode, an unknown variant) is shown as said. A
                // transport failure keeps the generic "Didn't switch".
                let explained = (error as? APIError).flatMap { $0.code != nil ? $0.message : nil }
                failedSwitch = ModeSwitchState.PendingSwitch(
                    targetBase: base, requestedAt: requestedAt, reason: explained)
            }
        }
    }

    private func reload() async {
        let token = reloadGuard.begin()
        // Captured before either network call — this is what lets
        // syncPendingSwitch tell "a tap that happened while I was already in
        // flight" apart from "a tap that happened before I even started",
        // which is the only distinction that matters for not clobbering it.
        let reloadStartedAt = Date()
        do {
            let all = try await state.backend.devices()
            guard reloadGuard.isLatest(token) else { return }
            guard let fresh = all.first(where: { $0.id == device.id }) else {
                // Removed from the account (or moved to another): it used to keep
                // the last reading on screen, ONLINE, polling for ever in silence.
                errorTitle = "\(device.name) is no longer on this account"
                errorMessage = "The server no longer lists this device for your account. Go back to Your Devices."
                hasLoaded = true
                return
            }
            device = fresh
            now = Date()
            syncPendingSwitch(from: fresh, reloadStartedAt: reloadStartedAt)
            settleProtection()
            let fetchedEvents = try await state.backend.events(deviceID: device.id, limit: 200)
            guard reloadGuard.isLatest(token) else { return }
            events = fetchedEvents
            // The library, the device's availability and its rules. Each says
            // "not supported by this server" on its own when the endpoint is
            // missing, and none of them fails the screen.
            await loadPresetLibrary()
            await loadRules()
            // The notifier dedupes by event id, so calling this on every poll
            // announces each alert exactly once.
            // AlertWatcher owns notifications now, from the app root, so it
            // works on every screen. Announcing here as well duplicated the
            // work on the one screen that was already the busiest.
            errorMessage = nil
            errorTitle = nil
        } catch let error as APIError where error.isAuthFailure {
            guard reloadGuard.isLatest(token) else { return }
            state.expireSession(error.message)
        } catch {
            guard reloadGuard.isLatest(token) else { return }
            errorMessage = error.localizedDescription
            errorTitle = (error as? APIError)?.title
        }
        if reloadGuard.isLatest(token) { hasLoaded = true }
    }

    /// Reconcile this app instance's optimistic "I just tapped this" state
    /// against the server's own authoritative record of what's still
    /// outstanding (`desiredPreset`) — the same field cloud/app.py's
    /// convergence logic clears once the device actually confirms, already
    /// alias- and custom-monitoring-aware. Trusting it here means this view
    /// never has to duplicate that reasoning, and it's also what lets
    /// `pendingSwitch` correctly reflect a mode change made from somewhere
    /// else (another client, the on-device dashboard) rather than only ever
    /// tracking this app's own taps.
    private func syncPendingSwitch(from fresh: Device, reloadStartedAt: Date) {
        guard ModeSwitchState.shouldSyncPendingSwitch(current: pendingSwitch, reloadStartedAt: reloadStartedAt) else {
            return  // a newer optimistic tap exists; this fetch predates it and must not touch it
        }
        // The device (or the backend for it) refused the switch: show its
        // reason now, instead of a spinner that ends in a generic timeout.
        if let pending = pendingSwitch,
           let reason = DeviceRefusal.forPreset(pending.targetBase, status: fresh.status, since: pending.requestedAt) {
            failedSwitch = ModeSwitchState.PendingSwitch(targetBase: pending.targetBase,
                                                         requestedAt: pending.requestedAt, reason: reason)
            pendingSwitch = nil
            return
        }
        guard let desired = fresh.desiredPreset else {
            pendingSwitch = nil  // nothing outstanding server-side
            return
        }
        let desiredBase = Preset.split(desired).base
        if pendingSwitch?.targetBase == desiredBase {
            return  // still the switch we're already tracking — keep its original timestamp for the timeout
        }
        // The server has an outstanding command this view wasn't already
        // tracking — most often this app's own request having just landed
        // server-side, occasionally another client's. Either way the server
        // is authoritative for WHAT is pending; "now" is the best available
        // timestamp for something first noticed this moment.
        pendingSwitch = ModeSwitchState.PendingSwitch(targetBase: desiredBase, requestedAt: Date())
    }
}

/// An Ask Jupiter screen to present, with the request it opens with.
private struct JupiterSheet: Identifiable {
    let id = UUID()
    let prompt: String
}
