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
    @State private var showAI = false
    @State private var showZones = false
    @State private var selectedEvent: EventItem?
    @State private var showDemoTrigger = false
    @StateObject private var notifier = AlertNotifier.shared
    @State private var showRename = false
    @State private var newName = ""
    @State private var hasLoaded = false
    /// What is being typed into the inline Ask Jupiter bar.
    @State private var jupiterDraft = ""
    /// The request handed to the Ask Jupiter screen when it opens (and sent
    /// straight away when non-empty).
    @State private var jupiterRequest = ""
    @State private var showHistory = false

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
            VStack(spacing: SD.Space.md) {
                // Above the feed, not below the fold: this view polls every 4s and
                // keeps showing the last frame and event list when a refresh fails.
                if let errorMessage {
                    ConnectionBanner(message: errorMessage) { await reload() }
                }

                // The camera is the centrepiece and always comes first.
                FeedCard(device: device, alertActive: alertActive)

                statusRow

                askJupiterSection
                modeCard
                manageCard

                // Last on purpose: detail for someone who went looking for it,
                // not something competing with the picture.
                healthCard
                    .id("health")
            }
            .padding(.horizontal, SD.Space.md)
            .padding(.top, SD.Space.xs)
            .padding(.bottom, SD.Space.xl)
        }
        .task {
            guard DebugLaunch.scrollToBottom else { return }
            try? await Task.sleep(for: .seconds(1))
            withAnimation { proxy.scrollTo("health", anchor: .bottom) }
        }
        }
        .scrollDismissesKeyboard(.interactively)
        .orynScreen()
        .navigationTitle(device.name)
        .navigationBarTitleDisplayMode(.inline)
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
                        .frame(width: 32, height: 32)
                        .contentShape(Rectangle())
                }
                .accessibilityLabel("Device options")
            }
        }
        .alert("Rename device", isPresented: $showRename) {
            TextField("Name", text: $newName)
            Button("Save") {
                Task {
                    try? await state.backend.renameDevice(deviceID: device.id, name: newName)
                    await reload()
                }
            }
            Button("Cancel", role: .cancel) {}
        }
        .sheet(isPresented: $showAI, onDismiss: { jupiterRequest = "" }) {
            AIConfigureView(device: device, initialPrompt: jupiterRequest) { await reload() }
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
                    errorMessage = "Couldn't open that alert. Open Event history below to find it."
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
                jupiterRequest = DebugLaunch.jupiterPrompt ?? ""
                showAI = true
            } else if DebugLaunch.openHistory {
                showHistory = true
            }
        }
        .onReceive(refreshTimer) { _ in
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

    // MARK: Status

    /// Online, alert state and what it is watching for, in one quiet line under
    /// the picture. The second pill is the device's own monitoring state
    /// (DeviceStatus): "All clear" only when the device reports `ready`, never
    /// for a failed, stopped, starting or silent one.
    private var statusRow: some View {
        let pill = MonitoringStatusPresentation.pill(device.monitoringPresentation, alertActive: alertActive)
        return HStack(spacing: SD.Space.xs) {
            SDStatusPill(
                status: device.online ? .online : .offline,
                pulses: device.online
            )
            SDStatusPill(status: pill.tone.sdStatus, text: pill.text)
            Spacer(minLength: SD.Space.xs)
            HStack(spacing: 5) {
                if customMonitoringActive {
                    Image(systemName: "sparkles")
                        .font(.system(size: 10, weight: .semibold))
                }
                // Rules running ALONGSIDE the mode leave the mode in charge, so
                // the mode is what this line names.
                Text(customReplacesModes
                     ? (device.monitoring?.title ?? "AI Custom Monitoring")
                     : Preset.title(for: device.activePreset))
                    .font(SD.Typography.caption.weight(.medium))
                    .lineLimit(1)
            }
            .foregroundStyle(customMonitoringActive ? SD.Colors.textPrimary : SD.Colors.textTertiary)
        }
    }

    // MARK: Health
    //
    // Every row here is something the Pi observed and sent up this cycle. When
    // the device stops reporting the backend withholds the whole report, so
    // these disappear rather than freezing on their last happy values.

    @ViewBuilder private var healthCard: some View {
        VStack(alignment: .leading, spacing: SD.Space.xs) {
            SDSectionHeader(title: "Device health")
                .padding(.horizontal, SD.Space.xxs)
            if let health = device.health {
                SDCard {
                    VStack(alignment: .leading, spacing: SD.Space.sm) {
                        healthRow("Camera", systemImage: "camera", health.camera, metric: "fps", suffix: " fps")
                        OrynRowDivider(inset: 22 + SD.Space.sm)
                        healthRow("AI", systemImage: "cpu", health.accelerator)
                        OrynRowDivider(inset: 22 + SD.Space.sm)
                        healthRow("Monitoring", systemImage: "eye", monitoringComponent(health))
                        OrynRowDivider(inset: 22 + SD.Space.sm)
                        healthRow("Network", systemImage: "wifi", health.network)
                    }
                }
            } else {
                // Online but silent about itself: an older device build that
                // predates health reporting. Offline: the backend withholds a
                // stale report. Either way, say so instead of leaving a gap.
                SDCard {
                    HStack(alignment: .top, spacing: SD.Space.sm) {
                        Image(systemName: "info.circle")
                            .foregroundStyle(SD.Colors.textTertiary)
                        Text(device.online
                             ? "This device isn't reporting its status yet. Update it to see camera and AI health here."
                             : "Health appears here while the device is online.")
                            .font(SD.Typography.caption)
                            .foregroundStyle(SD.Colors.textTertiary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
        }
    }

    /// The "Monitoring" row: the device's monitoring state (DeviceStatus), in
    /// the health card's own row format. It used to show the inference loop's
    /// health, which reads "ok" on a device that is stopped, or failed because
    /// the face model is gone. The inference component remains the fallback
    /// for a backend that predates the status contract.
    private func monitoringComponent(_ health: DeviceHealthReport) -> HealthComponent? {
        guard let presentation = device.monitoringPresentation else { return health.inference }
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

    // MARK: Ask Jupiter
    //
    // Was "Configure with AI". Same flow underneath: the request goes to
    // `/v1/ai/configure`, comes back as a draft, and nothing changes on the
    // device until the person reviews it and taps Activate in AIConfigureView.

    private var askJupiterSection: some View {
        VStack(alignment: .leading, spacing: SD.Space.xs) {
            AskJupiterBar(text: $jupiterDraft) {
                guard let request = JupiterPrompt.normalized(jupiterDraft) else { return }
                jupiterRequest = request
                jupiterDraft = ""
                showAI = true
            }
            HStack(spacing: SD.Space.xxs) {
                Text("Tell \(device.name) what to watch for. You review it before anything changes.")
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textTertiary)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: SD.Space.xs)
                Button {
                    jupiterRequest = ""
                    showAI = true
                } label: {
                    Text("Examples")
                        .font(SD.Typography.caption.weight(.semibold))
                        .foregroundStyle(SD.Colors.textSecondary)
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, SD.Space.xxs)
        }
    }

    // MARK: Manage

    /// What used to be scattered between the toolbar menu and a long activity
    /// card: the places you go *from* a device, as rows.
    private var manageCard: some View {
        VStack(alignment: .leading, spacing: SD.Space.xs) {
            SDSectionHeader(title: "Manage")
                .padding(.horizontal, SD.Space.xxs)
            SDCard(padding: SD.Space.xxs) {
                VStack(spacing: 0) {
                    Button { showHistory = true } label: {
                        OrynNavRow(
                            icon: "clock.arrow.circlepath",
                            title: "Event history",
                            detail: historyDetail,
                            trailing: alertActive ? "ALERT" : nil
                        )
                    }
                    .buttonStyle(SDRowButtonStyle(minHeight: 60))

                    OrynRowDivider()

                    Button { showZones = true } label: {
                        OrynNavRow(
                            icon: "square.dashed",
                            title: "Detection areas",
                            detail: "Draw where rules apply"
                        )
                    }
                    .buttonStyle(SDRowButtonStyle(minHeight: 60))

                    OrynRowDivider()

                    NavigationLink {
                        ReportsView(device: device)
                    } label: {
                        OrynNavRow(
                            icon: "chart.bar",
                            title: "Report",
                            detail: "Patterns across recent events"
                        )
                    }
                    .buttonStyle(SDRowButtonStyle(minHeight: 60))

                    if state.isDemo {
                        OrynRowDivider()
                        Button { showDemoTrigger = true } label: {
                            OrynNavRow(
                                icon: "play.circle",
                                title: "Run a demo alert",
                                detail: "Simulated: see an alert end to end"
                            )
                        }
                        .buttonStyle(SDRowButtonStyle(minHeight: 60))
                    }
                }
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
    private var activeVariant: String? { Preset.split(device.activePreset).variant }

    private var modeCard: some View {
        VStack(alignment: .leading, spacing: SD.Space.xs) {
        SDSectionHeader(title: "Monitoring mode")
            .padding(.horizontal, SD.Space.xxs)
        SDCard {
            VStack(alignment: .leading, spacing: SD.Space.sm) {
                // What is ACTUALLY running, when it is not one of the presets
                // below. The device reports this itself, so it cannot disagree
                // with what the device is doing.
                if customMonitoringActive, let custom = device.monitoring {
                    VStack(alignment: .leading, spacing: SD.Space.xs) {
                        HStack(spacing: SD.Space.sm) {
                            OrynMark(height: 16, color: SD.Colors.signal)
                                .frame(width: 26)
                            VStack(alignment: .leading, spacing: 1) {
                                Text(custom.title)
                                    .font(SD.Typography.body)
                                    .foregroundStyle(SD.Colors.textPrimary)
                                Text(customReplacesModes
                                     ? "Set by Jupiter · replaces the modes below"
                                     : "Set by Jupiter · runs alongside the mode below")
                                    .font(SD.Typography.caption)
                                    .foregroundStyle(SD.Colors.textSecondary)
                            }
                            Spacer(minLength: 0)
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundStyle(SD.Colors.success)
                        }

                        Text(custom.summary)
                            .font(SD.Typography.callout)
                            .foregroundStyle(SD.Colors.textSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                            .padding(.leading, 26 + SD.Space.sm)
                    }
                    .padding(.vertical, SD.Space.xxs)

                    Text(customReplacesModes
                         ? "Pick a mode below to go back to standard monitoring."
                         : "Pick the mode below again to turn these rules off.")
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textTertiary)

                    Rectangle().fill(SD.Colors.border).frame(height: 1)
                }

                // The device refused the latest rules (all or nothing) and kept
                // what it was running. They are not shown as running anywhere.
                if let notice = MonitoringStatusPresentation.rejectedRulesNotice(device.status) {
                    HStack(alignment: .top, spacing: SD.Space.sm) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.system(size: 13))
                            .frame(width: 26)
                            .foregroundStyle(SD.Colors.danger)
                        Text(notice)
                            .font(SD.Typography.caption)
                            .foregroundStyle(SD.Colors.danger)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .padding(.vertical, SD.Space.xxs)

                    Rectangle().fill(SD.Colors.border).frame(height: 1)
                }

                ForEach(Preset.catalog) { preset in
                    modeRow(preset)
                    if preset.id != Preset.catalog.last?.id {
                        Rectangle().fill(SD.Colors.border).frame(height: 1)
                    }
                }
            }
        }
        }
    }

    @ViewBuilder
    private func modeRow(_ preset: Preset) -> some View {
        // The single source of truth for this row's state — see
        // ModeSwitchState.swift for why "active"/"switching" alone weren't
        // enough: no distinct failure state, no offline state, no timeout.
        // `now: .now` is what lets a stuck switch age past switchTimeout and
        // flip to failed on the app's own 4s refresh cycle, with no separate
        // timer needed.
        // Only custom rules that REPLACE the modes stop a mode row reading active;
        // rules running alongside (an additive role) leave the mode running.
        let rowState = ModeSwitchState.state(
            for: preset.name, activeBase: activeBase, customMonitoringActive: customReplacesModes,
            pending: pendingSwitch, deviceOnline: device.online, failure: failedSwitch, now: .now,
            unavailable: MonitoringStatusPresentation.presetUnavailableReason(preset.name, status: device.status)
        )
        let isActive = rowState == .active
        let isUnavailable = rowState.isUnavailable

        VStack(alignment: .leading, spacing: SD.Space.xs) {
            Button { setMode(preset.name) } label: {
                HStack(spacing: SD.Space.sm) {
                    Image(systemName: preset.icon)
                        .font(.system(size: 15))
                        .frame(width: 26)
                        .foregroundStyle(isActive ? SD.Colors.textPrimary : SD.Colors.textDisabled)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(preset.title)
                            .font(SD.Typography.body)
                            .foregroundStyle(isUnavailable ? SD.Colors.textDisabled : SD.Colors.textPrimary)
                        switch rowState {
                        case .failed(let message):
                            Text(message)
                                .font(SD.Typography.caption)
                                .foregroundStyle(SD.Colors.danger)
                        case .unavailable(let reason):
                            Text(reason)
                                .font(SD.Typography.caption)
                                .foregroundStyle(SD.Colors.textTertiary)
                        case .queuedOffline:
                            Text("Device offline — will apply when it reconnects")
                                .font(SD.Typography.caption)
                                .foregroundStyle(SD.Colors.warning)
                        case .inactive, .switching, .active:
                            Text(preset.subtitle)
                                .font(SD.Typography.caption)
                                .foregroundStyle(SD.Colors.textTertiary)
                        }
                    }
                    Spacer(minLength: SD.Space.xxs)
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
                    case .inactive, .unavailable:
                        EmptyView()
                    }
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            // The backend would refuse it (role_forbids_preset): nothing to tap.
            .disabled(isUnavailable)

            // Tuning variants, shown only for the mode currently running.
            if isActive && !preset.variants.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: SD.Space.xs) {
                        ForEach(preset.variants) { variant in
                            let selected = (activeVariant ?? preset.variants.first?.name) == variant.name
                            Button {
                                setMode("\(preset.name):\(variant.name)")
                            } label: {
                                Text(variant.title)
                                    .font(SD.Typography.caption.weight(.semibold))
                                    .padding(.horizontal, SD.Space.sm)
                                    .padding(.vertical, SD.Space.xxs + 3)
                                    .background(
                                        Capsule().fill(selected ? SD.Colors.accent : SD.Colors.surfaceRaised)
                                    )
                                    .overlay(
                                        Capsule().strokeBorder(
                                            selected ? .clear : SD.Colors.border, lineWidth: 1
                                        )
                                    )
                                    .foregroundStyle(
                                        selected ? SD.Colors.textOnAccent : SD.Colors.textSecondary
                                    )
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    .padding(.leading, 26 + SD.Space.sm)
                    .padding(.vertical, 2)
                }
            }
        }
        .padding(.vertical, SD.Space.xxs)
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
            if let fresh = try await state.backend.devices().first(where: { $0.id == device.id }) {
                guard reloadGuard.isLatest(token) else { return }
                device = fresh
                syncPendingSwitch(from: fresh, reloadStartedAt: reloadStartedAt)
            }
            let fetchedEvents = try await state.backend.events(deviceID: device.id, limit: 200)
            guard reloadGuard.isLatest(token) else { return }
            events = fetchedEvents
            // The notifier dedupes by event id, so calling this on every poll
            // announces each alert exactly once.
            // AlertWatcher owns notifications now, from the app root, so it
            // works on every screen. Announcing here as well duplicated the
            // work on the one screen that was already the busiest.
            errorMessage = nil
        } catch let error as APIError where error.isAuthFailure {
            guard reloadGuard.isLatest(token) else { return }
            state.expireSession(error.message)
        } catch {
            guard reloadGuard.isLatest(token) else { return }
            errorMessage = error.localizedDescription
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

// MARK: - Event row

struct EventRow: View {
    let event: EventItem

    private var tint: Color {
        switch event.severity {
        case "loud": SD.Colors.danger
        case "soft": SD.Colors.warning
        default: SD.Colors.textTertiary
        }
    }

    /// Event types arrive as backend identifiers ("phone_use_while_distracted").
    /// Showing those raw is the fastest way to make a consumer product feel like
    /// a debug console.
    private var title: String {
        event.type
            .replacingOccurrences(of: "_", with: " ")
            .capitalized
    }

    var body: some View {
        HStack(alignment: .top, spacing: SD.Space.sm) {
            Circle()
                .fill(tint)
                .frame(width: 8, height: 8)
                .padding(.top, 6)
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(SD.Typography.body)
                    .foregroundStyle(SD.Colors.textPrimary)
                if let signals = event.signals, !signals.isEmpty {
                    Text(signals)
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textSecondary)
                        .lineLimit(2)
                }
            }
            Spacer(minLength: SD.Space.xxs)
            if let date = event.date {
                // Abbreviated ("3 min. ago") rather than the live ".relative"
                // style ("2 min, 52 sec"), which was wide enough to wrap the
                // event title onto two lines.
                Text(date, format: .relative(presentation: .numeric, unitsStyle: .abbreviated))
                    .font(SD.Typography.mono)
                    .foregroundStyle(SD.Colors.textTertiary)
                    .lineLimit(1)
                    .fixedSize()
            }
        }
        .padding(.vertical, SD.Space.xxs)
        .accessibilityElement(children: .combine)
    }
}
