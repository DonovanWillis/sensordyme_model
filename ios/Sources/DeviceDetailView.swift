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

    private let refreshTimer = Timer.publish(every: 4, on: .main, in: .common).autoconnect()

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
        ScrollView {
            VStack(spacing: SD.Space.md) {
                // Above the feed, not below the fold: this view polls every 4s and
                // keeps showing the last frame and event list when a refresh fails.
                if let errorMessage {
                    ConnectionBanner(message: errorMessage) { await reload() }
                }

                FeedCard(device: device, alertActive: alertActive)

                statusRow
                healthCard
                configureCard
                modeCard
                activityCard
            }
            .padding(.horizontal, SD.Space.md)
            .padding(.bottom, SD.Space.lg)
        }
        .background(SD.Colors.background)
        .navigationTitle(device.name)
        .navigationBarTitleDisplayMode(.inline)
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
                        Label("Areas", systemImage: "square.dashed")
                    }
                    if state.isDemo {
                        Button {
                            showDemoTrigger = true
                        } label: {
                            Label("Run a demo alert", systemImage: "play.circle")
                        }
                    }
                    NavigationLink {
                        ReportsView(device: device)
                    } label: {
                        Label("Full report", systemImage: "chart.bar")
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
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
        .sheet(isPresented: $showAI) {
            AIConfigureView(device: device) { await reload() }
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
        .onChange(of: notifier.pendingEventID) { _, id in
            guard let id else { return }
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
                    errorMessage = "Couldn't open that alert — try Recent Activity below."
                }
                notifier.pendingEventID = nil
            }
        }
        .task {
            await notifier.refreshAuthorization()
            if !notifier.authorized { await notifier.requestPermission() }
        }
        .refreshable { await reload() }
        .task { await reload() }
        .onReceive(refreshTimer) { _ in Task { await reload() } }
    }

    // MARK: Status

    private var statusRow: some View {
        HStack(spacing: SD.Space.xs) {
            SDStatusPill(
                status: device.online ? .online : .offline,
                pulses: device.online
            )
            SDStatusPill(
                status: alertActive ? .alert : .online,
                text: alertActive ? "Alert" : "All clear"
            )
            Spacer()
            Text(customMonitoringActive
                 ? (device.monitoring?.title ?? "AI Custom Monitoring")
                 : Preset.title(for: device.activePreset))
                .font(SD.Typography.caption.weight(.semibold))
                .foregroundStyle(customMonitoringActive ? SD.Colors.accent : SD.Colors.textSecondary)
                .lineLimit(1)
        }
    }

    // MARK: Health
    //
    // Every row here is something the Pi observed and sent up this cycle. When
    // the device stops reporting the backend withholds the whole report, so
    // these disappear rather than freezing on their last happy values.

    @ViewBuilder private var healthCard: some View {
        if let health = device.health {
            SDCard {
                VStack(alignment: .leading, spacing: SD.Space.sm) {
                    SDSectionHeader(title: "Device health")
                    healthRow("Camera", systemImage: "camera.fill", health.camera, metric: "fps", suffix: " fps")
                    healthRow("AI", systemImage: "cpu", health.accelerator)
                    healthRow("Monitoring", systemImage: "eye.fill", health.inference)
                    healthRow("Network", systemImage: "wifi", health.network)
                }
            }
        } else if device.online {
            // Online but silent about itself: an older device build that predates
            // health reporting. Say so plainly instead of showing nothing.
            SDCard {
                Label(
                    "This device isn't reporting its status yet. Update it to see camera and AI health here.",
                    systemImage: "info.circle"
                )
                .font(SD.Typography.caption)
                .foregroundStyle(SD.Colors.textSecondary)
            }
        }
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
                    Circle().fill(component.status.tint).frame(width: 8, height: 8)
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

    // MARK: Configure with AI

    private var configureCard: some View {
        Button { showAI = true } label: {
            SDCard {
                HStack(spacing: SD.Space.sm) {
                    ZStack {
                        RoundedRectangle(cornerRadius: SD.Radius.sm, style: .continuous)
                            .fill(SD.Colors.accentMuted)
                            .frame(width: 38, height: 38)
                        Image(systemName: "wand.and.stars")
                            .foregroundStyle(SD.Colors.accent)
                    }
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Configure with AI")
                            .font(SD.Typography.headline)
                            .foregroundStyle(SD.Colors.textPrimary)
                        Text("Describe what to watch for in your own words")
                            .font(SD.Typography.caption)
                            .foregroundStyle(SD.Colors.textSecondary)
                            .lineLimit(2)
                    }
                    Spacer(minLength: SD.Space.xxs)
                    Image(systemName: "chevron.right")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(SD.Colors.textTertiary)
                }
            }
        }
        .buttonStyle(.plain)
    }

    // MARK: Mode

    /// Whether a custom AI configuration has replaced the base mode.
    private var customMonitoringActive: Bool { device.monitoring?.active == true }

    private var activeBase: String { Preset.split(device.activePreset).base }
    private var activeVariant: String? { Preset.split(device.activePreset).variant }

    private var modeCard: some View {
        SDCard {
            VStack(alignment: .leading, spacing: SD.Space.sm) {
                SDSectionHeader(title: "Monitoring mode")

                // What is ACTUALLY running, when it is not one of the presets
                // below. The device reports this itself, so it cannot disagree
                // with what the device is doing.
                if customMonitoringActive, let custom = device.monitoring {
                    VStack(alignment: .leading, spacing: SD.Space.xs) {
                        HStack(spacing: SD.Space.sm) {
                            Image(systemName: "sparkles")
                                .frame(width: 26)
                                .foregroundStyle(SD.Colors.accent)
                            VStack(alignment: .leading, spacing: 1) {
                                Text(custom.title)
                                    .font(SD.Typography.body)
                                    .foregroundStyle(SD.Colors.textPrimary)
                                Text("Active — this replaces the modes below")
                                    .font(SD.Typography.caption)
                                    .foregroundStyle(SD.Colors.accent)
                            }
                            Spacer(minLength: 0)
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundStyle(SD.Colors.accent)
                        }

                        Text(custom.summary)
                            .font(SD.Typography.callout)
                            .foregroundStyle(SD.Colors.textSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                            .padding(.leading, 26 + SD.Space.sm)
                    }
                    .padding(.vertical, SD.Space.xxs)

                    Text("Pick a mode below to go back to standard monitoring.")
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textTertiary)

                    Divider().overlay(SD.Colors.border)
                }

                ForEach(Preset.catalog) { preset in
                    modeRow(preset)
                    if preset.id != Preset.catalog.last?.id {
                        Divider().overlay(SD.Colors.border)
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
        let rowState = ModeSwitchState.state(
            for: preset.name, activeBase: activeBase, customMonitoringActive: customMonitoringActive,
            pending: pendingSwitch, deviceOnline: device.online, failure: failedSwitch, now: .now
        )
        let isActive = rowState == .active

        VStack(alignment: .leading, spacing: SD.Space.xs) {
            Button { setMode(preset.name) } label: {
                HStack(spacing: SD.Space.sm) {
                    Image(systemName: preset.icon)
                        .frame(width: 26)
                        .foregroundStyle(isActive ? SD.Colors.accent : SD.Colors.textTertiary)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(preset.title)
                            .font(SD.Typography.body)
                            .foregroundStyle(SD.Colors.textPrimary)
                        switch rowState {
                        case .failed(let message):
                            Text(message)
                                .font(SD.Typography.caption)
                                .foregroundStyle(SD.Colors.danger)
                        case .queuedOffline:
                            Text("Device offline — will apply when it reconnects")
                                .font(SD.Typography.caption)
                                .foregroundStyle(SD.Colors.warning)
                        case .inactive, .switching, .active:
                            Text(preset.subtitle)
                                .font(SD.Typography.caption)
                                .foregroundStyle(SD.Colors.textSecondary)
                        }
                    }
                    Spacer(minLength: SD.Space.xxs)
                    switch rowState {
                    case .active:
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(SD.Colors.success)
                    case .switching:
                        ProgressView().controlSize(.small)
                    case .queuedOffline:
                        Image(systemName: "wifi.slash")
                            .foregroundStyle(SD.Colors.warning)
                    case .failed:
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(SD.Colors.danger)
                    case .inactive:
                        EmptyView()
                    }
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

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

    // MARK: Activity

    private var activityCard: some View {
        SDCard {
            VStack(alignment: .leading, spacing: SD.Space.sm) {
                SDSectionHeader(title: "Recent activity")

                if !events.isEmpty {
                    EventTimeline(events: events) { selectedEvent = $0 }
                        .padding(.bottom, SD.Space.xxs)
                    Divider().overlay(SD.Colors.border)
                }

                if events.isEmpty {
                    if !hasLoaded {
                        SDLoadingState(message: "Loading activity…")
                    } else {
                        SDEmptyState(
                            icon: "checkmark.circle",
                            title: "Nothing to report",
                            message: "Events appear here as soon as \(device.name) sees something worth telling you about."
                        )
                    }
                } else {
                    ForEach(events.prefix(25)) { event in
                        Button { selectedEvent = event } label: {
                            EventRow(event: event)
                                .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        if event.id != events.prefix(25).last?.id {
                            Divider().overlay(SD.Colors.border)
                        }
                    }
                }
            }
        }
    }

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
                failedSwitch = ModeSwitchState.PendingSwitch(targetBase: base, requestedAt: requestedAt)
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
                Text(date, style: .relative)
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textTertiary)
            }
        }
        .padding(.vertical, SD.Space.xxs)
        .accessibilityElement(children: .combine)
    }
}
