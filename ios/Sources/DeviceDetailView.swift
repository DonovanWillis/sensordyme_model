import SwiftUI

/// The screen a SensorDyme owner actually lives in: the camera, what it's
/// watching for, and what it has seen. Technical detail belongs in Device
/// Health, not here — nobody buying a baby monitor should meet the word "YOLO".
struct DeviceDetailView: View {
    @EnvironmentObject private var state: AppState
    @State var device: Device
    @State private var events: [EventItem] = []
    @State private var switchingTo: String?
    @State private var errorMessage: String?
    @State private var showAI = false
    @State private var showZones = false
    @State private var showRename = false
    @State private var newName = ""
    @State private var hasLoaded = false

    private let refreshTimer = Timer.publish(every: 4, on: .main, in: .common).autoconnect()

    /// A loud alert in the last minute lights up the feed border + status.
    private var alertActive: Bool {
        guard let latest = events.first(where: { $0.severity == "loud" }), let date = latest.date else {
            return false
        }
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
            Text(Preset.title(for: device.activePreset))
                .font(SD.Typography.caption.weight(.semibold))
                .foregroundStyle(SD.Colors.textSecondary)
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

    private var activeBase: String { Preset.split(device.activePreset).base }
    private var activeVariant: String? { Preset.split(device.activePreset).variant }

    private var modeCard: some View {
        SDCard {
            VStack(alignment: .leading, spacing: SD.Space.sm) {
                SDSectionHeader(title: "Monitoring mode")

                ForEach(Preset.catalog) { preset in
                    modeRow(preset)
                    if preset.id != Preset.catalog.last?.id {
                        Divider().overlay(SD.Colors.border)
                    }
                }

                if let switchingTo {
                    Label(
                        "Switching to \(Preset.title(for: switchingTo)) — applies on the device's next sync.",
                        systemImage: "arrow.triangle.2.circlepath"
                    )
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textSecondary)
                }
            }
        }
    }

    @ViewBuilder
    private func modeRow(_ preset: Preset) -> some View {
        let isActive = activeBase == preset.name
        let isSwitching = Preset.split(switchingTo).base == preset.name

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
                        Text(preset.subtitle)
                            .font(SD.Typography.caption)
                            .foregroundStyle(SD.Colors.textSecondary)
                    }
                    Spacer(minLength: SD.Space.xxs)
                    if isActive {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(SD.Colors.success)
                    } else if isSwitching {
                        ProgressView().controlSize(.small)
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
                        EventRow(event: event)
                        if event.id != events.prefix(25).last?.id {
                            Divider().overlay(SD.Colors.border)
                        }
                    }
                }
            }
        }
    }

    private func setMode(_ name: String) {
        guard name != device.activePreset, Preset.split(switchingTo).base.isEmpty else { return }
        switchingTo = name
        Task {
            do {
                try await state.backend.setPreset(deviceID: device.id, preset: name)
            } catch {
                errorMessage = error.localizedDescription
                switchingTo = nil
            }
        }
    }

    private func reload() async {
        do {
            if let fresh = try await state.backend.devices().first(where: { $0.id == device.id }) {
                device = fresh
                if let switchingTo, fresh.activePreset == switchingTo {
                    self.switchingTo = nil
                }
            }
            events = try await state.backend.events(deviceID: device.id, limit: 200)
            errorMessage = nil
        } catch let error as APIError where error.isAuthFailure {
            state.expireSession(error.message)
        } catch {
            errorMessage = error.localizedDescription
        }
        hasLoaded = true
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
