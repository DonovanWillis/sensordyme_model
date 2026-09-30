import Combine
import SwiftUI

/// The home screen: every device, its state at a glance, and one tap to the
/// things people actually came to do.
///
/// Deliberately shows only what the backend can substantiate — online state,
/// active mode, last contact. Camera / AI-accelerator / Wi-Fi tiles are absent
/// rather than optimistic: the device does not report that health upstream yet,
/// and a green "AI running" light that is really just "the row exists in the
/// database" is the kind of thing that falls apart in a live demo.
struct DevicesView: View {
    @EnvironmentObject private var state: AppState
    @State private var devices: [Device] = []
    @State private var errorMessage: String?
    @State private var showClaim = false
    @State private var showSetup = false
    @State private var hasLoaded = false
    @StateObject private var link = DeviceLink()
    @StateObject private var notifier = AlertNotifier.shared
    /// The device a tapped notification belongs to, so the alert can be opened
    /// from anywhere in the app rather than only from the device screen.
    @State private var notificationTarget: Device?
    @StateObject private var refreshing = ProgressTracker()
    /// Same problem as DeviceDetailView's own copy of this: three independent
    /// triggers (appear, a 10s timer, pull-to-refresh) can all have a
    /// reload() in flight at once, with no guarantee their responses arrive
    /// in the order they were sent — without this, a slow, stale fetch can
    /// complete after a fresh one and silently overwrite it. See
    /// LatestRequestGuard's own docs.
    @State private var reloadGuard = LatestRequestGuard()

    private let refreshTimer = Timer.publish(every: 10, on: .main, in: .common).autoconnect()

    var body: some View {
        NavigationStack {
            ScrollView {
                LazyVStack(spacing: SD.Space.sm) {
                    // Shown whether or not devices are loaded: a failed refresh
                    // leaves stale cards on screen that look perfectly current.
                    if let errorMessage {
                        ConnectionBanner(message: errorMessage) { await reload() }
                            .padding(.bottom, SD.Space.xxs)
                    }

                    if !devices.isEmpty {
                        SDSectionHeader(title: devices.count == 1 ? "Your device" : "Your devices")
                            .padding(.top, SD.Space.xxs)
                    }
                    // The backend's view of the same device, so the banner can
                    // tell "switched off" from "not on this Wi-Fi" instead of
                    // blaming the network for both.
                    DeviceLinkBanner(
                        link: link,
                        backendDevice: devices.first,
                        backendReachable: errorMessage == nil
                    )

                    ForEach(devices) { device in
                        NavigationLink(value: device) {
                            DeviceCard(device: device)
                        }
                        .buttonStyle(.plain)
                    }

                    if devices.isEmpty {
                        emptyOrLoading
                    }
                }
                .padding(.horizontal, SD.Space.md)
                .padding(.bottom, SD.Space.lg)
            }
            .background(SD.Colors.background)
            .navigationDestination(for: Device.self) { DeviceDetailView(device: $0) }
            // A tapped notification opens its device, which then consumes
            // pendingEventID and shows the alert. Without this the id was set
            // and nothing was listening unless that screen was already open —
            // so the tap appeared to do nothing.
            .navigationDestination(item: $notificationTarget) { DeviceDetailView(device: $0) }
            // A cold launch from a tapped notification races this against
            // `.task { await reload() }`: pendingEventID can already be set
            // before `devices` has ever been fetched. Falling back straight to
            // `devices.first` — the old behaviour — evaluates to nil on an
            // empty array, and since .onChange only fires on pendingEventID
            // actually CHANGING (not on `devices` populating a moment later),
            // that was a permanent miss: notificationTarget stayed nil forever,
            // no navigation, no error, the tap silently did nothing. The empty
            // case now waits for the reload already in flight and retries once.
            // `initial: true`: a cold launch FROM a tapped notification can have
            // AlertNotifier already holding a non-nil pendingEventID by the
            // moment this view's body first runs and this modifier attaches —
            // onChange only fires on the value actually CHANGING afterward,
            // never for whatever it already was, so without this the tap that
            // launched the app was silently ignored. Same root cause as
            // DeviceDetailView's own onChange just below in the navigation
            // stack; see its comment for the in-app (not cold-launch) case.
            .onChange(of: notifier.pendingEventID, initial: true) { _, id in
                guard id != nil, notificationTarget == nil else { return }
                if let match = devices.first(where: { $0.id == notifier.pendingDeviceID }) {
                    notificationTarget = match
                } else if devices.isEmpty {
                    Task {
                        await reload()
                        notificationTarget = devices.first(where: { $0.id == notifier.pendingDeviceID })
                            ?? devices.first
                    }
                } else {
                    notificationTarget = devices.first
                }
            }
            .navigationTitle("SensorDyme")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    HStack(spacing: SD.Space.xs) {
                        // Sits beside the add button so a background refresh is
                        // visible without occupying a row of its own.
                        SDProgress(phase: refreshing.phase)
                        Button { showSetup = true } label: {
                            Label("Connect over Bluetooth", systemImage: "sensor.tag.radiowaves.forward")
                        }
                        .tint(SD.Colors.accent)
                        Button { showClaim = true } label: {
                            Label("Add device", systemImage: "plus")
                        }
                        .tint(SD.Colors.accent)
                    }
                }
            }
            .sheet(isPresented: $showClaim) { SetupWizardView { await reload() } }
            .sheet(isPresented: $showSetup) {
                SetupFlowView()
                    .onDisappear { Task { await link.ensureConnection() } }
            }
            .refreshable { await reload() }
            .task {
                await reload()
                // Re-resolve the device's address on launch and whenever the
                // network changes, so a move never requires typing an IP.
                link.start()
                await link.ensureConnection()
            }
            .onReceive(refreshTimer) { _ in Task { await reload() } }
        }
        .tint(SD.Colors.accent)
    }

    /// Distinguishes "still loading" from "genuinely nothing here" — showing an
    /// empty state during the first fetch reads as "you have no devices", which
    /// is alarming when you do.
    @ViewBuilder private var emptyOrLoading: some View {
        if !hasLoaded && errorMessage == nil {
            SDLoadingState(message: "Looking for your devices…")
        } else if errorMessage == nil {
            SDEmptyState(
                icon: "sensor.tag.radiowaves.forward",
                title: "No devices yet",
                message: "Add your SensorDyme with the claim code printed on the card in the box — or connect one over Bluetooth to put it on your Wi-Fi.",
                actionTitle: "Add device",
                action: { showClaim = true }
            )
        }
    }

    private func reload() async {
        let token = reloadGuard.begin()
        refreshing.begin()
        do {
            let fetched = try await state.backend.devices()
            guard reloadGuard.isLatest(token) else { return }
            devices = fetched
            errorMessage = nil
            refreshing.succeed()
        } catch let error as APIError where error.isAuthFailure {
            guard reloadGuard.isLatest(token) else { return }
            refreshing.fail()
            state.expireSession(error.message)
        } catch {
            guard reloadGuard.isLatest(token) else { return }
            refreshing.fail()
            errorMessage = error.localizedDescription
        }
        if reloadGuard.isLatest(token) { hasLoaded = true }
    }
}

// MARK: - Device card

struct DeviceCard: View {
    let device: Device

    private var status: SDStatus { device.online ? .online : .offline }

    /// "Offline" alone leaves people wondering whether it just happened or has
    /// been down all week, which is the first thing anyone wants to know.
    ///
    /// Checks `device.monitoring?.active`, same as DeviceDetailView one tap
    /// away — Models.swift's own doc comment on `CustomMonitoring` names the
    /// exact bug this guards against: showing the base mode as selected while
    /// a custom configuration is what the device is really doing, so someone
    /// who asked it to count people is told from the home screen that it's
    /// watching for driver distraction. That was fixed on the device screen;
    /// this card had the identical unconditional read of `activePreset` and
    /// wasn't.
    private var subtitle: String {
        if device.online {
            if device.monitoring?.active == true {
                return device.monitoring?.title ?? "AI Custom Monitoring"
            }
            return Preset.title(for: device.activePreset)
        }
        guard let lastSeen = device.lastSeen else { return "Not seen yet" }
        let elapsed = Date().timeIntervalSince(Date(timeIntervalSince1970: lastSeen))
        return "Last seen \(Self.elapsed.localizedString(fromTimeInterval: -elapsed))"
    }

    private static let elapsed: RelativeDateTimeFormatter = {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .full
        return formatter
    }()

    var body: some View {
        SDCard {
            HStack(spacing: SD.Space.sm) {
                ZStack {
                    RoundedRectangle(cornerRadius: SD.Radius.md, style: .continuous)
                        .fill(device.online ? SD.Colors.accentMuted : SD.Colors.surfaceRaised)
                        .frame(width: 46, height: 46)
                    Image(systemName: "video.fill")
                        .font(.system(size: 18))
                        .foregroundStyle(device.online ? SD.Colors.accent : SD.Colors.textTertiary)
                }

                VStack(alignment: .leading, spacing: SD.Space.xxs) {
                    Text(device.name)
                        .font(SD.Typography.headline)
                        .foregroundStyle(SD.Colors.textPrimary)
                        .lineLimit(1)
                    Text(subtitle)
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textSecondary)
                        .lineLimit(1)
                }

                Spacer(minLength: SD.Space.xs)

                SDStatusPill(status: status, pulses: device.online)

                Image(systemName: "chevron.right")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(SD.Colors.textTertiary)
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(device.name), \(status.label), \(subtitle)")
    }
}

// MARK: - Pairing

struct ClaimDeviceView: View {
    @EnvironmentObject private var state: AppState
    @Environment(\.dismiss) private var dismiss
    @State private var code = ""
    @State private var errorMessage: String?
    @State private var busy = false
    let onClaimed: () async -> Void

    var body: some View {
        NavigationStack {
            Form {
                Section("Claim code") {
                    TextField("1A2B-3C4D-5E6F", text: $code)
                        .textInputAutocapitalization(.characters)
                        .autocorrectionDisabled()
                        .font(SD.Typography.body.monospacedDigit())
                    Text("Printed on the card that came in the box.")
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textSecondary)
                }
                if let errorMessage {
                    Section {
                        Label(errorMessage, systemImage: "exclamationmark.triangle.fill")
                            .font(SD.Typography.callout)
                            .foregroundStyle(SD.Colors.danger)
                    }
                }
            }
            .navigationTitle("Add device")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button(busy ? "Pairing…" : "Pair", action: pair)
                        .disabled(code.isEmpty || busy)
                }
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
        }
        .tint(SD.Colors.accent)
    }

    private func pair() {
        busy = true
        errorMessage = nil
        Task {
            defer { busy = false }
            do {
                try await state.backend.claim(code: code.trimmingCharacters(in: .whitespaces))
                await onClaimed()
                dismiss()
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }
}
