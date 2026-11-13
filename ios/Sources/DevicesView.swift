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
    /// A card zooms into its device's home (Continuity.swift).
    @Namespace private var zoomNS
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var devices: [Device] = []
    @State private var errorMessage: String?
    /// The banner's headline for `errorMessage` when the server answered with
    /// something unusable (ServerReply); nil = it couldn't be reached.
    @State private var errorTitle: String?
    /// Advanced by the refresh timer whether or not a reload runs, so a reply
    /// that stops being refreshed stops claiming ONLINE (ConnectionTruth).
    @State private var now = Date()
    @Environment(\.dynamicTypeSize) private var typeSize
    @State private var showClaim = false
    @State private var showSetup = false
    @State private var hasLoaded = false
    @StateObject private var link = DeviceLink()
    @StateObject private var notifier = AlertNotifier.shared
    /// The device a tapped notification belongs to, so the alert can be opened
    /// from anywhere in the app rather than only from the device screen.
    @State private var notificationTarget: Device?
    /// Why a tapped notification could not be opened, said rather than silent.
    @State private var alertOpenProblem: String?
    @StateObject private var refreshing = ProgressTracker()
    /// Same problem as DeviceDetailView's own copy of this: three independent
    /// triggers (appear, a 10s timer, pull-to-refresh) can all have a
    /// reload() in flight at once, with no guarantee their responses arrive
    /// in the order they were sent — without this, a slow, stale fetch can
    /// complete after a fresh one and silently overwrite it. See
    /// LatestRequestGuard's own docs.
    @State private var reloadGuard = LatestRequestGuard()

    private let refreshTimer = Timer.publish(every: 10, on: .main, in: .common).autoconnect()
    @State private var timedReloadInFlight = false

    /// Debug-only: the device a launch argument asked to open (see DebugLaunch).
    @State private var launchTarget: Device?

    /// Counted with the same rule as each card's pill (ConnectionTruth), so the
    /// summary can't say "1 online" above a card that says Stale or Unknown.
    private var onlineCount: Int {
        devices.filter { $0.isLive(now: now, lastRefreshFailed: errorMessage != nil) }.count
    }

    var body: some View {
        NavigationStack {
            routed
            // The home screen draws its own header (brand, actions, title) so
            // the ORYN lockup is never squeezed into a toolbar slot. Pushed
            // screens keep the standard navigation bar.
            .navigationTitle("Your Devices")
            .toolbar(.hidden, for: .navigationBar)
            .sheet(isPresented: $showClaim) { SetupWizardView { await reload() } }
            .sheet(isPresented: $showSetup) {
                SetupFlowView()
                    .onDisappear { Task { await link.ensureConnection() } }
            }
            .refreshable { await reload() }
            .task {
                await reload()
                if DebugLaunch.openDevice, launchTarget == nil { launchTarget = devices.first }
                // Re-resolve the device's address on launch and whenever the
                // network changes, so a move never requires typing an IP.
                link.start()
                await link.ensureConnection()
            }
            .onReceive(refreshTimer) { tick in
                now = tick
                // Same rule as DeviceDetailView: one timed reload at a time, and
                // none into a backend AppState is already retrying on its own.
                guard !timedReloadInFlight, state.periodicRefreshMayRun else { return }
                timedReloadInFlight = true
                Task {
                    await reload()
                    timedReloadInFlight = false
                }
            }
        }
        .tint(SD.Colors.accent)
    }

    /// The list with its destinations and notification routing.
    private var routed: some View {
            list
            .orynScreen()
            // Said once; gone after the person moves on, not pinned to the home screen.
            .onDisappear { alertOpenProblem = nil }
            .navigationDestination(for: Device.self) { device in
                DeviceDetailView(device: device)
                    .sdZoomDestination(.device(device.id), in: zoomNS, enabled: !reduceMotion)
            }
            .navigationDestination(item: $launchTarget) { DeviceDetailView(device: $0) }
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
            // Never another device as a fallback (NotificationRouting): that
            // screen ignored the alert, so the tap opened the wrong camera and
            // showed nothing.
            .onChange(of: notifier.pendingEventID, initial: true) { _, id in
                guard id != nil, notificationTarget == nil else { return }
                if devices.isEmpty {
                    Task {
                        await reload()
                        routeNotification()
                    }
                } else {
                    routeNotification()
                }
            }
    }

    private var list: some View {
            ScrollView {
                LazyVStack(alignment: .leading, spacing: SD.Space.md) {
                    header
                        .padding(.bottom, SD.Space.xxs)

                    // Shown whether or not devices are loaded: a failed refresh
                    // leaves stale cards on screen that look perfectly current.
                    if let errorMessage {
                        ConnectionBanner(message: errorMessage, title: errorTitle) { await reload() }
                            .padding(.bottom, SD.Space.xxs)
                    }
                    if let alertOpenProblem {
                        ConnectionBanner(message: alertOpenProblem, title: "Couldn't open that alert")
                            .padding(.bottom, SD.Space.xxs)
                    }

                    if !devices.isEmpty {
                        fleetSummary
                    }
                    // The backend's view of the same device, so the banner can
                    // tell "switched off" from "not on this Wi-Fi" instead of
                    // blaming the network for both.
                    DeviceLinkBanner(
                        link: link,
                        backendDevice: linkedBackendDevice,
                        backendReachable: errorMessage == nil,
                        accountHasDevices: !devices.isEmpty
                    )

                    ForEach(devices) { device in
                        NavigationLink(value: device) {
                            DeviceCard(device: device, zoomNS: zoomNS, zoomEnabled: !reduceMotion,
                                       lastRefreshFailed: errorMessage != nil, now: now)
                        }
                        .buttonStyle(SDPressStyle(scale: 0.985))
                        .accessibilityIdentifier("device.card")
                        .accessibilityHint("Opens the live view and controls")
                        .padding(.bottom, SD.Space.xs)
                    }

                    if devices.isEmpty {
                        emptyOrLoading
                    }
                }
                .padding(.horizontal, SD.Space.md)
                .padding(.bottom, SD.Space.lg)
            }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: SD.Space.lg) {
            HStack(spacing: SD.Space.xs) {
                OrynNavBrand()
                Spacer()
                // Beside the actions so a background refresh is visible
                // without occupying a row of its own. Working and failed only:
                // a green tick every 10 s said "the list refreshed" in the
                // colour that means "your device is fine".
                SDProgress(phase: refreshing.phase == .done ? .idle : refreshing.phase)
                headerButton("dot.radiowaves.left.and.right", label: "Connect over Bluetooth", id: "devices.bluetooth") {
                    showSetup = true
                }
                headerButton("plus", label: "Add device", id: "devices.add") { showClaim = true }
            }
            .padding(.top, SD.Space.xs)

            Text("Your Devices")
                .sdScaledFont(32, relativeTo: .largeTitle, weight: .semibold)
                .kerning(-0.6)
                .foregroundStyle(SD.Colors.textPrimary)
                .accessibilityAddTraits(.isHeader)
                .accessibilityIdentifier("devices.title")
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func headerButton(_ icon: String, label: String, id: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: icon)
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(SD.Colors.textPrimary)
                .frame(width: 38, height: 38)
                .background(Circle().fill(SD.Colors.surfaceRaised))
                // The circle stays 38 pt as designed; the target is the 44 pt minimum.
                .frame(width: 44, height: 44)
                .contentShape(Rectangle())
        }
        .buttonStyle(SDPressStyle(scale: 0.92))
        .accessibilityLabel(label)
        .accessibilityIdentifier(id)
    }

    /// One line under the title: how many devices, how many are up. Counted
    /// from the same `online` flag the cards show, so the two always agree.
    private var fleetSummary: some View {
        // Side by side normally; stacked at accessibility text sizes, where the
        // one-line row clipped both labels (audit).
        let layout = typeSize.isAccessibilitySize
            ? AnyLayout(VStackLayout(alignment: .leading, spacing: SD.Space.xxs))
            : AnyLayout(HStackLayout(spacing: SD.Space.xs))
        return layout {
            Text("\(devices.count) \(devices.count == 1 ? "device" : "devices")")
                .foregroundStyle(SD.Colors.textSecondary)
            if !typeSize.isAccessibilitySize {
                Text("·").foregroundStyle(SD.Colors.textTertiary).accessibilityHidden(true)
            }
            HStack(spacing: 6) {
                Circle()
                    .fill(onlineCount > 0 ? SD.Colors.success : SD.Colors.offline)
                    .frame(width: 6, height: 6)
                Text("\(onlineCount) online")
                    .foregroundStyle(onlineCount > 0 ? SD.Colors.textSecondary : SD.Colors.textTertiary)
            }
            if !typeSize.isAccessibilitySize { Spacer() }
        }
        .font(SD.Typography.callout.monospacedDigit())
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
    }

    /// The backend's record of the camera this phone links, for the banner.
    ///
    /// Was `devices.first`, which on a two-device account described the OTHER
    /// device: "hasn't reported in 3 days… switched off" about a camera that was
    /// running, using its neighbour's last_seen. Nil when it is ambiguous,
    /// which the banner says plainly rather than guessing.
    private var linkedBackendDevice: Device? {
        FeedLink.linkedDeviceIndex(
            hardwareIDs: devices.map(\.hardwareId), linkedID: state.linkedDeviceID
        ).map { devices[$0] }
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
                message: "Add your SensorDyme with the claim code printed on the card in the box, or connect one over Bluetooth to put it on your Wi-Fi.",
                actionTitle: "Add device",
                action: { showClaim = true }
            )
        }
    }

    private func routeNotification() {
        guard notifier.pendingEventID != nil else { return }
        if devices.isEmpty, errorMessage != nil {
            alertOpenProblem = "Your devices couldn't be loaded, so that alert can't be opened yet. "
                + "Try again once the server is back."
            notifier.pendingEventID = nil
            return
        }
        let route = NotificationRouting.route(pendingDeviceID: notifier.pendingDeviceID, deviceIDs: devices.map(\.id))
        if case .open(let id) = route, let match = devices.first(where: { $0.id == id }) {
            alertOpenProblem = nil
            notificationTarget = match
        } else {
            alertOpenProblem = NotificationRouting.message(route)
            notifier.pendingEventID = nil
        }
    }

    private func reload() async {
        let token = reloadGuard.begin()
        refreshing.begin()
        do {
            let fetched = try await state.backend.devices()
            guard reloadGuard.isLatest(token) else { return }
            devices = fetched
            state.recordAccountDevices(fetched)
            errorMessage = nil
            errorTitle = nil
            now = Date()
            refreshing.succeed()
        } catch let error as APIError where error.isAuthFailure {
            guard reloadGuard.isLatest(token) else { return }
            refreshing.fail()
            state.expireSession(error.message)
        } catch {
            guard reloadGuard.isLatest(token) else { return }
            refreshing.fail()
            errorMessage = error.localizedDescription
            errorTitle = (error as? APIError)?.title
        }
        if reloadGuard.isLatest(token) { hasLoaded = true }
    }
}

// MARK: - Device card

struct DeviceCard: View {
    let device: Device
    @Environment(\.dynamicTypeSize) private var typeSize
    var zoomNS: Namespace.ID? = nil
    var zoomEnabled = true
    /// The latest refresh failed: `online` is from an older reply, so the card
    /// says Unknown instead of a stale ONLINE (ConnectionTruth).
    var lastRefreshFailed = false
    /// The list's clock, so an old reply or old telemetry is judged against now.
    var now = Date()

    private var presence: ConnectionTruth.Presence {
        device.presence(now: now, lastRefreshFailed: lastRefreshFailed)
    }

    private var status: SDStatus { SDStatus(presence) }
    private var live: Bool { presence.isOnline }

    /// "Offline" alone leaves people wondering whether it just happened or has
    /// been down all week, which is the first thing anyone wants to know.
    ///
    /// Checks what custom rules are doing (`device.customRules`: the device's
    /// reported mode, else `monitoring.active`), same as DeviceDetailView one tap
    /// away — Models.swift's own doc comment on `CustomMonitoring` names the
    /// exact bug this guards against: showing the base mode as selected while
    /// a custom configuration is what the device is really doing, so someone
    /// who asked it to count people is told from the home screen that it's
    /// watching for driver distraction. That was fixed on the device screen;
    /// this card had the identical unconditional read of `activePreset` and
    /// wasn't.
    ///
    /// Online, the mode is prefixed with the device's monitoring state whenever
    /// it is not simply watching ("Stopped · Driver Safety", "Not monitoring ·
    /// ..."), from the backend's status contract (DeviceStatus.swift). The card
    /// used to read like a healthy device whatever the device reported.
    ///
    /// Stale or offline, the line says how old the last report is instead
    /// (ConnectionTruth.cardLine): a mode named under old telemetry reads as
    /// what the device is doing now.
    private var subtitle: String {
        if let line = ConnectionTruth.cardLine(presence) { return line }
        // Rules running alongside the mode (an additive role) leave the
        // mode in charge, so the mode is what is named.
        let modeTitle = device.customRules == .replacing
            ? (device.monitoring?.title ?? "AI Custom Monitoring")
            : Preset.title(for: device.activePreset)
        return MonitoringStatusPresentation.cardSubtitle(
            modeTitle: modeTitle, presentation: device.monitoringPresentation(live: live))
    }

    /// The mode glyph shown in the preview tile — what the camera is doing,
    /// not a generic camera icon.
    private var modeIcon: String {
        if device.customRules == .replacing { return "sparkles" }
        let base = Preset.split(device.activePreset).base
        return Preset.catalog.first(where: { $0.name == base })?.icon ?? "video"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: SD.Space.sm) {
            Group {
                if let zoomNS {
                    preview.sdZoomSource(.device(device.id), in: zoomNS, cornerRadius: SD.Radius.camera,
                                         enabled: zoomEnabled)
                } else {
                    preview
                }
            }
            HStack(alignment: .firstTextBaseline, spacing: SD.Space.sm) {
                VStack(alignment: .leading, spacing: 3) {
                    // Wraps instead of truncating: at the largest text sizes a
                    // one- or two-line limit cut the name off (audit: clipped).
                    Text(device.name)
                        .font(SD.Typography.headline)
                        .foregroundStyle(SD.Colors.textPrimary)
                        .multilineTextAlignment(.leading)
                        .fixedSize(horizontal: false, vertical: true)
                    Text(subtitle)
                        .font(SD.Typography.callout)
                        .foregroundStyle(SD.Colors.textSecondary)
                        .multilineTextAlignment(.leading)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: SD.Space.xs)
                Image(systemName: "chevron.right")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(SD.Colors.textSecondary)
                    .accessibilityHidden(true)
            }
            .padding(.horizontal, SD.Space.xxs)
        }
        .contentShape(Rectangle())
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(device.name), \(ConnectionTruth.spoken(presence)), \(subtitle)")
    }

    /// A camera-shaped tile standing in for a thumbnail. The backend holds no
    /// frames (privacy by design — frames never leave the LAN), so this is a
    /// representation, not a picture: never a fake snapshot.
    private var preview: some View {
        ZStack {
            SD.Colors.canvas
            RadialGradient(colors: [Color.white.opacity(live ? 0.05 : 0.02), .clear],
                           center: .center, startRadius: 4, endRadius: 240)
            Image(systemName: modeIcon)
                .font(.system(size: 30, weight: .ultraLight))
                .foregroundStyle(live ? SD.Colors.textSecondary : SD.Colors.textTertiary)
                .accessibilityHidden(true)
            VStack {
                // Side by side normally; stacked at accessibility sizes, where the
                // pill was cut to "ONLI…" beside the wrapping id.
                (typeSize.isAccessibilitySize
                    ? AnyLayout(VStackLayout(alignment: .leading, spacing: SD.Space.xs))
                    : AnyLayout(HStackLayout())) {
                    SDStatusPill(status: status, text: ConnectionTruth.label(presence))
                    if !typeSize.isAccessibilitySize { Spacer() }
                    if let id = device.hardwareId {
                        // No shrink-to-fit: minimumScaleFactor stops the id following
                        // the reader's text size (audit: Dynamic Type partially
                        // unsupported); it wraps at large sizes instead.
                        Text(id)
                            .font(SD.Typography.mono)
                            .foregroundStyle(SD.Colors.textSecondary)
                            .multilineTextAlignment(.trailing)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                Spacer()
            }
            .padding(SD.Space.md)
        }
        // A floor, not a fixed height: the pill and id grow with Dynamic Type.
        .frame(minHeight: 168)
        .clipShape(RoundedRectangle(cornerRadius: SD.Radius.camera, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: SD.Radius.camera, style: .continuous)
                .strokeBorder(Color.white.opacity(0.08), lineWidth: 1)
        )
    }
}

/// A faint rule-of-thirds grid — a viewfinder, so a
/// device tile reads as a camera without pretending to be footage.
struct CameraGrid: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        for fraction in [1.0 / 3, 2.0 / 3] {
            path.move(to: CGPoint(x: rect.width * fraction, y: 0))
            path.addLine(to: CGPoint(x: rect.width * fraction, y: rect.height))
            path.move(to: CGPoint(x: 0, y: rect.height * fraction))
            path.addLine(to: CGPoint(x: rect.width, y: rect.height * fraction))
        }
        return path
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
            Group {
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
            .orynRows()
        }
        .orynFormStyle()
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
