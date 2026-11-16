import SwiftUI

// M12: a camera paired with this iPhone, opened directly on the local network
// (docs/LOCAL_ACCESS.md) -- with the account server off, with no internet, or
// with no account at all. Live video over the local session, stop/resume and
// mode over signed requests, the camera's own history. Every state shown here
// is the camera's own fresh word, or said to be old.

/// A paired camera in a navigation path, by hardware id.
struct LocalDeviceRoute: Hashable {
    let hardwareID: String
}

/// Why the person is looking at a direct connection, said once at the top.
enum LocalContext {
    /// Signed in, but the account server isn't answering.
    case accountUnreachable
    /// Signed in and the server answers; this camera just isn't on the account.
    case notOnAccount
    /// No account on this iPhone.
    case noAccount

    var title: String {
        switch self {
        case .accountUnreachable: "Direct connection"
        case .notOnAccount: "Paired with this iPhone only"
        case .noAccount: "Not connected to an account"
        }
    }

    var message: String {
        switch self {
        case .accountUnreachable:
            "Your account server can't be reached, so this is a direct connection to the camera on this network. Events made now are kept on the camera and backed up when the server is back."
        case .notOnAccount:
            "This camera isn't on your account: no backup and no remote notifications. Add it with its claim code to get them."
        case .noAccount:
            "Not connected to an account — no backup, no remote notifications. Live view, stop and resume, and the camera's own history work on this network."
        }
    }
}

/// One paired camera in a list. Says only what is known: the device's own
/// fresh word, or how old the last one is.
struct LocalDeviceCard: View {
    let device: TrustedDevice
    @ObservedObject private var local = LocalAccessCenter.shared
    @Environment(\.dynamicTypeSize) private var typeSize
    var now = Date()

    private var presence: TrustedDevices.Presence { local.presence(device, now: now) }

    private var status: SDStatus {
        switch presence {
        case .live(let protection): protection == "protecting" ? .online : (protection == "stopped" ? .offline : .warning)
        case .searching: .connecting
        case .needsPairing: .warning
        case .notReachable, .awaitingDevice: .offline
        }
    }

    var body: some View {
        HStack(alignment: .top, spacing: SD.Space.sm) {
            Image(systemName: "video")
                .font(.system(size: 18, weight: .light))
                .foregroundStyle(TrustedDevices.isLive(presence) ? SD.Colors.textSecondary : SD.Colors.textTertiary)
                .frame(width: 40, height: 40)
                .background(RoundedRectangle(cornerRadius: SD.Radius.md, style: .continuous).fill(SD.Colors.canvas))
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 4) {
                Text(device.name)
                    .font(SD.Typography.headline)
                    .foregroundStyle(SD.Colors.textPrimary)
                    .fixedSize(horizontal: false, vertical: true)
                Text(TrustedDevices.detail(presence, now: now))
                    .font(SD.Typography.callout)
                    .foregroundStyle(SD.Colors.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                (typeSize.isAccessibilitySize
                    ? AnyLayout(VStackLayout(alignment: .leading, spacing: SD.Space.xxs))
                    : AnyLayout(HStackLayout(spacing: SD.Space.xs))) {
                    SDStatusPill(status: status, text: TrustedDevices.label(presence))
                    Text(device.hardwareID)
                        .font(SD.Typography.mono)
                        .foregroundStyle(SD.Colors.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: SD.Space.xs)
            Image(systemName: "chevron.right")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(SD.Colors.textSecondary)
                .accessibilityHidden(true)
        }
        .padding(SD.Space.md)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: SD.Radius.lg, style: .continuous).fill(SD.Colors.surface))
        .contentShape(Rectangle())
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(device.name), \(TrustedDevices.label(presence)), \(TrustedDevices.detail(presence, now: now))")
    }
}

/// The paired cameras, as a section of a list (the device list, the
/// no-account screen). Opening one pushes LocalDeviceView.
struct LocalDevicesSection: View {
    let devices: [TrustedDevice]
    let caption: String?
    var now = Date()

    var body: some View {
        VStack(alignment: .leading, spacing: SD.Space.sm) {
            SDSectionHeader(title: "Paired with this iPhone")
            if let caption {
                Text(caption)
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("local.caption")
            }
            ForEach(devices) { device in
                NavigationLink(value: LocalDeviceRoute(hardwareID: device.hardwareID)) {
                    LocalDeviceCard(device: device, now: now)
                }
                .buttonStyle(SDPressStyle(scale: 0.985))
                .accessibilityIdentifier("local.card")
                .accessibilityHint("Opens the camera directly on this network")
            }
        }
    }
}

/// One paired camera, directly.
struct LocalDeviceView: View {
    let hardwareID: String
    @EnvironmentObject private var state: AppState
    @ObservedObject private var local = LocalAccessCenter.shared
    @Environment(\.dismiss) private var dismiss
    @State private var presets: LocalPresets?
    @State private var events: [LocalEvent] = []
    @State private var eventsFetchedAt = Date()
    @State private var eventsProblem: String?
    @State private var busy = false
    @State private var note: (text: String, confirmed: Bool)?
    @State private var switching: String?
    @State private var confirmStop = false
    @State private var confirmForget = false
    @State private var now = Date()
    /// Held, never observed: frames redraw the camera alone, never this screen.
    @State private var feedSession = LiveFeedSession()
    /// M13.1: what the AI saw, drawn over the picture. Off: the clean picture
    /// is the ordinary view, and it is not remembered between visits.
    @State private var aiView = false
    @State private var showPresentation = false
    /// The last mode switch's failure, or the camera not confirming it.
    @State private var modeFailure: String?
    @State private var lastPoll = Date.distantPast
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var device: TrustedDevice? { local.device(hardwareID) }

    /// The camera's address when this iPhone can open its stream.
    private var streamURL: URL? {
        guard device?.isUsable == true, let address = local.address(hardwareID) else { return nil }
        return URL(string: address)
    }

    /// Presentation mode, from this screen's own state (PresentationSource.swift):
    /// the camera's fresh reading, its own modes and history; nothing from the account.
    private var presentationSource: PresentationSource {
        LocalPresentation.source(
            name: device?.name ?? hardwareID, hardwareID: hardwareID, url: streamURL, presence: presence,
            state: local.readings[hardwareID.uppercased()]?.state, presets: presets, events: events,
            fetchedAt: eventsFetchedAt, now: now, pending: switching, failure: modeFailure,
            isAlert: { $0.isCritical })
    }

    private var context: LocalContext {
        if !state.isSignedIn || state.isDemo { return .noAccount }
        if case .connected = state.backendStatus { return .notOnAccount }
        return .accountUnreachable
    }

    private var presence: TrustedDevices.Presence {
        device.map { local.presence($0, now: now) } ?? .notReachable(lastSeen: nil)
    }

    private var protection: String? {
        if case .live(let protection) = presence { return protection }
        return nil
    }

    private var live: Bool { TrustedDevices.isLive(presence) }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: SD.Space.lg) {
                ConnectionBanner(message: context.message, title: context.title)
                    .accessibilityIdentifier("local.banner")
                if let device { header(device) }
                camera
                presentButton
                monitoring
                if live, let presets, !presets.presets.isEmpty { modes(presets) }
                history
                forgetButton
            }
            .padding(.horizontal, SD.Space.md)
            .padding(.bottom, SD.Space.xl)
        }
        .orynScreen()
        .navigationTitle(device?.name ?? hardwareID)
        .navigationBarTitleDisplayMode(.inline)
        .confirmationDialog(ProtectionControl.stopConfirmationTitle, isPresented: $confirmStop, titleVisibility: .visible) {
            Button("Stop monitoring", role: .destructive) { send(stop: true) }
            Button("Keep monitoring", role: .cancel) {}
        } message: {
            Text(ProtectionControl.stopConfirmationMessage)
        }
        .confirmationDialog("Forget this camera on this iPhone?", isPresented: $confirmForget, titleVisibility: .visible) {
            Button("Forget camera", role: .destructive) {
                Task {
                    await local.forget(hardwareID)
                    dismiss()
                }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This iPhone stops trusting the camera and asks it to forget this iPhone. To use it again, pair it over Bluetooth or through your account.")
        }
        // M13.1: presentation mode over the direct connection. Rebuilt from this
        // screen's state on every update (the 4 s read, a mode confirmed); the
        // mode switch is the signed local command (`choose`).
        .fullScreenCover(isPresented: $showPresentation) {
            PresentationModeView(content: presentationSource, session: feedSession, onSwitch: { choose($0) })
                .task { await poll() }
        }
        .task { await poll() }
    }

    /// While on screen: the camera's own state every 4 s, its modes and
    /// history every 12 s. Nothing runs once the screen is gone.
    ///
    /// Run by this screen and by presentation mode over it (a full-screen
    /// cover can end the screen's own task), whichever is alive; a loop that
    /// finds the other has just read the camera skips its turn, so the camera
    /// is never read twice as often.
    private func poll() async {
        var tick = 0
        while !Task.isCancelled {
            if Date().timeIntervalSince(lastPoll) >= 3 {
                lastPoll = Date()
                now = Date()
                await local.refresh(hardwareID)
                now = Date()
                if tick % 3 == 0, local.isReady(hardwareID) {
                    if presets == nil || tick % 6 == 0 { presets = await local.presets(hardwareID) }
                    await loadEvents()
                }
                tick += 1
            }
            try? await Task.sleep(for: .seconds(4))
        }
    }

    // MARK: header

    private func header(_ device: TrustedDevice) -> some View {
        VStack(alignment: .leading, spacing: SD.Space.xs) {
            HStack(spacing: SD.Space.xs) {
                SDStatusPill(status: live ? (protection == "protecting" ? .online : .offline)
                                 : (presence == .searching ? .connecting : .offline),
                             text: TrustedDevices.label(presence))
                Text(device.hardwareID)
                    .font(SD.Typography.mono)
                    .foregroundStyle(SD.Colors.textSecondary)
            }
            Text(TrustedDevices.detail(presence, now: now))
                .font(SD.Typography.callout)
                .foregroundStyle(SD.Colors.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityIdentifier("local.status")
            if let problem = local.problems[hardwareID.uppercased()] {
                Label(problem, systemImage: "exclamationmark.triangle.fill")
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.warning)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("local.problem")
            }
        }
        .accessibilityElement(children: .combine)
    }

    // MARK: camera

    @ViewBuilder private var camera: some View {
        let shape = RoundedRectangle(cornerRadius: SD.Radius.camera, style: .continuous)
        Group {
            if let url = streamURL {
                ZStack {
                    MJPEGView(url: url, session: feedSession, hardwareID: hardwareID, managedLocally: true,
                              aiView: aiView)
                    CameraOffCover(reader: feedSession.reader)
                }
                // The same chrome as the account's camera (CameraViewport):
                // the live state and, with the AI view on, what the analysis
                // says; the Simulated badge whenever the stream says so.
                .overlay(alignment: .topLeading) {
                    VStack(alignment: .leading, spacing: 6) {
                        MJPEGLiveIndicator(reader: feedSession.reader, monitoringStopped: protection == "stopped")
                            .dynamicTypeSize(...DynamicTypeSize.xxxLarge)
                            .accessibilityShowsLargeContentViewer()
                        if aiView {
                            AIViewChrome(model: feedSession.reader.overlayModel, compact: true)
                                .transition(.opacity)
                        }
                    }
                    .padding(SD.Space.sm)
                }
                .overlay(alignment: .topTrailing) {
                    SimulatedBadgeIfNeeded(model: feedSession.reader.overlayModel)
                        .padding(SD.Space.sm)
                }
                .overlay(alignment: .bottomLeading) {
                    HStack(spacing: SD.Space.xs) {
                        AIViewToggle(isOn: $aiView)
                        if aiView { AIPerfReadout(model: feedSession.reader.overlayModel) }
                    }
                    .dynamicTypeSize(...DynamicTypeSize.xxxLarge)
                    .accessibilityShowsLargeContentViewer()
                    .padding(SD.Space.xxs)
                    .padding(.leading, SD.Space.xxs)
                }
            } else {
                CameraPlaceholder(interrupted: true)
                    .frame(minHeight: 200)
            }
        }
        .frame(maxWidth: .infinity)
        .clipShape(shape)
        .overlay(shape.strokeBorder(Color.white.opacity(0.08), lineWidth: 1))
    }

    // MARK: presentation

    private var presentButton: some View {
        Button {
            var transaction = Transaction()
            transaction.disablesAnimations = reduceMotion
            withTransaction(transaction) { showPresentation = true }
        } label: {
            Label("Present", systemImage: "play.rectangle")
        }
        .buttonStyle(SDButtonStyle(kind: .secondary))
        .accessibilityLabel("Presentation mode")
        .accessibilityHint("Full screen: the camera, what the AI sees, and the mode switch, directly on this network")
        .accessibilityIdentifier("local.present")
    }

    // MARK: monitoring and stop / resume

    private var monitoringText: String {
        switch protection {
        case "protecting"?: return "Monitoring — the camera reports it is protecting."
        case "stopped"?: return "Stopped — nothing is being watched."
        case "not_protecting"?:
            let reasons = local.readings[hardwareID.uppercased()]?.state.policy?.protection?.reasons ?? []
            return "Not protecting" + (reasons.isEmpty ? "." : ": " + reasons.joined(separator: "; ") + ".")
        default:
            return live ? "Connected; the camera didn't say whether it is monitoring."
                : "Unknown — the camera isn't answering on this network, so nothing here says what it is doing."
        }
    }

    private var monitoring: some View {
        VStack(alignment: .leading, spacing: SD.Space.sm) {
            SDSectionHeader(title: "Monitoring")
            Text(monitoringText)
                .font(SD.Typography.body)
                .foregroundStyle(SD.Colors.textPrimary)
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityIdentifier("local.monitoring")
            let stopped = protection == "stopped"
            Button {
                if stopped { send(stop: false) } else { confirmStop = true }
            } label: {
                HStack(spacing: 6) {
                    if busy { ProgressView().controlSize(.small) }
                    Image(systemName: stopped ? "play.fill" : "pause.fill")
                        .font(.system(size: 11, weight: .bold))
                        .accessibilityHidden(true)
                    Text(busy ? (stopped ? "Resuming…" : "Stopping…") : (stopped ? "Resume" : "Stop"))
                }
            }
            .buttonStyle(SDButtonStyle(kind: stopped ? .primary : .secondary))
            .disabled(!live || busy || device?.enrolment != .active)
            .accessibilityLabel(busy ? (stopped ? "Resuming" : "Stopping")
                                : (stopped ? "Resume, directly on the camera" : "Stop, directly on the camera"))
            .accessibilityIdentifier("local.protection")
            if let note {
                Text(note.text)
                    .font(SD.Typography.callout)
                    .foregroundStyle(note.confirmed ? SD.Colors.textSecondary : SD.Colors.warning)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("local.protectionNote")
            } else if !live {
                Text("Stop and resume work once the camera answers on this network.")
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func send(stop: Bool) {
        busy = true
        note = nil
        Task {
            defer { busy = false }
            switch await local.setProtection(hardwareID, stop: stop) {
            case .success(let outcome):
                note = (outcome.text, outcome.isConfirmed)
            case .failure(let error):
                note = (error.message, false)
            }
            now = Date()
        }
    }

    // MARK: modes

    private func modes(_ presets: LocalPresets) -> some View {
        VStack(alignment: .leading, spacing: SD.Space.sm) {
            SDSectionHeader(title: "Mode")
            ForEach(presets.presets) { preset in
                let active = presets.activeBase == preset.name
                Button {
                    choose(preset.name)
                } label: {
                    HStack {
                        Image(systemName: Preset.icon(for: preset.name))
                            .frame(width: 24)
                            .accessibilityHidden(true)
                        Text(preset.title)
                            .font(SD.Typography.body)
                            .fixedSize(horizontal: false, vertical: true)
                        Spacer()
                        if switching == preset.name {
                            ProgressView().controlSize(.small)
                        } else if active {
                            Image(systemName: "checkmark").accessibilityLabel("Active")
                        }
                    }
                    .foregroundStyle(SD.Colors.textPrimary)
                    .frame(minHeight: 44)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .disabled(switching != nil || active)
                .accessibilityAddTraits(active ? .isSelected : [])
                .accessibilityIdentifier("local.mode.\(preset.name)")
            }
            Text("Chosen on the camera directly. Your account learns the change from the camera's next report.")
                .font(SD.Typography.caption)
                .foregroundStyle(SD.Colors.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// Switch mode over the signed local path (`X-SD-Local`), from this screen's
    /// list or from presentation. Selected only once the camera reports it
    /// (`LocalPresets.reports`), then the list is read again from the camera.
    private func choose(_ name: String) {
        guard switching == nil else { return }
        switching = name
        note = nil
        modeFailure = nil
        Task {
            defer { switching = nil }
            switch await local.setPreset(hardwareID, name: name) {
            case .success(let outcome):
                note = (outcome.text, outcome.isConfirmed)
                if !outcome.isConfirmed { modeFailure = outcome.text }
                presets = await local.presets(hardwareID)
            case .failure(let error):
                // A 409 is the camera's role floor: its own reason, verbatim.
                note = (error.message, false)
                modeFailure = error.message
            }
            now = Date()
        }
    }

    // MARK: history

    private func loadEvents() async {
        do {
            let page = try await local.events(hardwareID, limit: 50)
            events = page.events
            eventsFetchedAt = Date()
            eventsProblem = page.fault
        } catch let error as LocalAccessError {
            eventsProblem = error.message
        } catch {
            eventsProblem = error.localizedDescription
        }
    }

    private var history: some View {
        VStack(alignment: .leading, spacing: SD.Space.sm) {
            SDSectionHeader(title: "On this camera")
            if let eventsProblem {
                Text(eventsProblem)
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.warning)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if events.isEmpty {
                Text(live ? "Nothing recorded on the camera yet." : "The camera's history is read from it directly, once it answers.")
                    .font(SD.Typography.callout)
                    .foregroundStyle(SD.Colors.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                ForEach(events.prefix(30), id: \.eventKey) { event in
                    LocalEventRow(event: event, fetchedAt: eventsFetchedAt)
                }
            }
        }
        .accessibilityIdentifier("local.history")
    }

    private var forgetButton: some View {
        Button(role: .destructive) {
            confirmForget = true
        } label: {
            Label("Forget this camera on this iPhone", systemImage: "trash")
        }
        .buttonStyle(SDButtonStyle(kind: .destructive))
        .padding(.top, SD.Space.md)
        .accessibilityIdentifier("local.forget")
    }
}

/// One event from the camera's own journal.
struct LocalEventRow: View {
    let event: LocalEvent
    let fetchedAt: Date

    private var item: EventItem { LocalHistory.eventItem(event, fetchedAt: fetchedAt) }

    private var when: String {
        if let age = event.ageS { return TrustedDevices.age(age + Date().timeIntervalSince(fetchedAt)) }
        return item.deviceDate.map { $0.formatted(date: .abbreviated, time: .shortened) + " (camera clock)" } ?? "time unknown"
    }

    var body: some View {
        HStack(alignment: .top, spacing: SD.Space.sm) {
            Circle()
                .fill(item.severityStatus.tint)
                .frame(width: 8, height: 8)
                .padding(.top, 6)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                Text(item.title + (event.simulated == true ? " (simulated)" : ""))
                    .font(SD.Typography.body)
                    .foregroundStyle(SD.Colors.textPrimary)
                    .fixedSize(horizontal: false, vertical: true)
                Text("\(when) · \(LocalHistory.syncLabel(event.sync))")
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .accessibilityElement(children: .combine)
    }
}

/// Signed out, or never signed in: the cameras paired with this iPhone over
/// Bluetooth, opened directly. Labelled as what it is.
struct LocalOnlyHomeView: View {
    @ObservedObject private var local = LocalAccessCenter.shared
    @State private var now = Date()
    @StateObject private var watcher = AlertWatcher()
    @EnvironmentObject private var state: AppState

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: SD.Space.lg) {
                ConnectionBanner(message: LocalContext.noAccount.message, title: LocalContext.noAccount.title)
                if local.devices.isEmpty {
                    SDEmptyState(icon: "video", title: "No paired cameras",
                                 message: "Set up a camera nearby over Bluetooth to pair it with this iPhone.")
                } else {
                    LocalDevicesSection(devices: local.devices, caption: nil, now: now)
                }
            }
            .padding(.horizontal, SD.Space.md)
            .padding(.bottom, SD.Space.xl)
        }
        .orynScreen()
        .navigationTitle("Paired cameras")
        .navigationBarTitleDisplayMode(.inline)
        .navigationDestination(for: LocalDeviceRoute.self) { LocalDeviceView(hardwareID: $0.hardwareID) }
        .task {
            watcher.start(state: state)
            while !Task.isCancelled {
                now = Date()
                await local.refreshAll()
                try? await Task.sleep(for: .seconds(5))
            }
        }
        .onDisappear { watcher.stop() }
    }
}
