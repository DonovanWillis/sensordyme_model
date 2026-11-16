import SwiftUI
import UIKit

// MARK: - Presentation mode (M13): the camera, what the AI sees, nothing else
//
// A full-screen, camera-first view of one device for showing it to someone:
// the live picture with the AI view, the mode it runs, its monitoring state
// (InstrumentState, the device screen's own truth logic), what driver
// monitoring is doing (DriverMonitoringPanel), an alert when one arrives, and
// a one-tap switch between the modes this device can run.
//
// Nothing here decides anything: the state is the device screen's, the mode
// switch is the device screen's `setMode` (the M12 signed local path when the
// camera answers this iPhone directly, the account otherwise), the selected
// mode is what the DEVICE reports, and the alert is the existing foreground
// alert path (AlertWatcher; its sound policy, AlertSound.decide, is unchanged).
// No developer detail beyond the device's own measured frame rate and timings
// in the AI view.

/// The alert as it arrives: shown while the newest alert is fresh, or while
/// the device's live analysis says an alert condition is active. Its words
/// are the device's: the live "why" line when this iPhone sees the analysis,
/// else the alert in the history with its recorded signals. It cross-fades
/// in; nothing pulses. Observes the throttled overlay model only.
private struct PresentationAlertBanner: View {
    @ObservedObject var model: LiveOverlayModel
    let latestAlert: EventItem?
    let fallback: String?
    let fresh: Bool

    var body: some View {
        if fresh || model.summary.alertActive {
            let panel = model.summary.panel
            // An alert condition now: said as the condition, never as an
            // older event's words. Otherwise the fresh alert, in its words.
            let text = model.summary.alertActive
                ? OverlayWhyLine.bannerText(conditionHeadline: panel.state.headline, why: panel.why)
                : (panel.why?.text ?? fallback)
            HStack(alignment: .top, spacing: SD.Space.sm) {
                Image(systemName: "bell.badge.fill")
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(SD.Colors.danger)
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 2) {
                    Text(latestAlert?.isSimulated == true || model.summary.simulated
                         ? "Driver alert · simulated" : "Driver alert")
                        .font(SD.Typography.headline)
                        .foregroundStyle(SD.Colors.textPrimary)
                    if let text {
                        Text(text)
                            .font(SD.Typography.callout)
                            .foregroundStyle(SD.Colors.textPrimary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                Spacer(minLength: 0)
            }
            .padding(SD.Space.sm)
            .background(RoundedRectangle(cornerRadius: SD.Radius.lg, style: .continuous).fill(SD.Colors.dangerMuted))
            .overlay(RoundedRectangle(cornerRadius: SD.Radius.lg, style: .continuous)
                .strokeBorder(SD.Colors.danger, lineWidth: 1.5))
            .accessibilityElement(children: .combine)
            .accessibilityIdentifier("present.alert")
            .transition(.opacity)
        }
    }
}

/// One mode the switch offers: a preset, or a variant of the running preset.
struct PresentationModeOption: Identifiable, Equatable {
    /// What `setMode` sends: "driver_safety" or "driver_safety:strict".
    let id: String
    /// "Driver Safety · Strict": what VoiceOver says.
    let title: String
    /// "Strict": what the chip says (the mode's name is in the header).
    let short: String
    let base: String
    /// The device reports this mode running (never the tap).
    let selected: Bool
}

enum PresentationModeOptions {
    /// The modes to offer. The presets this device can run; when that is only
    /// the running one (a role allows one mode, e.g. a driver-safety device),
    /// its variants instead, so there is still a real choice to make.
    static func make(catalog: PresetCatalog, activePreset: String?, customReplacesModes: Bool,
                     isAvailable: (String) -> Bool) -> [PresentationModeOption] {
        let active = Preset.split(activePreset)
        let available = catalog.presets.filter { isAvailable($0.id) }
        if available.count == 1, let only = available.first, !only.variants.isEmpty {
            let current = active.base == only.id && !customReplacesModes
                ? (active.variant ?? only.variants.first?.id) : nil
            return only.variants.map { variant in
                PresentationModeOption(id: "\(only.id):\(variant.id)", title: "\(only.title) · \(variant.title)",
                                       short: variant.title, base: only.id, selected: current == variant.id)
            }
        }
        return available.map { preset in
            PresentationModeOption(id: preset.id, title: preset.title, short: preset.title, base: preset.id,
                                   selected: active.base == preset.id && !customReplacesModes)
        }
    }
}

struct PresentationModeView: View {
    @EnvironmentObject private var state: AppState
    @Environment(\.dismiss) private var dismiss
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    let device: Device
    let instrument: InstrumentState
    /// The presence word ("Online", "Stale" ...), as the device screen shows it.
    let presence: String
    let modeTitle: String
    /// Jupiter's rules running on the device, as the device reports them.
    let customRule: (title: String, summary: String, replacesModes: Bool)?
    let options: [PresentationModeOption]
    /// The device screen's switch in flight (its base), and the last failure.
    let pendingBase: String?
    let failure: String?
    /// The newest alert in the last minute (the device screen's `alertActive` evidence).
    let latestAlert: EventItem?
    let lastDriverEvent: EventItem?
    let session: LiveFeedSession
    let onSwitch: (String) -> Void

    /// Presentation shows the AI view; one tap returns to the clean picture.
    @State private var aiView = true
    /// The option tapped here, for its spinner until the device reports it.
    @State private var requested: String?
    @State private var switchTaps = 0
    @State private var videoAspect: CGFloat?

    private var source: CameraSource { CameraSource(state: state, device: device) }

    var body: some View {
        GeometryReader { geo in
            let landscape = geo.size.width > geo.size.height
            ZStack {
                Color.black.ignoresSafeArea()
                if landscape {
                    HStack(alignment: .top, spacing: SD.Space.md) {
                        VStack(spacing: SD.Space.sm) {
                            header
                            video.frame(maxHeight: .infinity)
                        }
                        ScrollView {
                            sidePanel.padding(.vertical, SD.Space.xs)
                        }
                        .frame(width: min(360, geo.size.width * 0.38))
                    }
                    .padding(.horizontal, SD.Space.md)
                    .padding(.vertical, SD.Space.xs)
                } else {
                    VStack(spacing: SD.Space.sm) {
                        header.padding(.horizontal, SD.Space.md)
                        video.padding(.horizontal, SD.Space.xs)
                        ScrollView {
                            sidePanel
                                .padding(.horizontal, SD.Space.md)
                                .padding(.bottom, SD.Space.md)
                        }
                    }
                    .padding(.top, SD.Space.xs)
                }
            }
        }
        .preferredColorScheme(.dark)
        .statusBarHidden(true)
        .persistentSystemOverlays(.hidden)
        .accessibilityAction(.escape) { dismiss() }
        .onAppear { UIApplication.shared.isIdleTimerDisabled = true }
        .onDisappear { UIApplication.shared.isIdleTimerDisabled = false }
        .onChange(of: alerting) { _, now in
            // Said once when it arrives, for whoever can't see the screen.
            if now { AccessibilityNotification.Announcement("Driver alert. \(alertDetail ?? "")").post() }
        }
        .onChange(of: options) { _, new in
            if let requested, new.first(where: { $0.id == requested })?.selected == true { self.requested = nil }
        }
        .onChange(of: failure) { _, new in if new != nil { requested = nil } }
        .onChange(of: pendingBase) { old, new in
            // The device screen stopped waiting (confirmed, failed or timed out).
            if old != nil, new == nil, failure == nil,
               let requested, options.first(where: { $0.id == requested })?.selected != true {
                self.requested = nil
            }
        }
        .animation(SD.Motion.resolved(SD.Motion.status, reduceMotion: reduceMotion), value: alerting)
    }

    // MARK: Header: exit, mode, monitoring state, simulated

    private var header: some View {
        HStack(alignment: .center, spacing: SD.Space.sm) {
            Button { dismiss() } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(SD.Colors.textPrimary)
                    .frame(width: 40, height: 40)
                    .sdFloatingSurface(Circle(), shadow: false)
                    .frame(width: 44, height: 44)
                    .contentShape(Rectangle())
            }
            .buttonStyle(SDPressStyle(scale: 0.92))
            .accessibilityLabel("Exit presentation")
            .accessibilityIdentifier("present.close")

            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Image(systemName: Preset.icon(for: Preset.split(device.activePreset).base))
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(SD.Colors.textSecondary)
                        .accessibilityHidden(true)
                    Text(modeTitle)
                        .font(SD.Typography.headline)
                        .foregroundStyle(SD.Colors.textPrimary)
                        .lineLimit(2)
                        .minimumScaleFactor(0.8)
                }
                .accessibilityElement(children: .ignore)
                .accessibilityLabel("Mode: \(modeTitle)")
                .accessibilityIdentifier("present.mode")
                HStack(spacing: 5) {
                    Image(systemName: instrument.symbol)
                        .foregroundStyle(instrument.kind.tint)
                        .accessibilityHidden(true)
                    Text(instrument.word)
                        .foregroundStyle(SD.Colors.textPrimary)
                    Text("· \(presence)")
                        .foregroundStyle(SD.Colors.textSecondary)
                }
                .font(SD.Typography.callout.weight(.medium))
                .lineLimit(1)
                .minimumScaleFactor(0.8)
                .accessibilityElement(children: .ignore)
                .accessibilityLabel(instrument.word)
                .accessibilityValue(presence)
                .accessibilityIdentifier("present.state")
            }
            Spacer(minLength: SD.Space.xs)
            if case .live = source.kind {
                VStack(alignment: .trailing, spacing: 6) {
                    MJPEGLiveIndicator(reader: session.reader, monitoringStopped: instrument.kind == .stopped)
                    SimulatedBadgeIfNeeded(model: session.reader.overlayModel)
                }
                .dynamicTypeSize(...DynamicTypeSize.xxLarge)
                .accessibilityShowsLargeContentViewer()
            }
        }
    }

    // MARK: The picture with the AI view

    private var alerting: Bool { instrument.kind == .alert && instrument.live }

    /// What the alert banner says: the device's own words for the newest alert.
    private var alertDetail: String? {
        guard let latestAlert else { return nil }
        let words = latestAlert.signals.flatMap { $0.isEmpty ? nil : $0 }
        return words.map { "\(latestAlert.title): \($0)" } ?? latestAlert.title
    }

    private var video: some View {
        let shape = RoundedRectangle(cornerRadius: SD.Radius.camera, style: .continuous)
        return ZStack {
            switch source.kind {
            case .demo:
                SimulatedFeedView(preset: device.activePreset ?? "driver_safety", alert: alerting)
                    .aspectRatio(4 / 3, contentMode: .fit)
                    .dynamicTypeSize(...DynamicTypeSize.xxxLarge)
            case .live(let url):
                MJPEGView(url: url, session: session, hardwareID: device.hardwareId, aiView: aiView)
                CameraOffCover(reader: session.reader)
            case .none:
                CameraPlaceholder(interrupted: instrument.isInterrupted)
            }
        }
        // The picture's own proportions (FeedAspectRatioKey), so the frame, the
        // alert outline and the chips hug the picture instead of letterboxing.
        .onPreferenceChange(FeedAspectRatioKey.self) { videoAspect = $0 }
        .aspectRatio(videoAspect ?? 4 / 3, contentMode: .fit)
        .frame(maxWidth: .infinity)
        .clipShape(shape)
        .overlay(shape.strokeBorder(alerting ? SD.Colors.danger : Color.white.opacity(0.1),
                                    lineWidth: alerting ? 3 : 1))
        .overlay(alignment: .topLeading) {
            if case .live = source.kind, aiView {
                AIViewChrome(model: session.reader.overlayModel, compact: true)
                    .padding(SD.Space.sm)
                    .transition(.opacity)
            }
        }
        .overlay(alignment: .bottomLeading) {
            if case .live = source.kind {
                HStack(spacing: SD.Space.xs) {
                    AIViewToggle(isOn: $aiView)
                    if aiView { AIPerfReadout(model: session.reader.overlayModel) }
                }
                .dynamicTypeSize(...DynamicTypeSize.xxLarge)
                .padding(SD.Space.xxs)
                .padding(.leading, SD.Space.xxs)
            }
        }
    }

    /// The newest alert is fresh: the banner is for an event ARRIVING. The
    /// danger outline stays for the device screen's minute (`alerting`);
    /// the banner itself steps back after 15 s, so a demo with repeated
    /// events doesn't sit under a permanent red block. Re-evaluated on the
    /// device screen's 4 s refresh.
    private var alertIsFresh: Bool {
        guard alerting, let date = latestAlert?.date else { return false }
        return Date().timeIntervalSince(date) < 15
    }

    // MARK: Below (portrait) / beside (landscape) the picture

    private var sidePanel: some View {
        VStack(alignment: .leading, spacing: SD.Space.md) {
            PresentationAlertBanner(model: session.reader.overlayModel, latestAlert: latestAlert,
                                    fallback: alertDetail, fresh: alertIsFresh)
            if let rule = customRule {
                HStack(alignment: .top, spacing: SD.Space.sm) {
                    OrynMark(height: 13, color: SD.Colors.signal)
                        .frame(width: 22, height: 18)
                        .accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("\(rule.title) · set by Jupiter")
                            .font(SD.Typography.label)
                            .foregroundStyle(SD.Colors.textPrimary)
                            .fixedSize(horizontal: false, vertical: true)
                        Text(rule.summary)
                            .font(SD.Typography.caption)
                            .foregroundStyle(SD.Colors.textSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                .accessibilityElement(children: .combine)
                .accessibilityIdentifier("present.rule")
            }

            if Preset.split(device.activePreset).base == "driver_safety", customRule?.replacesModes != true {
                DriverMonitoringPanel(model: session.reader.overlayModel, monitoring: instrument,
                                      hasLiveFeed: { if case .live = source.kind { return true }; return false }(),
                                      lastEvent: lastDriverEvent, onDark: true)
            }

            modeSwitch
        }
    }

    private var modeSwitch: some View {
        VStack(alignment: .leading, spacing: SD.Space.xs) {
            SDEyebrow("Switch mode")
            if options.isEmpty {
                Text("No other mode is available on this device.")
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textSecondary)
            } else {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: SD.Space.xs) {
                        ForEach(options) { option in modeChip(option) }
                    }
                }
            }
            if let failure {
                Label(failure, systemImage: "exclamationmark.triangle.fill")
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.danger)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("present.switchFailed")
            } else if requested != nil || pendingBase != nil {
                Text("Switching — waiting for \(device.name) to confirm")
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("present.switching")
            }
        }
        .sensoryFeedback(.impact(weight: .light), trigger: switchTaps)
    }

    private func modeChip(_ option: PresentationModeOption) -> some View {
        let waiting = requested == option.id && !option.selected
        return Button {
            guard !option.selected else { return }
            switchTaps += 1
            requested = option.id
            onSwitch(option.id)
        } label: {
            HStack(spacing: 6) {
                if waiting {
                    ProgressView().controlSize(.mini).tint(SD.Colors.textPrimary)
                } else if option.selected {
                    Image(systemName: "checkmark")
                        .font(.system(size: 11, weight: .bold))
                }
                Text(option.short)
                    .font(SD.Typography.label.weight(.semibold))
                    .lineLimit(1)
            }
            .foregroundStyle(option.selected ? SD.Colors.textOnAccent : SD.Colors.textPrimary)
            .padding(.horizontal, SD.Space.md)
            .frame(minHeight: 44)
            .background(Capsule().fill(option.selected ? SD.Colors.accent : Color.white.opacity(0.1)))
            .contentShape(Capsule())
        }
        .buttonStyle(SDPressStyle(scale: 0.95))
        .accessibilityLabel(option.title)
        .accessibilityValue(option.selected ? "Active" : waiting ? "Switching" : "")
        .accessibilityAddTraits(option.selected ? .isSelected : [])
        .accessibilityIdentifier("present.mode.\(option.id)")
    }
}
