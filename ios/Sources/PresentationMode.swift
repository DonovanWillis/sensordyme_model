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
// Nothing here decides anything. It is fed by whichever screen opened it
// (PresentationSource.swift): the account device screen, or (M13.1) a paired
// camera opened directly on the local network with the account server
// possibly down. The state is that screen's, the mode switch is that screen's
// (the account's `setMode`, which itself goes local when the camera answers
// this iPhone, or the M12 signed local command), the selected mode is what
// the DEVICE reports, and the alert is the screen's own evidence (the
// foreground alert path's sound policy, AlertSound.decide, is unchanged).
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

// PresentationModeOption(s), PresentationRule and PresentationSource: PresentationSource.swift.

struct PresentationModeView: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// Everything shown, built by the screen underneath on every update.
    let content: PresentationSource
    let session: LiveFeedSession
    /// The screen's own mode switch (an option's id).
    let onSwitch: (String) -> Void

    /// Presentation shows the AI view; one tap returns to the clean picture.
    @State private var aiView = true
    /// The option tapped here, for its spinner until the device reports it.
    @State private var requested: String?
    @State private var switchTaps = 0
    @State private var videoAspect: CGFloat?

    private var instrument: InstrumentState { content.instrument }
    private var options: [PresentationModeOption] { content.options }
    private var pendingBase: String? { content.pending }
    private var failure: String? { content.failure }
    private var latestAlert: EventItem? { content.latestAlert }

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
                        // The mode switch is pinned under the scrolling panel:
                        // an alert banner arriving above never moves it under a
                        // finger that is about to tap.
                        VStack(alignment: .leading, spacing: SD.Space.sm) {
                            ScrollView {
                                sidePanel.padding(.vertical, SD.Space.xs)
                            }
                            modeSwitch
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
                        modeSwitch
                            .padding(.horizontal, SD.Space.md)
                            .padding(.bottom, SD.Space.xs)
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
                    Image(systemName: Preset.icon(for: Preset.split(content.activePreset).base))
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(SD.Colors.textSecondary)
                        .accessibilityHidden(true)
                    Text(content.modeTitle)
                        .font(SD.Typography.headline)
                        .foregroundStyle(SD.Colors.textPrimary)
                        .lineLimit(2)
                        .minimumScaleFactor(0.8)
                }
                .accessibilityElement(children: .ignore)
                .accessibilityLabel("Mode: \(content.modeTitle)")
                .accessibilityIdentifier("present.mode")
                HStack(spacing: 5) {
                    Image(systemName: instrument.symbol)
                        .foregroundStyle(instrument.kind.tint)
                        .accessibilityHidden(true)
                    Text(instrument.word)
                        .foregroundStyle(SD.Colors.textPrimary)
                    Text("· \(content.presence)")
                        .foregroundStyle(SD.Colors.textSecondary)
                }
                .font(SD.Typography.callout.weight(.medium))
                .lineLimit(1)
                .minimumScaleFactor(0.8)
                .accessibilityElement(children: .ignore)
                .accessibilityLabel(instrument.word)
                .accessibilityValue(content.presence)
                .accessibilityIdentifier("present.state")
            }
            Spacer(minLength: SD.Space.xs)
            if content.hasLivePicture {
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
            switch content.picture {
            case .demo:
                SimulatedFeedView(preset: content.activePreset ?? "driver_safety", alert: alerting)
                    .aspectRatio(4 / 3, contentMode: .fit)
                    .dynamicTypeSize(...DynamicTypeSize.xxxLarge)
            case .live(let url):
                MJPEGView(url: url, session: session, hardwareID: content.hardwareID,
                          managedLocally: content.managedLocally, aiView: aiView)
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
            if content.hasLivePicture, aiView {
                AIViewChrome(model: session.reader.overlayModel, compact: true)
                    .padding(SD.Space.sm)
                    .transition(.opacity)
            }
        }
        .overlay(alignment: .bottomLeading) {
            if content.hasLivePicture {
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
            switch content.rule {
            case .none:
                EmptyView()
            case .running(let title, let summary, _):
                ruleRow(title: "\(title) · set by Jupiter", detail: summary)
            case .runningUnnamed:
                // The device says Jupiter's rules run; their words are the
                // account's, and this connection cannot read them.
                ruleRow(title: PresentationRule.unnamedTitle, detail: PresentationRule.unnamedDetail)
            }

            if content.showsDriverPanel {
                DriverMonitoringPanel(model: session.reader.overlayModel, monitoring: instrument,
                                      hasLiveFeed: content.hasLivePicture,
                                      lastEvent: content.lastDriverEvent, historyNotes: content.historyNotes,
                                      onDark: true)
            }
        }
    }

    private func ruleRow(title: String, detail: String) -> some View {
        HStack(alignment: .top, spacing: SD.Space.sm) {
            OrynMark(height: 13, color: SD.Colors.signal)
                .frame(width: 22, height: 18)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(SD.Typography.label)
                    .foregroundStyle(SD.Colors.textPrimary)
                    .fixedSize(horizontal: false, vertical: true)
                Text(detail)
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("present.rule")
    }

    private var modeSwitch: some View {
        VStack(alignment: .leading, spacing: SD.Space.xs) {
            SDEyebrow("Switch mode")
            if options.isEmpty {
                Text(content.optionsNote ?? "No other mode is available on this device.")
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
                Text("Switching — waiting for \(content.deviceName) to confirm")
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("present.switching")
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
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
