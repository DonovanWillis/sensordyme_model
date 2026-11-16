import SwiftUI

// MARK: - The AI view (M13, sd-overlay/1, docs/LIVE_OVERLAY.md)
//
// What the device's AI saw on the frame on screen, drawn by the app over the
// CLEAN picture: thin rounded boxes (the driver highlighted, everything else
// muted, a phone labelled by whose it is), small tracker ids, the face outline
// and eye contours (never the full mesh), the dashed driver-seat region, the
// hands when the hand model ran. Every mark comes from that frame's document
// (LiveOverlay) and every word from LiveOverlay.swift's tested functions.
//
// Not here, by design: confidence percentages, eye gaze, invented motion
// (nothing animates on its own; marks move because the device saw them move),
// and anything from another frame.

// MARK: Drawing (per frame, inside MJPEGFrame's picture rect)

struct AIOverlayCanvas: View {
    let overlay: LiveOverlay

    var body: some View {
        Canvas { context, size in
            func rect(_ b: LiveOverlay.Box) -> CGRect {
                CGRect(x: b.x1 * size.width, y: b.y1 * size.height,
                       width: max(1, b.width * size.width), height: max(1, b.height * size.height))
            }

            func tag(_ text: String, at corner: CGPoint, color: Color, above: Bool = true, trailing: Bool = false) {
                let resolved = context.resolve(Text(text).font(.system(size: 10, weight: .semibold))
                    .foregroundColor(color))
                let measured = resolved.measure(in: CGSize(width: 240, height: 40))
                var plate = CGRect(x: trailing ? corner.x - measured.width - 10 : corner.x,
                                   y: above ? corner.y - measured.height - 5 : corner.y + 3,
                                   width: measured.width + 10, height: measured.height + 4)
                // Inside the picture: a box at the top edge puts its tag inside it.
                if plate.minY < 0 { plate.origin.y = corner.y + 3 }
                plate.origin.x = min(max(0, plate.origin.x), max(0, size.width - plate.width))
                plate.origin.y = min(max(0, plate.origin.y), max(0, size.height - plate.height))
                context.fill(Path(roundedRect: plate, cornerRadius: 4, style: .continuous),
                             with: .color(.black.opacity(0.66)))
                context.draw(resolved, in: plate.insetBy(dx: 5, dy: 2))
            }

            // 1. The driver's seat region: where the device looks for the driver.
            if let region = overlay.driverRegion {
                let r = rect(region.box).insetBy(dx: 1, dy: 1)
                context.stroke(Path(roundedRect: r, cornerRadius: 12, style: .continuous),
                               with: .color(.white.opacity(0.38)),
                               style: StrokeStyle(lineWidth: 1, dash: [6, 5]))
                // Top-right, inside: clear of the chips along the left and bottom.
                tag("Driver seat", at: CGPoint(x: r.maxX - 6, y: r.minY + 4), color: .white.opacity(0.8),
                    above: false, trailing: true)
            }

            // 2. Boxes: the detector's latest run. Faded when carried from an
            // earlier run (det_fresh false) -- the device says they are not new.
            // One path per stroke style, one stroke call each (M13.1): a frame
            // with a dozen boxes is a handful of strokes, not a dozen; the
            // tags are drawn after, on the outer context.
            if let detections = overlay.detections {
                var layer = context
                layer.opacity = overlay.boxesAreCarried ? 0.45 : 1
                var strokes: [(color: Color, width: CGFloat, path: Path)] = []
                var tags: [(label: String, at: CGPoint, color: Color)] = []
                for d in detections {
                    let style = AIBoxStyle.make(d, alertActive: overlay.alertActive)
                    let r = rect(d.box)
                    let box = Path(roundedRect: r, cornerRadius: 6, style: .continuous)
                    if let i = strokes.firstIndex(where: { $0.color == style.color && $0.width == style.lineWidth }) {
                        strokes[i].path.addPath(box)
                    } else {
                        strokes.append((style.color, style.lineWidth, box))
                    }
                    if let label = style.label {
                        tags.append((label, CGPoint(x: r.minX, y: r.minY),
                                     style.color.opacity(overlay.boxesAreCarried ? 0.6 : 1)))
                    }
                }
                for s in strokes { layer.stroke(s.path, with: .color(s.color), lineWidth: s.width) }
                // Drawn on the outer context so the tag's plate keeps its contrast.
                for t in tags { tag(t.label, at: t.at, color: t.color) }
            }

            // 3. Hands, only on frames where the hand model ran: one path.
            if let hands = overlay.hands, !hands.isEmpty {
                var path = Path()
                for hand in hands { path.addRoundedRect(in: rect(hand), cornerSize: CGSize(width: 5, height: 5), style: .continuous) }
                context.stroke(path, with: .color(SD.Colors.textSecondary.opacity(0.7)),
                               style: StrokeStyle(lineWidth: 1, dash: [2, 3]))
            }

            // 4. The face outline and the eyes: a thin contour, never the mesh.
            // The oval is one path; both eyes are one path.
            func contour(_ names: [String]) -> Path? {
                var path = Path()
                for name in names {
                    guard let points = overlay.drawableContour(name), let first = points.first else { continue }
                    path.move(to: CGPoint(x: first.x * size.width, y: first.y * size.height))
                    for p in points.dropFirst() { path.addLine(to: CGPoint(x: p.x * size.width, y: p.y * size.height)) }
                    path.closeSubpath()
                }
                return path.isEmpty ? nil : path
            }
            if let oval = contour(["oval"]) {
                context.stroke(oval, with: .color(SD.Colors.textPrimary.opacity(0.6)),
                               style: StrokeStyle(lineWidth: 1, lineJoin: .round))
            }
            if let eyes = contour(["eye_l", "eye_r"]) {
                let eyesClosed = overlay.face?.eyesClosed == true
                context.stroke(eyes, with: .color(eyesClosed ? SD.Colors.warning : SD.Colors.textPrimary.opacity(0.9)),
                               style: StrokeStyle(lineWidth: 1.2, lineJoin: .round))
            }
        }
        .allowsHitTesting(false)
    }
}

/// How one detection box is drawn: the driver bright, a phone by whose it is,
/// everything else muted. The labels are words and tracker ids only.
struct AIBoxStyle {
    let color: Color
    let lineWidth: CGFloat
    let label: String?

    static func make(_ d: LiveOverlay.Detection, alertActive: Bool) -> AIBoxStyle {
        let id = d.track.map { " #\($0)" } ?? ""
        switch (d.label, d.role) {
        case ("person", .driver?):
            return AIBoxStyle(color: alertActive ? SD.Colors.danger : SD.Colors.textPrimary, lineWidth: 1.6,
                              label: "Driver\(id)")
        case ("person", _):
            return AIBoxStyle(color: SD.Colors.textTertiary, lineWidth: 1, label: "Person\(id)")
        case ("phone", .driver?):
            return AIBoxStyle(color: alertActive ? SD.Colors.danger : SD.Colors.warning, lineWidth: 1.6,
                              label: "Phone · driver\(id)")
        case ("phone", .other?):
            return AIBoxStyle(color: SD.Colors.textTertiary, lineWidth: 1, label: "Phone · other\(id)")
        case ("phone", nil):
            return AIBoxStyle(color: SD.Colors.textSecondary, lineWidth: 1.2, label: "Phone\(id)")
        default:
            return AIBoxStyle(color: SD.Colors.textTertiary, lineWidth: 1,
                              label: DriverBehaviourCopy.readable(d.label) + id)
        }
    }
}

// MARK: Chips over the picture

/// The plate every chip over footage sits on: opaque-ish black, so its
/// contrast never depends on the picture (design system §3).
private struct FootagePlate: ViewModifier {
    var tint: Color? = nil
    func body(content: Content) -> some View {
        content
            .padding(.horizontal, 9)
            .padding(.vertical, 5)
            .background(Capsule().fill(Color.black.opacity(0.7)))
            .overlay(Capsule().strokeBorder((tint ?? .white).opacity(tint == nil ? 0.08 : 0.55), lineWidth: 1))
    }
}

extension View {
    fileprivate func footagePlate(tint: Color? = nil) -> some View { modifier(FootagePlate(tint: tint)) }
}

/// "SIMULATED": the source is a virtual device or a replay. On every frame
/// whose stream said so (SimulatedBadge), in the clean view as well.
struct SimulatedSourceBadge: View {
    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "testtube.2")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(SD.Colors.warning)
                .accessibilityHidden(true)
            Text("SIMULATED")
                .sdEyebrowFont()
                .foregroundStyle(SD.Colors.textPrimary)
                .lineLimit(1)
        }
        .footagePlate(tint: SD.Colors.warning)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Simulated video: a virtual device, not a real camera")
        .accessibilityIdentifier("ai.simulated")
    }
}

/// Shows the Simulated badge whenever the stream says so. Observes the
/// throttled overlay model only.
struct SimulatedBadgeIfNeeded: View {
    @ObservedObject var model: LiveOverlayModel
    var body: some View {
        if model.summary.simulated {
            SimulatedSourceBadge()
                .dynamicTypeSize(...DynamicTypeSize.xxxLarge)
                .accessibilityShowsLargeContentViewer()
                .transition(.opacity)
        }
    }
}

/// The AI view switch: the clean picture is the default; this draws what the
/// AI saw over it.
struct AIViewToggle: View {
    @Binding var isOn: Bool
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        Button {
            withAnimation(SD.Motion.resolved(SD.Motion.feedback, reduceMotion: reduceMotion)) { isOn.toggle() }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: isOn ? "viewfinder.circle.fill" : "viewfinder.circle")
                    .font(.system(size: 13, weight: .semibold))
                    .accessibilityHidden(true)
                Text("AI view")
                    .font(SD.Typography.caption.weight(.semibold))
                    .lineLimit(1)
            }
            .foregroundStyle(isOn ? SD.Colors.textOnAccent : SD.Colors.textPrimary)
            .padding(.horizontal, 11)
            .frame(minHeight: 32)
            .background(Capsule().fill(isOn ? SD.Colors.accent : Color.black.opacity(0.7)))
            .overlay(Capsule().strokeBorder(Color.white.opacity(isOn ? 0 : 0.12), lineWidth: 1))
            .frame(minHeight: 44)
            .contentShape(Rectangle())
        }
        .buttonStyle(SDPressStyle(scale: 0.95))
        .sensoryFeedback(.selection, trigger: isOn)
        .accessibilityLabel("AI view")
        .accessibilityValue(isOn ? "On, showing what the AI sees" : "Off, picture only")
        .accessibilityAddTraits(isOn ? .isSelected : [])
        .accessibilityIdentifier("ai.toggle")
    }
}

extension LiveAnalysisState.Kind {
    var symbol: String {
        switch self {
        case .watching: "eye"
        case .behaviour: "exclamationmark.circle"
        case .alert: "exclamationmark.triangle.fill"
        case .faceNotVisible: "person.crop.circle.badge.questionmark"
        case .detectionUnavailable: "eye.slash"
        case .insufficientEvidence: "questionmark.circle"
        case .notAnalysed: "minus.circle"
        case .waiting: "hourglass"
        case .unsupported: "nosign"
        }
    }

    var tint: Color {
        switch self {
        case .alert: SD.Colors.danger
        case .behaviour, .faceNotVisible, .detectionUnavailable, .insufficientEvidence: SD.Colors.warning
        case .watching: SD.Colors.textPrimary
        case .notAnalysed, .waiting, .unsupported: SD.Colors.textTertiary
        }
    }
}

extension BehaviourStripItem.Tone {
    var tint: Color {
        switch self {
        case .confirmed: SD.Colors.danger
        case .developing: SD.Colors.warning
        case .neutral: SD.Colors.textSecondary
        }
    }
}

/// One behaviour: "Head turned away · 2.1 s", amber while developing, the
/// alert colour once confirmed. The phase is also a word for VoiceOver.
struct BehaviourChip: View {
    let item: BehaviourStripItem
    var body: some View {
        HStack(spacing: 6) {
            Circle().fill(item.tone.tint).frame(width: 7, height: 7).accessibilityHidden(true)
            Text(item.text)
                .font(SD.Typography.caption.weight(.semibold).monospacedDigit())
                .foregroundStyle(SD.Colors.textPrimary)
                .lineLimit(1)
        }
        .footagePlate(tint: item.tone == .neutral ? nil : item.tone.tint)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(item.spoken)
    }
}

/// The AI view's words over the picture: what the analysis says right now,
/// the behaviour strip, eyes closed (only when the device says so), the head
/// indicator, the why line and the measured performance.
struct AIViewChrome: View {
    @ObservedObject var model: LiveOverlayModel
    /// The inline camera: the state and the strip only.
    var compact = false

    var body: some View {
        let summary = model.summary
        let panel = summary.panel
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: panel.state.kind.symbol)
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(panel.state.kind.tint)
                    .accessibilityHidden(true)
                Text(panel.state.kind == .behaviour || panel.state.kind == .alert
                     ? (panel.state.kind == .alert ? "Alert condition" : "Developing") : panel.state.headline)
                    .font(SD.Typography.caption.weight(.semibold))
                    .foregroundStyle(SD.Colors.textPrimary)
                    .lineLimit(1)
            }
            .footagePlate()
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("AI view: \(panel.state.headline)")
            .accessibilityValue(panel.state.detail ?? "")
            .accessibilityIdentifier("ai.state")

            ForEach(compact ? Array(panel.behaviours.prefix(1)) : panel.behaviours) { item in
                BehaviourChip(item: item)
                    .accessibilityIdentifier("ai.behaviour")
            }
            if !compact {
                if panel.eyesClosed {
                    Label("Eyes closed", systemImage: "eye.slash")
                        .font(SD.Typography.caption.weight(.semibold))
                        .foregroundStyle(SD.Colors.warning)
                        .footagePlate(tint: SD.Colors.warning)
                        .accessibilityIdentifier("ai.eyesClosed")
                }
                if let head = panel.head {
                    HeadIndicatorChip(head: head)
                }
                if let why = panel.why {
                    Text(why.text)
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textPrimary)
                        .lineLimit(2)
                        .footagePlate(tint: why.isAlert ? SD.Colors.danger : nil)
                        .accessibilityLabel("Why: \(why.text)")
                        .accessibilityIdentifier("ai.why")
                }
                if let unavailable = panel.unavailable {
                    Label(unavailable, systemImage: "exclamationmark.triangle")
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.warning)
                        .lineLimit(2)
                        .footagePlate()
                }
            }
        }
        .dynamicTypeSize(...DynamicTypeSize.xxLarge)
        .accessibilityShowsLargeContentViewer()
        .animation(.easeOut(duration: 0.2), value: summary.key)
    }
}

/// The device's measured performance: "14 fps · Detector 11.8 ms · Frame 83 ms".
struct AIPerfReadout: View {
    @ObservedObject var model: LiveOverlayModel
    var body: some View {
        if let perf = model.summary.perf {
            Text(perf)
                .font(SD.Typography.caption2Mono)
                .foregroundStyle(SD.Colors.textSecondary)
                .lineLimit(1)
                .footagePlate()
                .dynamicTypeSize(...DynamicTypeSize.xxLarge)
                .accessibilityLabel("Measured on the device: \(perf)")
                .accessibilityIdentifier("ai.perf")
        }
    }
}

/// "Head · Left" with a small dial: the needle is the device's head yaw. Head
/// orientation, never gaze.
struct HeadIndicatorChip: View {
    let head: HeadIndicator
    var body: some View {
        HStack(spacing: 7) {
            HeadDial(yawDeg: head.yawDeg)
                .frame(width: 22, height: 13)
                .accessibilityHidden(true)
            Text(head.text)
                .font(SD.Typography.caption.weight(.semibold))
                .foregroundStyle(SD.Colors.textPrimary)
                .lineLimit(1)
        }
        .footagePlate()
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(head.calibrated ? head.text : "\(head.text), not yet calibrated to this driver")
        .accessibilityIdentifier("ai.head")
    }
}

/// A half dial, -90 to 90 degrees of head yaw. Positive yaw (the device's
/// "Left", frame-relative) points the needle to the picture's left.
struct HeadDial: View {
    let yawDeg: Double?
    var body: some View {
        Canvas { context, size in
            let center = CGPoint(x: size.width / 2, y: size.height)
            let radius = min(size.width / 2, size.height) - 1
            var arc = Path()
            arc.addArc(center: center, radius: radius, startAngle: .degrees(180), endAngle: .degrees(0), clockwise: false)
            context.stroke(arc, with: .color(SD.Colors.textTertiary), lineWidth: 1)
            guard let yaw = yawDeg else { return }
            // 0 = straight up; +90 = left.
            let angle = (-90 - yaw) * .pi / 180
            var needle = Path()
            needle.move(to: center)
            needle.addLine(to: CGPoint(x: center.x + cos(angle) * radius, y: center.y + sin(angle) * radius))
            context.stroke(needle, with: .color(SD.Colors.textPrimary), style: StrokeStyle(lineWidth: 1.5, lineCap: .round))
        }
    }
}

// MARK: - The distracted-driving panel

/// What driver monitoring is doing, in one place on the device screen and in
/// presentation mode: the device's monitoring state, the live analysis (state,
/// behaviour in progress with its duration, eyes closed, head), and the last
/// event in the device's own words. With no live picture on this iPhone it
/// says so and shows what the account knows.
struct DriverMonitoringPanel: View {
    @ObservedObject var model: LiveOverlayModel
    /// The device's monitoring state word, from the existing status logic.
    let monitoring: InstrumentState
    /// The live analysis is available on this iPhone (the camera is reachable).
    let hasLiveFeed: Bool
    /// The newest driver event in the history, for when no live one is known.
    let lastEvent: EventItem?
    /// Per history `event_key`, where that row is kept ("In the camera's
    /// history · Not yet backed up"): the live why line names its row when the
    /// device sent the key (sd-overlay/1 `last_event.event_key`).
    var historyNotes: [String: String] = [:]
    var onDark = false

    private static let relative: RelativeDateTimeFormatter = {
        let f = RelativeDateTimeFormatter()
        f.unitsStyle = .short
        return f
    }()

    var body: some View {
        let panel = model.summary.panel
        VStack(alignment: .leading, spacing: SD.Space.sm) {
            HStack(alignment: .firstTextBaseline, spacing: SD.Space.xs) {
                SDEyebrow("Distracted driving")
                Spacer(minLength: SD.Space.xs)
                HStack(spacing: 5) {
                    Image(systemName: monitoring.symbol)
                        .foregroundStyle(monitoring.kind.tint)
                        .accessibilityHidden(true)
                    Text(monitoring.word)
                        .foregroundStyle(SD.Colors.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .font(SD.Typography.caption.weight(.medium))
            }
            .accessibilityElement(children: .combine)
            .accessibilityLabel("Distracted driving, monitoring: \(monitoring.word)")

            if hasLiveFeed {
                liveRows(panel)
            } else {
                Label("Live analysis appears when this iPhone can see the camera on the same network.",
                      systemImage: "video.slash")
                    .font(SD.Typography.callout)
                    .foregroundStyle(SD.Colors.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("driver.panel.noFeed")
            }

            lastEventRow(panel)
        }
        .padding(SD.Space.md)
        .background(
            RoundedRectangle(cornerRadius: SD.Radius.camera, style: .continuous)
                .fill(onDark ? Color.white.opacity(0.06) : SD.Colors.surface)
        )
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("driver.panel")
    }

    @ViewBuilder private func liveRows(_ panel: DriverPanelModel) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: SD.Space.sm) {
            Image(systemName: panel.state.kind.symbol)
                .foregroundStyle(panel.state.kind.tint)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                Text(panel.state.headline)
                    .font(SD.Typography.headline.monospacedDigit())
                    .foregroundStyle(SD.Colors.textPrimary)
                    .fixedSize(horizontal: false, vertical: true)
                if let detail = panel.state.detail {
                    Text(detail)
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(panel.state.headline)
        .accessibilityValue(panel.state.detail ?? "")
        .accessibilityIdentifier("driver.panel.state")

        let others = panel.behaviours.filter { $0.text != panel.state.headline }
        if !others.isEmpty || panel.eyesClosed || panel.head != nil {
            FlowChips(items: others, eyesClosed: panel.eyesClosed, head: panel.head)
        }
        if let unavailable = panel.unavailable {
            Label(unavailable, systemImage: "exclamationmark.triangle")
                .font(SD.Typography.caption)
                .foregroundStyle(SD.Colors.warning)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// The newest event: the live one (at most 30 s old, from the device) when
    /// there is one, else the newest driver event the history holds.
    @ViewBuilder private func lastEventRow(_ panel: DriverPanelModel) -> some View {
        if let why = panel.why {
            eventRow(title: why.title, words: why.because, when: why.ageText,
                     history: why.eventKey.flatMap { historyNotes[$0] })
        } else if let event = lastEvent {
            eventRow(title: event.title, words: event.signals.flatMap { $0.isEmpty ? nil : $0 },
                     when: event.date.map { Self.relative.localizedString(for: $0, relativeTo: .now) },
                     history: event.eventKey.flatMap { historyNotes[$0] })
        }
    }

    private func eventRow(title: String, words: String?, when: String?, history: String?) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(alignment: .firstTextBaseline, spacing: SD.Space.xs) {
                Text("Last event")
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textTertiary)
                Text(title)
                    .font(SD.Typography.caption.weight(.semibold))
                    .foregroundStyle(SD.Colors.textPrimary)
                    .fixedSize(horizontal: false, vertical: true)
                if let when {
                    Text("· \(when)")
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textTertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            if let words {
                // The device's own explanation, verbatim.
                Text("“\(words)”")
                    .font(SD.Typography.callout)
                    .foregroundStyle(SD.Colors.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let history {
                // The same event's row in the history (matched by its key).
                Label(history, systemImage: "list.bullet.rectangle")
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textTertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(.top, SD.Space.xxs)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Last event: \(title)\(when.map { ", \($0)" } ?? "")")
        .accessibilityValue([words.map { "The device says: \($0)" }, history].compactMap { $0 }.joined(separator: ". "))
        .accessibilityIdentifier("driver.panel.lastEvent")
    }
}

/// Secondary behaviours and the face chips, wrapping onto lines as needed.
private struct FlowChips: View {
    let items: [BehaviourStripItem]
    let eyesClosed: Bool
    let head: HeadIndicator?

    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: SD.Space.xs) { content }
            VStack(alignment: .leading, spacing: SD.Space.xs) { content }
        }
    }

    @ViewBuilder private var content: some View {
        ForEach(items) { BehaviourChip(item: $0) }
        if eyesClosed {
            Label("Eyes closed", systemImage: "eye.slash")
                .font(SD.Typography.caption.weight(.semibold))
                .foregroundStyle(SD.Colors.warning)
                .footagePlate(tint: SD.Colors.warning)
        }
        if let head { HeadIndicatorChip(head: head) }
    }
}
