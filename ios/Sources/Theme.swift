import SwiftUI

// MARK: - ORYN EDGE design system
//
// One source of truth for colour, type, spacing, radius and the components
// built from them. Screens compose these tokens rather than hard-coding values,
// so the app reads as one ORYN product instead of a dozen separately-styled
// screens. Change a token here and every screen follows.
//
// The palette is ORYN's own (ORYN-WEBSITE `--color-ink-*`, `--color-bone-*`,
// `--color-steel-*`, `--color-signal`, `--color-data-*`), so the app and the web
// platform are visibly the same company:
//
//   * ink   — near-black graphite surfaces, stacked by elevation
//   * bone  — warm off-white for type and the primary action
//   * steel — the greys between them, for secondary text
//   * signal — ORYN orange, used sparingly as a brand moment, never as chrome
//   * data  — status only (ok / attention / alert), so a colour always *means*
//             something
//
// ORYN EDGE is dark by design. The SD namespace and its token names are kept
// from the SensorDyme app so every screen adopted the new identity without its
// logic being touched.

enum SD {

    // MARK: Colour

    enum Colors {
        // Surfaces, darkest to lightest. Each step is one level of elevation;
        // there are no shadows in a dark UI — a lighter surface does that job.
        /// The page.
        static let background = Color(hex: 0x0A0B0C)   // ink-900
        /// Cards and grouped rows.
        static let surface = Color(hex: 0x0F1112)      // ink-850
        /// Things inside cards: chips, tiles, pressed rows.
        static let surfaceRaised = Color(hex: 0x141618) // ink-800
        /// Text inputs and the Ask Jupiter bar.
        static let field = Color(hex: 0x191C1E)        // ink-750

        /// Behind video. Pure black: footage reads best on a neutral field and a
        /// tinted box around a night-vision frame looks like a rendering fault.
        static let canvas = Color(hex: 0x000000)

        // Text
        static let textPrimary = Color(hex: 0xF2F1ED)   // bone-50
        static let textSecondary = Color(hex: 0xB6BABE) // steel-200
        /// Captions and metadata. steel-300 rather than steel-400: 400 is under
        /// 4.5:1 on ink and these are small.
        static let textTertiary = Color(hex: 0x8F959B)  // steel-300
        /// Placeholders and disabled glyphs only.
        static let textDisabled = Color(hex: 0x6C7278)  // steel-400
        static let textOnAccent = Color(hex: 0x0A0B0C)  // ink on bone

        // Lines. Hairlines carry the structure, so they stay quiet.
        static let border = Color(hex: 0x202326)        // ink-700
        static let borderStrong = Color(hex: 0x2B2F32)  // ink-600
        /// A focused field or the selected option.
        static let borderFocus = Color(hex: 0xF2F1ED).opacity(0.42)

        /// The interactive colour: bone. If it is the brightest thing on the
        /// screen and it isn't a status, you can tap it.
        static let accent = Color(hex: 0xF2F1ED)
        static let accentMuted = Color(hex: 0x202326)

        /// ORYN signal orange. A brand moment — the live indicator, Jupiter's
        /// focus — not a general-purpose accent.
        static let signal = Color(hex: 0xFF6A2B)

        // Status. Deliberately distinct from bone so "interactive" and
        // "something is happening" can never be confused.
        static let success = Color(hex: 0x8DBF63)       // data-veg
        static let successMuted = Color(hex: 0x131A10)
        static let warning = Color(hex: 0xE9B949)       // data-amber
        static let warningMuted = Color(hex: 0x1D180C)
        static let danger = Color(hex: 0xE5544B)        // data-alert
        static let dangerMuted = Color(hex: 0x231110)
        static let offline = Color(hex: 0x6C7278)       // steel-400
    }

    // MARK: Typography

    /// SF Pro for reading, SF Mono for labels — the pairing ORYN uses on the web
    /// (Geist / Geist Mono). A short scale: fewer, more distinct steps make the
    /// hierarchy obvious. Relative sizes, so Dynamic Type still works.
    enum Typography {
        static let display = Font.system(size: 30, weight: .semibold)
        static let title = Font.system(.title2).weight(.semibold)
        /// The device's status readout ("All clear", "Stopped"): readable at arm's
        /// length, one step under the device name.
        static let readout = Font.system(.title3).weight(.semibold)
        static let headline = Font.system(.headline).weight(.semibold)
        static let body = Font.system(.body)
        static let callout = Font.system(.callout)
        static let label = Font.system(.subheadline).weight(.medium)
        static let caption = Font.system(.caption)
        /// Uppercase section labels and tags. Pair with `.tracking(SD.Typography.eyebrowKerning)`
        /// — or use `SDEyebrow`, which does both.
        /// A text style, not a point size: caption2 is 11 pt at the default size, the
        /// design value, and follows Dynamic Type from there. A fixed size never scaled,
        /// and a ScaledMetric size was flagged by the accessibility audit as partial.
        /// Caption (12 pt), not caption2 (11 pt). caption2 is pinned at 11 pt for
        /// every size from Extra Small to Large, so the audit reported every
        /// eyebrow as "Dynamic Type partially unsupported"; two M8 experiments
        /// ruled out the font construction (`system(_:design:)`, `.monospaced()`,
        /// `fontDesign`) and left the text style. Caption follows every size.
        static let eyebrow = Font.caption.weight(.medium).monospaced()
        static let eyebrowKerning: CGFloat = 1.4
        /// Small machine values: addresses, ids, timings.
        static let mono = Font.system(.caption, design: .monospaced)
        /// A short machine value under another label (a key's count).
        static let caption2Mono = Font.caption.monospacedDigit().weight(.medium)
        /// Numbers that update in place — monospaced digits stop the layout
        /// twitching on every refresh.
        static let metric = Font.system(.title3).monospacedDigit().weight(.medium)
    }

    // MARK: Spacing — a 4pt grid

    enum Space {
        static let xxs: CGFloat = 4
        static let xs: CGFloat = 8
        static let sm: CGFloat = 12
        static let md: CGFloat = 16
        static let lg: CGFloat = 24
        static let xl: CGFloat = 32
        static let xxl: CGFloat = 48
    }

    // MARK: Radius
    //
    // Tighter than the old app. ORYN's shapes are precise rather than soft.

    enum Radius {
        static let sm: CGFloat = 6
        static let md: CGFloat = 10
        static let lg: CGFloat = 14
        static let xl: CGFloat = 18
        /// The camera viewport. Nested surfaces step down 22 → 14 → 10 so their
        /// corners stay concentric with the frame around them.
        static let camera: CGFloat = 22
    }

    // MARK: Elevation

    /// Kept for API compatibility. A dark UI shows depth with surface steps,
    /// not shadows, so this draws nothing.
    struct Elevation: ViewModifier {
        var strength: CGFloat = 1
        func body(content: Content) -> some View { content }
    }

    // MARK: Appearance

    /// Retained so a saved preference still decodes. ORYN EDGE is always dark;
    /// see SensorDymeApp, which no longer reads this to pick a colour scheme.
    enum Appearance: String, CaseIterable, Identifiable {
        case system, light, dark

        var id: String { rawValue }

        var title: String {
            switch self {
            case .system: "System"
            case .light: "Light"
            case .dark: "Dark"
            }
        }

        var icon: String {
            switch self {
            case .system: "iphone"
            case .light: "sun.max"
            case .dark: "moon"
            }
        }

        var colorScheme: ColorScheme? {
            switch self {
            case .system: nil
            case .light: .light
            case .dark: .dark
            }
        }
    }

    // MARK: Motion
    //
    // Native springs, applied at the container with `.animation(_:value:)` keyed
    // to a real state value — never per element, never on a timer, never per
    // telemetry packet. See docs/IOS_DESIGN_SYSTEM.md §4.

    enum Motion {
        /// Acknowledging a touch: press states, small chrome changes.
        static let feedback = Animation.spring(response: 0.18, dampingFraction: 0.9)
        /// Most in-place changes. One spring, so the app feels like one product.
        static let standard = Animation.spring(response: 0.34, dampingFraction: 0.88)
        /// Large spatial changes: expand/collapse, the Jupiter dock opening.
        static let spatial = Animation.spring(response: 0.5, dampingFraction: 0.86)
        /// Content arriving in a sheet.
        static let sheet = Animation.spring(response: 0.42, dampingFraction: 0.9)
        /// A status readout changing. No bounce: a truth changing is not a toy.
        static let status = Animation.easeInOut(duration: 0.35)
        /// Content revealed in sequence (capabilities in a mode's detail).
        static let reveal = Animation.easeOut(duration: 0.28)
        /// The step between staggered items, and how many items stagger at most.
        static let revealStep: Double = 0.04
        static let revealLimit = 8
        static let gentle = Animation.easeInOut(duration: 0.25)
        /// For acknowledging a touch. Anything slower reads as lag.
        static let quick = Animation.easeOut(duration: 0.12)

        /// The animation to use under the reader's Reduce Motion setting:
        /// spatial and staggered motion becomes an instant change; a status
        /// cross-fade stays (it is an opacity change, not movement).
        static func resolved(_ animation: Animation, reduceMotion: Bool) -> Animation? {
            guard reduceMotion else { return animation }
            return animation == status || animation == gentle || animation == quick ? animation : nil
        }

        /// The delay before item `index` of a staggered reveal.
        static func revealDelay(_ index: Int, reduceMotion: Bool) -> Double {
            reduceMotion ? 0 : Double(min(index, revealLimit)) * revealStep
        }
    }
}

extension Color {
    init(hex: UInt32) {
        self.init(
            red: Double((hex >> 16) & 0xFF) / 255,
            green: Double((hex >> 8) & 0xFF) / 255,
            blue: Double(hex & 0xFF) / 255
        )
    }
}

extension View {
    func sdElevation(_ strength: CGFloat = 1) -> some View {
        modifier(SD.Elevation(strength: strength))
    }
}

// MARK: - Labels

/// The ORYN section label: small, uppercase, monospaced, tracked out.
struct SDEyebrow: View {
    let text: String
    var color: Color = SD.Colors.textTertiary

    init(_ text: String, color: Color = SD.Colors.textTertiary) {
        self.text = text
        self.color = color
    }

    var body: some View {
        // No tracking: with it (and with kerning before it) the audit reported
        // every eyebrow as clipped text; without it, none (M8 experiment). The
        // monospaced capitals carry the spacing on their own.
        // No line limit: a long label (a device name in a Jupiter header) wraps
        // as far as the reader's text size needs; two lines clipped it (audit).
        Text(text.uppercased())
            .sdEyebrowFont()
            .foregroundStyle(color)
            .fixedSize(horizontal: false, vertical: true)
    }
}

// MARK: - Dynamic Type

/// One of ORYN's fixed design sizes that still follows the reader's text size:
/// the design value is the default and scales with the nearest text style.
/// `Font.system(size:)` alone never scales, so an 11 pt label stayed 11 pt at
/// the largest accessibility sizes.
struct SDScaledFont: ViewModifier {
    @ScaledMetric private var size: CGFloat
    private let weight: Font.Weight
    private let design: Font.Design

    init(size: CGFloat, relativeTo style: Font.TextStyle, weight: Font.Weight, design: Font.Design) {
        _size = ScaledMetric(wrappedValue: size, relativeTo: style)
        self.weight = weight
        self.design = design
    }

    func body(content: Content) -> some View {
        content.font(.system(size: size, weight: weight, design: design))
    }
}

extension View {
    func sdScaledFont(_ size: CGFloat, relativeTo style: Font.TextStyle,
                      weight: Font.Weight = .regular, design: Font.Design = .default) -> some View {
        modifier(SDScaledFont(size: size, relativeTo: style, weight: weight, design: design))
    }

    /// `SD.Typography.eyebrow` (already a Dynamic Type style).
    func sdEyebrowFont() -> some View {
        font(SD.Typography.eyebrow)
    }
}

// MARK: - Surfaces

/// A grouped surface, for the few places a container means something (a form
/// group in a sheet, a set of controls). Open layouts (sections separated by
/// space and type) are the default; this is not a wrapper for every block.
/// One surface step up from the page and no outline: the step is the edge.
struct SDCard<Content: View>: View {
    var padding: CGFloat = SD.Space.md
    var raised = false
    @ViewBuilder var content: Content

    var body: some View {
        content
            .padding(padding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: SD.Radius.lg, style: .continuous)
                    .fill(raised ? SD.Colors.surfaceRaised : SD.Colors.surface)
            )
    }
}

/// A section's label on open space, with an optional action at the trailing
/// edge (44 pt to touch). Sections are grouped by this and by spacing, not by
/// boxes.
struct SDSectionHeader: View {
    let title: String
    var action: (title: String, perform: () -> Void)?

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            SDEyebrow(title)
                .accessibilityAddTraits(.isHeader)
            Spacer()
            if let action {
                Button(action.title, action: action.perform)
                    .font(SD.Typography.caption.weight(.semibold))
                    .foregroundStyle(SD.Colors.textSecondary)
                    .frame(minHeight: 44)
                    .contentShape(Rectangle())
            }
        }
    }
}

// MARK: - Buttons

struct SDButtonStyle: ButtonStyle {
    enum Kind { case primary, secondary, tertiary, destructive }

    var kind: Kind = .primary
    var fullWidth = true

    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(SD.Typography.label.weight(.semibold))
            .foregroundStyle(foreground)
            .frame(maxWidth: fullWidth ? .infinity : nil, minHeight: kind == .tertiary ? 40 : 50)
            .padding(.horizontal, fullWidth ? 0 : SD.Space.md)
            .background(background)
            .clipShape(RoundedRectangle(cornerRadius: SD.Radius.md, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: SD.Radius.md, style: .continuous)
                    .strokeBorder(borderColor, lineWidth: 1)
            )
            .opacity(isEnabled ? (configuration.isPressed ? 0.78 : 1) : 0.4)
            .scaleEffect(configuration.isPressed ? 0.985 : 1)
            .animation(SD.Motion.quick, value: configuration.isPressed)
            .contentShape(Rectangle())
    }

    private var foreground: Color {
        switch kind {
        case .primary: SD.Colors.textOnAccent
        case .secondary: SD.Colors.textPrimary
        case .tertiary: SD.Colors.textSecondary
        case .destructive: SD.Colors.danger
        }
    }

    private var borderColor: Color {
        switch kind {
        case .secondary: SD.Colors.borderStrong
        case .destructive: SD.Colors.danger.opacity(0.35)
        case .primary, .tertiary: .clear
        }
    }

    @ViewBuilder private var background: some View {
        switch kind {
        case .primary: SD.Colors.accent
        case .destructive: SD.Colors.dangerMuted
        case .secondary: SD.Colors.surface
        case .tertiary: Color.clear
        }
    }
}

// MARK: - Status

/// A single vocabulary for "is this thing working", used everywhere a device,
/// camera, radio or service reports itself — so one glance means the same thing
/// on every screen.
enum SDStatus {
    case online, connecting, offline, warning, alert
    /// The app can't tell (the latest refresh failed): neither online nor offline.
    case unknown

    var label: String {
        switch self {
        case .online: "Online"
        case .connecting: "Connecting"
        case .offline: "Offline"
        case .unknown: "Unknown"
        case .warning: "Needs attention"
        case .alert: "Alert"
        }
    }

    var tint: Color {
        switch self {
        case .online: SD.Colors.success
        case .connecting: SD.Colors.warning
        case .offline: SD.Colors.offline
        case .unknown: SD.Colors.textTertiary
        case .warning: SD.Colors.warning
        case .alert: SD.Colors.danger
        }
    }

    var wash: Color {
        switch self {
        case .online: SD.Colors.successMuted
        case .connecting, .warning: SD.Colors.warningMuted
        case .offline, .unknown: SD.Colors.surfaceRaised
        case .alert: SD.Colors.dangerMuted
        }
    }

    var icon: String {
        switch self {
        case .online: "checkmark.circle.fill"
        case .connecting: "arrow.triangle.2.circlepath"
        case .offline: "moon.zzz.fill"
        case .unknown: "questionmark.circle"
        case .warning: "exclamationmark.triangle.fill"
        case .alert: "bell.badge.fill"
        }
    }
}

extension SDStatus {
    /// The chip for a device's presence (ConnectionTruth). Stale is amber:
    /// it is not offline, and it must not look like the green of a live device.
    init(_ presence: ConnectionTruth.Presence) {
        switch presence {
        case .online: self = .online
        case .stale: self = .warning
        case .offline: self = .offline
        case .unknown: self = .unknown
        }
    }
}

/// Compact status chip: a dot and a short word in a quiet capsule. It carries a
/// word as well as a colour, because colour alone fails colour-blind users and
/// bright sun. Sentence case: mono capitals are kept for section labels.
///
/// No pulse. A breathing dot says "alive" about a device whose state the dot
/// does not know (`pulses` is accepted for source compatibility and ignored).
struct SDStatusPill: View {
    let status: SDStatus
    var text: String?
    var pulses = false

    var body: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(status.tint)
                .frame(width: 6, height: 6)
            // Wraps rather than truncating: at the largest text sizes one line
            // cut "Online" short (seen at AX XXXL).
            Text(text ?? status.label)
                .font(SD.Typography.caption.weight(.semibold))
                .foregroundStyle(status == .offline || status == .unknown ? SD.Colors.textSecondary : status.tint)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.leading, 8)
        .padding(.trailing, 10)
        .padding(.vertical, 4)
        .background(Capsule().fill(status.wash))
        .accessibilityElement(children: .combine)
        .accessibilityLabel(text ?? status.label)
    }
}

// MARK: - Screen states
//
// Loading / empty / error handled once, so no screen invents its own and none
// leaves the user staring at a blank rectangle.

struct SDLoadingState: View {
    var message: String = "Loading…"

    var body: some View {
        VStack(spacing: SD.Space.sm) {
            ProgressView().tint(SD.Colors.textSecondary)
            Text(message)
                .font(SD.Typography.callout)
                .foregroundStyle(SD.Colors.textTertiary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, SD.Space.xl)
        // One element that says what is loading, rather than an unlabelled
        // spinner followed by the text.
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(message)
        .accessibilityAddTraits(.updatesFrequently)
        .accessibilityIdentifier("state.loading")
    }
}

struct SDEmptyState: View {
    let icon: String
    let title: String
    let message: String
    var actionTitle: String?
    var action: (() -> Void)?

    var body: some View {
        VStack(spacing: SD.Space.sm) {
            Image(systemName: icon)
                .font(.system(size: 22, weight: .light))
                .foregroundStyle(SD.Colors.textTertiary)
                .frame(height: 32)
                .padding(.bottom, SD.Space.xxs)
                .accessibilityHidden(true)  // decorative; the title says it
            VStack(spacing: SD.Space.sm) {
                Text(title)
                    .font(SD.Typography.headline)
                    .foregroundStyle(SD.Colors.textPrimary)
                    .accessibilityAddTraits(.isHeader)
                Text(message)
                    .font(SD.Typography.callout)
                    .foregroundStyle(SD.Colors.textTertiary)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .accessibilityElement(children: .combine)
            .accessibilityIdentifier("state.empty")
            if let actionTitle, let action {
                Button(actionTitle, action: action)
                    .buttonStyle(SDButtonStyle(kind: .secondary, fullWidth: false))
                    .padding(.top, SD.Space.xs)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, SD.Space.xl)
        .padding(.horizontal, SD.Space.md)
    }
}

/// Failure state. Says what happened and what to do about it — never
/// "Something went wrong", which tells the user nothing they can act on.
struct SDErrorState: View {
    let title: String
    let message: String
    var retry: (() async -> Void)?

    @State private var retrying = false

    var body: some View {
        VStack(spacing: SD.Space.sm) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 22, weight: .light))
                .foregroundStyle(SD.Colors.warning)
                .frame(height: 32)
                .accessibilityHidden(true)
            Text(title)
                .font(SD.Typography.headline)
                .foregroundStyle(SD.Colors.textPrimary)
            Text(message)
                .font(SD.Typography.callout)
                .foregroundStyle(SD.Colors.textTertiary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
            if let retry {
                Button {
                    retrying = true
                    Task { await retry(); retrying = false }
                } label: {
                    if retrying { ProgressView().tint(SD.Colors.textPrimary) } else { Text("Try again") }
                }
                .buttonStyle(SDButtonStyle(kind: .secondary, fullWidth: false))
                .disabled(retrying)
                .padding(.top, SD.Space.xxs)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, SD.Space.xl)
        .padding(.horizontal, SD.Space.md)
    }
}

/// Inline banner for a problem that must not hide the content behind it.
/// The device screens poll on a timer and keep their last-known data when a
/// refresh fails, so without this a dead backend looks exactly like a healthy
/// one — the screen just quietly stops changing.
struct ConnectionBanner: View {
    let message: String
    /// nil: from the message (ReplyError.bannerTitle), so an unreadable reply
    /// is never headed "Can't reach the server".
    var title: String? = nil
    var retry: (() async -> Void)?

    @State private var retrying = false

    var body: some View {
        HStack(alignment: .top, spacing: SD.Space.sm) {
            Rectangle()
                .fill(SD.Colors.warning)
                .frame(width: 2)
                .clipShape(Capsule())
            VStack(alignment: .leading, spacing: SD.Space.xxs + 2) {
                VStack(alignment: .leading, spacing: SD.Space.xxs + 2) {
                    Text(title ?? ReplyError.bannerTitle(for: message))
                        .font(SD.Typography.label)
                        .foregroundStyle(SD.Colors.textPrimary)
                    Text(message)
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                // Read as one sentence; the retry button below stays a separate,
                // activatable element. Combining the whole banner folded the button
                // into a static description that VoiceOver could not press.
                .accessibilityElement(children: .combine)
                .accessibilityIdentifier("banner.message")
                if let retry {
                    Button {
                        retrying = true
                        Task { await retry(); retrying = false }
                    } label: {
                        if retrying {
                            ProgressView().controlSize(.small).tint(SD.Colors.textPrimary)
                        } else {
                            Text("Try again")
                                .font(SD.Typography.caption.weight(.semibold))
                                .foregroundStyle(SD.Colors.accent)
                        }
                    }
                    // The text stays small; the target is the 44 pt minimum.
                    .frame(minWidth: 44, minHeight: 44, alignment: .leading)
                    .contentShape(Rectangle())
                    .disabled(retrying)
                    .accessibilityLabel(retrying ? "Trying again" : "Try again")
                    .accessibilityIdentifier("banner.retry")
                }
            }
            Spacer(minLength: 0)
        }
        .padding(.vertical, SD.Space.sm)
        .padding(.horizontal, SD.Space.sm + 2)
        .background(
            RoundedRectangle(cornerRadius: SD.Radius.lg, style: .continuous)
                .fill(SD.Colors.warningMuted)
        )
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("banner.connection")
    }
}

// MARK: - Floating surfaces

/// The one translucent surface in the app: things that float over content (the
/// Ask Jupiter dock, the camera's controls). Native Liquid Glass on iOS 26, a
/// thin material over ink before it, and an opaque ink field with a hairline
/// under Reduce Transparency, so it stays legible with every blur switched off.
struct SDFloatingSurface<S: Shape>: ViewModifier {
    let shape: S
    var shadow = true
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency

    func body(content: Content) -> some View {
        if reduceTransparency {
            content
                .background(shape.fill(SD.Colors.field))
                .overlay(shape.stroke(SD.Colors.borderStrong, lineWidth: 1))
        } else if #available(iOS 26.0, *) {
            content
                .background(shape.fill(SD.Colors.background.opacity(0.55)))
                .glassEffect(.regular, in: shape)
                .shadow(color: .black.opacity(shadow ? 0.35 : 0), radius: 24, y: 8)
        } else {
            content
                .background(shape.fill(SD.Colors.field.opacity(0.72)))
                .background(.ultraThinMaterial, in: shape)
                // One highlight hairline, brighter at the top, reads as a lit edge.
                .overlay(shape.stroke(
                    LinearGradient(colors: [Color.white.opacity(0.14), Color.white.opacity(0.03)],
                                   startPoint: .top, endPoint: .bottom), lineWidth: 1))
                .shadow(color: .black.opacity(shadow ? 0.35 : 0), radius: 24, y: 8)
        }
    }
}

extension View {
    func sdFloatingSurface<S: Shape>(_ shape: S, shadow: Bool = true) -> some View {
        modifier(SDFloatingSurface(shape: shape, shadow: shadow))
    }
}

/// A contextual action key on the device screen (Activity, Areas, Report): a
/// symbol over a short word on a raised surface, like a control on a dash.
/// 64 pt tall at the default size, well above the 44 pt minimum.
struct InstrumentKey: View {
    let icon: String
    let title: String
    /// A short state word under the title ("Alert"); nil for none.
    var badge: String? = nil
    var badgeColor: Color = SD.Colors.textTertiary

    var body: some View {
        VStack(spacing: 6) {
            Image(systemName: icon)
                .font(.system(size: 17, weight: .regular))
                .foregroundStyle(SD.Colors.textPrimary)
                .frame(height: 22)
                .accessibilityHidden(true)
            Text(title)
                .font(SD.Typography.caption.weight(.medium))
                .foregroundStyle(SD.Colors.textSecondary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
            if let badge {
                Text(badge)
                    .font(SD.Typography.caption2Mono)
                    .foregroundStyle(badgeColor)
            }
        }
        .padding(.vertical, SD.Space.sm)
        .padding(.horizontal, SD.Space.xxs)
        .frame(maxWidth: .infinity, minHeight: 64)
        .background(
            RoundedRectangle(cornerRadius: SD.Radius.lg, style: .continuous)
                .fill(SD.Colors.surfaceRaised)
        )
        .contentShape(RoundedRectangle(cornerRadius: SD.Radius.lg, style: .continuous))
    }
}

/// Press feedback for surfaces that are themselves the control (keys, the
/// camera, mode rows): a slight scale and dim at once on press, settling on
/// release.
struct SDPressStyle: ButtonStyle {
    var scale: CGFloat = 0.97

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? scale : 1)
            .opacity(configuration.isPressed ? 0.82 : 1)
            .animation(configuration.isPressed ? nil : SD.Motion.feedback, value: configuration.isPressed)
    }
}

// MARK: - Metrics (advanced / diagnostics surfaces)

/// A labelled number. Consumer screens should not show these; they belong in
/// Device Health, where someone has deliberately gone looking for detail.
struct SDMetric: View {
    let label: String
    let value: String
    var status: SDStatus?

    var body: some View {
        VStack(alignment: .leading, spacing: SD.Space.xxs) {
            SDEyebrow(label)
            HStack(spacing: SD.Space.xxs + 2) {
                Text(value)
                    .font(SD.Typography.metric)
                    .foregroundStyle(SD.Colors.textPrimary)
                if let status {
                    Circle().fill(status.tint).frame(width: 6, height: 6)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - Brand mark (compatibility)
//
// The SensorDyme hexagon is retired. These names stay so existing call sites
// compile, and now draw ORYN's real mark from the asset catalog (OrynBrand.swift).

struct LogoMark: View {
    var size: CGFloat = 64
    /// Retained for source compatibility; the ORYN mark is always bone.
    var accented = false

    var body: some View {
        OrynMark(height: size)
    }
}

struct LogoLockup: View {
    var markSize: CGFloat = 72

    var body: some View {
        OrynEdgeLockup(markHeight: markSize)
    }
}

// MARK: - Compatibility shims
//
// Kept so existing screens compile. New code should use SDCard / SDButtonStyle
// / SD.Colors directly.

enum Brand {
    static let ink = SD.Colors.canvas
    static let electric = SD.Colors.accent
    static let surface = SD.Colors.surface
    static let gradient = SD.Colors.accent
    static let alertGradient = SD.Colors.danger
    static let okGradient = SD.Colors.success
}

struct BrandCard: ViewModifier {
    func body(content: Content) -> some View {
        content
            .padding(SD.Space.md)
            .background(
                RoundedRectangle(cornerRadius: SD.Radius.lg, style: .continuous)
                    .fill(SD.Colors.surface)
            )
            .overlay(
                RoundedRectangle(cornerRadius: SD.Radius.lg, style: .continuous)
                    .strokeBorder(SD.Colors.border, lineWidth: 1)
            )
    }
}

extension View {
    func brandCard() -> some View { modifier(BrandCard()) }
}

struct BrandButtonStyle: ButtonStyle {
    var prominent = true

    func makeBody(configuration: Configuration) -> some View {
        SDButtonStyle(kind: prominent ? .primary : .secondary).makeBody(configuration: configuration)
    }
}

/// The hero backdrop behind sign-in: ink with a faint graphite glow at the top,
/// where the logo sits. Static — a safety product's first screen should be calm.
struct AnimatedBrandBackground: View {
    var body: some View {
        ZStack {
            SD.Colors.background
            RadialGradient(
                colors: [Color(hex: 0x202326).opacity(0.9), SD.Colors.background.opacity(0)],
                center: UnitPoint(x: 0.5, y: 0.0),
                startRadius: 10,
                endRadius: 520
            )
        }
        .ignoresSafeArea()
    }
}
