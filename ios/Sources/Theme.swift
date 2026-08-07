import SwiftUI

// MARK: - SensorDyme design system
//
// One source of truth for colour, type, spacing, radius, elevation, and the
// components built from them. Screens compose these tokens rather than
// hard-coding values, so the product reads as one system instead of a dozen
// separately-styled screens.
//
// The previous theme used a three-stop electric→violet→magenta gradient as its
// identity, applied to buttons, avatars, borders, logo, and backgrounds alike.
// A gradient everywhere is a gradient nowhere — and saturated magenta is a
// strange thing to see on a device that watches sleeping infants and elderly
// parents. This system inverts that: neutrals carry the interface, one accent
// marks what is interactive, and saturated colour is reserved for *status*, so
// when something turns amber or red it means something every time.

enum SD {

    // MARK: Colour

    /// Semantic colours, all adaptive. A phone in a bright car interior is a
    /// real use case, so nothing here assumes a dark UI.
    enum Colors {
        // Foundation. True white and true black, as specified.
        //
        // A white page with white cards needs *something* to separate them, and
        // the honest choices are a tint or a line. Lines win here: tinted cards
        // on white drift grey as they stack, while a hairline stays crisp and
        // keeps the page feeling like paper rather than a stack of panels.
        // Dark mode mirrors it — black page, barely-lifted cards, hairline.
        static let background = adaptive(light: 0xFFFFFF, dark: 0x000000)
        static let surface = adaptive(light: 0xFFFFFF, dark: 0x0C0C0D)
        static let surfaceRaised = adaptive(light: 0xF7F7F8, dark: 0x161618)

        /// Behind video, in both modes. Footage reads best on a neutral dark
        /// field, and a white box around a night-vision frame looks broken.
        static let canvas = Color(hex: 0x000000)

        // Text
        static let textPrimary = adaptive(light: 0x000000, dark: 0xFFFFFF)
        static let textSecondary = adaptive(light: 0x5A5A5E, dark: 0xA8A8AD)
        static let textTertiary = adaptive(light: 0x8E8E93, dark: 0x707075)
        static let textOnAccent = Color.white

        // Lines. The only thing separating a white card from a white page, so
        // they carry real weight here rather than being decorative.
        static let border = adaptive(light: 0xE5E5E7, dark: 0x2A2A2E)
        static let borderStrong = adaptive(light: 0xC9C9CE, dark: 0x3C3C42)

        /// The one interactive colour. If it is blue, you can tap it.
        static let accent = adaptive(light: 0x0A62F0, dark: 0x4C8DFF)
        static let accentMuted = adaptive(light: 0xEDF3FE, dark: 0x101B2E)

        // Status. Deliberately distinct from the accent so "interactive" and
        // "something is happening" can never be confused.
        static let success = adaptive(light: 0x0B7A45, dark: 0x30D07A)
        static let successMuted = adaptive(light: 0xEBF7F0, dark: 0x0A1F15)
        static let warning = adaptive(light: 0x9A5B00, dark: 0xF5A524)
        static let warningMuted = adaptive(light: 0xFCF3E8, dark: 0x241A0B)
        static let danger = adaptive(light: 0xC00F0C, dark: 0xFF5F57)
        static let dangerMuted = adaptive(light: 0xFDEDEC, dark: 0x260F0E)
        static let offline = adaptive(light: 0x8E8E93, dark: 0x636368)

        private static func adaptive(light: UInt32, dark: UInt32) -> Color {
            Color(uiColor: UIColor { traits in
                UIColor(Color(hex: traits.userInterfaceStyle == .dark ? dark : light))
            })
        }
    }

    // MARK: Typography

    /// A deliberately short scale. Fewer, more distinct steps make hierarchy
    /// obvious; a dozen near-identical sizes make everything look the same.
    /// All sizes are relative so Dynamic Type still works.
    enum Typography {
        static let display = Font.system(size: 32, weight: .bold, design: .default)
        static let title = Font.system(.title2, design: .default).weight(.semibold)
        static let headline = Font.system(.headline, design: .default)
        static let body = Font.system(.body, design: .default)
        static let callout = Font.system(.callout, design: .default)
        static let label = Font.system(.subheadline, design: .default).weight(.medium)
        static let caption = Font.system(.caption, design: .default)
        /// Numbers that update in place (latency, FPS, temperature) — monospaced
        /// digits stop the layout twitching on every refresh.
        static let metric = Font.system(.title3, design: .rounded).monospacedDigit().weight(.semibold)
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

    enum Radius {
        static let sm: CGFloat = 8
        static let md: CGFloat = 12
        static let lg: CGFloat = 16
        static let xl: CGFloat = 20
    }

    // MARK: Elevation

    /// Shadows soft enough to read as depth rather than decoration, and skipped
    /// entirely in dark mode where a raised surface colour does the job better.
    struct Elevation: ViewModifier {
        @Environment(\.colorScheme) private var scheme
        var strength: CGFloat = 1

        func body(content: Content) -> some View {
            content.shadow(
                color: .black.opacity(scheme == .dark ? 0 : 0.06 * strength),
                radius: 12 * strength, x: 0, y: 4 * strength
            )
        }
    }

    // MARK: Appearance

    /// The user's theme choice. `system` is the default because most people
    /// want the app to match the phone, but an explicit choice has to be
    /// honoured — someone who runs their phone dark and wants SensorDyme white
    /// in a bright car is making a real request, not a mistake.
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

        /// nil means "don't override", which is what lets `system` work.
        var colorScheme: ColorScheme? {
            switch self {
            case .system: nil
            case .light: .light
            case .dark: .dark
            }
        }
    }

    // MARK: Motion

    enum Motion {
        /// One spring for everything that moves, so the app feels like one product.
        static let standard = Animation.spring(response: 0.32, dampingFraction: 0.85)
        static let gentle = Animation.easeInOut(duration: 0.25)
        /// For acknowledging a touch. Anything slower reads as lag on the one
        /// interaction where lag is least forgivable.
        static let quick = Animation.easeOut(duration: 0.12)
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

// MARK: - Surfaces

/// The standard content container: one surface colour, one hairline, one radius.
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
            .overlay(
                RoundedRectangle(cornerRadius: SD.Radius.lg, style: .continuous)
                    .strokeBorder(SD.Colors.border, lineWidth: 1)
            )
            .sdElevation(raised ? 1 : 0.5)
    }
}

struct SDSectionHeader: View {
    let title: String
    var action: (title: String, perform: () -> Void)?

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(title)
                .font(SD.Typography.label)
                .foregroundStyle(SD.Colors.textSecondary)
            Spacer()
            if let action {
                Button(action.title, action: action.perform)
                    .font(SD.Typography.caption.weight(.semibold))
                    .foregroundStyle(SD.Colors.accent)
            }
        }
    }
}

// MARK: - Buttons

struct SDButtonStyle: ButtonStyle {
    enum Kind { case primary, secondary, tertiary, destructive }

    var kind: Kind = .primary
    var fullWidth = true

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(SD.Typography.headline)
            .foregroundStyle(foreground)
            .frame(maxWidth: fullWidth ? .infinity : nil)
            .padding(.vertical, 14)
            .padding(.horizontal, fullWidth ? 0 : SD.Space.md)
            .background(background)
            .clipShape(RoundedRectangle(cornerRadius: SD.Radius.md, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: SD.Radius.md, style: .continuous)
                    .strokeBorder(kind == .secondary ? SD.Colors.borderStrong : .clear, lineWidth: 1)
            )
            .opacity(configuration.isPressed ? 0.86 : 1)
            .scaleEffect(configuration.isPressed ? 0.985 : 1)
            .animation(SD.Motion.standard, value: configuration.isPressed)
    }

    private var foreground: Color {
        switch kind {
        case .primary, .destructive: SD.Colors.textOnAccent
        case .secondary: SD.Colors.textPrimary
        case .tertiary: SD.Colors.accent
        }
    }

    @ViewBuilder private var background: some View {
        switch kind {
        case .primary: SD.Colors.accent
        case .destructive: SD.Colors.danger
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

    var label: String {
        switch self {
        case .online: "Online"
        case .connecting: "Connecting"
        case .offline: "Offline"
        case .warning: "Needs attention"
        case .alert: "Alert"
        }
    }

    var tint: Color {
        switch self {
        case .online: SD.Colors.success
        case .connecting: SD.Colors.warning
        case .offline: SD.Colors.offline
        case .warning: SD.Colors.warning
        case .alert: SD.Colors.danger
        }
    }

    var wash: Color {
        switch self {
        case .online: SD.Colors.successMuted
        case .connecting, .warning: SD.Colors.warningMuted
        case .offline: SD.Colors.surfaceRaised
        case .alert: SD.Colors.dangerMuted
        }
    }

    var icon: String {
        switch self {
        case .online: "checkmark.circle.fill"
        case .connecting: "arrow.triangle.2.circlepath"
        case .offline: "moon.zzz.fill"
        case .warning: "exclamationmark.triangle.fill"
        case .alert: "bell.badge.fill"
        }
    }
}

/// Compact status chip. Carries a text label as well as colour — colour alone
/// fails colour-blind users and is unreadable in bright sun.
struct SDStatusPill: View {
    let status: SDStatus
    var text: String?
    var pulses = false

    @State private var pulse = false

    var body: some View {
        HStack(spacing: SD.Space.xxs + 2) {
            Circle()
                .fill(status.tint)
                .frame(width: 7, height: 7)
                .opacity(pulses && pulse ? 0.35 : 1)
                .animation(
                    pulses ? SD.Motion.gentle.repeatForever(autoreverses: true) : nil,
                    value: pulse
                )
            Text(text ?? status.label)
                .font(SD.Typography.caption.weight(.semibold))
                .foregroundStyle(status == .offline ? SD.Colors.textSecondary : status.tint)
        }
        .padding(.horizontal, SD.Space.xs)
        .padding(.vertical, SD.Space.xxs + 1)
        .background(Capsule().fill(status.wash))
        .onAppear { if pulses { pulse = true } }
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
            ProgressView()
            Text(message)
                .font(SD.Typography.callout)
                .foregroundStyle(SD.Colors.textSecondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, SD.Space.xl)
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
                .font(.system(size: 34, weight: .light))
                .foregroundStyle(SD.Colors.textTertiary)
            Text(title)
                .font(SD.Typography.headline)
                .foregroundStyle(SD.Colors.textPrimary)
            Text(message)
                .font(SD.Typography.callout)
                .foregroundStyle(SD.Colors.textSecondary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
            if let actionTitle, let action {
                Button(actionTitle, action: action)
                    .buttonStyle(SDButtonStyle(kind: .tertiary, fullWidth: false))
                    .padding(.top, SD.Space.xxs)
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
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 30))
                .foregroundStyle(SD.Colors.warning)
            Text(title)
                .font(SD.Typography.headline)
                .foregroundStyle(SD.Colors.textPrimary)
            Text(message)
                .font(SD.Typography.callout)
                .foregroundStyle(SD.Colors.textSecondary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
            if let retry {
                Button {
                    retrying = true
                    Task { await retry(); retrying = false }
                } label: {
                    if retrying { ProgressView() } else { Text("Try again") }
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
    var title = "Can't reach the backend"
    var retry: (() async -> Void)?

    @State private var retrying = false

    var body: some View {
        HStack(alignment: .top, spacing: SD.Space.sm) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(SD.Typography.headline)
                .foregroundStyle(SD.Colors.warning)
            VStack(alignment: .leading, spacing: SD.Space.xxs + 2) {
                Text(title)
                    .font(SD.Typography.label)
                    .foregroundStyle(SD.Colors.textPrimary)
                Text(message)
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                if let retry {
                    Button {
                        retrying = true
                        Task { await retry(); retrying = false }
                    } label: {
                        if retrying {
                            ProgressView().controlSize(.small)
                        } else {
                            Text("Try again").font(SD.Typography.caption.weight(.semibold))
                        }
                    }
                    .disabled(retrying)
                    .padding(.top, 2)
                }
            }
            Spacer(minLength: 0)
        }
        .padding(SD.Space.sm + 2)
        .background(
            RoundedRectangle(cornerRadius: SD.Radius.md, style: .continuous)
                .fill(SD.Colors.warningMuted)
        )
        .overlay(
            RoundedRectangle(cornerRadius: SD.Radius.md, style: .continuous)
                .strokeBorder(SD.Colors.warning.opacity(0.35), lineWidth: 1)
        )
        .accessibilityElement(children: .combine)
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
            Text(label.uppercased())
                .font(.system(size: 10, weight: .semibold))
                .kerning(0.6)
                .foregroundStyle(SD.Colors.textTertiary)
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

// MARK: - Brand mark

/// The SensorDyme mark: a rounded hexagon with a knocked-out S. Drop a real
/// asset named "Logo" into an asset catalog to replace this 1:1.
struct LogoMark: View {
    var size: CGFloat = 64
    /// Accent-filled rather than gradient-filled. The mark is an identity, not
    /// a decoration, and it has to stay legible at 22pt in a navigation bar.
    var accented = false

    var body: some View {
        ZStack {
            Hexagon()
                .fill(accented ? SD.Colors.accent : SD.Colors.textPrimary)
                .overlay(
                    Hexagon().stroke(
                        accented ? SD.Colors.accent : SD.Colors.textPrimary,
                        style: StrokeStyle(lineWidth: size * 0.14, lineJoin: .round)
                    )
                )
            Text("S")
                .font(.system(size: size * 0.62, weight: .black, design: .rounded))
                .italic()
                .foregroundStyle(SD.Colors.background)
        }
        .frame(width: size, height: size * 1.12)
        .accessibilityLabel("SensorDyme")
    }
}

struct LogoLockup: View {
    var markSize: CGFloat = 72

    var body: some View {
        VStack(spacing: markSize * 0.26) {
            LogoMark(size: markSize, accented: true)
            Text("SENSORDYME")
                .font(.system(size: markSize * 0.26, weight: .semibold, design: .default))
                .kerning(markSize * 0.055)
                .foregroundStyle(SD.Colors.textPrimary)
        }
    }
}

struct Hexagon: Shape {
    func path(in rect: CGRect) -> Path {
        let w = rect.width, h = rect.height
        let inset = w * 0.07
        let points = [
            CGPoint(x: w / 2, y: inset),
            CGPoint(x: w - inset, y: h * 0.27),
            CGPoint(x: w - inset, y: h * 0.73),
            CGPoint(x: w / 2, y: h - inset),
            CGPoint(x: inset, y: h * 0.73),
            CGPoint(x: inset, y: h * 0.27),
        ]
        var path = Path()
        path.addLines(points)
        path.closeSubpath()
        return path
    }
}

// MARK: - Compatibility shims
//
// Kept so existing screens compile while they migrate to the tokens above.
// New code should use SDCard / SDButtonStyle / SD.Colors directly.

enum Brand {
    static let ink = SD.Colors.canvas
    static let electric = SD.Colors.accent
    static let surface = SD.Colors.surface
    /// Was a three-stop electric→violet→magenta sweep used on every surface;
    /// now a flat accent, so "brand" and "status" stop competing for attention.
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
            .sdElevation(0.5)
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

/// A calm, very slow accent wash for hero surfaces. The old version cycled hue
/// through magenta on a six-second loop behind the sign-in form, which is a lot
/// of movement on the first screen of a safety product.
struct AnimatedBrandBackground: View {
    @State private var shift = false

    var body: some View {
        LinearGradient(
            colors: [SD.Colors.accent.opacity(0.16), SD.Colors.accent.opacity(0.03)],
            startPoint: shift ? .topLeading : .top,
            endPoint: .bottomTrailing
        )
        .animation(.easeInOut(duration: 14).repeatForever(autoreverses: true), value: shift)
        .onAppear { shift = true }
        .ignoresSafeArea()
    }
}
