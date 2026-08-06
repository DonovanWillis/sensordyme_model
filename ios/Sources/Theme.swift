import SwiftUI

// MARK: - SensorDyme brand system
// Modern + sleek: near-black surfaces, one electric gradient used for emphasis and
// motion, generous rounding. The logo is the hexagonal S mark (vector recreation —
// drop the real PNG into an asset catalog named "Logo" to replace it 1:1).

enum Brand {
    static let ink = Color(hex: 0x0B0D14)
    static let surface = Color(hex: 0x141826)
    static let surfaceLight = Color(hex: 0xF4F6FB)

    static let electric = Color(hex: 0x4C6FFF)
    static let violet = Color(hex: 0x9B4DFF)
    static let magenta = Color(hex: 0xFF4D9B)

    /// The signature vibrant transition used across the app.
    static let gradient = LinearGradient(
        colors: [electric, violet, magenta],
        startPoint: .topLeading, endPoint: .bottomTrailing
    )

    static let alertGradient = LinearGradient(
        colors: [Color(hex: 0xFF3B30), magenta],
        startPoint: .topLeading, endPoint: .bottomTrailing
    )

    static let okGradient = LinearGradient(
        colors: [Color(hex: 0x1FB970), Color(hex: 0x2BD9A9)],
        startPoint: .topLeading, endPoint: .bottomTrailing
    )
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

// MARK: - Logo

/// Vector recreation of the SensorDyme mark: a rounded hexagon with a knocked-out S.
struct LogoMark: View {
    var size: CGFloat = 64
    var gradient = false

    var body: some View {
        ZStack {
            Hexagon()
                .fill(gradient ? AnyShapeStyle(Brand.gradient) : AnyShapeStyle(Color.primary))
                .overlay(
                    Hexagon().stroke(
                        gradient ? AnyShapeStyle(Brand.gradient) : AnyShapeStyle(Color.primary),
                        style: StrokeStyle(lineWidth: size * 0.14, lineJoin: .round)
                    )
                )
            Text("S")
                .font(.system(size: size * 0.62, weight: .black, design: .rounded))
                .italic()
                .foregroundStyle(.background)
        }
        .frame(width: size, height: size * 1.12)
        .accessibilityLabel("SensorDyme")
    }
}

struct LogoLockup: View {
    var markSize: CGFloat = 72

    var body: some View {
        VStack(spacing: markSize * 0.28) {
            LogoMark(size: markSize, gradient: true)
            Text("SENSORDYME")
                .font(.system(size: markSize * 0.30, weight: .heavy, design: .rounded))
                .kerning(markSize * 0.06)
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

// MARK: - Reusable styles

struct BrandButtonStyle: ButtonStyle {
    var prominent = true

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.headline)
            .foregroundStyle(prominent ? .white : .primary)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 14)
            .background(
                Group {
                    if prominent {
                        RoundedRectangle(cornerRadius: 16).fill(Brand.gradient)
                    } else {
                        RoundedRectangle(cornerRadius: 16).fill(.thinMaterial)
                    }
                }
            )
            .opacity(configuration.isPressed ? 0.75 : 1)
            .scaleEffect(configuration.isPressed ? 0.98 : 1)
            .animation(.spring(duration: 0.25), value: configuration.isPressed)
    }
}

struct BrandCard: ViewModifier {
    func body(content: Content) -> some View {
        content
            .padding(16)
            .background(RoundedRectangle(cornerRadius: 20).fill(.background.secondary))
            .overlay(RoundedRectangle(cornerRadius: 20).strokeBorder(.quaternary, lineWidth: 1))
    }
}

extension View {
    func brandCard() -> some View { modifier(BrandCard()) }
}

/// Slow-moving animated gradient used behind hero surfaces ("vibrant transitions").
struct AnimatedBrandBackground: View {
    @State private var animate = false

    var body: some View {
        LinearGradient(
            colors: [Brand.electric.opacity(0.55), Brand.violet.opacity(0.45), Brand.magenta.opacity(0.4)],
            startPoint: animate ? .topLeading : .bottomTrailing,
            endPoint: animate ? .bottomTrailing : .topLeading
        )
        .hueRotation(.degrees(animate ? 12 : -12))
        .animation(.easeInOut(duration: 6).repeatForever(autoreverses: true), value: animate)
        .onAppear { animate = true }
        .ignoresSafeArea()
    }
}
