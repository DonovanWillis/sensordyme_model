import SwiftUI

// MARK: - Small pieces of the device screen

/// The device screen names the device in its own header, under the camera, so
/// the navigation bar carries only the back button and the menu (iOS 18+).
/// The title is still set for the back button's label and VoiceOver's
/// screen change; iOS 17 shows it as before.
struct HidesNavigationTitle: ViewModifier {
    func body(content: Content) -> some View {
        if #available(iOS 18.0, *) {
            content.toolbar(removing: .title)
        } else {
            content
        }
    }
}

/// A hairline with a short highlight travelling along it: "connecting", while
/// nothing has been heard yet. Shown only in that state and removed on the
/// first answer, so it can never imply a device is working. Still under
/// Reduce Motion.
struct ConnectingHairline: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var travel = false

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                Capsule().fill(SD.Colors.border)
                if !reduceMotion {
                    Capsule()
                        .fill(LinearGradient(colors: [.clear, SD.Colors.textSecondary.opacity(0.8), .clear],
                                             startPoint: .leading, endPoint: .trailing))
                        .frame(width: geometry.size.width * 0.3)
                        .offset(x: travel ? geometry.size.width : -geometry.size.width * 0.3)
                }
            }
        }
        .frame(height: 1)
        .clipped()
        .onAppear {
            guard !reduceMotion else { return }
            withAnimation(.easeInOut(duration: 1.4).repeatForever(autoreverses: false)) { travel = true }
        }
        .accessibilityHidden(true)
    }
}

/// The library's editorial voice: a short context for each built-in mode, said
/// above its name when it leads the library. Words for where a mode is used,
/// never a capability; the capabilities come from the device.
enum ModeEditorial {
    static func context(_ id: String) -> String? {
        switch id {
        case "driver_safety": "Distracted driving"
        case "baby_monitoring": "Child monitoring"
        case "elderly_monitoring": "Care at home"
        case "security_monitoring": "Property monitoring"
        case "occupancy": "Occupancy"
        case "workplace_safety": "Workplace safety"
        default: nil
        }
    }
}
