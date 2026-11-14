import SwiftUI

// MARK: - Continuity
//
// One place for the spatial transitions that join the screens: a device card
// zooms into its home, a mode row into its detail, the camera into full
// screen, the Jupiter dock into its review sheet. Native iOS 18 zoom
// transitions (`matchedTransitionSource` + `.navigationTransition(.zoom)`),
// which keep the system's own interactive dismissal (drag down, edge swipe).
//
// iOS 17, and anyone with Reduce Motion on, get the standard push or sheet:
// the zoom is never the only way to understand where something came from.

/// The transition namespaces' ids. Typed, so a source and its destination
/// cannot drift apart by a typo.
enum ZoomID: Hashable {
    case device(Int)
    case camera
    case mode(String)
    case jupiter
}

extension View {
    /// Marks this view as where a zoom transition starts (and returns to).
    /// `cornerRadius` clips the source the way it is drawn, so the zoom grows
    /// out of the shape the person tapped.
    @ViewBuilder
    func sdZoomSource(_ id: ZoomID, in namespace: Namespace.ID, cornerRadius: CGFloat = SD.Radius.lg,
                      enabled: Bool = true) -> some View {
        if #available(iOS 18.0, *), enabled {
            matchedTransitionSource(id: id, in: namespace) { source in
                source
                    .clipShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
                    .background(SD.Colors.background)
            }
        } else {
            self
        }
    }

    /// Applied to the presented or pushed view: it zooms out of the source.
    @ViewBuilder
    func sdZoomDestination(_ id: ZoomID, in namespace: Namespace.ID, enabled: Bool = true) -> some View {
        if #available(iOS 18.0, *), enabled {
            navigationTransition(.zoom(sourceID: id, in: namespace))
        } else {
            self
        }
    }
}
