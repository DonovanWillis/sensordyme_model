import Foundation

/// When the Ask Jupiter dock is the full prompt field and when it steps aside
/// as a compact button, from the device screen's scroll position.
///
/// Foundation-only and unit-tested (ios/Tests/JupiterDockBehaviorTests.swift);
/// `JupiterDockModel` feeds it the scroll view's geometry.
///
/// - At the top of the page (and in the pull-to-refresh bounce) the dock is
///   the full field: that is where it is discovered.
/// - Scrolling down, past the top, by a deliberate distance minimises it, so
///   reading the page is not done through a bar that covers a full row.
/// - Scrolling back up by a shorter distance, or reaching the end of the page,
///   brings it back.
/// - Never while the person is using it: focused, or with a request typed.
/// - At accessibility text sizes the dock is always compact (`DockForm`): the
///   field would be a tall bar over a page that already scrolls a lot, and the
///   full Ask Jupiter screen gives the request the whole width.
enum JupiterDockBehavior {
    /// Offsets this close to the top count as "at the top".
    static let topZone: Double = 56
    /// Downward travel, in one direction, that minimises the dock.
    static let minimiseTravel: Double = 48
    /// Upward travel that brings it back (shorter: asking is a smaller ask).
    static let restoreTravel: Double = 28
    /// This close to the end of the page counts as "at the end".
    static let endZone: Double = 12

    /// What the dock looks like.
    enum DockForm: Equatable {
        /// The prompt field, full width.
        case field
        /// A compact button that opens the field (or, at accessibility sizes,
        /// the Ask Jupiter screen).
        case button
    }

    /// The scroll state the decision is made from.
    struct Tracker: Equatable {
        private(set) var minimised = false
        private var lastOffset: Double?
        /// Distance travelled in the current direction (+ down, - up).
        private var travel: Double = 0

        init() {}

        /// One scroll sample: `offset` from the top of the content (0 at rest,
        /// negative in the pull-down bounce) and the largest offset the content
        /// allows. `engaged`: the dock is focused or holds a typed request.
        /// Returns whether the minimised state changed.
        @discardableResult
        mutating func scrolled(offset: Double, maxOffset: Double, engaged: Bool) -> Bool {
            let before = minimised
            defer { lastOffset = offset }
            guard !engaged else {
                travel = 0
                minimised = false
                return before != minimised
            }
            if offset <= JupiterDockBehavior.topZone || offset >= maxOffset - JupiterDockBehavior.endZone {
                travel = 0
                minimised = false
                return before != minimised
            }
            guard let last = lastOffset else { return false }
            let delta = offset - last
            if delta == 0 { return false }
            // A change of direction starts the count again.
            if (delta > 0) != (travel > 0) && travel != 0 { travel = 0 }
            travel += delta
            if travel >= JupiterDockBehavior.minimiseTravel {
                minimised = true
            } else if travel <= -JupiterDockBehavior.restoreTravel {
                minimised = false
            }
            return before != minimised
        }

        /// The person tapped the compact dock: it is the field again until the
        /// next deliberate scroll down.
        mutating func restore() {
            minimised = false
            travel = 0
        }
    }

    /// The dock's form for the reader's text size and the scroll state.
    static func form(accessibilitySize: Bool, minimised: Bool) -> DockForm {
        accessibilitySize || minimised ? .button : .field
    }
}
