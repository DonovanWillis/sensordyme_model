import Foundation

/// Ensures only the most recently STARTED of several overlapping async
/// requests gets to apply its result.
///
/// `DeviceDetailView` polls on a timer, refreshes on pull-to-refresh, and
/// reloads on appear — three independent triggers, each awaiting a network
/// round trip of unpredictable duration, with nothing stopping two from being
/// in flight at once. Without this, whichever one's response happens to
/// arrive LAST wins, regardless of which one was issued last — so a slow,
/// stale response from a request started before a mode switch can complete
/// after a fast, fresh one that already reflects it, and silently put the
/// screen back to the old state. That is "the mode switch looked like it
/// didn't take even though it did" and "an event briefly showed then
/// vanished" from the same cause: correctness depending on network timing.
///
/// The fix is a monotonically increasing token, not cancellation. Cancelling
/// the in-flight URLSession task would also work but adds plumbing this
/// doesn't need — a superseded request finishing and being ignored costs
/// nothing but the bytes already in flight, and is far simpler to reason
/// about than propagating cancellation through `async throws` call chains.
///
/// Not an actor: every call site in this app touches it from the main actor
/// already (it exists to protect `@State` on a SwiftUI View), and an actor
/// here would only add unnecessary `await` at every call site for no
/// additional safety.
final class LatestRequestGuard {
    private var generation = 0

    /// Call once when starting a new request. Keep the returned token and
    /// pass it to `isLatest` before applying that request's result.
    @discardableResult
    func begin() -> Int {
        generation += 1
        return generation
    }

    /// True if `token` is still the most recently started request — its
    /// result is safe to apply. False means a newer request has since begun,
    /// and this (now-stale) result must be discarded rather than applied.
    func isLatest(_ token: Int) -> Bool {
        token == generation
    }
}
