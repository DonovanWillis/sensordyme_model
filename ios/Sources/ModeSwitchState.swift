import Foundation

/// What a single mode row on the Monitoring mode card should show, derived
/// from explicit facts rather than inferred from strings that mean something
/// else. Before this, "is this row active" was `activeBase == preset.name`
/// and "is this row switching" was `Preset.split(switchingTo).base ==
/// preset.name` — two independent boolean checks with no shared state
/// machine between them, no distinct failure state (a failed switch just
/// cleared `switchingTo` and showed a generic banner, so the row silently
/// reverted to "inactive" with nothing explaining why), and no distinct
/// offline state (a switch requested against an offline device span forever,
/// because nothing timed it out or knew the device wasn't there to confirm).
enum ModeRowState: Equatable {
    /// Not the active mode, and nothing is happening to it.
    case inactive
    /// A switch to this mode was just requested and the device is online —
    /// expected to confirm within a poll cycle or two.
    case switching
    /// A switch to this mode was requested, but the device isn't currently
    /// online to receive it. Not a failure: the command is stored server-side
    /// (`desired_preset`) and will apply the moment the device reconnects —
    /// see cloud/app.py's convergence logic. Showing this as "switching"
    /// would imply imminent completion that may not come for hours.
    case queuedOffline
    /// The device has confirmed this mode is what it's actually running.
    case active
    /// A switch to this mode was requested and did not succeed — either the
    /// request itself failed, or it never confirmed before the timeout.
    case failed(String)
}

enum ModeSwitchState {
    /// A switch that hasn't confirmed within this long is treated as failed
    /// rather than left spinning forever. Generous relative to the uplink's
    /// default 5s poll interval (core/uplink.py's poll_interval_s) — a normal
    /// switch confirms within one or two poll cycles, so 20s gives real
    /// margin for a slow network without approaching what a user would
    /// perceive as "stuck."
    static let switchTimeout: TimeInterval = 20

    /// One request the UI is tracking: which mode was asked for, and when —
    /// the "when" is what makes the timeout above possible to enforce.
    ///
    /// `targetBase` is the preset's base name with any ":variant" suffix
    /// already stripped (Preset.split does this in Models.swift). Splitting
    /// happens on the caller's side rather than in this file on purpose: this
    /// file has no dependency on Models.swift, which is what lets
    /// ios/Tests/run.sh compile and run it standalone, concatenated only with
    /// its own test file — the same convention every other pure-logic file in
    /// this app follows.
    struct PendingSwitch: Equatable {
        let targetBase: String
        let requestedAt: Date
    }

    /// Pure derivation — no I/O, no SwiftUI, fully unit-testable. Every
    /// argument is a fact already available on DeviceDetailView; this
    /// function's only job is to not let any of them get inferred wrong.
    static func state(
        for presetName: String,
        activeBase: String,
        customMonitoringActive: Bool,
        pending: PendingSwitch?,
        deviceOnline: Bool,
        failure: PendingSwitch?,
        now: Date
    ) -> ModeRowState {
        // A failure is scoped to the specific mode it happened for. Once the
        // user taps a DIFFERENT mode, the old failure is superseded (the
        // caller clears it) rather than following them to a row it no longer
        // describes.
        if let failure, failure.targetBase == presetName {
            return .failed("Didn't switch — tap to try again")
        }

        if let pending, pending.targetBase == presetName {
            if now.timeIntervalSince(pending.requestedAt) > switchTimeout {
                // Ambiguous whether the command is still sitting queued
                // server-side or genuinely lost — either way, "still
                // switching" stops being an honest description at this point.
                return .failed("Didn't confirm in time — tap to try again")
            }
            return deviceOnline ? .switching : .queuedOffline
        }

        // No base mode reads as active while a custom configuration is in
        // charge — see DeviceDetailView's existing customMonitoringActive
        // handling, unchanged by this state machine.
        if !customMonitoringActive && activeBase == presetName {
            return .active
        }

        return .inactive
    }

    /// Whether a `reload()` that started at `reloadStartedAt` is allowed to
    /// overwrite (or clear) `current`.
    ///
    /// The race this closes: `reload()` is one of three independent triggers
    /// (on-appear, a timer, pull-to-refresh) and takes two network round
    /// trips. A user's tap can write a fresh `pendingSwitch` optimistically
    /// AFTER an in-flight reload's fetch already left the phone but BEFORE
    /// that reload's response comes back and tries to reconcile — at which
    /// point the fetch describes server state from before the tap even
    /// existed. Applying it would silently erase a switch the user just
    /// started, with no error and no visual explanation: the exact "looks
    /// like nothing happened" shape as the bugs LatestRequestGuard already
    /// fixed, but LatestRequestGuard's own token ordering can't catch this
    /// one — `pendingSwitch` is written by setMode() and read by reload(),
    /// two independent guards that don't know about each other.
    ///
    /// The fix is ordering by TIME rather than by request sequence: a
    /// `pendingSwitch` written after this reload's fetch began is provably
    /// describing something the fetch could not have known about, so this
    /// reload must leave it alone. The NEXT reload cycle starts after that
    /// write and will resolve it correctly.
    static func shouldSyncPendingSwitch(current: PendingSwitch?, reloadStartedAt: Date) -> Bool {
        guard let current else { return true }  // nothing to protect
        return current.requestedAt <= reloadStartedAt
    }
}
