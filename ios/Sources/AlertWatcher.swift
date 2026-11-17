import SwiftUI

/// Watches every device for new alerts, from anywhere in the app.
///
/// Notifications used to be posted only by DeviceDetailView's refresh loop,
/// which runs only while that one screen is open. So an alert reached you
/// exactly when you were already looking at the device and least needed
/// telling, and never when you were on the Devices list, in Settings, or
/// looking at something else — which is every moment a notification is for.
///
/// **The honest limit**, worth stating plainly: these are local notifications,
/// so the app posts them. The watcher runs only while the app is in the
/// foreground (RootView stops it on any other scene phase), so backgrounded,
/// suspended or force-quit, nothing arrives from the app. That is the
/// difference between this and APNs, and it is why this is a demo path rather
/// than the shipping one.
///
/// Each start is a new watch: the first poll of each device only records where
/// the history ends (AlertNotifier.beginWatching), so nothing that arrived
/// while the app was not watching is announced, or sounded, as if it were new.
@MainActor
final class AlertWatcher: ObservableObject {
    private let notifier = AlertNotifier.shared
    private var task: Task<Void, Never>?

    /// How often to check for alerts.
    ///
    /// Deliberately slower than it first shipped. At 5s this ran alongside
    /// DeviceDetailView's own 4s refresh, so with the device screen open the
    /// app made two round trips every couple of seconds for the same data —
    /// which is felt as lag on every other interaction, because they share a
    /// connection and the main actor.
    ///
    /// A notification arriving within fifteen seconds is indistinguishable from
    /// one arriving within five, for an event that persisted long enough to be
    /// confirmed in the first place.
    private let interval: Duration = .seconds(15)

    func start(state: AppState) {
        guard task == nil else { return }
        notifier.beginWatching()
        // Each watch starts from where every camera's journal ends: the backlog
        // is history, never news (LocalAlerts.advance).
        localCursors.removeAll()
        task = Task { [weak self] in
            while !Task.isCancelled {
                let local = await self?.poll(state: state) ?? false
                try? await Task.sleep(for: local ? (self?.localInterval ?? .seconds(5)) : (self?.interval ?? .seconds(5)))
            }
        }
    }

    /// M12 (C6): where each paired camera's journal was last read, by hardware id.
    private var localCursors: [String: Int] = [:]
    /// Polling the camera on this network is a small local request, so it runs
    /// more often than the account poll.
    private let localInterval: Duration = .seconds(5)

    func stop() {
        task?.cancel()
        task = nil
    }

    /// One poll. True when it read the cameras directly (the local path).
    private func poll(state: AppState) async -> Bool {
        await notifier.refreshAuthorization()
        guard notifier.authorized else { return false }
        // M12 (C6): the account server is not connected (or there is no
        // account), but a paired camera answers on this network: read its own
        // journal for new events, through the same notification and sound
        // rules. Foreground only, like everything here.
        if !state.isDemo, !(state.isSignedIn && state.periodicRefreshMayRun) {
            return await pollLocal()
        }
        await pollAccount(state: state)
        return false
    }

    private func pollLocal() async -> Bool {
        let center = LocalAccessCenter.shared
        var polled = false
        for device in center.devices where center.isReady(device.hardwareID) {
            let key = device.hardwareID.uppercased()
            let cursor = localCursors[key]
            // First look: just where the journal ends (the newest row).
            guard let page = try? await center.events(device.hardwareID, after: cursor, limit: cursor == nil ? 1 : 50)
            else { continue }
            polled = true
            let (fresh, next) = LocalAlerts.advance(cursor: cursor, page: page)
            localCursors[key] = next
            if !fresh.isEmpty {
                await notifier.announceLocal(fresh, deviceName: device.name, deviceID: device.backendDeviceID)
            }
        }
        return polled
    }

    private func pollAccount(state: AppState) async {
        guard state.isSignedIn else { return }

        // Don't poll into a backend that is known not to be there. Each attempt
        // otherwise costs a 15-second request timeout plus a Bonjour search, so a
        // 15-second loop against an absent server never stops working — it just
        // runs continuously, draining the battery to rediscover the same absence.
        //
        // Nothing is lost by waiting: AppState is already retrying on its own
        // schedule, and the moment it reconnects this resumes on the next tick.
        //
        // `periodicRefreshMayRun`, not `backendStatus == .connected`: the demo
        // never resolves a backend (it has none), so the connected-only check
        // meant a demo alert never reached the lock screen unless a real backend
        // happened to be reachable — the opposite of what a demo needs.
        guard state.periodicRefreshMayRun else { return }

        // A failed poll is silent on purpose. The backend being briefly
        // unreachable is normal — the phone changes network, the Mac sleeps —
        // and surfacing it here would put an error in front of someone who did
        // not ask for one. The device screen already reports reachability.
        guard let devices = try? await state.backend.devices() else { return }
        for device in devices {
            guard let events = try? await state.backend.events(deviceID: device.id, limit: 25)
            else { continue }
            // Dedup lives in AlertNotifier and is persisted, so polling from
            // here as well as from the device screen cannot double-announce.
            await notifier.announce(events, device: device, simulated: state.isDemo)
        }
    }
}
