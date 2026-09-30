import SwiftUI

/// Watches every device for new alerts, from anywhere in the app.
///
/// Notifications used to be posted only by DeviceDetailView's refresh loop,
/// which runs only while that one screen is open. So an alert reached you
/// exactly when you were already looking at the device and least needed
/// telling, and never when you were on the Devices list, in Settings, or
/// looking at something else — which is every moment a notification is for.
///
/// **The honest limit**, unchanged and worth stating plainly: these are local
/// notifications, so the app posts them. It must be running or very recently
/// backgrounded — iOS suspends timers in a backgrounded app and decides for
/// itself when to let one run again. Force-quit the app and nothing arrives.
/// That is the difference between this and APNs, and it is why this is a demo
/// path rather than the shipping one.
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
        task = Task { [weak self] in
            while !Task.isCancelled {
                await self?.poll(state: state)
                try? await Task.sleep(for: self?.interval ?? .seconds(5))
            }
        }
    }

    func stop() {
        task?.cancel()
        task = nil
    }

    private func poll(state: AppState) async {
        await notifier.refreshAuthorization()
        guard notifier.authorized, state.isSignedIn else { return }

        // Don't poll into a backend that is known not to be there. Each attempt
        // otherwise costs a 15-second request timeout plus a Bonjour search, so a
        // 15-second loop against an absent server never stops working — it just
        // runs continuously, draining the battery to rediscover the same absence.
        //
        // Nothing is lost by waiting: AppState is already retrying on its own
        // schedule, and the moment it reconnects this resumes on the next tick.
        guard case .connected = state.backendStatus else { return }

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
