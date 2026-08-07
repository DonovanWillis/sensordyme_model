import SwiftUI
import UserNotifications

/// SensorDyme alerts on the lock screen, with no push service and no paid account.
///
/// **Why local rather than push.** A real product uses APNs, which needs a paid
/// Apple Developer membership. Local notifications need neither an entitlement
/// nor a server — the app schedules them itself — and on the lock screen they
/// are indistinguishable from a push: SensorDyme's name, SensorDyme's icon,
/// SensorDyme's sound, and tapping one opens the alert.
///
/// The backend already has an ntfy path (free, no account) but its
/// notifications are branded "ntfy" and open the ntfy app, which is the wrong
/// thing to show someone evaluating the product.
///
/// **The honest limit**, which matters for how you demo: the app posts these,
/// so it must be running or recently backgrounded. Force-quit the app and
/// nothing arrives. iOS also decides how often a backgrounded app may poll. For
/// a demo — phone in hand, app open or just backgrounded — this behaves exactly
/// like the real thing. For a shipping product it does not replace APNs.
@MainActor
final class AlertNotifier: NSObject, ObservableObject {
    static let shared = AlertNotifier()

    /// Set when a notification is tapped, so the app can open that alert.
    @Published var pendingEventID: Int?
    @Published var authorized = false
    @Published var lastError: String?

    /// Event ids already announced. Persisted because the poller re-reads the
    /// same events every few seconds, and re-announcing them would turn one
    /// event into a stream of duplicate alerts.
    private var announced: Set<Int> {
        get { Set(UserDefaults.standard.array(forKey: "announcedEvents") as? [Int] ?? []) }
        set {
            // Bounded: the tail is old enough that it can never re-arrive.
            UserDefaults.standard.set(Array(newValue.suffix(500)), forKey: "announcedEvents")
        }
    }

    private override init() {
        super.init()
        UNUserNotificationCenter.current().delegate = self
    }

    func requestPermission() async {
        do {
            authorized = try await UNUserNotificationCenter.current()
                .requestAuthorization(options: [.alert, .sound, .badge])
        } catch {
            lastError = error.localizedDescription
            authorized = false
        }
    }

    func refreshAuthorization() async {
        let settings = await UNUserNotificationCenter.current().notificationSettings()
        authorized = settings.authorizationStatus == .authorized
            || settings.authorizationStatus == .provisional
    }

    /// Announce any events not yet seen. Safe to call on every poll.
    func announce(_ events: [EventItem], device: Device, simulated: Bool) async {
        guard authorized else { return }
        var seen = announced

        // Oldest first, so a burst arrives in the order it happened.
        for event in events.reversed() {
            guard let id = event.eventID, !seen.contains(id) else { continue }
            // Only things worth interrupting someone for. Informational events
            // belong in the timeline, not on the lock screen — a product that
            // buzzes for everything gets its notifications switched off.
            guard event.severityStatus != .online else {
                seen.insert(id)
                continue
            }
            await post(event, device: device, simulated: simulated)
            seen.insert(id)
        }
        announced = seen
    }

    private func post(_ event: EventItem, device: Device, simulated: Bool) async {
        let content = UNMutableNotificationContent()
        content.title = event.isCritical ? "SensorDyme Alert" : "SensorDyme"
        content.subtitle = event.title
        content.body = body(for: event, device: device)
        content.sound = event.isCritical ? .defaultCritical : .default
        content.userInfo = ["eventID": event.eventID ?? -1, "deviceID": device.id]
        content.threadIdentifier = "device-\(device.id)"

        if simulated {
            // Never let a demo alert be mistaken for a real detection. This is
            // the difference between a demo and a misrepresentation.
            content.subtitle = "\(event.title) (simulated)"
        }
        // `.timeSensitive` needs an entitlement a free account may not carry, so
        // this stays at the default level rather than risk failing to provision.
        content.interruptionLevel = .active

        let request = UNNotificationRequest(
            identifier: "event-\(event.eventID ?? Int.random(in: 0...99999))",
            content: content,
            trigger: nil   // deliver now
        )
        do {
            try await UNUserNotificationCenter.current().add(request)
        } catch {
            lastError = error.localizedDescription
        }
    }

    private func body(for event: EventItem, device: Device) -> String {
        var parts = [device.name]
        if let signals = event.signals, !signals.isEmpty {
            parts.append(signals.replacingOccurrences(of: "_", with: " "))
        }
        if let confidence = event.confidence, confidence > 0 {
            parts.append("\(Int(confidence * 100))% confidence")
        }
        return parts.joined(separator: " · ")
    }

    /// Forget what has been announced — so a rehearsed demo can be run twice.
    func resetAnnounced() {
        UserDefaults.standard.removeObject(forKey: "announcedEvents")
    }
}

extension AlertNotifier: UNUserNotificationCenterDelegate {
    /// Show the banner even while the app is open, so a demo works without
    /// backgrounding the app first.
    nonisolated func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification
    ) async -> UNNotificationPresentationOptions {
        [.banner, .sound, .list]
    }

    nonisolated func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        didReceive response: UNNotificationResponse
    ) async {
        let info = response.notification.request.content.userInfo
        guard let id = info["eventID"] as? Int, id >= 0 else { return }
        await MainActor.run { AlertNotifier.shared.pendingEventID = id }
    }
}
