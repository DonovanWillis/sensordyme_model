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
    /// Which device the tapped alert belongs to, so the app can open the right
    /// one. Carried in the notification's userInfo since it was first posted;
    /// nothing read it back until tapping had to work from any screen.
    @Published var pendingDeviceID: Int?
    @Published var authorized = false
    @Published var lastError: String?

    /// Minimum gap between notifications, in seconds.
    ///
    /// Not a detection setting. Jupiter keeps detecting and recording at full
    /// rate; this governs only how often the phone is allowed to interrupt.
    static let cooldown: TimeInterval = 20

    /// How many red alerts were recorded but not notified because the cooldown
    /// was active. Shown so the suppression is visible rather than silent —
    /// "3 more alerts in this period" is information; quietly dropping them is
    /// the same behaviour without the honesty.
    @Published private(set) var suppressedDuringCooldown = 0

    /// When the last notification was posted. Persisted so relaunching the app
    /// during a busy period does not immediately fire another one.
    private var lastNotifiedAt: Date? {
        get {
            let t = UserDefaults.standard.double(forKey: "lastNotifiedAt")
            return t > 0 ? Date(timeIntervalSince1970: t) : nil
        }
        set { UserDefaults.standard.set(newValue?.timeIntervalSince1970 ?? 0, forKey: "lastNotifiedAt") }
    }

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
        var suppressed = 0

        // Oldest first, so a burst arrives in the order it happened.
        for event in events.reversed() {
            guard let id = event.eventID else { continue }

            // The whole decision lives in NotificationPolicy, which is a pure
            // function with tests. Inlining it here is how it drifted before:
            // "not purely informational" quietly admitted both severities, and
            // on a real device that was 942 "loud" plus 608 "soft" in a day
            // from a driver-monitor doing exactly its job.
            let decision = NotificationPolicy.decide(
                isCritical: event.isCritical,
                isConfigured: event.isConfigured,
                alreadySeen: seen.contains(id),
                now: Date(),
                lastNotifiedAt: lastNotifiedAt,
                cooldown: Self.cooldown
            )
            seen.insert(id)   // considered, whatever we decide — never reconsidered

            switch decision {
            case .notify:
                await post(event, device: device, simulated: simulated)
                lastNotifiedAt = Date()
            case .suppress(.withinCooldown):
                // Recorded, not interrupted. Already in the event history.
                suppressed += 1
            case .suppress:
                break
            }
        }

        announced = seen
        if suppressed > 0 {
            suppressedDuringCooldown += suppressed
        }
    }

    private func post(_ event: EventItem, device: Device, simulated: Bool) async {
        let content = UNMutableNotificationContent()
        content.title = event.isCritical ? "SensorDyme Alert" : "SensorDyme"
        content.subtitle = event.title
        content.body = body(for: event, device: device) + " Tap to view details."
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
        // The device id has always been in the payload; read it back so the app
        // can open the right device from whatever screen it happens to be on.
        let deviceID = info["deviceID"] as? Int
        await MainActor.run {
            AlertNotifier.shared.pendingDeviceID = deviceID
            AlertNotifier.shared.pendingEventID = id
        }
    }
}
