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

    /// Minimum gap between notifications, in seconds. Product spec, stated
    /// explicitly: at most one notification every 20 seconds, globally per
    /// account — matches cloud/notify.py's COOLDOWN_S exactly, so the two
    /// notification paths (local, posted by this app; push, delivered by the
    /// OS through the backend) never disagree about the same moment.
    ///
    /// Not a detection setting. The device keeps detecting and recording at
    /// full rate; this governs only how often the phone is allowed to
    /// interrupt.
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

    /// Per device, the newest event id already considered. Persisted because
    /// the poller re-reads the same events every few seconds, and re-announcing
    /// them would turn one event into a stream of duplicate alerts. Per device
    /// rather than one global set: see `NotificationPolicy.freshEvents`.
    private static let marksKey = "announcedMarks"
    /// The global newest-500 set this replaces. Removed, never read.
    private static let legacyAnnouncedKey = "announcedEvents"

    private var marks: [String: Int] {
        get { UserDefaults.standard.dictionary(forKey: Self.marksKey) as? [String: Int] ?? [:] }
        set { UserDefaults.standard.set(newValue, forKey: Self.marksKey) }
    }

    private override init() {
        super.init()
        UNUserNotificationCenter.current().delegate = self
        UserDefaults.standard.removeObject(forKey: Self.legacyAnnouncedKey)
    }

    /// Forget which events were announced, for a different account or the demo.
    /// Event ids are per backend (the demo's start at 1, like a fresh server's),
    /// so another account's ids would silence this one's first real alerts.
    /// `lastNotifiedAt` is kept: the cooldown still applies, so the first poll
    /// afterwards cannot turn old history into a burst of alerts.
    func forgetAnnounced() {
        UserDefaults.standard.removeObject(forKey: Self.marksKey)
    }

    func requestPermission() async {
        // Automated Simulator runs (docs/VIRTUAL_DEVICE.md): the system prompt would
        // cover the screen being checked. DEBUG builds only; Release always asks.
        if DebugLaunch.noPermissionPrompt { return }
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
        let key = String(device.id)
        var allMarks = marks
        let (fresh, mark) = NotificationPolicy.freshEvents(
            ids: events.compactMap(\.eventID), mark: allMarks[key]
        )
        allMarks[key] = mark
        // Stored before anything is awaited. The decision below is made, and
        // `lastNotifiedAt` moved, before posting too: posting suspends, and a
        // second poll arriving in that gap (a quick inactive -> active, which
        // restarts the watcher) used to read the old state and post the same
        // event again.
        marks = allMarks

        var byID: [Int: EventItem] = [:]
        for event in events { if let id = event.eventID, byID[id] == nil { byID[id] = event } }

        var toPost: [EventItem] = []
        var suppressed = 0
        // Oldest first, so a burst arrives in the order it happened.
        for id in fresh {
            guard let event = byID[id] else { continue }

            // The whole decision lives in NotificationPolicy, which is a pure
            // function with tests. Inlining it here is how it drifted before:
            // "not purely informational" quietly admitted both severities, and
            // on a real device that was 942 "loud" plus 608 "soft" in a day
            // from a driver-monitor doing exactly its job.
            let decision = NotificationPolicy.decide(
                isCritical: event.isCritical,
                isConfigured: event.isConfigured,
                alreadySeen: false,
                now: Date(),
                lastNotifiedAt: lastNotifiedAt,
                cooldown: Self.cooldown
            )
            switch decision {
            case .notify:
                toPost.append(event)
                lastNotifiedAt = Date()
            case .suppress(.withinCooldown):
                // Recorded, not interrupted. Already in the event history.
                suppressed += 1
            case .suppress:
                break
            }
        }

        if suppressed > 0 {
            suppressedDuringCooldown += suppressed
        }
        for event in toPost {
            await post(event, device: device, simulated: simulated)
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

        if simulated || event.isSimulated {
            // Never let a demo alert be mistaken for a real detection. This is
            // the difference between a demo and a misrepresentation. The event
            // itself says so too (M8.3): a virtual device's alert, reaching a
            // real account, is labelled the same way.
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

    /// Let a rehearsed demo alert through at once.
    ///
    /// Every demo trigger is a new event, so it is never "already announced";
    /// what holds a second rehearsal back is the 20-second cooldown. Clearing
    /// the per-device marks instead would only make the next poll treat
    /// everything as history (see `NotificationPolicy.freshEvents`).
    func resetAnnounced() {
        lastNotifiedAt = nil
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
