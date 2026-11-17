import Foundation

// Run with ios/Tests/run.sh (concatenated with ios/Sources/AlertSound.swift and
// the sources named below).
// run.sh deps: NotificationPolicy EventModel
//
// The iPhone's alert sound for a new alert: only an alert sounds, only once per
// event, only something happening now, never for the backlog the app finds when
// it starts watching, and a simulated one says so.

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

func sound(_ type: String = "distraction", severity: String? = "loud", experimental: Bool = false,
           late: Bool = false, sounded: Bool = false, enabled: Bool = true) -> AlertSound.Decision {
    AlertSound.decide(type: type, severity: severity, experimental: experimental,
                      deliveredLate: late, alreadySounded: sounded, soundEnabled: enabled)
}

print("the driver-safety alerts are the device's validated, mandatory ones:")
for type in ["distraction", "eyes_down", "eyes_up", "phone_use", "drowsiness", "attention_lost"] {
    check(AlertSound.isDriverSafetyAlert(type: type), "\(type) is a driver-safety alert")
    check(sound(type) == .play, "a new loud \(type) plays the sound")
}
check(AlertSound.isDriverSafetyAlert(type: "eyes_down:cab"), "a zone-scoped type is read by its base type")
for type in ["phone_detected", "occupancy_change", "high_risk", "erratic_hand_movement", "fall"] {
    check(!AlertSound.isDriverSafetyAlert(type: type), "\(type) is not a driver-safety alert")
}

print("\nonly an alert sounds:")
check(sound("phone_detected") == .silent(.notAnAlert),
      "phone_detected never sounds, whatever severity it carries (a visible phone is not phone use)")
check(sound("occupancy_change") == .silent(.notAnAlert), "a log-only type never sounds")
check(sound("high_risk", severity: "log", experimental: true) == .silent(.notAnAlert),
      "an experimental observation never sounds")
check(sound("high_risk", severity: "loud", experimental: true) == .silent(.notAnAlert),
      "not even when older firmware tagged it loud")
check(sound("distraction", severity: "log") == .silent(.notAnAlert), "a log-severity record never sounds")
check(sound("fall") == .play, "another alert the app already notifies about keeps its sound (unchanged)")

print("\nonce per event:")
check(sound(sounded: true) == .silent(.alreadySounded), "an event that has sounded never sounds again")
check(AlertSound.identity(eventKey: "dev1-0042", eventID: 7) == "key:dev1-0042",
      "identity is the device's event key when it sent one")
check(AlertSound.identity(eventKey: nil, eventID: 7) == "id:7", "else the server row id")
check(AlertSound.identity(eventKey: "", eventID: 7) == "id:7", "an empty key is no key")
check(AlertSound.identity(eventKey: nil, eventID: nil) == nil, "an event with neither has no identity")
do {
    var log = SoundedEvents()
    check(log.insert("key:a"), "the first sighting is recorded")
    check(!log.insert("key:a"), "the same event a second time is refused")
    check(log.contains("key:a") && !log.contains("key:b"), "and only that event is remembered")
    for i in 0..<(SoundedEvents.capacity + 10) { log.insert("id:\(i)") }
    check(log.identities.count == SoundedEvents.capacity, "bounded to \(SoundedEvents.capacity)")
    check(!log.contains("key:a") && log.contains("id:\(SoundedEvents.capacity + 9)"),
          "the oldest are forgotten first")
    let restored = SoundedEvents(log.identities + ["x"])
    check(restored.identities.count == SoundedEvents.capacity, "a restored list is bounded too")
}

print("\nonly something happening now:")
check(sound(late: true) == .silent(.deliveredLate),
      "an alert the backend received a minute or more after it happened is shown silently")
let lateJSON = #"{"id": 9, "timestamp": "2026-11-14T10:00:00Z", "type": "distraction", "severity": "loud", "occurred_at": 1000, "received_at": 1300, "time_basis": "device_monotonic"}"#
let onTimeJSON = #"{"id": 10, "timestamp": "2026-11-14T10:00:00Z", "type": "distraction", "severity": "loud", "occurred_at": 1000, "received_at": 1002, "time_basis": "device_monotonic"}"#
let late = try! JSONDecoder().decode(EventItem.self, from: lateJSON.data(using: .utf8)!)
let onTime = try! JSONDecoder().decode(EventItem.self, from: onTimeJSON.data(using: .utf8)!)
check(late.deliveryDelay != nil, "a 5-minute backlog reads as delivered late (what the notifier passes)")
check(onTime.deliveryDelay == nil, "a 2-second arrival does not")

print("\nthe person can turn it off:")
check(sound(enabled: false) == .silent(.turnedOff), "off means silent, for every alert")

print("\na simulated alert never sounds (M12), on either path:")
check(AlertSound.decide(type: "distraction", severity: "loud", experimental: false, deliveredLate: false,
                        alreadySounded: false, soundEnabled: true, simulated: true) == .silent(.simulated),
      "a simulated driver alert is silent")
check(AlertSound.decide(type: "distraction", severity: "loud", experimental: false, deliveredLate: false,
                        alreadySounded: false, soundEnabled: true, simulated: false) == .play,
      "the same real alert plays")

print("\na simulated alert says so in the headline:")
check(AlertSound.title(type: "distraction", critical: true, simulated: false) == "Driver alert", "a real driver alert")
check(AlertSound.title(type: "phone_use", critical: true, simulated: true) == "Simulated · Driver alert",
      "a simulated one is labelled before anything else")
check(AlertSound.title(type: "fall", critical: true, simulated: false) == "SensorDyme Alert",
      "another red alert keeps its headline")
check(AlertSound.title(type: "custom", critical: false, simulated: true) == "Simulated · SensorDyme",
      "and a configured rule's")

// The whole path for one watch, as AlertNotifier runs it: the backlog the app
// finds when it starts watching is never announced, so it never reaches the
// sound decision; a new alert sounds once; its re-read does not.
print("\nend to end over one foreground watch:")
do {
    var mark: Int? = nil
    var baselined = false
    var log = SoundedEvents()
    var played: [Int] = []
    func poll(_ events: [(id: Int, type: String)]) {
        let (fresh, next) = NotificationPolicy.freshEvents(ids: events.map(\.id), mark: mark, baselined: baselined)
        mark = next
        baselined = true
        for id in fresh {
            let type = events.first { $0.id == id }!.type
            let identity = AlertSound.identity(eventKey: nil, eventID: id)!
            if sound(type, sounded: log.contains(identity)) == .play {
                log.insert(identity)
                played.append(id)
            }
        }
    }
    poll([(3, "distraction"), (2, "phone_use")])
    check(played.isEmpty, "launch: the backlog never sounds")
    poll([(4, "eyes_down"), (3, "distraction")])
    check(played == [4], "a new eyes_down sounds once")
    poll([(4, "eyes_down"), (3, "distraction")])
    check(played == [4], "the same event re-read does not sound again")
    poll([(5, "phone_detected"), (4, "eyes_down")])
    check(played == [4], "a new phone_detected does not sound")
}

print("\n\(failures == 0 ? "PASSED" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)
