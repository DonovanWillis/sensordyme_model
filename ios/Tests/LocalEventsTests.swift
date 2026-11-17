import Foundation

// Run with ios/Tests/run.sh
// run.sh deps: EventModel AlertSound NotificationPolicy
//
// The camera's own history (GET /api/events) merged with the account's, one
// row per event_key; honest sync labels; and the alert sound for an event that
// arrived over the local link: never for the backlog, never twice for the same
// event_key whichever way it came, never for an event older than 60 s, never
// for a simulated, experimental or log-only one.

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

let pageJSON = """
{"schema":"sd-local-events/1","latest_seq":3,"boot_id":"b112b491","retention":{"max_events":2000,"kept":3,"oldest":"x"},
 "fault":null,"events":[
  {"seq":3,"event_key":"k3","type":"drowsiness","timestamp":"2026-11-15T10:00:30+00:00","severity":"loud",
   "confidence":0.91,"signals":"eyes closed 2.1 s","duration":2.1,"experimental":false,"simulated":false,
   "alert":{"decision":"alerted","output":"sounded","rev":2},"age_s":4.5,"sync":"pending"},
  {"seq":2,"event_key":"k2","type":"monitoring_ready","timestamp":"2026-11-15T10:00:00+00:00","severity":"log",
   "confidence":null,"signals":"","duration":0.0,"experimental":false,"simulated":false,"alert":{"weird":1},"age_s":null,"sync":"local_only"},
  {"seq":1,"event_key":"k1","type":"phone_use","timestamp":"2026-11-15T09:59:00+00:00","severity":"loud",
   "confidence":0.8,"signals":"","duration":1.0,"experimental":false,"simulated":false,"alert":null,"age_s":95.0,"sync":"sent"}]}
"""
let page = try! JSONDecoder().decode(LocalEventsPage.self, from: Data(pageJSON.utf8))
let fetchedAt = Date(timeIntervalSince1970: 1_800_000_000)

print("the camera's page decodes:")
check(page.events.count == 3 && page.latestSeq == 3 && page.bootID == "b112b491", "events, latest_seq, boot_id")
check(page.events[0].alert?.output == "sounded", "an alert report this build reads")
check(page.events[1].alert?.output == nil && page.events[1].ageS == nil, "an unreadable report is absent; no age is nil")

print("\nas app events:")
let item = LocalHistory.eventItem(page.events[0], fetchedAt: fetchedAt)
check(item.eventKey == "k3" && item.eventID == nil, "keyed by event_key, no server id")
check(item.date == fetchedAt.addingTimeInterval(-4.5), "age_s placed on the phone's clock, not the camera's")
check(LocalHistory.eventItem(page.events[1], fetchedAt: fetchedAt).date
      == LocalHistory.eventItem(page.events[1], fetchedAt: fetchedAt).deviceDate,
      "no age: the camera's own timestamp")
check(item.severity == "loud" && item.signals == "eyes closed 2.1 s" && item.alert?.output == "sounded",
      "severity, signals and the alert report carried over (the same reading as an account event)")

print("\nmerged with the account's history, no duplicates:")
var accountRow = EventItem(eventID: 41, timestamp: "2026-11-15T09:59:00+00:00", type: "phone_use", confidence: 0.8,
                           signals: "", severity: "loud", source: nil)
accountRow.eventKey = "k1"
accountRow.occurredAt = fetchedAt.timeIntervalSince1970 - 95
let merged = LocalHistory.merge(account: [accountRow], local: page.events, fetchedAt: fetchedAt)
check(merged.events.count == 3, "three occurrences, not four")
check(merged.events.contains { $0.eventID == 41 } && !merged.events.contains { $0.eventKey == "k1" && $0.eventID == nil },
      "the account row is kept for k1 (it has the server id), the camera's copy dropped")
check(merged.events.first?.eventKey == "k3", "newest first")
check(merged.notes["key:k3"] == "Not yet backed up", "pending -> \"Not yet backed up\"")
check(merged.notes["key:k2"] == "On this camera only · Time from the camera's clock (an earlier power-on)",
      "local_only -> \"On this camera only\", and an event with no age says whose clock its time is")
check(merged.notes["id:41"] == nil, "an account row reads exactly as before")
check(LocalHistory.syncLabel("sent") == "Sent to your account" && LocalHistory.syncLabel(nil) == "Backup status unknown",
      "sent, and unknown")
let twice = LocalHistory.merge(account: [], local: page.events + page.events, fetchedAt: fetchedAt)
check(twice.events.count == 3, "the same camera row twice is one row")

print("\nforeground polling: the backlog is never news:")
let first = LocalAlerts.advance(cursor: nil, page: page)
check(first.fresh.isEmpty && first.cursor == 3, "the first poll only records where the journal ends")
var later = page
later.events.insert(LocalEvent(seq: 4, eventKey: "k4", type: "distraction", timestamp: "t", severity: "loud", ageS: 2), at: 0)
later.latestSeq = 4
let second = LocalAlerts.advance(cursor: first.cursor, page: later)
check(second.fresh.map(\.eventKey) == ["k4"] && second.cursor == 4, "the next one returns only what is new")
check(LocalAlerts.advance(cursor: 4, page: later).fresh.isEmpty, "and nothing twice")
var reset = page
reset.latestSeq = 1
check(LocalAlerts.advance(cursor: 4, page: reset) == ([], 1), "a journal that went backwards re-baselines, never replays")

print("\nthe sound, under the M11 rules:")
let fresh = LocalEvent(seq: 4, eventKey: "k4", type: "distraction", timestamp: "t", severity: "loud", ageS: 2)
var sounded = SoundedEvents()
check(LocalAlerts.soundDecision(fresh, sounded: sounded, soundEnabled: true) == .play, "a fresh driver alert plays")
check(LocalAlerts.soundDecision(fresh, sounded: sounded, soundEnabled: false) == .silent(.turnedOff), "unless turned off")
var old = fresh; old.ageS = 61
check(LocalAlerts.soundDecision(old, sounded: sounded, soundEnabled: true) == .silent(.deliveredLate), "age_s 61 s: silent")
var edge = fresh; edge.ageS = 60
check(LocalAlerts.soundDecision(edge, sounded: sounded, soundEnabled: true) == .silent(.deliveredLate), "60 s is already late")
var noAge = fresh; noAge.ageS = nil
check(LocalAlerts.soundDecision(noAge, sounded: sounded, soundEnabled: true) == .silent(.deliveredLate),
      "no age (an earlier power-on): silent")
var simulated = fresh; simulated.simulated = true
check(LocalAlerts.soundDecision(simulated, sounded: sounded, soundEnabled: true) == .silent(.simulated), "simulated: silent")
var experimental = fresh; experimental.experimental = true
check(LocalAlerts.soundDecision(experimental, sounded: sounded, soundEnabled: true) == .silent(.notAnAlert), "experimental: silent")
var logged = fresh; logged.severity = "log"
check(LocalAlerts.soundDecision(logged, sounded: sounded, soundEnabled: true) == .silent(.notAnAlert), "log severity: silent")
let visible = LocalEvent(seq: 5, eventKey: "k5", type: "phone_detected", timestamp: "t", severity: "loud", ageS: 2)
check(LocalAlerts.soundDecision(visible, sounded: sounded, soundEnabled: true) == .silent(.notAnAlert), "a log-only type: silent")

print("\nonce per event_key across both transports:")
// The account path sounded it first: its identity is key:<event_key>.
let fromAccount = AlertSound.identity(eventKey: "k4", eventID: 41)!
sounded.insert(fromAccount)
check(LocalAlerts.soundDecision(fresh, sounded: sounded, soundEnabled: true) == .silent(.alreadySounded),
      "an event that already sounded from the account does not sound again locally")
var other = SoundedEvents()
other.insert(AlertSound.identity(eventKey: "k4", eventID: nil)!)
check(AlertSound.decide(type: "distraction", severity: "loud", experimental: false, deliveredLate: false,
                        alreadySounded: other.contains(AlertSound.identity(eventKey: "k4", eventID: 41)!),
                        soundEnabled: true) == .silent(.alreadySounded),
      "and one that sounded locally does not sound again when the account delivers it later")

print("\n\(failures == 0 ? "all passed" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)
