import Foundation

// Standalone, run by ios/Tests/run.sh (concatenated with ios/Sources/EventModel.swift).
//
// The backend now records which exact model produced each event. These pin
// the app's half of that contract: the JSON cloud/app.py returns decodes, and
// an event from before model tracking (no "model", or null) still decodes.

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

let sha = String(repeating: "e6", count: 32)
let hailo = """
{"module": "driver_attention",
 "detector": {"name": "damoyolo_tinynasL20_T.hef", "sha256": "\(sha)", "backend": "hailo"}}
""".data(using: .utf8)!

print("a backend model identity decodes:")
let decoded = try? JSONDecoder().decode(EventModel.self, from: hailo)
check(decoded?.module == "driver_attention", "module")
check(decoded?.detector?.name == "damoyolo_tinynasL20_T.hef", "the file that ran, not the .pt cache key")
check(decoded?.detector?.sha256 == sha, "the full sha256 survives")
check(decoded?.detector?.backend == "hailo", "backend")
check(decoded?.summary == "damoyolo_tinynasL20_T.hef · e6e6e6e6e6e6", "a short summary for display")

print("\nidentities the backend can legitimately send:")
let noDetector = #"{"module": "fall_detection", "detector": null}"#.data(using: .utf8)!
let pose = try? JSONDecoder().decode(EventModel.self, from: noDetector)
check(pose != nil && pose?.detector == nil, "a module with no object detector")
check(pose?.summary == "fall_detection", "is summarised by its module")
let unhashed = #"{"module": "m", "detector": {"name": "yolo11n.pt", "sha256": null, "backend": "ultralytics"}}"#
let cpu = try? JSONDecoder().decode(EventModel.self, from: unhashed.data(using: .utf8)!)
check(cpu?.summary == "yolo11n.pt", "a detector whose file could not be hashed is named, not invented")

// These decode the app's real `EventItem` (EventModel.swift), against rows
// shaped exactly like cloud/app.py's device_events():
//   {**dict(row), "model": _stored_model(row["model"])} with the SELECTed columns
//   id, timestamp, type, confidence, signals, severity, source, duration, model.
print("\nreal event rows decode as EventItem:")
let current = """
{"events": [
 {"id": 812, "timestamp": "2026-10-15T01:13:52.642123+00:00", "type": "fall",
  "confidence": 0.94, "signals": "rapid_drop + stillness", "severity": "loud",
  "source": "module", "duration": 2.1,
  "model": {"module": "fall_detection", "detector": null}},
 {"id": 811, "timestamp": "2026-10-15T01:12:00+00:00", "type": "person",
  "confidence": null, "signals": "", "severity": "soft", "source": "custom_rule",
  "duration": null, "model": null}
]}
""".data(using: .utf8)!
struct EventsBody: Decodable { let events: [EventItem] }
let rows = (try? JSONDecoder().decode(EventsBody.self, from: current))?.events ?? []
check(rows.count == 2, "a current backend's rows decode, unknown columns (duration) and all")
check(rows.first?.eventID == 812, "the row id is the event id, so /explain can be asked")
check(rows.first?.model?.summary == "fall_detection", "the model identity rides along")
check(rows.first?.date != nil, "a timestamp with microseconds and an offset parses")
check(rows.last?.date != nil,
      "a whole-second timestamp parses too (isoformat drops a zero fraction)")
check(rows.last?.confidence == nil, "null confidence: no graded evidence, not 0%")
check(rows.last?.isConfigured == true, "source custom_rule is something the user asked for")
check(rows.first?.title == "Fall", "the type reads as a title")

print("\nhistory is still readable:")
let old = #"{"events": [{"timestamp": "2026-01-02T03:04:05.000+00:00", "type": "drowsiness", "confidence": 0.71}]}"#
let oldRows = (try? JSONDecoder().decode(EventsBody.self, from: old.data(using: .utf8)!))?.events
check(oldRows?.count == 1, "an event from a backend that predates ids, severity, source and model")
check(oldRows?.first?.eventID == nil && oldRows?.first?.model == nil, "...with those simply absent")
check(oldRows?.first?.alertEvidence == AlertDeliveryEvidence.none,
      "history says nothing about an alert output, so the app claims none")
check(oldRows?.first?.date != nil && oldRows?.first?.date == oldRows?.first?.deviceDate,
      "without a backend time, the device's own timestamp is shown")

// ---------------------------------------------------------------------- M8.3
// Rows shaped exactly like cloud/app.py's _event_row (docs/EVENT_PIPELINE.md).
func decodeRows(_ json: String) -> [EventItem] {
    (try? JSONDecoder().decode(EventsBody.self, from: json.data(using: .utf8)!))?.events ?? []
}

print("\nM8.3 rows decode with identity, both clocks, evidence and provenance:")
let m83 = decodeRows("""
{"events": [
 {"id": 901, "timestamp": "1970-01-01T00:02:11+00:00", "type": "drowsiness", "confidence": 0.9,
  "signals": "eyes closed", "severity": "loud", "source": "module", "duration": 2.0, "model": null,
  "event_key": "8f14e45fceea167a5a36dedd4bea2543", "received_at": 1763020000.5,
  "occurred_at": 1763019998.0, "time_basis": "device_monotonic", "simulated": false,
  "alert": {"decision": "alerted", "output": "sounded", "rev": 2, "episode": "0123456789ab-4",
            "category": "critical",
            "outputs": [{"name": "gpio_buzzer", "kind": "buzzer", "audible": true, "played": 1,
                         "failed": 0, "skipped": 0, "pending": 0, "unconfirmed": 0}]}},
 {"id": 900, "timestamp": "2026-11-13T08:00:00+00:00", "type": "distraction", "confidence": 0.8,
  "signals": "", "severity": "loud", "source": "module", "duration": 3.0, "model": null,
  "event_key": "k900", "received_at": 1763020000.0, "occurred_at": 1763016400.0,
  "time_basis": "device_clock", "simulated": true, "alert": null}
]}
""")
check(m83.count == 2, "both rows decode")
let sounded = m83.first
check(sounded?.eventKey == "8f14e45fceea167a5a36dedd4bea2543", "the device's own event identity")
check(sounded?.date == Date(timeIntervalSince1970: 1763019998.0),
      "shown when it happened on the backend's clock, not the device's 1970 stamp")
check(sounded?.deviceDate == Date(timeIntervalSince1970: 131), "the device's claim is still readable, verbatim")
check((sounded?.deviceClockError ?? 0) < -1_000_000_000, "a device clock that far off is flagged")
check(sounded?.alertEvidence == .sounded(outputs: ["buzzer"]), "the device REPORTED its buzzer played")
check(sounded?.alertEvidence.headline == "Alert sounded on the device", "and only then does the app say so")
check(sounded?.isSimulated == false, "a real device's event")
let late = m83.last
check(late?.isSimulated == true, "a simulated event is marked")
check(late?.deliveryDelay == 3600, "an hour between happening and arriving is visible")
check(late?.deviceClockError == nil, "the device clock was the evidence: nothing to compare it with")
check(late?.alertEvidence.headline == "Detected; no alert output confirmed",
      "a stored event is a detection, never proof a warning sounded")

print("\nthe app never claims an alert sounded without the device saying so:")
func evidence(_ output: String?, simulated: Bool = false, error: String? = nil) -> AlertDeliveryEvidence {
    let report = output.map {
        EventAlertReport(decision: "alerted", output: $0,
                         outputs: [EventAlertReport.Output(name: "gpio_buzzer", kind: "buzzer", audible: true,
                                                           played: $0 == "sounded" ? 1 : 0, error: error)])
    }
    return AlertDeliveryEvidence(report, eventSimulated: simulated)
}
let confirmed = "Detected; no alert output confirmed"
check(evidence(nil).headline == confirmed, "no report: detected, nothing confirmed")
check(evidence("pending").headline == confirmed, "pending is not sounded")
check(evidence("unconfirmed").headline == confirmed, "an output that can't report is not sounded")
check(evidence("none").headline == confirmed, "no output commanded")
check(evidence("failed", error: "GPIO17 busy").detail.contains("GPIO17 busy"), "a failure says why")
check(evidence("failed").headline == confirmed, "a failed output is not sounded")
check(evidence("simulated").headline == "Simulated alert (no sound)", "a simulated output says so")
check(evidence("sounded", simulated: true) == .simulated, "a simulated event never reads as a real sound")
check(evidence("made-up").headline == confirmed, "an unknown value is not evidence")
check(evidence("sounded").confirmedSound && !evidence("pending").confirmedSound, "confirmedSound is sounded only")

print("\nthe timeline merges refreshes without duplicates, losses or reordering:")
func row(_ id: Int, at occurred: Double, rev: Int? = nil, output: String? = nil) -> EventItem {
    var item = EventItem(eventID: id, timestamp: "2026-11-13T08:00:00+00:00", type: "t\(id)", confidence: nil,
                         signals: nil, severity: "loud", source: "module")
    item.occurredAt = occurred
    if let output { item.alert = EventAlertReport(decision: "alerted", output: output, rev: rev) }
    return item
}
let shown = [row(3, at: 300), row(2, at: 200), row(1, at: 100)]
let refreshed = EventFeed.merge(shown, [row(4, at: 400), row(3, at: 300, rev: 2, output: "sounded")])
check(refreshed.map(\.eventID) == [4, 3, 2, 1], "a smaller newer page adds to the timeline, losing nothing")
check(refreshed[1].alert?.output == "sounded", "a newer report for a shown event replaces it in place")
let stale = EventFeed.merge(refreshed, [row(3, at: 300, rev: 1, output: "pending")])
check(stale[1].alert?.output == "sounded", "a stale response never replaces a newer report")
check(Set(EventFeed.merge(refreshed, refreshed).map(\.id)).count == 4, "the same page twice is one copy")
// Out-of-order arrival: row 6 arrived after row 5 but happened before it.
let arrival = EventFeed.merge([], [row(6, at: 450), row(5, at: 500)])
check(arrival.map(\.eventID) == [5, 6], "ordered by when it happened, not by when it arrived")
check(EventFeed.ordered([row(7, at: 100), row(8, at: 100)]).map(\.eventID) == [8, 7], "ties: newest row first")
check(Set([row(1, at: 1).id, row(1, at: 99, rev: 3, output: "sounded").id]).count == 1,
      "an event's identity does not change when its report does")

print("\n\(failures == 0 ? "PASSED" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)
