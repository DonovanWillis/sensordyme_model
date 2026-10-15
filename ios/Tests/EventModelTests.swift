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

print("\n\(failures == 0 ? "PASSED" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)
