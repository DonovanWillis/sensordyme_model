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

print("\nhistory is still readable:")
struct Row: Decodable { let type: String; let confidence: Double?; let model: EventModel? }
let old = #"{"type": "drowsiness", "confidence": 0.71}"#.data(using: .utf8)!
let oldRow = try? JSONDecoder().decode(Row.self, from: old)
check(oldRow != nil && oldRow?.model == nil, "an event from before model tracking (no key)")
let nulls = #"{"type": "frequent_glancing", "confidence": null, "model": null}"#.data(using: .utf8)!
let nullRow = try? JSONDecoder().decode(Row.self, from: nulls)
check(nullRow != nil && nullRow?.confidence == nil, "null confidence: no graded evidence, not 0%")

print("\n\(failures == 0 ? "PASSED" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)
