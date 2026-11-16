import Foundation

// Standalone, run by ios/Tests/run.sh (concatenated with ios/Sources/DeviceDiagnostics.swift).
//
// The developer diagnostics screen reads `GET /api/diagnostics` (sd-diagnostics/1,
// docs/DIAGNOSTICS.md). The first payload below is a REAL body captured from the
// device runtime on a Mac (tests/test_m10_observe.py's DebugModule, fake camera),
// trimmed only for length; the second is the same shape as a Pi with a detector.

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

let macBody = """
{"schema": "sd-diagnostics/1", "enabled": true,
 "note": "developer diagnostics: local, token-gated, never uploaded; no frames or secrets",
 "platform": {"system": "Darwin", "machine": "arm64", "python": "3.12.12", "board": null, "is_raspberry_pi": false,
              "hailo_runtime": "not loaded", "simulated": false, "class": "desktop",
              "note": "measured on a desktop host: NOT a Raspberry Pi measurement"},
 "snapshot_age_s": 0.2,
 "frame": {"module": "debuggy", "frame_id": 6, "frame_age_ms": 0.0, "simulated": false, "source": "camera",
   "detections": {"state": "idle", "reason": "idle", "items": null},
   "face": {"face": "landmark model did not run on this frame"}, "hands": null,
   "evidence": {"driver_association": {"method": "largest_face", "known": true},
     "hand_phone": {"phone_detected": true, "phone_confidence": 0.61, "hand_overlaps_phone": false,
                    "phone_hold": "unknown", "engine_phone_evidence": {"state": "visible"}},
     "state": "DISTRACTED", "gaze_zone": "Left", "abstaining": {"phone_use": "no hand"},
     "events_fired_this_frame": []},
   "open_conditions": [{"source": "module_tracker", "condition": "distraction", "state": "developing",
                        "running_s": 1.4}]},
 "warming_up": false,
 "capabilities": {"state": "ready", "monitoring": true,
   "available": ["camera", "face", "hands", "inference", "objects", "pose"],
   "impaired": ["alert_output", "audible_alert"], "paused": [],
   "detail": {"alert_output": {"state": "impaired", "reason": "no alert output is configured"}}},
 "unknown_or_degraded": ["alert_output: no alert output is configured",
                         "audible_alert: no audible alert output is configured"],
 "versions": {"software": "0.2", "model_version": "unknown", "detector": null,
   "detector_pin_sha256": "abababababababababababababababababababababababababababababababab", "decoder": "legacy",
   "mediapipe": {"mediapipe_face_mesh": "mediapipe-0.10.14", "mediapipe_hands": "mediapipe-0.10.14",
                 "mediapipe_pose": "mediapipe-0.10.14", "installed": "mediapipe-0.10.14"}},
 "latency": {"schema": "sd-stage-timing/1", "window_samples": 256,
   "stages": {"camera_read": {"n": 10, "p50_ms": 0.019, "p95_ms": 0.028, "max_ms": 0.03, "mean_ms": 0.019,
                              "last_ms": 0.01, "count": 10},
              "detector_total": null, "hailo_inference": null,
              "behaviour_reasoning": {"n": 10, "p50_ms": 1.52, "p95_ms": 2.041, "max_ms": 2.327, "mean_ms": 1.581,
                                      "last_ms": 1.514, "count": 10}},
   "processed_fps": 29.6, "rss_mb": 412.3, "rss_source": "peak (getrusage)",
   "soc_temperature_c": null, "throttle_now": null}}
""".data(using: .utf8)!

print("a real device body decodes:")
let mac = try? DeviceDiagnostics.decode(macBody)
check(mac != nil, "sd-diagnostics/1 decodes")
if let mac {
    check(DiagnosticsPresentation.hostBanner(mac) == "Desktop host (Darwin arm64): NOT a Raspberry Pi measurement",
          "a Mac figure is labelled as a desktop, never as a Pi")
    let det = DiagnosticsPresentation.detectionRows(mac.frame)
    check(det == [.init(title: "Detector", value: "can't look right now: idle")],
          "no detector output reads as 'can't look', never as 'nothing detected'")
    let conditions = DiagnosticsPresentation.conditionRows(mac.frame)
    check(conditions == [.init(title: "Distraction (developing)", value: "1.4 s")], "open condition with its running time")
    let latency = DiagnosticsPresentation.latencyRows(mac.latency)
    check(latency.first == .init(title: "Camera read", value: "p50 0.02 ms · p95 0.03 ms · max 0.03 ms"),
          "stages in pipeline order")
    check(latency.contains(.init(title: "Hailo inference", value: "no samples yet")),
          "a stage with no sample says so, never 0 ms")
    check(latency.contains(.init(title: "Behaviour reasoning", value: "p50 1.5 ms · p95 2.0 ms · max 2.3 ms")),
          "behaviour reasoning shown")
    check(latency.contains(.init(title: "Memory", value: "412 MB (peak)")), "a peak RSS is labelled peak")
    check(latency.contains(.init(title: "SoC temperature", value: "not available on this host")),
          "no SoC sensor on a Mac: said, not invented")
    check(latency.contains(.init(title: "Processed rate", value: "29.6 fps")), "processed fps")
    let evidence = DiagnosticsPresentation.flatten(mac.frame?.evidence)
    check(evidence.contains(.init(title: "Driver association › Method", value: "largest_face")),
          "driver association readable")
    check(evidence.contains(.init(title: "Hand phone › Phone confidence", value: "0.61")), "hand-phone confidence")
    check(evidence.contains(.init(title: "Events fired this frame", value: "none")), "an empty list reads none")
    let versions = DiagnosticsPresentation.flatten(mac.versions)
    check(versions.contains(.init(title: "Decoder", value: "legacy")), "decoder version")
    check(versions.contains(.init(title: "Mediapipe › Installed", value: "mediapipe-0.10.14")), "mediapipe pin")
    check(versions.contains(.init(title: "Detector", value: "not reported")), "a null detector is not reported, not blank")
    check(mac.unknownOrDegraded?.count == 2, "degraded reasons decode")
}

print("\na Pi with a detector:")
let piBody = """
{"schema": "sd-diagnostics/1", "platform": {"class": "pi", "board": "Raspberry Pi 5 Model B Rev 1.0"},
 "frame": {"detections": {"state": "ok", "fresh": true, "age_s": 0.03, "total": 3,
   "items": [{"label": "person", "confidence": 0.912, "box": [1, 2, 3, 4], "class_id": 0, "track_id": 4},
             {"label": "cell phone", "confidence": 0.4449, "box": [5, 6, 7, 8], "class_id": 67, "track_id": null}]},
   "open_conditions": []},
 "latency": {"stages": {"hailo_inference": {"n": 256, "p50_ms": 9.81, "p95_ms": 12.04, "max_ms": 131.0}},
             "processed_fps": null, "soc_temperature_c": 63.4, "throttle_now": []}}
""".data(using: .utf8)!
let pi = try? DeviceDiagnostics.decode(piBody)
check(pi != nil, "a partial body decodes (every field optional but the schema)")
if let pi {
    check(DiagnosticsPresentation.hostBanner(pi) == "Measured on a Raspberry Pi (Raspberry Pi 5 Model B Rev 1.0)", "Pi banner")
    let det = DiagnosticsPresentation.detectionRows(pi.frame)
    check(det.first == .init(title: "person", value: "91% · track 4"), "label, confidence and track")
    check(det.dropFirst().first == .init(title: "cell phone", value: "44%"), "no track id: none shown")
    check(det.last == .init(title: "More", value: "1 lower-confidence detections not shown"), "the bound is said")
    check(DiagnosticsPresentation.conditionRows(pi.frame) == [.init(title: "Open conditions", value: "none")], "none open")
    let latency = DiagnosticsPresentation.latencyRows(pi.latency)
    check(latency.contains(.init(title: "Hailo inference", value: "p50 9.8 ms · p95 12.0 ms · max 131 ms")), "Pi stage")
    check(latency.contains(.init(title: "Processed rate", value: "not measured yet")), "no fps yet")
    check(latency.contains(.init(title: "Throttling now", value: "none")), "throttle state")
    check(latency.contains(.init(title: "SoC temperature", value: "63.4 °C")), "temperature")
}

print("\nrefusals and the gate:")
check((try? DeviceDiagnostics.decode(#"{"schema": "sd-diagnostics/2"}"#.data(using: .utf8)!)) == nil,
      "another schema is refused, not misread")
check(!DiagnosticsAccess.available(reported: nil, token: "t"), "older firmware (no flag): never offered")
check(!DiagnosticsAccess.available(reported: DiagnosticsFlag(enabled: false, schema: nil), token: "t"), "switched off: not offered")
check(!DiagnosticsAccess.available(reported: DiagnosticsFlag(enabled: true, schema: nil), token: "  "), "no token: not offered")
check(DiagnosticsAccess.available(reported: DiagnosticsFlag(enabled: true, schema: "sd-diagnostics/1"), token: "abc"),
      "on + token: offered")
check(DiagnosticsAccess.problem(status: 404, hasToken: true).contains("developer_diagnostics"), "404 names the switch")
check(DiagnosticsAccess.problem(status: 403, hasToken: true).contains("dashboard.token"), "403 names the token config")
check(DiagnosticsAccess.problem(status: 401, hasToken: true).contains("rejected"), "401 with a token: rejected")
check(DiagnosticsAccess.problem(status: 401, hasToken: false).contains("Add it"), "401 without one: add it")
let flag = try? JSONDecoder().decode(DiagnosticsFlag.self, from: #"{"enabled": true, "schema": "sd-diagnostics/1"}"#.data(using: .utf8)!)
check(flag?.enabled == true, "the /api/health diagnostics flag decodes")

print("\n\(failures == 0 ? "PASSED" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)
