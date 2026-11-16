import Foundation

// Standalone, run by ios/Tests/run.sh (concatenated with Sources/LiveOverlay.swift).
//
// sd-overlay/1 (docs/LIVE_OVERLAY.md): the device sends a clean frame and a
// description of what its AI saw on that frame. What these pin: the document
// decodes as the contract says (the example from the doc), "could not look"
// (det null) never reads as "nothing there" (det []), a simulated source is
// badged, unknown fields are ignored, an oversized header is refused before it
// is decoded, and every word drawn from it -- the behaviour strip, the why
// line, the perf readout, the panel's state -- is the device's, never a guess.

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

/// base64url, no padding: exactly what core/overlay.py encode_header() sends.
func header(_ json: String) -> String {
    Data(json.utf8).base64EncodedString()
        .replacingOccurrences(of: "+", with: "-").replacingOccurrences(of: "/", with: "_")
        .replacingOccurrences(of: "=", with: "")
}

func decode(_ json: String) -> LiveOverlay? {
    if case .success(let overlay) = LiveOverlay.decode(header: header(json)) { return overlay }
    return nil
}

func points(_ n: Int, cx: Double, cy: Double, r: Double) -> String {
    (0..<n).map { i -> String in
        let a = Double(i) / Double(n) * 2 * Double.pi
        return String(format: "[%.4f,%.4f]", cx + r * cos(a), cy + r * sin(a))
    }.joined(separator: ",")
}

// The doc's example, with its elided contours filled in at the real sizes
// (oval 36 points, each eye 16).
let example = """
{"schema":"sd-overlay/1","module":"driver_attention","mode":"driver_safety",
 "analysed":true,"sim":false,"source":"camera","w":640,"h":480,"frame_id":1812,
 "det":[{"label":"person","box":[0.1,0.1,0.5,1.0],"track":3,"role":"driver"},
        {"label":"phone","box":[0.3125,0.4167,0.375,0.5417],"track":9,"role":"driver"}],
 "det_age_s":0.0,"det_fresh":true,
 "face":{"visible":true,"contours":{"oval":[\(points(36, cx: 0.3, cy: 0.3, r: 0.1))],
         "eye_l":[\(points(16, cx: 0.33, cy: 0.28, r: 0.02))],"eye_r":[\(points(16, cx: 0.27, cy: 0.28, r: 0.02))]},
         "eyes_closed":false,
         "head_pose":{"yaw_deg":34.2,"pitch_deg":-3.1,"calibrated":true,"direction":"Left","kind":"head_pose"}},
 "hands":[[0.2969,0.3958,0.3906,0.5625]],
 "driver":{"box":[0.1,0.1,0.5,1.0],"basis":"configured_seat","identity":"driver","people":1},
 "driver_region":{"box":[0.05,0.0,0.55,1.0],"source":"configured"},
 "behaviours":[{"type":"distraction","phase":"confirmed","for_s":3.4}],
 "state":"DISTRACTED","alert_active":true,
 "last_event":{"type":"distraction","signals":"head turned away 3.4 s","severity":"loud","age_s":0.4},
 "unavailable":[],
 "perf":{"fps":14.0,"object_detector_ms":11.8,"face_landmarks_ms":9.4,"hand_landmarks_ms":7.0,"frame_age_ms":83}}
"""

print("== the documented example decodes as the contract says")
if let o = decode(example) {
    check(o.analysed && !o.sim && o.source == "camera", "analysed, real camera")
    check(o.width == 640 && o.height == 480 && o.frameID == 1812, "frame size and id")
    check(o.module == "driver_attention" && o.mode == "driver_safety", "module and mode")
    check(o.detections?.count == 2, "two boxes")
    let person = o.detections?.first
    check(person?.label == "person" && person?.track == 3 && person?.role == .driver, "the driver, tracked as #3")
    check(person?.box == LiveOverlay.Box(x1: 0.1, y1: 0.1, x2: 0.5, y2: 1.0), "normalised box kept exactly")
    check(o.detections?.last?.label == "phone" && o.detections?.last?.role == .driver, "the driver's phone")
    check(o.detFresh == true && !o.boxesAreCarried, "fresh boxes are not faded")
    check(o.face?.visible == true && o.face?.eyesClosed == false, "face visible, eyes open")
    check(o.face?.contours["oval"]?.count == 36 && o.face?.contours["eye_l"]?.count == 16, "contour sizes")
    check(o.drawableContour("oval")?.count == 36 && o.drawableContour("eye_r")?.count == 16, "real contours are drawn whole")
    check(o.face?.headPose?.direction == "Left" && o.face?.headPose?.yawDeg == 34.2, "head pose")
    check(o.hands?.count == 1, "one hand box")
    check(o.driver?.basis == "configured_seat" && o.driverRegion?.source == "configured", "driver and seat region")
    check(o.behaviours == [LiveOverlay.Behaviour(type: "distraction", phase: .confirmed, forS: 3.4)], "behaviour phase")
    check(o.state == "DISTRACTED" && o.alertActive, "state and alert")
    check(o.lastEvent?.signals == "head turned away 3.4 s", "the device's own words")
    check(o.unavailable.isEmpty, "nothing unavailable")
    check(o.perf?.fps == 14.0 && o.perf?.componentMs["object_detector"] == 11.8
          && o.perf?.componentMs["face_landmarks"] == 9.4 && o.perf?.frameAgeMs == 83, "measured perf")
    check(o.perf?.componentMs["frame_age"] == nil, "frame age is not a component")

    let strip = BehaviourStripItem.make(o)
    check(strip == [BehaviourStripItem(type: "distraction", text: "Head turned away · 3.4 s", tone: .confirmed)],
          "strip: \(strip.map(\.text))")
    let why = OverlayWhyLine.make(o)
    check(why?.text == "Distraction: head turned away 3.4 s" && why?.isAlert == true, "why: \(why?.text ?? "nil")")
    check(why?.ageText == "just now", "a 0.4 s old event is just now")
    check(OverlayPerfReadout.text(o.perf) == "14 fps · Detector 11.8 ms · Frame 83 ms",
          "perf: \(OverlayPerfReadout.text(o.perf) ?? "nil")")
    check(OverlayPerfReadout.components(o.perf) == ["Detector 11.8 ms", "Face 9.4 ms", "Hands 7.0 ms"], "components")
    let state = LiveAnalysisState.make(o, overlayCapable: true, live: true)
    check(state.kind == .alert && state.headline == "Head turned away · 3.4 s", "panel: \(state.headline)")
    check(HeadIndicator.make(o) == HeadIndicator(text: "Head · Left · pitch \u{2212}3°", yawDeg: 34.2, calibrated: true,
                                                 direction: "Left", pitchDeg: -3.1),
          "head indicator: \(HeadIndicator.make(o)?.text ?? "nil")")
    check(!SimulatedBadge.shows(current: o, streamSeenSimulated: false), "a real camera has no Simulated badge")
    let panel = DriverPanelModel.make(o, overlayCapable: true, live: true)
    check(!panel.eyesClosed && panel.unavailable == nil, "no eyes-closed chip, nothing unavailable")
    check(!panel.spoken.lowercased().contains("gaze") && !panel.spoken.contains("looking at"), "never gaze")
    check(!panel.spoken.contains("%"), "no percentages")
} else {
    check(false, "the example decodes")
}

print("== null vs empty detections")
let base = #""schema":"sd-overlay/1","analysed":true,"w":640,"h":480,"face":{"visible":true}"#
if let none = decode("{\(base),\"det\":null}"), let empty = decode("{\(base),\"det\":[]}"),
   let absent = decode("{\(base)}") {
    check(none.detections == nil, "det null: could not look")
    check(empty.detections == [], "det []: looked, nothing there")
    check(absent.detections == nil, "det absent: not said, never 'nothing there'")
    check(LiveAnalysisState.make(none, overlayCapable: true, live: true).headline == "Detection unavailable",
          "null says detection unavailable")
    check(LiveAnalysisState.make(empty, overlayCapable: true, live: true).kind != .detectionUnavailable,
          "an empty scene is not 'unavailable'")
} else {
    check(false, "null/empty/absent decode")
}

print("== not analysed, no overlay, not live, unsupported")
if let notAnalysed = decode(#"{"schema":"sd-overlay/1","analysed":false,"w":0,"h":0,"det":null,"face":null,"unavailable":["perception"]}"#) {
    let s = LiveAnalysisState.make(notAnalysed, overlayCapable: true, live: true)
    check(s.kind == .notAnalysed && s.headline == "Not analysed", "analysed false: Not analysed")
    check(BehaviourStripItem.make(notAnalysed).isEmpty, "nothing drawn from an unanalysed frame")
    let panel = DriverPanelModel.make(notAnalysed, overlayCapable: true, live: true)
    check(panel.unavailable == nil && panel.why == nil && panel.head == nil, "no claims from it")
}
check(LiveAnalysisState.make(nil, overlayCapable: true, live: true).kind == .notAnalysed,
      "a frame without a document (camera fallback) is Not analysed")
check(LiveAnalysisState.make(nil, overlayCapable: false, live: true).kind == .unsupported,
      "a camera that doesn't send the analysis says so")
check(LiveAnalysisState.make(decode(example), overlayCapable: true, live: false).kind == .waiting,
      "nothing is claimed from a picture that isn't live")
check(DriverPanelModel.make(decode(example), overlayCapable: true, live: false).behaviours.isEmpty,
      "no behaviour strip over a stalled picture")

print("== face states and insufficient evidence")
if let hidden = decode(#"{"schema":"sd-overlay/1","analysed":true,"det":[],"face":{"visible":false},"behaviours":[{"type":"face_not_visible","phase":"confirmed","for_s":null}]}"#) {
    let s = LiveAnalysisState.make(hidden, overlayCapable: true, live: true)
    check(s.kind == .faceNotVisible, "face not visible: \(s.headline)")
    check(BehaviourStripItem.make(hidden).first?.tone == .neutral, "no face is never the alert colour")
    check(BehaviourStripItem.make(hidden).first?.text == "Face not visible", "no duration when the device gives none")
}
if let notRun = decode(#"{"schema":"sd-overlay/1","analysed":true,"det":[],"face":null,"behaviours":[]}"#) {
    check(LiveAnalysisState.make(notRun, overlayCapable: true, live: true).kind == .insufficientEvidence,
          "face model not run: insufficient evidence, never 'watching'")
}
if let calibrating = decode(#"{"schema":"sd-overlay/1","analysed":true,"det":[],"face":{"visible":true,"head_pose":{"yaw_deg":2,"calibrated":false,"direction":"Forward"}},"behaviours":[]}"#) {
    check(LiveAnalysisState.make(calibrating, overlayCapable: true, live: true).headline == "Calibrating to the driver",
          "uncalibrated head pose says so")
}

print("== eyes closed only when true")
let eyes = { (value: String) -> Bool in
    DriverPanelModel.make(decode(#"{"schema":"sd-overlay/1","analysed":true,"det":[],"face":{"visible":true\#(value)}}"#),
                          overlayCapable: true, live: true).eyesClosed
}
check(eyes(#","eyes_closed":true"#), "eyes_closed true: the chip")
check(!eyes(#","eyes_closed":false"#), "eyes_closed false: no chip")
check(!eyes(""), "eyes_closed absent (no calibrated threshold): no chip")

print("== the Simulated badge")
if let sim = decode(#"{"schema":"sd-overlay/1","analysed":true,"sim":true,"source":"sim"}"#),
   let replay = decode(#"{"schema":"sd-overlay/1","analysed":true,"source":"replay"}"#),
   let real = decode(#"{"schema":"sd-overlay/1","analysed":true,"sim":false,"source":"camera"}"#) {
    check(SimulatedBadge.shows(current: sim, streamSeenSimulated: false), "sim: true")
    check(SimulatedBadge.shows(current: replay, streamSeenSimulated: false), "a replay source is simulated too")
    check(!SimulatedBadge.shows(current: real, streamSeenSimulated: false), "a real camera is not")
    check(SimulatedBadge.shows(current: nil, streamSeenSimulated: true),
          "a frame with no document on a simulated stream keeps the badge")
    check(!SimulatedBadge.shows(current: nil, streamSeenSimulated: false), "nothing known: no badge")
}

print("== tolerant: unknown fields, wrong types, junk entries")
let tolerant = #"""
{"schema":"sd-overlay/1","analysed":true,"future_field":{"x":[1,2,3]},"w":"640","h":480,
 "det":[{"label":"person","box":[0.1,0.2,0.3]},{"box":[1.4,-0.2,0.5,0.6],"track":"x"},"junk",
        {"label":"dog","box":[0.9,0.9,0.1,0.1],"confidence":0.97,"role":"pilot"}],
 "face":{"visible":true,"contours":{"oval":[[0.1,0.2],"bad",[0.3]],"eye_l":"nope"}},
 "behaviours":[{"type":"distraction"},{"phase":"developing"},{"type":"new_thing","phase":"developing","for_s":12.7},
               {"type":"eyes_down","phase":"warming","for_s":1}],
 "perf":{"fps":true,"object_detector_ms":"fast","new_model_ms":4.25},
 "unavailable":["object_detector",7],"alert_active":"yes"}
"""#
if let o = decode(tolerant) {
    check(o.width == 0 && o.height == 480, "a string width is not a number")
    check(o.detections?.count == 2, "malformed boxes dropped, others kept: \(o.detections?.count ?? -1)")
    check(o.detections?.first?.box == LiveOverlay.Box(x1: 0.5, y1: 0, x2: 1, y2: 0.6) && o.detections?.first?.label == "object",
          "clamped, ordered, unnamed")
    check(o.detections?.first?.track == nil, "a non-numeric track is not a track")
    check(o.detections?.last?.role == nil, "an unknown role is no role")
    check(o.face?.contours["oval"]?.count == 1 && o.face?.contours["eye_l"] == nil, "junk points dropped")
    check(o.drawableContour("oval") == nil, "one point is not an outline")
    check(o.behaviours?.count == 2, "behaviours without a type or phase dropped")
    let strip = BehaviourStripItem.make(o)
    check(strip.map(\.text) == ["New thing · 12 s", "Head down · 1.0 s"], "unknown type readable: \(strip.map(\.text))")
    check(strip.last?.tone == .neutral, "an unknown phase is neutral")
    check(o.perf?.fps == nil, "a Bool is not an fps")
    check(o.perf?.componentMs == ["new_model": 4.25], "unknown components kept by name")
    check(o.unavailable == ["object_detector"], "non-string entries dropped")
    check(!o.alertActive, "\"yes\" is not true")
    check(DriverPanelModel.unavailableText(o.unavailable) == "Not working: Object detection", "unavailable in words")
}

print("== refused: oversized, not base64, not JSON, another schema")
let big = String(repeating: "A", count: LiveOverlay.maxHeaderBytes + 1)
if case .failure(.tooLarge(let n)) = LiveOverlay.decode(header: big) {
    check(n == LiveOverlay.maxHeaderBytes + 1, "oversized refused before decoding")
} else { check(false, "oversized refused") }
check(LiveOverlay.decode(header: "!!!") == .failure(.notBase64), "not base64")
check(LiveOverlay.decode(header: header("[1,2]")) == .failure(.notJSON), "a JSON list is not a document")
check(LiveOverlay.decode(header: header("not json")) == .failure(.notJSON), "not JSON")
check(LiveOverlay.decode(header: header(#"{"schema":"sd-overlay/2","analysed":true}"#))
      == .failure(.unsupportedSchema("sd-overlay/2")), "a future schema is not misread")
check(LiveOverlay.decode(header: header(#"{"analysed":true}"#)) == .failure(.unsupportedSchema(nil)), "no schema")
check(LiveOverlay.base64URLDecode("eyJhIjoxfQ") == Data(#"{"a":1}"#.utf8), "unpadded base64url")
check(LiveOverlay.base64URLDecode("eyJhIjoxfQ==") == Data(#"{"a":1}"#.utf8), "padded is fine too")
check(decode(#"{"schema":"sd-overlay/1","analysed":true,"last_event":{"type":"distraction","signals":"\#(String(repeating: "x", count: 1000))"}}"#)?
        .lastEvent?.signals?.count == LiveOverlay.maxText, "long text bounded")

print("== the virtual device's scripted face mesh")
// What /stream sent from sim.virtual_device (m13 probe): the scripted mesh
// places a few points and parks the rest at (0.5, 0.5).
let parked = Array(repeating: "[0.5,0.5]", count: 35).joined(separator: ",")
let simEye = "[0.522,0.43],[0.5,0.5],[0.5,0.5],[0.535,0.443],[0.5,0.5],[0.557,0.443],[0.5,0.5],[0.5,0.5],[0.569,0.43],[0.5,0.5],[0.5,0.5],[0.557,0.418],[0.5,0.5],[0.535,0.418],[0.5,0.5],[0.5,0.5]"
if let o = decode(#"{"schema":"sd-overlay/1","analysed":true,"sim":true,"face":{"visible":true,"contours":{"oval":[\#(parked),[0.5,0.638]],"eye_l":[\#(simEye)]},"head_pose":{"yaw_deg":-0.0,"pitch_deg":180.0,"calibrated":true,"direction":"Forward"}}}"#) {
    check(o.drawableContour("oval") == nil, "a parked oval is not drawn")
    check(o.drawableContour("eye_l")?.count == 6, "the six real eye points are: \(o.drawableContour("eye_l")?.count ?? -1)")
    check(HeadIndicator.make(o)?.text == "Head · Forward", "head direction from the device")
}

print("== durations and readouts")
check(DriverBehaviourCopy.duration(2.14) == "2.1 s", "2.1 s")
check(DriverBehaviourCopy.duration(12.9) == "12 s", "whole seconds from 10 s")
check(DriverBehaviourCopy.duration(65) == "1 min 5 s", "minutes")
check(DriverBehaviourCopy.duration(120) == "2 min", "whole minutes")
check(DriverBehaviourCopy.duration(-1) == "0.0 s", "never negative")
check(DriverBehaviourCopy.label("eyes_down") == "Head down", "eyes_down is about the head")
check(OverlayPerfReadout.text(LiveOverlay.Perf(fps: 7.46, componentMs: [:], frameAgeMs: nil)) == "7.5 fps", "low fps keeps a decimal")
check(OverlayPerfReadout.text(LiveOverlay.Perf(fps: nil, componentMs: ["object_detector": 142.6], frameAgeMs: 1203.4))
      == "Detector 143 ms · Frame 1203 ms", "large numbers whole")
check(OverlayPerfReadout.text(LiveOverlay.Perf()) == nil, "nothing measured: no readout")
check(OverlayPerfReadout.text(nil) == nil, "no perf: no readout")
check(HeadIndicator.make(decode(#"{"schema":"sd-overlay/1","analysed":true,"face":{"visible":true,"head_pose":{"yaw_deg":178,"direction":"Left"}}}"#))?.yawDeg == nil,
      "an implausible yaw draws no needle")

print("== strip order and why-line severity")
if let o = decode(#"{"schema":"sd-overlay/1","analysed":true,"behaviours":[{"type":"phone_use","phase":"developing","for_s":1.2},{"type":"eyes_down","phase":"developing","for_s":2.5},{"type":"drowsiness","phase":"confirmed","for_s":0.5}],"last_event":{"type":"monitoring_ready","signals":"all monitoring policies operational","severity":"log","age_s":9}}"#) {
    check(BehaviourStripItem.make(o).map(\.type) == ["drowsiness", "eyes_down", "phone_use"], "confirmed first, then longest")
    check(BehaviourStripItem.make(o, limit: 2).count == 2, "bounded")
    check(OverlayWhyLine.make(o) == nil, "a log record is not a why line")
    let s = LiveAnalysisState.make(o, overlayCapable: true, live: true)
    check(s.kind == .alert && s.headline == "Eyes closed · 0.5 s", "confirmed drowsiness leads")
}
if let dev = decode(#"{"schema":"sd-overlay/1","analysed":true,"det":[],"face":{"visible":true},"behaviours":[{"type":"distraction","phase":"developing","for_s":2.1}]}"#) {
    let s = LiveAnalysisState.make(dev, overlayCapable: true, live: true)
    check(s.kind == .behaviour && s.headline == "Head turned away · 2.1 s" && s.detail == "Developing — not yet an alert",
          "developing: \(s.headline)")
    check(BehaviourStripItem.make(dev).first?.spoken == "Head turned away · 2.1 s, developing", "phase spoken")
}

print("== chrome throttling")
let summary = { (json: String) in
    LiveOverlaySummary.make(decode(json), overlayCapable: true, live: true, streamSeenSimulated: false)
}
let a = summary(#"{"schema":"sd-overlay/1","analysed":true,"det":[],"face":{"visible":true},"behaviours":[],"perf":{"fps":14,"frame_age_ms":80}}"#)
let b = summary(#"{"schema":"sd-overlay/1","analysed":true,"det":[],"face":{"visible":true},"behaviours":[],"perf":{"fps":14,"frame_age_ms":95}}"#)
let c = summary(#"{"schema":"sd-overlay/1","analysed":true,"det":[],"face":{"visible":true},"behaviours":[{"type":"distraction","phase":"developing","for_s":0.1}],"perf":{"fps":14,"frame_age_ms":95}}"#)
let d = summary(#"{"schema":"sd-overlay/1","analysed":true,"det":[],"face":{"visible":true},"behaviours":[{"type":"distraction","phase":"developing","for_s":0.2}],"perf":{"fps":14,"frame_age_ms":95}}"#)
check(LiveOverlaySummary.shouldPublish(old: nil, new: a, sinceLast: 0), "first summary published")
check(!LiveOverlaySummary.shouldPublish(old: a, new: a, sinceLast: 5), "unchanged: nothing")
check(!LiveOverlaySummary.shouldPublish(old: a, new: b, sinceLast: 0.3), "a frame-age digit waits")
check(LiveOverlaySummary.shouldPublish(old: a, new: b, sinceLast: 1.0), "perf once a second")
check(LiveOverlaySummary.shouldPublish(old: b, new: c, sinceLast: 0.01), "a behaviour starting is immediate")
check(!LiveOverlaySummary.shouldPublish(old: c, new: d, sinceLast: 0.05), "a ticking duration waits")
check(LiveOverlaySummary.shouldPublish(old: c, new: d, sinceLast: 0.25), "and then shows")

print("== alert banner words and an ageing why line")
let freshWhy = OverlayWhyLine(title: "Distraction", because: "head turned away 3.4 s", ageText: "just now",
                              isAlert: true, ageS: 0.4)
let oldWhy = OverlayWhyLine(title: "Distraction", because: "head turned away 3.4 s", ageText: "22 s ago",
                            isAlert: true, ageS: 22)
check(OverlayWhyLine.bannerText(conditionHeadline: "Head turned away · 3.4 s", why: freshWhy)
      == "Head turned away · 3.4 s — Distraction: head turned away 3.4 s", "a fresh why line is the reason")
check(OverlayWhyLine.bannerText(conditionHeadline: "Head down · 3.2 s", why: oldWhy) == "Head down · 3.2 s",
      "an older event is never given as the reason for a new condition")
let w1 = summary(#"{"schema":"sd-overlay/1","analysed":true,"det":[],"face":{"visible":true},"last_event":{"type":"distraction","signals":"x","severity":"loud","age_s":1.0}}"#)
let w2 = summary(#"{"schema":"sd-overlay/1","analysed":true,"det":[],"face":{"visible":true},"last_event":{"type":"distraction","signals":"x","severity":"loud","age_s":1.1}}"#)
check(w1.key == w2.key, "an event growing older is not a change of meaning")
check(!LiveOverlaySummary.shouldPublish(old: w1, new: w2, sinceLast: 0.05), "so it is throttled like a tick")

print("== M13.1: head pitch in the Head chip, never gaze")
func head(_ pose: String) -> HeadIndicator? {
    HeadIndicator.make(decode(#"{"schema":"sd-overlay/1","analysed":true,"face":{"visible":true,"head_pose":\#(pose)}}"#))
}
check(head(#"{"yaw_deg":2,"pitch_deg":4.4,"calibrated":true,"direction":"Forward"}"#)?.text == "Head · Forward · pitch +4°",
      "a positive pitch is signed")
check(head(#"{"yaw_deg":2,"pitch_deg":-0.4,"calibrated":false,"direction":"Forward"}"#)?.text == "Head · Forward · pitch 0°",
      "an uncalibrated wrapped pitch near 0 reads 0°")
check(head(#"{"yaw_deg":2,"pitch_deg":-90,"direction":"Down"}"#)?.pitchDeg == -90, "the ±90 edge is kept")
check(head(#"{"yaw_deg":2,"pitch_deg":-178.5,"direction":"Forward"}"#)?.text == "Head · Forward",
      "an unwrapped raw pitch (older firmware) is not shown")
check(head(#"{"yaw_deg":2,"direction":"Left"}"#)?.text == "Head · Left", "no pitch: the direction alone")
check(head(#"{"yaw_deg":2,"pitch_deg":true,"direction":"Left"}"#)?.pitchDeg == nil, "a bool is not a pitch")
for pose in [#"{"pitch_deg":12,"direction":"Up"}"#, #"{"pitch_deg":-12,"direction":"Down"}"#] {
    let text = head(pose)?.text.lowercased() ?? ""
    check(text.hasPrefix("head") && !text.contains("gaze") && !text.contains("looking"), "Head, never gaze: \(text)")
}
let p1 = summary(#"{"schema":"sd-overlay/1","analysed":true,"det":[],"face":{"visible":true,"head_pose":{"yaw_deg":1,"pitch_deg":3.0,"direction":"Forward"}}}"#)
let p2 = summary(#"{"schema":"sd-overlay/1","analysed":true,"det":[],"face":{"visible":true,"head_pose":{"yaw_deg":1,"pitch_deg":7.0,"direction":"Forward"}}}"#)
let p3 = summary(#"{"schema":"sd-overlay/1","analysed":true,"det":[],"face":{"visible":true,"head_pose":{"yaw_deg":1,"pitch_deg":7.0,"direction":"Up"}}}"#)
check(p1.key == p2.key && p1 != p2, "a pitch digit moving is a tick, not a change of meaning")
check(!LiveOverlaySummary.shouldPublish(old: p1, new: p2, sinceLast: 0.05), "so it is throttled")
check(p2.key != p3.key, "the device's direction word changing is news")

print("== M13.1: last_event.event_key ties the why line to its history row")
let keyed = decode(#"{"schema":"sd-overlay/1","analysed":true,"last_event":{"type":"distraction","signals":"head turned away 3.4 s","severity":"loud","age_s":0.4,"event_key":"evt-7f3a"}}"#)
check(keyed?.lastEvent?.eventKey == "evt-7f3a", "event_key decoded")
check(OverlayWhyLine.make(keyed)?.eventKey == "evt-7f3a", "and carried by the why line")
let unkeyed = decode(#"{"schema":"sd-overlay/1","analysed":true,"last_event":{"type":"distraction","severity":"loud","event_key":""}}"#)
check(unkeyed?.lastEvent?.eventKey == nil, "an empty key is no key")
let longKey = String(repeating: "k", count: 65)
check(decode(#"{"schema":"sd-overlay/1","last_event":{"type":"distraction","severity":"loud","event_key":"\#(longKey)"}}"#)?
        .lastEvent?.eventKey == nil, "a key longer than the device ever sends (64) is refused")
check(decode(#"{"schema":"sd-overlay/1","last_event":{"type":"distraction","severity":"loud","event_key":7}}"#)?
        .lastEvent?.eventKey == nil, "a non-string key is ignored")

print(failures == 0 ? "\nall passed" : "\n\(failures) FAILED")
exit(failures == 0 ? 0 : 1)
