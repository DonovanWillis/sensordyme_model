import Foundation

// MARK: - sd-overlay/1: what the AI saw on one frame (docs/LIVE_OVERLAY.md)
//
// The device sends the CLEAN camera frame and, in the same multipart part, an
// `X-SD-Overlay` header: base64url (no padding) of compact JSON describing what
// its perception saw on THAT frame (core/overlay.py). This file decodes it into
// a typed value and decides, as pure functions, every word the app draws from
// it -- the behaviour strip, the "why" line, the performance readout, the
// driver panel's state and the Simulated badge -- so ios/Tests/run.sh pins them.
//
// Foundation-only on purpose (LiveOverlayTests). The drawing lives in
// AIViewOverlay.swift.
//
// Truthfulness rules this file enforces (the device's own, restated):
//  * `det: null` means "could not look" and is never shown as an empty scene;
//    `det: []` means "looked, found nothing".
//  * No confidence numbers: the document carries none, and nothing here
//    invents one.
//  * Head orientation is "Head", never gaze or "looking at".
//  * A simulated or replayed source is said on every frame.
//  * An unknown field is ignored; a missing one is "not said", never a default
//    that claims something.

struct LiveOverlay: Equatable {
    static let schema = "sd-overlay/1"
    /// The header is refused above this many bytes (the device's own test bounds
    /// it below 4 KB; four times that is a document from something else).
    static let maxHeaderBytes = 16 * 1024
    /// Most of each list kept; anything beyond is dropped, never drawn.
    static let maxBoxes = 48
    static let maxHands = 8
    static let maxBehaviours = 12
    static let maxContourPoints = 96
    static let maxText = 240

    /// A normalised rectangle, 0-1 of the analysed frame, corners ordered.
    struct Box: Equatable {
        var x1: Double, y1: Double, x2: Double, y2: Double
        var width: Double { x2 - x1 }
        var height: Double { y2 - y1 }

        /// From `[x1, y1, x2, y2]`; nil unless four finite numbers. Clamped to
        /// the frame and ordered, so a box can never be drawn off the picture.
        init?(_ value: Any?) {
            guard let list = value as? [Any], list.count == 4 else { return nil }
            let numbers = list.compactMap { LiveOverlay.number($0) }
            guard numbers.count == 4 else { return nil }
            let c = numbers.map { min(1, max(0, $0)) }
            x1 = min(c[0], c[2]); x2 = max(c[0], c[2])
            y1 = min(c[1], c[3]); y2 = max(c[1], c[3])
        }

        init(x1: Double, y1: Double, x2: Double, y2: Double) {
            self.x1 = x1; self.y1 = y1; self.x2 = x2; self.y2 = y2
        }
    }

    struct Point: Equatable, Hashable {
        var x: Double, y: Double
    }

    enum Role: String, Equatable {
        case driver, other
    }

    struct Detection: Equatable {
        var label: String
        var box: Box
        var track: Int?
        /// `driver` / `other` when the module decided one; nil otherwise.
        var role: Role?
    }

    struct HeadPose: Equatable {
        var yawDeg: Double?
        var pitchDeg: Double?
        var calibrated: Bool?
        /// The device's own classification: Left / Right / Up / Down / Forward
        /// (frame-relative; core/overlay.py, modules/driver_attention.py).
        var direction: String?
    }

    struct Face: Equatable {
        /// False: the face model looked and found no face.
        var visible: Bool
        /// FaceMesh outline points in drawing order, by name (oval, eye_l, eye_r).
        var contours: [String: [Point]] = [:]
        /// Only when the device has a calibrated threshold; absent otherwise.
        var eyesClosed: Bool?
        var headPose: HeadPose?
    }

    struct Driver: Equatable {
        var box: Box?
        /// How the driver was associated ("configured_seat", "calibrated_seat_region" ...).
        var basis: String?
        var identity: String?
        var people: Int?
    }

    struct Region: Equatable {
        var box: Box
        var source: String?
    }

    struct Behaviour: Equatable {
        enum Phase: Equatable {
            /// The condition holds, not yet long enough to alert.
            case developing
            /// The module's alert condition.
            case confirmed
            case other(String)
        }
        var type: String
        var phase: Phase
        /// How long, as the device measured it; nil when it does not say.
        var forS: Double?
    }

    struct LastEvent: Equatable {
        var type: String
        /// The device's own words (`contributing_signals`).
        var signals: String?
        var severity: String?
        var ageS: Double?
    }

    struct Perf: Equatable {
        /// Processed frames per second, measured on the device.
        var fps: Double?
        /// `<component>_ms`: each model's own run-time EMA, by component name.
        var componentMs: [String: Double] = [:]
        /// Capture to overlay build, on the device's clock.
        var frameAgeMs: Double?
    }

    var module: String?
    var mode: String?
    var analysed: Bool
    var sim: Bool
    var source: String?
    var width: Int
    var height: Int
    var frameID: Int?
    /// nil: detection did not run or is unavailable ("could not look").
    /// []: it looked and found nothing.
    var detections: [Detection]?
    var detAgeS: Double?
    /// False: the boxes are the detector's last run, carried to this frame.
    var detFresh: Bool?
    /// nil: the face model was not run on this frame.
    var face: Face?
    /// nil: the hand model did not run on this frame (it is scheduled).
    var hands: [Box]?
    var driver: Driver?
    var driverRegion: Region?
    /// nil: this module reports no behaviours at all; []: none in progress.
    var behaviours: [Behaviour]?
    var state: String?
    var alertActive: Bool
    var lastEvent: LastEvent?
    var unavailable: [String]
    var perf: Perf?

    // MARK: Decoding

    enum DecodeError: Error, Equatable {
        case tooLarge(Int)
        case notBase64
        case notJSON
        case unsupportedSchema(String?)
    }

    /// Decode an `X-SD-Overlay` header value. Refuses an oversized header
    /// before decoding anything, and a document of another schema; tolerates
    /// unknown and missing fields.
    static func decode(header: String) -> Result<LiveOverlay, DecodeError> {
        let trimmed = header.trimmingCharacters(in: .whitespaces)
        guard trimmed.utf8.count <= maxHeaderBytes else { return .failure(.tooLarge(trimmed.utf8.count)) }
        guard let data = base64URLDecode(trimmed) else { return .failure(.notBase64) }
        guard let object = try? JSONSerialization.jsonObject(with: data),
              let root = object as? [String: Any] else { return .failure(.notJSON) }
        let schema = root["schema"] as? String
        guard schema == Self.schema else { return .failure(.unsupportedSchema(schema)) }
        return .success(LiveOverlay(root: root))
    }

    /// base64url without padding (RFC 4648 §5); standard base64 is accepted too.
    static func base64URLDecode(_ text: String) -> Data? {
        var s = text.replacingOccurrences(of: "-", with: "+").replacingOccurrences(of: "_", with: "/")
        guard !s.isEmpty else { return nil }
        let remainder = s.count % 4
        if remainder == 1 { return nil }
        if remainder > 0 { s += String(repeating: "=", count: 4 - remainder) }
        return Data(base64Encoded: s)
    }

    init(root: [String: Any]) {
        module = Self.text(root["module"])
        mode = Self.text(root["mode"])
        // Absent is "not shown to be analysed", never assumed true.
        analysed = root["analysed"] as? Bool ?? false
        source = Self.text(root["source"])
        sim = (root["sim"] as? Bool ?? false) || source == "sim" || source == "replay"
        width = Self.number(root["w"]).map { Int(max(0, $0)) } ?? 0
        height = Self.number(root["h"]).map { Int(max(0, $0)) } ?? 0
        frameID = Self.number(root["frame_id"]).map { Int($0) }

        if let list = root["det"] as? [Any] {
            detections = list.prefix(Self.maxBoxes).compactMap { item in
                guard let d = item as? [String: Any], let box = Box(d["box"]) else { return nil }
                return Detection(label: Self.text(d["label"]) ?? "object", box: box,
                                 track: Self.number(d["track"]).map { Int($0) },
                                 role: Self.text(d["role"]).flatMap(Role.init(rawValue:)))
            }
        } else {
            detections = nil  // null, absent or not a list: could not look
        }
        detAgeS = Self.number(root["det_age_s"])
        detFresh = root["det_fresh"] as? Bool

        if let f = root["face"] as? [String: Any] {
            var face = Face(visible: f["visible"] as? Bool ?? false)
            if let contours = f["contours"] as? [String: Any] {
                for (name, value) in contours {
                    guard let points = value as? [Any] else { continue }
                    face.contours[name] = points.prefix(Self.maxContourPoints).compactMap { p in
                        guard let pair = p as? [Any], pair.count == 2,
                              let x = Self.number(pair[0]), let y = Self.number(pair[1]) else { return nil }
                        return Point(x: min(1, max(0, x)), y: min(1, max(0, y)))
                    }
                }
            }
            face.eyesClosed = f["eyes_closed"] as? Bool
            if let h = f["head_pose"] as? [String: Any] {
                face.headPose = HeadPose(yawDeg: Self.number(h["yaw_deg"]), pitchDeg: Self.number(h["pitch_deg"]),
                                         calibrated: h["calibrated"] as? Bool, direction: Self.text(h["direction"]))
            }
            self.face = face
        } else {
            face = nil
        }

        hands = (root["hands"] as? [Any]).map { $0.prefix(Self.maxHands).compactMap { Box($0) } }

        if let d = root["driver"] as? [String: Any] {
            driver = Driver(box: Box(d["box"]), basis: Self.text(d["basis"]), identity: Self.text(d["identity"]),
                            people: Self.number(d["people"]).map { Int($0) })
        }
        if let r = root["driver_region"] as? [String: Any], let box = Box(r["box"]) {
            driverRegion = Region(box: box, source: Self.text(r["source"]))
        }

        behaviours = (root["behaviours"] as? [Any]).map { list in
            list.prefix(Self.maxBehaviours).compactMap { item in
                guard let b = item as? [String: Any], let type = Self.text(b["type"]) else { return nil }
                let phase: Behaviour.Phase
                switch Self.text(b["phase"]) {
                case "developing": phase = .developing
                case "confirmed": phase = .confirmed
                case let other?: phase = .other(other)
                case nil: return nil
                }
                return Behaviour(type: type, phase: phase, forS: Self.number(b["for_s"]).map { max(0, $0) })
            }
        }
        state = Self.text(root["state"])
        alertActive = root["alert_active"] as? Bool ?? false
        if let e = root["last_event"] as? [String: Any], let type = Self.text(e["type"]) {
            lastEvent = LastEvent(type: type, signals: Self.text(e["signals"]), severity: Self.text(e["severity"]),
                                  ageS: Self.number(e["age_s"]))
        }
        unavailable = (root["unavailable"] as? [Any])?.compactMap { Self.text($0) } ?? []
        if let p = root["perf"] as? [String: Any] {
            var perf = Perf(fps: Self.number(p["fps"]), frameAgeMs: Self.number(p["frame_age_ms"]))
            for (key, value) in p where key.hasSuffix("_ms") && key != "frame_age_ms" {
                if let ms = Self.number(value) { perf.componentMs[String(key.dropLast(3))] = ms }
            }
            self.perf = perf
        }
    }

    /// A finite JSON number (never a Bool: JSONSerialization bridges true to 1).
    static func number(_ value: Any?) -> Double? {
        guard let n = value as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID() else { return nil }
        let d = n.doubleValue
        return d.isFinite ? d : nil
    }

    /// A non-empty string, trimmed to `maxText` characters.
    static func text(_ value: Any?) -> String? {
        guard let s = value as? String else { return nil }
        let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        return t.isEmpty ? nil : String(t.prefix(maxText))
    }

    // MARK: Questions the views ask

    /// The Simulated badge: this frame's own document says so.
    var isSimulated: Bool { sim }

    /// Whether a component the overlay depends on is reported not working.
    func isUnavailable(_ component: String) -> Bool { unavailable.contains(component) }

    /// The boxes are the detector's latest run carried to this frame: drawn faded.
    var boxesAreCarried: Bool { detFresh == false }

    /// A contour fit to draw, or nil. A FaceMesh point repeated three or more
    /// times in one contour is not a measured landmark (a stand-in that only
    /// places some points, e.g. the virtual device's scripted mesh, parks the
    /// rest on one spot); those are dropped, and a contour left with too few
    /// points to be an outline is not drawn at all -- never completed by guessing.
    func drawableContour(_ name: String) -> [Point]? {
        guard let face, face.visible, let points = face.contours[name] else { return nil }
        var counts: [Point: Int] = [:]
        for p in points { counts[p, default: 0] += 1 }
        let kept = points.filter { (counts[$0] ?? 0) < 3 }
        let minimum = name == "oval" ? 12 : 5
        return kept.count >= minimum ? kept : nil
    }
}

// MARK: - Words

/// What a behaviour or event type is called on screen. Head orientation is
/// "head", never gaze; a type the app does not know is said as the device
/// named it, made readable.
enum DriverBehaviourCopy {
    static func label(_ type: String) -> String {
        switch type {
        case "distraction": return "Head turned away"
        case "eyes_down": return "Head down"
        case "eyes_up": return "Head up"
        case "phone_use": return "Phone in use"
        case "drowsiness": return "Eyes closed"
        case "face_not_visible", "attention_lost": return "Face not visible"
        default: return readable(type)
        }
    }

    /// "phone_use_while_distracted" -> "Phone use while distracted"
    static func readable(_ type: String) -> String {
        let words = type.replacingOccurrences(of: "_", with: " ").replacingOccurrences(of: ":", with: " in ")
        return words.prefix(1).uppercased() + words.dropFirst()
    }

    /// "2.1 s", "12 s", "1 min 5 s".
    static func duration(_ seconds: Double) -> String {
        let s = max(0, seconds)
        if s < 10 { return String(format: "%.1f s", s) }
        if s < 60 { return "\(Int(s.rounded(.down))) s" }
        let whole = Int(s.rounded(.down))
        return whole % 60 == 0 ? "\(whole / 60) min" : "\(whole / 60) min \(whole % 60) s"
    }
}

/// One item of the behaviour strip: "Head turned away · 2.1 s".
struct BehaviourStripItem: Equatable, Identifiable {
    enum Tone: Equatable {
        /// Developing: amber, the condition holds but has not been sustained.
        case developing
        /// Confirmed: the alert colour.
        case confirmed
        case neutral
    }
    let type: String
    let text: String
    let tone: Tone
    var id: String { type }
    static let all = Int.max

    /// The strip, most urgent first: confirmed, then developing, each by how
    /// long it has held. At most `limit` items.
    static func make(_ overlay: LiveOverlay?, limit: Int = 3) -> [BehaviourStripItem] {
        guard let overlay, overlay.analysed, let behaviours = overlay.behaviours else { return [] }
        func rank(_ b: LiveOverlay.Behaviour) -> Int {
            switch b.phase {
            case .confirmed: return 0
            case .developing: return 1
            case .other: return 2
            }
        }
        let sorted = behaviours.enumerated().sorted { a, b in
            let ra = rank(a.element), rb = rank(b.element)
            if ra != rb { return ra < rb }
            let fa = a.element.forS ?? -1, fb = b.element.forS ?? -1
            if fa != fb { return fa > fb }
            return a.offset < b.offset
        }.map(\.element)
        var seen = Set<String>()
        return sorted.compactMap { b -> BehaviourStripItem? in
            guard seen.insert(b.type).inserted else { return nil }
            let label = DriverBehaviourCopy.label(b.type)
            let text = b.forS.map { "\(label) · \(DriverBehaviourCopy.duration($0))" } ?? label
            let tone: Tone
            switch b.phase {
            // No face is a state of the evidence, not a behaviour of the
            // driver: said, never in the alert colour.
            case _ where b.type == "face_not_visible": tone = .neutral
            case .confirmed: tone = .confirmed
            case .developing: tone = .developing
            case .other: tone = .neutral
            }
            return BehaviourStripItem(type: b.type, text: text, tone: tone)
        }.prefix(limit).map { $0 }
    }

    /// What VoiceOver says for the item: the phase in words, not by colour.
    var spoken: String {
        switch tone {
        case .confirmed: return "\(text), confirmed"
        case .developing: return "\(text), developing"
        case .neutral: return text
        }
    }
}

/// The "why" line: the newest event, in the device's own words.
struct OverlayWhyLine: Equatable {
    let title: String
    /// The device's `signals`, verbatim.
    let because: String?
    let ageText: String?
    let isAlert: Bool
    var ageS: Double? = nil

    /// Only events that are alerts (severity loud or soft): a "log" record
    /// such as "monitoring ready" is not a reason for anything on screen.
    static func make(_ overlay: LiveOverlay?) -> OverlayWhyLine? {
        guard let event = overlay?.lastEvent else { return nil }
        let severity = event.severity ?? ""
        guard severity == "loud" || severity == "soft" else { return nil }
        return OverlayWhyLine(
            title: DriverBehaviourCopy.readable(event.type),
            because: event.signals,
            ageText: event.ageS.map { $0 < 1 ? "just now" : "\(DriverBehaviourCopy.duration($0)) ago" },
            isAlert: severity == "loud",
            ageS: event.ageS)
    }

    /// What an alert banner says while the analysis reports an alert condition
    /// NOW: the condition itself ("Head down · 3.2 s"), with the device's why
    /// line only when that event is from the last few seconds -- an older one
    /// is about something else and must not be read as the reason for this.
    static func bannerText(conditionHeadline: String, why: OverlayWhyLine?, freshWithin: Double = 5) -> String {
        guard let why, let age = why.ageS, age <= freshWithin else { return conditionHeadline }
        return "\(conditionHeadline) — \(why.text)"
    }

    /// "Distraction: head turned away 3.4 s"
    var text: String { because.map { "\(title): \($0)" } ?? title }
}

/// The compact performance readout: "14 fps · Detector 11.8 ms · Frame 83 ms".
/// Every number is the device's own measurement from this frame's document.
enum OverlayPerfReadout {
    static let componentNames: [(key: String, label: String)] = [
        ("object_detector", "Detector"), ("face_landmarks", "Face"), ("hand_landmarks", "Hands"),
    ]

    static func fps(_ value: Double) -> String {
        value < 10 ? String(format: "%.1f fps", value) : "\(Int(value.rounded())) fps"
    }

    static func ms(_ value: Double) -> String {
        value < 100 ? String(format: "%.1f ms", value) : "\(Int(value.rounded())) ms"
    }

    /// Processed fps, detector run time, frame age; nil when the document
    /// carries none of them.
    static func text(_ perf: LiveOverlay.Perf?) -> String? {
        guard let perf else { return nil }
        var parts: [String] = []
        if let value = perf.fps { parts.append(fps(value)) }
        if let detector = perf.componentMs["object_detector"] { parts.append("Detector \(ms(detector))") }
        if let age = perf.frameAgeMs { parts.append("Frame \(Int(age.rounded())) ms") }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }

    /// Every component's run time, for the longer readout.
    static func components(_ perf: LiveOverlay.Perf?) -> [String] {
        guard let perf else { return [] }
        return componentNames.compactMap { name in perf.componentMs[name.key].map { "\(name.label) \(ms($0))" } }
    }
}

/// The Simulated badge: shown for a frame whose document says so, and kept for
/// the rest of that stream once any document on it has (a camera fallback part
/// carries no document, and must not make a simulated stream look real).
enum SimulatedBadge {
    static func shows(current: LiveOverlay?, streamSeenSimulated: Bool) -> Bool {
        streamSeenSimulated || current?.isSimulated == true
    }
}

/// The head-orientation indicator. "Head" always; never gaze.
struct HeadIndicator: Equatable {
    /// "Head · Left"
    let text: String
    /// Needle angle, -90...90 degrees (yaw), or nil when no usable yaw.
    let yawDeg: Double?
    let calibrated: Bool

    static func make(_ overlay: LiveOverlay?) -> HeadIndicator? {
        guard let face = overlay?.face, face.visible, let pose = face.headPose else { return nil }
        let direction = pose.direction ?? "—"
        let yaw = pose.yawDeg.flatMap { abs($0) <= 90 ? $0 : nil }
        return HeadIndicator(text: "Head · \(direction)", yawDeg: yaw, calibrated: pose.calibrated == true)
    }
}

// MARK: - The distracted-driving panel

/// What the live analysis says right now, in one headline: the panel's state
/// and the AI view's status chip. Read from the newest overlay only.
struct LiveAnalysisState: Equatable {
    enum Kind: Equatable {
        /// The camera does not send the analysis (older software).
        case unsupported
        /// No picture yet / not live.
        case waiting
        /// A frame with no analysis behind it.
        case notAnalysed
        /// The detector could not look (`det: null`).
        case detectionUnavailable
        /// The face model looked and found no face.
        case faceNotVisible
        /// Analysed, but nothing on this frame supports a statement about the driver.
        case insufficientEvidence
        /// Watching, a behaviour in progress.
        case behaviour
        /// An alert condition is active.
        case alert
        /// Watching; nothing in progress.
        case watching
    }

    let kind: Kind
    let headline: String
    let detail: String?

    /// - Parameters:
    ///   - overlay: the newest frame's document (nil: that frame had none).
    ///   - overlayCapable: the stream answered `X-Stream-Overlay: sd-overlay/1`;
    ///     nil while no stream has answered.
    ///   - live: frames are arriving (FeedLiveness).
    static func make(_ overlay: LiveOverlay?, overlayCapable: Bool?, live: Bool) -> LiveAnalysisState {
        guard live else {
            return LiveAnalysisState(kind: .waiting, headline: "Waiting for live video",
                                     detail: "The analysis appears with the camera's picture.")
        }
        if overlayCapable == false {
            return LiveAnalysisState(kind: .unsupported, headline: "Live analysis not available",
                                     detail: "This camera's software doesn't send what its AI sees.")
        }
        guard let overlay, overlay.analysed else {
            return LiveAnalysisState(kind: .notAnalysed, headline: "Not analysed",
                                     detail: "This frame has no analysis behind it.")
        }
        let strip = BehaviourStripItem.make(overlay, limit: BehaviourStripItem.all)
            .filter { $0.type != "face_not_visible" }
        let confirmed = strip.first { $0.tone == .confirmed }
        if overlay.alertActive || confirmed != nil {
            return LiveAnalysisState(kind: .alert, headline: confirmed?.text ?? "Alert condition active",
                                     detail: "Confirmed by the device")
        }
        if let first = strip.first {
            return LiveAnalysisState(kind: .behaviour, headline: first.text,
                                     detail: first.tone == .developing ? "Developing — not yet an alert" : nil)
        }
        let faceMissing = overlay.face.map { !$0.visible } ?? false
            || overlay.behaviours?.contains(where: { $0.type == "face_not_visible" }) == true
        if faceMissing {
            return LiveAnalysisState(kind: .faceNotVisible, headline: "Face not visible",
                                     detail: "Nothing can be said about the driver's head or eyes.")
        }
        if overlay.detections == nil {
            return LiveAnalysisState(kind: .detectionUnavailable, headline: "Detection unavailable",
                                     detail: "The object detector couldn't look at this frame.")
        }
        if overlay.face == nil {
            return LiveAnalysisState(kind: .insufficientEvidence, headline: "Insufficient evidence",
                                     detail: "The face model didn't run on this frame.")
        }
        if overlay.face?.headPose?.calibrated == false {
            return LiveAnalysisState(kind: .watching, headline: "Calibrating to the driver",
                                     detail: "Watching; head orientation is not yet relative to this driver.")
        }
        return LiveAnalysisState(kind: .watching, headline: "Watching the driver",
                                 detail: "No behaviour in progress")
    }
}

/// Everything the distracted-driving panel shows, from the newest overlay.
struct DriverPanelModel: Equatable {
    let state: LiveAnalysisState
    let behaviours: [BehaviourStripItem]
    let why: OverlayWhyLine?
    /// Components the device reports not working, in words.
    let unavailable: String?
    let head: HeadIndicator?
    let eyesClosed: Bool

    static func make(_ overlay: LiveOverlay?, overlayCapable: Bool?, live: Bool) -> DriverPanelModel {
        let state = LiveAnalysisState.make(overlay, overlayCapable: overlayCapable, live: live)
        let usable = live && overlay?.analysed == true
        return DriverPanelModel(
            state: state,
            behaviours: usable ? BehaviourStripItem.make(overlay) : [],
            why: usable ? OverlayWhyLine.make(overlay) : nil,
            unavailable: usable ? unavailableText(overlay?.unavailable ?? []) : nil,
            head: usable ? HeadIndicator.make(overlay) : nil,
            eyesClosed: usable && overlay?.face?.visible == true && overlay?.face?.eyesClosed == true)
    }

    static func componentTitle(_ name: String) -> String {
        switch name {
        case "object_detector": return "Object detection"
        case "face_landmarks": return "Face tracking"
        case "hand_landmarks": return "Hand tracking"
        case "perception": return "Perception"
        default: return DriverBehaviourCopy.readable(name)
        }
    }

    /// "Not working: Object detection, Hand tracking"
    static func unavailableText(_ components: [String]) -> String? {
        guard !components.isEmpty else { return nil }
        return "Not working: " + components.map(componentTitle).joined(separator: ", ")
    }

    /// The panel as one VoiceOver sentence.
    var spoken: String {
        var parts = [state.headline]
        if let detail = state.detail { parts.append(detail) }
        parts += behaviours.dropFirst(state.kind == .behaviour || state.kind == .alert ? 1 : 0).map(\.spoken)
        if eyesClosed { parts.append("Eyes closed") }
        if let head { parts.append(head.text) }
        if let why { parts.append("Last event: \(why.text)") }
        if let unavailable { parts.append(unavailable) }
        return parts.joined(separator: ". ")
    }
}

/// The overlay as the views see it, refreshed per frame for drawing and
/// throttled for the words around the picture (so the chrome does not
/// re-render 14 times a second for a frame-age digit).
struct LiveOverlaySummary: Equatable {
    var panel: DriverPanelModel
    var simulated: Bool
    var perf: String?
    var perfComponents: [String]
    var alertActive: Bool
    var mode: String?

    static func make(_ overlay: LiveOverlay?, overlayCapable: Bool?, live: Bool,
                     streamSeenSimulated: Bool) -> LiveOverlaySummary {
        LiveOverlaySummary(
            panel: DriverPanelModel.make(overlay, overlayCapable: overlayCapable, live: live),
            simulated: SimulatedBadge.shows(current: overlay, streamSeenSimulated: streamSeenSimulated),
            perf: live && overlay?.analysed == true ? OverlayPerfReadout.text(overlay?.perf) : nil,
            perfComponents: live && overlay?.analysed == true ? OverlayPerfReadout.components(overlay?.perf) : [],
            alertActive: live && overlay?.analysed == true && overlay?.alertActive == true,
            mode: overlay?.mode)
    }

    /// What changes the meaning of the screen: a state, a behaviour starting
    /// or ending or being confirmed, the badge, the why line. Not a duration
    /// ticking, a needle moving or a timing digit.
    struct Key: Equatable {
        let kind: LiveAnalysisState.Kind
        let behaviours: [String]
        let why: String?
        let unavailable: String?
        let eyesClosed: Bool
        let head: String?
        let simulated: Bool
        let alertActive: Bool
        let mode: String?
    }

    var key: Key {
        Key(kind: panel.state.kind, behaviours: panel.behaviours.map { "\($0.type):\($0.tone)" },
            why: panel.why.map { "\($0.title)|\($0.because ?? "")|\($0.isAlert)" }, unavailable: panel.unavailable, eyesClosed: panel.eyesClosed,
            head: panel.head?.text, simulated: simulated, alertActive: alertActive, mode: mode)
    }

    /// Whether a new summary must be published now: a change of meaning at
    /// once; a running duration or the head needle at most every
    /// `tickInterval`; the perf numbers at most every `perfInterval` seconds.
    static func shouldPublish(old: LiveOverlaySummary?, new: LiveOverlaySummary, sinceLast: TimeInterval,
                              tickInterval: TimeInterval = 0.2, perfInterval: TimeInterval = 1) -> Bool {
        guard let old else { return true }
        if old == new { return false }
        if old.key != new.key { return true }
        if old.panel != new.panel { return sinceLast >= tickInterval }
        return sinceLast >= perfInterval
    }
}
