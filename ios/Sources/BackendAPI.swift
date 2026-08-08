import Foundation

// MARK: - Shared API surface, so the app runs identically against the real cloud or the demo device.

/// One validated detection rule from Configure-with-AI. Every field was checked
/// server-side against the device's capability taxonomy before reaching the app.
/// One monitoring rule: watch `target` for `predicate`, optionally in relation
/// to `secondary`. Mutable so the review screen can edit before activation —
/// the user approves a rule, not a black box.
struct AIRule: Codable, Equatable, Identifiable {
    var target: String        // "person", "dog", "lying_down"
    var predicate: String     // present | absent | in_zone | near | holding
    var secondary: String     // second object for relational rules, or ""
    var description: String
    var zone: String          // zone the user must draw, or ""
    var persist_s: Double
    var min_count: Int
    var confidence: Double
    var severity: String      // info | warning | critical
    var cooldown_s: Double
    var schedule: String      // "23:00-06:00" or ""
    var notify: String        // push | sms | silent
    /// Present when the rule is an approximation (proximity is evidence of
    /// "with", not proof of interaction). Carried on the rule so the caveat
    /// survives to the review screen rather than living only in the reply.
    var caveat: String?

    var id: String { "\(predicate):\(target):\(secondary)" }

    /// The rule in the user's terms rather than the schema's.
    var headline: String {
        let verb: String
        switch predicate {
        case "absent": verb = "is gone"
        case "in_zone": verb = "is in \(zone.isEmpty ? "the area" : zone)"
        case "near", "holding": verb = "is with \(secondary)"
        default: verb = min_count > 1 ? "appears (\(min_count)+)" : "appears"
        }
        return "\(target.capitalized) \(verb)"
    }
}

/// A part of the request the device honestly cannot do, with why and what it'd take.
struct AIUnsupported: Codable, Equatable, Identifiable {
    let request: String
    let reason: String
    let requires: String
    /// supported | partial | needs_model | needs_training_data | needs_hardware | unsupported
    var support: String?
    var alternatives: [String]?

    var id: String { request + reason }

    /// The takeaway, phrased as a next step rather than a refusal.
    var route: String {
        switch support {
        case "needs_training_data": "Add examples to teach it"
        case "needs_model": "Needs an extra model"
        case "needs_hardware": "Needs extra hardware"
        default: "Not possible from camera alone"
        }
    }
}

struct AIProposal: Codable, Equatable {
    let preset: String        // "" for a custom configuration belonging to no preset
    let variant: String
    let explanation: String
    var rules: [AIRule]
    let settings: [String: String]
    let unsupported: [AIUnsupported]
    let source: String        // "claude" | "unavailable" | "demo" — shown so demos are honest
    var mode: String?         // "custom" | "preset"

    /// A custom configuration is the normal outcome, not a failure to match.
    var isCustom: Bool { preset.isEmpty }

    /// Nothing was understood well enough to act on.
    var isEmpty: Bool { rules.isEmpty }
}

protocol BackendAPI {
    func signup(email: String, password: String) async throws -> AuthResponse
    func login(email: String, password: String) async throws -> AuthResponse
    func devices() async throws -> [Device]
    func claim(code: String) async throws
    func events(deviceID: Int, limit: Int) async throws -> [EventItem]
    func setPreset(deviceID: Int, preset: String) async throws
    func setNtfyTopic(_ topic: String) async throws
    func aiConfigure(deviceID: Int, prompt: String) async throws -> AIProposal
    func setPhone(_ phone: String, smsOnLoud: Bool) async throws
    func renameDevice(deviceID: Int, name: String) async throws
    func zones(deviceID: Int) async throws -> [MonitorZone]
    func setZones(deviceID: Int, zones: [MonitorZone]) async throws
    func explain(deviceID: Int, eventID: Int) async throws -> AlertExplanation
    func triggerDemoEvent(_ scenario: DemoScenario) async throws -> EventItem
    func placeOrder(quantity: Int, address: String) async throws -> Int
    func orders() async throws -> [Order]
}

/// A rehearsable demo event. Waiting ~20s for a synthesised alert is not a
/// demo; pressing a button and having the whole pipeline run is.
///
/// Every scenario maps to an event type the detection code already emits, so a
/// demo exercises the real path — rule severity, notification, explanation —
/// rather than a parallel one built to look good.
struct DemoScenario: Identifiable, Hashable, CaseIterable {
    let id: String
    let title: String
    let detail: String
    let icon: String
    let type: String
    let signals: String
    let severity: String

    static let distractedDriving = DemoScenario(
        id: "distraction", title: "Distracted driving",
        detail: "Phone in hand with eyes off the road", icon: "car.fill",
        type: "phone_use", signals: "phone + hand_overlap + gaze_off", severity: "loud")

    static let drowsyDriving = DemoScenario(
        id: "drowsy", title: "Drowsy driver",
        detail: "Sustained eye closure at speed", icon: "eye.slash.fill",
        type: "drowsiness", signals: "sustained_closure + head_nod", severity: "loud")

    static let poolSafety = DemoScenario(
        id: "pool", title: "Child near pool",
        detail: "Person in the pool zone", icon: "figure.pool.swim",
        type: "zone_violation:pool", signals: "person + zone_entry", severity: "loud")

    static let fall = DemoScenario(
        id: "fall", title: "Fall detected",
        detail: "Rapid drop then stillness on the floor", icon: "figure.fall",
        type: "fall", signals: "rapid_drop + stillness", severity: "loud")

    static let lingering = DemoScenario(
        id: "lingering", title: "Someone lingering",
        detail: "A person stayed at the entry", icon: "person.fill.questionmark",
        type: "person_lingering", signals: "person + dwell", severity: "soft")

    static let allCases: [DemoScenario] = [
        .distractedDriving, .drowsyDriving, .poolSafety, .fall, .lingering,
    ]
}

private struct OrderPlacedResponse: Codable {
    let order_id: Int
}

extension APIClient: BackendAPI {
    func aiConfigure(deviceID: Int, prompt: String) async throws -> AIProposal {
        try await postJSON("v1/ai/configure", body: ["device_id": deviceID, "prompt": prompt])
    }

    func zones(deviceID: Int) async throws -> [MonitorZone] {
        let response: ZonesResponse = try await getJSON("v1/devices/\(deviceID)/zones")
        return response.zones
    }

    func explain(deviceID: Int, eventID: Int) async throws -> AlertExplanation {
        try await getJSON("v1/devices/\(deviceID)/events/\(eventID)/explain")
    }

    func triggerDemoEvent(_ scenario: DemoScenario) async throws -> EventItem {
        // Deliberately unavailable against a real device: injecting a fake event
        // into a live safety system's history is exactly the kind of thing that
        // should be impossible, not merely discouraged.
        throw APIError(message: "Simulated events are only available on the demo device. Switch to the demo device to rehearse alerts.")
    }

    func setZones(deviceID: Int, zones: [MonitorZone]) async throws {
        struct OK: Codable { let ok: Bool }
        let payload = zones.map { ["name": $0.name, "kind": $0.kind, "polygon": $0.polygon] }
        let _: OK = try await postJSON("v1/devices/\(deviceID)/zones", body: ["zones": payload])
    }

    func setPhone(_ phone: String, smsOnLoud: Bool) async throws {
        struct OK: Codable { let ok: Bool }
        let _: OK = try await postJSON("v1/account/phone", body: ["phone": phone, "sms_on_loud": smsOnLoud])
    }

    func renameDevice(deviceID: Int, name: String) async throws {
        struct OK: Codable { let ok: Bool }
        let _: OK = try await postJSON("v1/devices/\(deviceID)/name", body: ["name": name])
    }

    func placeOrder(quantity: Int, address: String) async throws -> Int {
        let response: OrderPlacedResponse = try await postJSON(
            "v1/orders", body: ["quantity": quantity, "address": address]
        )
        return response.order_id
    }

    func orders() async throws -> [Order] {
        guard let baseURL else {
            throw APIError(message: "No cloud backend configured. Open Settings → Cloud backend and enter your server's address.")
        }
        var req = URLRequest(url: baseURL.appendingPathComponent("v1/orders"))
        if let token { req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        let (data, _) = try await URLSession.shared.data(for: req)
        return try JSONDecoder().decode(OrdersResponse.self, from: data).orders
    }
}

// MARK: - Mock device: a complete simulated SensorDyme unit living inside the app.

/// Drives the whole app with no backend and no hardware: an online demo device that
/// emits realistic alerts over time, honors mode switches (with a realistic sync
/// delay), keeps history for reports, and answers AI configuration prompts.
@MainActor
final class MockAPI: BackendAPI {
    /// Zones drawn during the walkthrough. In-memory only: the demo device
    /// has no camera, so persisting them would outlive their meaning.
    private var demoZones: [MonitorZone] = []

    static let shared = MockAPI()

    private var activePreset = "driver_safety"
    private var desiredPreset: String?
    private var switchRequestedAt: Date?
    private var history: [EventItem] = []
    private var lastSynthesized = Date()
    private var deviceName = "Demo Unit"
    private var mockOrders: [Order] = []
    private var nextOrderID = 1001

    private init() {
        seedHistory()
    }

    // Demo scenarios per mode: (type, signals, severity, avg interval seconds)
    private var scenario: [(String, String, String, Double)] {
        switch Preset.split(activePreset).base {
        case "baby_monitoring":
            [("stir_detected", "movement in crib", "log", 30),
             ("face_covered", "blanket near face 4s", "loud", 150),
             ("crying_detected", "audio pattern match 0.91", "soft", 90)]
        case "elderly_monitoring":
            [("movement", "kitchen zone", "log", 40),
             ("fall", "rapid drop + 2.1s stillness", "loud", 240),
             ("inactivity_warning", "no movement 25 min", "soft", 180)]
        case "security_monitoring":
            [("occupancy_change", "1 -> 2 people", "soft", 60),
             ("person_detected", "entry zone", "log", 45),
             ("unknown_person_lingering", "entry zone 40s", "loud", 200)]
        default:
            [("eyes_off_road", "gaze yaw 31deg for 2.4s", "soft", 45),
             ("phone_use_while_distracted", "phone + hand_overlap + gaze_off", "loud", 120),
             ("drowsiness", "sustained_eye_closure 1.8s", "loud", 160),
             ("vlm_judgment", "vlm:attentive — eyes forward, hands on wheel", "log", 25)]
        }
    }

    private func seedHistory() {
        var seeded: [EventItem] = []
        var t = Date().addingTimeInterval(-6 * 3600)
        while t < Date().addingTimeInterval(-60) {
            let (type, signals, severity, gap) = scenario.randomElement()!
            seeded.append(makeEvent(type: type, signals: signals, severity: severity, at: t))
            t = t.addingTimeInterval(gap * Double.random(in: 0.6...1.6))
        }
        history = seeded.reversed()
    }

    private var nextEventID = 1

    private func makeEvent(type: String, signals: String, severity: String, at: Date) -> EventItem {
        defer { nextEventID += 1 }
        return EventItem(
            // Demo events carry ids too, so tapping one opens an explanation
            // exactly as it does against a real device.
            eventID: nextEventID,
            timestamp: ISO8601DateFormatter.flexible.string(from: at),
            type: type,
            confidence: Double.random(in: 0.72...0.97),
            signals: signals,
            severity: severity
        )
    }

    private func synthesizeLiveEvents() {
        // roughly one new event every ~20s of demo use, matching the active scenario
        guard Date().timeIntervalSince(lastSynthesized) > 20 else { return }
        lastSynthesized = Date()
        let (type, signals, severity, _) = scenario.randomElement()!
        history.insert(makeEvent(type: type, signals: signals, severity: severity, at: Date()), at: 0)
    }

    private func settleModeSwitch() {
        if let desired = desiredPreset, let at = switchRequestedAt, Date().timeIntervalSince(at) > 2.5 {
            activePreset = desired
            desiredPreset = nil
            switchRequestedAt = nil
            history.insert(
                makeEvent(type: "preset_changed", signals: "mode -> \(desired)", severity: "log", at: Date()),
                at: 0
            )
            seedHistoryTailForNewMode()
        }
    }

    private func seedHistoryTailForNewMode() {
        lastSynthesized = Date().addingTimeInterval(-15)  // first event of the new mode arrives quickly
    }

    // MARK: BackendAPI
    func signup(email: String, password: String) async throws -> AuthResponse {
        AuthResponse(token: "demo", email: email.isEmpty ? "demo@sensordyme.app" : email)
    }

    func login(email: String, password: String) async throws -> AuthResponse {
        AuthResponse(token: "demo", email: email.isEmpty ? "demo@sensordyme.app" : email)
    }

    func devices() async throws -> [Device] {
        settleModeSwitch()
        return [
            Device(id: 1, name: deviceName, activePreset: activePreset,
                   desiredPreset: desiredPreset, online: true,
                   lastSeen: Date().timeIntervalSince1970,
                   health: Self.demoHealth)
        ]
    }

    /// Health for the walkthrough device. Every line says "simulated" — a demo
    /// that claims a Hailo-8 is running when no hardware is attached is the
    /// exact thing that destroys credibility when someone asks to see the Pi.
    private static let demoHealth = DeviceHealthReport(
        camera: HealthComponent(state: "ok", detail: "simulated feed", metrics: ["fps": .number(12)]),
        inference: HealthComponent(state: "ok", detail: "simulated analysis", metrics: nil),
        accelerator: HealthComponent(state: "unknown", detail: "no hardware attached (demo)", metrics: nil),
        network: HealthComponent(state: "ok", detail: "simulated connection", metrics: nil),
        system: HealthComponent(state: "ok", detail: "simulated", metrics: nil),
        cloudConnected: true
    )

    func claim(code: String) async throws {
        try await Task.sleep(for: .seconds(1.0))  // pairing handshake feel
    }

    func setPhone(_ phone: String, smsOnLoud: Bool) async throws {}

    func renameDevice(deviceID: Int, name: String) async throws {
        deviceName = name
    }

    func placeOrder(quantity: Int, address: String) async throws -> Int {
        try await Task.sleep(for: .seconds(0.8))
        let order = Order(id: nextOrderID, quantity: quantity, address: address,
                          status: "placed", createdAt: Date().timeIntervalSince1970)
        nextOrderID += 1
        mockOrders.insert(order, at: 0)
        return order.id
    }

    func orders() async throws -> [Order] {
        mockOrders
    }

    func events(deviceID: Int, limit: Int) async throws -> [EventItem] {
        settleModeSwitch()
        synthesizeLiveEvents()
        return Array(history.prefix(limit))
    }

    func setPreset(deviceID: Int, preset: String) async throws {
        desiredPreset = preset
        switchRequestedAt = Date()
    }

    func setNtfyTopic(_ topic: String) async throws {}

    func zones(deviceID: Int) async throws -> [MonitorZone] { demoZones }


    func setZones(deviceID: Int, zones: [MonitorZone]) async throws { demoZones = zones }

    func triggerDemoEvent(_ scenario: DemoScenario) async throws -> EventItem {
        try await Task.sleep(for: .milliseconds(500))  // the device thinking
        let event = makeEvent(type: scenario.type, signals: scenario.signals,
                              severity: scenario.severity, at: Date())
        history.insert(event, at: 0)
        return event
    }

    func explain(deviceID: Int, eventID: Int) async throws -> AlertExplanation {
        try await Task.sleep(for: .milliseconds(400))
        // Mirrors the real deterministic path, including the rule sentence —
        // a demo that explains alerts differently from the product teaches the
        // wrong thing to whoever is watching.
        return AlertExplanation(
            headline: "Fall", severity: "critical", when: "2:41 PM",
            facts: [
                ExplanationFact(label: "Rapid drop", detail: ""),
                ExplanationFact(label: "Stillness", detail: ""),
                ExplanationFact(label: "Detector confidence", detail: "94%"),
                ExplanationFact(label: "Held for", detail: "8 seconds"),
                ExplanationFact(label: "Area", detail: "living room"),
            ],
            rule: "You were alerted because your rule alerts after 5 seconds.",
            confidence: 0.94, source: "facts",
            narrative: "Fall SensorDyme saw: Rapid drop; Stillness; Detector confidence — 94%; Held for — 8 seconds; Area — living room. You were alerted because your rule alerts after 5 seconds.")
    }

    func aiConfigure(deviceID: Int, prompt: String) async throws -> AIProposal {
        try await Task.sleep(for: .seconds(1.2))  // feels like a model thinking

        let p = prompt.lowercased()
        func has(_ words: String...) -> Bool { words.contains { p.contains($0) } }
        func rule(_ target: String, _ predicate: String, _ desc: String, secondary: String = "",
                  zone: String = "", persist: Double = 3, severity: String = "warning",
                  cooldown: Double = 120, caveat: String? = nil) -> AIRule {
            AIRule(target: target, predicate: predicate, secondary: secondary, description: desc,
                   zone: zone, persist_s: persist, min_count: 1, confidence: 0.6,
                   severity: severity, cooldown_s: cooldown, schedule: "", notify: "push",
                   caveat: caveat)
        }

        // The demo mirrors the real capability layer, refusals included. A
        // walkthrough that only ever succeeds teaches the wrong expectation and
        // falls apart the first time someone asks for something unsupported.
        if has("dog", "cat", "pet", "couch") {
            return AIProposal(
                preset: "", variant: "",
                explanation: "Watching for your dog on the couch. Both are objects the camera recognises, so this runs as a custom rule with no mode change.",
                rules: [rule("dog", "near", "Alert when the dog is on the couch", secondary: "couch",
                             caveat: "Overlap shows the dog is with the couch, not certainly on it.")],
                settings: [:], unsupported: [], source: "demo", mode: "custom")
        }
        if has("can", "package", "parcel", "delivery") {
            return AIProposal(
                preset: "", variant: "",
                explanation: "I can't set that up. The detector doesn't recognise that object, and I'd rather say so than quietly watch for something else.",
                rules: [], settings: [:],
                unsupported: [AIUnsupported(
                    request: prompt,
                    reason: "the current vision model doesn't recognise that object",
                    requires: "labelled examples to fine-tune a detector for it",
                    support: "needs_training_data",
                    alternatives: ["bottle", "cup", "backpack", "suitcase"])],
                source: "demo", mode: "custom")
        }
        if has("floor", "ground", "fall", "elder", "grandma", "grandpa") {
            return AIProposal(
                preset: "elderly_monitoring", variant: "standard",
                explanation: "Switching to Elderly Monitoring and adding a rule for someone remaining on the floor. Posture comes from body-pose landmarks, and lying in bed does not count: the rule needs them low in the frame too.",
                rules: [rule("on_floor", "present", "Alert if someone stays on the floor",
                             persist: 30, severity: "critical", cooldown: 60)],
                settings: [:], unsupported: [], source: "demo", mode: "preset")
        }
        if has("empty", "nobody", "leaves") {
            return AIProposal(
                preset: "", variant: "",
                explanation: "Watching for the room becoming empty.",
                rules: [rule("person", "absent", "Alert when nobody is in view", persist: 10)],
                settings: [:], unsupported: [], source: "demo", mode: "custom")
        }
        if has("drive", "driving", "car", "phone", "drowsy", "road") {
            return AIProposal(
                preset: "driver_safety", variant: "strict",
                explanation: "Switching to Driver Distraction (Strict). Phone use while looking away alerts within a second; quick mirror checks never alarm.",
                rules: [rule("cell phone", "present", "Alert on phone use while driving",
                             persist: 1, severity: "critical", cooldown: 30)],
                settings: [:], unsupported: [], source: "demo", mode: "preset")
        }
        return AIProposal(
            preset: "", variant: "",
            explanation: "I couldn't turn that into something this camera can watch for. Nothing has been changed.",
            rules: [], settings: [:],
            unsupported: [AIUnsupported(
                request: prompt,
                reason: "no detectable object or condition matched the request",
                requires: "rephrase using an object the camera knows, or add a model for it",
                support: "unsupported", alternatives: [])],
            source: "demo", mode: "custom")
    }
}

// MARK: - APIClient generic POST helper used by aiConfigure

extension APIClient {
    func getJSON<T: Decodable>(_ path: String) async throws -> T {
        guard let baseURL else {
            throw APIError(message: "No cloud backend configured. Open Settings → Cloud backend and enter your server's address.")
        }
        var req = URLRequest(url: baseURL.appendingPathComponent(path))
        req.httpMethod = "GET"
        req.timeoutInterval = 30
        if let token { req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        let (data, response) = try await URLSession.shared.data(for: req)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard status != 401 else {
            throw APIError(message: "Your session has expired. Please sign in again.", isAuthFailure: true)
        }
        guard (200..<300).contains(status) else {
            let detail = (try? JSONSerialization.jsonObject(with: data) as? [String: Any])?["detail"]
            throw APIError(message: (detail as? String) ?? "request failed (\(status))")
        }
        return try JSONDecoder().decode(T.self, from: data)
    }

    func postJSON<T: Decodable>(_ path: String, body: [String: Any]) async throws -> T {
        guard let baseURL else {
            throw APIError(message: "No cloud backend configured. Open Settings → Cloud backend and enter your server's address.")
        }
        var req = URLRequest(url: baseURL.appendingPathComponent(path))
        req.httpMethod = "POST"
        req.timeoutInterval = 60
        if let token { req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await URLSession.shared.data(for: req)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard status != 401 else {
            throw APIError(message: "Your session has expired. Please sign in again.", isAuthFailure: true)
        }
        guard (200..<300).contains(status) else {
            let detail = (try? JSONSerialization.jsonObject(with: data) as? [String: Any])?["detail"]
            throw APIError(message: (detail as? String) ?? "request failed (\(status))")
        }
        return try JSONDecoder().decode(T.self, from: data)
    }
}
