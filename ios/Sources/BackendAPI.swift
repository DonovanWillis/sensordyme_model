import Foundation

// MARK: - Shared API surface, so the app runs identically against the real cloud or the demo device.

/// One validated detection rule from Configure-with-AI. Every field was checked
/// server-side against the device's capability taxonomy before reaching the app.
/// One monitoring rule: watch `target` for `predicate`, optionally in relation
/// to `secondary`. Mutable so the review screen can edit before activation —
/// the user approves a rule, not a black box.
struct AIRule: Codable, Equatable, Identifiable, Sendable {
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
    /// Every other key the backend sent with this rule (a timezone, anything a
    /// newer schema adds), carried back unchanged when the edited rule is sent
    /// for activation — so the app cannot drop what it does not know about.
    var extra: [String: JSONValue] = [:]

    var id: String { "\(predicate):\(target):\(secondary)" }

    /// The rule in the user's terms rather than the schema's. See RuleWording.
    var headline: String {
        RuleWording.headline(target: target, predicate: predicate, secondary: secondary,
                             zone: zone, minCount: min_count)
    }

    /// The rule as a JSON object, for a request body.
    func jsonObject() throws -> Any {
        try JSONSerialization.jsonObject(with: JSONEncoder().encode(self))
    }

    /// The rule as it is sent for activation: stamped with the phone's IANA
    /// time zone, which the device evaluates `schedule` ("23:00-06:00") in. The
    /// Pi's own clock zone is whatever the image shipped with, so "11pm" meant
    /// 11pm somewhere else. An older backend ignores the key.
    func forActivation(timeZone: TimeZone = .current) -> AIRule {
        var rule = self
        rule.extra["timezone"] = .string(timeZone.identifier)
        return rule
    }
}

// In an extension so the memberwise initialiser MockAPI uses survives.
extension AIRule {
    private struct Key: CodingKey {
        let stringValue: String
        init(_ name: String) { stringValue = name }
        init?(stringValue: String) { self.stringValue = stringValue }
        var intValue: Int? { nil }
        init?(intValue: Int) { nil }
    }

    private static let knownKeys: Set<String> = [
        "target", "predicate", "secondary", "description", "zone", "persist_s", "min_count",
        "confidence", "severity", "cooldown_s", "schedule", "notify", "caveat",
    ]

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: Key.self)
        func text(_ name: String, _ fallback: String = "") throws -> String {
            try c.decodeIfPresent(String.self, forKey: Key(name)) ?? fallback
        }
        func number(_ name: String, _ fallback: Double) throws -> Double {
            try c.decodeIfPresent(Double.self, forKey: Key(name)) ?? fallback
        }
        target = try c.decode(String.self, forKey: Key("target"))
        predicate = try text("predicate", "present")
        secondary = try text("secondary")
        description = try text("description")
        zone = try text("zone")
        persist_s = try number("persist_s", 3)
        // An integer field a Python backend may still render as 2.0.
        if let count = try? c.decodeIfPresent(Int.self, forKey: Key("min_count")) {
            min_count = count
        } else {
            min_count = Int(try number("min_count", 1))
        }
        confidence = try number("confidence", 0.6)
        severity = try text("severity", "warning")
        cooldown_s = try number("cooldown_s", 60)
        schedule = try text("schedule")
        notify = try text("notify", "push")
        caveat = try c.decodeIfPresent(String.self, forKey: Key("caveat"))
        var others: [String: JSONValue] = [:]
        for key in c.allKeys where !Self.knownKeys.contains(key.stringValue) {
            others[key.stringValue] = try c.decode(JSONValue.self, forKey: key)
        }
        extra = others
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: Key.self)
        for (name, value) in extra where !Self.knownKeys.contains(name) {
            try c.encode(value, forKey: Key(name))
        }
        try c.encode(target, forKey: Key("target"))
        try c.encode(predicate, forKey: Key("predicate"))
        try c.encode(secondary, forKey: Key("secondary"))
        try c.encode(description, forKey: Key("description"))
        try c.encode(zone, forKey: Key("zone"))
        try c.encode(persist_s, forKey: Key("persist_s"))
        try c.encode(min_count, forKey: Key("min_count"))
        try c.encode(confidence, forKey: Key("confidence"))
        try c.encode(severity, forKey: Key("severity"))
        try c.encode(cooldown_s, forKey: Key("cooldown_s"))
        try c.encode(schedule, forKey: Key("schedule"))
        try c.encode(notify, forKey: Key("notify"))
        try c.encodeIfPresent(caveat, forKey: Key("caveat"))
    }
}

// JSONValue lives in PresetLibrary.swift (Foundation-only, so its tests can use it).

/// A part of the request the device honestly cannot do, with why and what it'd take.
struct AIUnsupported: Codable, Equatable, Identifiable, Sendable {
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

/// How confident Jupiter is that it correctly understood the REQUEST — not to
/// be confused with `AIRule.confidence` (an alert-trigger threshold the rule
/// carries, shown to the user as "Sensitivity") or an event's own detection
/// confidence shown in EventDetailView. See jupiter/confidence.py's module
/// docstring on the backend for why these three are kept separate.
///
/// Deterministic, not a number the model reported about itself: computed from
/// how cleanly its response parsed and how many of its fields arrived
/// well-formed. A score below 1.0 always comes with `basis` explaining why —
/// there is no bare number with nothing behind it.
struct AIInterpretationConfidence: Codable, Equatable, Sendable {
    let score: Double
    let basis: [String]

    /// Below this, something about the model's OWN output was malformed —
    /// not merely that part of the request was unsupported (that case is
    /// already covered by `AIProposal.unsupported` and would be redundant to
    /// call out twice). A pure partial-understanding deduction alone does not
    /// cross this line; see jupiter/confidence.py's fixed deduction constants.
    static let reviewThreshold = 0.85
    var worthFlagging: Bool { score < Self.reviewThreshold }
}

/// Jupiter's own metadata about how a proposal was produced — which model,
/// how sure it was, how long it took. Optional because the demo backend and
/// older server versions don't send it; nothing here is load-bearing for
/// activating a rule.
struct AIJupiterMeta: Codable, Equatable, Sendable {
    let provider: String?
    let model: String?
    let interpretation_confidence: AIInterpretationConfidence?
}

struct AIProposal: Codable, Equatable, Sendable {
    let preset: String        // "" for a custom configuration belonging to no preset
    let variant: String
    let explanation: String
    var rules: [AIRule]
    let settings: [String: String]
    var unsupported: [AIUnsupported]
    /// Where the proposal came from, so the UI never implies a model was
    /// consulted when it was not. The values the backend actually emits are
    /// `"jupiter"` (the local model interpreted the request), `"fallback"`
    /// (keyword matching, because no model was reachable but the request still
    /// matched a known preset), `"unavailable"` (nothing could be interpreted —
    /// the only value the UI branches on) and `"demo"` from MockAPI.
    ///
    /// The list here previously said `"claude" | "unavailable" | "demo"`, which
    /// predates the move to Jupiter and omitted `"fallback"` entirely — the value
    /// most proposals carry on a backend with no model running.
    let source: String
    var mode: String?         // "custom" | "preset"
    var jupiter: AIJupiterMeta? = nil
    /// Which stored draft this proposal is (POST /v1/ai/configure). Sent back
    /// on activation so the backend can refuse — 409 — a draft that has been
    /// replaced since this one was reviewed. Nil from an older backend.
    var draft_version: String? = nil

    /// A custom configuration is the normal outcome, not a failure to match.
    var isCustom: Bool { preset.isEmpty }

    /// Nothing was understood well enough to act on. A keyword-fallback
    /// proposal carries a mode and no rules, and that is something to activate.
    var isEmpty: Bool { rules.isEmpty && preset.isEmpty }
}

/// What `POST /v1/devices/{id}/preset` confirmed about an activation.
///
/// The backend answers `{ok, desired_preset, rules_active, rules_version,
/// unsupported}`: how many of the sent rules will actually run, and every rule
/// it could not run, with why. A rule never runs beside a NAMED monitoring mode —
/// for a preset proposal the rules come back in `unsupported` with
/// `rules_active: 0` while the mode switch itself still happens. An older backend
/// omits `rules_active`: no confirmation, and the app then claims no more than
/// it always did.
struct ActivationResult: Decodable, Equatable, Sendable {
    /// How many of the sent rules will run; nil when the backend did not say.
    var rulesActive: Int?
    /// Fingerprint of the running rule set ("" when none).
    var rulesVersion: String?
    /// Rules the backend could not run, with why.
    var unsupported: [AIUnsupported]?

    var confirmsRules: Bool { rulesActive != nil }

    init(rulesActive: Int? = nil, rulesVersion: String? = nil, unsupported: [AIUnsupported]? = nil) {
        self.rulesActive = rulesActive
        self.rulesVersion = rulesVersion
        self.unsupported = unsupported
    }

    enum CodingKeys: String, CodingKey {
        case rulesActive = "rules_active"
        case rulesVersion = "rules_version"
        case unsupported
    }
}

protocol BackendAPI {
    func signup(email: String, password: String) async throws -> AuthResponse
    func login(email: String, password: String) async throws -> AuthResponse
    /// Revokes THIS session's token server-side, immediately, everywhere it's
    /// used — not just the local copy this device is about to forget. Best
    /// effort: the caller signs out locally regardless of whether this
    /// succeeds, since a person must always be able to sign out even offline.
    func logout() async throws
    func devices() async throws -> [Device]
    func claim(code: String) async throws
    func events(deviceID: Int, limit: Int) async throws -> [EventItem]
    func setPreset(deviceID: Int, preset: String) async throws
    func setNtfyTopic(_ topic: String) async throws
    func aiConfigure(deviceID: Int, prompt: String) async throws -> AIProposal
    /// Activate a reviewed proposal: its preset, and the rules AS EDITED.
    /// Throws an `APIError` with status 409 when the stored draft has changed
    /// since this proposal was fetched.
    func activate(deviceID: Int, proposal: AIProposal) async throws -> ActivationResult
    func setPhone(_ phone: String, smsOnLoud: Bool) async throws
    func renameDevice(deviceID: Int, name: String) async throws
    func zones(deviceID: Int) async throws -> [MonitorZone]
    func setZones(deviceID: Int, zones: [MonitorZone]) async throws
    func explain(deviceID: Int, eventID: Int) async throws -> AlertExplanation
    func triggerDemoEvent(_ scenario: DemoScenario) async throws -> EventItem
    func placeOrder(quantity: Int, address: String) async throws -> Int
    func orders() async throws -> [Order]

    // MARK: Preset library, custom rules, stop/resume (docs/IOS_APP_STATUS.md)
    //
    // An older backend answers these with an uncoded 404; callers check
    // `APIError.endpointMissing` and say "not supported by this server".

    /// GET /v1/presets
    func presetCatalog() async throws -> PresetCatalog
    /// GET /v1/devices/{id}/presets
    func devicePresets(deviceID: Int) async throws -> DevicePresets
    /// POST /v1/devices/{id}/preset with `settings`. Refusals carry a code
    /// (role_forbids_preset, incompatible_policy, invalid_setting, unknown_preset).
    func activatePreset(deviceID: Int, preset: String, settings: [String: JSONValue]) async throws -> ActivationResult
    /// GET /v1/devices/{id}/rules
    func deviceRules(deviceID: Int) async throws -> DeviceRules
    /// PATCH /v1/devices/{id}/rules/{rule_id}
    func setRuleEnabled(deviceID: Int, ruleID: String, enabled: Bool) async throws
    /// POST /v1/devices/{id}/protection. The returned id is what the device
    /// echoes once it has applied the request (ProtectionControl).
    func setProtection(deviceID: Int, action: ProtectionAction, confirm: Bool) async throws -> ProtectionCommand
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
    func logout() async throws {
        struct OK: Codable { let ok: Bool }
        let _: OK = try await postJSON("v1/logout", body: [:])
    }

    func aiConfigure(deviceID: Int, prompt: String) async throws -> AIProposal {
        try await postJSON("v1/ai/configure", body: ["device_id": deviceID, "prompt": prompt])
    }

    func activate(deviceID: Int, proposal: AIProposal) async throws -> ActivationResult {
        // The edited rules, not just the preset. Sending only the preset is
        // what made every control on the review screen decorative: the backend
        // promoted the draft stored at preview time, unedited, and for a named
        // preset promoted nothing at all — while the app said "Active".
        var body: [String: Any] = ["preset": proposal.preset]
        if !proposal.rules.isEmpty {
            body["rules"] = try proposal.rules.map { try $0.forActivation().jsonObject() }
            if let version = proposal.draft_version { body["draft_version"] = version }
        }
        return try await postJSON("v1/devices/\(deviceID)/preset", body: body)
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
        // Through getJSON, which checks the status: decoding a 401's
        // {"detail": ...} as OrdersResponse showed "The data couldn't be read
        // because it is missing" and never noticed the session had expired.
        let response: OrdersResponse = try await getJSON("v1/orders")
        return response.orders
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
    /// Settings per preset base, as the demo device "runs" them.
    private var presetSettings: [String: [String: JSONValue]] = [:]
    /// Custom rules activated through Ask Jupiter, as the demo device runs them.
    private var customRules: [DeviceRule] = []
    private var customInstruction = ""
    private var nextRuleID = 1
    /// Stop/resume: what the device has applied, and what is still in flight.
    private var protectionStopped = false
    private var lastCommandID: String?
    private var pendingProtection: (action: ProtectionAction, id: String, at: Date)?

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
            severity: severity,
            // Demo events come from a preset, not a configured rule. Marking
            // them as configured would make them all notify, which is exactly
            // the spam this distinction exists to prevent.
            source: "module"
        )
    }

    private func synthesizeLiveEvents() {
        // A stopped device watches nothing, so it reports nothing.
        guard !protectionStopped else { return }
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

    /// Nothing to revoke — the demo device has no real backend session.
    func logout() async throws {}

    func devices() async throws -> [Device] {
        settleModeSwitch()
        settleProtection()
        let running = customRules.filter { $0.enabled && $0.status == "running" }
        let customActive = !customRules.isEmpty
        return [
            Device(id: 1, name: deviceName, activePreset: activePreset,
                   desiredPreset: desiredPreset, online: true,
                   lastSeen: Date().timeIntervalSince1970,
                   health: Self.demoHealth,
                   // Only what was activated in this walkthrough: rules Ask
                   // Jupiter turned on, never a configuration nobody made.
                   monitoring: customActive
                       ? CustomMonitoring(instruction: customInstruction, ruleCount: customRules.count, active: true)
                       : nil,
                   status: demoStatus(customActive: customActive, running: running.map(\.id)))
        ]
    }

    /// The status contract for the walkthrough device: a general-purpose role
    /// (every mode can be picked), nothing refused. The capabilities are what
    /// the simulated health above claims, no more. Stopping it really stops it.
    private func demoStatus(customActive: Bool, running: [String]) -> DeviceStatus {
        DeviceStatus(
            monitoringState: protectionStopped ? "stopped" : "ready",
            monitoringActive: !protectionStopped,
            protection: protectionStopped ? "stopped" : "protecting",
            reasons: protectionStopped ? ["stopped from the app"] : [],
            protectionReasons: protectionStopped ? ["monitoring was stopped on request"] : [],
            mode: protectionStopped ? "off" : (customActive ? "ai_custom" : "standard"),
            role: "general", roleTitle: "General purpose", allowedPresets: nil,
            capabilities: .init(available: ["camera", "inference", "alert_output"], impaired: [], starting: []),
            rules: .init(active: running, additive: false, rejected: nil),
            stale: false, reportedAt: Date().timeIntervalSince1970,
            lastCommandID: lastCommandID
        )
    }

    /// A stop/resume lands after a short delay, like a real device's poll.
    private func settleProtection() {
        guard let pending = pendingProtection, Date().timeIntervalSince(pending.at) > 2.5 else { return }
        protectionStopped = pending.action == .stop
        lastCommandID = pending.id
        pendingProtection = nil
        history.insert(
            makeEvent(type: pending.action == .stop ? "monitoring_stopped" : "monitoring_resumed",
                      signals: "requested from the app", severity: "log", at: Date()),
            at: 0)
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
        // A named mode replaces AI Custom rules, as on a real device.
        customRules = []
        customInstruction = ""
    }

    // MARK: Preset library, rules, stop/resume (demo)

    /// The demo catalog. Validation is said as it is: the demo device is
    /// simulated, so nothing here is validated on a device.
    static let demoCatalog: PresetCatalog = {
        func setting(_ key: String, _ title: String, _ kind: PresetSetting.Kind, _ value: JSONValue,
                     min: Double? = nil, max: Double? = nil, step: Double? = nil, unit: String? = nil,
                     options: [PresetSetting.Option] = []) -> PresetSetting {
            PresetSetting(key: key, title: title, kind: kind, defaultValue: value,
                          min: min, max: max, step: step, unit: unit, options: options)
        }
        let sensitivity = setting("sensitivity", "Sensitivity", .number, .number(0.6), min: 0.3, max: 0.9, step: 0.1)
        var presets = Preset.catalog.map { preset in
            CatalogPreset(
                id: preset.name, version: "demo", title: preset.title, description: preset.subtitle,
                variants: preset.variants.map { PresetVariantInfo(id: $0.name, title: $0.title) },
                validation: PresetValidation(status: .unevaluated, evidence: ["Demo catalog — the demo device is simulated."]))
        }
        for index in presets.indices {
            switch presets[index].id {
            case "driver_safety":
                presets[index].requiredCapabilities = ["face", "gaze"]
                presets[index].optionalCapabilities = ["phone_use"]
                presets[index].hardware = ["IR camera facing the driver"]
                presets[index].temporalPolicies = [
                    TemporalPolicy(event: "eyes_off_road", description: "Eyes off the road", sustainS: 2),
                    TemporalPolicy(event: "drowsiness", description: "Eyes closed", sustainS: 1.5),
                ]
                presets[index].settings = [
                    sensitivity,
                    setting("eyes_off_s", "Eyes off the road for", .number, .number(2), min: 1, max: 5, step: 0.5, unit: "s"),
                    setting("chime", "Chime on the device", .bool, .bool(true)),
                ]
                presets[index].validation = PresetValidation(
                    status: .benchmarkedOffline,
                    evidence: ["Benchmarked offline on a public driver dataset (demo figure)."])
                presets[index].limitations = ["Sunglasses can hide the eyes.", "Needs the face in view."]
            case "baby_monitoring":
                presets[index].requiredCapabilities = ["pose", "audio"]
                presets[index].settings = [
                    sensitivity,
                    setting("cry_alert", "Alert on crying", .bool, .bool(true)),
                ]
                presets[index].limitations = ["Can't see through a fully covered crib."]
            case "elderly_monitoring":
                presets[index].requiredCapabilities = ["pose"]
                presets[index].temporalPolicies = [
                    TemporalPolicy(event: "fall", description: "On the floor after a fall", sustainS: 5),
                    TemporalPolicy(event: "inactivity_warning", description: "No movement", sustainS: 1500),
                ]
                presets[index].settings = [
                    setting("inactivity_min", "Alert after no movement for", .number, .number(25),
                            min: 5, max: 120, step: 5, unit: "min"),
                ]
                presets[index].limitations = ["Falls out of the camera's view are missed."]
            case "security_monitoring":
                presets[index].requiredCapabilities = ["person"]
                presets[index].settings = [
                    setting("linger_s", "Lingering after", .number, .number(40), min: 10, max: 300, step: 10, unit: "s"),
                    setting("arm", "Armed", .choice, .string("home"),
                            options: [.init(value: "home", title: "Home"), .init(value: "away", title: "Away")]),
                ]
            default: break
            }
        }
        presets.append(CatalogPreset(
            id: "pool_safety", version: "demo", title: "Pool Safety",
            description: "Someone entering the pool area",
            requiredCapabilities: ["person", "zones"],
            validation: PresetValidation(status: .unevaluated),
            limitations: ["Planned. Not available on any device yet."]))
        return PresetCatalog(catalogVersion: "demo-1", presets: presets, source: .server)
    }()

    func presetCatalog() async throws -> PresetCatalog {
        Self.demoCatalog
    }

    func devicePresets(deviceID: Int) async throws -> DevicePresets {
        settleModeSwitch()
        let active = Preset.split(activePreset).base
        return DevicePresets(
            deviceID: 1, catalogVersion: Self.demoCatalog.catalogVersion, activePreset: activePreset, stale: false,
            presets: Self.demoCatalog.presets.map { preset in
                if preset.id == "pool_safety" {
                    return DevicePresetState(id: preset.id, available: false, code: "planned",
                                             reason: "Waiting on a validated pool-zone model",
                                             validationStatus: preset.validation.status)
                }
                return DevicePresetState(
                    id: preset.id, available: true, active: preset.id == active && customRules.isEmpty,
                    validationStatus: preset.validation.status,
                    settings: PresetSettingsForm.initialValues(preset.settings, current: presetSettings[preset.id] ?? [:]))
            })
    }

    func activatePreset(deviceID: Int, preset: String, settings: [String: JSONValue]) async throws -> ActivationResult {
        let base = Preset.split(preset).base
        guard let entry = Self.demoCatalog.preset(base) else {
            throw APIError(message: "no such preset: \(preset)", status: 400, code: "unknown_preset")
        }
        if base == "pool_safety" {
            throw APIError(message: "Pool Safety is planned and not available on this device yet", status: 409,
                           code: "incompatible_policy")
        }
        if let problem = PresetSettingsForm.problems(entry.settings, values: settings).first {
            let title = entry.settings.first(where: { $0.key == problem.key })?.title ?? problem.key
            throw APIError(message: "\(title): \(problem.value)", status: 400, code: "invalid_setting")
        }
        presetSettings[base] = settings
        try await setPreset(deviceID: deviceID, preset: preset)
        return ActivationResult(rulesActive: 0, rulesVersion: "", unsupported: [])
    }

    func deviceRules(deviceID: Int) async throws -> DeviceRules {
        // Rules waiting for an area start once that area is drawn.
        for index in customRules.indices where customRules[index].status == "waiting" {
            if let zone = customRules[index].zone, demoZones.contains(where: { $0.name == zone }) {
                customRules[index].status = "running"
            }
        }
        return DeviceRules(rulesVersion: customRules.isEmpty ? "" : "demo-\(nextRuleID)",
                           additive: false, rules: customRules)
    }

    func setRuleEnabled(deviceID: Int, ruleID: String, enabled: Bool) async throws {
        guard let index = customRules.firstIndex(where: { $0.id == ruleID }) else {
            throw APIError(message: "no such rule", status: 404, code: "rule_not_found")
        }
        customRules[index].enabled = enabled
        if enabled {
            let waiting = customRules[index].zone.map { zone in !demoZones.contains(where: { $0.name == zone }) } ?? false
            customRules[index].status = waiting ? "waiting" : "running"
        } else {
            customRules[index].status = "disabled"
        }
    }

    func setProtection(deviceID: Int, action: ProtectionAction, confirm: Bool) async throws -> ProtectionCommand {
        if action == .stop && !confirm {
            throw APIError(message: "stopping monitoring requires confirm: true", status: 400, code: "confirmation_required")
        }
        let id = String(UUID().uuidString.prefix(12)).lowercased()
        pendingProtection = (action, id, Date())
        return ProtectionCommand(commandID: id, desired: action.rawValue)
    }

    /// Mirrors the real backend: a custom proposal's rules all run (the demo has
    /// no capability layer to refuse any), and a named mode switches while its
    /// rules come back unsupported, because rules never run beside a mode.
    func activate(deviceID: Int, proposal: AIProposal) async throws -> ActivationResult {
        guard !proposal.preset.isEmpty else {
            // The rules the demo device now runs, listed by GET rules. A rule
            // that needs an area waits until that area is drawn.
            customRules = proposal.rules.map { rule in
                defer { nextRuleID += 1 }
                let zone = rule.zone.isEmpty ? nil : rule.zone
                let waiting = zone.map { zone in !demoZones.contains(where: { $0.name == zone }) } ?? false
                return DeviceRule(
                    id: "r\(nextRuleID)", enabled: true, summary: rule.headline,
                    targets: [rule.target] + (rule.secondary.isEmpty ? [] : [rule.secondary]),
                    zone: zone, dwellS: rule.persist_s, requiredCapabilities: [],
                    status: waiting ? "waiting" : "running", reason: nil)
            }
            customInstruction = proposal.explanation
            return ActivationResult(rulesActive: proposal.rules.count, rulesVersion: "demo", unsupported: [])
        }
        try await setPreset(deviceID: deviceID, preset: proposal.preset)
        let moved = proposal.rules.map { rule in
            AIUnsupported(
                request: rule.description,
                reason: "a rule can't run alongside a monitoring mode",
                requires: "AI Custom monitoring",
                support: "unsupported", alternatives: [])
        }
        return ActivationResult(rulesActive: 0, rulesVersion: "", unsupported: moved)
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
                ExplanationFact(label: "Confidence", detail: "94%"),
                ExplanationFact(label: "Held for", detail: "8 seconds"),
                ExplanationFact(label: "Area", detail: "living room"),
            ],
            rule: "You were alerted because your rule alerts after 5 seconds.",
            confidence: 0.94, source: "facts",
            narrative: "Fall SensorDyme saw: Rapid drop; Stillness; Confidence — 94%; Held for — 8 seconds; Area — living room. You were alerted because your rule alerts after 5 seconds.")
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

extension APIError {
    /// The server answered, and it does not have this endpoint (an older
    /// backend). Not a failure of the request: a capability the server lacks.
    var endpointMissing: Bool { ServerSupport.endpointMissing(status: status, code: code) }
}

// MARK: - APIClient generic POST helper used by aiConfigure

extension APIClient {
    func getJSON<T: Decodable>(_ path: String) async throws -> T {
        guard let baseURL else {
            // `noBackendMessage` rather than a literal: AppState knows how many
            // attempts have failed, and this does not. Naming Settings on the
            // first failure is what made automatic discovery look broken while
            // it was still running. See BackendResolution.failureMessage.
            throw APIError(message: noBackendMessage, isUnreachable: true)
        }
        guard BackendResolution.cleartextPermitted(baseURL.absoluteString) else {
            throw APIError(message: BackendResolution.cleartextRefusal(host: baseURL.host ?? "that server"))
        }
        var req = URLRequest(url: baseURL.appendingPathComponent(path))
        req.httpMethod = "GET"
        req.timeoutInterval = 30
        if let token { req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        let (data, response) = try await URLSession.shared.data(for: req)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        if let error = Self.httpError(status: status, data: data, sentToken: token != nil) {
            throw error
        }
        return try Self.decodeReply(T.self, from: data)
    }

    func postJSON<T: Decodable>(_ path: String, body: [String: Any]) async throws -> T {
        guard let baseURL else {
            // `noBackendMessage` rather than a literal: AppState knows how many
            // attempts have failed, and this does not. Naming Settings on the
            // first failure is what made automatic discovery look broken while
            // it was still running. See BackendResolution.failureMessage.
            throw APIError(message: noBackendMessage, isUnreachable: true)
        }
        guard BackendResolution.cleartextPermitted(baseURL.absoluteString) else {
            throw APIError(message: BackendResolution.cleartextRefusal(host: baseURL.host ?? "that server"))
        }
        var req = URLRequest(url: baseURL.appendingPathComponent(path))
        req.httpMethod = "POST"
        req.timeoutInterval = 60
        if let token { req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await URLSession.shared.data(for: req)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        if let error = Self.httpError(status: status, data: data, sentToken: token != nil) {
            throw error
        }
        return try Self.decodeReply(T.self, from: data)
    }
}
