import Foundation

// MARK: - Shared API surface, so the app runs identically against the real cloud or the demo device.

/// One validated detection rule from Configure-with-AI. Every field was checked
/// server-side against the device's capability taxonomy before reaching the app.
struct AIRule: Codable, Equatable {
    let detection: String     // taxonomy id, e.g. "zone_violation"
    let description: String
    let zone: String          // zone the user must draw, or ""
    let persist_s: Double
    let min_count: Int
    let confidence: Double
    let severity: String      // info | warning | critical
    let cooldown_s: Double
    let schedule: String      // "23:00-06:00" or "" (enforcement planned)
    let notify: String        // push | sms | silent
}

/// A part of the request the device honestly cannot do, with why and what it'd take.
struct AIUnsupported: Codable, Equatable {
    let request: String
    let reason: String
    let requires: String
}

struct AIProposal: Codable, Equatable {
    let preset: String
    let variant: String
    let explanation: String
    let rules: [AIRule]
    let settings: [String: String]
    let unsupported: [AIUnsupported]
    let source: String        // "claude" | "fallback" — shown so demos are honest
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
    func placeOrder(quantity: Int, address: String) async throws -> Int
    func orders() async throws -> [Order]
}

private struct OrderPlacedResponse: Codable {
    let order_id: Int
}

extension APIClient: BackendAPI {
    func aiConfigure(deviceID: Int, prompt: String) async throws -> AIProposal {
        try await postJSON("v1/ai/configure", body: ["device_id": deviceID, "prompt": prompt])
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
    static let shared = MockAPI()

    private var activePreset = "driver_distraction"
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

    private func makeEvent(type: String, signals: String, severity: String, at: Date) -> EventItem {
        EventItem(
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
                   lastSeen: Date().timeIntervalSince1970)
        ]
    }

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

    func aiConfigure(deviceID: Int, prompt: String) async throws -> AIProposal {
        try await Task.sleep(for: .seconds(1.2))  // feels like a model thinking
        let p = prompt.lowercased()

        func has(_ words: String...) -> Bool { words.contains { p.contains($0) } }
        func rule(_ det: String, _ desc: String, zone: String = "", persist: Double = 2,
                  severity: String = "warning", cooldown: Double = 60) -> AIRule {
            AIRule(detection: det, description: desc, zone: zone, persist_s: persist,
                   min_count: 1, confidence: 0.6, severity: severity, cooldown_s: cooldown,
                   schedule: "", notify: "push")
        }

        if has("baby", "infant", "crib", "nursery") {
            return AIProposal(
                preset: "baby_monitoring", variant: "nap_time",
                explanation: "Switching to Baby Monitoring (Nap Time). You'll be alerted if the baby leaves the safe zone you draw around the crib, or is unusually still for 5 minutes. Crying and face-covered detection aren't supported by the camera alone — listed below with what they'd need.",
                rules: [
                    rule("safe_zone_exit", "Alert if the baby leaves the crib zone", zone: "crib", severity: "critical"),
                    rule("unusual_stillness", "Alert after 5 minutes of unusual stillness", persist: 300),
                ],
                settings: [:],
                unsupported: [
                    AIUnsupported(request: "Crying alerts", reason: "the device has no microphone",
                                  requires: "a microphone + audio classifier"),
                    AIUnsupported(request: "Face-covered detection",
                                  reason: "adult face models aren't reliable enough on infants for a safety claim",
                                  requires: "an infant-specific vision model"),
                ],
                source: "demo"
            )
        }
        if has("fall", "elder", "grandma", "grandpa", "senior") {
            return AIProposal(
                preset: "elderly_monitoring", variant: "standard",
                explanation: "Switching to Elderly Monitoring. A confirmed fall — rapid drop then stillness, so sitting down never triggers it — alerts immediately and stays active until movement resumes.",
                rules: [
                    rule("fall", "Loud alert on a confirmed fall", persist: 1.5, severity: "critical", cooldown: 30),
                    rule("stillness", "Warn after 30 minutes without movement", persist: 1800),
                ],
                settings: [:], unsupported: [], source: "demo"
            )
        }
        if has("security", "intruder", "burglar", "door", "package") {
            return AIProposal(
                preset: "security_monitoring", variant: "away",
                explanation: "Switching to Security (Away). Person-count changes alert immediately; draw a zone over the entry and lingering there past 30 seconds triggers a loud alert. Package tracking isn't supported yet — noted below.",
                rules: [
                    rule("person_detected", "Alert when the number of people changes", persist: 1, cooldown: 30),
                    rule("person_lingering", "Loud alert if someone lingers in the entry zone", zone: "entry", persist: 30, severity: "critical"),
                ],
                settings: [:],
                unsupported: has("package") ? [
                    AIUnsupported(request: "Package placed/removed alerts",
                                  reason: "needs object-permanence tracking against a scene baseline",
                                  requires: "a custom-trained package model"),
                ] : [],
                source: "demo"
            )
        }
        if has("drive", "driving", "car", "phone", "drowsy", "road") {
            return AIProposal(
                preset: "driver_distraction", variant: "strict",
                explanation: "Switching to Driver Distraction (Strict). Phone use while looking away triggers the loudest alert within a second; drowsiness fires on sustained eye closure. Quick mirror checks never alarm — that's built into the glance tolerance.",
                rules: [
                    rule("phone_use", "Loud alert on phone use while eyes off road", persist: 1, severity: "critical", cooldown: 30),
                    rule("eyes_closed", "Loud alert on sustained eye closure", persist: 1.5, severity: "critical", cooldown: 30),
                ],
                settings: [:], unsupported: [], source: "demo"
            )
        }
        return AIProposal(
            preset: activePreset, variant: "",
            explanation: "I kept the current mode. I couldn't map \"\(prompt)\" onto a detection this camera reliably supports — the honest gaps are listed below.",
            rules: [], settings: [:],
            unsupported: [AIUnsupported(request: prompt,
                                        reason: "no matching detection in this mode's capability set",
                                        requires: "see the capability matrix, or rephrase using presence, zones, stillness, phone use, or falls")],
            source: "demo"
        )
    }
}

// MARK: - APIClient generic POST helper used by aiConfigure

extension APIClient {
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
