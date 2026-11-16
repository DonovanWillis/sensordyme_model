import Foundation

/// The device's developer diagnostics (`GET /api/diagnostics`, schema
/// `sd-diagnostics/1`, docs/DIAGNOSTICS.md): what it detects, the evidence its
/// decisions rest on, the conditions it has open, its model versions and where
/// each frame's time goes.
///
/// Developer-only. The app offers it only when the device's own health report
/// says diagnostics are switched on (`dashboard.developer_diagnostics: true`)
/// AND this phone holds the device token (Settings > Live feed). The device
/// refuses it without the token anyway; the app simply never asks.
///
/// Foundation-only so ios/Tests/DeviceDiagnosticsTests.swift can decode real
/// payloads and pin the wording without Xcode.
struct DeviceDiagnostics: Decodable, Equatable {
    struct Platform: Decodable, Equatable {
        /// "pi" | "desktop" | "simulated": the one field that says whose numbers these are.
        let measurementClass: String?
        let note: String?
        let board: String?
        let machine: String?
        let system: String?

        enum CodingKeys: String, CodingKey {
            case measurementClass = "class"
            case note, board, machine, system
        }
    }

    struct Detection: Decodable, Equatable {
        let label: String
        let confidence: Double?
        let trackID: Int?

        enum CodingKeys: String, CodingKey {
            case label, confidence
            case trackID = "track_id"
        }
    }

    struct Detections: Decodable, Equatable {
        let state: String?
        let reason: String?
        let fresh: Bool?
        let ageS: Double?
        let total: Int?
        /// Nil when the detector could not look (not the same as "nothing seen").
        let items: [Detection]?

        enum CodingKeys: String, CodingKey {
            case state, reason, fresh, total, items
            case ageS = "age_s"
        }
    }

    struct OpenCondition: Decodable, Equatable {
        let source: String?
        let condition: String?
        let state: String?
        let runningS: Double?

        enum CodingKeys: String, CodingKey {
            case source, condition, state
            case runningS = "running_s"
        }
    }

    struct Frame: Decodable, Equatable {
        let module: String?
        let frameID: Int?
        let frameAgeMs: Double?
        let simulated: Bool?
        let detections: Detections?
        let face: DiagnosticValue?
        let evidence: DiagnosticValue?
        let openConditions: [OpenCondition]?

        enum CodingKeys: String, CodingKey {
            case module, simulated, detections, face, evidence
            case frameID = "frame_id"
            case frameAgeMs = "frame_age_ms"
            case openConditions = "open_conditions"
        }
    }

    struct Capabilities: Decodable, Equatable {
        let state: String?
        let monitoring: Bool?
        let available: [String]?
        let impaired: [String]?
        let paused: [String]?
    }

    struct StageStat: Decodable, Equatable {
        let n: Int?
        let p50Ms: Double?
        let p95Ms: Double?
        let maxMs: Double?

        enum CodingKeys: String, CodingKey {
            case n
            case p50Ms = "p50_ms"
            case p95Ms = "p95_ms"
            case maxMs = "max_ms"
        }
    }

    struct Latency: Decodable, Equatable {
        /// A stage with no sample yet is `null` on the wire: nil here, never 0.
        let stages: [String: StageStat?]?
        let processedFps: Double?
        let rssMb: Double?
        let rssSource: String?
        let socTemperatureC: Double?
        let throttleNow: [String]?

        enum CodingKeys: String, CodingKey {
            case stages
            case processedFps = "processed_fps"
            case rssMb = "rss_mb"
            case rssSource = "rss_source"
            case socTemperatureC = "soc_temperature_c"
            case throttleNow = "throttle_now"
        }
    }

    let schema: String
    let platform: Platform?
    let snapshotAgeS: Double?
    let frame: Frame?
    let warmingUp: Bool?
    let capabilities: Capabilities?
    let unknownOrDegraded: [String]?
    let versions: DiagnosticValue?
    let latency: Latency?

    enum CodingKeys: String, CodingKey {
        case schema, platform, frame, capabilities, versions, latency
        case snapshotAgeS = "snapshot_age_s"
        case warmingUp = "warming_up"
        case unknownOrDegraded = "unknown_or_degraded"
    }

    static let supportedSchema = "sd-diagnostics/1"

    /// Decodes a body, refusing any other schema rather than misreading it.
    static func decode(_ data: Data) throws -> DeviceDiagnostics {
        let decoded = try JSONDecoder().decode(DeviceDiagnostics.self, from: data)
        guard decoded.schema == supportedSchema else {
            throw DeviceDiagnosticsError(message: "The device sent diagnostics in a format this app doesn't read (\(decoded.schema)).")
        }
        return decoded
    }

    /// `GET <base>/api/diagnostics` with the device token. Errors carry a sentence
    /// a developer can act on (DiagnosticsAccess.problem).
    static func fetch(baseURL: String, token: String, timeout: TimeInterval = 6) async throws -> DeviceDiagnostics {
        guard let url = URL(string: baseURL)?.appendingPathComponent("api/diagnostics") else {
            throw DeviceDiagnosticsError(message: "invalid device address")
        }
        let cleaned = token.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty else { throw DeviceDiagnosticsError(message: DiagnosticsAccess.problem(status: 401, hasToken: false)) }
        var request = URLRequest(url: url)
        request.timeoutInterval = timeout
        request.cachePolicy = .reloadIgnoringLocalCacheData
        request.setValue(cleaned, forHTTPHeaderField: "X-Device-Token")
        let (data, response) = try await URLSession.shared.data(for: request)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard status == 200 else {
            throw DeviceDiagnosticsError(message: DiagnosticsAccess.problem(status: status, hasToken: true))
        }
        return try decode(data)
    }
}

struct DeviceDiagnosticsError: LocalizedError, Equatable {
    let message: String
    var errorDescription: String? { message }
}

/// What the device's /api/health says about diagnostics (`diagnostics` block).
struct DiagnosticsFlag: Decodable, Equatable {
    let enabled: Bool?
    let schema: String?
}

/// When the app may offer the developer diagnostics screen, and what a refusal means.
enum DiagnosticsAccess {
    /// Only when the device reports them switched on AND the phone holds the device token.
    static func available(reported: DiagnosticsFlag?, token: String) -> Bool {
        reported?.enabled == true && !token.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    static func problem(status: Int, hasToken: Bool) -> String {
        switch status {
        case 401:
            return hasToken ? "The device rejected this phone's device token. Check Settings > Live feed > Device token."
                : "Diagnostics need the device token. Add it under Settings > Live feed > Device token."
        case 403:
            return "The device has no dashboard.token configured, so it never serves diagnostics."
        case 404:
            return "Diagnostics are switched off on this device (dashboard.developer_diagnostics)."
        case 0:
            return "The device didn't answer (is it on and on this Wi-Fi?)."
        default:
            return "The device couldn't assemble its diagnostics (HTTP \(status)); see the device log."
        }
    }
}

/// Plain rows for the diagnostics screen. Everything shown comes from the
/// payload; a value the device did not report reads "not reported", never 0.
enum DiagnosticsPresentation {
    struct Row: Equatable, Hashable {
        let title: String
        let value: String
    }

    /// Whose numbers these are, first and loudest: a Mac figure is never a Pi figure.
    static func hostBanner(_ diagnostics: DeviceDiagnostics) -> String {
        let platform = diagnostics.platform
        switch platform?.measurementClass {
        case "pi": return "Measured on a Raspberry Pi" + (platform?.board.map { " (\($0))" } ?? "")
        case "desktop":
            let host = [platform?.system, platform?.machine].compactMap { $0 }.joined(separator: " ")
            return "Desktop host\(host.isEmpty ? "" : " (\(host))"): NOT a Raspberry Pi measurement"
        case "simulated": return "SIMULATED: not a hardware measurement"
        default: return "Host not reported: treat these numbers as unverified"
        }
    }

    static func percent(_ value: Double?) -> String {
        guard let value, value.isFinite else { return "no confidence" }
        return "\(Int((value * 100).rounded()))%"
    }

    static func detectionRows(_ frame: DeviceDiagnostics.Frame?) -> [Row] {
        guard let detections = frame?.detections else { return [Row(title: "Detections", value: "not reported")] }
        guard let items = detections.items else {
            let why = detections.reason ?? detections.state ?? "not available"
            return [Row(title: "Detector", value: "can't look right now: \(why)")]
        }
        if items.isEmpty { return [Row(title: "Detections", value: "nothing detected")] }
        var rows = items.map { item in
            Row(title: item.label,
                value: percent(item.confidence) + (item.trackID.map { " · track \($0)" } ?? ""))
        }
        if let total = detections.total, total > items.count {
            rows.append(Row(title: "More", value: "\(total - items.count) lower-confidence detections not shown"))
        }
        return rows
    }

    static func conditionRows(_ frame: DeviceDiagnostics.Frame?) -> [Row] {
        let conditions = frame?.openConditions ?? []
        if conditions.isEmpty { return [Row(title: "Open conditions", value: "none")] }
        return conditions.map { c in
            let name = (c.condition ?? "condition").replacingOccurrences(of: "_", with: " ")
            let state = c.state.map { " (\($0))" } ?? ""
            return Row(title: name.prefix(1).uppercased() + name.dropFirst() + state,
                       value: c.runningS.map { String(format: "%.1f s", $0) } ?? "running")
        }
    }

    /// Pipeline order, with names a person can read.
    static let stageNames: [(key: String, name: String)] = [
        ("camera_read", "Camera read"), ("frame_age", "Frame age at pickup"),
        ("detector_total", "Detector (whole run)"), ("hailo_preprocess", "Hailo preprocess"),
        ("hailo_inference", "Hailo inference"), ("decoder_cpu", "Decoder (CPU)"),
        ("face_landmarks", "Face landmarks"), ("hands", "Hands"), ("tracking", "Tracking"),
        ("behaviour_reasoning", "Behaviour reasoning"), ("alert_dispatch", "Alert decision"),
        ("alert_output", "Alert output (command to GPIO)"), ("frame_to_decision", "Frame to decision"),
        ("frame_interval", "Between analysed frames"),
    ]

    static func ms(_ value: Double?) -> String {
        guard let value, value.isFinite else { return "–" }
        if value >= 100 { return String(format: "%.0f ms", value) }
        return value < 1 ? String(format: "%.2f ms", value) : String(format: "%.1f ms", value)
    }

    static func latencyRows(_ latency: DeviceDiagnostics.Latency?) -> [Row] {
        guard let latency else { return [Row(title: "Latency", value: "not reported")] }
        var rows: [Row] = []
        for (key, name) in stageNames {
            guard let entry = latency.stages?[key] else { continue }
            guard let stat = entry else {
                rows.append(Row(title: name, value: "no samples yet"))
                continue
            }
            rows.append(Row(title: name, value: "p50 \(ms(stat.p50Ms)) · p95 \(ms(stat.p95Ms)) · max \(ms(stat.maxMs))"))
        }
        rows.append(Row(title: "Processed rate", value: latency.processedFps.map { String(format: "%.1f fps", $0) } ?? "not measured yet"))
        if let rss = latency.rssMb {
            let peak = (latency.rssSource ?? "").hasPrefix("peak") ? " (peak)" : ""
            rows.append(Row(title: "Memory", value: String(format: "%.0f MB", rss) + peak))
        }
        rows.append(Row(title: "SoC temperature", value: latency.socTemperatureC.map { String(format: "%.1f °C", $0) } ?? "not available on this host"))
        if let throttle = latency.throttleNow {
            rows.append(Row(title: "Throttling now", value: throttle.isEmpty ? "none" : throttle.joined(separator: ", ")))
        }
        return rows
    }

    /// A nested JSON value as "path: value" lines, sorted, bounded.
    static func flatten(_ value: DiagnosticValue?, limit: Int = 40) -> [Row] {
        var rows: [Row] = []
        func walk(_ v: DiagnosticValue, _ path: String) {
            guard rows.count < limit else { return }
            switch v {
            case .object(let dict):
                if dict.isEmpty { rows.append(Row(title: path.isEmpty ? "–" : path, value: "none")); return }
                for key in dict.keys.sorted() {
                    walk(dict[key]!, path.isEmpty ? readable(key) : "\(path) › \(readable(key))")
                }
            case .array(let items):
                if items.isEmpty { rows.append(Row(title: path, value: "none")); return }
                if items.allSatisfy({ $0.isScalar }) {
                    rows.append(Row(title: path, value: items.map(\.text).joined(separator: ", ")))
                } else {
                    for (i, item) in items.enumerated() { walk(item, "\(path) \(i + 1)") }
                }
            default:
                rows.append(Row(title: path.isEmpty ? "value" : path, value: v.text))
            }
        }
        if let value { walk(value, "") }
        return rows
    }

    static func readable(_ key: String) -> String {
        let words = key.replacingOccurrences(of: "_", with: " ")
        return words.prefix(1).uppercased() + words.dropFirst()
    }
}

/// Any JSON value, for the payload's open-ended evidence and version blocks.
enum DiagnosticValue: Decodable, Equatable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case null
    case array([DiagnosticValue])
    case object([String: DiagnosticValue])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() { self = .null }
        else if let b = try? container.decode(Bool.self) { self = .bool(b) }
        else if let n = try? container.decode(Double.self) { self = .number(n) }
        else if let s = try? container.decode(String.self) { self = .string(s) }
        else if let a = try? container.decode([DiagnosticValue].self) { self = .array(a) }
        else { self = .object(try container.decode([String: DiagnosticValue].self)) }
    }

    var isScalar: Bool {
        switch self {
        case .array, .object: return false
        default: return true
        }
    }

    var text: String {
        switch self {
        case .string(let s): return s
        case .number(let n):
            if n == n.rounded() && abs(n) < 1e9 { return String(Int(n)) }
            return String(format: "%.3g", n)
        case .bool(let b): return b ? "yes" : "no"
        case .null: return "not reported"
        case .array(let a): return a.map(\.text).joined(separator: ", ")
        case .object(let o): return "\(o.count) fields"
        }
    }

    subscript(key: String) -> DiagnosticValue? {
        if case .object(let dict) = self { return dict[key] }
        return nil
    }
}
