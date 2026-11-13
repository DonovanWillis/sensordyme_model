import Foundation

// The preset library as the BACKEND publishes it (GET /v1/presets, GET
// /v1/devices/{id}/presets, POST /v1/devices/{id}/preset with settings), and
// how the app puts it into words. Foundation only, so
// ios/Tests/PresetLibraryTests.swift can pin it down.
//
// The app used to carry the whole catalog as a Swift literal (Preset.catalog in
// Models.swift). That list is now only the offline/demo fallback, labelled as
// such wherever it is shown: it knows titles and variants, and nothing about
// what a device can run, what a preset needs, or how far it has been validated.
//
// Every field decodes leniently. A backend that predates a field, or sends a
// value this build does not know, must degrade to "not said", never fail the
// decode of the whole catalog.

// MARK: - Any JSON value

/// Any JSON value: a rule's unknown keys carried through untouched, a preset
/// setting's value, an alert pattern.
enum JSONValue: Codable, Equatable, Hashable, Sendable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case array([JSONValue])
    case object([String: JSONValue])
    case null

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null }
        else if let value = try? c.decode(Bool.self) { self = .bool(value) }
        else if let value = try? c.decode(Double.self) { self = .number(value) }
        else if let value = try? c.decode(String.self) { self = .string(value) }
        else if let value = try? c.decode([JSONValue].self) { self = .array(value) }
        else { self = .object(try c.decode([String: JSONValue].self)) }
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .string(let value): try c.encode(value)
        case .number(let value): try c.encode(value)
        case .bool(let value): try c.encode(value)
        case .array(let value): try c.encode(value)
        case .object(let value): try c.encode(value)
        case .null: try c.encodeNil()
        }
    }

    var doubleValue: Double? {
        if case .number(let value) = self { return value }
        if case .string(let text) = self { return Double(text) }
        return nil
    }

    var boolValue: Bool? {
        if case .bool(let value) = self { return value }
        return nil
    }

    var stringValue: String? {
        switch self {
        case .string(let value): return value
        case .number(let value): return value == value.rounded() ? String(Int(value)) : String(value)
        case .bool(let value): return value ? "true" : "false"
        default: return nil
        }
    }

    /// The value as Foundation JSON, for a JSONSerialization request body.
    var foundationObject: Any {
        switch self {
        case .string(let value): return value
        case .number(let value): return value
        case .bool(let value): return value
        case .array(let values): return values.map(\.foundationObject)
        case .object(let values): return values.mapValues(\.foundationObject)
        case .null: return NSNull()
        }
    }
}

/// An array that drops the elements it cannot decode instead of failing the
/// whole array: one malformed preset must not take the catalog down with it.
struct LossyArray<Element: Decodable>: Decodable {
    var elements: [Element]

    init(from decoder: Decoder) throws {
        var container = try decoder.unkeyedContainer()
        var out: [Element] = []
        while !container.isAtEnd {
            if let value = try? container.decode(Element.self) {
                out.append(value)
            } else {
                _ = try? container.decode(JSONValue.self)  // step past it
            }
        }
        elements = out
    }
}

private struct AnyKey: CodingKey {
    let stringValue: String
    init(_ name: String) { stringValue = name }
    init?(stringValue: String) { self.stringValue = stringValue }
    var intValue: Int? { nil }
    init?(intValue: Int) { nil }
}

private extension KeyedDecodingContainer where K == AnyKey {
    func string(_ key: String) -> String? {
        (try? decodeIfPresent(String.self, forKey: AnyKey(key))) ?? nil
    }
    func double(_ key: String) -> Double? {
        if let value = (try? decodeIfPresent(Double.self, forKey: AnyKey(key))) ?? nil { return value }
        return string(key).flatMap(Double.init)
    }
    func bool(_ key: String) -> Bool? {
        (try? decodeIfPresent(Bool.self, forKey: AnyKey(key))) ?? nil
    }
    func strings(_ key: String) -> [String] {
        ((try? decodeIfPresent(LossyArray<String>.self, forKey: AnyKey(key))) ?? nil)?.elements ?? []
    }
    func value(_ key: String) -> JSONValue? {
        (try? decodeIfPresent(JSONValue.self, forKey: AnyKey(key))) ?? nil
    }
    func list<T: Decodable>(_ key: String, _ type: T.Type) -> [T] {
        ((try? decodeIfPresent(LossyArray<T>.self, forKey: AnyKey(key))) ?? nil)?.elements ?? []
    }
}

// MARK: - Whether the server has an endpoint at all

/// The device names a running mode `preset[:variant][@key=value,...]`: the
/// `@...` part carries preset settings (docs/PRESET_LIBRARY.md). Everything that
/// compares or titles a mode must read the name through this, or a mode with
/// settings looks like an unknown variant.
enum PresetName {
    static func parse(_ name: String?) -> (base: String, variant: String?, settings: [String: String]) {
        guard let name, !name.isEmpty else { return ("", nil, [:]) }
        let at = name.firstIndex(of: "@")
        let mode = at.map { String(name[..<$0]) } ?? name
        var settings: [String: String] = [:]
        if let at {
            for pair in name[name.index(after: at)...].split(separator: ",") {
                let kv = pair.split(separator: "=", maxSplits: 1).map(String.init)
                if kv.count == 2, !kv[0].isEmpty { settings[kv[0]] = kv[1] }
            }
        }
        let parts = mode.split(separator: ":", maxSplits: 1).map(String.init)
        return (parts.first ?? "", parts.count > 1 ? parts[1] : nil, settings)
    }
}

enum ServerSupport {
    /// A 404/405 with no machine-readable code is FastAPI's own "no such
    /// route": an older backend that does not have this endpoint. A coded 404
    /// (`device_not_found`, `rule_not_found`) is a real answer from a backend
    /// that does, and must not be mistaken for it.
    static func endpointMissing(status: Int, code: String?) -> Bool {
        (status == 404 || status == 405) && (code ?? "").isEmpty
    }

    static let notSupported = "Not supported by this server yet. It needs a newer SensorDyme backend."
}

// MARK: - Catalog (GET /v1/presets)

/// How far a preset's detection has been checked. Only `validatedOnDevice`
/// is ever shown as validated.
enum ValidationStatus: String, Equatable, Sendable {
    case unevaluated
    case benchmarkedOffline = "benchmarked_offline"
    case evaluatedInDomain = "evaluated_in_domain"
    case validatedOnDevice = "validated_on_device"
    /// Not reported, or a value this build does not know.
    case unknown

    init(raw: String?) {
        self = raw.flatMap(ValidationStatus.init(rawValue:)) ?? .unknown
    }
}

struct PresetValidation: Equatable, Sendable {
    var status: ValidationStatus
    /// One line per piece of evidence, as the backend described it.
    var evidence: [String]

    init(status: ValidationStatus = .unknown, evidence: [String] = []) {
        self.status = status
        self.evidence = evidence
    }
}

struct PresetVariantInfo: Equatable, Identifiable, Sendable {
    let id: String
    let title: String
}

struct TemporalPolicy: Equatable, Sendable {
    let event: String
    let description: String
    let sustainS: Double?
}

struct PresetSetting: Equatable, Identifiable, Sendable {
    enum Kind: Equatable, Sendable {
        case number, bool, choice
        /// A type this build cannot render. Shown read-only, never sent.
        case unsupported(String)
    }

    struct Option: Equatable, Sendable, Identifiable {
        let value: String
        let title: String
        var id: String { value }
    }

    let key: String
    let title: String
    let kind: Kind
    let defaultValue: JSONValue?
    var min: Double? = nil
    var max: Double? = nil
    var step: Double? = nil
    var unit: String? = nil
    var options: [Option] = []

    var id: String { key }
}

struct CatalogPreset: Equatable, Identifiable, Sendable {
    let id: String
    var version: String? = nil
    var title: String
    var description: String = ""
    var variants: [PresetVariantInfo] = []
    var requiredCapabilities: [String] = []
    var optionalCapabilities: [String] = []
    var hardware: [String] = []
    var temporalPolicies: [TemporalPolicy] = []
    var alertPatterns: [String: JSONValue] = [:]
    var settings: [PresetSetting] = []
    var exclusiveGroup: String? = nil
    var validation = PresetValidation()
    var limitations: [String] = []
}

struct PresetCatalog: Equatable, Sendable {
    enum Source: Equatable, Sendable {
        /// Published by the backend this app is talking to.
        case server
        /// The list built into the app, because the server did not publish
        /// one. Carries why, for the label shown with it.
        case builtIn(reason: String)
    }

    var catalogVersion: String?
    var presets: [CatalogPreset]
    var source: Source = .server

    var isBuiltIn: Bool {
        if case .builtIn = source { return true }
        return false
    }

    func preset(_ id: String) -> CatalogPreset? {
        presets.first { $0.id == id }
    }

    /// "driver_safety:strict" -> "Driver Safety · Strict" from this catalog.
    func title(for name: String?) -> String? {
        guard let name, !name.isEmpty else { return nil }
        let parts = name.split(separator: ":", maxSplits: 1).map(String.init)
        guard let preset = preset(parts[0]) else { return nil }
        if parts.count > 1, parts[1] != "standard",
           let variant = preset.variants.first(where: { $0.id == parts[1] }) {
            return "\(preset.title) · \(variant.title)"
        }
        return preset.title
    }

    /// The app's own list, used only when the server publishes none.
    static func builtIn(
        _ items: [(id: String, title: String, description: String, variants: [(id: String, title: String)])],
        reason: String
    ) -> PresetCatalog {
        PresetCatalog(
            catalogVersion: nil,
            presets: items.map { item in
                CatalogPreset(
                    id: item.id, title: item.title, description: item.description,
                    variants: item.variants.map { PresetVariantInfo(id: $0.id, title: $0.title) },
                    validation: PresetValidation(status: .unknown))
            },
            source: .builtIn(reason: reason))
    }

    static let builtInBecauseOldServer =
        "Built-in list: this server doesn't publish its preset catalog, so requirements, settings and validation aren't shown."
    static let builtInBecauseUnreachable =
        "Built-in list: the preset catalog couldn't be loaded. Requirements, settings and validation aren't shown."
}

extension PresetCatalog: Decodable {
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: AnyKey.self)
        catalogVersion = c.string("catalog_version")
        presets = c.list("presets", CatalogPreset.self)
        source = .server
    }
}

extension CatalogPreset: Decodable {
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: AnyKey.self)
        guard let id = c.string("id"), !id.isEmpty else {
            throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: "preset without id"))
        }
        self.id = id
        version = c.string("version")
        title = c.string("title") ?? id.replacingOccurrences(of: "_", with: " ").capitalized
        description = c.string("description") ?? ""
        variants = c.list("variants", PresetVariantInfo.self)
        requiredCapabilities = c.strings("required_capabilities")
        optionalCapabilities = c.strings("optional_capabilities")
        hardware = c.strings("hardware")
        temporalPolicies = c.list("temporal_policies", TemporalPolicy.self)
        if case .object(let patterns)? = c.value("alert_patterns") { alertPatterns = patterns } else { alertPatterns = [:] }
        settings = c.list("settings", PresetSetting.self)
        if case .object(let compat)? = c.value("compatibility"), let group = compat["exclusive_group"]?.stringValue {
            exclusiveGroup = group
        } else {
            exclusiveGroup = nil
        }
        validation = (try? c.decodeIfPresent(PresetValidation.self, forKey: AnyKey("validation"))) ?? nil
            ?? PresetValidation(status: .unknown)
        limitations = c.strings("limitations")
    }
}

extension PresetVariantInfo: Decodable {
    init(from decoder: Decoder) throws {
        if let single = try? decoder.singleValueContainer(), let id = try? single.decode(String.self) {
            self.init(id: id, title: id.replacingOccurrences(of: "_", with: " ").capitalized)
            return
        }
        let c = try decoder.container(keyedBy: AnyKey.self)
        guard let id = c.string("id") ?? c.string("name") else {
            throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: "variant without id"))
        }
        self.init(id: id, title: c.string("title") ?? id.replacingOccurrences(of: "_", with: " ").capitalized)
    }
}

extension TemporalPolicy: Decodable {
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: AnyKey.self)
        self.init(event: c.string("event") ?? "", description: c.string("description") ?? "",
                  sustainS: c.double("sustain_s"))
    }
}

extension PresetValidation: Decodable {
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: AnyKey.self)
        status = ValidationStatus(raw: c.string("status"))
        // Evidence has no fixed shape yet: a sentence, or an object naming a
        // dataset and a metric. Reduced to lines a person can read.
        let raw = ((try? c.decodeIfPresent([JSONValue].self, forKey: AnyKey("evidence"))) ?? nil) ?? []
        evidence = raw.compactMap(Self.line)
    }

    private static func line(_ value: JSONValue) -> String? {
        switch value {
        case .string(let text): return text.isEmpty ? nil : text
        case .object(let fields):
            for key in ["summary", "description", "title", "name", "source", "dataset"] {
                if let text = fields[key]?.stringValue, !text.isEmpty { return text }
            }
            return nil
        default: return nil
        }
    }
}

extension PresetSetting: Decodable {
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: AnyKey.self)
        guard let key = c.string("key"), !key.isEmpty else {
            throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: "setting without key"))
        }
        let type = c.string("type") ?? ""
        let kind: Kind
        switch type {
        case "number", "int", "integer", "float": kind = .number
        case "bool", "boolean": kind = .bool
        case "enum", "choice": kind = .choice
        default: kind = .unsupported(type)
        }
        let options: [Option] = (((try? c.decodeIfPresent([JSONValue].self, forKey: AnyKey("options"))) ?? nil) ?? [])
            .compactMap { value in
                switch value {
                case .object(let fields):
                    guard let raw = (fields["value"] ?? fields["id"])?.stringValue else { return nil }
                    return Option(value: raw, title: fields["title"]?.stringValue ?? raw)
                default:
                    guard let raw = value.stringValue else { return nil }
                    return Option(value: raw, title: raw.replacingOccurrences(of: "_", with: " ").capitalized)
                }
            }
        self.init(
            key: key,
            title: c.string("title") ?? key.replacingOccurrences(of: "_", with: " ").capitalized,
            kind: kind, defaultValue: c.value("default"),
            min: c.double("min"), max: c.double("max"), step: c.double("step"),
            unit: c.string("unit"), options: options)
    }
}

// MARK: - Per device (GET /v1/devices/{id}/presets)

struct DevicePresetState: Equatable, Identifiable, Sendable {
    let id: String
    var available: Bool
    /// null | role_forbids_preset | capability_impaired | capability_unsupported | device_stale | planned
    var code: String?
    var reason: String?
    var active: Bool
    var validationStatus: ValidationStatus?
    var settings: [String: JSONValue]

    init(id: String, available: Bool = true, code: String? = nil, reason: String? = nil,
         active: Bool = false, validationStatus: ValidationStatus? = nil, settings: [String: JSONValue] = [:]) {
        self.id = id
        self.available = available
        self.code = code
        self.reason = reason
        self.active = active
        self.validationStatus = validationStatus
        self.settings = settings
    }
}

struct DevicePresets: Equatable, Sendable {
    var deviceID: Int?
    var catalogVersion: String?
    var activePreset: String?
    var stale: Bool
    var presets: [DevicePresetState]

    func state(for id: String) -> DevicePresetState? {
        presets.first { $0.id == id }
    }
}

extension DevicePresetState: Decodable {
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: AnyKey.self)
        guard let id = c.string("id"), !id.isEmpty else {
            throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: "preset state without id"))
        }
        var settings: [String: JSONValue] = [:]
        if case .object(let values)? = c.value("settings") { settings = values }
        self.init(
            id: id,
            // Missing means "not said": treated as available, and the POST is
            // still the authority (it refuses with a code).
            available: c.bool("available") ?? true,
            code: c.string("code"), reason: c.string("reason"),
            active: c.bool("active") ?? false,
            validationStatus: c.string("validation_status").map { ValidationStatus(raw: $0) },
            settings: settings)
    }
}

extension DevicePresets: Decodable {
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: AnyKey.self)
        deviceID = (try? c.decodeIfPresent(Int.self, forKey: AnyKey("device_id"))) ?? nil
        catalogVersion = c.string("catalog_version")
        activePreset = c.string("active_preset")
        stale = c.bool("stale") ?? false
        presets = c.list("presets", DevicePresetState.self)
    }
}

// MARK: - In words

/// What to say about a preset's validation. Never "validated" unless it was
/// validated on a device.
struct ValidationPresentation: Equatable {
    let chip: String
    let detail: String
    /// True only for `validated_on_device`; the views colour nothing else green.
    let isValidated: Bool

    static func make(_ status: ValidationStatus) -> ValidationPresentation {
        switch status {
        case .validatedOnDevice:
            return .init(chip: "Validated on device",
                         detail: "Checked on real SensorDyme hardware against labelled recordings.",
                         isValidated: true)
        case .evaluatedInDomain:
            return .init(chip: "Not yet validated",
                         detail: "Evaluated on recordings like the ones it's meant for, but not yet validated on a device.",
                         isValidated: false)
        case .benchmarkedOffline:
            return .init(chip: "Not yet validated",
                         detail: "Only measured offline on a public dataset. Not checked on a device, or in a real home or car.",
                         isValidated: false)
        case .unevaluated:
            return .init(chip: "Not yet validated",
                         detail: "Not evaluated yet. Treat its alerts as unproven.",
                         isValidated: false)
        case .unknown:
            return .init(chip: "Validation unknown",
                         detail: "This server doesn't say how far this preset has been validated.",
                         isValidated: false)
        }
    }
}

/// Whether a preset can be picked on a device, and why not.
struct PresetAvailability: Equatable {
    /// Tapping it would be refused or would run something that cannot work.
    let blocksSelection: Bool
    /// One line for the row; nil when there is nothing to say.
    let note: String?

    static let open = PresetAvailability(blocksSelection: false, note: nil)

    /// - Parameter roleTitle: the device's role in words (`status.role_title`),
    ///   used when the backend's reason is written for developers.
    static func make(_ state: DevicePresetState?, roleTitle: String? = nil) -> PresetAvailability {
        guard let state else { return .open }
        let reason = state.reason.flatMap(readableReason)
        if state.available { return .open }
        switch state.code {
        case "device_stale":
            // Not a refusal: the request is stored and applied when the device
            // reports again. Said, not blocked.
            return .init(blocksSelection: false,
                         note: "Device isn't reporting — can't check it can run this" + (reason.map { " (\($0))" } ?? ""))
        case "role_forbids_preset":
            return .init(blocksSelection: true,
                         note: reason ?? "Not available for this device's role" + (roleTitle.map { " (\($0))" } ?? ""))
        case "capability_impaired":
            return .init(blocksSelection: true,
                         note: "Needs something that isn't working right now" + (reason.map { ": \($0)" } ?? ""))
        case "capability_unsupported":
            return .init(blocksSelection: true,
                         note: "This device can't do this" + (reason.map { ": \($0)" } ?? ""))
        case "planned":
            return .init(blocksSelection: true, note: "Planned — not available yet" + (reason.map { ". \($0)" } ?? ""))
        default:
            return .init(blocksSelection: true, note: reason ?? "Not available on this device")
        }
    }
}

extension PresetAvailability {
    /// The backend's reason when it is a sentence for a person; nil when it is
    /// written for developers, e.g. "this device's role is 'driver_safety':
    /// preset 'elderly_monitoring' would stop driver_attention; allowed:
    /// driver_safety" — which the mode list used to show verbatim.
    static func readableReason(_ raw: String) -> String? {
        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.isEmpty { return nil }
        let identifier = text.range(of: "[a-z0-9]+_[a-z0-9_]+", options: .regularExpression) != nil
        if identifier || text.contains("'") && text.contains(":") || text.contains(";") { return nil }
        return text
    }
}

/// Why a POST /v1/devices/{id}/preset was refused, in words, keeping the
/// backend's own detail (it names the setting or the conflicting policy).
enum PresetActivationError {
    static func message(code: String?, detail: String) -> String {
        let detail = detail.trimmingCharacters(in: .whitespacesAndNewlines)
        func with(_ head: String) -> String { detail.isEmpty ? head + "." : "\(head): \(detail)" }
        switch code {
        case "role_forbids_preset": return with("This device's role doesn't allow this preset")
        case "draft_superseded": return with("Something newer replaced this request — review it again")
        case "incompatible_policy": return with("This can't run together with what's already active")
        case "invalid_setting": return with("A setting was refused")
        case "unknown_preset": return with("This server doesn't know that preset")
        default: return detail.isEmpty ? "The server refused the change." : detail
        }
    }
}

/// The settings sheet's logic: starting values, checks, and what is sent.
enum PresetSettingsForm {
    /// What the device runs now, else the schema default, for every setting
    /// this build can edit.
    static func initialValues(_ settings: [PresetSetting], current: [String: JSONValue]) -> [String: JSONValue] {
        var values: [String: JSONValue] = [:]
        for setting in settings {
            if let now = current[setting.key] {
                values[setting.key] = now
            } else if let fallback = setting.defaultValue {
                values[setting.key] = fallback
            } else {
                switch setting.kind {
                case .bool: values[setting.key] = .bool(false)
                case .number: values[setting.key] = .number(setting.min ?? 0)
                case .choice: if let first = setting.options.first { values[setting.key] = .string(first.value) }
                case .unsupported: break
                }
            }
        }
        return values
    }

    /// A number snapped to the setting's step and held inside its range.
    static func normalized(_ value: Double, for setting: PresetSetting) -> Double {
        var out = value
        if let step = setting.step, step > 0 {
            let origin = setting.min ?? 0
            out = origin + ((out - origin) / step).rounded() * step
            // Undo float noise from the multiply ("0.30000000000000004").
            out = (out * 1_000_000).rounded() / 1_000_000
        }
        if let lower = setting.min { out = Swift.max(lower, out) }
        if let upper = setting.max { out = Swift.min(upper, out) }
        return out
    }

    /// Why a value cannot be sent, per key. Empty when everything is valid.
    static func problems(_ settings: [PresetSetting], values: [String: JSONValue]) -> [String: String] {
        var problems: [String: String] = [:]
        for setting in settings {
            guard let value = values[setting.key] else { continue }
            switch setting.kind {
            case .number:
                guard let number = value.doubleValue else { problems[setting.key] = "Enter a number"; continue }
                if let lower = setting.min, number < lower { problems[setting.key] = "At least \(format(lower, setting.unit))" }
                if let upper = setting.max, number > upper { problems[setting.key] = "At most \(format(upper, setting.unit))" }
            case .bool:
                if value.boolValue == nil { problems[setting.key] = "Must be on or off" }
            case .choice:
                if !setting.options.contains(where: { $0.value == value.stringValue }) {
                    problems[setting.key] = "Pick one of the options"
                }
            case .unsupported:
                break
            }
        }
        return problems
    }

    /// The `settings` object sent with the activation: only keys this build
    /// can edit, never a type it could not render.
    static func payload(_ settings: [PresetSetting], values: [String: JSONValue]) -> [String: JSONValue] {
        var out: [String: JSONValue] = [:]
        for setting in settings {
            if case .unsupported = setting.kind { continue }
            if let value = values[setting.key] { out[setting.key] = value }
        }
        return out
    }

    /// Whether the device's reported settings match what was sent.
    static func confirmed(sent: [String: JSONValue], reported: [String: JSONValue]) -> Bool {
        sent.allSatisfy { key, value in
            guard let now = reported[key] else { return false }
            if let a = value.doubleValue, let b = now.doubleValue { return abs(a - b) < 1e-9 }
            return now == value
        }
    }

    static func format(_ number: Double, _ unit: String?) -> String {
        let text = number == number.rounded() ? String(Int(number)) : String(format: "%g", number)
        guard let unit, !unit.isEmpty else { return text }
        return "\(text) \(unit)"
    }
}
