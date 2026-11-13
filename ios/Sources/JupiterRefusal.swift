import Foundation

/// Why Ask Jupiter will not set something up, from the typed `code` the backend
/// puts on every `unsupported` entry (jupiter/policy.py UNSUPPORTED_CODES,
/// M8.4). One headline per reason, so a refusal reads the same wherever it
/// comes from -- the policy guard, the grammar, a language model, validation --
/// and the app never has to parse the backend's prose to say why.
///
/// Foundation-only (ios/Tests/JupiterRefusalTests.swift).
enum JupiterRefusal {
    /// Refused by policy: nothing added to the device would make it possible.
    static let policyCodes: Set<String> = [
        "identity", "emotion", "intent", "audio", "intoxication", "medical", "experimental_signal",
    ]

    /// The one-line reason, as a heading. `support` (needs_training_data, ...)
    /// refines a "not detectable" answer and is the whole answer from an older
    /// backend that sends no code.
    static func headline(code: String?, support: String?) -> String {
        switch code {
        case "identity": return "Never identifies people"
        case "emotion": return "Can't read feelings or mood"
        case "intent": return "Never judges intent"
        case "audio": return "No microphone: sounds can't be heard"
        case "intoxication": return "Can't judge intoxication"
        case "medical": return "Not a medical monitor"
        case "repetition_window": return "Can't count repeats over a time window"
        case "experimental_signal": return "Experimental: recorded, never an alert"
        case "needs_area": return "Needs an area you draw"
        case "needs_second_object": return "Needs a second object"
        case "invalid_rule": return "Not a rule the device can run"
        case "not_beside_mode": return "Not turned on with a mode switch"
        case "interpreter_unavailable": return "Couldn't be read, so nothing was guessed"
        case "no_target": return "Names nothing the camera can watch"
        default: return bySupport(support)
        }
    }

    private static func bySupport(_ support: String?) -> String {
        switch support {
        case "needs_training_data": return "The camera doesn't recognise this yet"
        case "needs_model": return "Needs an extra model"
        case "needs_hardware": return "Needs extra hardware"
        default: return "Not possible from the camera alone"
        }
    }

    /// How `requires` is introduced: what it would take, or (for a policy
    /// refusal, which nothing enables) why not.
    static func requiresPrefix(code: String?) -> String {
        code.map { policyCodes.contains($0) } == true ? "Why" : "To enable"
    }

    /// True when the request can never be set up, whatever is added.
    static func isPolicy(code: String?) -> Bool { code.map { policyCodes.contains($0) } ?? false }
}

/// The typed policy the backend returns for each previewed rule (`policies`,
/// jupiter/policy.py Policy.as_dict): whether its area is drawn, and in plain
/// language what it will and will not watch. Optional everywhere: an older
/// backend sends none, and the review screen then says only what it did before.
struct AIPolicyPreview: Codable, Equatable, Sendable {
    struct Zone: Codable, Equatable, Sendable {
        let name: String
        let drawn: Bool
    }

    struct Explanation: Codable, Equatable, Sendable {
        let summary: String?
        let monitors: [String]?
        let does_not_monitor: [String]?
        let status: String?
    }

    let rule_id: String?
    let status: String?
    let zone: Zone?
    let explanation: Explanation?

    /// The policy for a rule: by the id the preview gave it, else by position
    /// when the two lists line up one to one.
    static func matching(_ policies: [AIPolicyPreview]?, ruleID: String?, index: Int, ruleCount: Int)
        -> AIPolicyPreview? {
        guard let policies, !policies.isEmpty else { return nil }
        if let ruleID, let byID = policies.first(where: { $0.rule_id == ruleID }) { return byID }
        return policies.count == ruleCount && policies.indices.contains(index) ? policies[index] : nil
    }
}
