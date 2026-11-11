import Foundation

// Product capability tiers as the BACKEND publishes them (GET /v1/capability-tiers,
// derived from core/model_registry.yaml; docs/MODEL_REGISTRY.md "Capability tiers"),
// and how the app puts them into words. Foundation only, so
// ios/Tests/CapabilityTierTests.swift can pin it down.
//
// The one rule: the app says "validated for production" only when the backend's
// `production_validated` is true AND the tier is production_permitted AND the
// model is shipping. A research or experiment-only capability can never read as
// validated, whatever string arrives.

/// The tier ladder, lowest first. `unknown` is anything this build does not know.
enum CapabilityTier: String, Equatable, Sendable, CaseIterable {
    case existsInCode = "exists_in_code"
    case syntheticTests = "synthetic_tests"
    case imageLevelBenchmark = "image_level_benchmark"
    case videoEvidence = "video_evidence"
    case deviceVerified = "device_verified"
    case productionPermitted = "production_permitted"
    case unknown

    init(raw: String?) {
        self = raw.flatMap(CapabilityTier.init(rawValue:)) ?? .unknown
    }
}

/// One entry of GET /v1/capability-tiers. Unknown or missing fields decode to
/// the most conservative value.
struct CapabilityTierEntry: Decodable, Equatable, Sendable {
    var capability: String
    /// objects | face | hands | pose: which perception capability it rests on.
    var kind: String?
    var tier: CapabilityTier
    var modelDeployment: String?
    var deviceVerified: Bool
    var productionValidated: Bool

    enum CodingKeys: String, CodingKey {
        case capability, tier, kind
        case modelDeployment = "model_deployment"
        case deviceVerified = "device_verified"
        case productionValidated = "production_validated"
    }

    init(capability: String, tier: CapabilityTier, modelDeployment: String? = nil,
         deviceVerified: Bool = false, productionValidated: Bool = false, kind: String? = nil) {
        self.capability = capability
        self.kind = kind
        self.tier = tier
        self.modelDeployment = modelDeployment
        self.deviceVerified = deviceVerified
        self.productionValidated = productionValidated
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        capability = (try? c.decodeIfPresent(String.self, forKey: .capability)) ?? ""
        kind = try? c.decodeIfPresent(String.self, forKey: .kind)
        tier = CapabilityTier(raw: try? c.decodeIfPresent(String.self, forKey: .tier))
        modelDeployment = try? c.decodeIfPresent(String.self, forKey: .modelDeployment)
        deviceVerified = (try? c.decodeIfPresent(Bool.self, forKey: .deviceVerified)) ?? false
        productionValidated = (try? c.decodeIfPresent(Bool.self, forKey: .productionValidated)) ?? false
    }
}

struct CapabilityTiersResponse: Decodable, Equatable, Sendable {
    var capabilities: [CapabilityTierEntry]

    /// The product capabilities resting on any of `kinds` (a preset's
    /// required and optional perception capabilities), most validated last.
    func entries(forKinds kinds: [String]) -> [CapabilityTierEntry] {
        capabilities.filter { $0.kind.map(kinds.contains) ?? false }
    }
}

/// What to say about one capability's tier.
struct CapabilityTierPresentation: Equatable {
    let chip: String
    let detail: String
    /// True only for a shipping model at production_permitted that the backend
    /// also marks production_validated. The views colour nothing else green.
    let isProductionValidated: Bool

    static func make(_ entry: CapabilityTierEntry) -> CapabilityTierPresentation {
        let shipping = entry.modelDeployment == "shipping"
        if !shipping {
            return .init(chip: "Research only",
                         detail: "An experiment. Not part of what this device runs.",
                         isProductionValidated: false)
        }
        switch entry.tier {
        case .productionPermitted where entry.productionValidated:
            return .init(chip: "Validated for production",
                         detail: "Verified on SensorDyme hardware and approved for use.",
                         isProductionValidated: true)
        case .productionPermitted, .deviceVerified:
            return .init(chip: "Verified on device",
                         detail: "Checked on SensorDyme hardware. Not yet approved for production.",
                         isProductionValidated: false)
        case .videoEvidence:
            return .init(chip: "Not yet validated",
                         detail: "Measured on recorded video, not yet on a device.",
                         isProductionValidated: false)
        case .imageLevelBenchmark:
            return .init(chip: "Not yet validated",
                         detail: "Benchmarked on public still images only, not on video or a device.",
                         isProductionValidated: false)
        case .syntheticTests:
            return .init(chip: "Not yet validated",
                         detail: "Only checked with simulated inputs. Treat its alerts as unproven.",
                         isProductionValidated: false)
        case .existsInCode:
            return .init(chip: "Not yet validated",
                         detail: "Implemented but not measured.",
                         isProductionValidated: false)
        case .unknown:
            return .init(chip: "Validation unknown",
                         detail: "This server doesn't say how far this has been checked.",
                         isProductionValidated: false)
        }
    }
}
