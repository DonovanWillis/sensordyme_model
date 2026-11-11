import Foundation

// Run with ios/Tests/run.sh

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

func decode<T: Decodable>(_ type: T.Type, _ json: String) -> T? {
    do { return try JSONDecoder().decode(type, from: Data(json.utf8)) } catch {
        print("  decode error: \(error)")
        return nil
    }
}

print("the documented contract decodes:")
let json = """
{"registry_version": "2026-11-11.2", "tiers": ["exists_in_code"], "capabilities": [
  {"capability": "phone_detection", "kind": "objects", "model": "damo", "model_deployment": "shipping",
   "decoder": "legacy", "deployed_target": "hailo8l_int8", "tier": "image_level_benchmark",
   "device_verified": false, "production_validated": false, "evidence": ["x"], "experiment_evidence": []},
  {"capability": "hand_box_detection", "model_deployment": "candidate", "tier": "exists_in_code"},
  {"capability": "odd", "tier": "something_new", "production_validated": "yes"}
]}
"""
let response = decode(CapabilityTiersResponse.self, json)
check(response?.capabilities.count == 3, "three entries")
check(response?.capabilities[0].tier == .imageLevelBenchmark, "known tier")
check(response?.capabilities[2].tier == .unknown, "unknown tier is .unknown")
check(response?.capabilities[2].productionValidated == false, "a malformed flag decodes as false")
check(response?.capabilities[0].kind == "objects", "kind decodes")
check(response?.entries(forKinds: ["objects"]).map(\.capability) == ["phone_detection"], "filtered by kind")
check(response?.entries(forKinds: ["face"]).isEmpty == true, "no entries for an unrelated kind")

print("\nonly a shipping, production_permitted, production_validated capability is validated:")
for tier in CapabilityTier.allCases {
    for deployment in ["shipping", "candidate", "dev_only", nil] as [String?] {
        for flag in [false, true] {
            let entry = CapabilityTierEntry(capability: "c", tier: tier, modelDeployment: deployment,
                                            deviceVerified: flag, productionValidated: flag)
            let shown = CapabilityTierPresentation.make(entry)
            let expected = tier == .productionPermitted && deployment == "shipping" && flag
            if shown.isProductionValidated != expected {
                check(false, "\(tier.rawValue) / \(deployment ?? "nil") / \(flag)")
            }
            if !expected && shown.chip.lowercased().contains("validated for production") {
                check(false, "chip overclaims for \(tier.rawValue) / \(deployment ?? "nil")")
            }
        }
    }
}
check(true, "every tier x deployment x flag combination")

print("\nresearch capabilities say so, whatever tier the server sends:")
let research = CapabilityTierEntry(capability: "hand_box_detection", tier: .productionPermitted,
                                   modelDeployment: "candidate", deviceVerified: true, productionValidated: true)
check(CapabilityTierPresentation.make(research).chip == "Research only", "candidate model is research only")
check(!CapabilityTierPresentation.make(research).isProductionValidated, "never validated")

print("\ntoday's tiers read as not validated:")
for tier in [CapabilityTier.syntheticTests, .imageLevelBenchmark, .existsInCode] {
    let shown = CapabilityTierPresentation.make(CapabilityTierEntry(capability: "c", tier: tier, modelDeployment: "shipping"))
    check(shown.chip == "Not yet validated" && !shown.isProductionValidated, tier.rawValue)
}

print(failures == 0 ? "\nall passed" : "\n\(failures) FAILED")
exit(failures == 0 ? 0 : 1)
