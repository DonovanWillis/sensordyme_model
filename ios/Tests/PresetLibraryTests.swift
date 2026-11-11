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

print("the catalog decodes the documented contract:")
let catalogJSON = """
{"catalog_version": "2026.11.1", "presets": [
  {"id": "driver_safety", "version": "3", "title": "Driver Safety", "description": "Eyes and phone",
   "variants": [{"id": "standard", "title": "Standard"}, {"id": "strict", "title": "Strict"}],
   "required_capabilities": ["face", "gaze"], "optional_capabilities": ["phone_use"], "hardware": ["ir_camera"],
   "temporal_policies": [{"event": "eyes_off_road", "description": "Eyes off the road", "sustain_s": 2}],
   "alert_patterns": {"eyes_off_road": "soft"},
   "settings": [
     {"key": "sensitivity", "title": "Sensitivity", "type": "number", "default": 0.6, "min": 0.3, "max": 0.9, "step": 0.1},
     {"key": "chime", "title": "Chime", "type": "bool", "default": true},
     {"key": "profile", "title": "Profile", "type": "enum", "default": "car", "options": ["car", "truck"]},
     {"key": "curve", "title": "Curve", "type": "spline"}
   ],
   "compatibility": {"exclusive_group": "primary"},
   "validation": {"status": "benchmarked_offline", "evidence": ["StateFarm accuracy 0.91", {"dataset": "DMD"}]},
   "limitations": ["Sunglasses hide the eyes"]},
  {"title": "No id here"},
  {"id": "pool_safety", "validation": {"status": "something_new"}}
]}
"""
let catalog = decode(PresetCatalog.self, catalogJSON)
check(catalog?.catalogVersion == "2026.11.1", "catalog_version")
check(catalog?.presets.count == 2, "a preset without an id is dropped, not fatal")
check(catalog?.source == .server, "decoded catalogs are from the server")
let driver = catalog?.preset("driver_safety")
check(driver?.variants.map(\.id) == ["standard", "strict"], "variants")
check(driver?.requiredCapabilities == ["face", "gaze"], "required capabilities")
check(driver?.hardware == ["ir_camera"], "hardware")
check(driver?.temporalPolicies.first?.sustainS == 2, "temporal policy sustain_s")
check(driver?.exclusiveGroup == "primary", "compatibility.exclusive_group")
check(driver?.validation.status == .benchmarkedOffline, "validation status")
check(driver?.validation.evidence == ["StateFarm accuracy 0.91", "DMD"], "evidence reduced to lines")
check(driver?.limitations == ["Sunglasses hide the eyes"], "limitations")
check(driver?.settings.count == 4, "every setting decodes")
check(driver?.settings.first?.kind == .number && driver?.settings.first?.step == 0.1, "number setting")
check(driver?.settings[2].options.map(\.value) == ["car", "truck"], "enum options from bare strings")
check(driver?.settings[3].kind == .unsupported("spline"), "an unknown setting type is kept, as unsupported")
let pool = catalog?.preset("pool_safety")
check(pool?.title == "Pool Safety", "a missing title falls back to the id")
check(pool?.validation.status == .unknown, "an unknown validation value reads as unknown")
check(catalog?.title(for: "driver_safety:strict") == "Driver Safety · Strict", "title with variant")
check(catalog?.title(for: "driver_safety:standard") == "Driver Safety", "standard variant is not named")
check(catalog?.title(for: "nope") == nil, "unknown preset has no catalog title")
check(decode(PresetCatalog.self, "{}")?.presets.isEmpty == true, "an empty object is an empty catalog, not an error")

print("\nper-device availability decodes:")
let deviceJSON = """
{"device_id": 3, "catalog_version": "2026.11.1", "active_preset": "driver_safety:strict", "stale": false,
 "presets": [
   {"id": "driver_safety", "available": true, "code": null, "reason": null, "active": true,
    "validation_status": "benchmarked_offline", "settings": {"sensitivity": 0.7, "chime": false}},
   {"id": "baby_monitoring", "available": false, "code": "role_forbids_preset", "reason": "Driver role", "active": false},
   {"id": "elderly_monitoring", "available": false, "code": "capability_impaired", "reason": "pose: model missing"},
   {"id": "security_monitoring", "available": false, "code": "device_stale"},
   {"id": "pool_safety", "available": false, "code": "planned"},
   {"id": "x", "available": false, "code": "capability_unsupported", "reason": "no IR camera"},
   {"id": "y", "available": false, "code": "brand_new_code"}
 ]}
"""
let device = decode(DevicePresets.self, deviceJSON)
check(device?.activePreset == "driver_safety:strict", "active_preset")
check(device?.state(for: "driver_safety")?.settings["sensitivity"] == .number(0.7), "current settings")
check(device?.state(for: "driver_safety")?.validationStatus == .benchmarkedOffline, "per-device validation status")
check(decode(DevicePresets.self, #"{"presets": [{"id": "a"}]}"#)?.presets.first?.available == true,
      "a missing `available` is not a refusal (the POST stays the authority)")

func availability(_ id: String) -> PresetAvailability { PresetAvailability.make(device?.state(for: id)) }
check(availability("driver_safety") == .open, "available is open")
check(availability("baby_monitoring") == PresetAvailability(blocksSelection: true, note: "Driver role"),
      "role refusal blocks with the backend's reason")
check(availability("elderly_monitoring").blocksSelection
      && availability("elderly_monitoring").note?.contains("pose: model missing") == true,
      "impaired capability blocks and says which")
check(!availability("security_monitoring").blocksSelection
      && availability("security_monitoring").note?.contains("isn't reporting") == true,
      "a stale device is said, not blocked (the request queues)")
check(availability("pool_safety").blocksSelection && availability("pool_safety").note?.hasPrefix("Planned") == true,
      "planned blocks and says planned")
check(availability("x").note == "This device can't do this: no IR camera", "unsupported capability")
check(availability("y").blocksSelection && availability("y").note == "Not available on this device",
      "an unknown code still blocks, with a generic reason")
check(PresetAvailability.make(nil) == .open, "no per-device answer is open (old server)")

print("\nvalidation is never overstated:")
for status in [ValidationStatus.unevaluated, .benchmarkedOffline, .evaluatedInDomain, .unknown] {
    let p = ValidationPresentation.make(status)
    check(!p.isValidated && !p.chip.hasPrefix("Validated"), "\(status) is not shown as validated (\(p.chip))")
}
check(ValidationPresentation.make(.validatedOnDevice).isValidated, "only validated_on_device is validated")
check(ValidationPresentation.make(.benchmarkedOffline).detail.contains("offline"), "benchmark says offline")

print("\nsettings form:")
let settings = driver?.settings ?? []
let initial = PresetSettingsForm.initialValues(settings, current: ["sensitivity": .number(0.7)])
check(initial["sensitivity"] == .number(0.7), "current value wins over the default")
check(initial["chime"] == .bool(true) && initial["profile"] == .string("car"), "defaults fill the rest")
check(initial["curve"] == nil, "an unsupported setting gets no value")
let sensitivity = settings[0]
check(PresetSettingsForm.normalized(0.74, for: sensitivity) == 0.7, "snaps to step")
check(PresetSettingsForm.normalized(5, for: sensitivity) == 0.9, "clamps to max")
check(PresetSettingsForm.normalized(-1, for: sensitivity) == 0.3, "clamps to min")
check(PresetSettingsForm.problems(settings, values: initial).isEmpty, "valid values have no problems")
let bad = PresetSettingsForm.problems(settings, values: ["sensitivity": .number(2), "profile": .string("boat"),
                                                         "chime": .string("yes")])
check(bad["sensitivity"]?.hasPrefix("At most") == true, "out of range number")
check(bad["profile"] != nil, "an option that isn't offered")
check(bad["chime"] != nil, "a bool that isn't a bool")
let payload = PresetSettingsForm.payload(settings, values: initial.merging(["curve": .string("x")]) { a, _ in a })
check(payload.keys.sorted() == ["chime", "profile", "sensitivity"], "unsupported types are never sent")
check(PresetSettingsForm.confirmed(sent: ["sensitivity": .number(0.7)], reported: ["sensitivity": .number(0.7000000001)]),
      "a reported number within float noise confirms")
check(!PresetSettingsForm.confirmed(sent: ["chime": .bool(true)], reported: [:]), "a missing key does not confirm")
check(PresetSettingsForm.format(2, "s") == "2 s" && PresetSettingsForm.format(0.5, nil) == "0.5", "number format")

print("\nrefusals keep the backend's detail:")
check(PresetActivationError.message(code: "invalid_setting", detail: "sensitivity must be <= 0.9")
      == "A setting was refused: sensitivity must be <= 0.9", "invalid_setting")
check(PresetActivationError.message(code: "incompatible_policy", detail: "pool rules need security")
      .contains("pool rules need security"), "incompatible_policy")
check(PresetActivationError.message(code: "role_forbids_preset", detail: "").hasSuffix("."), "no detail still reads")
check(PresetActivationError.message(code: nil, detail: "boom") == "boom", "uncoded keeps the detail")

print("\nan endpoint the server doesn't have:")
check(ServerSupport.endpointMissing(status: 404, code: nil), "uncoded 404 is a missing endpoint")
check(ServerSupport.endpointMissing(status: 405, code: nil), "405 too")
check(!ServerSupport.endpointMissing(status: 404, code: "device_not_found"), "a coded 404 is a real answer")
check(!ServerSupport.endpointMissing(status: 404, code: "rule_not_found"), "rule_not_found is a real answer")
check(!ServerSupport.endpointMissing(status: 500, code: nil), "500 is not 'unsupported'")

print("\nthe built-in fallback is labelled and claims nothing:")
let builtIn = PresetCatalog.builtIn([(id: "driver_safety", title: "Driver Safety", description: "d",
                                     variants: [(id: "standard", title: "Standard")])],
                                    reason: PresetCatalog.builtInBecauseOldServer)
check(builtIn.isBuiltIn, "built-in is marked")
check(builtIn.presets.first?.validation.status == .unknown, "built-in presets claim no validation")
check(builtIn.presets.first?.settings.isEmpty == true, "built-in presets offer no settings")

print("\nJSONValue round trip:")
let value = decode(JSONValue.self, #"{"a": [1, true, "x", null]}"#)
check(value == .object(["a": .array([.number(1), .bool(true), .string("x"), .null])]), "decodes")
check((value?.foundationObject as? [String: Any]) != nil, "converts for a request body")

print(failures == 0 ? "\nall passed" : "\n\(failures) FAILED")
exit(failures == 0 ? 0 : 1)
