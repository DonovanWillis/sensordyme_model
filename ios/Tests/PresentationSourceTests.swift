import Foundation

// Run with ios/Tests/run.sh
// run.sh deps: PresetLibrary InstrumentState ConnectionTruth DevicePresence DeviceStatus PerceptionStatus CapabilityTier StreamLiveness MJPEGStream TrustedDevices LocalAccess BLEProofOfPossession LocalEvents EventModel AlertSound NotificationPolicy LocalDeviceClient
//
// M13.1: presentation mode fed by a paired camera opened directly (the M12
// local path, the account server possibly down). What these pin: the mode
// chips are the camera's own list, limited to what its role allows, and a chip
// is selected only for the mode the CAMERA reports; Jupiter's rules are said
// as running (from the device's monitoring mode) but never given words the
// phone doesn't have; the monitoring word is the camera's fresh one, never
// "protecting" from the cache; an alert is evidence from the last minute only;
// and the live why line can name its history row.

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

let driverVariants: [LocalPresets.Variant] = [
    .init(name: "standard", title: "Standard"), .init(name: "strict", title: "Strict"),
    .init(name: "teen_driver", title: "Teen Driver"),
]
func presets(_ active: String?) -> LocalPresets {
    LocalPresets(active: active, presets: [
        .init(name: "driver_safety", title: "Driver Safety", variants: driverVariants),
        .init(name: "baby_monitoring", title: "Baby Monitoring",
              variants: [.init(name: "standard", title: "Standard")]),
        .init(name: "occupancy", title: "Occupancy", variants: []),
    ])
}
let driverRole = LocalDeviceState.Role(name: "driver_safety", title: "Driver Safety", allowedPresets: ["driver_safety"])

print("== the mode chips: the camera's list, its role, what it reports")
let variantChips = LocalPresentation.options(presets("driver_safety"), role: driverRole, rule: .none)
check(variantChips.map(\.id) == ["driver_safety:standard", "driver_safety:strict", "driver_safety:teen_driver"],
      "a role that allows one preset offers its variants: \(variantChips.map(\.id))")
check(variantChips.filter(\.selected).map(\.id) == ["driver_safety:standard"],
      "the bare preset the camera reports is its first variant")
check(variantChips.map(\.short) == ["Standard", "Strict", "Teen Driver"], "in the camera's own titles")
let strictChips = LocalPresentation.options(presets("driver_safety:strict"), role: driverRole, rule: .none)
check(strictChips.filter(\.selected).map(\.id) == ["driver_safety:strict"], "the camera reports Strict: Strict selected")
check(strictChips.filter(\.selected).count == 1, "only one mode is ever selected")
let anyRole = LocalPresentation.options(presets("occupancy"), role: nil, rule: .none)
check(anyRole.map(\.id) == ["driver_safety", "baby_monitoring", "occupancy"],
      "a role that does not limit presets offers all of them")
check(anyRole.filter(\.selected).map(\.id) == ["occupancy"], "the running one selected")
let replaced = LocalPresentation.options(presets("driver_safety"), role: driverRole,
                                         rule: .runningUnnamed(replacesModes: true))
check(!replaced.contains(where: \.selected), "Jupiter's rules replacing the mode: no preset is shown as running")
check(LocalPresentation.options(nil, role: driverRole, rule: .none).isEmpty, "no list read yet: no chips")
check(LocalPresentation.options(presets(nil), role: driverRole, rule: .none).allSatisfy { !$0.selected },
      "the camera reports no mode: nothing selected")

print("== the mode title, in the camera's words")
check(LocalPresentation.modeTitle(presets("driver_safety:strict")) == "Driver Safety · Strict", "variant title")
check(LocalPresentation.modeTitle(presets("driver_safety:standard")) == "Driver Safety", "Standard is the plain name")
check(LocalPresentation.modeTitle(nil) == "Mode not read from the camera yet", "not read: said")
check(LocalPresentation.modeTitle(presets("mystery")) == "mystery", "an unknown preset is named as the camera named it")

print("== Jupiter's rules: from the device's monitoring mode, never with invented words")
check(PresentationRule.fromDeviceMode("ai_custom") == .runningUnnamed(replacesModes: true), "ai_custom replaces")
check(PresentationRule.fromDeviceMode("standard_plus_custom") == .runningUnnamed(replacesModes: false), "additive")
check(PresentationRule.fromDeviceMode("standard") == .none && PresentationRule.fromDeviceMode(nil) == .none,
      "standard / unknown: none")

print("== the monitoring word: the camera's fresh one")
let protecting = LocalPresentation.instrument(.live(protection: "protecting"), alertActive: false)
check(protecting.kind == .ready && protecting.word == "Protecting" && protecting.live, "protecting, live")
check(LocalPresentation.instrument(.live(protection: "stopped"), alertActive: false).kind == .stopped, "stopped")
check(LocalPresentation.instrument(.live(protection: "not_protecting"), alertActive: false).word == "Not protecting",
      "not protecting")
check(LocalPresentation.instrument(.live(protection: "protecting"), alertActive: true).kind == .alert, "an alert")
for presence in [TrustedDevices.Presence.notReachable(lastSeen: Date()), .searching, .needsPairing, .awaitingDevice] {
    let i = LocalPresentation.instrument(presence, alertActive: false)
    check(!i.live && i.kind != .ready && i.word != "Protecting", "not live: never protecting (\(i.word))")
}
check(LocalPresentation.presence(.live(protection: nil)) == "Direct connection", "how it is reached")
check(LocalPresentation.presence(.notReachable(lastSeen: nil)) == "Offline", "or that it isn't")

print("== alerts and the history")
let now = Date(timeIntervalSince1970: 1_800_000_000)
func event(_ seq: Int, _ type: String, severity: String = "loud", age: Double?, sync: String = "pending",
           simulated: Bool = true) -> LocalEvent {
    LocalEvent(seq: seq, eventKey: "k\(seq)", type: type, timestamp: "2026-11-16T10:00:00+00:00", severity: severity,
               signals: "head turned away 3.4 s", simulated: simulated, ageS: age, sync: sync)
}
let loud: (EventItem) -> Bool = { $0.severity == "loud" }
let history = [event(3, "monitoring_ready", severity: "log", age: 2), event(2, "distraction", age: 10, sync: "sent"),
               event(1, "phone_use", age: 300)]
check(LocalPresentation.latestAlert(history, fetchedAt: now, now: now, isAlert: loud)?.eventKey == "k2",
      "the newest alert in the last minute (a log record is not one)")
check(LocalPresentation.latestAlert(history, fetchedAt: now, now: now.addingTimeInterval(55), isAlert: loud) == nil,
      "a minute later it is history, not an alert")
check(LocalPresentation.latestAlert([event(4, "distraction", age: nil)], fetchedAt: now, now: now, isAlert: loud) == nil,
      "a row from an earlier power-on (no age) is never news")
let notes = LocalPresentation.historyNotes(history)
check(notes["k2"] == "In the camera's history · Sent to your account", "a row's note: \(notes["k2"] ?? "nil")")
check(notes["k3"] == "In the camera's history · Not yet backed up", "pending")

print("== the whole source for a paired camera")
let url = URL(string: "http://127.0.0.1:8899")!
let state = LocalDeviceState(module: "driver_attention", alertActive: false,
                             policy: .init(protection: .init(state: "protecting"), mode: "standard", role: driverRole))
let src = LocalPresentation.source(name: "Car", hardwareID: "SD-1", url: url, presence: .live(protection: "protecting"),
                                   state: state, presets: presets("driver_safety"), events: history.reversed(),
                                   fetchedAt: now, now: now, pending: nil, failure: nil, isAlert: loud)
check(src.picture == .live(url) && src.managedLocally && src.hardwareID == "SD-1", "the local stream, by its camera id")
check(src.instrument.kind == .alert && src.latestAlert?.eventKey == "k2", "a fresh alert in the camera's history")
check(src.lastDriverEvent?.eventKey == "k2", "the newest driver event, whatever the page order")
check(src.showsDriverPanel, "driver safety: the distracted-driving panel")
check(src.rule == .none && src.options.count == 3 && src.optionsNote == nil, "modes offered, no rule")
check(src.modeTitle == "Driver Safety" && src.presence == "Direct connection", "title and presence")

let ruled = LocalPresentation.source(
    name: "Car", hardwareID: "SD-1", url: url, presence: .live(protection: "protecting"),
    state: LocalDeviceState(policy: .init(protection: .init(state: "protecting"), mode: "ai_custom")),
    presets: presets("driver_safety"), events: [], fetchedAt: now, now: now, pending: nil, failure: nil, isAlert: loud)
check(ruled.rule == .runningUnnamed(replacesModes: true) && !ruled.showsDriverPanel,
      "Jupiter's rules replacing the mode: said, and the driver panel steps aside")
check(ruled.instrument.kind == .ready && ruled.latestAlert == nil, "no alert evidence: no alert")

let gone = LocalPresentation.source(name: "Car", hardwareID: "SD-1", url: nil, presence: .notReachable(lastSeen: nil),
                                    state: state, presets: presets("driver_safety"), events: history, fetchedAt: now,
                                    now: now, pending: nil, failure: nil, isAlert: loud)
check(gone.picture == .none && gone.options.isEmpty, "not answering: no picture, no chips")
check(gone.optionsNote == "The camera's modes appear once it answers on this network.", "and it says why")
check(gone.instrument.kind == .offline && gone.latestAlert == nil, "an old reading is not an alert now")
check(gone.rule == .none, "nor evidence of a rule")

print("== the account's option logic is the same function")
let catalog = PresetCatalog(catalogVersion: nil, presets: [
    CatalogPreset(id: "driver_safety", title: "Driver Safety",
                  variants: [PresetVariantInfo(id: "standard", title: "Standard"), PresetVariantInfo(id: "strict", title: "Strict")]),
])
let account = PresentationModeOptions.make(catalog: catalog, activePreset: "driver_safety:strict@sustain=2",
                                           customReplacesModes: false, isAvailable: { _ in true })
check(account.filter(\.selected).map(\.id) == ["driver_safety:strict"], "settings after @ do not change the mode")

print(failures == 0 ? "\nall passed" : "\n\(failures) FAILED")
exit(failures == 0 ? 0 : 1)
