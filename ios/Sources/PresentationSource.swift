import Foundation

// MARK: - What presentation mode shows, from whichever screen opened it (M13.1)
//
// Presentation mode (PresentationMode.swift) is one view. It is fed either by
// the account device screen (DeviceDetailView) or by a paired camera opened
// directly on the local network (LocalDeviceView, M12, with the account server
// possibly down). Each screen builds a `PresentationSource` from its OWN state
// on every update and hands presentation a closure for the mode switch (the
// account's setMode, or the M12 signed local command). So presentation never
// decides anything and never disagrees with the screen under it.
//
// What a direct connection cannot know is said, not invented: Jupiter's rule
// wording lives in the account (`PresentationRule.runningUnnamed`), and the
// modes are the camera's own list, filtered by the role it reports.
//
// Foundation-only, so ios/Tests/PresentationSourceTests.swift runs it.

/// One mode the switch offers: a preset, or a variant of the running preset.
struct PresentationModeOption: Identifiable, Equatable {
    /// What the switch sends: "driver_safety" or "driver_safety:strict".
    let id: String
    /// "Driver Safety · Strict": what VoiceOver says.
    let title: String
    /// "Strict": what the chip says (the mode's name is in the header).
    let short: String
    let base: String
    /// The device reports this mode running (never the tap).
    let selected: Bool
}

enum PresentationModeOptions {
    /// The modes to offer. The presets this device can run; when that is only
    /// the running one (a role allows one mode, e.g. a driver-safety device),
    /// its variants instead, so there is still a real choice to make.
    static func make(catalog: PresetCatalog, activePreset: String?, customReplacesModes: Bool,
                     isAvailable: (String) -> Bool) -> [PresentationModeOption] {
        let active = PresetName.parse(activePreset)
        let available = catalog.presets.filter { isAvailable($0.id) }
        if available.count == 1, let only = available.first, !only.variants.isEmpty {
            let current = active.base == only.id && !customReplacesModes
                ? (active.variant ?? only.variants.first?.id) : nil
            return only.variants.map { variant in
                PresentationModeOption(id: "\(only.id):\(variant.id)", title: "\(only.title) · \(variant.title)",
                                       short: variant.title, base: only.id, selected: current == variant.id)
            }
        }
        return available.map { preset in
            PresentationModeOption(id: preset.id, title: preset.title, short: preset.title, base: preset.id,
                                   selected: active.base == preset.id && !customReplacesModes)
        }
    }
}

/// Jupiter's rules, as far as the screen that opened presentation knows them.
enum PresentationRule: Equatable {
    case none
    /// The account's words for the rule the device runs.
    case running(title: String, summary: String, replacesModes: Bool)
    /// The device reports Jupiter's rules running (its monitoring mode), but
    /// their wording is kept by the account, which this screen cannot read.
    case runningUnnamed(replacesModes: Bool)

    var replacesModes: Bool {
        switch self {
        case .none: false
        case .running(_, _, let replaces), .runningUnnamed(let replaces): replaces
        }
    }

    /// From the device's own monitoring mode (core/monitoring_mode.py).
    static func fromDeviceMode(_ mode: String?) -> PresentationRule {
        switch mode {
        case "ai_custom": .runningUnnamed(replacesModes: true)
        case "standard_plus_custom": .runningUnnamed(replacesModes: false)
        default: .none
        }
    }

    /// What presentation says for an unnamed rule.
    static let unnamedTitle = "Rules set by Jupiter are running"
    static let unnamedDetail = "Their wording is kept by your account, so a direct connection can't show it."
}

/// The driver-monitoring event types (the driver_safety role's alerts).
enum DriverEvents {
    static let types: Set<String> = [
        "distraction", "eyes_down", "eyes_up", "phone_use", "drowsiness", "attention_lost",
    ]

    /// The newest driver event in a newest-first history (not an experimental note).
    static func newest(_ events: [EventItem]) -> EventItem? {
        events.first { types.contains($0.type) && !$0.isExperimental }
    }
}

struct PresentationSource {
    enum Picture: Equatable {
        /// The demo account: the simulated feed.
        case demo
        /// The camera's live stream at this address.
        case live(URL)
        /// No picture on this iPhone (and the placeholder says why).
        case none
    }

    var deviceName: String
    /// The camera's SD- id, so the stream uses its local session.
    var hardwareID: String?
    /// The address comes from the paired-camera connection (LocalAccessCenter),
    /// not from the linked camera's DeviceLink.
    var managedLocally = false
    /// What the device reports running ("driver_safety:strict"), nil if unknown.
    var activePreset: String?
    var picture: Picture
    var instrument: InstrumentState
    /// "Online", "Direct connection" ...: how this iPhone reaches the device.
    var presence: String
    var modeTitle: String
    var rule: PresentationRule = .none
    var options: [PresentationModeOption]
    /// Said instead of the chips when there are none to offer.
    var optionsNote: String? = nil
    /// A switch in flight (its id or base), and the last one's failure.
    var pending: String? = nil
    var failure: String? = nil
    /// The newest alert in the last minute (the screen's alert evidence).
    var latestAlert: EventItem? = nil
    var lastDriverEvent: EventItem? = nil
    /// Per history `event_key`, what that row says about where it is kept: the
    /// live "why" line names its row when the device sent the key.
    var historyNotes: [String: String] = [:]

    /// The distracted-driving panel belongs to Driver Safety, whenever the
    /// device runs it and Jupiter's rules haven't replaced it.
    var showsDriverPanel: Bool {
        PresetName.parse(activePreset).base == "driver_safety" && !rule.replacesModes
    }

    var hasLivePicture: Bool {
        if case .live = picture { return true }
        return false
    }
}

// MARK: - A paired camera, directly (M12 local path)

enum LocalPresentation {
    /// The camera's own preset list as a catalog, for the shared option logic.
    static func catalog(_ presets: LocalPresets) -> PresetCatalog {
        PresetCatalog(catalogVersion: nil, presets: presets.presets.map { entry in
            CatalogPreset(id: entry.name, title: entry.title, description: entry.description ?? "",
                          variants: (entry.variants ?? []).map { PresetVariantInfo(id: $0.name, title: $0.title) })
        }, source: .server)
    }

    /// The modes to offer: the camera's list, only those its role allows
    /// (a mode it would refuse is never offered), selected from what it reports.
    static func options(_ presets: LocalPresets?, role: LocalDeviceState.Role?,
                        rule: PresentationRule) -> [PresentationModeOption] {
        guard let presets else { return [] }
        let allowed = role?.allowedPresets.map(Set.init)
        return PresentationModeOptions.make(catalog: catalog(presets), activePreset: presets.active,
                                            customReplacesModes: rule.replacesModes,
                                            isAvailable: { allowed?.contains($0) ?? true })
    }

    /// "Driver Safety · Strict", in the camera's own titles.
    static func modeTitle(_ presets: LocalPresets?) -> String {
        guard let presets else { return "Mode not read from the camera yet" }
        return catalog(presets).title(for: presets.active) ?? presets.active ?? "No mode reported"
    }

    /// How this iPhone reaches the camera, for the header.
    static func presence(_ presence: TrustedDevices.Presence) -> String {
        TrustedDevices.isLive(presence) ? "Direct connection" : TrustedDevices.label(presence)
    }

    /// The monitoring readout for a camera read directly: its own fresh word,
    /// or what is known about why there is none. Never "protecting" from the cache.
    static func instrument(_ presence: TrustedDevices.Presence, alertActive: Bool) -> InstrumentState {
        func make(_ kind: InstrumentState.Kind, _ word: String, _ tone: StatusTone, live: Bool) -> InstrumentState {
            InstrumentState(kind: kind, word: word, tone: tone, symbol: InstrumentState.symbol(kind), live: live)
        }
        switch presence {
        case .live(let protection):
            if alertActive { return make(.alert, "Alert", .critical, live: true) }
            switch protection {
            case "protecting": return make(.ready, "Protecting", .ok, live: true)
            case "stopped": return make(.stopped, "Stopped", .neutral, live: true)
            case "not_protecting": return make(.degraded, "Not protecting", .caution, live: true)
            default: return make(.unknown, "Connected", .neutral, live: true)
            }
        case .searching: return make(.connecting, "Looking…", .neutral, live: false)
        case .awaitingDevice: return make(.connecting, "Waiting for camera", .neutral, live: false)
        case .notReachable: return make(.offline, "Offline", .neutral, live: false)
        case .needsPairing: return make(.unknown, "Pair again", .caution, live: false)
        }
    }

    /// The newest alert in the camera's history within `window` seconds, on
    /// this iPhone's clock (the row's `age_s`); a row with no age (an earlier
    /// power-on) is never news.
    static func latestAlert(_ events: [LocalEvent], fetchedAt: Date, now: Date, window: TimeInterval = 60,
                            isAlert: (EventItem) -> Bool) -> EventItem? {
        let items = events.filter { $0.ageS != nil }.map { LocalHistory.eventItem($0, fetchedAt: fetchedAt) }
        guard let newest = items.first(where: isAlert), let date = newest.date,
              now.timeIntervalSince(date) < window else { return nil }
        return newest
    }

    /// What each row of the camera's history says about where it is kept.
    static func historyNotes(_ events: [LocalEvent]) -> [String: String] {
        var notes: [String: String] = [:]
        for event in events where notes[event.eventKey] == nil {
            notes[event.eventKey] = "In the camera's history · \(LocalHistory.syncLabel(event.sync))"
        }
        return notes
    }

    /// Everything presentation shows for a paired camera opened directly.
    static func source(name: String, hardwareID: String, url: URL?, presence: TrustedDevices.Presence,
                       state: LocalDeviceState?, presets: LocalPresets?, events: [LocalEvent], fetchedAt: Date,
                       now: Date, pending: String?, failure: String?,
                       isAlert: (EventItem) -> Bool) -> PresentationSource {
        let live = TrustedDevices.isLive(presence)
        let events = events.sorted { $0.seq > $1.seq }  // newest first, whatever the page's order
        // Only a fresh read is evidence: a cached state says nothing about now.
        let rule = live ? PresentationRule.fromDeviceMode(state?.policy?.mode) : .none
        let latest = latestAlert(events, fetchedAt: fetchedAt, now: now, isAlert: isAlert)
        let alertActive = live && (latest != nil || state?.alertActive == true)
        let items = events.map { LocalHistory.eventItem($0, fetchedAt: fetchedAt) }
        let options = live ? options(presets, role: state?.policy?.role, rule: rule) : []
        let note: String? = options.isEmpty
            ? (!live || presets == nil ? "The camera's modes appear once it answers on this network." : nil)
            : nil
        return PresentationSource(
            deviceName: name, hardwareID: hardwareID, managedLocally: true, activePreset: presets?.active,
            picture: url.map { .live($0) } ?? .none,
            instrument: instrument(presence, alertActive: alertActive), presence: self.presence(presence),
            modeTitle: modeTitle(presets), rule: rule, options: options, optionsNote: note,
            pending: pending, failure: failure,
            latestAlert: alertActive ? latest : nil, lastDriverEvent: DriverEvents.newest(items),
            historyNotes: historyNotes(events))
    }
}
