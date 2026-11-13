import Foundation

/// M8.3: what the app may claim about a device, checked against the evidence
/// in the reply it is showing (docs/STATE_MODEL.md, "App invariants").
///
/// The device screen builds its readout the way this does: presence from
/// ConnectionTruth, `live` = presence online, the monitoring phrase from
/// MonitoringStatusPresentation, the readout from InstrumentState, capability
/// and component rows from CapabilityHealth / PerceptionPresentation, and the
/// stop/resume note from ProtectionControl. Every rule here names a claim and
/// the evidence it needs; `violations` lists each claim made without it.
///
/// The same rules as sim/invariants.py `status_violations` on the backend side.
/// Foundation only: ios/Tests/PresentationInvariantsTests.swift runs it over
/// generated status combinations.
enum PresentationInvariants {
    struct Input {
        var status: DeviceStatus?
        /// The backend's `online`, `last_seen`, and when the reply arrived.
        var online: Bool
        var lastSeen: Double?
        var receivedAt: Date?
        var lastRefreshFailed: Bool = false
        var alertActive: Bool = false
        /// A stop/resume this app sent and is tracking, if any.
        var request: ProtectionRequest? = nil
        var now: Date
    }

    /// Everything the screen would show for `input`.
    struct Shown {
        let presence: ConnectionTruth.Presence
        let live: Bool
        let monitoring: MonitoringStatusPresentation?
        let instrument: InstrumentState
        let capabilities: [CapabilityHealthRow]
        let components: [PerceptionComponentRow]
        let protection: ProtectionPhase
    }

    static func show(_ input: Input) -> Shown {
        let status = input.status
        let reading = ConnectionTruth.Reading(
            online: input.online, stale: status?.stale, lastSeen: input.lastSeen, receivedAt: input.receivedAt,
            linkStale: status?.linkStale, healthStale: status?.monitoringEvidenceStale,
            healthAge: status?.healthFreshness?.ageS)
        let presence = ConnectionTruth.presence(reading, lastRefreshFailed: input.lastRefreshFailed, now: input.now)
        let live = presence.isOnline
        let monitoring = MonitoringStatusPresentation.make(status: status, online: live)
        let instrument = InstrumentState.make(presence: presence, monitoring: monitoring,
                                              alertActive: input.alertActive, loaded: true)
        let capabilities = CapabilityHealth.rows(status: status, online: live).rows
        let components = PerceptionPresentation.componentRows(status: status, live: live)
        let protection = ProtectionControl.phase(input.request, report: ProtectionReport(status: status, live: live),
                                                 now: input.now)
        return Shown(presence: presence, live: live, monitoring: monitoring, instrument: instrument,
                     capabilities: capabilities, components: components, protection: protection)
    }

    static let analysis: Set<String> = CapabilityHealth.analysisCapabilities

    /// Each claim the screen makes for `input` without the evidence for it.
    static func violations(_ input: Input) -> [String] {
        let shown = show(input)
        let status = input.status
        var out: [String] = []
        let state = status?.monitoringState
        let stopped = state == "stopped" || status?.protection == "stopped"
        let impaired = Set((status?.capabilities?.impaired ?? []).map(\.name))
        let pipelineDown = impaired.contains("camera") || impaired.contains("inference")
        let fresh = input.online && status?.stale != true && status?.linkStale != true
            && status?.monitoringEvidenceStale != true && !input.lastRefreshFailed

        // P1 presence: "Online" only on a current reply with fresh evidence.
        if shown.presence.isOnline && !fresh {
            out.append("online_without_fresh_evidence: presence online, online=\(input.online) "
                       + "stale=\(String(describing: status?.stale)) healthStale=\(String(describing: status?.healthStale))")
        }
        if shown.presence.isOnline, let age = status?.healthFreshness?.ageS,
           status?.healthFreshness?.verified == true, age >= ConnectionTruth.liveWindow {
            out.append("online_with_old_health: health \(age) s old")
        }
        // P2 "All clear" / the ok tone / the ready readout: only live and ready.
        // (The pre-contract rendering, no status at all, is the one exception.)
        let readyClaim = shown.instrument.kind == .ready || shown.instrument.tone == .ok
            || shown.monitoring?.tone == .ok
        if readyClaim && status != nil && !(shown.live && state == "ready" && !pipelineDown && !stopped) {
            out.append("ready_without_evidence: \(shown.instrument.kind) live=\(shown.live) state=\(state ?? "nil")")
        }
        // P3 active visuals and the monitoring phrases need live telemetry.
        if shown.instrument.isActive && !shown.live {
            out.append("active_visuals_while_not_live: \(shown.instrument.kind) presence=\(shown.presence)")
        }
        if !shown.live, let phase = shown.monitoring?.phase, phase != .unknown {
            out.append("current_state_while_not_live: \(phase)")
        }
        if shown.live, let phase = shown.monitoring?.phase {
            if phase == .ready || phase == .degraded, stopped || pipelineDown {
                out.append("monitoring_claim_contradicted: \(phase) while stopped=\(stopped) pipelineDown=\(pipelineDown)")
            }
        }
        // P4 capability rows: nothing "producing results" without live
        // evidence, while stopped, without frames / inference, or while the
        // model behind it failed.
        for row in shown.capabilities where row.state == .available {
            if !shown.live { out.append("capability_available_while_not_live: \(row.name)") }
            if stopped && (analysis.contains(row.name) || row.name == "inference") {
                out.append("capability_available_while_stopped: \(row.name)")
            }
            if pipelineDown && analysis.contains(row.name) {
                out.append("capability_available_without_pipeline: \(row.name)")
            }
            if let component = PerceptionPresentation.componentForCapability[row.name],
               let comp = status?.perception?.components[component], comp.state == "failed" || comp.state == "unavailable" {
                out.append("capability_available_with_model_\(comp.state ?? "?"): \(row.name)")
            }
        }
        // P5 perception rows: no component "Working" unless live and not stopped.
        for row in shown.components where row.tone == .ok {
            if !shown.live { out.append("component_working_while_not_live: \(row.name)") }
            if stopped { out.append("component_working_while_stopped: \(row.name)") }
        }
        // P6 stop/resume: a success only on the device's live ack with the
        // state asked for; "monitoring again" only while protecting.
        if case .confirmed(let text) = shown.protection, let request = input.request {
            let report = ProtectionReport(status: status, live: shown.live)
            if !shown.live { out.append("protection_confirmed_while_not_live") }
            if !ProtectionControl.agrees(request.action, report) {
                out.append("protection_confirmed_with_other_state: \(request.action) vs \(status?.protection ?? "nil")")
            }
            let echoed = request.commandID != nil && status?.lastCommandID == request.commandID
            let latched = status?.protectionCommand?.id == request.commandID
                && status?.protectionCommand?.outcome == "confirmed"
            if request.commandID != nil && !echoed && !latched && status?.protectionCommand != nil {
                out.append("protection_confirmed_before_ack: \(request.action)")
            }
            if text.contains("monitoring again") && status?.protection != "protecting" {
                out.append("resume_claims_monitoring: protection=\(status?.protection ?? "nil")")
            }
        }
        return out
    }
}
