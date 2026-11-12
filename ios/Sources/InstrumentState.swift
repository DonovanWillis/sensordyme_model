import Foundation

/// The device screen's single status readout: presence, monitoring state and a
/// recent alert, as one of a small set of visual states.
///
/// Presentation only. Every input is already decided elsewhere (presence by
/// ConnectionTruth, the monitoring state by MonitoringStatusPresentation, the
/// alert window by the device screen), and this type only chooses how to SHOW
/// them together. The readout's word is exactly the monitoring pill's text
/// (`MonitoringStatusPresentation.pill`): VoiceOver, the journeys and the old
/// pill all read the same word.
///
/// Foundation-only, so ios/Tests/run.sh covers it.
struct InstrumentState: Equatable {
    enum Kind: String, Equatable, CaseIterable {
        /// Nothing has been heard yet on this screen.
        case connecting
        /// The device confirmed `ready` on fresh telemetry (or an online device
        /// on a backend that predates the status contract, as before).
        case ready
        /// An alert in the last minute.
        case alert
        case starting, degraded, stopped, failed
        /// The device itself is not reporting.
        case offline
        /// The reply says online, but the telemetry in it is old.
        case stale
        /// The app cannot tell (refresh failed, or nothing current).
        case unknown
    }

    let kind: Kind
    /// The readout's word, identical to the monitoring pill's text.
    let word: String
    let tone: StatusTone
    /// An SF Symbol that differs for every kind, so a state is never told by
    /// colour alone.
    let symbol: String

    /// Whether the screen may show "active" visuals (the live hairline, the
    /// camera's state frame in an active tone). Only for a device whose
    /// telemetry is live.
    var isActive: Bool {
        switch kind {
        case .ready, .alert, .starting, .degraded: true
        default: false
        }
    }

    /// The monitoring has stopped speaking for now: the camera placeholder is
    /// dimmed and the live accents are withdrawn at once.
    var isInterrupted: Bool {
        switch kind {
        case .offline, .stale, .unknown, .connecting: true
        default: false
        }
    }

    /// An indeterminate progress line is honest only while something is being
    /// waited for.
    var showsProgress: Bool { kind == .connecting }

    static func make(presence: ConnectionTruth.Presence,
                     monitoring: MonitoringStatusPresentation?,
                     alertActive: Bool,
                     loaded: Bool) -> InstrumentState {
        let pill = MonitoringStatusPresentation.pill(monitoring, alertActive: alertActive)
        let kind: Kind
        if alertActive {
            kind = .alert
        } else {
            switch presence {
            case .offline: kind = .offline
            case .stale: kind = .stale
            case .unknown(.notLoaded): kind = loaded ? .unknown : .connecting
            case .unknown: kind = .unknown
            case .online:
                switch monitoring?.phase {
                case nil, .ready?: kind = pill.tone == .ok ? .ready : .unknown
                case .starting?: kind = .starting
                case .degraded?: kind = .degraded
                case .stopped?: kind = .stopped
                case .failed?: kind = .failed
                case .unknown?: kind = .unknown
                }
            }
        }
        // Never an "ok" tone unless the device is live and confirmed ready: a
        // pill that says All clear over old telemetry is exactly what this
        // screen must not show.
        let tone: StatusTone = (kind == .ready) ? pill.tone : (pill.tone == .ok ? .neutral : pill.tone)
        return InstrumentState(kind: kind, word: pill.text, tone: tone, symbol: symbol(kind))
    }

    static func symbol(_ kind: Kind) -> String {
        switch kind {
        case .connecting: "antenna.radiowaves.left.and.right"
        case .ready: "checkmark.shield"
        case .alert: "bell.and.waves.left.and.right"
        case .starting: "hourglass"
        case .degraded: "exclamationmark.triangle"
        case .stopped: "pause.circle"
        case .failed: "xmark.octagon"
        case .offline: "wifi.slash"
        case .stale: "clock.badge.exclamationmark"
        case .unknown: "questionmark.circle"
        }
    }
}
