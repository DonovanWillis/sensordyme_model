import SwiftUI

/// The steps of setting up a device, and which one is happening now.
///
/// Setup is a chain of slow things over two different radios, and a single
/// spinner cannot say which one is slow. Someone watching "Connecting…" for
/// twenty seconds has no way to tell a device negotiating DHCP from an app that
/// stopped trying — so they force-quit, and a working setup is lost at the last
/// step.
///
/// Every step here is derived from something the device actually reported. There
/// is no timer advancing the list on its own and no percentage: when the outcome
/// of a step is genuinely unknown, it shows an indeterminate spinner and says so.
/// A progress bar that is really a clock is a lie that gets believed exactly
/// once.
struct SetupProgressView: View {
    let steps: [SetupStep]

    var body: some View {
        VStack(alignment: .leading, spacing: SD.Space.sm) {
            ForEach(steps) { step in
                HStack(alignment: .firstTextBaseline, spacing: SD.Space.sm) {
                    marker(for: step.status)
                        .frame(width: 18)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(step.title)
                            .font(SD.Typography.callout)
                            .foregroundStyle(colour(for: step.status))
                        if let detail = step.detail, step.status != .pending {
                            Text(detail)
                                .font(SD.Typography.caption)
                                .foregroundStyle(SD.Colors.textTertiary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                    Spacer(minLength: 0)
                }
                .accessibilityElement(children: .combine)
                .accessibilityLabel("\(step.title), \(step.status.spokenDescription)")
            }
        }
        .animation(SD.Motion.standard, value: steps)
    }

    @ViewBuilder private func marker(for status: SetupStep.Status) -> some View {
        switch status {
        case .done:
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(SD.Colors.success)
        case .active:
            ProgressView().controlSize(.small)
        case .pending:
            Image(systemName: "circle")
                .foregroundStyle(SD.Colors.textTertiary.opacity(0.5))
        case .failed:
            Image(systemName: "exclamationmark.circle.fill")
                .foregroundStyle(SD.Colors.danger)
        }
    }

    private func colour(for status: SetupStep.Status) -> Color {
        switch status {
        case .done: SD.Colors.textPrimary
        case .active: SD.Colors.textPrimary
        case .pending: SD.Colors.textTertiary
        case .failed: SD.Colors.danger
        }
    }
}

struct SetupStep: Identifiable, Equatable {
    enum Status: Equatable {
        case pending
        case active
        case done
        case failed

        var spokenDescription: String {
            switch self {
            case .pending: "not started"
            case .active: "in progress"
            case .done: "done"
            case .failed: "failed"
            }
        }
    }

    let id: String
    let title: String
    var detail: String?
    var status: Status
}

extension SetupStep {
    /// The setup checklist for a given provisioning state.
    ///
    /// One function, so the list can never disagree with itself — every step's
    /// status comes from the same reading of the same state. Building this
    /// inline in the view is how a screen ends up showing "connected" above
    /// "connecting".
    static func checklist(
        for state: BLEProvisioner.State,
        networksFound: Int,
        scanning: Bool,
        stateBeforeFailure: BLEProvisioner.State? = nil
    ) -> [SetupStep] {
        // Which step a failure belongs to: the one that was in flight. Marking
        // the whole list failed would hide how far setup actually got, which is
        // the most useful thing to know when deciding what to try next.
        let failedStage: Int? = {
            guard case .failed = state else { return nil }
            return Self.stage(of: stateBeforeFailure ?? .scanning)
        }()
        // How far the flow got. Steps before this are done.
        let currentStage = failedStage ?? Self.stage(of: state)

        func status(_ index: Int) -> Status {
            if let failedStage {
                if index < failedStage { return .done }
                return index == failedStage ? .failed : .pending
            }
            if index < currentStage { return .done }
            return index == currentStage ? .active : .pending
        }

        var networkDetail: String?
        if currentStage >= 2 {
            if networksFound > 0 {
                networkDetail = networksFound == 1
                    ? "1 network in range"
                    : "\(networksFound) networks in range"
            } else if scanning {
                networkDetail = "the device is looking…"
            } else {
                networkDetail = "the device could not see any Wi-Fi from here"
            }
        }

        var steps: [SetupStep] = [
            SetupStep(id: "find", title: "Find your SensorDyme", status: status(0)),
            SetupStep(id: "connect", title: "Connect over Bluetooth", status: status(1)),
            SetupStep(id: "scan", title: "Look for Wi-Fi networks",
                      detail: networkDetail, status: status(2)),
            SetupStep(id: "send", title: "Send the Wi-Fi details", status: status(3)),
            SetupStep(id: "join", title: "Device joins the network",
                      detail: currentStage >= 4 ? "this can take up to a minute" : nil,
                      status: status(4)),
            SetupStep(id: "ready", title: "SensorDyme ready", status: status(5)),
        ]

        if case let .failed(reason) = state, let failedStage,
           failedStage < steps.count {
            steps[failedStage].detail = reason
        }
        if case let .success(ip) = state {
            steps[5].status = .done
            steps[5].detail = ip.map { "on your network at \($0)" }
        }
        return steps
    }

    /// How far the flow has got. Steps before this one are done.
    private static func stage(of state: BLEProvisioner.State) -> Int {
        switch state {
        case .idle, .bluetoothOff, .scanning: 0
        case .connecting: 1
        case .ready: 2
        case .sending: 3
        case .deviceConnecting: 4
        case .success: 5
        case .failed: 0
        }
    }
}
