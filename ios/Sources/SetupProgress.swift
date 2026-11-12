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

    /// "Step 3 of 6", from the same reading as the rows below it.
    private var position: (index: Int, count: Int)? {
        guard let current = steps.firstIndex(where: { $0.status == .active || $0.status == .failed }) else {
            return nil
        }
        return (current + 1, steps.count)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: SD.Space.sm) {
            if let position {
                Text("Step \(position.index) of \(position.count)")
                    .font(SD.Typography.caption.weight(.medium).monospacedDigit())
                    .foregroundStyle(SD.Colors.textTertiary)
                    .accessibilityAddTraits(.isHeader)
            }
            VStack(alignment: .leading, spacing: 0) {
                ForEach(Array(steps.enumerated()), id: \.element.id) { index, step in
                    row(step, isLast: index == steps.count - 1)
                }
            }
        }
        .animation(SD.Motion.standard, value: steps)
    }

    /// One step on a vertical spine: the marker, a hairline to the next step,
    /// the step's name and (once it has started) what the device reported.
    private func row(_ step: SetupStep, isLast: Bool) -> some View {
        HStack(alignment: .top, spacing: SD.Space.sm) {
            VStack(spacing: 0) {
                marker(for: step.status)
                    .frame(width: 20, height: 20)
                if !isLast {
                    Rectangle()
                        .fill(step.status == .done ? SD.Colors.textTertiary : SD.Colors.border)
                        .frame(width: 1)
                        .frame(minHeight: 14, maxHeight: .infinity)
                }
            }
            VStack(alignment: .leading, spacing: 1) {
                Text(step.title)
                    .font(step.status == .active ? SD.Typography.label.weight(.semibold) : SD.Typography.callout)
                    .foregroundStyle(colour(for: step.status))
                if let detail = step.detail, step.status != .pending {
                    Text(detail)
                        .font(SD.Typography.caption)
                        .foregroundStyle(step.status == .failed ? SD.Colors.danger : SD.Colors.textTertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(.bottom, isLast ? 0 : SD.Space.sm)
            Spacer(minLength: 0)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(step.title), \(step.status.spokenDescription)")
    }

    @ViewBuilder private func marker(for status: SetupStep.Status) -> some View {
        switch status {
        case .done:
            // Bone, not green: each step is something the device reported,
            // and green is kept for "the device is working".
            Image(systemName: "checkmark")
                .font(.system(size: 9, weight: .bold))
                .foregroundStyle(SD.Colors.textOnAccent)
                .frame(width: 16, height: 16)
                .background(Circle().fill(SD.Colors.textSecondary))
        case .active:
            ProgressView().controlSize(.small)
        case .pending:
            Circle()
                .strokeBorder(SD.Colors.borderStrong, lineWidth: 1.5)
                .frame(width: 12, height: 12)
        case .failed:
            Image(systemName: "exclamationmark.circle.fill")
                .foregroundStyle(SD.Colors.danger)
        }
    }

    private func colour(for status: SetupStep.Status) -> Color {
        switch status {
        case .done: SD.Colors.textSecondary
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
    /// The onboarding checklist for one state of the journey.
    ///
    /// One function, so the list can never disagree with itself: every row's
    /// status comes from the same reading of the same state. A failure marks
    /// the row that was in flight and leaves the completed ones standing --
    /// how far setup got is the most useful thing to know about what to try
    /// next. Every row is derived from something reported; no timer moves it.
    static func checklist(
        for state: Onboarding.State,
        requiresCode: Bool,
        signedIn: Bool,
        networksFound: Int = 0,
        scanning: Bool = false,
        now: Date = Date()
    ) -> [SetupStep] {
        var stages = Onboarding.Stage.allCases
        if !requiresCode { stages.removeAll { $0 == .prove } }
        if !signedIn { stages.removeAll { $0 == .account || $0 == .mode } }

        let current = state.stage
        let finished: Bool = { if case .finished = state { return true }; return false }()
        let failed: Bool = { if case .failed = state { return true }; return false }()

        return stages.map { stage in
            let status: Status
            if finished || stage < current {
                status = .done
            } else if stage == current {
                status = failed ? .failed : .active
            } else {
                status = .pending
            }
            return SetupStep(id: "\(stage)", title: stage.title,
                             detail: detail(stage, state: state, networksFound: networksFound, scanning: scanning, now: now),
                             status: status)
        }
    }

    private static func detail(_ stage: Onboarding.Stage, state: Onboarding.State, networksFound: Int,
                               scanning: Bool, now: Date) -> String? {
        if case .failed(let failure) = state, failure.stage == stage { return failure.reason }
        switch (stage, state) {
        case (.prove, .enteringSetupCode(_, let lockedUntil?)) where lockedUntil > now:
            return "the device is refusing codes for \(Int(lockedUntil.timeIntervalSince(now).rounded(.up))) s"
        case (.prove, .proving):
            return "the device checks the code, then proves it back"
        case (.network, .choosingNetwork):
            if networksFound > 0 { return networksFound == 1 ? "1 network in range" : "\(networksFound) networks in range" }
            return scanning ? "the device is looking…" : "the device could not see any Wi-Fi from here"
        case (.network, .sending):
            return "sent encrypted, to this device only"
        case (.join, .joining):
            return "this can take up to a minute"
        case (.online, .waitingForDevice(let elapsed)):
            return elapsed > 0 ? "waiting for it to reach the server (\(elapsed) s)" : "waiting for it to reach the server"
        default:
            return nil
        }
    }
}
