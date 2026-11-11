import SwiftUI

/// The device's custom rules as the backend lists them (GET
/// /v1/devices/{id}/rules): what each one is, whether the device is really
/// running it, and a switch to turn it off or on (PATCH). Rules are added by
/// asking Jupiter; this card links there.
struct CustomRulesCard: View {
    enum Load: Equatable {
        case loading
        case loaded(DeviceRules)
        /// This server has no rules endpoint.
        case unsupported
        case failed(String)
    }

    let load: Load
    /// Online and reporting; rules are never shown running otherwise.
    let deviceLive: Bool
    /// Rule ids with a toggle request in flight.
    let busy: Set<String>
    /// Why the last toggle failed, if it did.
    let toggleError: String?
    let onToggle: (DeviceRule, Bool) -> Void
    let onAskJupiter: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: SD.Space.xs) {
            SDSectionHeader(title: "Custom rules", action: ("Ask Jupiter", onAskJupiter))
                .padding(.horizontal, SD.Space.xxs)
            SDCard {
                VStack(alignment: .leading, spacing: SD.Space.sm) {
                    content
                    if let toggleError {
                        Label(toggleError, systemImage: "exclamationmark.circle.fill")
                            .font(SD.Typography.caption)
                            .foregroundStyle(SD.Colors.danger)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private var content: some View {
        switch load {
        case .loading:
            HStack(spacing: SD.Space.xs) {
                ProgressView().controlSize(.small).tint(SD.Colors.textSecondary)
                caption("Loading rules…")
            }
        case .unsupported:
            caption("This server can't list rules yet, so their status isn't shown here. " + ServerSupport.notSupported)
        case .failed(let message):
            caption("Couldn't load rules: \(message)")
        case .loaded(let rules) where rules.rules.isEmpty:
            caption("No custom rules. Ask Jupiter to watch for something specific, like a dog on the couch.")
        case .loaded(let rules):
            if rules.additive == true {
                caption("These run alongside the monitoring mode.")
            }
            ForEach(Array(rules.rules.enumerated()), id: \.element.id) { index, rule in
                row(rule)
                if index < rules.rules.count - 1 {
                    Rectangle().fill(SD.Colors.border).frame(height: 1)
                }
            }
        }
    }

    private func row(_ rule: DeviceRule) -> some View {
        let presentation = RuleStatusPresentation.make(rule, deviceLive: deviceLive)
        return HStack(alignment: .top, spacing: SD.Space.sm) {
            VStack(alignment: .leading, spacing: SD.Space.xxs) {
                Text(rule.summary)
                    .font(SD.Typography.body)
                    .foregroundStyle(rule.enabled ? SD.Colors.textPrimary : SD.Colors.textDisabled)
                    .fixedSize(horizontal: false, vertical: true)
                SDStatusPill(status: presentation.tone.sdStatus, text: presentation.chip)
                if let detail = presentation.detail {
                    Text(detail)
                        .font(SD.Typography.caption)
                        .foregroundStyle(presentation.tone == .critical ? SD.Colors.danger : SD.Colors.textTertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: SD.Space.xs)
            if busy.contains(rule.id) {
                ProgressView().controlSize(.small).tint(SD.Colors.textSecondary)
            } else {
                Toggle("", isOn: Binding(get: { rule.enabled }, set: { onToggle(rule, $0) }))
                    .labelsHidden()
                    // The rule is the label; the switch's own value says on or off.
                    // "Turn off X, switch, on" said the action and the state at once.
                    .accessibilityLabel(rule.summary)
                    .accessibilityIdentifier("rule.toggle")
            }
        }
        .padding(.vertical, 2)
    }

    private func caption(_ text: String) -> some View {
        Text(text)
            .font(SD.Typography.caption)
            .foregroundStyle(SD.Colors.textTertiary)
            .fixedSize(horizontal: false, vertical: true)
    }
}

extension RuleTone {
    var sdStatus: SDStatus {
        switch self {
        case .ok: .online
        case .caution: .warning
        case .critical: .alert
        case .neutral: .offline
        }
    }
}
