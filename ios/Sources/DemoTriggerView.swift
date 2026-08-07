import SwiftUI

/// Rehearsable alerts: pick a scenario, watch the whole pipeline run.
///
/// A demo that depends on producing a real dangerous situation is not a demo —
/// nobody is going to fall over in front of an investor, and waiting for a
/// synthesised event to appear on its own turns a pitch into an awkward pause.
///
/// Each scenario emits an event type the detection code already produces, so
/// pressing one exercises the real path end to end: severity classification,
/// the notification on the lock screen, the explanation assembled from stored
/// facts. Nothing here is a parallel implementation built to look good.
///
/// Simulated events are labelled as simulated everywhere they surface. The
/// difference between a demo and a misrepresentation is whether the audience
/// can tell which is which.
struct DemoTriggerView: View {
    @EnvironmentObject private var state: AppState
    @Environment(\.dismiss) private var dismiss
    @StateObject private var notifier = AlertNotifier.shared
    let device: Device
    let onTriggered: () async -> Void

    @State private var firing: String?
    @State private var fired: String?
    @State private var errorMessage: String?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: SD.Space.md) {
                    intro
                    if !notifier.authorized {
                        permissionCard
                    }
                    ForEach(DemoScenario.allCases) { scenario in
                        scenarioRow(scenario)
                    }
                    if let errorMessage {
                        Label(errorMessage, systemImage: "exclamationmark.circle.fill")
                            .font(SD.Typography.callout)
                            .foregroundStyle(SD.Colors.danger)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    footnote
                }
                .padding(.horizontal, SD.Space.md)
                .padding(.bottom, SD.Space.xl)
            }
            .background(SD.Colors.background)
            .navigationTitle("Demo alerts")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Done") { dismiss() } }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Reset") { notifier.resetAnnounced() }
                        .font(SD.Typography.caption)
                }
            }
            .task { await notifier.refreshAuthorization() }
        }
        .tint(SD.Colors.accent)
    }

    private var intro: some View {
        VStack(alignment: .leading, spacing: SD.Space.xxs) {
            Text("Run the real pipeline")
                .font(SD.Typography.title)
                .foregroundStyle(SD.Colors.textPrimary)
            Text("Each one creates an event, classifies its severity, sends the notification, and writes it to the timeline — the same path a real detection takes.")
                .font(SD.Typography.callout)
                .foregroundStyle(SD.Colors.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.top, SD.Space.xs)
    }

    private var permissionCard: some View {
        SDCard {
            VStack(alignment: .leading, spacing: SD.Space.xs) {
                Label("Notifications are off", systemImage: "bell.slash")
                    .font(SD.Typography.headline)
                    .foregroundStyle(SD.Colors.warning)
                Text("Alerts will still appear in the timeline, but nothing will reach your lock screen.")
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textSecondary)
                Button("Allow notifications") {
                    Task { await notifier.requestPermission() }
                }
                .buttonStyle(SDButtonStyle(kind: .secondary, fullWidth: false))
            }
        }
    }

    private func scenarioRow(_ scenario: DemoScenario) -> some View {
        Button { fire(scenario) } label: {
            SDCard {
                HStack(spacing: SD.Space.sm) {
                    ZStack {
                        RoundedRectangle(cornerRadius: SD.Radius.sm, style: .continuous)
                            .fill(scenario.severity == "loud"
                                  ? SD.Colors.dangerMuted : SD.Colors.warningMuted)
                            .frame(width: 40, height: 40)
                        Image(systemName: scenario.icon)
                            .foregroundStyle(scenario.severity == "loud"
                                             ? SD.Colors.danger : SD.Colors.warning)
                    }
                    VStack(alignment: .leading, spacing: 2) {
                        Text(scenario.title)
                            .font(SD.Typography.headline)
                            .foregroundStyle(SD.Colors.textPrimary)
                        Text(scenario.detail)
                            .font(SD.Typography.caption)
                            .foregroundStyle(SD.Colors.textSecondary)
                            .lineLimit(2)
                    }
                    Spacer(minLength: SD.Space.xxs)
                    if firing == scenario.id {
                        ProgressView().controlSize(.small)
                    } else if fired == scenario.id {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(SD.Colors.success)
                    } else {
                        Image(systemName: "play.fill")
                            .font(.caption)
                            .foregroundStyle(SD.Colors.accent)
                    }
                }
            }
        }
        .buttonStyle(.plain)
        .disabled(firing != nil)
    }

    private var footnote: some View {
        Label(
            "Simulated events are labelled as simulated in the notification and the timeline.",
            systemImage: "info.circle"
        )
        .font(SD.Typography.caption)
        .foregroundStyle(SD.Colors.textTertiary)
        .fixedSize(horizontal: false, vertical: true)
    }

    private func fire(_ scenario: DemoScenario) {
        firing = scenario.id
        errorMessage = nil
        Task {
            defer { firing = nil }
            do {
                _ = try await state.backend.triggerDemoEvent(scenario)
                await onTriggered()
                fired = scenario.id
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }
}
