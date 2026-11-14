import SwiftUI

/// Developer diagnostics (M10, docs/DIAGNOSTICS.md): the device's
/// `sd-diagnostics/1` payload, readable. Reached only from Settings > Live feed,
/// and only when the device reports diagnostics on and this phone holds its token
/// (DiagnosticsAccess). Everything shown is what the device answered; the
/// wording lives in DiagnosticsPresentation (tested in DeviceDiagnosticsTests).
struct DeviceDiagnosticsView: View {
    let baseURL: String
    let token: String

    @State private var diagnostics: DeviceDiagnostics?
    @State private var errorMessage: String?
    @State private var fetchedAt: Date?
    @State private var live = true

    /// How often the screen re-asks while it is open (the device builds a
    /// snapshot at most twice a second, and only while someone is asking).
    private static let refreshInterval: Duration = .seconds(2)

    var body: some View {
        Form {
            Group {
                Section {
                    if let diagnostics {
                        Label(DiagnosticsPresentation.hostBanner(diagnostics),
                              systemImage: diagnostics.platform?.measurementClass == "pi" ? "cpu" : "exclamationmark.triangle")
                            .font(SD.Typography.callout.weight(.semibold))
                            .foregroundStyle(diagnostics.platform?.measurementClass == "pi" ? SD.Colors.textPrimary : SD.Colors.warning)
                            .fixedSize(horizontal: false, vertical: true)
                            .accessibilityIdentifier("diagnostics.host")
                        if let module = diagnostics.frame?.module {
                            row(.init(title: "Module", value: module))
                        }
                        if let age = diagnostics.snapshotAgeS {
                            row(.init(title: "Snapshot age", value: String(format: "%.1f s", age)))
                        }
                        if diagnostics.warmingUp == true {
                            Text("The device starts a snapshot when asked; it appears on the next refresh.")
                                .font(SD.Typography.caption)
                                .foregroundStyle(SD.Colors.textSecondary)
                        }
                    } else if let errorMessage {
                        Label(errorMessage, systemImage: "exclamationmark.circle")
                            .font(SD.Typography.caption)
                            .foregroundStyle(SD.Colors.danger)
                            .fixedSize(horizontal: false, vertical: true)
                    } else {
                        HStack { ProgressView(); Text("Asking the device…") }
                    }
                    Toggle("Refresh every 2 s", isOn: $live)
                } header: {
                    Text("Host")
                } footer: {
                    if let fetchedAt {
                        Text("As the device reported it at \(fetchedAt.formatted(date: .omitted, time: .standard)).")
                    }
                }

                if let diagnostics {
                    rows("Detections", DiagnosticsPresentation.detectionRows(diagnostics.frame))
                    rows("Face", DiagnosticsPresentation.flatten(diagnostics.frame?.face))
                    rows("Evidence", DiagnosticsPresentation.flatten(diagnostics.frame?.evidence))
                    rows("Open conditions", DiagnosticsPresentation.conditionRows(diagnostics.frame))
                    capabilitiesSection(diagnostics)
                    rows("Latency per stage", DiagnosticsPresentation.latencyRows(diagnostics.latency))
                    rows("Models and versions", DiagnosticsPresentation.flatten(diagnostics.versions))
                }
            }
            .orynRows()
        }
        .orynFormStyle()
        .navigationTitle("Developer diagnostics")
        .navigationBarTitleDisplayMode(.inline)
        .refreshable { await load() }
        .task(id: live) {
            await load()
            while live && !Task.isCancelled {
                try? await Task.sleep(for: Self.refreshInterval)
                if Task.isCancelled { break }
                await load()
            }
        }
    }

    @ViewBuilder
    private func capabilitiesSection(_ diagnostics: DeviceDiagnostics) -> some View {
        let caps = diagnostics.capabilities
        let degraded = diagnostics.unknownOrDegraded ?? []
        Section("Monitoring") {
            row(.init(title: "State", value: caps?.state ?? "not reported"))
            row(.init(title: "Available", value: (caps?.available ?? []).joined(separator: ", ").ifEmpty("none")))
            if let impaired = caps?.impaired, !impaired.isEmpty {
                row(.init(title: "Impaired", value: impaired.joined(separator: ", ")))
            }
            if degraded.isEmpty {
                row(.init(title: "Unknown or degraded", value: "nothing reported"))
            } else {
                ForEach(degraded, id: \.self) { reason in
                    Label(reason, systemImage: "exclamationmark.triangle")
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.warning)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    private func rows(_ title: String, _ items: [DiagnosticsPresentation.Row]) -> some View {
        Section(title) {
            if items.isEmpty {
                row(.init(title: title, value: "not reported"))
            }
            ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                row(item)
            }
        }
    }

    private func row(_ item: DiagnosticsPresentation.Row) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(item.title)
                .font(SD.Typography.caption)
                .foregroundStyle(SD.Colors.textTertiary)
            Text(item.value)
                .font(SD.Typography.callout.monospacedDigit())
                .foregroundStyle(SD.Colors.textPrimary)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
        }
        .accessibilityElement(children: .combine)
    }

    private func load() async {
        do {
            diagnostics = try await DeviceDiagnostics.fetch(baseURL: baseURL, token: token)
            fetchedAt = Date()
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
            // Stop polling on a refusal: asking every 2 s will not change a switch or a token.
            if error is DeviceDiagnosticsError { live = false }
        }
    }
}

private extension String {
    func ifEmpty(_ replacement: String) -> String { isEmpty ? replacement : self }
}
