import SwiftUI

/// What the device says about itself once setup is finished.
///
/// Every line comes from the device's own `/api/health`. Nothing is listed
/// because it is expected to be there: a tick beside "AI HAT+" that really
/// means "we shipped a Pi with one" is worth less than no tick at all, and it
/// is exactly the sort of claim that falls apart when someone asks a question
/// about it in a demo.
///
/// Components the device does not report are shown as unknown rather than
/// omitted or assumed good — "we didn't ask" and "it works" are different
/// answers.
struct DeviceReadinessCard: View {
    let baseURL: String?

    @State private var health: DeviceHealth?
    @State private var failure: String?
    @State private var loading = true

    var body: some View {
        SDCard {
            VStack(alignment: .leading, spacing: SD.Space.sm) {
                SDSectionHeader(title: "Your SensorDyme")

                if loading {
                    HStack {
                        ProgressView().controlSize(.small)
                        Text("Checking…")
                            .font(SD.Typography.callout)
                            .foregroundStyle(SD.Colors.textSecondary)
                    }
                } else if let failure {
                    // Reached over Wi-Fi a moment ago, so this is worth saying
                    // plainly rather than dressing up as a hardware fault.
                    Label(failure, systemImage: "exclamationmark.circle")
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.warning)
                        .fixedSize(horizontal: false, vertical: true)
                } else if let health {
                    ForEach(rows(from: health), id: \.name) { row in
                        HStack(spacing: SD.Space.sm) {
                            Image(systemName: row.healthy ? "checkmark.circle.fill" : "exclamationmark.circle.fill")
                                .foregroundStyle(row.healthy ? SD.Colors.success : SD.Colors.warning)
                            VStack(alignment: .leading, spacing: 1) {
                                Text(row.name)
                                    .font(SD.Typography.body)
                                    .foregroundStyle(SD.Colors.textPrimary)
                                if !row.detail.isEmpty {
                                    Text(row.detail)
                                        .font(SD.Typography.caption)
                                        .foregroundStyle(SD.Colors.textTertiary)
                                        .lineLimit(2)
                                }
                            }
                            Spacer(minLength: 0)
                        }
                    }
                }
            }
        }
        .task { await load() }
    }

    private struct Row {
        let name: String
        let detail: String
        let healthy: Bool
    }

    /// Present the device's own component names in words a person recognises.
    private func rows(from health: DeviceHealth) -> [Row] {
        let labels = [
            "camera": "Camera",
            "inference": "Detection",
            "accelerator": "AI HAT+",
            "wifi": "Wi-Fi",
            "dashboard": "Live feed",
            "provisioning": "Bluetooth",
            "cloud_uplink": "Cloud link",
        ]
        return health.components
            .sorted { $0.key < $1.key }
            .map { key, component in
                Row(name: labels[key] ?? key.replacingOccurrences(of: "_", with: " ").capitalized,
                    detail: component.detail,
                    healthy: component.state.lowercased() == "ok")
            }
    }

    private func load() async {
        defer { loading = false }
        guard let baseURL, let url = URL(string: baseURL)?.appendingPathComponent("api/health") else {
            failure = "No address for this device yet."
            return
        }
        var request = URLRequest(url: url)
        request.timeoutInterval = 6
        do {
            let (data, _) = try await URLSession.shared.data(for: request)
            health = try JSONDecoder().decode(DeviceHealth.self, from: data)
        } catch {
            failure = "Couldn't read the device's status just now — it may still be starting up."
        }
    }
}
