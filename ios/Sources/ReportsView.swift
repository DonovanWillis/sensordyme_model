import Charts
import SwiftUI

// MARK: - Reports: an activity estimate, alerts per hour, breakdown by type — computed
// on the phone from the loaded history. Not a validated safety measure, and labelled
// as an estimate everywhere it appears (screen and shared text).

struct DeviceReport {
    let safetyScore: Int
    let totalAlerts: Int
    let loudAlerts: Int
    let hourly: [(hour: Date, count: Int)]
    let byType: [(type: String, count: Int)]

    static func compute(from events: [EventItem]) -> DeviceReport {
        // An experimental observation (M8.4) is never counted as an alert.
        let alerts = events.filter { !$0.isExperimental && ($0.severity == "loud" || $0.severity == "soft") }
        let loud = alerts.filter { $0.severity == "loud" }.count

        var hourBuckets: [Date: Int] = [:]
        let calendar = Calendar.current
        for event in alerts {
            guard let date = event.date else { continue }
            let bucket = calendar.date(bySettingHour: calendar.component(.hour, from: date),
                                       minute: 0, second: 0, of: date) ?? date
            hourBuckets[bucket, default: 0] += 1
        }

        var typeCounts: [String: Int] = [:]
        for event in alerts {
            typeCounts[event.type, default: 0] += 1
        }

        // Demo-grade arithmetic, not a model: start at 100, loud alerts cost
        // 6, soft cost 2. Shown only as an estimate.
        let soft = alerts.count - loud
        let score = max(0, 100 - loud * 6 - soft * 2)

        return DeviceReport(
            safetyScore: score,
            totalAlerts: alerts.count,
            loudAlerts: loud,
            hourly: hourBuckets.sorted { $0.key < $1.key }.map { ($0.key, $0.value) },
            byType: typeCounts.sorted { $0.value > $1.value }.prefix(6).map { ($0.key, $0.value) }
        )
    }
}

struct ReportsView: View {
    @EnvironmentObject private var state: AppState
    let device: Device
    @State private var report: DeviceReport?
    /// Why the history couldn't be loaded. Was `try?`: a failure left the
    /// spinner up for ever.
    @State private var loadError: String?

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                if let report {
                    scoreCard(report)
                    hourlyCard(report)
                    typeCard(report)
                } else if let loadError {
                    SDErrorState(title: "Couldn't build the report", message: loadError) { await load() }
                } else {
                    ProgressView().padding(60)
                }
            }
            .padding()
        }
        .orynScreen()
        .navigationTitle("Report")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            if let report {
                ShareLink(item: shareText(report)) {
                    Image(systemName: "square.and.arrow.up")
                }
            }
        }
        .task { await load() }
    }

    private func load() async {
        do {
            let events = try await state.backend.events(deviceID: device.id, limit: 1000)
            report = DeviceReport.compute(from: events)
            loadError = nil
        } catch let error as APIError where error.isAuthFailure {
            state.expireSession(error.message)
        } catch {
            loadError = error.localizedDescription
        }
    }

    private func shareText(_ report: DeviceReport) -> String {
        let top = report.byType.prefix(3).map { "\($0.type) ×\($0.count)" }.joined(separator: ", ")
        return """
        SensorDyme report — \(device.name)
        Activity score (estimate, not a validated safety measure): \(report.safetyScore)/100
        \(report.totalAlerts) alerts (\(report.loudAlerts) serious)\(top.isEmpty ? "" : "\nMost frequent: \(top)")
        """
    }

    private func scoreCard(_ report: DeviceReport) -> some View {
        HStack(spacing: 20) {
            ZStack {
                Circle().stroke(SD.Colors.borderStrong, lineWidth: 10)
                Circle()
                    .trim(from: 0, to: Double(report.safetyScore) / 100)
                    .stroke(SD.Colors.accent, style: StrokeStyle(lineWidth: 10, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                Text("\(report.safetyScore)")
                    .font(.system(size: 34, weight: .heavy, design: .rounded))
            }
            .frame(width: 110, height: 110)

            VStack(alignment: .leading, spacing: 6) {
                Text("Activity score").font(.headline)
                SDEyebrow("Estimate", color: SD.Colors.warning)
                Text("\(report.totalAlerts) alerts in the loaded history — \(report.loudAlerts) serious.")
                    .font(.subheadline)
                    .foregroundStyle(SD.Colors.textSecondary)
                Text("Worked out on this phone from alert counts. Not a validated measure of safety.")
                    .font(.caption)
                    .foregroundStyle(SD.Colors.textTertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer()
        }
        .brandCard()
    }

    private func hourlyCard(_ report: DeviceReport) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Alerts by hour").font(.headline)
            if report.hourly.isEmpty {
                Text("No alerts yet").foregroundStyle(SD.Colors.textSecondary)
            } else {
                Chart(report.hourly, id: \.hour) { item in
                    BarMark(
                        x: .value("Hour", item.hour, unit: .hour),
                        y: .value("Alerts", item.count)
                    )
                    .foregroundStyle(SD.Colors.accent)
                    .cornerRadius(4)
                }
                .frame(height: 160)
            }
        }
        .brandCard()
    }

    private func typeCard(_ report: DeviceReport) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("What happened").font(.headline)
            Chart(report.byType, id: \.type) { item in
                SectorMark(
                    angle: .value("Count", item.count),
                    innerRadius: .ratio(0.6),
                    angularInset: 2
                )
                .cornerRadius(4)
                .foregroundStyle(by: .value("Type", item.type))
            }
            .frame(height: 220)
            .chartLegend(position: .bottom, spacing: 8)
        }
        .brandCard()
    }
}
