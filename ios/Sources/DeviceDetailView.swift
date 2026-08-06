import SwiftUI

struct DeviceDetailView: View {
    @EnvironmentObject private var state: AppState
    @State var device: Device
    @State private var events: [EventItem] = []
    @State private var switchingTo: String?
    @State private var errorMessage: String?
    @State private var showAI = false
    @State private var showRename = false
    @State private var newName = ""

    private let refreshTimer = Timer.publish(every: 4, on: .main, in: .common).autoconnect()

    /// A loud alert in the last minute lights up the feed border + status.
    private var alertActive: Bool {
        guard let latest = events.first(where: { $0.severity == "loud" }), let date = latest.date else {
            return false
        }
        return Date().timeIntervalSince(date) < 60
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                FeedCard(device: device, alertActive: alertActive)

                statusRow

                aiButton

                modeCard

                historyCard

                if let errorMessage {
                    Text(errorMessage).font(.callout).foregroundStyle(.red)
                }
            }
            .padding()
        }
        .navigationTitle(device.name)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            NavigationLink {
                ReportsView(device: device)
            } label: {
                Label("Report", systemImage: "chart.bar.fill")
            }
            Menu {
                Button {
                    newName = device.name
                    showRename = true
                } label: {
                    Label("Rename device", systemImage: "pencil")
                }
            } label: {
                Image(systemName: "ellipsis.circle")
            }
        }
        .alert("Rename device", isPresented: $showRename) {
            TextField("Name", text: $newName)
            Button("Save") {
                Task {
                    try? await state.backend.renameDevice(deviceID: device.id, name: newName)
                    await reload()
                }
            }
            Button("Cancel", role: .cancel) {}
        }
        .sheet(isPresented: $showAI) {
            AIConfigureView(device: device) { await reload() }
        }
        .refreshable { await reload() }
        .task { await reload() }
        .onReceive(refreshTimer) { _ in Task { await reload() } }
    }

    private var statusRow: some View {
        HStack(spacing: 12) {
            statusChip(
                icon: device.online ? "wifi" : "wifi.slash",
                label: device.online ? "Online" : "Offline",
                style: device.online ? AnyShapeStyle(Brand.okGradient) : AnyShapeStyle(.quaternary)
            )
            statusChip(
                icon: alertActive ? "exclamationmark.triangle.fill" : "checkmark.shield.fill",
                label: alertActive ? "Alert" : "All clear",
                style: alertActive ? AnyShapeStyle(Brand.alertGradient) : AnyShapeStyle(.quaternary)
            )
            statusChip(
                icon: "square.grid.2x2.fill",
                label: Preset.title(for: device.activePreset),
                style: AnyShapeStyle(Brand.gradient)
            )
        }
    }

    private func statusChip(icon: String, label: String, style: AnyShapeStyle) -> some View {
        HStack(spacing: 6) {
            Image(systemName: icon).font(.caption)
            Text(label).font(.caption.bold()).lineLimit(1)
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 12).padding(.vertical, 8)
        .background(Capsule().fill(style))
        .frame(maxWidth: .infinity)
    }

    private var aiButton: some View {
        Button {
            showAI = true
        } label: {
            HStack {
                Image(systemName: "sparkles")
                VStack(alignment: .leading, spacing: 1) {
                    Text("Configure with AI").font(.headline)
                    Text("Describe what to watch — the model sets everything up")
                        .font(.caption)
                        .opacity(0.85)
                }
                Spacer()
                Image(systemName: "chevron.right").font(.footnote.bold())
            }
            .foregroundStyle(.white)
            .padding(16)
            .background(RoundedRectangle(cornerRadius: 20).fill(Brand.gradient))
        }
        .buttonStyle(.plain)
    }

    private var activeBase: String { Preset.split(device.activePreset).base }
    private var activeVariant: String? { Preset.split(device.activePreset).variant }

    private var modeCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Mode").font(.headline)
            ForEach(Preset.catalog) { preset in
                let isActive = activeBase == preset.name
                Button {
                    setMode(preset.name)
                } label: {
                    HStack {
                        Image(systemName: preset.icon)
                            .frame(width: 30)
                            .foregroundStyle(isActive ? AnyShapeStyle(Brand.gradient) : AnyShapeStyle(.secondary))
                        VStack(alignment: .leading, spacing: 2) {
                            Text(preset.title).foregroundStyle(.primary)
                            Text(preset.subtitle).font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                        if isActive {
                            Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                        } else if Preset.split(switchingTo).base == preset.name {
                            ProgressView()
                        }
                    }
                    .padding(.vertical, 6)
                }
                .buttonStyle(.plain)

                // tuning variants for the active mode (Teen Driver, Away, Newborn, ...)
                if isActive && !preset.variants.isEmpty {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 8) {
                            ForEach(preset.variants) { variant in
                                let selected = (activeVariant ?? preset.variants.first?.name) == variant.name
                                Button {
                                    setMode("\(preset.name):\(variant.name)")
                                } label: {
                                    Text(variant.title)
                                        .font(.caption.bold())
                                        .padding(.horizontal, 12).padding(.vertical, 7)
                                        .background(
                                            Capsule().fill(selected ? AnyShapeStyle(Brand.gradient) : AnyShapeStyle(.thinMaterial))
                                        )
                                        .foregroundStyle(selected ? .white : .primary)
                                }
                                .buttonStyle(.plain)
                            }
                        }
                        .padding(.vertical, 2)
                    }
                }

                if preset.id != Preset.catalog.last?.id {
                    Divider()
                }
            }
            if let switchingTo {
                Text("Switching to \(Preset.title(for: switchingTo))… the device applies it on its next sync.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
        .brandCard()
    }

    private var historyCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Activity").font(.headline)
                Spacer()
                NavigationLink("Full report") { ReportsView(device: device) }
                    .font(.subheadline)
            }
            if events.isEmpty {
                Text("No events yet").foregroundStyle(.secondary)
            }
            ForEach(events.prefix(25)) { event in
                EventRow(event: event)
                if event.id != events.prefix(25).last?.id {
                    Divider()
                }
            }
        }
        .brandCard()
    }

    private func setMode(_ name: String) {
        guard name != device.activePreset, Preset.split(switchingTo).base.isEmpty else { return }
        switchingTo = name
        Task {
            do {
                try await state.backend.setPreset(deviceID: device.id, preset: name)
            } catch {
                errorMessage = error.localizedDescription
                switchingTo = nil
            }
        }
    }

    private func reload() async {
        do {
            if let fresh = try await state.backend.devices().first(where: { $0.id == device.id }) {
                device = fresh
                if let switchingTo, fresh.activePreset == switchingTo {
                    self.switchingTo = nil
                }
            }
            events = try await state.backend.events(deviceID: device.id, limit: 200)
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}

struct EventRow: View {
    let event: EventItem

    private var color: Color {
        switch event.severity {
        case "loud": .red
        case "soft": .orange
        default: .secondary.opacity(0.5)
        }
    }

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Circle().fill(color).frame(width: 9, height: 9)
            VStack(alignment: .leading, spacing: 2) {
                Text(event.type).font(.subheadline.weight(.semibold))
                if let signals = event.signals, !signals.isEmpty {
                    Text(signals).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                }
            }
            Spacer()
            if let date = event.date {
                Text(date, style: .relative)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 2)
    }
}
