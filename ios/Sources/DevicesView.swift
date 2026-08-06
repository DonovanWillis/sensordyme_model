import SwiftUI

struct DevicesView: View {
    @EnvironmentObject private var state: AppState
    @State private var devices: [Device] = []
    @State private var errorMessage: String?
    @State private var showClaim = false

    private let refreshTimer = Timer.publish(every: 10, on: .main, in: .common).autoconnect()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 14) {
                    // Shown whether or not devices are loaded: a failed refresh
                    // leaves stale cards on screen that look perfectly current.
                    if let errorMessage {
                        ConnectionBanner(message: errorMessage) { await reload() }
                    }
                    if devices.isEmpty && errorMessage == nil {
                        ContentUnavailableView(
                            "No devices yet",
                            systemImage: "video.slash",
                            description: Text("Pair your SensorDyme device with its claim code.")
                        )
                        .padding(.top, 60)
                    }
                    ForEach(devices) { device in
                        NavigationLink(value: device) {
                            DeviceCard(device: device)
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding()
            }
            .navigationDestination(for: Device.self) { device in
                DeviceDetailView(device: device)
            }
            .navigationTitle("Devices")
            .toolbar {
                ToolbarItem(placement: .principal) {
                    LogoMark(size: 22, gradient: true)
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        showClaim = true
                    } label: {
                        Label("Set up device", systemImage: "plus")
                    }
                }
            }
            .sheet(isPresented: $showClaim) {
                SetupWizardView { await reload() }
            }
            .refreshable { await reload() }
            .task { await reload() }
            .onReceive(refreshTimer) { _ in Task { await reload() } }
        }
    }

    private func reload() async {
        do {
            devices = try await state.backend.devices()
            errorMessage = nil
        } catch let error as APIError where error.isAuthFailure {
            state.expireSession(error.message)
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}

struct DeviceCard: View {
    let device: Device

    var body: some View {
        HStack(spacing: 14) {
            ZStack {
                RoundedRectangle(cornerRadius: 14)
                    .fill(device.online ? AnyShapeStyle(Brand.gradient) : AnyShapeStyle(.quaternary))
                    .frame(width: 52, height: 52)
                Image(systemName: "video.fill")
                    .foregroundStyle(.white)
            }
            VStack(alignment: .leading, spacing: 3) {
                Text(device.name).font(.headline)
                HStack(spacing: 6) {
                    Circle()
                        .fill(device.online ? .green : .gray.opacity(0.4))
                        .frame(width: 8, height: 8)
                    Text(device.online ? Preset.title(for: device.activePreset) : "Offline")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }
            Spacer()
            Image(systemName: "chevron.right")
                .font(.footnote.bold())
                .foregroundStyle(.tertiary)
        }
        .brandCard()
    }
}

struct ClaimDeviceView: View {
    @EnvironmentObject private var state: AppState
    @Environment(\.dismiss) private var dismiss
    @State private var code = ""
    @State private var errorMessage: String?
    @State private var busy = false
    let onClaimed: () async -> Void

    var body: some View {
        NavigationStack {
            Form {
                Section("Claim code") {
                    TextField("e.g. 1A2B-3C4D-5E6F", text: $code)
                        .autocapitalization(.allCharacters)
                        .autocorrectionDisabled()
                    Text("Printed on the card in the box (or by cloud/manage.py create-device).")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                if let errorMessage {
                    Text(errorMessage).foregroundStyle(.red)
                }
            }
            .navigationTitle("Pair Device")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button(busy ? "Pairing…" : "Pair") {
                        busy = true
                        Task {
                            defer { busy = false }
                            do {
                                try await state.backend.claim(code: code.trimmingCharacters(in: .whitespaces))
                                await onClaimed()
                                dismiss()
                            } catch {
                                errorMessage = error.localizedDescription
                            }
                        }
                    }
                    .disabled(code.isEmpty || busy)
                }
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
        }
    }
}
