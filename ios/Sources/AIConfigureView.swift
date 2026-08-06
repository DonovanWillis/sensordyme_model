import SwiftUI

// MARK: - AI setup: describe what you want in plain language; the model configures the device.

struct AIConfigureView: View {
    @EnvironmentObject private var state: AppState
    @Environment(\.dismiss) private var dismiss
    let device: Device
    let onApplied: () async -> Void

    @State private var prompt = ""
    @State private var thinking = false
    @State private var proposal: AIProposal?
    @State private var applied = false
    @State private var errorMessage: String?

    private let examples = [
        "Watch my baby and alert me right away if her face gets covered",
        "Alert me if my grandpa falls or hasn't moved in 30 minutes",
        "Make phone use in the car trigger the loudest alarm instantly",
        "Watch the front door and alert me if anyone lingers",
    ]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    header

                    VStack(alignment: .leading, spacing: 10) {
                        TextField("Describe what the device should watch for…", text: $prompt, axis: .vertical)
                            .lineLimit(3...6)
                            .padding(12)
                            .background(RoundedRectangle(cornerRadius: 14).fill(.background.secondary))
                            .overlay(RoundedRectangle(cornerRadius: 14).strokeBorder(.quaternary))

                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 8) {
                                ForEach(examples, id: \.self) { example in
                                    Button {
                                        prompt = example
                                    } label: {
                                        Text(example)
                                            .font(.caption)
                                            .lineLimit(1)
                                            .padding(.horizontal, 12).padding(.vertical, 8)
                                            .background(Capsule().fill(.thinMaterial))
                                    }
                                    .buttonStyle(.plain)
                                }
                            }
                        }
                    }

                    Button {
                        configure()
                    } label: {
                        if thinking {
                            HStack(spacing: 10) {
                                ProgressView().tint(.white)
                                Text("Configuring…")
                            }
                        } else {
                            Label("Configure with AI", systemImage: "sparkles")
                        }
                    }
                    .buttonStyle(BrandButtonStyle())
                    .disabled(prompt.trimmingCharacters(in: .whitespaces).isEmpty || thinking)

                    if let errorMessage {
                        Text(errorMessage).font(.callout).foregroundStyle(.red)
                    }

                    if let proposal {
                        proposalCard(proposal)
                    }
                }
                .padding()
            }
            .navigationTitle("AI Setup")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private var header: some View {
        HStack(spacing: 14) {
            LogoMark(size: 40, gradient: true)
            VStack(alignment: .leading, spacing: 2) {
                Text("Tell it what to watch").font(.headline)
                Text("Plain language in, a full device configuration out.")
                    .font(.subheadline).foregroundStyle(.secondary)
            }
        }
    }

    private func proposalCard(_ proposal: AIProposal) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Image(systemName: "wand.and.stars")
                Text(Preset.title(for: proposal.preset)).font(.headline)
                Spacer()
                if applied {
                    Label("Applied", systemImage: "checkmark.circle.fill")
                        .font(.subheadline.bold())
                        .foregroundStyle(.green)
                }
            }

            Text(proposal.explanation)
                .font(.subheadline)

            if !proposal.rules.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(proposal.rules, id: \.self) { rule in
                        Label(rule, systemImage: "bolt.fill")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            if !proposal.settings.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        ForEach(proposal.settings.sorted(by: { $0.key < $1.key }), id: \.key) { key, value in
                            Text("\(key): \(value)")
                                .font(.caption2.monospaced())
                                .padding(.horizontal, 10).padding(.vertical, 6)
                                .background(Capsule().fill(.thinMaterial))
                        }
                    }
                }
            }

            if !applied {
                Button {
                    apply(proposal)
                } label: {
                    Label("Apply to \(device.name)", systemImage: "arrow.down.circle.fill")
                }
                .buttonStyle(BrandButtonStyle())
            }
        }
        .padding(16)
        .background(RoundedRectangle(cornerRadius: 20).fill(.background.secondary))
        .overlay(RoundedRectangle(cornerRadius: 20).strokeBorder(Brand.gradient, lineWidth: 1.5))
        .transition(.move(edge: .bottom).combined(with: .opacity))
    }

    private func configure() {
        thinking = true
        errorMessage = nil
        applied = false
        Task {
            defer { thinking = false }
            do {
                let result = try await state.backend.aiConfigure(deviceID: device.id, prompt: prompt)
                withAnimation(.spring(duration: 0.5)) { proposal = result }
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    private func apply(_ proposal: AIProposal) {
        Task {
            do {
                try await state.backend.setPreset(deviceID: device.id, preset: proposal.preset)
                withAnimation { applied = true }
                await onApplied()
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }
}
