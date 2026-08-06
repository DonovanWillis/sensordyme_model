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
            LogoMark(size: 40, accented: true)
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
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(proposal.rules, id: \.detection) { rule in
                        HStack(alignment: .top, spacing: 8) {
                            Image(systemName: rule.severity == "critical" ? "bolt.fill" : "bell.fill")
                                .font(.caption)
                                .foregroundStyle(rule.severity == "critical" ? .red : .orange)
                                .padding(.top, 2)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(rule.description).font(.caption)
                                Text(ruleDetail(rule))
                                    .font(.caption2.monospaced())
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            }

            // The honest part: what the device can't do, and what it would take.
            if !proposal.unsupported.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Label("Not possible with this camera", systemImage: "exclamationmark.triangle")
                        .font(.caption.bold())
                        .foregroundStyle(.orange)
                    ForEach(proposal.unsupported, id: \.request) { item in
                        VStack(alignment: .leading, spacing: 2) {
                            Text(item.request).font(.caption)
                            Text("\(item.reason) — needs \(item.requires)")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .padding(10)
                .background(RoundedRectangle(cornerRadius: 12).fill(.orange.opacity(0.08)))
            }

            if proposal.source == "fallback" {
                Label("Configured offline (AI service unavailable) — using safe defaults", systemImage: "wifi.slash")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
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

    private func ruleDetail(_ rule: AIRule) -> String {
        var parts: [String] = ["\(rule.detection)"]
        if !rule.zone.isEmpty { parts.append("zone: \(rule.zone) (draw it in device settings)") }
        if rule.persist_s > 0 { parts.append("after \(Int(rule.persist_s))s") }
        if !rule.schedule.isEmpty { parts.append("hours \(rule.schedule) (scheduling coming soon)") }
        parts.append("cooldown \(Int(rule.cooldown_s))s")
        return parts.joined(separator: " · ")
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
