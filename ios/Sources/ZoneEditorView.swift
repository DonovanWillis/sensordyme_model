import SwiftUI

/// Draw the areas SensorDyme watches, over the camera's own view.
///
/// A camera cannot know what a kitchen is. It can know that a person's feet are
/// inside a shape you drew and named "kitchen" — which is why every rule about a
/// place depends on this screen existing.
///
/// Points are stored normalised (0..1) against the feed, not in screen
/// coordinates, so a zone drawn on a phone means the same region on a 4608px
/// sensor and survives rotating the device or changing capture resolution.
struct ZoneEditorView: View {
    @EnvironmentObject private var state: AppState
    @Environment(\.dismiss) private var dismiss
    let device: Device

    @State private var zones: [MonitorZone] = []
    @State private var draft: [CGPoint] = []
    @State private var draftName = ""
    @State private var draftKind = "restricted"
    @State private var editing = false
    @State private var loading = true
    @State private var errorMessage: String?
    @State private var saving = false
    @StateObject private var saveProgress = ProgressTracker()
    /// The live picture's real width / height, reported by the feed once a
    /// frame decodes (FeedAspectRatioKey). The canvas takes this shape so the
    /// picture fills it exactly and a tap normalised against the canvas is a
    /// point normalised against the camera frame — which is what the device
    /// tests polygons against. 4:3 (the shipped 640x480) until a frame arrives.
    @State private var feedAspect: CGFloat = 4 / 3

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: SD.Space.md) {
                    canvas
                    if editing {
                        draftControls
                    } else {
                        zoneList
                    }
                    if let errorMessage {
                        Label(errorMessage, systemImage: "exclamationmark.circle.fill")
                            .font(SD.Typography.callout)
                            .foregroundStyle(SD.Colors.danger)
                    }
                }
                .padding(.horizontal, SD.Space.md)
                .padding(.bottom, SD.Space.xl)
            }
            .orynScreen()
            .navigationTitle("Detection areas")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    if !editing {
                        Button { startDrawing() } label: { Label("Add", systemImage: "plus") }
                    }
                }
            }
            .task { await load() }
        }
        .tint(SD.Colors.accent)
    }

    // MARK: Canvas

    private var canvas: some View {
        GeometryReader { geometry in
            ZStack {
                FeedCard(device: device, alertActive: false)

                // Existing zones, dimmed while drawing so the new one is legible.
                ForEach(zones) { zone in
                    ZoneShape(points: zone.points)
                        .fill(colour(for: zone.kind).opacity(editing ? 0.08 : 0.18))
                        .overlay(
                            ZoneShape(points: zone.points)
                                .stroke(colour(for: zone.kind).opacity(editing ? 0.4 : 1),
                                        lineWidth: 2)
                        )
                }

                if editing {
                    ZoneShape(points: draft)
                        .fill(SD.Colors.accent.opacity(0.22))
                        .overlay(ZoneShape(points: draft).stroke(SD.Colors.accent, lineWidth: 2))
                    ForEach(Array(draft.enumerated()), id: \.offset) { _, point in
                        Circle()
                            .fill(SD.Colors.accent)
                            .frame(width: 12, height: 12)
                            .position(x: point.x * geometry.size.width,
                                      y: point.y * geometry.size.height)
                    }
                }
            }
            .contentShape(Rectangle())
            .onTapGesture { location in
                guard editing else { return }
                // Normalise on the way in: storing screen points would tie the
                // zone to the phone it was drawn on.
                draft.append(CGPoint(
                    x: min(max(location.x / geometry.size.width, 0), 1),
                    y: min(max(location.y / geometry.size.height, 0), 1)
                ))
            }
        }
        .aspectRatio(feedAspect, contentMode: .fit)
        .onPreferenceChange(FeedAspectRatioKey.self) { aspect in
            if let aspect, aspect.isFinite, aspect > 0 { feedAspect = aspect }
        }
        .clipShape(RoundedRectangle(cornerRadius: SD.Radius.lg, style: .continuous))
    }

    // MARK: Drawing

    private var draftControls: some View {
        SDCard {
            VStack(alignment: .leading, spacing: SD.Space.sm) {
                Text(draft.count < 3
                     ? "Tap the picture to outline the area — at least three corners."
                     : "\(draft.count) corners. Add more, or name it below.")
                    .font(SD.Typography.callout)
                    .foregroundStyle(SD.Colors.textSecondary)

                TextField("Name it (e.g. kitchen, pool)", text: $draftName)
                    .font(SD.Typography.body)
                    .textInputAutocapitalization(.never)
                    .padding(SD.Space.xs)
                    .background(RoundedRectangle(cornerRadius: SD.Radius.sm).fill(SD.Colors.surfaceRaised))

                Picker("Kind", selection: $draftKind) {
                    Text("Keep out").tag("restricted")
                    Text("Stay inside").tag("safe")
                    Text("Line").tag("boundary")
                }
                .pickerStyle(.segmented)

                Text(kindExplanation)
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textTertiary)
                    .fixedSize(horizontal: false, vertical: true)

                HStack(spacing: SD.Space.xs) {
                    Button("Undo point") { if !draft.isEmpty { draft.removeLast() } }
                        .buttonStyle(SDButtonStyle(kind: .secondary, fullWidth: false))
                        .disabled(draft.isEmpty)
                    Button("Cancel") { cancelDrawing() }
                        .buttonStyle(SDButtonStyle(kind: .tertiary, fullWidth: false))
                    Spacer()
                }

                Button {
                    saveDraft()
                } label: {
                    HStack(spacing: SD.Space.xs) {
                        Text("Save area")
                        SDProgress(phase: saveProgress.phase)
                    }
                }
                .buttonStyle(SDButtonStyle(kind: .primary))
                    .disabled(draft.count < 3 || draftName.trimmingCharacters(in: .whitespaces).isEmpty || saving)
            }
        }
    }

    private var kindExplanation: String {
        switch draftKind {
        case "safe": "Alerts when someone leaves this area — for a crib or a bed."
        case "boundary": "Alerts when someone crosses it in either direction."
        default: "Alerts when someone enters — for a pool, stairs, or a stove."
        }
    }

    // MARK: List

    @ViewBuilder private var zoneList: some View {
        if loading {
            SDLoadingState(message: "Loading areas…")
        } else if zones.isEmpty {
            SDEmptyState(
                icon: "square.dashed",
                title: "No areas yet",
                message: "Rules about places need an area to point at. Draw one and name it, then say things like \"alert me if someone enters the kitchen\".",
                actionTitle: "Draw an area",
                action: { startDrawing() }
            )
        } else {
            SDCard {
                VStack(alignment: .leading, spacing: SD.Space.sm) {
                    SDSectionHeader(title: zones.count == 1 ? "1 area" : "\(zones.count) areas")
                    ForEach(zones) { zone in
                        HStack(spacing: SD.Space.sm) {
                            RoundedRectangle(cornerRadius: 3)
                                .fill(colour(for: zone.kind))
                                .frame(width: 4, height: 30)
                            VStack(alignment: .leading, spacing: 1) {
                                Text(zone.name)
                                    .font(SD.Typography.body)
                                    .foregroundStyle(SD.Colors.textPrimary)
                                Text(zone.meaning)
                                    .font(SD.Typography.caption)
                                    .foregroundStyle(SD.Colors.textSecondary)
                            }
                            Spacer()
                            Button(role: .destructive) {
                                remove(zone)
                            } label: {
                                Image(systemName: "trash")
                            }
                            .buttonStyle(.plain)
                            .foregroundStyle(SD.Colors.danger)
                        }
                        if zone.id != zones.last?.id {
                            Divider().overlay(SD.Colors.border)
                        }
                    }
                }
            }
        }
    }

    private func colour(for kind: String) -> Color {
        switch kind {
        case "safe": SD.Colors.success
        case "boundary": SD.Colors.accent
        default: SD.Colors.danger
        }
    }

    // MARK: Actions

    private func startDrawing() {
        draft = []
        draftName = ""
        draftKind = "restricted"
        withAnimation(SD.Motion.standard) { editing = true }
    }

    private func cancelDrawing() {
        withAnimation(SD.Motion.standard) { editing = false }
        draft = []
    }

    private func saveDraft() {
        let name = draftName.trimmingCharacters(in: .whitespaces).lowercased()
        guard draft.count >= 3, !name.isEmpty else { return }
        // Replace rather than duplicate: two areas with one name would make any
        // rule naming it ambiguous.
        var updated = zones.filter { $0.name != name }
        updated.append(MonitorZone.from(points: draft, name: name, kind: draftKind))
        persist(updated) { withAnimation(SD.Motion.standard) { editing = false } }
    }

    private func remove(_ zone: MonitorZone) {
        persist(zones.filter { $0.id != zone.id })
    }

    private func persist(_ updated: [MonitorZone], onSuccess: @escaping () -> Void = {}) {
        saving = true
        errorMessage = nil
        Task {
            defer { saving = false }
            do {
                try await saveProgress.track {
                    try await state.backend.setZones(deviceID: device.id, zones: updated)
                }
                zones = updated
                draft = []
                onSuccess()
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    private func load() async {
        do {
            zones = try await state.backend.zones(deviceID: device.id)
            errorMessage = nil
        } catch let error as APIError where error.isAuthFailure {
            state.expireSession(error.message)
        } catch {
            errorMessage = error.localizedDescription
        }
        loading = false
    }
}

/// A closed polygon in normalised coordinates, scaled to whatever it is drawn in.
struct ZoneShape: Shape {
    let points: [CGPoint]

    func path(in rect: CGRect) -> Path {
        var path = Path()
        guard let first = points.first else { return path }
        path.move(to: CGPoint(x: first.x * rect.width, y: first.y * rect.height))
        for point in points.dropFirst() {
            path.addLine(to: CGPoint(x: point.x * rect.width, y: point.y * rect.height))
        }
        // Only close a real area: two points are a line being drawn, and closing
        // it would flash a filled sliver on every tap.
        if points.count >= 3 { path.closeSubpath() }
        return path
    }
}
