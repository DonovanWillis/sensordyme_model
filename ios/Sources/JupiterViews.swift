import SwiftUI

// MARK: - What Jupiter can be asked
//
// Suggestions are only requests the deterministic policy grammar reads in full
// (docs/PROMPT_POLICIES.md, "Grammar the parser reads"; jupiter/policy.py
// `parse`). Each one was checked against `parse(text, zones=["marked area"])`
// and yields exactly one policy: lingering, in_zone, absent, count. "The marked
// area" binds to the device's only drawn area, or waits for an area of that
// name (the preview says so before anything is activated).
//
// Anything else is labelled as what it is: the one example of a request
// Jupiter refuses is there to show the refusal and its reason.
enum JupiterSuggestions {
    struct Suggestion: Identifiable, Hashable {
        let text: String
        /// The policy it becomes, in the person's words.
        let kind: String
        var id: String { text }
    }

    static let supported: [Suggestion] = [
        .init(text: "Alert me if someone stays inside the marked area for longer than 10 seconds",
              kind: "Lingering"),
        .init(text: "Tell me when someone enters the marked area", kind: "Entering an area"),
        .init(text: "Let me know when nobody is in the marked area for 5 minutes", kind: "Area empty"),
        .init(text: "Alert me if more than 3 people are in the marked area", kind: "People count"),
        .init(text: "Tell me if a dog enters the marked area", kind: "Animal in an area"),
    ]

    /// Refused before any interpreter sees it (identity), with the reason.
    static let refused = Suggestion(
        text: "Tell me if someone I don't recognise enters the yard",
        kind: "Not possible: SensorDyme never identifies people")
}


// MARK: - The dock

/// What the device screen's scroll view tells the dock: where the page is.
/// Held by the device screen in `@State` and read only by the dock, so a
/// scroll redraws the dock (and only when its form changes), never the page.
@Observable
final class JupiterDockModel {
    /// The dock is the compact button because the page was scrolled down.
    private(set) var minimised = false
    /// Focused, or holding a typed request: the dock stays the field.
    @ObservationIgnored var engaged = false
    @ObservationIgnored private var tracker = JupiterDockBehavior.Tracker()

    /// One scroll sample from the device screen (`onScrollGeometryChange`).
    func scrolled(offset: Double, maxOffset: Double) {
        if tracker.scrolled(offset: offset, maxOffset: maxOffset, engaged: engaged) {
            minimised = tracker.minimised
        }
    }

    func restore() {
        tracker.restore()
        if minimised { minimised = false }
    }
}

/// Ask Jupiter, floating above the tab bar on the device screen.
///
/// At rest it is a light prompt field that opens in place when focused (what
/// Jupiter can be asked, a comfortable field) and hands the request to the
/// review sheet, which zooms out of it. Nothing changes on the device from
/// here: the sheet shows what Jupiter understood and asks before activating.
///
/// It steps aside while the page is read (JupiterDockBehavior): scrolled down,
/// it becomes a compact button at the trailing edge that a tap turns back into
/// the field. At accessibility text sizes it is always that button, and the
/// button opens the full Ask Jupiter screen, where the request has the whole
/// width. The device screen sets the dock as its bottom safe-area inset, so the
/// page always ends above it and nothing on the page is unreachable behind it.
///
/// The field is one stable view while it shows, so focus survives the screen's
/// 4 s refreshes (journey 10) and the suggestions appearing.
struct JupiterDock: View {
    let deviceName: String
    let model: JupiterDockModel
    var zoomNamespace: Namespace.ID
    var zoomEnabled = true
    /// The request, normalised (JupiterPrompt), to open the review sheet with.
    var onSubmit: (String) -> Void
    /// The full Ask Jupiter screen with no request yet (all examples).
    var onOpenFull: () -> Void

    /// Owned here, not by the device screen, so typing redraws the dock only.
    @State private var draft = ""
    @FocusState private var focused: Bool
    @State private var submitted = 0
    @State private var focusWhenShown = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.dynamicTypeSize) private var typeSize

    /// The height of the dock in either form, so the page's inset does not
    /// jump when it changes form.
    static let height: CGFloat = 52

    private var canSubmit: Bool { JupiterPrompt.canSubmit(draft) }

    private var form: JupiterDockBehavior.DockForm {
        // In use, it stays the field whatever the page did.
        if focused || !draft.isEmpty, !typeSize.isAccessibilitySize { return .field }
        return JupiterDockBehavior.form(accessibilitySize: typeSize.isAccessibilitySize, minimised: model.minimised)
    }

    var body: some View {
        HStack(spacing: 0) {
            Spacer(minLength: 0)
            switch form {
            case .field:
                field
                    .transition(.opacity)
            case .button:
                compactButton
                    .transition(.opacity)
            }
        }
        .animation(SD.Motion.resolved(SD.Motion.spatial, reduceMotion: reduceMotion), value: form)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("jupiter.dock")
        .onChange(of: focused) { _, isFocused in
            model.engaged = isFocused || !draft.isEmpty
            if isFocused { model.restore() }
        }
        .onChange(of: draft.isEmpty) { _, empty in model.engaged = focused || !empty }
        .task {
            guard DebugLaunch.focusJupiter else { return }
            try? await Task.sleep(for: .seconds(2))
            focused = true
        }
    }

    // MARK: Field

    private var field: some View {
        VStack(alignment: .leading, spacing: SD.Space.sm) {
            if focused {
                suggestions
                    .transition(.opacity)
            }
            HStack(alignment: .center, spacing: SD.Space.sm) {
                OrynMark(height: 17, color: focused ? SD.Colors.signal : SD.Colors.textPrimary)
                    .accessibilityHidden(true)
                TextField("", text: $draft,
                          prompt: Text("Ask Jupiter").foregroundStyle(SD.Colors.textTertiary))
                    .font(SD.Typography.body)
                    .foregroundStyle(SD.Colors.textPrimary)
                    .tint(SD.Colors.signal)
                    .focused($focused)
                    .submitLabel(.send)
                    .onSubmit(submit)
                    .accessibilityLabel("Ask Jupiter")
                    .accessibilityIdentifier("jupiter.input")
                Button(action: submit) {
                    Image(systemName: "arrow.up")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(canSubmit ? SD.Colors.textOnAccent : SD.Colors.textTertiary)
                        .frame(width: 32, height: 32)
                        .background(Circle().fill(canSubmit ? SD.Colors.accent : Color.white.opacity(0.08)))
                        // 32 pt as drawn; 44 pt to touch.
                        .frame(width: 44, height: 44)
                        .contentShape(Rectangle())
                }
                .buttonStyle(SDPressStyle(scale: 0.9))
                .disabled(!canSubmit)
                .sensoryFeedback(.impact(weight: .light), trigger: submitted)
                .accessibilityLabel("Send to Jupiter")
                .accessibilityIdentifier("jupiter.send")
            }
            if JupiterPrompt.showsCount(draft) {
                Text("\(JupiterPrompt.length(draft)) / \(JupiterPrompt.maxLength)")
                    .font(SD.Typography.mono)
                    .foregroundStyle(JupiterPrompt.isOverLimit(draft) ? SD.Colors.danger : SD.Colors.textTertiary)
                    .frame(maxWidth: .infinity, alignment: .trailing)
            }
        }
        .padding(.leading, SD.Space.md)
        .padding(.trailing, SD.Space.xxs)
        .padding(.vertical, focused ? SD.Space.sm : 2)
        .frame(minHeight: Self.height)
        .sdFloatingSurface(RoundedRectangle(cornerRadius: focused ? 24 : 26, style: .continuous))
        .sdZoomSource(.jupiter, in: zoomNamespace, cornerRadius: 26, enabled: zoomEnabled)
        .animation(SD.Motion.resolved(SD.Motion.spatial, reduceMotion: reduceMotion), value: focused)
        .onAppear {
            guard focusWhenShown else { return }
            focusWhenShown = false
            DispatchQueue.main.async { focused = true }
        }
    }

    // MARK: Compact

    /// The dock stepped aside: the Jupiter mark on the floating surface, the
    /// same 52 pt as the field so the page does not move. Fixed in size at
    /// every text size, like a tab bar item, with the Large Content Viewer for
    /// its name.
    private var compactButton: some View {
        Button {
            if typeSize.isAccessibilitySize {
                onOpenFull()
            } else {
                // The field is focused once it is on screen (onAppear below):
                // focus asked for in the same update as the field's arrival
                // can land on nothing.
                focusWhenShown = true
                model.restore()
            }
        } label: {
            OrynMark(height: 20, color: SD.Colors.signal)
                .accessibilityHidden(true)
                .frame(width: Self.height, height: Self.height)
                .contentShape(Circle())
        }
        .buttonStyle(SDPressStyle(scale: 0.92))
        .sdFloatingSurface(Circle())
        .sdZoomSource(.jupiter, in: zoomNamespace, cornerRadius: Self.height / 2, enabled: zoomEnabled)
        .accessibilityLabel("Ask Jupiter")
        .accessibilityHint(typeSize.isAccessibilitySize
                           ? "Opens Ask Jupiter to describe what \(deviceName) should watch for."
                           : "Opens the Ask Jupiter field.")
        .accessibilityShowsLargeContentViewer {
            Label("Ask Jupiter", systemImage: "sparkles")
        }
        .accessibilityIdentifier("jupiter.open")
    }

    // MARK: Suggestions

    private var suggestions: some View {
        VStack(alignment: .leading, spacing: SD.Space.xs) {
            HStack(alignment: .firstTextBaseline) {
                SDEyebrow("Jupiter can watch for")
                Spacer(minLength: SD.Space.xs)
                Button("All examples", action: onOpenFull)
                    .font(SD.Typography.caption.weight(.semibold))
                    .foregroundStyle(SD.Colors.textSecondary)
                    .frame(minHeight: 44)
                    .contentShape(Rectangle())
                    .accessibilityLabel("Jupiter examples")
                    .accessibilityIdentifier("detail.examples")
            }
            .padding(.trailing, SD.Space.sm)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: SD.Space.xs) {
                    ForEach(JupiterSuggestions.supported) { suggestion in
                        Button { draft = suggestion.text } label: {
                            Text(suggestion.text)
                                .font(SD.Typography.caption)
                                .foregroundStyle(SD.Colors.textSecondary)
                                .lineLimit(2)
                                .multilineTextAlignment(.leading)
                                .frame(width: 210, alignment: .leading)
                                .padding(.horizontal, SD.Space.sm)
                                .padding(.vertical, SD.Space.xs)
                                .frame(minHeight: 44)
                                .background(
                                    RoundedRectangle(cornerRadius: SD.Radius.md, style: .continuous)
                                        .fill(Color.white.opacity(0.06))
                                )
                        }
                        .buttonStyle(SDPressStyle())
                        .accessibilityLabel(suggestion.text)
                        .accessibilityHint("Fills in the request. \(suggestion.kind).")
                    }
                }
                .padding(.trailing, SD.Space.sm)
            }
        }
    }

    private func submit() {
        guard canSubmit, let request = JupiterPrompt.normalized(draft) else { return }
        submitted += 1
        focused = false
        draft = ""
        onSubmit(request)
    }
}

// MARK: - The device screen's side

/// Reports the device screen's scroll position to the dock (iOS 18+; on iOS 17
/// the dock simply stays the field at default text sizes).
struct JupiterDockScrollTracking: ViewModifier {
    let model: JupiterDockModel

    private struct Sample: Equatable {
        var offset: Double
        var maxOffset: Double
    }

    func body(content: Content) -> some View {
        if #available(iOS 18.0, *) {
            content.onScrollGeometryChange(for: Sample.self) { geometry in
                let top = geometry.contentOffset.y + geometry.contentInsets.top
                let max = geometry.contentSize.height + geometry.contentInsets.top + geometry.contentInsets.bottom
                    - geometry.containerSize.height
                return Sample(offset: top, maxOffset: max)
            } action: { _, sample in
                model.scrolled(offset: sample.offset, maxOffset: sample.maxOffset)
            }
        } else {
            content
        }
    }
}
