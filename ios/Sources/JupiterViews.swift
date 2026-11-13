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

/// Ask Jupiter, floating above the tab bar on the device screen: a light
/// prompt surface that opens in place when focused (what Jupiter can be asked,
/// a comfortable field) and hands the request to the review sheet, which zooms
/// out of it. Nothing changes on the device from here: the sheet shows what
/// Jupiter understood and asks before activating.
///
/// The field is one stable view whatever the dock's state, so focus survives
/// the screen's 4 s refreshes (journey 10) and the suggestions appearing.
struct JupiterDock: View {
    let deviceName: String
    /// The request, normalised (JupiterPrompt), to open the review sheet with.
    var onSubmit: (String) -> Void
    /// The full Ask Jupiter screen with no request yet (all examples).
    var onOpenFull: () -> Void

    /// Owned here, not by the device screen, so typing redraws the dock only.
    @State private var draft = ""
    @FocusState private var focused: Bool
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.dynamicTypeSize) private var typeSize

    private var canSubmit: Bool { JupiterPrompt.canSubmit(draft) }

    var body: some View {
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
        .frame(minHeight: 52)
        .sdFloatingSurface(RoundedRectangle(cornerRadius: focused ? 24 : 26, style: .continuous))
        .animation(SD.Motion.resolved(SD.Motion.spatial, reduceMotion: reduceMotion), value: focused)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("jupiter.dock")
        .task {
            guard DebugLaunch.focusJupiter else { return }
            try? await Task.sleep(for: .seconds(2))
            focused = true
        }
    }

    @State private var submitted = 0

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
            // At accessibility sizes the chips would be clipped or tower over the
            // keyboard; "All examples" opens the full list instead.
            if !typeSize.isAccessibilitySize {
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
    }

    private func submit() {
        guard canSubmit, let request = JupiterPrompt.normalized(draft) else { return }
        submitted += 1
        focused = false
        draft = ""
        onSubmit(request)
    }
}
