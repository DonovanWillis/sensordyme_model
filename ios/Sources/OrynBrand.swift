import SwiftUI
import UIKit

// MARK: - ORYN brand assets
//
// The mark, wordmark and lockup are ORYN's own vector artwork
// (ORYN-WEBSITE/public/brand/*.svg), carried in Assets.xcassets as template
// images with their vector data preserved. They are never redrawn as text:
// these views only size them at the artwork's true aspect ratio and tint them.

/// ORYN's symbol on its own. Used wherever the brand appears small: navigation,
/// the Ask Jupiter bar, empty states.
struct OrynMark: View {
    var height: CGFloat = 24
    var color: Color = SD.Colors.textPrimary

    /// From the artwork's viewBox (117.23 × 144.33).
    static let aspect: CGFloat = 117.23 / 144.33

    var body: some View {
        Image("OrynMark")
            .renderingMode(.template)
            .resizable()
            .aspectRatio(Self.aspect, contentMode: .fit)
            .frame(width: height * Self.aspect, height: height)
            .foregroundStyle(color)
            .accessibilityLabel("ORYN")
    }
}

/// The ORYN wordmark on its own.
struct OrynWordmark: View {
    var height: CGFloat = 18
    var color: Color = SD.Colors.textPrimary

    /// From the artwork's viewBox (485.5 × 71.45).
    static let aspect: CGFloat = 485.5 / 71.45

    var body: some View {
        Image("OrynWordmark")
            .renderingMode(.template)
            .resizable()
            .aspectRatio(Self.aspect, contentMode: .fit)
            .frame(width: height * Self.aspect, height: height)
            .foregroundStyle(color)
            .accessibilityLabel("ORYN")
    }
}

/// The app's full identity: ORYN's lockup with the EDGE product line beneath,
/// set as an ORYN label rather than a second logo.
struct OrynEdgeLockup: View {
    var markHeight: CGFloat = 64
    var showsProduct = true

    var body: some View {
        VStack(spacing: markHeight * 0.34) {
            OrynMark(height: markHeight)
            VStack(spacing: markHeight * 0.16) {
                OrynWordmark(height: markHeight * 0.3)
                if showsProduct {
                    Text("EDGE")
                        .font(.system(size: max(10, markHeight * 0.17), weight: .medium, design: .monospaced))
                        .kerning(markHeight * 0.11)
                        .foregroundStyle(SD.Colors.textTertiary)
                        // Kerning adds trailing space after the last letter;
                        // offset it so the word centres on its ink.
                        .padding(.leading, markHeight * 0.11)
                }
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("ORYN EDGE")
    }
}

/// The compact brand for navigation bars: mark plus "ORYN EDGE" in the label face.
struct OrynNavBrand: View {
    var body: some View {
        HStack(spacing: 8) {
            OrynMark(height: 15)
            Text("ORYN EDGE")
                .font(.system(size: 12, weight: .semibold, design: .monospaced))
                .kerning(2)
                .foregroundStyle(SD.Colors.textPrimary)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("ORYN EDGE")
    }
}

// MARK: - Inputs

/// A single ORYN text input: graphite field, hairline, a brighter hairline when
/// focused. Wraps whatever field it is given so secure / plain / vertical text
/// fields all share one look.
struct OrynFieldChrome<Field: View>: View {
    var icon: String?
    var focused: Bool
    var invalid = false
    @ViewBuilder var field: Field

    var body: some View {
        HStack(spacing: SD.Space.sm) {
            if let icon {
                Image(systemName: icon)
                    .font(.system(size: 14, weight: .regular))
                    .foregroundStyle(focused ? SD.Colors.textPrimary : SD.Colors.textDisabled)
                    .frame(width: 18)
            }
            field
                .font(SD.Typography.body)
                .foregroundStyle(SD.Colors.textPrimary)
                .tint(SD.Colors.signal)
        }
        .padding(.horizontal, SD.Space.md - 2)
        .frame(minHeight: 52)
        .background(
            RoundedRectangle(cornerRadius: SD.Radius.md, style: .continuous)
                .fill(SD.Colors.field)
        )
        .overlay(
            RoundedRectangle(cornerRadius: SD.Radius.md, style: .continuous)
                .strokeBorder(
                    invalid ? SD.Colors.danger.opacity(0.6)
                        : focused ? SD.Colors.borderFocus : SD.Colors.borderStrong,
                    lineWidth: 1
                )
        )
        .animation(SD.Motion.gentle, value: focused)
    }
}

/// Placeholder text in ORYN's disabled grey. SwiftUI's default placeholder is a
/// system grey that reads blue-ish on ink.
func orynPrompt(_ text: String) -> Text {
    Text(text).foregroundStyle(SD.Colors.textTertiary)
}

// MARK: - Ask Jupiter

/// The search-style input that is the front door to Jupiter, ORYN's language
/// layer: the ORYN mark on the left, "Ask Jupiter", and a minimal send arrow.
///
/// Purely an input. What a submitted request does — draft, review, activate —
/// is unchanged and lives in AIConfigureView; this bar only collects the words.
struct AskJupiterBar: View {
    @Binding var text: String
    var placeholder = "Ask Jupiter"
    var busy = false
    /// Multi-line growth for the full Ask Jupiter screen; one line inline.
    var expands = false
    var autoFocus = false
    var onSubmit: () -> Void

    @FocusState private var focused: Bool

    private var canSubmit: Bool { !busy && JupiterPrompt.canSubmit(text) }

    var body: some View {
        VStack(alignment: .trailing, spacing: SD.Space.xxs) {
            HStack(alignment: expands ? .top : .center, spacing: SD.Space.sm) {
                OrynMark(height: 18, color: focused ? SD.Colors.signal : SD.Colors.textPrimary)
                    .padding(.top, expands ? 3 + 7 : 0)
                    .animation(SD.Motion.gentle, value: focused)

                Group {
                    if expands {
                        TextField("", text: $text, prompt: orynPrompt(placeholder), axis: .vertical)
                            .lineLimit(1...5)
                            .padding(.vertical, 7)  // the margin the bar's padding gave up
                    } else {
                        TextField("", text: $text, prompt: orynPrompt(placeholder))
                            .lineLimit(1)
                    }
                }
                .font(SD.Typography.body)
                .foregroundStyle(SD.Colors.textPrimary)
                .tint(SD.Colors.signal)
                .focused($focused)
                .submitLabel(.send)
                .onSubmit(submit)
                .accessibilityLabel(placeholder)
                .accessibilityIdentifier("jupiter.input")

                Group {
                    if busy {
                        ProgressView()
                            .controlSize(.small)
                            .tint(SD.Colors.textPrimary)
                            .frame(width: 44, height: 44)
                            .accessibilityLabel("Jupiter is working")
                    } else {
                        Button(action: submit) {
                            Image(systemName: "arrow.up")
                                .font(.system(size: 14, weight: .semibold))
                                .foregroundStyle(canSubmit ? SD.Colors.textOnAccent : SD.Colors.textDisabled)
                                .frame(width: 30, height: 30)
                                .background(
                                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                                        .fill(canSubmit ? SD.Colors.accent : SD.Colors.surfaceRaised)
                                )
                                // 30 pt as drawn; 44 pt to touch.
                                .frame(width: 44, height: 44)
                                .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .disabled(!canSubmit)
                        .accessibilityLabel("Send to Jupiter")
                        .accessibilityIdentifier("jupiter.send")
                    }
                }
                .animation(SD.Motion.quick, value: canSubmit)
            }
            .padding(.leading, SD.Space.md)
            // The send button's 44 pt target carries 7 pt of its own margin, so the
            // padding shrinks by as much and the bar keeps its drawn size.
            .padding(.trailing, SD.Space.sm - 2 - 7)
            .padding(.vertical, SD.Space.sm - 1 - 7)
            .frame(minHeight: 54)
            .background(
                RoundedRectangle(cornerRadius: SD.Radius.xl, style: .continuous)
                    .fill(SD.Colors.field)
            )
            // An outline only while focused: the field's edge is its surface.
            .overlay(
                RoundedRectangle(cornerRadius: SD.Radius.xl, style: .continuous)
                    .strokeBorder(focused ? SD.Colors.borderFocus : Color.clear, lineWidth: 1)
            )
            .animation(SD.Motion.gentle, value: focused)
            .onAppear { if autoFocus { focused = true } }

            if JupiterPrompt.showsCount(text) {
                Text("\(JupiterPrompt.length(text)) / \(JupiterPrompt.maxLength)")
                    .font(SD.Typography.mono)
                    .foregroundStyle(JupiterPrompt.isOverLimit(text) ? SD.Colors.danger : SD.Colors.textTertiary)
                    .padding(.trailing, SD.Space.xxs)
            }
        }
    }

    private func submit() {
        guard canSubmit else { return }
        focused = false
        onSubmit()
    }
}

// MARK: - Rows

/// One navigable row inside a card: icon tile, title, optional detail, chevron.
/// Wrap in a Button or NavigationLink with `SDRowButtonStyle`.
struct OrynNavRow: View {
    let icon: String
    let title: String
    var detail: String?
    var trailing: String?
    var showsChevron = true

    var body: some View {
        HStack(spacing: SD.Space.sm) {
            Image(systemName: icon)
                .font(.system(size: 14, weight: .regular))
                .foregroundStyle(SD.Colors.textSecondary)
                .frame(width: 32, height: 32)
                .background(
                    RoundedRectangle(cornerRadius: SD.Radius.sm + 2, style: .continuous)
                        .fill(SD.Colors.surfaceRaised)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: SD.Radius.sm + 2, style: .continuous)
                        .strokeBorder(SD.Colors.border, lineWidth: 1)
                )
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(SD.Typography.body)
                    .foregroundStyle(SD.Colors.textPrimary)
                if let detail {
                    Text(detail)
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textTertiary)
                        .lineLimit(2)
                }
            }
            Spacer(minLength: SD.Space.xs)
            if let trailing {
                Text(trailing)
                    .font(SD.Typography.mono)
                    .foregroundStyle(SD.Colors.textTertiary)
            }
            if showsChevron {
                Image(systemName: "chevron.right")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(SD.Colors.textDisabled)
            }
        }
        .accessibilityElement(children: .combine)
    }
}

/// A hairline between rows, inset to start after the icon tile.
struct OrynRowDivider: View {
    var inset: CGFloat = 32 + SD.Space.sm + SD.Space.sm

    var body: some View {
        Rectangle()
            .fill(SD.Colors.border)
            .frame(height: 1)
            .padding(.leading, inset)
    }
}

// MARK: - Forms

extension View {
    /// ORYN treatment for a SwiftUI `Form` / `List`: the page shows through,
    /// rows sit on card surfaces with hairline separators.
    func orynFormStyle() -> some View {
        self
            .scrollContentBackground(.hidden)
            .background(SD.Colors.background)
            // Without an explicit bar background, a Form with a hidden content
            // background lost its large title on iOS 26 (seen in Settings).
            .toolbarBackground(SD.Colors.background, for: .navigationBar)
            .tint(SD.Colors.accent)
    }

    /// Applies ORYN row chrome to every row of the content it is attached to.
    func orynRows() -> some View {
        self
            .listRowBackground(SD.Colors.surface)
            .listRowSeparatorTint(SD.Colors.border)
    }

    /// The standard screen background, extending under the bars.
    func orynScreen() -> some View {
        self
            .background(SD.Colors.background.ignoresSafeArea())
            .toolbarBackground(SD.Colors.background, for: .navigationBar)
    }
}

// MARK: - System chrome

/// UIKit-level appearance for the parts SwiftUI does not expose: navigation
/// bars, the tab bar, segmented controls and alert buttons. Applied once at
/// launch, before the first window, so no screen ever renders in the old theme.
enum OrynAppearance {
    static func apply() {
        let ink = UIColor(SD.Colors.background)
        let deepInk = UIColor(Color(hex: 0x060707)) // ink-950
        let bone = UIColor(SD.Colors.textPrimary)
        let steel = UIColor(SD.Colors.textDisabled)
        let hairline = UIColor(SD.Colors.border)

        // Navigation bars
        let nav = UINavigationBarAppearance()
        nav.configureWithOpaqueBackground()
        nav.backgroundColor = ink
        nav.shadowColor = .clear
        nav.titleTextAttributes = [
            .foregroundColor: bone,
            .font: UIFont.systemFont(ofSize: 16, weight: .semibold),
        ]
        nav.largeTitleTextAttributes = [
            .foregroundColor: bone,
            .font: UIFont.systemFont(ofSize: 32, weight: .semibold),
            .kern: -0.6,
        ]
        let navButton = UIBarButtonItemAppearance()
        navButton.normal.titleTextAttributes = [
            .foregroundColor: bone,
            .font: UIFont.systemFont(ofSize: 16, weight: .medium),
        ]
        nav.buttonAppearance = navButton
        nav.doneButtonAppearance = navButton
        UINavigationBar.appearance().standardAppearance = nav
        UINavigationBar.appearance().scrollEdgeAppearance = nav
        UINavigationBar.appearance().compactAppearance = nav
        UINavigationBar.appearance().tintColor = bone

        // Tab bar
        let tab = UITabBarAppearance()
        tab.configureWithOpaqueBackground()
        tab.backgroundColor = deepInk
        tab.shadowColor = hairline
        for layout in [tab.stackedLayoutAppearance, tab.inlineLayoutAppearance, tab.compactInlineLayoutAppearance] {
            layout.normal.iconColor = steel
            layout.normal.titleTextAttributes = [
                .foregroundColor: steel,
                .font: UIFont.systemFont(ofSize: 10, weight: .medium),
            ]
            layout.selected.iconColor = bone
            layout.selected.titleTextAttributes = [
                .foregroundColor: bone,
                .font: UIFont.systemFont(ofSize: 10, weight: .semibold),
            ]
        }
        UITabBar.appearance().standardAppearance = tab
        UITabBar.appearance().scrollEdgeAppearance = tab

        // Segmented controls (sensitivity, notify, appearance pickers)
        let segmented = UISegmentedControl.appearance()
        segmented.selectedSegmentTintColor = UIColor(SD.Colors.borderStrong)
        segmented.backgroundColor = UIColor(SD.Colors.surfaceRaised)
        segmented.setTitleTextAttributes([
            .foregroundColor: UIColor(SD.Colors.textTertiary),
            .font: UIFont.systemFont(ofSize: 13, weight: .medium),
        ], for: .normal)
        segmented.setTitleTextAttributes([
            .foregroundColor: bone,
            .font: UIFont.systemFont(ofSize: 13, weight: .semibold),
        ], for: .selected)

        // Toggles: bone track would swallow the white knob, so "on" is ORYN signal.
        UISwitch.appearance().onTintColor = UIColor(SD.Colors.signal)

        // Alert and confirmation-dialog buttons
        UIView.appearance(whenContainedInInstancesOf: [UIAlertController.self]).tintColor = bone

        UIRefreshControl.appearance().tintColor = UIColor(SD.Colors.textTertiary)
    }
}
