import SwiftUI

/// A small spinner that resolves into a tick or a cross, then fades.
///
/// The gap this fills: a spinner that simply disappears leaves no evidence
/// anything happened. On a slow action the user looks away, looks back, and
/// cannot tell whether it finished, failed, or never started — so they press the
/// button again. Holding a result for a beat before fading is what turns
/// "something was happening" into "that worked".
///
/// Deliberately small and quiet. This marks the outcome of one action next to
/// the control that started it; anything that needs a paragraph belongs in an
/// error state, not here.
struct SDProgress: View {
    enum Phase: Equatable {
        case idle
        case working
        case done
        case failed
    }

    let phase: Phase
    var size: CGFloat = 16

    var body: some View {
        Group {
            switch phase {
            case .idle:
                Color.clear.frame(width: size, height: size)
            case .working:
                ProgressView()
                    .controlSize(.small)
                    .frame(width: size, height: size)
            case .done:
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: size))
                    .foregroundStyle(SD.Colors.success)
                    .transition(.scale(scale: 0.6).combined(with: .opacity))
            case .failed:
                Image(systemName: "exclamationmark.circle.fill")
                    .font(.system(size: size))
                    .foregroundStyle(SD.Colors.danger)
                    .transition(.scale(scale: 0.6).combined(with: .opacity))
            }
        }
        .animation(SD.Motion.standard, value: phase)
        .accessibilityLabel(label)
        // A tick that is only a colour is invisible to a colour-blind user and
        // silent to VoiceOver; the state is always announced.
        .accessibilityHidden(phase == .idle)
    }

    private var label: String {
        switch phase {
        case .idle: ""
        case .working: "Working"
        case .done: "Done"
        case .failed: "Failed"
        }
    }
}

/// Drives an `SDProgress` through working → done/failed → idle.
///
/// Owning the timing here rather than in each screen is the point: every action
/// in the app then shows its result for the same length of time, and no screen
/// has to remember to clear the tick.
@MainActor
final class ProgressTracker: ObservableObject {
    @Published private(set) var phase: SDProgress.Phase = .idle

    /// How long a finished result stays visible. Long enough to be noticed
    /// after glancing away, short enough not to look stuck.
    private let holdSeconds: Double
    private var reset: Task<Void, Never>?

    init(holdSeconds: Double = 1.6) {
        self.holdSeconds = holdSeconds
    }

    /// Run `work`, showing progress and then its outcome.
    ///
    /// Rethrows, so the caller still handles the error — this only reports that
    /// one happened. Swallowing it here would hide failures behind a small red
    /// icon, which is precisely the wrong place for them.
    @discardableResult
    func track<T>(_ work: () async throws -> T) async rethrows -> T {
        reset?.cancel()
        phase = .working
        do {
            let result = try await work()
            finish(.done)
            return result
        } catch {
            finish(.failed)
            throw error
        }
    }

    func begin() {
        reset?.cancel()
        phase = .working
    }

    func succeed() { finish(.done) }
    func fail() { finish(.failed) }

    private func finish(_ outcome: SDProgress.Phase) {
        phase = outcome
        reset = Task { [weak self] in
            guard let self else { return }
            try? await Task.sleep(for: .seconds(holdSeconds))
            guard !Task.isCancelled else { return }
            self.phase = .idle
        }
    }
}

/// Attaches a progress marker to any control, trailing its label.
struct ProgressAffordance<Content: View>: View {
    @ObservedObject var tracker: ProgressTracker
    @ViewBuilder var content: Content

    var body: some View {
        HStack(spacing: SD.Space.xs) {
            content
            SDProgress(phase: tracker.phase)
        }
    }
}
