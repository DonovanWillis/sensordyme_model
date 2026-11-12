import SwiftUI

// MARK: - The camera on the device screen
//
// The centrepiece of the device home: the real picture when the phone can
// reach the device on the LAN (MJPEG, one shared stream), the demo feed in demo
// mode, and otherwise a placeholder that says plainly why there is no picture.
// Never a fake snapshot: the backend holds no frames (privacy by design).

extension StatusTone {
    /// The design-system colour for a tone.
    var tint: Color {
        switch self {
        case .ok: SD.Colors.success
        case .caution: SD.Colors.warning
        case .critical: SD.Colors.danger
        case .neutral: SD.Colors.textTertiary
        }
    }
}

/// Whether this phone can show a device's live picture, and from where. The
/// same rule FeedCard has always used (FeedLink), in one place.
struct CameraSource {
    enum Kind { case demo, live(URL), none }
    let kind: Kind

    @MainActor
    init(state: AppState, device: Device) {
        if state.isDemo {
            kind = .demo
        } else if !state.deviceLocalURL.isEmpty, let url = URL(string: state.deviceLocalURL),
                  FeedLink.showsLiveFeed(deviceHardwareID: device.hardwareId,
                                         linkedID: state.linkedDeviceID,
                                         accountDeviceCount: state.accountDeviceCount) {
            kind = .live(url)
        } else {
            kind = .none
        }
    }

    var hasPicture: Bool {
        if case .none = kind { return false }
        return true
    }
}

/// The inline camera: picture, a frame whose tone follows the device's state,
/// and the controls floating over it.
struct CameraViewport: View {
    @EnvironmentObject private var state: AppState
    let device: Device
    let instrument: InstrumentState
    let session: LiveFeedSession
    var onExpand: () -> Void

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var source: CameraSource { CameraSource(state: state, device: device) }

    /// The frame is quiet unless the state calls for it: an alert or a failure
    /// in danger, a degradation in amber. Ready is not outlined in green: the
    /// readout below says it, and a green box around a picture reads as "safe".
    private var frameColor: Color {
        switch instrument.kind {
        case .alert, .failed: SD.Colors.danger
        case .degraded: SD.Colors.warning.opacity(0.7)
        default: Color.white.opacity(0.08)
        }
    }

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: SD.Radius.camera, style: .continuous)
        ZStack {
            switch source.kind {
            case .demo:
                SimulatedFeedView(preset: device.activePreset ?? "driver_safety", alert: instrument.kind == .alert)
                    .aspectRatio(4 / 3, contentMode: .fit)
            case .live(let url):
                MJPEGView(url: url, session: session)
            case .none:
                CameraPlaceholder(interrupted: instrument.isInterrupted)
            }
        }
        .frame(maxWidth: .infinity)
        .clipShape(shape)
        .overlay(shape.strokeBorder(frameColor, lineWidth: instrument.kind == .alert ? 2 : 1))
        .overlay(alignment: .topLeading) {
            if case .live = source.kind {
                MJPEGLiveIndicator(reader: session.reader)
                    .padding(SD.Space.sm)
            }
        }
        .overlay(alignment: .topTrailing) {
            if source.hasPicture {
                Button(action: onExpand) {
                    Image(systemName: "arrow.up.left.and.arrow.down.right")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(SD.Colors.textPrimary)
                        .frame(width: 36, height: 36)
                        .background(Circle().fill(Color.black.opacity(0.6)))
                        .frame(width: 44, height: 44)
                        .contentShape(Rectangle())
                }
                .buttonStyle(SDPressStyle(scale: 0.92))
                .padding(SD.Space.xxs)
                .accessibilityLabel("Full screen camera")
                .accessibilityIdentifier("camera.expand")
            }
        }
        .animation(SD.Motion.resolved(SD.Motion.status, reduceMotion: reduceMotion), value: instrument.kind)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("camera.viewport")
    }
}

/// Why there is no picture, in the camera's own shape: a viewfinder, the
/// reason, and nothing pretending to be footage. Dimmed while the device isn't
/// reporting, so the screen visibly stops looking live.
struct CameraPlaceholder: View {
    var interrupted = false

    var body: some View {
        ZStack {
            SD.Colors.canvas
            // A faint vignette lifts the centre; static, drawn once.
            RadialGradient(colors: [Color.white.opacity(interrupted ? 0.02 : 0.05), .clear],
                           center: .center, startRadius: 4, endRadius: 260)
            ViewfinderCorners()
                .stroke(Color.white.opacity(interrupted ? 0.12 : 0.22),
                        style: StrokeStyle(lineWidth: 1.5, lineCap: .round))
                .padding(SD.Space.md)
            VStack(spacing: SD.Space.xs) {
                Image(systemName: interrupted ? "video.slash" : "video")
                    .font(.system(size: 22, weight: .light))
                    .foregroundStyle(SD.Colors.textTertiary)
                    .accessibilityHidden(true)
                Text("Live picture on your Wi-Fi")
                    .font(SD.Typography.label)
                    .foregroundStyle(SD.Colors.textSecondary)
                // No longer "set the device address in Settings": nobody types an
                // address any more; the device is found over Bonjour, and the
                // only thing a person can do about an empty address is get the
                // device powered on and onto this Wi-Fi.
                Text("The picture comes straight from your SensorDyme, so it needs to be switched on and on this Wi-Fi. ORYN EDGE finds it by itself.")
                    .font(SD.Typography.caption)
                    .multilineTextAlignment(.center)
                    .foregroundStyle(SD.Colors.textTertiary)
                    // The camera keeps its shape at every text size; the
                    // sentence shrinks to fit rather than being cut off.
                    .minimumScaleFactor(0.5)
            }
            .padding(.horizontal, SD.Space.xl)
            .padding(.vertical, SD.Space.lg)
            .accessibilityElement(children: .combine)
        }
        .frame(minHeight: 200)
        .aspectRatio(16 / 10, contentMode: .fit)
    }
}

/// Four corner marks, like a viewfinder: a camera without pretending to be one.
struct ViewfinderCorners: Shape {
    var length: CGFloat = 18

    func path(in rect: CGRect) -> Path {
        var path = Path()
        let l = min(length, rect.width / 4, rect.height / 4)
        // top-left
        path.move(to: CGPoint(x: rect.minX, y: rect.minY + l))
        path.addLine(to: CGPoint(x: rect.minX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.minX + l, y: rect.minY))
        // top-right
        path.move(to: CGPoint(x: rect.maxX - l, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.minY + l))
        // bottom-right
        path.move(to: CGPoint(x: rect.maxX, y: rect.maxY - l))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY))
        path.addLine(to: CGPoint(x: rect.maxX - l, y: rect.maxY))
        // bottom-left
        path.move(to: CGPoint(x: rect.minX + l, y: rect.maxY))
        path.addLine(to: CGPoint(x: rect.minX, y: rect.maxY))
        path.addLine(to: CGPoint(x: rect.minX, y: rect.maxY - l))
        return path
    }
}

// MARK: - Full screen

/// The camera, full screen: the same stream as the inline preview (no second
/// connection), the device's state along the bottom, and a close control.
/// Drag down or use the close button (or the VoiceOver escape gesture) to go
/// back.
struct LiveCameraView: View {
    @EnvironmentObject private var state: AppState
    @Environment(\.dismiss) private var dismiss
    let device: Device
    let instrument: InstrumentState
    /// The presence word ("Online", "Stale" ...), as the device screen shows it.
    let presence: String
    let session: LiveFeedSession

    @State private var drag: CGFloat = 0

    var body: some View {
        let source = CameraSource(state: state, device: device)
        ZStack {
            Color.black.ignoresSafeArea()
            Group {
                switch source.kind {
                case .demo:
                    SimulatedFeedView(preset: device.activePreset ?? "driver_safety",
                                      alert: instrument.kind == .alert)
                        .aspectRatio(4 / 3, contentMode: .fit)
                case .live(let url):
                    MJPEGView(url: url, session: session)
                case .none:
                    CameraPlaceholder(interrupted: true)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .offset(y: max(0, drag))
        }
        .overlay(alignment: .top) { topBar }
        .overlay(alignment: .bottom) { bottomBar }
        .gesture(
            DragGesture(minimumDistance: 12)
                .onChanged { drag = $0.translation.height }
                .onEnded { value in
                    if value.translation.height > 120 { dismiss() } else {
                        withAnimation(SD.Motion.standard) { drag = 0 }
                    }
                }
        )
        .preferredColorScheme(.dark)
        .accessibilityAction(.escape) { dismiss() }
    }

    private var topBar: some View {
        HStack(alignment: .center, spacing: SD.Space.sm) {
            if case .live = CameraSource(state: state, device: device).kind {
                MJPEGLiveIndicator(reader: session.reader)
            }
            Spacer()
            Button { dismiss() } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(SD.Colors.textPrimary)
                    .frame(width: 40, height: 40)
                    .sdFloatingSurface(Circle(), shadow: false)
                    .frame(width: 44, height: 44)
                    .contentShape(Rectangle())
            }
            .buttonStyle(SDPressStyle(scale: 0.92))
            .accessibilityLabel("Close full screen camera")
            .accessibilityIdentifier("camera.close")
        }
        .padding(.horizontal, SD.Space.md)
        .padding(.top, SD.Space.xs)
    }

    private var bottomBar: some View {
        VStack(alignment: .leading, spacing: SD.Space.xxs) {
            Text(device.name)
                .font(SD.Typography.headline)
                .foregroundStyle(SD.Colors.textPrimary)
                .fixedSize(horizontal: false, vertical: true)
            HStack(spacing: SD.Space.xs) {
                Image(systemName: instrument.symbol)
                    .foregroundStyle(instrument.tone.tint)
                    .accessibilityHidden(true)
                Text(instrument.word)
                    .foregroundStyle(SD.Colors.textPrimary)
                Text("·").foregroundStyle(SD.Colors.textTertiary).accessibilityHidden(true)
                Text(presence)
                    .foregroundStyle(SD.Colors.textSecondary)
            }
            .font(SD.Typography.callout)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, SD.Space.lg)
        .padding(.top, SD.Space.xl)
        .padding(.bottom, SD.Space.md)
        .background(
            LinearGradient(colors: [.clear, Color.black.opacity(0.85)], startPoint: .top, endPoint: .bottom)
                .ignoresSafeArea()
        )
        .accessibilityElement(children: .combine)
    }
}
