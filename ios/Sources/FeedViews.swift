import SwiftUI

// MARK: - Live feed area: real MJPEG on the LAN, simulated viewport in demo mode.

struct FeedCard: View {
    @EnvironmentObject private var state: AppState
    let device: Device
    let alertActive: Bool

    private var isLiveFeed: Bool {
        !state.isDemo && !state.deviceLocalURL.isEmpty && URL(string: state.deviceLocalURL) != nil
    }

    var body: some View {
        ZStack {
            if state.isDemo {
                SimulatedFeedView(preset: device.activePreset ?? "driver_safety", alert: alertActive)
            } else if isLiveFeed, let url = URL(string: state.deviceLocalURL) {
                MJPEGView(url: url)  // MJPEGView appends /stream (and the device token) itself
            } else {
                feedPlaceholder
            }
        }
        // The live feed sizes itself to the camera's real aspect ratio (never
        // cropped — see MJPEGView); the simulated feed and placeholder keep a
        // fixed card height.
        .frame(height: isLiveFeed ? nil : 230)
        .frame(maxWidth: .infinity)
        .clipShape(RoundedRectangle(cornerRadius: 22))
        .overlay(
            RoundedRectangle(cornerRadius: 22)
                .strokeBorder(alertActive ? AnyShapeStyle(SD.Colors.danger) : AnyShapeStyle(.quaternary),
                              lineWidth: alertActive ? 3 : 1)
        )
        .animation(.easeInOut(duration: 0.4), value: alertActive)
    }

    private var feedPlaceholder: some View {
        ZStack {
            SD.Colors.canvas
            VStack(spacing: 10) {
                Image(systemName: "video.slash.fill").font(.title).foregroundStyle(SD.Colors.textSecondary)
                Text("Live feed streams on your home/car network.\nSet the device address in Settings.")
                    .font(.caption)
                    .multilineTextAlignment(.center)
                    .foregroundStyle(SD.Colors.textSecondary)
            }
        }
    }
}

// MARK: - Simulated demo feed: stylized viewport with scan motion and detection chips.

struct SimulatedFeedView: View {
    let preset: String
    let alert: Bool

    private var sceneIcon: String {
        switch Preset.split(preset).base {
        case "baby_monitoring": "figure.child"
        case "elderly_monitoring": "figure.walk"
        case "security_monitoring": "door.left.hand.open"
        default: "car.side.fill"
        }
    }

    var body: some View {
        TimelineView(.animation(minimumInterval: 1 / 30)) { context in
            let t = context.date.timeIntervalSinceReferenceDate
            ZStack {
                SD.Colors.canvas

                // subject silhouette with gentle "breathing" motion
                Image(systemName: sceneIcon)
                    .font(.system(size: 90))
                    .foregroundStyle(.white.opacity(0.16))
                    .scaleEffect(1 + 0.02 * sin(t * 1.4))
                    .offset(x: 12 * sin(t * 0.35), y: 8)

                // tracking box drifting like a live detector
                RoundedRectangle(cornerRadius: 10)
                    .strokeBorder(alert ? AnyShapeStyle(SD.Colors.danger) : AnyShapeStyle(SD.Colors.accent), lineWidth: 2)
                    .frame(width: 130, height: 150)
                    .offset(x: 12 * sin(t * 0.35), y: 4 + 4 * sin(t * 0.9))

                // sweeping scan line
                LinearGradient(colors: [.clear, SD.Colors.accent.opacity(0.35), .clear],
                               startPoint: .top, endPoint: .bottom)
                    .frame(height: 46)
                    .offset(y: CGFloat(((t * 40).truncatingRemainder(dividingBy: 300)) - 150))
                    .blendMode(.screen)

                VStack {
                    HStack {
                        Label("LIVE", systemImage: "dot.radiowaves.left.and.right")
                            .font(.caption2.bold())
                            .padding(.horizontal, 10).padding(.vertical, 5)
                            .background(Capsule().fill(.red.opacity(0.85)))
                            .foregroundStyle(.white)
                        Spacer()
                        Text(Preset.title(for: preset))
                            .font(.caption2.bold())
                            .padding(.horizontal, 10).padding(.vertical, 5)
                            .background(Capsule().fill(.ultraThinMaterial))
                    }
                    Spacer()
                    HStack {
                        Label("demo device — simulated feed", systemImage: "sparkles")
                            .font(.caption2)
                            .foregroundStyle(.white.opacity(0.65))
                        Spacer()
                    }
                }
                .padding(12)
            }
        }
    }
}

// MARK: - Real MJPEG stream reader (the device dashboard's /stream endpoint on the LAN).

struct MJPEGView: View {
    let url: URL
    @EnvironmentObject private var state: AppState
    @StateObject private var reader = MJPEGReader()

    var body: some View {
        ZStack {
            SD.Colors.canvas
            if let image = reader.image {
                // FULL frame, always: fit (letterbox) at the camera's real aspect
                // ratio, never fill+crop — cropping a safety camera hides exactly
                // the edges where events happen. The container adopts the stream's
                // aspect ratio so letterboxing only appears if the width forces it.
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: .infinity)
                    .aspectRatio(image.size.height > 0 ? image.size.width / image.size.height : 4 / 3,
                                 contentMode: .fit)
            } else {
                ProgressView().tint(.white)
                    .frame(height: 230)
            }
            if reader.isReconnecting {
                VStack {
                    Spacer()
                    Label("reconnecting…", systemImage: "wifi.exclamationmark")
                        .font(.caption2)
                        .padding(6)
                        .background(.black.opacity(0.6), in: Capsule())
                        .foregroundStyle(.white)
                        .padding(8)
                }
            }
        }
        .onAppear { reader.start(url: url, token: state.deviceToken) }
        .onDisappear { reader.stop() }
    }
}

@MainActor
final class MJPEGReader: NSObject, ObservableObject, URLSessionDataDelegate {
    @Published var image: UIImage?
    @Published var isReconnecting = false

    private var session: URLSession?
    private var buffer = Data()
    private var url: URL?
    private var token = ""
    private var stopped = true
    private var retryAttempt = 0
    private var retryTask: Task<Void, Never>?

    func start(url: URL, token: String = "") {
        stopSession()
        self.url = url
        self.token = token
        stopped = false
        retryAttempt = 0
        connect()
    }

    func stop() {
        stopped = true
        retryTask?.cancel()
        retryTask = nil
        stopSession()
    }

    private func stopSession() {
        session?.invalidateAndCancel()
        session = nil
        buffer.removeAll()
    }

    private func connect() {
        guard let url, !stopped else { return }
        var request = URLRequest(url: url.appendingPathComponent("stream"))
        request.timeoutInterval = 15
        if !token.isEmpty {
            request.setValue(token, forHTTPHeaderField: "X-Device-Token")  // dashboard.token support
        }
        let session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
        self.session = session
        session.dataTask(with: request).resume()
    }

    /// Any drop (device rebooted, Wi-Fi blip, app came back from background)
    /// retries with capped exponential backoff instead of spinning forever.
    private func scheduleReconnect() {
        guard !stopped else { return }
        isReconnecting = true
        stopSession()
        let delay = min(8.0, 0.5 * pow(2.0, Double(retryAttempt)))
        retryAttempt += 1
        retryTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            guard let self, !Task.isCancelled else { return }
            await MainActor.run { self.connect() }
        }
    }

    nonisolated func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        Task { @MainActor in
            self.buffer.append(data)
            // JPEG frames are delimited by SOI (FFD8) ... EOI (FFD9)
            while let start = self.buffer.range(of: Data([0xFF, 0xD8])),
                  let end = self.buffer.range(of: Data([0xFF, 0xD9]), in: start.lowerBound..<self.buffer.endIndex) {
                let frame = self.buffer.subdata(in: start.lowerBound..<end.upperBound)
                self.buffer.removeSubrange(..<end.upperBound)
                if let image = UIImage(data: frame) {
                    self.image = image
                    self.isReconnecting = false
                    self.retryAttempt = 0  // healthy again: future drops start with a short retry
                }
            }
            if self.buffer.count > 4_000_000 { self.buffer.removeAll() }  // corrupt stream guard
        }
    }

    nonisolated func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        Task { @MainActor in
            self.scheduleReconnect()
        }
    }
}
