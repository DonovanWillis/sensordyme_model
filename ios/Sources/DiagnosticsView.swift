import SwiftUI

/// Connection doctor: runs the whole chain in order and shows exactly which
/// stage fails — backend config, backend reachability, auth, device config,
/// device reachability, live stream. Each failure says what to fix.
struct DiagnosticsView: View {
    @EnvironmentObject private var state: AppState
    @State private var results: [StageResult] = []
    @State private var running = false

    struct StageResult: Identifiable {
        let id = UUID()
        let name: String
        let passed: Bool?   // nil = skipped
        let detail: String
    }

    var body: some View {
        Form {
            Section {
                Button {
                    Task { await runAll() }
                } label: {
                    if running {
                        HStack { ProgressView(); Text("Running checks…") }
                    } else {
                        Label(results.isEmpty ? "Run all checks" : "Run again", systemImage: "stethoscope")
                    }
                }
                .disabled(running)
            } footer: {
                Text("Checks run top to bottom: account → cloud → device → live feed. The first ✗ is where your problem is.")
            }

            if !results.isEmpty {
                Section("Results") {
                    ForEach(results) { result in
                        HStack(alignment: .top, spacing: 10) {
                            Image(systemName: icon(for: result.passed))
                                .foregroundStyle(color(for: result.passed))
                            VStack(alignment: .leading, spacing: 2) {
                                Text(result.name).font(.subheadline.weight(.semibold))
                                Text(result.detail).font(.caption).foregroundStyle(SD.Colors.textSecondary)
                            }
                        }
                    }
                }
            }
        }
        .navigationTitle("Connection doctor")
    }

    private func icon(for passed: Bool?) -> String {
        switch passed {
        case .some(true): "checkmark.circle.fill"
        case .some(false): "xmark.circle.fill"
        case .none: "minus.circle"
        }
    }

    private func color(for passed: Bool?) -> Color {
        switch passed {
        case .some(true): .green
        case .some(false): .red
        case .none: .secondary
        }
    }

    // ------------------------------------------------------------------ checks
    private func runAll() async {
        running = true
        defer { running = false }
        results = []

        if state.isDemo {
            results.append(.init(name: "Demo mode", passed: false,
                                 detail: "You're in demo mode — everything is simulated. Sign out and log into a real account to test the real chain."))
        }

        // 1. cloud backend configured
        let base = state.baseURLString
        if base.isEmpty {
            results.append(.init(name: "Cloud backend configured", passed: false,
                                 detail: "No backend address set. Settings → Cloud backend."))
        } else if base.contains("localhost") || base.contains("127.0.0.1") {
            #if targetEnvironment(simulator)
            results.append(.init(name: "Cloud backend configured", passed: true, detail: base))
            #else
            results.append(.init(name: "Cloud backend configured", passed: false,
                                 detail: "\(base) — on a phone, localhost is the phone itself. Use the server's LAN address or public URL."))
            #endif
        } else {
            results.append(.init(name: "Cloud backend configured", passed: true, detail: base))
        }

        // 2. cloud reachable
        if let url = URL(string: base)?.appendingPathComponent("v1/health") {
            do {
                var req = URLRequest(url: url); req.timeoutInterval = 6
                let (_, resp) = try await URLSession.shared.data(for: req)
                let ok = (resp as? HTTPURLResponse)?.statusCode == 200
                results.append(.init(name: "Cloud backend reachable", passed: ok,
                                     detail: ok ? "server answered" : "server answered with an error — is this really the SensorDyme backend?"))
            } catch let error as URLError {
                results.append(.init(name: "Cloud backend reachable", passed: false,
                                     detail: APIClient.explain(error, host: URL(string: base)?.host ?? base)))
            } catch {
                results.append(.init(name: "Cloud backend reachable", passed: false, detail: error.localizedDescription))
            }
        } else {
            results.append(.init(name: "Cloud backend reachable", passed: nil, detail: "skipped — no valid backend address"))
        }

        // 3. account/auth
        if state.isSignedIn && !state.isDemo {
            do {
                _ = try await state.api.devices()
                results.append(.init(name: "Account & session", passed: true, detail: "signed in as \(state.accountEmail); token accepted"))
            } catch {
                results.append(.init(name: "Account & session", passed: false, detail: error.localizedDescription))
            }
        } else if !state.isDemo {
            results.append(.init(name: "Account & session", passed: nil, detail: "not signed in"))
        }

        // 4. device address configured
        let device = state.deviceLocalURL
        if device.isEmpty {
            results.append(.init(name: "Device address configured", passed: false,
                                 detail: "No device address. Settings → Live feed → Find my device (phone must be on the same Wi-Fi as the device)."))
        } else {
            results.append(.init(name: "Device address configured", passed: true, detail: device))
        }

        // 5. device reachable (its on-device health monitor answers)
        if !device.isEmpty {
            do {
                let health = try await DeviceHealth.fetch(baseURL: device, token: state.deviceToken)
                let summary = health.components
                    .filter { $0.value.state != "ok" }
                    .map { "\($0.key): \($0.value.state)" }
                    .joined(separator: ", ")
                results.append(.init(name: "Device reachable", passed: true,
                                     detail: "overall \(health.state)\(summary.isEmpty ? "" : " (\(summary))")"))
            } catch {
                results.append(.init(name: "Device reachable", passed: false,
                                     detail: "\(error.localizedDescription) Same Wi-Fi? Device powered?"))
            }
        } else {
            results.append(.init(name: "Device reachable", passed: nil, detail: "skipped"))
        }

        // 6. live stream produces bytes
        if !device.isEmpty, let url = URL(string: device)?.appendingPathComponent("stream") {
            var req = URLRequest(url: url); req.timeoutInterval = 6
            if !state.deviceToken.isEmpty { req.setValue(state.deviceToken, forHTTPHeaderField: "X-Device-Token") }
            do {
                let (bytes, resp) = try await URLSession.shared.bytes(for: req)
                let status = (resp as? HTTPURLResponse)?.statusCode ?? 0
                if status == 401 {
                    results.append(.init(name: "Live stream", passed: false,
                                         detail: "device rejected the token (401) — the device's dashboard.token and Settings → Device token must match"))
                } else if status == 200 {
                    var got = 0
                    for try await _ in bytes.prefix(2048) { got += 1 }  // proves frames flow
                    results.append(.init(name: "Live stream", passed: got > 0,
                                         detail: got > 0 ? "receiving video data" : "connected but no video bytes — is the camera running?"))
                } else {
                    results.append(.init(name: "Live stream", passed: false, detail: "unexpected response (\(status))"))
                }
            } catch {
                results.append(.init(name: "Live stream", passed: false, detail: error.localizedDescription))
            }
        } else {
            results.append(.init(name: "Live stream", passed: nil, detail: "skipped"))
        }
    }
}
