import Foundation

/// Launch arguments for driving the app to a screen without touching it — used
/// to screenshot and check every ORYN EDGE screen in the simulator, e.g.
///
///     xcrun simctl launch booted com.sensordyme.app -orynDemo -orynOpenDevice
///
/// DEBUG builds only. In a Release build every flag is false and nothing here
/// can change what the app does.
enum DebugLaunch {
    #if DEBUG
    private static let args = ProcessInfo.processInfo.arguments

    /// Enter demo mode on launch (simulated device, no account needed).
    static var demo: Bool { args.contains("-orynDemo") }
    /// Push the first device's screen once the list loads.
    static var openDevice: Bool { args.contains("-orynOpenDevice") }
    /// Open Ask Jupiter on the device screen.
    static var openJupiter: Bool { args.contains("-orynOpenJupiter") }
    /// Open the device's event history.
    static var openHistory: Bool { args.contains("-orynOpenHistory") }
    /// Scroll the device screen to its last section.
    static var scrollToBottom: Bool { args.contains("-orynScrollBottom") }
    /// Select a tab: devices, shop or settings.
    static var tab: String? { value(after: "-orynTab") }
    /// Text to pre-fill (and send) in Ask Jupiter.
    static var jupiterPrompt: String? { value(after: "-orynPrompt") }
    /// Don't ask for notification permission (the prompt would cover the screen).
    static var noPermissionPrompt: Bool { args.contains("-orynNoPrompt") }
    /// Sign in with an existing backend session (`-orynSession <token> <email>`), so the
    /// app can be driven against a local backend + virtual device (docs/VIRTUAL_DEVICE.md)
    /// without typing. The token comes from a real /v1/login; nothing is bypassed.
    static var session: (token: String, email: String)? {
        guard let index = args.firstIndex(of: "-orynSession"), args.indices.contains(index + 2) else { return nil }
        return (args[index + 1], args[index + 2])
    }

    private static func value(after flag: String) -> String? {
        guard let index = args.firstIndex(of: flag), args.indices.contains(index + 1) else { return nil }
        return args[index + 1]
    }
    #else
    static let demo = false
    static let openDevice = false
    static let openJupiter = false
    static let openHistory = false
    static let scrollToBottom = false
    static let tab: String? = nil
    static let jupiterPrompt: String? = nil
    static let session: (token: String, email: String)? = nil
    static let noPermissionPrompt = false
    #endif
}
