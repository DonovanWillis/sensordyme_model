import Foundation

/// Which device's live feed this phone may show.
///
/// The phone links ONE camera on the LAN (its Bonjour `SD-XXXXXXXX` id and
/// address). Nothing used to connect that to a backend device, so every device
/// on an account streamed the one linked camera — two devices, one feed, the
/// wrong one on half the screens. The backend now reports each device's
/// hardware id (the same SD- id, from its health report), and this is the rule.
///
/// Foundation-only, so ios/Tests/FeedLinkTests.swift can run it outside Xcode.
enum FeedLink {
    /// - Parameters:
    ///   - deviceHardwareID: the backend device's reported SD- id; nil from
    ///     firmware or a backend that predates it.
    ///   - linkedID: the camera this phone linked (AppStorage "deviceServiceName"),
    ///     an SD- id, or a legacy Bonjour service name from older app versions.
    ///   - accountDeviceCount: devices on the signed-in account.
    static func showsLiveFeed(deviceHardwareID: String?, linkedID: String, accountDeviceCount: Int) -> Bool {
        let hardware = (deviceHardwareID ?? "").trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
        let linked = linkedID.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
        if hardware.hasPrefix("SD-"), linked.hasPrefix("SD-") {
            return hardware == linked
        }
        // One side unknown: only an account with exactly one device is
        // unambiguous. That keeps today's behaviour for a single camera while the
        // firmware catches up, and refuses to guess between several.
        return accountDeviceCount == 1
    }
}
