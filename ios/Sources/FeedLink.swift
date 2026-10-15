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
        let hardware = normalised(deviceHardwareID)
        let linked = normalised(linkedID)
        if hardware.hasPrefix("SD-"), linked.hasPrefix("SD-") {
            return hardware == linked
        }
        // One side unknown: only an account with exactly one device is
        // unambiguous. That keeps today's behaviour for a single camera while the
        // firmware catches up, and refuses to guess between several.
        return accountDeviceCount == 1
    }

    // MARK: - Which camera on the network to link

    /// The camera to adopt out of the ones Bonjour found, or nil to adopt none.
    ///
    /// The old rule took `found.first` whenever nothing was linked yet, so on a
    /// network with two SensorDymes the phone linked whichever answered first —
    /// often a neighbour's — and then kept it for ever, because a linked id is
    /// matched before anything else. Setup was worse: after provisioning camera B
    /// it went looking for whatever was linked before (camera A) and reported B
    /// ready at A's address.
    ///
    /// - Parameters:
    ///   - found: SD- ids (or legacy service names) Bonjour returned.
    ///   - linked: the id already linked, or "".
    ///   - accountHardwareIDs: SD- ids the backend reports for this account's
    ///     devices; empty when unknown.
    ///   - expecting: the id of the device that was just provisioned, when the
    ///     caller knows exactly which camera it is looking for.
    static func chooseCamera(
        found: [String],
        linked: String,
        accountHardwareIDs: [String] = [],
        expecting: String? = nil
    ) -> String? {
        // Looking for one specific device: that one, or nothing.
        if let expecting = expecting.map(normalised), !expecting.isEmpty {
            return found.first { normalised($0) == expecting }
        }
        let linkedID = normalised(linked)
        if !linkedID.isEmpty {
            if let mine = found.first(where: { normalised($0) == linkedID }) { return mine }
            // A legacy service name matches nothing any more; adopt only when
            // exactly one device answers, so this cannot rebind to a neighbour.
            if !linkedID.hasPrefix("SD-"), found.count == 1 { return found[0] }
            return nil
        }
        // Nothing linked yet. A camera the backend says is on this account wins.
        let account = Set(accountHardwareIDs.map(normalised).filter { $0.hasPrefix("SD-") })
        if !account.isEmpty {
            let ours = found.filter { account.contains(normalised($0)) }
            return ours.count == 1 ? ours[0] : nil
        }
        // No account ids to go on: only a lone camera is unambiguous.
        return found.count == 1 ? found[0] : nil
    }

    /// Whether a cached address that just answered is still the linked camera.
    ///
    /// An address is a location, not an identity: after a DHCP swap, or avahi
    /// handing `sensordyme.local` to whichever of two default-named Pis booted
    /// first, the cached address answers as a DIFFERENT SensorDyme — and with a
    /// shared token it streams, under the wrong device's card. When the device
    /// identifies itself (authorised `/api/health`), that identity must match.
    static func cachedAddressIsLinkedCamera(reportedID: String?, linkedID: String) -> Bool {
        let reported = normalised(reportedID)
        let linked = normalised(linkedID)
        guard reported.hasPrefix("SD-"), linked.hasPrefix("SD-") else { return true }
        return reported == linked
    }

    /// Index of the backend device the linked camera belongs to, for anything
    /// that has to describe "your device" — or nil when that is ambiguous.
    static func linkedDeviceIndex(hardwareIDs: [String?], linkedID: String) -> Int? {
        let linked = normalised(linkedID)
        if linked.hasPrefix("SD-"),
           let index = hardwareIDs.firstIndex(where: { normalised($0) == linked }) {
            return index
        }
        return hardwareIDs.count == 1 ? 0 : nil
    }

    private static func normalised(_ value: String?) -> String {
        (value ?? "").trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
    }
}
