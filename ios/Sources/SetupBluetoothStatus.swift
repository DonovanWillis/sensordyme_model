import Foundation

/// Why Bluetooth setup cannot start, said as the one thing that is actually
/// wrong, with the fix that matches it.
///
/// The onboarding state machine folds every radio problem into
/// `.bluetoothUnavailable` (and an unsupported radio into nothing at all while
/// it is still looking), which is right for its job: nothing can happen until
/// Bluetooth works. The person, though, needs to know WHICH problem it is,
/// because the fixes differ: permission is in Settings → ORYN EDGE, power is in
/// Control Centre, a restriction is whoever manages the phone, and an iPhone
/// without Bluetooth LE cannot do this at all. This reads what the radio
/// reported (CBManagerState) and the app's Bluetooth permission
/// (CBManagerAuthorization); it never changes the flow.
///
/// Foundation-only and unit-tested (ios/Tests/SetupBluetoothStatusTests.swift).
enum SetupBluetoothStatus {
    /// CBManagerState, as far as setup cares.
    enum Radio: Equatable { case on, off, unauthorized, unsupported, unknown }
    /// CBManagerAuthorization.
    enum Access: Equatable { case notDetermined, denied, restricted, allowed }

    enum Problem: Equatable {
        /// The person said no to Bluetooth for this app (or turned it off for it).
        case permissionDenied
        /// Screen Time or device management blocks Bluetooth for apps.
        case restricted
        /// Bluetooth is switched off.
        case poweredOff
        /// This device has no Bluetooth LE (the iOS Simulator, for one).
        case unsupported

        var title: String {
            switch self {
            case .permissionDenied: "Allow Bluetooth for ORYN EDGE"
            case .restricted: "Bluetooth is restricted on this iPhone"
            case .poweredOff: "Bluetooth is off"
            case .unsupported: "This device can't use Bluetooth"
            }
        }

        var detail: String {
            switch self {
            case .permissionDenied:
                "Setup reaches your SensorDyme over Bluetooth, and ORYN EDGE isn't allowed to use it. "
                    + "Turn on Bluetooth for ORYN EDGE in Settings, then come back: setup carries on by itself."
            case .restricted:
                "A Screen Time or device-management setting stops apps from using Bluetooth. "
                    + "Whoever manages this iPhone can allow it. Until then, use another iPhone to set up the device."
            case .poweredOff:
                "Turn Bluetooth on in Control Centre or in Settings → Bluetooth. "
                    + "Setup carries on by itself as soon as it is on."
            case .unsupported:
                "Setting up Wi-Fi needs Bluetooth Low Energy, which this device doesn't have. "
                    + "Use an iPhone with Bluetooth, or the device's setup network (Add device)."
            }
        }

        var symbol: String {
            switch self {
            case .permissionDenied: "hand.raised"
            case .restricted: "lock"
            case .poweredOff: "power"
            case .unsupported: "nosign"
            }
        }

        /// The app's own page in Settings is the fix only for the permission.
        /// iOS offers apps no link to Bluetooth itself, so no button pretends to.
        var opensAppSettings: Bool { self == .permissionDenied }
    }

    /// The problem, or nil when the radio is usable (or has not answered yet).
    static func problem(radio: Radio?, access: Access) -> Problem? {
        switch radio {
        case .unsupported?: return .unsupported
        case .unauthorized?: return access == .restricted ? .restricted : .permissionDenied
        case .off?: return .poweredOff
        case .on?: return nil
        case .unknown?, nil:
            // Before the radio answers, a refusal already on record is still the answer.
            switch access {
            case .denied: return .permissionDenied
            case .restricted: return .restricted
            case .notDetermined, .allowed: return nil
            }
        }
    }
}
