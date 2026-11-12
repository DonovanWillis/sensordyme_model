import Foundation

// The setup screen's side of Bluetooth proof of possession
// (docs/BLE_PROOF_OF_POSSESSION.md): read the setup code from the label, then
// drive BLEProofOfPossession.Session over the auth characteristic (5d1e0007):
//
//   hello -> challenge -> proof -> proven (the device proves the code back)
//         -> encrypted creds -> accepted; the join outcome then follows on the
//            status characteristic exactly as in the legacy flow.
//
// No CoreBluetooth here: BLEProvisioner writes what this returns and feeds it
// what the device notifies, so ios/Tests/BLEPoPFlowTests.swift can run the
// whole exchange against a simulated device, and
// tests/test_m8_onboarding_crosslang.py runs it against the device's own
// Python state machine (sim/ble_peripheral.py). NOT validated over a physical
// Bluetooth link.

/// What the person typed or pasted: the printed code ("XXXX-XXXX-XXXX") or the
/// QR text ("SDSETUP:1:<device id>:<code>").
enum SetupCodeEntry {
    enum Problem: Error, Equatable {
        case empty
        case malformed
        case wrongDevice(label: String, connected: String)
        case deviceUnknown
    }

    /// The label for the device this phone is connected to. A QR text naming
    /// another device is refused: being discoverable proves nothing, and the
    /// label is what says which device is yours.
    static func label(from input: String, connectedDeviceID: String?) -> Result<BLEProofOfPossession.SetupLabel, Problem> {
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return .failure(.empty) }
        if text.uppercased().hasPrefix("SDSETUP:") {
            guard let label = try? BLEProofOfPossession.parseLabel(text) else { return .failure(.malformed) }
            if let connectedDeviceID, label.deviceID != connectedDeviceID {
                return .failure(.wrongDevice(label: label.deviceID, connected: connectedDeviceID))
            }
            return .success(label)
        }
        guard let code = try? BLEProofOfPossession.normalize(code: text) else { return .failure(.malformed) }
        // A typed code carries no device id: the connected device's own id is
        // the salt, so a code for another device simply fails the proof.
        guard let connectedDeviceID else { return .failure(.deviceUnknown) }
        return .success(.init(deviceID: connectedDeviceID, code: code))
    }

    static func message(_ problem: Problem) -> String {
        switch problem {
        case .empty: "Enter the setup code from the label on your SensorDyme."
        case .malformed: "That isn't a setup code. It has 12 letters and digits, like ABCD-EFGH-JKMN."
        case .wrongDevice(let label, let connected):
            "This label is for \(label), but the phone is connected to \(connected). Use the label on this device."
        case .deviceUnknown: "The device hasn't said who it is yet. Wait a moment and try again."
        }
    }
}

/// One proof-of-possession exchange. Every reply moves it one step or fails it;
/// credentials are produced only after the device has proven the code back.
///
/// Two ways to drive it:
///   * prove first (`init(session:)`): hello -> proof -> `.proven`; the
///     credentials follow later with `sendCredentials`, within the device's
///     proof window (120 s). Onboarding uses this, so a wrong code or a
///     lockout is reported before anyone types a Wi-Fi password.
///   * one shot (`init(session:ssid:psk:)`): the credentials go out the
///     moment the device has proven itself.
final class BLEPoPFlow {
    enum Step: Equatable {
        case ready, awaitingChallenge, awaitingProven, proven, awaitingAccepted, accepted
        case failed(String)
    }

    enum Action: Equatable {
        /// Write this to the auth characteristic.
        case write(Data)
        /// The device holds the code and proved it back: credentials may be sent.
        case proven
        /// Credentials accepted: follow the status characteristic as before.
        case accepted
        case failed(String)
    }

    /// Why the exchange stopped, in terms the caller can act on.
    struct Rejection: Equatable {
        enum Kind: Equatable {
            /// The device's answers (services/ble_pop.py `_error`).
            case wrongCode, locked, busy, expired, unsupportedVersion, refusedMessage, other
            /// Found on the phone: the device is not the one on the label,
            /// cannot prove the code, or speaks something this app does not.
            case wrongDevice, notAuthentic, malformedReply
        }

        let kind: Kind
        let message: String
        /// The device's own wait, when it gave one (`retry_after_s`).
        let retryAfterSeconds: Int?

        /// Whether the link should be dropped: the device on the other end is
        /// not one this phone should say anything more to.
        var endsLink: Bool {
            switch kind {
            case .wrongDevice, .notAuthentic, .malformedReply: true
            default: false
            }
        }
    }

    private(set) var step: Step = .ready
    private(set) var rejection: Rejection?
    let session: BLEProofOfPossession.Session
    private var pending: (ssid: String, psk: String?)?

    /// Prove first; credentials later with `sendCredentials`.
    init(session: BLEProofOfPossession.Session) {
        self.session = session
    }

    /// One shot: credentials go out as soon as the device has proven itself.
    /// - Parameter psk: nil for an open network.
    init(session: BLEProofOfPossession.Session, ssid: String, psk: String?) {
        self.session = session
        self.pending = (ssid, psk)
    }

    func start() -> Action {
        guard step == .ready else { return fail(.init(kind: .other, message: "the exchange already started", retryAfterSeconds: nil)) }
        step = .awaitingChallenge
        return .write(session.helloMessage())
    }

    /// The credentials, once the device has proven itself.
    func sendCredentials(ssid: String, psk: String?) -> Action {
        guard step == .proven else {
            return fail(.init(kind: .other, message: "The device hasn't confirmed the setup code yet.", retryAfterSeconds: nil))
        }
        do {
            let creds = try session.credentialsMessage(ssid: ssid, psk: psk)
            step = .awaitingAccepted
            return .write(creds)
        } catch let failure as BLEProofOfPossession.Failure {
            return fail(Self.rejection(failure))
        } catch {
            return fail(.init(kind: .other, message: "the exchange failed: \(error.localizedDescription)", retryAfterSeconds: nil))
        }
    }

    func handle(reply: Data) -> Action {
        do {
            switch step {
            case .awaitingChallenge:
                let proof = try session.proofMessage(challenge: reply)
                step = .awaitingProven
                return .write(proof)
            case .awaitingProven:
                try session.verify(proven: reply)
                step = .proven
                if let pending {
                    self.pending = nil
                    return sendCredentials(ssid: pending.ssid, psk: pending.psk)
                }
                return .proven
            case .awaitingAccepted:
                let msg = (try? JSONSerialization.jsonObject(with: reply)) as? [String: Any]
                if msg?["op"] as? String == "accepted" {
                    step = .accepted
                    return .accepted
                }
                if msg?["op"] as? String == "error" {
                    return fail(Self.rejection(code: msg?["error"] as? String, retryAfter: msg?["retry_after_s"] as? Int))
                }
                return fail(.init(kind: .malformedReply, message: "the device sent an unexpected reply", retryAfterSeconds: nil))
            case .ready, .proven, .accepted, .failed:
                return fail(.init(kind: .other, message: "unexpected reply from the device", retryAfterSeconds: nil))
            }
        } catch let failure as BLEProofOfPossession.Failure {
            return fail(Self.rejection(failure))
        } catch {
            return fail(.init(kind: .other, message: "the exchange failed: \(error.localizedDescription)", retryAfterSeconds: nil))
        }
    }

    private func fail(_ rejection: Rejection) -> Action {
        self.rejection = rejection
        step = .failed(rejection.message)
        return .failed(rejection.message)
    }

    static func rejection(_ failure: BLEProofOfPossession.Failure) -> Rejection {
        switch failure {
        case .rejected(let code, let retryAfter):
            return rejection(code: code, retryAfter: retryAfter)
        case .wrongDevice:
            return .init(kind: .wrongDevice, message: message(failure), retryAfterSeconds: nil)
        case .deviceNotAuthentic, .notProven:
            return .init(kind: .notAuthentic, message: message(failure), retryAfterSeconds: nil)
        case .malformedReply:
            return .init(kind: .malformedReply, message: message(failure), retryAfterSeconds: nil)
        case .invalidCode, .invalidLabel, .keyDerivationFailed:
            return .init(kind: .other, message: message(failure), retryAfterSeconds: nil)
        }
    }

    static func rejection(code: String?, retryAfter: Int?) -> Rejection {
        let kind: Rejection.Kind = switch code {
        case "wrong_code": .wrongCode
        case "locked": .locked
        case "busy": .busy
        case "unexpected": .expired
        case "unsupported_version": .unsupportedVersion
        case "malformed": .refusedMessage
        default: .other
        }
        return .init(kind: kind, message: deviceError(code, retryAfter: retryAfter), retryAfterSeconds: retryAfter)
    }

    /// A failure in the person's terms, with what to do.
    static func message(_ failure: BLEProofOfPossession.Failure) -> String {
        switch failure {
        case .invalidCode, .invalidLabel:
            return "That setup code isn't valid. Check the label and try again."
        case .wrongDevice(let expected, let got):
            return "This is \(got), not \(expected). Connect to the device the label belongs to."
        case .deviceNotAuthentic:
            return "The device couldn't prove it holds this setup code, so nothing was sent. Make sure this is your SensorDyme."
        case .notProven:
            return "The device didn't confirm the setup code, so nothing was sent."
        case .rejected(let code, let retryAfter):
            return deviceError(code, retryAfter: retryAfter)
        case .malformedReply:
            return "The device answered in a way this app doesn't understand. It may need a software update."
        case .keyDerivationFailed:
            return "This phone couldn't prepare the secure setup. Try again."
        }
    }

    static func deviceError(_ code: String?, retryAfter: Int?) -> String {
        // The device's codes (services/ble_pop.py `_error`).
        switch code {
        case "wrong_code":
            // The device attaches its wait to the wrong code that starts a lockout.
            let lock = retryAfter.map { " That was too many wrong codes: the device will refuse for \($0) s." } ?? ""
            return "The setup code was wrong. Check the label and try again." + lock
        case "locked":
            let wait = retryAfter.map { " Try again in \($0) s." } ?? " Wait a while and try again."
            return "Too many wrong codes, so the device is refusing for now." + wait
        case "busy":
            return "The device is busy with another setup. Wait a moment and try again."
        case "unexpected":
            return "The device lost track of this setup (it may have timed out). Start again."
        case "unsupported_version":
            return "The device uses a different setup version. Update the app or the device."
        case "malformed":
            return "The device couldn't read what this phone sent, so nothing was applied. Try again."
        default:
            return "The device refused the setup" + (code.map { " (\($0))" } ?? "") + "."
        }
    }
}
