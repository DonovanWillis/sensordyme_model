import Foundation

/// The "Camera only while stopped" control (M8.4, owner Decision C), as pure
/// decisions: what a tap on the switch does, what the switch shows, and what
/// a refusal says. Foundation-only (ios/Tests/CameraModeControlTests.swift).
///
/// The switch shows what the DEVICE reports (`status.camera_mode.camera_only`
/// through `CameraModePresentation`), never the request: it reads ON only once
/// the device has reported camera-only on. While a request is being delivered
/// the switch is held where the device is and the line under it says so.
enum CameraModeControl {
    /// What a tap on the switch leads to.
    enum Action: Equatable {
        /// Nothing (the device can't change it, or it already is that way).
        case none
        /// Turning ON: ask the owner first; nothing is sent until they confirm.
        case askToConfirm
        /// Send `POST /v1/devices/{id}/camera-mode {camera_only, confirm}`.
        case send(cameraOnly: Bool, confirm: Bool)
    }

    /// The switch's position: the device's own report, nothing else.
    static func switchOn(_ presentation: CameraModePresentation) -> Bool {
        presentation.kind == .cameraOnly
    }

    /// Whether the switch can be touched: the device supports the setting and
    /// no request is on its way (a second tap would only race the first).
    static func switchEnabled(_ presentation: CameraModePresentation, sending: Bool) -> Bool {
        presentation.canChange && !presentation.pending && !sending
    }

    /// What flipping the switch to `on` does.
    static func toggled(to on: Bool, presentation: CameraModePresentation, sending: Bool) -> Action {
        guard switchEnabled(presentation, sending: sending), on != switchOn(presentation) else { return .none }
        return on ? .askToConfirm : .send(cameraOnly: false, confirm: false)
    }

    /// After the owner confirms turning it on.
    static let confirmedOn = Action.send(cameraOnly: true, confirm: true)

    /// The confirmation, in plain words, before anything is sent.
    static let confirmTitle = "Allow the camera while monitoring is stopped?"
    static let confirmMessage =
        "While monitoring is stopped, you'll be able to watch live video from this SensorDyme in the app. "
        + "Nothing is monitored and no alerts are raised during that time. The device switches it on only "
        + "after it verifies the signed request, and you can turn it off here at any time."
    static let confirmButton = "Allow camera while stopped"

    /// A refusal, in the owner's terms. `disablesSwitch`: the request can never
    /// succeed as things are (the device's software is too old), so the switch
    /// should not be offered.
    struct Refusal: Equatable {
        let message: String
        let disablesSwitch: Bool
    }

    static func refusal(status: Int, code: String?, message: String) -> Refusal {
        switch code {
        case "firmware_unsupported":
            return Refusal(message: "This SensorDyme's software can't do this yet. Update it, then try again.",
                           disablesSwitch: true)
        case "confirmation_required":
            return Refusal(message: "Turning the camera on needs your confirmation. Nothing was changed.",
                           disablesSwitch: false)
        default:
            break
        }
        if status == 404 && code == nil {
            return Refusal(message: "This server doesn't support camera settings yet. Nothing was changed.",
                           disablesSwitch: true)
        }
        if status == 0 {
            return Refusal(message: "Couldn't reach the server, so nothing was changed. \(message)",
                           disablesSwitch: false)
        }
        return Refusal(message: "Nothing was changed: \(message)", disablesSwitch: false)
    }

    /// The line under the switch right after the server accepted a request,
    /// until the device's own report catches up.
    static func sentNote(cameraOnly: Bool) -> String {
        cameraOnly ? "Sent. The camera stays off until the device confirms it."
            : "Sent. Waiting for the device to confirm the camera is off while stopped."
    }
}
