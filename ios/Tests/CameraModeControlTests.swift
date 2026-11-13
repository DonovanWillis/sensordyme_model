import Foundation

// Standalone, run by ios/Tests/run.sh (concatenated with ios/Sources/CameraModeControl.swift).
// run.sh deps: StreamLiveness MJPEGStream
//
// M8.4 (owner Decision C): the "Camera only while stopped" switch shows what
// the DEVICE reports, never the request; turning it on is confirmed first;
// refusals say what happened.

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

func mode(_ json: String) -> DeviceCameraMode? {
    try? JSONDecoder().decode(DeviceCameraMode.self, from: Data(json.utf8))
}
let off = CameraModePresentation.resolve(mode(#"{"supported": true, "camera_only": false}"#))
let on = CameraModePresentation.resolve(mode(#"{"supported": true, "camera_only": true}"#))
let turningOn = CameraModePresentation.resolve(
    mode(#"{"supported": true, "camera_only": false, "desired_camera_only": true, "pending": true}"#))
let turningOff = CameraModePresentation.resolve(
    mode(#"{"supported": true, "camera_only": true, "desired_camera_only": false, "pending": true}"#))
let unsupported = CameraModePresentation.resolve(mode(#"{"supported": false}"#))
let unknown = CameraModePresentation.resolve(nil)

print("the switch is what the device reports:")
check(!CameraModeControl.switchOn(off) && CameraModeControl.switchOn(on), "off and on, as reported")
check(!CameraModeControl.switchOn(turningOn), "never on before the device reports it")
check(CameraModeControl.switchOn(turningOff), "still on until the device reports it off")
check(!CameraModeControl.switchEnabled(turningOn, sending: false), "held while a request is delivered")
check(!CameraModeControl.switchEnabled(off, sending: true), "held while the request is in flight")
check(!CameraModeControl.switchEnabled(unsupported, sending: false), "no switch for older device software")
check(!CameraModeControl.switchEnabled(unknown, sending: false), "no switch when nothing is reported")

print("\na tap:")
check(CameraModeControl.toggled(to: true, presentation: off, sending: false) == .askToConfirm,
      "turning on asks first; nothing is sent yet")
check(CameraModeControl.confirmedOn == .send(cameraOnly: true, confirm: true), "confirmed: sent with confirm")
check(CameraModeControl.toggled(to: false, presentation: on, sending: false)
      == .send(cameraOnly: false, confirm: false), "turning off needs no confirmation")
check(CameraModeControl.toggled(to: true, presentation: on, sending: false) == .none, "no change, no request")
check(CameraModeControl.toggled(to: true, presentation: unsupported, sending: false) == .none,
      "nothing for unsupported software")
check(CameraModeControl.confirmMessage.contains("live video") && CameraModeControl.confirmMessage.contains("Nothing is monitored"),
      "the confirmation says the owner will see live video while nothing is monitored")

print("\nrefusals:")
let firmware = CameraModeControl.refusal(status: 409, code: "firmware_unsupported", message: "update it first")
check(firmware.disablesSwitch && firmware.message.contains("Update it"), "old software: update it, no switch")
let confirm = CameraModeControl.refusal(status: 400, code: "confirmation_required", message: "")
check(!confirm.disablesSwitch && confirm.message.contains("Nothing was changed"), "confirmation required")
check(CameraModeControl.refusal(status: 404, code: nil, message: "Not Found").disablesSwitch,
      "an older server without the endpoint: no switch")
check(CameraModeControl.refusal(status: 0, code: nil, message: "offline").message.contains("nothing was changed"),
      "offline: nothing was changed")
check(CameraModeControl.sentNote(cameraOnly: true).contains("until the device confirms"),
      "after sending ON, the camera is said to stay off until the device confirms")

print("\n\(failures == 0 ? "PASSED" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)
