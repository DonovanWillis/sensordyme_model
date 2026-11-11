import CryptoKit
import Foundation

// Run with ios/Tests/run.sh
// run.sh deps: BLEProofOfPossession

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

/// A device that holds `code`, speaking protocol v1 over the auth characteristic.
final class FakeDevice {
    let deviceID: String
    let code: String
    let nonceDevice = Data((0..<16).map { UInt8($0 + 100) })
    var keys: BLEProofOfPossession.SessionKeys?
    var received: (ssid: String, psk: String?)?
    var lieAboutProof = false

    init(deviceID: String, code: String) {
        self.deviceID = deviceID
        self.code = code
    }

    func json(_ object: [String: Any]) -> Data {
        try! JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    }

    func reply(to data: Data) -> Data {
        let msg = try! JSONSerialization.jsonObject(with: data) as! [String: Any]
        switch msg["op"] as? String {
        case "hello":
            let na = Data(base64Encoded: msg["na"] as! String)!
            let codeKey = try! BLEProofOfPossession.codeKey(code: code, deviceID: deviceID)
            keys = BLEProofOfPossession.sessionKeys(codeKey: codeKey, deviceID: deviceID, nonceApp: na,
                                                   nonceDevice: nonceDevice)
            return json(["op": "challenge", "v": 1, "id": deviceID, "nd": nonceDevice.base64EncodedString()])
        case "proof":
            let tag = Data(base64Encoded: msg["tag"] as! String)!
            guard tag == BLEProofOfPossession.appProof(keys!) else {
                return json(["op": "error", "error": "wrong_code", "message": "no"])
            }
            var proof = BLEProofOfPossession.deviceProof(keys!)
            if lieAboutProof { proof[0] ^= 0xFF }
            return json(["op": "proven", "tag": proof.base64EncodedString()])
        case "creds":
            let iv = Data(base64Encoded: msg["iv"] as! String)!
            let ct = Data(base64Encoded: msg["ct"] as! String)!
            let box = try! AES.GCM.SealedBox(nonce: AES.GCM.Nonce(data: iv), ciphertext: ct.dropLast(16),
                                             tag: ct.suffix(16))
            let plain = try! AES.GCM.open(box, using: keys!.enc,
                                          authenticating: Data("wifi-credentials".utf8) + keys!.transcript)
            let creds = try! JSONSerialization.jsonObject(with: plain) as! [String: Any]
            received = (creds["ssid"] as! String, creds["psk"] as? String)
            return json(["op": "accepted"])
        default:
            return json(["op": "error", "error": "malformed"])
        }
    }
}

/// Runs the flow against the device until it ends; returns the last action.
func run(_ flow: BLEPoPFlow, _ device: FakeDevice) -> (BLEPoPFlow.Action, writes: Int) {
    var action = flow.start()
    var writes = 0
    while case .write(let data) = action {
        writes += 1
        action = flow.handle(reply: device.reply(to: data))
        if writes > 10 { break }
    }
    return (action, writes)
}

let id = "SD-0A1B2C3D"
let code = "ABCD-EFGH-JKMN"

print("setup code entry:")
func entry(_ text: String, _ connected: String?) -> Result<BLEProofOfPossession.SetupLabel, SetupCodeEntry.Problem> {
    SetupCodeEntry.label(from: text, connectedDeviceID: connected)
}
check(try! entry(code, id).get() == .init(deviceID: id, code: "ABCDEFGHJKMN"), "typed code + connected id")
check(try! entry("abcd efgh jkmn", id).get().code == "ABCDEFGHJKMN", "case and spaces normalised")
check(try! entry("SDSETUP:1:\(id):\(code)", id).get().deviceID == id, "QR text for this device")
check(entry("SDSETUP:1:SD-OTHER:\(code)", id) == .failure(.wrongDevice(label: "SD-OTHER", connected: id)),
      "a QR label for another device is refused")
check(entry("SDSETUP:1:\(id):\(code)", nil) == .success(.init(deviceID: id, code: "ABCDEFGHJKMN")),
      "a QR label works before the device has said its id")
check(entry(code, nil) == .failure(.deviceUnknown), "a typed code needs the device's id")
check(entry("", id) == .failure(.empty), "empty")
check(entry("ABCD", id) == .failure(.malformed), "too short")
check(entry("SDSETUP:2:\(id):\(code)", id) == .failure(.malformed), "unknown label version")
check(SetupCodeEntry.message(.wrongDevice(label: "A", connected: "B")).contains("label is for A"), "says which device")

print("\nthe full exchange with the right code:")
let device = FakeDevice(deviceID: id, code: code)
let label = try! entry(code, id).get()
let flow = BLEPoPFlow(session: try! BLEProofOfPossession.Session(label: label), ssid: "Home", psk: "s3cret")
let (end, writes) = run(flow, device)
check(end == .accepted, "accepted")
check(writes == 3, "hello, proof, creds: three writes")
check(device.received?.ssid == "Home" && device.received?.psk == "s3cret", "the device decrypted the credentials")
check(flow.step == .accepted, "flow ends accepted")

print("\nan open network sends psk null:")
let openDevice = FakeDevice(deviceID: id, code: code)
let openFlow = BLEPoPFlow(session: try! BLEProofOfPossession.Session(label: label), ssid: "Cafe", psk: nil)
check(run(openFlow, openDevice).0 == .accepted && openDevice.received?.psk == nil, "psk is null")

print("\na wrong code sends nothing secret:")
let wrong = FakeDevice(deviceID: id, code: "ZZZZ-ZZZZ-ZZZZ")
let wrongFlow = BLEPoPFlow(session: try! BLEProofOfPossession.Session(label: label), ssid: "Home", psk: "s3cret")
let (wrongEnd, wrongWrites) = run(wrongFlow, wrong)
check(wrongEnd == .failed("The setup code was wrong. Check the label and try again."), "the device's refusal in words")
check(wrongWrites == 2 && wrong.received == nil, "no credentials were written")

print("\na device that can't prove the code gets nothing:")
let impostor = FakeDevice(deviceID: id, code: code)
impostor.lieAboutProof = true
let impostorFlow = BLEPoPFlow(session: try! BLEProofOfPossession.Session(label: label), ssid: "Home", psk: "s3cret")
let (impostorEnd, impostorWrites) = run(impostorFlow, impostor)
if case .failed(let why) = impostorEnd { check(why.contains("couldn't prove"), "refused as not authentic") }
else { check(false, "an impostor must fail") }
check(impostorWrites == 2 && impostor.received == nil, "credentials never left the phone")

print("\na challenge from another device id is refused:")
let other = FakeDevice(deviceID: "SD-FFFFFFFF", code: code)
let otherFlow = BLEPoPFlow(session: try! BLEProofOfPossession.Session(label: label), ssid: "Home", psk: "x")
if case .failed(let why) = run(otherFlow, other).0 { check(why.contains("SD-FFFFFFFF"), "names the other device") }
else { check(false, "another device must fail") }

print("\ndevice errors and out-of-order replies:")
let locked = BLEPoPFlow(session: try! BLEProofOfPossession.Session(label: label), ssid: "Home", psk: "x")
_ = locked.start()
let lockedReply = Data(#"{"op":"error","error":"locked","retry_after_s":31}"#.utf8)
check(locked.handle(reply: lockedReply) == .failed("Too many wrong codes, so the device is refusing for now. Try again in 31 s."),
      "lockout with the device's wait")
let early = BLEPoPFlow(session: try! BLEProofOfPossession.Session(label: label), ssid: "Home", psk: "x")
check(early.handle(reply: Data("{}".utf8)) == .failed("unexpected reply from the device"), "a reply before start")
check(BLEPoPFlow.deviceError("busy", retryAfter: nil).contains("busy"), "busy")
check(BLEPoPFlow.deviceError(nil, retryAfter: nil) == "The device refused the setup.", "no code")

print("\n\(failures == 0 ? "all passed" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)
