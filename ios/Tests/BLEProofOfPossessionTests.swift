import CryptoKit
import Foundation

// Standalone, runnable with ios/Tests/run.sh (concatenated after
// ios/Sources/BLEProofOfPossession.swift). The vectors below are
// tests/ble_pop_vectors.json verbatim, produced by the device's Python
// implementation (scripts/ble_setup_code.py vectors);
// tests/test_ble_pop.py::test_the_swift_tests_embed_the_same_vectors keeps the
// two copies identical, so both languages are checked against the same bytes.

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

func hex(_ data: Data) -> String { data.map { String(format: "%02x", $0) }.joined() }
func hex(_ key: SymmetricKey) -> String { key.withUnsafeBytes { hex(Data($0)) } }
func unhex(_ text: String) -> Data {
    var out = Data()
    var index = text.startIndex
    while index < text.endIndex {
        let next = text.index(index, offsetBy: 2)
        out.append(UInt8(text[index..<next], radix: 16)!)
        index = next
    }
    return out
}
func object(_ data: Data) -> [String: Any] { (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:] }
func reply(_ object: [String: Any]) -> Data { try! JSONSerialization.data(withJSONObject: object) }
func failure(_ body: () throws -> Void) -> BLEProofOfPossession.Failure? {
    do { try body(); return nil } catch let error as BLEProofOfPossession.Failure { return error } catch { return nil }
}

let vectorsJSON = #"""
{
 "protocol": 1,
 "pbkdf2_iterations": 600000,
 "cases": [
  {
   "name": "basic",
   "device_id": "SD-1A2B3C4D",
   "code": "ABCD-EFGH-JKMN",
   "na": "00112233445566778899aabbccddeeff",
   "nd": "ffeeddccbbaa99887766554433221100",
   "iv": "000102030405060708090a0b",
   "ssid": "HomeWiFi",
   "psk": "correct horse battery staple",
   "qr": "SDSETUP:1:SD-1A2B3C4D:ABCDEFGHJKMN",
   "code_key": "7305a2ddb6ed90e98be12005678dfe10dbb2b05af777d9295aa03c0dabca23f7",
   "k_mac": "32ecff9fa9157d8d672f6ff2fa72025240f22e970f634b4b78a69a4cfec0cc12",
   "k_enc": "ea609dbd840da06e6372c322c2575ce0652326274e1d832eb63a86eb77592346",
   "transcript": "53656e736f7244796d6520424c4520506f502076317c0b53442d314132423343344400112233445566778899aabbccddeeffffeeddccbbaa99887766554433221100",
   "app_proof": "029176b1d6b2fb003cc529882cd6e158c418194b861dfcfbe7fa0013750b995d",
   "device_proof": "360aa40f9556222b31e2f5635624f0108756601e3c46e19f2f2fb4d0e7ac8bb3",
   "creds_ct": "560d73e28aa9330bc4ee9457c6fd4c19bcb7164873f66407d4ef0dfbf2c60264d2a039b090f57c576e008ab6ae8d6d63d635116477f5a235b5f1af8c903918d209d45756f6511247"
  },
  {
   "name": "unicode_open_network",
   "device_id": "SD-00000000",
   "code": "0123-4567-89VW",
   "na": "0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f",
   "nd": "f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0",
   "iv": "a1a1a1a1a1a1a1a1a1a1a1a1",
   "ssid": "Café ☕ 5G",
   "psk": null,
   "qr": "SDSETUP:1:SD-00000000:0123456789VW",
   "code_key": "dd28e97006389ce537bc4698bc328dfa6190d1cf500f0d1d3d683b4854a8d1da",
   "k_mac": "b4ad57d14fa8a52d2d127d5ff6129b31e6c599ba401498198308ea86e0fde7b8",
   "k_enc": "5744c342ebd25784d5cfb80a73f2804a4d07fdd1ae16930da945359a85adcb74",
   "transcript": "53656e736f7244796d6520424c4520506f502076317c0b53442d30303030303030300f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0ff0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0",
   "app_proof": "ac8413bbed7eba5dca9cb33f0c417d29e60ce0af9a9d30c451922c6f70bb43ad",
   "device_proof": "3dea30c5b1f2c7ad2025ced19b5fd521aaf89400d1eed7845fc9cbeb69b9c519",
   "creds_ct": "f9de8388f7d09588334bc5dc2f5e8f599ec9e6ae48c6a8fd71ea1380dacddcfc29bf8ef6bfa494c54ada82bc02100ff0eb5d"
  }
 ]
}
"""#

typealias PoP = BLEProofOfPossession
let vectors = object(Data(vectorsJSON.utf8))
check(vectors["protocol"] as? Int == PoP.protocolVersion, "protocol version matches the device")
check((vectors["pbkdf2_iterations"] as? Int).map(UInt32.init) == PoP.pbkdf2Iterations, "PBKDF2 cost matches the device")

for vector in vectors["cases"] as! [[String: Any]] {
    let name = vector["name"] as! String
    let id = vector["device_id"] as! String
    let na = unhex(vector["na"] as! String), nd = unhex(vector["nd"] as! String), iv = unhex(vector["iv"] as! String)
    print("vector \(name):")
    let label = try! PoP.parseLabel(vector["qr"] as! String)
    check(label == PoP.SetupLabel(deviceID: id, code: try! PoP.normalize(code: vector["code"] as! String)),
          "the QR payload parses to the label's id and code")
    let codeKey = try! PoP.codeKey(code: label.code, deviceID: id)
    check(hex(codeKey) == vector["code_key"] as! String, "PBKDF2 code key")
    let keys = PoP.sessionKeys(codeKey: codeKey, deviceID: id, nonceApp: na, nonceDevice: nd)
    check(hex(keys.mac) == vector["k_mac"] as! String && hex(keys.enc) == vector["k_enc"] as! String, "HKDF session keys")
    check(hex(keys.transcript) == vector["transcript"] as! String, "transcript")
    check(hex(PoP.appProof(keys)) == vector["app_proof"] as! String, "app proof")
    check(PoP.isAuthentic(keys, deviceTag: unhex(vector["device_proof"] as! String)), "device proof verifies")
    let psk = vector["psk"] as? String
    let sealed = try! PoP.sealCredentials(keys, ssid: vector["ssid"] as! String, psk: psk, iv: iv)
    check(hex(sealed) == vector["creds_ct"] as! String, "AES-GCM credentials are byte-identical to the device's")

    // The same exchange through the Session API, against the device's recorded answers.
    let session = try! PoP.Session(label: label, nonceApp: na)
    check(object(session.helloMessage())["na"] as? String == na.base64EncodedString(), "hello carries the app nonce")
    let challenge = reply(["op": "challenge", "v": 1, "id": id, "nd": nd.base64EncodedString()])
    let proof = try! session.proofMessage(challenge: challenge)
    check(object(proof)["tag"] as? String == unhex(vector["app_proof"] as! String).base64EncodedString(),
          "session proof message")
    check(failure { _ = try session.credentialsMessage(ssid: "x", psk: nil, iv: iv) } == .notProven,
          "no credentials before the device has proved itself")
    try! session.verify(proven: reply(["op": "proven", "tag": unhex(vector["device_proof"] as! String).base64EncodedString()]))
    let creds = object(try! session.credentialsMessage(ssid: vector["ssid"] as! String, psk: psk, iv: iv))
    check(creds["ct"] as? String == unhex(vector["creds_ct"] as! String).base64EncodedString(), "session credentials message")
}

print("\nrefusals:")
do {
    let label = try! PoP.parseLabel("SDSETUP:1:SD-1A2B3C4D:ABCDEFGHJKMN")
    let nd = Data(repeating: 9, count: 16)
    let session = try! PoP.Session(label: label)
    let other = reply(["op": "challenge", "v": 1, "id": "SD-99999999", "nd": nd.base64EncodedString()])
    check(failure { _ = try session.proofMessage(challenge: other) } == .wrongDevice(expected: "SD-1A2B3C4D", got: "SD-99999999"),
          "a challenge from a device other than the label's is refused")
    let ours = reply(["op": "challenge", "v": 1, "id": "SD-1A2B3C4D", "nd": nd.base64EncodedString()])
    _ = try! session.proofMessage(challenge: ours)
    let forged = reply(["op": "proven", "tag": Data(repeating: 0, count: 32).base64EncodedString()])
    check(failure { try session.verify(proven: forged) } == .deviceNotAuthentic, "a device that cannot prove the code is refused")
    check(failure { _ = try session.credentialsMessage(ssid: "x", psk: "y") } == .notProven, "and gets no credentials")
    let locked = reply(["op": "error", "error": "locked", "retry_after_s": 31])
    check(failure { try session.verify(proven: locked) } == .rejected("locked", retryAfterSeconds: 31),
          "a lockout is surfaced with its wait")
    let v2 = reply(["op": "challenge", "v": 2, "id": "SD-1A2B3C4D", "nd": nd.base64EncodedString()])
    check(failure { _ = try session.proofMessage(challenge: v2) } == .malformedReply("expected a v1 challenge"),
          "another protocol version is refused")
}
check((try? PoP.normalize(code: "abcd efgh-jkmn")) == "ABCDEFGHJKMN", "typed codes are normalised")
check((try? PoP.normalize(code: "O1IL-0000-0000")) == "011100000000", "look-alikes fold as on the device")
check(failure { _ = try PoP.normalize(code: "ABCD-EFGH-JKMU") } == .invalidCode, "U is not in the alphabet")
check(failure { _ = try PoP.parseLabel("WIFI:S:home;;") } == .invalidLabel, "a non-SensorDyme QR is refused")
check(PoP.randomBytes(16) != PoP.randomBytes(16) && PoP.randomBytes(12).count == 12, "nonces are random")

print("\n\(failures == 0 ? "PASSED" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)
