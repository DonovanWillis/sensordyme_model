"""M12: the iPhone's sd-local-v1 client against the device's own Python (docs/LOCAL_ACCESS.md).

Two halves of one contract, checked against each other:

* ios/Tests/LocalAccessTests.swift embeds tests/local_access_vectors.json
  verbatim; this fails if the copy drifts.
* The app's own Swift (LocalAccess.swift, LocalDeviceClient.swift, ...) is run
  with `swift` against the REAL core/dashboard.py DashboardServer and
  core/local_auth.py LocalAuthority on 127.0.0.1 (with a dashboard token set,
  so nothing is let in for being on loopback): Swift signs with the vector's
  TEST phone key, the device verifies it; the device signs the session, Swift
  verifies it against the enrolled key; a signed stop reaches the runtime as
  the local app and is confirmed only once /api/state says stopped; a revoked
  phone gets 403 and is told to pair again.

macOS with the Swift toolchain only (skipped elsewhere). NOT a phone: no
iPhone, Secure Enclave or Wi-Fi is involved.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from core import local_auth as la
from core.dashboard import DashboardServer
from sim.local_client import PhoneKey

REPO = Path(__file__).resolve().parents[1]
VECTORS = REPO / "tests" / "local_access_vectors.json"
SWIFT_TEST = REPO / "ios" / "Tests" / "LocalAccessTests.swift"
SOURCES = ["LocalDeviceClient", "LocalAccess", "BLEProofOfPossession", "LocalEvents", "EventModel", "AlertSound"]
DEVICE_ID = "SD-1A2B3C4D"


def _swiftc() -> str | None:
    return shutil.which("swiftc")


def test_the_swift_tests_embed_the_same_vectors():
    swift = SWIFT_TEST.read_text("utf-8")
    start, end = swift.index('#"""') + len('#"""'), swift.index('"""#')
    assert json.loads(swift[start:end]) == json.loads(VECTORS.read_text(encoding="utf-8"))


#: The program: the app's sources, then this driver. Prints one JSON line per step.
DRIVER = r'''
import Foundation

func unhex(_ text: String) -> Data {
    var out = Data(); var i = text.startIndex
    while i < text.endIndex { let n = text.index(i, offsetBy: 2); out.append(UInt8(text[i..<n], radix: 16)!); i = n }
    return out
}
func say(_ object: [String: Any]) {
    print(String(data: try! JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]), encoding: .utf8)!)
    fflush(stdout)
}
let args = CommandLine.arguments
let mode = args[1]
let signer = SoftwareSigner(scalar: unhex(args[2]))!
if mode == "sign" {
    let message = Data(base64Encoded: args[3])!
    say(["public_key": LocalAccess.b64u(signer.publicKey), "client_id": signer.clientID,
         "sig": LocalAccess.b64u(try! signer.sign(message))])
    exit(0)
}
let url = URL(string: args[3])!
let devicePublicKey = LocalAccess.b64uDecode(args[4], length: 65)!
let client = LocalDeviceClient(deviceID: args[5], signer: signer, devicePublicKey: devicePublicKey, baseURL: url,
                               pause: { try? await Task.sleep(for: .seconds($0)) })
func describe(_ error: Error) -> String {
    if let e = error as? LocalAccessError { return "\(e) | \(e.message)" }
    return "\(error)"
}
do {
    let session = try await client.validSession()
    say(["step": "session", "ok": !session.isEmpty])
    let state = try await client.state()
    say(["step": "state", "protection": state.protection ?? "none"])
    let page = try await client.events(limit: 10)
    say(["step": "events", "count": page.events.count, "keys": page.events.map(\.eventKey),
         "sync": page.events.map { $0.sync ?? "" }])
    let stop = try await client.setProtection(stop: true, attempts: 20, interval: 0.25)
    say(["step": "stop", "confirmed": stop.isConfirmed, "text": stop.text])
    if mode == "revoke-wait" {
        say(["step": "waiting"])
        _ = readLine()
        do {
            _ = try await client.state()
            say(["step": "after-revoke", "error": "none"])
        } catch {
            say(["step": "after-revoke", "error": describe(error),
                 "forget": (error as? LocalAccessError)?.forgetsCredential ?? false])
        }
    }
} catch {
    say(["step": "error", "error": describe(error)])
}
'''


@pytest.fixture(scope="module")
def program(tmp_path_factory) -> Path:
    swiftc = _swiftc()
    if swiftc is None:
        pytest.skip("needs the Swift toolchain (macOS)")
    folder = tmp_path_factory.mktemp("swift-local-client")
    main = folder / "main.swift"
    main.write_text("".join((REPO / "ios" / "Sources" / f"{name}.swift").read_text("utf-8") + "\n"
                            for name in SOURCES) + DRIVER, "utf-8")
    binary = folder / "local-client"
    built = subprocess.run([swiftc, "-O", "-o", str(binary), str(main)],  # noqa: S603 - the repo's own sources
                           capture_output=True, text=True, timeout=600, check=False)
    assert built.returncode == 0, built.stderr[-3000:]
    return binary


def _vectors() -> dict:
    return json.loads(VECTORS.read_text("utf-8"))


def test_python_verifies_what_swift_signs(program):
    v = _vectors()
    message = v["request"]["message"].encode()
    import base64

    out = subprocess.run([str(program), "sign", v["phone"]["scalar_hex"],  # noqa: S603 - built above
                          base64.b64encode(message).decode()], capture_output=True, text=True, timeout=60,
                         check=False)
    assert out.returncode == 0, out.stderr
    answer = json.loads(out.stdout)
    assert answer["public_key"] == v["phone"]["public_key"]
    assert answer["client_id"] == v["phone"]["client_id"] == la.client_id_for(la.b64u_decode(answer["public_key"]))
    sig = la.b64u_decode(answer["sig"], 64)
    assert sig is not None and la.verify_signature(la.b64u_decode(v["phone"]["public_key"]), message, sig)
    assert not la.verify_signature(la.b64u_decode(v["phone"]["public_key"]), message + b"!", sig)


@pytest.fixture
def device(tmp_path):
    """The real dashboard on loopback WITH a token: loopback alone authorises nothing."""
    identity = la.DeviceIdentity.load_or_create(tmp_path / "local_identity.pem")
    authority = la.LocalAuthority(DEVICE_ID, identity, la.ClientStore(tmp_path / "local_clients.json"))
    from core.local_events import LocalEventJournal

    journal = LocalEventJournal(tmp_path / "events.jsonl")
    server = DashboardServer(host="127.0.0.1", port=0, token="legacy-token-for-the-test",
                             presets={"driver_safety": {"module": "driver_attention"}},
                             local_authority=authority, event_journal=journal)
    server.set_policy_state({"protection": {"state": "protecting", "role": "driver_safety"}})
    url = server.start()
    stop = threading.Event()

    def runtime() -> None:
        # What the runtime does with a dashboard stop: applies it a moment later.
        while not stop.is_set():
            request = server.pop_pending_protection()
            if request is not None:
                applied.append(request)
                time.sleep(0.5)
                state = "stopped" if request.get("action") == "stop" else "protecting"
                server.set_policy_state({"protection": {"state": state, "role": "driver_safety",
                                                        "by": request.get("by")}})
            time.sleep(0.05)

    applied: list[dict] = []
    thread = threading.Thread(target=runtime, daemon=True)
    thread.start()
    yield server, url, authority, journal, applied
    stop.set()
    thread.join(2)
    server.close()


def _record(journal) -> str:
    from datetime import UTC, datetime

    from core.events import Event

    event = Event(datetime.now(UTC), "distraction", 2.5, 0.81, "driver_attention", "gaze_off")
    journal.record(event, "loud")
    return event.event_id


def _run(program, url, authority, mode="run"):
    v = _vectors()
    return subprocess.Popen([str(program), mode, v["phone"]["scalar_hex"], url,  # noqa: S603 - built above
                             la.b64u(authority.identity.public_key), DEVICE_ID],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _lines(process, until: str | None = None, timeout: float = 60) -> list[dict]:
    lines, deadline = [], time.monotonic() + timeout
    while time.monotonic() < deadline:
        raw = process.stdout.readline()
        if not raw:
            break
        lines.append(json.loads(raw))
        if until and lines[-1].get("step") == until:
            break
    return lines


def test_the_swift_client_against_the_real_dashboard(program, device):
    _, url, authority, journal, applied = device
    phone = PhoneKey.from_scalar(int(_vectors()["phone"]["scalar_hex"], 16))
    key = _record(journal)
    # Enrolled the way Bluetooth PoP enrols it (the device's own authority).
    authority.enroll(la.b64u(phone.public_key), "Swift iPhone", "ble")
    process = _run(program, url, authority)
    lines = _lines(process)
    process.wait(30)
    steps = {line["step"]: line for line in lines}
    assert "error" not in steps, steps
    assert steps["session"]["ok"], "the device-signed session verified in Swift against the enrolled key"
    assert steps["state"]["protection"] == "protecting", "a read with X-SD-Session"
    assert steps["events"]["keys"] == [key] and steps["events"]["sync"] == ["local_only"], \
        "GET /api/events: the same event_key the backend would store; no uplink here, so local_only"
    assert steps["stop"]["confirmed"] is True, "confirmed only once /api/state said stopped"
    assert applied and applied[0]["action"] == "stop" and applied[0]["confirm"] is True
    assert applied[0]["by"] == "local_app" and applied[0]["client"] == phone.client_id, \
        "the device verified Swift's signature over the exact body bytes and recorded which phone"


def test_a_swift_client_the_device_never_enrolled_gets_nothing(program, device):
    _, url, authority, _, applied = device
    process = _run(program, url, authority)
    lines = _lines(process)
    process.wait(30)
    assert lines[-1]["step"] == "error" and "forgetCredential" in lines[-1]["error"], lines
    assert "paired with the camera again" in lines[-1]["error"]
    assert applied == []


def test_a_revoked_swift_client_is_cut_off_and_told_to_pair_again(program, device):
    _, url, authority, _, _ = device
    phone = PhoneKey.from_scalar(int(_vectors()["phone"]["scalar_hex"], 16))
    authority.enroll(la.b64u(phone.public_key), "Swift iPhone", "ble")
    process = _run(program, url, authority, mode="revoke-wait")
    _lines(process, until="waiting")
    assert authority.revoke(phone.client_id)
    process.stdin.write("\n")
    process.stdin.flush()
    lines = _lines(process, until="after-revoke")
    process.wait(30)
    after = lines[-1]
    assert after["step"] == "after-revoke" and after["forget"] is True, after
    assert "paired with the camera again" in after["error"]


def test_a_lookalike_device_cannot_open_a_session_for_the_swift_client(program, device, tmp_path):
    _, url, authority, _, applied = device
    phone = PhoneKey.from_scalar(int(_vectors()["phone"]["scalar_hex"], 16))
    authority.enroll(la.b64u(phone.public_key), "Swift iPhone", "ble")
    other = la.DeviceIdentity.load_or_create(tmp_path / "other.pem")  # the key the phone THINKS it enrolled with
    v = _vectors()
    process = subprocess.Popen([str(program), "run", v["phone"]["scalar_hex"], url,  # noqa: S603 - built above
                                la.b64u(other.public_key), DEVICE_ID], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    lines = _lines(process)
    process.wait(30)
    assert lines[-1]["step"] == "error" and "notAuthentic" in lines[-1]["error"], lines
    assert applied == [], "nothing was sent to a device that could not prove it is the enrolled one"
