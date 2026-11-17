"""M12: a phone that proves the setup code over Bluetooth enrols for local access.

docs/LOCAL_ACCESS.md §4.1. The `enroll` op runs inside a PROVEN BLE PoP
session, sealed both ways with the session key: a phone in range that never
saw the label cannot enrol, and a look-alike peripheral cannot swap its own
device key into the reply. Works with no network at all.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from core import local_auth as la
from services import ble_pop as pop
from services.ble_pop import PopAuthenticator, State
from sim.local_client import PhoneKey

DEVICE = "SD-1A2B3C4D"
CODE = "ABCD-EFGH-JKMN"
VECTORS = Path(__file__).with_name("local_access_vectors.json")


def _msg(**kw) -> bytes:
    return json.dumps(kw).encode()


def _reply(r) -> dict:
    return json.loads(r.encode())


class App:
    def __init__(self, code=CODE):
        self.code = code
        self.key = PhoneKey()

    def handshake(self, auth):
        self.na = os.urandom(16)
        challenge = _reply(auth.handle(_msg(op="hello", v=1, na=pop.b64e(self.na))))
        nd = pop._b64d(challenge["nd"], 16)
        self.keys = pop.session_keys(pop.derive_code_key(pop.normalize_setup_code(self.code), DEVICE),
                                     challenge["id"], self.na, nd)
        return _reply(auth.handle(_msg(op="proof", tag=pop.b64e(pop.app_proof(self.keys)))))

    def enroll_message(self, name="Owner iPhone"):
        iv = os.urandom(12)
        ct = pop.seal_enrollment(self.keys, la.b64u(self.key.public_key), name, iv)
        return _msg(op="enroll", iv=pop.b64e(iv), ct=pop.b64e(ct))


@pytest.fixture
def authority(tmp_path):
    return la.LocalAuthority(DEVICE, la.DeviceIdentity.load_or_create(tmp_path / "id.pem"),
                             la.ClientStore(tmp_path / "clients.json"))


@pytest.fixture
def auth(authority):
    a = PopAuthenticator(device_id=DEVICE, setup_code=CODE,
                         enroller=lambda key, name: authority.enroll(key, name, "ble"))
    a.start()
    return a


def test_a_proven_phone_enrols_and_learns_the_device_key_then_may_still_send_wifi(auth, authority):
    app = App()
    assert app.handshake(auth)["op"] == "proven"
    reply = _reply(auth.handle(app.enroll_message()))
    assert reply["op"] == "enrolled"
    opened = pop.open_enrolled(app.keys, pop._b64d(reply["iv"], 12), pop._b64d(reply["ct"]))
    assert opened == {"client_id": app.key.client_id, "device_id": DEVICE,
                      "device_public_key": la.b64u(authority.identity.public_key)}
    assert authority.store.clients[app.key.client_id].source == "ble"
    assert auth.state is State.PROVEN  # Wi-Fi credentials may follow
    iv = os.urandom(12)
    creds = auth.handle(_msg(op="creds", iv=pop.b64e(iv), ct=pop.b64e(pop.seal_credentials(app.keys, "Home", "pw",
                                                                                           iv))))
    assert creds.credentials == ("Home", "pw")


def test_enrolment_needs_the_proof(auth, authority):
    app = App()
    app.na = os.urandom(16)
    challenge = _reply(auth.handle(_msg(op="hello", v=1, na=pop.b64e(app.na))))
    app.keys = pop.session_keys(pop.derive_code_key(CODE, DEVICE), challenge["id"], app.na,
                                pop._b64d(challenge["nd"], 16))
    reply = _reply(auth.handle(app.enroll_message()))
    assert reply["op"] == "error" and reply["error"] == "unexpected"
    assert authority.store.clients == {}


def test_a_phone_with_the_wrong_code_never_reaches_enrolment(auth, authority):
    app = App(code="ZZZZ-ZZZZ-ZZZZ")
    assert app.handshake(auth)["error"] == "wrong_code"
    assert _reply(auth.handle(app.enroll_message()))["op"] == "error"
    assert authority.store.clients == {}


def test_an_enrolment_sealed_to_another_session_is_refused(auth, authority, tmp_path):
    first = App()
    first.handshake(auth)
    stale = first.enroll_message()
    auth.stop()
    auth.start()  # the window reopens: a new session with new nonces and keys
    second = App()
    assert second.handshake(auth)["op"] == "proven"
    reply = _reply(auth.handle(stale))
    assert reply["error"] == "malformed"
    assert authority.store.clients == {}


def test_one_enrolment_per_session(auth, authority):
    app = App()
    app.handshake(auth)
    assert _reply(auth.handle(app.enroll_message()))["op"] == "enrolled"
    again = _reply(auth.handle(app.enroll_message()))
    assert again["op"] == "error" and again["error"] == "unexpected"


def test_a_device_without_local_access_says_so():
    a = PopAuthenticator(device_id=DEVICE, setup_code=CODE)
    a.start()
    app = App()
    app.handshake(a)
    reply = _reply(a.handle(app.enroll_message()))
    assert reply["op"] == "error" and reply["error"] == "unsupported"


def test_a_refused_enrolment_is_an_error_reply_not_a_crash(authority):
    def full(key, name):
        raise la.LocalAuthError("too_many_clients", 409)

    a = PopAuthenticator(device_id=DEVICE, setup_code=CODE, enroller=full)
    a.start()
    app = App()
    app.handshake(a)
    reply = _reply(a.handle(app.enroll_message()))
    assert reply == {"op": "error", "error": "refused", "message": "the device could not enrol this phone",
                     "reason": "too_many_clients"}
    assert a.state is State.PROVEN


def test_the_messages_fit_the_auth_characteristic(auth):
    app = App()
    app.handshake(auth)
    request = app.enroll_message(name="x" * 64)
    assert len(request) <= pop.MAX_MESSAGE_BYTES
    assert len(auth.handle(request).encode()) <= pop.MAX_MESSAGE_BYTES


def test_the_ble_vectors_match_this_implementation():
    v = json.loads(VECTORS.read_text())["ble_enroll"]
    keys = pop.session_keys(pop.derive_code_key(v["setup_code"], v["device_id"]), v["device_id"],
                            base64.b64decode(v["na"]), base64.b64decode(v["nd"]))
    public_key = json.loads(v["enroll_plaintext"])["public_key"]
    assert base64.b64encode(pop.seal_enrollment(keys, public_key, v["name"],
                                                base64.b64decode(v["enroll_iv"]))).decode() == v["enroll_ct"]
    opened = pop.open_enrolled(keys, base64.b64decode(v["enrolled_iv"]), base64.b64decode(v["enrolled_ct"]))
    assert json.dumps(opened, separators=(",", ":"), sort_keys=True) == v["enrolled_plaintext"]


def test_the_local_access_vectors_are_current_and_verify():
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    assert subprocess.run([sys.executable, str(root / "scripts" / "local_access_vectors.py"), "--check"],  # noqa: S603 - fixed argv
                          cwd=root, check=False).returncode == 0
    v = json.loads(VECTORS.read_text())
    phone = la.b64u_decode(v["phone"]["public_key"], 65)
    device = la.b64u_decode(v["device"]["public_key"], 65)
    assert la.client_id_for(phone) == v["phone"]["client_id"]
    assert la.verify_signature(phone, v["request"]["message"].encode(), la.b64u_decode(v["request"]["phone_signature"],
                                                                                      64))
    assert la.verify_signature(device, v["session"]["message"].encode(),
                               la.b64u_decode(v["session"]["device_signature"], 64))
