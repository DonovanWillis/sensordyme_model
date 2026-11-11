"""Proof of possession for Bluetooth provisioning (services/ble_pop.py), device side.

Protocol, state machine, lockout, config gate, GATT wiring and the provisioning
service, all without a radio. Nothing here is a claim about physical BLE.
"""

from __future__ import annotations

import hashlib
import hmac
import itertools
import json
import logging
import os
from pathlib import Path
from typing import ClassVar

import pytest

pytest.importorskip("cryptography")

from services import ble_pop as pop
from services.ble_pop import Limits, PopAuthenticator, State

VECTORS = Path(__file__).with_name("ble_pop_vectors.json")
DEVICE = "SD-1A2B3C4D"
CODE = "ABCD-EFGH-JKMN"


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _msg(**kw) -> bytes:
    return json.dumps(kw).encode()


def _reply(r) -> dict:
    return json.loads(r.encode())


class App:
    """The phone's half, built from the module's primitives (the Swift client mirrors it)."""

    def __init__(self, code=CODE, device_id=DEVICE):
        self.code, self.device_id = code, device_id
        self.keys = None

    def hello(self, auth):
        self.na = os.urandom(16)
        return auth.handle(_msg(op="hello", v=1, na=pop.b64e(self.na)))

    def prove(self, auth, challenge: dict):
        nd = pop._b64d(challenge["nd"], 16)
        key = pop.derive_code_key(pop.normalize_setup_code(self.code), self.device_id)
        self.keys = pop.session_keys(key, challenge["id"], self.na, nd)
        self.proof_msg = _msg(op="proof", tag=pop.b64e(pop.app_proof(self.keys)))
        return auth.handle(self.proof_msg)

    def creds_message(self, ssid="HomeWiFi", psk="secret-passphrase", iv=None):
        iv = iv or os.urandom(12)
        ct = pop.seal_credentials(self.keys, ssid, psk, iv)
        return _msg(op="creds", iv=pop.b64e(iv), ct=pop.b64e(ct))

    def handshake(self, auth):
        challenge = _reply(self.hello(auth))
        assert challenge["op"] == "challenge", challenge
        return _reply(self.prove(auth, challenge))


class _Lines(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@pytest.fixture
def device_log():
    """Everything the BLE and provisioning loggers emit (the project's loggers do not reach caplog)."""
    handler = _Lines()
    loggers = [logging.getLogger(f"sensordyme.{n}") for n in ("ble", "provisioning")]
    for logger in loggers:
        logger.addHandler(handler)
    yield handler.lines
    for logger in loggers:
        logger.removeHandler(handler)


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def auth(clock):
    a = PopAuthenticator(device_id=DEVICE, setup_code=CODE, clock=clock)
    a.start()
    return a


# ---------------------------------------------------------------------- vectors
def test_the_shared_vectors_match_this_implementation():
    from scripts.ble_setup_code import vectors

    assert json.loads(VECTORS.read_text(encoding="utf-8")) == vectors()


@pytest.mark.parametrize("case", json.loads(VECTORS.read_text(encoding="utf-8"))["cases"], ids=lambda c: c["name"])
def test_the_vectors_against_independent_stdlib_primitives(case):
    """PBKDF2 and HMAC recomputed with hashlib/hmac; AES-GCM opened by the device path."""
    code = pop.normalize_setup_code(case["code"]).encode()
    salt = b"SensorDyme BLE PoP v1|" + case["device_id"].encode()
    assert hashlib.pbkdf2_hmac("sha256", code, salt, 600_000).hex() == case["code_key"]
    t = bytes.fromhex(case["transcript"])
    k_mac = bytes.fromhex(case["k_mac"])
    assert hmac.new(k_mac, b"app-proof" + t, "sha256").hexdigest() == case["app_proof"]
    assert hmac.new(k_mac, b"device-proof" + t, "sha256").hexdigest() == case["device_proof"]
    keys = pop.SessionKeys(k_mac, bytes.fromhex(case["k_enc"]), t)
    plain = pop.open_credentials(keys, bytes.fromhex(case["iv"]), bytes.fromhex(case["creds_ct"]))
    assert json.loads(plain) == {"ssid": case["ssid"], "psk": case["psk"]}


def test_the_swift_tests_embed_the_same_vectors():
    swift = (Path(__file__).resolve().parents[1] / "ios/Tests/BLEProofOfPossessionTests.swift").read_text("utf-8")
    start, end = swift.index('#"""') + len('#"""'), swift.index('"""#')
    assert json.loads(swift[start:end]) == json.loads(VECTORS.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------- setup code
def test_setup_code_forms():
    assert pop.normalize_setup_code("abcd efgh-jkmn") == "ABCDEFGHJKMN"
    assert pop.normalize_setup_code("O1IL-0000-0000") == "0111" + "0" * 8  # look-alikes folded
    assert pop.format_setup_code("ABCDEFGHJKMN") == "ABCD-EFGH-JKMN"
    for bad in ("", "ABCD-EFGH-JKM", "ABCD-EFGH-JKMNP", "ABCD-EFGH-JKMU", None):
        with pytest.raises(pop.SetupCodeError):
            pop.normalize_setup_code(bad)
    codes = {pop.generate_setup_code() for _ in range(50)}
    assert len(codes) == 50 and all(len(pop.normalize_setup_code(c)) == 12 for c in codes)


def test_qr_payload_round_trip_and_refusals():
    text = pop.qr_payload(DEVICE, CODE)
    assert text == "SDSETUP:1:SD-1A2B3C4D:ABCDEFGHJKMN"
    assert pop.parse_qr_payload(text) == (DEVICE, "ABCDEFGHJKMN")
    for bad in ("SDSETUP:2:SD-1:ABCDEFGHJKMN", "WIFI:S:x;;", "SDSETUP:1::ABCDEFGHJKMN", ""):
        with pytest.raises(pop.SetupCodeError):
            pop.parse_qr_payload(bad)


def test_the_code_file_is_written_once_0600_and_loaded_only_if_private(tmp_path):
    path = tmp_path / "ble_setup_code"
    pop.write_setup_code(str(path), CODE)
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert pop.load_setup_code(str(path)) == "ABCDEFGHJKMN"
    with pytest.raises(pop.SetupCodeError, match="already exists"):
        pop.write_setup_code(str(path), pop.generate_setup_code())
    assert pop.load_setup_code(str(path)) == "ABCDEFGHJKMN", "a refused rewrite left the original"
    assert [p.name for p in tmp_path.iterdir()] == ["ble_setup_code"], "no temp file left behind"
    path.chmod(0o640)
    with pytest.raises(pop.SetupCodeError, match="readable by other accounts"):
        pop.load_setup_code(str(path))
    path.chmod(0o600)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(pop.SetupCodeError, match="regular file"):
        pop.load_setup_code(str(link))
    with pytest.raises(pop.SetupCodeError, match="no setup code"):
        pop.load_setup_code(str(tmp_path / "missing"))


# ---------------------------------------------------------------------- protocol
def test_the_happy_path_is_mutual_and_delivers_the_credentials(auth):
    app = App()
    proven = app.handshake(auth)
    assert proven["op"] == "proven" and auth.state is State.PROVEN
    # The app checks the device back before sending anything secret.
    assert pop._b64d(proven["tag"], 32) == pop.device_proof(app.keys)
    reply = auth.handle(app.creds_message())
    assert _reply(reply) == {"op": "accepted"}
    assert reply.credentials == ("HomeWiFi", "secret-passphrase")
    assert auth.state is State.CREDENTIALS_RECEIVED
    auth.apply_result(True)
    assert auth.state is State.APPLIED
    assert [h[1] for h in auth.history] == [State.ADVERTISING, State.CHALLENGED, State.PROVEN,
                                            State.CREDENTIALS_RECEIVED, State.APPLIED]


def test_an_open_network_and_a_failed_join_then_a_fresh_handshake(auth):
    app = App()
    app.handshake(auth)
    assert auth.handle(app.creds_message("Cafe", None)).credentials == ("Cafe", "")
    auth.apply_result(False)
    assert auth.state is State.FAILED
    assert auth.handle(app.creds_message()).message["error"] == "unexpected", "a failed session is over"
    assert App().handshake(auth)["op"] == "proven"


def test_a_wrong_code_spends_the_challenge(auth):
    wrong = App(code="ZZZZ-ZZZZ-ZZZZ")
    challenge = _reply(wrong.hello(auth))
    assert _reply(wrong.prove(auth, challenge))["error"] == "wrong_code"
    assert auth.state is State.ADVERTISING
    right = App()
    right.na = wrong.na
    # The right code against the SAME (spent) challenge is not accepted: one guess per challenge.
    assert _reply(right.prove(auth, challenge))["error"] == "unexpected"


def test_a_device_that_does_not_know_the_code_cannot_fake_its_proof(auth):
    app = App()
    challenge = _reply(app.hello(auth))
    proven = _reply(app.prove(auth, challenge))
    impostor = pop.session_keys(pop.derive_code_key("ZZZZZZZZZZZZ", DEVICE), DEVICE, app.na,
                                pop._b64d(challenge["nd"], 16))
    assert pop.device_proof(impostor) != pop._b64d(proven["tag"], 32)


def test_a_proof_for_another_device_id_is_refused(clock):
    """Same code, different device: the id is in every key, so the label for one unit opens no other."""
    other = PopAuthenticator(device_id="SD-99999999", setup_code=CODE, clock=clock)
    other.start()
    app = App()
    challenge = _reply(app.hello(other))
    assert challenge["id"] == "SD-99999999"
    keys = pop.session_keys(pop.derive_code_key("ABCDEFGHJKMN", DEVICE), DEVICE, app.na,
                            pop._b64d(challenge["nd"], 16))
    assert _reply(other.handle(_msg(op="proof", tag=pop.b64e(pop.app_proof(keys)))))["error"] == "wrong_code"


def test_replayed_proof_and_credentials_from_an_earlier_session_are_refused(auth):
    app = App()
    assert app.handshake(auth)["op"] == "proven"
    recorded_proof, recorded_creds = app.proof_msg, app.creds_message()
    auth.handle(recorded_creds)
    auth.apply_result(True)
    # A recording of that whole session, replayed into a new one with the same app nonce:
    auth.handle(_msg(op="hello", v=1, na=pop.b64e(app.na)))
    assert _reply(auth.handle(recorded_proof))["error"] == "wrong_code", "the device nonce is fresh"
    assert _reply(auth.handle(recorded_creds))["error"] == "unexpected"


def test_credentials_need_a_proof_and_must_authenticate(auth):
    app = App()
    app.keys = pop.session_keys(b"\0" * 32, DEVICE, b"\1" * 16, b"\2" * 16)
    assert _reply(auth.handle(app.creds_message()))["error"] == "unexpected"
    app.handshake(auth)
    msg = json.loads(app.creds_message())
    ct = bytearray(pop._b64d(msg["ct"]))
    ct[0] ^= 1
    assert _reply(auth.handle(_msg(op="creds", iv=msg["iv"], ct=pop.b64e(bytes(ct)))))["error"] == "malformed"
    assert auth.state is State.PROVEN, "a forged message changes nothing"
    bad_json = pop.seal_credentials(app.keys, "x", "y", b"\3" * 12)
    assert auth.handle(_msg(op="creds", iv=pop.b64e(b"\3" * 12), ct=pop.b64e(bad_json))).credentials == ("x", "y")


@pytest.mark.parametrize("data", [
    b"not json", b"[1]", b"\xff\xfe", b"{}", _msg(op="nope"), _msg(op="hello", v=2, na=pop.b64e(b"0" * 16)),
    _msg(op="hello", v=1, na=pop.b64e(b"0" * 15)), _msg(op="hello", v=1, na="!!!"), _msg(op="proof", tag="AA=="),
    _msg(op="creds", iv=pop.b64e(b"0" * 11), ct=pop.b64e(b"0" * 40)), b"{" + b" " * 600 + b"}",
])
def test_malformed_messages_are_answered_and_change_nothing(auth, data):
    before = auth.state
    reply = _reply(auth.handle(data))
    assert reply["op"] == "error" and reply["error"] in ("malformed", "unsupported_version", "unexpected")
    assert auth.state is before


def test_lockout_after_repeated_wrong_codes_survives_reconnecting(auth, clock):
    wrong = App(code="ZZZZ-ZZZZ-ZZZZ")
    for _ in range(4):
        assert wrong.handshake(auth)["error"] == "wrong_code"
    fifth = wrong.handshake(auth)
    assert fifth["error"] == "wrong_code" and fifth["retry_after_s"] == 31
    locked = _reply(App().hello(auth))
    assert locked["error"] == "locked" and auth.state is State.ADVERTISING, "even the right code waits"
    clock.t += 31
    assert wrong.handshake(auth)["retry_after_s"] == 61, "each further failure doubles the lockout"
    clock.t += 61
    assert App().handshake(auth)["op"] == "proven"
    auth.stop()
    auth.start()
    for _ in range(4):
        assert "retry_after_s" not in wrong.handshake(auth), "a success reset the count"


def test_lockout_is_capped(clock):
    a = PopAuthenticator(device_id=DEVICE, setup_code=CODE, clock=clock,
                         limits=Limits(max_failures=1, lockout_base_s=1000, lockout_max_s=1500))
    a.start()
    wrong = App(code="ZZZZ-ZZZZ-ZZZZ")
    for _ in range(3):
        reply = wrong.handshake(a)
        clock.t += reply["retry_after_s"]
    assert reply["retry_after_s"] == 1501


def test_timeouts(auth, clock):
    app = App()
    app.hello(auth)
    clock.t += 31
    assert auth.tick() is State.ADVERTISING
    app.handshake(auth)
    clock.t += 121
    assert auth.tick() is State.ADVERTISING and not auth.accepts_credentials()
    app.handshake(auth)
    auth.handle(app.creds_message())
    clock.t += 121
    assert auth.tick() is State.FAILED, "a join that never reports is a failure, not a wedge"


def test_a_second_phone_cannot_take_over_a_proven_session(auth):
    App().handshake(auth)
    assert _reply(App().hello(auth))["error"] == "busy"
    assert auth.state is State.PROVEN


def test_every_transition_outside_the_table_is_illegal(clock):
    a = PopAuthenticator(device_id=DEVICE, setup_code=CODE, clock=clock)
    illegal = 0
    for src, dst in itertools.product(State, State):
        a.state = src
        if dst in pop.TRANSITIONS[src]:
            a._go(dst, "test")
        else:
            illegal += 1
            with pytest.raises(pop.IllegalTransitionError):
                a._go(dst, "test")
    assert illegal == len(State) ** 2 - sum(len(v) for v in pop.TRANSITIONS.values())
    a.state = State.ADVERTISING
    a.apply_result(True)  # an outcome with no credentials pending is ignored, not a transition
    assert a.state is State.ADVERTISING


def test_resolve_auth():
    assert pop.resolve_auth({}) == "legacy"
    assert pop.resolve_auth({"auth": "pop"}, "production") == "pop"
    with pytest.raises(pop.SetupCodeError, match="refused when deployment: production"):
        pop.resolve_auth({"auth": "legacy"}, "production")
    with pytest.raises(pop.SetupCodeError, match="must be one of"):
        pop.resolve_auth({"auth": "none"})


def test_the_schema_refuses_unauthenticated_setup_in_production():
    from core.config_schema import validate

    base = {"module": "m", "network": {"provisioning": {"method": "ble"}}}
    assert not any("auth" in e or "deployment" in e for e in validate(base)[0])
    errors, warnings = validate({**base, "deployment": "production"})
    assert any("auth: legacy" in e for e in errors)
    assert not any("deployment" in w for w in warnings)
    ok = {**base, "deployment": "production", "network": {"provisioning": {"method": "ble", "auth": "pop"}}}
    assert not any("auth" in e for e in validate(ok)[0])
    assert any("`deployment:`" in e for e in validate({**base, "deployment": "prod"})[0])
    assert any("network.provisioning.auth" in e for e in
               validate({**base, "network": {"provisioning": {"auth": "maybe"}}})[0])


# ---------------------------------------------------------------------- GATT wiring
def test_the_gatt_server_refuses_plain_credentials_under_pop(device_log):
    from services.ble_gatt import BLEProvisioningServer

    got = []
    plain = json.dumps({"ssid": "Evil", "psk": "x"}).encode()
    legacy = BLEProvisioningServer("n", {}, on_credentials=lambda s, p: got.append(s))
    assert legacy.handle_credentials_write(plain) and got == ["Evil"]
    guarded = BLEProvisioningServer("n", {}, on_credentials=lambda s, p: got.append(s), on_auth=lambda d: b'{"op":"x"}')
    assert not guarded.handle_credentials_write(plain)
    assert got == ["Evil"] and "refused" in device_log[-1]
    assert guarded.handle_auth_write(b"{}") == b'{"op":"x"}' and guarded._auth_reply == b'{"op":"x"}'

    def boom(_):
        raise RuntimeError("secret detail")

    crashing = BLEProvisioningServer("n", {}, on_credentials=lambda s, p: None, on_auth=boom)
    assert json.loads(crashing.handle_auth_write(b"{}"))["error"] == "internal"
    assert "secret detail" not in "\n".join(device_log)


# ---------------------------------------------------------------------- provisioning service
class _Wifi:
    def __init__(self):
        self.online, self.joined = False, []

    def is_online(self):
        return self.online

    def current_ssid(self):
        return None

    def ip_address(self):
        return "10.0.0.2" if self.online else None

    def watch(self, on_change, stop):
        return None

    def provision(self, ssid, psk):
        self.joined.append((ssid, psk))
        self.online = True
        return True, "joined"

    def scan(self):
        return []


class _BLE:
    instances: ClassVar[list] = []

    def __init__(self, device_name, device_info, on_credentials, scan_networks=None, on_command=None,
                 on_auth=None):
        self.device_info, self.on_credentials, self.on_auth = device_info, on_credentials, on_auth
        self.statuses = []
        _BLE.instances.append(self)

    def start(self):
        pass

    def stop(self):
        pass

    def set_status(self, status):
        self.statuses.append(status)

    def set_scans_paused(self, paused):
        pass

    def refresh_networks(self, force=False):
        pass


def _service(monkeypatch, tmp_path, config, deployment="development"):
    import services.ble_gatt as ble_gatt
    import services.provisioning as prov

    monkeypatch.setattr(ble_gatt, "BLEProvisioningServer", _BLE)
    monkeypatch.setattr(prov, "device_id", lambda: DEVICE)
    _BLE.instances = []
    wifi = _Wifi()
    cfg = {"setup_code_path": str(tmp_path / "code"), **config}
    return prov.ProvisioningService(wifi, cfg, deployment=deployment), wifi


def test_the_service_provisions_only_after_proof(monkeypatch, tmp_path, device_log):
    pop.write_setup_code(str(tmp_path / "code"), CODE)
    service, wifi = _service(monkeypatch, tmp_path, {"auth": "pop"}, deployment="production")
    service._open_ble_window()
    ble = _BLE.instances[-1]
    assert ble.device_info["auth"] == "pop" and ble.on_auth is not None
    assert json.loads(ble.on_auth(_msg(op="creds", iv="AA==", ct="AA=="))).get("op") == "error"
    app = App()
    app.na = b"\7" * 16
    challenge = json.loads(ble.on_auth(_msg(op="hello", v=1, na=pop.b64e(app.na))))
    nd = pop._b64d(challenge["nd"], 16)
    app.keys = pop.session_keys(pop.derive_code_key("ABCDEFGHJKMN", DEVICE), DEVICE, app.na, nd)
    proven = json.loads(ble.on_auth(_msg(op="proof", tag=pop.b64e(pop.app_proof(app.keys)))))
    assert proven["op"] == "proven"
    assert json.loads(ble.on_auth(app.creds_message("HomeWiFi", "pa55-phrase"))) == {"op": "accepted"}
    service._connect_thread.join(5)
    assert wifi.joined == [("HomeWiFi", "pa55-phrase")]
    assert service._pop.state is State.APPLIED
    assert ble.statuses[-1]["provision"]["state"] == "ok"
    for secret in ("ABCDEFGHJKMN", "ABCD-EFGH-JKMN", "pa55-phrase", proven["tag"]):
        assert secret not in "\n".join(device_log)
    assert device_log, "the log capture saw the session"


def test_legacy_in_production_keeps_bluetooth_shut_and_says_why(monkeypatch, tmp_path):
    service, _ = _service(monkeypatch, tmp_path, {"auth": "legacy"}, deployment="production")
    service.state = "waiting"
    assert not service._may_open_ble(0.0)
    healthy, detail = service.status()
    assert not healthy and "refused when deployment: production" in detail
    assert _BLE.instances == []


def test_pop_without_a_usable_code_file_keeps_bluetooth_shut(monkeypatch, tmp_path):
    service, _ = _service(monkeypatch, tmp_path, {"auth": "pop"})
    service._open_ble_window()
    assert _BLE.instances == [] and "setup code unusable" in service._ble_failure
    (tmp_path / "code").write_text("ABCD-EFGH-JKMN\n")
    (tmp_path / "code").chmod(0o644)
    service._ble_failed_at = None
    service._open_ble_window()
    assert _BLE.instances == [] and "readable by other accounts" in service._ble_failure
    (tmp_path / "code").chmod(0o600)
    service._open_ble_window()
    assert len(_BLE.instances) == 1, "a code written later is picked up on retry"


def test_legacy_development_is_unchanged(monkeypatch, tmp_path):
    service, wifi = _service(monkeypatch, tmp_path, {})
    service._open_ble_window()
    ble = _BLE.instances[-1]
    assert ble.on_auth is None and ble.device_info["auth"] == "legacy"
    ble.on_credentials("HomeWiFi", "pw")
    service._connect_thread.join(5)
    assert wifi.joined == [("HomeWiFi", "pw")]
