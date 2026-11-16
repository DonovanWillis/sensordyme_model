"""M12 sd-local-v1: per-phone local access to the device (docs/LOCAL_ACCESS.md).

The gap (docs/PORTABLE_OPERATION.md C1-C3): the only local credential was one
shared `dashboard.token` pasted by hand, and the shipped config leaves it unset,
so a new customer's camera refused the app entirely. These tests pin the
replacement on the real DashboardServer, bound to the LAN with NO token -- the
shipping configuration.
"""

from __future__ import annotations

import http.client
import json
import os
import stat
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from core import local_auth as la
from core.dashboard import DashboardServer
from sim.local_client import LocalPhone, PhoneKey

DEVICE = "SD-1A2B3C4D"


@pytest.fixture
def authority(tmp_path):
    identity = la.DeviceIdentity.load_or_create(tmp_path / "local_identity.pem")
    store = la.ClientStore(tmp_path / "local_clients.json")
    return la.LocalAuthority(DEVICE, identity, store)


@pytest.fixture
def device(authority):
    """The shipping shape: LAN bind, no dashboard token, local access on."""
    server = DashboardServer(host="0.0.0.0", port=0, token=None,  # noqa: S104 - the LAN bind under test
                             presets={"driver_safety": {"module": "driver_attention"},
                                      "general": {"module": "general"}},
                             local_authority=authority)
    url = server.start()
    yield server, url
    server.close()


def _enrolled_phone(url: str, authority: la.LocalAuthority, name: str = "Owner iPhone") -> LocalPhone:
    """A phone enrolled the way Bluetooth PoP enrols it (authority.enroll, source ble)."""
    phone = LocalPhone(url, name=name)
    phone.trust(authority.enroll(la.b64u(phone.key.public_key), name, "ble"))
    return phone


# ------------------------------------------------------------------ the shipping default stays closed
def test_with_no_token_and_no_enrolled_phone_everything_stays_refused(device):
    _, url = device
    phone = LocalPhone(url)
    for path in ("/api/state", "/api/health", "/api/history", "/api/events", "/stream", "/api/local/clients"):
        assert phone.call("GET", path).status == 401, path
    assert phone.call("POST", "/api/protection", body={"action": "stop", "confirm": True}).status == 401


def test_hello_gives_a_challenge_and_a_fingerprint_and_nothing_else(device, authority):
    _, url = device
    hello = LocalPhone(url).hello()
    assert set(hello) == {"contract", "device_id", "device_key_fp", "challenge", "challenge_ttl_s",
                          "token_enrollment"}
    assert hello["contract"] == "sd-local-v1" and hello["device_id"] == DEVICE
    assert hello["device_key_fp"] == authority.identity.fingerprint
    assert hello["token_enrollment"] is False
    assert len(hello["challenge"]) == 32


def test_a_phone_cannot_enrol_itself_over_the_network_without_the_device_token(device):
    _, url = device
    phone = LocalPhone(url)
    reply = phone.call("POST", "/api/local/enroll", body={"public_key": la.b64u(phone.key.public_key)})
    assert reply.status == 401 and reply.body["code"] == "token_required"
    # A signature from an unenrolled key opens nothing either.
    assert phone.open_session().status == 403


# ------------------------------------------------------------------ an enrolled phone, no token anywhere
def test_an_enrolled_phone_reads_state_and_the_stream_with_a_device_signed_session(device, authority):
    _, url = device
    phone = _enrolled_phone(url, authority)
    reply = phone.open_session()  # also verifies the DEVICE's signature on the session
    assert reply.status == 200 and reply.body["expires_in_s"] == int(la.SESSION_TTL_S)
    assert phone.get("/api/state").status == 200
    assert phone.get("/api/health").status == 200
    # The stream answers with the multipart header (closed right after).
    parts = urlsplit(url)
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=5)
    conn.request("GET", "/stream", headers={la.SESSION_HEADER: phone.session})
    resp = conn.getresponse()
    assert resp.status == 200 and resp.getheader("Content-Type").startswith("multipart/x-mixed-replace")
    conn.close()


def test_a_session_from_a_device_whose_key_differs_is_rejected_by_the_phone(device, authority, tmp_path):
    _, url = device
    phone = _enrolled_phone(url, authority)
    phone.device_public_key = la.DeviceIdentity.generate().public_key  # a look-alike's key
    with pytest.raises(AssertionError, match="does not verify"):
        phone.open_session()


def test_a_read_session_can_never_command(device, authority):
    server, url = device
    phone = _enrolled_phone(url, authority)
    phone.open_session()
    reply = phone.call("POST", "/api/protection", body={"action": "stop", "confirm": True},
                       headers={la.SESSION_HEADER: phone.session})
    assert reply.status == 401
    assert server.pop_pending_protection() is None


def test_a_signed_stop_reaches_the_runtime_marked_as_the_local_app(device, authority):
    server, url = device
    phone = _enrolled_phone(url, authority)
    reply = phone.signed("POST", "/api/protection", {"action": "stop", "confirm": True})
    assert reply.status == 202
    assert server.pop_pending_protection() == {"action": "stop", "confirm": True, "by": "local_app",
                                               "client": phone.key.client_id}


def test_a_signed_stop_still_needs_confirmation(device, authority):
    server, url = device
    phone = _enrolled_phone(url, authority)
    assert phone.signed("POST", "/api/protection", {"action": "stop"}).status == 400
    assert server.pop_pending_protection() is None


def test_role_floors_apply_to_signed_presets(device, authority):
    server, url = device
    server.preset_guard = lambda name: "role 'driver_safety' requires driver_attention" if name == "general" \
        else None
    phone = _enrolled_phone(url, authority)
    reply = phone.signed("POST", "/api/preset", {"preset": "general"})
    assert reply.status == 409 and reply.body["refused"] is True
    assert server.pop_pending_preset() is None
    assert phone.signed("POST", "/api/preset", {"preset": "driver_safety"}).status == 202


def test_a_replayed_signed_command_is_refused(device, authority):
    _, url = device
    phone = _enrolled_phone(url, authority)
    hello = phone.hello()
    body = json.dumps({"action": "resume"}).encode()
    message = la.request_message(DEVICE, phone.key.client_id, hello["challenge"], "POST", "/api/protection", body)
    header = la.format_header(phone.key.client_id, hello["challenge"], phone.key.sign(message))
    first = phone.call("POST", "/api/protection", raw=body, headers={la.HEADER: header})
    second = phone.call("POST", "/api/protection", raw=body, headers={la.HEADER: header})
    assert first.status == 202
    assert second.status == 401 and second.body["code"] == "stale_challenge"


def test_a_tampered_body_or_path_breaks_the_signature(device, authority):
    _, url = device
    phone = _enrolled_phone(url, authority)
    hello = phone.hello()
    signed_body = json.dumps({"action": "resume"}).encode()
    message = la.request_message(DEVICE, phone.key.client_id, hello["challenge"], "POST", "/api/protection",
                                 signed_body)
    header = la.format_header(phone.key.client_id, hello["challenge"], phone.key.sign(message))
    sent = json.dumps({"action": "stop", "confirm": True}).encode()
    reply = phone.call("POST", "/api/protection", raw=sent, headers={la.HEADER: header})
    assert reply.status == 401 and reply.body["code"] == "bad_signature"


def test_a_signature_for_another_device_is_refused(tmp_path):
    other = la.LocalAuthority("SD-99999999", la.DeviceIdentity.generate(), la.ClientStore(None))
    key = PhoneKey()
    other.enroll(la.b64u(key.public_key), "x", "ble")
    challenge = other.issue_challenge()
    message = la.request_message(DEVICE, key.client_id, challenge, "POST", "/api/protection", b"{}")
    with pytest.raises(la.LocalAuthError) as exc:
        other.verify_request(la.format_header(key.client_id, challenge, key.sign(message)), "POST",
                             "/api/protection", b"{}", "10.0.0.5")
    assert exc.value.reason == "bad_signature"


# ------------------------------------------------------------------ revocation and lifecycle
def test_revoking_a_phone_ends_its_session_and_its_stream_at_once(device, authority):
    _, url = device
    owner = _enrolled_phone(url, authority, "Owner")
    lost = _enrolled_phone(url, authority, "Lost phone")
    lost.open_session()
    assert lost.get("/api/state").status == 200
    reply = owner.signed("POST", "/api/local/revoke", {"client_id": lost.key.client_id})
    assert reply.status == 200 and reply.body["revoked"] == lost.key.client_id
    after = lost.get("/api/state")
    assert after.status == 401 and after.body["code"] == "session_invalid"
    again = lost.open_session()
    assert again.status == 403 and again.body["code"] == "unknown_client"
    assert not authority.client_active(lost.key.client_id)
    assert authority.client_active(owner.key.client_id)


def test_sessions_expire_and_are_bound_to_the_peer(authority):
    now = [1000.0]
    auth = la.LocalAuthority(DEVICE, authority.identity, la.ClientStore(None), clock=lambda: now[0])
    key = PhoneKey()
    auth.enroll(la.b64u(key.public_key), "p", "ble")
    challenge = auth.issue_challenge()
    body = json.dumps({"nonce": "ab" * 16}).encode()
    header = la.format_header(key.client_id, challenge,
                              key.sign(la.request_message(DEVICE, key.client_id, challenge, "POST",
                                                          "/api/local/session", body)))
    session = auth.open_session(header, "/api/local/session", body, "10.0.0.5")["session"]
    assert auth.check_session(session, "10.0.0.5") is not None
    assert auth.check_session(session, "10.0.0.6") is None  # another host on the LAN
    now[0] += la.SESSION_TTL_S + 1
    assert auth.check_session(session, "10.0.0.5") is None


def test_challenges_expire_and_are_single_use(authority):
    now = [50.0]
    auth = la.LocalAuthority(DEVICE, authority.identity, la.ClientStore(None), clock=lambda: now[0])
    key = PhoneKey()
    auth.enroll(la.b64u(key.public_key), "p", "ble")
    challenge = auth.issue_challenge()
    now[0] += la.CHALLENGE_TTL_S + 1
    message = la.request_message(DEVICE, key.client_id, challenge, "POST", "/api/protection", b"{}")
    with pytest.raises(la.LocalAuthError) as exc:
        auth.verify_request(la.format_header(key.client_id, challenge, key.sign(message)), "POST",
                            "/api/protection", b"{}", "p")
    assert exc.value.reason == "stale_challenge"


def test_repeated_bad_signatures_lock_the_peer_out_with_backoff(authority):
    now = [0.0]
    auth = la.LocalAuthority(DEVICE, authority.identity, la.ClientStore(None), clock=lambda: now[0])
    key = PhoneKey()
    auth.enroll(la.b64u(key.public_key), "p", "ble")
    forged = la.format_header(key.client_id, "0" * 32, bytes(64))
    for _ in range(la.FAIL_LIMIT):
        with pytest.raises(la.LocalAuthError):
            auth.verify_request(forged, "POST", "/x", b"", "10.0.0.9")
    with pytest.raises(la.LocalAuthError) as exc:
        auth.verify_request(forged, "POST", "/x", b"", "10.0.0.9")
    assert exc.value.reason == "locked_out" and exc.value.status == 429
    # Another peer is unaffected; the lockout ends.
    with pytest.raises(la.LocalAuthError) as other:
        auth.verify_request(forged, "POST", "/x", b"", "10.0.0.10")
    assert other.value.reason != "locked_out"
    now[0] += la.LOCKOUT_BASE_S + 1
    challenge = auth.issue_challenge()
    good = la.format_header(key.client_id, challenge,
                            key.sign(la.request_message(DEVICE, key.client_id, challenge, "POST", "/x", b"")))
    assert auth.verify_request(good, "POST", "/x", b"", "10.0.0.9").client_id == key.client_id


def test_a_phone_idle_past_the_expiry_must_enrol_again(tmp_path, authority):
    wall = [1_000_000.0]
    store = la.ClientStore(tmp_path / "c.json")
    auth = la.LocalAuthority(DEVICE, authority.identity, store, idle_expiry_days=30, wall=lambda: wall[0])
    key = PhoneKey()
    auth.enroll(la.b64u(key.public_key), "p", "ble")
    wall[0] += 31 * 86400
    challenge = auth.issue_challenge()
    header = la.format_header(key.client_id, challenge,
                              key.sign(la.request_message(DEVICE, key.client_id, challenge, "POST", "/x", b"")))
    with pytest.raises(la.LocalAuthError) as exc:
        auth.verify_request(header, "POST", "/x", b"", "p")
    assert exc.value.reason == "unknown_client"
    assert la.ClientStore(tmp_path / "c.json").clients == {}  # persisted


def test_enrolled_phones_survive_a_restart_and_a_tampered_row_is_dropped(tmp_path, authority):
    path = tmp_path / "clients.json"
    auth = la.LocalAuthority(DEVICE, authority.identity, la.ClientStore(path))
    a, b = PhoneKey(), PhoneKey()
    auth.enroll(la.b64u(a.public_key), "A", "ble")
    auth.enroll(la.b64u(b.public_key), "B", "account")
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    doc = json.loads(path.read_text())
    doc["clients"][1]["client_id"] = a.client_id  # B's key under A's id
    path.write_text(json.dumps(doc))
    os.chmod(path, 0o600)
    reloaded = la.ClientStore(path)
    assert set(reloaded.clients) == {a.client_id}


def test_a_loosely_permissioned_store_trusts_no_phone(tmp_path, authority):
    path = tmp_path / "clients.json"
    auth = la.LocalAuthority(DEVICE, authority.identity, la.ClientStore(path))
    auth.enroll(la.b64u(PhoneKey().public_key), "A", "ble")
    os.chmod(path, 0o644)
    loose = la.ClientStore(path)
    assert loose.clients == {} and loose.fault is not None


def test_the_device_identity_is_private_and_a_loose_one_is_refused_not_replaced(tmp_path):
    path = tmp_path / "id.pem"
    first = la.DeviceIdentity.load_or_create(path)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert la.DeviceIdentity.load_or_create(path).public_key == first.public_key
    os.chmod(path, 0o640)
    with pytest.raises(la.LocalAuthError) as exc:
        la.DeviceIdentity.load_or_create(path)
    assert exc.value.reason == "identity_unusable"
    assert "PRIVATE" in path.read_text()  # untouched
    assert first.fingerprint in repr(first) and "PRIVATE" not in repr(first)


def test_build_authority_turns_local_access_off_with_a_reason_not_a_crash(tmp_path):
    (tmp_path / "local_identity.pem").write_text("junk")
    os.chmod(tmp_path / "local_identity.pem", 0o600)
    authority, why = la.build_authority(DEVICE, {}, tmp_path)
    assert authority is None and "unusable" in why
    off, why_off = la.build_authority(DEVICE, {"enabled": False}, tmp_path)
    assert off is None and "turned off" in why_off


def test_the_store_caps_phones(authority):
    for _ in range(la.MAX_CLIENTS):
        authority.enroll(la.b64u(PhoneKey().public_key), "p", "ble")
    with pytest.raises(la.LocalAuthError) as exc:
        authority.enroll(la.b64u(PhoneKey().public_key), "p", "ble")
    assert exc.value.reason == "too_many_clients"


def test_invalid_public_keys_are_refused(authority):
    for bad in ("", "AAAA", la.b64u(b"\x04" + bytes(64)), la.b64u(b"\x02" + bytes(32))):
        with pytest.raises(la.LocalAuthError):
            authority.enroll(bad, "p", "ble")


# ------------------------------------------------------------------ the legacy token keeps working
def test_the_legacy_token_still_opens_everything_and_can_enrol_a_phone(authority):
    server = DashboardServer(host="0.0.0.0", port=0, token="legacy-token-1234567890",  # noqa: S104 - LAN bind under test, test-only token
                             presets={"driver_safety": {"module": "driver_attention"}}, local_authority=authority)
    url = server.start()
    try:
        phone = LocalPhone(url)
        assert phone.call("GET", "/api/state", headers={"X-Device-Token": "legacy-token-1234567890"}).status == 200
        assert phone.enroll_with_token("wrong").status == 401
        reply = phone.enroll_with_token("legacy-token-1234567890")
        assert reply.status == 201 and reply.body["client_id"] == phone.key.client_id
        assert authority.store.clients[phone.key.client_id].source == "token"
        assert phone.open_session().status == 200
        assert phone.hello()["token_enrollment"] is True
    finally:
        server.close()


def test_without_an_authority_the_dashboard_behaves_exactly_as_before():
    server = DashboardServer(host="0.0.0.0", port=0, token=None)  # noqa: S104 - the LAN bind under test
    url = server.start()
    try:
        phone = LocalPhone(url)
        assert phone.call("GET", "/api/local/hello").status == 404
        assert phone.call("GET", "/api/state").status == 401
    finally:
        server.close()


def test_nothing_secret_appears_in_the_report_or_the_client_view(authority, tmp_path):
    key = PhoneKey()
    authority.enroll(la.b64u(key.public_key), "Owner", "ble")
    text = json.dumps({"report": authority.report(), "view": authority.clients_view()})
    pem = Path(tmp_path / "local_identity.pem").read_text()
    assert "PRIVATE" not in text and pem.splitlines()[1] not in text
    assert key.client_id in text
