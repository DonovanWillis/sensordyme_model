"""sd-auth-v1 (core/command_auth.py): key derivation, request and command tags, refusals."""

from __future__ import annotations

import copy

import pytest

from core import command_auth as a

KEY = "dk_" + "11" * 24
OTHER = "dk_" + "22" * 24
NONCE = "ab" * 16
COMMAND = {"set_protection": {"action": "stop", "confirm": True, "id": "c0ffee"},
           "set_zones": [{"name": "door", "points": [[0.1, 0.25], [0.5, 0.1], [0.5, 0.5]]}],
           "zones_version": "z1"}


def _ring(*secrets, revoked=()):
    keys = [a.DeviceKey(s) for s in secrets]
    return a.Keyring(keys[0], keys[1] if len(keys) > 1 else None, frozenset(revoked))


def _signed(command=COMMAND, key=KEY, device="SD-0A1B2C3D", rn=NONCE):
    return a.sign_command(a.DeviceKey(key), command, device=device, request_nonce=rn)


def _verify(envelope, command=COMMAND, ring=None, devices=("SD-0A1B2C3D",), rn=NONCE):
    return a.verify_command(ring or _ring(KEY), command, envelope, devices=devices, request_nonce=rn)


# ------------------------------------------------------------------ HKDF (RFC 5869 appendix A)
def test_hkdf_matches_rfc5869_test_case_1():
    okm = a.hkdf_sha256(bytes([0x0b] * 22), bytes(range(0x0d)), bytes(range(0xf0, 0xfa)), 42)
    assert okm.hex() == ("3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf"
                         "34007208d5b887185865")


def test_hkdf_matches_rfc5869_test_case_3_empty_salt_and_info():
    okm = a.hkdf_sha256(bytes([0x0b] * 22), b"", b"", 42)
    assert okm.hex() == ("8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d"
                         "9d201395faa4b61a96c8")


def test_derived_keys_are_distinct_per_purpose_and_never_printed():
    key = a.DeviceKey(KEY)
    assert len({key._request, key._command, bytes.fromhex(key.kid)}) == 3
    assert key._request != KEY.encode() and key._command != KEY.encode()
    for text in (repr(key), str(key), f"{key}"):
        assert KEY not in text and key._request.hex() not in text and key.kid in text
    assert a.key_id(KEY) == key.kid and a.key_id(OTHER) != key.kid


def test_a_short_key_cannot_derive_keys():
    with pytest.raises(ValueError, match="at least 16 bytes"):
        a.DeviceKey("dk_test")


# ------------------------------------------------------------------ commands
def test_a_signed_command_verifies():
    envelope = _signed()
    assert _verify(envelope)["cid"] == envelope["cid"]
    assert envelope["types"] == ["set_protection", "set_zones"]
    assert KEY not in str(envelope)


def test_key_order_and_spacing_do_not_matter():
    reordered = {"zones_version": "z1", "set_zones": COMMAND["set_zones"], "set_protection":
                 {"id": "c0ffee", "confirm": True, "action": "stop"}}
    _verify(_signed(), command=reordered)


@pytest.mark.parametrize(("tamper", "reason"), [
    (lambda c: c["set_protection"].update(action="resume"), "body_mismatch"),
    (lambda c: c["set_zones"][0]["points"][0].__setitem__(0, 0.9), "body_mismatch"),
    (lambda c: c.update(zones_version="z2"), "body_mismatch"),
    (lambda c: c.update(set_preset="driver_safety"), "type_mismatch"),
    (lambda c: c.pop("set_zones"), "type_mismatch"),
])
def test_command_substitution_is_refused(tamper, reason):
    envelope = _signed()
    command = copy.deepcopy(COMMAND)
    tamper(command)
    with pytest.raises(a.AuthError) as err:
        _verify(envelope, command=command)
    assert err.value.reason == reason


@pytest.mark.parametrize("field", ["kid", "dev", "cid", "nonce", "rn", "iat", "body_sha256", "types"])
def test_every_bound_field_is_covered_by_the_tag(field):
    envelope = _signed()
    forged = dict(envelope)
    forged[field] = {
        "kid": a.key_id(OTHER), "dev": "SD-FFFFFFFF", "cid": "ee" * 12, "nonce": "cd" * 16,
        "rn": "ef" * 16, "iat": envelope["iat"] + 1, "body_sha256": "00" * 32, "types": ["set_preset"],
    }[field]
    with pytest.raises(a.AuthError) as err:
        _verify(forged, ring=_ring(KEY, OTHER), devices=("SD-0A1B2C3D", "SD-FFFFFFFF"), rn=forged["rn"])
    assert err.value.reason == "bad_signature"


def test_a_command_addressed_to_another_device_is_refused():
    with pytest.raises(a.AuthError) as err:
        _verify(_signed(device="SD-FFFFFFFF"))
    assert err.value.reason == "wrong_device"


def test_a_response_to_an_earlier_request_is_refused_without_any_clock():
    with pytest.raises(a.AuthError) as err:
        _verify(_signed(rn="00" * 16))
    assert err.value.reason == "stale_response"


def test_another_devices_key_is_unknown():
    with pytest.raises(a.AuthError) as err:
        _verify(_signed(key=OTHER))
    assert err.value.reason == "unknown_key"


def test_rotation_accepts_the_previous_key_until_it_is_revoked():
    old_envelope = _signed(key=OTHER)
    _verify(old_envelope, ring=_ring(KEY, OTHER))
    with pytest.raises(a.AuthError) as err:
        _verify(old_envelope, ring=_ring(KEY, OTHER, revoked={a.key_id(OTHER)}))
    assert err.value.reason == "revoked_key"


@pytest.mark.parametrize(("envelope", "reason"), [
    (None, "malformed"), ([], "malformed"), ("sig", "malformed"),
    ({"v": "sd-auth-v0"}, "unsupported_version"), ({"v": "sd-auth-v2"}, "unsupported_version"),
    ({"v": a.CONTRACT}, "malformed"),
])
def test_malformed_and_unknown_versions_are_refused(envelope, reason):
    with pytest.raises(a.AuthError) as err:
        _verify(envelope)
    assert err.value.reason == reason


@pytest.mark.parametrize(("field", "value"), [
    ("kid", "XYZ"), ("sig", "00"), ("iat", "1"), ("iat", True), ("iat", -1), ("types", "set_preset"),
    ("types", ["set preset\n"]), ("dev", "SD-0A1B2C3D\nkid"), ("rn", NONCE.upper()), ("body_sha256", "g" * 64),
])
def test_malformed_fields_are_refused_before_the_tag(field, value):
    envelope = dict(_signed())
    envelope[field] = value
    with pytest.raises(a.AuthError) as err:
        _verify(envelope)
    assert err.value.reason == "malformed"


def test_a_non_mapping_command_is_refused():
    with pytest.raises(a.AuthError):
        _verify(_signed(), command=["set_preset"])


# ------------------------------------------------------------------ requests
def _request(body=b'{"events":[]}', ts=1_800_000_000, nonce=NONCE, key=KEY):
    return a.parse_request_header(a.sign_request(a.DeviceKey(key), "POST", "/v1/ingest", body, nonce=nonce,
                                                 issued_at=ts))


def test_a_signed_request_verifies_once():
    nonces = a.NonceCache()
    auth = _request()
    assert a.verify_request(_ring(KEY), auth, "POST", "/v1/ingest", b'{"events":[]}', nonces=nonces,
                            now=1_800_000_010).kid == a.key_id(KEY)
    with pytest.raises(a.AuthError) as err:
        a.verify_request(_ring(KEY), auth, "POST", "/v1/ingest", b'{"events":[]}', nonces=nonces,
                         now=1_800_000_011)
    assert err.value.reason == "replayed_nonce"


@pytest.mark.parametrize(("method", "path", "body"), [
    ("GET", "/v1/ingest", b'{"events":[]}'), ("POST", "/v1/other", b'{"events":[]}'),
    ("POST", "/v1/ingest", b'{"events":[1]}'),
])
def test_a_request_tag_covers_method_path_and_body(method, path, body):
    with pytest.raises(a.AuthError) as err:
        a.verify_request(_ring(KEY), _request(), method, path, body, nonces=a.NonceCache(), now=1_800_000_000)
    assert err.value.reason == "bad_signature"


@pytest.mark.parametrize("offset", [-301, 301, -10 ** 6])
def test_a_request_outside_the_clock_window_is_refused(offset):
    with pytest.raises(a.AuthError) as err:
        a.verify_request(_ring(KEY), _request(), "POST", "/v1/ingest", b'{"events":[]}', nonces=a.NonceCache(),
                         now=1_800_000_000 - offset)
    assert err.value.reason == "clock_skew"


@pytest.mark.parametrize(("header", "reason"), [
    ("", "malformed"), ("Bearer dk_x", "malformed"),
    ("sd-auth-v2 kid=0123456789abcdef,ts=1,nonce=" + NONCE + ",sig=" + "0" * 64, "unsupported_version"),
    ("sd-auth-v1 kid=0123456789abcdef,ts=1,nonce=" + NONCE + ",sig=" + "0" * 63, "malformed"),
    ("sd-auth-v1 kid=0123456789abcdef,ts=1,nonce=" + NONCE + ",sig=" + "0" * 64 + "\n", "malformed"),
])
def test_bad_request_headers_are_refused(header, reason):
    with pytest.raises(a.AuthError) as err:
        a.parse_request_header(header)
    assert err.value.reason == reason


def test_the_nonce_cache_is_bounded_and_expires():
    cache = a.NonceCache(max_entries=3, ttl_s=10)
    assert all(cache.add(f"{i:032x}", now=0) for i in range(5))
    assert len(cache._seen) == 3
    assert not cache.add(f"{4:032x}", now=1)
    assert cache.add(f"{4:032x}", now=20), "expired after its ttl"


def test_error_messages_never_carry_key_material():
    envelope = _signed()
    envelope["sig"] = "00" * 32
    with pytest.raises(a.AuthError) as err:
        _verify(envelope)
    derived = a.DeviceKey(KEY)
    for secret in (KEY, derived._command.hex(), derived._request.hex(), _signed()["sig"]):
        assert secret not in str(err.value)
