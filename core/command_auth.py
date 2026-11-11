"""Device <-> backend authentication, contract sd-auth-v1 (docs/DEVICE_AUTH_CONTRACT.md).

Shared by the backend (cloud/app.py: verifies device requests, signs commands)
and the device (core/uplink.py: signs requests, verifies commands). Standard
library only: HKDF-SHA256 (RFC 5869) over hmac, HMAC-SHA256 tags compared with
hmac.compare_digest. No home-made primitives.

The per-device root secret is the existing device key (`cloud.device_key`). It
is never sent in `required` mode and never used directly as a MAC key: separate
keys are derived for each purpose, plus a public key id (`kid`) that names the
key in headers and logs without revealing it.

Two messages carry a tag:

* request (device -> backend), header `X-SD-Auth`: binds key id, method, path,
  the SHA-256 of the exact body bytes, a fresh request nonce and the device's
  issued-at time. The backend checks the time against its own clock and keeps
  a bounded nonce cache, so a captured request cannot be replayed.
* command envelope (backend -> device), `command_auth` beside `command` in the
  ingest response: binds key id, the device it is addressed to, the command
  types, the SHA-256 of the canonical command body, a command id, a response
  nonce, the issued-at time and **the nonce of the request it answers**. That
  last binding is the replay defence on the device, and it needs no clock: an
  old envelope answers an old request, never the one the device just sent.

Nothing here logs. Rejections are reported as reason codes; key material and
tags never appear in an exception message or a repr.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import time
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Optional

CONTRACT = "sd-auth-v1"

#: Modes for `cloud.command_auth` (device) and SENSORDYME_DEVICE_AUTH (backend).
#: off        -- legacy only: bearer key, unsigned commands. Refused by the deploy preflight.
#: permissive -- legacy auth still works; signed envelopes are verified when present, and an
#:               invalid one is never executed. A backend that does not sign is still obeyed.
#: required   -- device: signed requests only (the key never travels), unsigned or invalid
#:               commands refused. Backend: bearer-key requests refused.
MODES = ("off", "permissive", "required")
DEFAULT_MODE = "permissive"

#: Largest |request issued-at - backend clock| the backend accepts.
CLOCK_SKEW_S = 300
#: Request nonces the backend remembers (each for 2 * CLOCK_SKEW_S).
NONCE_CACHE_MAX = 50_000

_SALT = b"sensordyme/" + CONTRACT.encode()
_HEX = re.compile(r"^[0-9a-f]+$")
_KID = re.compile(r"^[0-9a-f]{16}$")
_NONCE = re.compile(r"^[0-9a-f]{32}$")
_CID = re.compile(r"^[0-9a-f]{24}$")
_SIG = re.compile(r"^[0-9a-f]{64}$")
_DEV = re.compile(r"^(SD-(SIM-)?[0-9A-F]{8}|kid:[0-9a-f]{16})$")
_TYPE = re.compile(r"^[a-z_]{1,32}$")
_PATH = re.compile(r"^/[A-Za-z0-9/_.-]{0,255}$")
_HEADER = re.compile(
    r"^" + re.escape(CONTRACT) + r" kid=([0-9a-f]{16}),ts=(\d{1,12}),nonce=([0-9a-f]{32}),sig=([0-9a-f]{64})$"
)


class AuthError(ValueError):
    """A message failed verification. `reason` is a stable code, safe to log."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        super().__init__(f"{reason}: {detail}" if detail else reason)


# ------------------------------------------------------------------ keys
def hkdf_sha256(ikm: bytes, salt: bytes, info: bytes, length: int = 32) -> bytes:
    """HKDF-SHA256, RFC 5869 (extract, then expand)."""
    if length > 255 * 32:
        raise ValueError("HKDF output too long")
    prk = hmac.new(salt or b"\x00" * 32, ikm, hashlib.sha256).digest()
    okm, block = b"", b""
    for counter in range(1, -(-length // 32) + 1):
        block = hmac.new(prk, block + info + bytes([counter]), hashlib.sha256).digest()
        okm += block
    return okm[:length]


class DeviceKey:
    """One device key and what is derived from it. The repr names only the key id."""

    __slots__ = ("_command", "_request", "kid")

    def __init__(self, secret: str) -> None:
        raw = str(secret or "").encode()
        if len(raw) < 16:
            raise ValueError("a device key must be at least 16 bytes")
        self.kid = hkdf_sha256(raw, _SALT, b"key-id", 8).hex()
        self._request = hkdf_sha256(raw, _SALT, b"request-mac", 32)
        self._command = hkdf_sha256(raw, _SALT, b"command-mac", 32)

    def __repr__(self) -> str:
        return f"DeviceKey(kid={self.kid})"

    __str__ = __repr__


@dataclass
class Keyring:
    """The keys one side holds for one device: current, an optional previous one
    (accepted for verification during a rotation), and revoked key ids."""

    current: DeviceKey
    previous: Optional[DeviceKey] = None
    revoked: frozenset = field(default_factory=frozenset)

    @classmethod
    def from_config(cls, cloud: dict) -> "Keyring":
        previous = cloud.get("previous_device_key") or None
        revoked = cloud.get("revoked_key_ids") or ()
        return cls(DeviceKey(cloud.get("device_key", "")),
                   DeviceKey(previous) if previous else None,
                   frozenset(str(k).lower() for k in revoked))

    def for_kid(self, kid: str) -> DeviceKey:
        if kid in self.revoked:
            raise AuthError("revoked_key", kid)
        for key in (self.current, self.previous):
            if key is not None and key.kid == kid:
                return key
        raise AuthError("unknown_key", kid)


def key_id(secret: str) -> str:
    """Public key id of a device key (16 hex). Safe to store, log and index."""
    return DeviceKey(secret).kid


# ------------------------------------------------------------------ encoding
def canonical_json(value) -> bytes:
    """The bytes a command body is hashed as. Both sides recompute it from the
    parsed object, so transport formatting (spacing, key order) does not matter."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def new_nonce() -> str:
    return secrets.token_hex(16)


def _mac(key: bytes, fields: Iterable[str]) -> str:
    return hmac.new(key, "\n".join(fields).encode(), hashlib.sha256).hexdigest()


# ------------------------------------------------------------------ requests (device -> backend)
def sign_request(key: DeviceKey, method: str, path: str, body: bytes, *, nonce: str,
                 issued_at: Optional[int] = None) -> str:
    """The X-SD-Auth header value for one request."""
    ts = int(time.time() if issued_at is None else issued_at)
    if not _PATH.fullmatch(path) or not _NONCE.fullmatch(nonce):
        raise ValueError("bad path or nonce")
    sig = _mac(key._request, (CONTRACT, "request", key.kid, method.upper(), path, sha256_hex(body),
                              nonce, str(ts)))
    return f"{CONTRACT} kid={key.kid},ts={ts},nonce={nonce},sig={sig}"


@dataclass(frozen=True)
class RequestAuth:
    kid: str
    issued_at: int
    nonce: str
    sig: str


def parse_request_header(header: str) -> RequestAuth:
    """Parse X-SD-Auth strictly. Raises AuthError (unsupported_version / malformed)."""
    text = str(header or "")
    if not text.startswith(CONTRACT + " "):
        raise AuthError("unsupported_version" if " kid=" in text else "malformed")
    match = _HEADER.fullmatch(text)
    if not match:
        raise AuthError("malformed")
    kid, ts, nonce, sig = match.groups()
    return RequestAuth(kid, int(ts), nonce, sig)


class NonceCache:
    """Bounded memory of recently seen nonces, each kept for `ttl_s`."""

    def __init__(self, max_entries: int = NONCE_CACHE_MAX, ttl_s: float = 2 * CLOCK_SKEW_S) -> None:
        self._seen: OrderedDict[str, float] = OrderedDict()
        self._max, self._ttl = max_entries, ttl_s

    def add(self, nonce: str, now: float) -> bool:
        """Record `nonce`; False if it was already seen (a replay)."""
        while self._seen and (next(iter(self._seen.values())) < now - self._ttl or len(self._seen) >= self._max):
            self._seen.popitem(last=False)
        if nonce in self._seen:
            return False
        self._seen[nonce] = now
        return True


def verify_request(keyring: Keyring, auth: RequestAuth, method: str, path: str, body: bytes, *,
                   nonces: NonceCache, now: Optional[float] = None,
                   skew_s: float = CLOCK_SKEW_S) -> DeviceKey:
    """Check one signed request; returns the key that signed it (for the reply)."""
    now = time.time() if now is None else now
    key = keyring.for_kid(auth.kid)
    expected = _mac(key._request, (CONTRACT, "request", key.kid, method.upper(), path, sha256_hex(body),
                                   auth.nonce, str(auth.issued_at)))
    if not hmac.compare_digest(expected, auth.sig):
        raise AuthError("bad_signature")
    if abs(auth.issued_at - now) > skew_s:
        raise AuthError("clock_skew", f"device clock off by {int(auth.issued_at - now)} s")
    if not nonces.add(auth.nonce, now):
        raise AuthError("replayed_nonce")
    return key


# ------------------------------------------------------------------ commands (backend -> device)
def command_types(command: dict) -> list[str]:
    return sorted(k for k in command if k.startswith("set_"))


def sign_command(key: DeviceKey, command: dict, *, device: str, request_nonce: str,
                 issued_at: Optional[int] = None, command_id: Optional[str] = None,
                 nonce: Optional[str] = None) -> dict:
    """The `command_auth` envelope for `command`, addressed to `device` and bound
    to the request (by its nonce) that this response answers."""
    if not _DEV.fullmatch(device) or not _NONCE.fullmatch(request_nonce):
        raise ValueError("bad device id or request nonce")
    envelope = {
        "v": CONTRACT,
        "kid": key.kid,
        "dev": device,
        "types": command_types(command),
        "body_sha256": sha256_hex(canonical_json(command)),
        "cid": command_id or secrets.token_hex(12),
        "nonce": nonce or new_nonce(),
        "rn": request_nonce,
        "iat": int(time.time() if issued_at is None else issued_at),
    }
    envelope["sig"] = _mac(key._command, _command_fields(envelope))
    return envelope


def _command_fields(envelope: dict) -> tuple:
    return (CONTRACT, "command", envelope["kid"], envelope["dev"], ",".join(envelope["types"]),
            envelope["body_sha256"], envelope["cid"], envelope["nonce"], envelope["rn"], str(envelope["iat"]))


def _well_formed_envelope(envelope) -> dict:
    if not isinstance(envelope, dict):
        raise AuthError("malformed", "envelope is not an object")
    if envelope.get("v") != CONTRACT:
        raise AuthError("unsupported_version", str(envelope.get("v"))[:32])
    checks = (("kid", _KID), ("dev", _DEV), ("cid", _CID), ("nonce", _NONCE), ("rn", _NONCE), ("sig", _SIG))
    for name, pattern in checks:
        if not isinstance(envelope.get(name), str) or not pattern.fullmatch(envelope[name]):
            raise AuthError("malformed", name)
    body_hash = envelope.get("body_sha256")
    if not isinstance(body_hash, str) or len(body_hash) != 64 or not _HEX.fullmatch(body_hash):
        raise AuthError("malformed", "body_sha256")
    types = envelope.get("types")
    if not isinstance(types, list) or len(types) > 16 or not all(isinstance(t, str) and _TYPE.fullmatch(t) for t in types):
        raise AuthError("malformed", "types")
    iat = envelope.get("iat")
    if not isinstance(iat, int) or isinstance(iat, bool) or not 0 <= iat < 10 ** 12:
        raise AuthError("malformed", "iat")
    return envelope


def verify_command(keyring: Keyring, command, envelope, *, devices: Iterable[str],
                   request_nonce: str) -> dict:
    """Check the envelope of one received command. Returns the envelope.

    `devices`: the ids this device answers to (its hardware id as sent in this
    request's health, and "kid:<its kid>"). `request_nonce`: the nonce of the
    request this response answers. Order: shape, version, key, tag, then the
    bound fields, so nothing unauthenticated is interpreted beyond its shape.
    """
    env = _well_formed_envelope(envelope)
    if not isinstance(command, dict):
        raise AuthError("malformed", "command is not an object")
    key = keyring.for_kid(env["kid"])
    if not hmac.compare_digest(_mac(key._command, _command_fields(env)), env["sig"]):
        raise AuthError("bad_signature")
    if env["dev"] not in set(devices):
        raise AuthError("wrong_device", env["dev"])
    if not hmac.compare_digest(env["rn"], request_nonce):
        raise AuthError("stale_response")
    if env["types"] != command_types(command):
        raise AuthError("type_mismatch")
    try:
        body = canonical_json(command)
    except (TypeError, ValueError) as exc:
        raise AuthError("malformed", "command is not canonical JSON") from exc
    if not hmac.compare_digest(sha256_hex(body), env["body_sha256"]):
        raise AuthError("body_mismatch")
    return env
