"""A reference sd-local-v1 phone, in Python (docs/LOCAL_ACCESS.md).

What the iOS app's `LocalDeviceClient` does, written against the same contract,
so the device side is tested end to end without a phone and the canonical
strings have a second implementation to agree with. Test and simulation use
only: the key lives in memory.
"""

from __future__ import annotations

import json
import secrets
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

from core import local_auth as la


@dataclass
class Reply:
    status: int
    body: Any
    headers: dict


class PhoneKey:
    """A P-256 key pair, as the phone's Keychain / Secure Enclave holds it."""

    def __init__(self, private_key=None) -> None:
        ec, _hashes, serialization, *_ = la._ec()
        self._key = private_key or ec.generate_private_key(ec.SECP256R1())
        self.public_key = self._key.public_key().public_bytes(
            serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
        self.client_id = la.client_id_for(self.public_key)

    @classmethod
    def from_scalar(cls, scalar: int) -> PhoneKey:
        ec, *_ = la._ec()
        return cls(ec.derive_private_key(scalar, ec.SECP256R1()))

    def sign(self, message: bytes) -> bytes:
        ec, hashes, _, _, _, decode = la._ec()
        r, s = decode(self._key.sign(message, ec.ECDSA(hashes.SHA256())))
        return r.to_bytes(32, "big") + s.to_bytes(32, "big")


class LocalPhone:
    """Talks sd-local-v1 to one device URL."""

    def __init__(self, base_url: str, key: Optional[PhoneKey] = None, name: str = "Test iPhone") -> None:
        self.base_url = base_url.rstrip("/")
        self.key = key or PhoneKey()
        self.name = name
        self.device_public_key: Optional[bytes] = None
        self.device_id: Optional[str] = None
        self.session: Optional[str] = None

    # -- transport -----------------------------------------------------------------
    def call(self, method: str, path: str, body: Optional[dict] = None, headers: Optional[dict] = None,
             raw: Optional[bytes] = None, timeout: float = 5.0) -> Reply:
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(self.base_url + path, data=data, method=method,  # noqa: S310 - test client, http only
                                     headers=dict(headers or {}))
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - test client, local URL
                text = resp.read()
                return Reply(resp.status, _maybe_json(text), dict(resp.headers))
        except urllib.error.HTTPError as exc:
            return Reply(exc.code, _maybe_json(exc.read()), dict(exc.headers))

    # -- protocol --------------------------------------------------------------------
    def hello(self) -> dict:
        reply = self.call("GET", "/api/local/hello")
        assert reply.status == 200, reply
        return reply.body

    def signed(self, method: str, path: str, body: Optional[dict] = None, *, challenge: Optional[str] = None,
               raw: Optional[bytes] = None) -> Reply:
        hello = self.hello()
        device_id = hello["device_id"]
        challenge = challenge or hello["challenge"]
        data = raw if raw is not None else json.dumps(body if body is not None else {}).encode()
        message = la.request_message(device_id, self.key.client_id, challenge, method, path, data)
        header = la.format_header(self.key.client_id, challenge, self.key.sign(message))
        return self.call(method, path, raw=data, headers={la.HEADER: header})

    def enroll_with_token(self, token: str) -> Reply:
        reply = self.call("POST", "/api/local/enroll",
                          body={"public_key": la.b64u(self.key.public_key), "name": self.name},
                          headers={"X-Device-Token": token})
        if reply.status == 201:
            self.trust(reply.body)
        return reply

    def trust(self, enrollment: dict) -> None:
        """Remember the device's key from an authorised enrolment (never from hello)."""
        self.device_public_key = la.b64u_decode(enrollment["device_public_key"], 65)
        self.device_id = enrollment["device_id"]

    def open_session(self) -> Reply:
        nonce = secrets.token_hex(16)
        hello = self.hello()
        body = json.dumps({"nonce": nonce}).encode()
        message = la.request_message(hello["device_id"], self.key.client_id, hello["challenge"], "POST",
                                     "/api/local/session", body)
        header = la.format_header(self.key.client_id, hello["challenge"], self.key.sign(message))
        reply = self.call("POST", "/api/local/session", raw=body, headers={la.HEADER: header})
        if reply.status == 200:
            if self.device_public_key is None:
                raise AssertionError("a session from a device this phone never enrolled with")
            signed = la.session_message(hello["device_id"], self.key.client_id, hello["challenge"], nonce,
                                        reply.body["session"], reply.body["expires_in_s"])
            sig = la.b64u_decode(reply.body["device_sig"], 64)
            if sig is None or not la.verify_signature(self.device_public_key, signed, sig):
                raise AssertionError("the device's session signature does not verify")
            self.session = reply.body["session"]
        return reply

    def get(self, path: str) -> Reply:
        return self.call("GET", path, headers={la.SESSION_HEADER: self.session or ""})


def _maybe_json(raw: bytes) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        return raw
