"""Per-phone local access to the device, contract sd-local-v1 (docs/LOCAL_ACCESS.md).

The problem it solves (docs/PORTABLE_OPERATION.md C1-C3): the device already
serves its live feed, state, history and stop/resume/preset on its own LAN
port, but the only credential was ONE shared `dashboard.token` that a person
had to copy from config.yaml into the app. That token never expires, cannot be
revoked for one lost phone, and the shipped config leaves it unset -- so a new
customer's camera refused the app entirely.

sd-local-v1 replaces "paste a shared secret" with a credential per phone:

* Each phone holds a P-256 signing key that never leaves it (the Secure
  Enclave where the phone has one). The device stores only the PUBLIC key,
  under a `client_id` that is the first 16 hex of SHA-256(public key) -- an id
  nobody can choose for a key they do not hold.
* A phone is ENROLLED by one of three authorised paths, never by being on the
  network:
    1. Bluetooth, inside a proven BLE PoP session (the phone proved the setup
       code printed on the device's label; services/ble_pop.py `enroll`);
    2. the owner's backend account, as a SIGNED sd-auth-v1 command
       (`set_local_clients`, core/uplink.py): the backend relays public keys
       only, so it never learns anything that would let it act locally;
    3. a holder of the legacy `dashboard.token` (`POST /api/local/enroll`),
       which is how an install that already had a token migrates.
* The device has its own P-256 identity key (generated once, 0600). The phone
  learns its public key at enrolment, over the same authorised path, and from
  then on checks the device's signature on every session it opens, so a
  look-alike on the LAN cannot pose as the camera.
* Replay defence needs no clock: the device issues single-use challenges
  (`GET /api/local/hello`), and every signature covers one. A Pi with no RTC
  and no NTP still refuses a replayed command.
* Reads (state, health, history, events, the stream) use a short-lived bearer
  SESSION the phone opens with a signed challenge; it is bound to the client and
  the peer address and dies when the client is revoked. Every write (stop,
  resume, preset, revoke) is signed individually over a fresh challenge -- a
  sniffed session can watch for at most its lifetime, never command.

What it does NOT do (stated in docs/LOCAL_ACCESS.md): the transport is plain
HTTP on the LAN, so a session and the video can be sniffed by someone on the
same network for the session's lifetime; commands cannot be forged or replayed.

Primitives: ECDSA P-256 / SHA-256 from `cryptography` (already in the device set
for BLE PoP), SHA-256, `secrets`. Nothing here logs key material, signatures or
sessions; refusals are stable reason codes.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.config_lkg import atomic_write_bytes

CONTRACT = "sd-local-v1"
#: Seconds a challenge from /api/local/hello can be used (once).
CHALLENGE_TTL_S = 60.0
#: Challenges outstanding at once; the oldest is dropped beyond this.
MAX_CHALLENGES = 256
#: Seconds a session lives. The app renews before it ends.
SESSION_TTL_S = 15 * 60.0
MAX_SESSIONS = 64
#: Phones enrolled at once. A household has a few; a cap bounds the store.
MAX_CLIENTS = 16
#: A phone not seen for this many days must enrol again (0 disables).
DEFAULT_IDLE_EXPIRY_DAYS = 180
#: `last_seen` reaches the SD card at most this often per phone.
LAST_SEEN_PERSIST_S = 3600.0
#: Failed signatures from one peer within FAIL_WINDOW_S before it is locked out.
FAIL_LIMIT = 10
FAIL_WINDOW_S = 60.0
LOCKOUT_BASE_S = 60.0
LOCKOUT_MAX_S = 3600.0
MAX_NAME = 64

HEADER = "X-SD-Local"
SESSION_HEADER = "X-SD-Session"

_CHALLENGE = re.compile(r"^[0-9a-f]{32}$")
_CLIENT_ID = re.compile(r"^[0-9a-f]{16}$")
_NONCE = re.compile(r"^[0-9a-f]{32}$")
_HEADER_RE = re.compile(
    r"^" + re.escape(CONTRACT) + r" client=([0-9a-f]{16}),challenge=([0-9a-f]{32}),sig=([A-Za-z0-9_-]{86})$"
)
_SOURCES = ("ble", "account", "token")


class LocalAuthError(Exception):
    """A refused local request. `reason` is a stable code, safe to log and to return."""

    def __init__(self, reason: str, status: int = 401, retry_after_s: Optional[float] = None) -> None:
        self.reason = reason
        self.status = status
        self.retry_after_s = retry_after_s
        super().__init__(reason)


# ---------------------------------------------------------------------- encoding
def b64u(data: bytes) -> str:
    """URL-safe base64 without padding (what the phone sends and expects)."""
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def b64u_decode(value: Any, length: Optional[int] = None) -> Optional[bytes]:
    """Bytes from URL-safe base64 (padding optional), or None for anything else."""
    if not isinstance(value, str) or not value or len(value) > 512:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_-]+={0,2}", value):
        return None
    try:
        raw = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (binascii.Error, ValueError):
        return None
    if length is not None and len(raw) != length:
        return None
    return raw


def client_id_for(public_key: bytes) -> str:
    """The id a public key is enrolled under: 16 hex of SHA-256(X9.63 point)."""
    return hashlib.sha256(public_key).hexdigest()[:16]


def key_fingerprint(public_key: bytes) -> str:
    """A short, shareable name for a public key (never the key itself)."""
    return hashlib.sha256(public_key).hexdigest()[:16]


def body_digest(body: bytes) -> str:
    return hashlib.sha256(body or b"").hexdigest()


def request_message(device_id: str, client_id: str, challenge: str, method: str, path: str, body: bytes) -> bytes:
    """What a phone signs for one request (docs/LOCAL_ACCESS.md §3)."""
    return "\n".join((CONTRACT, "request", device_id, client_id, challenge, method.upper(), path,
                      body_digest(body))).encode("utf-8")


def session_message(device_id: str, client_id: str, challenge: str, client_nonce: str, session: str,
                    expires_in_s: int) -> bytes:
    """What the DEVICE signs when it opens a session, so the phone knows who answered."""
    return "\n".join((CONTRACT, "session", device_id, client_id, challenge, client_nonce,
                      hashlib.sha256(session.encode("ascii")).hexdigest(), str(int(expires_in_s)))).encode("utf-8")


def format_header(client_id: str, challenge: str, signature: bytes) -> str:
    return f"{CONTRACT} client={client_id},challenge={challenge},sig={b64u(signature)}"


def parse_header(value: Any) -> tuple[str, str, bytes]:
    if not isinstance(value, str):
        raise LocalAuthError("missing_signature")
    match = _HEADER_RE.match(value.strip())
    if match is None:
        raise LocalAuthError("malformed_signature")
    sig = b64u_decode(match.group(3), 64)
    if sig is None:
        raise LocalAuthError("malformed_signature")
    return match.group(1), match.group(2), sig


# ---------------------------------------------------------------------- crypto
def _ec():
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import (
        decode_dss_signature,
        encode_dss_signature,
    )

    return ec, hashes, serialization, InvalidSignature, encode_dss_signature, decode_dss_signature


def public_key_from_x963(raw: bytes):
    """A P-256 public key from its uncompressed X9.63 form; ValueError if it is not one."""
    ec, *_ = _ec()
    if len(raw) != 65 or raw[0] != 0x04:
        raise ValueError("not an uncompressed P-256 point")
    return ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), raw)  # validates the point


def verify_signature(public_key: bytes, message: bytes, signature: bytes) -> bool:
    """ECDSA P-256/SHA-256 over `message`; `signature` is raw r||s (64 bytes), as CryptoKit makes it."""
    ec, hashes, _, invalid, encode, _ = _ec()
    if len(signature) != 64:
        return False
    try:
        key = public_key_from_x963(public_key)
    except ValueError:
        return False
    r = int.from_bytes(signature[:32], "big")
    s = int.from_bytes(signature[32:], "big")
    if not (r > 0 and s > 0):
        return False
    try:
        key.verify(encode(r, s), message, ec.ECDSA(hashes.SHA256()))
    except invalid:
        return False
    return True


class DeviceIdentity:
    """The device's own P-256 key: generated once, kept 0600, never sent anywhere."""

    def __init__(self, private_key) -> None:
        self._key = private_key
        _, _, serialization, *_ = _ec()
        self.public_key: bytes = private_key.public_key().public_bytes(
            serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
        self.fingerprint = key_fingerprint(self.public_key)

    def __repr__(self) -> str:  # never the key
        return f"DeviceIdentity(fp={self.fingerprint})"

    @classmethod
    def generate(cls) -> DeviceIdentity:
        ec, *_ = _ec()
        return cls(ec.generate_private_key(ec.SECP256R1()))

    @classmethod
    def load_or_create(cls, path: str | os.PathLike) -> DeviceIdentity:
        """The identity at `path`, creating it (0600) the first time.

        A file that is a link, not ours, or readable by other accounts is
        REFUSED (LocalAuthError "identity_unusable"), never repaired or
        replaced: replacing it would silently invalidate every phone's
        trust in this device, and a loose copy is not a secret.
        """
        _, _, serialization, *_ = _ec()
        p = Path(path)
        try:
            info = os.lstat(p)
        except FileNotFoundError:
            identity = cls.generate()
            pem = identity._key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                              serialization.NoEncryption())
            if not atomic_write_bytes(p, pem, mode=0o600):
                raise LocalAuthError("identity_unwritable", 503) from None
            return identity
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise LocalAuthError("identity_unusable", 503)
        if info.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise LocalAuthError("identity_unusable", 503)
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise LocalAuthError("identity_unusable", 503)
        try:
            key = serialization.load_pem_private_key(p.read_bytes(), password=None)
        except (OSError, ValueError, TypeError):
            raise LocalAuthError("identity_unusable", 503) from None
        ec, *_ = _ec()
        if not isinstance(key, ec.EllipticCurvePrivateKey) or key.curve.name != "secp256r1":
            raise LocalAuthError("identity_unusable", 503)
        return cls(key)

    def sign(self, message: bytes) -> bytes:
        """Raw r||s, the form the phone's CryptoKit verifies."""
        ec, hashes, _, _, _, decode = _ec()
        r, s = decode(self._key.sign(message, ec.ECDSA(hashes.SHA256())))
        return r.to_bytes(32, "big") + s.to_bytes(32, "big")


# ---------------------------------------------------------------------- the store
@dataclass
class Client:
    client_id: str
    public_key: bytes
    name: str
    source: str
    enrolled_at: float
    last_seen: float
    persisted_seen: float = 0.0

    def view(self) -> dict:
        """What may be shown about a phone: never its key, only a fingerprint."""
        return {"client_id": self.client_id, "name": self.name, "source": self.source,
                "enrolled_at": round(self.enrolled_at), "last_seen": round(self.last_seen),
                "key_fp": key_fingerprint(self.public_key)}


class ClientStore:
    """Enrolled phones, in one 0600 JSON file written atomically.

    A file that is unreadable, malformed, a link or loosely permissioned
    leaves the store EMPTY and says why (`fault`): no phone is trusted from a
    file someone else could have written.
    """

    def __init__(self, path: Optional[str | os.PathLike]) -> None:
        self.path = Path(path) if path else None
        self.fault: Optional[str] = None
        self.clients: dict[str, Client] = {}
        self._stamp: Optional[tuple[int, int]] = None
        self._load()

    def _file_stamp(self) -> Optional[tuple[int, int]]:
        try:
            info = os.lstat(self.path) if self.path is not None else None
        except OSError:
            return None
        return None if info is None else (info.st_mtime_ns, info.st_size)

    def refresh(self) -> bool:
        """Reload when the file changed under us (scripts/local_clients.py). True if reloaded.

        Without this, a revocation made on the device by hand was written back by
        the running service's next save.
        """
        if self.path is None or self._file_stamp() == self._stamp:
            return False
        self.clients, self.fault = {}, None
        self._load()
        return True

    def _load(self) -> None:
        if self.path is None:
            return
        self._stamp = self._file_stamp()
        try:
            info = os.lstat(self.path)
        except FileNotFoundError:
            return
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) \
                or info.st_mode & (stat.S_IRWXG | stat.S_IRWXO) \
                or (hasattr(os, "getuid") and info.st_uid != os.getuid()):
            self.fault = "the enrolled-phone store is not a private file; no phone is trusted until it is fixed"
            return
        try:
            doc = json.loads(self.path.read_bytes())
        except (OSError, ValueError):
            self.fault = "the enrolled-phone store could not be read; no phone is trusted"
            return
        rows = doc.get("clients") if isinstance(doc, dict) and doc.get("contract") == CONTRACT else None
        if not isinstance(rows, list):
            self.fault = "the enrolled-phone store has an unknown format; no phone is trusted"
            return
        for row in rows[:MAX_CLIENTS]:
            client = _client_from_row(row)
            if client is not None:
                self.clients[client.client_id] = client

    def save(self) -> bool:
        if self.path is None:
            return True
        rows = [{"client_id": c.client_id, "public_key": b64u(c.public_key), "name": c.name, "source": c.source,
                 "enrolled_at": c.enrolled_at, "last_seen": c.last_seen}
                for c in sorted(self.clients.values(), key=lambda c: c.enrolled_at)]
        body = json.dumps({"contract": CONTRACT, "clients": rows}, indent=1, sort_keys=True).encode("utf-8")
        ok = atomic_write_bytes(self.path, body, mode=0o600)
        if ok:
            self._stamp = self._file_stamp()
            for c in self.clients.values():
                c.persisted_seen = c.last_seen
        return ok


def _client_from_row(row: Any) -> Optional[Client]:
    if not isinstance(row, dict):
        return None
    key = b64u_decode(row.get("public_key"), 65)
    if key is None:
        return None
    try:
        public_key_from_x963(key)
    except ValueError:
        return None
    client_id = client_id_for(key)
    if row.get("client_id") != client_id:
        return None
    source = row.get("source") if row.get("source") in _SOURCES else "token"
    try:
        enrolled = float(row.get("enrolled_at") or 0.0)
        seen = float(row.get("last_seen") or enrolled)
    except (TypeError, ValueError):
        return None
    return Client(client_id, key, _clean_name(row.get("name")), source, enrolled, seen, seen)


def _clean_name(value: Any) -> str:
    text = str(value or "").strip()
    text = "".join(ch for ch in text if ch.isprintable())[:MAX_NAME]
    return text or "iPhone"


# ---------------------------------------------------------------------- the authority
@dataclass
class _Session:
    client_id: str
    peer: Optional[str]
    expires: float


@dataclass
class _Failures:
    times: list[float] = field(default_factory=list)
    locked_until: float = 0.0
    lockouts: int = 0


class LocalAuthority:
    """Everything the dashboard asks about a local request. Thread-safe.

    `clock` is monotonic (challenges, sessions, lockouts); `wall` is only for
    `enrolled_at` / `last_seen` and idle expiry, which must survive a reboot.
    """

    def __init__(self, device_id: str, identity: DeviceIdentity, store: ClientStore, *,
                 idle_expiry_days: float = DEFAULT_IDLE_EXPIRY_DAYS,
                 clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time,
                 on_change: Optional[Callable[[], None]] = None) -> None:
        self.device_id = device_id
        self.identity = identity
        self.store = store
        self._idle_expiry_s = max(0.0, float(idle_expiry_days)) * 86400.0
        self._clock = clock
        self._wall = wall
        self._lock = threading.RLock()
        self._challenges: OrderedDict[str, float] = OrderedDict()
        self._sessions: OrderedDict[str, _Session] = OrderedDict()
        self._failures: dict[str, _Failures] = {}
        #: Called after the set of enrolled phones changes (the uplink reports it).
        self.on_change = on_change
        self.refusals = 0
        with self._lock:
            self._expire_idle()

    # -- challenges ---------------------------------------------------------------
    def issue_challenge(self) -> str:
        now = self._clock()
        challenge = secrets.token_hex(16)
        with self._lock:
            self._prune(now)
            self._challenges[challenge] = now + CHALLENGE_TTL_S
            while len(self._challenges) > MAX_CHALLENGES:
                self._challenges.popitem(last=False)
        return challenge

    def _consume_challenge(self, challenge: str, now: float) -> bool:
        expires = self._challenges.pop(challenge, None)
        return expires is not None and now <= expires

    def _prune(self, now: float) -> None:
        for c in [c for c, exp in self._challenges.items() if exp < now]:
            del self._challenges[c]
        for s in [s for s, sess in self._sessions.items() if sess.expires < now]:
            del self._sessions[s]

    # -- failures -----------------------------------------------------------------
    def _check_lockout(self, peer: Optional[str], now: float) -> None:
        record = self._failures.get(peer or "")
        if record is not None and now < record.locked_until:
            raise LocalAuthError("locked_out", 429, retry_after_s=round(record.locked_until - now, 1))

    def _fail(self, peer: Optional[str], reason: str, now: float, status: int = 401) -> LocalAuthError:
        self.refusals += 1
        record = self._failures.setdefault(peer or "", _Failures())
        record.times = [t for t in record.times if now - t < FAIL_WINDOW_S] + [now]
        if len(record.times) >= FAIL_LIMIT:
            record.lockouts += 1
            record.locked_until = now + min(LOCKOUT_MAX_S, LOCKOUT_BASE_S * 2 ** (record.lockouts - 1))
            record.times = []
        if len(self._failures) > 1024:  # bounded: forget the quietest peers
            for key in list(self._failures)[:256]:
                if self._failures[key].locked_until < now:
                    del self._failures[key]
        return LocalAuthError(reason, status)

    # -- verification ---------------------------------------------------------------
    def verify_request(self, header: Any, method: str, path: str, body: bytes, peer: Optional[str]) -> Client:
        """The enrolled phone that signed this request, or LocalAuthError.

        The challenge is consumed whatever the outcome: one guess per challenge.
        """
        now = self._clock()
        with self._lock:
            self._check_lockout(peer, now)
            try:
                client_id, challenge, signature = parse_header(header)
            except LocalAuthError as exc:
                raise self._fail(peer, exc.reason, now) from None
            fresh = self._consume_challenge(challenge, now)
            self._sync_store()
            self._expire_idle()
            client = self.store.clients.get(client_id)
            if client is None:
                # Not enrolled, or revoked, or expired: the app must enrol again.
                raise self._fail(peer, "unknown_client", now, 403)
            if not fresh:
                raise self._fail(peer, "stale_challenge", now)
            message = request_message(self.device_id, client_id, challenge, method, path, body)
            if not verify_signature(client.public_key, message, signature):
                raise self._fail(peer, "bad_signature", now)
            self._failures.pop(peer or "", None)
            self._seen(client)
            return client

    def open_session(self, header: Any, path: str, body: bytes, peer: Optional[str]) -> dict:
        """POST /api/local/session: a signed challenge in, a device-signed session out."""
        try:
            payload = json.loads(body or b"{}")
        except ValueError:
            payload = None
        nonce = payload.get("nonce") if isinstance(payload, dict) else None
        if not isinstance(nonce, str) or not _NONCE.match(nonce):
            raise LocalAuthError("malformed_request", 400)
        client = self.verify_request(header, "POST", path, body, peer)
        _, challenge, _ = parse_header(header)
        session = secrets.token_urlsafe(32)
        ttl = int(SESSION_TTL_S)
        with self._lock:
            self._prune(self._clock())
            self._sessions[session] = _Session(client.client_id, peer, self._clock() + SESSION_TTL_S)
            while len(self._sessions) > MAX_SESSIONS:
                self._sessions.popitem(last=False)
        signature = self.identity.sign(session_message(self.device_id, client.client_id, challenge, nonce,
                                                       session, ttl))
        return {"contract": CONTRACT, "session": session, "expires_in_s": ttl, "client_id": client.client_id,
                "device_id": self.device_id, "device_key_fp": self.identity.fingerprint,
                "device_sig": b64u(signature)}

    def check_session(self, session: Any, peer: Optional[str]) -> Optional[Client]:
        """The phone a session belongs to, or None (unknown, expired, other peer, revoked)."""
        if not isinstance(session, str) or not 20 <= len(session) <= 64:
            return None
        now = self._clock()
        with self._lock:
            found = None
            for token, sess in self._sessions.items():
                # compare_digest over every entry: no early exit on a prefix.
                if hmac.compare_digest(token.encode("ascii", "replace"), session.encode("ascii", "replace")):
                    found = (token, sess)
            if found is None:
                return None
            self._sync_store()
            token, sess = found
            if sess.expires < now or (sess.peer is not None and peer is not None and sess.peer != peer):
                return None
            client = self.store.clients.get(sess.client_id)
            if client is None:
                self._sessions.pop(token, None)
                return None
            self._seen(client)
            return client

    def client_active(self, client_id: str) -> bool:
        """For a stream in progress: the phone is still enrolled (expiry is checked at connect)."""
        with self._lock:
            self._sync_store()
            return client_id in self.store.clients

    def _sync_store(self) -> None:
        """Pick up a change made on disk; sessions of phones no longer there end."""
        if self.store.refresh():
            gone = {s.client_id for s in self._sessions.values()} - set(self.store.clients)
            self._drop_sessions(gone)

    def _seen(self, client: Client) -> None:
        client.last_seen = self._wall()
        if client.last_seen - client.persisted_seen >= LAST_SEEN_PERSIST_S:
            self.store.save()

    def _expire_idle(self) -> None:
        if self._idle_expiry_s <= 0:
            return
        cutoff = self._wall() - self._idle_expiry_s
        stale = [cid for cid, c in self.store.clients.items() if c.last_seen < cutoff]
        if stale:
            for cid in stale:
                del self.store.clients[cid]
            self._drop_sessions(set(stale))
            self.store.save()
            self._changed()

    # -- enrolment and revocation ------------------------------------------------------
    def enroll(self, public_key_b64: Any, name: Any, source: str) -> dict:
        """Trust one phone's public key. Idempotent for a key already enrolled."""
        if source not in _SOURCES:
            raise LocalAuthError("bad_source", 400)
        key = b64u_decode(public_key_b64, 65)
        if key is None:
            raise LocalAuthError("bad_public_key", 400)
        try:
            public_key_from_x963(key)
        except ValueError:
            raise LocalAuthError("bad_public_key", 400) from None
        client_id = client_id_for(key)
        with self._lock:
            self._sync_store()
            if self.store.fault is not None:
                raise LocalAuthError("store_unusable", 503)
            existing = self.store.clients.get(client_id)
            if existing is None:
                if len(self.store.clients) >= MAX_CLIENTS:
                    raise LocalAuthError("too_many_clients", 409)
                now = self._wall()
                self.store.clients[client_id] = Client(client_id, key, _clean_name(name), source, now, now)
            else:
                existing.name = _clean_name(name) if name else existing.name
                existing.last_seen = self._wall()
            if not self.store.save():
                if existing is None:
                    self.store.clients.pop(client_id, None)
                raise LocalAuthError("store_unwritable", 503)
        self._changed()
        return self.enrollment_reply(client_id)

    def enrollment_reply(self, client_id: str) -> dict:
        return {"contract": CONTRACT, "client_id": client_id, "device_id": self.device_id,
                "device_public_key": b64u(self.identity.public_key), "device_key_fp": self.identity.fingerprint}

    def revoke(self, client_id: Any) -> bool:
        """Forget one phone and end its sessions now. True when it was enrolled."""
        if not isinstance(client_id, str) or not _CLIENT_ID.match(client_id):
            return False
        with self._lock:
            self._sync_store()
            removed = self.store.clients.pop(client_id, None)
            if removed is None:
                return False
            self._drop_sessions({client_id})
            self.store.save()
        self._changed()
        return True

    def apply_remote(self, change: Any) -> dict:
        """A `set_local_clients` command from the owner's account (verified by the uplink).

        `{"add": [{"public_key", "name"}], "revoke": [client_id]}`. Returns what
        was applied; malformed entries are skipped, never half-applied.
        """
        applied: dict[str, list] = {"added": [], "revoked": [], "refused": []}
        if not isinstance(change, dict):
            return applied
        for entry in (change.get("add") or [])[:MAX_CLIENTS] if isinstance(change.get("add"), list) else []:
            if not isinstance(entry, dict):
                continue
            try:
                reply = self.enroll(entry.get("public_key"), entry.get("name"), "account")
            except LocalAuthError as exc:
                applied["refused"].append(exc.reason)
                continue
            applied["added"].append(reply["client_id"])
        for client_id in (change.get("revoke") or [])[:64] if isinstance(change.get("revoke"), list) else []:
            if self.revoke(client_id):
                applied["revoked"].append(client_id)
        return applied

    def _drop_sessions(self, client_ids: set[str]) -> None:
        for token in [t for t, s in self._sessions.items() if s.client_id in client_ids]:
            del self._sessions[token]

    def _changed(self) -> None:
        callback = self.on_change
        if callback is not None:
            with contextlib.suppress(Exception):
                callback()

    # -- reporting -----------------------------------------------------------------
    def clients_view(self) -> list[dict]:
        with self._lock:
            self._sync_store()
            return [c.view() for c in sorted(self.store.clients.values(), key=lambda c: c.enrolled_at)]

    def report(self) -> dict:
        """For health and the uplink: the device's public key and the enrolled ids, nothing secret."""
        with self._lock:
            self._sync_store()
            return {"contract": CONTRACT, "device_public_key": b64u(self.identity.public_key),
                    "device_key_fp": self.identity.fingerprint,
                    "clients": sorted(self.store.clients), "fault": self.store.fault}


def build_authority(device_id: str, config: Optional[dict], data_dir: str | os.PathLike = "data",
                    on_change: Optional[Callable[[], None]] = None) -> tuple[Optional[LocalAuthority], Optional[str]]:
    """The device's LocalAuthority from `local_access:` config, or (None, why).

    Enabled by default: it adds no way in that is not enrolled through an
    authorised path, and it is what lets a phone reach the camera without a
    pasted token. `local_access.enabled: false` turns it off.
    """
    cfg = config if isinstance(config, dict) else {}
    if cfg.get("enabled", True) is False:
        return None, "local access is turned off (local_access.enabled: false)"
    base = Path(cfg.get("dir") or data_dir)
    try:
        identity = DeviceIdentity.load_or_create(cfg.get("identity_path") or base / "local_identity.pem")
    except LocalAuthError as exc:
        return None, f"the device's local identity key is unusable ({exc.reason}); local access is off"
    except ImportError:
        return None, "the cryptography package is missing; local access is off"
    store = ClientStore(cfg.get("clients_path") or base / "local_clients.json")
    try:
        days = float(cfg.get("idle_expiry_days", DEFAULT_IDLE_EXPIRY_DAYS))
    except (TypeError, ValueError):
        days = DEFAULT_IDLE_EXPIRY_DAYS
    return LocalAuthority(device_id, identity, store, idle_expiry_days=days, on_change=on_change), None
