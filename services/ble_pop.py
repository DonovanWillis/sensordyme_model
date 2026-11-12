"""Proof of possession for Bluetooth Wi-Fi provisioning (protocol v1).

Closes the M6 production blocker "BLE provisioning is unauthenticated" for the
credential write (docs/SECURITY_REVIEW_M6.md): with `network.provisioning.auth:
pop`, the device accepts Wi-Fi credentials only from a phone that has proved it
holds the device's SETUP CODE -- a per-device secret printed on its label and in
its QR code (`SDSETUP:1:<device id>:<code>`). Being discoverable, or being in
Bluetooth range, proves nothing: a device that has not seen the code never
belongs to the phone.

Protocol (all bytes base64 in JSON on the auth characteristic, 5d1e0007):

    code key   K_code = PBKDF2-HMAC-SHA256(code, "SensorDyme BLE PoP v1|" + id, 600 000 iterations)
    app   -> {"op":"hello","v":1,"na":<16 random bytes>}
    device-> {"op":"challenge","v":1,"id":<device id>,"nd":<16 random bytes>}
    both:     K_mac || K_enc = HKDF-SHA256(K_code, salt = na || nd, info = "...session|" + id, 64)
              T = "SensorDyme BLE PoP v1|" || len(id) || id || na || nd   (the transcript)
    app   -> {"op":"proof","tag": HMAC(K_mac, "app-proof" || T)}
    device-> {"op":"proven","tag": HMAC(K_mac, "device-proof" || T)}     (mutual: the app checks it)
    app   -> {"op":"creds","iv":<12 bytes>,"ct": AES-256-GCM(K_enc, iv, {"ssid","psk"}, aad="wifi-credentials" || T)}
    device-> {"op":"accepted"}; the join outcome follows on the status characteristic as before.

Why these choices (established primitives only, all from `cryptography`):
  * Both nonces and the device id are in every key and tag, so a recorded
    session cannot be replayed, and a proof for one device or one session is
    worthless for another.
  * The code is stretched with PBKDF2 at a fixed, versioned cost. Anyone who
    records a proof (or runs a fake device to collect one) can try codes offline;
    60 bits of code x 600 000 iterations puts that out of reach. The cost is NOT
    negotiated: a fake device could otherwise ask for 1 iteration.
  * A wrong proof consumes the challenge (one guess per challenge), and repeated
    failures lock the window out with exponential backoff that survives
    reconnecting (it is held by the device, not the connection).
  * The device proves itself back before the phone sends anything secret, so a
    look-alike peripheral cannot collect the Wi-Fi passphrase.
  * A PAKE (SPAKE2/OPAQUE) would remove the offline-guessing surface entirely;
    no vetted implementation exists on both CryptoKit and this device's stack.

The state machine is explicit (`TRANSITIONS`); every other transition raises.
Nothing here touches the radio: services/ble_gatt.py carries the bytes and
services/provisioning.py applies the credentials. NOT validated on physical BLE
hardware -- the protocol, state machine and the Swift client are verified by
tests and shared vectors only.
"""

from __future__ import annotations

import base64
import binascii
import functools
import json
import os
import secrets
import stat
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional

PROTOCOL_VERSION = 1
PREFIX = b"SensorDyme BLE PoP v1|"
PBKDF2_ITERATIONS = 600_000
NONCE_BYTES = 16
GCM_IV_BYTES = 12
TAG_BYTES = 32
#: Crockford base32: no I, L, O, U, so a label cannot be misread.
CODE_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
CODE_LENGTH = 12  # 60 bits
QR_PREFIX = "SDSETUP"
#: The largest auth write accepted: a GATT attribute is at most 512 bytes.
MAX_MESSAGE_BYTES = 512


class State(StrEnum):
    IDLE = "idle"                                    # Bluetooth window closed
    ADVERTISING = "advertising"                      # waiting for a phone's hello
    CHALLENGED = "challenged"                        # challenge sent, proof pending
    PROVEN = "proven"                                # the phone holds the setup code
    CREDENTIALS_RECEIVED = "credentials_received"    # decrypted, handed to the join
    APPLIED = "applied"                              # the join succeeded
    FAILED = "failed"                                # the join failed; a new handshake retries


#: Every legal transition. Anything else raises IllegalTransitionError.
TRANSITIONS: dict[State, frozenset[State]] = {
    State.IDLE: frozenset({State.ADVERTISING}),
    State.ADVERTISING: frozenset({State.CHALLENGED, State.IDLE}),
    # A second hello restarts the challenge (a phone that reconnected); a wrong
    # proof or a timeout returns to advertising and the challenge is spent.
    State.CHALLENGED: frozenset({State.CHALLENGED, State.PROVEN, State.ADVERTISING, State.IDLE}),
    State.PROVEN: frozenset({State.CREDENTIALS_RECEIVED, State.ADVERTISING, State.IDLE}),
    State.CREDENTIALS_RECEIVED: frozenset({State.APPLIED, State.FAILED, State.IDLE}),
    State.APPLIED: frozenset({State.CHALLENGED, State.ADVERTISING, State.IDLE}),
    State.FAILED: frozenset({State.CHALLENGED, State.ADVERTISING, State.IDLE}),
}


class IllegalTransitionError(RuntimeError):
    """A transition outside TRANSITIONS was attempted."""


class SetupCodeError(ValueError):
    """The setup code (or the file holding it) is unusable."""


# ---------------------------------------------------------------------- setup code
def normalize_setup_code(code: str) -> str:
    """The canonical 12-symbol form of a code as a person might type it.

    Dashes and spaces are ignored, case is folded, and the Crockford look-alikes
    are mapped (O->0, I/L->1). Anything else outside the alphabet is refused.
    """
    if not isinstance(code, str):
        raise SetupCodeError("setup code must be text")
    cleaned = code.upper().replace("-", "").replace(" ", "")
    cleaned = cleaned.translate(str.maketrans({"O": "0", "I": "1", "L": "1"}))
    if len(cleaned) != CODE_LENGTH or any(c not in CODE_ALPHABET for c in cleaned):
        raise SetupCodeError(f"setup code must be {CODE_LENGTH} characters of {CODE_ALPHABET}")
    return cleaned


def format_setup_code(code: str) -> str:
    """XXXX-XXXX-XXXX, as printed on the label."""
    canonical = normalize_setup_code(code)
    return "-".join(canonical[i:i + 4] for i in range(0, CODE_LENGTH, 4))


def generate_setup_code() -> str:
    """A fresh random code (factory provisioning; never rotated by the device)."""
    return format_setup_code("".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH)))


def qr_payload(device_id: str, code: str) -> str:
    return f"{QR_PREFIX}:{PROTOCOL_VERSION}:{device_id}:{normalize_setup_code(code)}"


def parse_qr_payload(text: str) -> tuple[str, str]:
    """(device id, canonical code) from a scanned label."""
    parts = (text or "").strip().split(":")
    if len(parts) != 4 or parts[0] != QR_PREFIX or parts[1] != str(PROTOCOL_VERSION) or not parts[2]:
        raise SetupCodeError("not a SensorDyme v1 setup code")
    return parts[2], normalize_setup_code(parts[3])


def load_setup_code(path: str) -> str:
    """The device's setup code from `path`, refusing a file others can read.

    The file is written once at the factory (scripts/ble_setup_code.py) and only
    read here. A group- or world-readable file, a symlink, or a file owned by
    another user is refused rather than used: the code is the whole of the
    proof, and a copy readable by any local account is not a secret.
    """
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        raise SetupCodeError(f"no setup code at {path}") from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise SetupCodeError(f"{path} must be a regular file, not a link")
    if info.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise SetupCodeError(f"{path} is readable by other accounts (mode {stat.S_IMODE(info.st_mode):o}); "
                             "it must be 0600")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise SetupCodeError(f"{path} is owned by uid {info.st_uid}, not this service")
    # Every way the file can fail to give up a code is a SetupCodeError, which
    # the provisioning service answers by keeping Bluetooth shut, saying why
    # in health and retrying. An unreadable file (mode 0000) or non-ASCII
    # content used to escape as PermissionError / UnicodeDecodeError and crash
    # the provisioning thread instead. The message never includes the content.
    try:
        with open(path, encoding="ascii") as f:
            text = f.read()
    except UnicodeDecodeError:
        raise SetupCodeError(f"{path} is not a setup code (not ASCII text)") from None
    except OSError as exc:
        raise SetupCodeError(f"{path} cannot be read ({type(exc).__name__}: {exc.strerror or 'error'})") from None
    return normalize_setup_code(text.strip())


def write_setup_code(path: str, code: str) -> None:
    """Write a code atomically with mode 0600. Refuses to replace an existing one."""
    canonical = normalize_setup_code(code)
    if os.path.lexists(path):
        raise SetupCodeError(f"{path} already exists: the label printed for it would stop working")
    directory = os.path.dirname(os.path.abspath(path))
    tmp = os.path.join(directory, f".{os.path.basename(path)}.{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="ascii") as f:
            f.write(format_setup_code(canonical) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)  # fails if `path` appeared meanwhile: never overwrites
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------- crypto
def _crypto():
    """`cryptography` primitives, imported on use so the rest of the device runs without it."""
    from cryptography.hazmat.primitives import hashes, hmac
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    return hashes, hmac, AESGCM, HKDF, PBKDF2HMAC


@functools.lru_cache(maxsize=8)
def derive_code_key(code: str, device_id: str) -> bytes:
    hashes, _, _, _, pbkdf2 = _crypto()
    kdf = pbkdf2(algorithm=hashes.SHA256(), length=32, salt=PREFIX + device_id.encode("utf-8"),
                 iterations=PBKDF2_ITERATIONS)
    return kdf.derive(normalize_setup_code(code).encode("ascii"))


def transcript(device_id: str, nonce_app: bytes, nonce_device: bytes) -> bytes:
    ident = device_id.encode("utf-8")
    if not 0 < len(ident) < 256:
        raise ValueError("device id must be 1-255 bytes")
    return PREFIX + bytes([len(ident)]) + ident + nonce_app + nonce_device


@dataclass(frozen=True)
class SessionKeys:
    mac: bytes
    enc: bytes
    transcript: bytes


def session_keys(code_key: bytes, device_id: str, nonce_app: bytes, nonce_device: bytes) -> SessionKeys:
    hashes, _, _, hkdf, _ = _crypto()
    okm = hkdf(algorithm=hashes.SHA256(), length=64, salt=nonce_app + nonce_device,
               info=PREFIX + b"session|" + device_id.encode("utf-8")).derive(code_key)
    return SessionKeys(okm[:32], okm[32:], transcript(device_id, nonce_app, nonce_device))


def _tag(keys: SessionKeys, label: bytes) -> bytes:
    hashes, hmac, _, _, _ = _crypto()
    h = hmac.HMAC(keys.mac, hashes.SHA256())
    h.update(label + keys.transcript)
    return h.finalize()


def app_proof(keys: SessionKeys) -> bytes:
    return _tag(keys, b"app-proof")


def device_proof(keys: SessionKeys) -> bytes:
    return _tag(keys, b"device-proof")


def _verify_tag(keys: SessionKeys, label: bytes, tag: bytes) -> bool:
    from cryptography.exceptions import InvalidSignature

    hashes, hmac, _, _, _ = _crypto()
    h = hmac.HMAC(keys.mac, hashes.SHA256())
    h.update(label + keys.transcript)
    try:
        h.verify(tag)  # constant time
    except InvalidSignature:
        return False
    return True


def seal_credentials(keys: SessionKeys, ssid: str, psk: Optional[str], iv: bytes) -> bytes:
    """The app's half, used by tests, vectors and scripts/ble_provision_client.py-style clients."""
    _, _, aesgcm, _, _ = _crypto()
    # Canonical (sorted keys, no spaces, UTF-8) so both implementations seal identical bytes.
    plaintext = json.dumps({"ssid": ssid, "psk": psk}, ensure_ascii=False, separators=(",", ":"),
                           sort_keys=True).encode()
    return aesgcm(keys.enc).encrypt(iv, plaintext, b"wifi-credentials" + keys.transcript)


def open_credentials(keys: SessionKeys, iv: bytes, ciphertext: bytes) -> Optional[bytes]:
    from cryptography.exceptions import InvalidTag

    _, _, aesgcm, _, _ = _crypto()
    try:
        return aesgcm(keys.enc).decrypt(iv, ciphertext, b"wifi-credentials" + keys.transcript)
    except InvalidTag:
        return None


def b64e(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _b64d(value, length: Optional[int] = None) -> Optional[bytes]:
    if not isinstance(value, str):
        return None
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return None
    if length is not None and len(raw) != length:
        return None
    return raw


# ---------------------------------------------------------------------- device side
@dataclass
class Reply:
    """What the device answers to one auth write, and credentials to apply (if any)."""
    message: dict
    credentials: Optional[tuple[str, str]] = None

    def encode(self) -> bytes:
        return json.dumps(self.message, separators=(",", ":")).encode()


@dataclass
class Limits:
    challenge_ttl_s: float = 30.0       # a proof must follow its challenge within this
    proven_ttl_s: float = 120.0         # credentials must follow a proof within this
    apply_ttl_s: float = 120.0          # a join with no outcome after this counts as failed
    max_failures: int = 5               # wrong proofs before a lockout
    lockout_base_s: float = 30.0        # first lockout; doubles per further failure
    lockout_max_s: float = 3600.0


@dataclass
class _Session:
    nonce_app: bytes
    nonce_device: bytes
    keys: SessionKeys
    at: float


@dataclass
class PopAuthenticator:
    """The device's half of the protocol: one window, one session at a time.

    Thread-safety: called from the BLE event loop only (writes, reads) plus
    `apply_result` from the join worker; the GIL-atomic state swap is all that
    crosses threads, and every method re-checks the state it acts on.
    """

    device_id: str
    setup_code: str
    clock: Callable[[], float] = time.monotonic
    rng: Callable[[int], bytes] = os.urandom
    limits: Limits = field(default_factory=Limits)

    def __post_init__(self) -> None:
        self._code_key = derive_code_key(normalize_setup_code(self.setup_code), self.device_id)
        self.state = State.IDLE
        self._entered = self.clock()
        self._session: Optional[_Session] = None
        self._failures = 0
        self._locked_until = 0.0
        self.history: list[tuple[State, State, str]] = []

    # -------------------------------------------------------------- transitions
    def _go(self, new: State, why: str) -> None:
        if new not in TRANSITIONS[self.state]:
            raise IllegalTransitionError(f"{self.state.value} -> {new.value} ({why})")
        self.history.append((self.state, new, why))
        self.state = new
        self._entered = self.clock()
        if new in (State.IDLE, State.ADVERTISING):
            self._session = None

    def start(self) -> None:
        self._go(State.ADVERTISING, "window opened")

    def stop(self) -> None:
        if self.state is not State.IDLE:
            self._go(State.IDLE, "window closed")

    def tick(self) -> State:
        """Expire whatever has waited too long. Called before every event."""
        age = self.clock() - self._entered
        if self.state is State.CHALLENGED and age > self.limits.challenge_ttl_s:
            self._go(State.ADVERTISING, "challenge expired")
        elif self.state is State.PROVEN and age > self.limits.proven_ttl_s:
            self._go(State.ADVERTISING, "proof expired without credentials")
        elif self.state is State.CREDENTIALS_RECEIVED and age > self.limits.apply_ttl_s:
            self._go(State.FAILED, "join reported no outcome")
        return self.state

    def locked_for(self) -> float:
        return max(0.0, self._locked_until - self.clock())

    def accepts_credentials(self) -> bool:
        return self.tick() is State.PROVEN

    def apply_result(self, ok: bool) -> None:
        """The outcome of the join this protocol handed over."""
        if self.tick() is State.CREDENTIALS_RECEIVED:
            self._go(State.APPLIED if ok else State.FAILED, "join succeeded" if ok else "join failed")

    # -------------------------------------------------------------- messages
    def handle(self, data: bytes) -> Reply:
        """Answer one write on the auth characteristic. Never raises on input."""
        self.tick()
        if len(data) > MAX_MESSAGE_BYTES:
            return _error("malformed", "message too long")
        try:
            msg = json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return _error("malformed", "not JSON")
        if not isinstance(msg, dict):
            return _error("malformed", "not an object")
        op = msg.get("op")
        if op == "hello":
            return self._hello(msg)
        if op == "proof":
            return self._proof(msg)
        if op == "creds":
            return self._creds(msg)
        return _error("malformed", "unknown op")

    def _hello(self, msg: dict) -> Reply:
        if msg.get("v") != PROTOCOL_VERSION:
            return _error("unsupported_version", f"this device speaks v{PROTOCOL_VERSION}")
        nonce_app = _b64d(msg.get("na"), NONCE_BYTES)
        if nonce_app is None:
            return _error("malformed", "na must be 16 bytes")
        if self.state not in (State.ADVERTISING, State.CHALLENGED, State.APPLIED, State.FAILED):
            return _error("busy", f"setup is {self.state.value}")
        wait = self.locked_for()
        if wait > 0:
            return _error("locked", "too many wrong codes", retry_after_s=int(wait) + 1)
        nonce_device = self.rng(NONCE_BYTES)
        keys = session_keys(self._code_key, self.device_id, nonce_app, nonce_device)
        self._go(State.CHALLENGED, "hello")
        self._session = _Session(nonce_app, nonce_device, keys, self.clock())
        return Reply({"op": "challenge", "v": PROTOCOL_VERSION, "id": self.device_id, "nd": b64e(nonce_device)})

    def _proof(self, msg: dict) -> Reply:
        tag = _b64d(msg.get("tag"), TAG_BYTES)
        if tag is None:
            return _error("malformed", "tag must be 32 bytes")
        if self.state is not State.CHALLENGED or self._session is None:
            return _error("unexpected", f"no challenge outstanding (setup is {self.state.value})")
        session = self._session
        if _verify_tag(session.keys, b"app-proof", tag):
            self._failures = 0
            self._go(State.PROVEN, "proof verified")
            self._session = session
            return Reply({"op": "proven", "tag": b64e(device_proof(session.keys))})
        self._failures += 1
        self._go(State.ADVERTISING, "wrong code")  # the challenge is spent: one guess each
        extra = {}
        if self._failures >= self.limits.max_failures:
            backoff = self.limits.lockout_base_s * 2 ** (self._failures - self.limits.max_failures)
            self._locked_until = self.clock() + min(backoff, self.limits.lockout_max_s)
            extra["retry_after_s"] = int(self.locked_for()) + 1
        return _error("wrong_code", "the setup code does not match this device", **extra)

    def _creds(self, msg: dict) -> Reply:
        iv = _b64d(msg.get("iv"), GCM_IV_BYTES)
        ciphertext = _b64d(msg.get("ct"))
        if iv is None or ciphertext is None or len(ciphertext) <= 16:
            return _error("malformed", "iv must be 12 bytes and ct non-empty")
        if self.state is not State.PROVEN or self._session is None:
            return _error("unexpected", f"credentials need a proof first (setup is {self.state.value})")
        plaintext = open_credentials(self._session.keys, iv, ciphertext)
        if plaintext is None:
            return _error("malformed", "credentials did not authenticate")
        from services.ble_gatt import parse_credentials

        credentials = parse_credentials(plaintext)
        if credentials is None:
            return _error("malformed", "credentials must be {ssid, psk}")
        self._go(State.CREDENTIALS_RECEIVED, "credentials decrypted")
        return Reply({"op": "accepted"}, credentials=credentials)


def _error(code: str, message: str, **extra) -> Reply:
    return Reply({"op": "error", "error": code, "message": message, **extra})


# ---------------------------------------------------------------------- config gate
AUTH_MODES = ("pop", "legacy")


def resolve_auth(config: dict, deployment: str = "development") -> str:
    """The credential-write policy for `network.provisioning`, or SetupCodeError.

    `auth: pop` requires the proof above. `auth: legacy` is the M6 "Just Works"
    write (anyone in range during the window) and is refused outright when the
    top-level `deployment:` is `production`: there is no insecure fallback in a
    production configuration, so such a device keeps its Bluetooth window
    closed rather than open it unauthenticated.
    """
    mode = str(config.get("auth", "legacy"))
    if mode not in AUTH_MODES:
        raise SetupCodeError(f"network.provisioning.auth must be one of {AUTH_MODES}, got {mode!r}")
    if mode == "legacy" and deployment == "production":
        raise SetupCodeError("network.provisioning.auth: legacy (unauthenticated Bluetooth setup) "
                             "is refused when deployment: production; use auth: pop")
    return mode
