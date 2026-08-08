"""SensorDyme cloud backend: accounts, device pairing, alert relay, remote mode control.

The service the iOS app and the devices both talk to:

    device (core/uplink.py)  --X-Device-Key-->  POST /v1/ingest   (events up, commands down)
    iOS app (ios/)           --JWT------------>  /v1/signup /v1/login /v1/devices ...

Privacy: this backend stores EVENTS ONLY (type, time, confidence, signals) — frames
never leave the device, so there is nothing visual to store or leak here.

Storage is SQLite (zero-ops, fine for one family of devices; swap for Postgres when
real customers arrive). Push notifications go out through a pluggable provider:
`ntfy` works today with no accounts (subscribe to your topic in the ntfy iOS app);
swap in APNs (token + p8 key from an Apple Developer account) for the production app.

Run locally:   .venv/bin/uvicorn cloud.app:app --port 8787
Create device: .venv/bin/python cloud/manage.py create-device "Car unit"
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import jwt
from cloud.notify import COOLDOWN_S as NOTIFY_COOLDOWN_S, gate
from core.presets import resolve_preset, same_preset
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

DB_PATH = os.environ.get("SENSORDYME_DB", str(Path(__file__).parent / "sensordyme.db"))
JWT_TTL_S = 30 * 24 * 3600
NTFY_BASE = os.environ.get("SENSORDYME_NTFY_BASE", "https://ntfy.sh")


def _load_jwt_secret() -> str:
    """A signing key that survives restarts, however the server was launched.

    Every token is signed with this key and verified with it 30 days later, so it
    must be *stable*. Two earlier behaviours made it unstable, and both produced
    `401 invalid token: Signature verification failed` on a token that was
    perfectly valid when issued:

      * an unset env var silently fell back to a committed default, so a server
        started from a plain shell and one started from a configured shell
        disagreed about every token; and
      * the documented deploy command was `export SENSORDYME_JWT_SECRET=$(openssl
        rand -hex 32)`, which mints a *new* key on every invocation — so each
        restart invalidated every session that existed before it.

    An explicit env var still wins (that is how you share one key across several
    server processes). Otherwise the key is generated once and persisted next to
    the database, which keeps sessions alive across restarts without ever
    shipping a shared default that anyone could forge tokens with.
    """
    env = os.environ.get("SENSORDYME_JWT_SECRET", "").strip()
    if env:
        return env

    path = Path(os.environ.get("SENSORDYME_JWT_SECRET_FILE", Path(DB_PATH).parent / ".jwt_secret"))
    try:
        existing = path.read_text().strip()
        if existing:
            return existing
    except FileNotFoundError:
        pass

    generated = secrets.token_hex(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 0600 before the write, so the key is never briefly world-readable.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(generated)
    return generated


JWT_SECRET = _load_jwt_secret()

app = FastAPI(title="SensorDyme Cloud", version="0.1")


# ------------------------------------------------------------------------- storage
def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@contextmanager
def db():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL,
                pw_hash TEXT NOT NULL, salt TEXT NOT NULL,
                ntfy_topic TEXT, created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS devices (
                id INTEGER PRIMARY KEY, name TEXT NOT NULL,
                claim_code TEXT UNIQUE NOT NULL, device_key TEXT UNIQUE NOT NULL,
                owner_id INTEGER REFERENCES users(id),
                active_preset TEXT, desired_preset TEXT, monitoring TEXT,
                last_seen REAL, created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY, device_id INTEGER NOT NULL REFERENCES devices(id),
                timestamp TEXT NOT NULL, type TEXT NOT NULL,
                confidence REAL, signals TEXT, severity TEXT,
                received_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_device ON events(device_id, id DESC);
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id),
                quantity INTEGER NOT NULL, address TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'placed', created_at REAL NOT NULL
            );
            """
        )
        # additive columns for existing databases (sqlite has no IF NOT EXISTS for columns)
        for statement in (
            "ALTER TABLE users ADD COLUMN phone TEXT",
            "ALTER TABLE users ADD COLUMN sms_on_loud INTEGER DEFAULT 1",
            "ALTER TABLE devices ADD COLUMN ai_rules TEXT",  # validated AI-configure proposal (JSON)
            "ALTER TABLE devices ADD COLUMN health TEXT",  # last device-reported health (JSON)
            "ALTER TABLE devices ADD COLUMN zones TEXT",  # user-drawn areas (JSON)
            # What custom configuration the device says is in charge. Without
            # this the app showed the base mode as selected while a custom
            # configuration was the thing actually running.
            "ALTER TABLE devices ADD COLUMN monitoring TEXT",
        ):
            try:
                conn.execute(statement)
            except sqlite3.OperationalError:
                pass  # column already exists


init_db()


# ------------------------------------------------------------------------- auth helpers
def _hash_pw(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000).hex()


def _issue_token(user_id: int) -> str:
    return jwt.encode({"uid": user_id, "exp": int(time.time()) + JWT_TTL_S}, JWT_SECRET, algorithm="HS256")


def current_user(authorization: str = Header(default="")) -> int:
    """FastAPI dependency: resolve the JWT bearer token to a user id."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    try:
        payload = jwt.decode(authorization[7:], JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise HTTPException(401, f"invalid token: {exc}") from exc
    return int(payload["uid"])


_HEALTH_STALE_S = 30  # matches the `online` window


def _health_payload(row: sqlite3.Row, now: float) -> Optional[dict]:
    """A device's last health report, or None if it can no longer be trusted.

    Health is only meaningful while the device is talking to us. Once it stops,
    the last report is a photograph of a moment that has passed — and rendering
    "Camera: capturing" next to a device that has been unplugged for an hour is
    exactly the failure this whole pipeline exists to prevent.
    """
    last_seen = row["last_seen"]
    if not last_seen or now - last_seen >= _HEALTH_STALE_S:
        return None
    try:
        report = json.loads(row["health"] or "null")
    except (ValueError, TypeError):
        return None
    return report if isinstance(report, dict) else None


def _rules_version(ai_rules_json: Optional[str]) -> str:
    """Stable fingerprint of a device's stored rule set (empty when it has none)."""
    if not ai_rules_json:
        return ""
    return hashlib.sha256(ai_rules_json.encode()).hexdigest()[:16]


def _zones_version(zones_json: Optional[str]) -> str:
    """Fingerprint of a device's zone set (empty when it has none)."""
    if not zones_json or zones_json == "[]":
        return ""
    return hashlib.sha256(zones_json.encode()).hexdigest()[:16]


def _rules_payload(ai_rules_json: Optional[str]) -> list[dict]:
    """The `rules` list out of a stored proposal, tolerant of anything malformed."""
    try:
        rules = json.loads(ai_rules_json or "{}").get("rules")
    except (ValueError, TypeError, AttributeError):
        return []
    return [r for r in rules if isinstance(r, dict)] if isinstance(rules, list) else []


def current_device(x_device_key: str = Header(default="")) -> sqlite3.Row:
    """FastAPI dependency: resolve X-Device-Key to a device row."""
    with db() as conn:
        row = conn.execute("SELECT * FROM devices WHERE device_key = ?", (x_device_key,)).fetchone()
    if row is None:
        raise HTTPException(401, "invalid device key")
    return row


# ------------------------------------------------------------------------- push + SMS
def send_push(user_row: sqlite3.Row, title: str, body: str) -> None:
    """Push provider. ntfy today (topic-based, no accounts); replace with APNs for the
    production iOS app (needs an Apple Developer p8 key — see cloud/README.md)."""
    topic = user_row["ntfy_topic"] if user_row else None
    if not topic:
        return
    try:
        import requests

        requests.post(f"{NTFY_BASE}/{topic}", data=body.encode(), headers={"Title": title}, timeout=5)
    except Exception:  # noqa: BLE001 - push failures must never break ingestion
        pass


def send_sms(user_row: sqlite3.Row, body: str) -> None:
    """SMS to the account's connected phone number on serious alerts.

    Uses Twilio when TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN / TWILIO_FROM are set;
    otherwise logs the message (so the flow is fully exercisable before buying a
    Twilio number)."""
    if user_row is None:
        return
    try:
        phone = user_row["phone"]
        enabled = bool(user_row["sms_on_loud"])
    except (KeyError, IndexError):
        return
    if not phone or not enabled:
        return
    sid = os.environ.get("TWILIO_ACCOUNT_SID")
    token = os.environ.get("TWILIO_AUTH_TOKEN")
    sender = os.environ.get("TWILIO_FROM")
    if not (sid and token and sender):
        print(f"sms (log-only, set TWILIO_* env to send for real) -> {phone}: {body}")
        return
    try:
        import requests

        requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json",
            data={"To": phone, "From": sender, "Body": body},
            auth=(sid, token),
            timeout=10,
        )
    except Exception:  # noqa: BLE001 - SMS failures must never break ingestion
        pass


# ------------------------------------------------------------------------- schemas
class Credentials(BaseModel):
    email: str
    password: str


class ClaimBody(BaseModel):
    claim_code: str


class PresetBody(BaseModel):
    preset: str


class NtfyBody(BaseModel):
    ntfy_topic: str


class PhoneBody(BaseModel):
    phone: str
    sms_on_loud: bool = True


class OrderBody(BaseModel):
    quantity: int
    address: str


class RenameBody(BaseModel):
    name: str


class ZonesBody(BaseModel):
    zones: list[dict] = []


class IngestBody(BaseModel):
    events: list[dict] = []
    active_preset: Optional[str] = None
    # The device's own view of its camera, inference, accelerator and network.
    health: Optional[dict] = None
    # Fingerprint of the rule set the device is currently enforcing. Lets ingest
    # re-deliver rules only when they actually changed, instead of every poll.
    rules_version: Optional[str] = None
    zones_version: Optional[str] = None
    # What custom configuration the device says is in charge, if any: the
    # instruction someone typed and whether it is overriding the base mode.
    monitoring: Optional[dict] = None


class AIConfigureBody(BaseModel):
    device_id: int
    prompt: str


# ------------------------------------------------------------------------- user API
@app.get("/v1/health")
def health():
    """Unauthenticated liveness probe — the app's Connection doctor hits this to
    distinguish 'wrong address' from 'server down' from 'auth problem'."""
    return {"ok": True, "service": "sensordyme-cloud", "time": time.time()}


@app.post("/v1/signup")
def signup(body: Credentials):
    email = body.email.strip().lower()
    if not email or "@" not in email or len(body.password) < 8:
        raise HTTPException(400, "valid email and a password of 8+ characters required")
    salt = secrets.token_hex(16)
    with db() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO users (email, pw_hash, salt, created_at) VALUES (?, ?, ?, ?)",
                (email, _hash_pw(body.password, salt), salt, time.time()),
            )
        except sqlite3.IntegrityError:
            raise HTTPException(409, "an account with this email already exists") from None
    return {"token": _issue_token(cur.lastrowid), "email": email}


@app.post("/v1/login")
def login(body: Credentials):
    with db() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (body.email.strip().lower(),)).fetchone()
    if row is None or _hash_pw(body.password, row["salt"]) != row["pw_hash"]:
        raise HTTPException(401, "wrong email or password")
    return {"token": _issue_token(row["id"]), "email": row["email"]}


@app.post("/v1/account/ntfy")
def set_ntfy(body: NtfyBody, uid: int = Depends(current_user)):
    with db() as conn:
        conn.execute("UPDATE users SET ntfy_topic = ? WHERE id = ?", (body.ntfy_topic.strip(), uid))
    return {"ok": True}


@app.post("/v1/account/phone")
def set_phone(body: PhoneBody, uid: int = Depends(current_user)):
    phone = body.phone.strip()
    if phone and not (phone.startswith("+") and phone[1:].replace(" ", "").isdigit() and len(phone) >= 8):
        raise HTTPException(400, "phone must be E.164 format, e.g. +15551234567 (or empty to remove)")
    with db() as conn:
        conn.execute(
            "UPDATE users SET phone = ?, sms_on_loud = ? WHERE id = ?",
            (phone or None, int(body.sms_on_loud), uid),
        )
    return {"ok": True}


@app.post("/v1/orders")
def place_order(body: OrderBody, uid: int = Depends(current_user)):
    if not (1 <= body.quantity <= 25) or len(body.address.strip()) < 10:
        raise HTTPException(400, "quantity 1-25 and a full shipping address are required")
    with db() as conn:
        cur = conn.execute(
            "INSERT INTO orders (user_id, quantity, address, created_at) VALUES (?, ?, ?, ?)",
            (uid, body.quantity, body.address.strip(), time.time()),
        )
    return {"ok": True, "order_id": cur.lastrowid, "status": "placed"}


@app.get("/v1/orders")
def list_orders(uid: int = Depends(current_user)):
    with db() as conn:
        rows = conn.execute(
            "SELECT id, quantity, address, status, created_at FROM orders WHERE user_id = ? ORDER BY id DESC",
            (uid,),
        ).fetchall()
    return {"orders": [dict(r) for r in rows]}


@app.post("/v1/devices/{device_id}/name")
def rename_device(device_id: int, body: RenameBody, uid: int = Depends(current_user)):
    name = body.name.strip()
    if not (1 <= len(name) <= 60):
        raise HTTPException(400, "name must be 1-60 characters")
    with db() as conn:
        _owned_device(conn, device_id, uid)
        conn.execute("UPDATE devices SET name = ? WHERE id = ?", (name, device_id))
    return {"ok": True}


@app.post("/v1/devices/claim")
def claim_device(body: ClaimBody, uid: int = Depends(current_user)):
    with db() as conn:
        row = conn.execute("SELECT * FROM devices WHERE claim_code = ?", (body.claim_code.strip(),)).fetchone()
        if row is None:
            raise HTTPException(404, "unknown claim code")
        if row["owner_id"] is not None and row["owner_id"] != uid:
            raise HTTPException(409, "device already claimed by another account")
        conn.execute("UPDATE devices SET owner_id = ? WHERE id = ?", (uid, row["id"]))
    return {"ok": True, "device_id": row["id"], "name": row["name"]}


def _owned_device(conn: sqlite3.Connection, device_id: int, uid: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM devices WHERE id = ? AND owner_id = ?", (device_id, uid)).fetchone()
    if row is None:
        raise HTTPException(404, "no such device on this account")
    return row


@app.get("/v1/devices")
def list_devices(uid: int = Depends(current_user)):
    with db() as conn:
        rows = conn.execute("SELECT * FROM devices WHERE owner_id = ?", (uid,)).fetchall()
    now = time.time()
    return {
        "devices": [
            {
                "id": r["id"],
                "name": r["name"],
                "active_preset": r["active_preset"],
                "desired_preset": r["desired_preset"],
                "monitoring": json.loads(r["monitoring"]) if r["monitoring"] else None,
                "online": bool(r["last_seen"] and now - r["last_seen"] < 30),
                "last_seen": r["last_seen"],
                # Whatever the device last said about itself. Withheld once the
                # device goes quiet: a stale report showing a healthy camera is
                # worse than no report, because the app would render it as live.
                "health": _health_payload(r, now),
            }
            for r in rows
        ]
    }


@app.get("/v1/devices/{device_id}/events")
def device_events(device_id: int, limit: int = 100, uid: int = Depends(current_user)):
    limit = max(1, min(limit, 1000))
    with db() as conn:
        _owned_device(conn, device_id, uid)
        # `id` is included so the app can ask why a specific alert fired.
        # Without it an event is unaddressable and /explain is unreachable.
        rows = conn.execute(
            "SELECT id, timestamp, type, confidence, signals, severity FROM events "
            "WHERE device_id = ? ORDER BY id DESC LIMIT ?",
            (device_id, limit),
        ).fetchall()
    return {"events": [dict(r) for r in rows]}


@app.post("/v1/devices/{device_id}/preset")
def set_preset(device_id: int, body: PresetBody, uid: int = Depends(current_user)):
    with db() as conn:
        _owned_device(conn, device_id, uid)
        # Store the current name for a mode, never a superseded one. The device
        # will apply and report the current name, so a stored alias could never
        # match on convergence and would re-send on every poll for ever.
        normalized = resolve_preset(body.preset)
        conn.execute("UPDATE devices SET desired_preset = ? WHERE id = ?", (normalized, device_id))
    return {"ok": True, "desired_preset": normalized}


# ------------------------------------------------------------------------- zones
# A zone is a named area the user draws over their own camera view. Rules about
# places ("the kitchen", "the pool area", "this room") are meaningless without
# them: a camera cannot know what a kitchen is, only that a person's feet are
# inside a shape someone drew and named.
#
# Stored in normalised 0..1 coordinates so one drawing survives any change of
# capture resolution, and delivered to the device over the same fingerprinted
# command channel as rules.

def _clean_zones(raw: list) -> list[dict]:
    """Validate a zone list. Bad entries are dropped, not repaired.

    A zone that arrives malformed would otherwise become an area of the wrong
    shape silently watching the wrong part of a room, which is worse than the
    rule plainly not working.
    """
    cleaned: list[dict] = []
    for entry in raw or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        kind = str(entry.get("kind") or "restricted")
        points = entry.get("polygon") or []
        if not name or kind not in ("restricted", "safe", "boundary"):
            continue
        try:
            polygon = [(float(p[0]), float(p[1])) for p in points]
        except (TypeError, ValueError, IndexError):
            continue
        # Three points is the minimum that encloses an area; anything outside
        # 0..1 is not a normalised coordinate and cannot be trusted.
        if len(polygon) < 3 or not all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in polygon):
            continue
        cleaned.append({
            "name": name,
            "kind": kind,
            "polygon": [[round(x, 4), round(y, 4)] for x, y in polygon],
        })
    return cleaned


@app.get("/v1/devices/{device_id}/zones")
def get_zones(device_id: int, uid: int = Depends(current_user)):
    with db() as conn:
        device = _owned_device(conn, device_id, uid)
    try:
        zones = json.loads(device["zones"] or "[]")
    except (ValueError, TypeError):
        zones = []
    return {"zones": zones if isinstance(zones, list) else []}


@app.post("/v1/devices/{device_id}/zones")
def set_zones(device_id: int, body: ZonesBody, uid: int = Depends(current_user)):
    zones = _clean_zones(body.zones)
    rejected = len(body.zones or []) - len(zones)
    with db() as conn:
        _owned_device(conn, device_id, uid)
        conn.execute("UPDATE devices SET zones = ? WHERE id = ?", (json.dumps(zones), device_id))
    return {"ok": True, "zones": zones, "rejected": rejected}


# ------------------------------------------------------------------------- AI configuration
# Natural language -> a validated detection configuration. The pipeline is:
#   1. Claude (structured outputs) proposes {preset, rules[], settings, unsupported[]}
#      grounded in the device's real capability taxonomy (core/taxonomy.py);
#   2. _validate_proposal re-checks every rule against the taxonomy server-side —
#      the model cannot claim a detection the device doesn't support;
#   3. anything unsupported comes back with WHY and what it would need, instead
#      of being silently accepted.
# Without an ANTHROPIC_API_KEY the deterministic keyword fallback still returns
# valid (if less nuanced) configurations, so the demo never dies with the network.

_MODES = (
    "driver_distraction", "driver_distraction_openpilot", "baby_monitoring",
    "elderly_monitoring", "security_monitoring",
)

_RULE_SCHEMA = {
    "type": "object",
    "properties": {
        # A rule is (target, predicate) rather than an opaque detection id tied to
        # one preset. That is what lets a request exist without belonging to any
        # of the five modes — the old shape could only name detections some
        # preset already provided, which is why everything collapsed into one.
        "target": {
            "type": "string",
            "description": "The object to watch, using an EXACT label from the capability sheet "
                           "(e.g. 'person', 'dog', 'bottle'). Empty for whole-scene conditions.",
        },
        "predicate": {
            "type": "string",
            "description": "What to check about the target, from the capability sheet "
                           "(e.g. 'present', 'in_zone', 'near', 'absent').",
        },
        "secondary": {
            "type": "string",
            "description": "Second object for relational predicates like 'near'/'holding' "
                           "(e.g. target='dog', predicate='near', secondary='couch'). Empty otherwise.",
        },
        "description": {"type": "string", "description": "Plain-language restatement of the user's rule"},
        "zone": {"type": "string", "description": "Zone name the user must draw for area rules; empty if none"},
        "persist_s": {"type": "number", "description": "Seconds the condition must hold before alerting"},
        "min_count": {"type": "integer", "description": "How many must be present (usually 1)"},
        "confidence": {"type": "number", "description": "Minimum detection confidence 0-1"},
        "severity": {"type": "string", "enum": ["info", "warning", "critical"]},
        "cooldown_s": {"type": "number", "description": "Seconds between repeat notifications"},
        "schedule": {"type": "string", "description": "Active hours like '23:00-06:00', or empty for always"},
        "notify": {"type": "string", "enum": ["push", "sms", "silent"]},
    },
    "required": ["target", "predicate", "secondary", "description", "zone", "persist_s",
                 "min_count", "confidence", "severity", "cooldown_s", "schedule", "notify"],
    "additionalProperties": False,
}

_PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        # "" means a custom configuration that belongs to no preset. Requests
        # must be allowed to not be one of the five modes; forcing a preset is
        # what turned "monitor for a can" into "watch for intruders".
        "preset": {"type": "string", "enum": ["", *list(_MODES)]},
        "variant": {"type": "string", "description": "Preset variant name, or empty for standard"},
        "explanation": {"type": "string", "description": "2-3 friendly sentences on what was set up"},
        "rules": {"type": "array", "items": _RULE_SCHEMA},
        "settings": {
            "type": "object",
            "description": "Module threshold overrides as string key/value pairs",
            "additionalProperties": False,
            "properties": {},
        },
        "unsupported": {
            "type": "array",
            "description": "Parts of the request the device cannot reliably do",
            "items": {
                "type": "object",
                "properties": {
                    "request": {"type": "string"},
                    "reason": {"type": "string"},
                    "requires": {"type": "string", "description": "What hardware/model/setup would enable it"},
                },
                "required": ["request", "reason", "requires"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["preset", "variant", "explanation", "rules", "settings", "unsupported"],
    "additionalProperties": False,
}


def _taxonomy():
    """core.taxonomy, imported lazily so the cloud backend can also run standalone."""
    try:
        from core import taxonomy
        return taxonomy
    except ImportError:
        return None


def _capabilities():
    """core.capabilities, imported lazily so the backend can run standalone."""
    try:
        from core import capabilities

        return capabilities
    except ImportError:
        return None


def _ai_system_prompt() -> str:
    caps = _capabilities()
    if caps is None:
        return _legacy_system_prompt()
    return (
        "You turn a person's plain-language monitoring request into a configuration for "
        "SensorDyme, a privacy-first camera device (video never leaves the device; only "
        "events do).\n\n"
        "SensorDyme is a GENERAL monitoring platform, not five fixed modes. Most requests "
        "are custom configurations. Only set `preset` when the request genuinely matches one "
        "of these whole-scenario modes: " + ", ".join(_MODES) + ". Otherwise leave `preset` "
        "empty — a custom rule set is the normal, expected outcome, not a failure.\n\n"
        "Express the request as rules of the form (target, predicate, secondary). Use ONLY "
        "the exact labels and predicates from the capability sheet below.\n\n"
        "Examples of the shape:\n"
        "  'alert if my dog gets on the couch'  -> target=dog, predicate=near, secondary=couch\n"
        "  'alert if someone enters the pool area' -> target=person, predicate=in_zone, zone=pool\n"
        "  'tell me when the room is empty'     -> target=person, predicate=absent\n"
        "  'alert if more than 3 people arrive' -> target=person, predicate=present, min_count=4\n\n"
        "HONESTY RULES — these are the product's core promise:\n"
        "- If a label is NOT in the sheet, do NOT substitute a preset or a loosely similar "
        "object. Put the request in `unsupported`, say plainly that the vision model does not "
        "recognise it, and state what would enable it (a model that includes that class, or "
        "labelled examples to train one). Naming a close alternative from the sheet is helpful; "
        "silently swapping to it is not.\n"
        "- Predicates marked UNAVAILABLE cannot be used. Put those requests in `unsupported` "
        "with the stated requirement.\n"
        "- Never claim identity, intent, intoxication, emotion or medical conditions from "
        "appearance. Report observable events only.\n"
        "- Area rules need a zone the user draws in the app: name it and say so.\n"
        "- Prefer conservative persistence (>=2s) and cooldowns (>=30s); a single ambiguous "
        "frame must never alert.\n"
        "- A request you cannot serve is fine to decline. An invented capability is not.\n\n"
        "CAPABILITY SHEET:\n" + caps.registry().describe_for_llm()
    )


def _legacy_system_prompt() -> str:
    tax = _taxonomy()
    capability_sheet = tax.summary_for_llm() if tax else ""
    return (
        "You translate a user's plain-language monitoring request into a configuration for "
        "SensorDyme, a privacy-first camera device (all video stays on the device; only events "
        "leave it). Pick the best preset, then express the user's specific wishes as structured "
        "rules using ONLY detection ids from the capability sheet below.\n\n"
        "HONESTY RULES (these are the product's core promise):\n"
        "- Only use detections marked SUPPORTED or PARTIAL. If part of the request needs an "
        "UNSUPPORTED/PLANNED detection, put it in `unsupported` with the reason and what it "
        "would require — never pretend the device can do it.\n"
        "- Area rules ('near the pool', 'in the storage room') use a zone the user must draw in "
        "the app: name the zone in the rule and mention drawing it in the explanation.\n"
        "- Time rules ('after 11pm') go in `schedule`; note in the explanation that scheduled "
        "enforcement is coming and the rule alerts at all hours until then.\n"
        "- Never claim identity, intent, intoxication, or medical conditions from appearance.\n"
        "- Prefer conservative persistence (>=2s) and sensible cooldowns (>=30s) so users are "
        "not flooded; single ambiguous frames must never alert.\n\n"
        "CAPABILITY SHEET:\n" + capability_sheet
    )


# Which model interprets custom requests. Jupiter (a local model) is the
# default, so a fresh install needs no third-party API key to configure a camera
# its owner already paid for. Anthropic remains selectable as a benchmark.
JUPITER_PROVIDER = os.environ.get("SENSORDYME_AI_PROVIDER", "jupiter").strip().lower()
JUPITER_MODEL = os.environ.get("JUPITER_MODEL", "").strip()
JUPITER_BASE_URL = os.environ.get("JUPITER_BASE_URL", "").strip()
# Off by default. An external provider must never be reached because it happens
# to be installed — sending a user's words off the box is an explicit decision.
JUPITER_ALLOW_EXTERNAL_FALLBACK = os.environ.get(
    "SENSORDYME_ALLOW_EXTERNAL_AI", ""
).strip().lower() in ("1", "true", "yes")


def _jupiter_config() -> dict:
    config = {}
    if JUPITER_MODEL:
        config["model"] = JUPITER_MODEL
    if JUPITER_BASE_URL:
        config["base_url"] = JUPITER_BASE_URL
    return config


def _ai_configure_llm(prompt: str) -> Optional[dict]:
    """Interpret a request into a structured proposal, or None to fall through.

    Jupiter and Anthropic are interchangeable here on purpose: both return a
    proposal that `_validate_proposal` then re-checks against the capability
    registry. Neither is trusted, so swapping the provider changes accuracy and
    cost, never what the device is willing to run.
    """
    try:
        from jupiter.interpret import interpret
    except ImportError:
        # Backend deployed without core/ + jupiter/ on the path. Say nothing
        # here; the caller's fallback explains it to the user.
        return None

    outcome = interpret(prompt, provider_name=JUPITER_PROVIDER, config=_jupiter_config())

    if not outcome.ok and JUPITER_ALLOW_EXTERNAL_FALLBACK and JUPITER_PROVIDER != "anthropic":
        # Explicitly enabled, and only after the local path failed.
        outcome = interpret(prompt, provider_name="anthropic")

    if not outcome.ok and not outcome.interpretation.rules:
        return None  # let the caller report why, rather than inventing a preset

    proposal = outcome.interpretation.as_proposal()
    proposal["source"] = outcome.provider or JUPITER_PROVIDER
    proposal["jupiter"] = {
        "provider": outcome.provider,
        "model": outcome.model,
        "latency_s": outcome.latency_s,
        "injection_suspected": outcome.injection_suspected,
        **outcome.meta,
    }
    return proposal


_AI_FALLBACKS = [
    (("baby", "infant", "crib", "nursery", "toddler"), "baby_monitoring",
     "Baby Monitoring: presence, safe/danger zones, and unusual-stillness alerts. Draw a safe "
     "zone around the crib (and any danger zones) in the app to enable area alerts.",
     [("unusual_stillness", "Alert if unusually still for a long period", 120.0, "warning")]),
    (("fall", "elder", "senior", "grandma", "grandpa"), "elderly_monitoring",
     "Elderly Monitoring: immediate fall alerts (rapid drop then stillness — sitting down "
     "won't trigger it), staying active until movement resumes.",
     [("fall", "Alert immediately on a confirmed fall", 1.5, "critical")]),
    (("security", "intruder", "door", "burglar", "lurk", "linger", "trespass"), "security_monitoring",
     "Security: person-count changes, plus restricted-zone and lingering alerts once you draw "
     "zones in the app.",
     [("zone_violation", "Alert when someone enters a restricted zone", 2.0, "critical")]),
    (("drive", "driving", "car", "phone", "drowsy", "road"), "driver_distraction",
     "Driver Distraction: loud alerts for phone use while looking away and sustained eye closure.",
     [("phone_use", "Alert on sustained phone use while driving", 2.0, "critical")]),
]


def _ai_fallback(prompt: str) -> dict:
    """Deterministic keyword configuration when Claude is unavailable. Returns the
    same structured shape, honestly labeled with source=fallback."""
    lowered = prompt.lower()
    for keywords, preset, explanation, rules in _AI_FALLBACKS:
        if any(k in lowered for k in keywords):
            return {
                "preset": preset, "variant": "", "explanation": explanation,
                "rules": [
                    {"detection": det, "description": desc, "zone": "", "persist_s": persist,
                     "min_count": 1, "confidence": 0.6, "severity": sev, "cooldown_s": 60.0,
                     "schedule": "", "notify": "push"}
                    for det, desc, persist, sev in rules
                ],
                "settings": {}, "unsupported": [], "source": "fallback",
            }
    # No keyword matched. Historically this returned security_monitoring as a
    # "safe default", which was neither safe nor a default: it silently pointed
    # the camera at something the user never asked for, and reported success.
    # Changing nothing is the honest outcome — say what is wrong and what fixes it.
    return {
        "preset": "", "variant": "", "rules": [], "settings": {},
        "explanation": (
            "I can't interpret custom requests right now — the language service that turns "
            "your words into a monitoring rule isn't configured on your backend, so I'd only "
            "be guessing. Nothing has been changed. You can still pick a monitoring mode "
            "manually in the meantime."
        ),
        "unsupported": [{
            "request": prompt,
            "reason": "the language service that interprets custom requests is not configured",
            "requires": "set ANTHROPIC_API_KEY on the backend and restart it",
            "support": "unsupported",
            "alternatives": [],
        }],
        "source": "unavailable",
    }


def _validate_proposal(data: dict) -> dict:
    """Re-check the model's claims against what the device can actually run.

    The model is capable and well-briefed, but it is still the thing being
    checked: a rule that reaches the device must be executable there. Rules the
    capability layer cannot serve are MOVED to `unsupported` carrying the
    specific gap, so the user learns what to add rather than watching a rule
    that silently never fires.

    Validation is against the capability registry, not against a preset's
    detection list. That is the substantive change: a rule no longer has to
    belong to one of the five modes in order to exist.
    """
    caps = _capabilities()
    if caps is None:
        return _legacy_validate_proposal(data)

    reg = caps.registry()
    preset = data.get("preset") if data.get("preset") in _MODES else ""
    valid_rules, unsupported = [], list(data.get("unsupported") or [])

    for rule in data.get("rules") or []:
        target = str(rule.get("target", "")).strip().lower()
        predicate = str(rule.get("predicate", "")).strip().lower() or "present"
        secondary = str(rule.get("secondary", "")).strip().lower()
        described = str(rule.get("description") or f"{predicate} {target}".strip())

        verdict = reg.assess(target, predicate)
        if not verdict.ok:
            unsupported.append({
                "request": described,
                "reason": verdict.reason,
                "requires": verdict.requires or "additional model or training data",
                "support": verdict.support.value,
                "alternatives": list(verdict.alternatives),
            })
            continue

        # A relational rule is only as good as its second object.
        if secondary:
            secondary_verdict = reg.assess(secondary, "present")
            if not secondary_verdict.ok:
                unsupported.append({
                    "request": described,
                    "reason": secondary_verdict.reason,
                    "requires": secondary_verdict.requires or "additional model or training data",
                    "support": secondary_verdict.support.value,
                    "alternatives": list(secondary_verdict.alternatives),
                })
                continue

        rule["target"] = target
        rule["predicate"] = predicate
        rule["secondary"] = secondary
        rule["persist_s"] = max(0.5, min(3600.0, float(rule.get("persist_s", 2.0) or 2.0)))
        rule["confidence"] = max(0.2, min(0.95, float(rule.get("confidence", 0.6) or 0.6)))
        rule["cooldown_s"] = max(10.0, min(86400.0, float(rule.get("cooldown_s", 60.0) or 60.0)))
        rule["min_count"] = max(1, min(100, int(rule.get("min_count", 1) or 1)))
        if rule.get("severity") not in ("info", "warning", "critical"):
            rule["severity"] = "warning"
        if verdict.support is caps.Support.PARTIAL:
            # The user is told the rule is an approximation, on the rule itself,
            # so the caveat survives into the review screen and the rule list.
            rule["caveat"] = verdict.reason
        valid_rules.append(rule)

    data["preset"] = preset
    data["mode"] = "preset" if preset else "custom"
    data["rules"] = valid_rules
    data["unsupported"] = unsupported
    return data


def _legacy_validate_proposal(data: dict) -> dict:
    """Taxonomy-based validation, for a backend running without core/ available."""
    tax = _taxonomy()
    preset = data.get("preset") if data.get("preset") in _MODES else ""
    valid_rules, unsupported = [], list(data.get("unsupported") or [])
    for rule in data.get("rules") or []:
        det_id = str(rule.get("detection", "") or rule.get("target", ""))
        det = tax.find(preset, det_id) if (tax and preset) else None
        if det is not None and det.support not in ("supported", "partial"):
            unsupported.append({
                "request": str(rule.get("description", det_id)),
                "reason": f"'{det_id}' is {det.support} on this device",
                "requires": det.requires or "future model/hardware support",
            })
            continue
        valid_rules.append(rule)
    data["preset"] = preset
    data["rules"] = valid_rules
    data["unsupported"] = unsupported
    return data


@app.post("/v1/ai/configure")
def ai_configure(body: AIConfigureBody, uid: int = Depends(current_user)):
    with db() as conn:
        _owned_device(conn, body.device_id, uid)
    result = _validate_proposal(_ai_configure_llm(body.prompt) or _ai_fallback(body.prompt))
    # Persist the validated proposal on the device so the app's Apply (set_preset)
    # and future device-side rule sync both have it.
    import json as _json

    with db() as conn:
        conn.execute("UPDATE devices SET ai_rules = ? WHERE id = ?",
                     (_json.dumps(result), body.device_id))
    return result


@app.get("/v1/devices/{device_id}/events/{event_id}/explain")
def explain_event(device_id: int, event_id: int, uid: int = Depends(current_user)):
    """Why this alert fired, grounded in what the device actually recorded.

    The explanation is assembled from the stored event and the rule it came
    from. Jupiter only rephrases that fact list — it is never asked what
    happened, so it cannot invent a cause. With no model reachable the
    deterministic text is returned unchanged, which is why this endpoint has no
    failure mode beyond the event not existing.
    """
    with db() as conn:
        device = _owned_device(conn, device_id, uid)
        row = conn.execute(
            "SELECT * FROM events WHERE id = ? AND device_id = ?", (event_id, device_id)
        ).fetchone()
    if row is None:
        raise HTTPException(404, "no such event on this device")

    try:
        from jupiter.explain import explain, parse_rule_for_event
    except ImportError:
        raise HTTPException(503, "explanations are unavailable on this backend")

    event = dict(row)
    rule = parse_rule_for_event(device["ai_rules"], event.get("type", ""))

    provider = None
    if JUPITER_PROVIDER not in ("", "off"):
        try:
            from jupiter.providers import get_provider

            candidate = get_provider(JUPITER_PROVIDER, _jupiter_config())
            ok, _ = candidate.available()
            provider = candidate if ok else None
        except Exception:  # noqa: BLE001 - the facts path never needs a model
            provider = None

    return explain(event, rule, provider=provider).as_dict()


# ------------------------------------------------------------------------- device API
@app.post("/v1/ingest")
def ingest(body: IngestBody, device: sqlite3.Row = Depends(current_device)):
    now = time.time()
    with db() as conn:
        conn.execute(
            "UPDATE devices SET last_seen = ?, active_preset = COALESCE(?, active_preset),"
            " health = COALESCE(?, health), monitoring = COALESCE(?, monitoring) WHERE id = ?",
            (
                now,
                body.active_preset,
                json.dumps(body.health) if body.health is not None else None,
                json.dumps(body.monitoring) if body.monitoring is not None else None,
                device["id"],
            ),
        )
        owner = (
            conn.execute("SELECT * FROM users WHERE id = ?", (device["owner_id"],)).fetchone()
            if device["owner_id"]
            else None
        )
        for e in body.events[:200]:
            conn.execute(
                "INSERT INTO events (device_id, timestamp, type, confidence, signals, severity, received_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    device["id"],
                    str(e.get("timestamp", "")),
                    str(e.get("type", "unknown")),
                    float(e.get("confidence", 0.0) or 0.0),
                    str(e.get("signals", "")),
                    str(e.get("severity", "log")),
                    now,
                ),
            )
        row = conn.execute(
            "SELECT desired_preset, ai_rules, zones FROM devices WHERE id = ?", (device["id"],)
        ).fetchone()
        desired, ai_rules_json, zones_json = row["desired_preset"], row["ai_rules"], row["zones"]
        # Convergence: once the device reports the desired preset as active, the
        # command is fulfilled — clear it. Without this, a stale desired_preset
        # keeps re-delivering forever and FIGHTS local mode switches made on the
        # device/dashboard (the device flips back within one poll interval).
        # Compare by mode, not by string. A device handed a superseded name
        # applies — and correctly reports — the current one, so raw equality
        # never matches: the backend concludes the command was ignored and
        # re-sends it every poll, fighting whatever mode the user picks locally.
        # This also converges rows written before names were normalised.
        if desired and same_preset(desired, body.active_preset or device["active_preset"]):
            conn.execute("UPDATE devices SET desired_preset = NULL WHERE id = ?", (device["id"],))
            desired = None

    # Push is the path that works when the app is closed, so it is the path
    # that most needs a filter. It had none: every "loud" event became a push,
    # which on a real device is around 960 in two days. The app filtered
    # locally, the server did not, so the channel that reaches someone when
    # they are NOT looking was the unfiltered one.
    #
    # Every event is still stored above and still appears in the timeline. This
    # decides interruptions only.
    for e in body.events:
        owner_id = owner["id"] if owner is not None else None
        send, held = gate.evaluate(owner_id, str(e.get("severity", "")))
        if not send:
            continue
        title = f"SensorDyme: {e.get('type', 'alert')}"
        detail = str(e.get("signals", "")) or "alert"
        if held:
            detail += f" (and {held} more in the last {int(NOTIFY_COOLDOWN_S)}s)"
        send_push(owner, title, detail)
        send_sms(owner, f"{title} — {detail} ({device['name']})")

    command = None
    if desired and desired != (body.active_preset or device["active_preset"]):
        command = {"set_preset": desired}

    # Custom rules are configured in the app but enforced on the device, so they
    # have to travel down this channel — without it a saved proposal changed the
    # text in the app and nothing about what the camera actually watched for.
    version = _rules_version(ai_rules_json)
    if version and version != (body.rules_version or ""):
        command = dict(command or {})
        command["set_rules"] = _rules_payload(ai_rules_json)
        # Send what the rules are FOR, not only what they are. The device shows
        # this back to the person so they can confirm it is watching the right
        # thing; without it the device stores rules with no idea what they mean.
        try:
            command["rules_instruction"] = str(
                (json.loads(ai_rules_json) or {}).get("explanation") or ""
            )
        except (TypeError, ValueError):
            command["rules_instruction"] = ""
        command["rules_version"] = version

    # Zones travel the same way rules do. A zone rule without its polygon is
    # inert, so the two have to arrive by the same mechanism or a rule can be
    # live on the device while the area it names is not.
    zone_version = _zones_version(zones_json)
    if zone_version and zone_version != (body.zones_version or ""):
        command = dict(command or {})
        try:
            command["set_zones"] = json.loads(zones_json)
        except (ValueError, TypeError):
            command.pop("set_zones", None)
        else:
            command["zones_version"] = zone_version

    return {"ok": True, "stored": len(body.events), "command": command}
