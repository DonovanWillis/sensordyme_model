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
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import jwt
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

DB_PATH = os.environ.get("SENSORDYME_DB", str(Path(__file__).parent / "sensordyme.db"))
JWT_SECRET = os.environ.get("SENSORDYME_JWT_SECRET", "dev-secret-change-in-production")
JWT_TTL_S = 30 * 24 * 3600
NTFY_BASE = os.environ.get("SENSORDYME_NTFY_BASE", "https://ntfy.sh")

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
                active_preset TEXT, desired_preset TEXT,
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


class IngestBody(BaseModel):
    events: list[dict] = []
    active_preset: Optional[str] = None


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
                "online": bool(r["last_seen"] and now - r["last_seen"] < 30),
                "last_seen": r["last_seen"],
            }
            for r in rows
        ]
    }


@app.get("/v1/devices/{device_id}/events")
def device_events(device_id: int, limit: int = 100, uid: int = Depends(current_user)):
    limit = max(1, min(limit, 1000))
    with db() as conn:
        _owned_device(conn, device_id, uid)
        rows = conn.execute(
            "SELECT timestamp, type, confidence, signals, severity FROM events "
            "WHERE device_id = ? ORDER BY id DESC LIMIT ?",
            (device_id, limit),
        ).fetchall()
    return {"events": [dict(r) for r in rows]}


@app.post("/v1/devices/{device_id}/preset")
def set_preset(device_id: int, body: PresetBody, uid: int = Depends(current_user)):
    with db() as conn:
        _owned_device(conn, device_id, uid)
        conn.execute("UPDATE devices SET desired_preset = ? WHERE id = ?", (body.preset, device_id))
    return {"ok": True, "desired_preset": body.preset}


# ------------------------------------------------------------------------- AI configuration
_AI_SYSTEM_PROMPT = (
    "You configure a SensorDyme monitoring device from a user's plain-language request. "
    "Available presets: driver_distraction, driver_distraction_openpilot, baby_monitoring, "
    "elderly_monitoring, security_monitoring. Answer ONLY a JSON object: "
    '{"preset": "<one of the presets>", "explanation": "<2-3 friendly sentences on what you set up>", '
    '"settings": {"<setting>": "<value>", ...}, "rules": ["<alert rule>", ...]}'
)

_AI_FALLBACKS = [
    (("baby", "infant", "crib", "nursery"), "baby_monitoring",
     "Baby Monitoring: face-covered detection at maximum sensitivity, crying alerts on."),
    (("fall", "elder", "senior", "grandma", "grandpa"), "elderly_monitoring",
     "Elderly Monitoring: immediate fall alerts plus a daytime inactivity check."),
    (("security", "intruder", "door", "package", "burglar"), "security_monitoring",
     "Security: alerts whenever the number of people in view changes."),
    (("drive", "driving", "car", "phone", "drowsy", "road"), "driver_distraction",
     "Driver Distraction: loud alerts for phone use while looking away and sustained eye closure."),
]


def _ai_fallback(prompt: str) -> dict:
    """Deterministic keyword configuration when no LLM key is configured."""
    lowered = prompt.lower()
    for keywords, preset, explanation in _AI_FALLBACKS:
        if any(k in lowered for k in keywords):
            return {"preset": preset, "explanation": explanation, "settings": {}, "rules": [f"From: {prompt}"]}
    return {
        "preset": "driver_distraction",
        "explanation": f"Kept the default mode and added your rule: {prompt}",
        "settings": {},
        "rules": [f"Notify when: {prompt}"],
    }


def _ai_configure_llm(prompt: str) -> Optional[dict]:
    """Ask an LLM (Gemini via its OpenAI-compatible endpoint) for a configuration."""
    import json as _json

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        import requests

        resp = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            json={
                "model": os.environ.get("SENSORDYME_AI_MODEL", "gemini-3.1-flash-lite"),
                "max_tokens": 1024,
                "messages": [
                    {"role": "system", "content": _AI_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=45,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        data = _json.loads(text)
        if data.get("preset") not in {p for _, p, _ in [(k, p, e) for k, p, e in _AI_FALLBACKS]} | {
            "driver_distraction_openpilot", "driver_distraction",
        }:
            return None
        return {
            "preset": data["preset"],
            "explanation": str(data.get("explanation", "")),
            "settings": {str(k): str(v) for k, v in (data.get("settings") or {}).items()},
            "rules": [str(r) for r in (data.get("rules") or [])],
        }
    except Exception:  # noqa: BLE001 - fall back to keywords on any LLM failure
        return None


@app.post("/v1/ai/configure")
def ai_configure(body: AIConfigureBody, uid: int = Depends(current_user)):
    with db() as conn:
        _owned_device(conn, body.device_id, uid)
    result = _ai_configure_llm(body.prompt) or _ai_fallback(body.prompt)
    return result


# ------------------------------------------------------------------------- device API
@app.post("/v1/ingest")
def ingest(body: IngestBody, device: sqlite3.Row = Depends(current_device)):
    now = time.time()
    with db() as conn:
        conn.execute(
            "UPDATE devices SET last_seen = ?, active_preset = COALESCE(?, active_preset) WHERE id = ?",
            (now, body.active_preset, device["id"]),
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
        desired = conn.execute("SELECT desired_preset FROM devices WHERE id = ?", (device["id"],)).fetchone()[
            "desired_preset"
        ]

    for e in body.events:
        if e.get("severity") == "loud" and owner is not None:
            title = f"SensorDyme: {e.get('type', 'alert')}"
            detail = str(e.get("signals", "")) or "alert"
            send_push(owner, title, detail)
            send_sms(owner, f"{title} — {detail} ({device['name']})")

    command = None
    if desired and desired != (body.active_preset or device["active_preset"]):
        command = {"set_preset": desired}
    return {"ok": True, "stored": len(body.events), "command": command}
