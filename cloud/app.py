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

import asyncio
import hashlib
import hmac
import json
import logging
import math
import os
import re
import secrets
import sqlite3
import time
from contextlib import asynccontextmanager, contextmanager, suppress
from pathlib import Path
from typing import Optional

import jwt
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from cloud import mdns
from cloud.notify import COOLDOWN_S as NOTIFY_COOLDOWN_S
from cloud.notify import gate
from cloud.ratelimit import (
    AI_CONCURRENCY,
    AI_RATE_LIMITER,
    CLAIM_LIMITER,
    LOGIN_LIMITER,
    SIGNUP_LIMITER,
)
from core.presets import load_presets, resolve_preset, same_preset

DB_PATH = os.environ.get("SENSORDYME_DB", str(Path(__file__).parent / "sensordyme.db"))
JWT_TTL_S = 30 * 24 * 3600
NTFY_BASE = os.environ.get("SENSORDYME_NTFY_BASE", "https://ntfy.sh")

# uvicorn configures handlers for its own loggers and leaves the root logger
# bare, so a logger named for this module would have its lines silently
# dropped. Logging through uvicorn's own channel puts them in the server log
# an operator is actually looking at.
log = logging.getLogger("uvicorn.error")


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
        if len(env.encode()) < 32:
            # Warned, not refused: refusing would take a running deployment
            # down on upgrade. HS256 with a short key is brute-forceable
            # offline from any one token, which is a forged session for anyone.
            logging.getLogger("uvicorn.error").warning(
                "SENSORDYME_JWT_SECRET is %d bytes; use at least 32 (e.g. "
                "`openssl rand -hex 32`, set once and kept) or unset it to have one "
                "generated and persisted", len(env.encode()),
            )
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

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Announce this server on the local network for as long as it is running.

    This is what lets a phone sign in without being told an address (see
    cloud/mdns.py). It is deliberately best-effort: if it fails, the server still
    serves, and Settings still accepts a typed-in address.
    """
    advertiser = mdns.Advertiser(mdns.advertised_port()) if mdns.enabled() else None
    # Registering probes the network for name conflicts and takes a couple of
    # seconds. On the event loop that is two seconds during which the server is
    # started but answering nothing, so it goes to a thread.
    if advertiser is not None:
        await asyncio.to_thread(advertiser.start)

    # Confirm, shortly after startup, that the port being announced is the port
    # the server is actually on. An advertised port nothing listens on sends the
    # app somewhere definite and wrong, which is worse than not advertising at
    # all. The usual cause: launching without `--port`, so uvicorn binds its own
    # default 8000 while SENSORDYME_PORT still says 8787.
    #
    # Deferred rather than checked inline, because uvicorn binds the socket AFTER
    # lifespan startup finishes — a check here would always find nothing and cry
    # wolf on every boot. Warn only; the server itself is fine either way.
    async def _warn_if_announcing_the_wrong_port(port: int) -> None:
        await asyncio.sleep(2)
        if not await asyncio.to_thread(mdns.verify_advertised_port, port):
            log.warning(
                "announcing port %d, but nothing is listening there — the app "
                "will be sent to a port this server is not on. Pass --port %d, "
                "or set SENSORDYME_PORT to the port uvicorn actually bound.",
                port, port,
            )

    port_check = (
        asyncio.create_task(_warn_if_announcing_the_wrong_port(advertiser.port))
        if advertiser is not None
        else None
    )
    try:
        yield
    finally:
        # Cancelled first: a shutdown inside the 2-second window must not leave a
        # task probing a port the server has already released.
        if port_check is not None:
            port_check.cancel()
        if advertiser is not None:
            await asyncio.to_thread(advertiser.stop)


#: Whether to publish the interactive API docs.
#:
#: Off by default. `/docs`, `/redoc` and `/openapi.json` were all answering 200
#: to anyone who could reach the backend — and since the launchd template binds
#: 0.0.0.0, that is everyone on the Wi-Fi. They describe every route, schema and
#: field, which is a map for probing the rest. Useful while developing, so this
#: is a switch rather than a deletion.
_DOCS_ENABLED = os.environ.get("SENSORDYME_API_DOCS", "").strip().lower() in {"1", "true", "yes", "on"}

app = FastAPI(
    title="SensorDyme Cloud",
    version="0.1",
    lifespan=_lifespan,
    docs_url="/docs" if _DOCS_ENABLED else None,
    redoc_url="/redoc" if _DOCS_ENABLED else None,
    openapi_url="/openapi.json" if _DOCS_ENABLED else None,
)


#: The largest request body accepted, on any route.
#:
#: FastAPI reads and parses a body BEFORE any auth dependency runs, so without
#: this one unauthenticated POST could make the server hold as much as it cared
#: to send -- measured at ~1 GB resident for a 300 MB body against a wrong
#: device key, on a deployment whose whole VM has 512 MB. The largest legitimate
#: request is a device flushing a full backlog (MAX_INGEST_EVENTS events), which
#: is well under 1 MB; 4 MB leaves room without leaving a lever.
MAX_BODY_BYTES = 4 * 1024 * 1024


class _BodySizeLimit:
    """Refuse oversized bodies with 413, by declared length and by bytes received.

    Pure ASGI rather than BaseHTTPMiddleware so the body is never buffered here,
    and so a chunked body with no Content-Length is counted as it arrives.
    """

    def __init__(self, app, limit: int) -> None:
        self.app = app
        self.limit = limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        for name, value in scope.get("headers") or []:
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = self.limit + 1
                if declared > self.limit:
                    await self._reject(send)
                    return

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.limit:
                    # FastAPI re-raises an HTTPException met while reading the
                    # body (anything else becomes a 400), so this surfaces as 413.
                    raise HTTPException(413, "request body too large")
            return message

        await self.app(scope, limited_receive, send)

    @staticmethod
    async def _reject(send) -> None:
        body = b'{"detail":"request body too large"}'
        await send({
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"connection", b"close"),
            ],
        })
        await send({"type": "http.response.body", "body": body})


app.add_middleware(_BodySizeLimit, limit=MAX_BODY_BYTES)


@app.exception_handler(RequestValidationError)
async def _validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """FastAPI's default 422, minus the echo of what was sent.

    The default includes each failing field's `input` -- so an over-long
    password came straight back in the response body, into any proxy or client
    log that records responses. `ctx` and `url` go too: `ctx` can hold
    non-serialisable values, and neither tells a client anything `msg` does not.
    """
    errors = [
        {key: value for key, value in error.items() if key not in ("input", "ctx", "url")}
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.exception_handler(OverflowError)
async def _overflow(_request: Request, _exc: OverflowError) -> JSONResponse:
    """An integer too large for SQLite (a 30-digit device id or limit) is a bad
    request, not a server fault."""
    return JSONResponse(status_code=400, content={"detail": "number out of range"})


@app.exception_handler(UnicodeEncodeError)
async def _unencodable(_request: Request, _exc: UnicodeEncodeError) -> JSONResponse:
    """JSON can carry a lone surrogate ("\\ud800") that UTF-8 cannot; hashing or
    storing it raised and answered 500."""
    return JSONResponse(status_code=400, content={"detail": "text is not valid Unicode"})


# ------------------------------------------------------------------------- storage
#: How long a statement waits for a write lock before giving up.
#:
#: WAL allows readers and one writer, so writes serialise. `/v1/ingest` writes on
#: every device poll (a device UPDATE, one INSERT per event, sometimes a
#: compare-and-swap on desired_preset) and every authenticated request reads
#: `revoked_tokens`, so brief contention is normal rather than exceptional.
#:
#: Python's default is 5 seconds, which is not obviously wrong — but the cost of
#: exceeding it is badly asymmetric here. An exceeded timeout raises
#: `sqlite3.OperationalError: database is locked`, which nothing catches, so it
#: becomes a 500 — and `core/uplink.py` treats any non-200 as a failed send and
#: re-queues the identical batch, so the one endpoint most likely to hit
#: contention is also the one whose client retries forever. Waiting longer is
#: strictly better than failing: the device is polling every couple of seconds
#: and has nothing else to do.
BUSY_TIMEOUT_S = 15.0


def _restrict_db_permissions() -> None:
    """Make the database readable only by its owner.

    It holds every account's email, password hash and salt, every device key,
    every claim code, shipping addresses and the whole event history — and it was
    created with the process umask, which on this machine meant `-rw-r--r--`.
    Any local account could read all of it.

    Contrast `_load_jwt_secret`, which opens its file 0o600 *before* writing so
    the secret is never briefly exposed. The same care had not been applied one
    file over. sqlite creates the database on first connect, so this runs after
    `init_db` rather than before.

    Best-effort: a database on a filesystem that cannot express these bits must
    not stop the backend from serving.
    """
    for path in (DB_PATH, f"{DB_PATH}-wal", f"{DB_PATH}-shm"):
        with suppress(OSError):
            os.chmod(path, 0o600)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=BUSY_TIMEOUT_S)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    # `timeout=` above covers the implicit transaction Python manages; the pragma
    # applies the same budget to every statement on this connection, including
    # ones inside an explicit transaction. Setting one without the other leaves a
    # path that still fails immediately.
    conn.execute(f"PRAGMA busy_timeout={int(BUSY_TIMEOUT_S * 1000)}")
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
            -- Explicit logout / revocation. A JWT is normally trusted purely
            -- by its signature for the full 30-day TTL, which means "sign
            -- out" on the phone was cosmetic -- it cleared the local copy
            -- and left the token itself valid anywhere else it existed
            -- (another device, a log, an interception) until it naturally
            -- expired. `jti` is generated once per token at issuance and
            -- travels with it; a row here makes that ONE token invalid
            -- immediately, everywhere, without touching any other session
            -- the account has. `exp` is stored alongside so expired rows can
            -- be pruned instead of growing this table forever -- a revoked
            -- token past its own natural expiry is redundant to keep, the
            -- signature check would already reject it.
            CREATE TABLE IF NOT EXISTS revoked_tokens (
                jti TEXT PRIMARY KEY, exp REAL NOT NULL
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
            # Whether an event came from something the user configured. Without
            # it the app cannot tell a preset's own warning from the one thing
            # someone explicitly asked to be told about.
            "ALTER TABLE events ADD COLUMN source TEXT",
            # A device-generated idempotency key, present from firmware that
            # sends one (see core/uplink.py's push_event). Lets ingest tell a
            # genuine resend apart from a second, distinct event.
            "ALTER TABLE events ADD COLUMN event_key TEXT",
            # Whether the user's last preset request was "AI Custom" (preset
            # == "") rather than a named mode. `ai_rules` itself is never
            # cleared by switching presets — reopening Configure-with-AI must
            # still show what was last configured — so without a separate
            # signal for "and is it supposed to be running right now", the
            # rules-delivery check below (comparing ai_rules' version against
            # what the device reports) had no way to tell "the device hasn't
            # picked this up yet" apart from "the user switched to a normal
            # preset a while ago". It resent the same set_rules command
            # forever, and the device kept reapplying it — silently pulling
            # the device back into custom monitoring within one poll interval
            # of every single preset switch, for any account that had ever
            # used Configure-with-AI even once.
            "ALTER TABLE devices ADD COLUMN custom_monitoring_wanted INTEGER DEFAULT 0",
            # Seconds the triggering condition was sustained before the event
            # fired (Event.duration) -- every module already computes this,
            # but it was silently dropped before reaching the cloud (see
            # core/uplink.py's push_event). Real evidence for notification
            # text ("remained down for 34 seconds") instead of a bare type
            # name. NULL for events from firmware that predates this field.
            "ALTER TABLE events ADD COLUMN duration REAL",
            # The proposal the user is LOOKING AT, which is not the same thing
            # as the proposal they asked to run. /v1/ai/configure is called by
            # AIConfigureView's `configure()` — the preview, before the person
            # has seen a word of the result, let alone tapped Activate — and
            # it used to write straight to `ai_rules`. On a device already in
            # AI Custom that is the column the ingest handler delivers from,
            # so merely previewing a new idea rewrote the running
            # configuration: the very next 2-second poll shipped, committed
            # and persisted rules nobody had accepted, and hot-swapped the
            # detection module underneath a live camera. Previews land here;
            # only set_preset (Activate) promotes a draft into `ai_rules`.
            "ALTER TABLE devices ADD COLUMN ai_rules_draft TEXT",
        ):
            try:
                conn.execute(statement)
            except sqlite3.OperationalError as exc:
                # ONLY the expected "this database already has that column".
                # Catching every OperationalError here meant a genuinely
                # broken migration — a typo'd statement, a missing table, a
                # locked database — started the server anyway, in a schema
                # nobody had checked, and surfaced later as a confusing
                # runtime error a long way from the cause.
                if "duplicate column name" not in str(exc).lower():
                    raise

        # Must run AFTER the ALTER TABLE above: event_key does not exist yet
        # on a database that predates it, and CREATE INDEX on a column that
        # doesn't exist is an error, not a no-op. A resent batch (device
        # treats a lost response as failure and retries the same events) must
        # not become duplicate rows. Partial index: older devices sending no
        # event_key are unaffected, since SQLite never treats two NULLs as
        # equal for uniqueness.
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_events_dedup "
            "ON events(device_id, event_key) WHERE event_key IS NOT NULL"
        )


init_db()
_restrict_db_permissions()


# ------------------------------------------------------------------------- auth helpers
def _hash_pw(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000).hex()


#: Hashed against when the email is unknown, so a login for an account that does
#: not exist costs what one for an account that does costs. Skipping PBKDF2 for
#: an unknown email answered ~20x faster, which says which emails have accounts.
_NO_SUCH_USER_SALT = secrets.token_hex(16)


def _issue_token(user_id: int) -> str:
    return jwt.encode(
        {"uid": user_id, "exp": int(time.time()) + JWT_TTL_S, "jti": secrets.token_hex(16)},
        JWT_SECRET, algorithm="HS256",
    )


def current_token(authorization: str = Header(default="")) -> dict:
    """FastAPI dependency: the decoded JWT payload, rejecting a revoked token.

    Separate from `current_user` (which most routes want, just the uid)
    because `/v1/logout` needs the jti too, and decoding twice per request
    for the one route that needs both fields is worse than one dependency
    two others can narrow.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    try:
        # Every claim this file goes on to rely on is required HERE, where a
        # missing one is a MissingRequiredClaimError — a PyJWTError, caught
        # below, answered with 401. Read off the payload later instead and a
        # validly-signed token with unexpected claims reaches raw dict access
        # on the hot auth path and comes back as a 500: an unhandled server
        # error for what is simply a token this server will not accept.
        #
        # `exp` is the one that matters most. PyJWT only enforces an
        # expiry that is PRESENT, so a token minted without one was accepted
        # for ever — a 30-day TTL that silently did not apply, and a session
        # nothing could age out. Requiring it makes the TTL real.
        #
        # `jti` is required too. THIS DELIBERATELY REVERSES an earlier
        # decision — do not "restore" it. The code here used to read the jti
        # off the payload with .get() and skip the revocation check when it
        # was absent, carrying the comment: "Tokens issued before revocation
        # existed have no jti — they simply cannot be revoked early and ride
        # out their remaining TTL, the same behavior every token had before
        # this. Never a reason to reject one."
        #
        # That reasoning treated "cannot be revoked" as a harmless property
        # of old tokens. It is not harmless: it is a token for which "sign
        # out" is cosmetic. The phone clears its local copy and reports
        # success while the credential stays valid anywhere else it exists —
        # another device, a log, an interception — until its own expiry. A
        # security property people rely on and that silently does not apply
        # is worse than one that is plainly absent, because nobody knows to
        # compensate for it.
        #
        # The cost is bounded and one-time: every token this backend issues
        # has carried a jti since revocation was added, so the only tokens
        # refused are older than that, and their owners sign in once more.
        # Reconsider only if that population ever becomes large enough that
        # a forced re-login is worse than an unrevokable session — which is
        # the trade, stated plainly, rather than the absolute the old
        # comment implied.
        payload = jwt.decode(
            authorization[7:], JWT_SECRET, algorithms=["HS256"],
            options={"require": ["exp", "uid", "jti"]},
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(401, f"invalid token: {exc}") from exc
    with db() as conn:
        revoked = conn.execute(
            "SELECT 1 FROM revoked_tokens WHERE jti = ?", (payload["jti"],)
        ).fetchone()
    if revoked is not None:
        raise HTTPException(401, "this session has been signed out")
    return payload


def current_user(payload: dict = Depends(current_token)) -> int:
    """FastAPI dependency: resolve the JWT bearer token to a user id.

    The type check is not defensive noise. `uid` is required by
    `current_token`, so it is always present — but a validly-signed token
    whose `uid` is a string, a list or null used to reach `int(...)` and
    raise ValueError/TypeError out of a dependency, which FastAPI turns into
    a 500. A token this server will not accept is a 401; a 500 says the
    server broke, which sends the app's Connection doctor looking for an
    outage that isn't there.
    """
    uid = payload["uid"]
    # Not `isinstance(uid, int)` alone: bool is a subclass of int, and
    # `True` would otherwise authenticate as user 1.
    if type(uid) is not int:
        raise HTTPException(401, "invalid token: uid claim is not a user id")
    return uid


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


def _monitoring_payload(monitoring_json: Optional[str]) -> Optional[dict]:
    """A device's stored monitoring config, or None if it cannot be read.

    Wrapped for the same reason every other JSON-column read in this file is,
    and it was the one that was not. `GET /v1/devices` is the app's busiest
    endpoint — it polls it continuously — so a single unreadable `monitoring`
    value (a legacy row, a partial write, a hand-edited database) turned the
    whole device list into a hard 500 for that account, with no way back short
    of editing the database. The stored value is input like any other; that it
    arrives from our own table rather than from a request body is exactly why
    it was easy to miss.

    Returning None is the same thing an absent value already means, so the app
    needs no new case: it renders as "no monitoring config", not as an error.
    """
    if not monitoring_json:
        return None
    try:
        config = json.loads(monitoring_json)
    except (ValueError, TypeError):
        log.warning("device has an unreadable monitoring config; reporting none")
        return None
    if not isinstance(config, dict):
        return None
    # The app decodes these three as non-optional (Models.swift CustomMonitoring),
    # and one undecodable device fails the decode of the WHOLE device list. The
    # dict arrives from the device verbatim, so its types are not ours to assume.
    try:
        rule_count = int(config.get("rule_count") or 0)
    except (TypeError, ValueError):
        rule_count = 0
    return {
        **config,
        "instruction": str(config.get("instruction") or ""),
        "rule_count": rule_count,
        "active": config.get("active") is True,
    }


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
#: How many times a push/SMS send is attempted before giving up, and the
#: delay between attempts. A transient provider blip (ntfy briefly down, a
#: Twilio rate limit) at the exact moment a genuinely new critical alert
#: lands used to lose that interruption permanently — the event stayed in
#: Recent Activity, but the one thing that reaches someone NOT already
#: looking at the app never arrived. 3 attempts, short backoff: enough to
#: ride out a blip measured in seconds, not so much that a real outage stalls
#: the background task for minutes.
NOTIFY_SEND_ATTEMPTS = 3
NOTIFY_RETRY_BACKOFF_S = 1.5


def _send_with_retry(send_once) -> None:
    """Call `send_once` (a zero-arg callable that raises on failure), retrying
    with a short backoff. Re-raises the LAST failure rather than swallowing
    it — retrying is not the same as succeeding, and a caller that can't tell
    the difference logs a delivery that never actually happened.

    UnicodeEncodeError is excluded from the retry, deliberately. Retrying is
    for a provider that MIGHT answer differently a second later; an encoding
    error is decided entirely by the bytes we are holding, so the second and
    third attempts are guaranteed to fail identically. That is not free: this
    runs on a FastAPI threadpool worker, and three attempts with the backoff
    below occupy one for ~4.5 seconds to reach a conclusion that was already
    available at attempt one. It still raises — the caller must never be told
    a notification was delivered — it just stops pretending the failure is
    transient.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(NOTIFY_SEND_ATTEMPTS):
        try:
            send_once()
            return
        except UnicodeEncodeError:
            raise
        except Exception as exc:  # noqa: BLE001 - genuinely any provider failure is retryable here
            last_exc = exc
            if attempt < NOTIFY_SEND_ATTEMPTS - 1:
                time.sleep(NOTIFY_RETRY_BACKOFF_S * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def _header_value(text: str) -> str:
    """A header-safe carrier for a title that contains non-ASCII characters.

    Every entry in `_EVENT_TITLES` opens with an emoji, and an HTTP header
    value is latin-1 by the spec urllib3 enforces — so `headers={"Title":
    "🔴 Potential Fall Detected"}` raised UnicodeEncodeError *before the
    request left the process*. The effect was exactly inverted delivery:
    every recognised safety event (fall, drowsiness, zone violation — the 22
    types this backend has copy for) failed to send, while an UNRECOGNISED
    type, whose title comes from `_readable_title`'s plain-ASCII fallback,
    was the only kind that got through.

    ntfy's documented convention for this is to send the UTF-8 bytes in the
    header and let the server decode them as UTF-8. Encoding to UTF-8 and
    decoding back as latin-1 produces a str whose code points are exactly
    those bytes, which is what urllib3 then puts on the wire unchanged.
    ASCII titles are unaffected: their UTF-8 bytes are themselves.
    """
    return text.encode("utf-8").decode("latin-1")


def send_push(user_row: sqlite3.Row, title: str, body: str) -> None:
    """Push provider. ntfy today (topic-based, no accounts); replace with APNs for the
    production iOS app (needs an Apple Developer p8 key — see cloud/README.md).

    Raises on failure (after retrying) rather than swallowing it — a caller
    that wants "never breaks the request this ran inside of" wraps this
    explicitly, same as every other fallible call in this file. Swallowing
    it HERE meant a bad ntfy response (a 500, a rejected topic) was
    genuinely invisible: not logged, not retried, not distinguishable from
    success anywhere in the system.
    """
    topic = user_row["ntfy_topic"] if user_row else None
    if not topic:
        return
    import requests

    def once() -> None:
        resp = requests.post(
            f"{NTFY_BASE}/{topic}",
            data=body.encode(),
            headers={"Title": _header_value(title)},
            timeout=5,
        )
        resp.raise_for_status()

    _send_with_retry(once)


def send_sms(user_row: sqlite3.Row, body: str) -> None:
    """SMS to the account's connected phone number on serious alerts.

    Uses Twilio when TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN / TWILIO_FROM are set;
    otherwise logs the message (so the flow is fully exercisable before buying a
    Twilio number). Raises on failure after retrying — see send_push."""
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
    import requests

    def once() -> None:
        resp = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json",
            data={"To": phone, "From": sender, "Body": body},
            auth=(sid, token),
            timeout=10,
        )
        resp.raise_for_status()

    _send_with_retry(once)


def _notify(owner: Optional[sqlite3.Row], title: str, detail: str, device_name: str) -> None:
    """Push, then SMS, as a background task after `ingest()` has already
    responded — a slow or retrying provider must never make a device wait
    for its next poll. Each is independently caught and logged: a push
    failure (after retries) must not cancel the SMS, and vice versa.
    """
    try:
        send_push(owner, title, detail)
    except Exception as exc:  # noqa: BLE001 - see send_push's own docstring
        log.warning("notify: push delivery failed after retries: %r", exc)
    try:
        send_sms(owner, f"{title} — {detail} ({device_name})")
    except Exception as exc:  # noqa: BLE001 - see send_sms's own docstring
        log.warning("notify: sms delivery failed after retries: %r", exc)


# ------------------------------------------------------------------------- alert copy
#: Human title per event type. A push banner reading "SensorDyme: fall" tells
#: nobody whether to be alarmed; unlisted types (a module this table hasn't
#: caught up with) fall back to `_readable_title` below rather than erroring.
_EVENT_TITLES = {
    "fall": "🔴 Potential Fall Detected",
    "drowsiness": "😴 Drowsiness Detected",
    "microsleep_warning": "⚠️ Microsleep Detected",
    "fatigue": "😴 Fatigue Detected",
    "fatigue_critical": "🔴 Critical Fatigue Detected",
    "fatigue_warning": "⚠️ Fatigue Warning",
    "fatigue_pattern": "⚠️ Fatigue Pattern Detected",
    "distraction": "⚠️ Distraction Detected",
    "attention_lost": "⚠️ Attention Lost",
    "eyes_off_road": "⚠️ Eyes Off Road",
    "eyes_down": "⚠️ Eyes Down",
    "frequent_glancing": "⚠️ Frequent Glancing Away",
    "phone_use": "📱 Phone Use Detected",
    "phone_detected": "📱 Phone Detected",
    "phone_in_hand": "📱 Phone In Hand",
    "eating_drinking": "🍔 Eating/Drinking Detected",
    "occupancy_change": "🚶 Occupancy Changed",
    "person_lingering": "🚨 Person Lingering",
    "zone_violation": "🚨 Zone Violation",
    "vehicle_in_zone": "🚗 Vehicle In Zone",
    "animal_in_zone": "🐾 Animal In Zone",
    "choking_hazard": "⚠️ Possible Choking Hazard",
}

#: Detail sentence per type, filled in with the event's own `duration` --
#: real evidence (how long the condition was sustained) that every module
#: already computes (Event.duration) but that used to be silently dropped
#: before reaching here (see core/uplink.py's push_event). Used only when
#: the module didn't already produce descriptive `signals` text of its own
#: (occupancy's zone events do; fall_detection's "fall" does not) -- signals
#: text, when present, is always better evidence than this generic fallback
#: and takes priority. Unlisted types get a generic template further down.
_EVENT_DETAIL_TEMPLATES = {
    "fall": "Person transitioned rapidly and remained down for {duration:.0f} seconds",
    "drowsiness": "Drowsiness signs sustained for {duration:.0f} seconds",
    "microsleep_warning": "Eyes closed long enough to indicate a microsleep ({duration:.0f}s)",
    "fatigue": "Fatigue signs sustained for {duration:.0f} seconds",
    "fatigue_critical": "Severe fatigue signs sustained for {duration:.0f} seconds",
    "fatigue_warning": "Early fatigue signs building for {duration:.0f} seconds",
    "fatigue_pattern": "A recurring fatigue pattern was confirmed over {duration:.0f} seconds",
    "distraction": "Attention off the task for {duration:.0f} seconds",
    "attention_lost": "Attention lost for {duration:.0f} seconds",
    "eyes_off_road": "Eyes off the road for {duration:.0f} seconds",
    "eyes_down": "Eyes down for {duration:.0f} seconds",
    "frequent_glancing": "Repeated glances away sustained for {duration:.0f} seconds",
    "phone_use": "Phone use sustained for {duration:.0f} seconds",
    "phone_detected": "Phone visible for {duration:.0f} seconds",
    "phone_in_hand": "Phone held for {duration:.0f} seconds",
    "eating_drinking": "Eating/drinking sustained for {duration:.0f} seconds",
    "person_lingering": "Someone lingered in a watched zone for {duration:.0f} seconds",
    "zone_violation": "A watched zone was violated for {duration:.0f} seconds",
    "vehicle_in_zone": "A vehicle was in a watched zone for {duration:.0f} seconds",
    "animal_in_zone": "An animal was in a watched zone for {duration:.0f} seconds",
}


def _readable_title(event_type: str) -> str:
    # The title travels as an HTTP header to ntfy; a device-supplied type with a
    # CR/LF in it raised InvalidHeader and tied up a worker through every retry.
    printable = "".join(ch for ch in event_type if ch.isprintable())[:80]
    return _EVENT_TITLES.get(event_type, f"SensorDyme: {printable}")


def _notification_detail(e: dict) -> str:
    """Best available evidence for a notification body.

    Priority: a module's own `signals` text (already descriptive — occupancy
    writes "zone=pool confirmed after 12.4s") beats the generic duration
    fallback beats the bare, contextless "alert" this used to always be.
    """
    signals = str(e.get("signals", "") or "").strip()
    if signals:
        return signals
    duration = e.get("duration")
    try:
        duration = float(duration) if duration is not None else 0.0
    except (TypeError, ValueError):
        duration = 0.0
    if not math.isfinite(duration):
        duration = 0.0
    if duration <= 0:
        return "alert"
    event_type = str(e.get("type", ""))
    template = _EVENT_DETAIL_TEMPLATES.get(event_type)
    if template:
        return template.format(duration=duration)
    return f"{_readable_title(event_type)} — sustained for {duration:.0f} seconds"


# ------------------------------------------------------------------------- schemas
#
# Upper bounds on the fields whose size costs the server real work. Not
# decoration: each of these had a floor ("at least 8 characters", "must be
# non-empty") and no ceiling, so the expensive direction was the unguarded one.
#
# Every limit here is far above any legitimate value, so no real request is
# rejected — the point is to bound the worst case, not to police input. Pydantic
# answers an over-long field with a 422 naming it, before the handler runs.
#
#: PBKDF2 runs 200_000 iterations synchronously on a threadpool worker, on both
#: signup and login. The per-IP limiter bounds one source; nothing bounded the
#: cost of a single attempt. No real password is near this.
MAX_PASSWORD_LEN = 1024
#: RFC 5321's maximum forward-path length.
MAX_EMAIL_LEN = 320
#: Goes straight to a language model, on an authenticated endpoint with no rate
#: limit of its own, so length is token spend.
MAX_AI_PROMPT_LEN = 4000
#: The device's own outbound queue is bounded at 500 events, so this cannot
#: reject a legitimate batch even when that queue is completely full. Set well
#: above it deliberately: core/uplink.py re-queues any non-200 and retries the
#: identical batch, so a cap a real device could hit would not shed load — it
#: would produce a permanent retry loop.
MAX_INGEST_EVENTS = 1000
#: Zones are drawn by hand on a phone screen.
MAX_ZONES = 64
MAX_ZONE_NAME_LEN = 60
MAX_ZONE_POINTS = 256


class Credentials(BaseModel):
    email: str = Field(max_length=MAX_EMAIL_LEN)
    password: str = Field(max_length=MAX_PASSWORD_LEN)


class ClaimBody(BaseModel):
    claim_code: str = Field(max_length=64)


class PresetBody(BaseModel):
    # Re-sent to the device on every poll until it converges, so its size is
    # paid on every ingest response, not once.
    preset: str = Field(max_length=128)


class NtfyBody(BaseModel):
    ntfy_topic: str = Field(max_length=256)


class PhoneBody(BaseModel):
    phone: str = Field(max_length=32)
    sms_on_loud: bool = True


class OrderBody(BaseModel):
    quantity: int
    address: str = Field(max_length=1000)


class RenameBody(BaseModel):
    name: str = Field(max_length=100)


class ZonesBody(BaseModel):
    zones: list[dict] = Field(default=[], max_length=MAX_ZONES)


class IngestBody(BaseModel):
    events: list[dict] = Field(default=[], max_length=MAX_INGEST_EVENTS)
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
    prompt: str = Field(max_length=MAX_AI_PROMPT_LEN)


# ------------------------------------------------------------------------- user API
@app.get("/v1/health")
def health():
    """Unauthenticated liveness probe — the app's Connection doctor hits this to
    distinguish 'wrong address' from 'server down' from 'auth problem'.

    Touches the database: a process that answers while its volume is missing or
    its file unreadable can serve nothing, and reporting it healthy kept Fly
    routing to it. One trivial query, on a connection that is opened anyway.
    """
    try:
        with db() as conn:
            conn.execute("SELECT 1").fetchone()
    except sqlite3.Error as exc:
        log.error("health: database unavailable (%s)", exc)
        return JSONResponse(
            status_code=503,
            content={"ok": False, "service": "sensordyme-cloud", "time": time.time(),
                     "detail": "database unavailable"},
        )
    return {"ok": True, "service": "sensordyme-cloud", "time": time.time()}


def _trusted_client_ip_header() -> str:
    """The header carrying the real client IP, when something in front of this
    process is known to set it. Empty — trust nothing — unless configured.

    Read per call rather than captured at import so a deployment can be
    reconfigured without a code change, and so this is testable.
    """
    return os.environ.get("SENSORDYME_CLIENT_IP_HEADER", "").strip()


def _client_ip(request: Request) -> str:
    """The caller's address, for rate limiting only (never trust this for auth).

    By default this is the direct TCP peer, not an `X-Forwarded-For` header —
    deliberately: a header is client-supplied and trivially spoofed by
    whoever is being rate limited, unless a trusted proxy in front of this
    process is known to overwrite it. Falling back to a fixed key when the
    peer is unknown (some ASGI transports in testing) means every such caller
    shares one bucket rather than the check silently never applying to them.

    Behind a proxy that assumption inverts, and the consequence is not a
    softer limit but a much worse one: the TCP peer is the proxy for EVERY
    request, so all callers share a single 10-attempts-per-5-minutes bucket
    and any one of them can lock every account out of login with ten tries.
    Deployed on Fly, `Fly-Client-IP` is set by Fly's own proxy and overwrites
    whatever the client sent, which is exactly the "known to overwrite it"
    condition the paragraph above reserves this for.

    So it stays opt-in and names the specific header to believe. An
    unconfigured deployment — anything on a LAN, and every test — behaves
    exactly as before, and no header is ever trusted by accident.
    """
    header = _trusted_client_ip_header()
    if header:
        # A proxy chain may append; the first entry is the original client.
        forwarded = request.headers.get(header, "")
        candidate = forwarded.split(",")[0].strip()
        if candidate:
            return candidate
        # Configured but absent means the request did not come through the
        # proxy it was promised to. Fall through to the peer rather than
        # returning "" — one shared bucket is a limit, "" is also a limit, but
        # neither should silently look like a distinct client.
    return request.client.host if request.client is not None else "unknown"


@app.post("/v1/signup")
def signup(body: Credentials, request: Request):
    if not SIGNUP_LIMITER.allow(_client_ip(request)):
        raise HTTPException(429, "too many attempts — try again in a few minutes")
    email = body.email.strip().lower()
    local, _, domain = email.rpartition("@")
    if not local or "@" in local or "." not in domain.strip(".") or len(body.password) < 8:
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
def login(body: Credentials, request: Request):
    if not LOGIN_LIMITER.allow(_client_ip(request)):
        raise HTTPException(429, "too many attempts — try again in a few minutes")
    with db() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (body.email.strip().lower(),)).fetchone()
    if row is None:
        _hash_pw(body.password, _NO_SUCH_USER_SALT)
        raise HTTPException(401, "wrong email or password")
    if not hmac.compare_digest(_hash_pw(body.password, row["salt"]), row["pw_hash"]):
        raise HTTPException(401, "wrong email or password")
    return {"token": _issue_token(row["id"]), "email": row["email"]}


@app.post("/v1/logout")
def logout(payload: dict = Depends(current_token)) -> dict:
    """Revoke the CURRENT token only. Signing out on one device must never
    sign out every device an account is using — this reads the jti straight
    off the token that called it, not anything account-wide."""
    jti = payload.get("jti")
    if jti is not None:
        with db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO revoked_tokens (jti, exp) VALUES (?, ?)",
                (jti, float(payload.get("exp", time.time() + JWT_TTL_S))),
            )
            # Opportunistic cleanup, on the same write this table gets
            # touched anyway, rather than a separate scheduled job for a
            # table that is usually empty or near it.
            conn.execute("DELETE FROM revoked_tokens WHERE exp < ?", (time.time(),))
    return {"ok": True}


#: What ntfy itself accepts as a topic name, and nothing else.
#
# The topic IS the credential: ntfy has no accounts, so anyone who knows the
# string receives that account's alerts. It was stored after nothing but a
# .strip() and then interpolated into `f"{NTFY_BASE}/{topic}"`, so the value
# chose the whole path — "../../anything", a query string, a second path
# segment. That is a request this backend never intended to make, aimed
# wherever the topic said, carrying whatever a safety alert says.
_NTFY_TOPIC_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@app.post("/v1/account/ntfy")
def set_ntfy(body: NtfyBody, uid: int = Depends(current_user)):
    topic = body.ntfy_topic.strip()
    # Empty is how the app turns push off, and stays allowed: send_push
    # returns early on a falsy topic.
    if topic and not _NTFY_TOPIC_RE.match(topic):
        raise HTTPException(
            400,
            "ntfy topic may only contain letters, numbers, - and _ (max 64 characters)",
        )
    with db() as conn:
        conn.execute("UPDATE users SET ntfy_topic = ? WHERE id = ?", (topic, uid))
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
def claim_device(body: ClaimBody, request: Request, uid: int = Depends(current_user)):
    if not CLAIM_LIMITER.allow(_client_ip(request)):
        raise HTTPException(429, "too many attempts — try again in a few minutes")
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
                "monitoring": _monitoring_payload(r["monitoring"]),
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
            "SELECT id, timestamp, type, confidence, signals, severity, source, duration FROM events "
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
        normalized = resolve_preset(body.preset.strip().rstrip(":"))
        # Only a mode that exists. The device ignores an unknown name ("ignoring
        # unknown preset") and never reports it back, so storing one made ingest
        # re-send it on every poll, for ever, with the app showing the switch as
        # pending indefinitely.
        if normalized:
            presets = load_presets()
            base, _, variant = normalized.partition(":")
            if base not in presets or (
                variant and variant not in (presets[base].get("variants") or {})
            ):
                raise HTTPException(400, f"unknown mode '{body.preset}'")
        # An empty preset IS the app's request for AI Custom (AIConfigureView's
        # Apply calls this with preset=""). Anything else is an explicit
        # request to leave custom monitoring in charge, whatever ai_rules
        # still says from a previous configuration.
        conn.execute(
            "UPDATE devices SET desired_preset = ?, custom_monitoring_wanted = ? WHERE id = ?",
            (normalized, 1 if normalized == "" else 0, device_id),
        )
        # This endpoint is the ONLY act of acceptance anywhere in the flow, so
        # it is where a previewed proposal (see /v1/ai/configure, which now
        # only ever writes a draft) becomes the one the device actually runs.
        #
        # Promotion is limited to the AI Custom request. preset == "" is what
        # AIConfigureView's Activate sends for a genuinely custom proposal,
        # and — being the only request that sets custom_monitoring_wanted — the
        # only one that turns rule delivery on at all. A named preset leaves
        # the draft where it is rather than promoting or discarding it, which
        # is the pre-existing contract for `ai_rules`: a configuration is kept
        # across a mode switch so that coming back to Configure-with-AI still
        # shows what was last configured, and delivery is gated on
        # custom_monitoring_wanted rather than on the column being empty.
        #
        # The draft is consumed on promotion, so a second Activate has nothing
        # left to re-promote and `ai_rules` stays the record of what is running.
        if normalized == "":
            draft = conn.execute(
                "SELECT ai_rules_draft FROM devices WHERE id = ?", (device_id,)
            ).fetchone()["ai_rules_draft"]
            if draft:
                conn.execute(
                    "UPDATE devices SET ai_rules = ?, ai_rules_draft = NULL WHERE id = ?",
                    (draft, device_id),
                )
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
        name = str(entry.get("name") or "").strip()[:MAX_ZONE_NAME_LEN]
        kind = str(entry.get("kind") or "restricted")
        points = entry.get("polygon") or []
        if not name or kind not in ("restricted", "safe", "boundary"):
            continue
        # Shipped to the device and point-in-polygon tested every frame; a
        # hand-drawn zone has a handful of points.
        if not isinstance(points, list) or len(points) > MAX_ZONE_POINTS:
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

def _modes() -> tuple[str, ...]:
    """The whole-scenario modes a proposal is allowed to name.

    Read from presets.yaml, never written out here. The hardcoded tuple this
    replaces had drifted to the point of being wrong in both directions: it
    still listed `driver_distraction` and `driver_distraction_openpilot`,
    which were consolidated into one mode and now exist only as aliases, and
    it did not list `driver_safety`, which is the mode that actually exists.

    Both halves of that mattered, because this list does two jobs at once. It
    is the enum in the structured-output schema, so the model could not emit
    `driver_safety` at all; and it is the server-side filter
    (`preset if preset in _MODES else ""`), so a model that emitted it anyway
    was silently downgraded to "no preset". Meanwhile `_ai_fallback` returned
    the superseded alias, which the iOS app's `Preset.catalog` has no entry
    for, so the mode picker rendered the raw string `driver_distraction` at
    the user.

    Deriving it means the list cannot drift from presets.yaml again.
    Superseded names still resolve (see `_resolve_mode`), so a proposal that
    names one is corrected rather than thrown away.
    """
    return tuple(load_presets())


def _resolve_mode(preset: object) -> str:
    """A proposal's `preset` as a mode that exists, or "" for none.

    Aliases are resolved rather than rejected: a model trained on, or a
    prompt quoting, a superseded name means the mode the name now refers to,
    and answering "" would silently discard a correct request.
    """
    name = str(preset or "").strip()
    if not name:
        return ""
    resolved = resolve_preset(name)
    return resolved if resolved in _modes() else ""


#: Snapshot for the schema and prompt strings built at import time. The
#: functions above stay the authority for anything evaluated per request.
_MODES = _modes()

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


# Which model interprets custom requests. Jupiter (a local model) is the
# default, so a fresh install needs no third-party API key to configure a camera
# its owner already paid for. Anthropic remains selectable as a benchmark.
#
# Model/base_url selection, per-role defaults, and the (off-by-default) cloud
# fallback now live in jupiter/router.py — this stays only as the cheap
# short-circuit for /v1/events/{id}/explain: "off" must skip even attempting a
# provider, not just fail to find one, so a deployment that wants facts-only
# explanations never pays for a provider construction + availability probe.
JUPITER_PROVIDER = os.environ.get("SENSORDYME_AI_PROVIDER", "jupiter").strip().lower()


def _ai_configure_llm(prompt: str) -> Optional[dict]:
    """Interpret a request into a structured proposal, or None to fall through.

    Jupiter and Anthropic are interchangeable here on purpose: both return a
    proposal that `_validate_proposal` then re-checks against the capability
    registry. Neither is trusted, so swapping the provider changes accuracy and
    cost, never what the device is willing to run.

    Routed through jupiter/router.py's "configure" role rather than a provider
    name and config assembled inline here — see that module for what decides
    the model (measured accuracy per model, not just whichever was installed
    first) and why the cloud fallback defaults to Sonnet rather than Opus.
    """
    try:
        from jupiter.interpret import interpret
        from jupiter.router import cloud_fallback_config, role_config
    except ImportError:
        # Backend deployed without core/ + jupiter/ on the path. Say nothing
        # here; the caller's fallback explains it to the user.
        return None

    provider_name, config = role_config("configure")
    outcome = interpret(prompt, provider_name=provider_name, config=config)

    fallback = cloud_fallback_config()
    if not outcome.ok and fallback["enabled"] and provider_name != fallback["provider"]:
        # Explicitly enabled, and only after the local path failed.
        outcome = interpret(
            prompt, provider_name=fallback["provider"], config={"model": fallback["model"]}
        )

    if not outcome.ok and not outcome.interpretation.rules:
        return None  # let the caller report why, rather than inventing a preset

    proposal = outcome.interpretation.as_proposal()
    proposal["source"] = outcome.provider or provider_name
    proposal["jupiter"] = {
        "provider": outcome.provider,
        "model": outcome.model,
        "latency_s": outcome.latency_s,
        "injection_suspected": outcome.injection_suspected,
        **outcome.meta,
    }
    return proposal


# Keyword -> (mode, explanation, rules), where a rule is
# (target, predicate, description, persist_s, severity).
#
# The rules are (target, predicate) pairs, which is the shape the device can
# actually execute — `core/rules.py`'s ConfiguredRule reads `target` and
# `predicate`, and evaluates the target as a DETECTION LABEL against each
# frame. They used to be keyed `detection` with EVENT-TYPE names ("fall",
# "phone_use", "zone_violation", "unusual_stillness"): names no detector ever
# produces as a label. `_validate_proposal` read `target`, found it absent,
# and assessed "" — which the capability layer accepts as a whole-frame
# condition — so the rule passed validation, was delivered, was stored, and
# then sat on the device permanently unable to fire. Silently: nothing in the
# app, the backend or the device said the rule was inert.
#
# That is the path taken when the language model is UNREACHABLE, which is
# exactly when it has to be trustworthy. Each pair below is checked against
# `core.capabilities.registry()` by tests/test_phase1_cloud_ai_fallback.py, and
# built through the device's own ConfiguredRule.from_dict, so "the fallback
# only ships rules that can fire" is a property under test rather than a
# claim in a comment.
#
# Predicates are kept to the ones core/rules.py evaluates explicitly
# (`present`, `absent`) rather than the wider set the registry declares, so
# what the description promises is what the device does.
_AI_FALLBACKS = [
    (("baby", "infant", "crib", "nursery", "toddler"), "baby_monitoring",
     "Baby Monitoring: the mode watches for unusual stillness, frantic movement and "
     "safe/danger zones. Added on top: an alert if the baby stops being visible at all. "
     "Draw a safe zone around the crib (and any danger zones) in the app to enable area alerts.",
     [("person", "absent", "Alert if nobody is visible in the camera's view for 30 seconds",
       30.0, "critical")]),
    (("fall", "elder", "senior", "grandma", "grandpa"), "elderly_monitoring",
     "Elderly Monitoring: immediate fall alerts (rapid drop then stillness — sitting down "
     "won't trigger it), staying active until movement resumes. Added on top: an alert if "
     "someone is on the floor, which also covers a slow slide down that never looks like a drop.",
     [("on_floor", "present", "Alert if someone is on the floor for more than 5 seconds",
       5.0, "critical")]),
    (("security", "intruder", "door", "burglar", "lurk", "linger", "trespass"), "security_monitoring",
     "Security: person-count changes, plus restricted-zone and lingering alerts once you draw "
     "zones in the app. Added on top: an alert for anyone who stays in view for half a minute, "
     "which needs no zone drawn to work.",
     [("person", "present", "Alert when someone stays in view for 30 seconds",
       30.0, "critical")]),
    (("drive", "driving", "car", "phone", "drowsy", "road"), "driver_safety",
     "Driver Safety: loud alerts for phone use while looking away and sustained eye closure. "
     "Added on top: an alert whenever a phone is visible for more than a couple of seconds.",
     [("cell phone", "present", "Alert when a phone is visible for more than 2 seconds",
       2.0, "critical")]),
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
                    {"target": target, "predicate": predicate, "secondary": "",
                     "description": desc, "zone": "", "persist_s": persist,
                     "min_count": 1, "confidence": 0.6, "severity": sev, "cooldown_s": 60.0,
                     "schedule": "", "notify": "push"}
                    for target, predicate, desc, persist, sev in rules
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
    preset = _resolve_mode(data.get("preset"))
    valid_rules, unsupported = [], list(data.get("unsupported") or [])

    for rule in data.get("rules") or []:
        # Resolve the target the way the DEVICE resolves it — `target`, then
        # the legacy `detection` key — because that is the string it will
        # look for as a detection label (core/rules.py, ConfiguredRule.
        # from_dict). Assessing only `target` is what let dead rules through:
        # a rule carrying `detection` and no `target` was assessed as "",
        # which the capability layer accepts as a whole-frame condition, so
        # it passed here and was then evaluated against a label no detector
        # produces. A rule with NEITHER is dropped outright by from_dict, so
        # it can never fire under any circumstances.
        target = str(rule.get("target") or rule.get("detection") or "").strip().lower()
        predicate = str(rule.get("predicate", "")).strip().lower() or "present"
        secondary = str(rule.get("secondary", "")).strip().lower()
        described = str(rule.get("description") or f"{predicate} {target}".strip())

        if not target:
            # Refused, not silently kept. A rule the device discards is worse
            # than a rule the user is told it cannot run: the first looks like
            # protection and is not.
            unsupported.append({
                "request": described,
                "reason": "this rule does not name anything to watch, so the device "
                          "would have nothing to look for.",
                "requires": "an object, posture or activity from the capability sheet",
                "support": caps.Support.UNSUPPORTED.value,
                "alternatives": [],
            })
            continue

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

        # Predicates that are meaningless without their second half. The
        # registry check above cannot see this -- it assessed "person" and
        # "in_zone" independently and both pass on their own.
        #
        # This duplicates jupiter/schema.py:145-156 ON PURPOSE. Until now that
        # was the ONLY place the check existed: this function did not look at
        # `zone` at all, and neither does core/rules.py's ConfiguredRule
        # .from_dict. So the property "an area rule only fires in its area" was
        # enforced exclusively inside the AI layer -- swap the interpreter and
        # it leaves with it, silently. An application safety rule may not be
        # delegated to the thing it is checking.
        zone = str(rule.get("zone") or "").strip()
        if predicate == "in_zone" and not zone:
            unsupported.append({
                "request": described,
                "reason": "'in_zone' needs a zone to watch, and none was named",
                "requires": "a zone the user draws in the app",
                "support": "unsupported",
                "alternatives": [],
            })
            continue
        if predicate in ("near", "holding", "near_mouth") and not secondary:
            unsupported.append({
                "request": described,
                "reason": f"'{predicate}' needs a second object, and none was named",
                "requires": "a second object for the relation",
                "support": "unsupported",
                "alternatives": [],
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
    return _decodable_proposal(data)


def _decodable_proposal(data: dict) -> dict:
    """Fill every field the app decodes as non-optional (BackendAPI.swift
    AIRule / AIUnsupported / AIProposal.settings).

    The rules come from a language model; one rule missing `zone` or `notify`
    made the app's decode of the WHOLE proposal fail, so the person saw an
    error instead of the rules that were fine.
    """
    rules = []
    for rule in data.get("rules") or []:
        if not isinstance(rule, dict):
            continue
        for key in ("secondary", "description", "zone", "schedule"):
            rule[key] = str(rule.get(key) or "")
        if rule.get("notify") not in ("push", "sms", "silent"):
            rule["notify"] = "push"
        rules.append(rule)
    data["rules"] = rules
    data["unsupported"] = [
        {**item, **{key: str(item.get(key) or "") for key in ("request", "reason", "requires")}}
        for item in (data.get("unsupported") or []) if isinstance(item, dict)
    ]
    settings = data.get("settings")
    data["settings"] = (
        {str(k): str(v) for k, v in settings.items()} if isinstance(settings, dict) else {}
    )
    return data


def _legacy_validate_proposal(data: dict) -> dict:
    """Taxonomy-based validation, for a backend running without core/ available."""
    tax = _taxonomy()
    preset = _resolve_mode(data.get("preset"))
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
    return _decodable_proposal(data)


@contextmanager
def _ai_slot(uid: int):
    """Yield whether this account may make a model call right now.

    Model calls are synchronous and can take up to a minute, on a shared thread
    pool. Without a per-account bound one account could occupy every worker and
    stall every other request on the server, devices' ingest included.
    """
    key = str(uid)
    # Concurrency first: a call refused for being one too many at once must not
    # also spend the per-minute budget, or a burst locks out the next real one.
    if not AI_CONCURRENCY.acquire(key):
        yield False
        return
    try:
        yield AI_RATE_LIMITER.allow(key)
    finally:
        AI_CONCURRENCY.release(key)


@app.post("/v1/ai/configure")
def ai_configure(body: AIConfigureBody, uid: int = Depends(current_user)):
    with db() as conn:
        _owned_device(conn, body.device_id, uid)
    with _ai_slot(uid) as allowed:
        if not allowed:
            raise HTTPException(429, "too many AI requests at once; try again in a moment")
        result = _validate_proposal(_ai_configure_llm(body.prompt) or _ai_fallback(body.prompt))
    # Persist the validated proposal as a DRAFT, so the app's Activate
    # (set_preset) has it to promote.
    #
    # Not into `ai_rules`, which is what this did and what made previewing
    # destructive. This endpoint is the PREVIEW: AIConfigureView calls it from
    # `configure()`, the moment the person stops typing, before the proposal
    # has been rendered and long before Activate. `ai_rules` is the column the
    # ingest handler delivers from, and on a device already in AI Custom
    # (custom_monitoring_wanted stays set — only set_preset ever writes it, so
    # a preview cannot clear it) the delivery gate is a content-hash mismatch,
    # which a preview flips by definition. So the next 2-second poll shipped
    # and committed a configuration the user was still reading, and the device
    # hot-swapped its detection module to match. Writing a draft leaves the
    # running configuration exactly where it is until someone accepts this one.
    # Scoped to the owner, not just to the id. Ownership was checked before the
    # model call above, and that call takes seconds — long enough for the device
    # to be re-claimed in the gap. Every other write in this file is gated by
    # `_owned_device` or a device-key lookup inside the same transaction; this was
    # the one that was not, so a claim landing mid-request wrote a draft onto a
    # device the caller no longer owned. Putting `owner_id` in the WHERE clause
    # makes the check and the write one atomic statement rather than two facts
    # that have to still agree.
    with db() as conn:
        updated = conn.execute(
            "UPDATE devices SET ai_rules_draft = ? WHERE id = ? AND owner_id = ?",
            (json.dumps(result), body.device_id, uid),
        ).rowcount
    if not updated:
        # Same 404 and same wording as `_owned_device`, so losing the device
        # mid-request is indistinguishable from never having had it — there is
        # nothing here worth telling an attacker apart.
        raise HTTPException(404, "no such device on this account")
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
        raise HTTPException(503, "explanations are unavailable on this backend") from None

    event = dict(row)
    rule = parse_rule_for_event(device["ai_rules"], event.get("type", ""))

    # Inside the per-account model budget, availability probe included: the
    # probe alone can take seconds against an unreachable model. Over budget,
    # explain from the facts alone -- this endpoint's contract is that it has no
    # failure mode beyond the event not existing, so the limit costs phrasing,
    # never an answer.
    with _ai_slot(uid) as allowed:
        provider = None
        if allowed and JUPITER_PROVIDER not in ("", "off"):
            try:
                from jupiter.router import get_role_provider

                candidate = get_role_provider("explain")
                ok, why = candidate.available()
                provider = candidate if ok else None
                if not ok:
                    # Degrading to the facts-only explanation is correct and must not
                    # fail the request — but it has to be *visible*. Reported at INFO
                    # because a model being unavailable is an expected state, not a
                    # fault, and it says which one so the reason is actionable.
                    log.info("explain: model unavailable (%s); using the facts-only path", why)
            except Exception as exc:  # noqa: BLE001 - the facts path never needs a model
                # Previously silent, which made a permanently misconfigured provider
                # indistinguishable from one deliberately switched off: every explain
                # quietly took the facts path for ever and no log line ever said so.
                log.warning(
                    "explain: could not reach the model provider (%s); using the "
                    "facts-only path", repr(exc),
                )
                provider = None
        return explain(event, rule, provider=provider).as_dict()


# ------------------------------------------------------------------------- device API
@app.post("/v1/ingest")
def ingest(body: IngestBody, background_tasks: BackgroundTasks, device: sqlite3.Row = Depends(current_device)):
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
        newly_inserted_events = []
        # No cap here on purpose: the device's own queue is bounded
        # (core/uplink.py's `queue_max`, default 500), so a batch is already
        # naturally small, and a partial batch used to be silently dropped —
        # the client clears its queue on ANY 200 response (uplink.py:180),
        # so a `[:200]` slice here meant events 201+ of a large backlog (a
        # long outage, then reconnect) were discarded with `stored` still
        # claiming the full count arrived. One SQLite transaction over the
        # whole list costs single-digit milliseconds even at the 500-event
        # ceiling; there is no real cost to processing all of it.
        for e in body.events:
            # OR IGNORE + the partial unique index on (device_id, event_key):
            # a device that treated a lost response as a failed send retries
            # the identical batch next poll. Without this, that retry is a
            # second row for the same alert — a duplicate in Recent Activity
            # and a second notification for something that already fired.
            # `event_key` is None (-> SQL NULL) for firmware that predates
            # it; NULL never collides with itself, so those rows insert
            # exactly as before — no behavior change for an older device.
            #
            # Every field is coerced defensively and the whole thing wrapped
            # per-event, not per-batch. `events: list[dict]` is unvalidated —
            # Pydantic checks the list holds dicts, never what's inside them —
            # so a malformed value (a string where confidence needs a float)
            # is a real, reachable input, not a hypothetical one. Reproduced
            # directly: one bad `confidence` raised an uncaught ValueError
            # INSIDE this loop, which propagated out of `with db()` before
            # `conn.commit()` ran — losing every good event in the same batch,
            # not just the bad one. Worse than losing one event: the device's
            # uplink treats the resulting 500 as a failed send and re-queues
            # the identical batch next poll — so the same malformed event
            # would 500 again, forever, permanently blocking every event
            # behind it in the queue. One skipped event, logged, is the
            # entire cost of getting this wrong; the alternative was total
            # queue deadlock.
            try:
                event_key = e.get("event_key")
                confidence_raw = e.get("confidence", 0.0)
                confidence = float(confidence_raw) if confidence_raw not in (None, "") else 0.0
                duration_raw = e.get("duration")
                duration = float(duration_raw) if duration_raw not in (None, "") else None
                # float() accepts "nan", "inf" and JSON's NaN / 1e999. Stored,
                # they make every later read of this device's events a 500:
                # the response encoder refuses non-finite floats.
                if not math.isfinite(confidence):
                    confidence = 0.0
                if duration is not None and not math.isfinite(duration):
                    duration = None
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO events (device_id, timestamp, type, confidence, signals,"
                    " severity, source, received_at, event_key, duration) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        device["id"],
                        str(e.get("timestamp", "")),
                        str(e.get("type", "unknown")),
                        confidence,
                        str(e.get("signals", "")),
                        str(e.get("severity", "log")),
                        str(e.get("source") or "module"),
                        now,
                        str(event_key) if event_key not in (None, "") else None,
                        duration,
                    ),
                )
                # rowcount is 0 when OR IGNORE silently dropped a retry of an
                # event_key already stored — that upload is not new information,
                # so it must not reach the notification loop below. Without this
                # check that loop re-evaluated every event in `body.events`
                # regardless of whether storage actually inserted a row, so a
                # retried batch (the device's own recovery path after a lost
                # response) could notify a second time for an alert already
                # notified on its first arrival — the exact duplicate this
                # gate exists to prevent, just moved from "database row" to
                # "notification".
                if cursor.rowcount > 0:
                    newly_inserted_events.append(e)
            except (TypeError, ValueError, sqlite3.Error) as exc:
                log.warning(
                    "ingest: skipped one malformed event from device %s (%s): key=%r type=%r",
                    device["id"], exc,
                    str(e.get("event_key"))[:64] if isinstance(e, dict) else None,
                    str(e.get("type"))[:64] if isinstance(e, dict) else None,
                )
                continue
        row = conn.execute(
            "SELECT desired_preset, ai_rules, zones, custom_monitoring_wanted FROM devices WHERE id = ?",
            (device["id"],),
        ).fetchone()
        desired, ai_rules_json, zones_json = row["desired_preset"], row["ai_rules"], row["zones"]
        custom_monitoring_wanted = bool(row["custom_monitoring_wanted"])
        # A device can report the same active_preset string while AI Custom
        # Monitoring runs on top of it — entering custom monitoring replaces the
        # running module without touching active_preset, so the field still names
        # whatever preset was selected before. A desired_preset equal to that
        # name is genuinely NOT fulfilled yet in that case: the device still has
        # to exit custom monitoring, which is real work "same string" hides.
        device_reports_custom_active = bool((body.monitoring or {}).get("active"))
        # Convergence: once the device reports the desired preset as active AND
        # not layered under custom monitoring, the command is fulfilled — clear
        # it. Without the first half, a stale desired_preset keeps re-delivering
        # forever and FIGHTS local mode switches made on the device/dashboard
        # (the device flips back within one poll interval). Without the second
        # half, asking to return to the preset that was active before custom
        # monitoring was turned on never converges — the string already matches
        # today, so the command is dropped before the device ever sees it.
        # Compare by mode, not by string. A device handed a superseded name
        # applies — and correctly reports — the current one, so raw equality
        # never matches: the backend concludes the command was ignored and
        # re-sends it every poll, fighting whatever mode the user picks locally.
        # This also converges rows written before names were normalised.
        if (
            desired
            and same_preset(desired, body.active_preset or device["active_preset"])
            and not device_reports_custom_active
        ):
            # Compare-and-swap, not an unconditional clear: `desired` was read
            # a few statements ago, and this whole request handler runs in
            # FastAPI's threadpool alongside every other one, including
            # POST .../preset (which writes a fresh desired_preset from a
            # user's tap). An unconditional UPDATE here would clear whatever
            # is in the column NOW, which is a stale write if a new command
            # landed in that gap -- silently discarding a mode switch the
            # user just made, with nothing to indicate it happened. Matching
            # the WHERE to the value just read makes the clear a no-op
            # instead whenever that's happened, leaving the fresh command
            # alone.
            conn.execute(
                "UPDATE devices SET desired_preset = NULL WHERE id = ? AND desired_preset = ?",
                (device["id"], desired),
            )
            desired = None

    # Push is the path that works when the app is closed, so it is the path
    # that most needs a filter. It had none: every "loud" event became a push,
    # which on a real device is around 960 in two days. The app filtered
    # locally, the server did not, so the channel that reaches someone when
    # they are NOT looking was the unfiltered one.
    #
    # Iterates `newly_inserted_events`, not `body.events` again, for two
    # reasons at once. First: every event THAT WAS STORED also still appears
    # in the timeline, so a notification must never fire for one that was
    # never written — "an alert was sent without being recorded" is exactly
    # the bug that guards against. Second, and the reason it is
    # `newly_inserted_events` rather than the full insert-attempted slice: a
    # retry of an already-stored event_key is OR-IGNOREd (rowcount 0) because
    # it is not new information, and it must not be new interruption either —
    # otherwise the device's own retry-on-lost-response recovery would
    # duplicate the notification for an alert that already fired, which is
    # the identical duplicate bug at the notification layer instead of the
    # database layer. Distinct alerts still each get their own notification:
    # each has its own event_key and therefore its own inserted row.
    #
    # A 20-second global cooldown per account on top of that — product spec,
    # stated explicitly: at most one notification every 20 seconds, with
    # every other qualifying alert in that window still stored and still
    # shown in Recent Activity, only the notification withheld. This is a
    # DIFFERENT question from the event_key filtering above and does not
    # replace it: `newly_inserted_events` keeps a retried duplicate from ever
    # reaching this loop at all (there would be nothing distinct to notify
    # about); the cooldown then rate-limits distinct, genuinely-new alerts
    # against each other. Both are always active together.
    for e in newly_inserted_events:
        owner_id = owner["id"] if owner is not None else None
        send, held = gate.evaluate(
            owner_id, str(e.get("severity", "")), source=str(e.get("source") or "module")
        )
        if not send:
            continue
        title = _readable_title(str(e.get("type", "alert")))
        detail = _notification_detail(e)
        if held:
            detail += f" (and {held} more in the last {int(NOTIFY_COOLDOWN_S)}s)"
        # A background task, not a direct call: the event is already
        # committed by this point, so neither a slow provider nor _notify's
        # own retries (see NOTIFY_SEND_ATTEMPTS) should make the DEVICE wait
        # for its response — this loop can queue several of these per batch,
        # and the device's next poll is what everything else here is trying
        # to keep prompt. `owner`/`device["name"]` are captured now, by
        # value, since the request-scoped `device` row (and the connection
        # that produced it) will be gone by the time this actually runs.
        background_tasks.add_task(_notify, owner, title, detail, device["name"])

    command = None
    # Same reasoning as the convergence check above: a matching name is not
    # "nothing to do" while the device still reports custom monitoring active —
    # the device has to exit custom monitoring, so the command must still go out.
    #
    # Deliberately truthy, not `is not None`: "" (AI Custom's sentinel, see
    # POST .../preset) is not a name to converge on — active_preset names the
    # underlying module and never becomes "" itself, so `desired != active_preset`
    # would be true forever and re-send this every poll, indefinitely. Custom
    # monitoring is activated by custom_monitoring_wanted + ai_rules below
    # instead, which converges on the right signal (monitoring.active) — see
    # tests/test_rule_delivery.py's coexistence test for why this one has no
    # set_preset command at all.
    if desired and (
        desired != (body.active_preset or device["active_preset"])
        or device_reports_custom_active
    ):
        command = {"set_preset": desired}

    # Custom rules are configured in the app but enforced on the device, so they
    # have to travel down this channel — without it a saved proposal changed the
    # text in the app and nothing about what the camera actually watched for.
    #
    # Gated on `custom_monitoring_wanted`, not just on the version mismatch.
    # `ai_rules` is never cleared by switching to a normal preset — reopening
    # Configure-with-AI later must still show what was last configured — so
    # the version comparison alone can't tell "not delivered yet" apart from
    # "the user asked for a different preset a while ago". Without this gate,
    # switching away from AI Custom did not stick: the very next poll saw the
    # same version mismatch, resent set_rules, and the device reapplied it —
    # pulling itself back into custom monitoring within one poll interval of
    # every switch, for any account that had ever used Configure-with-AI.
    version = _rules_version(ai_rules_json)
    if custom_monitoring_wanted and version and version != (body.rules_version or ""):
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

    # The count that actually landed in the database, not the size of the
    # request -- these silently diverged before (see the `[:200]` removal
    # above): a batch could report every event "stored" while most of them
    # never reached a row. A malformed-and-skipped or already-stored
    # (deduplicated retry) event legitimately isn't counted here; both are
    # accounted for elsewhere (a warning log, and the dedup this endpoint
    # relies on for retry-safety, respectively) -- this field means "how many
    # of these are now durable", which is the number a caller actually needs.
    return {"ok": True, "stored": len(newly_inserted_events), "command": command}
