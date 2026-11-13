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
import dataclasses
import hashlib
import hmac
import ipaddress
import json
import logging
import math
import os
import re
import secrets
import sqlite3
import threading
import time
from contextlib import asynccontextmanager, contextmanager, suppress
from pathlib import Path
from typing import Any, Optional

import jwt
import yaml
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from cloud import mdns
from cloud.notify import COOLDOWN_S as NOTIFY_COOLDOWN_S
from cloud.notify import gate
from cloud.ratelimit import (
    AI_CONCURRENCY,
    AI_GLOBAL_CONCURRENCY,
    AI_GLOBAL_KEY,
    AI_RATE_LIMITER,
    CLAIM_FAILURE_LIMITER,
    CLAIM_LIMITER,
    LOGIN_FAILURE_LIMITER,
    LOGIN_LIMITER,
    SIGNUP_LIMITER,
)
from core import command_auth as sd_auth
from core import health_stamp
from core.logging_setup import describe_failure, mask_phone
from core.presets import (
    SETTINGS_SEPARATOR,
    PresetError,
    catalog_entries,
    catalog_version,
    device_availability,
    effective_settings,
    encode_settings,
    load_planned,
    load_presets,
    parse_settings,
    resolve_preset,
    same_preset,
    validate_settings,
)

DB_PATH = os.environ.get("SENSORDYME_DB", str(Path(__file__).parent / "sensordyme.db"))
JWT_TTL_S = 30 * 24 * 3600
NTFY_BASE = os.environ.get("SENSORDYME_NTFY_BASE", "https://ntfy.sh")
#: How devices authenticate (sd-auth-v1, docs/DEVICE_AUTH_CONTRACT.md).
#: permissive (default): the bearer X-Device-Key still works, signed requests
#: too, and commands are signed whenever the device sends a nonce. required:
#: bearer-key requests are refused (production). off: never sign (rollback only).
#: Anything else stops the server: a typo must not silently mean "no auth".
DEVICE_AUTH_MODE = os.environ.get("SENSORDYME_DEVICE_AUTH", sd_auth.DEFAULT_MODE).strip().lower()
if DEVICE_AUTH_MODE not in sd_auth.MODES:
    raise RuntimeError(f"SENSORDYME_DEVICE_AUTH must be one of {sd_auth.MODES}, got {DEVICE_AUTH_MODE[:32]!r}")

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
            # Created 0600 below, but a restored backup or a copied data
            # directory keeps whatever mode it arrived with, and anyone who can
            # read this file can mint a session for any account (M8).
            with suppress(OSError):
                if path.stat().st_mode & 0o077:
                    os.chmod(path, 0o600)
                    logging.getLogger("uvicorn.error").warning(
                        "%s was readable by other users; its mode is now 0600", path)
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


class _NoStore:
    """`Cache-Control: no-store` on every /v1 answer (M8.3, SECURITY_REVIEW_M6 §6).

    The answers are one account's cameras, alert history and explanations, and
    the commands a device is to obey. Without a cache header the app's
    URLSession.shared kept each GET answer in its on-disk URLCache (reproduced:
    cached with no header, not cached with no-store), where it outlived sign-out
    and an account change on a shared phone; any HTTP cache in front of the
    backend could equally serve one account's answer to another request.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or not scope.get("path", "").startswith("/v1/"):
            await self.app(scope, receive, send)
            return

        async def send_no_store(message) -> None:
            if message["type"] == "http.response.start":
                headers = [(k, v) for k, v in message.get("headers", []) if k.lower() != b"cache-control"]
                message = {**message, "headers": [*headers, (b"cache-control", b"no-store")]}
            await send(message)

        await self.app(scope, receive, send_no_store)


app.add_middleware(_NoStore)


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


class CodedError(HTTPException):
    """An HTTP refusal the app must tell apart from others with the same status.

    Answered as {"detail": "<human sentence>", "code": "<machine code>"}:
    `detail` stays the plain string every client already shows, and `code` is
    what a client branches on (docs/APP_STATUS_CONTRACT.md lists the codes).
    Several different refusals share 409 -- a role forbidding a mode, a
    superseded proposal, firmware that cannot stop -- and the app used to treat
    every 409 as "the proposal changed".
    """

    def __init__(self, status_code: int, code: str, detail: str) -> None:
        super().__init__(status_code, detail)
        self.code = code
        #: More fields for the answer body (a device clock_skew's signed time).
        self.extra: dict = {}


@app.exception_handler(CodedError)
async def _coded_error(_request: Request, exc: CodedError) -> JSONResponse:
    content = {"detail": exc.detail, "code": exc.code}
    if exc.code == "clock_skew":
        content["server_time"] = int(time.time())
    content.update(getattr(exc, "extra", None) or {})
    return JSONResponse(status_code=exc.status_code, content=content)


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


@app.exception_handler(sqlite3.OperationalError)
async def _storage_unavailable(_request: Request, exc: sqlite3.OperationalError) -> JSONResponse:
    """A full disk, an I/O error or a lock held past BUSY_TIMEOUT_S: the
    request was fine and the server could not store it right now.

    503, not 500 and never a 4xx: core/uplink.py re-queues a batch on any
    non-4xx answer and drops it on most 4xx ones, so this is the status that
    keeps a device's events instead of discarding them. The transaction this
    happened in was never committed (`db()` commits only on a clean exit), so
    nothing half-written is left behind for the retry to collide with.
    """
    log.error("storage unavailable: %s", exc)
    return JSONResponse(status_code=503, content={"detail": "storage temporarily unavailable"})


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
            # A one-shot, explicitly confirmed stop/resume for the device (M0.1).
            # Cleared once the device reports the matching state, so a device
            # that reboots comes back protecting instead of being re-stopped.
            "ALTER TABLE devices ADD COLUMN desired_protection TEXT",
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
            # Which model produced the event: module + detector name, sha256
            # and backend, as JSON (see _model_identity). NULL for history and
            # for firmware that predates it — never back-filled with a guess.
            "ALTER TABLE events ADD COLUMN model TEXT",
            # The hardware's own id (SD-XXXXXXXX, the one it advertises over
            # Bonjour), from its health report. The app's only way to know which
            # camera on the LAN is which backend device. NULL until reported.
            "ALTER TABLE devices ADD COLUMN hardware_id TEXT",
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
            # How many times AI Custom has been activated on this device. Mixed
            # into the rules fingerprint (see _rules_version) so activating the
            # SAME rules again is still a delivery: a device that left custom
            # monitoring for a named preset keeps reporting the old version
            # (core/uplink.py never resets it), so an unchanged rule set used to
            # match and nothing was sent -- the app said AI Custom, the device
            # stayed in the preset.
            "ALTER TABLE devices ADD COLUMN ai_rules_activation INTEGER DEFAULT 0",
            # The draft_version that produced the current `ai_rules`, so a
            # retried Activate of the proposal already running is recognised
            # rather than refused as stale (see set_preset).
            "ALTER TABLE devices ADD COLUMN ai_rules_promoted_from TEXT",
            # The last mode request the backend itself dropped, and why (JSON:
            # preset, code, reason, at). Ingest drops a stored desired_preset
            # the device's role forbids; without a record the app's switch just
            # vanished. Cleared by the next accepted mode request.
            "ALTER TABLE devices ADD COLUMN preset_rejection TEXT",
            # sd-auth-v1 (core/command_auth.py, docs/DEVICE_AUTH_CONTRACT.md):
            # the public key id of device_key, so a signed request -- which never
            # carries the key -- can find its device. And the key being rotated
            # out, still accepted for signed requests until revoked
            # (`cloud/manage.py rotate-key` / `revoke-previous-key`).
            "ALTER TABLE devices ADD COLUMN auth_kid TEXT",
            "ALTER TABLE devices ADD COLUMN previous_device_key TEXT",
            "ALTER TABLE devices ADD COLUMN previous_auth_kid TEXT",
            # M7 companion monitoring nodes (docs/M7_IMPLEMENTATION_MAP.md): the
            # node states the owner asked for and the device has not reported yet
            # (JSON {node: {enabled, settings}}), delivered as `set_nodes` until
            # the device's perception.nodes agree, then cleared like desired_preset.
            "ALTER TABLE devices ADD COLUMN desired_nodes TEXT",
            # The last node request the device refused (JSON: node, code, reasons, at).
            "ALTER TABLE devices ADD COLUMN nodes_rejection TEXT",
            # Fingerprints of the device's last_rejection at the moment each kind
            # of request (preset, nodes) was written, JSON {kind: fingerprint}.
            # A refusal counts against a request only if it is NEWER than that
            # baseline: last_rejection persists on the device, and a refusal of
            # an earlier request for the same name must not cancel a new one.
            # Compared by content, not by clock -- the device's clock is its own.
            "ALTER TABLE devices ADD COLUMN rejection_baseline TEXT",
            # Health freshness (M8.1, core/health_stamp.py): which sample the stored
            # `health` is (boot id, generation, revision), when the BACKEND first
            # received that generation, when the device observed it (on the
            # backend's clock: received_at - the device's own monotonic age), how
            # many times the same sample was re-sent, the boot ids already
            # superseded by a reboot (JSON list), and the last report refused as
            # older than the stored one (JSON {reason, at, boot_id, generation}).
            "ALTER TABLE devices ADD COLUMN health_boot TEXT",
            "ALTER TABLE devices ADD COLUMN health_generation INTEGER",
            "ALTER TABLE devices ADD COLUMN health_revision INTEGER",
            "ALTER TABLE devices ADD COLUMN health_received_at REAL",
            "ALTER TABLE devices ADD COLUMN health_observed_at REAL",
            "ALTER TABLE devices ADD COLUMN health_repeats INTEGER DEFAULT 0",
            "ALTER TABLE devices ADD COLUMN health_boots TEXT",
            "ALTER TABLE devices ADD COLUMN health_ignored TEXT",
            # M8.3: the last stop/resume the app asked for and what became of it
            # (JSON {id, action, requested_at, applied_at, applied_state, ...}).
            # Unlike desired_protection it is never cleared, so the status can
            # say confirmed / mismatch / timed_out after delivery ends.
            "ALTER TABLE devices ADD COLUMN protection_request TEXT",
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
        # For retention (see _prune_events): an age cut-off without this is a
        # full scan of the largest table, on a write transaction.
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_received ON events(received_at)")
        _init_device_auth_tables(conn)
        _backfill_auth_kids(conn)


def _backfill_auth_kids(conn) -> None:
    """Fill auth_kid for device rows created without one (older rows, and rows
    inserted by tools that predate sd-auth-v1). Keys under 16 bytes get none:
    they can only use the legacy bearer path."""
    rows = conn.execute(
        "SELECT id, device_key, previous_device_key FROM devices"
        " WHERE auth_kid IS NULL OR (previous_device_key IS NOT NULL AND previous_auth_kid IS NULL)"
    ).fetchall()
    for row in rows:
        kids = []
        for secret in (row[1], row[2]):
            try:
                kids.append(sd_auth.key_id(secret) if secret else None)
            except ValueError:
                kids.append(None)
        conn.execute("UPDATE devices SET auth_kid = ?, previous_auth_kid = ? WHERE id = ?", (*kids, row[0]))


def _init_device_auth_tables(conn) -> None:
    """Device-auth state (docs/DEVICE_AUTH_CONTRACT.md §3, §6): request nonces,
    revoked keys, per-device enforcement and the last refusal."""
    conn.execute("CREATE TABLE IF NOT EXISTS auth_nonces (nonce TEXT PRIMARY KEY, seen_at REAL NOT NULL)"
                 " WITHOUT ROWID")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_auth_nonces_seen ON auth_nonces(seen_at)")
    # A revoked key is kept, so a request from its holder is recognised and
    # reported as revoked rather than as an unknown key. It authenticates
    # nothing: no lookup that grants access reads this table.
    conn.execute("CREATE TABLE IF NOT EXISTS revoked_device_keys (kid TEXT PRIMARY KEY,"
                 " device_id INTEGER NOT NULL, device_key TEXT NOT NULL, revoked_at REAL NOT NULL)")
    for statement in (
        # 'required': this device must sign, whatever SENSORDYME_DEVICE_AUTH says
        # (it can only tighten). NULL follows the backend mode.
        "ALTER TABLE devices ADD COLUMN auth_mode TEXT",
        # Set by `manage.py revoke-device`: every request from it is refused.
        "ALTER TABLE devices ADD COLUMN auth_revoked_at REAL",
        # 'signed' or 'bearer': how the device last authenticated.
        "ALTER TABLE devices ADD COLUMN last_auth TEXT",
        # JSON {reason, at}: the last refusal attributable to this device.
        "ALTER TABLE devices ADD COLUMN auth_refusal TEXT",
    ):
        try:
            conn.execute(statement)
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise


init_db()
_restrict_db_permissions()


# ------------------------------------------------------------------------- auth helpers
def _hash_pw(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000).hex()


#: Hashed against when the email is unknown, so a login for an account that does
#: not exist costs what one for an account that does costs. Skipping PBKDF2 for
#: an unknown email answered ~20x faster, which says which emails have accounts.
_NO_SUCH_USER_SALT = secrets.token_hex(16)


def _issue_token(user_id: int, email: str) -> str:
    """A session for `user_id`, bound to the account's email.

    `uid` alone is a row number, and row numbers are not identities across
    databases: replace the database while keeping the signing key (a Fly
    import over a deployment someone already signed in to, or a local
    `cloud/sensordyme.db` lost while `.jwt_secret` survived) and every live
    token silently became whoever holds that row number in the new file --
    full access to a stranger's cameras. `current_token` checks the email
    against the row, so such a token is refused instead.
    """
    return jwt.encode(
        {
            "uid": user_id,
            "email": email,
            "exp": int(time.time()) + JWT_TTL_S,
            "jti": secrets.token_hex(16),
        },
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
        #
        # `email` is required for the same kind of reason, and it is the same
        # kind of deliberate one-time cost: tokens issued before it existed are
        # refused, and their owners sign in once more. See _issue_token for
        # what an unbound uid allowed.
        payload = jwt.decode(
            authorization[7:], JWT_SECRET, algorithms=["HS256"],
            options={"require": ["exp", "uid", "jti", "email"]},
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(401, f"invalid token: {exc}") from exc
    uid = payload["uid"]
    # Checked here, before uid reaches a query: a validly-signed token whose
    # `uid` is a string, a list or null used to raise out of a dependency,
    # which FastAPI turns into a 500. Not `isinstance(uid, int)` alone: bool is
    # a subclass of int, and `True` would otherwise authenticate as user 1.
    if type(uid) is not int:
        raise HTTPException(401, "invalid token: uid claim is not a user id")
    with db() as conn:
        revoked = conn.execute(
            "SELECT 1 FROM revoked_tokens WHERE jti = ?", (payload["jti"],)
        ).fetchone()
        account = conn.execute("SELECT email FROM users WHERE id = ?", (uid,)).fetchone()
    if revoked is not None:
        raise HTTPException(401, "this session has been signed out")
    if account is None or not isinstance(payload["email"], str) or not hmac.compare_digest(
        account["email"].encode(), payload["email"].encode("utf-8", "replace")
    ):
        # The signature is good, but the account it names is not the one in
        # this database -- the database was replaced under the same key.
        raise HTTPException(401, "this session is for an account this server does not have; sign in again")
    return payload


def current_user(payload: dict = Depends(current_token)) -> int:
    """FastAPI dependency: resolve the JWT bearer token to a user id.

    `current_token` has already checked that `uid` is a real int naming an
    account whose email matches the token's, so this only narrows.
    """
    return payload["uid"]


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
    # A device still polling is not proof the report is current (M8.1): a
    # sample that stopped being refreshed is withheld like a silent device's.
    if _health_freshness(row, now)["state"] == "stale":
        return None
    try:
        report = json.loads(row["health"] or "null")
    except (ValueError, TypeError):
        return None
    return report if isinstance(report, dict) else None


#: Boot ids remembered as superseded, so a delayed report from before a reboot
#: is not taken for yet another reboot.
_HEALTH_BOOTS_KEPT = 8

_HEALTH_COLUMNS = ("health_boot", "health_generation", "health_revision", "health_received_at",
                   "health_observed_at", "health_repeats", "health_boots")


def _health_decision(stored: dict, health: dict, now: float) -> tuple[str, dict]:
    """What a reported health sample does to the stored one (M8.1).

    `stored` holds the _HEALTH_COLUMNS of the device row. Returns the outcome and
    the columns to write (`health` itself only when the content is stored):

      new          a newer generation of the current boot (or the first sample)
      reboot       a boot id never seen before: the generation restart is accepted
      revised      the same sample with newer monitoring keys: content stored, age kept
      repeat       the same sample again: nothing new, it keeps ageing
      out_of_order an older generation or revision of the current boot: ignored
      retired_boot a report from a boot a reboot already superseded: ignored
      invalid      a stamp that cannot be trusted: ignored, so it never refreshes
      unverified   no stamp (firmware before M8.1): stored, freshness unknown

    Ages are on the backend's clock only: receipt time minus the age the device
    measured on its monotonic clock. The device's wall clock is never used.
    """
    try:
        stamp = health_stamp.parse(health)
    except health_stamp.StampError as exc:
        return "invalid", {"health_ignored": json.dumps({"reason": "invalid", "detail": str(exc)[:120],
                                                         "at": now, "boot_id": None, "generation": None})}
    content = json.dumps(_finite(health))
    if stamp is None:
        return "unverified", {"health": content, "health_boot": None, "health_generation": None,
                              "health_revision": None, "health_received_at": now,
                              "health_observed_at": None, "health_repeats": 0}

    def ignored(reason: str) -> tuple[str, dict]:
        return reason, {"health_ignored": json.dumps({"reason": reason, "at": now, "boot_id": stamp.boot_id,
                                                      "generation": stamp.generation})}

    observed = now - (stamp.age_s or 0.0)
    try:
        retired = json.loads(stored.get("health_boots") or "[]")
    except (TypeError, ValueError):
        retired = []
    retired = [b for b in retired if isinstance(b, str)] if isinstance(retired, list) else []
    current = stored.get("health_boot")
    new_sample = {"health": content, "health_boot": stamp.boot_id, "health_generation": stamp.generation,
                  "health_revision": stamp.revision, "health_received_at": now,
                  "health_observed_at": observed, "health_repeats": 0}
    if stamp.boot_id in retired:
        return ignored("retired_boot")
    if current is None:
        return "new", new_sample
    if stamp.boot_id != current:
        retired = [*retired, current][-_HEALTH_BOOTS_KEPT:]
        return "reboot", {**new_sample, "health_boots": json.dumps(retired)}
    generation, revision = stored.get("health_generation") or 0, stored.get("health_revision") or 0
    if stamp.generation > generation:
        return "new", new_sample
    if stamp.generation < generation or stamp.revision < revision:
        return ignored("out_of_order")
    # The earliest estimate wins: another copy can only make the sample older.
    stored_observed = stored.get("health_observed_at")
    kept = min(observed, stored_observed) if isinstance(stored_observed, (int, float)) else observed
    if stamp.revision > revision:
        return "revised", {"health": content, "health_revision": stamp.revision, "health_observed_at": kept}
    return "repeat", {"health_observed_at": kept, "health_repeats": (stored.get("health_repeats") or 0) + 1}


def _store_health(conn: sqlite3.Connection, device_id: int, health: Optional[dict], now: float) -> str:
    """Apply a reported health sample to the device row; returns the outcome (M8.1).

    Called after ingest's first write, so this connection already holds the
    database's write lock: two overlapping polls from one device (a timed-out
    request still being served when the retry arrives) cannot both read the
    same stored generation.
    """
    if health is None:
        return "none"
    stored = conn.execute(f"SELECT {', '.join(_HEALTH_COLUMNS)} FROM devices WHERE id = ?",  # noqa: S608 - fixed names
                          (device_id,)).fetchone()
    outcome, updates = _health_decision(dict(stored) if stored is not None else {}, health, now)
    if updates:
        assignments = ", ".join(f"{name} = ?" for name in updates)
        conn.execute(f"UPDATE devices SET {assignments} WHERE id = ?",  # noqa: S608 - fixed names
                     (*updates.values(), device_id))
    if outcome in ("out_of_order", "retired_boot", "invalid"):
        log.info("ingest: health report from device %s not applied (%s)", device_id, outcome)
    return outcome


def _health_freshness(row: sqlite3.Row, now: float) -> dict:
    """How current the stored health is as evidence (M8.1, status `health_freshness`).

    Separate from link liveness (`last_seen`): a device can poll every 2 s while
    re-sending a sample minutes old. `fresh`/`stale` only for a stamped sample;
    `unverified` for firmware that does not stamp (judged by the link alone, as
    before M8.1); `absent` when no health was ever stored.
    """
    keys = row.keys()

    def col(name: str):
        return row[name] if name in keys else None

    try:
        stored = json.loads(row["health"] or "null")
    except (ValueError, TypeError):
        stored = None
    out: dict = {"state": "absent", "verified": False, "age_s": None, "threshold_s": _HEALTH_STALE_S,
                 "observed_at": None, "received_at": _reported_at(col("health_received_at")),
                 "device_generated_at": None, "boot_id": None, "generation": None,
                 "repeats": int(col("health_repeats") or 0),
                 "last_ignored": _loads_dict(col("health_ignored")) or None}
    if not isinstance(stored, dict):
        return out
    try:
        stamp = health_stamp.parse(stored)
    except health_stamp.StampError:
        stamp = None
    observed = _reported_at(col("health_observed_at"))
    if stamp is None or col("health_boot") != stamp.boot_id or observed is None:
        out["state"] = "unverified"
        return out
    age = max(0.0, now - observed)
    out.update(state="fresh" if age < _HEALTH_STALE_S else "stale", verified=True, age_s=round(age, 1),
               observed_at=observed, device_generated_at=stamp.generated_wall, boot_id=stamp.boot_id,
               generation=stamp.generation)
    return out


def _rules_version(ai_rules_json: Optional[str], activation: int = 0) -> str:
    """Stable fingerprint of a device's stored rule set (empty when it has none).

    `activation` (devices.ai_rules_activation) makes each Activate a new
    version even when the rules are byte-for-byte the same, so re-activating
    after a switch to a named preset is delivered again. 0 -- every row from
    before the column existed -- keeps the original fingerprint, so a deploy
    does not re-send every running rule set once.
    """
    if not ai_rules_json:
        return ""
    material = ai_rules_json if not activation else f"{activation}\n{ai_rules_json}"
    return hashlib.sha256(material.encode()).hexdigest()[:16]


def _zones_version(zones_json: Optional[str]) -> str:
    """Fingerprint of a device's zone set; empty only when none was ever set.

    An EMPTY set is a real set with its own fingerprint. It used to share the
    "nothing" fingerprint, which made deleting the last zone undeliverable:
    ingest only sends zones whose version is non-empty, so the device -- which
    treats `set_zones: []` as "clear them" (core/uplink.py) -- kept enforcing
    the deleted polygon until it rebooted.
    """
    if not zones_json:
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


def _stored_rules(ai_rules_json: Optional[str]) -> list[dict]:
    """Every rule in a stored proposal (disabled ones included), tolerant of anything malformed."""
    try:
        rules = json.loads(ai_rules_json or "{}").get("rules")
    except (ValueError, TypeError, AttributeError):
        return []
    return [r for r in rules if isinstance(r, dict)] if isinstance(rules, list) else []


def _rules_payload(ai_rules_json: Optional[str]) -> list[dict]:
    """The rules the device should RUN: the stored set without the ones the
    owner turned off (PATCH /v1/devices/{id}/rules/{rule_id}). A disabled rule
    is never delivered; the device runs the rest exactly as before."""
    return [r for r in _stored_rules(ai_rules_json) if r.get("enabled") is not False]


_RULE_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def _with_rule_ids(rules: list[dict]) -> list[dict]:
    """`rules` with a stable, unique `id` on each, keeping ids already given.

    A rule's id is what the app toggles it by. It is derived from the rule's
    content and position, so a row stored before ids existed reads the same
    ids every time until they are written down (the first PATCH does that).
    """
    seen: set[str] = set()
    out = []
    for index, rule in enumerate(rules):
        rule_id = rule.get("id")
        if not (isinstance(rule_id, str) and _RULE_ID.match(rule_id)) or rule_id in seen:
            material = json.dumps({k: v for k, v in rule.items() if k not in ("id", "enabled")},
                                  sort_keys=True, default=str)
            rule_id = "r" + hashlib.sha256(f"{index}\n{material}".encode()).hexdigest()[:10]
        seen.add(rule_id)
        out.append({**rule, "id": rule_id})
    return out


# ------------------------------------------------------------------ app status contract
# The one answer to "is this device watching right now, and for what?", computed
# here from what the device itself reported. docs/APP_STATUS_CONTRACT.md is the
# field-by-field contract; tests/test_app_status_contract.py pins it.

_MONITORING_STATES = frozenset({"starting", "ready", "degraded", "stopped", "failed"})
_PROTECTION_STATES = frozenset({"protecting", "not_protecting", "stopped"})
_RUN_MODES = frozenset({"off", "standard", "ai_custom", "standard_plus_custom"})
_MAX_REASONS = 20


def _strings(value, limit: int = _MAX_REASONS) -> list[str]:
    """A device-reported list as bounded strings (anything else: empty)."""
    if not isinstance(value, list):
        return []
    return [str(item)[:500] for item in value[:limit] if isinstance(item, (str, int, float))]


def _reported_at(value) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _role_name(*sources) -> Optional[str]:
    """The role name from whichever report carries one.

    The device sends it in two shapes: `monitoring.role` is core/roles.py's
    describe() dict, `health.monitoring.role` and `protection.role` are the bare
    name. Reading only the dict made a device that had sent the string look
    role-less, i.e. unrestricted.
    """
    for source in sources:
        name = source.get("name") if isinstance(source, dict) else source
        if isinstance(name, str) and name.strip():
            return name.strip()[:64]
    return None


def _role_facts(monitoring: dict, health_monitoring: dict, protection: dict):
    """(name, title, allowed_presets, additive) as the device reported its role.

    The backend's own role table (core/roles.py, the table set_preset enforces)
    wins for a role it knows; the device's describe() dict covers one it does
    not. allowed_presets None = every preset.
    """
    from core.roles import ROLES

    reported = monitoring.get("role")
    pushed_protection = monitoring.get("protection")
    pushed_protection = pushed_protection if isinstance(pushed_protection, dict) else {}
    name = _role_name(reported, health_monitoring.get("role"), protection.get("role"),
                      pushed_protection.get("role"))
    if name is None:
        return None, None, None, None
    role = ROLES.get(name)
    if role is not None:
        allowed = None if role.allowed_presets is None else sorted(role.allowed_presets)
        return name, role.title, allowed, role.additive_custom_rules
    described = reported if isinstance(reported, dict) else {}
    allowed_raw = described.get("allowed_presets")
    allowed = _strings(allowed_raw, 64) if isinstance(allowed_raw, list) else None
    additive = described.get("additive_custom_rules")
    title = described.get("title")
    return (name, str(title)[:64] if isinstance(title, str) else None, allowed,
            additive if isinstance(additive, bool) else None)


def _capabilities_summary(raw) -> Optional[dict]:
    """{available, impaired: [{name, reason}], starting, paused} from the device's map.

    `paused` (M8.2, additive): not running because monitoring was stopped on
    purpose; never a fault. Older apps ignore the key."""
    if not isinstance(raw, dict):
        return None
    available, impaired, starting, paused = [], [], [], []
    for name, cap in sorted(raw.items()):
        if not isinstance(name, str) or not isinstance(cap, dict):
            continue
        state = cap.get("state")
        if state == "available":
            available.append(name)
        elif state == "impaired":
            reason = cap.get("reason")
            impaired.append({"name": name, "reason": str(reason)[:500] if reason else None})
        elif state == "starting":
            starting.append(name)
        elif state == "paused":
            paused.append(name)
    return {"available": available, "impaired": impaired, "starting": starting, "paused": paused}


def _rejection(raw) -> Optional[dict]:
    """The device's last refusal (services/runtime.py _record_rejection), cleaned.

    `subject` is what was refused (a preset name, a rules version, a node id);
    the M7 wire contract also allows `preset`/`node` and a single `reason`, and
    an optional machine-readable `code`."""
    if not isinstance(raw, dict) or raw.get("kind") not in ("preset", "rules", "protection", "nodes"):
        return None
    subject = raw.get("subject") or raw.get("preset") or raw.get("node") or ""
    reasons = _strings(raw.get("reasons"))
    if not reasons and isinstance(raw.get("reason"), str) and raw["reason"]:
        reasons = [raw["reason"][:500]]
    code = raw.get("code")
    return {
        "kind": raw["kind"],
        "subject": str(subject)[:200],
        "reasons": reasons,
        "code": code[:64] if isinstance(code, str) and code else None,
        "at": _reported_at(raw.get("at")),
    }


def _rejection_fingerprint(raw) -> str:
    """Identity of one device refusal, so a refusal already seen is not re-counted."""
    cleaned = _rejection(raw)
    if cleaned is None:
        return ""
    return hashlib.sha256(json.dumps(cleaned, sort_keys=True).encode()).hexdigest()[:16]


def _stored_last_rejection(row: sqlite3.Row):
    """The device's last refusal as last pushed (the `monitoring` column)."""
    return (_monitoring_payload(row["monitoring"]) or {}).get("last_rejection")


def _baseline_with(row: sqlite3.Row, kind: str) -> str:
    """rejection_baseline with `kind` set to the refusal the device reports NOW.

    Written together with a new request: only a refusal that differs from this
    one can be the device's answer to that request."""
    baseline = _loads_dict(row["rejection_baseline"])
    baseline[kind] = _rejection_fingerprint(_stored_last_rejection(row))
    return json.dumps(baseline)


def _new_refusal(raw, kind: str, baseline_json: Optional[str]) -> Optional[dict]:
    """The device's refusal of `kind`, if it is newer than the request's baseline."""
    cleaned = _rejection(raw)
    if cleaned is None or cleaned["kind"] != kind:
        return None
    if _rejection_fingerprint(raw) == _loads_dict(baseline_json).get(kind):
        return None
    return cleaned


# ---------------------------------------------------------------- M7 perception (device truth)
#: docs/M7_IMPLEMENTATION_MAP.md wire contract, `monitoring.perception`.
_COMPONENT_STATES = ("ok", "stale", "failed", "unavailable", "idle", "paused")  # paused: M8.2
_NODE_STATES = ("active", "degraded", "inactive", "failed")
#: Which product capability kind each shared perception component provides.
COMPONENT_CAPABILITY = {
    "object_detector": "objects", "face_landmarks": "face", "hand_landmarks": "hands", "pose": "pose",
}
_MAX_COMPONENTS = 16
_MAX_NODES = 32


def _finite_or_none(value) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _count(value) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return max(0, value)


def _short(value, limit: int = 200) -> Optional[str]:
    return value[:limit] if isinstance(value, str) and value else None


def _node_metrics(raw) -> Optional[dict]:
    """A node's current values (M8: counts, per-area occupancy and dwell), bounded:
    numbers, booleans and short id lists only -- never free text, never accuracy."""
    if not isinstance(raw, dict):
        return None

    def scalar(value):
        if isinstance(value, bool) or value is None:
            return value
        return _finite_or_none(value)

    out: dict = {}
    for key, value in list(raw.items())[:16]:
        if not isinstance(key, str):
            continue
        if isinstance(value, dict):
            nested = {}
            for name, inner in list(value.items())[:16]:
                if isinstance(name, str) and isinstance(inner, dict):
                    nested[name[:64]] = {str(k)[:32]: scalar(v) for k, v in list(inner.items())[:8]}
                elif isinstance(name, str):
                    nested[name[:64]] = scalar(inner)
            out[key[:32]] = nested
        elif isinstance(value, list):
            out[key[:32]] = [v for v in value[:64] if isinstance(v, int) and not isinstance(v, bool)]
        else:
            out[key[:32]] = scalar(value)
    return out


def _perception(raw) -> Optional[dict]:
    """The device's perception block, bounded and typed; None if absent or unreadable.

    Forwarded as reported -- never completed with guesses. A component state or
    node state outside the contract's vocabulary reads "unknown", not "ok"."""
    if not isinstance(raw, dict):
        return None
    components = {}
    raw_components = raw.get("components")
    for name, comp in list((raw_components if isinstance(raw_components, dict) else {}).items())[:_MAX_COMPONENTS]:
        if not isinstance(name, str) or not isinstance(comp, dict):
            continue
        model = comp.get("model")
        state = comp.get("state")
        components[name[:64]] = {
            "state": state if state in _COMPONENT_STATES else "unknown",
            "model": ({k: _short(model.get(k), 128) for k in ("name", "sha256", "version")}
                      if isinstance(model, dict) else None),
            "last_run_age_s": _finite_or_none(comp.get("last_run_age_s")),
            "runs": _count(comp.get("runs")),
            "errors": _count(comp.get("errors")),
            "reason": _short(comp.get("reason"), 500),
            "consumers": _strings(comp.get("consumers"), 32) if isinstance(comp.get("consumers"), list) else [],
        }
    nodes = []
    raw_nodes = raw.get("nodes")
    for node in (raw_nodes if isinstance(raw_nodes, list) else [])[:_MAX_NODES]:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str) or not node["id"]:
            continue
        state = node.get("state")
        nodes.append({
            "id": node["id"][:64],
            "kind": _short(node.get("kind"), 32),
            "version": _short(node.get("version"), 64),
            "preset": _short(node.get("preset"), 64),
            "state": state if state in _NODE_STATES else "unknown",
            "required_capabilities": _strings(node.get("required_capabilities"), 16)
            if isinstance(node.get("required_capabilities"), list) else [],
            "missing_capabilities": _strings(node.get("missing_capabilities"), 16)
            if isinstance(node.get("missing_capabilities"), list) else [],
            "validation": _short(node.get("validation"), 64),
            "experimental": node.get("experimental") if isinstance(node.get("experimental"), bool) else None,
            "reason": _short(node.get("reason"), 500),
            # M8: what it measures now (people_counter: count, areas) and its
            # experimental parts (baby monitoring's pose features), as reported.
            "enabled_by": _short(node.get("enabled_by"), 16),
            "metrics": _node_metrics(node.get("metrics")),
            "experimental_features": [
                {"id": _short(f.get("id"), 64), "status": _short(f.get("status"), 200),
                 "enabled": f.get("enabled") if isinstance(f.get("enabled"), bool) else None}
                for f in (node.get("experimental_features") or [])[:8] if isinstance(f, dict)
            ] if isinstance(node.get("experimental_features"), list) else [],
        })
    tracking = raw.get("tracking")
    return {
        "schema": _short(raw.get("schema"), 64),
        "simulated": raw.get("simulated") if isinstance(raw.get("simulated"), bool) else None,
        "components": components,
        "nodes": nodes,
        # M8: the shared tracker's counters and cost (numbers only).
        "tracking": ({k: _finite_or_none(v) for k, v in list(tracking.items())[:16]
                      if isinstance(k, str) and not isinstance(v, bool) and isinstance(v, (int, float))}
                     if isinstance(tracking, dict) else None),
    }


def _reported_perception(live_health: dict, monitoring: dict) -> Optional[dict]:
    """Perception from a LIVE report: the health report's monitoring block, the
    health report itself, or the pushed monitoring block (firmware generations)."""
    live_monitoring = _dict_field(live_health, "monitoring")
    for source in (live_monitoring, live_health, monitoring):
        if isinstance(source.get("perception"), dict):
            return _perception(source["perception"])
    return None


def _bounded_dicts(raw, keys: tuple[str, ...], limit: int = 32) -> Optional[list[dict]]:
    """A list of the device's report dicts reduced to `keys` (policies, actuators)."""
    if not isinstance(raw, list):
        return None
    out = []
    for item in raw[:limit]:
        if not isinstance(item, dict):
            continue
        clean = {}
        for key in keys:
            value = item.get(key)
            if isinstance(value, list):
                clean[key] = _strings(value)
            elif isinstance(value, bool) or value is None:
                clean[key] = value
            elif isinstance(value, (int, float)):
                clean[key] = _finite_or_none(value)
            else:
                clean[key] = str(value)[:200]
        out.append(clean)
    return out


_POLICY_KEYS = ("id", "kind", "name", "can_evaluate", "operational", "unavailable_events", "reasons")
_ACTUATOR_KEYS = ("name", "kind", "audible", "ready", "detail", "commands_ok", "commands_failed",
                  "physically_verified")


# ---------------------------------------------------------------- M8.3 truth (docs/STATE_MODEL.md)
#: Capabilities that are analysis results, as opposed to the pipeline that feeds
#: them (camera, inference) and the hardware outputs (alert_output, audible_alert).
_ANALYSIS_CAPABILITIES = frozenset({"face", "objects", "hands", "pose", "attention"})
#: How long a stop/resume may go unacknowledged before its outcome is `timed_out`
#: (the app's ProtectionControl.timeout: stopping drains the pipeline).
PROTECTION_ACK_TIMEOUT_S = 45.0
#: A stop not delivered within this long is never delivered: a stop arriving
#: minutes after the person gave up on it would end protection behind their back.
#: A resume keeps being delivered until applied (the protective direction).
PROTECTION_STOP_DELIVERY_S = 30.0


def _make_coherent(status: dict) -> list[str]:
    """Apply the device's own rules to its own report, in the safe direction only.

    A report is normally coherent: services/runtime.py derives every field from
    one evaluation. One that is not (older firmware, a sample torn between two
    evaluations, a capability list from before M8.2) must never reach the app
    as a claim the device's own rules forbid. Each correction only withdraws a
    claim -- protecting, monitoring, a capability producing results -- never
    adds one, and is listed in `status.corrections`. A stale status is not
    touched: it already claims nothing.
    """
    if status["stale"]:
        return []
    fixes: list[str] = []
    state, protection = status["monitoring_state"], status["protection"]
    # An explicit stop is authoritative in whichever block reports it.
    if (state == "stopped") != (protection == "stopped") and "stopped" in (state, protection):
        fixes.append(f"stopped in one block but {state if protection == 'stopped' else protection} "
                     "in the other: reported as stopped")
        state = protection = "stopped"
    caps = status.get("capabilities")
    if isinstance(caps, dict):
        available = list(caps.get("available") or [])
        impaired = list(caps.get("impaired") or [])
        paused = list(caps.get("paused") or [])
        impaired_names = {i["name"]: i.get("reason") for i in impaired if isinstance(i, dict)}

        def withdraw(name: str, *, pause: bool, reason: str, into_impaired: bool = True) -> None:
            if name not in available:
                return
            available.remove(name)
            if pause:
                paused.append(name)
            elif into_impaired:
                impaired.append({"name": name, "reason": reason})
                impaired_names[name] = reason
            fixes.append(f"{name} withdrawn from available: {reason}")

        if state == "stopped":
            # M8.2: nothing analyses while stopped (a device before M8.2 said otherwise).
            for name in sorted(_ANALYSIS_CAPABILITIES | {"inference"}):
                withdraw(name, pause=True, reason="monitoring is stopped")
        # Nothing produces results without frames or a running inference loop.
        for pipe in ("camera", "inference"):
            if pipe in impaired_names:
                why = f"no analysis: {pipe} unavailable ({impaired_names[pipe] or 'impaired'})"
                for name in sorted(_ANALYSIS_CAPABILITIES):
                    withdraw(name, pause=False, reason=why)
        # A model that failed or is missing produces nothing (perception truth).
        # A missing one is left to the perception report, which says
        # `unsupported` (capability_states): switching cannot repair it.
        for component, comp in ((status.get("perception") or {}).get("components") or {}).items():
            kind = COMPONENT_CAPABILITY.get(component)
            if kind and isinstance(comp, dict) and comp.get("state") in ("failed", "unavailable"):
                withdraw(kind, pause=False, into_impaired=comp["state"] == "failed",
                         reason=f"{component.replace('_', ' ')} {comp['state']}: "
                                f"{comp.get('reason') or 'no reason given'}")
        # Every policy needs frames and inference (core/monitoring_state.py
        # BASE_REQUIREMENTS): without either nothing can be evaluated -- FAILED.
        lost = [p for p in ("camera", "inference") if p in impaired_names]
        if lost and state in ("ready", "degraded"):
            fixes.append(f"{state} while {', '.join(lost)} impaired: reported as failed")
            status["reasons"] = [*status["reasons"],
                                 *(f"required capability unavailable — {p}: {impaired_names[p]}" for p in lost)]
            state = "failed"
        status["capabilities"] = {**caps, "available": available, "impaired": impaired, "paused": paused}
    if state == "stopped" and isinstance(status.get("perception"), dict):
        for comp in (status["perception"].get("components") or {}).values():
            if isinstance(comp, dict) and comp.get("state") in ("ok", "stale", "idle"):
                comp["state"], comp["reason"] = "paused", "monitoring is stopped"
                fixes.append("a perception component withdrawn while stopped")
    if protection == "protecting" and state not in ("ready", "degraded"):
        fixes.append(f"protecting while {state}: reported as not protecting")
        protection = "not_protecting"
        status["protection_reasons"] = [*status["protection_reasons"], f"monitoring state is {state}"]
    status["monitoring_state"], status["protection"] = state, protection
    if state != "unknown":
        status["monitoring_active"] = state in ("ready", "degraded")
    return fixes


def _protection_command(row: sqlite3.Row, now: float) -> Optional[dict]:
    """The last stop/resume the app asked for, and its outcome (M8.3).

    outcome   pending    not applied yet, within PROTECTION_ACK_TIMEOUT_S
              confirmed  the device echoed THIS id with the requested state
              mismatch   the device echoed this id but reports another state
                         (it refused, or something else changed it first)
              timed_out  not applied within the timeout
    The outcome is latched from the report that first carried the echo, so a
    later local change cannot rewrite what this request did. `delivering` says
    whether the backend still sends it: a resume until applied, a stop only for
    PROTECTION_STOP_DELIVERY_S.
    """
    columns = row.keys()  # sqlite3.Row: `in row` would test the values
    record = _loads_dict(row["protection_request"] if "protection_request" in columns else None)
    action, command_id = record.get("action"), record.get("id")
    if action not in ("stop", "resume") or not isinstance(command_id, str):
        return None
    requested = _reported_at(record.get("requested_at"))
    age = max(0.0, now - requested) if requested is not None else None
    applied_at = _reported_at(record.get("applied_at"))
    applied_state = record.get("applied_state") if isinstance(record.get("applied_state"), str) else None
    reason = None
    if applied_at is not None:
        wanted = applied_state == "stopped" if action == "stop" else applied_state not in (None, "stopped")
        outcome = "confirmed" if wanted else "mismatch"
        if not wanted:
            reason = record.get("refusal") or f"the device reports {applied_state or 'no protection state'}"
    elif age is not None and age >= PROTECTION_ACK_TIMEOUT_S:
        outcome = "timed_out"
    else:
        outcome = "pending"
    delivering = applied_at is None and (
        action == "resume" or (age is not None and age < PROTECTION_STOP_DELIVERY_S))
    return {"id": command_id[:64], "action": action, "requested_at": requested,
            "age_s": None if age is None else round(age, 1), "timeout_s": PROTECTION_ACK_TIMEOUT_S,
            "outcome": outcome, "delivering": delivering, "applied_at": applied_at,
            "reported_state": applied_state, "reason": str(reason)[:500] if reason else None}


def _record_protection_ack(conn: sqlite3.Connection, device_id: int, reported_block: Optional[dict],
                           monitoring: Optional[dict]) -> None:
    """Latch the outcome of the last stop/resume from the first report echoing its id (M8.3).

    The state that report carries is what the request did: `stopped` for a stop
    that took effect, anything else for one the device refused. Later reports
    (a local resume, a restart) change the status, never this record.
    """
    if not reported_block or not isinstance(reported_block.get("last_command_id"), str):
        return
    stored = conn.execute("SELECT protection_request FROM devices WHERE id = ?", (device_id,)).fetchone()
    record = _loads_dict(stored["protection_request"] if stored is not None else None)
    if record.get("id") != reported_block["last_command_id"] or record.get("applied_at") is not None:
        return
    state = reported_block.get("state")
    record["applied_at"] = time.time()
    record["applied_state"] = state if state in _PROTECTION_STATES else None
    refusal = _new_refusal((monitoring or {}).get("last_rejection"), "protection",
                           json.dumps({"protection": record.get("refusal_baseline")}))
    if refusal is not None:
        record["refusal"] = "; ".join(refusal["reasons"]) or "the device refused it"
    conn.execute("UPDATE devices SET protection_request = ? WHERE id = ?", (json.dumps(record), device_id))


def _device_status(row: sqlite3.Row, now: float) -> dict:
    """The typed status the app renders (docs/APP_STATUS_CONTRACT.md).

    Liveness first: with no report in the last _HEALTH_STALE_S seconds (or
    none ever), monitoring_state and protection are "unknown" and
    monitoring_active is None. An old "ready" is never echoed: the last report
    is a photograph of a moment that has passed.
    """
    last_seen = row["last_seen"]
    link_stale = not last_seen or now - last_seen >= _HEALTH_STALE_S
    # M8.1: a successful poll is not proof the device is monitoring. A device
    # that keeps polling while its health sample stops being refreshed is
    # reachable (link_stale false) on stale evidence (health_stale true), and
    # `stale` -- what every claim below is gated on -- covers both.
    freshness = _health_freshness(row, now)
    health_stale = freshness["state"] == "stale"
    stale = link_stale or health_stale
    # A live device that sends no health at all (older software) is not stale:
    # it is "unknown" below for its own reason.
    health = None if stale else _health_payload(row, now)
    monitoring = _monitoring_payload(row["monitoring"]) or {}
    try:
        stored_health = json.loads(row["health"] or "null")
    except (ValueError, TypeError):
        stored_health = None
    stored_health = stored_health if isinstance(stored_health, dict) else {}
    # Role and capability facts read from the LAST report even when stale: they
    # describe configuration, not liveness. State and protection never do.
    stored_monitoring = stored_health.get("monitoring")
    stored_monitoring = stored_monitoring if isinstance(stored_monitoring, dict) else {}
    stored_protection = stored_health.get("protection")
    stored_protection = stored_protection if isinstance(stored_protection, dict) else {}
    role, role_title, allowed_presets, additive = _role_facts(monitoring, stored_monitoring, stored_protection)

    live = health or {}
    live_monitoring = _dict_field(live, "monitoring")
    reported_protection = live.get("protection")
    if not isinstance(reported_protection, dict):
        # Firmware that sends protection only in the monitoring block.
        reported_protection = _dict_field(monitoring, "protection")

    reasons: list[str] = []
    protection_reasons: list[str] = []
    if stale:
        state, protection, active = "unknown", "unknown", None
        if link_stale:
            reasons = [
                "the device has not reported yet" if not last_seen
                else f"no report from the device in the last {_HEALTH_STALE_S} s"
            ]
        else:
            reasons = [("the device is connected, but its monitoring status has not been updated for "
                        f"{int(freshness['age_s'] or 0)} s")]
        refused = _credential_refusal(row)
        if refused is not None:
            reasons.append(refused["detail"])
    else:
        state = live_monitoring.get("state")
        if state not in _MONITORING_STATES:
            # The state the device pushed with its monitoring report (same firmware
            # generation), else nothing this backend can trust.
            state = monitoring.get("state") if monitoring.get("state") in _MONITORING_STATES else "unknown"
        if state == "unknown":
            reasons = ["this device's software does not report its monitoring state"]
            active = None
        else:
            reasons = _strings(live_monitoring.get("reasons"))
            # Exactly the device's own definition (MonitoringReport.monitoring),
            # derived rather than read so it can never disagree with the state.
            active = state in ("ready", "degraded")
        protection = reported_protection.get("state")
        if protection not in _PROTECTION_STATES:
            protection = "unknown"
        protection_reasons = _strings(reported_protection.get("reasons"))

    mode = live_monitoring.get("mode") or monitoring.get("mode")
    mode = mode if mode in _RUN_MODES else None

    # Rules: what the DEVICE runs, never what the backend validated. A rule set the
    # device refused (all or nothing, services/runtime.py) is acknowledged by the
    # uplink all the same, so "delivered" is not "running".
    last_rejection = _rejection(monitoring.get("last_rejection"))
    wanted_version = (
        _rules_version(row["ai_rules"], row["ai_rules_activation"] or 0)
        if row["custom_monitoring_wanted"] else ""
    )
    rules_rejected = None
    if (last_rejection and last_rejection["kind"] == "rules" and wanted_version
            and last_rejection["subject"] == wanted_version):
        rules_rejected = {"version": wanted_version, "reasons": last_rejection["reasons"],
                          "at": last_rejection["at"]}
    running = live_monitoring.get("custom_rules") if not stale else None
    rules = {
        "active": _strings(running, 200) if isinstance(running, list) else None,
        "additive": additive,
        "rejected": rules_rejected,
    }

    preset_rejection = _loads_dict(row["preset_rejection"]) or None
    # M7: device truth only, and only while the device is live. None means "not
    # reported" (stale, or firmware without it) -- never filled in.
    perception = None if stale else _reported_perception(live, monitoring)
    last_command_id = reported_protection.get("last_command_id") if not stale else None
    status = {
        "monitoring_state": state,
        "monitoring_active": active,
        "protection": protection,
        "reasons": reasons,
        "protection_reasons": protection_reasons,
        "mode": mode,
        "role": role,
        "role_title": role_title,
        "allowed_presets": allowed_presets,
        "capabilities": None if stale else _capabilities_summary(live_monitoring.get("capabilities")),
        "rules": rules,
        "last_rejection": last_rejection,
        "preset_rejection": preset_rejection,
        "nodes_rejection": _loads_dict(row["nodes_rejection"]) or None,
        "perception": perception,
        # The id of the last protection command the device applied, so the app
        # converges on THIS request rather than inferring it from the state.
        "last_command_id": last_command_id[:64] if isinstance(last_command_id, str) else None,
        "policies": None if stale else _bounded_dicts(live_monitoring.get("policies"), _POLICY_KEYS),
        "actuators": None if stale else _bounded_dicts(live_monitoring.get("actuators"), _ACTUATOR_KEYS),
        "stale": stale,
        "reported_at": _reported_at(row["last_seen"]),
        # M8.1: the two halves of `stale`, and how old the evidence itself is.
        "link_stale": bool(link_stale),
        "health_stale": health_stale,
        "health_freshness": freshness,
        # M8.3: the last stop/resume and its acknowledged outcome.
        "protection_command": _protection_command(row, now),
    }
    # M8.3: never a claim the device's own rules forbid (see _make_coherent).
    status["corrections"] = _make_coherent(status)
    return status


#: Request nonces the backend remembers, in the database (sd-auth-v1 replay
#: refusal). They used to live in this process's memory, so a backend restart
#: forgot them and a request captured up to CLOCK_SKEW_S earlier was accepted
#: again (M8). The table is pruned of nonces older than 2 x CLOCK_SKEW_S --
#: those requests fail the clock check anyway -- and bounded.
REQUEST_NONCES_MAX = 1_000_000
_NONCE_PRUNE_EVERY_S = 30.0


class _StoredNonces:
    """sd_auth.NonceStore over the `auth_nonces` table: survives a restart, and
    is shared by every process on this database. INSERT OR IGNORE on the primary
    key is the atomic test-and-set, so no lock is needed."""

    def __init__(self, max_entries: int = REQUEST_NONCES_MAX, ttl_s: float = 2 * sd_auth.CLOCK_SKEW_S) -> None:
        self._max, self._ttl = max_entries, ttl_s
        self._next_prune = 0.0

    def add(self, nonce: str, now: float) -> bool:
        with db() as conn:
            if now >= self._next_prune:
                self._next_prune = now + _NONCE_PRUNE_EVERY_S
                conn.execute("DELETE FROM auth_nonces WHERE seen_at < ?", (now - self._ttl,))
                excess = conn.execute("SELECT COUNT(*) FROM auth_nonces").fetchone()[0] - self._max
                if excess > 0:
                    # Not reachable at a single backend's request rate (it would take
                    # over 1,600 signed requests a second); said loudly if it is.
                    log.warning("device auth: %d unexpired request nonces evicted (table full)", excess)
                    conn.execute("DELETE FROM auth_nonces WHERE nonce IN (SELECT nonce FROM auth_nonces"
                                 " ORDER BY seen_at LIMIT ?)", (excess,))
            inserted = conn.execute("INSERT OR IGNORE INTO auth_nonces (nonce, seen_at) VALUES (?, ?)",
                                    (nonce, now))
            return inserted.rowcount == 1


_REQUEST_NONCES = _StoredNonces()

#: What the app is told when the backend refused a device it could identify
#: (GET /v1/devices `credentials` and the stale status reason). Only refusals
#: that prove they came from the key's holder are recorded: anyone can send a
#: bad tag under a public key id, and that must not paint a healthy device red.
_CREDENTIAL_REFUSALS = {
    "revoked_key": "the backend refused this device's key: it has been revoked",
    "revoked_device": "this device's credentials have been revoked by the operator",
    "clock_skew": "the device's clock is too far from the backend's for a signed request",
    "signature_required": "the device sends unsigned requests, and the backend requires signed ones for it",
}


def _device_for_kid(kid: str) -> tuple[Optional[sqlite3.Row], Optional[sd_auth.Keyring]]:
    """The device row a key id names, and the keys it may sign with."""
    with db() as conn:
        query = "SELECT * FROM devices WHERE auth_kid = ? OR previous_auth_kid = ?"
        row = conn.execute(query, (kid, kid)).fetchone()
        if row is None:
            # A row inserted by a tool that predates sd-auth-v1 has no auth_kid yet.
            _backfill_auth_kids(conn)
            row = conn.execute(query, (kid, kid)).fetchone()
    if row is None:
        return None, None
    try:
        keyring = sd_auth.Keyring(sd_auth.DeviceKey(row["device_key"]),
                                  sd_auth.DeviceKey(row["previous_device_key"]) if row["previous_device_key"] else None)
    except ValueError:
        return None, None
    return row, keyring


def _revoked_key(kid: str) -> Optional[tuple[int, str]]:
    """(device id, key) of a revoked key, by its public id."""
    with db() as conn:
        row = conn.execute("SELECT device_id, device_key FROM revoked_device_keys WHERE kid = ?", (kid,)).fetchone()
    return (row[0], row[1]) if row is not None else None


def _record_refusal(device_id: int, reason: str) -> None:
    """Remember an authenticated refusal for the app. At most one write a minute
    per device and reason, so a device retrying every poll is not a write storm."""
    now = time.time()
    with db() as conn:
        conn.execute(
            "UPDATE devices SET auth_refusal = ? WHERE id = ? AND (auth_refusal IS NULL"
            " OR json_extract(auth_refusal, '$.reason') IS NOT ? OR json_extract(auth_refusal, '$.at') < ?)",
            (json.dumps({"reason": reason, "at": now}), device_id, reason, now - 60),
        )


def _note_auth(row: sqlite3.Row, kind: str) -> None:
    """Record how the device authenticated ('signed' / 'bearer'), on change only:
    an operator moves a device to `required` once it is seen signing."""
    if row["last_auth"] != kind:
        with db() as conn:
            conn.execute("UPDATE devices SET last_auth = ? WHERE id = ?", (kind, row["id"]))


def _credential_refusal(row: sqlite3.Row) -> Optional[dict]:
    """The device's last recorded refusal, unless a report was accepted after it."""
    refusal = _loads_dict(row["auth_refusal"])
    at = refusal.get("at")
    if not isinstance(at, (int, float)) or refusal.get("reason") not in _CREDENTIAL_REFUSALS:
        return None
    if row["last_seen"] and row["last_seen"] >= at:
        return None
    return {"reason": refusal["reason"], "detail": _CREDENTIAL_REFUSALS[refusal["reason"]], "at": at}


def _credentials_status(row: sqlite3.Row) -> dict:
    """GET /v1/devices `credentials`: whether the backend accepts this device."""
    refusal = _credential_refusal(row)
    if row["auth_revoked_at"]:
        state = "revoked"
    elif refusal is not None:
        state = "refused"
    else:
        state = "accepted" if row["last_seen"] else "unknown"
    return {
        "state": state,
        "auth": row["last_auth"] if row["last_auth"] in ("signed", "bearer") else None,
        "signature_required": DEVICE_AUTH_MODE == "required" or row["auth_mode"] == "required",
        "refusal": refusal,
    }


def _refuse_device(reason: str, kid: str = "", *, extra: Optional[dict] = None) -> CodedError:
    # Reason code and public key id only: never the key, the tag or the body.
    log.warning("device auth refused: %s%s", reason, f" (kid {kid})" if kid else "")
    if reason == "clock_skew":
        # Answered with server_time (see _coded_error): a Pi has no RTC. The
        # device uses it only to timestamp its next request; freshness is still
        # judged here, against this clock.
        error = CodedError(401, "clock_skew", "device clock is too far from the backend's")
    else:
        error = CodedError(401, f"device_auth_{reason}", "device authentication failed")
    error.extra = extra or {}
    return error


def _signed_by(secret: str, parsed: sd_auth.RequestAuth, method: str, path: str, body: bytes) -> bool:
    """Whether `secret` made this request's tag (no clock or nonce check)."""
    try:
        sd_auth.verify_request(sd_auth.Keyring(sd_auth.DeviceKey(secret)), parsed, method, path, body,
                               nonces=sd_auth.NonceCache(), skew_s=float("inf"))
    except (sd_auth.AuthError, ValueError):
        return False
    return True


def _bearer_revoked(presented: str) -> Optional[int]:
    """The device a revoked bearer key belonged to, if `presented` is exactly that key."""
    try:
        kid = sd_auth.key_id(presented)
    except ValueError:
        return None
    revoked = _revoked_key(kid)
    if revoked is not None and hmac.compare_digest(revoked[1].encode(), presented.encode()):
        return revoked[0]
    return None


async def current_device(request: Request, x_device_key: str = Header(default=""),
                         x_sd_auth: str = Header(default=""), x_sd_nonce: str = Header(default="")) -> sqlite3.Row:
    """FastAPI dependency: authenticate a device request (sd-auth-v1 or legacy key).

    A signed request (X-SD-Auth) is checked against the key its id names, the
    exact body bytes, the backend clock (+-CLOCK_SKEW_S) and the stored nonces.
    Otherwise X-Device-Key is the bearer secret, as before -- unless
    SENSORDYME_DEVICE_AUTH=required or the device's own auth_mode is. A revoked
    key or device is refused either way. `request.state.sd_auth` says which key
    to sign the reply's command with and which request nonce it answers.
    """
    request.state.sd_auth = None
    if x_sd_auth:
        if DEVICE_AUTH_MODE == "off":
            raise _refuse_device("disabled")
        try:
            parsed = sd_auth.parse_request_header(x_sd_auth)
        except sd_auth.AuthError as exc:
            raise _refuse_device(exc.reason) from None
        body = await request.body()
        root = request.scope.get("root_path") or ""
        path = request.url.path[len(root):] if root and request.url.path.startswith(root) else request.url.path
        row, keyring = _device_for_kid(parsed.kid)
        if row is None or keyring is None:
            revoked = _revoked_key(parsed.kid)
            if revoked is not None and _signed_by(revoked[1], parsed, request.method, path, body):
                _record_refusal(revoked[0], "revoked_key")
                raise _refuse_device("revoked_key", parsed.kid)
            raise _refuse_device("unknown_key", parsed.kid)
        try:
            # In the threadpool: recording the nonce is a database write, and a
            # write-lock wait here must not stall the event loop for every request.
            key = await run_in_threadpool(sd_auth.verify_request, keyring, parsed, request.method, path, body,
                                          nonces=_REQUEST_NONCES)
        except sd_auth.AuthError as exc:
            if exc.reason != "clock_skew":
                raise _refuse_device(exc.reason, parsed.kid) from None
            # The tag was checked before the clock, so this answer goes to the
            # key's holder: sign the time, bound to the request it answers.
            _record_refusal(row["id"], "clock_skew")
            now = int(time.time())
            clock = sd_auth.sign_clock(keyring.for_kid(parsed.kid), request_nonce=parsed.nonce, server_time=now)
            raise _refuse_device("clock_skew", parsed.kid,
                                 extra={"server_time": now, "clock_auth": clock}) from None
        if row["auth_revoked_at"]:
            _record_refusal(row["id"], "revoked_device")
            raise _refuse_device("revoked_device", parsed.kid)
        _note_auth(row, "signed")
        request.state.sd_auth = (key, parsed.nonce)
        return row
    with db() as conn:
        row = conn.execute("SELECT * FROM devices WHERE device_key = ?", (x_device_key,)).fetchone()
    if row is None:
        revoked_device = _bearer_revoked(x_device_key) if x_device_key else None
        if revoked_device is not None:
            _record_refusal(revoked_device, "revoked_key")
            raise _refuse_device("revoked_key")
        if DEVICE_AUTH_MODE == "required":
            raise _refuse_device("signature_required")
        raise HTTPException(401, "invalid device key")
    if DEVICE_AUTH_MODE == "required" or row["auth_mode"] == "required":
        _record_refusal(row["id"], "signature_required")
        raise _refuse_device("signature_required")
    if row["auth_revoked_at"]:
        _record_refusal(row["id"], "revoked_device")
        raise _refuse_device("revoked_device")
    _note_auth(row, "bearer")
    if DEVICE_AUTH_MODE != "off" and x_sd_nonce and re.fullmatch(r"[0-9a-f]{32}", x_sd_nonce):
        with suppress(ValueError):  # a key under 16 bytes: legacy only, nothing to sign with
            request.state.sd_auth = (sd_auth.DeviceKey(row["device_key"]), x_sd_nonce)
    return row


def _sign_command(request: Request, command: Optional[dict], hardware_id: Optional[str]) -> Optional[dict]:
    """The sd-auth-v1 envelope for `command`, when the request asked for one."""
    state = getattr(request.state, "sd_auth", None)
    if not command or state is None:
        return None
    key, nonce = state
    try:
        return sd_auth.sign_command(key, command, device=hardware_id or f"kid:{key.kid}", request_nonce=nonce)
    except (TypeError, ValueError):
        # Not canonical JSON (a NaN stored by an old writer): unsigned, which a
        # `required` device refuses -- visible there, never a 500 here.
        log.warning("device auth: could not sign a command for kid %s", key.kid)
        return None


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
        # The number masked (M8.3): a backend without Twilio configured logged
        # every account's phone number on every serious alert.
        print(f"sms (log-only, set TWILIO_* env to send for real) -> {mask_phone(phone)}: {body}")
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


def _notify(
    owner: Optional[sqlite3.Row], title: str, detail: str, device_name: str,
    push: bool = True, sms: bool = True,
) -> None:
    """Push, then SMS, as a background task after `ingest()` has already
    responded — a slow or retrying provider must never make a device wait
    for its next poll. Each is independently caught and logged: a push
    failure (after retries) must not cancel the SMS, and vice versa.

    Which channels to use is decided by the caller (`_notify_channels`).
    """
    if push:
        try:
            send_push(owner, title, detail)
        except Exception as exc:  # noqa: BLE001 - see send_push's own docstring
            # Type and status only: the exception's text carries the ntfy URL,
            # and the topic in it is the account's alert credential (M8.3).
            log.warning("notify: push delivery failed after retries: %s", describe_failure(exc))
    if sms:
        try:
            send_sms(owner, f"{title} — {detail} ({device_name})")
        except Exception as exc:  # noqa: BLE001 - see send_sms's own docstring
            log.warning("notify: sms delivery failed after retries: %s", describe_failure(exc))


def _notify_channels(event: dict) -> tuple[bool, bool]:
    """(push, sms) for one event that is allowed to interrupt someone.

    SMS is the account's "text me on LOUD alerts" setting (`sms_on_loud`), and
    it costs money per message. It used to go out for every event the gate let
    through -- including a custom rule's `soft` warning, and a rule the person
    set to "push". So:

      * an event carrying `notify` (a configured rule's own choice; the device
        sends it for custom-rule events) is honoured: "silent" -> nothing,
        "sms" -> push and SMS whatever the severity.
      * "push" is the DEFAULT rule setting and the only non-silent choice the
        app offers, so it means "notify the usual way", exactly like an event
        with no choice: push always, SMS only for a `loud` event. Reading it as
        "never SMS" silently stopped every critical custom rule from texting
        an owner who had turned on "text me on loud alerts".

    `send_sms` still applies the account's own switch and phone number.
    """
    choice = str(event.get("notify") or "").strip().lower()
    if choice == "silent":
        return False, False
    if choice == "sms":
        return True, True
    return True, str(event.get("severity", "")) == "loud"


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


#: Rules one Activate may carry. A proposal's rules come from one sentence; a
#: few is normal, and each is evaluated on the device every frame.
MAX_ACTIVATED_RULES = 32
#: Settings one mode request may carry (a preset defines a handful).
MAX_PRESET_SETTINGS = 32


class ProtectionBody(BaseModel):
    """Explicit stop/resume of monitoring. Stopping must be confirmed."""

    action: str = Field(pattern="^(stop|resume)$")
    confirm: bool = False


class PresetBody(BaseModel):
    # Re-sent to the device on every poll until it converges, so its size is
    # paid on every ingest response, not once.
    preset: str = Field(max_length=128)
    # Both optional, so an app that sends only `preset` behaves exactly as it
    # always has (see set_preset).
    #
    # draft_version: the `draft_version` /v1/ai/configure returned for the
    # proposal the person is looking at. Activate is refused (409) if a newer
    # preview has replaced it, instead of silently running something else.
    draft_version: Optional[str] = Field(default=None, max_length=64)
    # rules: the proposal's rules AS REVIEWED -- the person may have changed
    # sensitivity, Push/Silent, "confirm for" and "then wait". Re-validated
    # server-side exactly like a model's proposal; never trusted as sent.
    rules: Optional[list[dict]] = Field(default=None, max_length=MAX_ACTIVATED_RULES)
    # settings: owner-tuned values for the named mode, validated against its
    # catalogue schema (presets.yaml `settings:`, docs/PRESET_LIBRARY.md) and
    # delivered as the "mode:variant@key=value" name the device applies.
    settings: Optional[dict[str, Any]] = Field(default=None, max_length=MAX_PRESET_SETTINGS)
    # keep_rules: keep the custom rules running beside the new mode. Only an
    # additive role (core/roles.py) can; anywhere else it is refused with
    # `incompatible_policy` rather than silently dropping the rules. None: keep
    # them exactly when the device can (additive role, mandatory module).
    keep_rules: Optional[bool] = None


class NodeBody(BaseModel):
    """Turn one companion monitoring node on or off (docs/M7_IMPLEMENTATION_MAP.md)."""

    node: str = Field(max_length=64)
    enabled: bool
    settings: Optional[dict[str, Any]] = Field(default=None, max_length=MAX_PRESET_SETTINGS)


class RuleToggleBody(BaseModel):
    enabled: bool


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
    # `id` is the same stable identity the mDNS TXT record carries. The app pins
    # it, and checks it HERE as well as in the TXT record: every backend answers
    # to the same `sensordyme-backend.local` name, and a cached name or address
    # can lead to a different backend than the one that was found, so "it says
    # service=sensordyme-cloud" alone is not "it is my backend".
    try:
        with db() as conn:
            conn.execute("SELECT 1").fetchone()
    except sqlite3.Error as exc:
        log.error("health: database unavailable (%s)", exc)
        return JSONResponse(
            status_code=503,
            content={"ok": False, "service": "sensordyme-cloud", "id": mdns.backend_id(),
                     "time": time.time(), "detail": "database unavailable"},
        )
    return {"ok": True, "service": "sensordyme-cloud", "id": mdns.backend_id(), "time": time.time()}


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
            return _rate_limit_key(candidate)
        # Configured but absent means the request did not come through the
        # proxy it was promised to. Fall through to the peer rather than
        # returning "" — one shared bucket is a limit, "" is also a limit, but
        # neither should silently look like a distinct client.
    return _rate_limit_key(request.client.host) if request.client is not None else "unknown"


def _rate_limit_key(address: str) -> str:
    """One rate-limit bucket per HOST, not per address.

    An IPv6 host is normally given a whole /64 and picks addresses from it
    freely (privacy addresses rotate on their own), so keying on the full
    address gave one machine 2^64 fresh buckets: measured, 40 password guesses
    from 40 addresses in one /64 were all evaluated and none throttled. The /64
    is the unit a single host controls. IPv4-mapped IPv6 is the IPv4 address it
    carries. Anything that does not parse is used as-is.
    """
    try:
        ip = ipaddress.ip_address(address.strip())
    except ValueError:
        return address
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return str(ip.ipv4_mapped)
        return str(ipaddress.ip_network(f"{ip}/64", strict=False))
    return str(ip)


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
            raise CodedError(409, "account_exists", "an account with this email already exists") from None
    return {"token": _issue_token(cur.lastrowid, email), "email": email}


@app.post("/v1/login")
def login(body: Credentials, request: Request):
    if not LOGIN_LIMITER.allow(_client_ip(request)):
        raise HTTPException(429, "too many attempts — try again in a few minutes")
    email = body.email.strip().lower()
    # Per account as well as per address: see LOGIN_FAILURE_LIMITER. Checked
    # before the lookup and applied to unknown emails too, so a throttled
    # answer says nothing about whether an account exists.
    if LOGIN_FAILURE_LIMITER.blocked(email):
        raise HTTPException(429, "too many attempts — try again in a few minutes")
    with db() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if row is None:
        _hash_pw(body.password, _NO_SUCH_USER_SALT)
        LOGIN_FAILURE_LIMITER.record(email)
        raise HTTPException(401, "wrong email or password")
    if not hmac.compare_digest(_hash_pw(body.password, row["salt"]), row["pw_hash"]):
        LOGIN_FAILURE_LIMITER.record(email)
        raise HTTPException(401, "wrong email or password")
    return {"token": _issue_token(row["id"], row["email"]), "email": row["email"]}


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
    account = f"uid:{uid}"
    if CLAIM_FAILURE_LIMITER.blocked(account):
        raise HTTPException(429, "too many attempts — try again in a few minutes")
    with db() as conn:
        row = conn.execute("SELECT * FROM devices WHERE claim_code = ?", (body.claim_code.strip(),)).fetchone()
        if row is None:
            CLAIM_FAILURE_LIMITER.record(account)
            raise HTTPException(404, "unknown claim code")
        if row["owner_id"] is not None and row["owner_id"] != uid:
            CLAIM_FAILURE_LIMITER.record(account)
            raise CodedError(409, "device_already_claimed", "device already claimed by another account")
        # Conditional, not a blind overwrite: two accounts holding the same code
        # could both pass the check above before either wrote, and both got a
        # 200 while the last writer silently took the device.
        claimed = conn.execute(
            "UPDATE devices SET owner_id = ? WHERE id = ? AND (owner_id IS NULL OR owner_id = ?)",
            (uid, row["id"], uid),
        )
        if claimed.rowcount == 0:
            raise CodedError(409, "device_already_claimed", "device already claimed by another account")
    return {"ok": True, "device_id": row["id"], "name": row["name"]}


def _owned_device(conn: sqlite3.Connection, device_id: int, uid: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM devices WHERE id = ? AND owner_id = ?", (device_id, uid)).fetchone()
    if row is None:
        raise CodedError(404, "device_not_found", "no such device on this account")
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
                # Kept even when health is withheld as stale: it identifies the
                # camera, it does not claim it is healthy.
                "hardware_id": r["hardware_id"],
                # What the app renders as "is it watching": the typed contract in
                # docs/APP_STATUS_CONTRACT.md, "unknown" once the device goes quiet.
                "status": _device_status(r, now),
                # Whether the backend accepts this device's credentials, and why not
                # (revoked, clock, unsigned): "offline" alone hid a lockout.
                "credentials": _credentials_status(r),
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
            "SELECT id, timestamp, type, confidence, signals, severity, source, duration, model FROM events "
            "WHERE device_id = ? ORDER BY id DESC LIMIT ?",
            (device_id, limit),
        ).fetchall()
    return {"events": [{**dict(r), "model": _stored_model(r["model"])} for r in rows]}


def _stored_model(raw: Optional[str]) -> Optional[dict]:
    """The stored identity, or None for history (and for anything unreadable)."""
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


#: Why a rule sent with a NAMED mode is not run. Stated, not silently dropped.
#:
#: The device has exactly one way to run delivered rules: as AI Custom, which
#: replaces the mode's own module (services/runtime.py saves every delivered
#: rule set with `overrides_mode` on, and core/monitoring_mode.py runs only the
#: rules in that mode), and an explicit mode switch exits it. So "this mode,
#: plus these rules on top" is not something any device can do today --
#: delivering the rules would switch the mode's own detection OFF.
_NOT_BESIDE_A_MODE = (
    "Custom rules can't run alongside a monitoring mode on this device yet: "
    "running them replaces the mode's own detection. The mode was turned on "
    "without this rule."
)

#: The same refusal on a device whose role runs custom rules BESIDE its mode
#: (core/roles.py additive_custom_rules, M0.1). There "running them replaces the
#: mode's own detection" is false: activated on their own, rules run alongside
#: the mode, which keeps running. Only a rule sent WITH a mode switch is not run.
_BESIDE_A_MODE_ON_ITS_OWN = (
    "This device runs custom rules alongside its monitoring mode, but a rule sent "
    "together with a mode switch is not turned on. The mode was turned on without "
    "this rule; activate the rule on its own to add it beside the mode."
)


def _rules_run_beside(role, preset: str) -> bool:
    """Would custom rules run BESIDE `preset` on a device with `role` (additive)?"""
    if role is None or not role.additive_custom_rules:
        return False
    base = (preset or "").partition(SETTINGS_SEPARATOR)[0].partition(":")[0]
    module = (load_presets().get(base) or {}).get("module")
    return module in role.mandatory_modules


def _beside_a_mode_unsupported(rules: list, additive: bool = False) -> list[dict]:
    """`unsupported` entries for rules that came with a named mode.

    `additive`: the device runs rules beside this mode when they are activated
    on their own (see _BESIDE_A_MODE_ON_ITS_OWN), so the reason says that.
    """
    entries = []
    for rule in rules or []:
        if not isinstance(rule, dict):
            continue
        target = str(rule.get("target") or rule.get("detection") or "").strip()
        predicate = str(rule.get("predicate") or "").strip()
        entries.append({
            "request": str(rule.get("description") or f"{predicate} {target}".strip())[:500],
            "reason": _BESIDE_A_MODE_ON_ITS_OWN if additive else _NOT_BESIDE_A_MODE,
            "requires": "activate it on its own; it then runs beside the mode" if additive else
                        "activate it on its own as a custom configuration (no mode), "
                        "or device support for rules beside a mode",
            "support": "unsupported",
            "alternatives": [],
        })
    return entries


#: The fields a rule may carry back from the app. Everything else is dropped:
#: the rule is about to be stored and shipped to a device on every change.
_ACTIVATED_RULE_KEYS = (
    "target", "detection", "predicate", "secondary", "description", "zone",
    "persist_s", "min_count", "confidence", "severity", "cooldown_s",
    "schedule", "notify", "timezone",
    # M6: per-subject dwell, and the rule's identity / on-off state.
    "dwell_s", "id", "enabled",
)

#: An IANA zone name as the device's RuleEngine reads it ("Europe/London",
#: "America/Argentina/Buenos_Aires", "Etc/GMT+5"): the zone a rule's `schedule`
#: ("23:00-06:00") is meant in -- the owner's, sent by the app, since the
#: device's own clock zone is whatever the image shipped with. Anything else is
#: dropped rather than shipped to a device to interpret.
_TIMEZONE_RE = re.compile(r"^[A-Za-z0-9/_+\-]{1,64}$")


def _valid_timezone(value) -> Optional[str]:
    return value if isinstance(value, str) and _TIMEZONE_RE.match(value) else None


def _client_rules(rules: list) -> list[dict]:
    """The reviewed rules as the app sent them, reduced to known fields with
    bounded strings, ready for `_validate_proposal` (which does the real
    checking). `caveat` is dropped on purpose: validation re-derives it."""
    cleaned = []
    for rule in rules or []:
        if not isinstance(rule, dict):
            continue
        kept = {}
        for key in _ACTIVATED_RULE_KEYS:
            if key not in rule:
                continue
            value = rule[key]
            kept[key] = _clean_text(value)[:500] if isinstance(value, str) else value
        if "timezone" in kept and _valid_timezone(kept["timezone"]) is None:
            del kept["timezone"]
        if "enabled" in kept and not isinstance(kept["enabled"], bool):
            del kept["enabled"]
        if "id" in kept and not (isinstance(kept["id"], str) and _RULE_ID.match(kept["id"])):
            del kept["id"]
        cleaned.append(kept)
    return cleaned


def _loads_dict(raw: Optional[str]) -> dict:
    try:
        value = json.loads(raw or "{}")
    except (ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _dict_field(source: dict, key: str) -> dict:
    """source[key] when it is a dict (the same object, not a copy), else {}.

    Reads the field once, so the isinstance check narrows the value that is
    returned (`d.get(k) if isinstance(d.get(k), dict) else {}` checks one call
    and returns another, which a type checker cannot see is the same dict)."""
    value = source.get(key)
    return value if isinstance(value, dict) else {}


def _reported_role(device: sqlite3.Row):
    """The role the device last reported (core/roles.py), or None.

    Either shape: the describe() dict in the monitoring report, or the bare name
    the health report and the protection block carry (see _role_name).
    """
    from core.roles import ROLES

    reported = _monitoring_payload(device["monitoring"]) or {}
    try:
        health = json.loads(device["health"] or "null")
    except (ValueError, TypeError):
        health = None
    health = health if isinstance(health, dict) else {}
    health_monitoring = _dict_field(health, "monitoring")
    protection = _dict_field(reported, "protection")
    name = _role_name(reported.get("role"), health_monitoring.get("role"), protection.get("role"))
    return ROLES.get(name) if name is not None else None


def _device_supports(device: sqlite3.Row, feature: str) -> bool:
    """Whether the device's software reported `feature` (core/presets.py
    DEVICE_POLICY_FEATURES). Firmware that predates the list reports none."""
    features = (_monitoring_payload(device["monitoring"]) or {}).get("features")
    return isinstance(features, list) and feature in features


def _zone_names(device: sqlite3.Row) -> list[str]:
    """The names of the areas drawn for this device (POST .../zones)."""
    try:
        zones = json.loads(device["zones"] or "[]")
    except (ValueError, TypeError):
        return []
    if not isinstance(zones, list):
        return []
    return [str(z.get("name")) for z in zones if isinstance(z, dict) and z.get("name")]


def _role_refusal(device: sqlite3.Row, preset: str, presets: dict) -> Optional[str]:
    from core.roles import preset_violation

    role = _reported_role(device)
    return preset_violation(role, preset, presets) if role is not None else None


@app.post("/v1/devices/{device_id}/protection")
def set_protection(device_id: int, body: ProtectionBody, uid: int = Depends(current_user)):
    """Explicitly stop or resume monitoring on a device (M0.1).

    Stopping is never implied by any other command and must carry
    `confirm: true`. The device reports "stopped" until it is resumed or
    restarted; the request is delivered once and cleared on convergence.
    """
    if body.action == "stop" and not body.confirm:
        raise CodedError(400, "confirmation_required", "stopping monitoring requires confirm: true")
    with db() as conn:
        device = _owned_device(conn, device_id, uid)
        reported = _monitoring_payload(device["monitoring"])
        if reported and "protection" not in reported:
            # Firmware older than M0.1 cannot stop or resume. Answering ok here
            # made the app believe a device had stopped that never would, and
            # the request was re-sent on every poll for ever.
            raise CodedError(409, "firmware_unsupported",
                             "this device's software does not support stop/resume yet; update it first")
        # Each request has an id the device echoes once it has applied it, so a
        # stop still in flight cannot be mistaken for a later resume (or the
        # reverse) by comparing states alone.
        command_id = secrets.token_hex(6)
        requested_at = time.time()
        # M8.3: the request's record, kept after delivery ends so its outcome
        # (confirmed / mismatch / timed_out) can be reported, with the device's
        # current protection refusal as the baseline a new refusal must differ from.
        record = {"id": command_id, "action": body.action, "requested_at": requested_at,
                  "applied_at": None, "applied_state": None,
                  "refusal_baseline": _rejection_fingerprint(_stored_last_rejection(device))}
        conn.execute(
            "UPDATE devices SET desired_protection = ?, protection_request = ? WHERE id = ?",
            (json.dumps({"action": body.action, "confirm": body.confirm, "id": command_id,
                         "requested_at": requested_at}), json.dumps(record), device_id),
        )
    return {"ok": True, "desired_protection": body.action, "command_id": command_id,
            "timeout_s": PROTECTION_ACK_TIMEOUT_S}


@app.post("/v1/devices/{device_id}/preset")
def set_preset(device_id: int, body: PresetBody, uid: int = Depends(current_user)):
    """Switch mode, or (preset "") activate AI Custom.

    Response: {ok, desired_preset, rules_active, rules_version, unsupported}.
      rules_active   how many custom rules are now set to run (0 for a mode)
      rules_version  the fingerprint the device will report once it has them
                     ("" when no custom rules are active)
      unsupported    rules from this request that will NOT run, with why
    """
    with db() as conn:
        device = _owned_device(conn, device_id, uid)
        # Store the current name for a mode, never a superseded one. The device
        # will apply and report the current name, so a stored alias could never
        # match on convergence and would re-send on every poll for ever.
        normalized = resolve_preset(body.preset.strip().rstrip(":"))
        if body.settings and not normalized:
            raise CodedError(
                409, "incompatible_policy",
                "settings belong to a monitoring mode; name the mode they are for "
                "(activating custom rules takes no mode settings)",
            )
        # Only a mode that exists. The device ignores an unknown name ("ignoring
        # unknown preset") and never reports it back, so storing one made ingest
        # re-send it on every poll, for ever, with the app showing the switch as
        # pending indefinitely.
        if normalized:
            presets = load_presets()
            head = normalized.partition(SETTINGS_SEPARATOR)[0]
            base, _, variant = head.partition(":")
            if base not in presets or (
                variant and variant not in (presets[base].get("variants") or {})
            ):
                planned = load_planned().get(base)
                if planned is not None:
                    raise CodedError(400, "unknown_preset",
                                     f"'{base}' is planned, not available yet: {planned.get('reason') or ''}".strip())
                raise CodedError(400, "unknown_preset", f"unknown mode '{body.preset}'")
            # The device's role (as the device last reported it) may forbid this
            # mode. Refuse here with the reason; the device would refuse it too.
            refusal = _role_refusal(device, head, presets)
            if refusal:
                raise CodedError(409, "role_forbids_preset", refusal)
            # Owner settings: checked against the mode's schema, then carried in
            # the name the device applies ("mode:variant@key=value"). `settings`
            # in the body replaces any suffix the name already had.
            try:
                settings = (validate_settings(presets[base], body.settings) if body.settings is not None
                            else parse_settings(normalized, presets))
            except PresetError as exc:
                raise CodedError(400, "invalid_setting", exc.reason) from None
            normalized = encode_settings(head, settings)
            if settings and not _device_supports(device, "preset_settings"):
                # Older firmware reads "mode@key=value" as an unknown variant and
                # never converges; refuse instead of re-sending it for ever.
                raise CodedError(409, "firmware_unsupported",
                                 "this device's software cannot apply mode settings yet; update it first")
            # Role floors win over any setting (core/roles.py parameter floors).
            refusal = _role_refusal(device, normalized, presets)
            if refusal:
                raise CodedError(409, "role_forbids_preset", refusal)

        # Custom rules beside the new mode: only where the role runs them
        # additively beside this mode's module (core/roles.py). Asked for
        # anywhere else, it is a conflict -- one vision module runs at a time,
        # and on this device custom rules replace the mode -- never a silent drop.
        beside = bool(normalized) and _rules_run_beside(_reported_role(device), normalized)
        if body.keep_rules is True and normalized and not beside:
            raise CodedError(
                409, "incompatible_policy",
                f"custom rules cannot run beside '{normalized.partition(SETTINGS_SEPARATOR)[0]}' on this "
                "device: here they replace the mode. Choose the mode or the custom rules.",
            )
        keep_rules = (beside and body.keep_rules is not False
                      and bool(device["custom_monitoring_wanted"]) and bool(device["ai_rules"]))

        draft_json = device["ai_rules_draft"]
        # The proposal on screen must still be the one stored. A newer preview
        # (the debounce firing while the person reads the last one) used to be
        # what Activate promoted, so the device ran a configuration nobody had
        # seen. With no draft left, the only acceptable version is the one
        # already promoted -- a retried Activate, not a stale one.
        if body.draft_version is not None:
            expected = (
                _rules_version(draft_json) if draft_json
                else (device["ai_rules_promoted_from"] or "")
            )
            # Plain equality: the version is not a secret, and compare_digest
            # raises TypeError on non-ASCII text (a 500 for a bad request).
            if not expected or body.draft_version != expected:
                raise CodedError(
                    409, "draft_superseded",
                    "this proposal was replaced by a newer one; review it again before activating",
                )

        # An empty preset IS the app's request for AI Custom (AIConfigureView's
        # Apply calls this with preset=""). Anything else is an explicit
        # request to leave custom monitoring in charge, whatever ai_rules
        # still says from a previous configuration.
        conn.execute(
            "UPDATE devices SET desired_preset = ?, custom_monitoring_wanted = ?, preset_rejection = NULL,"
            " rejection_baseline = ? WHERE id = ?",
            (normalized, 1 if normalized == "" or keep_rules else 0, _baseline_with(device, "preset"),
             device_id),
        )

        if normalized != "":
            # A named mode never promotes or discards the draft (the
            # pre-existing contract: reopening Configure-with-AI still shows what
            # was last configured), and never runs rules -- see
            # _NOT_BESIDE_A_MODE. Rules that came with it are reported back.
            unsupported = _beside_a_mode_unsupported(body.rules or [], additive=beside)
            if not keep_rules:
                return {
                    "ok": True, "desired_preset": normalized, "rules_active": 0,
                    "rules_version": "", "unsupported": unsupported,
                }
            # The rules already running beside the mode stay: a new activation
            # re-delivers them with the switch, because a device that switches
            # mode exits custom monitoring before re-applying them.
            conn.execute(
                "UPDATE devices SET ai_rules_activation = COALESCE(ai_rules_activation, 0) + 1 WHERE id = ?",
                (device_id,),
            )
            kept = conn.execute(
                "SELECT ai_rules, ai_rules_activation FROM devices WHERE id = ?", (device_id,)
            ).fetchone()
            return {
                "ok": True, "desired_preset": normalized,
                "rules_active": len(_rules_payload(kept["ai_rules"])),
                "rules_version": _rules_version(kept["ai_rules"], kept["ai_rules_activation"] or 0),
                "unsupported": unsupported,
            }

        # AI Custom. This is the ONLY act of acceptance anywhere in the flow,
        # so it is where a previewed proposal (see /v1/ai/configure, which only
        # ever writes a draft) becomes the one the device actually runs.
        promoted: Optional[str] = None
        promoted_from = device["ai_rules_promoted_from"]
        rejected: list[dict] = []
        if body.rules is not None:
            # The reviewed rules win over the stored copy: the person may have
            # changed them. Everything else (explanation, source) comes from
            # the draft they were reviewing, or the running configuration.
            base_proposal = _loads_dict(draft_json) if draft_json else _loads_dict(device["ai_rules"])
            prior_unsupported = [u for u in (base_proposal.get("unsupported") or []) if isinstance(u, dict)]
            proposal = {
                **base_proposal,
                "preset": "",
                "rules": _client_rules(body.rules),
                "unsupported": [],
            }
            validated = _validate_proposal(proposal, device=device)
            rejected = list(validated["unsupported"])
            validated["unsupported"] = prior_unsupported + rejected
            promoted = json.dumps(validated)
            if draft_json:
                promoted_from = _rules_version(draft_json)
        elif draft_json:
            # Today's behaviour for an app that sends no rules: the draft as
            # previewed. The draft is consumed, so a second Activate has nothing
            # left to re-promote and `ai_rules` stays the record of what runs.
            promoted = draft_json
            promoted_from = _rules_version(draft_json)

        if promoted is not None:
            # Each rule gets the id the app toggles it by (PATCH .../rules/{id}).
            stored = _loads_dict(promoted)
            if isinstance(stored.get("rules"), list):
                stored["rules"] = _with_rule_ids([r for r in stored["rules"] if isinstance(r, dict)])
                promoted = json.dumps(stored)
            conn.execute(
                "UPDATE devices SET ai_rules = ?, ai_rules_draft = NULL, ai_rules_promoted_from = ?,"
                " ai_rules_activation = COALESCE(ai_rules_activation, 0) + 1 WHERE id = ?",
                (promoted, promoted_from, device_id),
            )
        else:
            # Re-activating what is already stored. Still a new activation, so
            # it is delivered even to a device that kept reporting the version
            # from before it left custom monitoring.
            conn.execute(
                "UPDATE devices SET ai_rules_activation = COALESCE(ai_rules_activation, 0) + 1"
                " WHERE id = ?",
                (device_id,),
            )
        row = conn.execute(
            "SELECT ai_rules, ai_rules_activation FROM devices WHERE id = ?", (device_id,)
        ).fetchone()
    active_rules = _rules_payload(row["ai_rules"])
    return {
        "ok": True,
        "desired_preset": normalized,
        "rules_active": len(active_rules),
        "rules_version": _rules_version(row["ai_rules"], row["ai_rules_activation"] or 0),
        "unsupported": rejected,
    }


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
        except (TypeError, ValueError, LookupError):
            # LookupError, not just IndexError: a point sent as an object
            # ({"x": .., "y": ..}) raises KeyError on p[0], which escaped as a 500.
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


# ------------------------------------------------------------------------- preset library (M6)
# The catalogue (presets.yaml, core/presets.py), what each preset needs and how
# far its models have been validated (derived from core/model_registry.py), and
# per device: which presets it can run right now, and the custom rules it runs.
# Field-by-field contract: docs/PRESET_LIBRARY.md.

_CATALOG_CACHE: dict[str, list[dict]] = {}
_CATALOG_LOCK = threading.Lock()
_TIERS_CACHE: dict[str, dict] = {}


def _catalog() -> list[dict]:
    """The catalogue entries, built once per process (files ship with the image)."""
    with _CATALOG_LOCK:
        if "entries" not in _CATALOG_CACHE:
            _CATALOG_CACHE["entries"] = catalog_entries()
        return _CATALOG_CACHE["entries"]


@app.get("/v1/presets")
def list_presets(uid: int = Depends(current_user)):
    return {"catalog_version": catalog_version(), "presets": _catalog()}


@app.get("/v1/capability-tiers")
def capability_tiers(uid: int = Depends(current_user)):
    """How far each product capability has been shown to work (core/model_registry.py).
    `production_validated` is the only field that may make the app say "validated"
    for production; it is derived, never stored, and false for every capability today."""
    from core.model_registry import capability_tiers as tiers
    from core.model_registry import load_registry

    with _CATALOG_LOCK:
        if "tiers" not in _TIERS_CACHE:
            registry = load_registry()
            served = tiers(registry)
            served["nodes"] = registry_node_tiers(registry)
            _TIERS_CACHE["tiers"] = served
        return _TIERS_CACHE["tiers"]


def _registry_document() -> dict:
    """core/model_registry.yaml as data, for sections the typed loader does not model."""
    from core.model_registry import DEFAULT_PATH

    try:
        data = yaml.safe_load(Path(DEFAULT_PATH).read_text())
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def registry_node_tiers(registry, document: Optional[dict] = None) -> list[dict]:
    """Each monitoring node's validation tier, never above the components it uses.

    Reads `monitoring_nodes` from the registry (a map or a list; absent = []). A
    node's served `tier` is the LOWEST of its declared tier and the claimed tier
    of every product capability it depends on: a node is only as validated as
    its weakest input (M7 directive: no node claims stronger evidence than its
    components). A dependency may name a product capability id or a capability
    kind (objects, face, hands, pose); a kind counts as its weakest capability."""
    from core.model_registry import TIERS, tier_rank

    document = _registry_document() if document is None else document
    raw = document.get("monitoring_nodes")
    if isinstance(raw, dict):
        items = [{"id": k, **(v if isinstance(v, dict) else {})} for k, v in raw.items()]
    elif isinstance(raw, list):
        items = [n for n in raw if isinstance(n, dict) and isinstance(n.get("id"), str)]
    else:
        return []
    by_id = {c.id: c.claimed_tier for c in registry.capabilities}
    by_kind: dict[str, list[str]] = {}
    for cap in registry.capabilities:
        by_kind.setdefault(cap.kind, []).append(cap.claimed_tier)
    out = []
    for node in items:
        declared = next((node[k] for k in ("claimed_tier", "tier", "validation") if node.get(k) in TIERS),
                        TIERS[0])
        deps = next((node[k] for k in ("product_capabilities", "depends_on", "capabilities", "requires")
                     if isinstance(node.get(k), list)), [])
        tier, capped_by = declared, []
        for dep in (d for d in deps if isinstance(d, str)):
            dep_tiers = [by_id[dep]] if dep in by_id else by_kind.get(dep, [])
            if not dep_tiers:
                # A dependency with no evidence entry at all: nothing supports it.
                dep_tiers = [TIERS[0]]
            weakest = min(dep_tiers, key=tier_rank)
            if tier_rank(weakest) < tier_rank(tier):
                tier = weakest
            if tier_rank(weakest) < tier_rank(declared):
                capped_by.append(dep)
        out.append({
            "id": node["id"][:64],
            "declared_tier": declared,
            "tier": tier,
            "capped_by": capped_by,
            "depends_on": [d for d in deps if isinstance(d, str)],
            "experimental": node.get("experimental") is True,
            "device_verified": tier_rank(tier) >= tier_rank("device_verified"),
            "production_validated": tier == "production_permitted",
        })
    return out


def _reported_preset(row: sqlite3.Row) -> tuple[Optional[str], Optional[str], dict]:
    """(base, variant, settings) of the preset the device last reported."""
    name = row["active_preset"]
    if not name:
        return None, None, {}
    resolved = resolve_preset(str(name))
    head = resolved.partition(SETTINGS_SEPARATOR)[0]
    base, _, variant = head.partition(":")
    try:
        settings = parse_settings(resolved)
    except PresetError:
        settings = {}
    return base, variant or None, settings


#: What each component state means for the capability it provides.
_COMPONENT_TO_CAPABILITY_STATE = {
    "ok": "available", "idle": "idle", "stale": "impaired", "failed": "impaired",
    "unavailable": "unsupported", "unknown": "unknown",
    "paused": "paused",  # M8.2: monitoring stopped on purpose; not a fault
}


def capability_states(status: dict) -> dict[str, dict]:
    """{capability kind: {state, reason, source}} from what the device REPORTS.

    States: available | idle (the device has it, nothing uses it now) | starting
    | paused (monitoring stopped on purpose, M8.2) | impaired | unsupported | unknown. A kind the device does not report is
    `unknown` -- a device reports only what its current mode uses, so silence
    is not evidence that a preset it is not running would work."""
    out: dict[str, dict] = {}
    caps = status.get("capabilities") or {}
    for name in caps.get("available") or []:
        out[name] = {"state": "available", "reason": None, "source": "monitoring"}
    for name in caps.get("starting") or []:
        out[name] = {"state": "starting", "reason": None, "source": "monitoring"}
    for name in caps.get("paused") or []:
        out[name] = {"state": "paused", "reason": "monitoring is stopped", "source": "monitoring"}
    for item in caps.get("impaired") or []:
        if isinstance(item, dict) and item.get("name"):
            reason = item.get("reason")
            unsupported = bool(reason) and re.search(r"\bnot?\b", str(reason)) and re.search(
                r"\b(configured|provided)\b", str(reason))
            out[item["name"]] = {"state": "unsupported" if unsupported else "impaired",
                                 "reason": reason, "source": "monitoring"}
    perception = status.get("perception") or {}
    for component, comp in (perception.get("components") or {}).items():
        kind = COMPONENT_CAPABILITY.get(component)
        if kind is None:
            continue
        state = _COMPONENT_TO_CAPABILITY_STATE.get(comp.get("state"), "unknown")
        known = out.get(kind)
        # The module's own impairment verdict wins over a healthy component: the
        # component may be fine while the module cannot use it.
        if known and known["state"] in ("impaired", "unsupported") and state in ("available", "idle"):
            continue
        # Stopped on purpose (M8.2): a component that is merely stale or idle
        # because nothing runs stays "paused"; a failed one is still a fault.
        if known and known["state"] == "paused" and comp.get("state") in ("stale", "idle", "ok"):
            continue
        out[kind] = {"state": state, "reason": comp.get("reason"), "source": "perception"}
    return out


def _preset_capabilities(entry: dict, states: dict[str, dict], stale: bool) -> tuple[list[dict], Optional[bool]]:
    """Per-capability rows for one preset, and whether it is ready (None = unknown)."""
    rows, ready = [], True
    required = list(entry.get("required_capabilities") or [])
    optional = [c for c in entry.get("optional_capabilities") or [] if c not in required]
    for name, is_required in [(c, True) for c in required] + [(c, False) for c in optional]:
        state = {"state": "unknown", "reason": None, "source": None} if stale else (
            states.get(name) or {"state": "unknown", "reason": None, "source": None})
        rows.append({"name": name, "required": is_required, **state})
        if not is_required:
            continue
        if state["state"] in ("impaired", "unsupported"):
            ready = False
        elif state["state"] != "available" and ready is not False:
            ready = None
    return rows, ready


@app.get("/v1/devices/{device_id}/presets")
def device_presets(device_id: int, uid: int = Depends(current_user)):
    with db() as conn:
        row = _owned_device(conn, device_id, uid)
    now = time.time()
    status = _device_status(row, now)
    role = _reported_role(row)
    presets = load_presets()
    base, variant, settings = _reported_preset(row)
    # A device in AI Custom (custom rules REPLACING the mode) runs no preset,
    # whatever its last selected preset was.
    active = base if base in presets and status["mode"] != "ai_custom" else None
    allowed = status["allowed_presets"]
    states = capability_states(status)
    reported_nodes = _reported_nodes(status["perception"]) or {}
    entries = []
    for entry in _catalog():
        code, reason = device_availability(
            entry, role=role, capabilities=status["capabilities"], stale=status["stale"], presets=presets)
        if code is None and role is None and isinstance(allowed, list) and entry["id"] not in allowed:
            # A role the backend does not know, described by the device itself.
            code, reason = "role_forbids_preset", f"this device's role allows only: {', '.join(allowed)}"
        if code is None and entry.get("availability") != "planned":
            # Device truth (M7): a component the device says it does not have at
            # all cannot be repaired by switching to it.
            for kind in entry.get("required_capabilities") or []:
                found = states.get(kind)
                if found and found["source"] == "perception" and found["state"] == "unsupported":
                    code = "capability_unsupported"
                    reason = f"{kind} is not available on this device ({found['reason'] or 'no model'})"
                    break
        preset = presets.get(entry["id"]) or {}
        is_active = entry["id"] == active
        capability_rows, ready = _preset_capabilities(entry, states, status["stale"])
        device_node = reported_nodes.get(entry["id"]) or next(
            (n for n in reported_nodes.values() if n.get("preset") == entry["id"]), None)
        entries.append({
            "id": entry["id"],
            # Selectable (not refused); NOT a claim that it would work -- see `ready`.
            "available": code is None,
            "code": code,
            "reason": reason,
            "active": is_active,
            # True: every required capability reported available. False: one is
            # impaired or missing. None: not known (not reported, or stale).
            "ready": ready if code is None else False,
            "capabilities": capability_rows,
            "validation_status": entry["validation"]["status"],
            # The device's own node validation claim when it reports one.
            "device_validation": (device_node or {}).get("validation"),
            "settings": effective_settings(preset, variant if is_active else None,
                                           settings if is_active else None) if preset else {},
        })
    return {
        "device_id": device_id,
        "catalog_version": catalog_version(),
        "active_preset": active,
        "active_variant": variant if active else None,
        "stale": status["stale"],
        "presets": entries,
    }


# ------------------------------------------------------------------ companion nodes (M7)
#: Used only when the device catalogue declares no companion nodes yet. Both read
#: shared person detections and run no model of their own; both are experimental
#: until the registry binds evidence to them.
_FALLBACK_NODE_CATALOG = {
    "people_counter": {
        "title": "People counter",
        "description": "People count, entries and exits, and presence in drawn areas, "
                       "from the shared person detections.",
        "required_capabilities": ["camera", "inference", "objects"],
        "experimental": True,
    },
    "zone_watch": {
        "title": "Zone watch",
        "description": "A person inside a drawn restricted area, and for how long.",
        "required_capabilities": ["camera", "inference", "objects"],
        "experimental": True,
    },
}
_NODE_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _as_node_catalog(nodes) -> dict[str, dict]:
    """{id: spec} from either shape a loader returns: a dict keyed by id, or the
    device's list of specs each carrying its own `id` (core.presets.load_companion_nodes)."""
    if isinstance(nodes, dict):
        return {str(k): (v if isinstance(v, dict) else {}) for k, v in nodes.items()}
    if isinstance(nodes, list):
        return {str(v["id"]): v for v in nodes if isinstance(v, dict) and v.get("id")}
    return {}


def node_catalog() -> dict[str, dict]:
    """Companion monitoring nodes a device can run beside its mode.

    From the device catalogue when it declares them (a core.presets loader, or a
    `nodes`/`companion_nodes` section in presets.yaml), else the fallback above."""
    try:
        from core import presets as preset_module
    except ImportError:
        return dict(_FALLBACK_NODE_CATALOG)
    for name in ("load_nodes", "load_companion_nodes"):
        loader = getattr(preset_module, name, None)
        if callable(loader):
            try:
                nodes = loader()
            except Exception:
                log.warning("node catalogue: %s failed; using the fallback", name, exc_info=True)
                continue
            catalog = _as_node_catalog(nodes)
            if catalog:
                return catalog
    try:
        document = preset_module._load(None)
    except (OSError, yaml.YAMLError):
        document = {}
    for key in ("nodes", "companion_nodes"):
        nodes = document.get(key)
        if isinstance(nodes, dict) and nodes:
            return {str(k): (v if isinstance(v, dict) else {}) for k, v in nodes.items()}
    return dict(_FALLBACK_NODE_CATALOG)


def _node_settings(entry: dict, settings: Optional[dict]) -> dict:
    """Owner settings for one node: scalars only, and only declared keys when declared."""
    if not settings:
        return {}
    declared = entry.get("settings")
    allowed = None
    if isinstance(declared, dict):
        allowed = set(declared)
    elif isinstance(declared, list):
        allowed = {s.get("key") for s in declared if isinstance(s, dict)}
    clean = {}
    for key, value in settings.items():
        if not isinstance(key, str) or not _KEY_PATTERN.fullmatch(key):
            raise CodedError(400, "invalid_setting", f"setting {str(key)[:64]!r} is not a valid name")
        if allowed is not None and key not in allowed:
            raise CodedError(400, "invalid_setting", f"node has no setting {key!r}")
        if isinstance(value, bool) or (isinstance(value, (int, float)) and math.isfinite(value)) or (isinstance(value, str) and len(value) <= 64):
            clean[key] = value
        else:
            raise CodedError(400, "invalid_setting", f"setting {key!r} must be a number, true/false or short text")
    return clean


_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _reported_nodes(perception: Optional[dict]) -> Optional[dict[str, dict]]:
    """{node id: reported node} from a perception block; None when not reported."""
    if not isinstance(perception, dict):
        return None
    return {n["id"]: n for n in perception.get("nodes") or [] if isinstance(n, dict) and n.get("id")}


def _node_applied(want: dict, reported: dict[str, dict], node_id: str) -> bool:
    """Whether the device's report already reflects the wanted on/off state.

    Enabled = reported in any state but `inactive`: a node the device holds but
    cannot run right now (`failed`, `degraded`) WAS applied -- re-sending would
    not repair it, and its state is shown from perception. Disabled = absent or
    `inactive`. Settings are not compared: the contract does not report them."""
    node = reported.get(node_id)
    running = node is not None and node.get("state") != "inactive"
    return running == bool(want.get("enabled"))


def _device_reports_perception(row: sqlite3.Row) -> Optional[bool]:
    """True/False: the device's last health report did/did not carry perception;
    None when it never reported health at all."""
    stored = _loads_dict(row["health"]) if row["health"] else None
    if not stored:
        return None
    monitoring = _monitoring_payload(row["monitoring"]) or {}
    return _reported_perception(stored, monitoring) is not None


@app.post("/v1/devices/{device_id}/nodes")
def set_node(device_id: int, body: NodeBody, uid: int = Depends(current_user)):
    """Turn a companion monitoring node on or off beside the device's mode.

    Queued like a mode switch: `desired_nodes` is delivered as `set_nodes` until
    the device's perception.nodes agree, then cleared. The device can still
    refuse (missing capability, role) -- that lands in status.nodes_rejection.
    Response: {ok, node, enabled, pending}."""
    catalog = node_catalog()
    node_id = body.node.strip()
    if not _NODE_ID.fullmatch(node_id) or node_id not in catalog:
        raise CodedError(400, "unknown_node",
                         f"no companion node {node_id[:64]!r}; known: {', '.join(sorted(catalog))}")
    entry = catalog[node_id]
    settings = _node_settings(entry, body.settings)
    with db() as conn:
        device = _owned_device(conn, device_id, uid)
        if _device_reports_perception(device) is False:
            # It would never acknowledge (no perception.nodes), so the request
            # would be re-sent on every poll for ever.
            raise CodedError(409, "firmware_unsupported",
                             "this device's software does not run companion monitoring nodes")
        desired = _loads_dict(device["desired_nodes"])
        desired[node_id] = {"enabled": body.enabled, "settings": settings}
        reported = _reported_nodes(_device_status(device, time.time())["perception"])
        if reported is not None and _node_applied(desired[node_id], reported, node_id) and not settings:
            # Already so on the device: nothing to deliver (idempotent repeat).
            desired.pop(node_id)
        rejection = _loads_dict(device["nodes_rejection"])
        conn.execute(
            "UPDATE devices SET desired_nodes = ?, nodes_rejection = ?, rejection_baseline = ? WHERE id = ?",
            (json.dumps(desired) if desired else None,
             None if not rejection or rejection.get("node") == node_id else device["nodes_rejection"],
             _baseline_with(device, "nodes"), device_id),
        )
    return {"ok": True, "node": node_id, "enabled": body.enabled, "pending": node_id in desired}


@app.get("/v1/devices/{device_id}/nodes")
def device_nodes(device_id: int, uid: int = Depends(current_user)):
    """Companion nodes: catalogue, what was asked, what the device reports, and tiers."""
    from core.model_registry import load_registry

    with db() as conn:
        row = _owned_device(conn, device_id, uid)
    status = _device_status(row, time.time())
    reported = _reported_nodes(status["perception"])
    tiers = {n["id"]: n for n in registry_node_tiers(load_registry())}
    desired = _loads_dict(row["desired_nodes"])
    entries = []
    for node_id, entry in sorted(node_catalog().items()):
        device_node = (reported or {}).get(node_id)
        tier = tiers.get(node_id)
        entries.append({
            "id": node_id,
            "title": str(entry.get("title") or node_id)[:64],
            "description": str(entry.get("description") or "")[:500],
            "required_capabilities": _strings(entry.get("required_capabilities"), 16)
            if isinstance(entry.get("required_capabilities"), list) else [],
            "experimental": entry.get("experimental") is not False,
            "desired": desired.get(node_id),
            # What the device runs, as it says; null when not reported (stale or old software).
            "reported": device_node,
            # The device's own claim first (it knows its models), then the registry's.
            "validation": (device_node or {}).get("validation") or (tier or {}).get("tier") or "exists_in_code",
            "registry_tier": tier,
        })
    return {
        "device_id": device_id,
        "stale": status["stale"],
        "reported": reported is not None,
        "nodes": entries,
        "rejection": status["nodes_rejection"],
    }


def _rule_view(rule: dict, *, index: Optional[int], status: str, reason: Optional[str]) -> dict:
    """One stored rule as GET /v1/devices/{id}/rules shows it."""
    from core.capabilities import registry

    target = str(rule.get("target") or rule.get("detection") or "").strip().lower()
    secondary = str(rule.get("secondary") or "").strip().lower()
    targets = [t for t in (target, secondary) if t]
    kinds: set[str] = set()
    reg = registry()
    for label in targets:
        found = {c.kind for c in reg.providers_of_label(label) if c.available} & {"objects", "pose", "face", "hands"}
        kinds |= found or {"objects"}
    view = {
        "id": rule.get("id"),
        "enabled": rule.get("enabled") is not False,
        "summary": str(rule.get("description") or f"{rule.get('predicate') or 'present'} {target}".strip()),
        "targets": targets,
        "zone": str(rule.get("zone") or "") or None,
        "required_capabilities": ["camera", "inference", *sorted(kinds)],
        "status": status,
        "reason": reason,
    }
    if rule.get("dwell_s") not in (None, ""):
        view["dwell_s"] = rule.get("dwell_s")
    return view


def _rules_overview(row: sqlite3.Row, now: float) -> dict:
    """The stored rules with what the DEVICE says about each (never inferred
    from what the backend accepted). Status precedence: disabled > waiting
    (area not drawn) > unknown (stale) > rejected > running | unknown."""
    from core.rules import _zone_key

    status = _device_status(row, now)
    wanted = bool(row["custom_monitoring_wanted"])
    version = _rules_version(row["ai_rules"], row["ai_rules_activation"] or 0)
    reported = _monitoring_payload(row["monitoring"]) or {}
    device_version = reported.get("version")
    running = status["rules"]["active"]
    rejected = status["rules"]["rejected"]
    drawn = {_zone_key(z) for z in _zone_names(row)}
    rules = _with_rule_ids(_stored_rules(row["ai_rules"]))
    out, delivered_index = [], 0
    for rule in rules:
        enabled = rule.get("enabled") is not False
        index = delivered_index if enabled else None
        if enabled:
            delivered_index += 1
        zone = str(rule.get("zone") or "").strip()
        if not enabled:
            state, reason = "disabled", "turned off in the app; not sent to the device"
        elif not wanted:
            state, reason = "disabled", ("custom rules are not active on this device (a monitoring mode was "
                                         "selected instead); activate them to run")
        elif zone and _zone_key(zone) not in drawn:
            state, reason = "waiting", f"area '{zone}' is not drawn yet; draw it in the app"
        elif status["stale"]:
            state, reason = "unknown", (status["reasons"] or ["the device is not reporting"])[0]
        elif rejected:
            state, reason = "rejected", "; ".join(rejected["reasons"]) or "the device refused this rule set"
        elif device_version != version or running is None:
            state, reason = "unknown", "not yet confirmed by the device"
        elif any(str(r).startswith(f"rule{index}:") for r in running):
            state, reason = "running", None
        else:
            state, reason = "unknown", "the device does not report this rule running"
        out.append(_rule_view(rule, index=index, status=state, reason=reason))
    return {
        "device_id": row["id"],
        "rules_version": version if wanted else "",
        "additive": bool(status["rules"]["additive"]),
        "custom_monitoring_wanted": wanted,
        "rules": out,
    }


@app.get("/v1/devices/{device_id}/rules")
def device_rules(device_id: int, uid: int = Depends(current_user)):
    with db() as conn:
        row = _owned_device(conn, device_id, uid)
    return _rules_overview(row, time.time())


@app.patch("/v1/devices/{device_id}/rules/{rule_id}")
def toggle_rule(device_id: int, rule_id: str, body: RuleToggleBody, uid: int = Depends(current_user)):
    """Turn one custom rule on or off. The stored set changes, so its version
    does too, and ingest delivers the enabled rules on the next poll (when
    custom rules are active); a disabled rule is never delivered."""
    with db() as conn:
        row = _owned_device(conn, device_id, uid)
        proposal = _loads_dict(row["ai_rules"])
        rules = _with_rule_ids(_stored_rules(row["ai_rules"]))
        match = next((r for r in rules if r["id"] == rule_id), None)
        if match is None:
            raise CodedError(404, "rule_not_found", f"no custom rule '{rule_id[:64]}' on this device")
        match["enabled"] = body.enabled
        proposal["rules"] = rules
        updated = conn.execute(
            "UPDATE devices SET ai_rules = ?, ai_rules_activation = COALESCE(ai_rules_activation, 0) + 1"
            " WHERE id = ? AND ai_rules IS ?",
            (json.dumps(proposal), device_id, row["ai_rules"]),
        ).rowcount
        if not updated:
            # Activated or toggled in the same instant; never overwrite that.
            raise CodedError(409, "draft_superseded", "the rules changed while this was being saved; reload them")
        row = conn.execute("SELECT * FROM devices WHERE id = ?", (device_id,)).fetchone()
    overview = _rules_overview(row, time.time())
    return {
        "ok": True,
        "rule": next(r for r in overview["rules"] if r["id"] == rule_id),
        "rules_version": overview["rules_version"],
        "rules_active": len(_rules_payload(row["ai_rules"])) if row["custom_monitoring_wanted"] else 0,
    }


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


#: Capabilities whose `available` is decided by probing THIS process's
#: environment (core/capabilities.py `_pose_available()`: is mediapipe importable
#: here). What the backend has installed says nothing about a device; the device
#: reports its own. Every other unavailable capability is unavailable everywhere
#: (no tracker, no eyewear or action model) and a device report cannot enable it.
_ENVIRONMENT_PROBED_CAPABILITIES = frozenset({"gesture.arm", "state.resting", "pose.body", "activity.gait"})


def device_capability_registry(device: Optional[sqlite3.Row], caps=None):
    """The capability registry as the TARGET DEVICE reports itself (M7), or None.

    None -- an old or never-reporting device, or one that reports no perception
    components -- means "keep today's behaviour" (the backend's own registry).
    Otherwise each capability of a kind the device reports a component for is
    adjusted to the device: a component it reports `unavailable` makes every
    capability of that kind unavailable, with the device's reason; a component
    it has makes an environment-probed capability available. Liveness does not
    matter here: what models a device has is configuration, not a moment."""
    caps = caps or _capabilities()
    if caps is None or device is None:
        return None
    stored = _loads_dict(device["health"]) if device["health"] else {}
    perception = _reported_perception(stored, _monitoring_payload(device["monitoring"]) or {})
    if perception is None:
        return None
    kinds = {COMPONENT_CAPABILITY[name]: comp for name, comp in perception["components"].items()
             if name in COMPONENT_CAPABILITY}
    if not kinds:
        return None
    adjusted = []
    for cap in caps.registry().capabilities:
        comp = kinds.get(cap.kind)
        if comp is None or comp["state"] == "unknown":
            adjusted.append(cap)
        elif comp["state"] == "unavailable":
            why = comp.get("reason") or "it has no model for it"
            adjusted.append(dataclasses.replace(
                cap, available=False,
                requires=f"{cap.requires or cap.summary}; this device reports no {cap.kind} capability ({why})"))
        elif not cap.available and cap.id in _ENVIRONMENT_PROBED_CAPABILITIES:
            adjusted.append(dataclasses.replace(cap, available=True))
        else:
            adjusted.append(cap)
    return caps.CapabilityRegistry(adjusted)


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


def _ai_configure_llm(prompt: str, reg=None) -> Optional[dict]:
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
    # The capability sheet the model is briefed with is the TARGET device's
    # (reg); its output is data, validated again below -- never executed.
    outcome = interpret(prompt, provider_name=provider_name, config=config, reg=reg)

    fallback = cloud_fallback_config()
    if not outcome.ok and fallback["enabled"] and provider_name != fallback["provider"]:
        # Explicitly enabled, and only after the local path failed.
        outcome = interpret(
            prompt, provider_name=fallback["provider"], config={"model": fallback["model"]}, reg=reg
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


# Keyword -> (mode, explanation). The path taken when the language model is
# UNREACHABLE -- which is the default on a deployment with no model server --
# so it has to be exactly as honest as the model path.
#
# A match names a MODE and nothing else. Each entry used to add a custom rule
# "on top" of the mode ("Added on top: an alert if someone is on the floor"),
# and the explanation promised it. No device could run it: delivered rules run
# only as AI Custom, which replaces the mode's own module (services/runtime.py
# saves every delivered rule set with `overrides_mode` on), and the app
# activates a mode proposal with its mode name, which never delivers rules. So
# the rule sat in a draft for ever while the person was told it was added.
# Shipping it with the mode would have been worse: it would have switched the
# mode's own detection (fall, drowsiness) OFF. See set_preset's
# _NOT_BESIDE_A_MODE.
_AI_FALLBACKS = [
    (("baby", "infant", "crib", "nursery", "toddler"), "baby_monitoring",
     ("Baby Monitoring: the mode watches for unusual stillness, frantic movement and "
      "safe/danger zones. Draw a safe zone around the crib (and any danger zones) in the "
      "app to enable area alerts.")),
    (("fall", "elder", "senior", "grandma", "grandpa"), "elderly_monitoring",
     ("Elderly Monitoring: immediate fall alerts (rapid drop then stillness — sitting down "
      "won't trigger it), staying active until movement resumes.")),
    (("security", "intruder", "door", "burglar", "lurk", "linger", "trespass"), "security_monitoring",
     ("Security: person-count changes, plus restricted-zone and lingering alerts once you draw "
      "zones in the app.")),
    (("drive", "driving", "car", "phone", "drowsy", "road"), "driver_safety",
     "Driver Safety: loud alerts for phone use while looking away and sustained eye closure."),
]


def _ai_fallback(prompt: str) -> dict:
    """Deterministic keyword configuration when the model is unavailable. Returns
    the same structured shape, honestly labeled with source=fallback: a mode,
    and no rules (see _AI_FALLBACKS for why)."""
    lowered = prompt.lower()
    for keywords, preset, explanation in _AI_FALLBACKS:
        if any(k in lowered for k in keywords):
            return {
                "preset": preset, "variant": "", "explanation": explanation,
                "rules": [], "settings": {}, "unsupported": [], "source": "fallback",
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


def _validate_proposal(data: dict, role=None, device: Optional[sqlite3.Row] = None) -> dict:
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

    # The target device's own capabilities when it reports them (M7), else the
    # backend's registry: a rule is checked against the camera that will run it.
    reg = device_capability_registry(device, caps) or caps.registry()
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
        # Through _number, not float(): rules now also arrive from the app's
        # review screen (set_preset), and "abc", NaN or 1e999 there was a 500.
        rule["persist_s"] = max(0.5, min(3600.0, _number(rule.get("persist_s"), 2.0)))
        rule["confidence"] = max(0.2, min(0.95, _number(rule.get("confidence"), 0.6)))
        rule["cooldown_s"] = max(10.0, min(86400.0, _number(rule.get("cooldown_s"), 60.0)))
        rule["min_count"] = int(max(1, min(100, _number(rule.get("min_count"), 1))))
        if rule.get("severity") not in ("info", "warning", "critical"):
            rule["severity"] = "warning"
        if "timezone" in rule and _valid_timezone(rule["timezone"]) is None:
            del rule["timezone"]
        if verdict.support is caps.Support.PARTIAL:
            # The user is told the rule is an approximation, on the rule itself,
            # so the caveat survives into the review screen and the rule list.
            rule["caveat"] = verdict.reason
        if rule.get("dwell_s") not in (None, ""):
            # Per-subject dwell (core/rules.py): an area is mandatory, and
            # persist_s mirrors it so firmware without dwell runs the nearest
            # honest equivalent ("the area occupied that long") instead of
            # refusing the set.
            # An unreadable value is left for the schema below to refuse.
            dwell = _number(rule.get("dwell_s"), -1.0)
            if dwell > 0:
                rule["dwell_s"] = max(1.0, min(3600.0, dwell))
                rule["persist_s"] = rule["dwell_s"]
        # The typed policy schema (jupiter/policy.py) is the last gate, whatever
        # produced the rule: a shape the schema refuses never reaches a device.
        problem = _policy_problem(rule, reg)
        if problem:
            unsupported.append({
                "request": described, "reason": problem, "requires": "a rule the device can evaluate as written",
                "support": "unsupported", "alternatives": [],
            })
            continue
        valid_rules.append(rule)

    if preset and valid_rules:
        # A mode proposal cannot carry rules the device would run: see
        # _NOT_BESIDE_A_MODE. Said here, at preview, rather than discovered
        # after Activate -- the review screen must not offer "Activate 1 rule"
        # for a rule that will never run.
        unsupported.extend(_beside_a_mode_unsupported(valid_rules, additive=_rules_run_beside(role, preset)))
        valid_rules = []

    data["preset"] = preset
    data["mode"] = "preset" if preset else "custom"
    data["rules"] = valid_rules
    data["unsupported"] = unsupported
    return _decodable_proposal(data)


def _policy_problem(rule: dict, reg=None) -> Optional[str]:
    """Why `rule` does not fit the typed policy schema, or None (it does)."""
    try:
        from jupiter.policy import PolicyError, policy_from_rule
    except ImportError:  # backend deployed without jupiter/: the checks above still ran
        return None
    try:
        policy_from_rule(rule, reg=reg)
    except PolicyError as exc:
        return str(exc)
    return None


def _policies_for(rules: list, zone_names: list[str]) -> list[dict]:
    """The typed policy (jupiter/policy.py) for each validated rule, in order."""
    try:
        from jupiter.policy import PolicyError, policy_from_rule
    except ImportError:
        return []
    out = []
    for rule in rules:
        try:
            out.append({"rule_id": rule.get("id"), **policy_from_rule(rule, zones=zone_names).as_dict()})
        except PolicyError:
            continue  # cannot happen after _validate_proposal; never invent one
    return out


def _number(value, default: float) -> float:
    """`value` as a finite float, or `default` for anything that is not one
    (missing, 0/empty as before, text, NaN, infinity, a bool)."""
    if isinstance(value, bool) or value in (None, "", 0):
        return float(default)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return number if math.isfinite(number) else float(default)


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


def _interpret_request(prompt: str, zone_names: list[str], reg=None) -> dict:
    """A proposal for `prompt`, from the most trustworthy interpreter that claims it.

      1. refusals (jupiter/policy.py refusal_for): identity, intent, emotion,
         sound, ... are refused with core/taxonomy.py's reason BEFORE any model
         sees them, so none can approximate "a stranger" as "a person";
      2. the deterministic policy parser, for the area/duration requests it
         reads completely ("someone stays inside this marked area for longer
         than 10 seconds") -- exact, and no model involved;
      3. the language model (Jupiter / Anthropic), whose output is validated
         like everything else;
      4. the keyword fallback.
    """
    try:
        from jupiter import policy as policy_module
    except ImportError:
        policy_module = None
    if policy_module is not None:
        refusals = policy_module.refusal_for(prompt)
        if refusals:
            return {
                "preset": "", "variant": "", "rules": [], "settings": {}, "unsupported": refusals,
                "explanation": "Nothing was set up. " + " ".join(r["reason"] for r in refusals),
                "source": "policy_guard",
            }
        parsed = policy_module.parse(prompt, zones=zone_names, reg=reg)
        if parsed is not None:
            return parsed.as_proposal()
    # Without device truth the call is exactly as before (reg=None, today's registry).
    proposal = _ai_configure_llm(prompt, reg=reg) if reg is not None else _ai_configure_llm(prompt)
    return proposal or _ai_fallback(prompt)


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
        # And across accounts: the per-account bound alone let twenty accounts
        # hold two calls each -- the whole default thread pool.
        if not AI_GLOBAL_CONCURRENCY.acquire(AI_GLOBAL_KEY):
            yield False
            return
        try:
            yield AI_RATE_LIMITER.allow(key)
        finally:
            AI_GLOBAL_CONCURRENCY.release(AI_GLOBAL_KEY)
    finally:
        AI_CONCURRENCY.release(key)


@app.post("/v1/ai/configure")
def ai_configure(body: AIConfigureBody, uid: int = Depends(current_user)):
    with db() as conn:
        device = _owned_device(conn, body.device_id, uid)
        role = _reported_role(device)
        zone_names = _zone_names(device)
    with _ai_slot(uid) as allowed:
        if not allowed:
            raise HTTPException(429, "too many AI requests at once; try again in a moment")
        # Interpreted and validated against the TARGET device's reported
        # capabilities (M7), not what this server process happens to have.
        reg = device_capability_registry(device)
        result = _validate_proposal(_interpret_request(body.prompt, zone_names, reg=reg), role=role,
                                    device=device)
    # The typed policy for every rule (docs/PROMPT_POLICIES.md): targets, area
    # and whether it is drawn, timing, capabilities, and plain-language
    # explanation of what is and is NOT monitored -- what the owner confirms.
    # Ids now, at preview, so the rules the owner reviews are the rules they
    # later toggle (Activate keeps ids it is given).
    result["rules"] = _with_rule_ids(result.get("rules") or [])
    result["policies"] = _policies_for(result["rules"], zone_names)
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
    draft_json = json.dumps(result)
    with db() as conn:
        updated = conn.execute(
            "UPDATE devices SET ai_rules_draft = ? WHERE id = ? AND owner_id = ?",
            (draft_json, body.device_id, uid),
        ).rowcount
    if not updated:
        # Same 404 and same wording as `_owned_device`, so losing the device
        # mid-request is indistinguishable from never having had it — there is
        # nothing here worth telling an attacker apart.
        raise HTTPException(404, "no such device on this account")
    # Which draft this is, so Activate can say "this one" (see set_preset) and
    # be refused if a newer preview has replaced it in the meantime.
    return {**result, "draft_version": _rules_version(draft_json)}


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
    # `signals` disambiguates when several rules watch the same target: the
    # device records the firing rule's own message there.
    rule = parse_rule_for_event(device["ai_rules"], event.get("type", ""), event.get("signals"))

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
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _model_identity(raw) -> Optional[dict]:
    """A device-reported model identity, validated at the boundary.

    Answers "which exact model produced this event": the module, and the object
    detector it ran — file name, sha256 of the file that was loaded, and the
    backend that executed it. Only these keys survive, every string is bounded,
    and a sha256 must look like one; anything else is dropped rather than
    stored, because a malformed identity must not cost the event itself.
    """
    if not isinstance(raw, dict):
        return None
    module = raw.get("module")
    out: dict = {"module": str(module)[:64] if isinstance(module, str) else None}
    detector = raw.get("detector")
    if isinstance(detector, dict) and isinstance(detector.get("name"), str):
        sha = detector.get("sha256")
        backend = detector.get("backend")
        out["detector"] = {
            "name": detector["name"][:128],
            "sha256": sha if isinstance(sha, str) and _SHA256.match(sha) else None,
            "backend": backend[:32] if isinstance(backend, str) else None,
        }
    else:
        out["detector"] = None
    return out if out["module"] or out["detector"] else None


#: A device's hardware id. `SD-SIM-` marks the virtual device (sim/); it is kept so
#: the app can link a simulated device's feed the same way, and is visibly simulated.
_HARDWARE_ID = re.compile(r"^SD-(SIM-)?[0-9A-F]{8}$")


def _hardware_id(health) -> Optional[str]:
    """The device's SD- id from its health report, if well-formed (core/identity.py)."""
    value = health.get("device_id") if isinstance(health, dict) else None
    return value if isinstance(value, str) and _HARDWARE_ID.match(value) else None


def _finite(value: Any) -> Any:
    """`value` with every non-finite float replaced by None.

    The request parser accepts NaN and 1e999; the response encoder refuses them.
    Stored verbatim in `health` or `monitoring`, one such value made every later
    GET /v1/devices for the account a 500 until the device reported again. The
    events loop below already guards its own floats; this is the same rule for
    the two blobs that are stored whole. (Current firmware sanitises before
    sending -- core/uplink.py's _json_safe -- so this is for any other client.)
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _clean_text(value)
    if isinstance(value, dict):
        return {_clean_text(str(k)): _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    return value


def _clean_text(value: str) -> str:
    """`value` with anything UTF-8 cannot carry (a lone surrogate) replaced.

    JSON can spell "\\ud800"; the request parser accepts it, `json.dumps`
    stores it escaped, and the response encoder then refuses it. One such
    string in a device's `health` or `monitoring` made every later
    GET /v1/devices for the account fail until the device reported again --
    `monitoring` is kept until replaced, so possibly for ever. In an event
    field it cost the event (sqlite refuses to bind it); in `active_preset` it
    failed the whole poll with a 4xx, which the device answers by DROPPING the
    batch, every poll.
    """
    return value.encode("utf-8", "replace").decode("utf-8")


# ------------------------------------------------------------------------- retention
def _env_number(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        log.warning("%s=%r is not a number; using %s", name, raw, default)
        return default
    return value if math.isfinite(value) else default


#: Events older than this many days are deleted. 0 keeps them for ever.
#:
#: Nothing used to delete an event, on a 1 GB volume that a camera writes to
#: every few seconds. A full disk is not a slow failure here: every ingest
#: then fails, and before _storage_unavailable existed it failed with a 200
#: that made devices throw their events away.
EVENT_RETENTION_DAYS = _env_number("SENSORDYME_EVENT_RETENTION_DAYS", 90.0)
#: Most events kept per device (newest kept). 0 means no cap. A runaway or
#: chatty device is bounded even inside the age window.
EVENT_MAX_PER_DEVICE = int(_env_number("SENSORDYME_EVENT_MAX_PER_DEVICE", 100_000))
#: How often retention runs, at most. It is triggered by ingest, after the
#: response, so an idle backend does no work and a busy one does it hourly.
EVENT_PRUNE_INTERVAL_S = 3600.0
#: Rows deleted per transaction, so a first run over a large backlog never
#: holds the write lock long enough to stall a device's poll.
_PRUNE_BATCH = 5000

_prune_lock = threading.Lock()
_next_prune_at = 0.0


def _prune_events(now: Optional[float] = None) -> int:
    """Apply the retention limits now. Returns how many events were deleted."""
    now = time.time() if now is None else now
    deleted = 0
    if EVENT_RETENTION_DAYS > 0:
        cutoff = now - EVENT_RETENTION_DAYS * 86400
        while True:
            with db() as conn:
                gone = conn.execute(
                    "DELETE FROM events WHERE id IN "
                    "(SELECT id FROM events WHERE received_at < ? LIMIT ?)",
                    (cutoff, _PRUNE_BATCH),
                ).rowcount
            deleted += gone
            if gone < _PRUNE_BATCH:
                break
    if EVENT_MAX_PER_DEVICE > 0:
        with db() as conn:
            device_ids = [r["id"] for r in conn.execute("SELECT id FROM devices").fetchall()]
        for device_id in device_ids:
            while True:
                with db() as conn:
                    # Cheap proof a device is under the cap, before the OFFSET
                    # walk below (which steps through up to EVENT_MAX_PER_DEVICE
                    # index entries — every device, every hour). Event ids are
                    # unique and increasing, so a device holds at most
                    # newest - oldest + 1 events: two index seeks.
                    oldest = conn.execute(
                        "SELECT id FROM events WHERE device_id = ? ORDER BY id ASC LIMIT 1",
                        (device_id,),
                    ).fetchone()
                    newest = conn.execute(
                        "SELECT id FROM events WHERE device_id = ? ORDER BY id DESC LIMIT 1",
                        (device_id,),
                    ).fetchone()
                    if oldest is None or newest["id"] - oldest["id"] + 1 <= EVENT_MAX_PER_DEVICE:
                        break
                    # The id of the oldest event to keep; everything older goes.
                    # Both lookups use idx_events_device.
                    boundary = conn.execute(
                        "SELECT id FROM events WHERE device_id = ? ORDER BY id DESC LIMIT 1 OFFSET ?",
                        (device_id, EVENT_MAX_PER_DEVICE - 1),
                    ).fetchone()
                    if boundary is None:
                        break
                    gone = conn.execute(
                        "DELETE FROM events WHERE id IN (SELECT id FROM events"
                        " WHERE device_id = ? AND id < ? ORDER BY id LIMIT ?)",
                        (device_id, boundary["id"], _PRUNE_BATCH),
                    ).rowcount
                deleted += gone
                if gone < _PRUNE_BATCH:
                    break
    if deleted:
        log.info("retention: deleted %d event(s)", deleted)
    return deleted


def _prune_events_if_due() -> None:
    """Run retention if the interval has passed and no other run is going.

    Never raises: retention failing must not show up as a failed request (it
    runs after the device has already had its answer).
    """
    global _next_prune_at  # noqa: PLW0603 - module-level retention schedule, guarded by _prune_lock
    now = time.time()
    if now < _next_prune_at or not _prune_lock.acquire(blocking=False):
        return
    try:
        _next_prune_at = now + EVENT_PRUNE_INTERVAL_S
        _prune_events(now)
    except Exception as exc:  # noqa: BLE001 - see docstring
        log.warning("retention: could not prune events (%r); will retry next interval", exc)
    finally:
        _prune_lock.release()


@app.post("/v1/ingest")
def ingest(body: IngestBody, background_tasks: BackgroundTasks, request: Request,
           device: sqlite3.Row = Depends(current_device)):
    now = time.time()
    # See _clean_text: an unencodable active_preset failed the whole poll.
    reported_preset = _clean_text(body.active_preset) if body.active_preset is not None else None
    with db() as conn:
        hardware_id = _hardware_id(body.health)
        if hardware_id and device["owner_id"]:
            # One camera, one row: a re-paired camera (a new device row for the
            # same hardware) would otherwise leave its id on the old row too, and
            # the app would show the live feed on the dead card as well.
            conn.execute(
                "UPDATE devices SET hardware_id = NULL WHERE owner_id = ? AND hardware_id = ? AND id != ?",
                (device["owner_id"], hardware_id, device["id"]),
            )
        conn.execute(
            "UPDATE devices SET last_seen = ?, active_preset = COALESCE(?, active_preset),"
            " monitoring = COALESCE(?, monitoring),"
            " hardware_id = COALESCE(?, hardware_id) WHERE id = ?",
            (
                now,
                reported_preset,
                json.dumps(_finite(body.monitoring)) if body.monitoring is not None else None,
                hardware_id,
                device["id"],
            ),
        )
        # `last_seen` is link liveness only. The health sample is judged by its
        # own stamp (M8.1): a re-sent or older sample never passes for a new one.
        # Never a reason to refuse the poll -- its events are stored below.
        _store_health(conn, device["id"], body.health, now)
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
                # Text the database cannot store (a lone surrogate) is replaced,
                # not allowed to cost the event. See _clean_text.
                e = _finite(e)  # noqa: PLW2901 - the cleaned event deliberately replaces the raw one
                event_key = e.get("event_key")
                # None is a statement — "no graded evidence" (core/confidence.py)
                # — and is stored as NULL, never coerced to a 0 that would read
                # as "certain it did not happen". Older firmware sends floats.
                confidence_raw = e.get("confidence")
                confidence = float(confidence_raw) if confidence_raw not in (None, "") else None
                duration_raw = e.get("duration")
                duration = float(duration_raw) if duration_raw not in (None, "") else None
                # float() accepts "nan", "inf" and JSON's NaN / 1e999. Stored,
                # they make every later read of this device's events a 500:
                # the response encoder refuses non-finite floats.
                if confidence is not None:
                    confidence = min(1.0, max(0.0, confidence)) if math.isfinite(confidence) else None
                model = _model_identity(e.get("model"))
                if duration is not None and not math.isfinite(duration):
                    duration = None
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO events (device_id, timestamp, type, confidence, signals,"
                    " severity, source, received_at, event_key, duration, model)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                        json.dumps(model, sort_keys=True) if model is not None else None,
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
            except (TypeError, ValueError, OverflowError) as exc:
                # Only what is wrong with THIS EVENT. A storage failure (a full
                # disk, an I/O error) is not: it used to be caught here too, so
                # the batch answered 200 -- and the device clears its queue on
                # any 200 -- while SQLite had rolled back rows already counted
                # as stored (measured: `stored: 3`, 1 row on disk) and
                # notifications went out for events that did not exist. It now
                # fails the request (503, see _storage_unavailable), nothing is
                # committed, and the device sends the same batch again.
                log.warning(
                    "ingest: skipped one malformed event from device %s (%s): key=%r type=%r",
                    device["id"], exc,
                    str(e.get("event_key"))[:64] if isinstance(e, dict) else None,
                    str(e.get("type"))[:64] if isinstance(e, dict) else None,
                )
                continue
        row = conn.execute(
            "SELECT desired_preset, ai_rules, zones, custom_monitoring_wanted, ai_rules_activation,"
            " desired_protection, desired_nodes, rejection_baseline, monitoring FROM devices WHERE id = ?",
            (device["id"],),
        ).fetchone()
        desired, ai_rules_json, zones_json = row["desired_preset"], row["ai_rules"], row["zones"]
        desired_protection = _loads_dict(row["desired_protection"]) if row["desired_protection"] else {}
        reported_block = (body.monitoring or {}).get("protection")
        reported_block = reported_block if isinstance(reported_block, dict) else None
        reported_protection = reported_block.get("state") if reported_block else None
        protection_action = desired_protection.get("action")
        if desired_protection.get("id") and reported_block is not None:
            # Converged when the device says it applied THIS request (by id).
            protection_done = reported_block.get("last_command_id") == desired_protection["id"]
        else:
            protection_done = (
                (protection_action == "stop" and reported_protection == "stopped")
                or (protection_action == "resume" and reported_protection not in (None, "stopped"))
                # A device reporting monitoring without any protection block runs
                # firmware that cannot do this: stop asking it every poll.
                or (body.monitoring is not None and reported_block is None)
            )
        # M8.3: a stop not delivered in PROTECTION_STOP_DELIVERY_S is withdrawn,
        # never delivered late; its outcome becomes `timed_out` (a resume is
        # delivered until applied: late protection is the safe direction).
        requested_at = _reported_at(desired_protection.get("requested_at"))
        stop_expired = (protection_action == "stop" and not protection_done and requested_at is not None
                        and time.time() - requested_at >= PROTECTION_STOP_DELIVERY_S)
        if protection_action and (protection_done or stop_expired):
            conn.execute(
                "UPDATE devices SET desired_protection = NULL WHERE id = ? AND desired_protection = ?",
                (device["id"], row["desired_protection"]),
            )
            protection_action = None
        _record_protection_ack(conn, device["id"], reported_block, body.monitoring)
        rules_activation = row["ai_rules_activation"] or 0
        custom_monitoring_wanted = bool(row["custom_monitoring_wanted"])
        # A device can report the same active_preset string while AI Custom
        # Monitoring runs on top of it — entering custom monitoring replaces the
        # running module without touching active_preset, so the field still names
        # whatever preset was selected before. A desired_preset equal to that
        # name is genuinely NOT fulfilled yet in that case: the device still has
        # to exit custom monitoring, which is real work "same string" hides.
        #
        # Except when the rules run BESIDE the mode (standard_plus_custom, an
        # additive role) and are still wanted (kept with the switch, M6): the
        # mode IS running, so the request is fulfilled. Counting it as
        # unfulfilled re-sent the mode every poll, and each re-send made the
        # device exit (and drop) the rules kept beside it. Rules no longer
        # wanted still have to be exited, so that case is unchanged.
        device_reports_custom_active = bool((body.monitoring or {}).get("active")) and not (
            (body.monitoring or {}).get("mode") == "standard_plus_custom" and custom_monitoring_wanted
        )
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
            and same_preset(desired, reported_preset or device["active_preset"])
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
        # The DEVICE refused the mode (M7): an unknown name, a module that would
        # not build, a role it enforces itself. It keeps running its previous
        # mode and says why in last_rejection. Re-sending would be refused again
        # on every poll, for ever, while the app spun "switching". Drop the
        # request and record the device's reason, exactly as for a role refusal
        # the backend finds itself. Only a refusal NEWER than the request counts
        # (rejection_baseline): the device never clears last_rejection, so an old
        # refusal of the same name must not cancel a fresh attempt.
        device_refusal = (
            _new_refusal(_stored_last_rejection(row), "preset", row["rejection_baseline"])
            if desired else None
        )
        if device_refusal and same_preset(device_refusal["subject"], desired):
            conn.execute(
                "UPDATE devices SET desired_preset = NULL, preset_rejection = ?"
                " WHERE id = ? AND desired_preset = ?",
                (json.dumps({
                    "preset": desired,
                    "code": device_refusal["code"] or "device_refused",
                    "reason": "; ".join(device_refusal["reasons"]) or "the device refused this mode",
                    "at": now,
                }), device["id"], desired),
            )
            desired = None
        # The row as THIS request left it: the report that carries the role may
        # be this very one, and `device` was read before it was stored.
        refusal = None
        if desired:
            current = conn.execute("SELECT * FROM devices WHERE id = ?", (device["id"],)).fetchone()
            refusal = _role_refusal(current, desired, load_presets())
        if refusal:
            # A preset the device's role forbids (stored before the device
            # reported its role, or before the upgrade) would be re-sent and
            # refused on every poll, for ever (M0.1 review). Drop it, and record
            # why: the app shows status.preset_rejection rather than a switch
            # that silently vanished.
            conn.execute(
                "UPDATE devices SET desired_preset = NULL, preset_rejection = ?"
                " WHERE id = ? AND desired_preset = ?",
                (json.dumps({"preset": desired, "code": "role_forbids_preset", "reason": refusal,
                             "at": now}), device["id"], desired),
            )
            desired = None

        # Companion nodes (M7): converge on what THIS report says the device runs.
        desired_nodes = _loads_dict(row["desired_nodes"])
        if desired_nodes:
            remaining = dict(desired_nodes)
            node_refusal = _new_refusal(_stored_last_rejection(row), "nodes", row["rejection_baseline"])
            if node_refusal and node_refusal["subject"] in remaining:
                remaining.pop(node_refusal["subject"])
                conn.execute(
                    "UPDATE devices SET nodes_rejection = ? WHERE id = ?",
                    (json.dumps({"node": node_refusal["subject"],
                                 "code": node_refusal["code"] or "device_refused",
                                 "reasons": node_refusal["reasons"] or ["the device refused this node"],
                                 "at": now}), device["id"]),
                )
            reported_nodes = _reported_nodes(_reported_perception(body.health or {}, body.monitoring or {}))
            if reported_nodes is not None:
                remaining = {k: v for k, v in remaining.items()
                             if not (isinstance(v, dict) and _node_applied(v, reported_nodes, k))}
            if remaining != desired_nodes:
                # Compare-and-swap, as for desired_preset: a node request written
                # since `row` was read is left for the next poll.
                conn.execute(
                    "UPDATE devices SET desired_nodes = ? WHERE id = ? AND desired_nodes = ?",
                    (json.dumps(remaining) if remaining else None, device["id"], row["desired_nodes"]),
                )
            desired_nodes = remaining

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
        push, sms = _notify_channels(e)
        if not (push or sms):
            # A rule set to "silent" asked not to interrupt anyone, so it must
            # not use up the account's cooldown either.
            continue
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
        background_tasks.add_task(_notify, owner, title, detail, device["name"], push, sms)

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
        desired != (reported_preset or device["active_preset"])
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
    version = _rules_version(ai_rules_json, rules_activation)
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

    # Companion nodes still to apply: the full wanted state of each, idempotent.
    if desired_nodes:
        command = dict(command or {})
        command["set_nodes"] = {
            node_id: {"enabled": bool(want.get("enabled")), "settings": want.get("settings") or {}}
            for node_id, want in sorted(desired_nodes.items()) if isinstance(want, dict)
        }

    # An explicit, confirmed stop/resume, until the device reports it (one-shot).
    if protection_action in ("stop", "resume"):
        command = dict(command or {})
        command["set_protection"] = {
            "action": protection_action, "confirm": desired_protection.get("confirm") is True,
            "id": desired_protection.get("id"),
        }

    # The count that actually landed in the database, not the size of the
    # request -- these silently diverged before (see the `[:200]` removal
    # above): a batch could report every event "stored" while most of them
    # never reached a row. A malformed-and-skipped or already-stored
    # (deduplicated retry) event legitimately isn't counted here; both are
    # accounted for elsewhere (a warning log, and the dedup this endpoint
    # relies on for retry-safety, respectively) -- this field means "how many
    # of these are now durable", which is the number a caller actually needs.
    # Retention, after the response: see _prune_events_if_due.
    background_tasks.add_task(_prune_events_if_due)
    response = {"ok": True, "stored": len(newly_inserted_events), "command": command}
    # sd-auth-v1: addressed to the hardware id reported in THIS request (what the
    # device knows it sent), bound to this request's nonce. Only when asked for,
    # so an older device's response is byte-for-byte what it was.
    envelope = _sign_command(request, command, hardware_id)
    if envelope is not None:
        response["command_auth"] = envelope
    return response
