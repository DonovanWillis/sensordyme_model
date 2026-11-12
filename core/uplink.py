"""Cloud uplink: connects the device to the SensorDyme cloud backend (cloud/).

Outbound-only, like any consumer camera device behind a NAT: a daemon thread
periodically (1) pushes queued alert events to the backend and (2) polls for
remote commands (currently: set_preset from the mobile app). Frames NEVER go
through this channel — only events and status, preserving the privacy design.

Failure-tolerant by design: if the backend is unreachable, events queue
(bounded) and the device keeps working standalone; the dashboard and local
alerts are never affected. With `spool_path` set (the runtime sets it beside
the event log) the backlog is also on disk, so an outage that outlasts a
restart or a power cut still delivers every queued event once (M8).
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

from core import command_auth as auth

#: Upper bound on the backlog. It must not exceed the backend's MAX_INGEST_EVENTS
#: (cloud/app.py): the whole backlog goes up in one POST, so a queue larger than
#: the backend accepts turns every flush after a long outage into a 422.
MAX_QUEUE = 1000

#: The ONLY answers that mean "this body is wrong, and will be wrong every time":
#: the request was read and refused for its content. A batch answered with one of
#: these is dropped so it cannot block every newer alert behind it forever.
#:
#: Deliberately an allow-list. It used to be "any 4xx except a retryable few",
#: which also dropped batches for answers that say nothing about the batch -- a
#: 404 from a wrong base_url path or a proxy, a 405 from a POST that a 301 had
#: turned into a GET -- so a misconfigured address silently discarded every
#: alert, each poll, for as long as it stayed wrong. Everything else is kept and
#: retried; a 413 for a single event (too big to ever fit) is handled separately.
_BODY_REJECTED_4XX = frozenset({400, 422})


class _RedirectedError(Exception):
    """The backend answered with a redirect, which this client refuses to follow.

    Following one re-sends X-Device-Key to wherever the Location points (requests
    strips only `Authorization` on a cross-host hop), and a 301/302 turns the POST
    into a GET that can never succeed.
    """

    def __init__(self, response) -> None:
        self.response = response
        self.location = (getattr(response, "headers", None) or {}).get("Location", "")
        super().__init__(f"HTTP {response.status_code} redirect to {self.location or '?'}")

#: Most events sent in one POST. A backlog after a long outage used to go up
#: whole, and a large one exceeds the backend's body limit (413); sending it in
#: slices also keeps each request short on a weak link. The rest stays queued
#: and goes on the following polls.
BATCH_MAX = 100

#: Longest wait between attempts while the backend is unreachable (M6).
#:
#: The worker used to retry at the fixed poll interval (2 s) for as long as an
#: outage lasted: a device with no network ran a DNS lookup + TCP connect every
#: two seconds indefinitely, and every device behind one backend outage came
#: back in lock-step. The wait now doubles per consecutive failure up to this
#: cap, with jitter, and drops back to the poll interval on the first success --
#: so once the network returns, events flow again within at most this long.
BACKOFF_CAP_S = 30.0

#: Protection command ids remembered for duplicate suppression (M6).
_COMMAND_ID_MEMORY = 64
#: Longest preset name / command id accepted from the backend.
_MAX_COMMAND_TEXT = 128
#: Signed-envelope command ids remembered (sd-auth-v1 `cid`), for duplicate refusal.
_ENVELOPE_ID_MEMORY = 256
#: The logical path a signed ingest request covers (docs/DEVICE_AUTH_CONTRACT.md):
#: never the URL's own path, so a reverse proxy's prefix does not break the tag.
_INGEST_PATH = "/v1/ingest"


def backoff_delay(failures: int, base_s: float, cap_s: float = BACKOFF_CAP_S,
                  rand=random.random) -> float:
    """Seconds to wait after `failures` consecutive failed syncs.

    0 failures -> the normal poll interval. Otherwise exponential growth from
    the poll interval, bounded by `cap_s`, with "equal jitter": half the
    ceiling is fixed (a retry is never immediate) and half random (a fleet does
    not reconnect in lock-step). A base of 0 (tests) stays 0.
    """
    base = max(0.0, float(base_s))
    if failures <= 0:
        return base
    ceiling = min(max(float(cap_s), base), base * (2 ** min(int(failures), 16)))
    return ceiling / 2 + rand() * ceiling / 2


def cleartext_remote(base_url: str) -> bool:
    """Whether `base_url` sends the device key and receives commands in cleartext
    over a network: plain http:// to anything but loopback."""
    parts = urlsplit(base_url or "")
    if parts.scheme.lower() != "http":
        return False
    return (parts.hostname or "").lower() not in ("localhost", "127.0.0.1", "::1")


def _json_safe(value):
    """`value` with every non-finite float replaced by None.

    `requests` encodes JSON with allow_nan=False and raises BEFORE sending, so a
    single NaN anywhere in a batch made that batch unsendable -- and because a
    failed batch is re-queued at the front, every later event and the heartbeat
    with it.
    """
    if hasattr(value, "item") and not isinstance(value, (dict, list, tuple, str, bytes)):
        # numpy scalars (float32 from a model, int64 from a count): requests
        # cannot encode them, and a batch it cannot encode is stuck for ever.
        try:
            value = value.item() if getattr(value, "ndim", 0) == 0 else value.tolist()
        except (AttributeError, ValueError, TypeError):
            return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _finite(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


#: More entries than any device runs is not a command, it is garbage.
_MAX_NODE_ENTRIES = 32

class CloudUplink:
    """Background sync loop between this device and the cloud backend."""

    def __init__(self, config: dict) -> None:
        self._base_url = str(config.get("base_url", "")).rstrip("/")
        self._device_key = str(config.get("device_key", ""))
        # Default kept in sync with config.yaml's cloud.poll_interval_s. This
        # is the dominant term in mode-switch latency, not HTTP/DB processing
        # (measured under 5ms end to end) -- see config.yaml's comment on that
        # key for the full reasoning.
        self._poll_interval_s = float(config.get("poll_interval_s", 2.0))
        self._queue: deque[dict] = deque(
            maxlen=min(max(int(config.get("queue_max", 500)), 1), MAX_QUEUE)
        )
        self._batch_max = BATCH_MAX
        self._lock = threading.Lock()
        self._active_preset: Optional[str] = None
        self._monitoring: Optional[dict] = None
        self._pending_preset: Optional[str] = None
        # Rule set currently enforced, reported upstream so the backend only
        # re-sends on a real change instead of on every poll.
        self._rules_version: str = ""
        self._rules_instruction: str = ""
        self._pending_rules: Optional[list[dict]] = None
        self._health: Optional[dict] = None
        self._zones_version: str = ""
        self._pending_zones: Optional[list[dict]] = None
        #: Explicit stop/resume from the backend (M0.1); applied by the main loop.
        self._pending_protection: Optional[dict] = None
        #: Companion monitoring nodes from the backend (M7, core/nodes.py):
        #: {node_id: {enabled, settings}}; applied by the main loop.
        self._pending_nodes: Optional[Any] = None
        self._stop = threading.Event()
        self._connected = False
        self._explained_failure = False
        #: Consecutive failed syncs; drives the reconnection backoff.
        self._failures = 0
        self._backoff_cap_s = float(config.get("max_backoff_s") or BACKOFF_CAP_S)
        #: Protection command ids already accepted (bounded), so a re-delivered
        #: or replayed stop/resume is not applied a second time.
        self._seen_command_ids: deque[str] = deque(maxlen=_COMMAND_ID_MEMORY)
        #: Events lost for good since start: evicted from a full backlog, or
        #: refused by the backend for their content. Reported in health.
        self.dropped_events = 0
        #: Where the backlog survives a restart or a power cut (M8); None = memory only.
        spool = config.get("spool_path")
        self._spool_path: Optional[Path] = Path(str(spool)) if spool else None
        self._spool_lines = 0
        self.restored_events = 0
        if self._spool_path is not None:
            self._load_spool()
        self._init_command_auth(config)
        import requests  # already a dependency (VLM judge backends)

        self._requests = requests
        # `verify` passed to every requests call below. True (the default) is
        # normal certificate verification against the system trust store --
        # exactly what a real, CA-signed cert (Let's Encrypt, Caddy) needs and
        # nothing more; this path needs no config at all. A path string pins
        # trust to one specific cert, for a self-signed backend (see
        # scripts/generate_dev_cert.py). False disables verification
        # entirely -- genuinely insecure, dev-only, and warned about loudly
        # exactly once rather than on every single poll.
        ca_cert_path = str(config.get("ca_cert_path", "") or "")
        insecure = bool(config.get("tls_insecure_skip_verify", False))
        if ca_cert_path:
            self._verify: bool | str = ca_cert_path
        elif insecure:
            self._verify = False
            print(
                "cloud_uplink: tls_insecure_skip_verify is ON — certificate verification is "
                "disabled and this connection can be intercepted. Development only; set "
                "ca_cert_path instead, or use a CA-signed certificate, before deploying."
            )
        else:
            self._verify = True
        if cleartext_remote(self._base_url):
            # Said once. Over plain HTTP the device key is readable by anyone on
            # the path, and the commands in the ingest response (preset, rules,
            # stop/resume) cannot be told apart from an attacker's. Kept working
            # -- LAN development backends are http:// -- but never silently.
            print(
                "cloud_uplink: base_url is plain http:// to a network host — the device key "
                "and every remote command travel unauthenticated. Use https:// outside "
                "development (docs/SECURITY_REVIEW_M6.md)."
            )

    def _init_command_auth(self, config: dict) -> None:
        """Device side of sd-auth-v1 (core/command_auth.py, docs/DEVICE_AUTH_CONTRACT.md).

        `cloud.command_auth`:
          off        -- exactly the pre-M6.5 behaviour: bearer key, unsigned commands obeyed.
          permissive -- (default) bearer key as before, plus a request nonce so a v1 backend
                        signs its commands; a signed command that fails verification is never
                        applied. A backend that does not sign is still obeyed (said once).
          required   -- the key itself never leaves the device: every request is signed, and
                        a command without a valid envelope is refused.
        An unknown mode is treated as `required` (fail closed).
        """
        mode = str(config.get("command_auth", auth.DEFAULT_MODE) or auth.DEFAULT_MODE).strip().lower()
        if mode not in auth.MODES:
            print(f"cloud_uplink: unknown cloud.command_auth {mode[:32]!r}; using 'required' (fail closed)")
            mode = "required"
        self._auth_mode = mode
        self._keyring: Optional[auth.Keyring] = None
        if mode != "off":
            try:
                self._keyring = auth.Keyring.from_config(config)
            except ValueError:
                # Too short to derive keys from. Only legacy bearer auth can work;
                # in `required` mode nothing is sent at all (see sync_once).
                print(f"cloud_uplink: the device key cannot be used for signed commands "
                      f"(cloud.command_auth: {mode}); it is shorter than 16 bytes. Re-issue it "
                      f"with `python cloud/manage.py create-device`.")
        #: Seconds added to the local clock for request timestamps only, learned
        #: from the backend's clock_skew answer (a Pi has no RTC). Never applied
        #: to the system clock, and never used to judge a command's freshness.
        self._clock_offset_s = 0.0
        self._seen_envelope_ids: deque[str] = deque(maxlen=_ENVELOPE_ID_MEMORY)
        self._unsigned_announced = False
        self._last_rejection: Optional[str] = None
        #: Commands refused by sd-auth-v1 verification since start (health / tests).
        self.command_auth_rejections = 0

    @property
    def command_auth_mode(self) -> str:
        return self._auth_mode

    def _refuse_command(self, reason: str) -> None:
        self.command_auth_rejections += 1
        if reason != self._last_rejection:
            # Reason code and nothing else: no key, tag or command body in the log.
            self._last_rejection = reason
            print(f"cloud_uplink: refused a remote command ({reason}); it was not applied")

    def _authenticated_command(self, data: dict, command: dict, nonce: Optional[str], health) -> dict:
        """`command` if it may be applied under the configured mode, else {}."""
        if not command or self._auth_mode == "off":
            return command
        envelope = data.get("command_auth")
        if envelope is None:
            if self._auth_mode == "required":
                self._refuse_command("missing_signature")
                return {}
            if not self._unsigned_announced:
                self._unsigned_announced = True
                print("cloud_uplink: the backend does not sign its commands (sd-auth-v1); they are "
                      "applied as before. Set cloud.command_auth: required once it does.")
            return command
        if self._keyring is None or nonce is None:
            self._refuse_command("no_usable_key")
            return {}
        devices = {f"kid:{k.kid}" for k in (self._keyring.current, self._keyring.previous) if k is not None}
        hardware_id = health.get("device_id") if isinstance(health, dict) else None
        if isinstance(hardware_id, str):
            devices.add(hardware_id)
        try:
            verified = auth.verify_command(self._keyring, command, envelope, devices=devices,
                                           request_nonce=nonce)
        except auth.AuthError as exc:
            self._refuse_command(exc.reason)
            return {}
        with self._lock:
            duplicate = verified["cid"] in self._seen_envelope_ids
            if not duplicate:
                self._seen_envelope_ids.append(verified["cid"])
        if duplicate:
            self._refuse_command("duplicate_command")
            return {}
        self._last_rejection = None
        return command

    def _learn_clock_offset(self, exc) -> None:
        """Adopt the backend's clock for request timestamps after a clock_skew refusal."""
        try:
            body = exc.response.json()
            server_time = float(body.get("server_time"))
        except Exception:  # noqa: BLE001 - any unreadable answer: keep the current offset
            return
        if body.get("code") == "clock_skew" and math.isfinite(server_time):
            self._clock_offset_s = server_time - time.time()
            print(f"cloud_uplink: device clock differs from the backend's by "
                  f"{self._clock_offset_s:+.0f} s; signing with the backend's time")

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True, name="cloud-uplink").start()

    def close(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------ main-loop side
    def push_event(
        self,
        event,
        severity: str,
        source: str = "module",
        model: Optional[dict] = None,
        notify: Optional[str] = None,
    ) -> None:
        """Queue one alert event for the next sync cycle.

        `source` distinguishes a preset's own detection from something the user
        explicitly configured. It exists because severity alone cannot answer
        "should this interrupt someone": a custom rule defaults to `warning`,
        which maps to `soft`, which the notification gate drops — so the one
        event a person asked to be told about was the one that never arrived,
        while the preset's own alerts kept coming through.

        `event_key` is generated once, here, and travels with this dict through
        every retry. A POST can succeed on the server (events inserted) while
        its response is lost to the device — a dropped connection, a timeout —
        which reads as failure and re-queues the same batch for the next poll.
        Without a stable key that arrives twice, the retry becomes a second row
        for the same alert: a duplicate in Recent Activity, and a second
        notification for something that already fired once. The server
        deduplicates on (device_id, event_key) — see cloud/app.py's ingest
        INSERT — so a resend is a no-op instead of a duplicate.

        `notify` is a rule's own delivery choice (e.g. "push", "sms", "silent"),
        sent only when given so the backend can honour it; events without one
        carry no `notify` key at all, exactly as before.
        """
        with self._lock:
            item = {
                "event_key": uuid.uuid4().hex,
                # Which exact model produced this: module + detector name,
                # sha256 and backend (DeviceRuntime.model_identity). Taken
                # NOW, at queue time, so an event queued before a module
                # switch keeps the identity of the model that produced it.
                "model": _json_safe(dict(model)) if model else None,
                "timestamp": event.timestamp.isoformat(),
                "type": event.type,
                # None is a statement ("no graded evidence"), not a missing 0:
                # see core/confidence.py. Sent as null, stored as NULL.
                "confidence": (
                    None if event.confidence is None
                    else round(_finite(event.confidence), 3)
                ),
                "signals": _json_safe(event.contributing_signals),
                # Seconds the condition was sustained before this fired --
                # already computed by every module (Event.duration), but
                # previously dropped here rather than sent, so a genuinely
                # informative number never reached a notification. Used as
                # a fallback when `signals` is empty (e.g. fall_detection's
                # "fall" event carries no signals text today) rather than
                # the bare, contextless "alert" that produced.
                "duration": round(_finite(event.duration), 3),
                "severity": severity,
                "source": source,
            }
            if notify is not None:
                item["notify"] = str(notify)
            if self._queue.maxlen is not None and len(self._queue) >= self._queue.maxlen:
                self.dropped_events += 1  # the deque evicts the oldest
            self._queue.append(item)
            self._spool_append(item)

    def set_monitoring(self, described: Optional[dict]) -> None:
        """What custom configuration, if any, is currently in charge.

        Reported because the device knowing is not enough: the app was showing
        the base mode as selected while a custom configuration was the thing
        actually running, so someone who asked it to count people was told it
        was watching for driver distraction.
        """
        with self._lock:
            self._monitoring = described

    def set_active_preset(self, name: Optional[str]) -> None:
        with self._lock:
            self._active_preset = name

    def refresh_health_monitoring(self, monitoring: dict, monitoring_active: bool, protection: dict) -> None:
        """Bring the monitoring part of the last health sample up to date now.

        The full health sample is taken every few seconds (it runs nmcli and
        vcgencmd); a stop, a resume or a monitoring failure must not wait for the
        next one, or the app shows the old state beside a newer pushed one. Only
        the three monitoring keys change; the rest of the sample is kept.
        """
        with self._lock:
            if isinstance(self._health, dict):
                self._health = {**self._health, "monitoring": monitoring,
                                "monitoring_active": monitoring_active, "protection": protection}

    def set_health(self, report: Optional[dict]) -> None:
        """Latest device health, sent with the next sync.

        Held as a snapshot rather than collected inside the worker so the health
        read happens on the runtime's own thread, where the frame bus, capture
        service and Wi-Fi manager actually live.
        """
        with self._lock:
            self._health = report

    def pop_pending_preset(self) -> Optional[str]:
        """Remote set_preset command awaiting the main loop, if any."""
        with self._lock:
            pending, self._pending_preset = self._pending_preset, None
            return pending

    def pop_pending_protection(self) -> Optional[dict]:
        """A remote stop/resume awaiting the main loop, if any."""
        with self._lock:
            pending, self._pending_protection = self._pending_protection, None
            return pending

    def pop_pending_nodes(self) -> Optional[Any]:
        """A set_nodes command awaiting the main loop: {node_id: {enabled, settings}}.

        Acknowledged by the device reporting the node in monitoring.perception
        (core/perception.py) or refusing it as last_rejection {kind: "nodes"},
        so it is not acked here.
        """
        with self._lock:
            pending, self._pending_nodes = self._pending_nodes, None
            return pending

    def pop_pending_zones(self) -> Optional[list[dict]]:
        """Zones drawn in the app, awaiting the main loop.

        None means nothing new; `[]` means the user deleted their zones and
        every area rule must stop matching rather than keep the old shapes.
        """
        with self._lock:
            pending, self._pending_zones = self._pending_zones, None
            return pending

    def rules_version(self) -> str:
        """The version the backend last sent, so the stored copy can be tagged."""
        with self._lock:
            return self._rules_version

    def pending_instruction(self) -> str:
        """The sentence a person typed, if the backend sent one along.

        Kept beside the rules because a rule list is what the machine executes
        and the sentence is what someone recognises when the app asks them to
        confirm the device is watching the right thing.
        """
        with self._lock:
            return self._rules_instruction

    def pop_pending_rules(self) -> Optional[list[dict]]:
        """AI-configured rules awaiting the main loop, if a new set arrived.

        None means "nothing new" — distinct from `[]`, which means the user
        deliberately cleared their rules and the device must stop enforcing them.
        """
        with self._lock:
            pending, self._pending_rules = self._pending_rules, None
            return pending

    @property
    def connected(self) -> bool:
        return self._connected

    def _looks_like_bare_ip(self) -> bool:
        """Whether base_url points at a numeric address rather than a name.

        A name resolves wherever the host goes; an IP is only true on the
        network it was written down on.
        """
        import re as _re

        return bool(_re.search(r"//(\d{1,3}\.){3}\d{1,3}(:|/|$)", self._base_url))

    # ------------------------------------------------------------------ worker
    def _run(self) -> None:
        while not self._stop.is_set():
            more_waiting = self.sync_once()
            self._stop.wait(0 if more_waiting else self.next_delay())

    def sync_once(self) -> bool:
        """One sync cycle: push queued events + health, apply any command answered.

        The body of the worker loop, unchanged, as a method so a caller with its
        own clock can drive it one cycle at a time (sim/, the virtual device's
        deterministic end-to-end tests). Never raises: a failure is handled
        exactly as it always was (batch kept or dropped, offline reported).
        Returns True when a full batch went up and more is waiting, i.e. the
        next cycle should run now rather than after poll_interval_s.
        """
        # Bound OUTSIDE the try. The except handler below re-queues `batch`, so if
        # anything raised before it was assigned the handler itself would raise
        # NameError -- and since this loop has no outer try, that kills the uplink
        # thread outright. The device would then keep detecting and alerting
        # locally while silently never reaching the backend again, which is the
        # worst failure shape available: no events, no error, no reconnect.
        batch: list = []
        more_waiting = False
        try:
            with self._lock:
                take = min(len(self._queue), self._batch_max)
                batch = [self._queue.popleft() for _ in range(take)]
                active = self._active_preset
                rules_version = self._rules_version
                zones_version = self._zones_version
                health = self._health
                monitoring = self._monitoring
            # 1) push events (an empty batch doubles as the online heartbeat)
            payload = _json_safe({
                "events": batch,
                "active_preset": active,
                "rules_version": rules_version,
                "zones_version": zones_version,
                "health": health,
                "monitoring": monitoring,
            })
            nonce = auth.new_nonce() if self._keyring is not None else None
            if self._auth_mode == "required":
                if self._keyring is None or nonce is None:
                    raise RuntimeError("cloud.command_auth is required but the device key is unusable")
                # The exact bytes are signed, so they are serialised here, not by requests.
                body = json.dumps(payload, allow_nan=False).encode()
                signed = auth.sign_request(self._keyring.current, "POST", _INGEST_PATH, body, nonce=nonce,
                                           issued_at=int(time.time() + self._clock_offset_s))
                transport: dict = {"data": body,
                                   "headers": {"Content-Type": "application/json", "X-SD-Auth": signed}}
            else:
                headers = {"X-Device-Key": self._device_key}
                if nonce is not None:
                    headers["X-SD-Nonce"] = nonce
                transport = {"json": payload, "headers": headers}
            resp = self._requests.post(
                f"{self._base_url}/v1/ingest",
                **transport,
                timeout=10,
                verify=self._verify,
                # Never follow: see _RedirectedError. A redirect is a base_url to
                # fix, reported below, with every event kept meanwhile.
                allow_redirects=False,
            )
            if 300 <= int(getattr(resp, "status_code", 200) or 200) < 400:
                raise _RedirectedError(resp)
            resp.raise_for_status()
            # 2) the ingest response carries any pending command
            data = resp.json()
            command = data.get("command") or {}
            if not isinstance(command, dict):
                command = {}
            command = self._authenticated_command(data, command, nonce, health)
            preset = command.get("set_preset")
            if isinstance(preset, str) and 0 < len(preset) <= _MAX_COMMAND_TEXT:
                with self._lock:
                    self._pending_preset = preset
            elif preset:
                print("cloud_uplink: ignored a malformed set_preset command")
            if "set_rules" in command:
                rules = command.get("set_rules")
                with self._lock:
                    self._pending_rules = rules if isinstance(rules, list) else []
                    # Acknowledge immediately: the main loop applies these on
                    # its next tick, and re-sending every poll until then
                    # would restart in-flight rule tracking each time.
                    self._rules_version = str(command.get("rules_version") or "")
                    self._rules_instruction = str(command.get("rules_instruction") or "")
            protection = command.get("set_protection")
            if isinstance(protection, dict) and protection.get("action") in ("stop", "resume"):
                self._accept_protection(protection)
            if "set_nodes" in command:
                nodes = command.get("set_nodes")
                if isinstance(nodes, (dict, list)) and len(nodes) <= _MAX_NODE_ENTRIES:
                    with self._lock:
                        self._pending_nodes = nodes
                else:
                    print("cloud_uplink: ignored a malformed set_nodes command")
            if "set_zones" in command:
                zones = command.get("set_zones")
                with self._lock:
                    self._pending_zones = zones if isinstance(zones, list) else []
                    self._zones_version = str(command.get("zones_version") or "")
            # A full batch that went through means a backlog: send the next
            # slice now rather than one per poll interval.
            more_waiting = bool(batch) and len(batch) >= self._batch_max
            if batch:
                # Delivered: they leave the on-disk backlog too. A power cut
                # before this line replays them after restart, and the backend
                # stores each event_key once.
                with self._lock:
                    self._spool_rewrite()
            self._failures = 0
            if not self._connected:
                self._connected = True
                self._explained_failure = False
                print(f"cloud_uplink: connected to {self._base_url}")
        except Exception as exc:  # noqa: BLE001 - the device must keep working offline
            if self._connected:
                print(f"cloud_uplink: connection lost ({exc}); device continues standalone")
            elif not self._explained_failure:
                # Said once, not every five seconds. A bare IP in base_url is
                # the commonest cause of a device that looks broken and is
                # not: the address was correct on the network where it was
                # configured, and means nothing on any other one.
                self._explained_failure = True
                # An HTTP error is not a reachability problem. The backend
                # answered — it refused. Reporting "cannot reach" for a
                # rejected device key sends someone to debug a network that
                # is working perfectly.
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if isinstance(exc, _RedirectedError):
                    print(
                        f"cloud_uplink: {self._base_url} answered HTTP {status}, redirecting "
                        f"to {exc.location or '(no Location)'}. Redirects are not followed "
                        f"(the device key would go with them): set cloud.base_url to the "
                        f"final address. Events are kept until then."
                    )
                elif status in (401, 403):
                    print(
                        f"cloud_uplink: {self._base_url} rejected this device's key "
                        f"(HTTP {status}). The backend is reachable; the key is wrong or "
                        f"unknown to it — re-issue with `python cloud/manage.py`."
                    )
                elif status is not None:
                    print(f"cloud_uplink: {self._base_url} returned HTTP {status}")
                else:
                    print(f"cloud_uplink: cannot reach {self._base_url} ({exc})")
                if status is None and self._looks_like_bare_ip():
                    print(
                        "cloud_uplink: base_url is a fixed IP address. If this device or "
                        "the backend has moved to a different network, that address now "
                        "points at nothing. Use the backend host's mDNS name instead — "
                        "e.g. http://your-macbook.local:8787 — which follows it between "
                        "networks. See docs/NETWORKING.md."
                    )
            self._connected = False
            self._failures += 1
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 401 and self._auth_mode == "required":
                self._learn_clock_offset(exc)
            if status == 413 and len(batch) > 1:
                # Too big, not wrong: send less next time rather than lose it.
                self._batch_max = max(1, len(batch) // 2)
                print(
                    f"cloud_uplink: batch of {len(batch)} too large for the backend; "
                    f"retrying in batches of {self._batch_max}"
                )
            elif batch and (status in _BODY_REJECTED_4XX or (status == 413 and len(batch) == 1)):
                # The backend read this batch and refused it; it will refuse
                # it again. Re-queuing it at the front is what used to block
                # every newer alert behind it forever.
                print(
                    f"cloud_uplink: backend refused {len(batch)} event(s) with HTTP "
                    f"{status}; dropping them so newer events are not blocked"
                )
                self.dropped_events += len(batch)
                batch = []
                with self._lock:
                    self._spool_rewrite()
            with self._lock:
                # Put the batch back in front of anything recorded since,
                # but never at the cost of those newer events.
                #
                # `appendleft` on a full deque evicts from the RIGHT — the
                # newest end. So re-queuing a failed batch used to discard
                # the most recent alerts in favour of the ones that had just
                # failed to send, which is backwards for a safety product:
                # during a long outage the device threw away what was
                # happening now to preserve what happened an hour ago. The
                # comment here claimed it dropped the oldest.
                capacity = self._queue.maxlen
                space = len(batch) if capacity is None else capacity - len(self._queue)
                if space > 0:
                    # Keep the newest of the failed batch when it cannot all fit.
                    for item in reversed(batch[-space:]):
                        self._queue.appendleft(item)
                dropped = len(batch) - max(space, 0)
                if dropped > 0:
                    self.dropped_events += dropped
                    self._spool_rewrite()
                    print(
                        f"cloud_uplink: backlog full — {dropped} older event(s) dropped; "
                        f"newer ones kept"
                    )
        return more_waiting

    # ------------------------------------------------------------------ backlog spool (M8)
    def backlog(self) -> dict:
        """The undelivered-event backlog, for health: depth, bound, losses, persistence."""
        with self._lock:
            return {"queued": len(self._queue), "capacity": self._queue.maxlen,
                    "dropped": self.dropped_events, "restored": self.restored_events,
                    "persistent": self._spool_path is not None}

    def _load_spool(self) -> None:
        """Re-queue events a previous run recorded but never delivered.

        The spool is JSON lines, appended as events are queued: a power cut can
        leave a torn last line, which is skipped (that one event was never fully
        recorded); anything else unreadable is skipped and counted, never fatal.
        """
        assert self._spool_path is not None
        try:
            raw = self._spool_path.read_bytes()
        except FileNotFoundError:
            return
        except OSError as exc:
            print(f"cloud_uplink: undelivered-event spool unreadable ({exc}); starting empty")
            return
        items: dict[str, dict] = {}
        damaged = 0
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                damaged += 1
                continue
            if not isinstance(item, dict) or not isinstance(item.get("event_key"), str) \
                    or not isinstance(item.get("type"), str):
                damaged += 1
                continue
            items.pop(item["event_key"], None)  # one entry per key, latest position
            items[item["event_key"]] = item
        restored = list(items.values())
        capacity = self._queue.maxlen
        if capacity is not None and len(restored) > capacity:
            self.dropped_events += len(restored) - capacity
            restored = restored[-capacity:]  # the newest, as the live backlog keeps
        self._queue.extend(restored)
        self.restored_events = len(restored)
        if restored or damaged:
            print(f"cloud_uplink: {len(restored)} undelivered event(s) restored from the last run"
                  + (f"; {damaged} damaged spool line(s) skipped" if damaged else ""))
        self._spool_rewrite()

    def _spool_append(self, item: dict) -> None:
        """Record one queued event durably (caller holds the lock)."""
        if self._spool_path is None:
            return
        capacity = self._queue.maxlen or MAX_QUEUE
        if self._spool_lines >= 2 * capacity:
            self._spool_rewrite()  # compaction: evicted events leave the file too
            return
        try:
            line = json.dumps(_json_safe(item), allow_nan=False, default=str)
            self._spool_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._spool_path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._spool_lines += 1
        except (OSError, TypeError, ValueError) as exc:
            # Losing durability is bad; losing the event or the alert is worse.
            print(f"cloud_uplink: could not record a queued event on disk ({exc}); kept in memory")

    def _spool_rewrite(self) -> None:
        """Make the spool exactly the in-memory backlog (caller holds the lock). Atomic."""
        if self._spool_path is None:
            return
        try:
            if not self._queue:
                with contextlib.suppress(FileNotFoundError):
                    self._spool_path.unlink()
                self._spool_lines = 0
                return
            from core.config_lkg import atomic_write_bytes

            body = "".join(json.dumps(_json_safe(item), allow_nan=False, default=str) + "\n"
                           for item in self._queue)
            if atomic_write_bytes(self._spool_path, body.encode("utf-8"), mode=0o600):
                self._spool_lines = len(self._queue)
        except (OSError, TypeError, ValueError) as exc:
            print(f"cloud_uplink: could not rewrite the undelivered-event spool ({exc})")

    def next_delay(self) -> float:
        """How long the worker waits before its next sync (see backoff_delay)."""
        return backoff_delay(self._failures, self._poll_interval_s, self._backoff_cap_s)

    def _accept_protection(self, protection: dict) -> None:
        """Queue a stop/resume for the main loop, at most once per command id.

        The backend re-sends a protection command until the device reports its
        id back, and a lost response -- or a replayed one -- delivers the same
        id again. Applying it twice is not harmless: a stop the owner has since
        resumed locally would silently stop monitoring again. A command without
        an id (an older backend) is accepted as before.
        """
        raw_id = protection.get("id")
        command_id = None if raw_id in (None, "") else str(raw_id)[:_MAX_COMMAND_TEXT]
        with self._lock:
            if command_id is not None:
                if command_id in self._seen_command_ids:
                    return
                self._seen_command_ids.append(command_id)
            self._pending_protection = {
                "action": protection["action"], "confirm": protection.get("confirm") is True,
                "id": command_id,
            }
