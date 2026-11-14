#!/usr/bin/env python3
"""Read-only: fetch the LIVE status and check it against the state-model invariants.

    V=~/hwval/sensordyme_model/scripts/hw_validation
    # the device's own sample (what the uplink sends; GET /api/health, token header)
    SENSORDYME_TOKEN=... $PY $V/check_live_status.py --device http://localhost:8899
    # what the app is told (GET /v1/devices, the owner's bearer token)
    SENSORDYME_BEARER=... $PY $V/check_live_status.py --backend https://<backend> [--device-id N]
    # both, sampled 6 times 5 s apart: transitions and device/backend agreement too
    $PY $V/check_live_status.py --device http://localhost:8899 --backend https://<backend> --samples 6
    # the video evidence: /stream part headers + /api/stream/stats (docs/VIDEO_TRANSPORT.md)
    $PY $V/check_live_status.py --device http://localhost:8899 --stream-seconds 6
    # re-check recorded evidence (no network)
    $PY $V/check_live_status.py --health-file e2e_health.json --devices-file devices.json

Invariants: sim/invariants.py `report_violations` (device sample) and
`status_violations` (backend row); user state and transitions:
core/state_model.py (`derive`, `transition_problem`). docs/STATE_MODEL.md.

Never writes to the device or the backend: GET only. Secrets come from
environment variables (`--token-env`, `--bearer-env`), never from argv, and
are never written to the report. The report keeps selected fields only (no
SSID, no full health dump).

Exit codes (as hwv_stage.py): 0 PASS, 1 FAIL (a contradiction, a broken
stream), 2 BLOCKED (unreachable, no token, nothing to check), 3 SIMULATED
(the data says it came from the virtual device: NOT hardware evidence).
"""

from __future__ import annotations

import argparse
import http.client
import itertools
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import state_model  # noqa: E402
from sim.invariants import report_violations, status_violations  # noqa: E402

PASS, FAIL, BLOCKED, SIMULATED = 0, 1, 2, 3
LABELS = {PASS: "PASS", FAIL: "FAIL", BLOCKED: "BLOCKED", SIMULATED: "SIMULATED"}
#: The app says LIVE only for a frame whose X-Frame-Age-Ms <= 2000 (docs/VIDEO_TRANSPORT.md).
LIVE_FRAME_AGE_MS = 2000
#: Parts in the first seconds of a stream may be the camera fallback until the
#: runtime starts drawing for its first viewer (ANNOTATION_GAP_S 1.5 s).
STREAM_WARMUP_S = 2.0
#: sim/identity.py: the virtual device's hardware id prefix ("nothing can mistake it for a Pi").
SIM_ID_PREFIX = "SD-SIM-"


class UnavailableError(Exception):
    """A source could not be read (unreachable, refused, not JSON): BLOCKED, not FAIL."""


# ------------------------------------------------------------------ fetching (GET only)
def _get_json(url: str, headers: dict[str, str], timeout: float = 8.0) -> Any:
    request = urllib.request.Request(url, headers=headers, method="GET")  # noqa: S310 - operator-given URL
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise UnavailableError(f"GET {_redact(url)} -> HTTP {exc.code}") from None
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise UnavailableError(f"GET {_redact(url)} -> {exc}") from None


def _redact(url: str) -> str:
    """A URL without its query string (a `?token=` must never reach a report)."""
    return url.split("?", 1)[0]


def http_status(url: str, headers: Optional[dict[str, str]] = None, timeout: float = 5.0) -> Optional[int]:
    """The status code of a GET (the body is not read beyond the headers), or None if unreachable."""
    parts = urllib.parse.urlsplit(url)
    conn_cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(parts.hostname or "localhost", parts.port, timeout=timeout)
    try:
        conn.request("GET", parts.path + (f"?{parts.query}" if parts.query else ""), headers=headers or {})
        return conn.getresponse().status
    except OSError:
        return None
    finally:
        conn.close()


def fetch_device_health(base: str, token: str) -> dict:
    return _get_json(base.rstrip("/") + "/api/health", {"X-Device-Token": token})


def fetch_stream_stats(base: str, token: str) -> dict:
    return _get_json(base.rstrip("/") + "/api/stream/stats", {"X-Device-Token": token})


def fetch_backend_devices(base: str, bearer: str) -> dict:
    return _get_json(base.rstrip("/") + "/v1/devices", {"Authorization": f"Bearer {bearer}"})


def read_stream_parts(base: str, token: str, seconds: float, max_parts: int = 400) -> dict:
    """Open GET /stream (token header) for `seconds` and return every part's headers.

    Opening the stream makes the device annotate and encode for one more
    viewer for that long; nothing else changes. JPEG bodies are read and
    discarded (only their length and SOI marker are kept).
    """
    parts = urllib.parse.urlsplit(base)
    conn_cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    # A silent stream (no part at all) ends the read at the socket timeout.
    conn = conn_cls(parts.hostname or "localhost", parts.port, timeout=seconds + 2.0)
    out: dict[str, Any] = {"status": None, "headers": {}, "parts": []}
    try:
        conn.request("GET", "/stream", headers={"X-Device-Token": token})
        response = conn.getresponse()
        out["status"] = response.status
        out["headers"] = {k: v for k, v in response.getheaders() if k.lower().startswith(("x-", "content-type"))}
        if response.status != 200:
            return out
        start = time.monotonic()
        while time.monotonic() - start < seconds and len(out["parts"]) < max_parts:
            line = response.readline()
            if not line:
                break
            if not line.strip().startswith(b"--frame"):
                continue
            headers: dict[str, str] = {}
            while True:
                header = response.readline()
                if not header or not header.strip():
                    break
                key, _, value = header.decode("latin-1").partition(":")
                headers[key.strip().lower()] = value.strip()  # header names are case-insensitive
            length = int(headers.get("content-length", "0") or 0)
            body = response.read(length) if length else b""
            out["parts"].append({"t": round(time.monotonic() - start, 3), "headers": headers,
                                 "bytes": len(body), "jpeg": body[:2] == b"\xff\xd8"})
    except OSError as exc:
        out["error"] = str(exc)
    finally:
        conn.close()
    return out


# ------------------------------------------------------------------ evaluation (pure; tested on fixtures)
def is_simulated(device: dict) -> bool:
    perception = (device.get("monitoring") or {}).get("perception") or {}
    return (bool(device.get("simulated")) or bool(perception.get("simulated"))
            or str(device.get("device_id") or "").startswith(SIM_ID_PREFIX))


def device_summary(device: dict) -> dict:
    """The fields a session log needs, nothing more (no network/SSID block)."""
    monitoring = device.get("monitoring") or {}
    protection = device.get("protection") or {}
    accelerator = device.get("accelerator") or {}
    stamp = device.get("stamp") or {}
    return {
        "device_id": device.get("device_id"),
        "monitoring_state": monitoring.get("state"),
        "monitoring_active": device.get("monitoring_active"),
        "protection": protection.get("state"),
        "reasons": monitoring.get("reasons"),
        "available": monitoring.get("available"),
        "impaired": monitoring.get("impaired"),
        "paused": monitoring.get("paused"),
        "accelerator": {"state": accelerator.get("state"), "detail": accelerator.get("detail"),
                        "metrics": accelerator.get("metrics")},
        "versions": monitoring.get("versions"),
        "model": monitoring.get("model"),
        "perception_schema": (monitoring.get("perception") or {}).get("schema"),
        "config_source": (device.get("config") or {}).get("source"),
        "saved_state_faults": (device.get("saved_state") or {}).get("faults"),
        "stamp": {k: stamp.get(k) for k in ("schema", "boot_id", "generation", "age_s")},
        "simulated": is_simulated(device),
    }


def check_device(health: dict) -> dict:
    """GET /api/health (or its `device` block) against report_violations."""
    device = health.get("device") if isinstance(health.get("device"), dict) else health
    violations = report_violations(device)
    return {"source": "device", "summary": device_summary(device),
            "violations": [{"invariant": i, "detail": d} for i, d in violations]}


def pick_row(devices: dict, device_id: Optional[int] = None, hardware_id: Optional[str] = None) -> dict:
    rows = devices.get("devices") if isinstance(devices, dict) else None
    if not isinstance(rows, list) or not rows:
        raise UnavailableError("GET /v1/devices returned no devices for this account")
    if device_id is not None:
        match = [r for r in rows if r.get("id") == device_id]
    elif hardware_id is not None:
        match = [r for r in rows if r.get("hardware_id") == hardware_id]
    else:
        match = rows if len(rows) == 1 else []
    if len(match) != 1:
        ids = [(r.get("id"), r.get("hardware_id")) for r in rows]
        raise UnavailableError(f"cannot pick one device from {ids}: pass --device-id")
    return match[0]


def check_row(row: dict, now: Optional[float] = None) -> dict:
    """One GET /v1/devices row against status_violations, plus its user-facing state."""
    status = row.get("status") if isinstance(row.get("status"), dict) else {}
    derived = state_model.derive(row, now if now is not None else time.time())
    health = row.get("health") if isinstance(row.get("health"), dict) else {}
    freshness = status.get("health_freshness") if isinstance(status.get("health_freshness"), dict) else {}
    command = status.get("protection_command") if isinstance(status.get("protection_command"), dict) else None
    return {
        "source": "backend",
        "summary": {
            "id": row.get("id"), "hardware_id": row.get("hardware_id"), "online": row.get("online"),
            "user_state": derived.state, "protecting": derived.protecting,
            "monitoring_state": status.get("monitoring_state"), "protection": status.get("protection"),
            "monitoring_active": status.get("monitoring_active"), "stale": status.get("stale"),
            "link_stale": status.get("link_stale"), "health_stale": status.get("health_stale"),
            "freshness": {k: freshness.get(k) for k in ("state", "age_s", "threshold_s", "boot_id", "generation")},
            "protection_command": None if command is None else {
                k: command.get(k) for k in ("action", "outcome", "id", "reported_state", "applied_at")},
            "simulated": bool(health.get("simulated"))
            or str(row.get("hardware_id") or "").startswith(SIM_ID_PREFIX),
        },
        "violations": [{"invariant": i, "detail": d} for i, d in status_violations(row)],
    }


def transition_findings(rows: list[dict]) -> list[dict]:
    """Consecutive user-facing states (checked rows, oldest first) against state_model.ALLOWED + guards."""
    findings = []
    for prev, cur in itertools.pairwise(rows):
        a, b = prev["summary"], cur["summary"]
        if a["user_state"] == b["user_state"]:
            continue
        cmd = b.get("protection_command") or {}
        fresh_cmd = cmd.get("id") != (a.get("protection_command") or {}).get("id") or \
            cmd.get("outcome") != (a.get("protection_command") or {}).get("outcome")
        confirmed = cmd.get("outcome") == "confirmed" and fresh_cmd
        boot_a, boot_b = a["freshness"].get("boot_id"), b["freshness"].get("boot_id")
        problem = state_model.transition_problem(
            a["user_state"], b["user_state"], stop_applied=confirmed and cmd.get("action") == "stop",
            resume_applied=confirmed and cmd.get("action") == "resume",
            restarted=bool(boot_a and boot_b and boot_a != boot_b))
        if problem:
            findings.append({"invariant": "state_transition", "detail": problem})
    return findings


def agreement(device_check: dict, row_check: dict) -> list[str]:
    """Where the device and the backend disagree right now (the uplink lags up to ~5 s)."""
    d, r = device_check["summary"], row_check["summary"]
    out = []
    if r.get("hardware_id") and d.get("device_id") and r["hardware_id"] != d["device_id"]:
        out.append(f"backend row is {r['hardware_id']}, the device is {d['device_id']}")
    if r.get("stale") is False and d.get("monitoring_state") != r.get("monitoring_state"):
        out.append(f"device says {d.get('monitoring_state')}, backend says {r.get('monitoring_state')}")
    return out


def check_stream(probe: dict, *, monitoring_running: Optional[bool], camera_while_stopped: bool = False) -> dict:
    """Judge /stream part headers (core/dashboard.py `_part_header`).

    monitoring_running True: frames must flow, seq strictly increasing, age <= 2000 ms,
    and (after the warm-up) every part `annotated`. False (stopped): Mission 8.4
    Decision C -- no camera picture leaves the device while stopped unless the
    owner enabled camera-only viewing (`camera_while_stopped`). None: state unknown.
    """
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: Any) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    status = probe.get("status")
    parts = [p for p in probe.get("parts", []) if p["headers"].get("content-type") == "image/jpeg"]
    check("GET /stream with the token header answers 200", status == 200, status)
    seqs = [int(p["headers"].get("x-frame-seq", -1)) for p in parts]
    ages = [int(p["headers"].get("x-frame-age-ms", -1)) for p in parts]
    sources = sorted({p["headers"].get("x-frame-source", "?") for p in parts})
    stream_ids = sorted({p["headers"].get("x-stream-id", "?") for p in parts})
    values = {"parts": len(parts), "seq_first_last": (seqs[0], seqs[-1]) if seqs else None,
              "age_ms_max": max(ages) if ages else None,
              "age_ms_p50": sorted(ages)[len(ages) // 2] if ages else None,
              "sources": sources, "stream_ids": stream_ids,
              "heartbeats": len(probe.get("parts", [])) - len(parts),
              "response_headers": probe.get("headers")}
    if status == 200 and parts:
        check("every part carries X-Frame-Seq / X-Frame-Age-Ms / X-Frame-Source (M8.3 headers)",
              all(s >= 0 for s in seqs) and all(a >= 0 for a in ages) and "?" not in sources, sources)
        check("X-Frame-Seq strictly increasing", all(b > a for a, b in itertools.pairwise(seqs)),
              values["seq_first_last"])
        check("one X-Stream-Id for the whole connection", len(stream_ids) == 1, stream_ids)
        check("every part is a JPEG (SOI marker)", all(p["jpeg"] for p in parts), len(parts))
    if monitoring_running is True:
        check("frames flow (>= 2 parts)", len(parts) >= 2, len(parts))
        if ages:
            check(f"X-Frame-Age-Ms <= {LIVE_FRAME_AGE_MS} (the app's LIVE criterion)",
                  max(ages) <= LIVE_FRAME_AGE_MS, values["age_ms_max"])
        late = sorted({p["headers"].get("x-frame-source") for p in parts if p["t"] >= STREAM_WARMUP_S})
        if late:
            check("after the warm-up every part is `annotated` (analysed) while monitoring runs",
                  late == ["annotated"], late)
    elif monitoring_running is False:
        # M8.4 Decision C (core/dashboard.py, core/camera_privacy.py): while the
        # owner has stopped monitoring, a REMOTE viewer gets no picture unless the
        # owner enabled camera-only mode (frames labelled `camera_only`); a viewer
        # on the device itself (loopback) may get `camera_local` frames for
        # maintenance. `camera` = the not-drawing fallback, never allowed remotely.
        def source(part: dict) -> Optional[str]:
            return part["headers"].get("x-frame-source")
        annotated = [p for p in parts if source(p) == "annotated"]
        remote_camera = [p for p in parts if source(p) == "camera"]
        camera_only = [p for p in parts if source(p) == "camera_only"]
        local = [p for p in parts if source(p) == "camera_local"]
        statuses = sorted({p["headers"].get("x-stream-status") for p in probe.get("parts", [])
                           if p["headers"].get("x-stream-status")})
        values["stream_statuses"] = statuses
        values["x_stream_camera"] = (probe.get("headers") or {}).get("x-stream-camera")
        check("no analysed (annotated) frame while stopped", not annotated, len(annotated))
        if camera_while_stopped:
            check("owner-enabled camera-only mode: frames flow, labelled `camera_only`", bool(camera_only),
                  f"{len(camera_only)} camera_only part(s)")
            check("no unlabelled camera fallback frame while stopped", not remote_camera, len(remote_camera))
        else:
            check("Decision C: no remote camera picture while stopped (camera-only mode off)",
                  not remote_camera and not camera_only,
                  f"{len(remote_camera)} camera + {len(camera_only)} camera_only part(s)")
        if local:
            # Allowed (local maintenance), but it proves nothing about remote viewers:
            # run this check from another machine for the Decision C verdict.
            check("loopback viewer: frames labelled `camera_local` (local maintenance, allowed); "
                  "re-run from another machine to verify Decision C remotely", True,
                  f"{len(local)} camera_local part(s)")
    return {"source": "stream", "checks": checks, "values": values}


def verdict(results: list[dict]) -> int:
    """FAIL on any violation or failed check; SIMULATED when any data says it is simulated."""
    if not results:
        return BLOCKED
    if any(r.get("blocked") for r in results):
        bad = any(r.get("violations") or any(not c["ok"] for c in r.get("checks", [])) for r in results)
        return FAIL if bad else BLOCKED
    if any(r.get("violations") or any(not c["ok"] for c in r.get("checks", [])) for r in results):
        return FAIL
    if any((r.get("summary") or {}).get("simulated") for r in results):
        return SIMULATED
    return PASS


# ------------------------------------------------------------------ orchestration
def run(args: argparse.Namespace, *, env: Optional[dict[str, str]] = None,
        sleep: Callable[[float], None] = time.sleep) -> dict:
    env = dict(os.environ if env is None else env)
    results: list[dict] = []
    rows: list[dict] = []
    agree_history: list[list[str]] = []
    token = env.get(args.token_env, "")
    bearer = env.get(args.bearer_env, "")
    samples = max(1, int(args.samples))
    stream_running: Optional[bool] = None

    for i in range(samples):
        device_check = row_check = None
        if args.health_file or args.device:
            try:
                if args.health_file:
                    health = json.loads(Path(args.health_file).read_text())
                elif not token:
                    raise UnavailableError(f"${args.token_env} is empty: export the dashboard token first (RB-0)")
                else:
                    health = fetch_device_health(args.device, token)
                device_check = check_device(health)
                device_check["sample"] = i
                results.append(device_check)
                state = device_check["summary"]["monitoring_state"]
                stream_running = (False if device_check["summary"]["protection"] == "stopped"
                                  else True if state in ("ready", "degraded") else None)
            except UnavailableError as exc:
                results.append({"source": "device", "sample": i, "blocked": str(exc)})
        if args.devices_file or args.backend:
            try:
                if args.devices_file:
                    devices = json.loads(Path(args.devices_file).read_text())
                elif not bearer:
                    raise UnavailableError(f"${args.bearer_env} is empty: sign in and export the bearer token first")
                else:
                    devices = fetch_backend_devices(args.backend, bearer)
                hint = (device_check or {}).get("summary", {}).get("device_id")
                row = pick_row(devices, args.device_id, None if args.device_id is not None else hint)
                row_check = check_row(row, None if not args.devices_file else _row_clock(row))
                row_check["sample"] = i
                results.append(row_check)
                rows.append(row_check)
            except UnavailableError as exc:
                results.append({"source": "backend", "sample": i, "blocked": str(exc)})
        if device_check and row_check:
            agree_history.append(agreement(device_check, row_check))
        if i + 1 < samples:
            sleep(float(args.interval))

    transitions = transition_findings(rows)
    if transitions:
        results.append({"source": "transitions", "violations": transitions})
    # A disagreement in one sample is uplink lag; in two consecutive samples it is a finding.
    persistent = [a for a, b in itertools.pairwise(agree_history) if a and b]
    if persistent:
        results.append({"source": "agreement", "violations": [{"invariant": "device_backend_agree", "detail": d}
                                                              for d in persistent[-1]]})
    elif agree_history and agree_history[-1]:
        results.append({"source": "agreement", "note": f"one-sample disagreement (uplink lag?): {agree_history[-1]}"
                                                       "; re-run with --samples 3"})

    if args.stream_seconds and args.device:
        if not token:
            results.append({"source": "stream", "blocked": f"${args.token_env} is empty"})
        else:
            unauth = http_status(args.device.rstrip("/") + "/stream")
            probe = read_stream_parts(args.device, token, float(args.stream_seconds))
            if probe.get("status") is None:
                results.append({"source": "stream", "blocked": f"GET /stream unreachable: {probe.get('error')}"})
                return finish(results)
            stream = check_stream(probe, monitoring_running=stream_running,
                                  camera_while_stopped=bool(args.camera_while_stopped))
            stream["checks"].insert(0, {"check": "GET /stream without a token answers 401", "ok": unauth == 401,
                                        "detail": unauth})
            try:
                stream["values"]["stats"] = fetch_stream_stats(args.device, token)
            except UnavailableError as exc:
                stream["values"]["stats_error"] = str(exc)
            if stream_running is None:
                stream["note"] = "monitoring state unknown: only the header format was judged"
            results.append(stream)
    return finish(results)


def finish(results: list[dict]) -> dict:
    code = verdict(results)
    return {"tool": "check_live_status", "verdict": LABELS[code], "exit_code": code, "utc": _utc(),
            "simulated_data": any((r.get("summary") or {}).get("simulated") for r in results),
            "results": results}


def _row_clock(row: dict) -> Optional[float]:
    """For a recorded row: judge it at the moment it was observed, not today."""
    status = row.get("status") if isinstance(row.get("status"), dict) else {}
    freshness = status.get("health_freshness") if isinstance(status.get("health_freshness"), dict) else {}
    observed = freshness.get("observed_at") or row.get("last_seen")
    return float(observed) if isinstance(observed, (int, float)) else None


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def print_report(report: dict) -> None:
    for r in report["results"]:
        src = r.get("source")
        if r.get("blocked"):
            print(f"[BLOCKED] {src}: {r['blocked']}")
            continue
        if r.get("summary"):
            s = r["summary"]
            if src == "device":
                print(f"[device ] {s['device_id']} state={s['monitoring_state']} active={s['monitoring_active']} "
                      f"protection={s['protection']} accel={s['accelerator']['state']} "
                      f"versions={s['versions']} simulated={s['simulated']}")
            else:
                print(f"[backend] id={s['id']} {s['hardware_id']} online={s['online']} user_state={s['user_state']} "
                      f"state={s['monitoring_state']} protection={s['protection']} stale={s['stale']} "
                      f"command={s['protection_command']}")
        for v in r.get("violations", []):
            print(f"  [FAIL] {v['invariant']}: {v['detail']}")
        for c in r.get("checks", []):
            print(f"  [{'ok  ' if c['ok'] else 'FAIL'}] {c['check']}: {c['detail']}")
        if r.get("note"):
            print(f"  note: {r['note']}")
        if src == "stream":
            v = r["values"]
            print(f"  stream: {v['parts']} parts, seq {v['seq_first_last']}, age p50/max {v['age_ms_p50']}/"
                  f"{v['age_ms_max']} ms, sources {v['sources']}")
    tail = " (virtual-device data: NOT hardware evidence)" if report["simulated_data"] else ""
    print(f"LIVE STATUS: {report['verdict']}{tail}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--device", help="device dashboard base URL, e.g. http://localhost:8899")
    p.add_argument("--backend", help="backend base URL (GET /v1/devices)")
    p.add_argument("--device-id", type=int, default=None, help="backend device id (default: match the device's id)")
    p.add_argument("--health-file", help="a recorded GET /api/health JSON instead of --device")
    p.add_argument("--devices-file", help="a recorded GET /v1/devices JSON instead of --backend")
    p.add_argument("--token-env", default="SENSORDYME_TOKEN", help="env var holding the dashboard token")
    p.add_argument("--bearer-env", default="SENSORDYME_BEARER", help="env var holding the app's bearer token")
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--interval", type=float, default=5.0)
    p.add_argument("--stream-seconds", type=float, default=0.0, help="also read /stream for this long")
    p.add_argument("--camera-while-stopped", action="store_true",
                   help="the owner ENABLED the camera-only view while stopped (Decision C); default: off")
    p.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not any((args.device, args.backend, args.health_file, args.devices_file)):
        print("BLOCKED: give --device and/or --backend (or --health-file / --devices-file)", file=sys.stderr)
        return BLOCKED
    report = run(args)
    print_report(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
        print(f"report: {args.out}")
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
