"""M13: event-to-alert latency IN SOFTWARE, on the virtual device (SIMULATED; THIS host, not a Pi).

Two parts, both labelled SIMULATED: the perception is scripted (sim/perception.py),
the buzzer is the simulated sink, the backend is cloud/app.py in-process.

Part 1, the decision path (sim.harness.SimStack, the production inference pass):
for every alert the scenarios raise, wall-clock (time.perf_counter, which the
SimClock does not touch) from the moment the inference loop picks up the
decisive frame to

  (a) decision      AlertManager.notify() returns for the alertable event
  (b) buzzer        the buzzer sink's play() is called (simulated sink; on a Pi
                    the GPIO pulse follows on the pattern-player thread: stage
                    alert_output, measurable only there)
  (c) local event   DashboardServer.push_event() returned (the event is in
                    /api/events from here on), plus one real HTTP GET of
                    /api/events over loopback to confirm it and time a read
  (d) uplink queued CloudUplink.push_event() returned (queued for the backend)
      uplink sent   one sync_once() to the in-process backend, timed; the event
                    is then confirmed in the backend's /v1/devices/{id}/events

Part 2, the uplink's scheduling delay (real threads, real time): the uplink
worker of the checkout under test (--repo) against the in-process backend, at
cloud.poll_interval_s 2.0; N alerts pushed at random moments; time from
push_event() to the backend's 200 for the batch carrying it. Before M13 this
waited for the rest of the poll interval; with M13 the push wakes the worker.

    python benchmarks/m13_event_latency.py --out event_latency.json
    python benchmarks/m13_event_latency.py --repo /path/to/base --part uplink --out base_uplink.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _repo_root(argv: list[str]) -> Path:
    for i, a in enumerate(argv):
        if a == "--repo" and i + 1 < len(argv):
            return Path(argv[i + 1]).resolve()
    return HERE.parent


REPO = _repo_root(sys.argv)
sys.path.insert(0, str(REPO))


def stats(values: list[float]):
    if not values:
        return None
    s = sorted(values)

    def q(p):
        pos = p * (len(s) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (pos - lo)

    return {"n": len(s), "p50": round(q(0.5), 4), "p90": round(q(0.9), 4), "mean": round(statistics.fmean(s), 4),
            "max": round(s[-1], 4)}


# ---------------------------------------------------------------------- part 1
def decision_path(scenarios: list[str], seconds: float, step: float) -> dict:
    import urllib.request

    from sim import scenario as sc
    from sim.harness import SimStack

    rows: list[dict] = []
    for name in scenarios:
        with tempfile.TemporaryDirectory(prefix="m13evt-") as tmp, SimStack(tmp) as stack:
            device = stack.boot(sc.get(name), dashboard_port=0)
            stack.claim()
            rt = device.runtime
            assert rt is not None and rt.dashboard is not None and rt.uplink is not None
            marks: dict = {}

            def wrap(owner, attr, key, marks=marks):
                original = getattr(owner, attr)

                def timed(*a, **k):
                    out = original(*a, **k)
                    marks.setdefault(key, []).append((time.perf_counter(), a, out))
                    return out

                setattr(owner, attr, timed)

            def pick(*a, _original=rt.bus.wait_for_fresh_stamped, _marks=marks, **k):
                out = _original(*a, **k)
                _marks["pick"] = time.perf_counter()
                return out

            rt.bus.wait_for_fresh_stamped = pick
            wrap(rt.alert_manager, "notify", "decision")
            wrap(device.buzzer, "play", "buzzer")
            wrap(rt.dashboard, "push_event", "local")
            wrap(rt.uplink, "push_event", "uplink_queued")
            url = rt.dashboard.url.rstrip("/") + "/api/events?limit=5"
            token = device.dashboard_token
            for _ in range(max(1, round(seconds / step))):
                for key in ("decision", "buzzer", "local", "uplink_queued"):
                    marks.pop(key, None)
                device.advance(step, step=step)
                decisions = [m for m in marks.get("decision", []) if m[2] is not None and getattr(m[2], "name", "")
                             != "SUPPRESSED"]
                if not decisions or "pick" not in marks:
                    continue
                t0 = marks["pick"]
                event = decisions[0][1][0]
                row = {"scenario": name, "event": event.type,
                       "a_decision_ms": (decisions[0][0] - t0) * 1e3}
                if marks.get("buzzer"):
                    row["b_buzzer_play_called_ms"] = (marks["buzzer"][0][0] - t0) * 1e3
                if marks.get("local"):
                    row["c_local_event_ms"] = (marks["local"][0][0] - t0) * 1e3
                if marks.get("uplink_queued"):
                    row["d_uplink_queued_ms"] = (marks["uplink_queued"][0][0] - t0) * 1e3
                g0 = time.perf_counter()
                req = urllib.request.Request(url, headers={"X-Device-Token": token})  # noqa: S310 - loopback
                with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310 - loopback, our dashboard
                    body = json.loads(resp.read())
                row["c_http_get_api_events_ms"] = (time.perf_counter() - g0) * 1e3
                row["c_event_in_api_events"] = any(e.get("type") == event.type for e in body.get("events", []))
                s0 = time.perf_counter()
                rt.uplink.sync_once()
                row["d_sync_once_to_backend_ms"] = (time.perf_counter() - s0) * 1e3
                row["d_event_stored_by_backend"] = event.type in stack.event_types()
                rows.append(row)
    keys = ("a_decision_ms", "b_buzzer_play_called_ms", "c_local_event_ms", "c_http_get_api_events_ms",
            "d_uplink_queued_ms", "d_sync_once_to_backend_ms")
    return {"alerts": rows, "summary_ms": {k: stats([r[k] for r in rows if k in r]) for k in keys}}


# ---------------------------------------------------------------------- part 2
def uplink_scheduling(trials: int, poll_interval_s: float, seed: int) -> dict:
    import types

    from core.uplink import CloudUplink
    from sim.backend import InProcessBackend
    from sim.uplink import TestClientTransport

    rng = random.Random(seed)  # noqa: S311 - reproducible probe timing, not a secret
    with tempfile.TemporaryDirectory(prefix="m13up-") as tmp:
        backend = InProcessBackend(Path(tmp) / "backend")
        try:
            factory = backend.provision()
            transport = TestClientTransport(backend.client)
            delivered: dict[str, float] = {}

            class Timed:
                def post(self, url, **kw):
                    resp = transport.post(url, **kw)
                    now = time.perf_counter()
                    payload = kw.get("json")
                    if payload is None and kw.get("data") is not None:
                        payload = json.loads(kw["data"])
                    for e in (payload or {}).get("events", []):
                        if getattr(resp, "status_code", 0) == 200:
                            delivered.setdefault(e["type"], now)
                    return resp

            uplink = CloudUplink({"base_url": backend.base_url, "device_key": factory["device_key"],
                                  "poll_interval_s": poll_interval_s})
            uplink._requests = Timed()
            worker = threading.Thread(target=uplink._run, daemon=True)
            worker.start()
            time.sleep(0.5)  # first heartbeat out of the way
            waits = []
            for i in range(trials):
                time.sleep(rng.uniform(0.0, poll_interval_s))  # a random phase of the poll cycle
                kind = f"m13_probe_{i}"
                ev = types.SimpleNamespace(timestamp=types.SimpleNamespace(isoformat=lambda: "2026-11-16T08:00:00"),
                                           type=kind, confidence=0.9, contributing_signals="SIMULATED probe",
                                           duration=1.0)
                t0 = time.perf_counter()
                uplink.push_event(ev, "loud")
                end = t0 + poll_interval_s * 3
                while kind not in delivered and time.perf_counter() < end:
                    time.sleep(0.001)
                if kind in delivered:
                    waits.append((delivered[kind] - t0) * 1e3)
            uplink.close()
            worker.join(5)
        finally:
            backend.close()
    return {"poll_interval_s": poll_interval_s, "trials": trials, "delivered": len(waits),
            "push_to_backend_200_ms": stats(waits)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", default=str(HERE.parent))
    p.add_argument("--part", choices=("all", "decision", "uplink"), default="all")
    p.add_argument("--scenarios", default="drowsy,phone,distracted")
    p.add_argument("--seconds", type=float, default=90.0)
    p.add_argument("--step", type=float, default=1 / 30)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--poll-interval", type=float, default=2.0)
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    import logging

    logging.disable(logging.WARNING)
    out = {
        "schema": "sd-m13-event-latency/1",
        "label": "SIMULATED (virtual device: scripted perception, simulated buzzer sink, in-process backend); "
                 "desktop host, NOT a Raspberry Pi",
        "conditions": {"platform": platform.platform(), "python": platform.python_version(),
                       "loadavg": list(os.getloadavg()), "repo": str(REPO), "step_s": args.step},
    }
    if args.part in ("all", "decision"):
        out["decision_path"] = decision_path([s for s in args.scenarios.split(",") if s], args.seconds, args.step)
    if args.part in ("all", "uplink"):
        out["uplink_scheduling"] = uplink_scheduling(args.trials, args.poll_interval, seed=13)
    text = json.dumps(out, indent=1, default=str)
    if args.out:
        args.out.write_text(text + "\n")
    if "decision_path" in out:
        for k, v in out["decision_path"]["summary_ms"].items():
            print(f"{k:28s} {v}")
        print("alerts:", [(r["scenario"], r["event"]) for r in out["decision_path"]["alerts"]])
    if "uplink_scheduling" in out:
        print("uplink push->backend:", out["uplink_scheduling"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
