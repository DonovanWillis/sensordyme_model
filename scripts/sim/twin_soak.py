"""M8.4 digital-twin soak: the virtual device + the real backend, for hours.

Two modes, both SIMULATED (scripted camera and models; never a Pi measurement):

  live      real processes, real time, real HTTP: `python -m sim.backend` (uvicorn)
            and a virtual device process (its real runtime threads, dashboard and
            /stream), driven by a repeating timeline (drowsiness, phone, distraction,
            network flaps with an alert raised offline, camera freeze, detector
            failure) while this controller stops/resumes monitoring through the API,
            hard-kills and restarts the device process, restarts the backend, reads
            /stream continuously and polls GET /v1/devices like the app. Measures
            memory/CPU/threads/fds of both processes, queue depth, event loss and
            duplicates (device-side ledger vs backend rows by event_key), stream
            stalls and reconnects, event and API tail latency, and every app row
            against the M8.3 truth invariants (sim.invariants.status_violations).

  simclock  the same timeline on the SimClock (tests' SimStack + InvariantMonitor),
            through a lossy/duplicating/reordering link (sim.faults.LinkFaults),
            with stop/resume, device restarts, power cuts and backend restarts:
            hours of device time in minutes, deterministic, every transition checked.

    python scripts/sim/twin_soak.py live --minutes 150 --out OUT
    python scripts/sim/twin_soak.py simclock --hours 24 --out OUT

Writes OUT/summary.json, OUT/summary.md and OUT/samples.csv (small files).
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import json
import os
import platform
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from sim.perception import Scene  # noqa: E402
from sim.scenario import (  # noqa: E402
    ASLEEP,
    ATTENTIVE,
    CALIBRATION_S,
    LOOKING_AWAY,
    PHONE_IN_LAP,
    Faults,
    Scenario,
    Segment,
)

DOWN = Faults(network_down=True)
#: One cycle of the soak timeline: (offset s, scene, faults). The tail from
#: QUIET_FROM is quiet: the controller's stop/resume and restarts happen there.
CYCLE: tuple[tuple[float, Scene, Faults], ...] = (
    (0.0, ATTENTIVE, Faults()),
    (20.0, ASLEEP, Faults()), (24.0, ATTENTIVE, Faults()),                       # drowsiness, online
    (60.0, PHONE_IN_LAP, Faults()), (66.0, ATTENTIVE, Faults()),                 # phone use
    (100.0, LOOKING_AWAY, Faults()), (105.0, ATTENTIVE, Faults()),               # distraction
    (130.0, ATTENTIVE, DOWN), (150.0, ATTENTIVE, Faults()),                      # 20 s flap
    (170.0, ATTENTIVE, DOWN), (180.0, ASLEEP, DOWN), (184.0, ATTENTIVE, DOWN),   # alert raised offline
    (215.0, ATTENTIVE, Faults()),                                                # ... 45 s outage ends
    (230.0, ATTENTIVE, Faults(camera="frozen")), (245.0, ATTENTIVE, Faults()),   # camera freeze 15 s
    (260.0, ATTENTIVE, Faults(detector_failing=True)), (280.0, ATTENTIVE, Faults()),  # detector 20 s
)
#: --quiet-timeline (M8.5 memory attribution): the same cycle length, nothing happening.
QUIET_CYCLE: tuple[tuple[float, Scene, Faults], ...] = ((0.0, ATTENTIVE, Faults()),)
CYCLE_S = 360.0
QUIET_FROM = 290.0
STOP_AT, RESUME_AT, DISRUPT_AT = 300.0, 325.0, 345.0
#: The last stretch of a soak is fault-free so every backlog drains before counting.
DRAIN_S = 150.0


def soak_scenario(total_s: float, *, cycle_s: float = CYCLE_S, reboot_every_s: float = 0.0,
                  quiet: bool = False) -> Scenario:
    """The repeating soak timeline: calibration, cycles, then a fault-free drain."""
    timeline = QUIET_CYCLE if quiet else CYCLE
    segments = [Segment(0.0)]
    start = CALIBRATION_S
    end = max(start, total_s - DRAIN_S)
    while start + cycle_s <= end:
        segments += [Segment(start + off, scene, faults) for off, scene, faults in timeline]
        start += cycle_s
    segments.append(Segment(max(start, segments[-1].start + 1.0), ATTENTIVE, Faults()))
    cycles = int((end - CALIBRATION_S) // cycle_s)
    reboots: tuple[float, ...] = ()
    if reboot_every_s > 0:
        every = max(1, round(reboot_every_s / cycle_s))
        reboots = tuple(CALIBRATION_S + k * cycle_s + DISRUPT_AT + 5.0 for k in range(every - 1, cycles, every))
    return Scenario("m84_soak", "M8.4 soak: repeating faults, alerts and recoveries, then a drain",
                    tuple(segments), reboots=reboots)


def fault_at(scenario: Scenario, t: float) -> Optional[tuple[str, float]]:
    """("frozen" | "detector", seconds into it) when that fault is in force at scenario time `t`."""
    seg = scenario.at(t)
    if seg.faults.camera == "frozen":
        return "frozen", t - seg.start
    if seg.faults.detector_failing:
        return "detector", t - seg.start
    return None


def cycle_offset(t: float, cycle_s: float = CYCLE_S) -> Optional[float]:
    if t < CALIBRATION_S:
        return None
    return (t - CALIBRATION_S) % cycle_s


def pct(values: list[float], p: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(p * len(ordered)))], 3)


def slope_per_hour(points: list[tuple[float, float]]) -> Optional[float]:
    """Least-squares slope of (seconds, value) in value per hour."""
    if len(points) < 3:
        return None
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return None if den == 0 else round(sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / den * 3600, 3)


def _git_commit() -> str:
    with contextlib.suppress(Exception):
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                              timeout=10).stdout.strip() or os.environ.get("SDJOB_COMMIT", "unknown")
    return os.environ.get("SDJOB_COMMIT", "unknown")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ====================================================================== live: device process
def run_device(args: argparse.Namespace) -> int:
    """The device process (spawned by the live controller; restarted after a kill)."""
    # M8.5: a memory sampler inside this process when the controller asked for one
    # (SDSOAK_MEMPROF; scripts/sim/memory_profile.py). Started first, so the
    # Python heap it reports includes the device's imports.
    sampler = None
    if os.environ.get("SDSOAK_MEMPROF"):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import memory_profile

        sampler = memory_profile.start_from_env(label="device")
    from sim.device import VirtualDevice
    from sim.faults import EventLedger
    from sim.uplink import SimUplink

    SimUplink.LEDGER = EventLedger(args.ledger)
    factory = json.loads(Path(args.factory).read_text())
    entry = factory["devices"][0]
    elapsed = max(0.0, time.time() - args.start_wall)
    scenario = soak_scenario(args.total_s, reboot_every_s=args.reboot_every_s, quiet=args.quiet_timeline)
    scenario = replace(scenario, reboots=tuple(t for t in scenario.reboots if t > elapsed + 5.0))
    device = VirtualDevice(state_dir=args.state_dir, backend_url=factory["backend_url"], scenario=scenario,
                           device_key=entry["device_key"], claim_code=entry["claim_code"],
                           dashboard_port=args.dashboard_port, fps=args.fps, log_level="warning")
    # Same timeline across process restarts: scenario time is wall time since the soak began.
    device.scenario_clock.origin = time.monotonic() - elapsed
    if args.driver_attention_override:
        # One factor at a time (M8.5): e.g. {"hands_policy": "always"}, merged into this
        # device's modules.driver_attention config at every (re)build.
        override = json.loads(args.driver_attention_override)
        base_config = device.config

        def config_with_override() -> dict:
            config = base_config()
            config["modules"]["driver_attention"].update(override)
            return config

        device.config = config_with_override  # type: ignore[method-assign]
    if sampler is not None:
        base_build = device.build

        def build_marked():
            sampler.mark("runtime boot")
            return base_build()

        device.build = build_marked  # type: ignore[method-assign]
    print(f"device token: {device.dashboard_token}", flush=True)
    device.run()
    return 0


# ====================================================================== live: controller
class Proc:
    def __init__(self, argv: list[str], log_path: Path, env: Optional[dict] = None) -> None:
        import psutil

        self.log = open(log_path, "a")  # noqa: SIM115 - closed in stop()
        self.proc = subprocess.Popen(argv, cwd=REPO, stdout=self.log, stderr=subprocess.STDOUT,
                                     start_new_session=True, env={**os.environ, **(env or {})})
        self.ps = psutil.Process(self.proc.pid)
        self.ps.cpu_percent(None)
        self.started = time.time()

    def alive(self) -> bool:
        return self.proc.poll() is None

    def sample(self) -> dict:
        import psutil

        try:
            with self.ps.oneshot():
                fds = self.ps.num_fds() if hasattr(self.ps, "num_fds") else None
                return {"rss_mb": round(self.ps.memory_info().rss / 2**20, 2), "cpu_pct": self.ps.cpu_percent(None),
                        "threads": self.ps.num_threads(), "fds": fds}
        except psutil.Error:
            return {}

    def stop(self, *, hard: bool = False, timeout: float = 30.0) -> None:
        if self.alive():
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.proc.pid, signal.SIGKILL if hard else signal.SIGINT)
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=10)
        self.log.close()


class StreamWatcher(threading.Thread):
    """Reads /stream for the whole soak like the app does; reconnects when cut."""

    def __init__(self, port: int, token_path: Path, windows) -> None:
        super().__init__(daemon=True)
        self.port, self.token_path, self.windows = port, token_path, windows
        self.stop_flag = threading.Event()
        self.frames = 0
        self.heartbeats = 0
        self.connects = 0
        self.connect_failures = 0
        self.gaps: list[float] = []
        self.ages: list[float] = []
        self.stalls: list[dict] = []
        self.expected_gaps = 0
        self.max_buffer = 0

    def run(self) -> None:
        from sim.stream_client import StreamClient

        last_frame: Optional[float] = None
        while not self.stop_flag.is_set():
            try:
                token = self.token_path.read_text().strip()
                client = StreamClient(self.port, token, heartbeat=True, keep_bodies=False)
            except OSError:
                self.connect_failures += 1
                self.stop_flag.wait(1.0)
                continue
            self.connects += 1
            try:
                while not self.stop_flag.is_set():
                    client.read_for(1.0)
                    for part in client.parts:
                        if part.is_heartbeat:
                            self.heartbeats += 1
                            continue
                        if not part.is_jpeg:
                            continue
                        # Arrival time of THIS part (monotonic), on the wall clock.
                        now = time.time() - (time.monotonic() - part.received_at)
                        self.frames += 1
                        with contextlib.suppress(KeyError, ValueError):
                            self.ages.append(part.int_header("X-Frame-Age-Ms"))
                        if last_frame is not None:
                            gap = now - last_frame
                            self.gaps.append(gap)
                            if gap > 2.0:
                                why = self.windows.explain(last_frame, now)
                                if why:
                                    self.expected_gaps += 1
                                else:
                                    self.stalls.append({"at": round(now, 1), "gap_s": round(gap, 2)})
                        last_frame = now
                    self.max_buffer = max(self.max_buffer, client.max_buffer)
                    client.parts.clear()
                    if client.closed_by_peer:
                        break
            except OSError:
                pass
            finally:
                client.close()
            self.stop_flag.wait(0.5)

    def summary(self) -> dict:
        gaps_ms = [g * 1000 for g in self.gaps]
        return {"frames": self.frames, "heartbeats": self.heartbeats, "connects": self.connects,
                "reconnects": max(0, self.connects - 1), "connect_failures": self.connect_failures,
                "gap_ms_p50": pct(gaps_ms, 0.5), "gap_ms_p99": pct(gaps_ms, 0.99),
                "gap_ms_max": round(max(gaps_ms), 1) if gaps_ms else None,
                "frame_age_ms_p50": pct(self.ages, 0.5), "frame_age_ms_p99": pct(self.ages, 0.99),
                "frame_age_ms_max": max(self.ages) if self.ages else None,
                "gaps_over_2s_explained": self.expected_gaps, "unexplained_stalls": len(self.stalls),
                "stall_examples": self.stalls[:10], "client_max_buffer_bytes": self.max_buffer}


class Windows:
    """Wall-time windows in which a gap or a missing alert is expected."""

    def __init__(self, start_wall: float, scenario: Scenario) -> None:
        self.start_wall = start_wall
        self.scenario = scenario
        self.disruptions: list[tuple[float, float, str]] = []
        self._lock = threading.Lock()

    def add(self, start: float, end: float, why: str) -> None:
        with self._lock:
            self.disruptions.append((start, end, why))

    def scenario_faults(self, wall: float) -> Faults:
        return self.scenario.at(wall - self.start_wall).faults

    def explain(self, a: float, b: float) -> Optional[str]:
        with self._lock:
            for start, end, why in self.disruptions:
                if start - 2 <= b and a <= end + 10:
                    return why
        for probe in (a, (a + b) / 2, b):
            faults = self.scenario_faults(probe)
            if faults.camera != "ok":
                return f"camera {faults.camera}"
        return None


class Api:
    def __init__(self, base: str) -> None:
        import requests

        self.base = base
        self.session = requests.Session()
        self.headers: dict[str, str] = {}
        self.latencies: list[float] = []
        self.errors = 0

    def call(self, method: str, path: str, **kwargs):
        t0 = time.perf_counter()
        try:
            response = self.session.request(method, self.base + path, headers=self.headers, timeout=10, **kwargs)
        except Exception:  # noqa: BLE001 - a soak counts every failed call, whatever it was
            self.errors += 1
            return None
        self.latencies.append((time.perf_counter() - t0) * 1000)
        return response

    def login(self, email: str, password: str) -> None:
        response = self.call("POST", "/v1/login", json={"email": email, "password": password})
        assert response is not None and response.status_code == 200, getattr(response, "text", None)
        self.headers = {"Authorization": f"Bearer {response.json()['token']}"}

    def all_events(self, device_id: int) -> list[dict]:
        rows: list[dict] = []
        before = None
        while True:
            params: dict[str, Any] = {"limit": 1000}
            if before is not None:
                params["before_id"] = before
            response = self.call("GET", f"/v1/devices/{device_id}/events", params=params)
            if response is None or response.status_code != 200:
                break
            page = response.json()["events"]
            rows += page
            if len(page) < 1000:
                break
            before = page[-1]["id"]
        return rows


def _wait_http(url: str, timeout: float = 90.0) -> bool:
    import requests

    deadline = time.time() + timeout
    while time.time() < deadline:
        with contextlib.suppress(Exception):
            if requests.get(url, timeout=2).status_code < 500:
                return True
        time.sleep(0.5)
    return False


def run_live(args: argparse.Namespace) -> dict:
    from core import state_model
    from sim.faults import EventLedger, compare
    from sim.invariants import status_violations

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="m84-soak-", dir=args.work_dir))
    total_s = args.minutes * 60.0
    start_wall = time.time()
    scenario = soak_scenario(total_s, reboot_every_s=args.reboot_every_s, quiet=args.quiet_timeline)
    windows = Windows(start_wall, scenario)
    backend_port, dash_port = _free_port(), _free_port()
    backend_dir, state_dir, ledger_path = work / "backend", work / "device", work / "ledger.jsonl"
    py = sys.executable
    device_env: dict[str, str] = {}
    if args.mem_profile != "off":
        device_env["SDSOAK_MEMPROF"] = json.dumps({
            "out": str(out.resolve()), "interval_s": args.mem_sample_s, "types_every_s": args.mem_types_s,
            "warmup_s": args.mem_warmup_s, "snapshot_s": args.mem_snapshot_s,
            "tracemalloc_frames": args.tracemalloc_frames if args.mem_profile == "tracemalloc" else 0})
    if args.malloc_arena_max:
        device_env["MALLOC_ARENA_MAX"] = str(args.malloc_arena_max)

    def start_backend() -> Proc:
        proc = Proc([py, "-m", "sim.backend", "--port", str(backend_port), "--data-dir", str(backend_dir)],
                    out / "backend.log")
        if not _wait_http(f"http://127.0.0.1:{backend_port}/v1/health"):
            raise RuntimeError("backend did not come up")
        return proc

    def start_device() -> Proc:
        return Proc([py, "-u", str(Path(__file__).resolve()), "_device", "--factory", str(backend_dir / "factory.json"),
                     "--state-dir", str(state_dir), "--dashboard-port", str(dash_port), "--fps", str(args.fps),
                     "--ledger", str(ledger_path), "--start-wall", repr(start_wall), "--total-s", repr(total_s),
                     "--reboot-every-s", repr(args.reboot_every_s),
                     *(["--quiet-timeline"] if args.quiet_timeline else []),
                     *(["--driver-attention-override", args.driver_attention_override]
                       if args.driver_attention_override else [])],
                    out / "device.log", env=device_env)

    backend = start_backend()
    factory = json.loads((backend_dir / "factory.json").read_text())
    device = start_device()
    api = Api(f"http://127.0.0.1:{backend_port}")
    api.login(factory["email"], factory["password"])
    claim = api.call("POST", "/v1/devices/claim", json={"claim_code": factory["devices"][0]["claim_code"]})
    assert claim is not None and claim.status_code == 200, getattr(claim, "text", None)
    device_id = int(claim.json()["device_id"])
    token_path = state_dir / "dashboard_token"
    deadline = time.time() + 120
    while not token_path.exists() and time.time() < deadline:
        time.sleep(0.5)
    # --viewers N: N concurrent /stream readers (the app, a second phone, ...); 0 = none.
    viewers = [StreamWatcher(dash_port, token_path, windows) for _ in range(args.viewers)]
    for viewer in viewers:
        viewer.start()

    kill_device_at = sorted(f * total_s for f in args.device_kills)
    restart_backend_at = sorted(f * total_s for f in args.backend_restarts)
    samples: list[dict] = []
    violations: collections.Counter = collections.Counter()
    violation_examples: list[str] = []
    user_states: list[str] = []
    false_protecting: list[dict] = []
    missed_degraded: list[dict] = []
    queue_depths: list[int] = []
    dropped_max = 0
    online_flips = 0
    last_online: Optional[bool] = None
    stops = resumes = stop_confirmed = resume_confirmed = 0
    commands: list[dict] = []
    done_cycle_actions: set[tuple[int, str]] = set()
    next_poll = next_ps = time.time()
    proc_samples: dict[str, list[tuple[float, dict]]] = {"device": [], "backend": []}
    device_lifetimes: list[tuple[float, float]] = [(time.time(), float("inf"))]

    def disrupt_device() -> None:
        nonlocal device
        t0 = time.time()
        device.stop(hard=True)
        device = start_device()
        windows.add(t0, time.time() + 25, "device process killed and restarted")
        device_lifetimes[-1] = (device_lifetimes[-1][0], t0)
        device_lifetimes.append((time.time(), float("inf")))

    def disrupt_backend() -> None:
        nonlocal backend
        t0 = time.time()
        backend.stop(hard=True)
        backend = start_backend()
        windows.add(t0, time.time() + 5, "backend restarted")

    try:
        while time.time() - start_wall < total_s:
            now = time.time()
            t = now - start_wall
            off = cycle_offset(t)
            cycle = int((t - CALIBRATION_S) // CYCLE_S) if off is not None else -1
            in_drain = t > total_s - DRAIN_S
            # ---- scheduled actions (the cycle's quiet tail), never in the final drain
            if off is not None and not in_drain:
                if args.stop_resume and off >= STOP_AT and (cycle, "stop") not in done_cycle_actions \
                        and off < RESUME_AT:
                    done_cycle_actions.add((cycle, "stop"))
                    r = api.call("POST", f"/v1/devices/{device_id}/protection", json={"action": "stop", "confirm": True})
                    if r is not None and r.status_code == 200:
                        stops += 1
                        commands.append({"action": "stop", "id": r.json()["command_id"], "at": now})
                        windows.add(now, now + (RESUME_AT - STOP_AT) + 10, "monitoring stopped")
                if args.stop_resume and off >= RESUME_AT and (cycle, "resume") not in done_cycle_actions \
                        and off < DISRUPT_AT:
                    done_cycle_actions.add((cycle, "resume"))
                    r = api.call("POST", f"/v1/devices/{device_id}/protection", json={"action": "resume"})
                    if r is not None and r.status_code == 200:
                        resumes += 1
                        commands.append({"action": "resume", "id": r.json()["command_id"], "at": now})
                if off >= DISRUPT_AT and (cycle, "disrupt") not in done_cycle_actions:
                    done_cycle_actions.add((cycle, "disrupt"))
                    if kill_device_at and t >= kill_device_at[0]:
                        kill_device_at.pop(0)
                        disrupt_device()
                    elif restart_backend_at and t >= restart_backend_at[0]:
                        restart_backend_at.pop(0)
                        disrupt_backend()
            if not device.alive():  # it crashed on its own: that is a finding, and it comes back
                windows.add(now, now + 25, "device process exited unexpectedly")
                violations["device_process_exited"] += 1
                if violations["device_process_exited"] > 5:
                    tail = (out / "device.log").read_text(errors="replace")[-3000:]
                    raise RuntimeError(f"the device process keeps exiting; aborting the soak:\n{tail}")
                device = start_device()
            if not backend.alive():
                violations["backend_process_exited"] += 1
                backend = start_backend()
            # ---- the app's view
            if now >= next_poll:
                next_poll = now + args.poll_s
                r = api.call("GET", "/v1/devices")
                if r is not None and r.status_code == 200:
                    rows = [d for d in r.json()["devices"] if d["id"] == device_id]
                    if rows:
                        row = rows[0]
                        status = row["status"]
                        for name, detail in status_violations(row):
                            violations[name] += 1
                            if len(violation_examples) < 20:
                                violation_examples.append(f"t={t:.0f}s {name}: {detail}")
                        derived = state_model.derive(row, time.time()).state
                        if not user_states or user_states[-1] != derived:
                            user_states.append(derived)
                        if last_online is not None and row["online"] and not last_online:
                            online_flips += 1
                        last_online = bool(row["online"])
                        health = row.get("health") or {}
                        backlog = health.get("uplink_backlog") or {}
                        if isinstance(backlog.get("queued"), int):
                            queue_depths.append(backlog["queued"])
                            dropped_max = max(dropped_max, int(backlog.get("dropped") or 0))
                        state = status.get("monitoring_state")
                        # Judged on THIS row's time (scenario time = wall time since the start),
                        # not health.simulation.scenario_time_s: the health sample is taken every
                        # 5 s while the monitoring state is pushed at each transition, so the two
                        # can be seconds apart and the older one would accuse a recovered device.
                        if not status.get("stale"):
                            current = fault_at(scenario, t)
                            if current is not None:
                                kind, into = current
                                if kind == "frozen" and into >= 10 and state in ("ready", "degraded"):
                                    false_protecting.append({"t": round(t, 1), "into_s": round(into, 1),
                                                             "state": state})
                                if kind == "detector" and into >= 10 and state == "ready":
                                    missed_degraded.append({"t": round(t, 1), "into_s": round(into, 1)})
                        for cmd in commands:
                            pc = status.get("protection_command") or {}
                            if pc.get("id") == cmd["id"] and pc.get("outcome") == "confirmed" and not cmd.get("confirmed"):
                                cmd["confirmed"] = time.time() - cmd["at"]
                        samples.append({"t": round(t, 1), "state": state, "user_state": derived,
                                        "online": row["online"], "stale": status.get("stale"),
                                        "queued": backlog.get("queued"), "dropped": backlog.get("dropped")})
            if now >= next_ps:
                next_ps = now + args.sample_s
                for name, proc in (("device", device), ("backend", backend)):
                    s = proc.sample()
                    if s:
                        proc_samples[name].append((t, s))
            time.sleep(0.2)

        # ---- the drain is over: count
        time.sleep(args.poll_s)
        stored = api.all_events(device_id)
        generated = EventLedger.read(ledger_path)
        integrity = compare(generated, stored)
    finally:
        for viewer in viewers:
            viewer.stop_flag.set()
        for viewer in viewers:
            viewer.join(timeout=10)
        device.stop()
        backend.stop()

    stops_confirmed = [c["confirmed"] for c in commands if c["action"] == "stop" and c.get("confirmed")]
    resumes_confirmed = [c["confirmed"] for c in commands if c["action"] == "resume" and c.get("confirmed")]
    del stop_confirmed, resume_confirmed
    latencies = [r["received_at"] - r["occurred_at"] for r in stored
                 if isinstance(r.get("received_at"), (int, float)) and isinstance(r.get("occurred_at"), (int, float))]
    online_lat = []
    for r in stored:
        occurred = r.get("occurred_at")
        if not isinstance(occurred, (int, float)) or not isinstance(r.get("received_at"), (int, float)):
            continue
        faults = windows.scenario_faults(occurred)
        if not faults.network_down and not windows.explain(occurred - 5, occurred + 5):
            online_lat.append(r["received_at"] - occurred)

    def proc_summary(name: str) -> dict:
        pts = proc_samples[name]
        if not pts:
            return {}
        rss = [s["rss_mb"] for _, s in pts]
        cpu = [s["cpu_pct"] for _, s in pts]
        out_d: dict[str, Any] = {"samples": len(pts), "rss_mb_first": rss[0], "rss_mb_last": rss[-1],
                                 "rss_mb_max": max(rss), "cpu_pct_mean": round(statistics.fmean(cpu), 1),
                                 "cpu_pct_p95": pct(cpu, 0.95),
                                 "threads_max": max(s.get("threads") or 0 for _, s in pts),
                                 "fds_max": max((s.get("fds") or 0) for _, s in pts)}
        if name == "device":
            start, end = device_lifetimes[-1][0] - start_wall, total_s
            life = [(tt, s["rss_mb"]) for tt, s in pts if start + 600 <= tt <= end]
            out_d["final_lifetime_s"] = round(end - start)
            out_d["rss_slope_mb_per_h_final_lifetime_after_10min"] = slope_per_hour(life)
            threads = [(tt, s["threads"]) for tt, s in pts if start + 600 <= tt <= end]
            out_d["threads_first_last_final_lifetime"] = [threads[0][1], threads[-1][1]] if threads else None
        else:
            out_d["rss_slope_mb_per_h_after_10min"] = slope_per_hour([(tt, s["rss_mb"]) for tt, s in pts if tt >= 600])
        return out_d

    summary = {
        "simulated": True, "mode": "live", "commit": os.environ.get("SDJOB_COMMIT") or _git_commit(),
        "platform": platform.platform(), "python": platform.python_version(),
        "planned_minutes": args.minutes, "completed_s": round(time.time() - start_wall, 1),
        "cycles": int((total_s - DRAIN_S - CALIBRATION_S) // CYCLE_S), "fps": args.fps,
        "processes": {"device": proc_summary("device"), "backend": proc_summary("backend")},
        "stops": stops, "resumes": resumes,
        "stop_confirm_s_p50": pct(stops_confirmed, 0.5), "stop_confirm_s_max": max(stops_confirmed, default=None),
        "stops_confirmed": len(stops_confirmed), "resumes_confirmed": len(resumes_confirmed),
        "resume_confirm_s_p50": pct(resumes_confirmed, 0.5), "resume_confirm_s_max": max(resumes_confirmed, default=None),
        "device_kills": len(args.device_kills), "backend_restarts": len(args.backend_restarts),
        "in_process_reboots_scheduled": len(scenario.reboots),
        "events": {k: (len(v) if isinstance(v, list) else v) for k, v in integrity.items()},
        "events_lost_keys": integrity["lost"][:20], "events_duplicate_keys": integrity["duplicate_keys"][:20],
        "event_types": dict(collections.Counter(r["type"] for r in stored)),
        "event_latency_s": {"all_p50": pct(latencies, 0.5), "all_p99": pct(latencies, 0.99),
                            "all_max": round(max(latencies), 2) if latencies else None,
                            "online_n": len(online_lat), "online_p50": pct(online_lat, 0.5),
                            "online_p95": pct(online_lat, 0.95), "online_p99": pct(online_lat, 0.99),
                            "online_max": round(max(online_lat), 2) if online_lat else None},
        "api_latency_ms": {"n": len(api.latencies), "p50": pct(api.latencies, 0.5), "p99": pct(api.latencies, 0.99),
                           "max": round(max(api.latencies), 1) if api.latencies else None, "errors": api.errors},
        "queue": {"max": max(queue_depths, default=None), "p95": pct([float(q) for q in queue_depths], 0.95),
                  "dropped_max": dropped_max},
        "reconnects_app_online_flips": online_flips,
        "stream": viewers[0].summary() if viewers else None,
        "viewers": args.viewers, "stop_resume": args.stop_resume, "quiet_timeline": args.quiet_timeline,
        "driver_attention_override": args.driver_attention_override, "mem_profile": args.mem_profile,
        "malloc_arena_max": args.malloc_arena_max,
        "invariant_violations": dict(violations), "invariant_examples": violation_examples,
        "false_protecting_samples": len(false_protecting), "false_protecting_examples": false_protecting[:10],
        "detector_fault_shown_ready_samples": len(missed_degraded),
        "user_state_transitions": len(user_states), "user_states_seen": sorted(set(user_states)),
        "app_samples": len(samples),
    }
    # M8.5: the per-sample process time series (the M8.4 summary kept only first/last/max).
    with open(out / "samples_proc_live.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "process", "rss_mb", "cpu_pct", "threads", "fds"])
        for name, pts in proc_samples.items():
            for tt, s in pts:
                writer.writerow([round(tt, 1), name, s.get("rss_mb"), s.get("cpu_pct"), s.get("threads"), s.get("fds")])
    if args.mem_profile != "off":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import memory_profile

        with contextlib.suppress(Exception):  # the soak's own results stand without the analysis
            summary["memory"] = memory_profile.analyze(out, warmup_s=args.mem_warmup_s)
    _write(out, summary, samples)
    return summary


# ====================================================================== simclock
def run_simclock(args: argparse.Namespace) -> dict:
    import gc

    import psutil

    from sim.faults import EventLedger, LinkFaults, compare
    from sim.harness import SimStack
    from sim.invariants import InvariantMonitor
    from sim.uplink import SimUplink

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total_s = args.hours * 3600.0
    scenario = soak_scenario(total_s)
    ledger = EventLedger()
    SimUplink.LEDGER = ledger
    proc = psutil.Process()
    wall0 = time.time()
    samples: list[dict] = []
    commands = {"stop": 0, "resume": 0, "stop_confirmed": 0, "resume_confirmed": 0}
    restarts = power_cuts = backend_restarts = 0
    unconfirmed: list[dict] = []
    disruptions: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="m84-simsoak-", dir=args.work_dir) as tmp, SimStack(Path(tmp)) as stack:
        monitor = InvariantMonitor(stack)
        link = LinkFaults(stack.transport, seed=args.seed, loss=args.loss, answer_loss=args.answer_loss,
                          duplicate=args.duplicate, hold=args.hold, hold_s=8.0, clock=stack.clock)
        origin = stack.clock.monotonic()

        def boot(kind: str = "boot"):
            device = (stack.boot(scenario, transport=link, honour_backoff=True) if kind == "boot" else
                      stack.restart(scenario, transport=link, honour_backoff=True) if kind == "restart" else
                      stack.cold_restart(scenario, transport=link, honour_backoff=True))
            device.scenario_clock.origin = origin
            return device

        device = boot()
        stack.claim()
        cycle = 0

        def run_to(t_target: float) -> None:
            now = stack.clock.monotonic() - origin
            if t_target > now:
                device.advance(t_target - now)

        while True:
            base = CALIBRATION_S + cycle * CYCLE_S
            if base + CYCLE_S > total_s - DRAIN_S:
                break
            run_to(base + STOP_AT)
            if stack.protection("stop", confirm=True).status_code == 200:
                commands["stop"] += 1
            run_to(base + RESUME_AT)
            st = stack.status()
            if (st.get("protection_command") or {}).get("outcome") == "confirmed":
                commands["stop_confirmed"] += 1
            else:
                unconfirmed.append({"cycle": cycle, "action": "stop", "after_s": RESUME_AT - STOP_AT,
                                    "command": st.get("protection_command"), "stale": st.get("stale"),
                                    "state": st.get("monitoring_state")})
            if stack.protection("resume").status_code == 200:
                commands["resume"] += 1
            run_to(base + DISRUPT_AT)
            st = stack.status()
            if (st.get("protection_command") or {}).get("outcome") == "confirmed":
                commands["resume_confirmed"] += 1
            else:
                unconfirmed.append({"cycle": cycle, "action": "resume", "after_s": DISRUPT_AT - RESUME_AT,
                                    "command": st.get("protection_command"), "stale": st.get("stale"),
                                    "state": st.get("monitoring_state")})
            if cycle % 7 == 3:
                device = boot("restart")
                restarts += 1
                disruptions.append({"cycle": cycle, "kind": "restart"})
            elif cycle % 11 == 6:
                device = boot("cold")
                power_cuts += 1
                disruptions.append({"cycle": cycle, "kind": "power_cut"})
            elif cycle % 13 == 9:
                stack.restart_backend()
                link._inner = stack.transport
                backend_restarts += 1
                disruptions.append({"cycle": cycle, "kind": "backend_restart"})
            run_to(base + CYCLE_S)
            gc.collect()
            samples.append({"t": round(stack.clock.monotonic() - origin), "wall_s": round(time.time() - wall0, 1),
                            "rss_mb": round(proc.memory_info().rss / 2**20, 1), "objects": len(gc.get_objects()),
                            "threads": threading.active_count(), "queue": len(device.runtime.uplink._queue),
                            "violations": len(monitor.violations)})
            cycle += 1
        run_to(total_s)
        link.enabled = False
        link.release_held(force=True)
        device.advance(30.0)
        stored = []
        before = None
        while True:
            params: dict[str, Any] = {"limit": 1000}
            if before is not None:
                params["before_id"] = before
            page = stack.api.get(f"/v1/devices/{stack.device_id}/events", params=params,
                                 headers=stack.headers).json()["events"]
            stored += page
            if len(page) < 1000:
                break
            before = page[-1]["id"]
        integrity = compare(ledger.entries, stored)
        monitor.check_events()
        final = stack.status()
        summary = {
            "simulated": True, "mode": "simclock", "commit": os.environ.get("SDJOB_COMMIT") or _git_commit(),
            "platform": platform.platform(), "python": platform.python_version(),
            "sim_hours": args.hours, "sim_seconds_completed": round(stack.clock.monotonic() - origin),
            "wall_s": round(time.time() - wall0, 1), "cycles": cycle,
            "link": {"seed": args.seed, "loss": args.loss, "answer_loss": args.answer_loss,
                     "duplicate": args.duplicate, "hold": args.hold, **vars(link.stats)},
            "commands": commands, "device_restarts": restarts, "power_cuts": power_cuts,
            "backend_restarts": backend_restarts,
            "events": {k: (len(v) if isinstance(v, list) else v) for k, v in integrity.items()},
            "events_lost_keys": integrity["lost"][:20],
            "event_types": dict(collections.Counter(r["type"] for r in stored)),
            "invariants": monitor.summary(), "violations": [str(v) for v in monitor.violations[:20]],
            "violations_by_invariant": dict(collections.Counter(v.invariant for v in monitor.violations)),
            "violations_by_detail": dict(collections.Counter(v.detail.split(" without ")[0].split(" is not ")[0]
                                                             for v in monitor.violations)),
            "violation_cycles_vs_disruptions": _violation_cycles(monitor.violations, origin, disruptions),
            "commands_unconfirmed": unconfirmed[:20],
            "harness_growth": {"ledger_entries": len(ledger.entries), "monitor_transitions": len(monitor.transitions),
                               "monitor_user_states": len(monitor.user_states),
                               "protection_effects": len(monitor.protection_effects)},
            "user_states_seen": sorted(set(monitor.user_states)),
            "final_state": final.get("monitoring_state"),
            "rss_mb_first_last": [samples[0]["rss_mb"], samples[-1]["rss_mb"]] if samples else None,
            "rss_slope_mb_per_sim_h": slope_per_hour([(s["t"], s["rss_mb"]) for s in samples[len(samples) // 5:]]),
            "objects_first_last": [samples[0]["objects"], samples[-1]["objects"]] if samples else None,
            "threads_first_last": [samples[0]["threads"], samples[-1]["threads"]] if samples else None,
            "queue_max": monitor.max_queue,
        }
    SimUplink.LEDGER = None
    _write(out, summary, samples)
    return summary


def _violation_cycles(violations, origin: float, disruptions: list[dict]) -> dict:
    """Which cycles the violations fell in, and what disruption ended the cycle before."""
    by_cycle = collections.Counter(int((v.sim_time - origin - CALIBRATION_S) // CYCLE_S) for v in violations)
    kinds = {d["cycle"]: d["kind"] for d in disruptions}
    after = collections.Counter(kinds.get(c, kinds.get(c - 1, "none")) for c in by_cycle)
    return {"cycles_with_violations": len(by_cycle), "disruption_in_or_before_that_cycle": dict(after)}


# ====================================================================== output
def _write(out: Path, summary: dict, samples: list[dict]) -> None:
    (out / f"summary_{summary['mode']}.json").write_text(json.dumps(summary, indent=1, default=str))
    if samples:
        with open(out / f"samples_{summary['mode']}.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(samples[0]))
            writer.writeheader()
            writer.writerows(samples)
    lines = [f"# M8.4 twin soak ({summary['mode']}) -- SIMULATED", ""]
    for key, value in summary.items():
        lines.append(f"- **{key}**: `{json.dumps(value, default=str)}`")
    (out / f"summary_{summary['mode']}.md").write_text("\n".join(lines) + "\n")


def _parse(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    live = sub.add_parser("live")
    live.add_argument("--minutes", type=float, default=150.0)
    live.add_argument("--out", required=True)
    live.add_argument("--work-dir", default=None)
    live.add_argument("--fps", type=float, default=15.0)
    live.add_argument("--poll-s", type=float, default=3.0)
    live.add_argument("--sample-s", type=float, default=15.0)
    live.add_argument("--reboot-every-s", type=float, default=1800.0,
                      help="in-process device restarts (the scenario's reboots); 0 = none")
    live.add_argument("--device-kills", type=float, nargs="*", default=[0.3, 0.5],
                      help="fractions of the run at which the device process is SIGKILLed and restarted")
    live.add_argument("--backend-restarts", type=float, nargs="*", default=[0.4, 0.6],
                      help="fractions of the run at which the backend process is SIGKILLed and restarted")
    # M8.5 memory attribution: vary one factor at a time.
    live.add_argument("--viewers", type=int, default=1, help="concurrent /stream readers (0 = none)")
    live.add_argument("--no-stop-resume", dest="stop_resume", action="store_false",
                      help="never stop/resume monitoring through the API")
    live.add_argument("--quiet-timeline", action="store_true",
                      help="no scripted events or faults: an attentive driver for the whole soak")
    live.add_argument("--driver-attention-override", default=None,
                      help='JSON merged into modules.driver_attention, e.g. \'{"hands_policy": "always"}\'')
    live.add_argument("--mem-profile", choices=("off", "basic", "tracemalloc"), default="off",
                      help="sample the device process from inside (scripts/sim/memory_profile.py)")
    live.add_argument("--mem-sample-s", type=float, default=15.0)
    live.add_argument("--mem-types-s", type=float, default=300.0, help="object-type census cadence")
    live.add_argument("--mem-warmup-s", type=float, default=600.0, help="tracemalloc baseline after this")
    live.add_argument("--mem-snapshot-s", type=float, default=600.0, help="tracemalloc diff cadence")
    live.add_argument("--tracemalloc-frames", type=int, default=1)
    live.add_argument("--malloc-arena-max", type=int, default=0,
                      help="MALLOC_ARENA_MAX for the device process (glibc); 0 = unset")
    simc = sub.add_parser("simclock")
    simc.add_argument("--hours", type=float, default=6.0)
    simc.add_argument("--out", required=True)
    simc.add_argument("--work-dir", default=None)
    simc.add_argument("--seed", type=int, default=84)
    simc.add_argument("--loss", type=float, default=0.05)
    simc.add_argument("--answer-loss", type=float, default=0.05)
    simc.add_argument("--duplicate", type=float, default=0.03)
    simc.add_argument("--hold", type=float, default=0.03)
    dev = sub.add_parser("_device")
    dev.add_argument("--factory", required=True)
    dev.add_argument("--state-dir", required=True)
    dev.add_argument("--dashboard-port", type=int, required=True)
    dev.add_argument("--fps", type=float, default=15.0)
    dev.add_argument("--ledger", required=True)
    dev.add_argument("--start-wall", type=float, required=True)
    dev.add_argument("--total-s", type=float, required=True)
    dev.add_argument("--reboot-every-s", type=float, default=0.0)
    dev.add_argument("--quiet-timeline", action="store_true")
    dev.add_argument("--driver-attention-override", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    args = _parse(argv)
    if args.mode == "_device":
        return run_device(args)
    summary = run_live(args) if args.mode == "live" else run_simclock(args)
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
