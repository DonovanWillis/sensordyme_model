"""M8.3 live-stream measurements against the VIRTUAL device (SIMULATED camera).

Starts the real device runtime + dashboard (`python -m sim.virtual_device
--with-backend`) and measures /stream the way the app reads it: delivered fps,
frame capture age (X-Frame-Age-Ms), inter-arrival gaps, device CPU/RSS per
client count, a slow client, reconnect time, stop/resume (video stays live,
X-Frame-Source: camera), a camera freeze (heartbeats say camera_stale), a
device reboot (stream id changes, seq resets) and a long soak (server RSS and
client buffer bounds).

Every number this prints is from the virtual device on this Mac. None of it is
a Raspberry Pi measurement; the Pi's encode cost and Wi-Fi are not modelled.

    ~/.venvs/sensordyme-m01/bin/python scripts/sim/stream_soak.py \
        --out ~/sensordyme-audit/2026-11-10/m83-evidence/stream --soak-minutes 10
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import psutil

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from sim.stream_client import StreamClient  # noqa: E402 - after the repo is on sys.path


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Device:
    """One virtual device process (with its own throwaway backend)."""

    def __init__(self, scenario: str, fps: float = 15.0) -> None:
        self.dash_port, self.backend_port = _free_port(), _free_port()
        self.log = tempfile.NamedTemporaryFile("w+", prefix=f"sim-{scenario}-", suffix=".log", delete=False)  # noqa: SIM115
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "sim.virtual_device", "--with-backend",
             "--backend", f"http://127.0.0.1:{self.backend_port}", "--scenario", scenario,
             "--dashboard-port", str(self.dash_port), "--fps", str(fps), "--log-level", "warning"],
            cwd=REPO, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        self.token = self._wait_token()
        self.ps = psutil.Process(self.proc.pid)
        self._wait_frames()
        self.started = time.monotonic()

    def _wait_token(self, timeout: float = 90) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            text = Path(self.log.name).read_text(errors="replace")
            match = re.search(r"device token: (\S+)", text)
            if match:
                return match.group(1)
            if self.proc.poll() is not None:
                raise RuntimeError(f"virtual device exited:\n{text[-3000:]}")
            time.sleep(0.2)
        raise RuntimeError("virtual device did not print its token")

    def _wait_frames(self, timeout: float = 60) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with StreamClient(self.dash_port, self.token, keep_bodies=False) as c:
                    if c.read_until(lambda cl: len(cl.jpeg_parts()) >= 2, timeout=3):
                        return
            except OSError:
                pass
            time.sleep(0.5)
        raise RuntimeError("no frames from the virtual device")

    def api(self, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.dash_port}{path}", data=data,
            headers={"X-Device-Token": self.token, "Content-Type": "application/json"},
            method="POST" if body is not None else "GET")
        with urllib.request.urlopen(req, timeout=5) as r:  # noqa: S310
            return json.loads(r.read() or b"{}")

    def sample(self, seconds: float) -> dict:
        """Device-process CPU % (one core = 100) and RSS over `seconds`."""
        self.ps.cpu_percent(None)
        rss = []
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            time.sleep(min(1.0, max(0.05, end - time.monotonic())))
            rss.append(self.ps.memory_info().rss)
        return {"cpu_pct": round(self.ps.cpu_percent(None), 1), "rss_mb": round(max(rss) / 2**20, 1)}

    def stop(self) -> None:
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGINT)
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)


def _pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    return values[min(len(values) - 1, int(p * len(values)))]


def _stats(client: StreamClient, seconds: float) -> dict:
    parts = client.jpeg_parts()
    arrivals = [p.received_at for p in parts]
    gaps = [(b - a) * 1000 for a, b in itertools.pairwise(arrivals)]
    ages = [p.int_header("X-Frame-Age-Ms") for p in parts]
    seqs = [p.int_header("X-Frame-Seq") for p in parts]
    skipped = sum(max(0, b - a - 1) for a, b in itertools.pairwise(seqs))
    return {
        "frames": len(parts),
        "fps": round(len(parts) / seconds, 2),
        "gap_ms_p50": round(_pct(gaps, 0.5), 1), "gap_ms_p95": round(_pct(gaps, 0.95), 1),
        "gap_ms_max": round(max(gaps), 1) if gaps else None,
        "age_ms_p50": _pct(ages, 0.5), "age_ms_p95": _pct(ages, 0.95), "age_ms_max": max(ages) if ages else None,
        "seq_skipped": skipped,
        "sources": sorted({p.headers.get("x-frame-source", "?") for p in parts}),
        "heartbeats": len(client.heartbeats()),
        "bytes_per_frame": round(client.raw_bytes / max(1, len(parts))),
        "client_max_buffer_bytes": client.max_buffer,
    }


def _read_clients(dev: Device, n: int, seconds: float, **kw) -> tuple[list[dict], dict]:
    clients = [StreamClient(dev.dash_port, dev.token, keep_bodies=False, **kw) for _ in range(n)]
    threads = [threading.Thread(target=c.read_for, args=(seconds,)) for c in clients]
    for t in threads:
        t.start()
    load = dev.sample(seconds - 0.5)
    for t in threads:
        t.join()
    out = [_stats(c, seconds) for c in clients]
    for c in clients:
        c.close()
    return out, load


def scenario_attentive(results: dict, seconds: float, soak_minutes: float) -> None:
    dev = Device("attentive")
    try:
        time.sleep(3)  # let the viewer from _wait_frames leave
        results["baseline_no_viewer"] = dev.sample(seconds)
        for n in (1, 3):
            stats, load = _read_clients(dev, n, seconds, heartbeat=True)
            results[f"viewers_{n}"] = {"device": load, "clients": stats}

        # a slow client (tiny receive buffer, reads 64 KB per second) beside a normal one
        slow = StreamClient(dev.dash_port, dev.token, keep_bodies=False, recv_buffer=8192)
        fast = StreamClient(dev.dash_port, dev.token, keep_bodies=False)
        rss_before = dev.ps.memory_info().rss
        t = threading.Thread(target=fast.read_for, args=(seconds,))
        t.start()
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            slow.read_for(0.05)
            time.sleep(1.0)
        t.join()
        results["slow_client"] = {
            "fast": _stats(fast, seconds), "slow": _stats(slow, seconds),
            "device_rss_delta_mb": round((dev.ps.memory_info().rss - rss_before) / 2**20, 1),
            "server_stats": {k: v for k, v in dev.api("/api/stream/stats").items() if k != "clients"},
        }
        slow.close()
        fast.close()

        # reconnect: connect -> first fresh frame, 20 times
        times = []
        for _ in range(20):
            t0 = time.monotonic()
            with StreamClient(dev.dash_port, dev.token, keep_bodies=False) as c:
                c.read_until(lambda cl: len(cl.jpeg_parts()) >= 1, timeout=10)
                if c.jpeg_parts():
                    times.append((c.jpeg_parts()[0].received_at - t0) * 1000)
            time.sleep(0.3)
        results["reconnect_to_first_frame_ms"] = {
            "n": len(times), "p50": round(_pct(times, 0.5), 1), "p95": round(_pct(times, 0.95), 1),
            "max": round(max(times), 1)}

        # monitoring stopped: video stays live, labelled camera
        dev.api("/api/protection", {"action": "stop", "confirm": True})
        time.sleep(2.5)
        with StreamClient(dev.dash_port, dev.token, keep_bodies=False, heartbeat=True) as c:
            c.read_for(5)
        results["monitoring_stopped"] = _stats(c, 5)
        dev.api("/api/protection", {"action": "resume"})
        time.sleep(2.5)
        with StreamClient(dev.dash_port, dev.token, keep_bodies=False, heartbeat=True) as c:
            c.read_for(5)
        results["monitoring_resumed"] = _stats(c, 5)

        # soak
        if soak_minutes > 0:
            soak = StreamClient(dev.dash_port, dev.token, keep_bodies=False, heartbeat=True)
            samples = []
            t_end = time.monotonic() + soak_minutes * 60
            reader_stop = threading.Event()

            def pump() -> None:
                while not reader_stop.is_set():
                    soak.read_for(1.0)
                    if len(soak.parts) > 20000:  # keep the client's own list bounded
                        soak.parts = soak.parts[-2000:]

            reader = threading.Thread(target=pump)
            reader.start()
            frames_at_start = dev.api("/api/stream/stats")["frames_sent"]
            while time.monotonic() < t_end:
                s = dev.sample(10)
                s["t_s"] = round(soak_minutes * 60 - (t_end - time.monotonic()))
                s["client_max_buffer_bytes"] = soak.max_buffer
                samples.append(s)
            reader_stop.set()
            reader.join()
            stats = dev.api("/api/stream/stats")
            soak.close()
            rss = [s["rss_mb"] for s in samples]
            results["soak"] = {
                "minutes": soak_minutes, "samples": samples,
                "rss_mb_first": rss[0], "rss_mb_last": rss[-1], "rss_mb_max": max(rss),
                "cpu_pct_median": statistics.median(s["cpu_pct"] for s in samples),
                "frames_sent": stats["frames_sent"] - frames_at_start,
                "fps_avg": round((stats["frames_sent"] - frames_at_start) / (soak_minutes * 60), 2),
                "client_max_buffer_bytes": soak.max_buffer,
                "client_send_timeouts": stats["client_send_timeouts"],
            }
    finally:
        dev.stop()


def scenario_camera_freeze(results: dict) -> None:
    dev = Device("camera_freeze")
    try:
        # the freeze runs from scenario t=17 s to t=37 s
        with StreamClient(dev.dash_port, dev.token, keep_bodies=False, heartbeat=True) as c:
            c.read_for(max(0.0, 45 - (time.monotonic() - dev.started)))
        parts = c.parts
        statuses = sorted({p.headers.get("x-stream-status") for p in parts if p.is_heartbeat})
        jpeg_times = [p.received_at for p in parts if p.is_jpeg]
        gaps = [b - a for a, b in itertools.pairwise(jpeg_times)]
        longest = max(gaps) if gaps else None
        first_beat = next((p for p in parts if p.is_heartbeat and p.headers.get("x-stream-status") == "camera_stale"), None)
        results["camera_freeze"] = {
            "heartbeat_statuses": statuses,
            "heartbeats": len([p for p in parts if p.is_heartbeat]),
            "longest_frame_gap_s": round(longest, 2) if longest else None,
            "first_camera_stale_age_ms": first_beat.int_header("X-Camera-Age-Ms") if first_beat else None,
            "frames": len(jpeg_times),
            "connection_survived": not c.closed_by_peer,
        }
    finally:
        dev.stop()


def scenario_reboot(results: dict) -> None:
    dev = Device("reboot")
    try:
        c = StreamClient(dev.dash_port, dev.token, keep_bodies=False, heartbeat=True)
        first_id = c.headers.get("x-stream-id")
        c.read_until(lambda cl: cl.closed_by_peer, timeout=80)
        dropped_at = time.monotonic()
        last_seq = c.jpeg_parts()[-1].int_header("X-Frame-Seq") if c.jpeg_parts() else None
        first_id = first_id or c.headers.get("x-stream-id")
        c.close()
        # reconnect on the app's backoff (0.5, 1, 2, 4, 8 s)
        attempt, delay = 0, 0.5
        new = None
        while time.monotonic() - dropped_at < 120:
            attempt += 1
            try:
                new = StreamClient(dev.dash_port, dev.token, keep_bodies=False, heartbeat=True)
                if new.read_until(lambda cl: len(cl.jpeg_parts()) >= 1, timeout=5):
                    break
                new.close()
            except OSError:
                pass
            time.sleep(delay)
            delay = min(8.0, delay * 2)
        assert new is not None
        results["device_reboot"] = {
            "stream_id_before": first_id, "stream_id_after": new.headers.get("x-stream-id"),
            "last_seq_before": last_seq,
            "first_seq_after": new.jpeg_parts()[0].int_header("X-Frame-Seq") if new.jpeg_parts() else None,
            "drop_to_first_frame_s": round(new.jpeg_parts()[0].received_at - dropped_at, 2) if new.jpeg_parts() else None,
            "attempts": attempt,
        }
        new.close()
    finally:
        dev.stop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--soak-minutes", type=float, default=10.0)
    parser.add_argument("--only", default="attentive,camera_freeze,reboot")
    args = parser.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {
        "label": "SIMULATOR / virtual device on a Mac (sim.virtual_device, 640x480 @ 15 fps scripted camera). "
                 "NOT Raspberry Pi numbers.",
        "host": {"cpu_count": psutil.cpu_count(), "platform": sys.platform},
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    only = set(args.only.split(","))
    if "attentive" in only:
        scenario_attentive(results, args.seconds, args.soak_minutes)
    if "camera_freeze" in only:
        scenario_camera_freeze(results)
    if "reboot" in only:
        scenario_reboot(results)
    results["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    path = out / "stream_measurements.json"
    path.write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k != "soak"}, indent=2))
    if "soak" in results:
        print(json.dumps({k: v for k, v in results["soak"].items() if k != "samples"}, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
