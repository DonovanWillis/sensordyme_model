"""M8.5 memory profiler: a time series of where a process's memory goes.

Three parts:

  MemorySampler  an in-process sampler thread. Every `interval_s` it appends one
                 JSON line: RSS, the Python heap as tracemalloc sees it (current
                 and peak; numpy buffers included), glibc malloc arenas
                 (mallinfo2, Linux: in use vs free-in-arena vs mmapped), gc
                 generation counts and collections, thread and fd counts. Every
                 `types_every_s` it adds the live object count and the top-N
                 object types; once `warmup_s` has passed it takes a tracemalloc
                 baseline and every `snapshot_s` writes the top allocation sites
                 diffed against it. Each line records what sampling cost.

  analyze        python scripts/sim/memory_profile.py analyze RUN_DIR [--warmup-s 600]
                 Reads RUN_DIR/mem_samples.jsonl (+ mem_snapshots.jsonl) and writes
                 memory_summary.json and memory_profile.png: per-process-lifetime
                 slopes (MB/h) of RSS, Python heap and malloc in-use, before and
                 after warm-up, in halves and over the last 30 min.

  simclock       python scripts/sim/memory_profile.py simclock --hours 2 --out OUT
                 The device alone on the SimClock (the backend is in-process, as in
                 tests): the soak timeline with stop/resume each cycle, sampled once
                 per cycle. Hours of device time; deterministic; for attribution
                 (no threads: no dashboard viewers, no live uplink worker).

The live twin soak (scripts/sim/twin_soak.py live --mem-profile ...) starts a
MemorySampler inside the virtual device process via SDSOAK_MEMPROF.

Overhead (measured, docs/MEMORY_PROFILE.md): the basic sampler (no tracemalloc)
costs a few ms per sample plus ~0.1-0.3 s per object census; tracemalloc slows
every allocation and its traces themselves occupy memory (they inflate RSS),
so RSS slopes are judged on runs WITHOUT tracemalloc and allocation sites on
runs WITH it.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import ctypes
import ctypes.util
import gc
import json
import os
import statistics
import sys
import threading
import time
import tracemalloc
from pathlib import Path
from typing import Any, Optional

ENV = "SDSOAK_MEMPROF"
SAMPLES = "mem_samples.jsonl"
SNAPSHOTS = "mem_snapshots.jsonl"


# ====================================================================== probes
class _MallInfo2(ctypes.Structure):
    _fields_ = [(name, ctypes.c_size_t) for name in (
        "arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks", "fsmblks", "uordblks", "fordblks", "keepcost")]


def _mallinfo2_fn():
    if not sys.platform.startswith("linux"):
        return None
    with contextlib.suppress(OSError, AttributeError):
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        fn = libc.mallinfo2  # glibc >= 2.33 (Pi OS bookworm: 2.36)
        fn.restype = _MallInfo2
        return fn
    return None


_MALLINFO2 = _mallinfo2_fn()


def malloc_info() -> Optional[dict]:
    """glibc's view, summed over every malloc arena: what the C heap holds in use,
    what it holds FREE (fragmentation: counted in RSS, not returned to the OS)
    and what it mmapped for large blocks. None off glibc."""
    if _MALLINFO2 is None:
        return None
    m = _MALLINFO2()
    mb = 2**20
    return {"arena_mb": round(m.arena / mb, 2), "inuse_mb": round(m.uordblks / mb, 2),
            "free_mb": round(m.fordblks / mb, 2), "mmap_mb": round(m.hblkhd / mb, 2),
            "releasable_mb": round(m.keepcost / mb, 2)}


def _rss_mb() -> Optional[float]:
    with contextlib.suppress(Exception):
        import psutil

        return round(psutil.Process().memory_info().rss / 2**20, 2)
    return None


def _os_threads() -> Optional[int]:
    """Native threads (MediaPipe/ORT/OpenCV pools included), not just Python's."""
    with contextlib.suppress(Exception):
        import psutil

        return psutil.Process().num_threads()
    return None


def _fds() -> Optional[int]:
    with contextlib.suppress(OSError):
        return len(os.listdir("/proc/self/fd" if os.path.isdir("/proc/self/fd") else "/dev/fd"))
    return None


def type_census(top: int = 25) -> tuple[int, dict[str, int]]:
    """(live gc-tracked objects, the `top` most numerous types)."""
    counts = collections.Counter(type(o).__qualname__ for o in gc.get_objects())
    return sum(counts.values()), dict(counts.most_common(top))


_OWN = (__file__, tracemalloc.__file__)


# Snapshot.filter_traces() is the slow part of a diff (measured: 4.5 s per 600k
# traces at 1 frame, 17 s at 6), so the profiler's own rows are dropped from the
# result instead of filtered out of the traces.
def top_sites(snapshot: tracemalloc.Snapshot, baseline: tracemalloc.Snapshot, n: int = 25) -> list[dict]:
    """Allocation sites (file:line of the allocating frame) that grew most since `baseline`."""
    stats = [s for s in snapshot.compare_to(baseline, "lineno") if s.traceback[0].filename not in _OWN]
    rows = []
    for s in stats[:n]:
        frame = s.traceback[0]
        rows.append({"site": f"{_short(frame.filename)}:{frame.lineno}", "size_diff_kib": round(s.size_diff / 1024, 1),
                     "count_diff": s.count_diff, "size_kib": round(s.size / 1024, 1), "count": s.count})
    return rows


def top_tracebacks(snapshot: tracemalloc.Snapshot, baseline: tracemalloc.Snapshot, n: int = 8) -> list[dict]:
    """The same diff grouped by whole traceback (needs tracemalloc frames > 1)."""
    stats = [s for s in snapshot.compare_to(baseline, "traceback") if s.traceback[-1].filename not in _OWN]
    return [{"size_diff_kib": round(s.size_diff / 1024, 1), "count_diff": s.count_diff,
             "trace": [f"{_short(f.filename)}:{f.lineno}" for f in s.traceback]} for s in stats[:n]]


_REPO = str(Path(__file__).resolve().parents[2]) + os.sep


def _short(path: str) -> str:
    if path.startswith(_REPO):
        return path[len(_REPO):]
    if "site-packages/" in path:
        return path.split("site-packages/", 1)[1]
    return path


# ====================================================================== sampler
class MemorySampler(threading.Thread):
    """Appends one JSON line per sample to OUT/mem_samples.jsonl (see module doc)."""

    def __init__(self, out_dir: Path | str, *, interval_s: float = 15.0, types_every_s: float = 300.0,
                 warmup_s: float = 600.0, snapshot_s: float = 600.0, tracemalloc_frames: int = 0,
                 top_types: int = 25, label: str = "") -> None:
        super().__init__(daemon=True, name="memory-sampler")
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.interval_s = float(interval_s)
        self.types_every_s = float(types_every_s)
        self.warmup_s = float(warmup_s)
        self.snapshot_s = float(snapshot_s)
        self.frames = int(tracemalloc_frames)
        self.top_types = int(top_types)
        self.label = label
        self.stop_flag = threading.Event()
        self.t0 = time.monotonic()
        self.pid = os.getpid()
        self._marks: list[str] = []
        self._lock = threading.Lock()
        #: The post-warm-up tracemalloc baseline lives ON DISK (a temp file, never an
        #: artifact): held in memory its hundreds of thousands of trace tuples would
        #: be counted by every later object census and RSS sample.
        self._baseline: Optional[Path] = None
        self._next_types = 0.0
        self._next_snapshot = 0.0
        if self.frames > 0 and not tracemalloc.is_tracing():
            tracemalloc.start(self.frames)

    def mark(self, what: str) -> None:
        """Note an event (a runtime boot, a viewer joining) in the next sample."""
        with self._lock:
            self._marks.append(what)

    def run(self) -> None:
        while not self.stop_flag.is_set():
            with contextlib.suppress(Exception):  # a profiler never takes the device down
                self.sample()
            self.stop_flag.wait(self.interval_s)

    def stop(self) -> None:
        self.stop_flag.set()
        if self._baseline is not None:
            with contextlib.suppress(OSError):
                self._baseline.unlink()

    def sample(self) -> dict:
        started = time.perf_counter()
        t = time.monotonic() - self.t0
        with self._lock:
            marks, self._marks = self._marks, []
        row: dict[str, Any] = {"t": round(t, 1), "wall": round(time.time(), 1), "pid": self.pid,
                               "label": self.label, "rss_mb": _rss_mb(),
                               "threads": threading.active_count(), "threads_os": _os_threads(), "fds": _fds(),
                               "gc_count": list(gc.get_count()),
                               "gc_collections": [s["collections"] for s in gc.get_stats()]}
        if tracemalloc.is_tracing():
            cur, peak = tracemalloc.get_traced_memory()
            row["py_heap_mb"], row["py_heap_peak_mb"] = round(cur / 2**20, 2), round(peak / 2**20, 2)
            row["tracemalloc_overhead_mb"] = round(tracemalloc.get_tracemalloc_memory() / 2**20, 2)
        malloc = malloc_info()
        if malloc is not None:
            row["malloc"] = malloc
        if marks:
            row["marks"] = marks
        if t >= self._next_types:
            self._next_types = t + self.types_every_s
            c0 = time.perf_counter()
            row["gc_objects"], row["types"] = type_census(self.top_types)
            row["census_ms"] = round((time.perf_counter() - c0) * 1000, 1)
        if tracemalloc.is_tracing() and t >= self.warmup_s and t >= self._next_snapshot:
            self._next_snapshot = t + self.snapshot_s
            s0 = time.perf_counter()
            snap = tracemalloc.take_snapshot()
            if self._baseline is None:
                import tempfile

                fd, name = tempfile.mkstemp(prefix=f"memprof-{self.pid}-", suffix=".tracemalloc")
                os.close(fd)
                snap.dump(name)
                self._baseline = Path(name)
                record = {"t": row["t"], "pid": self.pid, "baseline": True,
                          "py_heap_mb": row.get("py_heap_mb")}
            else:
                baseline = tracemalloc.Snapshot.load(str(self._baseline))
                record = {"t": row["t"], "pid": self.pid, "baseline_t": round(self.warmup_s, 1),
                          "py_heap_mb": row.get("py_heap_mb"), "sites": top_sites(snap, baseline)}
                if self.frames > 1:
                    record["tracebacks"] = top_tracebacks(snap, baseline)
                del baseline
            del snap
            row["snapshot_ms"] = round((time.perf_counter() - s0) * 1000, 1)
            record["snapshot_ms"] = row["snapshot_ms"]
            _append(self.out / SNAPSHOTS, record)
        row["cost_ms"] = round((time.perf_counter() - started) * 1000, 1)
        _append(self.out / SAMPLES, row)
        return row


def _append(path: Path, record: dict) -> None:
    with open(path, "a") as handle:
        handle.write(json.dumps(record) + "\n")


def config_from_env() -> Optional[dict]:
    raw = os.environ.get(ENV)
    if not raw:
        return None
    return json.loads(raw)


def start_from_env(label: str = "") -> Optional[MemorySampler]:
    """Start a sampler when SDSOAK_MEMPROF holds its JSON config ({"out": ..., ...})."""
    cfg = config_from_env()
    if cfg is None:
        return None
    cfg = dict(cfg)
    sampler = MemorySampler(cfg.pop("out"), label=label, **cfg)
    sampler.start()
    return sampler


# ====================================================================== analysis
def slope_per_hour(points: list[tuple[float, float]]) -> Optional[float]:
    """Least-squares slope of (seconds, value) in value per hour."""
    if len(points) < 3:
        return None
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return None if den == 0 else round(sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / den * 3600, 3)


def _series(rows: list[dict], key: str) -> list[tuple[float, float]]:
    out = []
    for r in rows:
        value: Any = r
        for part in key.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if isinstance(value, (int, float)):
            out.append((float(r["t"]), float(value)))
    return out


def window_stats(points: list[tuple[float, float]], warmup_s: float) -> dict:
    """Slopes of one series of one process lifetime: before/after warm-up, halves, last 30 min."""
    if not points:
        return {}
    end = points[-1][0]
    after = [p for p in points if p[0] >= warmup_s]
    mid = (warmup_s + end) / 2
    out: dict[str, Any] = {
        "first": points[0][1], "last": points[-1][1], "max": max(v for _, v in points),
        "at_warmup": after[0][1] if after else None,
        "slope_per_h_warmup": slope_per_hour([p for p in points if p[0] < warmup_s]),
        "slope_per_h_after_warmup": slope_per_hour(after),
        "slope_per_h_after_first_half": slope_per_hour([p for p in after if p[0] < mid]),
        "slope_per_h_after_second_half": slope_per_hour([p for p in after if p[0] >= mid]),
        "slope_per_h_last_30min": slope_per_hour([p for p in after if p[0] >= end - 1800]) if end - warmup_s >= 1800
        else None,
        "growth_after_warmup": round(points[-1][1] - after[0][1], 2) if after else None,
        "span_s": round(end - points[0][0], 1),
    }
    return out


def load_samples(run_dir: Path) -> list[dict]:
    path = run_dir / SAMPLES
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def lifetimes(rows: list[dict]) -> list[list[dict]]:
    """Split samples by process (a device kill starts a new pid)."""
    out: dict[int, list[dict]] = {}
    for r in rows:
        out.setdefault(int(r["pid"]), []).append(r)
    return list(out.values())


SERIES = ("rss_mb", "py_heap_mb", "malloc.inuse_mb", "malloc.free_mb", "malloc.arena_mb", "malloc.mmap_mb",
          "gc_objects", "threads", "threads_os", "fds")


def analyze(run_dir: Path, warmup_s: float = 600.0, plot: bool = True) -> dict:
    rows = load_samples(run_dir)
    summary: dict[str, Any] = {"run_dir": str(run_dir), "warmup_s": warmup_s, "samples": len(rows), "lifetimes": []}
    for life in lifetimes(rows):
        entry: dict[str, Any] = {"pid": life[0]["pid"], "samples": len(life), "span_s": life[-1]["t"] - life[0]["t"],
                                 "marks": [(r["t"], m) for r in life for m in r.get("marks", [])]}
        for key in SERIES:
            pts = _series(life, key)
            if pts:
                entry[key] = window_stats(pts, warmup_s)
        census = [r for r in life if "types" in r]
        if len(census) >= 2:
            # Not the census taken in the same sample as the tracemalloc baseline:
            # profiles from before the baseline moved to disk hold it from then on.
            base = next((r for r in census if r["t"] >= warmup_s and "snapshot_ms" not in r), census[0])
            last = census[-1]
            growth = {k: last["types"].get(k, 0) - base["types"].get(k, 0)
                      for k in set(base["types"]) | set(last["types"])}
            entry["types_growth_since_warmup"] = dict(sorted(growth.items(), key=lambda kv: -abs(kv[1]))[:15])
            entry["census_ms_max"] = max(r.get("census_ms", 0) for r in census)
        costs = [r.get("cost_ms", 0.0) for r in life]
        entry["sample_cost_ms_p50"] = round(statistics.median(costs), 1) if costs else None
        entry["sample_cost_ms_max"] = max(costs) if costs else None
        summary["lifetimes"].append(entry)
    snaps_path = run_dir / SNAPSHOTS
    if snaps_path.exists():
        snaps = [json.loads(line) for line in snaps_path.read_text().splitlines() if line.strip()]
        diffs = [s for s in snaps if "sites" in s]
        if diffs:
            last = diffs[-1]
            summary["tracemalloc_last_diff"] = {"t": last["t"], "pid": last["pid"], "py_heap_mb": last.get("py_heap_mb"),
                                                "sites": last["sites"][:15],
                                                "tracebacks": last.get("tracebacks", [])[:5]}
    (run_dir / "memory_summary.json").write_text(json.dumps(summary, indent=1))
    if plot and rows:
        with contextlib.suppress(ImportError):
            _plot(run_dir, rows)
    return summary


def _plot(run_dir: Path, rows: list[dict]) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=110)
    t0 = rows[0]["wall"]
    lives = lifetimes(rows)
    for key, label, color in (("rss_mb", "RSS", "#1f5fa8"), ("malloc.inuse_mb", "malloc in use", "#c2571a"),
                              ("malloc.free_mb", "malloc free (in arenas)", "#8c6d1f"),
                              ("py_heap_mb", "Python heap (tracemalloc)", "#2a8a4a")):
        for i, life in enumerate(lives):
            pts = [((r["wall"] - t0) / 60, value) for r in life
                   for value in [_dig(r, key)] if isinstance(value, (int, float))]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=1.4,
                        label=label if i == 0 else None)
    for r in rows:
        if r.get("marks"):
            ax.axvline((r["wall"] - t0) / 60, color="#999", lw=0.6, ls=":")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_xlabel("minutes since soak start")
    ax.set_ylabel("MB")
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_title(run_dir.name, fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "memory_profile.png")
    plt.close(fig)


def _dig(row: dict, key: str) -> Any:
    value: Any = row
    for part in key.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


# ====================================================================== simclock
def run_simclock(args: argparse.Namespace) -> dict:
    """The soak timeline on the SimClock, sampled once per cycle (see module doc)."""
    import tempfile

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import twin_soak as ts

    from sim.faults import EventLedger
    from sim.harness import SimStack
    from sim.uplink import SimUplink

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    SimUplink.LEDGER = EventLedger()
    total_s = args.hours * 3600.0
    scenario = ts.soak_scenario(total_s)
    # The sampler's clock here is SIM seconds: warm-up, snapshot cadence and slopes
    # are per simulated hour of device time.
    sampler = MemorySampler(out, interval_s=1e9, types_every_s=0.0, warmup_s=args.warmup_h * 3600,
                            snapshot_s=args.snapshot_h * 3600, tracemalloc_frames=args.tracemalloc_frames,
                            label="simclock")
    wall0 = time.time()
    with tempfile.TemporaryDirectory(prefix="m85-memsim-") as tmp, SimStack(Path(tmp)) as stack:
        device = stack.boot(scenario)
        origin = stack.clock.monotonic()
        device.scenario_clock.origin = origin
        stack.claim()
        cycle = 0
        while True:
            base = ts.CALIBRATION_S + cycle * ts.CYCLE_S
            if base + ts.CYCLE_S > total_s - ts.DRAIN_S:
                break
            device.advance(base + ts.STOP_AT - (stack.clock.monotonic() - origin))
            if args.stop_resume:
                stack.protection("stop", confirm=True)
            device.advance(ts.RESUME_AT - ts.STOP_AT)
            if args.stop_resume:
                stack.protection("resume")
            device.advance(base + ts.CYCLE_S - (stack.clock.monotonic() - origin))
            cycle += 1
            gc.collect()
            sampler.t0 = time.monotonic() - (stack.clock.monotonic() - origin)  # t = sim seconds
            row = sampler.sample()
            print(f"cycle {cycle} sim_h={(stack.clock.monotonic() - origin) / 3600:.2f} wall={time.time() - wall0:.0f}s "
                  f"rss={row.get('rss_mb')} heap={row.get('py_heap_mb')} objs={row.get('gc_objects')}", flush=True)
    SimUplink.LEDGER = None
    return analyze(out, warmup_s=args.warmup_h * 3600, plot=False)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("run_dir")
    a.add_argument("--warmup-s", type=float, default=600.0)
    a.add_argument("--no-plot", action="store_true")
    s = sub.add_parser("simclock")
    s.add_argument("--hours", type=float, default=2.0)
    s.add_argument("--out", required=True)
    s.add_argument("--warmup-h", type=float, default=0.5, help="sim hours before the tracemalloc baseline")
    s.add_argument("--snapshot-h", type=float, default=0.5, help="sim hours between tracemalloc diffs")
    s.add_argument("--tracemalloc-frames", type=int, default=1)
    s.add_argument("--no-stop-resume", dest="stop_resume", action="store_false")
    args = parser.parse_args(argv)
    if args.cmd == "analyze":
        summary = analyze(Path(args.run_dir), args.warmup_s, plot=not args.no_plot)
    else:
        summary = run_simclock(args)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
