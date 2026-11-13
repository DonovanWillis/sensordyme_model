"""M8.5: the memory profiler the soak uses (scripts/sim/memory_profile.py).

The profiler is evidence tooling: these tests pin that its time series and its
slopes say what docs/MEMORY_PROFILE.md claims they say.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("memory_profile", REPO / "scripts/sim/memory_profile.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("memory_profile", module)
    spec.loader.exec_module(module)
    return module


mp = _load()


def test_slope_is_per_hour():
    pts = [(t, 100.0 + t / 3600 * 6.0) for t in range(0, 7200, 60)]
    assert mp.slope_per_hour(pts) == 6.0
    assert mp.slope_per_hour(pts[:2]) is None


def test_window_stats_separate_warmup_from_steady_state():
    # Grows 30 MB in the first 10 min, then flat: a plateau, not a leak.
    pts = [(float(t), min(200.0 + t / 600 * 30.0, 230.0)) for t in range(0, 5400, 15)]
    w = mp.window_stats(pts, warmup_s=600)
    assert w["slope_per_h_warmup"] > 100
    assert abs(w["slope_per_h_after_warmup"]) < 0.01
    assert w["growth_after_warmup"] == 0.0
    assert w["slope_per_h_last_30min"] == 0.0


def test_sampler_writes_a_time_series_and_a_site_diff(tmp_path):
    import tracemalloc

    was_tracing = tracemalloc.is_tracing()
    sampler = mp.MemorySampler(tmp_path, interval_s=1e9, types_every_s=0.0, warmup_s=0.0, snapshot_s=0.0,
                               tracemalloc_frames=1, top_types=5)
    try:
        sampler.mark("boot")
        first = sampler.sample()  # takes the baseline
        hoard = [bytearray(4096) for _ in range(300)]  # ~1.2 MB retained at one site
        second = sampler.sample()
    finally:
        if not was_tracing:
            tracemalloc.stop()
    assert first["marks"] == ["boot"] and "marks" not in second
    for key in ("rss_mb", "threads", "gc_count", "gc_collections", "py_heap_mb", "gc_objects", "types", "cost_ms"):
        assert key in second, key
    assert len(second["types"]) == 5
    rows = [json.loads(line) for line in (tmp_path / mp.SAMPLES).read_text().splitlines()]
    assert [r["t"] for r in rows] == [first["t"], second["t"]]
    snaps = [json.loads(line) for line in (tmp_path / mp.SNAPSHOTS).read_text().splitlines()]
    assert snaps[0]["baseline"] is True
    top = snaps[1]["sites"][0]
    assert top["site"].endswith(f"test_m85_memory_profile.py:{_hoard_line()}"), top
    assert top["size_diff_kib"] >= 1100
    del hoard


def _hoard_line() -> int:
    for number, line in enumerate(Path(__file__).read_text().splitlines(), start=1):
        if "hoard = [bytearray" in line:
            return number
    raise AssertionError


def test_analyze_splits_process_lifetimes_and_reports_slopes(tmp_path):
    rows = []
    for pid, base in ((11, 200.0), (12, 205.0)):
        for i in range(241):
            t = i * 15.0
            rows.append({"t": t, "wall": 1000.0 + (0 if pid == 11 else 4000) + t, "pid": pid,
                         "rss_mb": base + (t / 3600 * 10.0 if pid == 12 else 0.0),
                         "threads": 20, "fds": 8, "gc_count": [0, 0, 0], "gc_collections": [1, 1, 1]})
    (tmp_path / mp.SAMPLES).write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = mp.analyze(tmp_path, warmup_s=600, plot=False)
    lives = {life["pid"]: life for life in summary["lifetimes"]}
    assert lives[11]["rss_mb"]["slope_per_h_after_warmup"] == 0.0
    assert lives[12]["rss_mb"]["slope_per_h_after_warmup"] == 10.0
    assert (tmp_path / "memory_summary.json").exists()


def test_malloc_info_is_glibc_only():
    info = mp.malloc_info()
    if sys.platform.startswith("linux"):
        assert info is None or {"arena_mb", "inuse_mb", "free_mb", "mmap_mb"} <= set(info)
    else:
        assert info is None
