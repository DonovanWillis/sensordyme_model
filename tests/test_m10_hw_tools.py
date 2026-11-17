"""M10 OBSERVE: the prepared hardware measurement commands (not run on a Pi yet).

  hwv_stage.py observe   records the runtime's own per-stage timings for N minutes
  hwv_stage.py arch      Arch A (production .hef) vs Arch B (+ a second .hef)
  stage_timing.py record the recording itself

Off a Pi they must refuse cleanly; with --fake they must run end to end against
fake_hailo and say SIMULATED everywhere (exit 3, never 0). `arch` must refuse
without the second .hef and must never create one.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HWV = REPO / "scripts" / "hw_validation"
sys.path.insert(0, str(HWV))

import hwv_stage as hs  # noqa: E402
import stage_timing as st  # noqa: E402

PY = sys.executable


def _run(args: list[str], tmp_path: Path, timeout: float = 240) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=tmp_path, capture_output=True, text=True, timeout=timeout, check=False,  # noqa: S603 - fixed argv
                          env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1"})


def test_arch_refuses_without_a_second_hef_and_never_creates_one(tmp_path):
    missing = tmp_path / "arch_b.hef"
    for extra in ([], ["--second-hef", str(missing)]):
        r = _run([PY, str(HWV / "hwv_stage.py"), "arch", "--fake", *extra, "--out", str(tmp_path / "out")], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == hs.BLOCKED, out
        assert "never" in out and "Traceback" not in out, out
    assert not missing.exists(), "the tool must never fabricate the second .hef"
    assert not list(tmp_path.rglob("*.hef")), "no .hef anywhere: nothing was fabricated"


def test_arch_refuses_a_second_hef_that_does_not_match_its_pin(tmp_path):
    second = tmp_path / "arch_b.hef"
    second.write_bytes(b"operator-provided placeholder")
    r = _run([PY, str(HWV / "hwv_stage.py"), "arch", "--fake", "--second-hef", str(second),
              "--second-hef-sha256", "0" * 64, "--out", str(tmp_path / "out")], tmp_path)
    assert r.returncode == hs.BLOCKED and "refusing it" in r.stdout + r.stderr


def test_arch_without_hardware_is_blocked_not_a_traceback(tmp_path):
    second = tmp_path / "arch_b.hef"
    second.write_bytes(b"operator-provided placeholder")
    r = _run([PY, str(HWV / "hwv_stage.py"), "arch", "--no-service-check", "--second-hef", str(second),
              "--out", str(tmp_path / "out")], tmp_path)
    out = r.stdout + r.stderr
    assert r.returncode == hs.BLOCKED and "Traceback" not in out, out


def test_arch_dry_run_runs_both_arms_abba_and_is_labelled_simulated(tmp_path):
    """The placeholder is the OPERATOR's file (written by this test), never the tool's."""
    second = tmp_path / "arch_b.hef"
    second.write_bytes(b"operator-provided placeholder for a dry run, not a model\n")
    out_dir = tmp_path / "out"
    r = _run([PY, str(HWV / "hwv_stage.py"), "arch", "--fake", "--second-hef", str(second), "--rounds", "1",
              "--block-seconds", "1", "--warmup", "1", "--out", str(out_dir)], tmp_path)
    out = r.stdout + r.stderr
    assert r.returncode == hs.SIMULATED, out
    assert "SIMULATED-PASS (stand-ins; not evidence)" in out and "not a hardware measurement" in out
    report = json.loads(next(out_dir.glob("arch_*.json")).read_text())
    assert report["simulated"] is True and report["exit_code"] == hs.SIMULATED
    values = report["values"]
    assert [b["arm"] for b in values["blocks"]] == ["A", "B", "B", "A"]
    assert values["platform"]["class"] == "simulated"
    assert values["conditions"]["detector_stride_frames"] >= 1 and values["conditions"]["camera_fps"] > 0
    assert values["models"]["second"]["pin"] == "UNPINNED" and len(values["models"]["second"]["sha256"]) == 64
    for block in values["blocks"]:
        has_b = (block["summary"].get("b_detector_total") or {}).get("n", 0) > 0
        assert has_b == (block["arm"] == "B"), "only Arch B runs the second detector"
        assert block["rss_mb"] is None or block["rss_mb"] > 0
    assert set(values["arms"]) == {"A", "B"} and "frame_total_p95" in values["b_minus_a"]
    (csv_path,) = out_dir.glob("arch_blocks_SIMULATED_*.csv")
    rows = list(csv.DictReader(csv_path.open()))
    assert rows and all(row["simulated"] == "True" and row["platform_class"] == "simulated" for row in rows)
    assert second.read_bytes().startswith(b"operator-provided"), "the second .hef is read, never written"


def test_observe_dry_run_records_a_simulated_performance_block(tmp_path):
    out_dir = tmp_path / "out"
    r = _run([PY, str(HWV / "hwv_stage.py"), "observe", "--fake", "--out", str(out_dir)], tmp_path)
    out = r.stdout + r.stderr
    assert r.returncode == hs.SIMULATED, out
    (rec_dir,) = out_dir.glob("observe_SIMULATED_*")
    summary = json.loads((rec_dir / "record.summary.json").read_text())
    assert summary["platform_classes"] == ["simulated"] and summary["is_pi_measurement"] is False
    assert summary["stages"]["hailo_inference"]["window_p95_ms"] is not None
    assert (rec_dir / "record.csv").exists() and (rec_dir / "record.jsonl").exists()
    assert summary["provenance"]["git_commit"] is None or len(summary["provenance"]["git_commit"]) >= 7


def test_observe_and_record_refuse_cleanly_without_a_running_service(tmp_path):
    r = _run([PY, str(HWV / "hwv_stage.py"), "observe", "--out", str(tmp_path / "out")], tmp_path)
    assert r.returncode == hs.BLOCKED and "Traceback" not in r.stdout + r.stderr
    r = _run([PY, str(HWV / "stage_timing.py"), "record", "--url", "http://127.0.0.1:9/api/health", "--minutes",
              "0.05", "--out", str(tmp_path / "rec")], tmp_path)
    out = r.stdout + r.stderr
    assert r.returncode == 2 and "BLOCKED" in out and "Traceback" not in out, out


def test_all_never_runs_the_operator_only_stages(tmp_path):
    assert set(hs._NOT_IN_ALL) == {"observe", "arch", "decoder-exp"}  # M14: decoder-exp is its own run
    assert {"observe", "arch"} <= set(hs.STAGES) and {"observe", "arch"} <= set(hs.RUNNERS)


# ---------------------------------------------------------------------- record, pure
def _payload(p95: float, platform_class: str = "pi", temp: float = 60.0) -> dict:
    stage = {"n": 256, "p50_ms": p95 / 2, "p95_ms": p95, "max_ms": p95 * 2, "mean_ms": p95 / 2, "last_ms": 1.0,
             "count": 1000}
    return {"device": {"monitoring": {"state": "ready"}},
            "performance": {"platform": {"class": platform_class}, "processed_fps": 29.5, "rss_mb": 410.0,
                            "soc_temperature_c": temp, "throttle_now": [], "throttled_raw": "0x0",
                            "stages": {"hailo_inference": stage, "face_landmarks": None}}}


def test_record_rows_flatten_the_performance_block_and_summaries_say_which_host():
    rows = [st.record_row(_payload(10.0 + i), float(i)) for i in range(5)]
    assert rows[0]["hailo_inference_p95_ms"] == 10.0 and "face_landmarks_p95_ms" not in rows[0]
    summary = st.summarize_record(rows)
    assert summary["is_pi_measurement"] is True and summary["throttled_any"] is False
    assert summary["stages"]["hailo_inference"]["worst_max_ms"] == 28.0
    mac = st.summarize_record([st.record_row(_payload(5.0, "desktop"), 0.0)])
    assert mac["is_pi_measurement"] is False, "a desktop recording is never summarised as a Pi measurement"
    with pytest.raises(ValueError, match="predates M10"):
        st.record_row({"device": {}}, 0.0)
