"""M14: `hwv_stage.py decoder-exp`, the legacy-vs-reference decoder experiment as one run.

Everything here runs on the Mac against the stand-in chip (fake_hailo) and the
existing stages; nothing is hardware evidence. Pins: the whole flow writes ONE
evidence directory with provenance, thermal samples, both decoders on the SAME
captured tensors (the product's decoders == the offline specs; the hand channel
80 is dropped; reference's multilabel keeps a phone that legacy's argmax loses),
calibration readiness without touching a threshold, pipeline fps per decoder,
and a gate that says KEEP LEGACY unless every criterion passes -- and never edits
config. A wrong .hef is refused before the chip is touched.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HWV = REPO / "scripts" / "hw_validation"
sys.path.insert(0, str(HWV))

import decoder_experiment as de  # noqa: E402
import hwv_stage as hs  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def dry_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("decoder_exp")
    config_before = _sha(REPO / "config.yaml")
    argv = [sys.executable, str(HWV / "hwv_stage.py"), "decoder-exp", "--fake", "--block-seconds", "0.5",
            "--warmup", "0", "--iterations", "10", "--out", str(out)]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=600, cwd=out, check=False,  # noqa: S603
                       env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO), "HOME": str(out)})
    found = sorted(out.glob("decoder_exp_SIMULATED_*"))
    assert len(found) == 1, f"exit {r.returncode}\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}"
    return r, found[0], config_before


def test_one_command_writes_one_evidence_directory_and_never_exits_0_when_fake(dry_run):
    r, ev, _ = dry_run
    assert r.returncode == hs.SIMULATED, r.stdout[-3000:] + r.stderr[-3000:]
    names = {p.name for p in ev.iterdir()}
    assert {"evidence.json", "DECISION.md", "SHA256SUMS", "thermal.csv", "decoders.json", "calibration.json",
            "pipeline.json", "capture", "stages"} <= names
    stages = {p.name.split("_")[0] for p in (ev / "stages").glob("*.json")}
    assert stages == {"pins", "model", "parse-hef", "capture", "parity", "latency"}
    for line in (ev / "SHA256SUMS").read_text().splitlines():  # every file hashed, and the hashes hold
        digest, rel = line.split("  ", 1)
        assert _sha(ev / rel) == digest
    assert "DECODER EXPERIMENT: SIMULATED (not evidence)" in r.stdout


def test_provenance_and_config_untouched(dry_run):
    _r, ev, config_before = dry_run
    prov = json.loads((ev / "evidence.json").read_text())["provenance"]
    assert re.fullmatch(r"[0-9a-f]{40}", prov["git_commit"] or "")
    assert prov["hef"]["sha256_before"] == prov["hef"]["sha256_after"] == prov["hef"]["pin"]
    assert "SIMULATED" in prov["hef"]["note"] and de.PRODUCTION_HEF_SHA256[:16] in prov["hef"]["note"]
    assert prov["config"]["sha256_before"] == prov["config"]["sha256_after"]
    assert prov["config"]["installed_decoder"] == "legacy"
    assert prov["config"]["camera"]["width"] == 640 and prov["config"]["gates"]["phone"] == 0.20
    assert prov["host"]["hostname"] and prov["started_utc"] and prov["finished_utc"]
    assert _sha(REPO / "config.yaml") == config_before  # the checkout's config was not written either


def test_both_decoders_on_the_same_captured_tensors(dry_run):
    _r, ev, _ = dry_run
    d = json.loads((ev / "decoders.json").read_text())
    assert d["samples"] == 6 and d["product_equals_offline_all"] is True and not d["problems"]
    assert d["decoders"]["reference"]["name"] == "device_reference"  # the product's reference, not upstream DAMO
    assert d["class_ge_80_emitted_any"] is False and d["channel80_max"] == pytest.approx(0.9)
    assert all(v["p50"] > 0 for v in d["decode_ms_this_cpu"].values())
    cal = json.loads((ev / "calibration.json").read_text())
    phone = cal["classes"]["67"]["decoders"]
    # The planted phone shares its anchor with a stronger 'remote': legacy's argmax loses it, reference keeps it.
    assert phone["legacy"]["n_at_gate"] == 0 and phone["reference"]["n_at_gate"] == 5
    assert cal["labels"]["simulated"] is True
    assert cal["classes"]["67"]["labelled"]["reference"]["at_production_gate"]["recall"] == 1.0
    assert "no threshold is proposed or changed" in cal["statement"]


def test_pipeline_thermal_and_the_gate(dry_run):
    r, ev, _ = dry_run
    pipe = json.loads((ev / "pipeline.json").read_text())
    assert pipe["conditions"]["order"] == "ABBA"
    assert all(p["decoder_seen_ok"] for p in pipe["per_decoder"].values())
    assert {b["decoder"] for b in pipe["blocks"]} == {"legacy", "reference"}
    thermal = (ev / "thermal.csv").read_text().splitlines()
    assert thermal[0].startswith("utc,elapsed_s,phase,soc_temp_c,throttled,arm_hz")
    assert {"capture", "parity", "latency", "end"} <= {line.split(",")[2] for line in thermal[1:]}
    gate = json.loads((ev / "evidence.json").read_text())["gate"]
    status = {c["id"]: c["status"] for c in gate["criteria"]}
    assert status["C10"] == "FAIL" and status["C11"] == "NOT EVALUATED" and status["C12"] == "PASS"
    assert gate["keep_legacy"] and gate["decision"].startswith("KEEP LEGACY") and gate["config_edited"] is False
    md = (ev / "DECISION.md").read_text()
    assert "KEEP LEGACY" in md and "do not prove real-world accuracy" in md and "Owner approval" in md
    assert ">>> KEEP LEGACY" in r.stdout


def test_a_wrong_hef_is_refused_before_the_chip_is_touched(tmp_path):
    install = tmp_path / "install"
    (install / "models").mkdir(parents=True)
    hef = install / "models" / "damoyolo_tinynasL20_T.hef"
    hef.write_bytes(b"not the production model")
    config = (REPO / "config.yaml").read_text()
    (install / "config.yaml").write_text(config)  # pin = production sha; the file is not it
    out = tmp_path / "out"
    code = hs.main(["decoder-exp", "--install-dir", str(install), "--out", str(out)])
    assert code == hs.BLOCKED
    (ev,) = out.glob("decoder_exp_*")
    doc = json.loads((ev / "evidence.json").read_text())
    assert "disagree" in doc["refused"] and doc["gate"]["keep_legacy"]
    assert not (ev / "capture").exists() and not (ev / "stages" / "capture").exists()
    assert (install / "config.yaml").read_text() == config


def test_production_pin_matches_every_other_pin():
    setup = re.search(r'^MODEL_SHA256="([0-9a-f]{64})"', (REPO / "setup_pi.sh").read_text(), re.M)
    assert setup and setup.group(1) == de.PRODUCTION_HEF_SHA256
    sys.path.insert(0, str(REPO))
    from core.accel import COCO80_ONLY_HEF_SHA256

    assert de.PRODUCTION_HEF_SHA256 in COCO80_ONLY_HEF_SHA256


def test_vcgencmd_parsing_and_throttle_bits():
    r = de.parse_vcgencmd("temp=48.3'C", "throttled=0x50005", "frequency(0)=2400000000")
    assert r == {"soc_temp_c": 48.3, "throttled": "0x50005", "throttle_now_bits": 5, "throttle_occurred_bits": 5,
                 "arm_hz": 2400000000}
    assert de.parse_vcgencmd(None, None, None)["soc_temp_c"] is None


def test_thermal_monitor_flags_a_new_throttle_but_not_an_old_one(tmp_path):
    readings = iter([de.parse_vcgencmd("temp=50.0'C", "throttled=0x50000", "x=1"),   # occurred before the run
                     de.parse_vcgencmd("temp=61.0'C", "throttled=0x50000", "x=1"),
                     de.parse_vcgencmd("temp=62.0'C", "throttled=0x50000", "x=1")])
    mon = de.ThermalMonitor(tmp_path / "t.csv", interval_s=3600, reader=lambda: next(readings))
    mon.start()
    mon.mark("work")
    mon.stop()
    s = mon.summary()
    assert s["available"] and not s["throttled_now_any"] and not s["throttle_newly_occurred"]
    assert s["soc_temp_c_peak"] == 62.0
    readings2 = iter([de.parse_vcgencmd("temp=50.0'C", "throttled=0x0", "x=1"),
                      de.parse_vcgencmd("temp=80.5'C", "throttled=0x80008", "x=1"),
                      de.parse_vcgencmd("temp=80.0'C", "throttled=0x80000", "x=1")])
    mon2 = de.ThermalMonitor(tmp_path / "t2.csv", interval_s=3600, reader=lambda: next(readings2))
    mon2.start()
    mon2.mark("work")
    mon2.stop()
    s2 = mon2.summary()
    assert s2["throttled_now_any"] and s2["throttle_newly_occurred"]


def _passing_inputs():
    def rep(stage, checks, values=None):
        r = hs.Report(stage, False)
        for name in checks:
            r.check(name, True)
        r.values.update(values or {})
        return r

    per = {"img": {"missed_or_moved": [], "ghosts": [], "score_delta": [0.0], "iou": [1.0]}}
    reports = {
        "pins": rep("pins", ["config pin is a sha256"]), "parse-hef": rep("parse-hef", ["six outputs"]),
        "model": rep("model", ["class map: 0=person, 67=cell phone"]), "capture": rep("capture", ["bundle complete"]),
        "parity": rep("parity", ["production decoder == offline legacy decoder on the device tensors",
                                 "[reference] every known-good image captured"],
                      {"decoders": {d: {"per_image": per, "matched_score_delta": {"p95": 0.01}} for d in de.SPECS}}),
        "latency": rep("latency", ["the detector is the Hailo path (infer and decode were timed)",
                                   "[reference decoder] predict() p95 <= the production stride budget 100 ms"]),
    }
    sha = de.PRODUCTION_HEF_SHA256
    ev = {"provenance": {"hef": {"path": "x", "sha256_before": sha, "pin": sha, "expected": sha, "sha256_after": sha},
                         "config": {"sha256_before": "a" * 64, "sha256_after": "a" * 64},
                         "host": {"board": "Raspberry Pi 5 Model B Rev 1.0"}, "hailo_simulated": False},
          "thermal": {"available": True, "throttled_now_any": False, "throttle_newly_occurred": False,
                      "throttled_values": ["0x0"], "soc_temp_c_peak": 60.0, "hailo_temp_c_peak": None},
          "decoders": {"product_equals_offline_all": True, "class_ge_80_emitted_any": False, "samples": 6},
          "pipeline": {"per_decoder": {d: {"achieved_fps_mean": 29.8, "late_fraction": 0.0, "decoder_seen_ok": True}
                                       for d in de.SPECS}}}
    return ev, reports


def _score(tmp_path: Path, name: str, false: int, caught: int) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps({"simulated": False, "evidence_level": "verified_labels",
                                "alerts": {"phone_use": {"false_alerts": false, "exposure_min": 60}},
                                "per_event": {"phone_use": {"caught": caught}}}))
    return path


def test_the_gate_keeps_legacy_without_in_cabin_evidence_even_when_the_bench_passes():
    ev, reports = _passing_inputs()
    gate = de.decision_gate(ev, reports, fake=False)
    assert gate["bench_criteria_passed"] is True
    assert gate["keep_legacy"] and gate["decision"].startswith("KEEP LEGACY") and "C11" in gate["decision"]


def test_the_gate_is_eligible_for_owner_review_only_when_everything_passes(tmp_path):
    ev, reports = _passing_inputs()
    ev["cabin_scores"] = [_score(tmp_path, "l.json", 4, 9), _score(tmp_path, "r.json", 3, 10)]
    gate = de.decision_gate(ev, reports, fake=False)
    assert not gate["keep_legacy"] and gate["decision"].startswith("ELIGIBLE FOR OWNER REVIEW")
    assert "production stays LEGACY" in gate["decision"] and gate["owner_approval_required"]
    # Any one failure -> KEEP LEGACY: more false alerts with reference, a throttle, a fake run.
    ev["cabin_scores"] = [_score(tmp_path, "l.json", 3, 9), _score(tmp_path, "r.json", 4, 10)]
    assert de.decision_gate(ev, reports, fake=False)["keep_legacy"]
    ev["cabin_scores"] = [_score(tmp_path, "l.json", 4, 9), _score(tmp_path, "r.json", 3, 10)]
    ev["thermal"]["throttle_newly_occurred"] = True
    assert de.decision_gate(ev, reports, fake=False)["keep_legacy"]
    ev["thermal"]["throttle_newly_occurred"] = False
    assert de.decision_gate(ev, reports, fake=True)["keep_legacy"]
    ev["pipeline"]["per_decoder"]["reference"]["achieved_fps_mean"] = 27.0  # < 0.97 x 29.8
    assert de.decision_gate(ev, reports, fake=False)["keep_legacy"]


def test_cabin_scores_must_be_verified_labels(tmp_path):
    bad = tmp_path / "cue.json"
    bad.write_text(json.dumps({"simulated": False, "evidence_level": "diagnostic_cue_schedule"}))
    ok, detail = de.cabin_comparison([bad, bad])
    assert ok is False and "not verified-label" in detail
    assert de.cabin_comparison(None)[0] is None


def test_labels_must_name_who_verified_them(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"format": de.LABELS_FORMAT, "images": {}}))
    with pytest.raises(SystemExit, match="verified_by"):
        de.load_labels(path)
    path.write_text(json.dumps({"format": "other", "verified_by": "A1", "method": "review"}))
    with pytest.raises(SystemExit, match="is not"):
        de.load_labels(path)


def test_decoder_exp_is_listed_but_never_part_of_all():
    assert "decoder-exp" in hs.STAGES and "decoder-exp" in hs._NOT_IN_ALL
