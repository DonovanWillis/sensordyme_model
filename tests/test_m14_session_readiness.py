"""M14: data-collection readiness of the cued in-cabin session tooling (cabin_session.py).

Pins that a consent-based plan exists for every condition the first sessions must
cover, that a plan run under the wrong condition (e.g. the day-only baseline at
night) is refused BEFORE anything is recorded instead of at scoring time, that
consent is recorded by reference for every adult and ids are pseudonyms, that a
session records what it never captured, and that a cue schedule stays diagnostic
(only `score --labels` produces verified ground truth). No device, no video.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "hw_validation"))

import cabin_session as cs  # noqa: E402

RUN = ["--subject", "S01", "--session", "S01-2026-11-17-a", "--vehicle", "V01", "--mount", "dash_centre"]


def _events(block: str) -> set:
    return {s.get("event") for s in cs.load_cues()["blocks"][block]["steps"]}


def test_every_required_behaviour_and_condition_has_a_block():
    blocks, presets = cs.load_cues()["blocks"], cs.load_cues()["presets"]
    assert _events("baseline") == {None}                                               # forward attention
    assert {"glance_mirror_left", "glance_mirror_rear"} <= _events("mirror_glances")   # brief normal glances
    assert "glance_cluster" in _events("cluster_glances")
    assert "eyes_off_road" in _events("off_road_prolonged")                            # prolonged looking away
    assert blocks["phone_mounted"]["cell"] == "P4_phone_mounted"                        # phone in mount
    assert "phone_use" in _events("phone_handheld")                                    # phone held by driver
    assert "passenger_phone_use" in _events("passenger_present")                       # passenger with phone
    assert "long_closure" in _events("closures")                                       # eye closure
    assert {blocks[b]["cell"] for b in ("eyewear_clear", "eyewear_sunglasses_ir_pass",  # eyewear
                                        "eyewear_sunglasses_ir_block")} == {
        "EW1_clear_glasses", "EW2_sunglasses_ir_pass", "EW3_sunglasses_ir_block"}
    assert blocks["occlusion_hand"]["cell"] == "O1_occlusion_hand"                     # partial occlusion
    assert blocks["occlusion_cap"]["cell"] == "O2_occlusion_hat"
    assert blocks["low_light_dusk"]["cell"] == "L2_dusk" and blocks["night_ir"]["cell"] == "L3_night_ir"  # NoIR
    assert {"displacement", "low_light", "night_ir", "occlusion"} <= set(presets)       # camera angle: displacement


@pytest.mark.parametrize(("preset", "axis", "value"), [("night_ir", "lighting", "night_ir"),
                                                         ("low_light", "lighting", "dusk"),
                                                         ("eyewear_clear", "eyewear", "clear"),
                                                         ("eyewear_sunglasses_ir_block", "eyewear",
                                                          "sunglasses_ir_block"),
                                                         ("core", "lighting", "day"),
                                                         ("passenger", "occupancy", "driver_plus_front_passenger")])
def test_each_preset_states_the_condition_it_must_be_run_under(preset, axis, value):
    need = cs.required_session_axes(cs.load_cues()["presets"][preset])
    assert value in need[axis]


def _plan(tmp_path: Path, preset: str) -> Path:
    assert cs.main(["plan", "--preset", preset, "--out", str(tmp_path)]) == 0
    return tmp_path


def test_the_day_baseline_at_night_is_refused_before_recording(tmp_path, capsys):
    """M10.1's doc said 'the same presets with --lighting night_ir'; that session could not be scored."""
    plan = _plan(tmp_path, "core")
    code = cs.main(["run", "--plan", str(plan), *RUN, "--lighting", "night_ir", "--eyewear", "none",
                    "--occupancy", "driver_only", "--fake"])
    assert code == 2 and "needs --lighting in ['day']" in capsys.readouterr().err
    assert not list(plan.glob("session*.json"))


def test_operator_sheet_names_the_condition_and_consent(tmp_path, capsys):
    _plan(tmp_path, "night_ir")
    sheet = (tmp_path / "operator_sheet.md").read_text()
    assert "--lighting night_ir" in sheet and "consent" in sheet and "No frames, video or audio" in sheet


@pytest.mark.parametrize("preset", ["night_ir", "low_light", "eyewear_clear", "eyewear_sunglasses_ir_pass",
                                    "eyewear_sunglasses_ir_block", "occlusion"])
def test_every_condition_preset_runs_and_scores_under_its_condition(tmp_path, preset):
    plan = _plan(tmp_path, preset)
    need = cs.required_session_axes(cs.load_cues()["presets"][preset])
    axes = {"lighting": "day", "eyewear": "none", "occupancy": "driver_only"}
    axes.update({a: v[0] for a, v in need.items() if a in axes})
    args = [f"--{a}={v}" for a, v in axes.items()]
    assert cs.main(["run", "--plan", str(plan), *RUN, *args, "--fake"]) == 0
    assert cs.main(["score", "--session", str(plan)]) == 0
    score = json.loads((plan / "score_SIMULATED.json").read_text())
    assert score["evidence_level"] == "SIMULATED" and score["accuracy_claim_permitted"] is False


def test_live_runs_need_consent_for_every_adult_and_pseudonyms(tmp_path, capsys):
    plan = _plan(tmp_path, "passenger")
    base = ["run", "--plan", str(plan), *RUN, "--lighting", "day", "--eyewear", "none",
            "--occupancy", "driver_plus_front_passenger", "--url", "http://127.0.0.1:9"]
    assert cs.main(base) == 2
    err = capsys.readouterr().err
    assert "--consent-ref" in err and "--passenger-consent-ref is required" in err
    assert cs.main([*base, "--consent-ref", "C-0001", "--passenger-consent-ref", "Jane Doe"]) == 2
    assert "not a pseudonym" in capsys.readouterr().err
    # With both consent ids the run gets as far as the (absent) device: BLOCKED, nothing recorded.
    assert cs.main([*base, "--consent-ref", "C-0001", "--passenger-consent-ref", "C-0002"]) == 2
    assert "BLOCKED" in capsys.readouterr().err
    assert not list(plan.glob("session*.json"))


def test_a_session_records_consent_by_reference_and_what_it_never_captured(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSORDYME_TOKEN", "tok-should-never-appear")
    plan = _plan(tmp_path, "core")
    assert cs.main(["run", "--plan", str(plan), *RUN, "--lighting", "day", "--eyewear", "none",
                    "--occupancy", "driver_only", "--consent-ref", "C-0001", "--fake"]) == 0
    session = json.loads((plan / "session_SIMULATED.json").read_text())
    assert session["consent"]["subject_ref"] == "C-0001"
    assert session["data_minimisation"]["frames"] is False and session["data_minimisation"]["video"] is False
    assert "diagnostics only" in session["evidence_scope"]
    assert "tok-should-never-appear" not in (plan / "session_SIMULATED.json").read_text()
    # Cue timing is never verified ground truth: without --labels the score is the cue-schedule level.
    session["simulated"] = False
    (plan / "session.json").write_text(json.dumps(session))
    assert cs.main(["score", "--session", str(plan)]) == 0
    score = json.loads((plan / "score_cue_diagnostic.json").read_text())
    assert score["evidence_level"] == cs.EVIDENCE_CUE and score["accuracy_claim_permitted"] is False
