"""M10.1 Priority 6: the cued stationary in-cabin session (scripts/hw_validation/cabin_session.py).

No device and no video here: the tool's plan / annotation / scoring logic on
constructed sessions, through the real evaluation.event_metrics and scenario
matrix. Pins that every preset makes a valid v2 annotation, that the scoring
counts what the priorities need (caught / latency, false alerts on legitimate
behaviour, drowsiness on an eyes-open look down as FALSE, abstention when the
device said it could not evaluate, voided cues not scored), that a --fake run is
labelled SIMULATED, and that no frame, video or token is ever written.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "hw_validation"))

import cabin_session as cs  # noqa: E402

from evaluation import annotations_v2 as av2  # noqa: E402
from evaluation import scenarios  # noqa: E402

META = {"subject": "S00", "session": "S00-t", "vehicle": "V00", "seat": "driver", "lighting": "day",
        "eyewear": "none", "occupancy": "driver_only", "mount": "dash_centre", "notes": ""}


def _session(blocks: list[str], predictions: list, *, unavailable=None, occupancy="driver_only") -> dict:
    steps, t = [], 0.0
    for s in cs.build_plan(blocks):
        steps.append({**s, "start_s": t, "end_s": t + s["s"]})
        t += s["s"]
    return {"format": cs.SESSION_FORMAT, "simulated": False, "started_utc": "2026-11-16T10:00:00+00:00",
            "duration_s": t, "meta": {**META, "occupancy": occupancy}, "steps": steps,
            "predictions": predictions, "unavailable": unavailable or {}}


def _step(session: dict, block: str, event: str, k: int = 0) -> dict:
    return [s for s in session["steps"] if s["block"] == block and s["event"] == event][k]


@pytest.mark.parametrize("preset", sorted(cs.load_cues()["presets"]))
def test_every_preset_makes_a_valid_annotation(preset):
    blocks = cs.load_cues()["presets"][preset]
    need = cs.required_session_axes(blocks)  # M14: condition presets need their own lighting / eyewear
    session = _session(blocks, [], occupancy=(need.get("occupancy") or ["driver_only"])[0])
    session["meta"].update({axis: values[0] for axis, values in need.items()})
    clips = cs.block_clips(session, set())
    assert clips
    for clip in clips:
        annotation, problems = av2.parse(clip["doc"], matrix=scenarios.load())
        assert annotation is not None, (clip["doc"]["item"], problems)


def test_a_block_whose_cell_needs_another_session_axis_is_refused():
    with pytest.raises(SystemExit, match="needs occupancy"):
        cs.block_clips(_session(["baseline"], [], occupancy="driver_plus_front_passenger"), set())


def test_every_block_names_a_matrix_cell_and_known_events():
    cues, matrix = cs.load_cues(), scenarios.load()
    vocab = set(av2.vocabulary()["events"])
    for bid, block in cues["blocks"].items():
        assert block["cell"] in matrix.cells, bid
        for step in block["steps"]:
            assert step["event"] is None or step["event"] in vocab, (bid, step)
            assert step["event"] is None or step["event"] in matrix.rules, (bid, step)


def test_the_priority_contrasts_are_all_covered():
    """M10.1 Priority 6 list -> the blocks that cover it."""
    blocks = cs.load_cues()["blocks"]
    events = {bid: {s.get("event") for s in b["steps"]} for bid, b in blocks.items()}
    attrs = {bid: [s.get("attrs") or {} for s in b["steps"]] for bid, b in blocks.items()}
    assert "passenger_phone_use" in events["passenger_present"]                       # driver vs passenger
    assert {"leaning_in"} <= {a.get("passenger") for a in attrs["passenger_present"]}
    assert {"slumped_toward_passenger"} <= {a.get("posture") for a in attrs["passenger_present"]}
    assert blocks["phone_mounted"]["cell"] == "P4_phone_mounted"                        # mounted vs handheld
    assert "phone_use" in events["phone_handheld"]
    assert {"glance_mirror_left", "glance_mirror_right", "glance_mirror_rear"} <= events["mirror_glances"]
    assert "eyes_off_road" in events["off_road_prolonged"]                            # prolonged off-road
    assert {"eyes_off_road", "long_closure"} <= events["look_down_vs_closure"]        # closed vs looking down
    assert "driver_absent" in events["driver_absent"]
    assert "displacement" in cs.load_cues()["presets"]                                # camera displacement


def test_scoring_catches_latency_false_alerts_and_look_down_drowsiness():
    s = _session(["calibrate", "mirror_glances", "look_down_vs_closure"], [])
    look = _step(s, "look_down_vs_closure", "eyes_off_road")
    closure = _step(s, "look_down_vs_closure", "long_closure")
    mirror = _step(s, "mirror_glances", "glance_mirror_left")
    s["predictions"] = [
        [look["start_s"] + 2.9, "eyes_down"],          # caught, 2.9 s from the cue
        [look["start_s"] + 3.5, "drowsiness"],         # drowsiness on an EYES-OPEN look down: false
        [closure["start_s"] + 2.4, "drowsiness"],      # caught
        [mirror["start_s"] + 1.0, "distraction"],      # an alert on a mirror check: false, legitimate behaviour
    ]
    r = cs.score_session(s, set())
    assert r["per_event"]["eyes_off_road"]["caught"] == 1
    assert r["per_event"]["eyes_off_road"]["latency_from_cue_s"]["min"] == pytest.approx(2.9)
    assert r["per_event"]["long_closure"]["caught"] == 1
    assert r["alerts"]["drowsiness"]["false_alerts"] == 1
    assert r["alerts"]["distraction"]["false_on_legitimate_behaviour"] == 1
    assert r["negatives"]["glance_mirror_left"]["alerted"] == 1
    assert r["power_checked_summary"]  # behind the power check: never a claim from one session
    assert any("reaction" in c for c in r["caveats"])


def test_a_missed_cue_while_the_device_said_unavailable_is_an_abstention():
    s = _session(["closures"], [])
    first = _step(s, "closures", "long_closure", 0)
    s["unavailable"] = {"drowsiness": [[first["start_s"] - 1, first["end_s"] + 1]]}
    r = cs.score_session(s, set())
    d = r["per_event"]["long_closure"]
    assert d["abstained"] == 1 and d["silent_misses"] == d["cued"] - 1


def test_voided_cues_are_not_scored():
    s = _session(["closures"], [])
    first = _step(s, "closures", "long_closure", 0)
    assert cs.score_session(s, {first["n"]})["per_event"]["long_closure"]["cued"] == \
        cs.score_session(s, set())["per_event"]["long_closure"]["cued"] - 1


def test_unavailable_alerts_follow_the_devices_own_policy_health():
    alerts = scenarios.load().alerts
    health = {"monitoring": {"state": "degraded", "policies": [
        {"unavailable_events": ["drowsiness", "fatigue_warning"]}]}}
    assert cs.unavailable_alerts(health, alerts) == {"drowsiness"}
    assert cs.unavailable_alerts({"monitoring": {"state": "stopped"}}, alerts) == set(alerts)
    assert cs.unavailable_alerts({"monitoring": {"state": "ready", "policies": []}}, alerts) == set()


def test_a_fake_run_is_labelled_simulated_and_writes_no_frames_or_token(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSORDYME_TOKEN", "secret-token-value")
    assert cs.main(["plan", "--blocks", "calibrate,closures", "--out", str(tmp_path)]) == 0
    assert cs.main(["run", "--plan", str(tmp_path), "--subject", "S00", "--session", "S00-f", "--vehicle", "V0",
                    "--lighting", "day", "--eyewear", "none", "--occupancy", "driver_only", "--mount",
                    "dash_centre", "--fake"]) == 0
    assert cs.main(["score", "--session", str(tmp_path)]) == 0
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["operator_sheet.md", "plan.json", "score_SIMULATED.json", "session_SIMULATED.json"]
    for p in tmp_path.iterdir():
        assert "secret-token-value" not in p.read_text()
    assert json.loads((tmp_path / "score_SIMULATED.json").read_text())["simulated"] is True


def test_run_without_a_device_is_blocked_not_a_traceback(tmp_path, capsys):
    cs.main(["plan", "--blocks", "calibrate", "--out", str(tmp_path)])
    code = cs.main(["run", "--plan", str(tmp_path), "--subject", "S0", "--session", "S0-x", "--vehicle", "V0",
                    "--lighting", "day", "--eyewear", "none", "--occupancy", "driver_only", "--mount",
                    "dash_centre", "--consent-ref", "C-0001", "--url", "http://127.0.0.1:9"])
    assert code == 2 and "BLOCKED" in capsys.readouterr().err
