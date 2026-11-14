"""M11: a cue schedule is NOT ground truth (scripts/hw_validation/cabin_session.py).

Pins the two evidence levels: cue-schedule scores are diagnostics with every claim withheld; accuracy
needs independently verified labels (named labeller who is not the operator, a real labelling method,
a synchronised clock, a consent reference), and even then one session never permits a claim. Shows the
difference on a session where the participant did not perform one cue and did something un-cued.
"""

from __future__ import annotations

import copy
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "hw_validation"))

import cabin_session as cs  # noqa: E402

T0 = datetime(2026, 11, 16, 10, 0, 0, tzinfo=UTC)
META = {"subject": "S00", "session": "S00-t", "vehicle": "V00", "seat": "driver", "lighting": "day",
        "eyewear": "none", "occupancy": "driver_only", "mount": "dash_centre", "notes": ""}


def _session(blocks: list[str], predictions: list) -> dict:
    steps, t = [], 0.0
    for s in cs.build_plan(blocks):
        steps.append({**s, "start_s": t, "end_s": t + s["s"]})
        t += s["s"]
    return {"format": cs.SESSION_FORMAT, "simulated": False, "started_utc": T0.isoformat(), "duration_s": t,
            "meta": dict(META), "steps": steps, "predictions": predictions, "unavailable": {}}


def _utc(seconds: float) -> str:
    return (T0 + timedelta(seconds=seconds)).isoformat()


PROVENANCE = {"annotator": "O1", "method": "live_observer", "independent_of_operator": True,
              "blinded_to_device_output": False,
              "clock": {"method": "same_host", "synchronized": True, "offset_s": 0.0, "uncertainty_s": 0.15},
              "consent_ref": "C-2026-11-16-S00"}


def _labels(events: list[tuple[str, float, float]], **prov) -> dict:
    return {"format": cs.LABELS_FORMAT, "session": "S00-t", "provenance": {**PROVENANCE, **prov},
            "labels": [{"type": t, "start_utc": _utc(a), "end_utc": _utc(b)} for t, a, b in events],
            "uncertain": []}


def _closures():
    """6 cued closures; the participant skips the 2nd, and closes their eyes un-cued during a rest step."""
    s = _session(["closures"], [])
    cued = [x for x in s["steps"] if x["event"] == "long_closure"]
    rest = next(x for x in s["steps"] if x["event"] is None and x["start_s"] > cued[2]["end_s"])
    observed = [("long_closure", c["start_s"] + 0.4, c["end_s"] + 0.6) for i, c in enumerate(cued) if i != 1]
    unscripted = ("long_closure", rest["start_s"] + 3.0, rest["start_s"] + 6.0)
    # the device alerts on every closure that HAPPENED, ~2.3 s into it, and on nothing else
    s["predictions"] = sorted([[round(a + 2.1, 3), "drowsiness"] for _t, a, _b in [*observed, unscripted]])
    return s, cued, observed, unscripted


def test_cue_schedule_score_is_diagnostic_with_every_claim_withheld():
    s, _cued, _obs, _un = _closures()
    r = cs.score_session(s, set())
    assert r["evidence_level"] == cs.EVIDENCE_CUE
    assert r["accuracy_claim_permitted"] is False
    assert r["label_provenance"] is None and r["cue_label_agreement"] is None
    assert any("not proof" in c for c in r["caveats"])
    assert _claims(r["power_checked_summary"]) and all(c is None for c in _claims(r["power_checked_summary"]))


def _claims(node) -> list:
    if isinstance(node, dict):
        return ([node["claim"]] if "claim" in node else []) + [c for v in node.values() for c in _claims(v)]
    if isinstance(node, list):
        return [c for v in node for c in _claims(v)]
    return []


def test_withholding_nulls_a_claim_that_passed_the_power_check():
    summary = {"detection": {"long_closure": {"status": "ok", "claim": {"recall": 0.97}}}}
    out = cs._withhold_claims(summary, "cue-schedule labels")
    assert out["detection"]["long_closure"]["claim"] is None
    assert out["detection"]["long_closure"]["claim_withheld"] == "cue-schedule labels"
    assert cs._power_ok(summary) is True and cs._power_ok({"detection": {}}) is False


def test_the_cue_schedule_misjudges_what_happened_and_verified_labels_do_not():
    s, cued, observed, _un = _closures()
    cue_mode = cs.score_session(s, set())["per_event"]["long_closure"]
    # cue mode: the skipped cue looks like a SILENT MISS, the un-cued closure's alert looks FALSE
    assert cue_mode["cued"] == len(cued)
    assert cue_mode["silent_misses"] >= 1
    labels, problems = cs.load_labels(_labels([*observed, _un]), s)
    assert problems == [] and labels is not None
    r = cs.score_session(s, set(), labels)
    d = r["per_event"]["long_closure"]
    assert r["evidence_level"] == cs.EVIDENCE_VERIFIED
    assert d["labelled"] == len(observed) + 1
    assert d["silent_misses"] == 0 and d["caught"] == d["labelled"]
    assert r["alerts"]["drowsiness"]["false_alerts"] == 0
    ag = r["cue_label_agreement"]
    assert ag["cued_behaviours"] == len(cued) and ag["performed_as_cued"] == len(cued) - 1
    assert ag["not_performed_cues"] == [cued[1]["n"]]
    assert ag["unscripted_by_type"] == {"long_closure": 1}
    assert r["label_provenance"]["annotator"] == "O1"
    assert any("not blinded" in c for c in r["caveats"])
    # one session never permits a claim, verified or not
    assert r["accuracy_claim_permitted"] is False


@pytest.mark.parametrize(("change", "fragment"), [
    ({"annotator": "cue-schedule"}, "independent labeller"),
    ({"annotator": ""}, "independent labeller"),
    ({"method": "cue_schedule"}, "not a label"),
    ({"independent_of_operator": False}, "cannot label"),
    ({"consent_ref": ""}, "consent"),
    ({"clock": {"method": "ntp", "synchronized": False, "offset_s": 0, "uncertainty_s": 0.2}}, "one clock"),
    ({"clock": {"method": "same_host", "synchronized": True, "offset_s": 0, "uncertainty_s": 2.0}}, "uncertainty"),
    ({"clock": {"method": "wristwatch", "synchronized": True, "offset_s": 0, "uncertainty_s": 0.1}}, "clock.method"),
])
def test_labels_without_verifiable_provenance_are_refused(change, fragment):
    s, _cued, observed, _un = _closures()
    labels, problems = cs.load_labels(_labels(observed, **change), s)
    assert labels is None
    assert any(fragment in p for p in problems), problems


def test_bad_label_contents_are_refused():
    s, *_ = _closures()
    doc = _labels([("not_a_behaviour", 1, 2), ("long_closure", 5, 4)])
    labels, problems = cs.load_labels(doc, s)
    assert labels is None
    assert any("not an annotation v2 event type" in p for p in problems)
    assert any("ends before it starts" in p for p in problems)
    wrong_session = copy.deepcopy(_labels([("long_closure", 1, 4)]))
    wrong_session["session"] = "S99-x"
    assert any("this session is" in p for p in cs.load_labels(wrong_session, s)[1])
    assert any("empty" in p for p in cs.load_labels(_labels([]), s)[1])


def test_clock_offset_moves_labels_onto_the_device_clock():
    s, *_ = _closures()
    doc = _labels([("long_closure", 10.0, 13.0)])
    doc["provenance"]["clock"] = {"method": "ntp", "synchronized": True, "offset_s": 0.5, "uncertainty_s": 0.2}
    labels, _ = cs.load_labels(doc, s)
    assert labels is not None
    assert labels["events"][0]["start_s"] == pytest.approx(9.5)


def test_void_cannot_be_mixed_with_verified_labels():
    s, _cued, observed, _un = _closures()
    labels, _ = cs.load_labels(_labels(observed), s)
    with pytest.raises(SystemExit, match="--void"):
        cs.score_session(s, {3}, labels)


def test_observe_writes_a_label_file_that_scoring_accepts(tmp_path, monkeypatch):
    s, cued, *_ = _closures()
    (tmp_path / "session.json").write_text(json.dumps(s))
    first = cued[0]
    clock = iter([T0.timestamp() + first["start_s"] + 0.4, T0.timestamp() + first["end_s"] + 0.3,
                  T0.timestamp() + 50.0, T0.timestamp() + 52.0, T0.timestamp() + 60.0, T0.timestamp() + 61.0,
                  T0.timestamp() + 62.0])
    args = cs.argparse.Namespace(session=tmp_path, session_id=None, annotator="O1", method="live_observer",
                                 clock="same_host", clock_uncertainty_s=None, consent_ref="C-1",
                                 independent=True, blinded=False)
    assert cs.observe(args, keys=iter(["c", "c", "u", "u", "z", "q"]), now=lambda: next(clock)) == 0
    doc = json.loads((tmp_path / "labels_O1.json").read_text())
    assert doc["session"] == "S00-t" and doc["provenance"]["tool"] == "cabin_session.py observe"
    assert [x["type"] for x in doc["labels"]] == ["long_closure"]
    assert len(doc["uncertain"]) == 1
    labels, problems = cs.load_labels(doc, s)
    assert problems == [] and labels is not None
    assert labels["events"][0]["start_s"] == pytest.approx(first["start_s"] + 0.4, abs=1e-3)


def test_observe_without_independence_is_written_but_refused_at_scoring(tmp_path):
    s, *_ = _closures()
    (tmp_path / "session.json").write_text(json.dumps(s))
    clock = iter([T0.timestamp() + 1, T0.timestamp() + 4, T0.timestamp() + 5, T0.timestamp() + 6])
    args = cs.argparse.Namespace(session=tmp_path, session_id=None, annotator="OP", method="live_observer",
                                 clock="same_host", clock_uncertainty_s=None, consent_ref="C-1",
                                 independent=False, blinded=False)
    cs.observe(args, keys=iter(["c", "c", "q"]), now=lambda: next(clock))
    problems = cs.load_labels(json.loads((tmp_path / "labels_OP.json").read_text()), s)[1]
    assert any("cannot label" in p for p in problems)


def test_score_cli_names_the_evidence_level_in_the_file(tmp_path, capsys):
    s, _cued, observed, _un = _closures()
    (tmp_path / "session.json").write_text(json.dumps(s))
    assert cs.main(["score", "--session", str(tmp_path)]) == 0
    assert (tmp_path / "score_cue_diagnostic.json").exists()
    assert "CUE-SCHEDULE DIAGNOSTICS" in capsys.readouterr().out
    (tmp_path / "labels_O1.json").write_text(json.dumps(_labels(observed)))
    assert cs.main(["score", "--session", str(tmp_path), "--labels", str(tmp_path / "labels_O1.json")]) == 0
    assert json.loads((tmp_path / "score_verified_O1.json").read_text())["evidence_level"] == cs.EVIDENCE_VERIFIED
    bad = _labels(observed, consent_ref="")
    (tmp_path / "labels_bad.json").write_text(json.dumps(bad))
    assert cs.main(["score", "--session", str(tmp_path), "--labels", str(tmp_path / "labels_bad.json")]) == 2
    assert "BLOCKED" in capsys.readouterr().err
