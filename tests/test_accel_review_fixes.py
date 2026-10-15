"""Regression tests for the 2026-10 review of the accelerator / detector / VLM /
comparison-module / benchmark code. One section per finding (F1..F16, F2 is a
documented open question with no code change). Each test reproduces the defect
as it was found and fails on the code before its fix.
"""

from __future__ import annotations

import json
import threading
import time
import types
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import test_hailo_lifecycle as hl
import yaml
from test_hailo_lifecycle import hailo  # noqa: F401 - pytest fixture: a fake HAT + real accel code

from core import accel
from core.vlm_judge import VLMConfirmTracker, VLMJudge, VLMVerdict, vlm_is_local

REPO = Path(__file__).resolve().parent.parent
_FRAME = np.zeros((48, 64, 3), dtype=np.uint8)


# ============================================================ F1: CI accuracy gate
def test_ci_runs_the_accuracy_harness_directly_so_exit_1_can_fail_the_job():
    """`make` exits 2 whenever a recipe fails, whatever the recipe returned, so
    `make accuracy-gate` turned the harness's regression code (1) into the
    workflow's "could not run, not a failure" branch (2): the gate could never
    go red. The step must run the harness itself."""
    workflow = yaml.safe_load((REPO / ".github" / "workflows" / "ci.yml").read_text())
    steps = workflow["jobs"]["accuracy-gate"]["steps"]
    gate = next(s for s in steps if "code=$?" in s.get("run", ""))  # the step that branches on the exit code
    script = gate["run"]
    assert "make accuracy-gate" not in script
    assert "benchmarks/accuracy_state_farm.py" in script
    assert "code=$?" in script and '1)' in script and "exit 1" in script

    # Same arguments as the Makefile target humans run, so the two cannot drift.
    makefile = (REPO / "Makefile").read_text()
    recipe = makefile[makefile.index("accuracy-gate:"):]
    recipe = recipe[: recipe.index("\n\n")]
    for arg in ("--per-class 100", "--compare benchmarks/results/baseline_phase1.json"):
        assert arg in recipe and arg in " ".join(script.replace("\\\n", " ").split())


# ============================================================ F3: gate threshold
def _per_class(pos_scores, neg_scores):
    def row(scores):
        return {"n": len(scores), "face": 0, "person": len(scores), "person_conf": [0.9],
                "phone_scores": list(scores), "multi_person": 0}
    pc = {c: row(pos_scores) for c in ("c1", "c2", "c3", "c4")}
    pc.update({c: row(neg_scores) for c in ("c0", "c9")})
    return {"per_class": pc, "skipped": 0, "elapsed_s": 0, "backend": "cpu"}


def _write_baseline(tmp_path, summary, shipped_conf, backend=None):
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"summary": summary, "shipped_conf": shipped_conf, "backend": backend}))
    return path


def test_phone_recall_is_compared_at_the_baselines_threshold_not_the_current_one(tmp_path):
    """Baseline shipped 0.35, config now ships 0.20. A model whose recall AT 0.35
    fell 0.70 -> 0.50 used to pass, because 0.20-recall (0.72) was compared with
    0.35-recall (0.70)."""
    from benchmarks.accuracy_state_farm import DEFAULT_SWEEP, EXIT_REGRESSION, compare, summarise

    sweep = tuple(sorted(set(DEFAULT_SWEEP) | {0.20, 0.35}))
    neg = [0.0] * 100
    base = summarise(_per_class([0.45] * 70 + [0.0] * 30, neg), sweep, 0.35)
    now = summarise(_per_class([0.45] * 50 + [0.25] * 22 + [0.0] * 28, neg), sweep, 0.20)
    now["backend"] = "cpu"

    assert compare(now, _write_baseline(tmp_path, base, 0.35), 0.03) == EXIT_REGRESSION


def test_an_unchanged_model_passes_its_own_baseline(tmp_path):
    from benchmarks.accuracy_state_farm import DEFAULT_SWEEP, EXIT_OK, compare, summarise

    sweep = tuple(sorted(set(DEFAULT_SWEEP) | {0.20}))
    scored = _per_class([0.45] * 70 + [0.25] * 10 + [0.0] * 20, [0.0] * 100)
    summary = summarise(scored, sweep, 0.20)
    summary["backend"] = "cpu"
    assert compare(summary, _write_baseline(tmp_path, summary, 0.35, backend="cpu"), 0.03) == EXIT_OK


def test_a_baseline_from_a_different_backend_cannot_be_compared(tmp_path):
    from benchmarks.accuracy_state_farm import DEFAULT_SWEEP, EXIT_CANNOT_RUN, compare, summarise

    summary = summarise(_per_class([0.45] * 10, [0.0] * 10), DEFAULT_SWEEP, 0.35)
    summary["backend"] = "hailo"
    assert compare(summary, _write_baseline(tmp_path, summary, 0.35, backend="cpu"), 0.03) == EXIT_CANNOT_RUN


def test_main_sweeps_the_baselines_threshold_even_when_config_moved_off_it(monkeypatch, tmp_path):
    """compare() reads recall at the baseline's conf, so main() must put that
    conf on this run's curve — including one that is not in DEFAULT_SWEEP."""
    from benchmarks import accuracy_state_farm as asf

    assert 0.33 not in asf.DEFAULT_SWEEP
    baseline = tmp_path / "b.json"
    baseline.write_text(json.dumps({"shipped_conf": 0.33, "summary": {}}))
    seen = {}

    def fake_score(sample, sweep, quiet=False, backend="auto"):
        seen["sweep"] = sweep
        return {"backend": "cpu", "detector": "x", "per_class": {}, "skipped": len(sample), "elapsed_s": 0.0}

    monkeypatch.setattr(asf, "load_sample", lambda *a, **k: [{"subject": "p", "classname": "c0", "img": "i"}])
    monkeypatch.setattr(asf, "score_sample", fake_score)
    asf.main(["--quiet", "--shipped-conf", "0.20", "--compare", str(baseline)])
    assert 0.33 in seen["sweep"] and 0.20 in seen["sweep"]

    committed = json.loads((REPO / "benchmarks" / "results" / "baseline_phase1.json").read_text())
    assert asf.baseline_conf(committed) == pytest.approx(0.35)


# ============================================================ F4 + F15: run_benchmarks
@pytest.fixture
def _restore_accel():
    yield
    accel.configure({}, info=accel.AcceleratorInfo(False, False, "none", "no HAT"))


@pytest.mark.usefixtures("_restore_accel")
def test_run_benchmarks_installs_the_accelerator_config_before_building_modules(tmp_path, monkeypatch):
    """It never called accel.configure(), so maybe_hailo_yolo() saw an empty
    config and the Hailo path could not be benchmarked at all."""
    from benchmarks import run_benchmarks as rb

    accel_block = {"enabled": True, "yolo_hef": "models/damoyolo_tinynasL20_T.hef", "yolo_hef_sha256": "ab" * 32}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"accelerator": accel_block, "modules": {"fake": {}}}))
    seen = {}

    def fake_benchmark(module_name, config, duration_s, video_path, profile="pi"):
        seen["cfg"] = dict(accel._accel_cfg)
        return {"module": module_name, "fps": 1.0, "frame_count": 1,
                "latency_ms": {"mean": 1.0, "p50": 1.0, "p95": 1.0, "p99": 1.0},
                "detector": {"name": "x.hef", "sha256": None, "backend": "hailo"},
                "object_detection_error": None}

    monkeypatch.setattr(rb, "benchmark_module", fake_benchmark)
    monkeypatch.setattr(rb, "RESULTS_DIR", tmp_path / "results")
    assert rb.main(["--config", str(config_path), "--modules", "fake", "--minutes", "0.001"]) == 0
    assert seen["cfg"] == accel_block


@pytest.mark.usefixtures("_restore_accel")
def test_run_benchmarks_fails_when_detection_was_disabled_during_the_run(tmp_path, monkeypatch):
    """A device with no usable detector used to produce Face-Mesh-only numbers
    labelled as the full production module."""
    from benchmarks import run_benchmarks as rb

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"modules": {"fake": {}}}))
    monkeypatch.setattr(rb, "benchmark_module", lambda *a, **k: {
        "module": "fake", "fps": 30.0, "frame_count": 10,
        "latency_ms": {"mean": 1.0, "p50": 1.0, "p95": 1.0, "p99": 1.0},
        "detector": None, "object_detection_error": "no object-detection backend available"})
    monkeypatch.setattr(rb, "RESULTS_DIR", tmp_path / "results")
    assert rb.main(["--config", str(config_path), "--modules", "fake", "--minutes", "0.001"]) == 1


class _SlowFirstFrameModule:
    """First process() call stands in for a lazy model load."""

    instances: ClassVar[list] = []

    def __init__(self, config):
        self.calls = 0
        self.closed = 0
        self._object_detection_error = None
        _SlowFirstFrameModule.instances.append(self)

    def process(self, frame):
        self.calls += 1
        if self.calls == 1:
            time.sleep(0.3)
        return []

    def close(self):
        self.closed += 1


def test_benchmark_excludes_warmup_frames_and_closes_the_module(monkeypatch):
    """The first-frame model load used to land in the latency samples."""
    from benchmarks import run_benchmarks as rb

    _SlowFirstFrameModule.instances.clear()
    monkeypatch.setattr(rb, "get_module_class", lambda name: _SlowFirstFrameModule)
    monkeypatch.setattr(rb, "_frame_source", lambda video, cam: iter([_FRAME] * 20))
    config = {"modules": {"driver_attention": {"yolo_every_n_frames": 3}}}

    result = rb.benchmark_module("driver_attention", config, duration_s=60.0, video_path="x")

    module = _SlowFirstFrameModule.instances[0]
    warmup = rb.warmup_frame_count(config["modules"]["driver_attention"])
    assert warmup == 4
    assert result["frame_count"] == 20 - warmup
    assert result["latency_ms"]["p99"] < 100.0, "the 300 ms first-frame load was timed"
    assert result["warmup_frames"] == warmup
    assert module.closed == 1
    assert result["object_detection_error"] is None


# ============================================================ F5: VLM latch
def _max_module_with_judge(judge):
    from modules.driver_attention_max import DriverAttentionMaxModule as M

    mod = M.__new__(M)
    mod.name = M.name
    mod._vlm = judge
    mod._vlm_confirm = VLMConfirmTracker(0.7, 2)
    mod._vlm_last_seq = 0
    for attr in ("_drowsy_active", "_fatigue_critical_active", "_phone_use_distracted_active",
                 "_eyes_down_active", "_off_road_active", "_fatigue_active", "_excessive_glancing_active"):
        setattr(mod, attr, False)
    mod._calibrator = types.SimpleNamespace(is_calibrated=True)
    mod._blink_rate_tracker = types.SimpleNamespace(is_active=False)
    return mod


def _confirm_phone_use(judge, mod, at):
    for seq in (1, 2):
        judge._verdict = VLMVerdict("phone_use", 0.9, "phone in hand", seq, at, 1.0)
        mod._poll_vlm([])
    assert mod._vlm_confirm.active_state == "phone_use"


def test_a_confirmed_vlm_state_clears_when_the_judge_disables_itself():
    """Two failures disabled the judge; no verdict ever arrived again, so the
    confirmed state — and is_alert_active() — stayed latched for the session."""
    judge = VLMJudge({"provider": "ollama", "max_consecutive_failures": 2},
                     post=lambda *a, **k: None, autostart=False)
    mod = _max_module_with_judge(judge)
    _confirm_phone_use(judge, mod, time.monotonic())

    judge._step([_FRAME])  # post() returns None -> AttributeError -> failure 1
    judge._step([_FRAME])  # failure 2 -> disabled
    assert judge.disabled_reason is not None
    mod._poll_vlm([])

    assert mod.is_alert_active() is False
    assert mod._resolve_state() == "ATTENTIVE"


def test_a_confirmed_vlm_state_expires_when_verdicts_stop_arriving(monkeypatch):
    import core.vlm_judge as vlm_module

    now = [1000.0]
    monkeypatch.setattr(vlm_module.time, "monotonic", lambda: now[0])
    judge = VLMJudge({"provider": "ollama", "interval_s": 8.0, "burst_frames": 3, "burst_spacing_s": 0.7},
                     post=lambda *a, **k: None, autostart=False)
    mod = _max_module_with_judge(judge)
    _confirm_phone_use(judge, mod, now[0])

    lifetime = judge.verdict_lifetime_s(judge.latest_verdict())
    now[0] += lifetime * 0.9  # one slow cycle: still current
    mod._poll_vlm([])
    assert mod.is_alert_active() is True
    now[0] += lifetime  # the judge has gone quiet (network down, rate-limit backoff)
    mod._poll_vlm([])
    assert mod.is_alert_active() is False


# ============================================================ F6: privacy notice
@pytest.mark.parametrize(
    ("config", "local"),
    [
        ({"provider": "ollama"}, True),
        ({"provider": "ollama", "base_url": "http://127.0.0.1:11434"}, True),
        ({"provider": "ollama", "base_url": "http://gpu-box.lan:11434"}, False),
        ({"provider": "openai_compatible", "base_url": "http://localhost:1234/v1"}, True),
        ({"provider": "openai_compatible", "base_url": "http://[::1]:8000/v1"}, True),
        ({"provider": "openai_compatible", "base_url": "https://localhost.attacker.example/v1"}, False),
        ({"provider": "openai_compatible", "base_url": "https://api.example.com/v1?via=localhost"}, False),
        ({"provider": "openai_compatible", "base_url": "https://127.0.0.1.example.com/v1"}, False),
        ({"provider": "anthropic"}, False),
    ],
)
def test_vlm_is_local_is_decided_by_the_host_frames_are_sent_to(config, local):
    """Remote Ollama and lookalike hosts printed "frames stay ON-DEVICE"."""
    assert vlm_is_local(config) is local


# ============================================================ F7 + F16: DashSentinel
def test_dashsentinel_resets_feature_history_when_the_face_is_lost(tmp_path):
    """Their app.py calls extractor.reset() on a no-face frame; the adapter did
    not, so closure/look-away history leaked into the next face."""
    from modules.base import get_module_class

    module = get_module_class("driver_attention_dashsentinel")(
        {"profile_path": str(tmp_path / "profile.json"), "save_profile_every_seconds": 9999}
    )
    ex = module._extractor
    ex.closed_frames, ex.look_away_frames, ex.bad_pose_frames = 30, 18, 24
    ex.yaw_history.extend([0.9] * 30)

    module.process(np.zeros((480, 640, 3), dtype=np.uint8))  # FaceMesh finds no face

    assert (ex.closed_frames, ex.look_away_frames, ex.bad_pose_frames) == (0, 0, 0)
    assert len(ex.yaw_history) == 0
    module.close()


def test_dashsentinel_close_saves_the_profile_and_is_idempotent(tmp_path):
    from modules.base import get_module_class

    profile = tmp_path / "profile.json"
    module = get_module_class("driver_attention_dashsentinel")(
        {"profile_path": str(profile), "save_profile_every_seconds": 9999}
    )
    module.process(np.zeros((480, 640, 3), dtype=np.uint8))
    assert not profile.exists(), "nothing saved yet: the periodic save is 9999s away"
    module.close()
    assert profile.exists(), "upstream saves the profile on exit; the adapter must too"
    module.close()  # second call is harmless
    assert module._face_mesh is None


# ============================================================ F8: openpilot backlog
def test_openpilot_does_not_replay_a_stall_backlog_as_instant_policy_time(monkeypatch):
    """After a 30 s stall, looking down for ~0.2 s reached the TERMINAL alert."""
    from test_openpilot_module import _driver_state

    import modules.driver_attention_openpilot as op

    states = {"current": _driver_state()}

    class _StubModel:
        def __init__(self, path):
            pass

        def run(self, frame):
            return states["current"], 0.0

    monkeypatch.setattr(op, "_OnnxDMModel", _StubModel)
    now = [1000.0]
    monkeypatch.setattr(op.time, "monotonic", lambda: now[0])
    module = op.DriverAttentionOpenpilotModule({"instant_distraction_sound": False, "phone_detection": False})
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    for _ in range(100):
        now[0] += 1 / 30
        module.process(frame)
    now[0] += 30.0  # loop stall
    module.process(frame)
    assert module._tick_accum < op.DT_DMON, "the stall backlog must be dropped, not queued"

    states["current"] = _driver_state(pitch_net=-0.7)  # looks down
    fired = []
    for _ in range(15):  # 0.5 s of real time
        now[0] += 1 / 30
        fired += [e.type for e in module.process(frame)]
    assert "driverDistracted" not in fired
    assert "promptDriverDistracted" not in fired
    assert module._dm.awareness > 0.9


def test_openpilot_close_releases_hands_and_is_idempotent(monkeypatch):
    import modules.driver_attention_openpilot as op

    monkeypatch.setattr(op, "_OnnxDMModel", lambda path: types.SimpleNamespace(run=None))
    module = op.DriverAttentionOpenpilotModule({"phone_detection": False})
    hands = types.SimpleNamespace(closed=0)
    hands.close = lambda: setattr(hands, "closed", hands.closed + 1)
    module._hands = hands
    module.close()
    module.close()
    assert hands.closed == 1


# ============================================================ F9: Hailo predict kwargs
def _bare_hailo(raw_output):
    model = accel.HailoYOLO.__new__(accel.HailoYOLO)
    model._input_h = model._input_w = 640
    model._lock = threading.Lock()
    model._closed = False
    model._session = None
    model._pipeline = types.SimpleNamespace(infer=lambda inputs: raw_output)
    model._input_name = "input"
    model._raw_outputs = False
    model.names = dict(accel.COCO_NAMES)
    return model


def test_hailo_predict_accepts_the_device_keyword_the_comparison_modules_pass(monkeypatch):
    """openpilot/skinnemoen/mac_max call predict(..., device=...). With a null
    pin (the only way they reach Hailo) that raised TypeError and permanently
    disabled their phone detection."""
    monkeypatch.setattr(accel, "record_inference", lambda: None)
    per_class = [np.zeros((0, 5), dtype=np.float32) for _ in range(80)]
    per_class[67] = np.array([[0.25, 0.25, 0.5, 0.5, 0.9]], dtype=np.float32)
    model = _bare_hailo({"nms": [per_class]})

    [result] = model.predict(np.zeros((480, 640, 3), np.uint8), verbose=False, classes=[67],
                             conf=0.35, device=None)
    assert [int(b.cls[0]) for b in result.boxes] == [67]


# ============================================================ F10: cannot-run exit codes
def test_a_sample_with_no_readable_images_cannot_run_rather_than_regress():
    from benchmarks.accuracy_state_farm import sample_problem

    assert sample_problem({"per_class": {}, "skipped": 600}, 600) is not None
    assert sample_problem({"per_class": {"c0": {"n": 50}}, "skipped": 550}, 600) is not None
    assert sample_problem({"per_class": {"c0": {"n": 599}}, "skipped": 1}, 600) is None


def test_main_returns_cannot_run_when_the_images_are_missing(monkeypatch, tmp_path):
    from benchmarks import accuracy_state_farm as asf

    rows = [{"subject": "p002", "classname": "c0", "img": f"img_{i}.jpg"} for i in range(20)]
    monkeypatch.setattr(asf, "load_sample", lambda *a, **k: rows)
    monkeypatch.setattr(asf, "score_sample", lambda sample, sweep, quiet=False, backend="auto": {
        "backend": "cpu", "detector": "yolo11n.pt", "per_class": {}, "skipped": len(sample), "elapsed_s": 0.0})
    monkeypatch.setattr(asf, "RESULTS_DIR", tmp_path)
    code = asf.main(["--quiet", "--compare", str(REPO / "benchmarks" / "results" / "baseline_phase1.json")])
    assert code == asf.EXIT_CANNOT_RUN


@pytest.mark.usefixtures("_restore_accel")
def test_a_detector_that_will_not_load_exits_cannot_run_not_regression(monkeypatch):
    """An exception escaping _load_detector exited 1 — the REGRESSION code."""
    from benchmarks import accuracy_state_farm as asf
    from core import detectors

    def no_backend(weights, device=None):
        raise RuntimeError("no object-detection backend available")

    monkeypatch.setattr(detectors, "get_yolo_model", no_backend)
    with pytest.raises(SystemExit) as exc:
        asf._load_detector("cpu")
    assert exc.value.code == asf.EXIT_CANNOT_RUN


# ============================================================ F11: plot_results
def test_plot_results_ignores_json_that_is_not_a_benchmark_result(tmp_path):
    """The committed baseline_phase1.json made every plot crash: KeyError 'module'."""
    from benchmarks import plot_results

    (tmp_path / "baseline_phase1.json").write_text(json.dumps({"dataset": "state_farm", "summary": {}}))
    (tmp_path / "protocol_x_report.json").write_text(json.dumps({"recall": 1.0}))
    bench = {"module": "occupancy", "fps": 20.0, "latency_ms": {"mean": 1, "p50": 1, "p95": 1, "p99": 1},
             "cpu_temp_c": []}
    (tmp_path / "occupancy_20260101T000000.json").write_text(json.dumps(bench))

    results = plot_results.load_results(results_dir=tmp_path)
    assert results == [bench]
    plot_results.plot_fps_bar_chart(results, tmp_path / "fps.png")


def test_plot_results_loads_the_real_results_directory_without_crashing():
    from benchmarks import plot_results

    assert all("module" in r for r in plot_results.load_results())


# ============================================================ F12: unusable verdicts
class _Post:
    def __init__(self, message, status=200):
        self.message, self.status = message, status

    def __call__(self, url, json=None, headers=None, timeout=None):
        body = {"choices": [{"message": self.message}]}
        return types.SimpleNamespace(status_code=self.status, headers={}, json=lambda: body,
                                     raise_for_status=lambda: None)


def _compat_judge(message, failures=3):
    return VLMJudge({"provider": "openai_compatible", "model": "x", "base_url": "http://localhost:1/v1",
                     "interval_s": 0.0, "max_consecutive_failures": failures},
                    post=_Post(message), autostart=False)


def test_a_judge_that_only_returns_unusable_answers_disables_itself():
    """Empty / unparseable answers were dropped silently for ever ("warming up")."""
    judge = _compat_judge({"role": "assistant", "content": None})  # e.g. reasoning ate max_tokens
    for _ in range(3):
        judge._step(_FRAME)
    assert judge.disabled_reason is not None
    assert "3 consecutive errors" in judge.disabled_reason


@pytest.mark.parametrize("content", ["Sure, the driver looks fine.", '{"state": "sleeping", "confidence": 1}', ""])
def test_each_unusable_answer_counts_as_one_failure(content):
    judge = _compat_judge({"role": "assistant", "content": content})
    judge._step(_FRAME)
    assert judge._failures == 1 and judge.latest_verdict() is None


def test_an_explicit_refusal_is_still_not_a_failure():
    judge = _compat_judge({"role": "assistant", "content": None, "refusal": "I can't help with that."})
    judge._step(_FRAME)
    assert judge._failures == 0 and judge.disabled_reason is None


def test_anthropic_max_tokens_is_configurable():
    from test_vlm_judge import _fake_response, _FakeClient

    client = _FakeClient(responses=[_fake_response()])
    judge = VLMJudge({"model": "claude-opus-4-8", "interval_s": 0.0, "max_tokens": 2048},
                     client=client, autostart=False)
    judge._step(_FRAME)
    assert client.requests[0]["max_tokens"] == 2048


# ============================================================ F13 + F14: Hailo lifecycle
def test_a_later_successful_load_clears_the_stale_refusal(hailo):  # noqa: F811
    """A transient init failure at boot kept status() reporting "Hailo init
    failed — running on CPU" after a lazy retry had the model on the chip."""
    hl.chip.fail_vdevice = True
    assert accel.maybe_hailo_yolo("yolo11n.pt") is None
    assert "Hailo init failed" in accel.status()[1]

    hl.chip.fail_vdevice = False
    assert accel.maybe_hailo_yolo("yolo11n.pt") is not None  # the retry, without a reconfigure
    healthy, detail = accel.status()
    assert healthy is True and "Hailo in use" in detail


def test_a_timed_out_close_does_not_wait_again_on_every_alias(hailo, monkeypatch):  # noqa: F811
    model = hl.detectors.get_yolo_model("yolo11n.pt")
    model.predict(hl.FRAME)
    release = threading.Event()

    def wedged(self, inputs):
        release.wait()
        return {"nms": [[np.zeros((0, 5), dtype=np.float32) for _ in range(80)]]}

    monkeypatch.setattr(hl.InferVStreams, "infer", wedged)
    worker = threading.Thread(target=lambda: model.predict(hl.FRAME), daemon=True)
    worker.start()
    time.sleep(0.05)
    monkeypatch.setattr(type(model), "_CLOSE_TIMEOUT_S", 0.3)
    try:
        model.close()  # times out once
        started = time.monotonic()
        model.close()
        model.close()
        assert time.monotonic() - started < 0.1, "each further close() re-waited the full timeout"
    finally:
        release.set()
        worker.join(timeout=2)


# ============================================================ F16: close() everywhere
def test_mac_max_close_stops_the_judge_and_releases_graphs_once():
    from modules.driver_attention_max import DriverAttentionMaxModule as M

    judge = VLMJudge({"provider": "ollama"}, post=lambda *a, **k: None, autostart=True)
    mod = M.__new__(M)
    mod._vlm = judge
    graphs = {name: types.SimpleNamespace(closed=0) for name in ("_face_mesh", "_hands")}
    for name, graph in graphs.items():
        graph.close = (lambda g: lambda: setattr(g, "closed", g.closed + 1))(graph)
        setattr(mod, name, graph)

    mod.close()
    mod.close()

    assert all(g.closed == 1 for g in graphs.values())
    judge._thread.join(timeout=2)
    assert not judge._thread.is_alive(), "the judge's worker thread is still running after close()"


def test_skinnemoen_close_releases_face_mesh_and_is_idempotent():
    from modules.base import get_module_class

    module = get_module_class("driver_attention_skinnemoen")({"phone_detection": False})
    module.close()
    module.close()
    assert module._face_mesh is None
