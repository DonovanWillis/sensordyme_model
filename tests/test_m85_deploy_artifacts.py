"""M8.5 (M84-5): model files are verified where they are loaded, and a mismatch fails closed.

* MediaPipe .tflite files inside the mediapipe wheel: core/model_registry.py
  verify_bundled_artifacts() hashes them where Python loads them from; main.py's
  enforce_bundled_artifacts() refuses the matching solution class BEFORE any module
  builds a graph; the runtime reports the capability IMPAIRED with the reason.
* openpilot's dmonitoring_model.onnx (benchmark module): pinned sha256 at download
  and at load.
"""

from __future__ import annotations

import hashlib
import io
import sys
import types
from pathlib import Path

import pytest

from core import model_registry as reg
from core import perception as pc

REPO = Path(__file__).resolve().parent.parent
FACE_A, FACE_B, HAND = b"face-detector-bytes", b"face-landmark-bytes", b"hand-bytes"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _registry():
    def card(model_id, capability, files):
        return {"id": model_id, "provides_capabilities": [capability], "evaluation": {"status": "unevaluated"},
                "artifacts": [{"path": f"mediapipe/modules/{name}", "sha256": _sha(data), "format": "tflite"}
                              for name, data in files.items()]}

    return reg.parse_registry({"schema_version": 1, "models": [
        card("mediapipe_face_mesh", "face", {"fd.tflite": FACE_A, "fl.tflite": FACE_B}),
        card("mediapipe_hands", "hands", {"hl.tflite": HAND}),
        {"id": "hailo", "provides_capabilities": ["objects"], "evaluation": {"status": "unevaluated"},
         "artifacts": [{"path": "models/m.hef", "sha256": "a" * 64, "format": "hef"}]},
    ]})


def _site(tmp_path, **override) -> Path:
    files = {"fd.tflite": FACE_A, "fl.tflite": FACE_B, "hl.tflite": HAND, **override}
    root = tmp_path / "site-packages"
    (root / "mediapipe" / "modules").mkdir(parents=True)
    for name, data in files.items():
        if data is not None:
            (root / "mediapipe" / "modules" / name).write_bytes(data)
    return root


# ---------------------------------------------------------------------- verification
def test_matching_files_verify_and_only_bundled_tflite_files_are_checked(tmp_path):
    checks = reg.verify_bundled_artifacts(_registry(), site_root=_site(tmp_path))
    assert [c.status for c in checks] == ["ok", "ok", "ok"]
    assert {c.model for c in checks} == {"mediapipe_face_mesh", "mediapipe_hands"}, "the .hef is core/accel.py's"
    assert reg.artifact_faults(checks) == {}


def test_a_modified_file_is_a_fault_of_its_capability_only(tmp_path):
    checks = reg.verify_bundled_artifacts(_registry(), site_root=_site(tmp_path, **{"fl.tflite": b"tampered"}))
    faults = reg.artifact_faults(checks)
    assert set(faults) == {"face"}
    assert "fl.tflite" in faults["face"] and "not the pinned" in faults["face"] and "refused" in faults["face"]


def test_a_missing_file_is_a_fault_and_an_absent_package_is_not(tmp_path):
    checks = reg.verify_bundled_artifacts(_registry(), site_root=_site(tmp_path, **{"hl.tflite": None}))
    assert reg.artifact_faults(checks) == {"hands": next(c.reason() for c in checks if c.status == "missing")}
    empty = tmp_path / "empty"
    empty.mkdir()
    absent = reg.verify_bundled_artifacts(_registry(), site_root=empty)
    assert {c.status for c in absent} == {"package_absent"} and reg.artifact_faults(absent) == {}


def test_the_installed_mediapipe_matches_the_registry_pins():
    """On this machine (the same bytes ship in the aarch64 wheel the lock pins)."""
    pytest.importorskip("mediapipe")
    checks = reg.verify_bundled_artifacts()
    assert len(checks) == 6 and all(c.status == "ok" for c in checks), [c.to_dict() for c in checks]
    assert reg.main(["verify-artifacts"]) == 0


def test_the_cli_fails_on_a_fault_and_on_an_absent_package(tmp_path, monkeypatch, capsys):
    bad = reg.verify_bundled_artifacts(_registry(), site_root=_site(tmp_path, **{"fd.tflite": b"x"}))
    monkeypatch.setattr(reg, "verify_bundled_artifacts", lambda *a, **k: bad)
    out = tmp_path / "a.json"
    assert reg.main(["verify-artifacts", "--json-out", str(out)]) == 1
    assert '"ok": false' in out.read_text() and "FAIL" in capsys.readouterr().out
    absent = [reg.ArtifactCheck("m", "mediapipe/x.tflite", "a" * 64, "package_absent", capabilities=("face",))]
    monkeypatch.setattr(reg, "verify_bundled_artifacts", lambda *a, **k: absent)
    assert reg.main(["verify-artifacts"]) == 1, "the staged install requires the package it just installed"
    assert reg.main(["verify-artifacts", "--allow-absent"]) == 0


# ---------------------------------------------------------------------- fail closed at boot
class _RealFaceMesh:
    loaded = 0

    def __init__(self, *args, **kwargs):
        _RealFaceMesh.loaded += 1  # the real class parses the .tflite here


@pytest.fixture
def fake_mediapipe(monkeypatch):
    solutions = types.SimpleNamespace(
        face_mesh=types.SimpleNamespace(FaceMesh=_RealFaceMesh),
        hands=types.SimpleNamespace(Hands=object), pose=types.SimpleNamespace(Pose=object))
    module = types.ModuleType("mediapipe")
    module.solutions = solutions  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mediapipe", module)
    yield solutions
    reg.enforce_bundled_artifacts(checks=[])  # leave no boot state behind


def test_a_mismatch_keeps_the_model_from_ever_loading_and_is_remembered(tmp_path, fake_mediapipe):
    checks = reg.verify_bundled_artifacts(_registry(), site_root=_site(tmp_path, **{"fd.tflite": b"tampered"}))
    faults = reg.enforce_bundled_artifacts(checks)
    assert set(faults) == {"face"} and reg.boot_artifact_faults() == faults
    graph = fake_mediapipe.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    assert _RealFaceMesh.loaded == 0, "the unverified model file is never parsed"
    assert graph.process(object()).multi_face_landmarks is None, "it finds nothing, it runs nothing"
    assert "fd.tflite" in graph.refused_reason
    assert fake_mediapipe.hands.Hands is object, "a verified model is untouched"


def test_when_the_check_cannot_run_every_mediapipe_model_is_refused(fake_mediapipe, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("registry unreadable")

    monkeypatch.setattr(reg, "verify_bundled_artifacts", broken)
    faults = reg.enforce_bundled_artifacts()
    assert set(faults) == {"face", "hands", "pose"} and "could not be checked" in faults["face"]
    assert fake_mediapipe.pose.Pose is not object


def test_main_verifies_before_it_builds_the_module():
    source = (REPO / "main.py").read_text()
    assert source.index("enforce_bundled_artifacts()") < source.index("module = module_cls(module_config)")


def test_a_blocked_component_stays_unavailable_whatever_runs():
    engine = pc.PerceptionEngine()
    engine.block(pc.FACE_LANDMARKS, "model integrity: fl.tflite ... refused")
    engine.graph_ran(pc.FACE_LANDMARKS, object(), 3.0)  # a run would clear `unavailable`; not a block
    status = engine.status()[pc.FACE_LANDMARKS]
    assert status.state == pc.UNAVAILABLE and "model integrity" in str(status.reason)


def test_the_runtime_reports_the_refused_capability_impaired_with_the_reason(tmp_path, monkeypatch):
    """A driver-safety device whose face model failed the boot check: FAILED, and says why."""
    from test_m01_role_floors import _driver_runtime

    from core import monitoring_state as ms

    reason = "model integrity: face_landmark_with_attention.tflite sha256 0123456789ab... is not the pinned"
    monkeypatch.setattr(reg, "_BOOT_FAULTS", {"face": reason})
    clock = {"now": 1000.0}
    monkeypatch.setattr("time.monotonic", lambda: clock["now"])
    runtime, *_ = _driver_runtime(tmp_path, clock, monkeypatch, closed={"value": False})
    try:
        evaluation = runtime._evaluate_monitoring()
        face = evaluation.capabilities["face"]
        assert face.state == ms.IMPAIRED and "model integrity" in str(face.reason)
        report = runtime.monitoring_report(evaluation)
        assert report.state is ms.MonitoringState.FAILED
        assert any("model integrity" in r for r in report.reasons)
        wire = runtime.perception_report()["components"][pc.FACE_LANDMARKS]
        assert wire["state"] == pc.UNAVAILABLE and "model integrity" in wire["reason"]
    finally:
        if runtime.logger is not None:
            runtime.logger.close()


# ---------------------------------------------------------------------- openpilot .onnx (benchmark module)
PINNED = b"the pinned openpilot model"


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@pytest.fixture
def openpilot(monkeypatch):
    from modules import driver_attention_openpilot as op

    monkeypatch.setattr(op, "_MODEL_SHA256", _sha(PINNED))
    return op


def test_a_download_that_is_not_the_pinned_model_is_refused_and_left_nowhere(tmp_path, openpilot, monkeypatch):
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: _Response(b"a different model"))
    target = tmp_path / "dmonitoring_model.onnx"
    with pytest.raises(RuntimeError, match="not the pinned"):
        openpilot._fetch_pinned_model(target, sha256=openpilot._MODEL_SHA256)
    assert not target.exists() and not list(tmp_path.iterdir()), "no file and no .part left behind"


def test_the_pinned_download_is_installed(tmp_path, openpilot, monkeypatch):
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: _Response(PINNED))
    target = tmp_path / "dmonitoring_model.onnx"
    openpilot._fetch_pinned_model(target, sha256=openpilot._MODEL_SHA256)
    assert target.read_bytes() == PINNED
    openpilot._verify_pinned_model(target, sha256=openpilot._MODEL_SHA256)


def test_an_existing_model_file_is_verified_before_it_is_loaded(tmp_path, openpilot, monkeypatch):
    pytest.importorskip("onnxruntime")
    import onnxruntime

    loaded = []
    monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *a, **k: loaded.append(a))
    stray = tmp_path / "dmonitoring_model.onnx"
    stray.write_bytes(b"whatever was lying around")
    with pytest.raises(RuntimeError, match="refusing to load"):
        openpilot._OnnxDMModel(str(stray))
    assert loaded == []


def test_the_shipped_pin_is_the_repo_copy_of_the_v0_9_7_model():
    from modules import driver_attention_openpilot as op

    assert op._MODEL_SHA256 == "3dd3982940d823c4fbb0429b733a0b78b0688d7d67aa76ff7b754a3e2f3d8683"
    assert "/v0.9.7/" in op._MODEL_URL and "urlretrieve" not in (REPO / "modules/driver_attention_openpilot.py").read_text()
    copy = REPO / "dmonitoring_model.onnx"
    if copy.exists():
        assert _sha(copy.read_bytes()) == op._MODEL_SHA256
