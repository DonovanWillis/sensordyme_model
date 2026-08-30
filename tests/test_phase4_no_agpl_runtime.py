"""Phase 4: the SHIPPING runtime must not REQUIRE the AGPL-3.0 `ultralytics` package.

WHY THIS FILE EXISTS. `ultralytics==8.3.28` is AGPL-3.0 (confirmed in the
installed dist-info METADATA; see docs/LICENSING.md for the full analysis).
SensorDyme both *conveys* the software on a device and *serves inference over a
network* from it (the MJPEG dashboard in core/dashboard.py), so AGPL §13 and the
§4/§6 conveying obligations are both in play. That makes "is ultralytics on the
shipped device?" a commercial-release blocker, not a preference.

WHAT IS AND IS NOT ASSERTED HERE. These tests do NOT claim ultralytics is
unusable — the Mac dev profile and benchmarks/ legitimately load it, and
requirements.txt still pins it for exactly that. What they pin down is the
weaker, sufficient property: nothing on the *device* install path needs it.

    - core/ and modules/ import cleanly with `ultralytics` absent.
    - The device requirements file names neither ultralytics nor torch.
    - setup_pi.sh and `make deploy` install THAT file, not the dev one.
    - When there is no accelerator AND no ultralytics, the failure is a clear,
      actionable message — never a bare ImportError, and never a silent
      degradation to a detector that finds nothing.

The last one is the safety property. A baby monitor that quietly stops running
object detection still serves a dashboard and still looks alive.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Every module the device can actually run (main.py's standard path). The
# mac_max variants are deliberately included too: modules/__init__.py imports
# EVERY sibling eagerly to run the @register_module decorators, so a top-level
# `import ultralytics` anywhere in modules/ would be imported on the device even
# though the device never constructs that class.
_SHIPPING_IMPORTS = ["core.detectors", "core.accel", "core.rules", "core.hand_perception", "modules"]


class _BlockUltralytics:
    """A sys.meta_path finder that makes `import ultralytics` fail as if it were
    not installed — which is exactly the state of a shipped device."""

    def find_spec(self, name, path=None, target=None):
        if name == "ultralytics" or name.startswith("ultralytics."):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        # Anything else: defer to the finders after this one on sys.meta_path.
        return


@pytest.fixture
def without_ultralytics():
    """Run the body with `ultralytics` unimportable and un-imported, then restore.

    Purging sys.modules matters: the dev venv HAS ultralytics, and something
    earlier in the suite may already have imported it. Leaving a cached module
    object behind would make these tests pass for the wrong reason.
    """
    finder = _BlockUltralytics()
    saved = {k: v for k, v in sys.modules.items() if k == "ultralytics" or k.startswith("ultralytics.")}
    saved.update({k: v for k, v in sys.modules.items() if k in _SHIPPING_IMPORTS})
    for name in list(saved):
        del sys.modules[name]
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        for name in [k for k in sys.modules if k == "ultralytics" or k.startswith("ultralytics.")]:
            del sys.modules[name]
        for name in _SHIPPING_IMPORTS:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


def _requirement_names(path: Path) -> set[str]:
    """Distribution names pinned in a requirements file, lowercased.

    Ignores comments, blank lines, and `-r`/`-c` includes — the includes are
    followed explicitly by the caller when that is the point of the assertion.
    """
    names: set[str] = set()
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = re.match(r"^([A-Za-z0-9._-]+)", line)
        if match:
            names.add(match.group(1).lower())
    return names


# --------------------------------------------------------------- import surface


@pytest.mark.parametrize("module_name", _SHIPPING_IMPORTS)
def test_shipping_imports_do_not_need_ultralytics(without_ultralytics, module_name):
    """The device's import graph must not touch ultralytics at all.

    This is stronger than "detectors.py imports it lazily": modules/__init__.py
    imports every module in the package, so this covers all of them at once.
    """
    importlib.import_module(module_name)
    assert "ultralytics" not in sys.modules, (
        f"importing {module_name} pulled in ultralytics — the AGPL package is "
        "back on the device import path"
    )


# ------------------------------------------------- the no-backend failure mode


def test_no_accelerator_and_no_ultralytics_fails_loudly_and_actionably(
    without_ultralytics, monkeypatch
):
    """THE REGRESSION THIS FILE IS FOR.

    A device with no .hef configured and no ultralytics installed used to die on
    `from ultralytics import YOLO` with a bare ModuleNotFoundError — a traceback
    that tells an installer nothing about which of the two fixes they need. It
    must instead raise an error that names the accelerator option AND the
    developer install, and it must NOT return a stub that detects nothing.
    """
    detectors = importlib.import_module("core.detectors")
    accel = importlib.import_module("core.accel")
    monkeypatch.setattr(accel, "maybe_hailo_yolo", lambda weights: None)
    monkeypatch.setattr(detectors, "_model_cache", {})

    with pytest.raises(RuntimeError) as excinfo:
        detectors.get_yolo_model("yolo11n.pt")

    message = str(excinfo.value)
    assert "yolo11n.pt" in message, "the error must name the weights that could not load"
    assert "accelerator.yolo_hef" in message, "must point at the accelerator fix"
    assert "requirements.txt" in message, "must point at the developer install"
    assert "AGPL" in message, "must say WHY it is not installed by default"
    # A ModuleNotFoundError escaping unwrapped is the old behavior, not the new one.
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_the_failure_is_not_a_silent_stub(without_ultralytics, monkeypatch):
    """Failing closed, not open: no backend must never yield a model object.

    The dangerous shape of this fix would be `return _NullModel()` so startup
    survives. The device would then serve a live dashboard while detecting
    nothing at all, which is worse than not booting.
    """
    detectors = importlib.import_module("core.detectors")
    accel = importlib.import_module("core.accel")
    monkeypatch.setattr(accel, "maybe_hailo_yolo", lambda weights: None)
    monkeypatch.setattr(detectors, "_model_cache", {})

    with pytest.raises(RuntimeError):
        detectors.get_yolo_model("yolo11n.pt")
    assert detectors._model_cache == {}, "a failed load must not be cached as success"


def test_ultralytics_is_still_available_to_developers():
    """Not a removal from the repo — a removal from the DEVICE.

    requirements.txt (the developer/CI install) must keep pinning it, so the Mac
    profile and benchmarks/accuracy_state_farm.py keep working. If this test
    ever fails, the dev path was broken in the name of the licence fix.
    """
    assert "ultralytics" in _requirement_names(REPO_ROOT / "requirements.txt")


# ------------------------------------------- the accelerator's call surface


def test_hailo_shaped_model_satisfies_every_consumer_attribute():
    """The Hailo shim is the ONLY detector on a shipping device, so it has to
    cover every attribute the consumers reach for.

    Collected by reading each call site rather than by guessing:
        .names          driver_attention.py:959,1076  driver_attention_max.py:469,517
                        driver_attention_skinnemoen.py:147  ..._openpilot.py:831
        .predict(...)   occupancy.py:215  baby_monitoring.py:1180  detectors.py:98
        .boxes          core/detectors.py:101
        .xyxy[0]        core/detectors.py:104
        .cls[0]         core/detectors.py:102
        .conf[0]        core/detectors.py:103
    """
    accel = importlib.import_module("core.accel")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    class _FakeHailo:
        """core.accel.HailoYOLO's surface, without the Pi-only hailo_platform import."""

        def __init__(self):
            self.names = dict(accel.COCO_NAMES)

        def predict(self, frame, verbose=False, classes=None, conf=0.25):
            box = accel._ShimBox(np.array([10.0, 20.0, 110.0, 220.0]), cls_id=67, conf=0.9)
            return [accel._ShimResult([box])]

    model = _FakeHailo()

    # .names — every consumer builds its class-id map from this mapping.
    assert model.names[0] == "person"
    assert model.names[67] == "cell phone"
    assert {cid for cid, name in model.names.items() if name == "cell phone"} == {67}

    # .predict(...) with the exact keyword set the modules pass.
    results = model.predict(frame, verbose=False, classes=[67], conf=0.25)
    for result in results:
        for box in result.boxes:
            assert int(box.cls[0]) == 67
            assert float(box.conf[0]) == pytest.approx(0.9)
            assert [int(v) for v in box.xyxy[0].tolist()] == [10, 20, 110, 220]

    # And it must survive core.detectors' own consumer, predict_roi, unchanged.
    detectors = importlib.import_module("core.detectors")
    [(roi_box, class_id, roi_score)] = detectors.predict_roi(
        model, frame, roi=(100, 100, 300, 300), classes=[67], conf=0.25
    )
    assert roi_box == (110, 120, 210, 320)
    assert class_id == 67
    assert roi_score == pytest.approx(0.9)


# ------------------------------------------------------- the shipping install


def test_device_requirements_exist_and_name_no_agpl_package():
    """requirements-device.txt is the file that reaches a customer's hardware."""
    device_file = REPO_ROOT / "requirements-device.txt"
    assert device_file.exists(), "the shipping requirements set must be a separate file"
    names = _requirement_names(device_file)
    assert "ultralytics" not in names
    # torch/torchvision arrive ONLY as ultralytics dependencies here (nothing in
    # the repo imports torch outside modules/driver_attention_max.py's mac-only
    # require_mps). Naming them keeps a future "just add torch" from quietly
    # re-opening the door, and they are ~800MB on a Pi besides.
    assert "torch" not in names
    assert "torchvision" not in names


def test_device_requirements_cover_what_the_device_actually_runs():
    """Removing a package is only safe if nothing shipping imported it.

    requirements.txt:13 noted that `requests` was "already a transitive dep of
    ultralytics" — that is precisely the kind of pin that breaks when the parent
    goes away, so the direct pins are asserted here rather than assumed.
    """
    names = _requirement_names(REPO_ROOT / "requirements-device.txt")
    for package in ["mediapipe", "opencv-python", "numpy", "pyyaml", "requests", "onnxruntime"]:
        assert package in names, f"the device imports {package} but the shipping pins omit it"


def test_dev_requirements_include_the_device_set_so_the_pins_cannot_drift():
    """One source of truth for the shared pins.

    Two hand-maintained lists of the same packages drift, and the way that shows
    up is a device running a different opencv than CI tested against.
    """
    text = (REPO_ROOT / "requirements.txt").read_text()
    assert "-r requirements-device.txt" in text
    shared = _requirement_names(REPO_ROOT / "requirements-device.txt")
    dev_only = _requirement_names(REPO_ROOT / "requirements.txt")
    assert not (shared & dev_only), (
        f"pinned in BOTH files (they will drift): {sorted(shared & dev_only)}"
    )


def test_pi_install_and_deploy_use_the_device_requirements():
    """The two paths that put packages onto real hardware.

    setup_pi.sh is the first install; `make deploy` is every install after that.
    Both were installing requirements.txt, so missing either one leaves
    ultralytics on the device regardless of what the file says.
    """
    setup_pi = (REPO_ROOT / "setup_pi.sh").read_text()
    assert "-r requirements-device.txt" in setup_pi
    assert not re.search(r"-r\s+requirements\.txt", setup_pi), (
        "setup_pi.sh still installs the developer requirements onto the device"
    )

    makefile = (REPO_ROOT / "Makefile").read_text()
    deploy = makefile[makefile.index("deploy:") :].split("\n\n")[0]
    assert "-r requirements-device.txt" in deploy
    assert not re.search(r"-r\s+requirements\.txt", deploy), (
        "make deploy still installs the developer requirements onto the device"
    )


def test_deploy_does_not_rsync_agpl_model_weights():
    """The hole the pip fix alone does not close.

    A .pt checkpoint is gitignored, but `make deploy` rsyncs the WORKING TREE,
    not the index — and ultralytics auto-downloads yolo11n.pt into the repo root
    on any dev run. So before this exclude, a deploy from a developer's Mac put
    an AGPL-3.0 model weight onto customer hardware even with requirements fixed.
    Weights carry the licence of the project that trained them (docs/LICENSING.md
    tables yolo11n → ultralytics/ultralytics → AGPL-3.0).
    """
    makefile = (REPO_ROOT / "Makefile").read_text()
    deploy = makefile[makefile.index("deploy:") :].split("\n\n")[0]
    rsync = deploy[deploy.index("rsync") :]
    assert "--exclude='*.pt'" in rsync, "deploy would copy .pt weights onto the device"
    # openpilot's dmonitoring model is a different licence and the device really
    # runs it — excluding *.onnx would break driver_attention_openpilot for a
    # licence problem it does not have.
    assert "--exclude='*.onnx'" not in rsync
