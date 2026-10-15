"""Test-suite setup: importable repo root, and isolation from the real runtime state.

The repo root is ALSO where a running installation keeps its state —
cloud/sensordyme.db (accounts, devices, events), cloud/.jwt_secret,
cloud/.backend_id and data/monitoring.json, all gitignored. Without this, any
test that imported cloud.app with default settings opened (or created) the real
database and secret, the health test minted a backend identity there, and the
preset-switch tests overwrote the real monitoring.json. On a checkout with live
state that is a test run silently editing production data.

Every one of those locations is pointed at a per-session temp directory BEFORE
any test module is imported (cloud.app and cloud.mdns read them at import).
Tests that need their own paths still set them per test, as they always have.
"""

import contextlib
import os
import sys
import tempfile
import weakref
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

_STATE = Path(tempfile.mkdtemp(prefix="sensordyme-test-state-"))
for _var, _name in (
    ("SENSORDYME_DB", "sensordyme.db"),
    ("SENSORDYME_JWT_SECRET_FILE", ".jwt_secret"),
    ("SENSORDYME_BACKEND_ID_FILE", ".backend_id"),
    ("SENSORDYME_MONITORING_PATH", "monitoring.json"),
):
    os.environ.setdefault(_var, str(_STATE / _name))
# Never announce a test backend on the developer's real network.
os.environ.setdefault("SENSORDYME_MDNS", "0")


#: Every MediaPipe solution graph (Pose, Hands, FaceMesh) built since tracking
#: was installed; see _close_mediapipe_graphs.
_LIVE_GRAPHS: "weakref.WeakSet" = weakref.WeakSet()
_GRAPH_TRACKING = {"installed": False}


def _install_graph_tracking() -> None:
    """Record each MediaPipe graph as it is built. Installed lazily, once
    mediapipe has been imported by the code under test, so this file never
    imports it itself (tests that pin import side effects stay meaningful)."""
    if _GRAPH_TRACKING["installed"]:
        return
    solution_base = sys.modules.get("mediapipe.python.solution_base")
    if solution_base is None:
        return
    original = solution_base.SolutionBase.__init__

    def tracking_init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        _LIVE_GRAPHS.add(self)

    solution_base.SolutionBase.__init__ = tracking_init
    _GRAPH_TRACKING["installed"] = True


@pytest.fixture(autouse=True)
def _close_mediapipe_graphs():
    """Close every MediaPipe graph a test built, when the test ends.

    A graph holds ~8 OS threads until close() — dropping the last reference
    and even gc.collect() do not release them (measured). The runtime closes a
    module on every swap and at shutdown, but tests build modules and drop
    them, so a full run accumulated ~250 live graphs, hit the per-process
    thread limit, and MediaPipe aborted the whole session from inside
    pthread_create.
    """
    _install_graph_tracking()
    yield
    _install_graph_tracking()
    for graph in list(_LIVE_GRAPHS):
        if getattr(graph, "_graph", None) is not None:
            with contextlib.suppress(Exception):  # teardown must not mask the test's result
                graph.close()
    _LIVE_GRAPHS.clear()


@pytest.fixture(autouse=True)
def _fresh_device_state(monkeypatch, tmp_path):
    """A per-test core.device_state file. It remembers the active preset and
    the app's zones across restarts, so a shared file would let one test's
    preset switch decide which module the next test's runtime boots into."""
    monkeypatch.setenv("SENSORDYME_DEVICE_STATE_PATH", str(tmp_path / "device_state.json"))


@pytest.fixture(autouse=True)
def _no_inherited_accelerator_evidence(monkeypatch):
    """Start every test with no completed accelerator inferences on record.

    core.accel keeps the count and time of completed Hailo inferences as module
    globals, and device_health reads them to decide whether a detector has died
    (inferences happened, then stopped). A test that drives a fake HailoYOLO
    leaves those globals set; once 10 s pass they read as "idle", and any later
    test calling device_health.collect() saw monitoring_active go False — so
    whether those tests passed depended on what ran before them. Only reset
    when the module is already imported, so tests that check import side
    effects see the same sys.modules they always did.
    """
    accel = sys.modules.get("core.accel")
    if accel is not None:
        monkeypatch.setattr(accel, "_inference_count", 0)
        monkeypatch.setattr(accel, "_last_inference_monotonic", None)
