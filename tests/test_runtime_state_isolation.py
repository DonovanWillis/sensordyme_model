"""The test suite must never touch a real installation's state.

The repo root is where a running backend keeps its database, JWT secret and
identity, and where a device keeps data/monitoring.json (all gitignored). Tests
once wrote all four there: harmless in a fresh clone, an edit to production data
in a checkout that is actually running SensorDyme. tests/conftest.py redirects
them; this pins that it still does.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _outside_repo(path) -> bool:
    return REPO not in Path(path).resolve().parents


def test_the_backend_state_paths_point_outside_the_repo():
    from cloud import app, mdns

    assert _outside_repo(app.DB_PATH), app.DB_PATH
    assert _outside_repo(mdns.BACKEND_ID_FILE), mdns.BACKEND_ID_FILE


def test_the_monitoring_config_path_points_outside_the_repo():
    from core import monitoring_config

    assert _outside_repo(monitoring_config.DEFAULT_PATH), monitoring_config.DEFAULT_PATH


def test_the_suite_never_advertises_on_the_real_network():
    from cloud import mdns

    assert mdns.enabled() is False
