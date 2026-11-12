"""First-time virtual pairing, end to end (scripts/ble_onboarding_e2e.py), then moving the same unit.

The app's onboarding code (Swift, compiled for this Mac) against the device's
Bluetooth provisioning code (Python, simulated radio) and the real backend,
with the virtual device runtime reporting in once its Wi-Fi is up. Asserted
from the backend's side and the device's side, not only the app's. macOS with
swiftc only; not a Bluetooth validation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("cryptography")
pytest.importorskip("fastapi")

from sim import ble_harness

REPO = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(ble_harness.swiftc() is None, reason="needs swiftc on macOS")


def _script():
    spec = importlib.util.spec_from_file_location("ble_onboarding_e2e", REPO / "scripts" / "ble_onboarding_e2e.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_first_time_pairing_then_reprovisioning_the_same_unit(tmp_path):
    summary = _script().run(tmp_path / "evidence")
    first = summary["scenarios"]["first-time"]
    again = summary["scenarios"].get("reprovision")
    assert first["ok"], first["checks"]
    assert again is not None and again["ok"], (again or {}).get("checks")
    assert summary["secrets_in_evidence"] == []
    assert summary["ok"]

    # The device side: proof before anything secret, then one join.
    assert first["device_auth_messages"] == [
        "app->device hello", "device->app challenge", "app->device proof", "device->app proven",
        "app->device creds", "device->app accepted"]
    # The account side: a refused claim, then the real one, then the mode.
    calls = first["api_calls"]
    assert "POST /v1/devices/claim -> 404" in calls and "POST /v1/devices/claim -> 200" in calls
    assert calls.index("POST /v1/devices/claim -> 404") < calls.index("POST /v1/devices/claim -> 200")
    assert any(c.startswith("POST /v1/devices/") and c.endswith("/preset -> 200") for c in calls)
    (row,) = first["backend_after"]
    assert row["online"] and row["hardware_id"].startswith("SD-SIM-")
    assert row["desired_preset"] or row["active_preset"], "a monitoring mode was set"

    # The same unit moved: no claim, same account row.
    assert not any("/claim" in c for c in again["api_calls"])
    (moved,) = again["backend_after"]
    assert moved["id"] == row["id"] and moved["hardware_id"] == row["hardware_id"]
    assert (tmp_path / "evidence" / "summary.json").exists()
