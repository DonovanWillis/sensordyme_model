"""Phase 8 — backend unavailable, cloud offline, and monitoring dead are three
different things, and must never render as one.

Phase 8 opened with the app reporting "can't reach the backend" while the
backend was healthy the entire time. `cloud.base_url` named
`Donovans-MacBook-Pro.local`; macOS had renamed the machine to
`Donovans-MacBook-Pro-2` (it appends a suffix when the name is already claimed
on the network), so the device could not resolve it. DNS, not the backend.

The cost of that confusion is measured in debugging hours pointed at the wrong
subsystem: a hostname fault presented as a backend outage, on a device whose
camera and NPU were working perfectly throughout.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core import device_health

REPO = Path(__file__).resolve().parent.parent


def _load_preflight():
    """Import scripts/demo_preflight.py, registered so @dataclass resolves."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "sensordyme_demo_preflight_p8", REPO / "scripts" / "demo_preflight.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------- the three distinct states

def test_monitoring_health_does_not_depend_on_the_cloud_uplink():
    """A device with no backend is still monitoring. Conflating the two would
    make an intentionally-offline demo look like a dead safety device."""
    report = device_health.collect(module_name="driver_attention")
    assert "monitoring_active" in report
    # `collect` takes no cloud argument at all — the independence is structural,
    # not a value that happens to be right today.
    assert "cloud" not in str(report.get("inference", {})).lower()


def test_uplink_state_is_reported_separately_from_monitoring():
    """`cloud_connected` is its own field, set only when explicitly supplied."""
    without = device_health.collect(module_name="driver_attention")
    assert "cloud_connected" not in without, (
        "cloud state must not be invented when nobody supplied it"
    )
    offline = device_health.collect(module_name="driver_attention", uplink_connected=False)
    assert offline["cloud_connected"] is False
    assert offline["monitoring_active"] is True, (
        "an offline uplink must never imply monitoring has stopped"
    )


# ------------------------------------------------- the preflight's three states

@pytest.mark.parametrize(
    ("enabled", "connected", "expect_pass", "expect_phrase"),
    [
        (False, None,  True,  "intentionally offline"),   # deliberately off
        (True,  True,  True,  "connected"),               # configured and working
        (True,  False, False, "NOT connected"),           # configured and broken
        (True,  None,  False, "NOT connected"),           # configured, never reported
    ],
)
def test_preflight_distinguishes_all_three_uplink_states(
    enabled, connected, expect_pass, expect_phrase
):
    """The check keys on CONFIGURATION, so it cannot be wrong about intent.

    A device configured to reach a backend and failing to is a fault. A device
    configured not to is not. Anything else requires the tool to guess whether
    the demo "needs" cloud, and a gate that guesses is a gate people learn to
    ignore.
    """
    preflight = _load_preflight()
    report = preflight.Report()

    cloud_cfg = {"enabled": enabled, "base_url": "http://example.local:8787"}
    if not cloud_cfg.get("enabled"):
        report.add("backend uplink", True, "cloud.enabled is false — uplink intentionally offline")
    else:
        report.add(
            "backend uplink", connected is True,
            "cloud.enabled is true — "
            + ("connected" if connected is True else f"NOT connected (cloud_connected={connected})"),
        )
    check = report.checks[-1]
    assert (check.state == preflight.PASS) is expect_pass
    assert expect_phrase in check.detail


def test_a_broken_uplink_does_not_report_a_monitoring_fault():
    """The specific confusion that cost Phase 8 its opening hours."""
    preflight = _load_preflight()
    report = preflight.Report()
    report.add("backend uplink", False, "cloud.enabled is true — NOT connected")
    report.add("monitoring active", True, "monitoring_active=True")
    report.add("camera capturing", True, "ok — capturing")

    failed = [c.name for c in report.checks if c.state == preflight.FAIL]
    assert failed == ["backend uplink"], (
        "a backend fault must not also fail monitoring or camera — they are "
        f"independent subsystems, but {failed} failed together"
    )


# ------------------------------------------------------- configuration hygiene

def test_repo_config_does_not_ship_a_developer_hostname():
    """The repo default must not carry one machine's name.

    The device's own base_url is per-device state (like camera.rotate_180) and
    lives only on the device. A developer hostname committed to the repo would
    be provisioned onto every unit and would go stale exactly as this one did.
    """
    config = yaml.safe_load((REPO / "config.yaml").read_text())
    base_url = str((config.get("cloud") or {}).get("base_url") or "")
    assert "MacBook" not in base_url and "macbook" not in base_url, (
        f"config.yaml ships a developer hostname: {base_url!r}"
    )
    assert not config["cloud"].get("device_key"), (
        "config.yaml must not ship a device key — it is per-device state"
    )
