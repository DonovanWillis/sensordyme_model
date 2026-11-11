"""M0.2 — deployment preflight (scripts/deploy_preflight.py).

A deploy never overwrites a device's config.yaml, so the code about to land is
checked against the config the device actually has. Nothing here touches a
device: configs are text, the device's model hash is a parameter.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from scripts import deploy_preflight as preflight

REPO = Path(__file__).resolve().parent.parent
REPO_CONFIG = (REPO / "config.yaml").read_text()
PINNED = yaml.safe_load(REPO_CONFIG)["accelerator"]["yolo_hef_sha256"]
EVALUATED_BUILD = "4c71eab80976a2c9f1b134877890abce02871f6d4d92f53ac5a3f8bf2f24add2"  # setup_pi.sh keeps it


def _config(**changes) -> str:
    config = yaml.safe_load(REPO_CONFIG)
    for dotted, value in changes.items():
        *parents, key = dotted.split("__")
        node = config
        for part in parents:
            node = node[part]
        if value is _DROP:
            node.pop(key, None)
        else:
            node[key] = value
    return yaml.safe_dump(config)


_DROP = object()


def test_the_shipped_driver_config_passes_for_a_driver_deployment():
    report = preflight.check(REPO_CONFIG, expect_role="driver_safety", hef_sha256=PINNED)
    assert report.ok, report.errors
    assert report.role == "driver_safety" and report.role_declared


def test_a_device_without_a_role_is_refused_for_a_driver_deployment_and_never_changed(tmp_path):
    """Item 7: existing devices are not silently classified as protected units."""
    text = _config(role=_DROP)
    path = tmp_path / "config.yaml"
    path.write_text(text)
    assert preflight.main(["--config", str(path), "--expect-role", "driver_safety"]) == 1
    assert path.read_text() == text, "the preflight must never edit a device's config"
    report = preflight.check(text, expect_role="driver_safety")
    assert any("MIGRATION: add `role: driver_safety`" in e for e in report.errors)
    assert report.role == "general" and not report.role_declared


def test_a_general_purpose_device_stays_deployable():
    text = _config(role=_DROP, module="fall_detection")
    report = preflight.check(text)
    assert report.ok, report.errors
    assert any("runs as 'general'" in w for w in report.warnings)


@pytest.mark.parametrize(("change", "reason"), [
    ({"modules__driver_attention__drowsy_seconds": 5.0}, "safety floor"),
    ({"alerts__sinks": ["console"]}, "audible"),
    ({"module": "occupancy"}, "requires `module:`"),
])
def test_a_weakened_driver_config_is_refused(change, reason):
    report = preflight.check(_config(**change), expect_role="driver_safety")
    assert not report.ok
    assert any(reason in e for e in report.errors), report.errors


def test_a_property_device_needs_a_detector_configured():
    text = _config(role="property", module="occupancy", accelerator__yolo_hef="")
    report = preflight.check(text, expect_role="property")
    assert any("needs object detection" in e for e in report.errors), report.errors


@pytest.mark.parametrize(("device_hef", "ok", "reason"), [
    (PINNED, True, None),
    (EVALUATED_BUILD, False, "not the build config.yaml pins"),
    ("", False, "no object-detection model"),
])
def test_the_devices_model_must_be_the_pinned_build(device_hef, ok, reason):
    """Audit F1: setup_pi.sh keeps an 'evaluated' .hef the runtime refuses at load."""
    report = preflight.check(REPO_CONFIG, expect_role="driver_safety", hef_sha256=device_hef)
    assert report.ok is ok
    if reason:
        assert any(reason in e for e in report.errors)


def test_an_unparseable_config_is_refused():
    assert not preflight.check("module: [unclosed").ok
    assert not preflight.check("- a list").ok


def test_make_deploy_runs_the_preflight_first():
    makefile = (REPO / "Makefile").read_text()
    assert re.search(r"^deploy: deploy-preflight\b", makefile, re.MULTILINE)
    recipe = makefile.split("deploy-preflight:", 1)[1].split("\ndeploy:", 1)[0]
    assert "scripts/deploy_preflight.py" in recipe
    assert "rsync" not in recipe and "scp -q pi@" in recipe, "the preflight reads, never writes"
