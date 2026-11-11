"""M6 security: the deploy preflight refuses the developer profile on a device and
names every exposure the device config would have; frames cannot leave the device.

* `profile: mac_max` is refused: it is the only path that uploads frames (the
  VLM judge, Gemini by default) and cannot run on a Pi anyway (no torch/MPS);
* the device image carries no torch, so the VLM judge cannot be constructed there;
* the uplink payload carries events and status only -- no image field exists;
* LAN dashboard without (or with a short) token, cleartext/unverified uplink and
  on-disk frame capture are warned about on every preflight.
"""

from pathlib import Path

import pytest
import yaml

from scripts import deploy_preflight as preflight

REPO = Path(__file__).resolve().parent.parent
BASE = yaml.safe_load((REPO / "config.yaml").read_text())


def _check(**overrides):
    config = yaml.safe_load(yaml.safe_dump(BASE))
    for dotted, value in overrides.items():
        node = config
        *parents, leaf = dotted.split("__")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return preflight.check(yaml.safe_dump(config))


def test_the_developer_profile_is_refused_on_a_device():
    report = _check(profile="mac_max")
    assert not report.ok
    assert any("not a device profile" in e and "uploads camera frames" in e for e in report.errors)


def test_the_device_profile_is_accepted():
    report = _check(profile="pi")
    assert not any("device profile" in e for e in report.errors)


def test_the_device_image_cannot_run_the_vlm_judge():
    """mac_max needs torch with MPS; requirements-device.txt installs no torch at all."""
    pins = [line.split("==")[0].strip().lower() for line in (REPO / "requirements-device.txt").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    assert "torch" not in pins and "torchvision" not in pins and "ultralytics" not in pins
    main_src = (REPO / "main.py").read_text()
    assert 'get_module_class(f"{module_name}_max")' in main_src
    assert main_src.index("require_mps()") < main_src.index('get_module_class(f"{module_name}_max")'), (
        "the _max (VLM) module must only be built after require_mps() refuses non-Apple hardware")


def test_the_uplink_payload_has_no_image_field():
    import types

    from core.uplink import CloudUplink

    uplink = CloudUplink({"base_url": "https://backend.example", "device_key": "dk_x", "poll_interval_s": 0})
    sent = []

    def post(url, json=None, **_kwargs):
        sent.append(json)
        uplink._stop.set()
        return types.SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=dict)

    uplink._requests = types.SimpleNamespace(post=post)
    event = types.SimpleNamespace(timestamp=types.SimpleNamespace(isoformat=lambda: "t"), type="fall",
                                  confidence=0.9, contributing_signals={"x": 1}, duration=1.0)
    uplink.push_event(event, "loud")
    uplink._run()
    body = sent[0]
    assert set(body) == {"events", "active_preset", "rules_version", "zones_version", "health", "monitoring"}
    assert set(body["events"][0]) <= {"event_key", "model", "timestamp", "type", "confidence", "signals",
                                      "duration", "severity", "source", "notify"}


def test_the_repo_config_keeps_frame_capture_and_frame_upload_off():
    assert BASE["alerts"].get("save_evidence", False) is False
    assert BASE["training"]["capture"]["enabled"] is False
    assert BASE["training"]["capture"]["capture_negatives"] is False
    assert BASE.get("profile", "pi") == "pi"


@pytest.mark.parametrize(("overrides", "needle"), [
    ({"dashboard__token": None}, "no token"),
    ({"dashboard__token": "abc123"}, "shorter than 16"),
    ({"cloud__enabled": True, "cloud__base_url": "http://backend.local:8787"}, "plain http://"),
    ({"cloud__enabled": True, "cloud__base_url": "https://b.example", "cloud__tls_insecure_skip_verify": True},
     "tls_insecure_skip_verify"),
    ({"alerts__save_evidence": True}, "alerts.save_evidence"),
    ({"training__capture__enabled": True}, "training.capture.enabled"),
])
def test_exposures_are_warned(overrides, needle):
    report = _check(**overrides)
    assert any(needle in w for w in report.warnings), report.warnings


def test_a_hardened_config_has_no_exposure_warnings():
    report = _check(dashboard__token="f" * 32, cloud__enabled=True, cloud__base_url="https://b.example")
    exposure = [w for w in report.warnings if not w.startswith("config:")
                and any(k in w for k in ("token", "http://", "tls_", "SD card"))]
    assert exposure == []


def test_warnings_do_not_refuse_a_deploy():
    report = _check(dashboard__token="abc", alerts__save_evidence=True)
    assert not any("token" in e or "SD card" in e for e in report.errors)
