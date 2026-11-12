"""M6: owner settings reach the device through the preset name, and only valid ones run.

The backend stores "preset:variant@key=value" as the desired preset; the device
applies the variant's overrides with the settings on top (services/runtime.py
`_switch_preset_now` via core/presets.py `preset_overrides`), reports the
canonical name back so the backend converges, survives a restart
(main.py `restored_preset`), and refuses — with a recorded reason — a setting
the preset's schema does not allow or a role floor forbids.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from core import device_state
from core.alerts import AlertManager
from core.events import Event
from core.logger import EventLogger
from core.presets import DEVICE_POLICY_FEATURES, same_preset
from main import restored_preset
from modules.base import VisionModule, register_module
from services.runtime import DeviceRuntime


@register_module("m6_settings_probe")
class _Probe(VisionModule):
    name = "m6_settings_probe"
    last_config: ClassVar[dict] = {}

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        type(self).last_config = dict(config)

    def process(self, frame: np.ndarray) -> list[Event]:
        return []


PRESETS = {
    "probe": {
        "title": "Probe", "module": "m6_settings_probe",
        "variants": {"strict": {"title": "Strict", "overrides": {"wait_s": 1.5, "other": 7}}},
        "settings": [
            {"key": "wait_s", "title": "Wait", "type": "number", "default": 2.0, "min": 1.0, "max": 3.0,
             "step": 0.5, "unit": "s"},
            {"key": "loud", "title": "Loud", "type": "bool", "default": True},
            {"key": "tone", "title": "Tone", "type": "enum", "default": "beep", "options": ["beep", "chime"]},
        ],
    },
    "other": {"title": "Other", "module": "m6_settings_probe"},
}


@pytest.fixture
def runtime(tmp_path):
    rt = DeviceRuntime(
        config={"module": "m6_settings_probe", "modules": {"m6_settings_probe": {"wait_s": 2.0, "base": 1}}},
        profile="pi", module=_Probe({}), module_name="m6_settings_probe",
        alert_manager=AlertManager(sinks=[]), alert_cfg={}, alert_event_types=set(),
        alert_types_for=lambda name, preset=None: set(), presets=PRESETS, active_preset="other",
        display=False, debug=False,
    )
    rt.logger = EventLogger(str(tmp_path / "events.csv"))
    try:
        yield rt
    finally:
        rt.logger.close()


def test_settings_apply_over_the_variant_and_the_name_is_canonical(runtime):
    runtime._switch_preset("probe:strict@wait_s=2.50,tone=chime")
    config = _Probe.last_config
    assert config["wait_s"] == 2.5          # the setting beats the variant's 1.5
    assert config["other"] == 7             # the variant's other override still applies
    assert config["base"] == 1              # config.yaml's block underneath
    assert config["tone"] == "chime"
    assert runtime.active_preset == "probe:strict@tone=chime,wait_s=2.5"
    # What the backend stored and what the device reports are the same mode.
    assert same_preset("probe:strict@wait_s=2.50,tone=chime", runtime.active_preset, PRESETS)


@pytest.mark.parametrize("name", [
    "probe@wait_s=9",          # above the schema's max
    "probe@wait_s=2.2",        # off the step grid
    "probe@nope=1",            # not a setting of this preset
    "probe@loud=yes",          # not a bool
    "probe@tone=siren",        # not an option
    "probe@wait_s",            # malformed suffix
])
def test_an_invalid_setting_is_refused_with_a_reason_and_nothing_changes(runtime, name):
    runtime._switch_preset(name)
    assert runtime.active_preset == "other"
    rejection = runtime._last_rejection
    assert rejection["kind"] == "preset" and rejection["reasons"]


def test_the_choice_survives_a_restart(runtime):
    runtime._switch_preset("probe@wait_s=3")
    assert device_state.saved_preset("m6_settings_probe") == "probe@wait_s=3"
    restored = restored_preset("m6_settings_probe", {"m6_settings_probe": {}}, PRESETS)
    assert restored is not None, "the saved choice must be restored"
    name, preset, overrides = restored
    assert name == "probe@wait_s=3" and overrides == {"wait_s": 3}
    assert preset["module"] == "m6_settings_probe"


def test_the_device_reports_what_it_understands(runtime):
    assert set(runtime._policy_state()["features"]) == set(DEVICE_POLICY_FEATURES)
    assert {"preset_settings", "dwell_in_zone"} <= set(DEVICE_POLICY_FEATURES)
