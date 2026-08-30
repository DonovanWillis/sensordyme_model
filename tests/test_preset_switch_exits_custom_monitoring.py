"""The bug this reproduces: switching presets used to leave custom monitoring
running underneath the newly-selected preset.

`_switch_preset()` swapped `self.module` to the chosen preset but never touched
`self.monitoring` (the saved custom rules + `overrides_mode`). Since
`self.monitoring.active` drives `_apply_monitoring_mode()`, and that runs on
every rules/zones poll from the cloud (every few seconds), the very next poll
would see `monitoring.active` still true and swap the device straight back to
AI_CUSTOM/occupancy — silently reverting the mode switch the user just made.
An explicit preset choice must always win over whatever custom rules were
running before it, immediately and durably.
"""

import numpy as np

from core.alerts import AlertManager
from core.events import Event
from core.logger import EventLogger
from core.monitoring_mode import MonitoringMode
from modules.base import VisionModule, register_module
from services.runtime import DeviceRuntime


@register_module("preset_switch_test_a")
class _ModuleA(VisionModule):
    name = "preset_switch_test_a"

    def process(self, frame: np.ndarray) -> list[Event]:
        return []


@register_module("preset_switch_test_b")
class _ModuleB(VisionModule):
    name = "preset_switch_test_b"

    def process(self, frame: np.ndarray) -> list[Event]:
        return []


_PRESETS = {
    "mode_a": {"title": "Mode A", "module": "preset_switch_test_a"},
    "mode_b": {"title": "Mode B", "module": "preset_switch_test_b"},
}


def _runtime(tmp_path) -> DeviceRuntime:
    config = {
        "modules": {"preset_switch_test_a": {}, "preset_switch_test_b": {}},
    }
    runtime = DeviceRuntime(
        config=config,
        profile="pi",
        module=_ModuleA({}),
        module_name="preset_switch_test_a",
        alert_manager=AlertManager(sinks=[]),
        alert_cfg={},
        alert_event_types=set(),
        alert_types_for=lambda name, preset=None: set(),
        presets=_PRESETS,
        active_preset="mode_a",
        display=False,
        debug=False,
    )
    # _switch_preset logs a switch event; only _build() wires a real logger.
    runtime.logger = EventLogger(str(tmp_path / "events.csv"))
    return runtime


def test_switching_preset_clears_active_custom_monitoring(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.monitoring.save(
        [{"target": "cell phone", "predicate": "present"}],
        instruction="watch for phones",
        overrides_mode=True,
    )
    assert runtime.monitoring.active is True

    runtime._switch_preset("mode_b")

    assert runtime.monitoring.active is False, "custom monitoring survived an explicit preset switch"
    assert runtime.rule_engine.rules == []
    assert runtime.mode is MonitoringMode.STANDARD
    assert runtime.module_name == "preset_switch_test_b"
    assert runtime.active_preset == "mode_b"


def test_switching_preset_does_not_revert_on_the_next_mode_apply(tmp_path):
    """The actual failure mode: a later _apply_monitoring_mode() call (which
    every rules/zones poll triggers) must not see monitoring.active still true
    and swap the device back to AI_CUSTOM."""
    runtime = _runtime(tmp_path)
    runtime.monitoring.save([{"target": "cell phone", "predicate": "present"}], overrides_mode=True)

    runtime._switch_preset("mode_b")
    runtime._apply_monitoring_mode()  # what a subsequent rules/zones poll calls

    assert runtime.mode is MonitoringMode.STANDARD
    assert runtime.module_name == "preset_switch_test_b", "mode reverted to AI_CUSTOM on the next poll"


def test_switching_between_two_presets_with_no_custom_monitoring_is_unaffected(tmp_path):
    """The ordinary case, unchanged: plain preset-to-preset switching still works
    and does not touch (or write) the monitoring file when nothing was active."""
    runtime = _runtime(tmp_path)
    assert runtime.monitoring.active is False

    runtime._switch_preset("mode_b")

    assert runtime.module_name == "preset_switch_test_b"
    assert runtime.mode is MonitoringMode.STANDARD
    assert runtime.monitoring.active is False


def test_exit_custom_monitoring_is_a_noop_when_nothing_is_active(tmp_path):
    runtime = _runtime(tmp_path)
    before = runtime.monitoring.updated_at
    runtime._exit_custom_monitoring()
    assert runtime.monitoring.updated_at == before  # no spurious write
