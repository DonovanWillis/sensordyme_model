"""M13: the `showcase` scenario (sim/scenario.py) and `/fresh?scenario=` (scripts/sim/uitest_control.py).

Presentation mode's XCUITest journey and its labelled fallback recording need a
virtual device that keeps producing a simulated distraction while someone
watches: `showcase` repeats one every 40 s (a phone in the lap every third
time). Pinned here: it calibrates first like every driver scenario, each look
away lasts at least as long as the `distracted` scenario's (which
tests/test_sim_e2e.py proves raises a distraction), it repeats for the length of
a demo, and the controller passes the scenario through to the device.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

from sim import scenario as sc

REPO = Path(__file__).resolve().parents[1]


def _intervals(scenario: sc.Scenario, scene: object) -> list[tuple[float, float]]:
    segments = scenario.segments
    return [(s.start, segments[i + 1].start) for i, s in enumerate(segments[:-1]) if s.scene == scene]


def test_showcase_is_listed_and_well_formed():
    show = sc.get("showcase")
    assert show.description
    starts = [s.start for s in show.segments]
    assert starts == sorted(starts) and starts[0] == 0.0
    assert show.at(0.0).scene == sc.ATTENTIVE
    assert min(s for s in starts if s > 0) >= sc.CALIBRATION_S, "nothing changes during calibration"


def test_showcase_repeats_a_distraction_long_enough_to_alert():
    show = sc.get("showcase")
    away = _intervals(show, sc.LOOKING_AWAY)
    phone = _intervals(show, sc.PHONE_IN_LAP)
    distracted = _intervals(sc.get("distracted"), sc.LOOKING_AWAY)[0]
    assert len(away) == 40 and len(phone) == 20, "60 cycles: two looks away for every phone"
    assert all(end - start >= distracted[1] - distracted[0] for start, end in away)
    first = away[0][0]
    assert first == sc.CALIBRATION_S + 18.0, "the first one comes ~30 s after boot"
    assert show.at(first + 3.0).scene == sc.LOOKING_AWAY
    assert show.at(first + 8.0).scene == sc.ATTENTIVE, "and ends: the driver looks back"
    assert show.at(sc.CALIBRATION_S + 2 * 40.0 + 20.0).scene == sc.PHONE_IN_LAP
    assert show.duration >= 30 * 60, "long enough for a demo"


def test_the_controller_starts_fresh_with_a_scenario():
    path = REPO / "scripts" / "sim" / "uitest_control.py"
    spec = importlib.util.spec_from_file_location("uitest_control_m13", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    params = inspect.signature(module.Stack.fresh).parameters
    assert params["scenario"].default == "attentive", "every other journey keeps the attentive device"
    source = path.read_text()
    assert 'scenario=q.get("scenario", "attentive")' in source
    assert "self.start_device(scenario)" in source
