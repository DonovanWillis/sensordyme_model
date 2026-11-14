"""M8.5: the device's Python heap stays flat after warm-up (docs/MEMORY_PROFILE.md).

The M8.4 live soak saw the virtual device's RSS climb ~33 MB/h over its final
lifetime. The M8.5 profile attributes that to the C allocator, not to retained
Python objects: tracemalloc's heap and the object census stayed flat. These
tests keep it that way, deterministically, on the SimClock:

  * the real runtime (inference pass, monitoring check, uplink sync, alerts,
    stop/resume through the backend) runs minutes of busy device time; after a
    warm-up, the live gc-tracked objects and pymalloc blocks may not grow with
    every frame -- a retained object per frame is thousands here;
  * a leak injected on purpose IS caught (the measurement is not blind);
  * every per-event memory the device keeps (alert deliveries and episodes,
    uplink follow-ups, dashboard event and risk history) is capped: driven far
    past its cap, it stays at the cap.
"""

from __future__ import annotations

import gc
import sys
from datetime import UTC, datetime

import pytest

from sim.harness import SimStack
from sim.scenario import (
    ASLEEP,
    ATTENTIVE,
    CALIBRATION_S,
    LOOKING_AWAY,
    PHONE_IN_LAP,
    Scenario,
    Segment,
)
from sim.uplink import SimUplink

pytest.importorskip("fastapi")
pytest.importorskip("mediapipe")

C = CALIBRATION_S
#: One busy minute, repeated: drowsiness, a phone, a distraction.
MINUTE = ((0.0, ATTENTIVE), (5.0, ASLEEP), (9.0, ATTENTIVE), (20.0, PHONE_IN_LAP), (26.0, ATTENTIVE),
          (38.0, LOOKING_AWAY), (43.0, ATTENTIVE))
WARMUP_MIN, MEASURE_MIN = 2, 3
STEPS_PER_MIN = 60 * 20  # VirtualDevice.advance: one frame + inference pass every 0.05 s
#: Allowed growth over the measured minutes. The busy timeline itself fills a
#: few bounded caches (alert deliveries, uplink follow-ups: tens of objects a
#: minute until they hit their caps); one retained object per frame is
#: MEASURE_MIN * STEPS_PER_MIN = 3600.
MAX_OBJECT_GROWTH = 1200
MAX_BLOCK_GROWTH = 2400


def _busy(minutes: int) -> Scenario:
    segments = [Segment(0.0)]
    for m in range(minutes):
        segments += [Segment(C + 60.0 * m + off, scene) for off, scene in MINUTE]
    return Scenario("m85_busy", "busy minutes for the heap check", tuple(segments))


def _heap() -> tuple[int, int]:
    gc.collect()
    return len(gc.get_objects()), sys.getallocatedblocks()


def _drive(stack: SimStack, device, minutes: int) -> None:
    for _ in range(minutes):
        device.advance(30.0)
        assert stack.protection("stop", confirm=True).status_code == 200
        device.advance(10.0)
        assert stack.protection("resume").status_code == 200
        device.advance(20.0)


def _growth(tmp_path, leak: bool = False) -> tuple[int, int, int]:
    SimUplink.LEDGER = None  # the harness's own event ledger grows by design
    with SimStack(tmp_path) as stack:
        device = stack.boot(_busy(WARMUP_MIN + MEASURE_MIN + 1))
        stack.claim()
        retained: list = []
        runtime = device.runtime
        assert runtime is not None
        if leak:  # one small list (gc-tracked, unlike a tuple of atoms) kept per inference pass
            inference_pass = runtime._inference_pass

            def leaky(*args, **kwargs):
                retained.append([len(retained)])
                return inference_pass(*args, **kwargs)

            runtime._inference_pass = leaky  # type: ignore[method-assign]
        device.advance(C)
        _drive(stack, device, WARMUP_MIN)
        objects0, blocks0 = _heap()
        _drive(stack, device, MEASURE_MIN)
        objects1, blocks1 = _heap()
        alerts = len(device.buzzer.played)
    return objects1 - objects0, blocks1 - blocks0, alerts


def test_the_device_python_heap_is_flat_after_warmup(tmp_path):
    objects, blocks, alerts = _growth(tmp_path)
    assert alerts >= 2 * (WARMUP_MIN + MEASURE_MIN), "the timeline raised no alerts: the test proves nothing"
    print(f"after warm-up, {MEASURE_MIN} busy min: objects {objects:+d}, pymalloc blocks {blocks:+d}, alerts {alerts}")
    assert objects < MAX_OBJECT_GROWTH, objects
    assert blocks < MAX_BLOCK_GROWTH, blocks


def test_the_heap_check_catches_one_object_retained_per_frame(tmp_path):
    objects, blocks, _ = _growth(tmp_path, leak=True)
    assert objects >= MAX_OBJECT_GROWTH and blocks >= MAX_BLOCK_GROWTH, (objects, blocks)


def _live(*names: str) -> dict[str, int]:
    counts = dict.fromkeys(names, 0)
    for obj in gc.get_objects():
        name = type(obj).__name__
        if name in counts:
            counts[name] += 1
    return counts


def test_a_restarted_runtime_releases_the_previous_one(tmp_path):
    """The live soak's RSS steps up at every in-process runtime rebuild (a scenario
    reboot). Nothing of the old runtime may stay reachable -- the step is the C
    allocator's, not a retained runtime, module, uplink or alert policy."""
    SimUplink.LEDGER = None
    names = ("SimRuntime", "DriverAttentionModule", "SimUplink", "AlertManager", "PerceptionEngine",
             "FrameBus", "HealthMonitor")
    with SimStack(tmp_path) as stack:
        scenario = _busy(2)
        device = stack.boot(scenario)
        stack.claim()
        device.advance(C + 30.0)
        counts = []
        for _ in range(3):
            device = stack.restart(scenario)
            device.advance(20.0)
            counts.append(_heap()[0])
            assert _live(*names) == dict.fromkeys(names, 1), _live(*names)
    assert max(counts) - min(counts) < 300, counts


# ---------------------------------------------------------------------- per-event caps
def _event(i: int, kind: str = "drowsiness"):
    from core.events import Event

    return Event(timestamp=datetime.now(UTC), type=kind, duration=1.0, confidence=0.9,
                 module_name="driver_attention")


def test_alert_policy_memory_is_capped_however_many_alerts():
    from core.alerts import AlertManager

    manager = AlertManager(sinks=[])
    n = 3 * manager._DELIVERY_MEMORY
    for i in range(n):
        manager.notify(_event(i, ("drowsiness", "phone_use", "distraction")[i % 3]), now=10.0 * i)
        if i % 7 == 0:
            manager.end_all(10.0 * i + 1)
    manager.end_all(10.0 * n)
    assert len(manager._deliveries) == manager._DELIVERY_MEMORY
    assert len(manager._recent) == manager._RECENT
    assert manager._open == {} and len(manager._last_start) <= 3


def test_uplink_alert_memory_is_capped_however_many_alerts():
    from core import uplink as up
    from core.alerts import AlertDelivery

    uplink = up.CloudUplink({"base_url": "https://backend.example", "device_key": "dk_x", "queue_max": 50})
    sink = object()
    deliveries = []
    for i in range(3 * up._ALERT_FOLLOWUP_MEMORY):
        delivery = AlertDelivery("alerted", f"boot:{i}", "critical")
        delivery.commanded(sink, reports=True)  # an output still owes its report: not final
        deliveries.append(delivery)
        uplink.push_event(_event(i), "critical", alert=delivery)
    assert len(uplink._alert_followups) == up._ALERT_FOLLOWUP_MEMORY
    assert len(uplink._episodes_queued) == up._ALERT_FOLLOWUP_MEMORY
    assert len(uplink._queue) == 50 and uplink.dropped_events == len(deliveries) - 50
    assert all(len(d._listeners) == 1 for d in deliveries)  # one per event, released with the delivery


def test_dashboard_history_is_capped(tmp_path):
    from core import dashboard as db

    dash = db.DashboardServer(host="127.0.0.1", port=0, module_name="driver_attention", token="t" * 24,
                              events_csv_path=str(tmp_path / "events.csv"))
    for i in range(3 * db._MAX_EVENTS):
        dash.push_event(_event(i), severity="loud")
    assert len(dash._events) == db._MAX_EVENTS
    assert dash._risk_series.maxlen == db._MAX_RISK_POINTS
