"""M8.3: stop / resume end to end, with an explicit, acknowledged outcome every time.

Real runtime, uplink and backend on the virtual device (sim/), deterministic
clock, the InvariantMonitor attached throughout (sim/invariants.py). What the
app reads is `status.protection_command` (docs/APP_STATUS_CONTRACT.md):

  pending    -> confirmed  the device echoed THIS id with the requested state
             -> mismatch   the device echoed it with another state
             -> timed_out  no echo within 45 s; a stop is withdrawn after 30 s
                           (never delivered late), a resume is still delivered
"""

from __future__ import annotations

import json

import pytest

from core import state_model
from sim.harness import SimStack
from sim.invariants import InvariantMonitor
from sim.scenario import ATTENTIVE, CALIBRATION_S, Faults, Scenario, Segment

C = CALIBRATION_S
ATTENTIVE_FOREVER = Scenario("attentive", "attentive", (Segment(0.0),))


@pytest.fixture
def stack(tmp_path):
    with SimStack(tmp_path) as s:
        yield s


def _ready(stack) -> tuple:
    monitor = InvariantMonitor(stack)
    device = stack.boot(ATTENTIVE_FOREVER)
    stack.claim()
    device.advance(C + 4.0)
    assert stack.status()["monitoring_state"] == "ready"
    return monitor, device


def _command(stack) -> dict:
    return stack.status()["protection_command"]


def test_stop_and_resume_are_confirmed_only_by_the_devices_echo(stack):
    monitor, device = _ready(stack)
    response = stack.protection("stop", confirm=True)
    assert response.status_code == 200
    command_id = response.json()["command_id"]
    assert response.json()["timeout_s"] == 45.0
    # Stored, not applied: never a success before the device answers.
    command = _command(stack)
    assert command["id"] == command_id and command["outcome"] == "pending" and command["delivering"] is True
    assert stack.status()["protection"] == "protecting"

    device.advance(6.0)
    status = stack.status()
    command = status["protection_command"]
    assert command["outcome"] == "confirmed" and command["reported_state"] == "stopped"
    assert command["applied_at"] is not None and command["delivering"] is False
    assert status["protection"] == "stopped" and status["last_command_id"] == command_id
    assert state_model.derive(stack.row(), stack.clock.wall()).state == state_model.STOPPED

    resume_id = stack.protection("resume").json()["command_id"]
    row = stack.row()
    assert state_model.derive(row, stack.clock.wall()).state == state_model.RESUMING, \
        "stopped with a resume not yet applied is resuming, never ready"
    device.advance(6.0)
    command = _command(stack)
    assert command["id"] == resume_id and command["outcome"] == "confirmed"
    assert command["reported_state"] in ("protecting", "not_protecting")
    device.advance(C + 10.0)
    assert stack.status()["monitoring_state"] == "ready"
    assert monitor.protection_effects[command_id] == 1 and monitor.protection_effects[resume_id] == 1
    assert state_model.STOPPED in monitor.user_states and monitor.user_states[-1] == state_model.READY
    monitor.assert_clean()


def test_a_stop_the_device_cannot_receive_times_out_and_is_never_applied_late(stack):
    monitor, device = _ready(stack)
    device.state.fault_overrides["network_down"] = True
    command_id = stack.protection("stop", confirm=True).json()["command_id"]
    device.advance(32.0)
    command = _command(stack)
    assert command["outcome"] == "pending" and command["delivering"] is False, \
        "past 30 s the stop is no longer delivered (the next poll withdraws it), and not yet timed out"
    device.state.fault_overrides["network_down"] = False
    device.advance(16.0)  # back online, past the 45 s acknowledgement timeout
    status = stack.status()
    command = status["protection_command"]
    assert command["id"] == command_id and command["outcome"] == "timed_out"
    assert command["delivering"] is False
    assert status["protection"] == "protecting", "a stop withdrawn after 30 s never stops the device later"
    device.advance(20.0)
    assert stack.status()["protection"] == "protecting" and device.runtime._stopped is None
    assert monitor.protection_effects[command_id] == 0
    monitor.assert_clean()


def test_a_resume_that_times_out_says_still_stopped_and_still_applies_when_the_device_returns(stack):
    monitor, device = _ready(stack)
    stack.protection("stop", confirm=True)
    device.advance(6.0)
    assert stack.status()["protection"] == "stopped"
    device.state.fault_overrides["network_down"] = True
    resume_id = stack.protection("resume").json()["command_id"]
    device.advance(50.0)
    row = stack.row()
    command = row["status"]["protection_command"]
    assert command["outcome"] == "timed_out" and command["delivering"] is True
    assert row["online"] is False and row["status"]["protection"] == "unknown", \
        "a silent device's protection is unknown, never its last stopped or a hoped-for protecting"
    assert device.runtime._stopped is not None, "the device itself is still stopped"
    device.state.fault_overrides["network_down"] = False
    device.advance(6.0)
    command = _command(stack)
    assert command["id"] == resume_id and command["outcome"] == "confirmed", \
        "late protection is the safe direction: the resume still applies, and is then confirmed"
    assert device.runtime._stopped is None
    monitor.assert_clean()


def test_the_device_confirming_a_different_state_is_a_mismatch_with_its_reason(stack):
    """The device echoes the id of a request it refused. A stop without confirm
    cannot come from POST .../protection; this is an older backend or a damaged
    row, injected into the same column the endpoint writes."""
    monitor, device = _ready(stack)
    command_id = stack.protection("stop", confirm=True).json()["command_id"]
    with stack.backend.app.db() as conn:
        conn.execute("UPDATE devices SET desired_protection = ? WHERE id = ?",
                     (json.dumps({"action": "stop", "confirm": False, "id": command_id,
                                  "requested_at": stack.clock.wall()}), stack.device_id))
    device.advance(6.0)
    status = stack.status()
    command = status["protection_command"]
    assert status["last_command_id"] == command_id, "the device answered this request"
    assert status["protection"] == "protecting"
    assert command["outcome"] == "mismatch" and command["reported_state"] == "protecting"
    assert "confirm" in command["reason"], command
    monitor.assert_clean()


def test_a_later_local_change_does_not_rewrite_what_the_request_did(stack):
    """The outcome is latched from the first echo: a restart (stops are not
    persisted, so the device comes back protecting) is the status's news, not
    the stop's."""
    monitor, device = _ready(stack)
    command_id = stack.protection("stop", confirm=True).json()["command_id"]
    device.advance(6.0)
    assert _command(stack)["outcome"] == "confirmed"
    device = stack.restart(ATTENTIVE_FOREVER)
    device.advance(C + 6.0)
    status = stack.status()
    assert status["protection"] == "protecting" and status["last_command_id"] is None
    assert status["protection_command"]["id"] == command_id
    assert status["protection_command"]["outcome"] == "confirmed"
    assert monitor.boots == 2
    monitor.assert_clean()


def test_a_newer_request_supersedes_and_duplicates_take_effect_once(stack):
    monitor, device = _ready(stack)
    first = stack.protection("stop", confirm=True).json()["command_id"]
    second = stack.protection("stop", confirm=True).json()["command_id"]
    assert _command(stack)["id"] == second
    device.advance(6.0)
    assert _command(stack)["outcome"] == "confirmed"
    # The first was replaced before delivery: it never took effect on its own.
    assert monitor.protection_effects[first] == 0 and monitor.protection_effects[second] == 1
    device.advance(10.0)
    assert monitor.protection_effects[second] == 1, "re-delivery never doubles the effect"
    monitor.assert_clean()


def test_stop_without_confirmation_is_refused_before_anything_is_stored(stack):
    monitor, device = _ready(stack)
    response = stack.protection("stop", confirm=False)
    assert response.status_code == 400 and response.json()["code"] == "confirmation_required"
    assert _command(stack) is None
    device.advance(4.0)
    assert stack.status()["protection"] == "protecting"
    monitor.assert_clean()


def test_a_frozen_link_during_a_stop_never_shows_stopped_from_old_evidence(stack):
    monitor, device = _ready(stack)
    stack.protection("stop", confirm=True)
    device.advance(6.0)
    assert stack.status()["protection"] == "stopped"
    device.state.fault_overrides["network_down"] = True
    device.advance(35.0)
    row = stack.row()
    assert row["online"] is False and row["status"]["protection"] == "unknown"
    assert state_model.derive(row, stack.clock.wall()).state == state_model.OFFLINE
    assert row["status"]["protection_command"]["outcome"] == "confirmed", \
        "the record of what the stop did stays; the CURRENT state is unknown"
    monitor.assert_clean()


def test_faults_hook_is_inert_by_default():
    assert Faults().soc_temp_c is None and Faults().throttle == ()
    assert Scenario("x", "x").at(0.0).scene == ATTENTIVE
