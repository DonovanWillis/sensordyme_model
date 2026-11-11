"""M6 security: what the device accepts from the backend link, and what it logs.

* a protection (stop/resume) command id is applied at most once -- a lost
  response or a replay re-delivers it, and a stale stop must not silently stop
  monitoring again after the owner resumed it;
* malformed command fields are refused at the uplink, not handed to the main loop;
* a plain-http uplink to a network host is announced, once;
* the device key never reaches the log, on any failure path.
"""

import types

import pytest
import requests

from core.uplink import CloudUplink, cleartext_remote

KEY = "dk_0123456789abcdef0123456789abcdef"


class _Once:
    def __init__(self, commands):
        self.commands = list(commands)

    def is_set(self):
        return not self.commands

    def wait(self, timeout=None):
        return self.is_set()

    def set(self):
        self.commands = []


def _uplink(commands, base_url="https://backend.example"):
    uplink = CloudUplink({"base_url": base_url, "device_key": KEY, "poll_interval_s": 0})
    stop = _Once(commands)

    def post(url, json=None, **_kwargs):
        command = stop.commands.pop(0)
        return types.SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                                     json=lambda: {"command": command})

    uplink._requests = types.SimpleNamespace(post=post)
    uplink._stop = stop
    return uplink


def test_a_protection_command_id_is_applied_once():
    stop = {"set_protection": {"action": "stop", "confirm": True, "id": "a1"}}
    uplink = _uplink([stop])
    uplink._run()
    assert uplink.pop_pending_protection() == {"action": "stop", "confirm": True, "id": "a1"}
    # The owner resumes locally; the same stop arrives again (re-sent or replayed).
    uplink._stop.commands = [stop]
    uplink._run()
    assert uplink.pop_pending_protection() is None, "a stale stop re-applied"


def test_distinct_ids_are_each_applied():
    uplink = _uplink([{"set_protection": {"action": "stop", "confirm": True, "id": "a1"}}])
    uplink._run()
    assert uplink.pop_pending_protection()["id"] == "a1"
    uplink._stop.commands = [{"set_protection": {"action": "resume", "id": "b2"}}]
    uplink._run()
    assert uplink.pop_pending_protection() == {"action": "resume", "confirm": False, "id": "b2"}


def test_a_command_without_an_id_is_still_accepted_for_older_backends():
    uplink = _uplink([{"set_protection": {"action": "resume"}}])
    uplink._run()
    assert uplink.pop_pending_protection() == {"action": "resume", "confirm": False, "id": None}


def test_the_seen_id_memory_is_bounded():
    uplink = _uplink([{"set_protection": {"action": "resume", "id": f"id{i}"}} for i in range(500)])
    uplink._run()
    assert len(uplink._seen_command_ids) <= 64


@pytest.mark.parametrize("bad", [{"name": "x"}, ["driver_safety"], 7, "p" * 500])
def test_a_malformed_set_preset_is_ignored(bad):
    uplink = _uplink([{"set_preset": bad}])
    uplink._run()
    assert uplink.pop_pending_preset() is None


def test_a_valid_set_preset_still_arrives():
    uplink = _uplink([{"set_preset": "driver_safety"}])
    uplink._run()
    assert uplink.pop_pending_preset() == "driver_safety"


def test_a_non_mapping_command_is_ignored():
    uplink = _uplink([["set_preset", "x"]])
    uplink._run()
    assert uplink.pop_pending_preset() is None and uplink.connected


@pytest.mark.parametrize(("url", "cleartext"), [
    ("http://backend.local:8787", True),
    ("http://192.168.1.20:8787", True),
    ("HTTP://Backend.example", True),
    ("http://localhost:8787", False),
    ("http://127.0.0.1:8787", False),
    ("https://backend.example", False),
    ("", False),
])
def test_cleartext_remote(url, cleartext):
    assert cleartext_remote(url) is cleartext


def test_a_cleartext_uplink_is_announced_once(capsys):
    CloudUplink({"base_url": "http://backend.local:8787", "device_key": KEY})
    out = capsys.readouterr().out
    assert out.count("plain http://") == 1
    CloudUplink({"base_url": "https://backend.example", "device_key": KEY})
    assert "plain http://" not in capsys.readouterr().out


@pytest.mark.parametrize("failure", ["connect", 401, 500, "redirect"])
def test_the_device_key_never_reaches_the_log(capsys, failure):
    uplink = CloudUplink({"base_url": "http://backend.local:8787", "device_key": KEY, "poll_interval_s": 0})
    calls = {"n": 0}

    def post(url, json=None, headers=None, **_kwargs):
        calls["n"] += 1
        uplink._stop.set()
        if failure == "connect":
            raise requests.ConnectionError(f"cannot connect to {url}")
        if failure == "redirect":
            return types.SimpleNamespace(status_code=302, headers={"Location": "http://evil.example/"},
                                         raise_for_status=lambda: None, json=dict)
        error = requests.HTTPError(f"{failure} error")
        error.response = types.SimpleNamespace(status_code=failure)
        raise error

    uplink._requests = types.SimpleNamespace(post=post)
    uplink._run()
    assert calls["n"] == 1
    assert KEY not in capsys.readouterr().out
