"""Direct mode (the device's own access point): docs/DIRECT_MODE.md, R2, R3, G5, G6.

Nothing here touches a real network. nmcli is tests/direct_mode_fakes.FakeNM
behind core.network._run, which also plays the root dispatcher script's part
(the firewall marker). Passphrases are generated at run time.
"""

from __future__ import annotations

import json
import logging
import os
import threading

import pytest
from direct_mode_fakes import FakeNM

import core.network as network
from core import direct_mode as dm
from core.direct_mode import DirectModeManager, PskError
from core.network import DIRECT_PROFILE, WifiManager

DEVICE = "SD-1A2B3C4D"
SSID = "SensorDyme-1A2B"


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """A device with a saved home network, a label passphrase and the firewall installed."""
    marker = str(tmp_path / "run" / "firewall-active")
    nm = FakeNM(marker=marker)
    nm.add_client("Home", active=True)
    monkeypatch.setattr(network, "nmcli_available", lambda: True)
    monkeypatch.setattr(network, "_run", nm.run)
    psk = dm.generate_psk()
    psk_path = str(tmp_path / "direct_mode_psk")
    dm.write_psk(psk_path, psk)
    firewall = []
    for name in ("direct_mode.nft", "90-sensordyme-direct"):
        (tmp_path / name).write_text("installed\n")
        firewall.append(str(tmp_path / name))
    clock = Clock()

    def make(**overrides) -> DirectModeManager:
        config = {"enabled": True, "psk_path": psk_path, "state_path": str(tmp_path / "state.json"),
                  "ap_timeout_s": 5, **overrides}
        return DirectModeManager(WifiManager(), config, device_id=DEVICE, clock=clock, monotonic=clock,
                                 sleep=clock.sleep, has_nmcli=lambda: True, firewall_marker=marker,
                                 firewall_files=tuple(firewall))

    return types_ns(nm=nm, psk=psk, psk_path=psk_path, make=make, clock=clock, tmp=tmp_path, marker=marker,
                    firewall=firewall)


def types_ns(**kw):
    import types
    return types.SimpleNamespace(**kw)


# ============================================================================ the label passphrase (G6)
def test_a_generated_passphrase_is_random_long_and_typable():
    seen = {dm.generate_psk() for _ in range(200)}
    assert len(seen) == 200
    for psk in seen:
        assert dm.check_psk(psk) == psk
        assert len(psk) == 19 and 8 <= len(psk) <= 63  # a valid WPA passphrase
        assert set(psk) <= set(dm.PSK_ALPHABET) | {"-"}
        assert not set(psk) & set("0o1liI"), "no look-alike characters on a printed label"


def test_the_passphrase_is_not_derived_from_the_board(monkeypatch):
    """The MAC and serial are broadcast; a passphrase made from them is public."""
    import core.identity as identity
    monkeypatch.setattr(identity, "_cpu_serial", lambda: "10000000deadbeef")
    monkeypatch.setattr(identity, "_mac_address", lambda: "d8:3a:dd:00:11:22")
    assert dm.generate_psk() != dm.generate_psk()


def test_the_passphrase_file_is_written_once_0600_and_loaded_only_if_private(tmp_path):
    path = str(tmp_path / "psk")
    psk = dm.generate_psk()
    dm.write_psk(path, psk)
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert dm.load_psk(path) == psk
    with pytest.raises(PskError, match="already exists"):
        dm.write_psk(path, dm.generate_psk())
    assert dm.load_psk(path) == psk, "an existing label is never replaced"

    os.chmod(path, 0o640)
    with pytest.raises(PskError, match="0600") as caught:
        dm.load_psk(path)
    assert psk not in str(caught.value)

    link = tmp_path / "link"
    os.chmod(path, 0o600)
    link.symlink_to(path)
    with pytest.raises(PskError, match="regular file"):
        dm.load_psk(str(link))
    with pytest.raises(PskError, match="no direct-mode passphrase"):
        dm.load_psk(str(tmp_path / "missing"))


@pytest.mark.parametrize("bad", ["", "password", "abcd-efgh-jkmn", "ABCD-EFGH-JKMN-PQRS", "abcd-efgh-jkmn-pqr0"])
def test_a_file_that_is_not_a_generated_passphrase_is_refused_without_echoing_it(tmp_path, bad):
    path = tmp_path / "psk"
    path.write_text(bad + "\n")
    path.chmod(0o600)
    with pytest.raises(PskError) as caught:
        dm.load_psk(str(path))
    assert not bad or bad not in str(caught.value)


def test_the_label_carries_the_ssid_the_passphrase_and_a_joinable_qr():
    psk = dm.generate_psk()
    text = dm.label_text(DEVICE, psk)
    assert SSID in text and psk in text
    assert f"WIFI:T:WPA;S:{SSID};P:{psk};;" in text
    assert dm.wifi_qr_payload('a;b:c"d', "p\\w") == 'WIFI:T:WPA;S:a\\;b\\:c\\"d;P:p\\\\w;;'


# ============================================================================ the profile (G5)
def test_the_access_point_profile_is_wpa2_ccmp_shared_and_never_autoconnects():
    psk = dm.generate_psk()
    args = network.ap_profile_args(SSID, psk)
    pairs = dict(zip(args[10::2], args[11::2], strict=True))
    assert args[:10] == ["connection", "add", "type", "wifi", "con-name", DIRECT_PROFILE,
                         "ifname", "wlan0", "ssid", SSID]
    assert pairs == {
        "connection.autoconnect": "no",
        "802-11-wireless.mode": "ap", "802-11-wireless.band": "bg", "802-11-wireless.channel": "6",
        "ipv4.method": "shared", "ipv4.addresses": "10.42.0.1/24", "ipv6.method": "disabled",
        "wifi-sec.key-mgmt": "wpa-psk", "wifi-sec.proto": "rsn",
        "wifi-sec.pairwise": "ccmp", "wifi-sec.group": "ccmp",
        "wifi-sec.psk": psk,
    }
    assert psk not in network.redacted_args(args)


def test_wpa3_is_sae_with_management_frame_protection():
    args = network.ap_profile_args(SSID, dm.generate_psk(), security="wpa3")
    pairs = dict(zip(args[10::2], args[11::2], strict=True))
    assert pairs["wifi-sec.key-mgmt"] == "sae"
    assert pairs["wifi-sec.pmf"] == "required"


@pytest.mark.parametrize(("security", "psk"), [
    ("none", "abcdefgh"), ("open", "abcdefgh"), ("", "abcdefgh"),
    ("wpa2", ""), ("wpa2", "short"), ("wpa2", "x" * 64), ("wpa2", "tab\there!"), ("wpa2", "ünïcödé!"),
])
def test_no_code_path_can_build_an_open_or_unusable_access_point(security, psk):
    with pytest.raises(ValueError, match=r"security must be one of|WPA passphrase"):
        network.ap_profile_args(SSID, psk, security=security)


def test_the_reserved_profile_name_cannot_be_provisioned_as_a_client_network():
    assert "reserved" in (network.credential_problem(DIRECT_PROFILE, "password1") or "")
    assert network.credential_problem("SensorDyme-1A2B", "password1") is None


def test_client_networks_exclude_the_access_point(env):
    env.nm.profiles[DIRECT_PROFILE] = {}
    assert WifiManager().client_networks() == ["Home"]
    assert DIRECT_PROFILE in WifiManager().saved_networks()


# ============================================================================ enter (R2)
def test_enter_starts_the_access_point_and_remembers_the_client_network(env):
    manager = env.make()
    assert manager.status()["mode"] == "client"
    ok, message = manager.enter("technician")
    assert ok, message
    assert SSID in message
    assert env.nm.active == DIRECT_PROFILE
    profile = env.nm.profiles[DIRECT_PROFILE]
    assert profile["802-11-wireless-security.psk"] == env.psk
    assert profile["802-11-wireless.ssid"] == SSID
    state = json.loads((env.tmp / "state.json").read_text())
    assert state["previous"] == "Home" and state["mode"] == "direct" and state["by"] == "explicit"
    assert env.psk not in json.dumps(state)
    status = manager.status()
    assert status == {"mode": "direct", "ssid": SSID, "since": env.clock.t, "last_error": None,
                      "fallback_enabled": False}


def test_entering_twice_is_a_no_op(env):
    manager = env.make()
    assert manager.enter("first")[0]
    before = len(env.nm.mutating())
    ok, message = manager.enter("second")
    assert ok and "already" in message
    assert len(env.nm.mutating()) == before


def test_an_access_point_that_does_not_come_up_rolls_back_to_the_previous_network(env):
    env.nm.ap_up_fails = True
    manager = env.make()
    ok, message = manager.enter("technician")
    assert not ok
    assert "returned to 'Home'" in message
    assert env.nm.active == "Home"
    status = manager.status()
    assert status["mode"] == "client"
    assert "AP mode not supported" in status["last_error"]
    assert not (env.tmp / "state.json").exists()


def test_an_access_point_without_its_firewall_is_rolled_back(env):
    """Fail closed: an AP whose firewall did not load must not stay up (G7)."""
    env.nm.firewall_loads = False
    manager = env.make(ap_timeout_s=3)
    ok, message = manager.enter("technician")
    assert not ok
    assert "firewall did not load within 3s" in message
    assert env.nm.active == "Home"
    assert ["connection", "down", "id", DIRECT_PROFILE] in env.nm.calls
    assert env.clock.t >= 1003, "it waited the full timeout before giving up"


def test_an_access_point_without_an_address_is_rolled_back(env):
    env.nm.ap_gets_ip = False
    ok, message = env.make(ap_timeout_s=2).enter("technician")
    assert not ok and "no address 10.42.0.1" in message
    assert env.nm.active == "Home"


def test_a_device_with_no_previous_network_says_so_on_rollback(env):
    env.nm.active = None
    env.nm.ap_up_fails = True
    ok, message = env.make().enter("technician")
    assert not ok
    assert "no previous network" in message
    assert env.nm.active is None


def test_a_profile_that_reads_back_open_is_deleted_and_never_brought_up(env):
    """If NetworkManager saved anything other than a WPA AP, it never broadcasts."""
    env.nm.add_drops = ("802-11-wireless-security.key-mgmt",)
    ok, message = env.make().enter("technician")
    assert not ok
    assert "key-mgmt" in message
    assert ["connection", "up", "id", DIRECT_PROFILE] not in env.nm.calls
    assert DIRECT_PROFILE not in env.nm.profiles
    assert env.nm.active == "Home", "the client network was never released"


def test_a_hand_edited_profile_is_replaced_not_trusted(env):
    env.nm.profiles[DIRECT_PROFILE] = {"connection.autoconnect": "yes", "802-11-wireless.mode": "ap",
                                       "802-11-wireless-security.key-mgmt": ""}
    assert env.make().enter("technician")[0]
    assert ["connection", "delete", "id", DIRECT_PROFILE] in env.nm.calls
    assert env.nm.profiles[DIRECT_PROFILE]["connection.autoconnect"] == "no"
    assert env.nm.profiles[DIRECT_PROFILE]["802-11-wireless-security.key-mgmt"] == "wpa-psk"


@pytest.mark.parametrize("breakage", ["disabled", "no_psk", "loose_psk", "no_firewall", "no_nmcli"])
def test_enter_refuses_and_touches_nothing_when_it_cannot_be_safe(env, breakage):
    overrides = {}
    if breakage == "disabled":
        overrides["enabled"] = False
    elif breakage == "no_psk":
        os.unlink(env.psk_path)
    elif breakage == "loose_psk":
        os.chmod(env.psk_path, 0o644)
    elif breakage == "no_firewall":
        os.unlink(env.firewall[1])
    manager = env.make(**overrides)
    if breakage == "no_nmcli":
        manager._has_nmcli = lambda: False
    ok, message = manager.enter("technician")
    assert not ok
    assert env.nm.mutating() == [], "nothing was changed"
    assert env.nm.active == "Home"
    status = manager.status()
    assert status["mode"] == "unavailable"
    assert status["last_error"]
    assert env.psk not in json.dumps(status) and env.psk not in message


def test_only_one_change_runs_at_a_time(env, monkeypatch):
    manager = env.make()
    entered = threading.Event()
    release = threading.Event()
    original = env.nm.run

    def slow(args, timeout=10):
        if args[:2] == ["connection", "up"]:
            entered.set()
            release.wait(2)
        return original(args, timeout)

    monkeypatch.setattr(network, "_run", slow)
    worker = threading.Thread(target=manager.enter, args=("first",))
    worker.start()
    assert entered.wait(2)
    assert manager.status()["mode"] == "entering"
    assert manager.enter("second") == (False, "a direct-mode change is already in progress")
    assert manager.leave("now") == (False, "a direct-mode change is already in progress")
    release.set()
    worker.join(2)
    assert manager.status()["mode"] == "direct"


def test_the_passphrase_is_never_logged_or_reported(env, caplog):
    env.nm.firewall_loads = False
    manager = env.make(ap_timeout_s=1)
    with caplog.at_level(logging.DEBUG):
        manager.enter("one")
        env.nm.firewall_loads = True
        manager.enter("two")
        manager.status()
        manager.leave("three")
        plan = "\n".join(manager.plan_enter() + manager.plan_leave())
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert env.psk not in logged
    assert env.psk not in plan
    assert "<redacted>" in plan


def test_the_status_is_json_with_exactly_the_agreed_keys(env):
    manager = env.make()
    for step in (lambda: None, lambda: manager.enter("x"), lambda: manager.leave("y")):
        step()
        status = manager.status()
        assert set(status) == {"mode", "ssid", "since", "last_error", "fallback_enabled"}
        json.dumps(status)


# ============================================================================ leave
def test_leave_restores_the_saved_client_network(env):
    manager = env.make()
    manager.enter("technician")
    env.clock.t += 60
    ok, message = manager.leave("done")
    assert ok and "back on 'Home'" in message
    assert env.nm.active == "Home"
    assert not os.path.exists(env.marker)
    assert not (env.tmp / "state.json").exists()
    assert manager.status() == {"mode": "client", "ssid": SSID, "since": env.clock.t, "last_error": None,
                                "fallback_enabled": False}


def test_leave_when_the_previous_network_was_forgotten_leaves_it_to_networkmanager(env):
    manager = env.make()
    manager.enter("technician")
    del env.nm.profiles["Home"]
    ok, message = manager.leave("done")
    assert ok and "any saved network" in message
    assert env.nm.active is None


def test_leave_when_the_previous_network_is_out_of_range_still_stops_the_ap(env):
    manager = env.make()
    manager.enter("technician")
    env.nm.client_up_fails.add("Home")
    ok, message = manager.leave("done")
    assert ok and "did not come back" in message and "Bluetooth" in message
    assert env.nm.active is None


def test_leave_when_not_in_direct_mode_changes_nothing(env):
    ok, message = env.make().leave("x")
    assert ok and message == "not in direct mode"
    assert env.nm.mutating() == []


def test_a_direct_mode_started_by_another_process_is_seen_and_left(env):
    """The CLI and the service are different processes; NetworkManager is the truth."""
    env.make().enter("cli")
    service = env.make()
    assert service.is_active()
    assert service.status()["mode"] == "direct"
    assert service.leave("app")[0]
    assert env.nm.active == "Home"


# ============================================================================ R3
def test_a_power_cycle_comes_back_to_the_client_network(env):
    """autoconnect=no: NetworkManager never raises the AP at boot; stale state is dropped."""
    manager = env.make()
    manager.enter("technician")
    # Power cycle: NetworkManager brings up what autoconnects, i.e. never the AP.
    env.nm.active = "Home" if env.nm.profiles[DIRECT_PROFILE]["connection.autoconnect"] == "no" else DIRECT_PROFILE
    os.unlink(env.marker)
    rebooted = env.make()
    rebooted.tick()
    assert env.nm.active == "Home"
    assert rebooted.status()["mode"] == "client"
    assert not (env.tmp / "state.json").exists()


def test_a_profile_made_sticky_by_hand_is_unstuck_and_left_at_boot(env):
    manager = env.make()
    manager.enter("technician")
    env.nm.profiles[DIRECT_PROFILE]["connection.autoconnect"] = "yes"  # someone edited it
    rebooted = env.make()
    rebooted.tick()
    assert env.nm.profiles[DIRECT_PROFILE]["connection.autoconnect"] == "no"
    assert env.nm.active == "Home"


def test_a_direct_mode_started_on_purpose_survives_a_service_restart(env):
    env.make().enter("technician")
    restarted = env.make()
    restarted.tick()
    assert env.nm.active == DIRECT_PROFILE


# ============================================================================ fallback policy
def test_the_fallback_is_off_by_default(env):
    env.nm.active = None
    manager = env.make()
    for _ in range(100):
        env.clock.t += 60
        manager.tick(env.clock.t)
    assert env.nm.active is None
    assert DIRECT_PROFILE not in env.nm.profiles
    assert manager.status()["fallback_enabled"] is False


def test_the_fallback_enters_after_the_offline_interval_and_leaves_when_home_returns(env):
    env.nm.active = None
    manager = env.make(fallback_enabled=True, fallback_after_s=600, return_check_s=300)
    assert manager.status()["fallback_enabled"] is True
    manager.tick(env.clock.t)
    env.clock.t += 599
    manager.tick(env.clock.t)
    assert env.nm.active is None, "not before the interval"
    env.clock.t += 2
    manager.tick(env.clock.t)
    assert env.nm.active == DIRECT_PROFILE
    assert json.loads((env.tmp / "state.json").read_text())["by"] == "fallback"

    env.clock.t += 301
    manager.tick(env.clock.t)
    assert env.nm.active == DIRECT_PROFILE, "home is not visible yet"
    env.nm.visible = ["Home"]
    env.clock.t += 301
    manager.tick(env.clock.t)
    assert env.nm.active == "Home"


def test_the_fallback_never_fires_while_online(env):
    manager = env.make(fallback_enabled=True, fallback_after_s=60)
    for _ in range(10):
        env.clock.t += 60
        manager.tick(env.clock.t)
    assert env.nm.active == "Home"


def test_an_explicit_direct_mode_is_not_left_by_the_fallback_policy(env):
    manager = env.make(fallback_enabled=True, return_check_s=1)
    manager.enter("technician")
    env.nm.visible = ["Home"]
    env.clock.t += 10
    manager.tick(env.clock.t)
    assert env.nm.active == DIRECT_PROFILE


def test_a_failed_fallback_start_waits_a_full_interval_before_retrying(env):
    env.nm.active = None
    env.nm.ap_up_fails = True
    manager = env.make(fallback_enabled=True, fallback_after_s=600)
    manager.tick(env.clock.t)
    env.clock.t += 601
    manager.tick(env.clock.t)
    ups = sum(1 for c in env.nm.calls if c == ["connection", "up", "id", DIRECT_PROFILE])
    assert ups == 1
    env.clock.t += 60
    manager.tick(env.clock.t)
    assert sum(1 for c in env.nm.calls if c == ["connection", "up", "id", DIRECT_PROFILE]) == 1


def test_a_disabled_manager_runs_no_nmcli_on_tick(env):
    manager = env.make(enabled=False, fallback_enabled=True)
    env.nm.calls.clear()
    manager.tick(env.clock.t + 10_000)
    assert env.nm.calls == []


# ============================================================================ dry-run plans
def test_the_enter_plan_is_what_enter_runs(env):
    manager = env.make()
    plan = manager.plan_enter()
    assert env.nm.mutating() == [], "planning changes nothing"
    manager.enter("technician")
    ran = [network.redacted_args(c) for c in env.nm.mutating()]
    import shlex
    planned = [shlex.split(line.split("    #", 1)[0])[1:] for line in plan if line.startswith("nmcli ")]
    # The success path: the planned add and up, in order (delete only when a profile existed).
    assert [c for c in planned if c[:2] in (["connection", "add"], ["connection", "up"])][:2] == ran
    assert planned[-1] == ["connection", "up", "id", "Home"], "the rollback names the previous network"
