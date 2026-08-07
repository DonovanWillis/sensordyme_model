"""What the device says when a network will not accept it.

A person mistyping a Wi-Fi password is the single most likely thing to go wrong
in setup, and it has to be survivable: a clear reason, and no wreckage left
behind that makes the next attempt worse.
"""

import subprocess
import types

import core.network as network
from core.network import WifiManager, explain_connect_failure


def fake_nmcli(monkeypatch, *, returncode=0, stdout="", stderr="", saved=""):
    calls = []

    def run(args, timeout=10):
        calls.append(args)
        if args[:3] == ["-t", "-f", "NAME,TYPE"]:
            return types.SimpleNamespace(returncode=0, stdout=saved, stderr="")
        return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(network, "nmcli_available", lambda: True)
    monkeypatch.setattr(network, "_run", run)
    return calls


def test_a_wrong_password_says_so_in_plain_words(monkeypatch):
    fake_nmcli(monkeypatch, returncode=4,
               stderr="Error: Connection activation failed: (7) Secrets were required, but not provided.")
    ok, message = WifiManager().connect("HomeNet", "wrong")
    assert ok is False
    assert "password" in message.lower()
    assert "HomeNet" in message
    assert "(7)" not in message


def test_a_network_out_of_range_is_not_blamed_on_the_password(monkeypatch):
    """Telling someone their password is wrong when the router is simply out of
    range sends them to fix something that was never broken."""
    fake_nmcli(monkeypatch, returncode=10,
               stderr="Error: No network with SSID 'Upstairs' found.")
    ok, message = WifiManager().connect("Upstairs", "hunter2")
    assert ok is False
    assert "closer" in message.lower() or "cannot see" in message.lower()
    assert "password" not in message.lower()


def test_an_unrecognised_error_is_passed_through_rather_than_guessed_at(monkeypatch):
    fake_nmcli(monkeypatch, returncode=1, stderr="Error: something entirely new happened")
    _, message = WifiManager().connect("HomeNet", "pw")
    assert "something entirely new happened" in message


def test_a_failed_join_does_not_leave_a_broken_profile_behind(monkeypatch):
    """nmcli creates the profile before the join fails. Left there, the device
    retries a network it can never join on every future boot."""
    calls = fake_nmcli(monkeypatch, returncode=4, stderr="Secrets were required", saved="")
    WifiManager().connect("Typo", "wrong")
    assert any(c[:2] == ["connection", "delete"] and c[2] == "Typo" for c in calls), \
        "the failed profile was left saved"


def test_a_network_that_already_worked_is_not_deleted_on_a_later_failure(monkeypatch):
    """Otherwise one bad retry against a known-good network forgets it, and the
    device stops auto-connecting somewhere it used to work."""
    calls = fake_nmcli(monkeypatch, returncode=4, stderr="Secrets were required",
                       saved="HomeNet:802-11-wireless\n")
    WifiManager().connect("HomeNet", "wrong")
    assert not any(c[:2] == ["connection", "delete"] for c in calls), \
        "a previously saved network was deleted after one failed attempt"


def test_a_stalled_join_times_out_rather_than_hanging_setup(monkeypatch):
    monkeypatch.setattr(network, "nmcli_available", lambda: True)

    def stall(args, timeout=10):
        if args[:3] == ["-t", "-f", "NAME,TYPE"]:
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        if args[:2] == ["connection", "delete"]:
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        raise subprocess.TimeoutExpired(cmd="nmcli", timeout=45)

    monkeypatch.setattr(network, "_run", stall)
    ok, message = WifiManager().connect("Slow", "pw")
    assert ok is False
    assert "time" in message.lower()


def test_saved_networks_are_listed(monkeypatch):
    fake_nmcli(monkeypatch, saved="HomeNet:802-11-wireless\nWired:802-3-ethernet\nCafe:802-11-wireless\n")
    assert WifiManager().saved_networks() == ["HomeNet", "Cafe"]


def test_the_password_never_appears_in_the_explanation():
    message = explain_connect_failure("Secrets were required, but not provided.", "HomeNet")
    assert "hunter2" not in message
    assert "password" in message.lower()


def test_a_newly_configured_network_becomes_the_preferred_one(monkeypatch):
    """Requirement: after moving a device somewhere new, it should come back up
    on the network it was just set up for -- not on whichever one NetworkManager
    happens to pick when several are in range."""
    calls = []

    def run(args, timeout=10):
        calls.append(args)
        if args[:3] == ["-t", "-f", "NAME,TYPE"]:
            return types.SimpleNamespace(
                returncode=0, stdout="OldNet:802-11-wireless\nNewNet:802-11-wireless\n", stderr="")
        if args[:2] == ["-t", "-f"] and "autoconnect-priority" in args[2]:
            return types.SimpleNamespace(
                returncode=0, stdout="connection.autoconnect-priority:5\n", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(network, "nmcli_available", lambda: True)
    monkeypatch.setattr(network, "_run", run)
    ok, _ = WifiManager().connect("NewNet", "pw")
    assert ok
    modify = [c for c in calls if c[:2] == ["connection", "modify"]]
    assert modify, "priority was never set"
    assert "6" in modify[0], f"new network did not outrank the existing 5: {modify[0]}"
    assert "connection.autoconnect" in modify[0]


def test_failing_to_set_priority_does_not_fail_the_join(monkeypatch):
    """A device that joined but could not be told to prefer the network has
    still joined; reporting failure would send someone to redo working setup."""
    def run(args, timeout=10):
        if args[:2] == ["connection", "modify"]:
            raise OSError("nmcli busy")
        if args[:3] == ["-t", "-f", "NAME,TYPE"]:
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(network, "nmcli_available", lambda: True)
    monkeypatch.setattr(network, "_run", run)
    ok, _ = WifiManager().connect("NewNet", "pw")
    assert ok is True


def test_the_journalctl_hint_is_stripped():
    """nmcli appends "Hint: use 'journalctl -xe NM_CONNECTION=<uuid>...'". Useful
    to an engineer at a terminal; noise to someone holding a phone, and it
    drowns the one sentence that matters."""
    raw = ("Error: Connection activation failed: The Wi-Fi network could not be found\\n"
           "Hint: use 'journalctl -xe NM_CONNECTION=b8f5c1c1 + NM_DEVICE=wlan0' to get more details.")
    message = explain_connect_failure(raw, "Upstairs")
    assert "journalctl" not in message
    assert "NM_CONNECTION" not in message


def test_nm_activation_wording_for_a_missing_network_is_recognised():
    """Observed verbatim on a Pi 5. The earlier patterns matched "not found" but
    not "could not be found", so this fell through to the raw text."""
    message = explain_connect_failure(
        "Error: Connection activation failed: The Wi-Fi network could not be found", "Upstairs")
    assert "Upstairs" in message
    assert "closer" in message.lower() or "cannot see" in message.lower()
    assert "activation failed" not in message.lower()
