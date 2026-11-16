"""A stateful fake NetworkManager for the direct-mode tests (no real nmcli, ever).

It answers the exact argv core/network.py builds, keeps profiles and the active
connection, and plays the root dispatcher script's part: when the direct-mode
profile comes up it writes the firewall marker (unless told the firewall fails).
"""

from __future__ import annotations

import os
import types
from typing import Optional

from core.network import DIRECT_PROFILE

_ALIASES = {
    "ssid": "802-11-wireless.ssid",
    "wifi-sec.key-mgmt": "802-11-wireless-security.key-mgmt",
    "wifi-sec.proto": "802-11-wireless-security.proto",
    "wifi-sec.pairwise": "802-11-wireless-security.pairwise",
    "wifi-sec.group": "802-11-wireless-security.group",
    "wifi-sec.pmf": "802-11-wireless-security.pmf",
    "wifi-sec.psk": "802-11-wireless-security.psk",
}


def _esc(value: str) -> str:
    return value.replace("\\", "\\\\").replace(":", "\\:")


def _ok(stdout: str = "", returncode: int = 0, stderr: str = ""):
    return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


class FakeNM:
    def __init__(self, marker: Optional[str] = None) -> None:
        self.calls: list[list[str]] = []
        self.profiles: dict[str, dict[str, str]] = {}
        self.active: Optional[str] = None
        self.marker = marker
        self.visible: list[str] = []
        # Knobs for failure cases.
        self.ap_up_fails = False      # `connection up` of the AP returns an error
        self.ap_gets_ip = True        # shared mode assigns 10.42.0.1/24
        self.firewall_loads = True    # the dispatcher writes the marker
        self.add_drops = ()           # setting names NetworkManager "loses" on add
        self.client_up_fails: set[str] = set()

    # ------------------------------------------------------------------ helpers
    def add_client(self, name: str, ssid: Optional[str] = None, active: bool = False) -> None:
        self.profiles[name] = {"802-11-wireless.ssid": ssid or name, "connection.autoconnect": "yes",
                               "802-11-wireless.mode": "infrastructure"}
        if active:
            self.active = name

    def mutating(self) -> list[list[str]]:
        return [c for c in self.calls if c and c[0] == "connection" and c[1] in ("add", "delete", "modify", "up", "down")]

    def _set_marker(self, present: bool) -> None:
        if not self.marker:
            return
        if present:
            os.makedirs(os.path.dirname(self.marker), exist_ok=True)
            open(self.marker, "w").close()
        elif os.path.exists(self.marker):
            os.unlink(self.marker)

    # ------------------------------------------------------------------ nmcli
    def run(self, args, timeout=10):  # one branch per nmcli form
        args = list(args)
        self.calls.append(args)
        if args == ["-t", "-f", "NAME,TYPE", "connection", "show"]:
            return _ok("".join(f"{_esc(n)}:802-11-wireless\n" for n in self.profiles))
        if args == ["-t", "-f", "NAME,TYPE,DEVICE", "connection", "show", "--active"]:
            return _ok(f"{_esc(self.active)}:802-11-wireless:wlan0\n" if self.active else "")
        if args == ["-t", "-f", "STATE", "general"]:
            if self.active == DIRECT_PROFILE:
                return _ok("connected (local only)\n")
            return _ok("connected\n" if self.active else "disconnected\n")
        if args[:4] == ["-t", "-f", "IP4.ADDRESS", "device"]:
            if self.active == DIRECT_PROFILE and self.ap_gets_ip:
                return _ok("IP4.ADDRESS[1]:10.42.0.1/24\n")
            if self.active and self.active != DIRECT_PROFILE:
                return _ok("IP4.ADDRESS[1]:192.168.1.50/24\n")
            return _ok("")
        if args[:2] == ["-t", "-f"] and args[3:5] == ["connection", "show"] and args[5] == "id":
            profile = self.profiles.get(args[6])
            if profile is None:
                return _ok(returncode=10, stderr=f"Error: {args[6]} - no such connection profile.")
            return _ok("".join(f"{f}:{_esc(profile.get(f, ''))}\n" for f in args[2].split(",")))
        if args[:3] == ["-t", "-f", "SSID,SIGNAL,SECURITY"]:
            return _ok("".join(f"{_esc(s)}:70:WPA2\n" for s in self.visible))
        if args[:2] == ["connection", "add"]:
            name = args[args.index("con-name") + 1]
            settings: dict[str, str] = {}
            rest = args[args.index("ifname") + 2:]
            for key, value in zip(rest[::2], rest[1::2], strict=True):
                settings[_ALIASES.get(key, key)] = value
            for dropped in self.add_drops:
                settings.pop(dropped, None)
            self.profiles[name] = settings
            return _ok()
        if args[:2] == ["connection", "delete"]:
            self.profiles.pop(args[-1], None)
            if self.active == args[-1]:
                self.active = None
            return _ok()
        if args[:2] == ["connection", "modify"]:
            profile = self.profiles.setdefault(args[3], {})
            rest = args[4:]
            for key, value in zip(rest[::2], rest[1::2], strict=True):
                profile[_ALIASES.get(key, key)] = value
            return _ok()
        if args[:2] == ["connection", "up"]:
            name = args[-1]
            if name not in self.profiles:
                return _ok(returncode=10, stderr="Error: unknown connection")
            if name == DIRECT_PROFILE and self.ap_up_fails:
                return _ok(returncode=4, stderr="Error: Connection activation failed: AP mode not supported")
            if name in self.client_up_fails:
                return _ok(returncode=4, stderr="Error: No network with SSID found.")
            if self.active == DIRECT_PROFILE and name != DIRECT_PROFILE:
                self._set_marker(False)
            self.active = name
            if name == DIRECT_PROFILE and self.firewall_loads:
                self._set_marker(True)
            return _ok()
        if args[:2] == ["connection", "down"]:
            if self.active == args[-1]:
                self.active = None
                if args[-1] == DIRECT_PROFILE:
                    self._set_marker(False)
            return _ok()
        return _ok()
