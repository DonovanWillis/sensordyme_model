"""Wi-Fi management: a thin, testable wrapper over NetworkManager's nmcli.

Single owner of every nmcli invocation in the codebase (the dashboard's /setup
page and the BLE provisioning service both come through here), so connect
behavior, timeouts, and the no-nmcli dry-run used by tests live in one place.

On boot the flow is: NetworkManager auto-joins any saved network by itself; the
provisioning service (services/provisioning.py) just watches `is_online()` and
opens BLE provisioning if nothing connects within the grace period.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from typing import Optional

from core.logging_setup import get_logger

log = get_logger("wifi")

_CONNECT_TIMEOUT_S = 45


def nmcli_available() -> bool:
    """True when NetworkManager's CLI exists (any Pi OS Bookworm; not macOS)."""
    return shutil.which("nmcli") is not None


def _run(args: list[str], timeout: float = 10) -> subprocess.CompletedProcess:
    return subprocess.run(["nmcli", *args], capture_output=True, text=True, timeout=timeout)


class WifiManager:
    """Query and change Wi-Fi state. Every method degrades gracefully without nmcli."""

    def is_online(self) -> bool:
        """True when NetworkManager reports full or site-local connectivity.

        Without nmcli (dev Mac) this returns True — the Mac's own network is
        assumed up, and provisioning never runs there anyway.
        """
        if not nmcli_available():
            return True
        try:
            result = _run(["-t", "-f", "STATE", "general"])
            # startswith, not `in`: "disconnected" contains "connected". Matches
            # "connected", "connected (site only)", "connected (local only)".
            return result.stdout.strip().startswith("connected")
        except Exception as exc:  # noqa: BLE001
            log.warning("nmcli state query failed: %s", exc)
            return False

    def current_ssid(self) -> Optional[str]:
        """The SSID currently connected on wlan0, or None."""
        if not nmcli_available():
            return None
        try:
            result = _run(["-t", "-f", "ACTIVE,SSID", "device", "wifi"])
            for line in result.stdout.splitlines():
                active, _, ssid = line.partition(":")
                if active == "yes" and ssid:
                    return ssid
        except Exception as exc:  # noqa: BLE001
            log.warning("nmcli ssid query failed: %s", exc)
        return None

    def ip_address(self) -> Optional[str]:
        """This device's first IPv4 address, or None."""
        if not nmcli_available():
            return None
        try:
            result = _run(["-t", "-f", "IP4.ADDRESS", "device", "show"])
            for line in result.stdout.splitlines():
                _, _, value = line.partition(":")
                addr = value.split("/")[0]
                if addr and not addr.startswith("127."):
                    return addr
        except Exception as exc:  # noqa: BLE001
            log.warning("nmcli ip query failed: %s", exc)
        return None

    def saved_wifi_names(self) -> list[str]:
        """Names of saved Wi-Fi connection profiles (credentials NetworkManager already has)."""
        if not nmcli_available():
            return []
        try:
            result = _run(["-t", "-f", "NAME,TYPE", "connection", "show"])
            return [
                line.partition(":")[0]
                for line in result.stdout.splitlines()
                if line.endswith(":802-11-wireless")
            ]
        except Exception as exc:  # noqa: BLE001
            log.warning("nmcli connection list failed: %s", exc)
            return []

    def scan(self, limit: int = 20) -> list[dict]:
        """Wi-Fi networks this device can see, strongest first.

        Scanned on the DEVICE, not the phone, and that distinction is the whole
        point: iOS cannot enumerate networks at all without an entitlement Apple
        effectively does not grant, and even if it could, the phone's list is
        the wrong one. The Pi is the machine that has to associate — it may sit
        behind a wall from the router, or reach a network the phone does not.
        Offering a network the phone can see and the Pi cannot produces a setup
        that fails with no explanation.

        Hidden networks are omitted: they have no SSID to show, and a blank row
        is worse than none. They remain reachable by typing the name.
        """
        if not nmcli_available():
            return []
        try:
            # --rescan yes forces a fresh sweep; the cached list goes stale
            # exactly when it matters, which is right after the device moves.
            result = _run(
                ["-t", "-f", "SSID,SIGNAL,SECURITY", "device", "wifi", "list", "--rescan", "yes"],
                timeout=20,
            )
        except Exception as exc:  # noqa: BLE001 - a failed scan must not break provisioning
            log.warning("nmcli wifi scan failed: %s", exc)
            return []

        seen: dict[str, dict] = {}
        for line in result.stdout.splitlines():
            # nmcli escapes colons inside fields as "\:", so split on unescaped ones.
            parts = re.split(r"(?<!\\):", line)
            if len(parts) < 3:
                continue
            ssid = parts[0].replace("\\:", ":").strip()
            if not ssid:
                continue  # hidden network
            try:
                signal = int(parts[1])
            except ValueError:
                signal = 0
            security = parts[2].strip()
            # The same SSID appears once per access point on a mesh; keep the
            # strongest, or the list is three copies of the same network.
            if ssid not in seen or signal > seen[ssid]["signal"]:
                seen[ssid] = {
                    "ssid": ssid,
                    "signal": signal,
                    "secured": bool(security and security != "--"),
                }
        return sorted(seen.values(), key=lambda n: n["signal"], reverse=True)[:limit]

    def connect(self, ssid: str, password: str) -> tuple[bool, str]:
        """Join `ssid`, saving the profile so NetworkManager auto-reconnects on
        every future boot. Returns (ok, human-readable message)."""
        if not ssid:
            return False, "network name required"
        if not nmcli_available():
            return True, f"dry-run (no nmcli on this platform): would join '{ssid}'"
        try:
            args = ["device", "wifi", "connect", ssid]
            if password:
                args += ["password", password]
            result = _run(args, timeout=_CONNECT_TIMEOUT_S)
            if result.returncode == 0:
                log.info("joined wifi '%s'", ssid)
                return True, f"joined '{ssid}' — the device will appear online in the app shortly"
            message = result.stderr.strip() or "could not join network"
            log.warning("wifi connect to '%s' failed: %s", ssid, message)
            return False, message
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)

    def status(self) -> tuple[bool, str]:
        """(healthy, detail) for the health monitor's `wifi` component (non-critical:
        the device is designed to keep detecting and alerting fully offline)."""
        if not nmcli_available():
            return True, "no NetworkManager on this platform (dev machine)"
        if self.is_online():
            ssid = self.current_ssid()
            ip = self.ip_address()
            return True, f"online{f' on {ssid!r}' if ssid else ''}{f' ({ip})' if ip else ''}"
        return False, "offline — device continues standalone; provisioning may be active"
