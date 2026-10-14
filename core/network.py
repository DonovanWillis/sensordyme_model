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

# Module-level so the string annotations on watch() resolve. It was imported only
# INSIDE watch(), so `threading` was undefined at module scope and anything calling
# typing.get_type_hints() on this class raised NameError -- benign today because the
# annotations are strings and nothing introspects them, but a trap for any future
# schema/DI tooling that does.
import threading
from typing import Optional

from core.logging_setup import get_logger

log = get_logger("wifi")

# Outer bound on a single join attempt: associate + 4-way handshake + DHCP.
# A wrong passphrase normally fails in a few seconds because the access point
# rejects the handshake outright — this is the cap for a network that accepts
# the key and then stalls (slow DHCP, a captive portal), not the expected wait.
# Kept deliberately: without it a stalled join hangs setup with no way out.
_CONNECT_TIMEOUT_S = 45


def explain_connect_failure(raw: str, ssid: str) -> str:
    """Turn an nmcli error into something a person can act on.

    nmcli says "Connection activation failed: (7) Secrets were required, but
    not provided." A consumer reads that and learns nothing about what to do
    next. The mapping is deliberately narrow: anything unrecognised is passed
    through verbatim rather than guessed at, because a confidently wrong
    explanation ("wrong password" when the router was simply off) sends someone
    to fix a thing that was never broken.
    """
    # nmcli appends a "Hint: use 'journalctl -xe NM_CONNECTION=<uuid>…'" line.
    # Useful to an engineer with a terminal; noise to someone holding a phone,
    # and it drowns the one sentence that matters.
    raw = raw.split("Hint:", 1)[0].strip()

    lowered = raw.lower()
    if "secrets were required" in lowered or "no secrets" in lowered:
        return f"That password was not accepted by '{ssid}'. Check it and try again."
    if "property is invalid" in lowered or ("invalid" in lowered and "key" in lowered):
        return "That password is not a valid Wi-Fi key — most are at least 8 characters."
    if ("no network with ssid" in lowered or "not found" in lowered
            or "could not be found" in lowered or "network not found" in lowered):
        return (
            f"The device cannot see '{ssid}' from where it is. Move it closer to "
            f"the router, or check the network is on."
        )
    if "timeout" in lowered or "timed out" in lowered:
        return f"'{ssid}' did not finish connecting in time. It may be out of range."
    return raw


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
            existing = set(self.saved_networks())
            result = _run(args, timeout=_CONNECT_TIMEOUT_S)
            if result.returncode == 0:
                log.info("joined wifi '%s'", ssid)
                self._prefer(ssid)
                return True, f"joined '{ssid}' — the device will appear online in the app shortly"
            raw = result.stderr.strip() or "could not join network"
            # Never log `password`; the SSID and nmcli's reason are enough to
            # diagnose this, and the passphrase is the one thing that must not
            # end up in a log file someone later pastes into a bug report.
            log.warning("wifi connect to '%s' failed: %s", ssid, raw)
            if ssid not in existing:
                # A failed `device wifi connect` still leaves the profile it
                # created behind. Left there, NetworkManager retries it on every
                # future boot — so one typo turns into a device that keeps
                # trying a network it can never join.
                self.forget_network(ssid)
            return False, explain_connect_failure(raw, ssid)
        except subprocess.TimeoutExpired:
            self.forget_network(ssid)
            return False, (
                f"'{ssid}' did not respond in time. It may be out of range of the "
                f"device, or the network may be asking for a sign-in page."
            )
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)

    def watch(self, on_change, stop: "threading.Event") -> "threading.Thread | None":
        """Call `on_change()` whenever NetworkManager's state moves. Non-blocking.

        Polling cannot be both cheap and quick: checking every ten seconds means
        up to ten seconds of a device that has lost its network doing nothing
        visible, and checking every second spawns a subprocess sixty times a
        minute on a Pi that is already busy running inference.

        `nmcli monitor` is the thing that already knows. It emits a line the
        moment anything changes, so the device reacts to a network appearing or
        vanishing rather than discovering it on the next tick.
        """
        import threading

        if not nmcli_available():
            return None

        def pump() -> None:
            while not stop.is_set():
                try:
                    process = subprocess.Popen(
                        ["nmcli", "monitor"],
                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("nmcli monitor unavailable (%s) — falling back to polling", exc)
                    return
                try:
                    for _ in process.stdout:  # type: ignore[union-attr]
                        if stop.is_set():
                            break
                        on_change()
                except Exception as exc:  # noqa: BLE001
                    log.warning("nmcli monitor stream died (%s) — restarting", exc)
                finally:
                    process.terminate()
                # nmcli monitor exits if NetworkManager restarts. Come back,
                # rather than silently degrading to no notifications at all.
                if not stop.is_set():
                    stop.wait(2)

        thread = threading.Thread(target=pump, daemon=True, name="nm-monitor")
        thread.start()
        return thread

    def active_profile(self) -> Optional[str]:
        """The NetworkManager profile currently carrying the Wi-Fi connection."""
        if not nmcli_available():
            return None
        try:
            result = _run(["-t", "-f", "NAME,TYPE,DEVICE", "connection", "show", "--active"])
        except Exception:  # noqa: BLE001
            return None
        for line in result.stdout.splitlines():
            parts = line.split(":")
            if len(parts) >= 2 and parts[1] == "802-11-wireless":
                return parts[0]
        return None

    def provision(self, ssid: str, password: str) -> tuple[bool, str]:
        """Move the device to `ssid`, keeping the current network until it works.

        `connect()` uses `nmcli device wifi connect`, which associates
        immediately — it gives up the working network before knowing whether the
        new one will accept it. That is the wrong order for a device someone has
        just carried into a new building: a mistyped password leaves it on
        nothing, out of Bluetooth range, with no way back.

        Here the profile is created and validated first, and the working
        connection is only released to activate it. If activation fails the new
        profile is deleted and the previous one brought back, so a failed attempt
        costs a few seconds of downtime rather than the device.
        """
        if not ssid:
            return False, "network name required"
        if not nmcli_available():
            return True, f"dry-run (no nmcli on this platform): would join '{ssid}'"

        previous = self.active_profile()
        existed = ssid in self.saved_networks()
        if previous == ssid and existed and self.is_online():
            # Already on this network with a password that demonstrably works.
            # Re-writing it gains nothing and risks everything: a typo (or a
            # stranger in Bluetooth range) used to replace the working
            # passphrase, and when the old secret cannot be read back
            # (agent-owned secrets) there is nothing to restore it from.
            return True, f"already on '{ssid}'"
        #: The passphrase a saved profile had before this attempt, so a failed
        #: attempt can put it back. None when there is nothing to restore.
        saved_psk: Optional[str] = None

        try:
            if not existed:
                args = [
                    "connection", "add", "type", "wifi", "con-name", ssid,
                    "ifname", "wlan0", "ssid", ssid,
                    "connection.autoconnect", "yes",
                ]
                if password:
                    args += ["wifi-sec.key-mgmt", "wpa-psk", "wifi-sec.psk", password]
                result = _run(args)
                if result.returncode != 0:
                    reason = result.stderr.strip() or "could not save that network"
                    log.warning("could not create profile for '%s': %s", ssid, reason)
                    return False, explain_connect_failure(reason, ssid)
            elif password:
                # Known network, new password: update in place rather than
                # creating a duplicate profile for the same SSID -- but keep the
                # old passphrase first. When `ssid` is the network the device is
                # ON, rollback has nowhere else to go, so a mistyped password
                # used to leave a working device off Wi-Fi with a broken
                # profile, across reboots.
                saved_psk = self._stored_psk(ssid)
                _run(["connection", "modify", ssid,
                      "wifi-sec.key-mgmt", "wpa-psk", "wifi-sec.psk", password])

            # Only now is the working connection at risk.
            result = _run(["connection", "up", ssid], timeout=_CONNECT_TIMEOUT_S)
            if result.returncode == 0 and self._settled_on(ssid):
                log.info("switched to wifi '%s'", ssid)
                self._prefer(ssid)
                return True, f"joined '{ssid}'"

            reason = result.stderr.strip() or "the network did not accept the connection"
            log.warning("switch to '%s' failed: %s", ssid, reason)
        except subprocess.TimeoutExpired:
            reason = "timed out"
        except Exception as exc:  # noqa: BLE001
            reason = str(exc)

        self._roll_back(ssid, created=not existed, previous=previous, restore_psk=saved_psk)
        return False, explain_connect_failure(reason, ssid)

    @staticmethod
    def _stored_psk(ssid: str) -> Optional[str]:
        """The passphrase NetworkManager holds for profile `ssid`, or None."""
        try:
            result = _run(["-s", "-g", "802-11-wireless-security.psk", "connection", "show", ssid])
        except Exception:  # noqa: BLE001 - absence means nothing to restore
            return None
        value = result.stdout.strip() if result.returncode == 0 else ""
        return value or None

    def _settled_on(self, ssid: str) -> bool:
        """Whether the device is really on `ssid` with an address.

        `nmcli connection up` returning 0 means activation began, not that the
        network works. Reporting success on that alone is how a device ends up
        "connected" to something that never gave it an address.
        """
        import time

        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if self.current_ssid() == ssid and self.ip_address() and self.is_online():
                return True
            time.sleep(0.5)
        return False

    def _roll_back(self, ssid: str, created: bool, previous: Optional[str],
                   restore_psk: Optional[str] = None) -> None:
        """Undo a failed switch and get the device back on something."""
        try:
            if created:
                # Leaving it saved means NetworkManager retries a network that
                # rejected us on every future boot.
                _run(["connection", "delete", ssid])
            elif restore_psk is not None:
                _run(["connection", "modify", ssid, "wifi-sec.psk", restore_psk])
                if previous == ssid:
                    log.info("switch failed — restoring '%s' with its previous password", ssid)
                    _run(["connection", "up", ssid], timeout=_CONNECT_TIMEOUT_S)
            if previous and previous != ssid:
                log.info("switch failed — returning to '%s'", previous)
                _run(["connection", "up", previous], timeout=_CONNECT_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001 - recovery must never raise
            log.error("could not restore the previous network: %s", exc)

    def _prefer(self, ssid: str) -> None:
        """Make `ssid` the network this device reaches for first on future boots.

        Most-recently-set-up wins. When someone moves a device to a new place
        and both the old and new networks are somehow in range, the one they
        just deliberately configured is the one they meant — and without an
        explicit order NetworkManager picks by its own criteria, which shows up
        as a device that mysteriously rejoins the wrong network after a reboot.

        Best effort: a device that joined a network but could not be told to
        prefer it is still a device that joined the network.
        """
        if not nmcli_available():
            return
        try:
            highest = 0
            for name in self.saved_networks():
                if name == ssid:
                    continue
                result = _run(["-t", "-f", "connection.autoconnect-priority",
                               "connection", "show", name])
                value = result.stdout.strip().rsplit(":", 1)[-1]
                if value.lstrip("-").isdigit():
                    highest = max(highest, int(value))
            _run(["connection", "modify", ssid,
                  "connection.autoconnect", "yes",
                  "connection.autoconnect-priority", str(highest + 1)])
        except Exception as exc:  # noqa: BLE001
            log.warning("could not set autoconnect priority for '%s': %s", ssid, exc)

    def saved_networks(self) -> list[str]:
        """SSIDs NetworkManager already has profiles for, strongest priority first."""
        if not nmcli_available():
            return []
        try:
            result = _run(["-t", "-f", "NAME,TYPE", "connection", "show"])
        except Exception:  # noqa: BLE001
            return []
        return [
            line.rsplit(":", 1)[0]
            for line in result.stdout.splitlines()
            if line.endswith(":802-11-wireless") and line.rsplit(":", 1)[0]
        ]

    def forget_network(self, ssid: str) -> bool:
        """Delete a saved profile. Used to clean up after a failed join."""
        if not nmcli_available() or not ssid:
            return False
        try:
            result = _run(["connection", "delete", ssid])
            return result.returncode == 0
        except Exception:  # noqa: BLE001
            return False

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
