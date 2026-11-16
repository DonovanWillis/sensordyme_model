"""Direct mode: the device runs its own Wi-Fi access point (docs/DIRECT_MODE.md).

A phone joins `SensorDyme-XXXX` with the passphrase printed on the device's
label and reaches the dashboard (and only the dashboard: core/ap_firewall.py).
The Pi 5 has one Wi-Fi radio, so while the AP is up the device is not on any
client network.

Rules this module keeps, each pinned by tests/test_direct_mode.py:

* **Never open, never a shared passphrase.** The PSK is random per device,
  stored 0600 and refused when loosely permissioned (like the BLE setup code),
  never derived from the MAC or serial, never logged and never in status().
  The profile is re-created and read back before every start, and refused
  unless it is a WPA access point (core/network.ap_profile_problem).
* **R2: entering is transactional.** If the AP (with its address and its
  firewall) is not up within `ap_timeout_s`, it is taken down and the client
  profile that was active before is brought back.
* **R3: a power cycle tries saved client networks first.** The profile has
  `autoconnect no`, re-asserted on the first tick; an AP that came up at boot
  because someone turned autoconnect on is taken down.
* **Fallback entry is OFF by default.** With `fallback_enabled`, the device
  enters direct mode after `fallback_after_s` with no network, and leaves again
  once a saved network is back in range.
* Leaving restores the saved client network.

R1 (Bluetooth setup stays open in direct mode) lives in
services/provisioning.py, which is handed `is_active`.

Every nmcli call goes through core/network.WifiManager. This module never runs
nmcli, nft or any other program itself.
"""

from __future__ import annotations

import json
import os
import secrets
import shlex
import stat
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from core import ap_firewall
from core.identity import device_id as _device_id
from core.identity import display_name
from core.logging_setup import get_logger
from core.network import (
    AP_VERIFY_FIELDS,
    DIRECT_PROFILE,
    WifiManager,
    ap_profile_args,
    nmcli_available,
    redacted_args,
)

log = get_logger("direct_mode")

# ---------------------------------------------------------------------- label passphrase (G6)
#: Lower case, no look-alikes (0/o, 1/l/i): typed on a phone from a printed label.
PSK_ALPHABET = "23456789abcdefghjkmnpqrstuvwxyz"
#: 16 symbols of 31: about 79 bits, far past an offline WPA2 dictionary attack.
PSK_SYMBOLS = 16
PSK_GROUP = 4
#: Where the factory writes the passphrase (scripts/direct_mode_psk.py), mode 0600.
DEFAULT_PSK_PATH = "/etc/sensordyme/direct_mode_psk"


class PskError(ValueError):
    """The direct-mode passphrase (or the file holding it) is unusable."""


def generate_psk() -> str:
    """A fresh random passphrase, `xxxx-xxxx-xxxx-xxxx`. From `secrets`, never from the board."""
    symbols = "".join(secrets.choice(PSK_ALPHABET) for _ in range(PSK_SYMBOLS))
    return "-".join(symbols[i:i + PSK_GROUP] for i in range(0, PSK_SYMBOLS, PSK_GROUP))


def check_psk(text: str) -> str:
    """`text` if it is a passphrase this module generates, else PskError (never echoing it)."""
    if not isinstance(text, str):
        raise PskError("the direct-mode passphrase must be text")
    groups = text.split("-")
    if (len(groups) != PSK_SYMBOLS // PSK_GROUP
            or any(len(g) != PSK_GROUP or any(c not in PSK_ALPHABET for c in g) for g in groups)):
        raise PskError(f"not a SensorDyme direct-mode passphrase ({PSK_SYMBOLS} symbols in groups of {PSK_GROUP})")
    return text


def load_psk(path: str) -> str:
    """The passphrase from `path`, refusing a file other accounts can read.

    The same rules as the BLE setup code (services/ble_pop.load_setup_code): a
    symlink, a non-regular file, any group/other permission bit, or another
    owner is refused. The message never includes the content.
    """
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        raise PskError(f"no direct-mode passphrase at {path}") from None
    except OSError as exc:
        raise PskError(f"{path} cannot be read ({type(exc).__name__})") from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise PskError(f"{path} must be a regular file, not a link")
    if info.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise PskError(f"{path} is readable by other accounts (mode {stat.S_IMODE(info.st_mode):o}); it must be 0600")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise PskError(f"{path} is owned by uid {info.st_uid}, not this service")
    try:
        with open(path, encoding="ascii") as f:
            text = f.read()
    except UnicodeDecodeError:
        raise PskError(f"{path} is not a direct-mode passphrase (not ASCII text)") from None
    except OSError as exc:
        raise PskError(f"{path} cannot be read ({type(exc).__name__}: {exc.strerror or 'error'})") from None
    return check_psk(text.strip())


def write_psk(path: str, psk: str) -> None:
    """Write a passphrase atomically with mode 0600. Refuses to replace an existing one.

    The printed label is how the owner joins; silently changing the passphrase
    would orphan it. Replacing it is a deliberate step: delete the file, generate,
    print a new label.
    """
    check_psk(psk)
    if os.path.lexists(path):
        raise PskError(f"{path} already exists: the label printed for it would stop working")
    directory = os.path.dirname(os.path.abspath(path))
    tmp = os.path.join(directory, f".{os.path.basename(path)}.{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="ascii") as f:
            f.write(psk + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)  # fails if `path` appeared meanwhile: never overwrites
    finally:
        os.unlink(tmp)


def _qr_escape(value: str) -> str:
    return "".join("\\" + c if c in '\\;,:"' else c for c in value)


def wifi_qr_payload(ssid: str, psk: str) -> str:
    """The standard Wi-Fi QR text (`WIFI:T:WPA;S:...;P:...;;`) the iPhone camera can join from."""
    return f"WIFI:T:WPA;S:{_qr_escape(ssid)};P:{_qr_escape(psk)};;"


def label_text(device_id: str, psk: str) -> str:
    """The direct-mode lines for the device label (beside the BLE setup code)."""
    ssid = display_name(device_id)
    return (f"SensorDyme {device_id}\n"
            f"Direct Wi-Fi: {ssid}\n"
            f"Wi-Fi password: {psk}\n"
            f"Wi-Fi QR: {wifi_qr_payload(ssid, psk)}")


# ---------------------------------------------------------------------- configuration
@dataclass(frozen=True)
class DirectModeConfig:
    """`network.direct_mode:` (config.yaml). Every default keeps direct mode off."""

    enabled: bool = False
    interface: str = "wlan0"
    security: str = "wpa2"
    channel: int = 6
    address: str = "10.42.0.1/24"
    ap_timeout_s: float = 30.0
    psk_path: str = DEFAULT_PSK_PATH
    state_path: str = "data/direct_mode.json"
    fallback_enabled: bool = False
    fallback_after_s: float = 600.0
    return_check_s: float = 300.0

    @classmethod
    def from_dict(cls, raw: Optional[dict]) -> DirectModeConfig:
        raw = raw or {}
        defaults = cls()
        return cls(
            enabled=raw.get("enabled", defaults.enabled) is True,
            interface=str(raw.get("interface", defaults.interface)),
            security=str(raw.get("security", defaults.security)),
            channel=int(raw.get("channel", defaults.channel)),
            address=defaults.address,
            ap_timeout_s=float(raw.get("ap_timeout_s", defaults.ap_timeout_s)),
            psk_path=str(raw.get("psk_path", defaults.psk_path)),
            state_path=str(raw.get("state_path", defaults.state_path)),
            fallback_enabled=raw.get("fallback_enabled", defaults.fallback_enabled) is True,
            fallback_after_s=float(raw.get("fallback_after_s", defaults.fallback_after_s)),
            return_check_s=float(raw.get("return_check_s", defaults.return_check_s)),
        )


def _shell(args: list[str], program: str = "nmcli") -> str:
    return shlex.join([program, *redacted_args(args)])


# ---------------------------------------------------------------------- the manager
class DirectModeManager:
    """Enter, leave and report direct mode. Thread-safe; one change at a time.

    Public interface (wired into the device's local API by the integrator):

        status() -> dict   {"mode", "ssid", "since", "last_error", "fallback_enabled"}
        enter(reason) -> (ok, message)
        leave(reason) -> (ok, message)
        tick(now=None)     the optional fallback policy and the R3 boot check
        is_active() -> bool    for services/provisioning.py (R1)

    The keyword arguments are test seams; production passes none of them.
    """

    def __init__(self, wifi: WifiManager, config: Optional[dict] = None, *,
                 device_id: Optional[str] = None,
                 clock: Callable[[], float] = time.time,
                 monotonic: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 has_nmcli: Optional[Callable[[], bool]] = None,
                 firewall_marker: str = ap_firewall.MARKER_PATH,
                 firewall_files: tuple[str, ...] = (ap_firewall.RULESET_PATH, ap_firewall.DISPATCHER_PATH)) -> None:
        self._wifi = wifi
        self.config = DirectModeConfig.from_dict(config)
        self._device_id = device_id
        self._clock = clock
        self._monotonic = monotonic
        self._sleep = sleep
        self._has_nmcli = has_nmcli or nmcli_available
        self._marker = firewall_marker
        self._firewall_files = firewall_files
        self._lock = threading.Lock()
        #: "entering" | "leaving" while a change runs; read by status() without the lock.
        self._transition: Optional[str] = None
        self._last_error: Optional[str] = None
        self._client_since: Optional[float] = None
        self._offline_since: Optional[float] = None
        self._last_return_check: Optional[float] = None
        self._booted = False

    # ------------------------------------------------------------------ facts
    @property
    def ssid(self) -> str:
        """The access point's name: `SensorDyme-XXXX` (core/identity). Not secret: it is broadcast."""
        return display_name(self._device_id or _device_id())

    def is_active(self) -> bool:
        """Whether NetworkManager has the direct-mode profile up, whoever started it."""
        return self._wifi.active_profile() == DIRECT_PROFILE

    def unavailable_reason(self) -> Optional[str]:
        """Why direct mode cannot be entered here, or None. Never reveals the passphrase."""
        if not self.config.enabled:
            return "direct mode is disabled (network.direct_mode.enabled: false)"
        if not self._has_nmcli():
            return "no NetworkManager on this platform"
        missing = [p for p in self._firewall_files if not os.path.exists(p)]
        if missing:
            return (f"the access-point firewall is not installed ({', '.join(missing)} missing); "
                    "run `sudo scripts/direct_mode.py install`")
        try:
            load_psk(self.config.psk_path)
        except PskError as exc:
            return f"direct-mode passphrase unusable: {exc}"
        return None

    def status(self) -> dict[str, Any]:
        """JSON-safe state. Never the passphrase.

        mode: client | direct | entering | leaving | unavailable. `unavailable`
        means direct mode cannot be entered and is not active; `last_error` then
        says why. `ssid` is always the access point's name (what a phone would
        join), whichever mode the device is in. `since` is the wall-clock time
        the current mode began, when this process knows it.
        """
        transition = self._transition
        state = self._load_state()
        active = False if transition else self.is_active()
        since: Optional[float] = None
        last_error = self._last_error
        if transition:
            mode = transition
        elif active:
            mode = "direct"
            since = state.get("since") if isinstance(state.get("since"), (int, float)) else None
        else:
            reason = self.unavailable_reason()
            mode = "unavailable" if reason else "client"
            since = self._client_since
            if reason:
                last_error = last_error or reason
        return {
            "mode": mode,
            "ssid": self.ssid,
            "since": since,
            "last_error": last_error,
            "fallback_enabled": bool(self.config.enabled and self.config.fallback_enabled),
        }

    # ------------------------------------------------------------------ enter (R2)
    def enter(self, reason: str, *, by: str = "explicit") -> tuple[bool, str]:
        """Start the access point, or put the previous network back and say why not."""
        if not self._lock.acquire(blocking=False):
            return False, "a direct-mode change is already in progress"
        try:
            return self._enter(str(reason or "requested"), by)
        finally:
            self._transition = None
            self._lock.release()

    def _enter(self, reason: str, by: str) -> tuple[bool, str]:
        problem = self.unavailable_reason()
        if problem:
            self._last_error = problem
            log.warning("direct mode not entered (%s): %s", reason, problem)
            return False, problem
        if self.is_active():
            return True, f"already in direct mode as '{self.ssid}'"
        psk = load_psk(self.config.psk_path)  # checked by unavailable_reason() just above
        previous = self._wifi.active_profile()
        self._transition = "entering"
        log.info("entering direct mode (%s); previous network: %s", reason, previous or "none")

        cfg = self.config
        ok, message = self._wifi.write_ap_profile(self.ssid, psk, security=cfg.security, interface=cfg.interface,
                                                  channel=cfg.channel, address=cfg.address)
        del psk
        if not ok:
            # Nothing was switched yet: the client connection is untouched.
            self._last_error = message
            return False, f"direct mode not started: {message}"

        self._save_state({"mode": "entering", "previous": previous, "since": self._clock(),
                          "reason": reason, "by": by})
        deadline = self._monotonic() + cfg.ap_timeout_s
        ok, message = self._wifi.activate_profile(DIRECT_PROFILE, timeout=cfg.ap_timeout_s)
        if ok:
            ok, message = self._wait_until_up(deadline)
        if not ok:
            restored = self._roll_back(previous)
            self._last_error = message
            self._clear_state()
            log.warning("direct mode did not start (%s); %s", message, restored)
            return False, f"direct mode did not start: {message}; {restored}"

        since = self._clock()
        self._save_state({"mode": "direct", "previous": previous, "since": since,
                          "reason": reason, "by": by})
        self._last_error = None
        self._offline_since = None
        self._last_return_check = self._monotonic()
        log.info("direct mode on as '%s' (%s)", self.ssid, reason)
        return True, f"direct mode on: join '{self.ssid}' with the password on the device label"

    def _wait_until_up(self, deadline: float) -> tuple[bool, str]:
        """The AP is up only with the profile active, its address, and its firewall loaded."""
        ip = self.config.address.split("/", 1)[0]
        missing = "the access point did not come up"
        while True:
            if self._wifi.active_profile() != DIRECT_PROFILE:
                missing = "the access point did not come up"
            elif not (self._wifi.interface_ipv4(self.config.interface) or "").startswith(ip + "/"):
                missing = f"the access point has no address {ip}"
            elif not os.path.exists(self._marker):
                missing = "the access-point firewall did not load"
            else:
                return True, "up"
            if self._monotonic() >= deadline:
                return False, f"{missing} within {self.config.ap_timeout_s:.0f}s"
            self._sleep(0.5)

    def _roll_back(self, previous: Optional[str]) -> str:
        """Take the AP down and bring `previous` back. Never raises; returns what happened."""
        try:
            self._wifi.deactivate_profile(DIRECT_PROFILE)
            if previous and previous != DIRECT_PROFILE:
                ok, message = self._wifi.activate_profile(previous)
                return f"returned to '{previous}'" if ok else f"could not return to '{previous}' ({message})"
        except Exception as exc:  # noqa: BLE001 - recovery must never raise
            log.error("direct-mode rollback failed: %s", exc)
            return f"rollback failed ({exc})"
        return "no previous network to return to; NetworkManager will join any saved network in range"

    # ------------------------------------------------------------------ leave
    def leave(self, reason: str) -> tuple[bool, str]:
        """Stop the access point and go back to the saved client network."""
        return self._leave_locked(reason)

    def _leave_locked(self, reason: str, target: Optional[str] = None) -> tuple[bool, str]:
        if not self._lock.acquire(blocking=False):
            return False, "a direct-mode change is already in progress"
        try:
            return self._leave(str(reason or "requested"), target)
        finally:
            self._transition = None
            self._lock.release()

    def _leave(self, reason: str, target: Optional[str]) -> tuple[bool, str]:
        """`target`: the network to rejoin when no previous one was recorded (the fallback's find)."""
        if not self.is_active():
            self._clear_state()
            return True, "not in direct mode"
        previous = self._load_state().get("previous") or target
        self._transition = "leaving"
        log.info("leaving direct mode (%s)", reason)
        ok, message = self._wifi.deactivate_profile(DIRECT_PROFILE)
        if not ok or self.is_active():
            self._last_error = f"could not stop the access point: {message}"
            return False, self._last_error
        self._clear_state()
        self._last_error = None
        self._client_since = self._clock()
        self._offline_since = None
        if isinstance(previous, str) and previous in self._wifi.client_networks():
            ok, message = self._wifi.activate_profile(previous)
            if ok:
                return True, f"direct mode off; back on '{previous}'"
            return True, (f"direct mode off; '{previous}' did not come back ({message}). NetworkManager keeps "
                          "trying saved networks, and Bluetooth setup opens while the device is offline")
        return True, ("direct mode off; NetworkManager will join any saved network in range, and Bluetooth "
                      "setup opens while the device is offline")

    # ------------------------------------------------------------------ policy
    def tick(self, now: Optional[float] = None) -> None:
        """Run the boot check once, then the fallback policy (only when enabled). Never raises."""
        try:
            self._tick(self._monotonic() if now is None else now)
        except Exception as exc:  # noqa: BLE001 - a policy tick must never take the caller down
            log.error("direct-mode tick failed: %s", exc)

    def _tick(self, now: float) -> None:
        if not self.config.enabled or self._transition:
            return
        if not self._booted:
            self._booted = True
            self._boot_check()
        if not self.config.fallback_enabled:
            return
        if self.is_active():
            self._offline_since = None
            state = self._load_state()
            due = self._last_return_check is None or now - self._last_return_check >= self.config.return_check_s
            if state.get("by") == "fallback" and due:
                self._last_return_check = now
                back = self._visible_saved_network()
                if back:
                    self._leave_locked(f"saved network '{back}' is back in range", target=back)
            return
        if self._wifi.is_online():
            self._offline_since = None
            return
        if self._offline_since is None:
            self._offline_since = now
            return
        if now - self._offline_since >= self.config.fallback_after_s:
            # Restart the clock whatever happens, so a failed start is retried
            # after another full interval rather than on every tick.
            self._offline_since = now
            self.enter(f"no network for {self.config.fallback_after_s:.0f}s (fallback)", by="fallback")

    def _boot_check(self) -> None:
        """R3: direct mode is never what a power cycle comes back to.

        The profile is written with autoconnect off, so NetworkManager cannot
        raise it at boot by itself. If someone turned autoconnect on by hand,
        it is turned off again, and an AP that came up because of it is taken
        down so the saved client networks get their turn. An AP that is up
        with autoconnect off was started on purpose in this boot (the CLI, or
        before a service restart) and is left alone.
        """
        if DIRECT_PROFILE not in self._wifi.saved_networks():
            if self._load_state():
                self._clear_state()  # left over from before a power cycle
            return
        field = AP_VERIFY_FIELDS[0]  # connection.autoconnect
        autoconnect = self._wifi.profile_settings(DIRECT_PROFILE, (field,)).get(field)
        sticky = autoconnect != "no"
        if sticky:
            log.warning("direct-mode profile had autoconnect=%s; setting it back to no", autoconnect)
            self._wifi.set_autoconnect(DIRECT_PROFILE, False)
        if not self.is_active():
            if self._load_state():
                self._clear_state()  # left over from before a power cycle
            return
        if sticky:
            log.warning("direct mode came up at boot by itself; leaving it for the saved networks")
            ok, message = self.leave("direct mode is never the boot default")
            if not ok:
                log.error("could not leave direct mode at boot: %s", message)

    def _visible_saved_network(self) -> Optional[str]:
        """The name of a saved client profile whose SSID a scan can see, or None.

        Whether the radio can scan while it is an access point is a hardware
        question (docs/DIRECT_MODE.md); a scan that finds nothing keeps the AP.
        """
        by_ssid = {self._wifi.profile_ssid(name) or name: name for name in self._wifi.client_networks()}
        if not by_ssid:
            return None
        for network in self._wifi.scan():
            name = by_ssid.get(str(network.get("ssid") or ""))
            if name:
                return name
        return None

    # ------------------------------------------------------------------ dry-run plans
    def plan_enter(self) -> list[str]:
        """The commands enter() would run, as shell lines, passphrase redacted. Runs nothing that changes state."""
        cfg = self.config
        previous = self._wifi.active_profile()
        add = ap_profile_args(self.ssid, "x" * 8, interface=cfg.interface, security=cfg.security,
                              channel=cfg.channel, address=cfg.address)
        lines = [
            f"# read: active profile is {previous or '(none)'}",
            _shell(["connection", "delete", "id", DIRECT_PROFILE]) + "    # only if it exists",
            _shell(add),
            _shell(["-t", "-f", ",".join(AP_VERIFY_FIELDS), "connection", "show", "id", DIRECT_PROFILE])
            + "    # refuse unless a WPA AP with autoconnect no",
            _shell(["connection", "up", "id", DIRECT_PROFILE]),
            (f"# wait up to {cfg.ap_timeout_s:.0f}s for: profile active, {cfg.address} on {cfg.interface}, "
             f"{self._marker} (written by the dispatcher after: nft -f {ap_firewall.RULESET_PATH})"),
            "# on timeout (rollback):",
            _shell(["connection", "down", "id", DIRECT_PROFILE]),
        ]
        if previous:
            lines.append(_shell(["connection", "up", "id", previous]))
        return lines

    def plan_leave(self) -> list[str]:
        previous = self._load_state().get("previous")
        lines = [_shell(["connection", "down", "id", DIRECT_PROFILE]),
                 f"# dispatcher then runs: nft delete table inet {ap_firewall.TABLE}"]
        if isinstance(previous, str) and previous:
            lines.append(_shell(["connection", "up", "id", previous]) + "    # if still saved")
        else:
            lines.append("# no saved previous network: NetworkManager autoconnects any saved network in range")
        return lines

    # ------------------------------------------------------------------ state file
    def _load_state(self) -> dict:
        try:
            data = json.loads(Path(self.config.state_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save_state(self, state: dict) -> None:
        path = Path(self.config.state_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
            tmp.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
            os.replace(tmp, path)
        except OSError as exc:
            log.warning("could not record direct-mode state at %s: %s", path, exc)

    def _clear_state(self) -> None:
        try:
            Path(self.config.state_path).unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            log.warning("could not clear direct-mode state: %s", exc)
