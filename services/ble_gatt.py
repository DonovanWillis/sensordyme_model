"""BlueZ GATT server for Bluetooth-LE Wi-Fi provisioning (Pi-only).

Exposes one primary service the SensorDyme app discovers while the device has
no Wi-Fi:

    Service  5d1e0001-...   SensorDyme provisioning
      char   5d1e0002  read          device info JSON {id, name, model, version, hostname}
      char   5d1e0003  write         Wi-Fi credentials JSON {"ssid": "...", "psk": "..."}
                                     (long/prepared writes are reassembled by offset)
      char   5d1e0004  read/notify   status JSON {"state", "ssid", "ip", "device_id",
                                     "reconfigurable", "provision": {"state":
                                     "connecting"|"ok"|"error", "ssid", "attempt",
                                     "ip"|"error", "message"}, "scanning"} -- <= 512 bytes
      char   5d1e0005  write         lightweight commands JSON {"action": "rescan" | "identify"}
      char   5d1e0006  read          networks JSON [{"ssid", "signal", "secured"}] (cached)
      char   5d1e0007  read/write/   proof-of-possession messages (services/ble_pop.py), present
                       notify        only with `network.provisioning.auth: pop`; the reply to the
                                     last write is read (or notified) back

Bluetooth is for setup/config/commands ONLY — frames never move over BLE; the
live feed always uses Wi-Fi (see core/dashboard.py's /stream).

Requires BlueZ with dbus-python + PyGObject (`apt install python3-dbus
python3-gi`, installed by setup_pi.sh). All imports are performed at start()
so this file is importable (and the rest of the codebase testable) on a Mac.

Security posture: pairing is BLE "Just Works" at the link layer either way.
With `network.provisioning.auth: pop` the credential characteristic refuses
plain writes; credentials arrive only encrypted on the auth characteristic,
after the phone has proved it holds the device's setup code
(services/ble_pop.py). With `auth: legacy` (development only; refused when
`deployment: production`) the credential write is unauthenticated and what
bounds the exposure is WHEN it is served: while the device is offline, and for
a few minutes after each boot (services/provisioning.py,
`reconfigure_window_s`).
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from collections.abc import Callable
from typing import Optional

from core.logging_setup import get_logger

log = get_logger("ble")

SERVICE_UUID = "5d1e0001-5344-4d45-b000-536e447a6d65"
CHAR_DEVICE_INFO_UUID = "5d1e0002-5344-4d45-b000-536e447a6d65"
CHAR_WIFI_CREDS_UUID = "5d1e0003-5344-4d45-b000-536e447a6d65"
CHAR_STATUS_UUID = "5d1e0004-5344-4d45-b000-536e447a6d65"
CHAR_COMMAND_UUID = "5d1e0005-5344-4d45-b000-536e447a6d65"
CHAR_NETWORKS_UUID = "5d1e0006-5344-4d45-b000-536e447a6d65"
CHAR_AUTH_UUID = "5d1e0007-5344-4d45-b000-536e447a6d65"

BLUEZ_SERVICE = "org.bluez"
GATT_MANAGER_IFACE = "org.bluez.GattManager1"
LE_AD_MANAGER_IFACE = "org.bluez.LEAdvertisingManager1"
GATT_SERVICE_IFACE = "org.bluez.GattService1"
GATT_CHRC_IFACE = "org.bluez.GattCharacteristic1"
LE_AD_IFACE = "org.bluez.LEAdvertisement1"
DBUS_OM_IFACE = "org.freedesktop.DBus.ObjectManager"
DBUS_PROP_IFACE = "org.freedesktop.DBus.Properties"


# A GATT attribute value is at most 512 octets. Not a tuning knob — it is the
# ceiling in the specification, and BlueZ truncates at exactly 512 with no error.
#
# The previous value was 4096, chosen as "generously large", which meant the cap
# never engaged: the payload sailed past 512, the transport cut it mid-string,
# and the app received unparseable JSON and showed no networks. It read as "this
# device cannot see any Wi-Fi" — the opposite of the truth, since the device
# could see twelve.
#
# Long reads (see ReadValue's offset handling) are what get us up to 512 rather
# than one MTU. They cannot get us past it.
_GATT_MAX_ATTRIBUTE_BYTES = 512
_MAX_NETWORK_PAYLOAD = 480  # headroom under the ceiling


def _fit_networks(networks: list) -> bytes:
    """Serialise as many networks as fit the payload budget, strongest first.

    Trimmed by BYTES, not by count. The previous cap took the first twelve and
    assumed that was small enough; twelve real SSIDs are about 700 bytes, so the
    cap never engaged and the value was truncated mid-JSON by the transport —
    which the app cannot parse at all, turning a full list into an empty one.
    """
    compact = []
    for network in networks:
        candidate = [*compact, network]
        encoded = json.dumps(candidate, separators=(",", ":")).encode()
        if len(encoded) > _MAX_NETWORK_PAYLOAD:
            break
        compact = candidate
    return json.dumps(compact, separators=(",", ":")).encode()


#: The status value is served whole on a read, so it must fit the attribute.
_MAX_STATUS_PAYLOAD = _GATT_MAX_ATTRIBUTE_BYTES


def _dumps(obj) -> bytes:
    # Compact, and UTF-8 rather than \uXXXX escapes: a non-ASCII SSID costs 4
    # bytes per character instead of 12 (an emoji is a surrogate pair).
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _clip(text: str, max_bytes: int) -> str:
    """`text` cut to at most `max_bytes` of UTF-8, never mid-character."""
    return text.encode("utf-8")[:max_bytes].decode("utf-8", "ignore")


def _encode_status(status: dict) -> bytes:
    """The status payload as bytes that always fit a GATT attribute (512).

    Every field is kept while it fits. Past 512 BlueZ cut the value mid-string,
    the app could not parse it, and a setup whose outcome was in that value
    waited out its 90-second deadline and blamed the password. Observed sizes:
    an error status for a 32-byte non-ASCII SSID was 531 bytes.
    Reductions, in order, only as needed: the redundant `provision.message` and
    `provision.ssid` (the app sent that SSID), then the free text clipped. The
    fields the app decides on -- state, provision.state/attempt/ip/error,
    device_id, scanning -- are never dropped.
    """
    payload = _dumps(status)
    if len(payload) <= _MAX_STATUS_PAYLOAD:
        return payload
    slim = dict(status)
    provision = dict(slim["provision"]) if isinstance(slim.get("provision"), dict) else None
    if provision is not None:
        provision.pop("message", None)
        provision.pop("ssid", None)
        slim["provision"] = provision
    payload = _dumps(slim)
    if len(payload) <= _MAX_STATUS_PAYLOAD:
        return payload
    for key in ("ssid", "error"):
        if isinstance(slim.get(key), str):
            slim[key] = _clip(slim[key], 64)
    if provision is not None and isinstance(provision.get("error"), str):
        provision["error"] = _clip(provision["error"], 160)
    payload = _dumps(slim)
    if len(payload) <= _MAX_STATUS_PAYLOAD:
        return payload
    keep = ("state", "attempt", "ip", "error")
    minimal = {
        "state": slim.get("state"),
        "device_id": slim.get("device_id"),
        "provision": (
            {k: (_clip(v, 60) if k == "error" and isinstance(v, str) else v)
             for k, v in provision.items() if k in keep}
            if provision is not None else slim.get("provision")
        ),
        "scanning": slim.get("scanning", False),
    }
    return _dumps(minimal)[:_MAX_STATUS_PAYLOAD]


def parse_credentials(data: bytes) -> Optional[tuple[str, str]]:
    """(ssid, psk) from a credential write, or None when it is not one.

    Only a JSON object with a string `ssid` is credentials. `psk` may be absent
    or null (an open network). Anything else -- a list, a string, `{"ssid":
    null}` -- is refused here rather than turned into a join attempt for a
    network literally called "None".
    """
    try:
        payload = json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    ssid = payload.get("ssid")
    psk = payload.get("psk")
    if not isinstance(ssid, str):
        return None
    if psk is None:
        psk = ""
    if not isinstance(psk, str):
        return None
    return ssid, psk


def dispatch_credentials(data: bytes, on_credentials: Callable[[str, str], None]) -> bool:
    """Handle one credential write: parse, log safely, hand to provisioning.

    The SSID is logged with %r and clipped: it comes from an unauthenticated
    write, and a raw newline in it used to forge whole journal lines. The
    passphrase is never logged. Content validation (length, control
    characters) happens in WifiManager.provision -- core.network.credential_problem
    -- so the refusal reaches the app as the provisioning error instead of
    being dropped silently here.
    """
    credentials = parse_credentials(data)
    if credentials is None:
        log.warning("BLE: malformed credential write ignored")
        return False
    ssid, psk = credentials
    log.info("BLE: received Wi-Fi credentials for %r", ssid[:64])
    on_credentials(ssid, psk)
    return True


def parse_command(data: bytes) -> Optional[dict]:
    """The command object from a command write, or None when it is not one."""
    try:
        payload = json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


class _WriteAssembler:
    """Reassembles a characteristic value written in pieces.

    A value longer than one MTU arrives as a long (prepared) write, and BlueZ
    hands each piece to WriteValue separately with its `offset`. Treating every
    piece as a whole value parsed each as broken JSON and dropped it, while the
    write itself was acknowledged -- so the app waited for an outcome to a write
    the device had thrown away.

    A value is dispatched once the bytes so far parse as JSON; until then they
    are held (bounded by the attribute size). A new write at offset 0 starts over.
    """

    def __init__(self, limit: int = _GATT_MAX_ATTRIBUTE_BYTES) -> None:
        self._limit = limit
        self._buf = bytearray()

    def feed(self, chunk: bytes, offset: int = 0) -> Optional[bytes]:
        if offset == 0:
            if self._buf:
                log.warning("BLE: discarded %d bytes of an incomplete or malformed write",
                            len(self._buf))
            self._buf = bytearray()
        elif offset != len(self._buf):
            log.warning("BLE: write at offset %d does not follow the %d bytes held; ignored",
                        offset, len(self._buf))
            self._buf = bytearray()
            return None
        self._buf.extend(chunk)
        if len(self._buf) > self._limit:
            log.warning("BLE: write longer than %d bytes ignored", self._limit)
            self._buf = bytearray()
            return None
        data = bytes(self._buf)
        try:
            json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None  # incomplete (more pieces to come) or malformed
        self._buf = bytearray()
        return data


class BLEUnavailable(RuntimeError):  # noqa: N818 - established exception name
    """BlueZ / dbus python bindings are not present on this platform."""


class BLEStartError(RuntimeError):
    """The bindings exist but the GATT server could not be brought up (no adapter,
    bluetoothd down, registration refused). Transient: worth retrying later."""


class BLEProvisioningServer:
    """Advertises the provisioning service and dispatches writes to callbacks.

    Args:
        device_name: The advertised local name (e.g. "SensorDyme-4F2A").
        device_info: Dict served by the device-info characteristic.
        on_credentials: Called with (ssid, psk) when the app writes credentials.
        on_command: Called with the parsed command dict for lightweight commands.
        on_auth: Proof-of-possession handler (bytes in, reply bytes out). When
            given, the auth characteristic is served and the plain credential
            characteristic refuses every write.
    """

    def __init__(
        self,
        device_name: str,
        device_info: dict,
        on_credentials: Callable[[str, str], None],
        scan_networks: Optional[Callable[[], list]] = None,
        on_command: Optional[Callable[[dict], None]] = None,
        on_auth: Optional[Callable[[bytes], bytes]] = None,
    ) -> None:
        self._device_name = device_name
        self._on_auth = on_auth
        #: The reply to the last auth write, served by reads of the auth characteristic.
        self._auth_reply = b"{}"
        self._auth_chrc = None
        self._device_info = device_info
        # Scanning happens on a worker thread and reads answer from this cache.
        #
        # It used to run inside read_value(), which executes on the GLib main
        # loop — the single thread that serves every BLE operation. One `nmcli
        # --rescan yes` takes up to 20 seconds, and for that whole time the
        # device could not answer a read, accept a write, or push a status
        # notification. Opening "Choose a Network" therefore froze the entire
        # connection, which is what made that screen slow.
        #
        # The radio is genuinely slow; that part cannot be fixed. What can be
        # fixed is making everything else wait for it.
        self._scan_networks = scan_networks
        self._networks: list = []
        self._scanning = False
        self._scan_lock = threading.Lock()
        self._on_credentials = on_credentials
        self._on_command = on_command
        self._status: dict = {"state": "provisioning"}
        #: monotonic() when the last scan finished; rate-limits implicit refreshes.
        self._scanned_at: Optional[float] = None
        #: Set while a join is using the radio: no implicit or explicit scans.
        self._scans_paused = False
        self._btmgmt_active = False
        self._thread: Optional[threading.Thread] = None
        self._mainloop = None
        self._status_chrc = None
        self._running = False
        self._ready = threading.Event()
        self._error: Optional[str] = None

    # ------------------------------------------------------------------ public API
    #: How long start() waits for the GATT server to be exported and registering.
    _START_TIMEOUT_S = 10.0
    #: A read of the networks characteristic starts a fresh scan only when the
    #: cached list is at least this old. See refresh_networks().
    _IMPLICIT_SCAN_MIN_AGE_S = 15.0

    def start(self) -> None:
        """Bring the GATT server + advertisement up.

        Raises BLEUnavailable off-Pi (no bindings) and BLEStartError when the
        bindings exist but the server could not be brought up -- returning
        normally used to mean only "a thread was started", so a server that died
        during setup was reported as an open setup window.
        """
        try:
            import dbus
            import dbus.mainloop.glib  # noqa: F401
            from gi.repository import GLib
        except ImportError as exc:
            raise BLEUnavailable(
                "BLE provisioning needs BlueZ D-Bus bindings (apt: python3-dbus python3-gi)"
            ) from exc

        self._ready = threading.Event()
        self._error = None
        # Created HERE, before the thread exists, so stop() always has a loop to
        # quit. It used to be created late in _run(); a stop() that arrived first
        # found None, quit nothing, and the thread went on to serve -- and
        # advertise -- for the life of the process on a device past its window.
        self._mainloop = GLib.MainLoop()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="ble-gatt")
        self._thread.start()
        if not self._ready.wait(self._START_TIMEOUT_S):
            log.warning("BLE: GATT setup still in progress after %.0fs", self._START_TIMEOUT_S)
        elif self._error is not None:
            reason = self._error
            self.stop()
            raise BLEStartError(reason)
        # Start scanning immediately rather than when the app asks. Setup takes
        # a person several seconds of tapping to reach the network screen; using
        # those seconds means the list is usually already there on arrival.
        self.refresh_networks()

    def is_alive(self) -> bool:
        """Whether the GATT server thread is still serving."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def failure(self) -> Optional[str]:
        """Why the server failed to start or stopped serving, if it did."""
        return self._error

    def set_scans_paused(self, paused: bool) -> None:
        """Hold off Wi-Fi scans while a join is using the radio."""
        with self._scan_lock:
            self._scans_paused = bool(paused)

    def refresh_networks(self, force: bool = False) -> None:
        """Kick off a scan on a worker thread. Returns immediately.

        Concurrent calls collapse into the running scan: the app re-reading, a
        rescan command, and the periodic refresh must not stack up several
        nmcli invocations against one radio.

        Implicit refreshes (a read of the networks characteristic) are also
        rate-limited to one per _IMPLICIT_SCAN_MIN_AGE_S. Without that the device
        and the app formed a loop: every finished scan pushed `scanning: false`,
        the app answered every such status by re-reading the list, and every
        read started the next scan -- back-to-back radio sweeps for as long as
        the phone stayed connected, the join included. `force` (the explicit
        "rescan" command) skips the age check. Nothing scans while paused.
        """
        if self._scan_networks is None:
            return
        with self._scan_lock:
            if self._scanning or self._scans_paused:
                return
            if (not force and self._scanned_at is not None
                    and time.monotonic() - self._scanned_at < self._IMPLICIT_SCAN_MIN_AGE_S):
                return
            self._scanning = True
        self.set_status(dict(self._status))  # surface scanning=True
        threading.Thread(target=self._scan_worker, daemon=True, name="wifi-scan").start()

    def _scan_worker(self) -> None:
        scan = self._scan_networks
        try:
            networks = (scan() if scan is not None else None) or []
        except Exception as exc:  # noqa: BLE001 - a failed scan is an empty list
            log.warning("BLE: wifi scan failed: %s", exc)
            networks = []
        with self._scan_lock:
            # Keep the previous list on a failed rescan. An empty screen where
            # networks were a moment ago reads as "gone", not as "ask again".
            if networks or not self._networks:
                self._networks = networks
            self._scanning = False
            self._scanned_at = time.monotonic()
        log.info("BLE: wifi scan found %d network(s)", len(networks))
        self.set_status(dict(self._status))  # surface scanning=False

    def stop(self) -> None:
        self._running = False
        self._stop_btmgmt_advertising()
        loop = self._mainloop
        if loop is not None:
            # If the loop is not running yet this quit is lost (GLib resets it on
            # run()); _serve() re-checks _running before running and a timer
            # re-checks it while running, so the thread still exits.
            with contextlib.suppress(Exception):
                loop.quit()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5)
            if thread.is_alive():
                log.warning("BLE provisioning thread did not exit within 5s; it will on its next check")
            self._thread = None
        log.info("BLE provisioning stopped")

    def _keep_running(self) -> bool:
        """GLib timer: quit the loop once stop() has asked, even if its quit() was lost."""
        if self._running:
            return True
        loop = self._mainloop
        if loop is not None:
            loop.quit()
        return False

    # ------------------------------------------------------------------ advertising fallback
    _BTMGMT_INSTANCE = "1"

    def _advertise_via_btmgmt(self) -> None:
        """Advertise through the kernel mgmt API (bypasses bluetoothd's broken path)."""
        import subprocess

        from core.privilege import privileged

        commands = [
            ["btmgmt", "connectable", "on"],
            ["btmgmt", "rm-adv", self._BTMGMT_INSTANCE],  # clear any stale instance
            # -c connectable, -g general-discoverable, -n local name in scan response
            ["btmgmt", "add-adv", "-u", SERVICE_UUID, "-c", "-g", "-n", self._BTMGMT_INSTANCE],
        ]
        for command in commands:
            subcommand = command[1]  # ["btmgmt", <subcommand>, ...]
            try:
                result = subprocess.run(  # noqa: S603 - fixed btmgmt argv built above
                    privileged(command), capture_output=True, text=True, timeout=10, check=False
                )
            except Exception as exc:  # noqa: BLE001
                log.error("btmgmt fallback failed (%s): %s", subcommand, exc)
                return
            if subcommand == "add-adv":
                # Judge it by the exit code, not by what it printed.
                #
                # btmgmt only writes "Instance added: 1" when stdout is a
                # terminal; under systemd it is a pipe and says nothing at all.
                # Matching on that text meant a *successful* advertisement was
                # reported as a failure on every boot — and because the failure
                # path left _btmgmt_active False, teardown then skipped rm-adv
                # and leaked the instance into the next run.
                if result.returncode == 0:
                    self._btmgmt_active = True
                    log.info("BLE advertising via kernel mgmt as '%s'", self._device_name)
                else:
                    # This is the last fallback: when it genuinely fails the
                    # device is invisible and cannot be set up at all, so report
                    # everything rather than a bare "failed:".
                    log.error(
                        "btmgmt add-adv failed (rc=%s) stdout=%r stderr=%r — "
                        "the device will NOT be visible over Bluetooth",
                        result.returncode,
                        result.stdout.strip()[:200],
                        result.stderr.strip()[:200],
                    )

    def _stop_btmgmt_advertising(self) -> None:
        """Remove our advertising instance. Attempted unconditionally.

        Not gated on `_btmgmt_active`: the flag records what this process
        believes it did, and a leaked instance is precisely the case where that
        belief was wrong. Removing an instance that is not there costs one
        harmless non-zero exit; leaving one behind blocks the next boot's
        advertisement and makes the device unsettable.
        """
        import subprocess

        from core.privilege import privileged

        try:
            subprocess.run(  # noqa: S603 - fixed argv
                privileged(["btmgmt", "rm-adv", self._BTMGMT_INSTANCE]),
                capture_output=True, timeout=10, check=False,
            )
        except Exception as exc:  # noqa: BLE001 - teardown must never raise
            log.debug("btmgmt rm-adv failed during teardown: %r", exc)
        self._btmgmt_active = False

    def is_advertising(self) -> bool:
        """Whether the controller is actually advertising, asked of the kernel.

        Observed rather than remembered. Every other signal here is what some
        earlier call *returned*; this is the device looking at itself, which is
        the only answer worth reporting to a person asking why their phone
        cannot see it.

        Asks `advinfo` for the registered instance count, NOT the `advertising`
        flag in `btmgmt info`. On the Pi 5's CYW43455 that flag stays clear
        while instance-based advertising is running, so reading it reports a
        visible device as invisible — the mirror image of the bug above.
        """
        import subprocess

        from core.privilege import privileged

        try:
            result = subprocess.run(  # noqa: S603 - fixed argv
                privileged(["btmgmt", "advinfo"]),
                capture_output=True, text=True, timeout=10, check=False,
            )
        except Exception:  # noqa: BLE001
            return False
        for line in result.stdout.splitlines():
            if "Instances list with" in line:
                digits = "".join(c for c in line if c.isdigit())
                return bool(digits) and int(digits) > 0
        return False

    def set_status(self, status: dict) -> None:
        """Update the status characteristic (and notify a subscribed app)."""
        self._status = dict(status)
        chrc = self._status_chrc
        if chrc is not None:
            try:
                # Scanning rides along with every status push so the app can say
                # "looking for networks" instead of showing an empty list that
                # is indistinguishable from "there are none".
                payload = dict(self._status)
                payload["scanning"] = self._scanning
                chrc.push_value(_encode_status(payload))
            except Exception as exc:  # noqa: BLE001
                log.warning("BLE status notify failed: %s", exc)

    def handle_credentials_write(self, data: bytes) -> bool:
        """A write on the plain credential characteristic. Refused under proof of possession."""
        if self._on_auth is not None:
            log.warning("BLE: plain credential write refused: this device requires proof of the setup code")
            return False
        return dispatch_credentials(data, self._on_credentials)

    def handle_auth_write(self, data: bytes) -> bytes:
        """A write on the auth characteristic: hand it to the protocol, keep the reply for reads."""
        if self._on_auth is None:
            return b"{}"
        try:
            reply = self._on_auth(data)
        except Exception as exc:  # noqa: BLE001 - a protocol fault must not kill the GATT loop
            log.error("BLE: proof-of-possession handler failed: %s", type(exc).__name__)
            reply = b'{"op":"error","error":"internal"}'
        self._auth_reply = reply
        chrc = self._auth_chrc
        if chrc is not None:
            try:
                chrc.push_value(reply)
            except Exception as exc:  # noqa: BLE001
                log.warning("BLE auth notify failed: %s", exc)
        return reply

    # ------------------------------------------------------------------ the GATT table
    def gatt_characteristics(self) -> list[tuple[str, tuple[str, ...]]]:
        """(uuid, flags) of every characteristic served, in registration order.

        The auth characteristic exists only under proof of possession.
        """
        table = [
            (CHAR_DEVICE_INFO_UUID, ("read",)),
            (CHAR_WIFI_CREDS_UUID, ("write",)),
            (CHAR_STATUS_UUID, ("read", "notify")),
            (CHAR_COMMAND_UUID, ("write",)),
            (CHAR_NETWORKS_UUID, ("read",)),
        ]
        if self._on_auth is not None:
            table.append((CHAR_AUTH_UUID, ("read", "write", "notify")))
        return table

    def gatt_read(self, uuid: str, offset: int = 0) -> bytes:
        """The value a read of `uuid` returns (offset 0 = a read from the start)."""
        if uuid == CHAR_DEVICE_INFO_UUID:
            return json.dumps(self._device_info).encode()
        if uuid == CHAR_STATUS_UUID:
            payload = dict(self._status)
            payload["scanning"] = self._scanning
            return _encode_status(payload)
        if uuid == CHAR_NETWORKS_UUID:
            return self.read_networks(offset)
        if uuid == CHAR_AUTH_UUID and self._on_auth is not None:
            return self._auth_reply
        return b"{}"

    def gatt_write(self, uuid: str, data: bytes) -> None:
        """One whole (reassembled) value written to `uuid`."""
        if uuid == CHAR_WIFI_CREDS_UUID:
            self.handle_credentials_write(data)
        elif uuid == CHAR_AUTH_UUID:
            self.handle_auth_write(data)
        elif uuid == CHAR_COMMAND_UUID:
            self.handle_command_write(data)

    def read_networks(self, offset: int = 0) -> bytes:
        """Wi-Fi networks the DEVICE can see, so the app can offer a list.

        Served from a cache refreshed on a worker thread. The read itself must
        return instantly -- it runs on the BLE event loop, so any wait here is a
        wait for every other BLE operation too. Never scan here.
        """
        with self._scan_lock:
            networks = list(self._networks)
        if offset == 0:
            # A read from the start is the app arriving on the network screen,
            # which is when a fresh list is worth starting -- if the cached one
            # is not already fresh (rate-limited inside).
            self.refresh_networks()
        return _fit_networks(networks)

    def handle_command_write(self, data: bytes) -> None:
        if self._on_command is None:
            return
        command = parse_command(data)
        if command is None:
            log.warning("BLE: malformed command write ignored")
            return
        self._on_command(command)

    # ------------------------------------------------------------------ GLib main loop thread
    def _run(self) -> None:
        try:
            self._serve()
        except Exception as exc:  # reported via failure / is_alive()
            if self._error is None:
                self._error = f"{type(exc).__name__}: {exc}"
            log.error("BLE provisioning server failed: %s", exc, exc_info=True)
        finally:
            self._ready.set()

    def _serve(self) -> None:
        import dbus
        import dbus.mainloop.glib
        import dbus.service
        from gi.repository import GLib

        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        # A PRIVATE connection, closed on the way out. dbus.SystemBus() is one
        # connection shared by the whole process, and every object exported on
        # it stays registered until removed -- this never removed them, so the
        # SECOND window in a process died at once with "Can't register the
        # object-path handler ... there is already a handler" while the service
        # reported it open. Every window after the first boot's closed was dead
        # until a reboot.
        bus = dbus.SystemBus(private=True)
        exported: list = []
        managers: list = []
        try:
            adapter_path = self._find_adapter(bus)
            if adapter_path is None:
                self._error = "no Bluetooth adapter found"
                log.error("no Bluetooth adapter found — BLE provisioning unavailable")
                return
            self._export_and_serve(bus, adapter_path, exported, managers, dbus, GLib)
        finally:
            # Teardown must never raise: every step is best effort.
            self._status_chrc = None
            self._auth_chrc = None
            for manager, method, path in managers:
                with contextlib.suppress(Exception):
                    getattr(manager, method)(path)
            for obj in exported:
                with contextlib.suppress(Exception):
                    obj.remove_from_connection()
            with contextlib.suppress(Exception):
                bus.close()
            if self._btmgmt_active:
                self._stop_btmgmt_advertising()

    def _export_and_serve(self, bus, adapter_path, exported, managers, dbus, glib) -> None:
        server = self

        # --- D-Bus scaffolding (BlueZ's standard GATT server shape, condensed) ---
        class Application(dbus.service.Object):
            PATH = "/com/sensordyme/provisioning"

            def __init__(self, bus_):
                self.objects = []
                super().__init__(bus_, self.PATH)

            def add(self, obj):
                self.objects.append(obj)

            @dbus.service.method(DBUS_OM_IFACE, out_signature="a{oa{sa{sv}}}")
            def GetManagedObjects(self):  # noqa: N802 - D-Bus/BlueZ method name
                return {o.path: o.properties() for o in self.objects}

        class Characteristic(dbus.service.Object):
            def __init__(self, bus_, index, uuid, flags, service_path):
                self.path = f"{service_path}/char{index}"
                self.uuid = uuid
                self.flags = flags
                self.service_path = service_path
                self.notifying = False
                self._assembler = _WriteAssembler()
                self._read_cache: Optional[bytes] = None
                super().__init__(bus_, self.path)

            def properties(self):
                return {
                    GATT_CHRC_IFACE: {
                        "Service": dbus.ObjectPath(self.service_path),
                        "UUID": self.uuid,
                        "Flags": dbus.Array(self.flags, signature="s"),
                    }
                }

            def read_value(self, offset=0) -> bytes:
                return b"{}"

            def write_value(self, data: bytes) -> None:
                pass

            @dbus.service.method(GATT_CHRC_IFACE, in_signature="a{sv}", out_signature="ay")
            def ReadValue(self, options):  # noqa: N802 - D-Bus/BlueZ method name
                # Honour the offset. This is how ATT transfers a value larger
                # than one MTU: the central issues Read Blob requests for
                # successive chunks. Ignoring it — which this did — silently
                # truncated every oversized value at the MTU boundary, so the
                # network list arrived as invalid JSON and the app showed no
                # networks at all in exactly the case that matters, a flat with
                # a dozen of them. It was worse on iOS, which negotiates a
                # smaller MTU than macOS.
                #
                # Later chunks come from the value serialised for offset 0, not a
                # fresh one: a scan finishing (or a status change) between two
                # Read Blobs spliced two different documents into invalid JSON.
                offset = int(options.get("offset", 0))
                if offset == 0 or self._read_cache is None:
                    self._read_cache = bytes(self.read_value(offset))
                return dbus.Array(self._read_cache[offset:], signature="y")

            @dbus.service.method(GATT_CHRC_IFACE, in_signature="aya{sv}")
            def WriteValue(self, value, options):  # noqa: N802 - D-Bus/BlueZ method name
                data = self._assembler.feed(bytes(value), int(options.get("offset", 0)))
                if data is not None:
                    self.write_value(data)

            @dbus.service.method(GATT_CHRC_IFACE)
            def StartNotify(self):  # noqa: N802 - D-Bus/BlueZ method name
                self.notifying = True

            @dbus.service.method(GATT_CHRC_IFACE)
            def StopNotify(self):  # noqa: N802 - D-Bus/BlueZ method name
                self.notifying = False

            # BlueZ reads properties off the ObjectManager dict, but also probes
            # the standard Properties interface on each object; without this it
            # errors out mid-registration.
            @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
            def GetAll(self, interface):  # noqa: N802 - D-Bus/BlueZ method name
                return self.properties().get(interface, {})

            @dbus.service.signal(DBUS_PROP_IFACE, signature="sa{sv}as")
            def PropertiesChanged(self, interface, changed, invalidated):  # noqa: N802 - D-Bus/BlueZ method name
                pass

            def push_value(self, data: bytes) -> None:
                if self.notifying:
                    self.PropertiesChanged(
                        GATT_CHRC_IFACE, {"Value": dbus.Array(data, signature="y")}, []
                    )

        class ProvisioningService(dbus.service.Object):
            PATH = "/com/sensordyme/provisioning/service0"

            def __init__(self, bus_):
                self.path = self.PATH
                super().__init__(bus_, self.path)

            def properties(self):
                return {
                    GATT_SERVICE_IFACE: {"UUID": SERVICE_UUID, "Primary": dbus.Boolean(True)}
                }

            @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
            def GetAll(self, interface):  # noqa: N802 - D-Bus/BlueZ method name
                return self.properties().get(interface, {})

        # Each characteristic serves what the server's gatt_read/gatt_write
        # say, so the virtual peripheral (sim/ble_peripheral.py) serves the
        # very same bytes without D-Bus.
        class DeviceInfoChrc(Characteristic):
            def read_value(self, offset=0):
                return server.gatt_read(CHAR_DEVICE_INFO_UUID, offset)

        class WifiCredsChrc(Characteristic):
            def write_value(self, data):
                server.gatt_write(CHAR_WIFI_CREDS_UUID, data)

        class AuthChrc(Characteristic):
            def read_value(self, offset=0):
                return server.gatt_read(CHAR_AUTH_UUID, offset)

            def write_value(self, data):
                server.gatt_write(CHAR_AUTH_UUID, data)

        class NetworksChrc(Characteristic):
            """Wi-Fi networks the DEVICE can see (see BLEProvisioningServer.read_networks)."""

            def read_value(self, offset=0):
                return server.gatt_read(CHAR_NETWORKS_UUID, offset)

        class StatusChrc(Characteristic):
            def read_value(self, offset=0):
                return server.gatt_read(CHAR_STATUS_UUID, offset)

        class CommandChrc(Characteristic):
            def write_value(self, data):
                server.gatt_write(CHAR_COMMAND_UUID, data)

        class Advertisement(dbus.service.Object):
            PATH = "/com/sensordyme/advertisement0"

            def __init__(self, bus_):
                self.path = self.PATH
                super().__init__(bus_, self.path)

            # GetAll takes ONE argument (the interface name) — the previous "ss"
            # signature never matched BlueZ's call, so the advertisement's
            # properties could not be read.
            @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
            def GetAll(self, interface):  # noqa: N802 - D-Bus/BlueZ method name
                return {
                    "Type": "peripheral",
                    "ServiceUUIDs": dbus.Array([SERVICE_UUID], signature="s"),
                    "LocalName": dbus.String(server._device_name),
                    "Discoverable": dbus.Boolean(True),
                }

            @dbus.service.method(LE_AD_IFACE)
            def Release(self):  # noqa: N802 - D-Bus/BlueZ method name
                pass

        app = Application(bus)
        exported.append(app)
        service = ProvisioningService(bus)
        exported.append(service)
        app.add(service)
        status_chrc = None
        classes = {
            CHAR_DEVICE_INFO_UUID: DeviceInfoChrc, CHAR_WIFI_CREDS_UUID: WifiCredsChrc,
            CHAR_STATUS_UUID: StatusChrc, CHAR_COMMAND_UUID: CommandChrc,
            CHAR_NETWORKS_UUID: NetworksChrc, CHAR_AUTH_UUID: AuthChrc,
        }
        characteristics = [(index, classes[uuid], uuid, list(flags))
                           for index, (uuid, flags) in enumerate(self.gatt_characteristics())]
        for index, cls, uuid, flags in characteristics:
            chrc = cls(bus, index, uuid, flags, service.PATH)
            exported.append(chrc)
            app.add(chrc)
            if uuid == CHAR_STATUS_UUID:
                status_chrc = chrc
            if uuid == CHAR_AUTH_UUID:
                self._auth_chrc = chrc

        # The adapter must be powered before anything can be registered on it.
        adapter_props = None
        try:
            adapter_props = dbus.Interface(
                bus.get_object(BLUEZ_SERVICE, adapter_path), DBUS_PROP_IFACE
            )
            adapter_props.Set("org.bluez.Adapter1", "Powered", dbus.Boolean(True))
        except Exception as exc:  # noqa: BLE001
            log.warning("could not power on the Bluetooth adapter: %s", exc)

        gatt_manager = dbus.Interface(
            bus.get_object(BLUEZ_SERVICE, adapter_path), GATT_MANAGER_IFACE
        )
        ad_manager = dbus.Interface(
            bus.get_object(BLUEZ_SERVICE, adapter_path), LE_AD_MANAGER_IFACE
        )
        # Show the device name in scan results: the kernel's scan-response
        # local-name comes from the adapter's Alias.
        try:
            if adapter_props is None:
                raise RuntimeError("no adapter properties interface")
            adapter_props.Set("org.bluez.Adapter1", "Alias", dbus.String(self._device_name))
        except Exception as exc:  # noqa: BLE001
            log.warning("could not set adapter alias: %s", exc)

        advertisement = Advertisement(bus)
        exported.append(advertisement)
        if self._mainloop is None:
            self._mainloop = glib.MainLoop()
        loop = self._mainloop

        # CRITICAL: register ASYNCHRONOUSLY. BlueZ answers RegisterApplication by
        # calling back into this process (GetManagedObjects on our ObjectManager).
        # A synchronous call blocks this thread before the main loop starts, so
        # that callback can never be served -> D-Bus times out after 25s with
        # NoReply. Async handlers let mainloop.run() below service the callback.
        def _registered(what: str):
            log.info("BLE %s registered", what)

        def _failed(what: str, error):
            log.error("BLE %s registration failed: %s", what, error)
            if not self._running:
                return  # stopping: never add an advertisement stop() already removed
            if what == "advertisement":
                # bluetoothd's advertising manager is broken on the Pi 5's
                # controller (BlueZ 5.66 + CYW43455: every AddAdvertising is
                # rejected with "Invalid Parameters"). The kernel's own mgmt
                # interface accepts the identical advertisement, so fall back to
                # it -- the GATT service above is unaffected either way.
                self._advertise_via_btmgmt()
            else:
                # Without the GATT application there is nothing to connect to:
                # end this window so the provisioning service sees it and retries.
                self._error = f"GATT application registration failed: {error}"
                loop.quit()

        if not self._running:
            return  # stop() arrived while this thread was still setting up
        self._status_chrc = status_chrc
        gatt_manager.RegisterApplication(
            Application.PATH, {},
            reply_handler=lambda: _registered("GATT application"),
            error_handler=lambda e: _failed("GATT application", e),
        )
        managers.append((gatt_manager, "UnregisterApplication", Application.PATH))
        ad_manager.RegisterAdvertisement(
            advertisement.PATH, {},
            reply_handler=lambda: _registered("advertisement"),
            error_handler=lambda e: _failed("advertisement", e),
        )
        # Unregister the advertisement before the application, as before.
        managers.insert(0, (ad_manager, "UnregisterAdvertisement", advertisement.PATH))

        glib.timeout_add(250, self._keep_running)
        log.info("BLE provisioning advertising as '%s'", self._device_name)
        self._ready.set()
        loop.run()  # until stop() quits it

    @staticmethod
    def _find_adapter(bus) -> Optional[str]:
        import dbus

        om = dbus.Interface(bus.get_object(BLUEZ_SERVICE, "/"), DBUS_OM_IFACE)
        for path, interfaces in om.GetManagedObjects().items():
            if GATT_MANAGER_IFACE in interfaces:
                return path
        return None
