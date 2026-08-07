"""BlueZ GATT server for Bluetooth-LE Wi-Fi provisioning (Pi-only).

Exposes one primary service the SensorDyme app discovers while the device has
no Wi-Fi:

    Service  5d1e0001-...   SensorDyme provisioning
      char   5d1e0002  read          device info JSON {name, model, version, hostname}
      char   5d1e0003  write         Wi-Fi credentials JSON {"ssid": "...", "psk": "..."}
      char   5d1e0004  read/notify   status JSON {"state": "...", "ssid": ..., "ip": ..., "error": ...}
      char   5d1e0005  write         lightweight commands JSON {"action": "rescan" | "identify"}

Bluetooth is for setup/config/commands ONLY — frames never move over BLE; the
live feed always uses Wi-Fi (see core/dashboard.py's /stream).

Requires BlueZ with dbus-python + PyGObject (`apt install python3-dbus
python3-gi`, installed by setup_pi.sh). All imports are performed at start()
so this file is importable (and the rest of the codebase testable) on a Mac.

Security posture (documented, deliberate): pairing is BLE "Just Works" — the
credential write window only exists while the device is unprovisioned, which
requires physical presence during setup, and closes the moment Wi-Fi connects.
"""

from __future__ import annotations

import json
import threading
from typing import Callable, Optional

from core.logging_setup import get_logger

log = get_logger("ble")

SERVICE_UUID = "5d1e0001-5344-4d45-b000-536e447a6d65"
CHAR_DEVICE_INFO_UUID = "5d1e0002-5344-4d45-b000-536e447a6d65"
CHAR_WIFI_CREDS_UUID = "5d1e0003-5344-4d45-b000-536e447a6d65"
CHAR_STATUS_UUID = "5d1e0004-5344-4d45-b000-536e447a6d65"
CHAR_COMMAND_UUID = "5d1e0005-5344-4d45-b000-536e447a6d65"
CHAR_NETWORKS_UUID = "5d1e0006-5344-4d45-b000-536e447a6d65"

BLUEZ_SERVICE = "org.bluez"
GATT_MANAGER_IFACE = "org.bluez.GattManager1"
LE_AD_MANAGER_IFACE = "org.bluez.LEAdvertisingManager1"
GATT_SERVICE_IFACE = "org.bluez.GattService1"
GATT_CHRC_IFACE = "org.bluez.GattCharacteristic1"
LE_AD_IFACE = "org.bluez.LEAdvertisement1"
DBUS_OM_IFACE = "org.freedesktop.DBus.ObjectManager"
DBUS_PROP_IFACE = "org.freedesktop.DBus.Properties"


class BLEUnavailable(RuntimeError):
    """BlueZ / dbus python bindings are not present on this platform."""


class BLEProvisioningServer:
    """Advertises the provisioning service and dispatches writes to callbacks.

    Args:
        device_name: The advertised local name (e.g. "SensorDyme-4F2A").
        device_info: Dict served by the device-info characteristic.
        on_credentials: Called with (ssid, psk) when the app writes credentials.
        on_command: Called with the parsed command dict for lightweight commands.
    """

    def __init__(
        self,
        device_name: str,
        device_info: dict,
        on_credentials: Callable[[str, str], None],
        scan_networks: Optional[Callable[[], list]] = None,
        on_command: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self._device_name = device_name
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
        self._btmgmt_active = False
        self._thread: Optional[threading.Thread] = None
        self._mainloop = None
        self._status_chrc = None
        self._running = False

    # ------------------------------------------------------------------ public API
    def start(self) -> None:
        """Bring the GATT server + advertisement up. Raises BLEUnavailable off-Pi."""
        try:
            import dbus  # noqa: F401
            import dbus.mainloop.glib  # noqa: F401
            from gi.repository import GLib  # noqa: F401
        except ImportError as exc:
            raise BLEUnavailable(
                "BLE provisioning needs BlueZ D-Bus bindings (apt: python3-dbus python3-gi)"
            ) from exc

        self._thread = threading.Thread(target=self._run, daemon=True, name="ble-gatt")
        self._running = True
        self._thread.start()
        # Start scanning immediately rather than when the app asks. Setup takes
        # a person several seconds of tapping to reach the network screen; using
        # those seconds means the list is usually already there on arrival.
        self.refresh_networks()

    def refresh_networks(self) -> None:
        """Kick off a scan on a worker thread. Returns immediately.

        Concurrent calls collapse into the running scan: the app re-reading, a
        rescan command, and the periodic refresh must not stack up several
        nmcli invocations against one radio.
        """
        if self._scan_networks is None:
            return
        with self._scan_lock:
            if self._scanning:
                return
            self._scanning = True
        self.set_status(dict(self._status))  # surface scanning=True
        threading.Thread(target=self._scan_worker, daemon=True, name="wifi-scan").start()

    def _scan_worker(self) -> None:
        try:
            networks = self._scan_networks() or []
        except Exception as exc:  # noqa: BLE001 - a failed scan is an empty list
            log.warning("BLE: wifi scan failed: %s", exc)
            networks = []
        with self._scan_lock:
            # Keep the previous list on a failed rescan. An empty screen where
            # networks were a moment ago reads as "gone", not as "ask again".
            if networks or not self._networks:
                self._networks = networks
            self._scanning = False
        log.info("BLE: wifi scan found %d network(s)", len(networks))
        self.set_status(dict(self._status))  # surface scanning=False

    def stop(self) -> None:
        self._running = False
        self._stop_btmgmt_advertising()
        if self._mainloop is not None:
            try:
                self._mainloop.quit()
            except Exception:  # noqa: BLE001
                pass
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        log.info("BLE provisioning stopped")

    # ------------------------------------------------------------------ advertising fallback
    _BTMGMT_INSTANCE = "1"

    def _advertise_via_btmgmt(self) -> None:
        """Advertise through the kernel mgmt API (bypasses bluetoothd's broken path)."""
        import subprocess

        commands = [
            ["sudo", "-n", "btmgmt", "connectable", "on"],
            ["sudo", "-n", "btmgmt", "rm-adv", self._BTMGMT_INSTANCE],  # clear any stale instance
            # -c connectable, -g general-discoverable, -n local name in scan response
            ["sudo", "-n", "btmgmt", "add-adv", "-u", SERVICE_UUID, "-c", "-g", "-n",
             self._BTMGMT_INSTANCE],
        ]
        for command in commands:
            subcommand = command[3]  # ["sudo", "-n", "btmgmt", <subcommand>, ...]
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=10)
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

        try:
            subprocess.run(
                ["sudo", "-n", "btmgmt", "rm-adv", self._BTMGMT_INSTANCE],
                capture_output=True, timeout=10,
            )
        except Exception:  # noqa: BLE001 - teardown must never raise
            pass
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

        try:
            result = subprocess.run(
                ["sudo", "-n", "btmgmt", "advinfo"],
                capture_output=True, text=True, timeout=10,
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
                chrc.push_value(json.dumps(payload).encode())
            except Exception as exc:  # noqa: BLE001
                log.warning("BLE status notify failed: %s", exc)

    # ------------------------------------------------------------------ GLib main loop thread
    def _run(self) -> None:
        import dbus
        import dbus.mainloop.glib
        import dbus.service
        from gi.repository import GLib

        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        bus = dbus.SystemBus()
        adapter_path = self._find_adapter(bus)
        if adapter_path is None:
            log.error("no Bluetooth adapter found — BLE provisioning unavailable")
            return

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
            def GetManagedObjects(self):
                return {o.path: o.properties() for o in self.objects}

        class Characteristic(dbus.service.Object):
            def __init__(self, bus_, index, uuid, flags, service_path):
                self.path = f"{service_path}/char{index}"
                self.uuid = uuid
                self.flags = flags
                self.service_path = service_path
                self.notifying = False
                super().__init__(bus_, self.path)

            def properties(self):
                return {
                    GATT_CHRC_IFACE: {
                        "Service": dbus.ObjectPath(self.service_path),
                        "UUID": self.uuid,
                        "Flags": dbus.Array(self.flags, signature="s"),
                    }
                }

            def read_value(self) -> bytes:
                return b"{}"

            def write_value(self, data: bytes) -> None:
                pass

            @dbus.service.method(GATT_CHRC_IFACE, in_signature="a{sv}", out_signature="ay")
            def ReadValue(self, options):
                return dbus.Array(self.read_value(), signature="y")

            @dbus.service.method(GATT_CHRC_IFACE, in_signature="aya{sv}")
            def WriteValue(self, value, options):
                self.write_value(bytes(value))

            @dbus.service.method(GATT_CHRC_IFACE)
            def StartNotify(self):
                self.notifying = True

            @dbus.service.method(GATT_CHRC_IFACE)
            def StopNotify(self):
                self.notifying = False

            # BlueZ reads properties off the ObjectManager dict, but also probes
            # the standard Properties interface on each object; without this it
            # errors out mid-registration.
            @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
            def GetAll(self, interface):
                return self.properties().get(interface, {})

            @dbus.service.signal(DBUS_PROP_IFACE, signature="sa{sv}as")
            def PropertiesChanged(self, interface, changed, invalidated):
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
            def GetAll(self, interface):
                return self.properties().get(interface, {})

        class DeviceInfoChrc(Characteristic):
            def read_value(self):
                return json.dumps(server._device_info).encode()

        class WifiCredsChrc(Characteristic):
            def write_value(self, data):
                try:
                    payload = json.loads(data.decode())
                    ssid, psk = str(payload.get("ssid", "")), str(payload.get("psk", ""))
                except (ValueError, UnicodeDecodeError):
                    log.warning("BLE: malformed credential write ignored")
                    return
                # Log the SSID only — never the passphrase.
                log.info("BLE: received Wi-Fi credentials for '%s'", ssid)
                server._on_credentials(ssid, psk)

        class NetworksChrc(Characteristic):
            """Wi-Fi networks the DEVICE can see, so the app can offer a list.

            Served from a cache refreshed on a worker thread. The read itself
            must return instantly — it runs on the BLE event loop, so any wait
            here is a wait for every other BLE operation too.
            """

            def read_value(self):
                # Answers from cache, always. Never scan here: this runs on the
                # GLib main loop and blocking it stalls the whole connection.
                with server._scan_lock:
                    networks = list(server._networks)
                # A read is the app saying it is on the network screen now, so
                # this is exactly when a fresh list is worth starting.
                server.refresh_networks()
                payload = json.dumps(networks).encode()
                # BLE reads are MTU-bounded and this stack does not implement
                # long reads. Twelve networks is more than anyone picks from and
                # stays comfortably inside a default MTU.
                if len(payload) > 480:
                    payload = json.dumps(networks[:12]).encode()
                return payload

        class StatusChrc(Characteristic):
            def read_value(self):
                payload = dict(server._status)
                payload["scanning"] = server._scanning
                return json.dumps(payload).encode()

        class CommandChrc(Characteristic):
            def write_value(self, data):
                if server._on_command is None:
                    return
                try:
                    server._on_command(json.loads(data.decode()))
                except (ValueError, UnicodeDecodeError):
                    log.warning("BLE: malformed command write ignored")

        class Advertisement(dbus.service.Object):
            PATH = "/com/sensordyme/advertisement0"

            def __init__(self, bus_):
                self.path = self.PATH
                super().__init__(bus_, self.path)

            # GetAll takes ONE argument (the interface name) — the previous "ss"
            # signature never matched BlueZ's call, so the advertisement's
            # properties could not be read.
            @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
            def GetAll(self, interface):
                return {
                    "Type": "peripheral",
                    "ServiceUUIDs": dbus.Array([SERVICE_UUID], signature="s"),
                    "LocalName": dbus.String(server._device_name),
                    "Discoverable": dbus.Boolean(True),
                }

            @dbus.service.method(LE_AD_IFACE)
            def Release(self):
                pass

        app = Application(bus)
        service = ProvisioningService(bus)
        app.add(service)
        for index, cls, uuid, flags in (
            (0, DeviceInfoChrc, CHAR_DEVICE_INFO_UUID, ["read"]),
            (1, WifiCredsChrc, CHAR_WIFI_CREDS_UUID, ["write"]),
            (2, StatusChrc, CHAR_STATUS_UUID, ["read", "notify"]),
            (3, CommandChrc, CHAR_COMMAND_UUID, ["write"]),
            (4, NetworksChrc, CHAR_NETWORKS_UUID, ["read"]),
        ):
            chrc = cls(bus, index, uuid, flags, service.PATH)
            app.add(chrc)
            if uuid == CHAR_STATUS_UUID:
                self._status_chrc = chrc

        # The adapter must be powered before anything can be registered on it.
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
            adapter_props.Set("org.bluez.Adapter1", "Alias", dbus.String(self._device_name))
        except Exception as exc:  # noqa: BLE001
            log.warning("could not set adapter alias: %s", exc)

        advertisement = Advertisement(bus)
        self._mainloop = GLib.MainLoop()

        # CRITICAL: register ASYNCHRONOUSLY. BlueZ answers RegisterApplication by
        # calling back into this process (GetManagedObjects on our ObjectManager).
        # A synchronous call blocks this thread before the main loop starts, so
        # that callback can never be served -> D-Bus times out after 25s with
        # NoReply. Async handlers let mainloop.run() below service the callback.
        def _registered(what: str):
            log.info("BLE %s registered", what)

        def _failed(what: str, error):
            log.error("BLE %s registration failed: %s", what, error)
            if what == "advertisement":
                # bluetoothd's advertising manager is broken on the Pi 5's
                # controller (BlueZ 5.66 + CYW43455: every AddAdvertising is
                # rejected with "Invalid Parameters"). The kernel's own mgmt
                # interface accepts the identical advertisement, so fall back to
                # it -- the GATT service above is unaffected either way.
                self._advertise_via_btmgmt()

        gatt_manager.RegisterApplication(
            Application.PATH, {},
            reply_handler=lambda: _registered("GATT application"),
            error_handler=lambda e: _failed("GATT application", e),
        )
        ad_manager.RegisterAdvertisement(
            advertisement.PATH, {},
            reply_handler=lambda: _registered("advertisement"),
            error_handler=lambda e: _failed("advertisement", e),
        )

        log.info("BLE provisioning advertising as '%s'", self._device_name)
        try:
            self._mainloop.run()  # until stop() quits it
        finally:
            for manager, path, kind in (
                (ad_manager, advertisement.PATH, "UnregisterAdvertisement"),
                (gatt_manager, Application.PATH, "UnregisterApplication"),
            ):
                try:
                    getattr(manager, kind)(path)
                except Exception:  # noqa: BLE001 - teardown must never raise
                    pass

    @staticmethod
    def _find_adapter(bus) -> Optional[str]:
        import dbus

        om = dbus.Interface(bus.get_object(BLUEZ_SERVICE, "/"), DBUS_OM_IFACE)
        for path, interfaces in om.GetManagedObjects().items():
            if GATT_MANAGER_IFACE in interfaces:
                return path
        return None
