"""ProvisioningService: the networking workflow the device follows on every boot.

    boot -> NetworkManager auto-joins any saved Wi-Fi
         -> online?  --yes-->  NORMAL (BLE off; mDNS + dashboard on Wi-Fi)
                 |no
                 v
         PROVISIONING: advertise over BLE as "SensorDyme-XXXX"
             app pairs -> writes {ssid, psk} -> nmcli joins & SAVES the profile
                 -> online -> BLE torn down -> NORMAL
             connect fails -> error pushed to the status characteristic; window stays open

BLE is for provisioning/config/commands only. The moment Wi-Fi is up the GATT
server is stopped; video streaming always rides Wi-Fi. If Wi-Fi later drops
(drove away from home), the device keeps detecting/alerting standalone — and
this service re-opens the BLE window so the app can hand it a new network
(e.g. a phone hotspot) without any SD-card surgery.

Method selection (`network.provisioning.method` in config.yaml):
    ble      -- this in-process GATT flow (default). The window opens the
                moment the device is offline and closes the moment Wi-Fi is up,
                so it is there when someone needs it and invisible otherwise.
    hotspot  -- legacy scripts/provisioning_hotspot.sh oneshot (kept for
                installs without BLE); this service stays dormant
    off      -- no provisioning; Wi-Fi is whatever NetworkManager already knows
"""

from __future__ import annotations

import socket
import threading
from typing import Optional

from core.logging_setup import get_logger
from core.identity import device_id, display_name
from core.network import WifiManager, nmcli_available

log = get_logger("provisioning")


def _ble_name() -> str:
    """The name shown in the app's Bluetooth list, derived from the device ID.

    Display only. The app identifies devices by CoreBluetooth's peripheral UUID
    and by the `id` field below — never by this string, which is why a stale
    adapter alias can no longer split one Pi into two list entries.
    """
    return display_name()


class ProvisioningService:
    """Wi-Fi watcher + BLE provisioning window. Runs as a supervised service."""

    name = "provisioning"

    # Ceiling on how long to wait between checks while the device is offline or
    # the setup window is open. Bounds "network vanished" -> "visible on
    # Bluetooth" to about a second rather than a full poll interval of apparent
    # nothing. A ceiling, not a fixed value: a caller that asked to be polled
    # faster than this meant it, and must not be slowed down by a floor.
    _UNSETTLED_POLL_CEILING_S = 1.0

    def __init__(self, wifi: WifiManager, config: dict, version: str = "dev") -> None:
        self._wifi = wifi
        self._method = str(config.get("method", "ble"))
        self._grace_s = float(config.get("grace_s", 0))
        self._poll_s = float(config.get("poll_s", 10))
        # How long after boot Bluetooth stays open even when Wi-Fi is working.
        #
        # Without this, moving a device to a new network requires first making it
        # fail: the setup window only existed while offline, so a working device
        # could not be reconfigured at all. The consumer answer is that a device
        # you just switched on is always reachable for a few minutes — long
        # enough to re-point it, short enough that it is not permanently open.
        self._reconfigure_window_s = float(config.get("reconfigure_window_s", 300))
        self._version = version
        self._ble = None  # BLEProvisioningServer, created lazily
        self._ble_unavailable = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._connect_thread: Optional[threading.Thread] = None
        # Somebody has successfully set this device up over Bluetooth since it
        # booted. The reconfigure window exists for a device nobody has touched
        # yet; once it has been pointed at a network on purpose, leaving the
        # window open is exposure with no remaining purpose.
        self._provisioned_this_session = False
        self.state = "starting"  # starting | online | waiting | provisioning | unavailable

    # ------------------------------------------------------------------ Service lifecycle
    def start(self, health=None) -> None:  # noqa: ANN001 - supervisor duck-type
        self._stop.clear()
        self._thread = threading.Thread(target=self.run, args=(self._stop,), daemon=True, name=self.name)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        # Wake the run loop rather than letting it sit out its interval.
        wake = getattr(self, "_wake", None)
        if wake is not None:
            wake.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None
        self._teardown_ble()

    def restart(self) -> None:
        self.stop()
        self.start()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------ state machine
    def run(self, stop: threading.Event) -> None:
        if self._method != "ble" or not nmcli_available():
            # Dev Mac, or hotspot/off installs: nothing to orchestrate in-process.
            self.state = "unavailable" if self._method == "ble" else self._method
            return

        import time

        offline_since: Optional[float] = None
        started_at = time.monotonic()

        # Wake the moment NetworkManager reports a change, instead of finding
        # out on the next tick. Measured on hardware, polling put 8.1 seconds
        # between "Wi-Fi vanished" and "visible on Bluetooth" — almost all of it
        # spent asleep in the settled interval, doing nothing.
        changed = threading.Event()
        self._wifi.watch(changed.set, stop)
        self._wake = changed

        while not stop.is_set():
            now = time.monotonic()
            online = self._wifi.is_online()

            if online:
                offline_since = None
                # A device that has just been switched on stays reachable over
                # Bluetooth for a few minutes even though its Wi-Fi is fine.
                # This is what makes "I moved it to a new house" possible: the
                # window used to exist only while offline, so reconfiguring a
                # working device meant first breaking it.
                reconfigurable = (
                    (now - started_at) < self._reconfigure_window_s
                    and not self._provisioned_this_session
                )
                if reconfigurable:
                    self.state = "reconfigurable"
                    if self._ble is None and not self._ble_unavailable:
                        self._open_ble_window()
                    self._push_status(self._describe())
                else:
                    if self._ble is not None:
                        ssid = self._wifi.current_ssid()
                        self._push_status(self._describe())
                        log.info(
                            "Wi-Fi is up (%s) and the setup window has elapsed — closing Bluetooth",
                            ssid,
                        )
                        self._teardown_ble()
                    self.state = "online"
            else:
                if offline_since is None:
                    offline_since = now
                    self.state = "waiting"
                # Open the window as soon as the device is offline. `grace_s`
                # used to default to 45, which meant someone standing next to a
                # freshly-moved device with the app open waited three quarters
                # of a minute for anything to appear on Bluetooth — long enough
                # to conclude it was broken and go looking for a bug.
                #
                # Advertising a few seconds longer on a normal boot costs almost
                # nothing: the window closes the instant Wi-Fi comes up, which
                # on a known network is well before anyone could connect to it.
                # Waiting costs the entire setup experience.
                if now - offline_since >= self._grace_s and self._ble is None and not self._ble_unavailable:
                    self._open_ble_window()
                if self._ble is not None:
                    self.state = "provisioning"
            # Poll quickly whenever something is expected to change, slowly when
            # nothing is. A settled device on known Wi-Fi has no reason to be
            # asked every second; a device that just lost its network is one
            # somebody is probably standing over, and up to ten seconds of
            # apparent nothing is exactly the pause that reads as broken.
            settled = online and self._ble is None
            interval = (
                self._poll_s if settled
                else min(self._poll_s, self._UNSETTLED_POLL_CEILING_S)
            )
            # The interval is now a backstop, not the mechanism: a state change
            # wakes this immediately, and the timeout only covers what nmcli
            # monitor might not report.
            changed.clear()
            changed.wait(timeout=interval)

    def _describe(self) -> dict:
        """The device's own account of where it is in setup.

        Sent over Bluetooth so the app can show real steps rather than a
        spinner. Every field here is observed, never assumed.
        """
        online = self._wifi.is_online()
        return {
            "state": "online" if online else self.state,
            "ssid": self._wifi.current_ssid() if online else None,
            "ip": self._wifi.ip_address() if online else None,
            "device_id": device_id(),
            "reconfigurable": self.state == "reconfigurable",
        }

    def status(self) -> tuple[bool, str]:
        """(healthy, detail) for the health monitor (non-critical component)."""
        if self.state == "provisioning":
            return True, "BLE provisioning window open (device unprovisioned or Wi-Fi lost)"
        if self.state == "reconfigurable":
            return True, "online; Bluetooth still open so the device can be moved to another network"
        if self.state == "unavailable":
            return False, "BLE provisioning requested but BlueZ bindings are missing"
        return True, f"state: {self.state}"

    # ------------------------------------------------------------------ BLE window
    def _open_ble_window(self) -> None:
        from services.ble_gatt import BLEProvisioningServer, BLEUnavailable

        name = _ble_name()
        try:
            self._ble = BLEProvisioningServer(
                device_name=name,
                device_info={
                    # The stable identity, so the app can recognise this device
                    # as the same one it already knows however it was found.
                    "id": device_id(),
                    "name": name,
                    "model": "SensorDyme",
                    "version": self._version,
                    "hostname": socket.gethostname(),
                },
                on_credentials=self._on_credentials,
                scan_networks=lambda: self._wifi.scan(),
                on_command=self._on_command,
            )
            self._ble.start()
            self.state = "provisioning"
            log.info("no Wi-Fi — Bluetooth setup open as '%s'; pair from the SensorDyme app", name)
        except BLEUnavailable as exc:
            self._ble = None
            self._ble_unavailable = True
            self.state = "unavailable"
            log.error("%s", exc)

    def _teardown_ble(self) -> None:
        if self._ble is not None:
            self._ble.stop()
            self._ble = None

    def _push_status(self, status: dict) -> None:
        if self._ble is not None:
            self._ble.set_status(status)

    # ------------------------------------------------------------------ app callbacks
    def _on_credentials(self, ssid: str, psk: str) -> None:
        """The app wrote Wi-Fi credentials: join + save; report the outcome over BLE.

        Returns immediately and joins on a worker thread. This is not an
        optimisation — it is the difference between the app showing progress and
        the app appearing to hang.

        This callback runs inside the BLE write handler, on the GLib main loop
        thread. Joining a network takes up to 45 seconds, and while that call
        blocked the loop, nothing could be delivered to the phone: the
        "connecting" status was queued before the join and could not leave until
        after it finished. The user saw "Sending…" and total silence for the
        entire attempt, which reads as a broken device rather than a working one
        doing something slow.
        """
        if self._connect_thread is not None and self._connect_thread.is_alive():
            # A second write while the first is still going would race two nmcli
            # invocations against the same radio.
            log.info("BLE: ignoring credentials for '%s' — a join is already running", ssid)
            return

        self._push_status({"state": "connecting", "ssid": ssid})
        self._connect_thread = threading.Thread(
            target=self._join_network, args=(ssid, psk),
            daemon=True, name="wifi-join",
        )
        self._connect_thread.start()

    def _join_network(self, ssid: str, psk: str) -> None:
        """Join `ssid` off the event loop, reporting the outcome over BLE."""
        try:
            ok, message = self._wifi.connect(ssid, psk)  # nmcli saves the profile on success
        except Exception as exc:  # noqa: BLE001 - the window must survive a failed join
            log.error("wifi join failed: %s", exc)
            self._push_status({"state": "error", "ssid": ssid, "error": str(exc)})
            return

        if ok:
            self._provisioned_this_session = True
            self._push_status(
                {"state": "online", "ssid": ssid, "ip": self._wifi.ip_address(), "message": message}
            )
            # The run() loop sees is_online() and closes the BLE window.
        else:
            self._push_status({"state": "error", "ssid": ssid, "error": message})

    def _on_command(self, command: dict) -> None:
        """Handle an app command. Runs on the BLE event loop — never block here.

        This previously shelled out to `nmcli --rescan yes` inline, which stalls
        every BLE operation for as long as the radio takes. Rescanning is now
        the BLE server's own non-blocking refresh; the app learns the outcome
        from the status push and re-reads the list.
        """
        action = command.get("action")
        if action == "rescan":
            if self._ble is not None:
                self._ble.refresh_networks()
        elif action == "identify":
            log.info("BLE identify requested")  # future: blink the status LED via the HAL
