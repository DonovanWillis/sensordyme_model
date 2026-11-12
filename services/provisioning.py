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

Who may write credentials (`network.provisioning.auth`):
    legacy   -- anyone in Bluetooth range while the window is open ("Just Works").
                Development only: refused outright when the top-level
                `deployment:` is `production`, and the window then stays shut.
    pop      -- only a phone that proves it holds this device's setup code (the
                label / QR code; `setup_code_path`), with the credentials sent
                encrypted (services/ble_pop.py). A missing or loosely-permissioned
                code file keeps the window shut and is reported in health; there
                is no fallback to legacy.
"""

from __future__ import annotations

import socket
import threading
from collections.abc import Callable
from typing import Any, Optional

from core.identity import device_id, display_name
from core.logging_setup import get_logger
from core.network import WifiManager, nmcli_available
from services.ble_pop import Limits, PopAuthenticator, SetupCodeError, load_setup_code, resolve_auth

log = get_logger("provisioning")

#: Where the factory writes the setup code (scripts/ble_setup_code.py), mode 0600.
DEFAULT_SETUP_CODE_PATH = "/etc/sensordyme/ble_setup_code"


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

    # How long to wait before trying to open Bluetooth again after the GATT
    # server failed to start or died (bluetoothd restarting, adapter rfkilled).
    # A retry, not a give-up: the window is the device's only way back onto a
    # network, so a transient BlueZ fault must not close it for the whole boot.
    _BLE_RETRY_S = 30.0

    # Methods whose in-process loop is dormant BY DESIGN (run() returns at once).
    _DORMANT_METHODS = ("off", "hotspot")

    def __init__(self, wifi: WifiManager, config: dict, version: str = "dev",
                 deployment: str = "development", *,
                 ble_server_factory: Optional[Callable[..., Any]] = None,
                 identity: Optional[tuple[str, str]] = None,
                 radio_available: Optional[Callable[[], bool]] = None,
                 pop_limits: Optional[Limits] = None) -> None:
        """
        The keyword-only arguments are seams for the virtual Bluetooth peripheral
        (sim/ble_peripheral.py), which runs THIS state machine with an in-memory
        GATT table and a simulated radio. Production passes none of them: the
        BlueZ server, core.identity, nmcli and the protocol's own limits.
        """
        self._wifi = wifi
        self._ble_server_factory = ble_server_factory
        self._identity = identity
        self._radio_available = radio_available
        self._pop_limits = pop_limits
        self._method = str(config.get("method", "ble"))
        # Proof of possession (services/ble_pop.py). A refused or unusable policy
        # is a reason the window cannot open, never a reason to open it unguarded.
        self._auth_error: Optional[str] = None
        try:
            self._auth_mode = resolve_auth(config, deployment)
        except SetupCodeError as exc:
            self._auth_mode = "refused"
            self._auth_error = str(exc)
        self._setup_code_path = str(config.get("setup_code_path", DEFAULT_SETUP_CODE_PATH))
        self._pop: Optional[PopAuthenticator] = None
        self._grace_s = float(config.get("grace_s", 0))
        self._poll_s = float(config.get("poll_s", 10))
        # Whether Bluetooth setup is reachable whenever the device is running.
        #
        # It first shipped as a few-minute window after boot, on the theory that
        # a shorter exposure is safer. In practice that made the product's
        # central promise impossible: a device that has been sitting on a shelf
        # working for a week is exactly the one you want to move to a new house,
        # and its window closed six days ago. The only way to reconfigure it was
        # to break its Wi-Fi first — which is the problem, not the solution.
        #
        # Keeping it open permanently was justified by authentication that was
        # never built: the comment here cited a core/pairing.py that does not
        # exist, and the credential characteristic is a plain BLE "write" with no
        # pairing behind it. Left open, anyone within Bluetooth range could move
        # a working camera onto a network of their choosing at any time.
        #
        # So the default is the window: open while offline, and for
        # `reconfigure_window_s` after every boot. Moving a working device to a
        # new house is a power-cycle away. `always_available: true` remains for
        # someone who accepts that exposure, and is the setting to revisit once
        # credential writes are actually authenticated.
        self._always_available = bool(config.get("always_available", False))
        self._reconfigure_window_s = float(config.get("reconfigure_window_s", 300))
        # How long Bluetooth stays up after a credential write has an outcome.
        #
        # The outcome travels over the very channel the success closes: a device
        # that joined Wi-Fi past its reconfigure window used to tear Bluetooth
        # down the moment NetworkManager said "connected" -- before the join
        # worker had even pushed "ok" -- so the app saw the link vanish mid-join
        # and reported a failure for a setup that had worked.
        self._outcome_linger_s = float(config.get("outcome_linger_s", 10.0))
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
        # The outcome of the most recent credential write, reported separately
        # from connectivity.
        #
        # These were one field, and the periodic connectivity push overwrote the
        # provisioning result about a second after it was set — so a device that
        # had just REFUSED a network reported "online" (on the old one), and the
        # app read that as the new network succeeding. A failed setup showed as
        # a successful one, which is the worst way for this to be wrong.
        self._provision: Optional[dict] = None
        #: Incremented on every credential write and echoed in `provision`, so
        #: the app can tell the answer to ITS write from a status describing an
        #: earlier attempt that was still in flight when it pressed Send.
        self._attempt = 0
        #: monotonic() when the current `provision` reached ok/error (linger clock).
        self._outcome_at: Optional[float] = None
        #: When the BLE server last failed to start or died, and why.
        self._ble_failed_at: Optional[float] = None
        self._ble_failure: Optional[str] = None
        #: Last values _describe() observed, so the credential write handler --
        #: which runs on the GLib loop thread -- can answer without running nmcli.
        self._last_described: dict = {}
        #: Last status pushed, so an unchanged description is not re-sent.
        self._last_pushed: Optional[dict] = None
        self.state = "starting"  # starting | online | waiting | provisioning | unavailable
        #: Set by _guarded_run when the thread dies of an unhandled exception, so
        #: status() can report the failure instead of the stale `state` string.
        self._crashed: Optional[str] = None
        self._health = None

    # ------------------------------------------------------------------ Service lifecycle
    def start(self, health=None) -> None:
        # Same three guarantees services/capture.py and services/supervisor.py both
        # make, and which this class -- which duck-types Service rather than
        # subclassing it -- was missing entirely:
        #
        #   1. refuse to double-run. `stop()` sets `self._thread = None` even when the
        #      join TIMED OUT, so a wedged thread (BLE calls can block) left nothing
        #      to detect it and a restart started a second one alongside it, both
        #      driving the same BlueZ objects.
        #   2. a FRESH stop Event per generation, never a cleared shared one. An
        #      abandoned thread polls the OLD object; clearing it told that thread to
        #      resume forever instead of exiting at its next check.
        #   3. crash containment via _guarded_run, so an exception marks the component
        #      failed instead of killing the thread silently. This one mattered most:
        #      status() reports `self.state`, a plain string the dead thread left
        #      behind, so a crashed provisioning service went on reporting HEALTHY
        #      indefinitely while Wi-Fi setup quietly stopped working.
        if self._thread is not None and self._thread.is_alive():
            log.warning("provisioning service is still running; not starting a second thread")
            return
        if health is not None:
            self._health = health
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._guarded_run, args=(self._stop,), daemon=True, name=self.name
        )
        self._thread.start()

    def _guarded_run(self, stop: threading.Event) -> None:
        """Crash containment: a provisioning crash must be visible, not silent."""
        try:
            self.run(stop)
        except Exception as exc:
            log.error("provisioning service crashed: %s", exc, exc_info=True)
            self._crashed = f"crashed: {exc}"
            health = getattr(self, "_health", None)
            if health is not None:
                health.report_failure(self.name, self._crashed)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        # Wake the run loop rather than letting it sit out its interval.
        wake = getattr(self, "_wake", None)
        if wake is not None:
            wake.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                # Keep the handle. Dropping it here is what let start() launch a
                # second thread beside a wedged one (BLE calls can block), both
                # driving the same BlueZ objects; start() refuses while it lives.
                log.warning("provisioning thread did not stop within %.0fs", timeout)
            else:
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
        if self._method != "ble" or not (self._radio_available or nmcli_available)():
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
            self._reap_dead_ble(now)
            online = self._wifi.is_online()
            joining = self._join_in_flight()
            # A join that has an outcome keeps the channel open for a while, so
            # the outcome is delivered (and re-readable) before Bluetooth closes.
            lingering = (
                self._outcome_at is not None
                and now - self._outcome_at < self._outcome_linger_s
            )
            window_left: Optional[float] = None

            if online:
                offline_since = None
                # A device that has just been switched on stays reachable over
                # Bluetooth for a few minutes even though its Wi-Fi is fine.
                # This is what makes "I moved it to a new house" possible: the
                # window used to exist only while offline, so reconfiguring a
                # working device meant first breaking it.
                if self._always_available:
                    reconfigurable = True
                else:
                    window_left = self._reconfigure_window_s - (now - started_at)
                    reconfigurable = window_left > 0 and not self._provisioned_this_session
                if reconfigurable:
                    if self._ble is None and self._may_open_ble(now):
                        self._open_ble_window()
                    # After opening, which sets "provisioning": the first status
                    # pushed used to say reconfigurable=false for that reason.
                    self.state = "reconfigurable"
                    self._push_status(self._describe(online=True))
                elif self._ble is not None and (joining or lingering):
                    # Online -- quite possibly BECAUSE of the join in flight --
                    # but the app is still waiting to hear how its write went.
                    self.state = "online"
                    self._push_status(self._describe(online=True))
                else:
                    if self._ble is not None:
                        self._push_status(self._describe(online=True))
                        log.info(
                            "Wi-Fi is up (%s) and the setup window has elapsed — closing Bluetooth",
                            self._last_described.get("ssid"),
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
                if (now - offline_since >= self._grace_s and self._ble is None
                        and self._may_open_ble(now)):
                    self._open_ble_window()
                if self._ble is not None:
                    self.state = "provisioning"
            # Poll quickly only while something is expected to change: offline
            # (someone is probably standing over it), a join in flight, or an
            # outcome being delivered. A device that is online -- including one
            # holding its reconfigure window open -- has no reason to be asked
            # every second: nmcli monitor wakes this loop on any real change, and
            # each tick costs several nmcli processes on a Pi running inference.
            if not online or joining or lingering:
                interval = min(self._poll_s, self._UNSETTLED_POLL_CEILING_S)
            else:
                interval = self._poll_s
                if self._ble is not None and window_left is not None and window_left > 0:
                    interval = min(interval, max(window_left, 0.05))  # close on time
            if self._ble is None and self._ble_failed_at is not None:
                interval = min(interval, self._BLE_RETRY_S)
            # The interval is now a backstop, not the mechanism: a state change
            # wakes this immediately, and the timeout only covers what nmcli
            # monitor might not report.
            changed.clear()
            changed.wait(timeout=interval)

    def _join_in_flight(self) -> bool:
        thread = self._connect_thread
        return thread is not None and thread.is_alive()

    def _describe(self, online: Optional[bool] = None) -> dict:
        """The device's own account of where it is in setup.

        Sent over Bluetooth so the app can show real steps rather than a
        spinner. Every field here is observed, never assumed.
        """
        if online is None:
            online = self._wifi.is_online()
        described = {
            "state": "online" if online else self.state,
            "ssid": self._wifi.current_ssid() if online else None,
            "ip": self._wifi.ip_address() if online else None,
            "device_id": self._device_id(),
            "reconfigurable": self.state == "reconfigurable",
            # Never merged into `state`: "which network am I on" and "did the
            # thing you just asked for work" are different questions, and the
            # answer to the first must not be mistaken for the second.
            "provision": self._provision,
        }
        self._last_described = described
        return described

    def _describe_cached(self) -> dict:
        """_describe() from the last observed values -- no nmcli, safe on the GLib loop."""
        described = dict(self._last_described) if self._last_described else {
            "state": self.state, "ssid": None, "ip": None,
            "device_id": self._device_id(),
            "reconfigurable": self.state == "reconfigurable",
        }
        described["provision"] = self._provision
        return described

    def status(self) -> tuple[bool, str]:
        """(healthy, detail) for the health monitor (non-critical component).

        `self.state` is a string the run loop last wrote, so on its own it describes
        the thread's last known intention, not whether the thread still exists. A
        crashed or exited thread left its final state behind and this reported it
        forever as healthy -- exactly the "guessing in its own favour" that
        core/device_health.py's own docstring says a monitoring product must not do.
        Liveness is therefore checked before the state is trusted.
        """
        crashed = getattr(self, "_crashed", None)
        if crashed:
            return False, crashed
        if self._method in self._DORMANT_METHODS:
            # run() returns at once by design: provisioning is off, or handled by
            # the legacy hotspot oneshot outside this process. Not a failure --
            # reporting it as one kept /api/health "degraded" for ever.
            if self._method == "off":
                return True, "Wi-Fi provisioning disabled (network.provisioning.method: off)"
            return True, "provisioning handled by the hotspot oneshot (method: hotspot)"
        # `unavailable` is a legitimate terminal state (dev Mac): run() returns
        # immediately by design, so a dead thread there is expected.
        if self.state != "unavailable" and self._thread is not None and not self._thread.is_alive():
            return False, f"provisioning thread is not running (last state: {self.state})"
        if (self._ble is None and self._ble_failure
                and self.state in ("waiting", "provisioning", "reconfigurable")):
            # Observed, not assumed: the window is wanted and is NOT open.
            return False, f"Bluetooth setup could not start ({self._ble_failure}); retrying"
        if self.state == "provisioning":
            return True, "offline — Bluetooth setup open so the app can hand it a network"
        if self.state == "reconfigurable":
            return True, "online; Bluetooth still open so the device can be moved to another network"
        if self.state == "unavailable":
            return False, "BLE provisioning requested but BlueZ bindings are missing"
        return True, f"state: {self.state}"

    def _device_id(self) -> str:
        return self._identity[0] if self._identity else device_id()

    # ------------------------------------------------------------------ BLE window
    def _may_open_ble(self, now: float) -> bool:
        if self._ble_unavailable:
            return False
        if self._auth_mode == "refused":
            if self._ble_failure != self._auth_error:
                log.error("Bluetooth setup stays closed: %s", self._auth_error)
                self._ble_failure = self._auth_error
            return False
        return self._ble_failed_at is None or now - self._ble_failed_at >= self._BLE_RETRY_S

    def _reap_dead_ble(self, now: float) -> None:
        """Notice a GATT server whose thread has died, instead of reporting it open.

        start() returning is not the server serving: bluetoothd can restart under
        it, or registration can fail after the fact. Left unchecked the service
        kept a dead server, said "Bluetooth setup open", and never tried again.
        """
        ble = self._ble
        if ble is None:
            return
        alive = getattr(ble, "is_alive", None)
        if alive is None or alive():
            return
        reason = getattr(ble, "failure", None) or "the Bluetooth server stopped unexpectedly"
        log.error("BLE provisioning is not serving (%s) — retrying in %.0fs", reason, self._BLE_RETRY_S)
        self._teardown_ble()
        self._ble_failed_at = now
        self._ble_failure = reason

    def _authenticator(self) -> Optional[PopAuthenticator]:
        """The proof-of-possession state machine, built once (PBKDF2 runs here, not per phone)."""
        if self._pop is None:
            code = load_setup_code(self._setup_code_path)  # SetupCodeError: caller fails closed
            self._pop = PopAuthenticator(device_id=self._device_id(), setup_code=code,
                                         limits=self._pop_limits or Limits())
        return self._pop

    def _open_ble_window(self) -> None:
        import time

        from services.ble_gatt import BLEProvisioningServer, BLEStartError, BLEUnavailable

        name = self._identity[1] if self._identity else _ble_name()
        ble = None
        extra = {}
        if self._auth_mode == "pop":
            try:
                pop = self._authenticator()
            except (SetupCodeError, ImportError) as exc:
                # Retried like any start failure: a code written later is picked up.
                self._ble_failed_at = time.monotonic()
                self._ble_failure = f"setup code unusable: {exc}"
                log.error("Bluetooth setup stays closed (auth: pop): %s", exc)
                return
            pop.stop()
            pop.start()
            extra["on_auth"] = self._on_auth
        try:
            ble = (self._ble_server_factory or BLEProvisioningServer)(
                device_name=name,
                device_info={
                    # The stable identity, so the app can recognise this device
                    # as the same one it already knows however it was found.
                    "id": self._device_id(),
                    "name": name,
                    "model": "SensorDyme",
                    "version": self._version,
                    "hostname": socket.gethostname(),
                    # How this device takes credentials, so the app knows to ask
                    # for the setup code ("pop") instead of writing them in clear.
                    "auth": self._auth_mode,
                },
                on_credentials=self._on_credentials,
                scan_networks=lambda: self._wifi.scan(),  # noqa: PLW0108 - late-binds self._wifi
                on_command=self._on_command,
                **extra,
            )
            self._ble = ble
            self._last_pushed = None
            ble.start()
            self._ble_failed_at = None
            self._ble_failure = None
            self.state = "provisioning"
            log.info("no Wi-Fi — Bluetooth setup open as '%s'; pair from the SensorDyme app", name)
        except BLEUnavailable as exc:
            self._ble = None
            self._ble_unavailable = True
            self.state = "unavailable"
            log.error("%s", exc)
        except BLEStartError as exc:
            # Transient until proven otherwise: retried after _BLE_RETRY_S.
            self._ble = None
            self._ble_failed_at = time.monotonic()
            self._ble_failure = str(exc)
            log.error("Bluetooth setup could not start: %s — retrying in %.0fs", exc, self._BLE_RETRY_S)

    def _teardown_ble(self) -> None:
        ble, self._ble = self._ble, None
        self._last_pushed = None
        if self._pop is not None:
            self._pop.stop()
        if ble is not None:
            ble.stop()

    def _push_status(self, status: dict, force: bool = False) -> None:
        # One read of the attribute: the run loop can tear the window down (and
        # set it to None) between a check and a use on a worker thread.
        ble = self._ble
        if ble is None:
            return
        if "provision" in status and status["provision"] is not self._provision:
            # Described before a credential write replaced the outcome (the run
            # loop and the BLE thread race): never re-publish the previous
            # attempt's answer on top of this attempt's "connecting".
            status = dict(status, provision=self._provision)
        if not force and status == self._last_pushed:
            return  # nothing new to say; a re-send is a notification for nothing
        self._last_pushed = dict(status)
        ble.set_status(status)

    # ------------------------------------------------------------------ app callbacks
    def _on_auth(self, data: bytes) -> bytes:
        """A proof-of-possession write. Runs on the BLE event loop; never blocks."""
        pop = self._pop
        if pop is None:
            return b'{"op":"error","error":"unavailable"}'
        reply = pop.handle(data)
        if reply.message.get("error") == "wrong_code":
            log.warning("BLE: a phone failed to prove the setup code")  # never the code or the tag
        if reply.credentials is not None:
            if self._join_in_flight():
                pop.apply_result(False)
                return b'{"op":"error","error":"busy","message":"a join is already running"}'
            self._on_credentials(*reply.credentials)
        return reply.encode()

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
        doing something slow. The "connecting" push is built from cached values
        for the same reason: three nmcli calls here were three more stalls.
        """
        if self._join_in_flight():
            # A second write while the first is still going would race two nmcli
            # invocations against the same radio.
            log.info("BLE: ignoring credentials for '%s' — a join is already running", ssid)
            return

        self._attempt += 1
        attempt = self._attempt
        self._outcome_at = None
        # Set (and pushed) before this handler returns, i.e. before the write
        # response: by the time the app's write completes, "connecting" with this
        # attempt number is what the status characteristic says.
        self._provision = {"state": "connecting", "ssid": ssid, "attempt": attempt}
        ble = self._ble
        if ble is not None:
            # No implicit Wi-Fi scans while the radio is associating.
            ble.set_scans_paused(True)
        self._push_status(self._describe_cached(), force=True)
        self._connect_thread = threading.Thread(
            target=self._join_network, args=(ssid, psk, attempt),
            daemon=True, name="wifi-join",
        )
        self._connect_thread.start()

    def _join_network(self, ssid: str, psk: str, attempt: int = 0) -> None:
        """Join `ssid` off the event loop, reporting the outcome over BLE."""
        import time

        try:
            # provision(), not connect(): the device may already be on a working
            # network that someone is standing a long way from. The new profile
            # is saved and validated before that connection is given up, and
            # restored if the new network refuses us.
            ok, message = self._wifi.provision(ssid, psk)
        except Exception as exc:  # noqa: BLE001 - the window must survive a failed join
            log.error("wifi join failed: %s", exc)
            ok, message = False, str(exc)

        if ok:
            try:
                ip = self._wifi.ip_address()
            except Exception:  # noqa: BLE001
                ip = None
            outcome = {"state": "ok", "ssid": ssid, "ip": ip, "message": message, "attempt": attempt}
        else:
            outcome = {"state": "error", "ssid": ssid, "error": message, "attempt": attempt}
        # Outcome first, linger clock second, "provisioned" last: the run loop
        # must never see a finished setup without also seeing the reason to keep
        # Bluetooth up long enough to report it.
        self._provision = outcome
        self._outcome_at = time.monotonic()
        if ok:
            self._provisioned_this_session = True
        if self._pop is not None:
            self._pop.apply_result(ok)
        ble = self._ble
        if ble is not None:
            ble.set_scans_paused(False)
        try:
            described = self._describe()
        except Exception:  # noqa: BLE001 - the outcome matters more than the rest
            described = self._describe_cached()
        self._push_status(described, force=True)
        self._wake_loop()

    def _wake_loop(self) -> None:
        wake = getattr(self, "_wake", None)
        if wake is not None:
            wake.set()

    def _on_command(self, command: dict) -> None:
        """Handle an app command. Runs on the BLE event loop — never block here.

        This previously shelled out to `nmcli --rescan yes` inline, which stalls
        every BLE operation for as long as the radio takes. Rescanning is now
        the BLE server's own non-blocking refresh; the app learns the outcome
        from the status push and re-reads the list.
        """
        if not isinstance(command, dict):
            return
        action = command.get("action")
        if action == "rescan":
            ble = self._ble
            if ble is not None:
                # Explicitly asked for: bypasses the implicit-refresh rate limit
                # (but still not while a join is using the radio).
                ble.refresh_networks(force=True)
        elif action == "identify":
            log.info("BLE identify requested")  # future: blink the status LED via the HAL
