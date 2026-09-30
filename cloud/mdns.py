"""Advertise this backend on the local network, so the app never needs an address.

The app used to require someone to type the server's address into Settings. That
is a setting only the person who set the server up can fill in, and it goes stale
on its own: DHCP moves the machine, and the address that was written down keeps
pointing at nothing. A new phone starts from zero and cannot sign in at all,
because sign-in needs an address before it needs a password.

So the server announces itself instead. It publishes `_sensordyme-cloud._tcp`
over mDNS, the app browses for it, and the address becomes a detail neither of
them has to remember. Same mechanism the device already uses for its own feed
(`_sensordyme._tcp`, see scripts/publish_mdns.py) — this is the cloud half.

Scope, stated plainly: mDNS is link-local. This makes the address irrelevant on
the network the server is on, and does nothing off it. A backend that answers
from anywhere needs a real public address; see cloud/README.md.

Everything here is best-effort. A backend that refuses to start because it could
not advertise itself would be a worse backend, so every failure below degrades to
"you can still type the address in Settings" and says so in the log.
"""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import threading
from typing import Optional

# "-api", not the more obvious "-cloud": RFC 6763 caps a service name at 15 bytes
# and `sensordyme-cloud` is 16, which zeroconf rejects outright. Keep any future
# rename under the limit, and keep it in step with NSBonjourServices in
# ios/Info.plist and ios/project.yml — a service the app has not declared is a
# service iOS will not let it see, and the failure is silent on both sides.
SERVICE_TYPE = "_sensordyme-api._tcp.local."
DEFAULT_PORT = 8787

#: How often to re-check whether the address and hostname we announced are still
#: the ones this machine has.
#:
#: This is not a tuning knob, it is the fix for the bug that made this whole
#: module useless in practice. The address and hostname were read ONCE, at
#: startup, and the mDNS record kept repeating them for the life of the process.
#: A laptop is the one kind of host where that assumption is always wrong: it
#: joins a different Wi-Fi network, it tethers to a phone, its DHCP lease moves,
#: and macOS renames it (`Donovans-MacBook-Pro-2` becomes `-4`) the moment it
#: sees a name collision. Because a LaunchAgent keeps the backend up for weeks,
#: the process that outlives all of those changes is exactly the one still
#: advertising the address it had on the network it left — which looks, from the
#: phone, precisely like "the server is gone".
#:
#: 15s is chosen against what it competes with: a DHCP move is noticed within one
#: interval, and the work per interval is one UDP socket open with no packet sent
#: (see `lan_ip`) plus a string compare.
REFRESH_INTERVAL_S = 15.0

# uvicorn configures handlers for its own loggers and leaves the root logger bare,
# so a logger named for this module would have its INFO lines silently dropped —
# including the one line that confirms the announcement actually went out. Logging
# through uvicorn's own channel puts these messages in the server log people read.
log = logging.getLogger("uvicorn.error")


def lan_ip() -> Optional[str]:
    """This machine's address on the LAN.

    Resolving our own hostname is the obvious approach and the wrong one: on a
    Mac it commonly answers 127.0.0.1, and advertising loopback tells every phone
    on the network to connect to itself. Opening a UDP socket toward an off-link
    address makes the routing table choose the interface that actually carries
    LAN traffic, and we read the address off the socket. No packet is sent.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 9))  # TEST-NET-1: reserved, never routed
        address = probe.getsockname()[0]
    except OSError:
        return None
    finally:
        probe.close()
    return None if address.startswith("127.") else address


def machine_identity() -> tuple[Optional[str], str]:
    """The address and short hostname this machine has *right now*.

    Read together and compared as a pair, because either one changing on its own
    is enough to make an existing announcement wrong: a new DHCP lease changes
    only the address, and a macOS name-collision rename changes only the host.
    """
    return lan_ip(), socket.gethostname().split(".")[0]


class Advertiser:
    """Publishes the service while the server runs, keeps it true, and withdraws
    it on shutdown.

    Withdrawing matters: an abandoned mDNS record outlives the process by minutes
    and points phones at a port nothing is listening on, which looks exactly like
    the bug this module exists to remove.

    Keeping it *true* matters for the same reason and was the part missing. A
    record that still names the address this machine had an hour ago is not a
    lesser version of a correct announcement — it is worse than silence, because
    silence makes the app look elsewhere while a confident wrong answer makes it
    connect to nothing and report the server as down. So the address and hostname
    are re-checked on a timer (`REFRESH_INTERVAL_S`) and the record is corrected
    the moment they diverge.

    The instance name deliberately does NOT change when the hostname does. It is
    what a browsing client keys on; churning it makes one server look like two
    appearing and disappearing. The address and the `server` field — the parts
    that decide whether a connection succeeds — are what get corrected.
    """

    def __init__(
        self,
        port: int,
        name: Optional[str] = None,
        refresh_interval_s: float = REFRESH_INTERVAL_S,
    ) -> None:
        self.port = port
        self.name = name or f"SensorDyme Cloud on {socket.gethostname().split('.')[0]}"
        self.refresh_interval_s = refresh_interval_s
        self._zeroconf = None
        self._info = None
        #: The address/host currently on the wire, so a refresh can tell whether
        #: anything actually changed without re-deriving it from ServiceInfo.
        self._address: Optional[str] = None
        self._host: Optional[str] = None
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._watcher: Optional[threading.Thread] = None

    # ----------------------------------------------------------------- internals

    def _build_info(self, address: str, host: str):
        from zeroconf import ServiceInfo

        return ServiceInfo(
            SERVICE_TYPE,
            f"{self.name}.{SERVICE_TYPE}",
            addresses=[socket.inet_aton(address)],
            port=self.port,
            # The app reads `path` to build its base URL, so the API can move
            # without every installed copy of the app needing to be told.
            properties={"v": "1", "path": "/", "api": "v1"},
            server=f"{host}.local.",
        )

    def _register_locked(self, address: str, host: str) -> bool:
        """Put the record on the wire. Caller holds the lock."""
        try:
            info = self._build_info(address, host)
            self._zeroconf.register_service(info)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001 - advertising is never worth a crash
            log.warning(
                "could not announce this server on the network (%s); "
                "the app will need its address typed into Settings",
                repr(exc),
            )
            return False
        self._info = info
        self._address = address
        self._host = host
        log.info(
            "announcing this server as '%s' at http://%s:%d (and http://%s.local:%d) — "
            "the app can find it without being given an address",
            self.name, address, self.port, host, self.port,
        )
        return True

    def _withdraw_locked(self) -> None:
        """Take the record off the wire but keep the Zeroconf instance alive, so
        the announcement can come back when the network does. Caller holds the lock."""
        if self._zeroconf is not None and self._info is not None:
            # Suppressed rather than logged: this runs on shutdown and on a lost
            # network, where the reason unregistering failed is never actionable.
            with contextlib.suppress(Exception):
                self._zeroconf.unregister_service(self._info)
        self._info = None
        self._address = None
        self._host = None

    # -------------------------------------------------------------------- public

    def refresh(self) -> bool:
        """Re-check this machine's identity and correct the record if it moved.

        Returns True when something was changed, so a caller (and a test) can
        tell a no-op tick from a real correction. Never raises: a refresh that
        fails leaves the previous record in place and says so in the log.
        """
        address, host = machine_identity()
        with self._lock:
            if self._zeroconf is None:
                return False

            # No LAN at all. Withdraw rather than keep repeating the last address
            # we had: a phone that finds nothing goes on looking, while a phone
            # that finds a dead address stops and reports the server as down.
            if address is None:
                if self._info is None:
                    return False
                log.info(
                    "this machine has no LAN address right now; withdrawing the "
                    "announcement until it has one again"
                )
                self._withdraw_locked()
                return True

            if self._info is None:
                log.info("a LAN address is available again (%s); re-announcing", address)
                return self._register_locked(address, host)

            if address == self._address and host == self._host:
                return False

            previous_address, previous_host = self._address, self._host
            # A changed hostname means a changed `server` record, which
            # update_service does not revise — so that case is a clean
            # re-registration. An address-only change (the common one: a new DHCP
            # lease) is an update in place, which browsing clients see as the same
            # service correcting itself rather than one leaving and another
            # arriving.
            if host != self._host:
                log.info(
                    "this machine was renamed (%s -> %s); re-announcing so the "
                    "hostname in the record resolves again",
                    previous_host, host,
                )
                self._withdraw_locked()
                return self._register_locked(address, host)

            try:
                info = self._build_info(address, host)
                self._zeroconf.update_service(info)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "this machine moved to %s but the announcement could not be "
                    "updated (%s); it still says %s, which the app will read as "
                    "the server being down",
                    address, repr(exc), previous_address,
                )
                return False
            self._info = info
            self._address = address
            log.info(
                "this machine moved (%s -> %s); the announcement now points at "
                "http://%s:%d",
                previous_address, address, address, self.port,
            )
            return True

    def start(self) -> bool:
        try:
            from zeroconf import Zeroconf
        except ImportError:
            log.warning(
                "zeroconf is not installed, so this server will not announce itself on "
                "the network; the app will need its address typed into Settings. "
                "Fix: pip install zeroconf"
            )
            return False

        address, host = machine_identity()
        if address is None:
            log.warning(
                "no LAN address found, so this server will not announce itself; "
                "the app will need its address typed into Settings"
            )
            return False

        with self._lock:
            try:
                self._zeroconf = Zeroconf()
            except Exception as exc:  # noqa: BLE001
                # repr, not str: the exception that actually happens here first
                # (BadTypeInNameException for an over-long service name) can carry
                # an empty message, and "could not announce this server: " with
                # nothing after it is a log line that costs more time than it saves.
                log.warning(
                    "could not announce this server on the network (%s); "
                    "the app will need its address typed into Settings",
                    repr(exc),
                )
                self._zeroconf = None
                return False
            registered = self._register_locked(address, host)

        if not registered:
            self.stop()
            return False

        self._start_watcher()
        return True

    def _start_watcher(self) -> None:
        if self.refresh_interval_s <= 0 or self._watcher is not None:
            return
        # A daemon thread so a refresh in progress can never hold up interpreter
        # exit; `stop()` still joins it for an orderly shutdown when there is one.
        self._watcher = threading.Thread(
            target=self._watch, name="sensordyme-mdns-refresh", daemon=True
        )
        self._watcher.start()

    def _watch(self) -> None:
        # Event.wait rather than sleep, so stop() is acted on immediately instead
        # of after up to a full interval of shutdown delay.
        while not self._stopping.wait(self.refresh_interval_s):
            try:
                self.refresh()
            except Exception as exc:  # noqa: BLE001 - a refresh must never kill the thread
                log.warning("mDNS refresh failed (%s); will try again", repr(exc))

    def stop(self) -> None:
        self._stopping.set()
        watcher, self._watcher = self._watcher, None
        if watcher is not None and watcher is not threading.current_thread():
            watcher.join(timeout=2 * max(self.refresh_interval_s, 0.5))
        with self._lock:
            self._withdraw_locked()
            if self._zeroconf is not None:
                with contextlib.suppress(Exception):
                    self._zeroconf.close()
            self._zeroconf = None


def advertised_port() -> int:
    """The port to announce.

    uvicorn owns the real port and does not tell the application what it chose,
    so this is told separately via SENSORDYME_PORT. Announcing a port the server
    is not on is worse than not announcing at all — it sends the app somewhere
    definite and wrong — so if you run on a non-default port, set this too.
    """
    raw = os.environ.get("SENSORDYME_PORT", "")
    try:
        return int(raw) if raw else DEFAULT_PORT
    except ValueError:
        log.warning("SENSORDYME_PORT=%r is not a number; announcing %d", raw, DEFAULT_PORT)
        return DEFAULT_PORT


def enabled() -> bool:
    """Off switch for deployments where announcing is pointless or unwanted."""
    return os.environ.get("SENSORDYME_MDNS", "1").strip().lower() not in {"0", "false", "no", "off"}
