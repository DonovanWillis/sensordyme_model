"""Advertise this backend on the local network, so the app never needs an address.

The app used to require someone to type the server's address into Settings. That
is a setting only the person who set the server up can fill in, and it goes stale
on its own: DHCP moves the machine, and the address that was written down keeps
pointing at nothing. A new phone starts from zero and cannot sign in at all,
because sign-in needs an address before it needs a password.

So the server announces itself instead. It publishes `_sensordyme-api._tcp`
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
import secrets
import socket
import threading
from pathlib import Path
from typing import Optional

# "-api", not the more obvious "-cloud": RFC 6763 caps a service name at 15 bytes
# and `sensordyme-cloud` is 16, which zeroconf rejects outright. Keep any future
# rename under the limit, and keep it in step with NSBonjourServices in
# ios/Info.plist and ios/project.yml — a service the app has not declared is a
# service iOS will not let it see, and the failure is silent on both sides.
SERVICE_TYPE = "_sensordyme-api._tcp.local."
DEFAULT_PORT = 8787

#: The `.local` name this backend publishes for itself, and the whole point of it
#: is that the backend OWNS it.
#:
#: The record used to carry the machine's own hostname, which seems like the
#: stable thing and is not. macOS renames a Mac the moment it sees a name
#: collision — this one went from `Donovans-MacBook-Pro-2` to `-4` — and every
#: client holding the old name is then pointed at a host that no longer exists.
#: It cost ten days of a device dutifully dialling `…-Pro-2.local` and being
#: reported as offline, while the backend was up the whole time and the config
#: had followed the documented advice to "use a name, not an IP". The name was
#: the problem: a name assigned by the OS is not ours to depend on.
#:
#: A name we publish ourselves changes only when this file does. mDNS resolves it
#: to whatever address the machine currently has, so the identity is stable and
#: the endpoint is disposable — which is the property every client needs and none
#: of them could get from `LocalHostName`.
#:
#: If two machines on one network both run this backend, zeroconf's conflict
#: detection appends a suffix to one of them; the `id` below is what tells them
#: apart for a client that cares.
STABLE_HOSTNAME = "sensordyme-backend"

#: Where this backend's stable identity is kept. Same approach as the JWT secret:
#: generated once, persisted, never shipped.
BACKEND_ID_FILE = os.environ.get(
    "SENSORDYME_BACKEND_ID_FILE", str(Path(__file__).resolve().parent / ".backend_id")
)


def backend_id() -> str:
    """A stable identifier for THIS backend, advertised in the TXT record.

    Two SensorDyme backends on one network are indistinguishable without it: the
    app would take whichever answered its health check first, which on a shared
    or campus network can be a colleague's Mac. The device side has had an
    identity for a while (`id=SD-…`, see scripts/publish_mdns.py) and the app
    already refuses to silently adopt a different device; this is the same
    guarantee for the backend.

    Not a secret and deliberately not derived from one — it identifies a backend,
    it does not authenticate it. Generated on first use and persisted so it
    survives restarts, because an identity that changes every boot is not one.
    """
    path = Path(BACKEND_ID_FILE)
    try:
        existing = path.read_text().strip()
        if existing:
            return existing
    except OSError:
        pass
    # An id that could not be persisted must still be ONE id for the life of the
    # process: /v1/health reports it too, and a health id that differs from the
    # TXT id advertised a moment earlier would make the app refuse this backend.
    if str(path) in _TEMPORARY_IDS:
        return _TEMPORARY_IDS[str(path)]
    minted = f"bk_{secrets.token_hex(6)}"
    try:
        path.write_text(minted + "\n")
    except OSError as exc:
        # Usable but not persisted: better to advertise an identity that changes
        # on restart than none at all, and the log says which happened.
        log.warning(
            "could not persist the backend id to %s (%s); using a temporary one "
            "that will change on restart", path, exc,
        )
        _TEMPORARY_IDS[str(path)] = minted
    return minted


#: Ids minted for a location that could not be written, by path. See backend_id.
_TEMPORARY_IDS: dict[str, str] = {}

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
        self.refresh_interval_s = refresh_interval_s
        #: Read once: an identity that changes while the process runs is not one.
        self.backend_id = backend_id()
        #: The service INSTANCE name — the client-facing key for this service.
        #:
        #: Derived from this backend's own id, not from the machine's hostname,
        #: for the same reason `server` is STABLE_HOSTNAME rather than the OS
        #: name. The hostname is not ours: this network (a university ResNet)
        #: hands out `dhcp-10-130-229-13.dyn.uncc.edu` via DHCP, so
        #: `socket.gethostname()` embeds the IP ADDRESS and the instance name
        #: changed with every lease. A browsing client saw its known server
        #: disappear and an unfamiliar one arrive on each move, which is the
        #: opposite of what an instance name is for.
        #:
        #: Short id suffix rather than the bare product name so two backends on
        #: one network stay distinguishable in the app's picker; the full id is
        #: still in the TXT record, which is what clients actually match on.
        self.name = name or f"SensorDyme Cloud ({self.backend_id.removeprefix('bk_')[:8]})"
        self._zeroconf = None
        self._info = None
        #: The address currently on the wire, so a refresh can tell whether
        #: anything actually changed without re-deriving it from ServiceInfo.
        #:
        #: Only the address: the `server` field is now a name this backend owns
        #: (STABLE_HOSTNAME), so a machine rename no longer changes the record and
        #: there is nothing to track about the hostname any more.
        self._address: Optional[str] = None
        #: The address the live Zeroconf instance was CONSTRUCTED against.
        #:
        #: python-zeroconf binds its multicast sockets to the interface addresses
        #: that exist when the instance is created, and does not rebind them
        #: afterwards. So an instance that outlives a DHCP move is listening and
        #: answering on an address that is gone. Measured on a campus network that
        #: moved this Mac 38 times across 12 subnets: the SERVICE record survived
        #: (macOS's own responder relays it, so browsing still worked), but the
        #: STABLE_HOSTNAME A record was answered on loopback only -- so a phone
        #: found the service, could not resolve the name in it, and reported
        #: "Couldn't find a SensorDyme on this network". A freshly built instance
        #: published the same name on en0 correctly at the same instant, which is
        #: what identified the instance's age as the variable.
        self._bound_address: Optional[str] = None
        #: True between a start() that found zeroconf installed and stop(). It,
        #: not `_zeroconf`, is what decides whether refresh() does anything: a
        #: Zeroconf that could not be built (no LAN at login, an interface
        #: flapping mid-move) is a reason to try again on the next tick, not to
        #: stop announcing for the rest of the process.
        self._running = False
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._watcher: Optional[threading.Thread] = None

    # ----------------------------------------------------------------- internals

    def _build_info(self, address: str):
        from zeroconf import ServiceInfo

        return ServiceInfo(
            SERVICE_TYPE,
            f"{self.name}.{SERVICE_TYPE}",
            addresses=[socket.inet_aton(address)],
            port=self.port,
            # `path` lets the API move without every installed app being told.
            # `id` is what distinguishes this backend from another one on the same
            # network — see `backend_id`.
            properties={"v": "1", "path": "/", "api": "v1", "id": self.backend_id},
            # A name this backend owns, NOT the machine's hostname. See
            # STABLE_HOSTNAME for the ten-day outage that distinction cost.
            server=f"{STABLE_HOSTNAME}.local.",
        )

    def _rebind_locked(self, address: str) -> bool:
        """Rebuild the Zeroconf instance so its sockets bind to `address`.

        The only way to rebind python-zeroconf is to construct a new one: there
        is no public API to re-bind an existing instance, and `update_service`
        on a stale one silently keeps using the dead sockets.
        """
        from zeroconf import Zeroconf

        previous = self._zeroconf
        self._info = None
        self._address = None
        if previous is not None:
            # Best effort: the old sockets are bound to an address that no longer
            # exists, so closing them can fail, and that is not a reason to stop.
            with contextlib.suppress(Exception):
                previous.close()
        try:
            self._zeroconf = Zeroconf()
        except Exception as exc:  # noqa: BLE001 - advertising is never worth a crash
            log.warning(
                "could not re-open mDNS for %s (%s); this server is not "
                "announcing itself until the next refresh",
                address, repr(exc),
            )
            self._zeroconf = None
            self._bound_address = None
            return False
        self._bound_address = address
        return True

    def _register_locked(self, address: str) -> bool:
        """Put the record on the wire. Caller holds the lock."""
        try:
            info = self._build_info(address)
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
        log.info(
            "announcing this server as '%s' (id %s) at http://%s:%d — reachable as "
            "http://%s.local:%d, a name that survives this machine being renamed "
            "or moved to another network",
            self.name, self.backend_id, address, self.port, STABLE_HOSTNAME, self.port,
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

    # -------------------------------------------------------------------- public

    def refresh(self) -> bool:
        """Re-check this machine's identity and correct the record if it moved.

        Returns True when something was changed, so a caller (and a test) can
        tell a no-op tick from a real correction. Never raises: a refresh that
        fails leaves the previous record in place and says so in the log.
        """
        address = lan_ip()
        with self._lock:
            if not self._running:
                return False

            if self._zeroconf is None:
                # Nothing on the wire because the instance could not be built
                # (or the network was absent) last time. Try again now.
                if address is None or not self._rebind_locked(address):
                    return False
                if not self._register_locked(address):
                    return False
                log.info("mDNS is available again; announcing at http://%s:%d", address, self.port)
                return True

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
                # The usual shape of a DHCP move is interface-down then
                # interface-up on a NEW address, so this path -- not the
                # address-changed one below -- is where most moves land. It has to
                # rebind too, or the re-announcement goes out on dead sockets.
                if address != self._bound_address and not self._rebind_locked(address):
                    return False
                return self._register_locked(address)

            if address == self._address:
                return False

            previous_address = self._address
            # A new address means the sockets under this instance are bound to an
            # address that is gone, so the instance is rebuilt rather than
            # updated in place.
            #
            # `update_service` was what this did, and it is where the bug lived:
            # it updates the record the instance holds and re-announces it over
            # those dead sockets. Browsing kept working because macOS's own
            # responder relays the service, which is exactly why this was so hard
            # to see -- the service was found and only the hostname in it could
            # not be resolved.
            #
            # The cost is that clients see the service leave and return instead of
            # correcting itself. That is the right trade: a brief re-announcement
            # is recoverable, and an unresolvable hostname is not.
            if not self._rebind_locked(address):
                log.warning(
                    "this machine moved to %s but mDNS could not be re-opened; it "
                    "is no longer announcing (was %s)",
                    address, previous_address,
                )
                return False
            if not self._register_locked(address):
                log.warning(
                    "this machine moved to %s but the announcement could not be "
                    "re-made; it is no longer announcing (was %s)",
                    address, previous_address,
                )
                return False
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

        # From here on a failure is retried by the watcher rather than final: a
        # LaunchAgent starts this at login, routinely before Wi-Fi has joined.
        self._stopping.clear()
        self._running = True
        self._start_watcher()

        address = lan_ip()
        if address is None:
            log.warning(
                "no LAN address yet, so this server is not announcing itself; it "
                "will as soon as this machine joins a network"
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
                    "will retry every %.0fs",
                    repr(exc), self.refresh_interval_s,
                )
                self._zeroconf = None
                return False
            self._bound_address = address
            return self._register_locked(address)

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
        self._running = False
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


def verify_advertised_port(port: int, timeout: float = 0.5) -> bool:
    """Is anything actually listening on the port we just told the network about?

    uvicorn owns the real port and never tells the application which one it chose,
    so `advertised_port()` is a *claim* read from the environment. When the two
    disagree the failure is silent and total: the phone gets a definite address,
    connects to nothing, and reports the server as down — worse than never having
    advertised, which is what the module docstring warns about but nothing checked.

    The specific trap: `SENSORDYME_PORT` defaults to 8787, but uvicorn's own
    default is 8000. Launch `uvicorn cloud.app:app` with no `--port` and the
    server binds 8000 while this announces 8787, confidently.

    A local TCP connect, not an HTTP request: this runs during startup, so the
    app is not serving yet and a request would deadlock. All we need to know is
    whether the port is bound.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(timeout)
    try:
        return probe.connect_ex(("127.0.0.1", port)) == 0
    except OSError:
        return False
    finally:
        probe.close()


def enabled() -> bool:
    """Off switch for deployments where announcing is pointless or unwanted."""
    return os.environ.get("SENSORDYME_MDNS", "1").strip().lower() not in {"0", "false", "no", "off"}
