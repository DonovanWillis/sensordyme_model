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

import logging
import os
import socket
from typing import Optional

# "-api", not the more obvious "-cloud": RFC 6763 caps a service name at 15 bytes
# and `sensordyme-cloud` is 16, which zeroconf rejects outright. Keep any future
# rename under the limit, and keep it in step with NSBonjourServices in
# ios/Info.plist and ios/project.yml — a service the app has not declared is a
# service iOS will not let it see, and the failure is silent on both sides.
SERVICE_TYPE = "_sensordyme-api._tcp.local."
DEFAULT_PORT = 8787

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
    """Publishes the service while the server runs, and withdraws it on shutdown.

    Withdrawing matters: an abandoned mDNS record outlives the process by minutes
    and points phones at a port nothing is listening on, which looks exactly like
    the bug this module exists to remove.
    """

    def __init__(self, port: int, name: Optional[str] = None) -> None:
        self.port = port
        self.name = name or f"SensorDyme Cloud on {socket.gethostname().split('.')[0]}"
        self._zeroconf = None
        self._info = None

    def start(self) -> bool:
        try:
            from zeroconf import ServiceInfo, Zeroconf
        except ImportError:
            log.warning(
                "zeroconf is not installed, so this server will not announce itself on "
                "the network; the app will need its address typed into Settings. "
                "Fix: pip install zeroconf"
            )
            return False

        address = lan_ip()
        if address is None:
            log.warning(
                "no LAN address found, so this server will not announce itself; "
                "the app will need its address typed into Settings"
            )
            return False

        host = socket.gethostname().split(".")[0]
        try:
            self._info = ServiceInfo(
                SERVICE_TYPE,
                f"{self.name}.{SERVICE_TYPE}",
                addresses=[socket.inet_aton(address)],
                port=self.port,
                # The app reads `path` to build its base URL, so the API can move
                # without every installed copy of the app needing to be told.
                properties={"v": "1", "path": "/", "api": "v1"},
                server=f"{host}.local.",
            )
            self._zeroconf = Zeroconf()
            self._zeroconf.register_service(self._info)
        except Exception as exc:  # noqa: BLE001 - advertising is never worth a crash
            # repr, not str: the exception that actually happens here first
            # (BadTypeInNameException for an over-long service name) can carry an
            # empty message, and "could not announce this server: " with nothing
            # after it is a log line that costs more time than it saves.
            log.warning(
                "could not announce this server on the network (%s); "
                "the app will need its address typed into Settings",
                repr(exc),
            )
            self.stop()
            return False

        log.info(
            "announcing this server as '%s' at http://%s:%d (and http://%s.local:%d) — "
            "the app can find it without being given an address",
            self.name, address, self.port, host, self.port,
        )
        return True

    def stop(self) -> None:
        try:
            if self._zeroconf is not None and self._info is not None:
                self._zeroconf.unregister_service(self._info)
        except Exception:  # noqa: BLE001 - shutdown must not raise
            pass
        finally:
            if self._zeroconf is not None:
                try:
                    self._zeroconf.close()
                except Exception:  # noqa: BLE001
                    pass
            self._zeroconf = None
            self._info = None


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
