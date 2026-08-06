"""Minimal systemd sd_notify client (no dependency on the sdnotify package).

Used for two things when running under systemd with `Type=notify`:

  * `notify_ready()` — tells systemd startup finished (camera open, module loaded),
    so `systemctl start` blocks until the device is actually working.
  * `notify_watchdog()` — the periodic keep-alive for `WatchdogSec=`. The health
    monitor sends it only while the core pipeline is verifiably alive, so a hung
    inference loop (not just a dead process) gets the service restarted.

Outside systemd (dev Mac, `python main.py` by hand) NOTIFY_SOCKET is unset and
every call is a no-op.
"""

from __future__ import annotations

import os
import socket


def _send(message: str) -> bool:
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return False
    if addr.startswith("@"):  # abstract socket namespace
        addr = "\0" + addr[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.connect(addr)
            sock.sendall(message.encode())
        return True
    except OSError:
        return False


def enabled() -> bool:
    """True when running under a systemd unit with NotifyAccess configured."""
    return bool(os.environ.get("NOTIFY_SOCKET"))


def notify_ready() -> bool:
    """Signal startup complete (Type=notify)."""
    return _send("READY=1")


def notify_watchdog() -> bool:
    """Send one watchdog keep-alive (WatchdogSec)."""
    return _send("WATCHDOG=1")


def notify_stopping() -> bool:
    """Signal orderly shutdown has begun."""
    return _send("STOPPING=1")
