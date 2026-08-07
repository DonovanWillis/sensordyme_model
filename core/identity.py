"""The device's permanent name for itself.

Everything else about a SensorDyme changes. Its IP changes when it moves between
networks, its hostname can be edited, its Bluetooth advertisement name and its
mDNS service name are two different strings produced by two different daemons,
and the app can meet it over Bluetooth on Monday and Wi-Fi on Tuesday.

If any of those stand in for identity, one physical device becomes two entries
in the app the moment they disagree — which is exactly what happened when the
Bluetooth list deduplicated by advertised name and showed both "SensorDyme" and
"SensorDyme-TEST" for a single Pi sitting on the desk.

So identity is derived from the board itself and published on every channel:

    SD-A1B2C3D4

Stable across reboots, IP changes, hostname changes, re-provisioning onto a new
Wi-Fi network, and factory-resetting the app. Two devices never collide; one
device never splits.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

_ID_RE = re.compile(r"^SD-[0-9A-F]{8}$")


def _cpu_serial() -> Optional[str]:
    """The Pi's SoC serial. Burned into the silicon — the most stable thing here."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("serial"):
                serial = line.split(":", 1)[1].strip()
                # A board with no serial reports all zeroes rather than omitting
                # the field, which would otherwise become everyone's device ID.
                if serial and set(serial) != {"0"}:
                    return serial
    except OSError:
        pass
    return None


def _mac_address() -> Optional[str]:
    """A network interface's MAC. Fixed per board, though wlan0 may be absent."""
    for interface in ("wlan0", "eth0"):
        try:
            mac = Path(f"/sys/class/net/{interface}/address").read_text().strip()
        except OSError:
            continue
        if mac and mac != "00:00:00:00:00:00":
            return mac
    return None


def _machine_id() -> Optional[str]:
    """systemd's per-installation ID. Survives reboots but not a reflash."""
    for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            value = Path(path).read_text().strip()
        except OSError:
            continue
        if value:
            return value
    return None


def device_id() -> str:
    """This device's stable ID, as `SD-XXXXXXXX`.

    Hashed rather than used raw: a CPU serial is a real hardware identifier, and
    broadcasting it in cleartext over mDNS to every device on a coffee-shop
    network hands out a permanent tracking token. The hash is stable and
    comparable without being reversible to the serial.
    """
    source = _cpu_serial() or _mac_address() or _machine_id()
    if source is None:
        # Nothing identifying available (a container, a dev Mac). A fixed
        # fallback keeps behaviour deterministic instead of inventing a new
        # identity on every call, which would look like a new device each boot.
        source = "sensordyme-unidentified"
    digest = hashlib.sha256(f"sensordyme:{source}".encode()).hexdigest()
    return f"SD-{digest[:8].upper()}"


def is_device_id(value: str) -> bool:
    """Whether `value` is a well-formed device ID."""
    return bool(_ID_RE.match(value or ""))


def display_name(device_id_value: Optional[str] = None) -> str:
    """The name a person sees: `SensorDyme-A1B2`.

    Short enough for a Bluetooth advertisement and recognisable next to the
    device, but never used as identity — two devices could be renamed to match,
    and one device can present more than one name.
    """
    identifier = device_id_value or device_id()
    return f"SensorDyme-{identifier.removeprefix('SD-')[:4]}"
