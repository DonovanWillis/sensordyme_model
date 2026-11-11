"""How the service runs its few privileged commands: through sudo, or directly.

The service needs privilege for exactly two things: the health monitor's
last-resort `systemctl reboot` (services/runtime.py) and BLE advertising through
the kernel mgmt API when bluetoothd cannot (`btmgmt`, services/ble_gatt.py).

    SENSORDYME_PRIVILEGE=sudo    (default; the stock sensordyme.service)
        `sudo -n <argv>`, allowed by the exact command lines in
        deploy/sudoers/sensordyme-reboot.
    SENSORDYME_PRIVILEGE=direct  (set only by deploy/systemd/sensordyme.service.hardened)
        `<argv>` as the service user. The hardened unit sets NoNewPrivileges, under
        which sudo can never gain root, so the grants move to the unit: `systemctl
        reboot` is authorised by polkit (deploy/polkit/20-sensordyme-reboot.rules)
        and btmgmt by CAP_NET_ADMIN, granted only by the optional drop-in
        deploy/systemd/sensordyme.service.d/btmgmt-advertising.conf.

Any other value is a configuration error and falls back to sudo, which under the
hardened unit fails closed (the caller logs it and stays up) rather than running
anything with more privilege than intended.
"""

from __future__ import annotations

import os

ENV = "SENSORDYME_PRIVILEGE"
SUDO_PREFIX = ("sudo", "-n")


def mode() -> str:
    return "direct" if os.environ.get(ENV, "sudo").strip().lower() == "direct" else "sudo"


def privileged(argv: list[str]) -> list[str]:
    """The argv to run for a privileged command under the configured mode."""
    return list(argv) if mode() == "direct" else [*SUDO_PREFIX, *argv]
