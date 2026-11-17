"""The firewall on the direct-mode access point (docs/DIRECT_MODE.md, G7).

While the device runs its own access point, a phone holding the label
passphrase is on the device's network. It may reach exactly:

* DHCP (UDP 67) and DNS (UDP/TCP 53), both served by NetworkManager's shared
  mode (dnsmasq), so the phone gets an address and iOS stops asking;
* mDNS (UDP 5353), so the app finds the device over Bonjour;
* the dashboard (TCP `dashboard.port`, 8899), which still needs the device token.

Everything else arriving on the AP interface is dropped, SSH included, and
nothing is forwarded to or from it: an AP client reaches the device and
nothing beyond it.

The service runs unprivileged and cannot load nftables rules, so the ruleset is
a file loaded by a root NetworkManager dispatcher script
(deploy/networkmanager/90-sensordyme-direct) only while the direct-mode profile
is up, and removed when it goes down. The script leaves a marker that the
direct-mode manager requires before it reports the AP as up; without the
marker it rolls the AP back (core/direct_mode.py).

This module only renders text. It never runs nft.
"""

from __future__ import annotations

import re

#: The nftables table this ruleset owns. Nothing else in the device uses it.
TABLE = "sensordyme_direct"

#: Where `scripts/direct_mode.py install` puts the rendered ruleset (root, 0644).
RULESET_PATH = "/etc/sensordyme/direct_mode.nft"

#: The dispatcher script, and its pre-up link (NetworkManager waits for pre-up
#: scripts before it finishes activating the connection).
DISPATCHER_PATH = "/etc/NetworkManager/dispatcher.d/90-sensordyme-direct"
PRE_UP_LINK = "/etc/NetworkManager/dispatcher.d/pre-up.d/90-sensordyme-direct"

#: Written by the dispatcher after the ruleset loaded, removed when it is unloaded.
MARKER_PATH = "/run/sensordyme-direct/firewall-active"

_IFNAME = re.compile(r"^[A-Za-z0-9_.-]{1,15}$")


def render_ruleset(interface: str = "wlan0", dashboard_port: int = 8899) -> str:
    """The nftables ruleset for the AP interface, as `nft -f` reads it.

    Loaded atomically: the leading `table` + `delete table` pair replaces any
    earlier copy in the same transaction, so a reload never leaves a gap.
    The base chains run at priority -10 with policy accept for every other
    interface; a drop here is final whatever another table accepts
    (NetworkManager's shared mode adds its own NAT/forward table).
    """
    if not _IFNAME.match(interface or ""):
        raise ValueError(f"not an interface name: {interface!r}")
    if not isinstance(dashboard_port, int) or isinstance(dashboard_port, bool) or not 1 <= dashboard_port <= 65535:
        raise ValueError(f"not a TCP port: {dashboard_port!r}")
    return f"""#!/usr/sbin/nft -f
# SensorDyme direct mode: what a phone on the device's own access point may reach.
# Rendered by core/ap_firewall.py. Loaded by the NetworkManager dispatcher script
# while the "sensordyme-direct" profile is up; removed when it goes down.
# Allowed on {interface}: DHCP, DNS, mDNS and the dashboard (TCP {dashboard_port}).
# Everything else is dropped, SSH included. Nothing is forwarded to or from it.

table inet {TABLE}
delete table inet {TABLE}

table inet {TABLE} {{
	chain input {{
		type filter hook input priority -10; policy accept;
		iifname != "{interface}" accept
		ct state established,related accept
		ct state invalid drop
		udp sport 68 udp dport 67 accept comment "DHCP"
		udp dport 53 accept comment "DNS"
		tcp dport 53 accept comment "DNS"
		udp dport 5353 accept comment "mDNS"
		tcp dport {dashboard_port} accept comment "dashboard"
		counter drop comment "everything else, SSH included"
	}}

	chain forward {{
		type filter hook forward priority -10; policy accept;
		iifname "{interface}" counter drop comment "no routing from the access point"
		oifname "{interface}" counter drop comment "no routing to the access point"
	}}
}}
"""
