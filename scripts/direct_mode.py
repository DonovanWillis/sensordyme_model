#!/usr/bin/env python3
"""Operator tool for direct mode, the device's own Wi-Fi access point (docs/DIRECT_MODE.md).

    scripts/direct_mode.py status
    scripts/direct_mode.py enter [--reason TEXT] [--dry-run]
    scripts/direct_mode.py leave [--reason TEXT] [--dry-run]
    sudo scripts/direct_mode.py install [--dry-run]      # the access-point firewall (root)
    scripts/direct_mode.py ruleset                       # print the firewall ruleset

`status`, `enter` and `leave` drive the same DirectModeManager the service
runs. Run them ON the device from the checkout, AS THE SERVICE USER
(`sudo -u <service user> .venv/bin/python scripts/direct_mode.py enter`): the
passphrase file must be owned by the account that reads it, and that account is
the one polkit lets change networks. They need `network.direct_mode.enabled:
true` in the config; with it false they refuse and say so.

`--dry-run` prints the nmcli commands (passphrase redacted) and the nft command
the dispatcher runs, without running any of them. The one thing it does run is
the read-only `nmcli -t -f NAME,TYPE,DEVICE connection show --active`, so the
rollback line can name the real previous network; without NetworkManager (a
development Mac) it runs nothing at all.

`install` (root) renders the ruleset for the configured interface and
dashboard port, checks it with `nft -c -f` when nft is present (nothing is
installed if nft rejects it), and installs it as /etc/sensordyme/direct_mode.nft,
the dispatcher script deploy/networkmanager/90-sensordyme-direct (root, 0755)
with its pre-up link, and the polkit grant
deploy/polkit/11-sensordyme-direct-mode.rules (wifi.share.protected only; never
wifi.share.open). Until it has run, `enter` refuses: the access point never
runs without its firewall.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

import yaml

from core import ap_firewall
from core.direct_mode import DirectModeManager
from core.network import WifiManager

REPO = Path(__file__).resolve().parent.parent
DISPATCHER_SOURCE = REPO / "deploy" / "networkmanager" / "90-sensordyme-direct"
#: NetworkManager asks polkit for wifi.share.protected before it raises a shared
#: Wi-Fi AP; the base rule (10-sensordyme-network.rules) does not grant it.
POLKIT_SOURCE = REPO / "deploy" / "polkit" / "11-sensordyme-direct-mode.rules"
POLKIT_TARGET = "/etc/polkit-1/rules.d/11-sensordyme-direct-mode.rules"


def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config if isinstance(config, dict) else {}


def _direct_config(config: dict) -> dict:
    network = config.get("network") or {}
    direct = network.get("direct_mode") if isinstance(network, dict) else None
    return direct if isinstance(direct, dict) else {}


def _dashboard_port(config: dict) -> int:
    dashboard = config.get("dashboard") or {}
    port = dashboard.get("port", 8899) if isinstance(dashboard, dict) else 8899
    return port if isinstance(port, int) and not isinstance(port, bool) and port > 0 else 8899


def _under(root: str, path: str) -> Path:
    return Path(root) / path.lstrip("/") if root else Path(path)


def install(config: dict, *, dry_run: bool, root: str = "") -> int:
    """Install the ruleset, the dispatcher script and the polkit grant. Root only, unless --root."""
    interface = str(_direct_config(config).get("interface", "wlan0"))
    ruleset = ap_firewall.render_ruleset(interface, _dashboard_port(config))
    ruleset_path = _under(root, ap_firewall.RULESET_PATH)
    dispatcher_path = _under(root, ap_firewall.DISPATCHER_PATH)
    link_path = _under(root, ap_firewall.PRE_UP_LINK)
    polkit_path = _under(root, POLKIT_TARGET)
    nft = shutil.which("nft")
    steps = [
        f"nft -c -f <rendered ruleset for {interface}>" + ("" if nft else "    # skipped: nft not installed here"),
        f"install -D -m 0644 -o root -g root <rendered ruleset> {ruleset_path}",
        f"install -D -m 0755 -o root -g root {DISPATCHER_SOURCE} {dispatcher_path}",
        f"ln -sfn ../{dispatcher_path.name} {link_path}",
        f"install -D -m 0644 -o root -g root {POLKIT_SOURCE} {polkit_path}",
    ]
    if dry_run:
        print("\n".join(steps))
        return 0
    if not root and hasattr(os, "geteuid") and os.geteuid() != 0:
        print("refused: install writes /etc and must run as root (sudo)", file=sys.stderr)
        return 1
    as_root = hasattr(os, "geteuid") and os.geteuid() == 0
    files = ((ruleset_path, ruleset, 0o644),
             (dispatcher_path, DISPATCHER_SOURCE.read_text(encoding="utf-8"), 0o755),
             (polkit_path, POLKIT_SOURCE.read_text(encoding="utf-8"), 0o644))
    for path, data, mode in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(data, encoding="utf-8")
        os.chmod(tmp, mode)
        if as_root:
            os.chown(tmp, 0, 0)
        if path == ruleset_path and nft and not root:
            # Checked before anything is replaced: a ruleset nft rejects is never installed.
            check = subprocess.run([nft, "-c", "-f", str(tmp)],  # fixed argv; -c only checks
                                   capture_output=True, text=True, check=False)
            if check.returncode != 0:
                tmp.unlink()
                print(f"refused: nft rejected the ruleset; nothing installed: {check.stderr.strip()}",
                      file=sys.stderr)
                return 1
        os.replace(tmp, path)
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(f"../{dispatcher_path.name}")
    print(f"installed {ruleset_path}, {dispatcher_path} (+ {link_path}) and {polkit_path}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO / "config.yaml"))
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    for name in ("enter", "leave"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--reason", default="operator CLI")
        cmd.add_argument("--dry-run", action="store_true")
    ins = sub.add_parser("install")
    ins.add_argument("--dry-run", action="store_true")
    ins.add_argument("--root", default="", help=argparse.SUPPRESS)  # tests: install under a scratch root
    sub.add_parser("ruleset")
    args = parser.parse_args(argv)

    config = _load_config(args.config)
    if args.cmd == "ruleset":
        print(ap_firewall.render_ruleset(str(_direct_config(config).get("interface", "wlan0")),
                                         _dashboard_port(config)), end="")
        return 0
    if args.cmd == "install":
        return install(config, dry_run=args.dry_run, root=args.root)

    # Relative paths (state_path) resolve the way they do for the service, whose
    # WorkingDirectory is the checkout.
    os.chdir(REPO)
    manager = DirectModeManager(WifiManager(), _direct_config(config))
    if args.cmd == "status":
        print(json.dumps(manager.status(), indent=1, sort_keys=True))
        return 0
    if args.dry_run:
        problem = manager.unavailable_reason()
        if problem:
            print(f"# would refuse now: {problem}")
        plan = manager.plan_enter() if args.cmd == "enter" else manager.plan_leave()
        print("\n".join(plan))
        return 0
    ok, message = manager.enter(args.reason) if args.cmd == "enter" else manager.leave(args.reason)
    print(message if ok else f"refused: {message}", file=sys.stdout if ok else sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

