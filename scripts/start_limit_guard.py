#!/usr/bin/env python3
"""Bounded reboot for a SensorDyme service that has exhausted its start limit.

sensordyme.service used `StartLimitAction=reboot`: five failed starts inside
StartLimitIntervalSec rebooted the Pi. systemd's start-limit counter does not
survive a reboot, so a failure that is deterministic at startup -- an import
error after a bad deploy, a corrupt presets.yaml, a broken venv -- rebooted the
device, failed five more times, rebooted again, for ever (audit 05, F12/F13 and
row 18: "StartLimitAction=reboot is not bounded across boots"). A reboot loop is
the worst state to leave a device in: no health, no dashboard, no BLE, and an
SSH window of seconds per cycle.

The unit now says `StartLimitAction=none` and `OnFailure=sensordyme-recover.service`,
which runs this script as root. It reboots ONLY when:

  * the unit failed because it hit its start limit (`Result=start-limit-hit`):
    an exit-78 config refusal, a manual stop or any other failure never reboots;
  * fewer than --max reboots of this kind happened within --window seconds,
    counted in a state file that survives reboots;
  * the new count was durably written first (a reboot that cannot be recorded
    could not be bounded, so it does not happen).

Past the budget the unit is left failed, the device stays up and reachable, and
the journal says why. After fixing the cause:
    sudo systemctl reset-failed sensordyme && sudo systemctl start sensordyme

Standard library only: it is installed root-owned at
/usr/local/lib/sensordyme/start_limit_guard.py (scripts/release/install_units.sh,
setup_pi.sh), never run from the service user's writable checkout.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Optional

DEFAULT_UNIT = "sensordyme.service"
DEFAULT_STATE = "/var/lib/sensordyme-recover/reboots.json"
DEFAULT_MAX_REBOOTS = 3
DEFAULT_WINDOW_S = 6 * 3600
TRIGGER_RESULT = "start-limit-hit"


def decide(result: str, history: list[float], now: float, *, max_reboots: int = DEFAULT_MAX_REBOOTS,
           window_s: float = DEFAULT_WINDOW_S) -> tuple[bool, list[float], str]:
    """(reboot?, history to persist, reason). Pure.

    A timestamp in the future (the clock went backwards: no RTC battery, no NTP
    yet) is kept and counted -- the conservative direction.
    """
    recent: list[float] = [t for t in history if isinstance(t, (int, float)) and (t > now or now - t < window_s)]
    if result != TRIGGER_RESULT:
        return False, recent, f"unit result is {result or 'unknown'!r}, not {TRIGGER_RESULT!r}: no reboot"
    if len(recent) >= max_reboots:
        return False, recent, (
            f"start limit hit, but {len(recent)} reboot(s) for this already happened within "
            f"{window_s:.0f}s (max {max_reboots}): leaving the service failed and the device up. "
            "Fix the cause, then: sudo systemctl reset-failed sensordyme && sudo systemctl start sensordyme"
        )
    return True, [*recent, now], f"start limit hit: rebooting ({len(recent) + 1} of {max_reboots} within {window_s:.0f}s)"


def load_history(path: Path) -> list[float]:
    """Reboot timestamps from `path`; [] when absent or unreadable (the write must
    then succeed before any reboot, so this permissive read cannot loop)."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    values = data.get("reboots") if isinstance(data, dict) else None
    return [float(v) for v in values if isinstance(v, (int, float))] if isinstance(values, list) else []


def save_history(path: Path, history: list[float]) -> bool:
    """Durably write the history (temp file + fsync + rename). False on any failure."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as fh:
            json.dump({"reboots": history}, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        return True
    except OSError as exc:
        print(f"start-limit guard: cannot record the reboot in {path}: {exc}")
        return False


def unit_result(unit: str, run: Callable = subprocess.run) -> str:
    try:
        out = run(["systemctl", "show", unit, "-p", "Result", "--value"],
                  capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"start-limit guard: cannot query {unit}: {exc}")
        return ""
    return (out.stdout or "").strip()


def main(argv: Optional[list[str]] = None, *, run: Callable = subprocess.run,
         clock: Callable[[], float] = time.time) -> int:
    parser = argparse.ArgumentParser(description="Bounded reboot after a start-limit failure")
    parser.add_argument("--unit", default=DEFAULT_UNIT)
    parser.add_argument("--state", default=DEFAULT_STATE)
    parser.add_argument("--max", type=int, default=DEFAULT_MAX_REBOOTS)
    parser.add_argument("--window", type=float, default=DEFAULT_WINDOW_S)
    parser.add_argument("--dry-run", action="store_true", help="decide and print; never write or reboot")
    args = parser.parse_args(argv)

    result = unit_result(args.unit, run)
    state = Path(args.state)
    reboot, history, reason = decide(result, load_history(state), clock(),
                                     max_reboots=args.max, window_s=args.window)
    print(f"start-limit guard: {args.unit}: {reason}")
    if not reboot or args.dry_run:
        return 0
    if not save_history(state, history):
        print("start-limit guard: reboot ABORTED: it could not be recorded, so it could not be bounded")
        return 1
    try:
        run(["systemctl", "reboot"], check=False, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"start-limit guard: reboot command failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
