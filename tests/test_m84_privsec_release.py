"""M8.4 safe updates: a rollback removes what the failed release added, whatever its mtime.

`make deploy` copies the release with `rsync -a`, which keeps each file's mtime
from the developer's machine. The M8 rollback removed an added file only when
its mtime was after the snapshot, so a module written before the deploy (any
file older than the deploy itself -- the normal case) survived the rollback;
modules/__init__.py imports every sibling module, so the "previous release"
then ran with the failed release's leftover in it. Reproduced:
~/sensordyme-audit/2026-11-10/m84-evidence/privsec/rsync_mtime_rollback_repro.txt.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from test_m8_lifecycle_release import (
    ACTIVE,
    CRASH_LOOP,
    STARTING,
    _device,
    _FakeSystemd,
    _guard,
    _tree,
)

from scripts.release import release_guard as rg


def test_a_file_the_failed_release_added_is_removed_even_with_an_old_mtime(tmp_path):
    root = _device(tmp_path)
    before = _tree(root)
    systemd = _FakeSystemd([[ACTIVE], CRASH_LOOP, [STARTING, ACTIVE]])
    guard = _guard(root, systemd)
    assert guard.snapshot() == 0
    # The deploy: a new module, copied with its source mtime (a day old), as rsync -a does.
    added = root / "core" / "new_module.py"
    added.write_text("import does_not_exist\n")
    day_ago = time.time() - 86400
    os.utime(added, (day_ago, day_ago))
    systemd.run(["sudo", "systemctl", "restart", "sensordyme"])  # the deploy's restart: it crash-loops
    assert guard.confirm_or_rollback(timeout_s=60, settle_s=5) == 1
    assert not added.exists(), "the failed release's module survived the rollback"
    assert _tree(root) == before


def test_a_file_that_was_already_there_is_never_removed(tmp_path):
    root = _device(tmp_path)
    systemd = _FakeSystemd([[ACTIVE], CRASH_LOOP, [STARTING, ACTIVE]])
    guard = _guard(root, systemd)
    assert guard.snapshot() == 0
    assert guard.rollback(reason="test") == 0
    assert (root / "main.py").read_text() == "VERSION = 'v1'\n"
    assert (root / "data" / "device_state.json").exists() and (root / "models" / "m.hef").exists()


def test_the_rollback_keeps_device_state_including_the_camera_privacy_setting(tmp_path):
    """data/ is never part of a release: a rollback cannot turn camera-only back on or off."""
    root = _device(tmp_path)
    state = root / "data" / "device_state.json"
    state.write_text('{"camera_privacy": {"schema": 1, "camera_only": false}}')
    guard = _guard(root, _FakeSystemd([[ACTIVE], CRASH_LOOP, [STARTING, ACTIVE]]))
    guard.snapshot()
    state.write_text('{"camera_privacy": {"schema": 1, "camera_only": true, "command_id": "owner"}}')
    guard.rollback(reason="test")
    assert "owner" in state.read_text(), "the rollback rewrote device state"
    assert Path(rg.__file__).exists()
