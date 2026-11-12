"""M8 security: device credential tooling (scripts/migrate_device_key.py).

A config.yaml that does not parse used to end the migration with PyYAML's
traceback, which quotes the offending line -- the device key, when that is the
line that is broken. The script now names the line and column only.
"""

from __future__ import annotations

import pytest

KEY = "dk_" + "5a" * 24


@pytest.mark.parametrize("broken", [
    f"cloud:\n  base_url: https://x\n  device_key: {KEY}: x\n",
    f"cloud:\n  device_key: \"{KEY}\n  poll_interval_s: 2\n",
    f"cloud:\n  device_key: {KEY}\n   poll_interval_s: [2\n",
])
def test_a_config_that_does_not_parse_is_refused_without_quoting_the_key(tmp_path, capsys, broken):
    from scripts.migrate_device_key import main

    config = tmp_path / "config.yaml"
    config.write_text(broken)
    assert main(["--config", str(config), "--store", str(tmp_path / "creds.json")]) == 1
    captured = capsys.readouterr()
    assert KEY not in captured.out + captured.err
    assert "is not valid YAML (line" in captured.err
    assert not (tmp_path / "creds.json").exists() and config.read_text() == broken
