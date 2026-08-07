"""Deploying code must never overwrite what a device knows about itself.

config.yaml on a deployed unit holds its device_key, its backend address, and
whether its camera is upside down. Those are facts about that one device. A
deploy that copies the repo's config over them unpairs the device from the
backend and flips the picture, and neither failure points back at the deploy.
"""

from pathlib import Path

from scripts.migrate_config import has_key, migrate

DEVICE_CONFIG = """\
camera:
  rotate_180: true

network:
  provisioning:
    method: ble
    grace_s: 45

cloud:
  enabled: true
  base_url: http://Donovans-MacBook-Pro.local:8787
  device_key: dk_secret_value
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


def test_a_value_the_device_already_set_is_never_changed(tmp_path):
    """grace_s is 45 here and the repo default is 0. The device wins: this file
    is the record of decisions made about this unit."""
    path = write(tmp_path, DEVICE_CONFIG)
    migrate(path)
    assert "grace_s: 45" in path.read_text()


def test_device_identity_survives_migration(tmp_path):
    path = write(tmp_path, DEVICE_CONFIG)
    migrate(path)
    text = path.read_text()
    assert "dk_secret_value" in text
    assert "Donovans-MacBook-Pro.local" in text
    assert "rotate_180: true" in text


def test_a_missing_key_is_added(tmp_path):
    path = write(tmp_path, DEVICE_CONFIG)
    added = migrate(path)
    assert "network.provisioning.reconfigure_window_s" in added
    assert "reconfigure_window_s: 300" in path.read_text()


def test_the_new_key_lands_inside_its_own_section(tmp_path):
    """Inserted at the wrong indent it silently belongs to another section, and
    the setting simply does nothing."""
    path = write(tmp_path, DEVICE_CONFIG)
    migrate(path)
    lines = path.read_text().splitlines()
    index = next(i for i, l in enumerate(lines) if "reconfigure_window_s" in l and not l.strip().startswith("#"))
    assert lines[index].startswith("    ")
    preceding = [l for l in lines[:index] if l.strip() and not l.strip().startswith("#")]
    assert any(l.strip() == "provisioning:" for l in preceding)


def test_running_twice_changes_nothing_the_second_time(tmp_path):
    path = write(tmp_path, DEVICE_CONFIG)
    migrate(path)
    after_first = path.read_text()
    assert migrate(path) == []
    assert path.read_text() == after_first


def test_a_backup_is_written_when_something_changes(tmp_path):
    path = write(tmp_path, DEVICE_CONFIG)
    migrate(path)
    assert list(tmp_path.glob("config.yaml.bak-*"))


def test_no_backup_when_nothing_changes(tmp_path):
    path = write(tmp_path, DEVICE_CONFIG)
    migrate(path)
    for old in tmp_path.glob("config.yaml.bak-*"):
        old.unlink()
    migrate(path)
    assert not list(tmp_path.glob("config.yaml.bak-*"))


def test_has_key_does_not_match_a_key_from_another_section(tmp_path):
    """`cloud.enabled` existing must not be read as `network.enabled` existing."""
    text = "network:\n  provisioning:\n    method: ble\ncloud:\n  enabled: true\n"
    lines = text.splitlines(keepends=True)
    assert has_key(lines, "cloud.enabled")
    assert not has_key(lines, "network.enabled")


def test_a_new_key_is_not_filed_under_the_next_section_s_comment(tmp_path):
    """A section's last lines are usually the heading comment for what follows.
    Inserting after them still parses, so nothing errors -- the file just
    silently claims the new setting belongs to something else."""
    text = (
        "network:\n"
        "  provisioning:\n"
        "    method: ble\n"
        "    poll_s: 10\n"
        "\n"
        "# Health monitoring and automatic recovery: every component is checked\n"
        "# continuously and failures escalate to a restart.\n"
        "health:\n"
        "  enabled: true\n"
    )
    path = tmp_path / "config.yaml"
    path.write_text(text)
    migrate(path)
    lines = path.read_text().splitlines()
    key_line = next(i for i, l in enumerate(lines) if "reconfigure_window_s:" in l)
    health_comment = next(i for i, l in enumerate(lines) if "Health monitoring" in l)
    assert key_line < health_comment, "new key landed under the health section's comment"


def test_the_migrated_file_still_parses_as_yaml(tmp_path):
    yaml = __import__("yaml")
    path = tmp_path / "config.yaml"
    path.write_text(DEVICE_CONFIG)
    migrate(path)
    parsed = yaml.safe_load(path.read_text())
    assert parsed["network"]["provisioning"]["reconfigure_window_s"] == 300
    assert parsed["network"]["provisioning"]["grace_s"] == 45
    assert parsed["cloud"]["device_key"] == "dk_secret_value"
