"""Deploying code must never overwrite what a device knows about itself.

config.yaml on a deployed unit holds its device_key, its backend address, and
whether its camera is upside down. Those are facts about that one device. A
deploy that copies the repo's config over them unpairs the device from the
backend and flips the picture, and neither failure points back at the deploy.
"""

from pathlib import Path

from scripts.migrate_config import has_key, migrate

# A device already carrying an OLDER alerts.loud_types/soft_types list -- the
# exact shape a real deployed unit had: unusual_stillness/person_absent/
# flailing/person_lingering/vehicle_in_zone/animal_in_zone are all ABSENT,
# which is precisely why those preset alert types could never produce a
# phone notification (services/runtime.py's severity formula defaulted them
# to "soft", and cloud/notify.py's NotificationGate requires "loud").
DEVICE_CONFIG_OLD_ALERTS = """\
alerts:
  sinks: [console, sound, gpio_buzzer]
  cooldown_s: 15
  loud_types: [drowsiness, phone_use, eyes_down, attention_lost, high_risk,
               fall, zone_violation, safe_zone_exit]
  soft_types: [fatigue_warning, frequent_glancing, elevated_risk]
  gpio_buzzer_pin: 17

cloud:
  device_key: dk_secret_value
"""

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
    index = next(i for i, line in enumerate(lines) if "reconfigure_window_s" in line and not line.strip().startswith("#"))
    assert lines[index].startswith("    ")
    preceding = [line for line in lines[:index] if line.strip() and not line.strip().startswith("#")]
    assert any(line.strip() == "provisioning:" for line in preceding)


def test_a_one_level_key_gets_its_comment_at_the_same_indent_as_the_key(tmp_path):
    """cloud.ca_cert_path sits one level deep (indent 2), unlike the
    network.provisioning.* migrations (indent 4) this file's insertion logic
    was first written for. A hardcoded 4-space comment indent looked correct
    only because every existing migration happened to be nested two levels —
    the bug was invisible until a one-level key exposed it: the key landed at
    indent 2 with its own explanatory comment floating at indent 4 above it."""
    path = write(tmp_path, DEVICE_CONFIG)
    migrate(path)
    lines = path.read_text().splitlines()
    key_index = next(i for i, line in enumerate(lines) if line.strip().startswith("ca_cert_path:"))
    comment_index = key_index - 1
    assert lines[comment_index].strip().startswith("#")
    key_indent = len(lines[key_index]) - len(lines[key_index].lstrip(" "))
    comment_indent = len(lines[comment_index]) - len(lines[comment_index].lstrip(" "))
    assert comment_indent == key_indent == 2


def test_tls_keys_are_added_under_cloud_off_by_default(tmp_path):
    path = write(tmp_path, DEVICE_CONFIG)
    added = migrate(path)
    assert "cloud.ca_cert_path" in added
    assert "cloud.tls_insecure_skip_verify" in added
    text = path.read_text()
    assert 'ca_cert_path: ""' in text
    assert "tls_insecure_skip_verify: false" in text


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
    key_line = next(i for i, line in enumerate(lines) if "reconfigure_window_s:" in line)
    health_comment = next(i for i, line in enumerate(lines) if "Health monitoring" in line)
    assert key_line < health_comment, "new key landed under the health section's comment"


# ------------------------------------------------------------------ list migrations
# A device that already HAS a key (unlike the missing-key migrations above)
# but whose VALUE needs a new member -- a repo fix to config.yaml's own
# loud_types list is invisible to an already-deployed device, since it has
# that key already and the missing-key rule leaves it untouched.
def test_a_missing_member_is_added_to_an_existing_list(tmp_path):
    path = write(tmp_path, DEVICE_CONFIG_OLD_ALERTS)
    added = migrate(path)
    assert "alerts.loud_types+=unusual_stillness" in added
    assert "alerts.loud_types+=flailing" in added
    text = path.read_text()
    assert "unusual_stillness" in text
    assert "flailing" in text


def test_a_member_already_present_is_not_duplicated(tmp_path):
    path = write(tmp_path, DEVICE_CONFIG_OLD_ALERTS)
    migrate(path)
    text_after_first = path.read_text()
    assert text_after_first.count("zone_violation") == 1  # already present before migrating


def test_running_the_list_migration_twice_changes_nothing_the_second_time(tmp_path):
    path = write(tmp_path, DEVICE_CONFIG_OLD_ALERTS)
    migrate(path)
    after_first = path.read_text()
    assert migrate(path) == []
    assert path.read_text() == after_first


def test_existing_loud_types_and_other_sections_are_preserved(tmp_path):
    path = write(tmp_path, DEVICE_CONFIG_OLD_ALERTS)
    migrate(path)
    text = path.read_text()
    for existing in ("drowsiness", "phone_use", "eyes_down", "attention_lost", "high_risk",
                      "fall", "zone_violation", "safe_zone_exit"):
        assert existing in text
    assert "dk_secret_value" in text  # unrelated device-identity value untouched
    assert "cooldown_s: 15" in text


def test_a_single_line_flow_sequence_is_handled_too(tmp_path):
    """The migration must work whether the device's list happens to be on one
    line or wrapped across many -- not assume the repo's own multi-line style."""
    text = "alerts:\n  loud_types: [drowsiness, phone_use]\n  soft_types: [fatigue_warning]\n"
    path = write(tmp_path, text)
    added = migrate(path)
    assert "alerts.loud_types+=unusual_stillness" in added
    result = path.read_text()
    assert "drowsiness" in result and "phone_use" in result and "unusual_stillness" in result
    # Substrings were all this checked, and they were all there -- above the
    # key, outside the list, in a file that no longer parsed. A device whose
    # config does not parse does not boot.
    parsed = __import__("yaml").safe_load(result)
    assert parsed["alerts"]["loud_types"][:2] == ["drowsiness", "phone_use"]
    assert "unusual_stillness" in parsed["alerts"]["loud_types"]
    assert parsed["alerts"]["soft_types"] == ["fatigue_warning"]


def test_list_migration_result_still_parses_as_yaml(tmp_path):
    yaml = __import__("yaml")
    path = write(tmp_path, DEVICE_CONFIG_OLD_ALERTS)
    migrate(path)
    parsed = yaml.safe_load(path.read_text())
    loud = parsed["alerts"]["loud_types"]
    for expected in ("unusual_stillness", "person_absent", "flailing",
                      "person_lingering", "vehicle_in_zone", "animal_in_zone",
                      "drowsiness", "zone_violation"):  # new + pre-existing, all present
        assert expected in loud
    assert parsed["cloud"]["device_key"] == "dk_secret_value"


def test_a_key_migration_and_a_list_migration_both_trigger_one_backup(tmp_path):
    path = write(tmp_path, DEVICE_CONFIG_OLD_ALERTS)
    added = migrate(path)
    assert any("+=" in a for a in added)  # a list migration ran
    assert any("+=" not in a for a in added)  # a key migration ran too (e.g. cloud.ca_cert_path)
    assert list(tmp_path.glob("config.yaml.bak-*"))


def test_the_migrated_file_still_parses_as_yaml(tmp_path):
    yaml = __import__("yaml")
    path = tmp_path / "config.yaml"
    path.write_text(DEVICE_CONFIG)
    migrate(path)
    parsed = yaml.safe_load(path.read_text())
    assert parsed["network"]["provisioning"]["reconfigure_window_s"] == 300
    assert parsed["network"]["provisioning"]["grace_s"] == 45
    assert parsed["cloud"]["device_key"] == "dk_secret_value"
    assert parsed["cloud"]["ca_cert_path"] == ""
    assert parsed["cloud"]["tls_insecure_skip_verify"] is False
