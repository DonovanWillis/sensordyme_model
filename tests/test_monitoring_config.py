"""What someone configured has to survive the device being switched off.

The rules a person set up lived only in memory and arrived only from the cloud.
So a reboot silently emptied them, and a device with no Wi-Fi could never be
configured at all — for a feature whose promise is "tell SensorDyme what to
watch for", that is the feature not existing exactly when it matters most.
"""

import json

from core.monitoring_config import MonitoringConfig, load_or_empty

RULES = [{"detection": "bottle", "predicate": "present", "min_count": 1,
          "description": "count the bottles"}]


def test_a_configuration_survives_a_restart(tmp_path):
    """The regression. A reboot used to lose it silently."""
    path = tmp_path / "monitoring.json"
    MonitoringConfig(path).save(RULES, instruction="Count how many bottles are visible.")

    restarted = load_or_empty(path)
    assert restarted.rules == RULES
    assert restarted.instruction == "Count how many bottles are visible."
    assert restarted.active is True


def test_the_sentence_someone_typed_is_kept_alongside_the_rules():
    """A rule list is what the machine runs; the sentence is what a person
    recognises when asked to confirm the device is doing the right thing."""
    config = MonitoringConfig("/dev/null")
    config.rules, config.instruction, config.overrides_mode = RULES, "Count the bottles.", True
    assert config.describe()["instruction"] == "Count the bottles."
    assert config.describe()["rule_count"] == 1


def test_no_stored_configuration_is_not_an_error(tmp_path):
    """A device that has never been configured runs its base mode."""
    config = load_or_empty(tmp_path / "absent.json")
    assert config.rules == []
    assert config.active is False


def test_a_corrupt_file_does_not_stop_the_device_monitoring(tmp_path):
    """Better to run the base mode than to refuse to start."""
    path = tmp_path / "monitoring.json"
    path.write_text("{ this is not json")
    config = load_or_empty(path)
    assert config.rules == []
    assert config.active is False


def test_clearing_returns_the_device_to_its_base_mode(tmp_path):
    path = tmp_path / "monitoring.json"
    config = MonitoringConfig(path)
    config.save(RULES, instruction="Count the bottles.")
    config.clear()

    assert load_or_empty(path).active is False


def test_a_configuration_with_rules_but_not_overriding_is_not_active(tmp_path):
    """Rules that run ALONGSIDE the base mode are a different thing from a
    custom mode that replaces it."""
    path = tmp_path / "monitoring.json"
    MonitoringConfig(path).save(RULES, overrides_mode=False)
    assert load_or_empty(path).active is False


def test_the_write_is_atomic_so_a_crash_cannot_half_write_it(tmp_path):
    """A half-written file makes the device forget its configuration on the
    next boot — the exact failure this module exists to prevent."""
    path = tmp_path / "monitoring.json"
    config = MonitoringConfig(path)
    config.save(RULES, instruction="Count the bottles.")
    # Valid JSON, and no temporary files left lying around.
    json.loads(path.read_text())
    assert list(tmp_path.glob("*.tmp")) == []


def test_saving_twice_replaces_rather_than_accumulates(tmp_path):
    path = tmp_path / "monitoring.json"
    config = MonitoringConfig(path)
    config.save(RULES, instruction="one")
    config.save([{"detection": "person", "predicate": "present"}], instruction="two")

    restored = load_or_empty(path)
    assert len(restored.rules) == 1
    assert restored.rules[0]["detection"] == "person"
    assert restored.instruction == "two"


def test_a_stale_instruction_never_outlives_the_rules_it_described(tmp_path):
    """Observed on hardware: the device stored the cloud's rules paired with a
    previous instruction, so it would have reported — confidently — that it was
    watching for something it was not. Silence is better than a wrong label."""
    path = tmp_path / "monitoring.json"
    config = MonitoringConfig(path)
    config.save(RULES, instruction="Count how many bottles are visible.")

    # New rules arrive with no instruction attached.
    config.save([{"detection": "person", "predicate": "present"}], instruction="")

    assert config.instruction == "", "the old instruction was carried onto new rules"
