"""A custom configuration replaces the base mode; it does not run beside it.

Someone who asks "count the bottles" should not keep receiving drowsiness and
phone-use alerts they never asked for — and should not pay for that detection
in CPU and heat on a device that is already at 58% of one core.

The distinction that matters: the base module still RUNS, because its per-frame
detections are what the rules are evaluated against. What is suppressed is its
own opinion about what deserves an alert.
"""

from core.monitoring_config import MonitoringConfig

RULES = [{"detection": "bottle", "predicate": "present", "min_count": 1}]


def test_a_custom_configuration_is_active_and_overrides(tmp_path):
    config = MonitoringConfig(tmp_path / "m.json")
    config.save(RULES, instruction="Count the bottles.", overrides_mode=True)
    assert config.active is True


def test_no_configuration_leaves_the_base_mode_in_charge(tmp_path):
    assert MonitoringConfig(tmp_path / "m.json").active is False


def test_rules_that_deliberately_run_alongside_do_not_override(tmp_path):
    """Adding a rule to an existing mode is a different intent from replacing
    the mode, and the config records which one was meant."""
    config = MonitoringConfig(tmp_path / "m.json")
    config.save(RULES, overrides_mode=False)
    assert config.rules and config.active is False


def test_clearing_hands_control_back_to_the_base_mode(tmp_path):
    config = MonitoringConfig(tmp_path / "m.json")
    config.save(RULES, instruction="Count the bottles.")
    assert config.active is True
    config.clear()
    assert config.active is False
    assert config.instruction == ""


def test_an_empty_rule_set_never_counts_as_an_active_override(tmp_path):
    """Otherwise clearing the rules would suppress the base mode's alerts and
    leave the device monitoring nothing at all — silently."""
    config = MonitoringConfig(tmp_path / "m.json")
    config.save([], instruction="", overrides_mode=True)
    assert config.active is False


def test_the_override_survives_a_restart(tmp_path):
    from core.monitoring_config import load_or_empty
    path = tmp_path / "m.json"
    MonitoringConfig(path).save(RULES, instruction="Count the bottles.", overrides_mode=True)
    assert load_or_empty(path).active is True
