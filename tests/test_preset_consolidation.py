"""One driver mode, and nobody stranded by the rename.

There used to be two driver presets that asked a consumer to choose between
inference engines to answer the same question. Merging them is easy; not
breaking the devices already holding the old name is the part that matters —
an unknown preset is a device stuck in a mode it cannot switch out of, and the
old name can be in a deployed config.yaml, a desired_preset row in the backend,
or a phone that has not been updated.
"""

from main import load_preset_aliases, load_presets, resolve_preset


def test_there_is_exactly_one_driver_mode():
    driver = [name for name in load_presets() if "driver" in name]
    assert driver == ["driver_safety"], driver


def test_the_four_consumer_modes_are_present():
    modes = set(load_presets())
    assert modes == {"driver_safety", "baby_monitoring", "elderly_monitoring", "security_monitoring"}


def test_the_driver_mode_is_named_for_what_it_does():
    assert load_presets()["driver_safety"]["title"] == "Driver Safety"


def test_a_device_still_on_the_old_name_keeps_working():
    presets = load_presets()
    assert resolve_preset("driver_distraction", presets) == "driver_safety"


def test_a_device_on_the_openpilot_name_also_lands_somewhere_real():
    presets = load_presets()
    assert resolve_preset("driver_distraction_openpilot", presets) == "driver_safety"


def test_a_variant_survives_the_rename():
    """The variant belongs to the preset, not to the name it used to have."""
    presets = load_presets()
    assert resolve_preset("driver_distraction:teen_driver", presets) == "driver_safety:teen_driver"


def test_a_variant_the_replacement_does_not_have_is_dropped_rather_than_carried():
    """openpilot had no teen_driver. Carrying it across would fail later, a long
    way from the rename that caused it."""
    presets = load_presets()
    resolved = resolve_preset("driver_distraction_openpilot:strict", presets)
    base, _, variant = resolved.partition(":")
    assert base == "driver_safety"
    if variant:
        assert variant in presets["driver_safety"]["variants"]


def test_a_current_name_is_returned_untouched():
    presets = load_presets()
    assert resolve_preset("baby_monitoring", presets) == "baby_monitoring"
    assert resolve_preset("driver_safety:strict", presets) == "driver_safety:strict"


def test_an_unknown_name_is_passed_through_rather_than_guessed_at():
    """Silently redirecting an unrecognised mode would hide a real config error."""
    presets = load_presets()
    assert resolve_preset("something_invented", presets) == "something_invented"


def test_the_aliases_all_point_at_modes_that_exist():
    presets, aliases = load_presets(), load_preset_aliases()
    for old, new in aliases.items():
        assert new in presets, f"alias {old} -> {new}, which is not a preset"


def test_the_openpilot_module_still_exists_for_benchmarking():
    """Consolidating the product decision must not delete the comparison engine."""
    from modules import available_modules
    assert "driver_attention_openpilot" in available_modules()
