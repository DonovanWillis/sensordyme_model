"""The migration must carry the notification invariant onto deployed devices.

tests/test_config.py::test_every_presets_alert_types_can_actually_notify holds
one property over the REPO's config.yaml: every event type a preset declares
as alert-worthy must be in `alerts.loud_types`, because services/runtime.py
defaults anything else to "soft" and cloud/notify.py's NotificationGate drops
"soft". That is the difference between a fall producing a push and an SMS, and
a fall producing a red screen that nobody who isn't already looking ever sees.

A deployed device never receives the repo's config.yaml -- `make deploy` rsyncs
with --exclude='config.yaml' precisely so a unit keeps its own device_key,
backend address and camera rotation. scripts/migrate_config.py's
LIST_MIGRATIONS is therefore the ONLY channel by which a loud_types fix ever
reaches a real device.

So the repo-side test was passing over a property that did not hold where it
matters. It stayed green because it read the repo's config, while
test_config_migration.py checked only that two individual members were added.
Five types -- `fall` among them -- were missing from LIST_MIGRATIONS entirely.

These tests close that: the first migrates a genuine historical config forward
and asserts the RESULT satisfies the same invariant; the second checks the
migration list against presets.yaml directly, so a newly-declared alert type
cannot go missing the same way.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from main import load_config, load_presets  # noqa: E402
from scripts.migrate_config import LIST_MIGRATIONS, migrate  # noqa: E402

# The alerts block a device deployed from commit 28e276e is carrying -- the
# release that introduced baby monitoring, zones and Configure-with-AI, and
# so the oldest config still plausibly running in the field. Reproduced
# verbatim rather than paraphrased: the bug is about what real devices hold,
# and a tidied-up fixture is exactly how it hid for so long.
DEPLOYED_CONFIG_28E276E = """\
camera:
  rotate_180: true

network:
  provisioning:
    method: ble
    grace_s: 45

alerts:
  sinks: [console, sound, gpio_buzzer] # gpio_buzzer drives the piezo on the Pi device
  save_evidence: false
  evidence_dir: data/evidence
  evidence_max_files: 50
  cooldown_s: 15 # minimum seconds between two alerts of the same type
  escalation_window_s: 60 # repeat within this window -> continuous tone
  loud_types: [drowsiness, phone_use, eyes_down, attention_lost, high_risk, # always the loud tone
               driverDistracted, driverUnresponsive, # openpilot terminal (red) alerts
               phoneUseDetected, eyesClosedDetected, # openpilot additive flags
               skinnemoen_distracted, skinnemoen_phone_use, # skinnemoen module
               dashsentinel_drowsy] # DashSentinel's DROWSY status
  soft_types: [fatigue_warning, frequent_glancing, elevated_risk, # always the soft chime
               promptDriverDistracted, promptDriverUnresponsive,
               distractionDetected,
               skinnemoen_calibration_complete,
               dashsentinel_warning, dashsentinel_no_face, dashsentinel_vision_degraded]
  gpio_buzzer_pin: 17 # BCM pin, Pi only

cloud:
  enabled: true
  base_url: http://Donovans-MacBook-Pro.local:8787
  device_key: dk_secret_value
"""


def _unreachable_alert_types(config: dict) -> list[str]:
    """`preset:event_type` for every alert type that cannot ever notify.

    Reproduces services/runtime.py's severity formula exactly, the same way
    tests/test_config.py does: every type reached here is by construction a
    member of alert_event_types, so the formula resolves to "loud" when the
    type is explicitly listed and "soft" otherwise -- and "soft" is what
    cloud/notify.py's NotificationGate silently drops.
    """
    loud = set(config["alerts"]["loud_types"])
    return [
        f"{preset_name}:{event_type}"
        for preset_name, preset in load_presets().items()
        for event_type in preset.get("alert_types", [])
        if event_type not in loud
    ]


@pytest.fixture()
def deployed_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(DEPLOYED_CONFIG_28E276E)
    return path


def test_the_deployed_config_starts_out_unable_to_notify(deployed_config):
    """The premise of the test below: this really is a config where a fall
    cannot reach a phone. If this ever comes back empty the fixture has
    drifted and the next test proves nothing."""
    unreachable = _unreachable_alert_types(load_config(deployed_config))
    assert "elderly_monitoring:fall" in unreachable


def test_migrating_a_deployed_config_restores_every_preset_alert(deployed_config):
    """THE deliverable. A real old config, migrated forward the way `make
    deploy` migrates it, must satisfy the same invariant the repo's own
    config is held to.

    Before the LIST_MIGRATIONS fix this reported eight still-unreachable
    types after a full migration, including elderly_monitoring:fall.
    """
    migrate(deployed_config)
    unreachable = _unreachable_alert_types(load_config(deployed_config))
    assert not unreachable, (
        "after migration a DEPLOYED device still cannot notify for these preset "
        f"alert types (missing from alerts.loud_types): {unreachable}. Add each to "
        "LIST_MIGRATIONS in scripts/migrate_config.py -- the repo's own config.yaml "
        "never reaches a deployed device (make deploy excludes it)."
    )


def test_migration_is_still_idempotent_with_the_added_members(deployed_config):
    """Adding five entries must not make the script re-write the file every
    time it runs -- the whole contract of this script is that it converges."""
    migrate(deployed_config)
    after_first = deployed_config.read_text()
    assert migrate(deployed_config) == []
    assert deployed_config.read_text() == after_first


def test_the_device_keeps_its_own_identity_through_the_bigger_migration(deployed_config):
    """The added members must not have widened what this script touches."""
    migrate(deployed_config)
    text = deployed_config.read_text()
    assert "dk_secret_value" in text
    assert "rotate_180: true" in text
    assert "grace_s: 45" in text  # the device's value, not the repo's 0
    assert "cooldown_s: 15" in text


def test_every_preset_alert_type_is_covered_by_a_list_migration():
    """The invariant stated against the source of truth, with no fixture in
    between: a type added to a preset's `alert_types` tomorrow must also be
    added to LIST_MIGRATIONS, or devices already in the field will silently
    never notify for it. This is the check that would have caught the
    original five, and it does not depend on any particular old config.
    """
    migrated_members = {
        member for dotted, member, _why in LIST_MIGRATIONS if dotted == "alerts.loud_types"
    }
    repo_loud = set(load_config(REPO_ROOT / "config.yaml")["alerts"]["loud_types"])

    uncovered = sorted(
        f"{preset_name}:{event_type}"
        for preset_name, preset in load_presets().items()
        for event_type in preset.get("alert_types", [])
        if event_type not in migrated_members
    )
    assert not uncovered, (
        "these preset alert types are loud in the repo but have no LIST_MIGRATIONS "
        f"entry, so no deployed device will ever get them: {uncovered}"
    )
    # And the two lists must agree: a member migrated onto devices that the
    # repo itself does not consider loud would be a different kind of drift.
    assert migrated_members <= repo_loud, (
        f"LIST_MIGRATIONS pushes types the repo's config.yaml does not list as loud: "
        f"{sorted(migrated_members - repo_loud)}"
    )


def test_a_migrated_deployed_config_is_as_loud_as_a_fresh_checkout(deployed_config):
    """The invariant above only covers presets that declare `alert_types`.
    Driver Safety declares none -- it alerts on main.ALERT_EVENT_TYPES_PI -- so
    `distraction` and `eyes_up`, made loud in the repo after this config
    shipped, stayed soft (no push) on every deployed unit while the preset
    check passed. Stated over the whole list instead: a unit migrated forward
    from the oldest field config tags exactly what a fresh checkout tags.

    M8.4 (owner decision A): EFFECTIVELY loud. The old config still lists
    high_risk/elevated_risk (migration never removes a device's own list
    members); they are inert because every reader gates them
    (core/behavior/experimental.gate_alert_lists, AlertManager.permits), and
    migration writes the explicit `experimental_alerts_audible: false`."""
    from core.behavior.experimental import gate_alert_lists

    migrate(deployed_config)
    device_alerts = load_config(deployed_config)["alerts"]
    assert "high_risk" in device_alerts["loud_types"], "the device's own list is left as it was"
    assert device_alerts["experimental_alerts_audible"] is False
    device_loud = set(gate_alert_lists(device_alerts)[0]["loud_types"])
    repo_loud = set(gate_alert_lists(load_config(REPO_ROOT / "config.yaml")["alerts"])[0]["loud_types"])
    assert device_loud == repo_loud, (
        f"missing on a migrated device: {sorted(repo_loud - device_loud)}; "
        f"extra: {sorted(device_loud - repo_loud)}"
    )


def test_every_pi_alert_type_the_repo_makes_loud_reaches_deployed_devices(deployed_config):
    """Same property from the alert-type side, independent of the fixture's
    contents: every type the pi profile alerts on and the repo tags loud is
    loud after migration."""
    from main import ALERT_EVENT_TYPES_PI

    migrate(deployed_config)
    device_loud = set(load_config(deployed_config)["alerts"]["loud_types"])
    repo_loud = set(load_config(REPO_ROOT / "config.yaml")["alerts"]["loud_types"])
    assert sorted((ALERT_EVENT_TYPES_PI & repo_loud) - device_loud) == []
