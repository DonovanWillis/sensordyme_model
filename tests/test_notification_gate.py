"""Which events may interrupt someone, decided where it can be enforced.

The server is the only part still running when the phone's app is not, so it
is the only place a filter can protect the channel that matters most. It had
none: every "loud" event became a push, roughly 960 in two days from a device
doing exactly its job.

Nothing here discards an event. Every one is stored and appears in the app's
timeline; this decides interruptions only.
"""

from cloud.notify import COOLDOWN_S, NotificationGate


def test_a_red_alert_notifies():
    assert NotificationGate().should_notify(1, "loud", now=1000.0) is True


def test_a_warning_does_not_notify():
    """It still reaches the timeline — it just does not reach a lock screen."""
    assert NotificationGate().should_notify(1, "soft", now=1000.0) is False


def test_the_worked_example_holds():
    """notify at :00, suppress :02 :05 :11 :18, eligible again at :20."""
    gate = NotificationGate()
    assert gate.should_notify(1, "loud", now=0.0) is True
    for t in (2.0, 5.0, 11.0, 18.0):
        assert gate.should_notify(1, "loud", now=t) is False, f"should be suppressed at {t}"
    assert gate.should_notify(1, "loud", now=20.0) is True


def test_a_sustained_incident_does_not_buzz_every_few_seconds():
    """Someone still on the floor is not new information every four seconds."""
    gate = NotificationGate()
    sent = sum(gate.should_notify(1, "loud", now=float(t)) for t in range(0, 60, 2))
    assert sent == 3, f"60s of continuous alerts should send 3, sent {sent}"


def test_suppressed_alerts_are_counted_not_silently_dropped():
    """"and 4 more" is information; quietly dropping them is the same behaviour
    without the honesty."""
    gate = NotificationGate()
    gate.should_notify(1, "loud", now=0.0)
    for t in (2.0, 5.0, 9.0):
        gate.should_notify(1, "loud", now=t)
    assert gate.suppressed_since_last(1) == 3


def test_the_count_resets_after_a_notification_goes_out():
    gate = NotificationGate()
    gate.should_notify(1, "loud", now=0.0)
    gate.should_notify(1, "loud", now=3.0)
    assert gate.suppressed_since_last(1) == 1
    gate.should_notify(1, "loud", now=25.0)
    assert gate.suppressed_since_last(1) == 0


def test_the_limit_is_per_account_not_per_device():
    """Two cameras in one house must not double the interruptions."""
    gate = NotificationGate()
    assert gate.should_notify(7, "loud", now=0.0) is True
    assert gate.should_notify(7, "loud", now=5.0) is False   # other camera, same person


def test_two_different_accounts_do_not_share_a_cooldown():
    gate = NotificationGate()
    assert gate.should_notify(1, "loud", now=0.0) is True
    assert gate.should_notify(2, "loud", now=0.5) is True


def test_an_unclaimed_device_notifies_nobody():
    """No owner means no one to interrupt — and must not raise on the hot path."""
    assert NotificationGate().should_notify(None, "loud", now=0.0) is False


def test_a_warning_inside_the_cooldown_does_not_consume_it():
    """A suppressed warning must not look like a sent notification, or a real
    red alert arriving next second would be wrongly held back."""
    gate = NotificationGate()
    assert gate.should_notify(1, "soft", now=0.0) is False
    assert gate.should_notify(1, "loud", now=1.0) is True


def test_the_default_cooldown_is_twenty_seconds():
    assert COOLDOWN_S == 20.0


def test_the_held_count_survives_the_send_that_reports_it():
    """The count is consumed by sending — reading it afterwards gives zero,
    because sending resets it. The first version did exactly that, silently
    dropping "and 4 more" from every message it was meant to appear on."""
    gate = NotificationGate()
    gate.evaluate(1, "loud", now=0.0)              # sent
    for t in (2.0, 5.0, 9.0):
        gate.evaluate(1, "loud", now=t)            # held back
    send, held = gate.evaluate(1, "loud", now=25.0)
    assert send is True
    assert held == 3, f"the next notification should report 3 held, got {held}"


def test_a_quiet_period_reports_nothing_held():
    gate = NotificationGate()
    gate.evaluate(1, "loud", now=0.0)
    send, held = gate.evaluate(1, "loud", now=100.0)
    assert send is True and held == 0


def test_something_the_user_configured_always_qualifies():
    """The regression that made this feature look broken.

    A custom rule defaults to severity "warning" -> "soft", which the gate
    dropped. So the one event someone had explicitly asked to be told about was
    the only one that never arrived, while the preset's own alerts kept coming
    through — the exact opposite of what configuring it meant.
    """
    gate = NotificationGate()
    assert gate.should_notify(1, "soft", now=0.0) is False           # a preset warning
    gate.reset()
    send, _ = gate.evaluate(1, "soft", now=0.0, source="custom_rule")
    assert send is True, "a configured rule did not notify"


def test_a_configured_event_still_respects_the_cooldown():
    """Honouring the intent must not reintroduce spam."""
    gate = NotificationGate()
    assert gate.evaluate(1, "soft", now=0.0, source="custom_rule")[0] is True
    assert gate.evaluate(1, "soft", now=5.0, source="custom_rule")[0] is False
    assert gate.evaluate(1, "soft", now=20.0, source="custom_rule")[0] is True


def test_a_presets_own_warning_is_still_not_worth_interrupting_for():
    gate = NotificationGate()
    assert gate.evaluate(1, "soft", now=0.0, source="module")[0] is False
