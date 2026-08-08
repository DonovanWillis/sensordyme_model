"""Which events are allowed to interrupt someone, decided where it can be enforced.

The same policy the app applies locally — only the red alert, and at most one
notification per cooldown window — but here, on the server, because the server
is the only part that is still running when the phone's app is not.

That distinction is the whole point of this module. Local notifications are
posted BY the app, so they need it alive; a push is delivered TO the phone by
the OS whether the app is open, backgrounded, force-quit or the phone locked.
Both paths must agree on what deserves an interruption, or the same event
buzzes twice — or worse, the path that works when you are not looking is the
unfiltered one, which is what was happening: the app filtered and the server
did not, so every one of ~960 red alerts became a push.

Deliberately independent of the delivery mechanism. ntfy today, APNs when
there is a paid Apple account; the decision does not change, only the
transport underneath it.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

# Severities the device emits. "loud" is the red alert the app shows as major;
# "soft" is a warning that belongs in the timeline, not on a lock screen.
CRITICAL_SEVERITY = "loud"

# Minimum gap between notifications to one account, in seconds.
#
# Not a detection setting: the device keeps detecting and recording at full
# rate. This governs only how often a person may be interrupted. Someone still
# on the floor is not new information every four seconds.
COOLDOWN_S = 20.0


class NotificationGate:
    """Per-account rate limit for outbound notifications.

    In memory on purpose. A cooldown is a statement about the last few seconds,
    so it does not need to survive a restart — and persisting it would mean a
    database write on the hot ingest path for something whose whole value is
    that it is cheap. A backend restart at worst allows one extra notification.
    """

    def __init__(self, cooldown_s: float = COOLDOWN_S) -> None:
        self._cooldown_s = cooldown_s
        self._last_sent: dict[int, float] = {}
        self._suppressed: dict[int, int] = {}
        self._lock = threading.Lock()

    def evaluate(self, user_id: Optional[int], severity: str,
                 now: Optional[float] = None) -> tuple[bool, int]:
        """(may interrupt now, how many were held back since the last one).

        Returns both together because the count is consumed by the act of
        sending: reading it afterwards gives zero, since sending resets it.
        Doing that was the first version of this, and it silently dropped the
        "and 4 more" from every message it was meant to appear on.

        A False first element never means the event is discarded — it is
        already stored, and the timeline shows it. It means only that the phone
        will not buzz for it.
        """
        if user_id is None or severity != CRITICAL_SEVERITY:
            return False, 0

        now = time.time() if now is None else now
        with self._lock:
            last = self._last_sent.get(user_id)
            if last is not None and now - last < self._cooldown_s:
                self._suppressed[user_id] = self._suppressed.get(user_id, 0) + 1
                return False, self._suppressed[user_id]
            held = self._suppressed.get(user_id, 0)
            self._last_sent[user_id] = now
            self._suppressed[user_id] = 0
            return True, held

    def should_notify(self, user_id: Optional[int], severity: str,
                      now: Optional[float] = None) -> bool:
        """Whether this event may interrupt `user_id` right now."""
        return self.evaluate(user_id, severity, now)[0]

    def suppressed_since_last(self, user_id: int) -> int:
        """How many red alerts were recorded but not sent since the last one.

        Reported rather than silently dropped: "and 4 more" is information a
        person can act on, and it is the honest way to say that quietening the
        phone did not quieten the device.
        """
        with self._lock:
            return self._suppressed.get(user_id, 0)

    def reset(self, user_id: Optional[int] = None) -> None:
        """Forget rate-limit state — for tests, and for a manual test send."""
        with self._lock:
            if user_id is None:
                self._last_sent.clear()
                self._suppressed.clear()
            else:
                self._last_sent.pop(user_id, None)
                self._suppressed.pop(user_id, None)


# One gate for the process. Notifications are per account, and the limit has to
# be shared across every device that account owns — two cameras in one house
# should not double the interruptions.
gate = NotificationGate()
