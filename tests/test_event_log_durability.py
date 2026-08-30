"""The event log is the device's only permanent record of what it saw.

The upload queue is bounded and in memory, so beyond the last few hundred
events this CSV *is* the history. flush() hands bytes to the OS, which is free
to extend the file's length before the data reaches the card — cut the power in
that window and the file comes back longer with the gap reading as NUL bytes.
Found on the real device: 944 NUL bytes across 3 lines.
"""

import datetime
import os

from core.events import Event
from core.logger import EventLogger, repair


def an_event(kind: str = "fall") -> Event:
    return Event(
        timestamp=datetime.datetime.now(datetime.UTC),
        type=kind, duration=1.0, confidence=0.9,
        module_name="test", contributing_signals="",
    )


def test_each_event_is_synced_to_the_card(tmp_path, monkeypatch):
    """The fix. flush() alone leaves data in a buffer the kernel may write
    after it has already extended the file."""
    synced = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real_fsync(fd))[1])

    path = tmp_path / "events.csv"
    with EventLogger(path) as logger:
        logger.log(an_event())
    assert synced, "the event log was never synced to disk"


def test_a_failing_sync_does_not_stop_the_device_monitoring(tmp_path, monkeypatch):
    """A dying card is worth knowing about; it is not worth crashing over."""
    def explode(fd):
        raise OSError("card gone")

    monkeypatch.setattr(os, "fsync", explode)
    path = tmp_path / "events.csv"
    with EventLogger(path) as logger:
        logger.log(an_event())          # must not raise
    assert path.exists()


def test_damage_from_an_interrupted_write_is_removed(tmp_path):
    path = tmp_path / "events.csv"
    path.write_bytes(
        b"timestamp,type\n"
        b"2026-08-07T10:00:00,fall\n"
        b"\x00\x00\x00\x00\x00\x00\n"          # what a power cut leaves
        b"2026-08-07T10:05:00,drowsiness\n"
    )
    assert repair(path) == 1
    assert b"\x00" not in path.read_bytes()


def test_repair_keeps_every_intact_row(tmp_path):
    """The damage is one row. Everything around it is the history someone may
    care about, so this removes the damage rather than the file."""
    path = tmp_path / "events.csv"
    path.write_bytes(
        b"timestamp,type\n"
        b"2026-08-07T10:00:00,fall\n"
        b"\x00\x00\x00\n"
        b"2026-08-07T10:05:00,drowsiness\n"
    )
    repair(path)
    text = path.read_text()
    assert "fall" in text and "drowsiness" in text
    assert text.count("\n") >= 2


def test_a_clean_file_is_left_completely_alone(tmp_path):
    path = tmp_path / "events.csv"
    original = b"timestamp,type\n2026-08-07T10:00:00,fall\n"
    path.write_bytes(original)
    assert repair(path) == 0
    assert path.read_bytes() == original


def test_opening_the_log_repairs_it(tmp_path):
    """Repaired on the way in, so every consumer — dashboard, exporter, a person
    with a spreadsheet — sees a clean file rather than each tolerating damage."""
    path = tmp_path / "events.csv"
    path.write_bytes(b"timestamp,type\n2026-08-07T10:00:00,fall\n\x00\x00\n")
    with EventLogger(path) as logger:
        logger.log(an_event("on_floor"))
    text = path.read_text()
    assert "\x00" not in text
    assert "fall" in text and "on_floor" in text


def test_repair_leaves_no_temporary_file_behind(tmp_path):
    path = tmp_path / "events.csv"
    path.write_bytes(b"a\n\x00\nb\n")
    repair(path)
    assert list(tmp_path.glob("*.repair")) == []


def test_a_missing_file_is_not_an_error(tmp_path):
    assert repair(tmp_path / "never-existed.csv") == 0
