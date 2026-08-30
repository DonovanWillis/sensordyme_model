"""The device dashboard's two cheap hardening fixes, and the uplink's unbound-name trap.

All three are small, and all three were sitting on paths that are LAN-reachable with
the shipped `dashboard.token: null`:

  * `/api/history` read the ENTIRE append-only event log into memory on every request
    before slicing off the last N rows. The log has no rotation, so the cost of one
    request grew for the life of the device -- an amplification vector that got worse
    the longer a device had been deployed, against a Pi already running inference.
  * The token comparison used `==`, which short-circuits on the first differing byte
    and leaks the token prefix to anyone who can time responses.
  * `CloudUplink._run`'s except handler re-queues `batch`, but `batch` was assigned
    inside the try. Anything raising before that assignment would make the handler
    itself raise NameError, killing the uplink thread with no outer try to catch it.
"""

from __future__ import annotations

import csv

import pytest

from core.dashboard import DashboardServer


def _write_log(path, rows: int) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "type", "duration_s", "confidence", "module",
                         "contributing_signals"])
        for i in range(rows):
            writer.writerow([f"2026-08-29T00:00:{i % 60:02d}+00:00", f"event_{i}",
                             "1.0", "0.9", "driver_attention", ""])


def _dashboard(path) -> DashboardServer:
    """A DashboardServer with only the fields history() touches."""
    server = DashboardServer.__new__(DashboardServer)
    server._events_csv_path = str(path)
    return server


def test_history_returns_the_newest_rows_first(tmp_path):
    log = tmp_path / "events.csv"
    _write_log(log, 500)

    rows = _dashboard(log).history(limit=10)

    assert len(rows) == 10
    assert rows[0]["type"] == "event_499", "newest row should be first"
    assert rows[-1]["type"] == "event_490"


def test_history_reads_only_the_tail_of_a_large_log(tmp_path, monkeypatch):
    """The regression: cost must not scale with the size of the log.

    Asserted by counting bytes actually read, which is the property that matters --
    a test on wall-clock time would be flaky on a loaded machine.
    """
    log = tmp_path / "events.csv"
    _write_log(log, 200_000)  # comfortably past the 512KB tail window
    size = log.stat().st_size
    assert size > DashboardServer._HISTORY_TAIL_BYTES * 2, "test log is not large enough"

    dashboard = _dashboard(log)
    read_bytes = 0

    class _CountingFile:
        """Delegates to the real handle, tallying bytes actually pulled off disk."""

        def __init__(self, inner):
            self._inner = inner

        def read(self, *args):
            nonlocal read_bytes
            data = self._inner.read(*args)
            read_bytes += len(data)
            return data

        def readline(self, *args):
            nonlocal read_bytes
            data = self._inner.readline(*args)
            read_bytes += len(data)
            return data

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *exc):
            return self._inner.__exit__(*exc)

    real_open = type(log).open

    def counting_open(self, *args, **kwargs):
        return _CountingFile(real_open(self, *args, **kwargs))

    monkeypatch.setattr(type(log), "open", counting_open)
    rows = dashboard.history(limit=200)

    assert len(rows) == 200
    assert rows[0]["type"] == "event_199999", "tail read lost the newest rows"
    assert read_bytes < size / 2, (
        f"read {read_bytes} of {size} bytes -- history is still reading the whole log"
    )


def test_history_survives_a_log_smaller_than_the_tail_window(tmp_path):
    """The small-file path must not drop rows by seeking past the start."""
    log = tmp_path / "events.csv"
    _write_log(log, 3)

    rows = _dashboard(log).history(limit=200)

    assert [r["type"] for r in rows] == ["event_2", "event_1", "event_0"]


def test_history_on_a_header_only_or_missing_log_is_empty(tmp_path):
    missing = tmp_path / "nope.csv"
    assert _dashboard(missing).history() == []

    empty = tmp_path / "empty.csv"
    _write_log(empty, 0)
    assert _dashboard(empty).history() == []


def test_history_does_not_emit_a_partial_row_from_the_seek_boundary(tmp_path):
    """Landing mid-row must drop that row, not surface a mangled one to the app."""
    log = tmp_path / "events.csv"
    _write_log(log, 50_000)

    rows = _dashboard(log).history(limit=1000)

    assert all(r["type"].startswith("event_") for r in rows), (
        "a partial line from the seek boundary leaked into the result"
    )
    assert all(r["timestamp"].startswith("2026-") for r in rows)


def test_token_comparison_is_constant_time():
    """Pin the compare_digest call, so a later refactor cannot quietly restore `==`."""
    import inspect

    source = inspect.getsource(DashboardServer)
    assert "compare_digest" in source, "the timing-safe token comparison was removed"
    assert "return sent == dashboard._token" not in source, (
        "the short-circuiting comparison is back; it leaks the token prefix by timing"
    )


def test_uplink_binds_batch_before_the_try_so_the_handler_cannot_raise_nameerror():
    """The except handler re-queues `batch`; it must always be bound when it runs.

    Checked structurally rather than by forcing the race: the failure needs an
    exception between entering the loop and the assignment, which is genuinely hard
    to provoke and is exactly why this sat unnoticed. What is assertable is that the
    binding now precedes the try.
    """
    import inspect

    from core.uplink import CloudUplink

    source = inspect.getsource(CloudUplink._run)
    batch_init = source.index("batch")
    try_index = source.index("try:")
    assert batch_init < try_index, (
        "`batch` is bound inside the try again -- if anything raises before the "
        "assignment the except handler raises NameError and kills the uplink thread"
    )
