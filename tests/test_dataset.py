"""Opt-in training capture and export.

The privacy properties are tested first and hardest: a monitoring device that
accumulates footage of someone's living room by default is exactly what buyers
are right to fear, so "off unless asked" and "nothing leaves without an explicit
command" are behaviours, not intentions.
"""

import json

import pytest

from core.dataset import (
    LABEL_CORRECT,
    LABEL_FALSE_POSITIVE,
    LABEL_UNREVIEWED,
    CaptureStore,
    Sample,
    split_by_session,
)


def fake_writer(path, frame):
    path.write_bytes(b"jpeg")


@pytest.fixture()
def store(tmp_path):
    return CaptureStore({"enabled": True, "dir": str(tmp_path / "training"),
                         "max_samples": 5, "min_interval_s": 0})


def capture(store, **kwargs):
    kwargs.setdefault("event_type", "person_detected")
    return store.capture(object(), writer=fake_writer, **kwargs)


# --------------------------------------------------------------------- privacy
def test_capture_is_off_unless_explicitly_enabled(tmp_path):
    store = CaptureStore({"dir": str(tmp_path)})
    assert store.enabled is False
    assert capture(store) is None


def test_disabled_store_writes_nothing_to_disk(tmp_path):
    directory = tmp_path / "training"
    store = CaptureStore({"dir": str(directory)})
    capture(store)
    assert not directory.exists()


def test_purge_removes_every_captured_frame(store):
    capture(store, now=1.0)
    capture(store, now=2.0)
    assert store.count == 2
    assert store.purge() == 2
    assert store.count == 0
    assert not list(store.directory.glob("*.jpg"))


def test_negatives_require_their_own_opt_in(tmp_path):
    """Frames with no event are footage of ordinary life; capturing them is a
    separate decision from capturing alerts."""
    store = CaptureStore({"enabled": True, "dir": str(tmp_path), "min_interval_s": 0})
    assert store.capture(object(), writer=fake_writer, event_type="") is None

    opted_in = CaptureStore({"enabled": True, "dir": str(tmp_path), "min_interval_s": 0,
                             "capture_negatives": True, "negative_interval_s": 0})
    assert opted_in.capture(object(), writer=fake_writer, event_type="") is not None


# --------------------------------------------------------------------- bounds
def test_capture_stops_at_the_limit(store):
    for i in range(10):
        capture(store, now=float(i))
    assert store.count == 5
    assert store.full


def test_rate_limit_prevents_flooding_on_a_sustained_event(tmp_path):
    """A condition that holds would otherwise fill the card with near-identical
    frames and crowd out everything else."""
    store = CaptureStore({"enabled": True, "dir": str(tmp_path), "min_interval_s": 10})
    assert capture(store, now=100.0) is not None
    assert capture(store, now=105.0) is None
    assert capture(store, now=111.0) is not None


def test_a_failed_write_never_raises_into_the_camera(store):
    def broken(path, frame):
        raise OSError("disk full")

    assert store.capture(object(), writer=broken, event_type="x") is None
    assert store.count == 0


# --------------------------------------------------------------------- metadata
def test_sample_records_what_the_device_believed(store):
    sample = capture(store, event_type="fall", rule_id="rule0:present:on_floor",
                     confidence=0.82, module="fall_detection", model_version="yolo11n")
    assert sample.event_type == "fall"
    assert sample.confidence == 0.82
    assert sample.model_version == "yolo11n"
    assert sample.review == LABEL_UNREVIEWED
    assert sample.session == store.session


def test_manifest_survives_a_restart(tmp_path):
    first = CaptureStore({"enabled": True, "dir": str(tmp_path), "min_interval_s": 0})
    capture(first, now=1.0)
    capture(first, now=2.0)

    second = CaptureStore({"enabled": True, "dir": str(tmp_path), "min_interval_s": 0})
    assert second.count == 2
    # A new run is a new session, or the split would treat two rooms as one.
    assert second.session != first.session


def test_a_corrupt_manifest_does_not_stop_the_camera(tmp_path):
    (tmp_path).mkdir(parents=True, exist_ok=True)
    (tmp_path / "manifest.jsonl").write_text("{not json at all\n")
    store = CaptureStore({"enabled": True, "dir": str(tmp_path)})
    assert store.enabled  # loaded without raising


# --------------------------------------------------------------------- review
def test_user_review_is_recorded_and_persisted(store):
    sample = capture(store, now=1.0)
    assert store.review(sample.id, LABEL_FALSE_POSITIVE, notes="that was a coat")

    lines = (store.directory / "manifest.jsonl").read_text().strip().split("\n")
    stored = json.loads(lines[-1])
    assert stored["review"] == LABEL_FALSE_POSITIVE
    assert stored["notes"] == "that was a coat"


def test_unknown_review_labels_are_rejected(store):
    sample = capture(store, now=1.0)
    assert not store.review(sample.id, "probably_fine")


def test_reviewing_a_missing_sample_reports_failure(store):
    assert not store.review("nope", LABEL_CORRECT)


# --------------------------------------------------------------------- splitting
def sample(session, index=0):
    return Sample(id=f"{session}{index}", session=session, captured_at=0.0, image="x.jpg")


def test_split_keeps_whole_sessions_together():
    """Frames from one recording are near-duplicates; splitting them across
    train and test leaks the answer and inflates every metric."""
    samples = [sample(f"s{i}", j) for i in range(10) for j in range(3)]
    splits = split_by_session(samples)

    seen = {}
    for name, items in splits.items():
        for item in items:
            assert seen.setdefault(item.session, name) == name, "a session spans two splits"


def test_split_covers_every_sample():
    samples = [sample(f"s{i}") for i in range(10)]
    splits = split_by_session(samples)
    assert sum(len(v) for v in splits.values()) == len(samples)


def test_split_of_nothing_is_empty_not_an_error():
    assert split_by_session([]) == {"train": [], "val": [], "test": []}


def test_a_single_session_goes_to_train_and_the_gap_is_visible():
    """One session cannot produce an honest held-out set — the export warns."""
    splits = split_by_session([sample("only", i) for i in range(5)])
    assert len(splits["train"]) == 5
    assert splits["test"] == []


# --------------------------------------------------------------------- export
def test_export_writes_splits_and_flags_what_would_ruin_training(tmp_path):
    import sys

    sys.path.insert(0, "scripts")
    from export_dataset import export

    source = tmp_path / "training"
    store = CaptureStore({"enabled": True, "dir": str(source), "min_interval_s": 0,
                          "max_samples": 100})
    for i in range(4):
        capture(store, now=float(i), event_type="fall")

    manifest = export(source, tmp_path / "out", reviewed_only=False, copy=True)

    assert manifest["samples"] == 4
    assert (tmp_path / "out" / "train" / "labels.jsonl").exists()
    assert (tmp_path / "out" / "manifest.json").exists()
    joined = " ".join(manifest["warnings"])
    assert "session" in joined          # one session cannot hold out a test set
    assert "negative" in joined         # all-positive data teaches the model to say yes
    assert "unreviewed" in joined       # nobody has verified these labels
