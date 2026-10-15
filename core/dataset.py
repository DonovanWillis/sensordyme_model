"""Opt-in training capture: turning what the device sees into labelled examples.

SensorDyme cannot get better at recognising a can, or an unsafe sleeping
position, without examples of one. Those examples can only come from real
cameras in real rooms — which is exactly the footage a monitoring product must
be most careful with.

So this module is built around one rule: **nothing leaves the device unless a
person exports it deliberately.** There is no upload path here, no network
import, and no background sync. Capture writes to local disk, bounded, and stops
on its own. Getting data off the device is a separate, explicit action
(scripts/export_dataset.py) that a human runs.

What a sample is
----------------
One frame, plus what the device believed at the time: which rule or event it
belongs to, the confidence, the model version, and a session id. The belief
matters as much as the pixels — a frame the device got *wrong* is the most
valuable training example there is, and it is worthless without knowing what it
predicted.

Sessions and leakage
--------------------
Every sample carries a `session` — one continuous run of the device. Frames
from one session are near-duplicates of each other, so splitting a dataset
randomly puts near-copies of training frames into the test set and produces a
model that scores beautifully and fails on a desk. Splits are by session, and
the session id is recorded at capture time because it cannot be recovered later.
"""

from __future__ import annotations

import contextlib
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional

# What the device thought about a sample. Feedback arrives later, from the app.
LABEL_UNREVIEWED = "unreviewed"
LABEL_CORRECT = "correct"          # the alert was right
LABEL_FALSE_POSITIVE = "false_positive"
LABEL_MISSED = "missed"            # something happened and the device said nothing
LABEL_NOT_USEFUL = "not_useful"    # right, but the user did not want to know

REVIEW_LABELS = (LABEL_UNREVIEWED, LABEL_CORRECT, LABEL_FALSE_POSITIVE,
                 LABEL_MISSED, LABEL_NOT_USEFUL)

#: The filenames capture() writes: "<12-hex session>_<16-hex id>.jpg". purge()
#: deletes by this shape, so a frame the manifest lost track of is still found,
#: while anything else someone put in the directory is left alone.
_CAPTURED_IMAGE = re.compile(r"^[0-9a-f]{12}_[0-9a-f]{16}\.jpg$")


@dataclass
class Sample:
    """One captured frame and everything known about it."""

    id: str
    session: str
    captured_at: float
    image: str                     # filename, relative to the capture directory
    event_type: str = ""           # what fired, or "" for a negative sample
    rule_id: str = ""
    confidence: float = 0.0
    module: str = ""
    model_version: str = ""
    review: str = LABEL_UNREVIEWED
    notes: str = ""
    # Detections present when the frame was captured, so a labeller starts from
    # the model's guess instead of an empty canvas.
    detections: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


class CaptureStore:
    """Local, bounded, opt-in store of training samples.

    Disabled by default. A monitoring device that quietly accumulates footage of
    someone's living room is the thing customers are right to fear, so capture
    must be switched on deliberately and must stop by itself.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        config = config or {}
        self.enabled = bool(config.get("enabled", False))
        self.directory = Path(str(config.get("dir", "data/training")))
        self.max_samples = int(config.get("max_samples", 500))
        # Rate limit: an event that fires continuously would otherwise fill the
        # card with a thousand near-identical frames and crowd out everything else.
        self.min_interval_s = float(config.get("min_interval_s", 5.0))
        self.capture_negatives = bool(config.get("capture_negatives", False))
        self.negative_interval_s = float(config.get("negative_interval_s", 300.0))

        self.session = uuid.uuid4().hex[:12]
        self._last_capture = 0.0
        self._last_negative = 0.0
        self._samples: list[Sample] = []
        self._manifest = self.directory / "manifest.jsonl"
        if self.enabled:
            self._load()

    # ------------------------------------------------------------------ state
    def _load(self) -> None:
        """Read an existing manifest so restarts extend a dataset rather than
        silently starting a second one beside it.

        Line by line: a damaged line (a power cut mid-append) is skipped, not
        the end of the manifest. Stopping at it used to hide every later sample
        on every later boot -- the count froze, so `max_samples` stopped bounding
        the card, and the next review() rewrote the manifest without them.
        """
        known = {f.name for f in fields(Sample)}
        try:
            text = self._manifest.read_text()
        except FileNotFoundError:
            return
        except (OSError, UnicodeDecodeError):
            # A corrupt manifest must not stop the camera. The images are still
            # on disk; the export tool reports the discrepancy.
            return
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    continue
                self._samples.append(Sample(**{k: v for k, v in record.items() if k in known}))
            except (ValueError, TypeError):
                continue

    @property
    def count(self) -> int:
        return len(self._samples)

    @property
    def full(self) -> bool:
        return self.count >= self.max_samples

    def status(self) -> dict:
        """For the app's capture screen — what has been kept, and how much room is left."""
        return {
            "enabled": self.enabled,
            "session": self.session,
            "samples": self.count,
            "max_samples": self.max_samples,
            "full": self.full,
            "directory": str(self.directory),
            "unreviewed": sum(1 for s in self._samples if s.review == LABEL_UNREVIEWED),
        }

    # ------------------------------------------------------------------ capture
    def should_capture(self, now: Optional[float] = None, negative: bool = False) -> bool:
        """Whether a sample may be taken right now."""
        if not self.enabled or self.full:
            return False
        now = time.time() if now is None else now
        if negative:
            if not self.capture_negatives:
                return False
            return now - self._last_negative >= self.negative_interval_s
        return now - self._last_capture >= self.min_interval_s

    def capture(
        self,
        frame,
        *,
        event_type: str = "",
        rule_id: str = "",
        confidence: float = 0.0,
        module: str = "",
        model_version: str = "",
        detections: Optional[list] = None,
        now: Optional[float] = None,
        writer=None,
    ) -> Optional[Sample]:
        """Write one frame and its metadata. Returns the sample, or None if declined.

        `writer` exists so this is testable without OpenCV and so a future
        encoder (clips rather than stills) can be swapped in without touching
        the bookkeeping.
        """
        negative = not event_type
        if not self.should_capture(now, negative=negative):
            return None
        now = time.time() if now is None else now

        sample_id = uuid.uuid4().hex[:16]
        filename = f"{self.session}_{sample_id}.jpg"
        self.directory.mkdir(parents=True, exist_ok=True)

        try:
            (writer or _write_jpeg)(self.directory / filename, frame)
        except Exception:  # noqa: BLE001 - capture is never worth killing the camera for
            return None

        sample = Sample(
            id=sample_id,
            session=self.session,
            captured_at=now,
            image=filename,
            event_type=event_type,
            rule_id=rule_id,
            confidence=round(float(confidence), 3),
            module=module,
            model_version=model_version,
            detections=detections or [],
        )
        self._samples.append(sample)
        self._append_manifest(sample)
        if negative:
            self._last_negative = now
        else:
            self._last_capture = now
        return sample

    def _append_manifest(self, sample: Sample) -> None:
        # Start on a fresh line if the previous append was cut off mid-line;
        # otherwise this record is glued onto the fragment and lost with it.
        prefix = ""
        try:
            with self._manifest.open("rb") as existing:
                existing.seek(0, 2)
                if existing.tell() > 0:
                    existing.seek(-1, 2)
                    if existing.read(1) != b"\n":
                        prefix = "\n"
        except FileNotFoundError:
            pass
        with self._manifest.open("a") as handle:
            handle.write(prefix + json.dumps(sample.as_dict()) + "\n")

    # ------------------------------------------------------------------ review
    def review(self, sample_id: str, label: str, notes: str = "") -> bool:
        """Record the user's verdict on a sample.

        This is the highest-value data in the system: a confirmed false positive
        is worth many ordinary frames, because it is an example the current model
        demonstrably gets wrong.
        """
        if label not in REVIEW_LABELS:
            return False
        for sample in self._samples:
            if sample.id == sample_id:
                sample.review = label
                sample.notes = notes
                self._rewrite_manifest()
                return True
        return False

    def _rewrite_manifest(self) -> None:
        """Rewrite via a temp file so an interrupted write cannot truncate the
        manifest and orphan every image referenced by it."""
        temporary = self._manifest.with_suffix(".jsonl.tmp")
        with temporary.open("w") as handle:
            for sample in self._samples:
                handle.write(json.dumps(sample.as_dict()) + "\n")
        temporary.replace(self._manifest)

    def samples(self) -> list[Sample]:
        return list(self._samples)

    def purge(self) -> int:
        """Delete every captured sample. The user's right to change their mind.

        Every captured frame on disk, not only the ones the manifest still
        lists: a manifest damaged by a power cut must not leave footage behind
        after someone asked for it to be deleted.
        """
        targets = {self.directory / sample.image for sample in self._samples}
        with contextlib.suppress(OSError):
            targets |= {
                path for path in self.directory.iterdir() if _CAPTURED_IMAGE.match(path.name)
            }
        removed = 0
        for path in targets:
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
        self._samples.clear()
        with contextlib.suppress(OSError):
            self._manifest.unlink()
        return removed


def _write_jpeg(path: Path, frame) -> None:
    """Default encoder. Imported lazily so the store is testable without OpenCV."""
    import cv2

    if not cv2.imwrite(str(path), frame):
        raise OSError(f"could not write {path}")


def split_by_session(samples: list[Sample], ratios=(0.7, 0.15, 0.15)) -> dict[str, list[Sample]]:
    """Split into train/val/test by SESSION, never by frame.

    Frames from one recording are near-duplicates. Splitting them randomly puts
    near-copies of training frames into the test set, which produces a model that
    scores wonderfully and fails the moment it sees a room it was not trained in.
    Whole sessions therefore move together, even though that makes the split
    sizes less exact.
    """
    sessions = sorted({s.session for s in samples})
    if not sessions:
        return {"train": [], "val": [], "test": []}

    train_end = max(1, round(len(sessions) * ratios[0]))
    val_end = max(train_end, round(len(sessions) * (ratios[0] + ratios[1])))
    buckets = {
        "train": set(sessions[:train_end]),
        "val": set(sessions[train_end:val_end]),
        "test": set(sessions[val_end:]),
    }
    return {
        name: [s for s in samples if s.session in ids]
        for name, ids in buckets.items()
    }
