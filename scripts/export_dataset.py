#!/usr/bin/env python3
"""Export captured samples into a training-ready dataset.

This is the ONLY way data leaves a SensorDyme device, and it is a command a
person runs deliberately. There is no background upload, and adding one would
undo the reason capture is safe to enable at all.

    python scripts/export_dataset.py --source data/training --out ~/sensordyme-dataset
    python scripts/export_dataset.py --source data/training --out /tmp/d --reviewed-only

What you get:

    <out>/train/images/…            frames, split BY SESSION
    <out>/val/images/…
    <out>/test/images/…
    <out>/<split>/labels.jsonl      one record per frame, with the device's
                                    prediction and the user's review
    <out>/manifest.json             counts, sessions, class balance, warnings

The split is by session rather than by frame on purpose: frames from one
recording are near-duplicates, so a random split leaks the test set into
training and yields a model that scores well and fails in a new room.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.dataset import LABEL_UNREVIEWED, Sample, split_by_session


def load_samples(source: Path) -> list[Sample]:
    manifest = source / "manifest.jsonl"
    if not manifest.exists():
        raise SystemExit(f"no manifest at {manifest} — has capture ever been enabled?")
    samples: list[Sample] = []
    for number, line in enumerate(manifest.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            samples.append(Sample(**json.loads(line)))
        except (ValueError, TypeError) as exc:
            print(f"  ! skipping malformed manifest line {number}: {exc}")
    return samples


def export(source: Path, out: Path, reviewed_only: bool, copy: bool) -> dict:
    samples = load_samples(source)
    if reviewed_only:
        samples = [s for s in samples if s.review != LABEL_UNREVIEWED]

    missing = [s for s in samples if not (source / s.image).exists()]
    if missing:
        print(f"  ! {len(missing)} sample(s) reference images that are gone; skipping them")
        present = {s.id for s in samples} - {s.id for s in missing}
        samples = [s for s in samples if s.id in present]

    if not samples:
        raise SystemExit("nothing to export")

    splits = split_by_session(samples)
    out.mkdir(parents=True, exist_ok=True)

    for split_name, split_samples in splits.items():
        images = out / split_name / "images"
        images.mkdir(parents=True, exist_ok=True)
        with (out / split_name / "labels.jsonl").open("w") as handle:
            for sample in split_samples:
                destination = images / sample.image
                if copy:
                    shutil.copy2(source / sample.image, destination)
                else:
                    shutil.move(str(source / sample.image), destination)
                handle.write(json.dumps(sample.as_dict()) + "\n")

    manifest = {
        "samples": len(samples),
        "sessions": sorted({s.session for s in samples}),
        "splits": {name: len(items) for name, items in splits.items()},
        "events": dict(Counter(s.event_type or "(negative)" for s in samples)),
        "reviews": dict(Counter(s.review for s in samples)),
        "warnings": _warnings(samples, splits),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _warnings(samples: list[Sample], splits: dict) -> list[str]:
    """Problems that would quietly ruin a training run.

    Said out loud here because every one of them produces a model that looks
    fine by its own metrics and disappoints in the room.
    """
    warnings: list[str] = []
    sessions = {s.session for s in samples}
    if len(sessions) < 3:
        warnings.append(
            f"only {len(sessions)} session(s): a split by session cannot hold out "
            "meaningfully different conditions. Record on different days, in "
            "different light, before training."
        )
    if not splits["test"]:
        warnings.append("the test split is empty — there are too few sessions to hold one out")

    negatives = sum(1 for s in samples if not s.event_type)
    if negatives == 0:
        warnings.append(
            "no negative samples: a detector trained only on frames containing the "
            "target learns to say yes. Enable capture_negatives."
        )
    unreviewed = sum(1 for s in samples if s.review == LABEL_UNREVIEWED)
    if unreviewed:
        warnings.append(
            f"{unreviewed} sample(s) are unreviewed — they carry the device's guess, "
            "not a verified label, and training on them teaches the model its own mistakes"
        )
    counts = Counter(s.event_type or "(negative)" for s in samples)
    if counts:
        most, least = counts.most_common()[0], counts.most_common()[-1]
        if least[1] and most[1] / least[1] > 10:
            warnings.append(
                f"class imbalance: '{most[0]}' has {most[1]} samples and '{least[0]}' has "
                f"{least[1]}. Balance them or weight the loss."
            )
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=Path("data/training"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reviewed-only", action="store_true",
                        help="export only samples a human has labelled")
    parser.add_argument("--move", action="store_true",
                        help="move rather than copy, freeing space on the device")
    args = parser.parse_args()

    manifest = export(args.source, args.out, args.reviewed_only, copy=not args.move)

    print(f"\nexported {manifest['samples']} sample(s) to {args.out}")
    print(f"  sessions : {len(manifest['sessions'])}")
    for name, count in manifest["splits"].items():
        print(f"  {name:<6}   : {count}")
    print(f"  events   : {manifest['events']}")
    if manifest["warnings"]:
        print("\nbefore you train:")
        for warning in manifest["warnings"]:
            print(f"  ! {warning}")


if __name__ == "__main__":
    main()
