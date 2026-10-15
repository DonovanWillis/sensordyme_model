"""Headless accuracy measurement against the State Farm distracted-driver dataset.

Why this exists
---------------
`run_benchmarks.py` measures latency/FPS. `accuracy_protocol.py` measures accuracy
but needs a HUMAN to act out timed scenarios in front of a camera, so it has never
been run. That left the product with no accuracy number of any kind for any vision
model, while a labelled 44,848-image driver dataset sat unused in `data/`.

This module closes that gap for the detector half of driver_attention: it scores the
SAME models the product actually loads, at the SAME thresholds config.yaml actually
ships, against a subject-disjoint held-out split, with no human in the loop. It runs
in ~30s for the default sample and is safe to gate CI on.

What it measures
----------------
  * phone PR curve  -- YOLO 'cell phone' recall/precision across a confidence sweep,
                       scored on the four phone classes (c1-c4) as positives and
                       c0/c9 (safe driving, talking to passenger) as negatives.
  * person recall   -- YOLO 'person' availability. Gates presence, zones, stillness,
                       occupancy, and the candidate set baby_monitoring's InfantIdentifier
                       chooses the infant from (a spurious box is a candidate too).
  * face  recall    -- MediaPipe FaceMesh availability. Gates EVERY drowsiness (EAR,
                       PERCLOS) and gaze (yaw/pitch) detection in the product; when
                       the face is not found those detections cannot run at all.

READ THIS BEFORE QUOTING ANY NUMBER FROM HERE
---------------------------------------------
State Farm's camera is mounted high on the PASSENGER side looking ACROSS the cabin.
SensorDyme's recommended mount is dash-centre, DRIVER-FACING. These are therefore
domain-shifted LOWER BOUNDS, not measurements of SensorDyme's operating accuracy --
face availability in particular should be materially higher in the intended geometry.
They bound the signal; they do not certify it. Their value is regression detection:
a change that moves these numbers moved something real, and that is exactly what the
threshold-tuning passes in this repo have historically lacked.

Usage
-----
    python benchmarks/accuracy_state_farm.py                     # default 60/class
    python benchmarks/accuracy_state_farm.py --per-class 150     # tighter CIs
    python benchmarks/accuracy_state_farm.py --sweep             # full PR curve
    python benchmarks/accuracy_state_farm.py --baseline out.json # write a baseline
    python benchmarks/accuracy_state_farm.py --compare out.json  # regression gate
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "state_farm"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

#: State Farm's own class definitions. c1-c4 are the four phone-in-hand poses, so a
#: 'cell phone' detection on them is a true positive by construction. c0 and c9 have
#: no phone, which makes them the honest negative set -- c5-c8 are excluded from the
#: negative set because a phone may legitimately be visible in frame during them.
CLASS_LABELS = {
    "c0": "safe driving",
    "c1": "texting - right",
    "c2": "phone - right",
    "c3": "texting - left",
    "c4": "phone - left",
    "c5": "operating radio",
    "c6": "drinking",
    "c7": "reaching behind",
    "c8": "hair and makeup",
    "c9": "talking to passenger",
}
PHONE_POSITIVE = ("c1", "c2", "c3", "c4")
PHONE_NEGATIVE = ("c0", "c9")

DEFAULT_SWEEP = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)


#: Exit codes, kept distinct ON PURPOSE. A caller has to be able to tell "the gate
#: failed" apart from "the gate could not run", and `raise SystemExit("some string")`
#: cannot: Python exits 1 for any non-integer argument, which is the same code a real
#: regression returns. That collision meant the only way to stop a missing dataset
#: from reddening CI was `continue-on-error`, which would have swallowed genuine
#: regressions too — i.e. disabled the exact thing the job exists to catch.
EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_CANNOT_RUN = 2


class DatasetMissing(SystemExit):
    """Raised when the dataset is absent. Exits EXIT_CANNOT_RUN, never EXIT_REGRESSION."""

    def __init__(self, message: str) -> None:
        print(message, file=sys.stderr)
        super().__init__(EXIT_CANNOT_RUN)


class BackendUnavailable(SystemExit):
    """Requested backend did not answer. Also EXIT_CANNOT_RUN — not a regression."""

    def __init__(self, message: str) -> None:
        print(message, file=sys.stderr)
        super().__init__(EXIT_CANNOT_RUN)


#: Largest fraction of the sample that may be unreadable before the run is
#: declared unable to measure anything. A sample whose images are missing (the
#: manifest survived but data/state_farm/train/ did not) scores every metric as
#: 0.0, which used to be reported as a four-metric REGRESSION (exit 1).
MAX_SKIPPED_FRACTION = 0.05


def sample_problem(scored: dict, requested: int) -> Optional[str]:
    """Why a scored sample cannot support a measurement, or None if it can."""
    scored_n = sum(s["n"] for s in scored["per_class"].values())
    if requested == 0 or scored_n == 0:
        return (
            f"no image in the {requested}-image sample could be read (looked under "
            f"{DATA_ROOT / 'train'}). The manifest exists but the images do not — re-run "
            f"scripts/prepare_state_farm.py. (exit {EXIT_CANNOT_RUN} = could not run, NOT a regression)"
        )
    skipped_fraction = scored["skipped"] / requested
    if skipped_fraction > MAX_SKIPPED_FRACTION:
        return (
            f"{scored['skipped']}/{requested} sampled images ({skipped_fraction:.0%}) could not be "
            f"read — more than the {MAX_SKIPPED_FRACTION:.0%} a comparable measurement tolerates. "
            f"(exit {EXIT_CANNOT_RUN} = could not run, NOT a regression)"
        )
    return None


def load_sample(per_class: int, seed: int, split: str = "val") -> list[dict]:
    """Deterministic per-class sample from the subject-disjoint manifest.

    Sampling per class rather than uniformly keeps every behaviour represented even
    at small n; the seed makes a run reproducible so two runs are comparable.
    """
    manifest = DATA_ROOT / f"{split}_manifest.csv"
    if not manifest.exists():
        raise DatasetMissing(
            f"missing {manifest}.\n"
            "The State Farm dataset is gitignored (~6GB). Prepare it with:\n"
            "    python scripts/prepare_state_farm.py\n"
            f"(exit {EXIT_CANNOT_RUN} = could not run, NOT a regression)"
        )
    rows = list(csv.DictReader(manifest.open()))
    by_class: dict[str, list[dict]] = {}
    for row in rows:
        by_class.setdefault(row["classname"], []).append(row)

    rng = random.Random(seed)  # noqa: S311 - seeded, reproducible sampling; not crypto
    sample: list[dict] = []
    for name in sorted(by_class):
        pool = by_class[name]
        sample.extend(rng.sample(pool, min(per_class, len(pool))))
    return sample


def _image_path(row: dict) -> Path:
    return DATA_ROOT / "train" / row["classname"] / row["img"]


def _load_detector(backend: str = "auto") -> tuple[object, str]:
    """The detector the PRODUCT would load, plus the backend that actually answered.

    This deliberately goes through `core.detectors.get_yolo_model()` rather than
    constructing `ultralytics.YOLO` directly. Constructing YOLO directly is what
    the first version of this harness did, and it silently made the harness
    incapable of measuring the Hailo backend at all -- it would happily report
    "accuracy" for a CPU model on a device whose whole pipeline was running on
    the NPU. A benchmark that cannot see the accelerator is worse than no
    benchmark, because the number looks authoritative.

    backend="auto"  -- honour config.yaml's `accelerator:` block (Hailo if a .hef
                       is configured and the runtime is up; CPU otherwise)
    backend="cpu"   -- force CPU by not configuring the accelerator at all, which
                       is exactly how a HAT-less device behaves

    Returns (model, backend_actually_used) where the backend string is derived
    from the returned object, never from what was requested.
    """
    import yaml

    from core import accel
    from core.detectors import get_yolo_model

    repo = Path(__file__).resolve().parent.parent
    config = yaml.safe_load((repo / "config.yaml").read_text())
    weights = (
        config.get("modules", {}).get("driver_attention", {}).get("yolo_weights", "yolo11n.pt")
    )

    if backend == "cpu":
        accel.configure({"enabled": False})
    else:
        accel.configure(config.get("accelerator", {}))

    try:
        model = get_yolo_model(weights)
    except Exception as exc:  # no backend / weights download failed: cannot run (re-raised)
        # An exception escaping here used to exit 1 with a traceback — the
        # REGRESSION code — for what is a "could not run" condition.
        raise BackendUnavailable(
            f"could not load the detector '{weights}': {exc}\n"
            f"(exit {EXIT_CANNOT_RUN} = could not run, NOT a regression)"
        ) from exc
    actual = "hailo" if type(model).__name__ == "HailoYOLO" else "cpu"
    if backend == "hailo" and actual != "hailo":
        # Refuse rather than silently mislabel. Publishing CPU numbers under a
        # Hailo heading is the specific dishonesty this harness exists to prevent.
        raise BackendUnavailable(
            "--backend hailo was requested but the accelerator did not answer.\n"
            f"  accel.status(): {accel.status()[1]}\n"
            "Refusing to report CPU numbers under a Hailo label."
        )
    return model, actual


def score_sample(sample: list[dict], sweep: tuple[float, ...], quiet: bool = False,
                 backend: str = "auto") -> dict:
    """Run the product's real models over `sample`, recording per-image raw scores.

    Scores are recorded ONCE at a floor confidence and thresholded afterwards, so a
    full sweep costs the same as a single threshold -- which is what makes the PR
    curve cheap enough to run in CI.
    """
    yolo, backend_used = _load_detector(backend)
    names = yolo.names
    try:
        phone_id = next(k for k, v in names.items() if v == "cell phone")
        person_id = next(k for k, v in names.items() if v == "person")
    except StopIteration as exc:
        raise BackendUnavailable(
            "the loaded detector has no 'cell phone'/'person' class — not a COCO detector, "
            f"cannot measure (exit {EXIT_CANNOT_RUN})"
        ) from exc

    try:
        import mediapipe as mp

        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
        )
    except Exception as exc:  # import/graph failure is "cannot run", not a regression (re-raised)
        raise BackendUnavailable(
            f"MediaPipe FaceMesh is unavailable: {exc}\n(exit {EXIT_CANNOT_RUN} = could not run, NOT a regression)"
        ) from exc

    floor = min(sweep)
    per_class: dict[str, dict] = {}
    started = time.perf_counter()
    skipped = 0

    for index, row in enumerate(sample):
        path = _image_path(row)
        image = cv2.imread(str(path))
        if image is None:
            skipped += 1
            continue

        stats = per_class.setdefault(
            row["classname"],
            {"n": 0, "face": 0, "person": 0, "person_conf": [], "phone_scores": [], "multi_person": 0},
        )
        stats["n"] += 1

        if face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).multi_face_landmarks:
            stats["face"] += 1

        people = yolo.predict(image, verbose=False, classes=[person_id], conf=0.25)[0]
        person_boxes = list(people.boxes or [])
        if person_boxes:
            stats["person"] += 1
            stats["person_conf"].append(float(max(b.conf[0] for b in person_boxes)))
            if len(person_boxes) > 1:
                stats["multi_person"] += 1

        phones = yolo.predict(image, verbose=False, classes=[phone_id], conf=floor)[0]
        stats["phone_scores"].append(
            float(max((b.conf[0] for b in (phones.boxes or [])), default=0.0))
        )

        if not quiet and (index + 1) % 100 == 0:
            print(f"  ...{index + 1}/{len(sample)}", file=sys.stderr)

    return {
        "backend": backend_used,
        # The file that actually ran (the .hef on the Hailo path), not the .pt
        # name that is only a cache key there.
        "detector": str(getattr(yolo, "model_name", None) or "unknown"),
        "per_class": per_class,
        "skipped": skipped,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }


def phone_curve(per_class: dict, sweep: tuple[float, ...]) -> list[dict]:
    """Recall / FP-rate / precision at each confidence threshold."""
    positives = [s for c in PHONE_POSITIVE for s in per_class.get(c, {}).get("phone_scores", [])]
    negatives = [s for c in PHONE_NEGATIVE for s in per_class.get(c, {}).get("phone_scores", [])]
    curve = []
    for threshold in sweep:
        true_pos = sum(1 for s in positives if s >= threshold)
        false_pos = sum(1 for s in negatives if s >= threshold)
        recall = true_pos / len(positives) if positives else 0.0
        fp_rate = false_pos / len(negatives) if negatives else 0.0
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision == precision and (precision + recall) > 0
            else 0.0
        )
        curve.append(
            {
                "conf": threshold,
                "recall": round(recall, 4),
                "fp_rate": round(fp_rate, 4),
                "precision": None if precision != precision else round(precision, 4),
                "f1": round(f1, 4),
                "true_pos": true_pos,
                "false_pos": false_pos,
                "n_pos": len(positives),
                "n_neg": len(negatives),
            }
        )
    return curve


def phone_calibration(per_class: dict, edges: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.01)) -> dict:
    """Is the detector's phone score a calibrated probability on labelled frames?

    A phone event's confidence IS this score (core/confidence.py, "model"), so
    this is the calibration measurement that number rests on: among frames whose
    best phone box scored in a band, how often was a phone really present?
    Frames with no phone box at all carry no score and are excluded.

    Two caveats travel with every number here. The observed rate depends on the
    sample's class mix (`base_rate`: 4 phone classes against 2 non-phone ones),
    so it is calibration ON THIS MIX, not a universal correction. And State Farm's
    camera geometry is not SensorDyme's (see the module docstring), so this is a
    measurement to watch, not a mapping to apply on device.
    """
    scored = [(s, True) for c in PHONE_POSITIVE for s in per_class.get(c, {}).get("phone_scores", []) if s > 0]
    scored += [(s, False) for c in PHONE_NEGATIVE for s in per_class.get(c, {}).get("phone_scores", []) if s > 0]
    n_pos = sum(len(per_class.get(c, {}).get("phone_scores", [])) for c in PHONE_POSITIVE)
    n_all = n_pos + sum(len(per_class.get(c, {}).get("phone_scores", [])) for c in PHONE_NEGATIVE)
    bins, ece = [], 0.0
    for lo, hi in itertools.pairwise(edges):
        members = [(s, y) for s, y in scored if lo <= s < hi]
        if not members:
            continue
        mean_score = sum(s for s, _ in members) / len(members)
        rate = sum(1 for _, y in members if y) / len(members)
        ece += len(members) / len(scored) * abs(mean_score - rate)
        bins.append({"band": [lo, min(hi, 1.0)], "n": len(members),
                     "mean_score": round(mean_score, 3), "phone_rate": round(rate, 3)})
    return {
        "bins": bins,
        "ece": round(ece, 3) if scored else None,
        "frames_with_a_score": len(scored),
        "base_rate": round(n_pos / n_all, 3) if n_all else None,
    }


def summarise(scored: dict, sweep: tuple[float, ...], shipped_conf: float) -> dict:
    per_class = scored["per_class"]
    total_n = sum(s["n"] for s in per_class.values())
    total_face = sum(s["face"] for s in per_class.values())
    total_person = sum(s["person"] for s in per_class.values())
    total_multi = sum(s["multi_person"] for s in per_class.values())
    all_person_conf = [c for s in per_class.values() for c in s["person_conf"]]

    curve = phone_curve(per_class, sweep)
    calibration = phone_calibration(per_class)
    at_shipped = min(curve, key=lambda r: abs(r["conf"] - shipped_conf))
    best_f1 = max(curve, key=lambda r: r["f1"])

    positives = [s for c in PHONE_POSITIVE for s in per_class.get(c, {}).get("phone_scores", [])]
    undetectable = sum(1 for s in positives if s == 0.0) / len(positives) if positives else 0.0

    return {
        "images_scored": total_n,
        "images_skipped": scored["skipped"],
        "elapsed_s": scored["elapsed_s"],
        "face_availability": round(total_face / total_n, 4) if total_n else 0.0,
        "person_availability": round(total_person / total_n, 4) if total_n else 0.0,
        "person_mean_conf": round(float(np.mean(all_person_conf)), 4) if all_person_conf else None,
        "multi_person_rate": round(total_multi / total_n, 4) if total_n else 0.0,
        "phone_curve": curve,
        "phone_at_shipped_conf": at_shipped,
        "phone_best_f1": best_f1,
        "phone_undetectable_at_any_conf": round(undetectable, 4),
        "phone_calibration": calibration,
        "per_class": {
            name: {
                "label": CLASS_LABELS.get(name, name),
                "n": s["n"],
                "face_rate": round(s["face"] / s["n"], 4) if s["n"] else 0.0,
                "person_rate": round(s["person"] / s["n"], 4) if s["n"] else 0.0,
            }
            for name, s in sorted(per_class.items())
        },
    }


def print_report(summary: dict, shipped_conf: float) -> None:
    print()
    print("=" * 74)
    print("SensorDyme accuracy baseline -- State Farm val split (subject-disjoint)")
    print("=" * 74)
    print(f"detector backend: {summary.get('backend', '?').upper()}")
    print(
        f"{summary['images_scored']} images scored in {summary['elapsed_s']}s"
        + (f" ({summary['images_skipped']} unreadable)" if summary["images_skipped"] else "")
    )
    print()

    print("SIGNAL AVAILABILITY  (what fraction of frames the signal exists at all)")
    face = summary["face_availability"]
    print(f"  MediaPipe FaceMesh finds a face : {100 * face:5.1f}%")
    print("      gates EVERY drowsiness (EAR/PERCLOS) and gaze (yaw/pitch) detection")
    print(f"  YOLO11n finds a person          : {100 * summary['person_availability']:5.1f}%"
          + (f"   (mean conf {summary['person_mean_conf']:.2f})" if summary["person_mean_conf"] else ""))
    print("      gates presence, zones, stillness, occupancy, baby-box heuristic")
    print(f"  >1 person box on a 1-person img : {100 * summary['multi_person_rate']:5.1f}%")
    print("      spurious boxes feed baby_monitoring's infant identification")
    print()

    print("PER CLASS")
    print(f"  {'class':<26} {'n':>4} {'face':>7} {'person':>8}")
    for name, row in summary["per_class"].items():
        print(
            f"  {name} {row['label']:<22} {row['n']:>4} "
            f"{100 * row['face_rate']:6.1f}% {100 * row['person_rate']:7.1f}%"
        )
    print()

    print("PHONE DETECTION -- operating curve")
    print(f"  {'conf':>6} {'recall':>9} {'FP rate':>9} {'precision':>10} {'F1':>7}")
    for row in summary["phone_curve"]:
        marker = "  <- shipped" if abs(row["conf"] - shipped_conf) < 1e-9 else ""
        precision = "  n/a" if row["precision"] is None else f"{100 * row['precision']:8.1f}%"
        print(
            f"  {row['conf']:>6.2f} {100 * row['recall']:8.1f}% "
            f"{100 * row['fp_rate']:8.1f}% {precision} {row['f1']:>7.3f}{marker}"
        )
    shipped = summary["phone_at_shipped_conf"]
    best = summary["phone_best_f1"]
    print()
    print(f"  at the shipped conf={shipped['conf']:.2f} : recall {100 * shipped['recall']:.1f}%, "
          f"F1 {shipped['f1']:.3f}  (n={shipped['n_pos']} pos / {shipped['n_neg']} neg)")
    print(f"  best F1 at      conf={best['conf']:.2f} : recall {100 * best['recall']:.1f}%, "
          f"F1 {best['f1']:.3f}")
    cal = summary.get("phone_calibration") or {}
    if cal.get("bins"):
        print(f"\n  Phone-score calibration (ECE {cal['ece']}, base rate {cal['base_rate']} on this mix):")
        for b in cal["bins"]:
            print(f"    score {b['band'][0]:.1f}-{b['band'][1]:.1f}: n={b['n']:4d}  mean score {b['mean_score']:.2f}"
                  f"  -> phone present {100 * b['phone_rate']:5.1f}%")
    print(f"  {100 * summary['phone_undetectable_at_any_conf']:.1f}% of phone-present frames yield NO box "
          "at any confidence")
    print("      -- a model ceiling; no threshold change recovers those")
    print()
    print("CAVEAT: State Farm's camera is passenger-side looking across the cabin;")
    print("SensorDyme's mount is dash-centre driver-facing. These are domain-shifted")
    print("LOWER BOUNDS, not SensorDyme's operating accuracy. Use them for regression")
    print("detection, not for certification.")
    print("=" * 74)


def _curve_row(curve: list[dict], conf: float) -> Optional[dict]:
    """The operating-curve row at exactly `conf`, or None if it was not swept."""
    return next((row for row in curve if abs(row["conf"] - conf) < 1e-9), None)


def baseline_conf(payload: dict) -> float:
    """The threshold the baseline's phone recall was measured at."""
    conf = payload.get("shipped_conf")
    if conf is None:
        conf = payload["summary"]["phone_at_shipped_conf"]["conf"]
    return float(conf)


def compare(summary: dict, baseline_path: Path, tolerance: float) -> int:
    """Regression gate: fail if a headline metric dropped by more than `tolerance`.

    Phone recall is compared at ONE threshold — the one the baseline was
    measured at — never "current shipped conf" against "baseline shipped conf".
    Recall only rises as the threshold falls, so comparing across a threshold
    change (baseline 0.35, config now 0.20) let a model that lost 20 points of
    recall at a fixed threshold pass as "improved".

    Returns EXIT_CANNOT_RUN (not a regression) when the two runs cannot be
    compared: the baseline's threshold was not swept, or the baseline records
    a different detector backend than this run used.
    """
    payload = json.loads(baseline_path.read_text())
    baseline = payload["summary"]
    conf = baseline_conf(payload)
    now_row = _curve_row(summary["phone_curve"], conf)
    was_row = _curve_row(baseline.get("phone_curve", []), conf) or baseline["phone_at_shipped_conf"]
    if now_row is None:
        print(
            f"cannot compare: this run did not sweep the baseline's phone threshold {conf:.2f} "
            f"(exit {EXIT_CANNOT_RUN} = could not run, NOT a regression)",
            file=sys.stderr,
        )
        return EXIT_CANNOT_RUN
    was_backend = payload.get("backend") or baseline.get("backend")
    now_backend = summary.get("backend")
    if was_backend and now_backend and was_backend != now_backend:
        print(
            f"cannot compare: the baseline was measured on the {was_backend!r} backend and this run "
            f"used {now_backend!r} — different detectors, not a regression signal. Write a baseline "
            f"for this backend with --baseline. (exit {EXIT_CANNOT_RUN})",
            file=sys.stderr,
        )
        return EXIT_CANNOT_RUN
    if not was_backend:
        print("note: the baseline does not record its backend; assuming it matches this run "
              f"({now_backend!r}).")
    current_conf = summary["phone_at_shipped_conf"]["conf"]
    if abs(current_conf - conf) > 1e-9:
        print(f"note: config now ships conf={current_conf:.2f}; phone recall is compared at the "
              f"baseline's conf={conf:.2f} so a threshold change cannot mask a model change.")
    checks = [
        ("face_availability", summary["face_availability"], baseline["face_availability"]),
        ("person_availability", summary["person_availability"], baseline["person_availability"]),
        (f"phone_recall_at_{conf:.2f}", now_row["recall"], was_row["recall"]),
        ("phone_best_f1", summary["phone_best_f1"]["f1"], baseline["phone_best_f1"]["f1"]),
    ]
    print("\nREGRESSION CHECK vs", baseline_path)
    failed = 0
    for name, now, was in checks:
        delta = now - was
        status = "ok"
        if delta < -tolerance:
            status = "REGRESSION"
            failed += 1
        elif delta > tolerance:
            status = "improved"
        print(f"  {name:<26} {was:.4f} -> {now:.4f}  ({delta:+.4f})  {status}")
    if failed:
        print(f"\n{failed} metric(s) regressed by more than {tolerance}")
    else:
        print(f"\nno metric regressed by more than {tolerance}")
    return EXIT_REGRESSION if failed else EXIT_OK


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--per-class", type=int, default=60,
                        help="images sampled per class (10 classes; default 60 = 600 images)")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed, for reproducibility")
    parser.add_argument("--split", default="val", choices=["val", "train"],
                        help="which manifest to sample (default val -- subject-disjoint)")
    parser.add_argument("--sweep", action="store_true",
                        help="score the full confidence sweep (default: sweep is always computed)")
    parser.add_argument("--shipped-conf", type=float, default=None,
                        help="the threshold to headline; default reads config.yaml")
    parser.add_argument("--baseline", type=Path, help="write results here as a new baseline")
    parser.add_argument("--compare", type=Path, help="compare against a baseline and exit nonzero on regression")
    parser.add_argument("--tolerance", type=float, default=0.03,
                        help="allowed metric drop before --compare fails (default 0.03)")
    parser.add_argument("--backend", default="auto", choices=["auto", "cpu", "hailo"],
                        help="auto = honour config.yaml accelerator block; cpu = force CPU; "
                             "hailo = require the accelerator and refuse to fall back")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    shipped_conf = args.shipped_conf
    if shipped_conf is None:
        # Read the value the product actually ships, so the headline number is always
        # the one the device really uses rather than a benchmark-only constant.
        import yaml
        config = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text())
        shipped_conf = float(
            config.get("modules", {}).get("driver_attention", {}).get("yolo_confidence_threshold", 0.35)
        )

    sweep_points = set(DEFAULT_SWEEP) | {shipped_conf}
    if args.compare and args.compare.exists():
        # compare() measures phone recall at the BASELINE's threshold, so that
        # threshold must be on this run's curve even if config has moved on.
        sweep_points.add(baseline_conf(json.loads(args.compare.read_text())))
    sweep = tuple(sorted(sweep_points))

    sample = load_sample(args.per_class, args.seed, args.split)
    if not args.quiet:
        subjects = sorted({r["subject"] for r in sample})
        print(f"sampling {len(sample)} images ({args.per_class}/class) from the "
              f"{args.split} split; subjects: {', '.join(subjects)}")

    scored = score_sample(sample, sweep, quiet=args.quiet, backend=args.backend)
    problem = sample_problem(scored, len(sample))
    if problem is not None:
        print(problem, file=sys.stderr)
        return EXIT_CANNOT_RUN
    summary = summarise(scored, sweep, shipped_conf)
    summary["backend"] = scored["backend"]
    print_report(summary, shipped_conf)

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": "state_farm",
        "split": args.split,
        "per_class": args.per_class,
        "seed": args.seed,
        "shipped_conf": shipped_conf,
        "backend": scored["backend"],
        "models": {"detector": scored["detector"], "face": "mediapipe face_mesh (refine_landmarks)"},
        "summary": summary,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"accuracy_state_farm_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")

    if args.baseline:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(payload, indent=2))
        print(f"wrote baseline {args.baseline}")

    if args.compare:
        if not args.compare.exists():
            print(f"\nno baseline at {args.compare} — nothing to compare against.\n"
                  f"Write one with --baseline {args.compare}", file=sys.stderr)
            return EXIT_CANNOT_RUN
        return compare(summary, args.compare, args.tolerance)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
