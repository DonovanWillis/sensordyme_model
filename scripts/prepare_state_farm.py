"""Prepare the State Farm Distracted Driver dataset for training.

Takes the raw competition archive (data/state_farm/imgs.zip) plus the official
driver_imgs_list.csv and produces an ultralytics-ready classification layout:

    data/state_farm/dataset/
        train/c0..c9/*.jpg      (~80% of images)
        val/c0..c9/*.jpg        (~20% of images)

The split is BY DRIVER (subject id), not by image: the dataset contains 26
drivers photographed in bursts, so a random per-image split leaks near-duplicate
frames of the same driver into validation and reports inflated accuracy. Val
subjects are chosen deterministically (greedy fill to ~20% of images over the
sorted subject list), so re-running the script always yields the same split.

Files are hard-linked into the dataset layout when possible (NTFS supports
this), so the extra layout costs no meaningful disk space; falls back to copy.

Usage:
    python scripts/prepare_state_farm.py                  # extract + split
    python scripts/prepare_state_farm.py --val-frac 0.25  # bigger holdout

Train afterwards with (already in requirements.txt via ultralytics):
    yolo classify train data=data/state_farm/dataset model=yolo11s-cls.pt \
        epochs=30 imgsz=224
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

CLASS_NAMES = {
    "c0": "safe driving",
    "c1": "texting - right hand",
    "c2": "talking on phone - right",
    "c3": "texting - left hand",
    "c4": "talking on phone - left",
    "c5": "operating the radio",
    "c6": "drinking",
    "c7": "reaching behind",
    "c8": "hair and makeup",
    "c9": "talking to passenger",
}

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "state_farm"
ZIP_PATH = RAW_DIR / "imgs.zip"
DRIVER_CSV = RAW_DIR / "driver_imgs_list.csv"
TRAIN_DIR = RAW_DIR / "train"  # extracted flat competition layout: train/c0..c9
DATASET_DIR = RAW_DIR / "dataset"  # ultralytics layout: dataset/{train,val}/c0..c9


def extract_train(zip_path: Path, out_dir: Path) -> None:
    """Extract only the labeled train/ images from the competition zip.

    The unlabeled test/ set (~79k images) is skipped: it has no ground truth and
    is useless for supervised training. Handles both `train/...` and
    `imgs/train/...` archive layouts.
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        prefix = next(
            (p for p in ("train/", "imgs/train/") if any(n.startswith(p) for n in names)),
            None,
        )
        if prefix is None:
            raise SystemExit(f"no train/ folder found inside {zip_path}")
        members = [n for n in names if n.startswith(prefix) and n.lower().endswith(".jpg")]
        print(f"extracting {len(members)} train images from {zip_path.name} ...")
        for i, name in enumerate(members, 1):
            # name looks like ".../train/c0/img_44733.jpg" -> keep "c0/img_44733.jpg"
            rel = Path(name[len(prefix):])
            dest = out_dir / rel
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
            if i % 2000 == 0:
                print(f"  {i}/{len(members)}")
    print("extraction done")


def load_driver_list(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def pick_val_subjects(rows: list[dict], val_frac: float) -> set[str]:
    """Deterministically pick whole subjects until ~val_frac of images are held out."""
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["subject"]] += 1
    total = sum(counts.values())
    target = total * val_frac
    val_subjects: set[str] = set()
    held = 0
    for subject in sorted(counts):
        if held >= target:
            break
        val_subjects.add(subject)
        held += counts[subject]
    print(
        f"val subjects: {sorted(val_subjects)} "
        f"({held}/{total} images, {held / total:.1%})"
    )
    return val_subjects


def link_or_copy(src: Path, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def build_dataset(rows: list[dict], val_subjects: set[str]) -> None:
    manifests = {"train": [], "val": []}
    missing = 0
    for row in rows:
        split = "val" if row["subject"] in val_subjects else "train"
        src = TRAIN_DIR / row["classname"] / row["img"]
        if not src.exists():
            missing += 1
            continue
        link_or_copy(src, DATASET_DIR / split / row["classname"] / row["img"])
        manifests[split].append(row)
    if missing:
        print(f"WARNING: {missing} images listed in driver_imgs_list.csv were not found")
    for split, split_rows in manifests.items():
        manifest = RAW_DIR / f"{split}_manifest.csv"
        with open(manifest, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["subject", "classname", "img"])
            writer.writeheader()
            writer.writerows(split_rows)
        print(f"{split}: {len(split_rows)} images -> {DATASET_DIR / split} (manifest: {manifest.name})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--val-frac", type=float, default=0.2, help="fraction of images to hold out by driver")
    args = parser.parse_args()

    if not DRIVER_CSV.exists():
        raise SystemExit(f"missing {DRIVER_CSV}")
    if not TRAIN_DIR.exists():
        if not ZIP_PATH.exists():
            raise SystemExit(f"missing both {TRAIN_DIR} and {ZIP_PATH}")
        extract_train(ZIP_PATH, TRAIN_DIR)

    rows = load_driver_list(DRIVER_CSV)
    val_subjects = pick_val_subjects(rows, args.val_frac)
    build_dataset(rows, val_subjects)
    print("\nclasses:")
    for code, name in CLASS_NAMES.items():
        print(f"  {code}: {name}")
    print(
        "\nready to train, e.g.:\n"
        "  yolo classify train data=data/state_farm/dataset model=yolo11s-cls.pt epochs=30 imgsz=224"
    )


if __name__ == "__main__":
    main()
