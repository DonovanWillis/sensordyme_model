"""A licence-filtered, provenance-recorded Open Images V7 subset for detector research.

    python -m research.openimages.oi all --data $SDJOB_DATA/openimages --out $SDJOB_OUT

Steps (each idempotent; ``all`` runs them in order):
  meta      fetch the official metadata CSVs and record their sha256
  select    choose images deterministically (seeded hash order) from the
            validation and test subsets, whose boxes are all human-drawn and
            exhaustive for every human-verified positive label
  download  fetch the selected images from the CVDF S3 bucket; sha256 every file
  build     validate, de-duplicate, group, split and write the manifest

Rights (verified 2026-11-11 against storage.googleapis.com/openimages/web/factsfigures_v7.html):
annotations are CC BY 4.0 (Google LLC); images are "listed as having a CC BY 2.0
license", with no warranty, and the site says to verify each image's licence.
So only images whose own ``License`` field is exactly CC BY 2.0 are kept, and
every file carries its author, original URL and licence for attribution.
People in these photos are still personal data: internal use only, never
redistributed, no faces published.

The S3 copies are CVDF re-encodes, so the metadata's OriginalMD5 (of the Flickr
original) cannot verify them; the manifest records our own sha256 instead.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import os
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
if os.environ.get("SD_PYLIB"):
    sys.path.insert(0, os.environ["SD_PYLIB"])  # job-private Pillow on the Windows worker

from evaluation import manifest as mf

VERSION = "oi7-sd-v1"
CC_BY_2 = "https://creativecommons.org/licenses/by/2.0/"
S3 = "https://open-images-dataset.s3.amazonaws.com/{subset}/{image_id}.jpg"
META = {
    "validation-bbox.csv": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "test-bbox.csv": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
    "validation-labels.csv": "https://storage.googleapis.com/openimages/v7/oidv7-val-annotations-human-imagelabels.csv",
    "test-labels.csv": "https://storage.googleapis.com/openimages/v7/oidv7-test-annotations-human-imagelabels.csv",
    "validation-images.csv": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
    "test-images.csv": "https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv",
    "hierarchy.json": "https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy.json",
}
SUBSETS = ("validation", "test")

#: Our label -> Open Images MIDs it covers (the class and its descendants in the 600-class hierarchy).
PERSON = "/m/01g317"
PERSON_CHILDREN = ("/m/04yx4", "/m/03bt1vf", "/m/01bl7v", "/m/05r655")  # man, woman, boy, girl
PHONE = "/m/050k8"  # Mobile phone (child of Telephone /m/07cx4)
TELEPHONE = "/m/07cx4"
CORDED_PHONE = "/m/0h8lkj8"
HAND = "/m/0k65p"
LABELS = {"person": (PERSON, *PERSON_CHILDREN), "phone": (PHONE,), "hand": (HAND,)}
#: Verified-label expansion: a positive implies its ancestors, a negative its descendants.
ANCESTORS = {**dict.fromkeys(PERSON_CHILDREN, (PERSON,)), PHONE: (TELEPHONE,), CORDED_PHONE: (TELEPHONE,)}
DESCENDANTS = {PERSON: PERSON_CHILDREN, TELEPHONE: (PHONE, CORDED_PHONE)}
TRACKED = {PERSON, *PERSON_CHILDREN, PHONE, TELEPHONE, CORDED_PHONE, HAND}

#: Selection caps per pool (images). Phones are rare, so every eligible one is kept.
CAPS = {"phone": 4000, "hand": 4500, "person": 3500, "phone_negative": 3000}
MAX_BYTES = 5_500_000_000


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _rank(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def _get(url: str, timeout: float = 120.0, tries: int = 4) -> bytes:
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - fixed https sources
                return r.read()
        except OSError:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


# -- meta ---------------------------------------------------------------------------------

def fetch_meta(data: Path) -> dict:
    meta_dir = data / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    record = {}
    for name, url in META.items():
        path = meta_dir / name
        if not path.exists():
            tmp = path.with_suffix(".part")
            tmp.write_bytes(_get(url, timeout=600))
            tmp.rename(path)
        record[name] = {"url": url, "sha256": _sha256(path), "bytes": path.stat().st_size}
    (meta_dir / "sources.json").write_text(json.dumps(record, indent=1))
    return record


# -- select -------------------------------------------------------------------------------

def _rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def _flag(value: str) -> int:
    return 1 if value.strip() == "1" else 0 if value.strip() == "0" else -1  # -1 = unknown


def select(data: Path, seed: int = 0) -> dict:
    meta_dir = data / "meta"
    stats: Counter = Counter()
    images: dict[str, dict] = {}
    for subset in SUBSETS:
        for row in _rows(meta_dir / f"{subset}-images.csv"):
            stats[f"images_{subset}"] += 1
            if row["License"] != CC_BY_2:
                stats[f"excluded_licence:{row['License'] or 'none'}"] += 1
                continue
            if row.get("Rotation", "").strip() not in ("", "0", "0.0"):
                stats["excluded_rotation"] += 1  # boxes refer to the upright image; avoid ambiguity
                continue
            images[row["ImageID"]] = {
                "id": row["ImageID"], "oi_subset": subset, "license": row["License"],
                "author": row["Author"], "author_profile": row["AuthorProfileURL"],
                "original_url": row["OriginalURL"], "landing_url": row["OriginalLandingURL"],
                "title": row["Title"], "s3_url": S3.format(subset=subset, image_id=row["ImageID"]),
                "boxes": [], "verified_positive": set(), "verified_negative": set(),
            }
    for subset in SUBSETS:
        for row in _rows(meta_dir / f"{subset}-labels.csv"):
            img = images.get(row["ImageID"])
            if img is None or row["LabelName"] not in TRACKED:
                continue
            if row["Confidence"].strip() in ("1", "1.0"):
                img["verified_positive"].update((row["LabelName"], *ANCESTORS.get(row["LabelName"], ())))
            else:
                img["verified_negative"].update((row["LabelName"], *DESCENDANTS.get(row["LabelName"], ())))
        for row in _rows(meta_dir / f"{subset}-bbox.csv"):
            img = images.get(row["ImageID"])
            if img is None or row["LabelName"] not in TRACKED:
                continue
            img["boxes"].append({
                "mid": row["LabelName"], "source": row["Source"],
                "xmin": float(row["XMin"]), "xmax": float(row["XMax"]),
                "ymin": float(row["YMin"]), "ymax": float(row["YMax"]),
                "occluded": _flag(row["IsOccluded"]), "truncated": _flag(row["IsTruncated"]),
                "group_of": _flag(row["IsGroupOf"]), "depiction": _flag(row["IsDepiction"]),
                "inside": _flag(row["IsInside"]),
            })

    def has(img: dict, mids: tuple[str, ...]) -> bool:
        return any(b["mid"] in mids for b in img["boxes"])

    pools = {
        "phone": [i for i in images.values() if has(i, LABELS["phone"])],
        "hand": [i for i in images.values() if has(i, LABELS["hand"])],
        "person": [i for i in images.values() if has(i, LABELS["person"])],
        # verified phone-free images, people present first: the realistic negatives
        "phone_negative": sorted(
            (i for i in images.values() if PHONE in i["verified_negative"] and not has(i, (PHONE,))),
            key=lambda i: (not has(i, LABELS["person"]), _rank(seed, i["id"]))),
    }
    chosen: dict[str, dict] = {}
    for pool, members in pools.items():
        stats[f"pool_{pool}_eligible"] = len(members)
        ordered = members if pool == "phone_negative" else sorted(members, key=lambda i: _rank(seed, i["id"]))
        taken = 0
        for img in ordered:
            if taken >= CAPS[pool]:
                break
            chosen.setdefault(img["id"], img)
            img.setdefault("pools", []).append(pool)
            taken += 1
        stats[f"pool_{pool}_taken"] = taken
    out = []
    for img in sorted(chosen.values(), key=lambda i: i["id"]):
        img["verified_positive"] = sorted(img["verified_positive"])
        img["verified_negative"] = sorted(img["verified_negative"])
        out.append(img)
    (data / "selection.jsonl").write_text("".join(json.dumps(i, sort_keys=True) + "\n" for i in out))
    summary = {"selected": len(out), "seed": seed, "caps": CAPS, "stats": dict(sorted(stats.items()))}
    (data / "selection_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


# -- download -----------------------------------------------------------------------------

def _load_selection(data: Path) -> list[dict]:
    return [json.loads(line) for line in (data / "selection.jsonl").read_text().splitlines() if line]


def download(data: Path, workers: int = 16, max_bytes: int = MAX_BYTES) -> dict:
    image_dir = data / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    selection = _load_selection(data)
    stats: Counter = Counter()
    total = sum(p.stat().st_size for p in image_dir.glob("*.jpg"))

    def fetch(img: dict) -> tuple[str, int, Optional[str]]:
        path = image_dir / f"{img['id']}.jpg"
        if path.exists():
            return img["id"], 0, None
        try:
            body = _get(img["s3_url"], timeout=60)
        except OSError as exc:
            return img["id"], 0, f"{type(exc).__name__}: {exc}"
        tmp = path.with_suffix(".part")
        tmp.write_bytes(body)
        tmp.rename(path)
        return img["id"], len(body), None

    failures = {}
    started = time.monotonic()
    with cf.ThreadPoolExecutor(workers) as pool:
        pending = []
        for img in selection:
            if total >= max_bytes:
                stats["skipped_budget"] += 1
                continue
            pending.append(pool.submit(fetch, img))
            if len(pending) >= workers * 4:  # keep the byte budget honest while streaming
                for fut in pending:
                    image_id, size, err = fut.result()
                    total += size
                    stats["downloaded" if size else ("failed" if err else "present")] += 1
                    if err:
                        failures[image_id] = err
                pending = []
        for fut in pending:
            image_id, size, err = fut.result()
            total += size
            stats["downloaded" if size else ("failed" if err else "present")] += 1
            if err:
                failures[image_id] = err
    summary = {"bytes_on_disk": total, "seconds": round(time.monotonic() - started, 1),
               "stats": dict(stats), "failures": failures}
    (data / "download_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


# -- build --------------------------------------------------------------------------------

def dhash(image, size: int = 8) -> int:
    """64-bit difference hash (PIL image in)."""
    small = image.convert("L").resize((size + 1, size))
    px = list(small.getdata())
    bits = 0
    for row in range(size):
        for col in range(size):
            bits = (bits << 1) | (px[row * (size + 1) + col + 1] > px[row * (size + 1) + col])
    return bits


def near_duplicate_pairs(hashes: list[int], max_distance: int = 4) -> list[tuple[int, int]]:
    """All index pairs whose 64-bit hashes differ in <= max_distance bits (vectorised)."""
    import numpy as np

    if not hashes:
        return []
    arr = np.array(hashes, dtype=np.uint64)
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    pairs = []
    for start in range(0, len(arr), 512):
        block = arr[start:start + 512]
        x = block[:, None] ^ arr[None, :]
        dist = table[x.view(np.uint8).reshape((*x.shape, 8))].sum(axis=2)
        rows, cols = np.nonzero(dist <= max_distance)
        pairs += [(start + int(r), int(c)) for r, c in zip(rows, cols, strict=True) if start + r < c]
    return pairs


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _box_problems(box: dict) -> list[str]:
    problems = []
    if not (0.0 <= box["xmin"] < box["xmax"] <= 1.0 and 0.0 <= box["ymin"] < box["ymax"] <= 1.0):
        problems.append("box outside [0,1] or empty")
    return problems


def build(data: Path, out: Path, seed: int = 0) -> dict:
    from PIL import Image

    selection = _load_selection(data)
    image_dir = data / "images"
    stats: Counter = Counter()
    quality: Counter = Counter()
    kept, hashes = [], []
    by_sha: dict[str, str] = {}
    for img in selection:
        path = image_dir / f"{img['id']}.jpg"
        if not path.exists():
            stats["missing"] += 1
            continue
        try:
            with Image.open(path) as im:
                im.load()
                width, height = im.size
                img_hash = dhash(im)
        except Exception as exc:  # noqa: BLE001 - any decode failure means a corrupt file
            stats["corrupt"] += 1
            img["corrupt"] = repr(exc)
            continue
        sha = _sha256(path)
        if sha in by_sha:
            stats["exact_duplicate"] += 1
            continue
        by_sha[sha] = img["id"]
        boxes_ok = []
        for box in img["boxes"]:
            problems = _box_problems(box)
            if problems:
                quality["box_invalid"] += 1
                continue
            px_w, px_h = (box["xmax"] - box["xmin"]) * width, (box["ymax"] - box["ymin"]) * height
            if px_w < 2 or px_h < 2:
                quality["box_under_2px"] += 1
            boxes_ok.append(box)
        positive = set(img["verified_positive"])
        box_mids = {b["mid"] for b in boxes_ok}
        for mid in (PHONE, HAND, PERSON):
            covered = {mid, *DESCENDANTS.get(mid, ())}
            if mid in positive and not box_mids & covered:
                quality[f"verified_positive_without_box:{mid}"] += 1
            if box_mids & covered and mid not in positive:
                quality[f"box_without_verified_positive:{mid}"] += 1
        img.update(boxes=boxes_ok, width=width, height=height, sha256=sha, bytes=path.stat().st_size)
        kept.append(img)
        hashes.append(img_hash)

    groups = _UnionFind()
    for img in kept:
        groups.union(img["id"], "author:" + (img["author_profile"] or img["author"] or img["id"]))
    pairs = near_duplicate_pairs(hashes)
    for a, b in pairs:
        groups.union(kept[a]["id"], kept[b]["id"])
    stats["near_duplicate_pairs"] = len(pairs)

    items = []
    for img in kept:
        group = groups.find(img["id"])
        group_id = hashlib.sha256(group.encode()).hexdigest()[:12]
        items.append(mf.Item(
            id=img["id"], path=f"images/{img['id']}.jpg", sha256=img["sha256"], bytes=img["bytes"],
            subject=group_id, session=group_id, split=None,
            conditions={"source": "openimages-v7", "oi_subset": img["oi_subset"], "pools": "+".join(img["pools"])},
            provenance={k: img[k] for k in ("license", "author", "author_profile", "original_url",
                                            "landing_url", "title", "s3_url")}
            | {"s3_copy": "CVDF re-encode of the Flickr original; sha256 is of this copy"},
        ))
    manifest = mf.Manifest(
        name="openimages-v7-sd-detection", version=VERSION,
        licence={"class": "commercial_training",
                 "id": "Open Images V7 annotations CC BY 4.0 (Google LLC); images individually CC BY 2.0 "
                       "(filtered on each image's License field; attribution in provenance)",
                 "url": "https://storage.googleapis.com/openimages/web/factsfigures_v7.html",
                 "notes": "No warranty from Google on image licences. Identifiable people: internal use, "
                          "no redistribution, no publication of faces. Counsel review recommended."},
        items=items,
    )
    mf.assign_splits(manifest, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=seed)
    problems = mf.validate(manifest)
    mf.save(manifest, data / "manifest.json")
    split_of = {i.id: i.split for i in manifest.items}
    with (data / "boxes.jsonl").open("w") as f:
        for img in kept:
            f.write(json.dumps({"id": img["id"], "split": split_of[img["id"]], "width": img["width"],
                                "height": img["height"], "boxes": img["boxes"],
                                "verified_positive": img["verified_positive"],
                                "verified_negative": img["verified_negative"]}, sort_keys=True) + "\n")

    distribution: dict = defaultdict(Counter)
    for img in kept:
        split = split_of[img["id"]]
        distribution[split]["images"] += 1
        for name, mids in LABELS.items():
            boxes = [b for b in img["boxes"] if b["mid"] in mids]
            if boxes:
                distribution[split][f"{name}_images"] += 1
                distribution[split][f"{name}_boxes"] += len(boxes)
                distribution[split][f"{name}_group_of"] += sum(b["group_of"] == 1 for b in boxes)
                distribution[split][f"{name}_occluded"] += sum(b["occluded"] == 1 for b in boxes)
                distribution[split][f"{name}_small"] += sum(
                    (b["xmax"] - b["xmin"]) * (b["ymax"] - b["ymin"]) < 0.0025 for b in boxes)
        if PHONE in img["verified_negative"]:
            distribution[split]["phone_verified_negative_images"] += 1
    summary = {
        "manifest_id": manifest.id, "version": VERSION, "items": len(items),
        "bytes": sum(i.bytes for i in items), "groups": len({i.subject for i in items}),
        "stats": dict(stats), "annotation_quality": dict(quality), "manifest_problems": problems,
        "distribution": {k: dict(v) for k, v in sorted(distribution.items())},
        "sha256_list_digest": hashlib.sha256("".join(sorted(i.sha256 for i in items)).encode()).hexdigest(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (data / "build_summary.json").write_text(json.dumps(summary, indent=1))
    for name in ("build_summary.json", "selection_summary.json", "download_summary.json", "manifest.json"):
        if (data / name).exists():
            (out / name).write_bytes((data / name).read_bytes())
    (out / "meta_sources.json").write_bytes((data / "meta" / "sources.json").read_bytes())
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    p.add_argument("step", choices=("meta", "select", "download", "build", "all"))
    p.add_argument("--data", required=True, help="dataset directory on the worker")
    p.add_argument("--out", required=True, help="small artifacts (summaries, manifest)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--max-bytes", type=int, default=MAX_BYTES)
    args = p.parse_args(argv)
    data, out = Path(args.data), Path(args.out)
    data.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    steps = ("meta", "select", "download", "build") if args.step == "all" else (args.step,)
    for step in steps:
        t0 = time.monotonic()
        if step == "meta":
            result = fetch_meta(data)
        elif step == "select":
            result = select(data, args.seed)
        elif step == "download":
            result = download(data, args.workers, args.max_bytes)
        else:
            result = build(data, out, args.seed)
        brief = {k: v for k, v in result.items() if k not in ("failures", "distribution")}
        print(f"[{step}] {time.monotonic() - t0:.1f}s {json.dumps(brief)[:1500]}", flush=True)
        if step == "build":
            print(json.dumps(result["distribution"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
