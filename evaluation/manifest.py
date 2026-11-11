"""Dataset manifests: exactly which files a number was measured on, and whether we may.

A manifest is a JSON document listing every file of a dataset version with its
sha256, the subject and session it came from, its split, its recording
conditions and its annotation file, plus the dataset's licence class and (for
our own recordings) the consent protocol and retention date.

Three rules are enforced here rather than remembered:

  * Leakage. Splits are assigned per *subject* (driver, room, household), never
    per file: frames of one person in train and test produce a model that scores
    well and fails on a stranger. A subject or session in two splits, or one
    file (same sha256) twice, is an error.
  * The held-out test split is frozen. ``assign_splits`` only places items that
    have no split yet; it never moves an existing one, so adding data cannot
    silently reshuffle the test set.
  * Licence and consent. Each dataset declares a licence class. Research-only
    data can never feed a shipped model; evaluation-only data can be measured
    on but not trained on. A retention date that has passed is an error: the
    data must be deleted, not evaluated.

    python -m evaluation.manifest verify  --manifest m.json --root /data/ds
    python -m evaluation.manifest build   --root /data/ds --meta meta.csv --name n \
        --version v --licence-class own_collection --licence-id ... --out m.json
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

SCHEMA = 1
SPLITS = ("train", "val", "test")

#: Licence classes, and what each may be used for.
LICENCE_USES = {
    "own_collection": {"training", "evaluation"},  # consented SensorDyme recordings
    "commercial_training": {"training", "evaluation"},  # licence permits training a sold model
    "evaluation_only": {"evaluation"},  # internal measurement only, never weights
    "research_only": set(),  # cannot touch a product model or its release evidence
}

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class Item:
    id: str
    path: str  # relative to the dataset root
    sha256: str
    bytes: int
    subject: str  # the unit of leakage: one driver / room / household
    session: str  # one continuous recording
    split: Optional[str] = None
    conditions: dict = field(default_factory=dict)  # lighting, ir, camera, position, eyewear...
    annotations: Optional[str] = None  # relative path of the annotation JSON
    provenance: dict = field(default_factory=dict)  # per-file source, licence, author, original URL

    @property
    def kind(self) -> str:
        suffix = Path(self.path).suffix.lower()
        return "video" if suffix in VIDEO_SUFFIXES else "image" if suffix in IMAGE_SUFFIXES else "other"


@dataclass
class Manifest:
    name: str
    version: str
    licence: dict  # {"class": ..., "id": ..., "url": ..., "notes": ...}
    items: list[Item]
    consent: dict = field(default_factory=dict)  # {"protocol": ..., "retention_until": "YYYY-MM-DD"}
    split_policy: dict = field(default_factory=dict)
    schema: int = SCHEMA

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def id(self) -> str:
        """Content id: any change to files, splits or metadata changes it."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def split(self, name: str) -> list[Item]:
        return [i for i in self.items if i.split == name]

    def usable_for(self, purpose: str) -> bool:
        return purpose in LICENCE_USES.get(str(self.licence.get("class")), set())


def load(path: str | Path) -> Manifest:
    raw = json.loads(Path(path).read_text())
    items = [Item(**item) for item in raw.pop("items", [])]
    return Manifest(items=items, **raw)


def save(manifest: Manifest, path: str | Path) -> None:
    Path(path).write_text(json.dumps(manifest.to_dict(), indent=1, sort_keys=True) + "\n")


def validate(manifest: Manifest, *, today: Optional[dt.date] = None) -> list[str]:
    """Every reason this manifest must not be used (metadata only; no file I/O)."""
    problems: list[str] = []
    if manifest.schema != SCHEMA:
        problems.append(f"schema {manifest.schema} is not {SCHEMA}")
    lclass = manifest.licence.get("class")
    if lclass not in LICENCE_USES:
        problems.append(f"licence class {lclass!r} is not one of {sorted(LICENCE_USES)}")
    if not manifest.licence.get("id"):
        problems.append("licence.id is required (the exact licence or agreement name)")
    if lclass == "own_collection":
        if not manifest.consent.get("protocol"):
            problems.append("own recordings need consent.protocol (the signed consent form version)")
        if not manifest.consent.get("retention_until"):
            problems.append("own recordings need consent.retention_until")
    retention = manifest.consent.get("retention_until")
    if retention:
        try:
            if dt.date.fromisoformat(retention) < (today or dt.datetime.now(dt.UTC).date()):
                problems.append(f"retention ended {retention}: delete this data, do not evaluate on it")
        except ValueError:
            problems.append(f"consent.retention_until {retention!r} is not YYYY-MM-DD")

    seen_ids: set[str] = set()
    seen_sha: dict[str, str] = {}
    split_of: dict[tuple[str, str], set[str]] = {}
    for item in manifest.items:
        if item.id in seen_ids:
            problems.append(f"duplicate item id {item.id}")
        seen_ids.add(item.id)
        if item.sha256 in seen_sha:
            problems.append(f"{item.id} is the same file as {seen_sha[item.sha256]} (sha256)")
        seen_sha.setdefault(item.sha256, item.id)
        if item.split is not None and item.split not in SPLITS:
            problems.append(f"{item.id}: split {item.split!r} is not one of {SPLITS}")
        if not item.subject or not item.session:
            problems.append(f"{item.id}: subject and session are required (they prevent leakage)")
        if Path(item.path).is_absolute() or ".." in Path(item.path).parts:
            problems.append(f"{item.id}: path must be relative to the dataset root")
        if item.split:
            split_of.setdefault(("subject", item.subject), set()).add(item.split)
            split_of.setdefault(("session", item.session), set()).add(item.split)
    for (kind, key), splits in sorted(split_of.items()):
        if len(splits) > 1:
            problems.append(f"leakage: {kind} {key!r} is in {sorted(splits)}")
    return problems


def _unit(seed: int, group: str) -> float:
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_splits(manifest: Manifest, ratios: dict[str, float], seed: int = 0) -> list[str]:
    """Place every item without a split, by subject; never move an existing one.

    A subject that already has items in a split keeps that split for its new
    items. Returns the ids that were placed.
    """
    if abs(sum(ratios.values()) - 1.0) > 1e-9 or set(ratios) - set(SPLITS):
        raise ValueError(f"ratios must cover only {SPLITS} and sum to 1: {ratios}")
    existing = {i.subject: i.split for i in manifest.items if i.split}
    placed = []
    for item in manifest.items:
        if item.split:
            continue
        split = existing.get(item.subject)
        if split is None:
            u, acc = _unit(seed, item.subject), 0.0
            for name in SPLITS:
                acc += ratios.get(name, 0.0)
                if u < acc:
                    split = name
                    break
            split = split or SPLITS[-1]
            existing[item.subject] = split
        item.split = split
        placed.append(item.id)
    manifest.split_policy = {"group_by": "subject", "seed": seed, "ratios": ratios}
    return placed


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _decodes(path: Path, kind: str) -> Optional[str]:
    import cv2

    if kind == "image":
        return None if cv2.imread(str(path)) is not None else "image does not decode"
    if kind == "video":
        cap = cv2.VideoCapture(str(path))
        try:
            ok, _ = cap.read()
            fps = cap.get(cv2.CAP_PROP_FPS)
        finally:
            cap.release()
        if not ok:
            return "video does not decode"
        if not fps or fps != fps or fps <= 0:
            return "video has no frame rate (replay timestamps would be undefined)"
    return None


def verify_files(manifest: Manifest, root: str | Path, items: Optional[list[Item]] = None,
                 *, decode: bool = True) -> list[str]:
    """Every file present, unchanged (sha256, size) and decodable."""
    root = Path(root)
    problems = []
    for item in manifest.items if items is None else items:
        path = root / item.path
        if not path.is_file():
            problems.append(f"{item.id}: missing {item.path}")
            continue
        if path.stat().st_size != item.bytes or file_sha256(path) != item.sha256:
            problems.append(f"{item.id}: {item.path} changed since the manifest was built")
            continue
        if decode and (why := _decodes(path, item.kind)):
            problems.append(f"{item.id}: {why}")
        if item.annotations and not (root / item.annotations).is_file():
            problems.append(f"{item.id}: missing annotations {item.annotations}")
    return problems


def dhash(gray, size: int = 8) -> int:
    """64-bit difference hash of a grayscale image (near-duplicate detection)."""
    import cv2

    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _middle_frame_hash(path: Path, kind: str) -> Optional[int]:
    import cv2

    if kind == "image":
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return None if img is None else dhash(img)
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2))
        ok, frame = cap.read()
    finally:
        cap.release()
    return dhash(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) if ok else None


def near_duplicates(manifest: Manifest, root: str | Path, max_distance: int = 4) -> list[str]:
    """Pairs of items in DIFFERENT splits that look nearly identical (re-encodes, crops)."""
    root = Path(root)
    hashes = [(i, _middle_frame_hash(root / i.path, i.kind)) for i in manifest.items if i.kind != "other"]
    found = []
    for a in range(len(hashes)):
        for b in range(a + 1, len(hashes)):
            (ia, ha), (ib, hb) = hashes[a], hashes[b]
            if ha is None or hb is None or ia.split == ib.split:
                continue
            if bin(ha ^ hb).count("1") <= max_distance:
                found.append(f"near-duplicate across splits: {ia.id} ({ia.split}) ~ {ib.id} ({ib.split})")
    return found


def build(root: str | Path, meta_rows: list[dict], *, name: str, version: str, licence: dict,
          consent: Optional[dict] = None) -> Manifest:
    """A manifest from metadata rows (id, path, subject, session, [split], [annotations], cond_*)."""
    root = Path(root)
    items = []
    for row in meta_rows:
        path = root / row["path"]
        items.append(Item(
            id=row["id"], path=row["path"], sha256=file_sha256(path), bytes=path.stat().st_size,
            subject=row["subject"], session=row["session"], split=row.get("split") or None,
            conditions={k[5:]: v for k, v in row.items() if k.startswith("cond_") and v},
            annotations=row.get("annotations") or None,
        ))
    return Manifest(name=name, version=version, licence=licence, items=items, consent=consent or {})


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build or verify a dataset manifest")
    sub = parser.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--manifest", required=True)
    v.add_argument("--root", required=True)
    v.add_argument("--no-decode", action="store_true")
    v.add_argument("--near-duplicates", action="store_true")
    b = sub.add_parser("build")
    b.add_argument("--root", required=True)
    b.add_argument("--meta", required=True, help="CSV: id,path,subject,session[,split,annotations,cond_*]")
    b.add_argument("--name", required=True)
    b.add_argument("--version", required=True)
    b.add_argument("--licence-class", required=True, choices=sorted(LICENCE_USES))
    b.add_argument("--licence-id", required=True)
    b.add_argument("--licence-url", default="")
    b.add_argument("--consent-protocol", default="")
    b.add_argument("--retention-until", default="")
    b.add_argument("--split", default="0.7,0.15,0.15", help="train,val,test ratios for unsplit items")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    if args.cmd == "build":
        with open(args.meta, newline="") as f:
            rows = list(csv.DictReader(f))
        consent = {k: v for k, v in (("protocol", args.consent_protocol),
                                     ("retention_until", args.retention_until)) if v}
        manifest = build(args.root, rows, name=args.name, version=args.version, consent=consent,
                         licence={"class": args.licence_class, "id": args.licence_id, "url": args.licence_url})
        ratios: dict[str, float] = dict(zip(SPLITS, (float(x) for x in args.split.split(",")), strict=True))
        assign_splits(manifest, ratios, args.seed)
        problems = validate(manifest)
        save(manifest, args.out)
        print(f"manifest {manifest.name} {manifest.version} id={manifest.id}: {len(manifest.items)} items")
    else:
        manifest = load(args.manifest)
        problems = validate(manifest) + verify_files(manifest, args.root, decode=not args.no_decode)
        if args.near_duplicates:
            problems += near_duplicates(manifest, args.root)
        print(f"manifest {manifest.name} {manifest.version} id={manifest.id}: {len(manifest.items)} items")
    for p in problems:
        print(f"  PROBLEM: {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
