"""Subject-separated splits that cannot leak, and a locked held-out test split.

Split unit: the SUBJECT, never a clip or a frame. Subjects are further linked
whenever they share something that would let a model recognise the test set
rather than generalise to it, and linked subjects always land in one split:

  * a session (one continuous recording);
  * a vehicle-day (same car, same day: same mount, light, cabin, background);
  * a clip (a passenger in someone else's clip is a person the model has seen).

``check`` refuses a manifest where any subject, session, vehicle-day or
co-occupant crosses splits, where own recordings lack the metadata to prove
otherwise, or where near-duplicate FRAMES (dHash over several frames per video,
reusing ``manifest.dhash``) appear in two splits. ``cross_manifest`` does the
same between a training manifest and an evaluation manifest.

The held-out test split is LOCKED: ``write_lock`` stamps every assignment and
the test membership with a sha256 over the canonical content. ``verify_lock``
refuses any later manifest whose test set differs or whose items moved.
Scoring the test split needs ``authorize_test_access`` with an explicit flag,
and every access is appended to ``<lock>.access.jsonl`` (so the number of looks
at the test set is on the record).

    python -m evaluation.leakage check --manifest m.json [--root DIR --frames 5]
    python -m evaluation.leakage assign --manifest m.json --split 0.7,0.15,0.15 --seed 0 --out m2.json
    python -m evaluation.leakage lock --manifest m.json --out split_lock.json
    python -m evaluation.leakage verify --manifest m.json --lock split_lock.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

from evaluation import manifest as mf

LOCK_SCHEMA = "sensordyme.eval.split_lock"
LOCK_VERSION = 1


class LeakageError(ValueError):
    """The split cannot be made (or kept) leak-free."""


class TestAccessRefusedError(PermissionError):
    """The held-out test split was requested without the explicit, recorded flag."""


# -- per-item metadata ---------------------------------------------------------------------

def _list(value) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return [v.strip() for v in str(value or "").split(";") if v.strip()]


def people(item: mf.Item) -> list[str]:
    """Everyone visible in the clip: its subject plus the listed occupants."""
    return [item.subject, *[p for p in _list(item.conditions.get("occupants")) if p != item.subject]]


def vehicle_day(item: mf.Item) -> Optional[str]:
    vehicle, day = item.conditions.get("vehicle"), item.conditions.get("recorded_on")
    return f"{vehicle}@{day}" if vehicle and day else None


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


def subject_groups(manifest: mf.Manifest) -> dict[str, str]:
    """subject -> group id: subjects linked by a shared session, vehicle-day or clip."""
    uf = _UnionFind()
    by_key: dict[str, list[str]] = defaultdict(list)
    for item in manifest.items:
        persons = people(item)
        for p in persons:
            uf.find(p)
        for p in persons[1:]:
            uf.union(persons[0], p)
        by_key[f"session:{item.session}"].append(item.subject)
        if (vd := vehicle_day(item)) is not None:
            by_key[f"vehicle_day:{vd}"].append(item.subject)
    for subjects in by_key.values():
        for s in subjects[1:]:
            uf.union(subjects[0], s)
    return {s: uf.find(s) for s in uf.parent}


def assign_subject_splits(manifest: mf.Manifest, ratios: dict[str, float], seed: int = 0) -> list[str]:
    """Place unsplit items by linked subject GROUP; never move an existing item.

    A group with items already in a split keeps that split for its new items; a
    group already spread over two splits cannot be repaired by placement and
    raises LeakageError. Returns the ids placed.
    """
    if abs(sum(ratios.values()) - 1.0) > 1e-9 or set(ratios) - set(mf.SPLITS):
        raise ValueError(f"ratios must cover only {mf.SPLITS} and sum to 1: {ratios}")
    group_of = subject_groups(manifest)
    existing: dict[str, set[str]] = defaultdict(set)
    for item in manifest.items:
        if item.split:
            for p in people(item):
                existing[group_of[p]].add(item.split)
    if spread := sorted(g for g, s in existing.items() if len(s) > 1):
        raise LeakageError(f"linked subject groups already span splits: {spread}")
    chosen = {g: next(iter(s)) for g, s in existing.items()}
    placed = []
    for item in manifest.items:
        if item.split:
            continue
        group = group_of[item.subject]
        if group not in chosen:
            u, acc, pick = mf._unit(seed, group), 0.0, None  # the same hashing as manifest.assign_splits
            for name in mf.SPLITS:
                acc += ratios.get(name, 0.0)
                if u < acc:
                    pick = name
                    break
            chosen[group] = pick or mf.SPLITS[-1]
        item.split = chosen[group]
        placed.append(item.id)
    manifest.split_policy = {"group_by": "subject+session+vehicle_day+co_occupant", "seed": seed,
                             "ratios": ratios, "assigner": "evaluation.leakage.assign_subject_splits"}
    return placed


# -- checks ----------------------------------------------------------------------------------

def metadata_problems(manifest: mf.Manifest) -> list[str]:
    """Leakage visible from metadata alone (no file I/O)."""
    problems = []
    split_of: dict[tuple[str, str], set[str]] = defaultdict(set)
    own = manifest.licence.get("class") == "own_collection"
    for item in manifest.items:
        if not item.split:
            problems.append(f"{item.id}: no split assigned")
            continue
        if own:
            missing = [k for k in ("vehicle", "recorded_on") if not item.conditions.get(k)]
            if missing:
                problems.append(f"{item.id}: own recordings need conditions {missing} "
                                f"(vehicle-day leakage cannot be ruled out without them)")
        for p in people(item):
            split_of[("person", p)].add(item.split)
        split_of[("session", item.session)].add(item.split)
        if (vd := vehicle_day(item)) is not None:
            split_of[("vehicle-day", vd)].add(item.split)
    for (kind, key), splits in sorted(split_of.items()):
        if len(splits) > 1:
            problems.append(f"leakage: {kind} {key!r} is in {sorted(splits)}")
    return problems


def frame_hashes(path: Path, kind: str, frames: int) -> list[int]:
    """dHash of ``frames`` evenly spaced frames (one for an image)."""
    import cv2

    if kind == "image":
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return [] if img is None else [mf.dhash(img)]
    cap = cv2.VideoCapture(str(path))
    out = []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return []
        n = max(1, min(frames, total))
        for k in range(n):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((k + 0.5) * total / n))
            ok, frame = cap.read()
            if ok:
                out.append(mf.dhash(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
    finally:
        cap.release()
    return out


def _is_uniform(h: int) -> bool:
    """An all-equal or all-increasing hash (a blank or flat frame) matches anything blank: not evidence."""
    return h in (0, (1 << 64) - 1)


def near_duplicate_frames(manifest: mf.Manifest, root: str | Path, *, frames: int = 5,
                          max_distance: int = 4) -> list[str]:
    """Items in DIFFERENT splits sharing a near-identical frame (re-cuts, re-encodes, overlapping clips)."""
    root = Path(root)
    hashed = [(i, [h for h in frame_hashes(root / i.path, i.kind, frames) if not _is_uniform(h)])
              for i in manifest.items if i.kind != "other"]
    found = []
    for a in range(len(hashed)):
        for b in range(a + 1, len(hashed)):
            (ia, ha), (ib, hb) = hashed[a], hashed[b]
            if ia.split == ib.split:
                continue
            best = min((bin(x ^ y).count("1") for x in ha for y in hb), default=None)
            if best is not None and best <= max_distance:
                found.append(f"near-duplicate frames across splits: {ia.id} ({ia.split}) ~ {ib.id} ({ib.split}),"
                             f" dHash distance {best}")
    return found


def check(manifest: mf.Manifest, *, root: Optional[str | Path] = None, frames: int = 0,
          max_distance: int = 4) -> list[str]:
    """Every reason this manifest's splits could leak (metadata; plus frames when ``root`` is given)."""
    problems = metadata_problems(manifest)
    if root is not None and frames > 0:
        problems += near_duplicate_frames(manifest, root, frames=frames, max_distance=max_distance)
    return problems


def cross_manifest(training: mf.Manifest, evaluation: mf.Manifest,
                   splits: tuple[str, ...] = ("val", "test")) -> list[str]:
    """A training manifest must share no person, session, vehicle-day or file with eval splits."""
    held = [i for i in evaluation.items if i.split in splits]
    keys: dict[str, set[str]] = defaultdict(set)
    for item in held:
        keys["person"].update(people(item))
        keys["session"].add(item.session)
        keys["file"].add(item.sha256)
        if (vd := vehicle_day(item)) is not None:
            keys["vehicle-day"].add(vd)
    problems = []
    for item in training.items:
        for kind, values in (("person", people(item)), ("session", [item.session]), ("file", [item.sha256]),
                             ("vehicle-day", [vd] if (vd := vehicle_day(item)) else [])):
            for v in values:
                if v in keys[kind]:
                    problems.append(f"leakage: training item {item.id} shares {kind} {v!r} "
                                    f"with {evaluation.name} {'/'.join(splits)}")
    return sorted(set(problems))


# -- the test-split lock ---------------------------------------------------------------------

def _canonical_hash(doc: dict) -> str:
    body = {k: v for k, v in doc.items() if k != "lock_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def lock_document(manifest: mf.Manifest, *, created_at: Optional[str] = None) -> dict:
    test = manifest.split("test")
    doc = {
        "schema": LOCK_SCHEMA, "version": LOCK_VERSION,
        "manifest": {"name": manifest.name, "version": manifest.version, "id_at_lock": manifest.id},
        "created_at": created_at or dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "split_policy": manifest.split_policy,
        "assignments": {i.id: [i.split, i.sha256] for i in sorted(manifest.items, key=lambda i: i.id)},
        "test": {"items": sorted(i.id for i in test),
                 "people": sorted({p for i in test for p in people(i)}),
                 "sessions": sorted({i.session for i in test}),
                 "vehicle_days": sorted({vd for i in test if (vd := vehicle_day(i))})},
    }
    doc["lock_sha256"] = _canonical_hash(doc)
    return doc


def write_lock(manifest: mf.Manifest, path: str | Path, *, created_at: Optional[str] = None) -> str:
    """Lock the splits of a leak-free manifest. Never overwrites a lock. Returns the lock hash."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists: a split lock is written once (a new lock is a new test set)")
    problems = mf.validate(manifest) + metadata_problems(manifest)
    if not manifest.split("test"):
        problems.append("no test items: nothing to hold out")
    if problems:
        raise LeakageError("; ".join(problems))
    doc = lock_document(manifest, created_at=created_at)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return doc["lock_sha256"]


def load_lock(path: str | Path) -> tuple[Optional[dict], list[str]]:
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        return None, [f"split lock {path}: {exc}"]
    if doc.get("schema") != LOCK_SCHEMA or doc.get("version") != LOCK_VERSION:
        return None, [f"split lock {path}: not a {LOCK_SCHEMA} v{LOCK_VERSION} document"]
    if doc.get("lock_sha256") != _canonical_hash(doc):
        return None, [f"split lock {path}: content does not match its lock_sha256 (edited after locking)"]
    return doc, []


def verify_lock(manifest: mf.Manifest, lock: dict) -> tuple[list[str], list[str]]:
    """(problems, notes). Items may be REMOVED (consent withdrawn); nothing may move or join test."""
    problems, notes = [], []
    if lock["manifest"]["name"] != manifest.name:
        problems.append(f"lock is for dataset {lock['manifest']['name']!r}, not {manifest.name!r}")
    current = {i.id: i for i in manifest.items}
    for item_id, (split, sha) in sorted(lock["assignments"].items()):
        item = current.get(item_id)
        if item is None:
            notes.append(f"{item_id} ({split}) removed since the lock (withdrawal or retention)")
            continue
        if item.split != split:
            problems.append(f"{item_id} moved from {split} to {item.split} after the lock")
        if item.sha256 != sha:
            problems.append(f"{item_id} changed (sha256) after the lock")
    locked_test = set(lock["test"]["items"])
    locked_people, locked_vd = set(lock["test"]["people"]), set(lock["test"]["vehicle_days"])
    for item in manifest.items:
        if item.id in lock["assignments"]:
            continue
        if item.split == "test":
            problems.append(f"{item.id} joined the locked test split (a new test set needs a new lock)")
        elif set(people(item)) & locked_people or (vehicle_day(item) in locked_vd):
            problems.append(f"{item.id} ({item.split}) shares a person or vehicle-day with the locked test split")
        else:
            notes.append(f"{item.id} added to {item.split} after the lock")
    removed_test = locked_test - set(current)
    if removed_test:
        notes.append(f"{len(removed_test)} locked test item(s) removed: the test set is smaller than locked")
    return problems, notes


def authorize_test_access(lock_path: str | Path, *, frozen_test: bool, purpose: str, commit: str,
                          config_sha256: str, now: Optional[str] = None) -> dict:
    """Allow one scoring of the locked test split, and append it to the access log."""
    if not frozen_test:
        raise TestAccessRefusedError("the test split is locked: pass --frozen-test for a final, untuned "
                                     "measurement (the access is recorded)")
    if not purpose.strip():
        raise TestAccessRefusedError("a test-split access needs a stated purpose (--purpose)")
    lock, problems = load_lock(lock_path)
    if lock is None:
        raise TestAccessRefusedError("; ".join(problems))
    log = Path(str(lock_path) + ".access.jsonl")
    previous = len(log.read_text().splitlines()) if log.exists() else 0
    record = {"at": now or dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
              "lock_sha256": lock["lock_sha256"], "commit": commit, "config_sha256": config_sha256,
              "purpose": purpose, "access_number": previous + 1}
    with log.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Leak-free subject splits and the test-split lock")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--manifest", required=True)
    c.add_argument("--root")
    c.add_argument("--frames", type=int, default=5, help="frames per video for near-duplicate dHash")
    c.add_argument("--training-manifest", action="append", default=[])
    a = sub.add_parser("assign")
    a.add_argument("--manifest", required=True)
    a.add_argument("--split", default="0.7,0.15,0.15")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", required=True)
    lk = sub.add_parser("lock")
    lk.add_argument("--manifest", required=True)
    lk.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--manifest", required=True)
    v.add_argument("--lock", required=True)
    args = p.parse_args(argv)
    manifest = mf.load(args.manifest)
    problems: list[str] = []
    if args.cmd == "check":
        problems = mf.validate(manifest) + check(manifest, root=args.root, frames=args.frames if args.root else 0)
        for path in args.training_manifest:
            problems += cross_manifest(mf.load(path), manifest)
    elif args.cmd == "assign":
        ratios: dict[str, float] = dict(zip(mf.SPLITS, (float(x) for x in args.split.split(",")), strict=True))
        try:
            placed = assign_subject_splits(manifest, ratios, args.seed)
        except LeakageError as exc:
            problems = [str(exc)]
        else:
            mf.save(manifest, args.out)
            print(f"placed {len(placed)} items; manifest id={manifest.id}")
    elif args.cmd == "lock":
        try:
            print(f"split lock {write_lock(manifest, args.out)}")
        except (LeakageError, FileExistsError) as exc:
            problems = [str(exc)]
    else:
        lock, problems = load_lock(args.lock)
        if lock is not None:
            problems, notes = verify_lock(manifest, lock)
            for n in notes:
                print(f"  note: {n}")
    for prob in problems:
        print(f"  PROBLEM: {prob}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
