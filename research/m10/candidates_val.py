"""M10 Task 3: every candidate detector through the DEVICE's own inference path, val split, one job.

    python -u -m research.m10.candidates_val --data $SDJOB_DATA/openimages/oi7-sd-v1 \
        --zoo-heads $SDJOB_DATA/models/damoyolo_tinynasL20_T.heads.onnx \
        --exp08-heads81 $SDJOB_DATA/m84/export_smoke/exp08.heads81.onnx \
        --preds $SDJOB_DATA/m10/$SDJOB_ID --out $SDJOB_OUT [--limit N]

What runs is core/accel.py HailoYOLO.predict, unmodified, on the stand-in hailo_platform
(scripts/hw_validation/fake_hailo.py) whose outputs come from FLOAT ONNX heads (as
research/m83/inference_path_val.py and research/m84/device_export.py). Same input (PIL decode,
fitted to a 640 max side, letterboxed by predict), same score floor (0.05), same class-wise NMS
(IoU 0.50), same decoders, for:

  zoo      the shipping weights (Hailo Model Zoo float ONNX 02d7ef7f..., heads graph de27bd7c...),
           COCO-80, decoders legacy (what runs) and reference (opt-in candidate);
  exp08    the EXP-08 device-layout export (research/m84/device_export.py, heads81 a0327a25...):
           person 0 / phone 67 / hand 80 of an 81-channel class head, decoded with the M10
           opt-in class map accelerator.yolo_extra_classes {80: hand}, decoders legacy/reference;
  exp08 stretch (diagnostic): the same EXP-08 heads on DAMO-YOLO's own stretch-resize input
           (640x640, no letterbox), reference decode, boxes mapped back per axis. Separates the
           letterbox-vs-training-preprocessing effect from the decoder.

Evidence level L1 (float on x86). Nothing here is INT8 (L2), a compiled .hef (L3) or a Pi (L4).
Timing printed here is x86 host time of the decode path at the product threshold; NOT a Pi number.
Val split only; the frozen test split is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "hw_validation"))

import numpy as np  # noqa: E402

#: Pinned inputs (M8.3 job 20261113-043857-f182d1; M8.4 job 20261113-090349-6d22e9).
ZOO_HEADS_SHA256 = "de27bd7cab8d391e36d1ac1b84850f88d474cd3307e6b2b990e9b3c49ce850ba"
EXP08_HEADS81_SHA256 = "a0327a25da034e942bb1f6529b7747cb5ddef8c821dc9545ac40ccf394ed7aa1"
FLOOR, PRODUCT_CONF = 0.05, 0.20
HAND_MAP = {80: "hand"}
LABELS = {0: "person", 67: "phone", 80: "hand"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    return round(float(s[min(len(s) - 1, int(q * (len(s) - 1)))]), 4)


def write_boxes(f, image_id: str, boxes, w: int, h: int, sx: float = 1.0, sy: float = 1.0) -> int:
    """Product boxes (frame px) -> scorer JSON lines {id, label, score, box normalised xyxy}."""
    n = 0
    for b in boxes:
        label = LABELS.get(int(b.cls[0]))
        if label is None:
            continue
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
        nb = [x1 * sx / w, y1 * sy / h, x2 * sx / w, y2 * sy / h]
        if nb[2] > nb[0] and nb[3] > nb[1]:
            f.write(json.dumps({"id": image_id, "label": label, "score": float(b.conf[0]), "box": nb}) + "\n")
            n += 1
    return n


def session(path: Path, threads: int):
    import onnxruntime as ort

    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    o.inter_op_num_threads = 1
    s = ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])
    if s.get_providers()[0] != "CPUExecutionProvider":
        raise SystemExit(f"unexpected providers {s.get_providers()}")
    return s


def run_model(name: str, heads_path: Path, extra, items, data: Path, preds: Path, threads: int,
              stretch: bool) -> dict:
    """One network, all images, both decoders (one ONNX run per image, shared by the decoders)."""
    import cv2
    import fake_hailo
    from PIL import Image

    from core import accel
    from core.accel import HailoYOLO
    from research.m83.inference_path_val import fit_640

    sess = session(heads_path, threads)
    base = fake_hailo.onnx_backend(sess)
    memo: dict = {"key": None, "heads": None, "ms": []}

    def backend(batch):
        key = hashlib.sha256(batch.tobytes()).digest()
        if key != memo["key"]:
            t0 = time.perf_counter()
            memo["heads"] = base(batch)
            memo["ms"].append((time.perf_counter() - t0) * 1e3)
            memo["key"] = key
        return {k: v.copy() for k, v in memo["heads"].items()}

    fake_hailo.install(backend)
    accel.record_inference = lambda: None
    model = HailoYOLO(f"{name}.hef (simulated: float heads ONNX)", decoder="legacy", extra_classes=extra)
    names = {k: model.names[k] for k in sorted(LABELS) if k in model.names}
    decode_ms: dict[str, list[float]] = {"legacy": [], "reference": []}
    orig = model._decode_raw

    files = {d: (preds / f"{name}_{d}_val.jsonl").open("w") for d in ("legacy", "reference")}
    if stretch:
        files["reference_stretch"] = (preds / f"{name}_reference_stretch_val.jsonl").open("w")
    counts = dict.fromkeys(files, 0)
    ch80 = {"max": 0.0, "images_with_ch80_ge_0.20": 0}
    t_start = time.perf_counter()
    try:
        for n, item in enumerate(items):
            if n and n % 200 == 0:
                print(f"  {name} {n}/{len(items)} {time.perf_counter() - t_start:.0f}s", flush=True)
            with Image.open(data / item.path) as im:
                fitted = fit_640(im.convert("RGB"))
            rgb = np.asarray(fitted)
            bgr = np.ascontiguousarray(rgb[..., ::-1])
            h, w = bgr.shape[:2]
            for d in ("legacy", "reference"):
                model.decoder = d
                (res,) = model.predict(bgr, conf=FLOOR)
                counts[d] += write_boxes(files[d], item.id, res.boxes, w, h)
            heads = memo["heads"]
            m80 = max(float(a[..., 80].max()) for a in heads.values() if a.shape[-1] == 81)
            ch80["max"] = max(ch80["max"], m80)
            ch80["images_with_ch80_ge_0.20"] += int(m80 >= PRODUCT_CONF)
            # host decode time at the product threshold (x86; NOT a Pi number), on the first 300 images
            if n < 300:
                for d in ("legacy", "reference"):
                    model.decoder = d
                    t0 = time.perf_counter()
                    orig(heads, 1.0, (640 - w) // 2, (640 - h) // 2, w, h, None, PRODUCT_CONF)
                    decode_ms[d].append((time.perf_counter() - t0) * 1e3)
            if stretch:
                # DAMO-YOLO's own eval input: the image stretched to 640x640 (no letterbox), RGB, 0..255
                sq = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
                sheads = base(np.ascontiguousarray(sq[None]))
                model.decoder = "reference"
                (sres,) = orig(sheads, 1.0, 0, 0, 640, 640, None, FLOOR)
                counts["reference_stretch"] += write_boxes(files["reference_stretch"], item.id, sres.boxes,
                                                           640, 640)  # normalised by 640 on both axes
    finally:
        for f in files.values():
            f.close()
        model.close()
        fake_hailo.uninstall()
    return {"heads_onnx": str(heads_path), "heads_onnx_sha256": sha256_file(heads_path), "class_map": extra,
            "names": names, "images": len(items), "boxes_written": counts, "class_channel_80": ch80,
            "wall_s": round(time.perf_counter() - t_start, 1),
            "x86_cpu_not_pi": {
                "onnx_heads_ms_p50": pct(memo["ms"], 0.5), "onnx_heads_ms_p90": pct(memo["ms"], 0.9),
                "threads": threads,
                "decode_ms_at_conf_0.20": {d: {"p50": pct(v, 0.5), "p90": pct(v, 0.9), "n": len(v)}
                                           for d, v in decode_ms.items()}},
            "preds": {k: str(preds / f"{name}_{k}_val.jsonl") for k in files}}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--zoo-heads", type=Path, required=True)
    p.add_argument("--exp08-heads81", type=Path, required=True)
    p.add_argument("--preds", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", default="val", choices=("val",))
    p.add_argument("--limit", type=int)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    a.preds.mkdir(parents=True, exist_ok=True)

    import cv2
    import onnxruntime as ort

    from evaluation import manifest as mf

    for path, want in ((a.zoo_heads, ZOO_HEADS_SHA256), (a.exp08_heads81, EXP08_HEADS81_SHA256)):
        got = sha256_file(path)
        if got != want:
            raise SystemExit(f"{path}: sha256 {got} != pinned {want}")
    manifest = mf.load(a.data / "manifest.json")
    items = sorted(manifest.split(a.split), key=lambda i: i.id)[: a.limit]
    bad = mf.verify_files(manifest, a.data, items, decode=False)
    if bad:
        raise SystemExit(f"{len(bad)} files differ from the manifest: {bad[:3]}")
    report = {
        "evidence_level": "L1 float32 ONNX on x86 CPU through core/accel.py HailoYOLO on fake_hailo; NOT INT8, "
                          "NOT a compiled .hef, NOT a Pi",
        "job": os.environ.get("SDJOB_ID"), "manifest_id": manifest.id, "split": a.split, "images": len(items),
        "floor": FLOOR, "nms_iou": 0.5,
        "versions": {"onnxruntime": ort.__version__, "opencv": cv2.__version__, "numpy": np.__version__,
                     "python": sys.version.split()[0]},
        "core_accel_sha256": sha256_file(ROOT / "core" / "accel.py"),
        "fake_hailo_sha256": sha256_file(ROOT / "scripts" / "hw_validation" / "fake_hailo.py"),
        "models": {},
    }
    report["models"]["zoo"] = run_model("zoo", a.zoo_heads, None, items, a.data, a.preds, a.threads, stretch=False)
    print(json.dumps(report["models"]["zoo"], indent=1), flush=True)
    report["models"]["exp08"] = run_model("exp08", a.exp08_heads81, HAND_MAP, items, a.data, a.preds, a.threads,
                                          stretch=True)
    print(json.dumps(report["models"]["exp08"], indent=1), flush=True)
    for m in report["models"].values():
        m["preds_sha256"] = {k: sha256_file(Path(v)) for k, v in m["preds"].items()}
    (a.out / "candidates_val.json").write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
