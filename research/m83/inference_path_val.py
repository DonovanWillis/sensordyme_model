"""M8.3: the DEVICE inference path on the float ONNX of the shipping model, Open Images val.

    python -u -m research.m83.inference_path_val --data $SDJOB_DATA/openimages/oi7-sd-v1 \
        --onnx $SDJOB_DATA/models/damoyolo_tinynasL20_T.onnx \
        --heads-onnx $SDJOB_DATA/models/damoyolo_tinynasL20_T.heads.onnx \
        --preds $SDJOB_DATA/m83_inference_path --out $SDJOB_OUT

What runs is core/accel.py HailoYOLO.predict itself, unmodified: a stand-in
hailo_platform (scripts/hw_validation/fake_hailo.py) whose .hef has the parse-hef
layout (six NHWC float32 outputs, 80/40/20, 68 + 81 channels) answers each
`pipeline.infer` by running the float ONNX heads on the exact uint8 tensor the
device code built. So letterbox, BGR->RGB, head pairing, decode, NMS, class
mapping and the map back to frame pixels are the product's; only the chip is
replaced by the float model it was compiled from. No quantisation is simulated:
INT8 parity is a Pi measurement (scripts/hw_validation/hwv_stage.py capture/parity).

Per val image (PIL decode, fitted to a 640 max side with PIL bilinear -- the same
input the Mission 4/5 bench used -- then handed to predict as a BGR frame):

  A  input parity     the uint8 tensor predict() sent == an independently built
                      letterbox (numpy, no core import), bit for bit
  B  decode parity    predict()'s boxes (frame px) == hwv_decode LEGACY + unmap on
                      the same heads (independent numpy decoder)
  C  candidate parity the opt-in reference decoder == hwv_decode with the
                      candidate's semantics (offset 0, multilabel, IoU 0.5, cap 500)
  D  accuracy         person/phone AP of the device path (bench.score), to set
                      beside the Mission 5 numbers for the same weights/decoder
  E  channel order    the same with R and B swapped at the chip (a deliberate
                      fault): if RGB is right, AP must drop
  F  native-res input the original full-resolution image straight into predict()
                      (cv2.INTER_LINEAR downscale, no antialias): what the device
                      would do with a higher-resolution camera mode
  G  timing           x86 CPU per-stage time of predict()'s host work (preprocess,
                      decode) at the product threshold 0.20; NOT a Pi number

Plus the known-good expectations for the Pi stages: for every image in
scripts/hw_validation/known_good/ the exact input tensor sha256, per-head value
statistics and the float detections at conf 0.05 (legacy decoder).

Val split only (the frozen test split is refused). No training. Model files are
verified by sha256.
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

ONNX_SHA256 = "02d7ef7f35da6abfecdf2e04505b560451eb2dd9eff1cf335ab8e5370c126657"  # research/m8/coco80_preds.py
HEAD_NODES = ("Mul_224", "Sigmoid_244", "Mul_264", "Sigmoid_284", "Mul_304", "Sigmoid_324")
FLOOR, PRODUCT_CONF = 0.05, 0.20
COCO_TO_LABEL = {0: "person", 67: "phone"}


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


def independent_letterbox(rgb: np.ndarray) -> np.ndarray:
    """(1, 640, 640, 3) uint8 for an RGB image whose longer side is <= 640 (scale 1). No core import."""
    h, w = rgb.shape[:2]
    if max(h, w) != 640:
        raise ValueError("independent_letterbox expects an image already fitted to a 640 max side")
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    top, left = (640 - h) // 2, (640 - w) // 2
    canvas[top:top + h, left:left + w] = rgb
    return canvas[None]


def fit_640(image):
    """PIL image -> PIL image with longer side 640 (bilinear), as research/detector_bench letterbox does."""
    from PIL import Image

    w, h = image.size
    s = 640 / max(w, h)
    new = (round(w * s), round(h * s))
    return image if new == (w, h) else image.resize(new, Image.Resampling.BILINEAR)


def rows_of(result) -> np.ndarray:
    b = result.boxes
    if not b:
        return np.zeros((0, 6))
    rows = np.array([[float(x.conf[0]), int(x.cls[0]), *map(float, x.xyxy[0])] for x in b])
    return rows[np.lexsort((rows[:, 2], rows[:, 1], -rows[:, 0]))]


def rows_of_dets(d) -> np.ndarray:
    if len(d) == 0:
        return np.zeros((0, 6))
    rows = np.column_stack([d.scores, d.classes, d.boxes])
    return rows[np.lexsort((rows[:, 2], rows[:, 1], -rows[:, 0]))]


def parity(a: np.ndarray, b: np.ndarray) -> tuple[bool, float]:
    if a.shape != b.shape:
        return False, float("inf")
    if len(a) == 0:
        return True, 0.0
    same_cls = bool((a[:, 1] == b[:, 1]).all())
    diff = float(np.abs(a - b).max())
    return same_cls and diff <= 1e-3, diff


def to_jsonl(f, image_id, rows, w, h):
    for score, cls, x1, y1, x2, y2 in rows:
        label = COCO_TO_LABEL.get(int(cls))
        if label is None:
            continue
        nb = [x1 / w, y1 / h, x2 / w, y2 / h]
        if nb[2] > nb[0] and nb[3] > nb[1]:
            f.write(json.dumps({"id": image_id, "label": label, "score": float(score), "box": nb}) + "\n")


def build_session(heads_onnx: Path, threads: int):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(heads_onnx), opts, providers=["CPUExecutionProvider"])
    return sess


def known_good(model, fake_hailo, hd, out: Path, captured: dict) -> dict:
    """Float expectations for scripts/hw_validation/known_good (device path, legacy, conf 0.05)."""
    import cv2

    kg = ROOT / "scripts" / "hw_validation" / "known_good"
    sources = json.loads((kg / "SOURCES.json").read_text())
    result = {"format": "sensordyme-known-good-float/1", "conf": FLOOR, "decoder": "legacy", "images": []}
    for row in sources["images"]:
        path = kg / row["file"]
        if sha256_file(path) != row["sha256"]:
            raise SystemExit(f"{path} does not match SOURCES.json")
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        (res,) = model.predict(frame, conf=FLOOR)
        batch, heads = captured["batch"], captured["heads"]
        stats = {}
        for name, arr in heads.items():
            a = arr.astype(np.float64)
            entry = {"shape": list(arr.shape), "min": float(a.min()), "max": float(a.max()),
                     "mean": float(a.mean()), "std": float(a.std())}
            if arr.shape[-1] == 81:
                flat = a[0].reshape(-1, 81)[:, :80]
                top = np.argsort(-flat.max(axis=0))[:5]
                entry["top_classes"] = [[int(c), round(float(flat[:, c].max()), 5)] for c in top]
            stats[name] = entry
        dets = [{"cls": int(b.cls[0]), "name": model.names[int(b.cls[0])], "score": round(float(b.conf[0]), 5),
                 "xyxy": [round(float(v), 2) for v in b.xyxy[0]]} for b in res.boxes]
        result["images"].append({"file": row["file"], "sha256": row["sha256"],
                                 "input_tensor_sha256": hashlib.sha256(batch.tobytes()).hexdigest(),
                                 "heads": stats, "detections": dets})
        print(f"  known_good {row['file']}: {len(dets)} dets >= {FLOOR}; top "
              f"{[(d['name'], d['score']) for d in dets[:3]]}", flush=True)
    (out / "known_good_expected_float.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--heads-onnx", type=Path, required=True)
    p.add_argument("--preds", type=Path, required=True, help="large outputs (on the worker's data volume)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", default="val", choices=("val",))
    p.add_argument("--limit", type=int)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--skip", default="", help="comma list of E,F to skip")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    args.preds.mkdir(parents=True, exist_ok=True)
    skip = {s.strip().upper() for s in args.skip.split(",") if s.strip()}

    import cv2
    import fake_hailo
    import hwv_decode as hd
    import onnxruntime as ort
    from PIL import Image

    from core import accel
    from core.accel import HailoYOLO
    from evaluation import manifest as mf
    from research.detector_bench import bench

    got = sha256_file(args.onnx)
    if got != ONNX_SHA256:
        raise SystemExit(f"{args.onnx}: sha256 {got} != pinned {ONNX_SHA256}")
    sess = build_session(args.heads_onnx, args.threads)
    out_meta = [(o.name, o.shape) for o in sess.get_outputs()]
    provenance = {
        "job": os.environ.get("SDJOB_ID"), "onnx_sha256": got, "heads_onnx_sha256": sha256_file(args.heads_onnx),
        "heads_onnx_outputs": out_meta, "providers": sess.get_providers(), "threads": args.threads,
        "versions": {"onnxruntime": ort.__version__, "opencv": cv2.__version__, "numpy": np.__version__,
                     "python": sys.version.split()[0]},
        "core_accel_sha256": sha256_file(ROOT / "core" / "accel.py"),
        "fake_hailo_sha256": sha256_file(ROOT / "scripts" / "hw_validation" / "fake_hailo.py"),
        "hwv_decode_sha256": sha256_file(ROOT / "scripts" / "hw_validation" / "hwv_decode.py"),
    }
    # The ONNX head shapes on a real input: NCHW? 80 or 81 class channels?
    probe = sess.run(None, {sess.get_inputs()[0].name: np.zeros((1, 3, 640, 640), np.float32)})
    provenance["heads_onnx_runtime_shapes"] = [list(a.shape) for a in probe]
    print(json.dumps(provenance, indent=1), flush=True)

    captured: dict = {}
    base = fake_hailo.onnx_backend(sess)
    swapped = fake_hailo.onnx_backend(sess, swap_rb=True)
    mode = {"swap": False}

    def backend(batch):
        heads = (swapped if mode["swap"] else base)(batch)
        captured["batch"], captured["heads"] = batch, heads
        return heads

    fake_hailo.install(backend)
    accel.record_inference = lambda: None  # no health bookkeeping in a batch job
    model = HailoYOLO("damoyolo_tinynasL20_T.hef (simulated: float ONNX)", decoder="legacy")
    scales_hd = hd.group_heads(dict(fake_hailo.OUTPUTS), 640)
    candidate = hd.DecoderSpec("candidate", anchor_offset=0.0, multilabel=True, nms_iou=0.5, max_det=500)

    # Time the host decode inside predict() without changing it.
    decode_ms: list[float] = []
    orig_decode = model._decode_raw

    def timed_decode(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig_decode(*a, **k)
        finally:
            decode_ms.append((time.perf_counter() - t0) * 1e3)

    model._decode_raw = timed_decode  # type: ignore[method-assign]

    kg = known_good(model, fake_hailo, hd, args.out, captured)

    manifest = mf.load(args.data / "manifest.json")
    items = sorted(manifest.split(args.split), key=lambda i: i.id)[: args.limit]
    bad = mf.verify_files(manifest, args.data, items, decode=False)
    if bad:
        raise SystemExit(f"{len(bad)} files differ from the manifest: {bad[:3]}")

    files = {k: (args.preds / f"{k}_{args.split}.jsonl").open("w") for k in ("device_legacy", "device_legacy_swaprb",
                                                                            "device_legacy_native_res")}
    stats = {"images": 0, "input_identical": 0, "legacy_parity": 0, "candidate_parity": 0,
             "legacy_max_diff": 0.0, "candidate_max_diff": 0.0, "mismatch_examples": [],
             "dets_at_floor": 0, "dets_at_product_conf": 0,
             # The ONNX class heads have 81 channels too, so channel 80 is the model's own
             # (mmdet-style background slot), not Hailo padding. Every decoder drops it;
             # record what it holds so that is a measured statement.
             "class_channel_80_max": 0.0, "class_channel_80_mean_sum": 0.0,
             "class_head_min": 1.0, "class_head_max": 0.0, "box_head_min": 0.0, "box_head_max": 0.0}
    timing = {"predict_total_ms": [], "infer_ms": [], "decode_ms_conf020": [], "preprocess_ms": [],
              "decode_reference_ms_conf020": []}
    t_start = time.perf_counter()
    for n, item in enumerate(items):
        if n and n % 200 == 0:
            print(f"  {n}/{len(items)} {time.perf_counter() - t_start:.0f}s", flush=True)
        with Image.open(args.data / item.path) as im:
            full = im.convert("RGB")
        fitted = fit_640(full)
        rgb = np.asarray(fitted)
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        h, w = bgr.shape[:2]

        # A + B at the scoring floor
        mode["swap"] = False
        (res,) = model.predict(bgr, conf=FLOOR)
        batch, heads = captured["batch"], captured["heads"]
        stats["images"] += 1
        for arr in heads.values():
            if arr.shape[-1] == 81:
                ch80 = arr[..., 80]
                stats["class_channel_80_max"] = max(stats["class_channel_80_max"], float(ch80.max()))
                stats["class_channel_80_mean_sum"] += float(ch80.mean()) / 3
                stats["class_head_min"] = min(stats["class_head_min"], float(arr[..., :80].min()))
                stats["class_head_max"] = max(stats["class_head_max"], float(arr[..., :80].max()))
            else:
                stats["box_head_min"] = min(stats["box_head_min"], float(arr.min()))
                stats["box_head_max"] = max(stats["box_head_max"], float(arr.max()))
        ref_batch = independent_letterbox(rgb)
        same_input = np.array_equal(batch, ref_batch)
        stats["input_identical"] += int(same_input)
        if not same_input:
            heads = fake_hailo.nchw_heads_to_hef(sess.run(None, {sess.get_inputs()[0].name:
                                                                np.ascontiguousarray(ref_batch.transpose(0, 3, 1, 2))
                                                                .astype(np.float32)}))
        left, top = (640 - w) // 2, (640 - h) // 2
        geo = {"mode": "letterbox", "scale": 1.0, "left": left, "top": top, "frame_w": w, "frame_h": h}
        dev_rows = rows_of(res)
        ok, diff = parity(dev_rows, rows_of_dets(hd.unmap(hd.decode(heads, scales_hd, hd.LEGACY, FLOOR), geo)))
        stats["legacy_parity"] += int(ok)
        stats["legacy_max_diff"] = max(stats["legacy_max_diff"], diff)
        if not ok and len(stats["mismatch_examples"]) < 5:
            stats["mismatch_examples"].append({"id": item.id, "kind": "legacy", "diff": diff})
        stats["dets_at_floor"] += len(dev_rows)
        to_jsonl(files["device_legacy"], item.id, dev_rows, w, h)

        # C: the opt-in candidate decoder on the same heads
        model.decoder = "reference"
        try:
            (cres,) = orig_decode(heads, 1.0, left, top, w, h, None, FLOOR)
        finally:
            model.decoder = "legacy"
        ok, diff = parity(rows_of(cres), rows_of_dets(hd.unmap(hd.decode(heads, scales_hd, candidate, FLOOR), geo)))
        stats["candidate_parity"] += int(ok)
        stats["candidate_max_diff"] = max(stats["candidate_max_diff"], diff)
        if not ok and len(stats["mismatch_examples"]) < 5:
            stats["mismatch_examples"].append({"id": item.id, "kind": "candidate", "diff": diff})

        # G: timing at the product threshold (host work only; infer is x86 ONNX, not the chip)
        decode_ms.clear()
        t0 = time.perf_counter()
        ti = {}

        def timed_backend(b, ti=ti):
            t = time.perf_counter()
            r = base(b)
            ti["infer"] = (time.perf_counter() - t) * 1e3
            captured["batch"], captured["heads"] = b, r
            return r

        fake_hailo.state.backend = timed_backend
        (r020,) = model.predict(bgr, conf=PRODUCT_CONF)
        total = (time.perf_counter() - t0) * 1e3
        fake_hailo.state.backend = backend
        timing["predict_total_ms"].append(total)
        timing["infer_ms"].append(ti["infer"])
        timing["decode_ms_conf020"].append(decode_ms[-1])
        timing["preprocess_ms"].append(total - ti["infer"] - decode_ms[-1])
        stats["dets_at_product_conf"] += len(r020.boxes)
        model.decoder = "reference"
        t1 = time.perf_counter()
        try:
            orig_decode(captured["heads"], 1.0, left, top, w, h, None, PRODUCT_CONF)
        finally:
            model.decoder = "legacy"
        timing["decode_reference_ms_conf020"].append((time.perf_counter() - t1) * 1e3)

        # E: channel-swap fault
        if "E" not in skip:
            mode["swap"] = True
            (sres,) = model.predict(bgr, conf=FLOOR)
            mode["swap"] = False
            to_jsonl(files["device_legacy_swaprb"], item.id, rows_of(sres), w, h)
        # F: full-resolution frame straight into predict (cv2 INTER_LINEAR inside)
        if "F" not in skip:
            full_bgr = np.ascontiguousarray(np.asarray(full)[..., ::-1])
            (nres,) = model.predict(full_bgr, conf=FLOOR)
            fh, fw = full_bgr.shape[:2]
            to_jsonl(files["device_legacy_native_res"], item.id, rows_of(nres), fw, fh)

    for f in files.values():
        f.close()
    wall = time.perf_counter() - t_start
    scores = {}
    for key in files:
        if (key.endswith("swaprb") and "E" in skip) or (key.endswith("native_res") and "F" in skip):
            continue
        rep = bench.score(args.data, args.preds / f"{key}_{args.split}.jsonl", args.split, args.out,
                          frozen_test=False, name=key)
        scores[key] = {label: {k: rep["classes"][label][k] for k in ("npos", "ap50", "ap50_95")}
                       for label in ("person", "phone")}
        scores[key]["preds_sha256"] = sha256_file(args.preds / f"{key}_{args.split}.jsonl")
    summary = {
        "provenance": provenance, "manifest_id": manifest.id, "split": args.split, "wall_s": round(wall, 1),
        "parity": stats, "scores": scores,
        "timing_x86_cpu_ms": {k: {"p50": pct(v, 0.5), "p95": pct(v, 0.95), "n": len(v)} for k, v in timing.items()},
        "known_good": [{"file": i["file"], "dets": len(i["detections"]),
                        "top": i["detections"][:3]} for i in kg["images"]],
        "comparison_mission5": {"source": "docs/DECODER_CANDIDATE.md, job 20261111-022003-26d0f7 (CUDA, research "
                                          "letterbox + core.accel._decode_raw)",
                                "legacy_person_ap50": 0.656, "legacy_phone_ap50": 0.728},
    }
    (args.out / "inference_path_summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print(json.dumps({k: summary[k] for k in ("parity", "scores", "timing_x86_cpu_ms")}, indent=1, default=str))
    model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
