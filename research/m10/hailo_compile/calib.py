"""Build the Dataflow Compiler calibration set with the DEVICE's preprocessing, bit for bit.

    python research/m10/hailo_compile/calib.py --frames DIR --out calib_dir [--min-frames 1024]

Input: in-cabin frames from the SensorDyme camera (Camera Module 3 Wide NoIR through core/camera.py:
640x480 BGR), saved as PNG (lossless; JPEG adds compression artefacts the device never sees), or the
.npz samples of a scripts/hw_validation/capture_hailo_outputs.py bundle (their `input` tensor is
already the exact uint8 tensor the chip received and is used as is).

Each frame goes through scripts/hw_validation/hwv_common.letterbox_bgr_to_rgb, which is pinned to
core.accel.HailoYOLO.predict by a test (tests/test_m5_hw_validation.py and tests/test_m10_research_tools.py):
letterbox to 640x640 (cv2.INTER_LINEAR, pad 114, centred), BGR->RGB, uint8 NHWC. No normalisation is
applied: the network takes raw 0..255 (research/m10/export_contract.py), so the calibration tensor is
the device tensor cast to float32.

Writes calib_dir/calib_uint8_nhwc.npy (N, 640, 640, 3) uint8 and calib_manifest.json (per-frame source
sha256, size, mean luma; totals; the preprocessing spec). Refuses fewer than --min-frames (default
1,024) unless --allow-small, refuses frames that are not 640x480 unless --allow-any-size, and refuses
duplicate frames (same sha256).

This script does not need the Hailo SDK. It has not been run on in-cabin data: none exists yet
(docs/research/INCABIN_PILOT_PROTOCOL.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "hw_validation"))

INPUT_W = INPUT_H = 640
CAMERA_SIZE = (640, 480)
MIN_FRAMES = 1024
PREPROCESSING = {
    "source": "scripts/hw_validation/hwv_common.letterbox_bgr_to_rgb == core.accel.HailoYOLO.predict",
    "letterbox": "scale to fit 640x640, cv2.INTER_LINEAR, pad (114,114,114), centred (640x480 -> top/bottom 80)",
    "colour": "BGR (camera) -> RGB", "layout": "NHWC", "dtype": "uint8 0..255", "normalisation": "none",
}


def device_tensor(frame_bgr: np.ndarray) -> np.ndarray:
    """(640, 640, 3) uint8 RGB, exactly what HailoYOLO.predict hands the chip for this BGR frame."""
    import hwv_common as hc

    rgb, _geo = hc.letterbox_bgr_to_rgb(frame_bgr, INPUT_W, INPUT_H)
    return rgb


def luma(rgb: np.ndarray) -> float:
    r, g, b = (rgb[..., i].astype(np.float64) for i in range(3))
    return float((0.299 * r + 0.587 * g + 0.114 * b).mean())


def collect(frames: Path, *, allow_any_size: bool) -> tuple[list[np.ndarray], list[dict]]:
    import cv2

    tensors, rows, seen = [], [], set()
    for path in sorted(p for p in frames.rglob("*") if p.is_file()):
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest in seen:
            raise SystemExit(f"duplicate frame {path} (sha256 {digest[:12]})")
        seen.add(digest)
        if path.suffix.lower() == ".npz":
            with np.load(path) as z:
                if "input" not in z:
                    continue
                t = np.asarray(z["input"]).reshape(INPUT_H, INPUT_W, 3)
            if t.dtype != np.uint8:
                raise SystemExit(f"{path}: capture input is {t.dtype}, expected uint8")
            src = "capture_bundle_input"
            size = None
        elif path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise SystemExit(f"{path}: not a readable image")
            size = (frame.shape[1], frame.shape[0])
            if size != CAMERA_SIZE and not allow_any_size:
                raise SystemExit(f"{path}: {size[0]}x{size[1]}, expected the device camera's 640x480 "
                                 "(pass --allow-any-size to accept)")
            t = device_tensor(frame)
            src = "image"
        else:
            continue
        tensors.append(t)
        rows.append({"file": str(path.relative_to(frames)), "sha256": digest, "kind": src,
                     "size": list(size) if size else None, "mean_luma": round(luma(t), 2)})
    return tensors, rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-frames", type=int, default=MIN_FRAMES)
    p.add_argument("--allow-small", action="store_true", help="dry runs only; never for a release compile")
    p.add_argument("--allow-any-size", action="store_true")
    a = p.parse_args(argv)
    tensors, rows = collect(a.frames, allow_any_size=a.allow_any_size)
    if len(tensors) < a.min_frames and not a.allow_small:
        raise SystemExit(f"{len(tensors)} frames < {a.min_frames}: the calibration set must be at least "
                         f"{MIN_FRAMES} in-cabin NoIR frames (day, night/IR, each seat position)")
    if not tensors:
        raise SystemExit("no frames found")
    a.out.mkdir(parents=True, exist_ok=True)
    arr = np.stack(tensors).astype(np.uint8)
    np.save(a.out / "calib_uint8_nhwc.npy", arr)
    lum = [r["mean_luma"] for r in rows]
    manifest = {"format": "sensordyme-dfc-calib/1", "frames": len(rows), "shape": list(arr.shape),
                "dtype": str(arr.dtype), "array_sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
                "preprocessing": PREPROCESSING, "release_grade": len(rows) >= MIN_FRAMES and not a.allow_small,
                "mean_luma": {"min": min(lum), "median": float(np.median(lum)), "max": max(lum),
                              "share_below_40": sum(x < 40 for x in lum) / len(lum)},
                "files": rows}
    (a.out / "calib_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
