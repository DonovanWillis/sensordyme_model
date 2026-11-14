#!/usr/bin/env python3
"""Build the known-good image set for the Hailo stages (scripts/hw_validation/known_good/).

No people and no faces. Every frame is 640x480 BGR, the production camera size
(config.yaml camera.width/height), so HailoYOLO.predict letterboxes it at scale
1.0: no resampling happens anywhere, and the uint8 tensor the Pi's chip receives
is bit-identical to the one the fleet's float model received when the expected
outputs were made (the stage checks this by sha256 of the input tensor).

Sources (all public domain / CC0; nothing here needs attribution, it is recorded
anyway in SOURCES.json):

  chelsea.png       scikit-image v0.24.0 data, "Chelsea the cat". CC0 by the
                    photographer (Stefan van der Walt).
  coffee.png        scikit-image v0.24.0 data, "Coffee cup". CC0 by the
                    photographer (Rachel Michetti).
  clock_motion.png  scikit-image v0.24.0 data, "Motion blurred clock". Released
                    into the public domain by the photographer (Stefan van der Walt).
  rocket.jpg        scikit-image v0.24.0 data, DSCOVR launch on Falcon 9. Public
                    domain (SpaceX).

The source files are verified against the sha256 values in scikit-image's own
data registry (skimage/data/_registry.py at v0.24.0) before use. Each photo is
pasted, unscaled, at the centre of a 640x480 canvas of grey 90 (cropped
centrally if larger). Two synthetic frames are generated: flat grey 114 (the
letterbox colour: the model sees one uniform field) and a geometric test card.

    python make_known_good.py --src <dir with the 4 downloaded files> --out known_good/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SKIMAGE_REF = "scikit-image v0.24.0, skimage/data"
SOURCES = {
    "chelsea.png": ("596aa1e7cb875eb79f437e310381d26b338a81c2da23439704a73c4651e8c4bb",
                    "CC0 (Stefan van der Walt)", "cat"),
    "coffee.png": ("cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7",
                   "CC0 (Rachel Michetti)", "cup on a saucer"),
    "clock_motion.png": ("f029226b28b642e80113d86622e9b215ee067a0966feaf5e60604a1e05733955",
                         "public domain (Stefan van der Walt)", "motion-blurred wall clock, greyscale"),
    "rocket.jpg": ("c2dd0de7c538df8d111e479619b129464d0269d0ae5fd18ca91d33a7fdfea95c",
                   "public domain (SpaceX)", "rocket launch"),
}
W, H, CANVAS = 640, 480, 90


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def paste_centre(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    h, w = image.shape[:2]
    y0, x0 = max(0, (h - H) // 2), max(0, (w - W) // 2)
    image = image[y0:y0 + min(h, H), x0:x0 + min(w, W)]
    h, w = image.shape[:2]
    canvas = np.full((H, W, 3), CANVAS, dtype=np.uint8)
    top, left = (H - h) // 2, (W - w) // 2
    canvas[top:top + h, left:left + w] = image
    return canvas


def test_card() -> np.ndarray:
    """Deterministic shapes and gradients; integer arithmetic only."""
    y, x = np.mgrid[0:H, 0:W]
    card = np.zeros((H, W, 3), dtype=np.uint8)
    card[..., 0] = (x * 255 // (W - 1)).astype(np.uint8)
    card[..., 1] = (y * 255 // (H - 1)).astype(np.uint8)
    card[..., 2] = (((x // 40) + (y // 40)) % 2 * 200).astype(np.uint8)
    cv2.rectangle(card, (60, 60), (220, 300), (20, 20, 20), -1)
    cv2.circle(card, (450, 240), 110, (240, 240, 240), -1)
    cv2.line(card, (0, 479), (639, 0), (0, 0, 255), 9)
    return card


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (want, licence, content) in SOURCES.items():
        data = (args.src / name).read_bytes()
        if sha256(data) != want:
            print(f"REFUSED: {name} sha256 {sha256(data)} != scikit-image registry {want}", file=sys.stderr)
            return 2
        decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        frame = paste_centre(decoded)
        out_name = f"kg_{Path(name).stem}.png"
        cv2.imwrite(str(args.out / out_name), frame)
        rows.append({"file": out_name, "content": content, "people": False,
                     "source": f"{SKIMAGE_REF}/{name}", "source_sha256": want, "licence": licence,
                     "transform": f"decoded with OpenCV, pasted unscaled at the centre of a {W}x{H} grey-{CANVAS} "
                                  "canvas (centre-cropped if larger)"})
    for out_name, frame, content in (("kg_flat_grey.png", np.full((H, W, 3), 114, np.uint8),
                                      "flat grey 114 (negative: nothing to detect)"),
                                     ("kg_test_card.png", test_card(), "synthetic geometric test card")):
        cv2.imwrite(str(args.out / out_name), frame)
        rows.append({"file": out_name, "content": content, "people": False, "source": "generated by "
                     "make_known_good.py", "licence": "public domain (generated)", "transform": None})
    for row in rows:
        data = (args.out / row["file"]).read_bytes()
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        row["sha256"] = sha256(data)
        row["shape"] = list(frame.shape)
        row["pixels_sha256"] = sha256(np.ascontiguousarray(frame).tobytes())
    (args.out / "SOURCES.json").write_text(json.dumps({
        "purpose": "known-good inputs for the Hailo stages; no people, no faces",
        "frame": {"width": W, "height": H, "order": "BGR on disk via OpenCV", "letterbox_scale": 1.0},
        "images": rows}, indent=1) + "\n")
    print(json.dumps([(r["file"], r["sha256"][:12]) for r in rows]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
