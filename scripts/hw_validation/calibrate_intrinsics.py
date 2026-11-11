#!/usr/bin/env python3
"""Calibrate the Camera Module 3 Wide (NoIR) from checkerboard images, with both
OpenCV models, and say what config value that implies.

RUNS ON A WORKER OR A MAC, on images captured on the Pi by
capture_calibration_frames.py. Never edits config.yaml: it prints the value an
owner may choose to set.

Models:
  pinhole  cv2.calibrateCamera with CALIB_RATIONAL_MODEL (k1..k6, p1, p2)
  fisheye  cv2.fisheye.calibrate (k1..k4, equidistant)

Outputs (JSON): camera matrix, distortion, RMS and per-view reprojection error,
principal-point offset from the image centre, and three different fields of
view, which must not be confused:

  edge-ray HFOV/VFOV/DFOV  the true angle between the rays through opposite
                           image edges (what a lens datasheet quotes; the
                           Wide's "102 deg horizontal" is this, at FULL sensor
                           width -- a 4:3 640x480 stream is a crop of the 16:9
                           sensor and sees less)
  paraxial HFOV            2*atan((w/2)/fx): the FOV a DISTORTION-FREE pinhole
                           with this focal length would have
  recommended              modules.driver_attention.camera_hfov_deg feeds
                           _camera_matrix(), which builds fx = (w/2)/tan(hfov/2),
                           centre principal point, ZERO distortion, for
                           solvePnP on face landmarks near the image centre.
                           The value that reproduces the calibrated fx there
                           is the paraxial HFOV, not the edge-ray one: with a
                           barrel-distorting wide lens, setting the datasheet
                           edge-ray angle gives a focal length that is too short.

Calibrate each resolution separately (640x480 production stream, and full res):
the stream is scaled AND possibly cropped from the sensor, so its intrinsics are
not the full-res ones divided by a constant unless ScalerCrop says so.

    python calibrate_intrinsics.py --images calib_640x480/ --pattern 9x6 \\
        --square-mm 25 --out intrinsics_640x480.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import hwv_common as hc  # noqa: E402


# ------------------------------------------------------------------ geometry (pure)
def board_points(cols: int, rows: int, square: float) -> np.ndarray:
    grid = np.zeros((rows * cols, 3), np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return grid


def _undistort(model: str, pts: np.ndarray, k: np.ndarray, d: np.ndarray) -> np.ndarray:
    import cv2

    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    if model == "fisheye":
        return cv2.fisheye.undistortPoints(pts, k, d).reshape(-1, 2)
    return cv2.undistortPoints(pts, k, d).reshape(-1, 2)


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.array([a[0], a[1], 1.0]), np.array([b[0], b[1], 1.0])
    cos = float(ra @ rb / (np.linalg.norm(ra) * np.linalg.norm(rb)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def fields_of_view(model: str, k: np.ndarray, d: np.ndarray, size: tuple[int, int]) -> dict:
    """Edge-ray H/V/D FOV (through the principal point's row/column) and paraxial HFOV."""
    w, h = size
    cx, cy = float(k[0, 2]), float(k[1, 2])
    pts = np.array([[-0.5, cy], [w - 0.5, cy], [cx, -0.5], [cx, h - 0.5],
                    [-0.5, -0.5], [w - 0.5, h - 0.5]])
    n = _undistort(model, pts, k, d)
    return {
        "edge_hfov_deg": _angle(n[0], n[1]),
        "edge_vfov_deg": _angle(n[2], n[3]),
        "edge_dfov_deg": _angle(n[4], n[5]),
        "paraxial_hfov_deg": math.degrees(2 * math.atan((w / 2.0) / float(k[0, 0]))),
        "paraxial_vfov_deg": math.degrees(2 * math.atan((h / 2.0) / float(k[1, 1]))),
    }


def recommend(result: dict, size: tuple[int, int]) -> dict:
    """The config values implied by the better-fitting model (lower RMS)."""
    fitted = {m: r for m, r in result.items() if isinstance(r, dict) and "rms" in r}
    if not fitted:
        return {"error": "no model converged"}
    best = min(fitted, key=lambda m: fitted[m]["rms"])
    r = fitted[best]
    w, h = size
    fx = r["camera_matrix"][0][0]
    cx, cy = r["camera_matrix"][0][2], r["camera_matrix"][1][2]
    return {
        "model_used": best,
        "image_size": [w, h],
        "camera_hfov_deg": round(r["fov"]["paraxial_hfov_deg"], 1),
        "why": "paraxial HFOV reproduces the calibrated fx in driver_attention._camera_matrix "
               "(zero-distortion pinhole, centred principal point)",
        "residuals_not_modelled_by_config": {
            "principal_point_offset_px": [round(cx - w / 2.0, 2), round(cy - h / 2.0, 2)],
            "yaw_bias_deg_from_cx": round(math.degrees(math.atan((cx - w / 2.0) / fx)), 2),
            "pitch_bias_deg_from_cy": round(math.degrees(math.atan((cy - h / 2.0) / fx)), 2),
            "edge_vs_paraxial_hfov_deg": round(r["fov"]["edge_hfov_deg"] - r["fov"]["paraxial_hfov_deg"], 1),
        },
        "yaml_snippet": f"modules:\n  driver_attention:\n    camera_hfov_deg: "
                        f"{round(r['fov']['paraxial_hfov_deg'], 1)}  # {best} calibration at {w}x{h}",
        "warning": "NOT applied. Changing camera_hfov_deg invalidates the device's stored "
                   "head-pose calibration (it recalibrates); owner approval required.",
    }


# ------------------------------------------------------------------ calibration
def calibrate_pinhole(obj: list, img: list, size: tuple[int, int]) -> dict:
    import cv2

    flags = cv2.CALIB_RATIONAL_MODEL
    rms, k, d, rvecs, tvecs = cv2.calibrateCamera(obj, img, size, None, None, flags=flags)
    per_view = []
    for o, i, rv, tv in zip(obj, img, rvecs, tvecs, strict=True):
        proj, _ = cv2.projectPoints(o, rv, tv, k, d)
        per_view.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - i.reshape(-1, 2)) ** 2, axis=1)))))
    return {"rms": float(rms), "camera_matrix": k.tolist(), "distortion": d.ravel().tolist(),
            "distortion_names": ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"],
            "per_view_rms": per_view, "fov": fields_of_view("pinhole", k, d, size)}


def calibrate_fisheye(obj: list, img: list, size: tuple[int, int]) -> dict:
    import cv2

    o = [np.asarray(x, np.float64).reshape(-1, 1, 3) for x in obj]
    i = [np.asarray(x, np.float64).reshape(-1, 1, 2) for x in img]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-7)
    base = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    note = None
    try:
        rms, k, d, rvecs, tvecs = cv2.fisheye.calibrate(
            o, i, size, np.zeros((3, 3)), np.zeros((4, 1)),
            flags=base | cv2.fisheye.CALIB_CHECK_COND, criteria=criteria)
    except cv2.error as exc:
        note = f"CALIB_CHECK_COND rejected a view ({str(exc).splitlines()[-1][:120]}); retried without it"
        rms, k, d, rvecs, tvecs = cv2.fisheye.calibrate(
            o, i, size, np.zeros((3, 3)), np.zeros((4, 1)), flags=base, criteria=criteria)
    per_view = []
    for ob, im, rv, tv in zip(o, i, rvecs, tvecs, strict=True):
        proj, _ = cv2.fisheye.projectPoints(ob, rv, tv, k, d)
        per_view.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - im.reshape(-1, 2)) ** 2, axis=1)))))
    out = {"rms": float(rms), "camera_matrix": k.tolist(), "distortion": d.ravel().tolist(),
           "distortion_names": ["k1", "k2", "k3", "k4"], "per_view_rms": per_view,
           "fov": fields_of_view("fisheye", k, d, size)}
    if note:
        out["note"] = note
    return out


def find_corners(paths: list[Path], cols: int, rows: int) -> tuple[list, list, tuple[int, int], list]:
    import cv2

    img_points, used, size = [], [], None
    for path in paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        this = (gray.shape[1], gray.shape[0])
        if size is None:
            size = this
        elif this != size:
            raise SystemExit(f"{path}: size {this} differs from {size}; calibrate one resolution per run")
        found, corners = cv2.findChessboardCornersSB(gray, (cols, rows), cv2.CALIB_CB_EXHAUSTIVE)
        if not found:
            found, corners = cv2.findChessboardCorners(gray, (cols, rows))
            if found:
                corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4))
        if found:
            img_points.append(corners.reshape(-1, 1, 2).astype(np.float32))
            used.append(path.name)
    return img_points, used, size or (0, 0), [p.name for p in paths if p.name not in used]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Wide-lens intrinsics: fisheye vs pinhole+rational")
    parser.add_argument("--images", type=Path, required=True, help="directory of checkerboard images")
    parser.add_argument("--pattern", default="9x6", help="INNER corners, cols x rows")
    parser.add_argument("--square-mm", type=float, default=25.0)
    parser.add_argument("--model", choices=("both", "pinhole", "fisheye"), default="both")
    parser.add_argument("--min-views", type=int, default=12)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    cols, rows = (int(v) for v in args.pattern.lower().split("x"))
    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    img_points, used, size, rejected = find_corners(paths, cols, rows)
    print(f"{len(used)}/{len(paths)} images with a detected {cols}x{rows} board at {size[0]}x{size[1]}")
    if len(used) < args.min_views:
        print(f"REFUSED: fewer than {args.min_views} usable views; capture more, covering the corners",
              file=sys.stderr)
        return 2
    obj = [board_points(cols, rows, args.square_mm) for _ in used]
    result: dict = {"utc": hc.utc_now(), "images_dir": str(args.images), "image_size": list(size),
                    "pattern": [cols, rows], "square_mm": args.square_mm, "views_used": used,
                    "views_rejected": rejected}
    capture_meta = args.images / "capture_meta.json"
    if capture_meta.exists():
        result["capture_meta"] = json.loads(capture_meta.read_text())
    if args.model in ("both", "pinhole"):
        result["pinhole_rational"] = calibrate_pinhole(obj, img_points, size)
    if args.model in ("both", "fisheye"):
        try:
            result["fisheye"] = calibrate_fisheye(obj, img_points, size)
        except Exception as exc:  # noqa: BLE001 - report, keep the pinhole result
            result["fisheye"] = {"error": repr(exc)}
    result["recommendation"] = recommend(
        {k: v for k, v in result.items() if k in ("pinhole_rational", "fisheye")}, size)
    if "pinhole_rational" in result and "rms" in result.get("fisheye", {}):
        a, b = result["pinhole_rational"]["fov"], result["fisheye"]["fov"]
        result["model_disagreement_deg"] = {k: round(a[k] - b[k], 2) for k in a}
    hc.write_json(args.out, result)
    for m in ("pinhole_rational", "fisheye"):
        r = result.get(m) or {}
        if "rms" in r:
            f = r["fov"]
            print(f"{m:>16}: rms {r['rms']:.3f}px  fx {r['camera_matrix'][0][0]:.1f}  "
                  f"edge HFOV {f['edge_hfov_deg']:.1f}  VFOV {f['edge_vfov_deg']:.1f}  "
                  f"DFOV {f['edge_dfov_deg']:.1f}  paraxial HFOV {f['paraxial_hfov_deg']:.1f}")
        elif r:
            print(f"{m:>16}: {r.get('error')}")
    print("\n" + result["recommendation"].get("yaml_snippet", ""))
    print(result["recommendation"].get("warning", ""))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
