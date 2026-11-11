#!/usr/bin/env python3
"""Capture checkerboard frames for calibrate_intrinsics.py. RUNS ON THE PI.

Two resolutions, calibrated separately:
  --stream  the PRODUCTION stream: core.camera.Camera with the installed
            config's camera block (640x480 RGB888 video configuration,
            rotate_180 honoured). This is the geometry the product sees.
  --full    a full-sensor still configuration via Picamera2 (IMX708: 4608x2592),
            for the lens itself.

The camera is single-user, so this refuses while sensordyme.service may hold it
(and prints the operator stop/start commands). Frames are written to --out only;
point the camera at the board, not at people. Camera metadata (ScalerCrop,
sensor mode, properties) goes to capture_meta.json: the stream's ScalerCrop is
what tells you how much of the sensor, and so of the lens's FOV, 640x480 keeps.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import hwv_common as hc  # noqa: E402


def _jsonable(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Checkerboard capture for intrinsics (Pi)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--stream", action="store_true", help="production 640x480 path")
    mode.add_argument("--full", action="store_true", help="full-resolution stills")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--interval", type=float, default=2.0, help="seconds between frames")
    parser.add_argument("--interactive", action="store_true", help="press Enter for each frame")
    parser.add_argument("--install-dir", type=Path, default=hc.DEFAULT_INSTALL_DIR)
    parser.add_argument("--code-root", type=Path, default=hc.REPO_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-service-check", action="store_true")
    args = parser.parse_args(argv)
    # Lazily imported code may come from the INSTALLED tree (--code-root): leave no
    # __pycache__ behind in it. Set here, not at import, so importing this file has
    # no process-wide side effect.
    sys.dont_write_bytecode = True

    ok, why = hc.service_gate(hc.service_state(), allow_unknown=args.no_service_check)
    print(f"service: {why}")
    if not ok:
        print("REFUSED.\n" + hc.operator_commands(), file=sys.stderr)
        return 2
    ok, why = hc.ensure_outside(args.out, args.install_dir)
    if not ok:
        print(f"REFUSED: {why}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    import cv2

    config = yaml.safe_load((args.install_dir / "config.yaml").read_text()) or {}
    cam_cfg = config.get("camera") or {}
    meta: dict = {"utc": hc.utc_now(), "mode": "stream" if args.stream else "full",
                  "camera_config": cam_cfg, "environment": hc.environment()}

    if args.stream:
        sys.path.insert(0, str(args.code_root.resolve()))
        from core.camera import Camera

        camera = Camera(cam_cfg)
        picam = camera._picam
        grab = lambda: camera.read(timeout_s=5.0)  # noqa: E731
        release = camera.release
    else:
        from picamera2 import Picamera2

        picam = Picamera2()
        kwargs = {}
        if cam_cfg.get("rotate_180"):
            from libcamera import Transform

            kwargs["transform"] = Transform(hflip=1, vflip=1)
        picam.configure(picam.create_still_configuration(main={"format": "RGB888"}, **kwargs))
        picam.start()
        time.sleep(1.0)
        grab = picam.capture_array  # RGB888 = B,G,R bytes: already what cv2.imwrite expects

        def release():
            picam.stop()
            picam.close()

    try:
        for attr in ("camera_properties", "camera_config"):
            meta[attr] = _jsonable(getattr(picam, attr, None))
        try:
            meta["first_frame_metadata"] = {k: _jsonable(v) for k, v in picam.capture_metadata().items()}
        except Exception as exc:  # noqa: BLE001 - informational only
            meta["first_frame_metadata"] = repr(exc)
        for i in range(args.count):
            if args.interactive:
                input(f"[{i + 1}/{args.count}] position the board, then Enter ")
            else:
                time.sleep(args.interval)
            frame = grab()
            name = args.out / f"calib_{i:03d}.png"
            cv2.imwrite(str(name), frame)
            print(f"  {name.name}  {frame.shape[1]}x{frame.shape[0]}", flush=True)
    finally:
        release()
    hc.write_json(args.out / "capture_meta.json", meta)
    print(f"-> {args.out}  (copy to the worker; ScalerCrop is in capture_meta.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
