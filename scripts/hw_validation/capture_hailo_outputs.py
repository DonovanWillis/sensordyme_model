#!/usr/bin/env python3
"""Capture the Hailo-8L's raw outputs for a set of images (or live frames), for
offline comparison against the float ONNX and between decoders.

RUNS ON THE PI. Read-only with respect to installed software: it reads the
installed config.yaml and .hef, verifies the .hef against the config's pin, and
writes ONLY under --out (refused if --out is inside the installed tree). It does
not start, stop or restart anything; if sensordyme.service may hold the chip it
refuses and prints the operator commands instead. Bytecode writing is disabled
so importing code from the installed tree leaves no __pycache__ behind.

The model is loaded by constructing core.accel.HailoYOLO itself, so HEF parsing,
VDevice/ConfigureParams, UINT8 input and FLOAT32 output vstreams are exactly the
production path. Letterbox preprocessing mirrors HailoYOLO.predict line for line
(using core.accel.letterbox_transform); "stretch" is the DAMO-YOLO reference
preprocessing. Each image is captured in every requested mode.

Per sample it saves an .npz (the exact uint8 input tensor and every raw output
tensor) and records in manifest.json: tensor names/shapes/dtypes/quant info,
geometry, per-stage timing (preprocess, infer, production decode, the legacy and
reference decoders from hwv_decode.py) and the production decoder's own boxes.

    # images (preferred: a fixed, shareable set)
    /home/pi/sensordyme/.venv/bin/python capture_hailo_outputs.py \\
        --images ~/hwval/images --out ~/hwval/capture_$(date +%Y%m%dT%H%M%S)

    # live frames (frames are WRITTEN TO DISK: needs --allow-saving-frames)
    ... --camera 20 --allow-saving-frames --out ~/hwval/live_...
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import hwv_common as hc  # noqa: E402
import hwv_decode as hd  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def _ms(t0: int, t1: int) -> float:
    return (t1 - t0) / 1e6


def list_images(directory: Path, limit: Optional[int] = None) -> list[Path]:
    files = sorted(p for p in Path(directory).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    return files[:limit] if limit else files


def resolve_model(install_dir: Path, config: dict) -> tuple[Path, Optional[str]]:
    accel = config.get("accelerator") or {}
    hef = accel.get("yolo_hef")
    if not hef:
        raise SystemExit("REFUSED: config has no accelerator.yolo_hef")
    path = Path(hef)
    if not path.is_absolute():
        path = install_dir / path  # relative to the service WorkingDirectory
    return path, accel.get("yolo_hef_sha256")


def _frames_from_images(paths: list[Path], root: Path):
    import cv2

    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR, like Camera.read()
        if frame is None:
            print(f"  skip (unreadable): {path}", file=sys.stderr)
            continue
        yield path.relative_to(root).as_posix(), frame, {
            "kind": "image", "path": path.relative_to(root).as_posix(),
            "sha256": hc.sha256_file(path)}


def _frames_from_camera(config: dict, count: int, interval_s: float, frames_dir: Path):
    import cv2

    from core.camera import Camera

    camera = Camera(config.get("camera") or {})
    try:
        picam = getattr(camera, "_picam", None)
        props = {}
        if picam is not None:
            for attr in ("camera_properties", "camera_config"):
                try:
                    props[attr] = getattr(picam, attr)
                except Exception as exc:  # noqa: BLE001 - informational only
                    props[attr] = repr(exc)
        yield None, None, {"camera_properties": props}
        for i in range(count):
            frame = camera.read(timeout_s=5.0)
            wall_ns, mono_ns = time.time_ns(), time.monotonic_ns()
            name = f"frame_{i:04d}.png"
            cv2.imwrite(str(frames_dir / name), frame)
            yield f"camera/{name}", frame, {
                "kind": "camera", "saved_frame": f"frames/{name}", "capture_wall_ns": wall_ns,
                "capture_monotonic_ns": mono_ns, "sha256": hc.sha256_file(frames_dir / name)}
            time.sleep(interval_s)
    finally:
        camera.release()


def _range_summary(outs: dict[str, np.ndarray], scales: list[hd.Scale]) -> dict:
    out = {}
    for s in scales:
        cls = np.asarray(outs[s.cls])
        box = np.asarray(outs[s.box])
        out[str(s.stride)] = {"cls_min": float(cls.min()), "cls_max": float(cls.max()),
                              "box_min": float(box.min()), "box_max": float(box.max())}
    return out


def _production_boxes(model, raw, geometry, conf) -> tuple[list[dict], float]:
    t0 = time.perf_counter_ns()
    result = model._decode_raw(raw, geometry["scale"], geometry["left"], geometry["top"],
                               geometry["frame_w"], geometry["frame_h"], None, conf)[0]
    elapsed = _ms(t0, time.perf_counter_ns())
    boxes = [{"xyxy": [round(float(v), 2) for v in b.xyxy[0]], "cls": int(b.cls[0]),
              "score": round(float(b.conf[0]), 5)} for b in result.boxes]
    return boxes, elapsed


def _capture_native(model, samples: list[dict], out_dir: Path) -> dict:
    """Second pass with the vstreams' native (quantised) output format.

    Optional and experimental: it re-uses the production network group with a
    different OutputVStreamParams. The float pass is complete before this runs,
    so a failure here costs nothing already captured.
    """
    from hailo_platform import FormatType, InferVStreams, OutputVStreamParams

    report: dict = {"format_type": "AUTO", "samples": 0, "dequant_check": {}}
    model._teardown()
    params = OutputVStreamParams.make(model._network_group, format_type=FormatType.AUTO)
    infos = {i.name: hc.describe_vstream_info(i) for i in model._hef.get_output_vstream_infos()}
    with InferVStreams(model._network_group, model._input_params, params) as pipe, \
            model._network_group.activate(model._network_group_params):
        for sample in samples:
            inp, floats = hc.load_sample(out_dir, sample)
            raw = pipe.infer({model._input_name: inp})
            names = [o["name"] for o in sample["outputs"]]
            native = [np.asarray(raw[n]) for n in names]
            native_file = sample["file"].replace(".npz", "_native.npz")
            np.savez_compressed(out_dir / native_file, **{f"q_out{i}": a for i, a in enumerate(native)})
            sample["native_file"] = native_file
            report["samples"] += 1
            for name, q in zip(names, native, strict=True):
                quant = infos.get(name, {}).get("quant_info") or {}
                zp, scale = quant.get("qp_zp"), quant.get("qp_scale")
                if isinstance(zp, float) and isinstance(scale, float):
                    deq = (q.astype(np.float64) - zp) * scale
                    err = float(np.abs(deq - floats[name]).max())
                    prev = report["dequant_check"].get(name, 0.0)
                    report["dequant_check"][name] = max(prev, err)
                report.setdefault("dtypes", {})[name] = str(q.dtype)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Capture raw Hailo outputs (Pi, read-only).")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", type=Path, help="directory of test images (recursed)")
    src.add_argument("--camera", type=int, metavar="N", help="capture N live frames instead")
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--install-dir", type=Path, default=hc.DEFAULT_INSTALL_DIR,
                        help="installed SensorDyme tree (read only)")
    parser.add_argument("--config", type=Path, default=None,
                        help="config.yaml to read (default: <install-dir>/config.yaml)")
    parser.add_argument("--code-root", type=Path, default=hc.REPO_ROOT,
                        help="tree whose core/accel.py is imported (default: this checkout; "
                             "pass the install dir to exercise the installed decoder)")
    parser.add_argument("--modes", default="letterbox,stretch",
                        help="comma list of letterbox,stretch")
    parser.add_argument("--conf", type=float, default=0.25, help="score threshold for decoding")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=5, help="untimed inferences first")
    parser.add_argument("--camera-interval", type=float, default=0.5)
    parser.add_argument("--allow-saving-frames", action="store_true",
                        help="required with --camera: frames are written to --out")
    parser.add_argument("--native-outputs", action="store_true",
                        help="also capture quantised outputs (second pass, experimental)")
    parser.add_argument("--no-compress", action="store_true")
    parser.add_argument("--no-service-check", action="store_true",
                        help="ONLY for a machine with no systemd service at all")
    args = parser.parse_args(argv)
    # Lazily imported code may come from the INSTALLED tree (--code-root): leave no
    # __pycache__ behind in it. Set here, not at import, so importing this file has
    # no process-wide side effect.
    sys.dont_write_bytecode = True

    # ---- gates: every refusal happens before the device is touched ----------
    state = hc.service_state()
    ok, why = hc.service_gate(state, allow_unknown=args.no_service_check)
    print(f"service: {why}")
    if not ok:
        print("REFUSED.\n" + hc.operator_commands(), file=sys.stderr)
        return 2
    holders = hc.device_holders()
    if holders:
        print(f"REFUSED: /dev/hailo0 is open in pid(s) {holders}. Stop them first.", file=sys.stderr)
        return 2
    if args.camera is not None and not args.allow_saving_frames:
        print("REFUSED: --camera writes frames to disk, which the product never does. Pass "
              "--allow-saving-frames, point the camera at a test scene, and delete --out after "
              "the worker has the bundle.", file=sys.stderr)
        return 2
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not modes or any(m not in hc.PREPROCESSORS for m in modes):
        print(f"REFUSED: --modes must be from {sorted(hc.PREPROCESSORS)}", file=sys.stderr)
        return 2
    ok, why = hc.ensure_outside(args.out, args.install_dir)
    if not ok:
        print(f"REFUSED: {why}", file=sys.stderr)
        return 2
    if args.out.exists() and any(args.out.iterdir()):
        print(f"REFUSED: {args.out} exists and is not empty", file=sys.stderr)
        return 2

    config_path = args.config or (args.install_dir / "config.yaml")
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes) or {}
    hef_path, pin = resolve_model(args.install_dir, config)
    actual = hc.sha256_file(hef_path)
    ok, why = hc.verify_pin(actual, pin)
    print(f"model: {hef_path}: {why}")
    if not ok:
        print(f"REFUSED: {why}", file=sys.stderr)
        return 2

    images = list_images(args.images, args.limit) if args.images else []
    n_items = len(images) if args.images else args.camera
    if n_items == 0:
        print("REFUSED: no images found", file=sys.stderr)
        return 2
    ok, why = hc.check_free_space(args.out, hc.estimate_bundle_bytes(n_items * len(modes),
                                                                     native=args.native_outputs))
    print(f"disk: {why}")
    if not ok:
        print(f"REFUSED: {why}", file=sys.stderr)
        return 2

    # ---- provenance, collected before the VDevice is opened -----------------
    sys.path.insert(0, str(args.code_root.resolve()))
    accel_file = args.code_root / "core" / "accel.py"
    installed_accel = args.install_dir / "core" / "accel.py"
    manifest: dict = {
        "format": hc.BUNDLE_FORMAT,
        "status": "running",
        "created_utc": hc.utc_now(),
        "argv": sys.argv if argv is None else ["capture_hailo_outputs.py", *argv],
        "design_statement": ("this script writes only under --out and never starts/stops a "
                             "service; model sha256 is re-checked after the run (sha256_after)"),
        "service": {"name": hc.SERVICE, "state_at_start": state},
        "install_dir": str(args.install_dir),
        "config": {"path": str(config_path), "sha256": hc.sha256_bytes(config_bytes)},
        "model": {"path": str(hef_path), "sha256": actual, "pin": pin},
        "code": {
            "code_root": str(args.code_root.resolve()),
            "git_commit": hc.git_commit(args.code_root),
            "git_dirty": hc.git_dirty(args.code_root),
            "core_accel_sha256": hc.sha256_file(accel_file),
            "installed_core_accel_sha256": (hc.sha256_file(installed_accel)
                                            if installed_accel.exists() else None),
            "hw_validation_sha256": {p.name: hc.sha256_file(p) for p in sorted(HERE.glob("*.py"))},
        },
        "hailo": hc.hailo_runtime_info(),
        "environment": hc.environment(),
        "decode": {"conf": args.conf,
                   "variants": {k: dataclasses.asdict(v) for k, v in hd.VARIANTS.items()}},
        "modes": modes,
        "samples": [],
    }
    manifest["code"]["same_as_installed"] = (
        manifest["code"]["core_accel_sha256"] == manifest["code"]["installed_core_accel_sha256"])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "samples").mkdir()
    frames_dir = args.out / "frames"
    if args.camera is not None:
        frames_dir.mkdir()
    hc.write_json(args.out / "manifest.json", manifest)

    from core.accel import HailoYOLO

    model = HailoYOLO(str(hef_path), sha256=actual)
    try:
        if hc.sha256_file(hef_path) != actual:
            print("REFUSED: the .hef changed while it was being loaded", file=sys.stderr)
            return 2
        in_w, in_h = model._input_w, model._input_h
        hef_outputs = [hc.describe_vstream_info(i) for i in model._hef.get_output_vstream_infos()]
        manifest["hef"] = {
            "input": hc.describe_vstream_info(model._hef.get_input_vstream_infos()[0]),
            "outputs": hef_outputs,
            "raw_outputs": bool(model._raw_outputs),
        }
        if not model._raw_outputs:
            print("REFUSED: this .hef has on-chip NMS; nothing raw to capture", file=sys.stderr)
            return 2
        names = [o["name"] for o in hef_outputs]
        scales = hd.group_heads({o["name"]: tuple(o["shape"]) for o in hef_outputs}, in_h)
        manifest["hef"]["scales"] = [dataclasses.asdict(s) for s in scales]
        pipeline = model._ensure_ready()

        if args.images:
            source = _frames_from_images(images, args.images)
            manifest["source"] = {"kind": "images", "dir": str(args.images.resolve()),
                                  "count": len(images)}
        else:
            source = _frames_from_camera(config, args.camera, args.camera_interval, frames_dir)
            manifest["source"] = {"kind": "camera", "camera_config": config.get("camera")}

        warm = args.warmup
        for image_id, frame, origin in source:
            if frame is None:
                manifest["source"].update(origin)
                continue
            for mode in modes:
                t0 = time.perf_counter_ns()
                rgb, geometry = hc.PREPROCESSORS[mode](frame, in_w, in_h)
                batch = np.expand_dims(rgb, axis=0)
                t1 = time.perf_counter_ns()
                while warm > 0:
                    pipeline.infer({model._input_name: batch})
                    warm -= 1
                t2 = time.perf_counter_ns()
                raw = pipeline.infer({model._input_name: batch})
                t3 = time.perf_counter_ns()
                outs = {n: np.array(raw[n], copy=True) for n in names}
                timing = {"preprocess": _ms(t0, t1), "infer": _ms(t2, t3)}
                production = None
                if mode == "letterbox":
                    production, timing["decode_production"] = _production_boxes(
                        model, outs, geometry, args.conf)
                counts = {}
                for name, spec in (("legacy", hd.LEGACY), ("reference", hd.REFERENCE)):
                    d0 = time.perf_counter_ns()
                    dets = hd.decode(outs, scales, spec, args.conf)
                    timing[f"decode_{name}"] = _ms(d0, time.perf_counter_ns())
                    counts[name] = len(dets)
                sample_id = f"{len(manifest['samples']):05d}_{mode}"
                fname = f"samples/{sample_id}.npz"
                hc.save_sample(args.out / fname, batch, [outs[n] for n in names],
                               compressed=not args.no_compress)
                manifest["samples"].append({
                    "id": sample_id, "image_id": image_id, "mode": mode, "file": fname,
                    "source": origin, "geometry": geometry,
                    "outputs": [{"index": i, "name": n, "shape": list(outs[n].shape),
                                 "dtype": str(outs[n].dtype)} for i, n in enumerate(names)],
                    "timing_ms": timing, "detection_counts": counts,
                    "production_detections": production,
                    "ranges": _range_summary(outs, scales),
                })
            print(f"  {image_id}: {len(modes)} mode(s) captured", flush=True)

        if args.native_outputs:
            try:
                manifest["native"] = _capture_native(model, manifest["samples"], args.out)
            except Exception as exc:  # noqa: BLE001 - optional pass; float capture is intact
                manifest["native"] = {"error": repr(exc)}
        manifest["status"] = "complete"
    finally:
        if manifest["status"] == "running":
            manifest["status"] = "incomplete"
        manifest["finished_utc"] = hc.utc_now()
        model.close()
        manifest["model"]["sha256_after"] = hc.sha256_file(hef_path)
        stages = sorted({k for s in manifest["samples"] for k in s["timing_ms"]})
        manifest["timing_summary_ms"] = {
            k: hd.percentiles([s["timing_ms"][k] for s in manifest["samples"] if k in s["timing_ms"]])
            for k in stages}
        hc.write_json(args.out / "manifest.json", manifest)

    print(f"\n{len(manifest['samples'])} samples -> {args.out}/manifest.json")
    for stage, stats in manifest["timing_summary_ms"].items():
        if stats:
            print(f"  {stage:<20} p50 {stats['p50']:7.2f} ms   p95 {stats['p95']:7.2f} ms")
    bad = [s["id"] for s in manifest["samples"]
           if any(r["cls_min"] < 0 or r["cls_max"] > 1 for r in s["ranges"].values())]
    if bad:
        print(f"  WARNING: class head outside [0,1] in {len(bad)} sample(s) - not sigmoid'd?")
    print("  Copy the whole directory to the worker; then delete it here if it holds frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
