#!/usr/bin/env python3
"""Processed FPS, per-stage latency and camera-to-alert latency analysis.

Three subcommands, chosen so that the production numbers come from metrics the
runtime ALREADY publishes, and the per-stage split comes from an out-of-process
harness rather than new runtime code:

  health   (Pi, service RUNNING, read-only). Polls the service's own
           /api/health and records what it reports: capture fps, the
           processed-frame interval distribution (frame_bus
           processed_interval_stats: frames, p50/p95/p99/max, stalls), Hailo
           inference count, temperature/throttling and the monitoring state.
           Processed FPS per window = processed_frames / window seconds.
           CAVEAT: those windows are drained per caller ("dashboard"); any
           other /api/health client (the app, a browser) running at the same
           time steals samples. Close them while measuring.

  stages   (Pi, service STOPPED: needs the camera and the chip). Builds the
           configured module the way benchmarks/run_benchmarks.py does and
           times, per frame: capture, preprocess, Hailo infer, decode,
           FaceMesh, Hands, and the module's own logic (rules, trackers,
           fusion: process() minus the timed sub-stages). Timing is by
           wrapping instance attributes of the already-built objects in this
           process only; no repo file changes. Single-threaded, so its FPS is
           a sequential figure; the production loop overlaps capture with
           processing (see the doc for how to read the two together). The
           runtime's alert dispatch and custom-rule evaluation are NOT inside
           module.process() and are not in this split.

  latency  (anywhere). Summarises hand-annotated trials from the high-frame-
           rate phone video procedure (docs/HAILO_HARDWARE_VALIDATION.md):
           --trials CSV  columns stimulus_ms, alert_onset_ms[, designed_delay_ms]
           --glass  CSV  columns displayed_ms, captured_ms[, offset_ms]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import hwv_common as hc  # noqa: E402
import hwv_decode as hd  # noqa: E402

STAGES = ("capture", "preprocess", "hailo_infer", "decode", "detector_total", "facemesh",
          "hands", "module_logic", "process_total", "frame_total")


# ------------------------------------------------------------------ health (pure)
def health_row(payload: dict, now_s: float, prev: Optional[dict]) -> dict:
    """One poll of /api/health -> a flat row. `prev` is the previous row (rates)."""
    device = payload.get("device") or {}
    cam = (device.get("camera") or {}).get("metrics") or {}
    inf = (device.get("inference") or {}).get("metrics") or {}
    acc = (device.get("accelerator") or {}).get("metrics") or {}
    sysm = (device.get("system") or {}).get("metrics") or {}
    row = {
        "t": now_s,
        "camera_fps": cam.get("fps"),
        "processed_frames": inf.get("processed_frames"),
        "processed_p50_ms": inf.get("processed_p50_ms"),
        "processed_p95_ms": inf.get("processed_p95_ms"),
        "processed_p99_ms": inf.get("processed_p99_ms"),
        "processed_max_ms": inf.get("processed_max_ms"),
        "processed_stalls_500ms": inf.get("processed_stalls_500ms"),
        "processed_stalls_1s": inf.get("processed_evidence_losing_stalls_1s"),
        "hailo_inferences": acc.get("inferences"),
        "temperature_c": sysm.get("temperature_c"),
        "throttle_now": sysm.get("throttle_now"),
        "load_1m": sysm.get("load_1m"),
        "monitoring_state": (device.get("monitoring") or {}).get("state"),
        "monitoring_active": device.get("monitoring_active"),
    }
    if prev is not None and now_s > prev["t"]:
        window = now_s - prev["t"]
        row["window_s"] = round(window, 3)
        frames = row["processed_frames"] or 0
        row["processed_fps"] = round(frames / window, 2)
        if row["hailo_inferences"] is not None and prev.get("hailo_inferences") is not None:
            row["hailo_infer_per_s"] = round((row["hailo_inferences"] - prev["hailo_inferences"]) / window, 2)
    return row


def summarize_health(rows: list[dict]) -> dict:
    windows = [r for r in rows if "processed_fps" in r]
    out = {"polls": len(rows), "windows": len(windows)}
    for key in ("processed_fps", "camera_fps", "hailo_infer_per_s", "processed_p95_ms",
                "processed_max_ms", "temperature_c"):
        out[key] = hd.percentiles([r[key] for r in windows if r.get(key) is not None])
    out["stalls_500ms_total"] = sum(r.get("processed_stalls_500ms") or 0 for r in windows)
    out["stalls_1s_total"] = sum(r.get("processed_stalls_1s") or 0 for r in windows)
    out["states_seen"] = sorted({str(r.get("monitoring_state")) for r in rows})
    out["throttled_any"] = any(bool(r.get("throttle_now")) for r in rows)
    return out


def _fetch(url: str, token: Optional[str], timeout: float = 5.0) -> dict:
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"not an http(s) URL: {url}")
    req = urllib.request.Request(url, headers={"X-Device-Token": token} if token else {})  # noqa: S310 - scheme checked
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - scheme checked
        return json.loads(resp.read().decode())


def cmd_health(args) -> int:
    token = os.environ.get(args.token_env) or None
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    prev = None
    deadline = time.monotonic() + args.minutes * 60
    try:
        _fetch(args.url, token)
    except urllib.error.URLError as exc:
        if isinstance(getattr(exc, "reason", None), ConnectionRefusedError):
            # Nothing listening at all is not a data point, it is "the service is not
            # running": polling a dead port for --minutes measured nothing (M8.3).
            print(f"BLOCKED: nothing is listening at {args.url} ({exc.reason}). Start sensordyme.service "
                  "first; this measurement reads the running service's own health.", file=sys.stderr)
            return 2
    except Exception as exc:  # noqa: BLE001 - any other first-poll failure is recorded by the loop below
        print(f"first poll failed ({exc!r}); continuing: later polls record it", flush=True)
    with out.open("w") as fh:
        while time.monotonic() < deadline:
            try:
                payload = _fetch(args.url, token)
                row = health_row(payload, time.monotonic(), prev)
            except Exception as exc:  # noqa: BLE001 - a failed poll is a data point
                row = {"t": time.monotonic(), "error": repr(exc)}
            row["utc"] = hc.utc_now()
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if "error" not in row:
                rows.append(row)
                prev = row
                if "processed_fps" in row:
                    print(f"{row['utc']}  processed {row['processed_fps']:5.1f} fps  camera "
                          f"{row['camera_fps']}  p95 {row['processed_p95_ms']} ms  "
                          f"temp {row['temperature_c']}  state {row['monitoring_state']}", flush=True)
            else:
                print(f"{row['utc']}  poll failed: {row['error']}", flush=True)
            time.sleep(args.interval)
    summary = summarize_health(rows)
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


# ------------------------------------------------------------------ stages (harness)
class StageRecorder:
    """Per-frame stage durations (ms). Sub-stages may fire 0..n times per frame."""

    def __init__(self) -> None:
        self.frames: list[dict[str, float]] = []
        self._current: dict[str, float] = {}

    def add(self, stage: str, ms: float) -> None:
        self._current[stage] = self._current.get(stage, 0.0) + ms

    def end_frame(self) -> dict[str, float]:
        frame = self._current
        if "process_total" in frame:
            inner = sum(frame.get(k, 0.0) for k in ("detector_total", "facemesh", "hands"))
            frame["module_logic"] = max(0.0, frame["process_total"] - inner)
        if "detector_total" in frame and "hailo_infer" in frame:
            frame["preprocess"] = max(0.0, frame["detector_total"] - frame["hailo_infer"]
                                      - frame.get("decode", 0.0))
        self.frames.append(frame)
        self._current = {}
        return frame

    def summary(self) -> dict:
        out: dict = {"frames": len(self.frames)}
        for stage in STAGES:
            values = [f[stage] for f in self.frames if stage in f]
            out[stage] = hd.percentiles(values, qs=(0.5, 0.95, 0.99))
            if out[stage]:
                out[stage]["frames_with_stage"] = len(values)
        totals = [f["frame_total"] for f in self.frames if "frame_total" in f]
        if totals:
            out["sequential_fps"] = len(totals) / (sum(totals) / 1000.0)
        procs = [f["process_total"] for f in self.frames if "process_total" in f]
        if procs:
            out["processing_bound_fps"] = len(procs) / (sum(procs) / 1000.0)
        return out


def timed(fn, recorder: StageRecorder, stage: str):
    def wrapper(*a, **kw):
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            recorder.add(stage, (time.perf_counter() - t0) * 1000.0)

    wrapper.__hwv_timed__ = stage  # type: ignore[attr-defined]
    return wrapper


class _TimedProxy:
    """Delegates everything to `target`, timing one method."""

    def __init__(self, target, method: str, recorder: StageRecorder, stage: str) -> None:
        self._hwv_target = target
        self._hwv_method = timed(getattr(target, method), recorder, stage)
        self._hwv_name = method

    def __getattr__(self, name):
        if name == self._hwv_name:
            return self._hwv_method
        return getattr(self._hwv_target, name)


def instrument_detector(model, recorder: StageRecorder) -> bool:
    """Time predict / pipeline.infer / _decode_raw on a HailoYOLO-shaped object.

    Instance attributes shadow the class methods for THIS object only.
    Returns True if it is Hailo-shaped (infer + decode timed), False otherwise
    (only predict is timed: e.g. the CPU path on a Mac).
    """
    if getattr(model.predict, "__hwv_timed__", None):
        return hasattr(model, "_ensure_ready")
    model.predict = timed(model.predict, recorder, "detector_total")
    if not (hasattr(model, "_ensure_ready") and hasattr(model, "_decode_raw")):
        return False
    ensure = model._ensure_ready

    def ensure_ready_timed():
        pipeline = ensure()
        return _TimedProxy(pipeline, "infer", recorder, "hailo_infer")

    model._ensure_ready = ensure_ready_timed
    model._decode_raw = timed(model._decode_raw, recorder, "decode")
    return True


def instrument_module(module, recorder: StageRecorder) -> dict:
    """Wrap whatever the module has built so far. Idempotent; call every frame."""
    found = {}
    for attr, stage in (("_face_mesh", "facemesh"), ("_hands", "hands")):
        obj = getattr(module, attr, None)
        if obj is not None and not isinstance(obj, _TimedProxy):
            setattr(module, attr, _TimedProxy(obj, "process", recorder, stage))
        found[stage] = obj is not None
    model = getattr(module, "_yolo_model", None)
    found["detector"] = model is not None
    if model is not None:
        found["hailo"] = instrument_detector(model, recorder)
    return found


def cmd_stages(args) -> int:
    state = hc.service_state()
    ok, why = hc.service_gate(state, allow_unknown=args.no_service_check)
    print(f"service: {why}")
    if not ok:
        print("REFUSED.\n" + hc.operator_commands(), file=sys.stderr)
        return 2
    out = Path(args.out).resolve()
    ok, why = hc.ensure_outside(out, args.install_dir)
    if not ok:
        print(f"REFUSED: {why}", file=sys.stderr)
        return 2
    import yaml

    config_path = Path(args.config or (args.install_dir / "config.yaml")).resolve()
    config = yaml.safe_load(config_path.read_text()) or {}
    accel = dict(config.get("accelerator") or {})
    if accel.get("yolo_hef") and not Path(accel["yolo_hef"]).is_absolute():
        accel["yolo_hef"] = str(Path(args.install_dir) / accel["yolo_hef"])  # in memory only
    config["accelerator"] = accel
    if accel.get("yolo_hef"):
        actual = hc.sha256_file(accel["yolo_hef"])
        ok, why = hc.verify_pin(actual, accel.get("yolo_hef_sha256"))
        print(f"model: {why}")
        if not ok:
            print(f"REFUSED: {why}", file=sys.stderr)
            return 2
    module_name = args.module or config.get("module")
    video = Path(args.video).resolve() if args.video else None

    sys.path.insert(0, str(Path(args.code_root).resolve()))
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)  # any relative write by the module lands here, never in the install
    from benchmarks.run_benchmarks import configure_accelerator, warmup_frame_count
    from modules import get_module_class

    print(f"accelerator: {configure_accelerator(config).detail}")
    module_cfg = config["modules"][module_name]
    module = get_module_class(module_name)(module_cfg)
    recorder = StageRecorder()

    import cv2

    camera = cap = None
    if video:
        cap = cv2.VideoCapture(str(video))
    else:
        from core.camera import Camera

        camera = Camera(config.get("camera") or {})

    def read_frame():
        if cap is not None:
            ok_, frame_ = cap.read()
            return frame_ if ok_ else None
        return camera.read(timeout_s=5.0)

    warm = warmup_frame_count(module_cfg) + args.warmup
    found: dict = {}
    try:
        for _ in range(warm):
            frame = read_frame()
            if frame is None:
                break
            module.process(frame)  # not instrumented yet: warm-up records nothing
        deadline = time.monotonic() + args.minutes * 60
        while time.monotonic() < deadline and len(recorder.frames) < args.max_frames:
            f0 = time.perf_counter()
            frame = read_frame()
            captured_at = time.monotonic()  # the runtime stamps frames the same way (frame_bus)
            f1 = time.perf_counter()
            if frame is None:
                break
            found = instrument_module(module, recorder)
            recorder.add("capture", (f1 - f0) * 1000.0)
            setter = getattr(module, "set_frame_time", None)
            if callable(setter):
                setter(captured_at)
            module.process(frame)
            f2 = time.perf_counter()
            recorder.add("process_total", (f2 - f1) * 1000.0)
            recorder.add("frame_total", (f2 - f0) * 1000.0)
            recorder.end_frame()
    finally:
        if cap is not None:
            cap.release()
        if camera is not None:
            camera.release()
        close = getattr(module, "close", None)
        if callable(close):
            close()

    result = {"utc": hc.utc_now(), "module": module_name, "config_sha256": hc.sha256_file(config_path),
              "code_root": str(Path(args.code_root).resolve()),
              "git_commit": hc.git_commit(Path(args.code_root)),
              "source": str(video) if video else "camera", "instrumented": found,
              "camera_fps_configured": (config.get("camera") or {}).get("fps"),
              "yolo_every_n_frames": module_cfg.get("yolo_every_n_frames"),
              "summary": recorder.summary(),
              "environment": hc.environment()}
    (out / "stages.json").write_text(json.dumps(result, indent=2, default=str))
    with (out / "stages_frames.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(STAGES), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(recorder.frames)
    print(json.dumps(result["summary"], indent=2))
    if not found.get("hailo"):
        print("WARNING: the detector was not Hailo-shaped (or never built): no infer/decode split.")
    return 0


# ------------------------------------------------------------------ latency (pure)
def analyse_trials(rows: list[dict]) -> dict:
    totals, pipeline = [], []
    for r in rows:
        total = float(r["alert_onset_ms"]) - float(r["stimulus_ms"])
        totals.append(total)
        if r.get("designed_delay_ms") not in (None, ""):
            pipeline.append(total - float(r["designed_delay_ms"]))
    return {"trials": len(rows), "stimulus_to_alert_ms": hd.percentiles(totals),
            "minus_designed_delay_ms": hd.percentiles(pipeline)}


def analyse_glass(rows: list[dict]) -> dict:
    lat = [float(r["captured_ms"]) - float(r.get("offset_ms") or 0.0) - float(r["displayed_ms"])
           for r in rows]
    return {"frames": len(rows), "glass_to_capture_ms": hd.percentiles(lat)}


def cmd_latency(args) -> int:
    result = {}
    if args.trials:
        with open(args.trials, newline="") as fh:
            result["trials"] = analyse_trials(list(csv.DictReader(fh)))
    if args.glass:
        with open(args.glass, newline="") as fh:
            result["glass"] = analyse_glass(list(csv.DictReader(fh)))
    print(json.dumps(result, indent=2))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Processed FPS / per-stage / end-to-end latency")
    sub = parser.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("health", help="poll the running service's /api/health (read-only)")
    h.add_argument("--url", default="http://127.0.0.1:8899/api/health")
    h.add_argument("--token-env", default="SENSORDYME_TOKEN",
                   help="env var holding the dashboard token (never printed)")
    h.add_argument("--interval", type=float, default=5.0)
    h.add_argument("--minutes", type=float, default=5.0)
    h.add_argument("--out", required=True, help="JSONL output path")
    h.set_defaults(func=cmd_health)

    s = sub.add_parser("stages", help="per-stage harness (service must be stopped)")
    s.add_argument("--install-dir", type=Path, default=hc.DEFAULT_INSTALL_DIR)
    s.add_argument("--config", default=None)
    s.add_argument("--code-root", default=str(hc.REPO_ROOT))
    s.add_argument("--module", default=None)
    s.add_argument("--video", default=None, help="video file instead of the live camera")
    s.add_argument("--minutes", type=float, default=2.0)
    s.add_argument("--max-frames", type=int, default=100000)
    s.add_argument("--warmup", type=int, default=10)
    s.add_argument("--out", required=True, help="output DIRECTORY (also the working dir)")
    s.add_argument("--no-service-check", action="store_true")
    s.set_defaults(func=cmd_stages)

    lat = sub.add_parser("latency", help="summarise annotated phone-video trials")
    lat.add_argument("--trials", default=None)
    lat.add_argument("--glass", default=None)
    lat.set_defaults(func=cmd_latency)

    args = parser.parse_args(argv)
    # Lazily imported code may come from the INSTALLED tree (--code-root): leave no
    # __pycache__ behind in it. Set here, not at import, so importing this file has
    # no process-wide side effect.
    sys.dont_write_bytecode = True
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(hc.run_cli(main))
