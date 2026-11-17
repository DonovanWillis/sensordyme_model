#!/usr/bin/env python3
"""The legacy-vs-reference decoder experiment as ONE run and ONE evidence directory (M14).

    $PY $V/hwv_stage.py decoder-exp --install-dir $D [--package ~/transfer-<id>] [--images DIR]
    python hwv_stage.py decoder-exp --fake            # Mac dry run: every phase against stand-ins, SIMULATED

Not a second validation system: a thin composition of the existing stages and
scripts, in this order, each writing its usual report under <evidence>/stages/:

  0. refusals before the chip is touched: service stopped, nothing holds
     /dev/hailo0, the production .hef's sha256 == config pin == PRODUCTION_HEF_SHA256;
  1. `pins`, `model`, `parse-hef` (hwv_stage.py);
  2. `capture` (capture_hailo_outputs.py) on the known-good images -> <evidence>/capture/,
     plus, optionally, an operator image set (--images) or live frames
     (--camera-frames N --allow-saving-frames) -> <evidence>/capture_extra/;
  3. `parity --decoder both` (INT8 vs float, per decoder, on the SAME captured tensors);
  4. same-tensor decoding (decoders.json): the product's own decoders (core.accel
     `_decode_raw`, device-free) and hwv_decode's offline specs on every captured
     INT8 tensor; class map; the dropped hand channel 80; decode time on this CPU;
  5. threshold-calibration readiness (calibration.json): per-class score
     distributions, counts at the production gates, count-matched thresholds and,
     with human-verified --labels, precision/recall at matched operating points.
     Read-only: nothing here changes a threshold;
  6. `latency --latency-decoder both` (stage_timing instrumentation; ABAB);
  7. pipeline fps (pipeline.json): the production module end to end, legacy and
     reference arms in ABBA blocks (stage_timing.run_block), same frame source and
     one camera configuration, every block started in the same thermal state;
  8. temperature / throttling sampled through every phase (thermal.csv);
  9. the decision gate (DECISION.md, evidence.json): KEEP LEGACY unless every
     criterion passes, and even then only "eligible for owner review".

It never writes config.yaml (hashed before and after; a change is a FAIL), never
starts or stops the service, and never changes a threshold or the decoder:
the reference decoder runs only inside this process.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import hwv_common as hc  # noqa: E402
import hwv_decode as hd  # noqa: E402

#: The shipping DAMO-YOLO .hef (setup_pi.sh MODEL_SHA256, core/model_registry.yaml; pinned by a test).
PRODUCTION_HEF_SHA256 = "e69136d3ebf3cc8ebaf3da5417a59962b139f3d3d67d715e9a9b1ae10963960e"
PERSON, PHONE, HAND_CHANNEL = 0, 67, 80
#: The two decoders compared: the product's legacy and the product's OWN reference (not upstream DAMO's).
SPECS = {"legacy": hd.LEGACY, "reference": hd.DEVICE_REFERENCE}
LABELS_FORMAT = "sensordyme.hwval.image_labels/1"
#: Provisional engineering tolerances, chosen before any hardware run (not product thresholds).
GATE = {"parity_score_delta_p95_slack": 0.02, "pipeline_fps_ratio_min": 0.97, "late_fraction_max": 0.01}
NOT_ACCURACY = ("A handful of images and one bench run do not prove real-world accuracy: the known-good set "
                "has no people and no phones, and in-cabin accuracy needs consented, independently labelled, "
                "subject-separated sessions past the scenario matrix's power check.")


# ------------------------------------------------------------------ thermal
def _vc(*args: str) -> Optional[str]:
    return hc.run_text(["vcgencmd", *args], timeout=3) if shutil.which("vcgencmd") else None


def parse_vcgencmd(temp: Optional[str], throttled: Optional[str], clock: Optional[str]) -> dict:
    """`temp=48.3'C`, `throttled=0x50000`, `frequency(0)=2400000000` -> numbers (None when absent)."""
    t = re.search(r"temp=([\d.]+)", temp or "")
    th = re.search(r"throttled=(0x[0-9a-fA-F]+)", throttled or "")
    hz = re.search(r"=(\d+)", clock or "")
    raw = th.group(1).lower() if th else None
    value = int(raw, 16) if raw else None
    return {"soc_temp_c": float(t.group(1)) if t else None, "throttled": raw,
            "throttle_now_bits": None if value is None else value & 0xF,
            "throttle_occurred_bits": None if value is None else (value >> 16) & 0xF,
            "arm_hz": int(hz.group(1)) if hz else None}


def hailo_chip_temp(model=None) -> dict:
    """The Hailo die temperature, read-only, or why it is unavailable.

    In-process through a model this process already holds (HailoRT
    Device.control.get_chip_temperature(); NOT YET RUN ON HARDWARE), else the
    operator's HAILO_TEMP_CMD (thermal_soak.sh's convention: a command verified
    safe for this HailoRT version that prints one number in C), run only while
    this process holds no VDevice."""
    import os

    if model is not None:
        try:
            device = model._vdevice.get_physical_devices()[0]
            reading = device.control.get_chip_temperature()
            return {"source": "hailort_control", "ts0_c": float(reading.ts0_temperature),
                    "ts1_c": float(reading.ts1_temperature)}
        except Exception as exc:  # noqa: BLE001 - informational; never disturbs the run
            return {"source": "unavailable", "why": f"in-process read failed: {exc!r}"[:200]}
    cmd = os.environ.get("HAILO_TEMP_CMD")
    if not cmd:
        return {"source": "unavailable", "why": "no model held and HAILO_TEMP_CMD not set"}
    out = hc.run_text(["bash", "-c", cmd], timeout=10) or ""
    m = re.search(r"-?\d+(\.\d+)?", out)
    return {"source": "HAILO_TEMP_CMD", "c": float(m.group(0)) if m else None, "raw": out[-200:]}


class ThermalMonitor:
    """SoC temperature, throttle state and ARM clock every `interval_s` on a background
    thread (vcgencmd is a separate process: the inference loop is never touched), plus
    one sample at every phase mark, appended to a CSV. Off a Pi every value is empty
    and `available` is False."""

    COLUMNS = ("utc", "elapsed_s", "phase", "soc_temp_c", "throttled", "arm_hz", "hailo_temp_c", "hailo_source")

    def __init__(self, path: Path, interval_s: float = 5.0, reader=None) -> None:
        self.path, self.interval_s = path, interval_s
        self._reader = reader or (lambda: parse_vcgencmd(_vc("measure_temp"), _vc("get_throttled"),
                                                         _vc("measure_clock", "arm")))
        self.rows: list[dict] = []
        self.phase = "start"
        self._t0 = time.monotonic()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def sample(self, hailo: Optional[dict] = None) -> dict:
        reading = self._reader()
        hailo = hailo or {}
        row = {"utc": hc.utc_now(), "elapsed_s": round(time.monotonic() - self._t0, 2), "phase": self.phase,
               **reading,
               "hailo_temp_c": hailo.get("ts0_c", hailo.get("c")), "hailo_source": hailo.get("source")}
        with self._lock:
            self.rows.append(row)
        return row

    def mark(self, phase: str, hailo: Optional[dict] = None) -> None:
        self.phase = phase
        self.sample(hailo)

    def start(self) -> ThermalMonitor:
        def loop():
            while not self._stop.wait(self.interval_s):
                self.sample()

        self.sample()
        self._thread = threading.Thread(target=loop, name="hwv-thermal", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        self.mark("end")
        with self.path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)

    def summary(self) -> dict:
        rows = list(self.rows)
        temps = [r["soc_temp_c"] for r in rows if r.get("soc_temp_c") is not None]
        arm = [r["arm_hz"] for r in rows if r.get("arm_hz") is not None]
        occurred = [r["throttle_occurred_bits"] for r in rows if r.get("throttle_occurred_bits") is not None]
        now = [r["throttle_now_bits"] for r in rows if r.get("throttle_now_bits") is not None]
        hailo = [r["hailo_temp_c"] for r in rows if r.get("hailo_temp_c") is not None]
        return {"rows": len(rows), "available": bool(temps or now),
                "phases": sorted({r["phase"] for r in rows}),
                "soc_temp_c_first": temps[0] if temps else None, "soc_temp_c_peak": max(temps) if temps else None,
                "throttled_values": sorted({r["throttled"] for r in rows if r.get("throttled")}),
                "throttled_now_any": any(now), "throttle_newly_occurred": bool(occurred) and any(
                    o & ~occurred[0] for o in occurred),
                "arm_hz_first": arm[0] if arm else None, "arm_hz_min": min(arm) if arm else None,
                "hailo_temp_c_peak": max(hailo) if hailo else None,
                "hailo_temp_sources": sorted({r["hailo_source"] for r in rows if r.get("hailo_source")})}


# ------------------------------------------------------------------ same-tensor decoding
def decoder_only_model(scales: list[dict], decoder: str):
    """core.accel.HailoYOLO with NO device, holding exactly the state its decoders read
    (the production decode code, on tensors already captured; as tests/test_m10_hand_class_decode.py)."""
    sys.path.insert(0, str(hc.REPO_ROOT))
    from core.accel import HailoYOLO

    model = HailoYOLO.__new__(HailoYOLO)
    model.decoder, model._nms_iou = decoder, HailoYOLO._DEFAULT_NMS_IOU
    model._set_extra_classes(None, sha256=None)  # production: no extra-class map; channel 80 dropped
    model._scales = [dict(s) for s in scales]
    return model


def _product_rows(model, outs: dict, geometry: dict, classes, conf: float) -> list[dict]:
    (res,) = model._decode_raw(outs, geometry["scale"], geometry["left"], geometry["top"], geometry["frame_w"],
                               geometry["frame_h"], classes, conf)
    return [{"cls": int(b.cls[0]), "score": float(b.conf[0]), "xyxy": [float(v) for v in b.xyxy[0]]}
            for b in res.boxes]


def _same(product: list[dict], offline: hd.Detections) -> bool:
    a = sorted((r["cls"], round(r["score"], 4), *np.round(r["xyxy"], 2)) for r in product)
    b = sorted((int(c), round(float(s), 4), *np.round(x, 2))
               for x, s, c in zip(offline.boxes, offline.scores, offline.classes, strict=True))
    return len(a) == len(b) and all(x[:2] == y[:2] and np.allclose(x[2:], y[2:], atol=0.01)
                                    for x, y in zip(a, b, strict=True))


def letterbox_samples(bundles: list[Path]) -> list[tuple[Path, dict, dict]]:
    out = []
    for bundle in bundles:
        manifest = hc.load_bundle(bundle)
        for s in manifest["samples"]:
            if s["mode"] == "letterbox":
                out.append((bundle, manifest, s))
    return out


def same_tensor_decode(bundles: list[Path], *, prod_classes: list[int], prod_conf: float, floor: float,
                       repeats: int = 10) -> tuple[dict, dict[str, dict[str, list[dict]]]]:
    """Both decoders on the SAME captured INT8 tensors. Returns (report, detections at the
    floor per decoder per image) -- the latter feeds the calibration readiness."""
    samples = letterbox_samples(bundles)
    per_image: dict[str, dict] = {}
    dets: dict[str, dict[str, list[dict]]] = {d: {} for d in SPECS}
    times: dict[str, list[float]] = {d: [] for d in SPECS}
    agreement = hd.Agreement(iou_threshold=0.5)
    problems: list[str] = []
    for k, (bundle, manifest, s) in enumerate(samples):
        _inp, outs = hc.load_sample(bundle, s)
        scale_dicts = manifest["hef"]["scales"]
        scales = [hd.Scale(**{f: d[f] for f in ("grid", "stride", "box", "cls", "num_classes")}) for d in scale_dicts]
        geometry = s["geometry"]
        image = f"{bundle.name}/{s['image_id']}"
        row: dict[str, Any] = {"bundle": bundle.name, "image_id": s["image_id"],
                               "channel80_max": max(float(np.asarray(outs[sc.cls])[..., HAND_CHANNEL].max())
                                                    for sc in scales if sc.num_classes > HAND_CHANNEL)
                               if any(sc.num_classes > HAND_CHANNEL for sc in scales) else None}
        at_gate: dict[str, hd.Detections] = {}
        for dec, spec in SPECS.items():
            model = decoder_only_model(scale_dicts, dec)
            product = _product_rows(model, outs, geometry, None, floor)
            offline = hd.unmap(hd.decode(outs, scales, spec, floor), geometry)
            same = _same(product, offline)
            if not same:
                problems.append(f"{image}: core.accel {dec} != hwv_decode {spec.name}")
            emitted = sorted({r["cls"] for r in product})
            dets[dec][image] = product
            gate = _product_rows(model, outs, geometry, prod_classes, prod_conf)
            at_gate[dec] = hd.Detections(np.array([r["xyxy"] for r in gate], float).reshape(-1, 4),
                                         np.array([r["score"] for r in gate], float),
                                         np.array([r["cls"] for r in gate], np.int64), np.zeros(len(gate), np.int64))
            row[dec] = {"product_equals_offline": same, "at_floor": len(product), "classes_emitted": emitted,
                        "class_ge_80_emitted": [c for c in emitted if c >= hd.NUM_CLASSES],
                        "at_production_query": len(gate),
                        "person_at_gate": sum(1 for r in gate if r["cls"] == PERSON),
                        "phone_at_gate": sum(1 for r in gate if r["cls"] == PHONE)}
        agreement.add(at_gate["legacy"], at_gate["reference"])
        # Decode time on THIS CPU, interleaved (ABAB...) so drift lands on both.
        models = {d: decoder_only_model(scale_dicts, d) for d in SPECS}
        for r in range(repeats):
            for dec in (list(SPECS) if (r + k) % 2 == 0 else list(reversed(SPECS))):
                t0 = time.perf_counter()
                models[dec]._decode_raw(outs, geometry["scale"], geometry["left"], geometry["top"],
                                        geometry["frame_w"], geometry["frame_h"], prod_classes, prod_conf)
                times[dec].append((time.perf_counter() - t0) * 1000.0)
        per_image[image] = row
    report = {
        "samples": len(samples), "floor_conf": floor,
        "production_query": {"classes": prod_classes, "conf": prod_conf,
                             "note": "the driver_attention module's own query: its yolo_classes + person, "
                                     "at yolo_confidence_threshold (person is gated again at "
                                     "person_confidence_threshold by the module)"},
        "decoders": {d: dataclass_dict(spec) for d, spec in SPECS.items()},
        "product_equals_offline_all": bool(samples) and not problems, "problems": problems,
        "class_ge_80_emitted_any": any(row[d]["class_ge_80_emitted"] for row in per_image.values() for d in SPECS),
        "channel80_max": max((r["channel80_max"] for r in per_image.values() if r["channel80_max"] is not None),
                             default=None),
        "decode_ms_this_cpu": {d: hd.percentiles(v, qs=(0.5, 0.95)) for d, v in times.items()},
        "legacy_vs_reference_at_production_query": agreement.summary(),
        "per_image": per_image,
    }
    return report, dets


def dataclass_dict(spec: hd.DecoderSpec) -> dict:
    import dataclasses

    return dataclasses.asdict(spec)


# ------------------------------------------------------------------ calibration readiness
def load_labels(path: Path) -> dict:
    """Human-verified boxes for the captured images: {"format", "verified_by", "method",
    "images": {"<bundle>/<image_id>": [{"cls", "xyxy"}]}}. Refused unless it names who
    verified it: a model's own output is never ground truth."""
    doc = json.loads(Path(path).read_text())
    if doc.get("format") != LABELS_FORMAT:
        raise SystemExit(f"REFUSED: {path} is not {LABELS_FORMAT}")
    if not str(doc.get("verified_by") or "").strip() or not str(doc.get("method") or "").strip():
        raise SystemExit(f"REFUSED: {path} must name `verified_by` (a pseudonym) and `method`")
    return doc


def _scored(dets: dict[str, list[dict]], labels: dict, cls: int) -> tuple[list[tuple[float, bool]], int]:
    """(score, true positive) per detection of `cls`, greedy by score at IoU 0.5; and the number of positives."""
    scored: list[tuple[float, bool]] = []
    npos = 0
    for image, gts in labels.items():
        gt = np.array([g["xyxy"] for g in gts if int(g["cls"]) == cls], float).reshape(-1, 4)
        npos += len(gt)
        rows = sorted((r for r in dets.get(image, []) if r["cls"] == cls), key=lambda r: -r["score"])
        used = np.zeros(len(gt), bool)
        for r in rows:
            ious = hd.box_iou(np.array([r["xyxy"]]), gt)[0] if len(gt) else np.zeros(0)
            ious = np.where(used, -1.0, ious)
            j = int(ious.argmax()) if len(ious) else -1
            hit = j >= 0 and ious[j] >= 0.5
            if hit:
                used[j] = True
            scored.append((r["score"], bool(hit)))
    return scored, npos


def _pr_at(scored: list[tuple[float, bool]], npos: int, thr: float) -> dict:
    kept = [hit for s, hit in scored if s >= thr]
    tp = sum(kept)
    return {"threshold": thr, "kept": len(kept), "tp": tp, "fp": len(kept) - tp,
            "precision": tp / len(kept) if kept else None, "recall": tp / npos if npos else None}


def calibration_readiness(dets: dict[str, dict[str, list[dict]]], gates: dict[int, float], floor: float,
                          labels: Optional[dict] = None) -> dict:
    """What the captured set says about thresholds, WITHOUT changing one."""
    bins = np.round(np.arange(floor, 1.0 + 1e-9, 0.05), 2).tolist()
    out: dict[str, Any] = {
        "statement": ("readiness only: no threshold is proposed or changed. Production gates stay as configured "
                      "until thresholds are re-derived on a labelled, subject-separated INT8 set "
                      "(docs/research/M101_HARDWARE_GATES.md §2 item 4) and the owner decides."),
        "images": sorted({i for d in dets.values() for i in d}), "floor_conf": floor, "classes": {}}
    for cls, gate in gates.items():
        entry: dict[str, Any] = {"production_gate": gate, "decoders": {}}
        for dec, per_image in dets.items():
            scores = sorted((r["score"] for rows in per_image.values() for r in rows if r["cls"] == cls),
                            reverse=True)
            hist, _ = np.histogram(scores, bins=[*bins, 1.0 + 1e-9]) if scores else (np.zeros(len(bins)), None)
            entry["decoders"][dec] = {"n_at_floor": len(scores), "n_at_gate": sum(1 for s in scores if s >= gate),
                                      "score_quantiles": hd.percentiles(scores, qs=(0.5, 0.9)),
                                      "histogram": {"bin_lo": bins, "count": [int(c) for c in hist]}}
        legacy_n = entry["decoders"]["legacy"]["n_at_gate"]
        ref_scores = sorted((r["score"] for rows in dets["reference"].values() for r in rows if r["cls"] == cls),
                            reverse=True)
        entry["count_matched_reference_threshold"] = {
            "value": ref_scores[legacy_n - 1] if 0 < legacy_n <= len(ref_scores) else None,
            "meaning": "the reference score that keeps as many boxes as legacy keeps at the production gate on "
                       "THIS captured set (a matched operating point for comparison, not a proposal)"}
        if labels is not None:
            pr: dict[str, Any] = {}
            scored_by = {dec: _scored(per_image, labels["images"], cls) for dec, per_image in dets.items()}
            for dec, (scored, npos) in scored_by.items():
                pr[dec] = {"positives": npos, "at_production_gate": _pr_at(scored, npos, gate)}
            legacy_p = pr["legacy"]["at_production_gate"]["precision"]
            ref_scored, npos = scored_by["reference"]
            matched = None
            if legacy_p is not None and npos:
                candidates = [_pr_at(ref_scored, npos, s) for s, _ in ref_scored]
                ok = [c for c in candidates if c["precision"] is not None and c["precision"] >= legacy_p - 1e-12]
                matched = max(ok, key=lambda c: (c["recall"], c["threshold"])) if ok else None
            pr["reference_at_legacy_precision"] = matched
            entry["labelled"] = pr
        out["classes"][str(cls)] = entry
    if labels is not None:
        out["labels"] = {"verified_by": labels.get("verified_by"), "method": labels.get("method"),
                         "simulated": bool(labels.get("simulated")), "images": len(labels.get("images") or {})}
    else:
        out["labels"] = None
        out["labelled_note"] = ("no --labels: precision/recall not computed. Score distributions alone cannot set "
                                "a threshold")
    return out


# ------------------------------------------------------------------ pipeline fps (both decoders)
def pipeline_fps(args, config: dict, install_dir: Path, thermal: ThermalMonitor) -> dict:
    """The production module end to end with each decoder, ABBA blocks, same frame source
    and camera configuration, each block started in the same thermal state. The decoder is
    selected for THIS process only (accel.configure on an in-memory copy)."""
    import hwv_stage as hs
    import stage_timing as st

    sys.path.insert(0, str(hc.REPO_ROOT))
    from core import accel, detectors
    from modules import get_module_class

    hef, pin = hs.hef_path_and_pin(install_dir, config)
    module_name = config.get("module") or "driver_attention"
    module_cfg = dict((config.get("modules") or {}).get(module_name) or {})
    every = max(1, int(module_cfg.get("yolo_every_n_frames", 3)))
    camera = dict(config.get("camera") or {})
    fps = float(camera.get("fps", 30))
    seconds = min(args.block_seconds, 2.0) if args.fake else float(args.block_seconds)
    order = list("ABBA") * (1 if args.fake else max(1, args.rounds))
    arms = {"A": "legacy", "B": "reference"}
    base = dict(config.get("accelerator") or {})
    base["yolo_hef"] = str(hef)
    info = (accel.AcceleratorInfo(True, True, "hailo", "SIMULATED Hailo (fake_hailo)")
            if getattr(sys.modules.get("hailo_platform"), "SIMULATED", False) else None)
    frame_args = argparse.Namespace(video=args.video, fake=args.fake)
    reset, next_frame, close_frames = hs._arch_frames(frame_args, config)
    source = "known-good images (SIMULATED)" if args.fake else (args.video or "live camera (frames never saved)")
    blocks: list[dict] = []
    target = thermal.sample().get("soc_temp_c")
    try:
        for index, arm in enumerate(order):
            decoder = arms[arm]
            thermal.phase = f"pipeline:{index}:{decoder}"
            before = st.wait_for_thermal_state(target, args.temp_tolerance_c, 0.0 if args.fake else args.cooldown_max_s)
            reset()
            acc = copy.deepcopy(base)
            acc["yolo_decoder"] = decoder  # this process only; config.yaml is never written
            accel.configure(acc, info=info)
            seen, chip_temp = None, {"source": "unavailable", "why": "no model loaded"}
            with contextlib.ExitStack() as stack:
                module = get_module_class(module_name)(module_cfg)
                stack.callback(detectors.release_all)
                stack.callback(getattr(module, "close", lambda: None))
                result = st.run_block(module=module, second=None, next_frame=next_frame, every=every, fps=fps,
                                      seconds=seconds, warmup=max(2, every + 1) + args.warmup)
                model = getattr(module, "_yolo_model", None)
                seen = getattr(model, "decoder", None)
                if model is not None and not args.fake:
                    chip_temp = hailo_chip_temp(model)
            accel.release_hailo_models()
            thermal.mark(f"pipeline:{index}:{decoder}:end", chip_temp)
            s = result["summary"]
            result.update({"block": index, "arm": arm, "decoder": decoder, "decoder_seen": seen,
                           "thermal_start": before, "hailo_temp_end": chip_temp,
                           "late_fraction": round(result["late_frames"] / result["frames"], 4)
                           if result["frames"] else None})
            blocks.append(result)
            print(f"  pipeline block {index} {decoder}: {result['achieved_fps']} fps, late {result['late_frames']}/"
                  f"{result['frames']}, detector p95 {(s.get('detector_total') or {}).get('p95')} ms, "
                  f"decode p95 {(s.get('decode') or {}).get('p95')} ms", flush=True)
    finally:
        close_frames()
        detectors.release_all()
        accel.release_hailo_models()
    per_decoder = {}
    for decoder in arms.values():
        bs = [b for b in blocks if b["decoder"] == decoder]
        fpss = [b["achieved_fps"] for b in bs if b["achieved_fps"] is not None]
        late = sum(b["late_frames"] for b in bs)
        frames = sum(b["frames"] for b in bs)
        dec_p95 = [(b["summary"].get("decode") or {}).get("p95") for b in bs]
        per_decoder[decoder] = {"blocks": len(bs), "achieved_fps_mean": round(float(np.mean(fpss)), 2) if fpss else None,
                                "frames": frames, "late_fraction": round(late / frames, 4) if frames else None,
                                "decode_p95_ms_mean": round(float(np.mean([v for v in dec_p95 if v is not None])), 3)
                                if any(v is not None for v in dec_p95) else None,
                                "decoder_seen_ok": all(b["decoder_seen"] == decoder for b in bs)}
    return {"conditions": {"module": module_name, "camera_config": camera, "camera_fps": fps,
                           "detector_stride_frames": every, "block_seconds": seconds, "order": "".join(order),
                           "frames": source, "one_camera_configuration_for_every_block": True,
                           "hef_sha256_pin": pin, "temp_tolerance_c": args.temp_tolerance_c},
            "blocks": blocks, "per_decoder": per_decoder}


# ------------------------------------------------------------------ the gate
def _criterion(cid: str, name: str, ok: Optional[bool], detail: Any, provisional: bool = False) -> dict:
    return {"id": cid, "criterion": name, "status": "PASS" if ok else ("NOT EVALUATED" if ok is None else "FAIL"),
            "detail": detail, "provisional": provisional}


def _checks(rep, prefix: str = "") -> list[dict]:
    return [c for c in (rep.checks if rep else []) if c["check"].startswith(prefix)]


def cabin_comparison(paths: Optional[list[Path]]) -> tuple[Optional[bool], Any]:
    """C11: two cabin_session.py score files (legacy run, reference run) from verified labels."""
    if not paths:
        return None, ("not provided: run the same consented cabin_session preset once per decoder (reference on a "
                      "development config, dev_config.py) and score both with --labels; pass "
                      "--cabin-scores LEGACY.json,REFERENCE.json")
    docs = [json.loads(Path(p).read_text()) for p in paths]
    bad = [str(p) for p, d in zip(paths, docs, strict=True)
           if d.get("simulated") or d.get("evidence_level") != "verified_labels"]
    if bad:
        return False, f"not verified-label, non-simulated scores: {bad}"

    def totals(d: dict) -> tuple[float, int, float]:
        alerts = d.get("alerts") or {}
        false = sum(a.get("false_alerts") or 0 for a in alerts.values())
        hours = max((a.get("exposure_min") or 0) for a in alerts.values()) / 60 if alerts else 0.0
        caught = sum(e.get("caught") or 0 for e in (d.get("per_event") or {}).values())
        return false, caught, hours

    (lf, lc, lh), (rf, rc, rh) = totals(docs[0]), totals(docs[1])
    lph, rph = (lf / lh if lh else None), (rf / rh if rh else None)
    ok = lph is not None and rph is not None and rph <= lph and rc >= lc
    return ok, {"legacy": {"false_per_hour": lph, "caught": lc}, "reference": {"false_per_hour": rph, "caught": rc},
                "note": "one session per decoder is diagnostics (cabin_session.py); the comparison is necessary, "
                        "never sufficient, for a claim"}


def decision_gate(ev: dict, reports: dict, fake: bool) -> dict:
    prov, thermal, decoders, pipeline = ev["provenance"], ev["thermal"], ev.get("decoders") or {}, ev.get("pipeline")
    hef = prov["hef"]
    c: list[dict] = []
    pins, phef = reports.get("pins"), reports.get("parse-hef")
    c.append(_criterion("C0", "pins agree (config / setup_pi.sh / model_registry.yaml), installed decoder legacy; "
                        "parse-hef: HAILO8L, six raw outputs, no on-chip NMS",
                        bool(pins) and pins.passed and bool(phef) and phef.passed,
                        [x["check"] for x in _checks(pins) + _checks(phef) if not x["ok"]] or "all ok"))
    c.append(_criterion("C1", "production .hef sha256 == config pin == PRODUCTION_HEF_SHA256, unchanged after the run",
                        hef["sha256_before"] == hef["pin"] == hef["expected"] == hef["sha256_after"],
                        {k: (v or "")[:16] for k, v in hef.items() if k != "path"}))
    cap = reports.get("capture")
    c.append(_criterion("C2", "capture complete: six heads, class heads in [0,1], inputs bit-identical to the float run",
                        bool(cap) and cap.passed, [x["check"] for x in _checks(cap) if not x["ok"]] or "all ok"))
    par = reports.get("parity")
    prod_eq = [x for x in _checks(par) if x["check"].startswith("production decoder == offline legacy")]
    c.append(_criterion("C3", "the device's production decoder == offline legacy on the device tensors",
                        bool(prod_eq) and all(x["ok"] for x in prod_eq), [x["detail"] for x in prod_eq]))
    c.append(_criterion("C4", "core.accel reference decoder == hwv_decode.DEVICE_REFERENCE on every captured tensor "
                        "(and legacy == LEGACY)", decoders.get("product_equals_offline_all"),
                        decoders.get("problems") or decoders.get("samples")))
    model_rep = reports.get("model")
    class_map = [x for x in _checks(model_rep) if x["check"].startswith("class map")]
    c.append(_criterion("C5", "class map 0=person, 67=cell phone; no decoder emits a class >= 80 (hand channel 80 "
                        "dropped)", bool(class_map) and all(x["ok"] for x in class_map)
                        and decoders.get("class_ge_80_emitted_any") is False,
                        {"channel80_max": decoders.get("channel80_max")}))
    ref_checks, leg = _checks(par, "[reference] "), (par.values.get("decoders") or {}) if par else {}

    def errors(dec: str) -> Optional[int]:
        per = (leg.get(dec) or {}).get("per_image")
        return None if per is None else sum(len(v.get("missed_or_moved", [])) + len(v.get("ghosts", []))
                                            for v in per.values() if not v.get("missing"))

    def p95(dec: str) -> Optional[float]:
        return ((leg.get(dec) or {}).get("matched_score_delta") or {}).get("p95")

    ref_err, leg_err = errors("reference"), errors("legacy")
    no_worse = (ref_err is not None and leg_err is not None and ref_err <= leg_err
                and (p95("reference") or 0.0) <= (p95("legacy") or 0.0) + GATE["parity_score_delta_p95_slack"])
    c.append(_criterion("C6", "INT8-vs-float: reference passes its own parity checks AND is no worse than legacy "
                        f"(errors <=, |score delta| p95 within +{GATE['parity_score_delta_p95_slack']})",
                        bool(ref_checks) and all(x["ok"] for x in ref_checks) and no_worse,
                        {"errors": {d: errors(d) for d in SPECS}, "score_delta_p95": {d: p95(d) for d in SPECS}},
                        provisional=True))
    lat = reports.get("latency")
    lat_ref = [x for x in _checks(lat) if x["check"].startswith("[reference decoder]")
               or x["check"].startswith("the detector is the Hailo path")]
    c.append(_criterion("C7", "reference predict() p95 inside the production stride budget, on the Hailo path",
                        len(lat_ref) == 2 and all(x["ok"] for x in lat_ref),
                        {"decode_p95_ms": {d: ((lat.values.get("burst_by_decoder") or {}).get(d, {}).get("decode")
                                               or {}).get("p95") for d in SPECS} if lat else None}))
    if pipeline:
        pd = pipeline["per_decoder"]
        lf, rf = pd["legacy"]["achieved_fps_mean"], pd["reference"]["achieved_fps_mean"]
        ok = (lf is not None and rf is not None and rf >= GATE["pipeline_fps_ratio_min"] * lf
              and (pd["reference"]["late_fraction"] or 0) < GATE["late_fraction_max"]
              and pd["legacy"]["decoder_seen_ok"] and pd["reference"]["decoder_seen_ok"])
        c.append(_criterion("C8", f"end-to-end fps with reference >= {GATE['pipeline_fps_ratio_min']} x legacy, < "
                            f"{GATE['late_fraction_max']:.0%} late, each block ran the decoder it claims",
                            ok, {"fps": {"legacy": lf, "reference": rf}}, provisional=True))
    else:
        c.append(_criterion("C8", "end-to-end fps per decoder", False, ev.get("pipeline_error") or "not run"))
    c.append(_criterion("C9", "never throttled during the run (no live throttle bit, no new 'occurred' bit)",
                        thermal["available"] and not thermal["throttled_now_any"]
                        and not thermal["throttle_newly_occurred"],
                        {k: thermal[k] for k in ("available", "throttled_values", "soc_temp_c_peak",
                                                 "hailo_temp_c_peak")}))
    real = (not fake) and "raspberry pi" in str(prov["host"].get("board") or "").lower() \
        and not prov.get("hailo_simulated")
    c.append(_criterion("C10", "measured on a Raspberry Pi with a real Hailo device (not --fake)", real,
                        {"board": prov["host"].get("board"), "simulated": fake}))
    c.append(_criterion("C11", "in-cabin event-level comparison (DECODER_CANDIDATE.md 'To promote' #3): reference "
                        "false alerts/h <= legacy and caught >= legacy, from verified labels",
                        *cabin_comparison(ev.get("cabin_scores"))))
    c.append(_criterion("C12", "config.yaml unchanged by this run (sha256 before == after)",
                        prov["config"]["sha256_before"] == prov["config"]["sha256_after"],
                        prov["config"]["sha256_before"][:16]))
    bench = [x for x in c if x["id"] not in ("C11",)]
    bench_ok = all(x["status"] == "PASS" for x in bench)
    everything = bench_ok and all(x["status"] == "PASS" for x in c)
    if everything:
        decision = ("ELIGIBLE FOR OWNER REVIEW -- production stays LEGACY. The reference decoder may be proposed for "
                    "a DEVELOPMENT unit only after the owner approves in writing; this tool changed no config")
    else:
        failed = [x["id"] for x in c if x["status"] != "PASS"]
        decision = f"KEEP LEGACY (criteria not passed: {', '.join(failed)})"
    return {"decision": decision, "keep_legacy": not everything, "bench_criteria_passed": bench_ok,
            "criteria": c, "owner_approval_required": True, "config_edited": False, "not_accuracy": NOT_ACCURACY,
            "simulated": fake}


def decision_markdown(gate: dict, ev: dict) -> str:
    sim = "SIMULATED (--fake): stand-ins, NOT evidence. " if gate["simulated"] else ""
    lines = [f"# Decoder experiment: {gate['decision']}", "", f"{sim}Generated {ev['provenance']['finished_utc']}.", "",
             "| id | criterion | status | detail |", "|---|---|---|---|"]
    for x in gate["criteria"]:
        mark = " (provisional)" if x["provisional"] else ""
        lines.append(f"| {x['id']} | {x['criterion']}{mark} | **{x['status']}** | "
                     f"`{json.dumps(x['detail'], default=str)[:220]}` |")
    lines += ["", f"**{gate['not_accuracy']}**", "",
              ("Owner approval is required for any change. This tool never edits config.yaml, a threshold or the "
               "decoder; production is `accelerator.yolo_decoder: legacy` until the owner decides otherwise."), ""]
    p = ev["provenance"]
    lines += ["## Provenance", "", f"- git commit: `{p['git_commit']}` (dirty: {p['git_dirty']})",
              f"- package: `{json.dumps(p['package'], default=str)}`",
              f"- .hef: `{p['hef']['sha256_before']}`", f"- config: `{p['config']['sha256_before']}`",
              f"- host: `{p['host'].get('hostname')}` / `{p['host'].get('board')}` / `{p['host'].get('platform')}`",
              f"- started {p['started_utc']}, finished {p['finished_utc']}", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------ provenance
def package_info(path: Optional[Path]) -> Optional[dict]:
    if path is None:
        return None
    p = Path(path).expanduser()
    record = p / "PACKAGE.json" if p.is_dir() else p
    if not record.is_file():
        return {"path": str(p), "error": "no PACKAGE.json"}
    doc = json.loads(record.read_text())
    return {"path": str(record), "sha256": hc.sha256_file(record), "release": doc.get("release"),
            "commit": doc.get("commit"), "complete": doc.get("complete"), "id": f"transfer-{doc.get('release')}"}


def write_sha256sums(root: Path) -> None:
    lines = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        lines.append(f"{hc.sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ fake chip for the dry run
def experiment_backend():
    """The --fake chip for this experiment: a person, a phone that SHARES its anchor with a
    stronger 'remote' (so legacy's argmax loses the phone and reference's multilabel keeps
    it), and a lit hand channel 80 (which both product decoders must drop). Flat frames
    see nothing. Coordinates are model-input pixels (640x480 frames letterbox at top 80)."""
    import fake_hailo

    objects = [(PERSON, 0.86, 200, 120, 440, 540, 32), (PHONE, 0.46, 300, 300, 340, 372, 8),
               (HAND_CHANNEL, 0.90, 460, 300, 520, 380, 16)]
    heads = fake_hailo.planted_heads(objects)
    stride, (x1, y1, x2, y2) = 8, (300, 300, 340, 372)
    col, row = round((x1 + x2) / 2 / stride - 0.5), round((y1 + y2) / 2 / stride - 0.5)
    cls_name = next(n for n, (g, _, c) in fake_hailo.OUTPUTS if g == 640 // stride and c == fake_hailo.CLASS_CHANNELS)
    heads[cls_name][0, row, col, 65] = 0.60  # 'remote' beats the phone at the phone's own anchor

    def run(batch):
        fake_hailo.zero_backend(batch)
        if int(batch.min()) == int(batch.max()):
            return fake_hailo.zero_backend(batch)
        return {k: v.copy() for k, v in heads.items()}

    labels = {"format": LABELS_FORMAT, "verified_by": "SIMULATED", "method": "planted objects (--fake)",
              "simulated": True, "objects_input_px": objects}
    return run, labels


def simulated_labels(template: dict, bundle: Path) -> dict:
    """Frame-pixel labels for the planted person and phone on every non-flat captured image."""
    images = {}
    for s in hc.load_bundle(bundle)["samples"]:
        if s["mode"] != "letterbox" or s["image_id"] == "kg_flat_grey.png":
            continue
        g = s["geometry"]
        images[f"{bundle.name}/{s['image_id']}"] = [
            {"cls": c, "xyxy": [(x1 - g["left"]) / g["scale"], (y1 - g["top"]) / g["scale"],
                                (x2 - g["left"]) / g["scale"], (y2 - g["top"]) / g["scale"]]}
            for c, _score, x1, y1, x2, y2, _st in template["objects_input_px"] if c in (PERSON, PHONE)]
    return {**{k: v for k, v in template.items() if k != "objects_input_px"}, "images": images}


# ------------------------------------------------------------------ the run
def run(args) -> int:
    import hwv_stage as hs

    started = hc.utc_now()
    tag = "SIMULATED_" if args.fake else ""
    ev_dir = (args.out / f"decoder_exp_{tag}{time.strftime('%Y%m%dT%H%M%S')}").resolve()
    ok, why = hc.ensure_outside(ev_dir, args.install_dir)
    if not ok:
        print(f"REFUSED: {why}", file=sys.stderr)
        return hs.BLOCKED
    if args.cabin_scores:
        cabin = [Path(p.strip()).expanduser() for p in args.cabin_scores.split(",") if p.strip()]
        if len(cabin) != 2 or not all(p.is_file() for p in cabin):
            print("REFUSED: --cabin-scores needs LEGACY.json,REFERENCE.json (two existing files)", file=sys.stderr)
            return hs.BLOCKED
    else:
        cabin = None
    labels = load_labels(args.labels) if args.labels else None
    ev_dir.mkdir(parents=True)
    stages_dir = ev_dir / "stages"
    print(f"== DECODER EXPERIMENT{' [--fake: stand-ins, NOT evidence]' if args.fake else ''} -> {ev_dir}", flush=True)
    reports: dict[str, Any] = {}
    collected: list = []
    ev: dict[str, Any] = {"format": "sensordyme.hwval.decoder_experiment/1", "simulated": bool(args.fake),
                          "cabin_scores": [str(p) for p in cabin] if cabin else None}
    thermal = ThermalMonitor(ev_dir / "thermal.csv", args.thermal_interval).start()
    real_install = args.install_dir
    code = hs.PASS
    with contextlib.ExitStack() as stack:
        sub = argparse.Namespace(**vars(args))
        sub.out, sub.bundle, sub.decoder, sub.latency_decoder = stages_dir, ev_dir / "capture", "both", "both"

        def stage(name: str) -> int:
            thermal.mark(name)
            before = len(collected)
            rc = hs.run_stage(name, sub, reports=collected, fake_wrap=False)
            if len(collected) > before:
                reports[name] = collected[-1]
            return rc

        # 0. pins (against the real install, or this checkout's config under --fake), then the refusals.
        stage("pins")
        fake_labels = None
        if args.fake:
            backend, fake_labels = experiment_backend()
            sub.install_dir = stack.enter_context(hs.fake_device(backend))
        config, cfg_path = hs.load_config(sub.install_dir)
        hef, pin = hs.hef_path_and_pin(sub.install_dir, config)
        expected = pin if args.fake else PRODUCTION_HEF_SHA256  # the fake .hef is not the model; labelled
        prov: dict[str, Any] = {
            "started_utc": started, "argv": sys.argv, "simulated": bool(args.fake),
            "git_commit": hc.git_commit(hc.REPO_ROOT), "git_dirty": hc.git_dirty(hc.REPO_ROOT),
            "package": package_info(args.package), "host": hc.environment(),
            "hailo": {"simulated": True} if args.fake else None,  # probed after the refusals, chip free
            "hailo_simulated": bool(args.fake),
            "hef": {"path": str(hef), "sha256_before": hc.sha256_file(hef) if hef.exists() else None, "pin": pin,
                    "expected": expected, "sha256_after": None},
            "config": {"path": str(cfg_path), "sha256_before": hc.sha256_file(cfg_path), "sha256_after": None,
                       "installed_decoder": (config.get("accelerator") or {}).get("yolo_decoder", "legacy"),
                       "role": config.get("role"), "camera": config.get("camera")},
            "install_dir": str(real_install),
        }
        if args.fake:
            prov["hef"]["note"] = (f"SIMULATED .hef; on the Pi `expected` is PRODUCTION_HEF_SHA256 "
                                   f"{PRODUCTION_HEF_SHA256[:16]}...")
        ev["provenance"] = prov
        da = (config.get("modules") or {}).get("driver_attention") or {}
        gates = {PERSON: float(da.get("person_confidence_threshold", 0.50)),
                 PHONE: float(da.get("yolo_confidence_threshold", 0.20))}
        prov["config"]["gates"] = {"person": gates[PERSON], "phone": gates[PHONE],
                                   "yolo_every_n_frames": da.get("yolo_every_n_frames")}
        refusal = None
        if not hef.exists():
            refusal = f"{hef} does not exist"
        elif not (prov["hef"]["sha256_before"] == pin == expected):
            refusal = (f".hef sha256 {str(prov['hef']['sha256_before'])[:16]} / pin {pin[:16]} / production "
                       f"{expected[:16]} disagree: refusing to run the experiment on this model")
        if refusal is None and not args.fake:
            gate_rep = hs.Report("decoder-exp", False)
            if hs.service_must_be(gate_rep, "stopped", sub) is not None:
                refusal = gate_rep.blocked
            elif not Path("/dev/hailo0").exists():
                refusal = "no /dev/hailo0: the AI HAT+ is not seated or its driver is not loaded"
        if refusal is None and not args.fake:
            prov["hailo"] = hc.hailo_runtime_info()
        if refusal:
            print(f"BLOCKED before the chip was touched: {refusal}", flush=True)
            ev["refused"] = refusal
            code = hs.BLOCKED
        else:
            codes = [stage(n) for n in ("model", "parse-hef", "capture")]
            bundles = [ev_dir / "capture"] if (ev_dir / "capture" / "manifest.json").exists() else []
            if args.images or args.camera_frames:
                thermal.mark("capture-extra")
                import capture_hailo_outputs as cap

                extra = ev_dir / "capture_extra"
                argv = ["--modes", "letterbox", "--conf", str(args.floor), "--warmup", "3",
                        "--install-dir", str(sub.install_dir), "--out", str(extra)]
                argv += (["--images", str(args.images)] if args.images else
                         ["--camera", str(args.camera_frames), "--allow-saving-frames"])
                if args.fake or args.no_service_check:
                    argv.append("--no-service-check")
                if args.camera_frames and not args.allow_saving_frames:
                    print("  capture-extra REFUSED: --camera-frames writes frames to disk; pass --allow-saving-frames "
                          "(owner-approved test scene only)", flush=True)
                elif cap.main(argv) == 0:
                    bundles.append(extra)
                else:
                    ev["capture_extra_error"] = "capture_hailo_outputs.py refused (see its message above)"
            codes.append(stage("parity"))
            thermal.mark("same-tensor-decode")
            if bundles:
                prod_classes = production_classes(config)
                ev["decoders"], dets = same_tensor_decode(bundles, prod_classes=prod_classes,
                                                          prod_conf=gates[PHONE], floor=args.floor)
                if labels is None and fake_labels is not None:
                    labels = simulated_labels(fake_labels, bundles[0])
                ev["calibration"] = calibration_readiness(dets, gates, args.floor, labels)
            codes.append(stage("latency"))
            thermal.mark("pipeline")
            try:
                ev["pipeline"] = pipeline_fps(args, config, sub.install_dir, thermal)
            except Exception as exc:  # noqa: BLE001 - recorded as a FAIL criterion with its reason
                import traceback

                traceback.print_exc()
                ev["pipeline_error"] = repr(exc)
            code = hs.FAIL if hs.FAIL in codes else hs.BLOCKED if hs.BLOCKED in codes else code
        thermal.stop()
        ev["thermal"] = thermal.summary()
        prov["hef"]["sha256_after"] = hc.sha256_file(hef) if hef.exists() else None
        prov["config"]["sha256_after"] = hc.sha256_file(cfg_path)
        prov["finished_utc"] = hc.utc_now()
    ev["stage_reports"] = {n: {"verdict": r.verdict()[0], "checks_failed": [c["check"] for c in r.checks if not c["ok"]]}
                           for n, r in reports.items()}
    gate = decision_gate(ev, reports, bool(args.fake)) if "refused" not in ev else {
        "decision": f"KEEP LEGACY (refused before the run: {ev['refused']})", "keep_legacy": True,
        "bench_criteria_passed": False, "criteria": [], "owner_approval_required": True, "config_edited": False,
        "not_accuracy": NOT_ACCURACY, "simulated": bool(args.fake)}
    ev["gate"] = gate
    for name, payload in (("decoders.json", ev.get("decoders")), ("calibration.json", ev.get("calibration")),
                          ("pipeline.json", ev.get("pipeline"))):
        if payload is not None:
            hc.write_json(ev_dir / name, payload)
    hc.write_json(ev_dir / "evidence.json", ev)
    (ev_dir / "DECISION.md").write_text(decision_markdown(gate, ev))
    write_sha256sums(ev_dir)
    print_summary(ev, gate)
    if not gate["bench_criteria_passed"] and code == hs.PASS:
        code = hs.FAIL
    if args.fake:
        code = hs.SIMULATED if code in (hs.PASS, hs.FAIL) and "refused" not in ev else code
    label = {hs.PASS: "PASS", hs.FAIL: "FAIL", hs.BLOCKED: "BLOCKED", hs.SIMULATED: "SIMULATED (not evidence)"}[code]
    print(f"DECODER EXPERIMENT: {label}   evidence {ev_dir}   (sha256 of SHA256SUMS "
          f"{hc.sha256_file(ev_dir / 'SHA256SUMS')[:16]})", flush=True)
    return code


def production_classes(config: dict) -> list[int]:
    """The driver_attention module's own detector query: its yolo_classes plus person."""
    sys.path.insert(0, str(hc.REPO_ROOT))
    from core.accel import COCO_NAMES

    wanted = set(((config.get("modules") or {}).get("driver_attention") or {}).get("yolo_classes") or {})
    return sorted({PERSON, *(i for i, n in COCO_NAMES.items() if n in wanted)})


def print_summary(ev: dict, gate: dict) -> None:
    d, cal, pipe, th = ev.get("decoders") or {}, ev.get("calibration") or {}, ev.get("pipeline"), ev["thermal"]
    print("\nSUMMARY (decoder experiment)")
    if d:
        print(f"  same tensors: {d['samples']} samples; product == offline: {d['product_equals_offline_all']}; "
              f"class >= 80 emitted: {d['class_ge_80_emitted_any']}; channel-80 max {d['channel80_max']}")
        for dec, t in d["decode_ms_this_cpu"].items():
            if t:
                print(f"  decode on this CPU [{dec}]: p50 {t['p50']:.3f} ms  p95 {t['p95']:.3f} ms")
    for cls, entry in (cal.get("classes") or {}).items():
        dd = entry["decoders"]
        print(f"  class {cls} at gate {entry['production_gate']}: legacy {dd['legacy']['n_at_gate']} / reference "
              f"{dd['reference']['n_at_gate']} boxes; count-matched reference threshold "
              f"{entry['count_matched_reference_threshold']['value']}")
        for dec in SPECS:
            pr = ((entry.get("labelled") or {}).get(dec) or {}).get("at_production_gate")
            if pr:
                print(f"    [{dec}] labelled P {pr['precision']} R {pr['recall']} (tp {pr['tp']} fp {pr['fp']})")
    if pipe:
        for dec, p in pipe["per_decoder"].items():
            print(f"  pipeline [{dec}]: {p['achieved_fps_mean']} fps, late {p['late_fraction']}, decode p95 "
                  f"{p['decode_p95_ms_mean']} ms, decoder seen ok {p['decoder_seen_ok']}")
    print(f"  thermal: available {th['available']}, peak SoC {th['soc_temp_c_peak']} C, throttled "
          f"{th['throttled_values']}, Hailo {th['hailo_temp_c_peak']} ({th['hailo_temp_sources']})")
    print("\nDECISION GATE")
    for x in gate["criteria"]:
        print(f"  {x['id']:<4} {x['status']:<14} {x['criterion']}{' (provisional)' if x['provisional'] else ''}")
    print(f"\n  >>> {gate['decision']}")
    print(f"  {gate['not_accuracy']}")
    print("  Owner approval is required for any change; this tool edited no config, threshold or decoder.\n")


def add_arguments(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("decoder-exp (M14)")
    g.add_argument("--package", type=Path, default=None, help="decoder-exp: the transfer package dir (PACKAGE.json)")
    g.add_argument("--images", type=Path, default=None,
                   help="decoder-exp: an extra, consented image set to capture for calibration readiness")
    g.add_argument("--camera-frames", type=int, default=0,
                   help="decoder-exp: capture N live frames instead of --images (needs --allow-saving-frames)")
    g.add_argument("--allow-saving-frames", action="store_true",
                   help="decoder-exp: --camera-frames writes frames under the evidence dir (owner-approved scene)")
    g.add_argument("--labels", type=Path, default=None,
                   help=f"decoder-exp: human-verified boxes ({LABELS_FORMAT}) for precision/recall")
    g.add_argument("--floor", type=float, default=0.05, help="decoder-exp: decode floor for score distributions")
    g.add_argument("--thermal-interval", type=float, default=5.0, help="decoder-exp: seconds between thermal samples")
    g.add_argument("--cabin-scores", default=None,
                   help="decoder-exp: LEGACY.json,REFERENCE.json cabin_session score files (verified labels), C11")


if __name__ == "__main__":  # pragma: no cover - the entry point is hwv_stage.py decoder-exp
    raise SystemExit(subprocess.call([sys.executable, str(HERE / "hwv_stage.py"), "decoder-exp", *sys.argv[1:]]))
