"""Developer diagnostics: what the device sees and decides, for the people building it.

`GET /api/diagnostics` on the dashboard (schema `sd-diagnostics/1`,
docs/DIAGNOSTICS.md). OFF unless `dashboard.developer_diagnostics: true`, and even
then only for a request carrying the device token: a device with no
`dashboard.token` refuses it outright, also on loopback, where /api/health would
answer. Never sent upstream, never shown in the consumer app.

What is in it, all bounded:

  * the current detections (label, confidence, box, track id), at most
    `MAX_DETECTIONS`, highest confidence first;
  * the driver association outcome and the active evidence (eye state, head
    pose / gaze, the hand-phone association and handling cue), as the module
    computed them on the last analysed frame;
  * the open conditions with how long each has been running (alert episodes
    and the module's developing / confirmed trackers);
  * the monitoring capabilities and every unknown / degraded reason;
  * model and version identifiers (detector sha256, decoder, MediaPipe pins);
  * the per-stage latencies (core/stage_timing.py), tagged with the host.

What is never in it: a frame or any pixel data, the device token, the cloud
device key, the Wi-Fi SSID or passphrase. `sanitize()` enforces it for anything
a module hands over: arrays and bytes are dropped, keys that look like secrets
are dropped, depth, list length and string length are capped.

Cost: the per-frame snapshot is built on the inference thread (where the module's
state is consistent), at most `SNAPSHOT_INTERVAL_S` apart, and only while a
diagnostics request has arrived within `DEMAND_TTL_S`. With the switch off, or
nobody asking, the inference loop pays one attribute check per frame.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Optional

SCHEMA = "sd-diagnostics/1"
MAX_DETECTIONS = 20
SNAPSHOT_INTERVAL_S = 0.5
DEMAND_TTL_S = 10.0
_MAX_DEPTH = 6
_MAX_ITEMS = 32
_MAX_STR = 240
#: Keys never carried, at any depth (case-insensitive substring match).
_SECRET_KEY = re.compile(r"token|secret|password|passwd|psk|ssid|device_key|api_key|apikey|credential|cookie|auth",
                         re.IGNORECASE)
#: Keys that name image data; dropped at any depth whatever the value is.
_IMAGE_KEY = re.compile(r"^(frame|image|jpeg|jpg|png|pixels|thumbnail|crop)s?$", re.IGNORECASE)

DISABLED_REASON = ("developer diagnostics are off on this device: set dashboard.developer_diagnostics: true "
                   "in config.yaml (and a dashboard.token) and restart the service")
UNCONFIGURED_REASON = ("developer diagnostics need dashboard.token to be set: they are never served "
                       "without the device token, not even on loopback")


def sanitize(value: Any, depth: int = 0) -> Any:
    """`value` as bounded JSON: no arrays/bytes, no secret-looking keys, capped sizes."""
    if depth > _MAX_DEPTH:
        return "<truncated>"
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, 4) if math.isfinite(value) else None
    if isinstance(value, str):
        return value if len(value) <= _MAX_STR else value[:_MAX_STR] + "…"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return None  # never raw bytes (a JPEG, a frame buffer)
    if type(value).__module__.startswith("numpy"):
        # A numpy scalar is a number; an array may be a frame and is never carried.
        if getattr(value, "ndim", 1) == 0 and hasattr(value, "item"):
            return sanitize(value.item(), depth)
        return None
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return sanitize(dataclasses.asdict(value), depth)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict) and not isinstance(value, Mapping):
        try:
            return sanitize(to_dict(), depth)
        except Exception:  # noqa: BLE001 - a broken to_dict loses that value, never the payload
            return None
    if isinstance(value, Mapping):
        out: dict = {}
        for key, item in value.items():
            if len(out) >= _MAX_ITEMS:
                out["…"] = f"{len(value) - _MAX_ITEMS} more"
                break
            name = str(key)
            if _SECRET_KEY.search(name) or _IMAGE_KEY.match(name):
                continue
            out[name[:64]] = sanitize(item, depth + 1)
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        return [sanitize(v, depth + 1) for v in items[:_MAX_ITEMS]]
    if hasattr(value, "value") and isinstance(getattr(value, "value", None), (str, int)):
        return value.value  # an Enum
    return str(value)[:_MAX_STR]


def detections(observation: Any, limit: int = MAX_DETECTIONS) -> dict:
    """The detector's current boxes, highest confidence first, or why there are none."""
    if observation is None:
        return {"state": "no analysed frame yet", "items": None}
    dets = getattr(observation, "detections", None)
    status = (getattr(observation, "components", {}) or {}).get("object_detector")
    if dets is None:
        reason = getattr(status, "reason", None) or getattr(status, "state", None) or "not available"
        return {"state": getattr(status, "state", "unknown"), "reason": reason, "items": None}
    ordered = sorted(dets, key=lambda d: -float(d.confidence))
    return {
        "state": getattr(status, "state", "ok"),
        "fresh": bool(getattr(observation, "fresh_detections", False)),
        "age_s": None if observation.detections_age_s is None else round(float(observation.detections_age_s), 3),
        "total": len(ordered),
        "items": [d.to_dict() for d in ordered[:limit]],
    }


def face_evidence(observation: Any) -> dict:
    """Eye state, head pose and gaze as the module derived them this frame."""
    if observation is None or getattr(observation, "face_visible", None) is None:
        return {"face": "landmark model did not run on this frame"}
    if not observation.face_visible:
        return {"face": "no face found"}
    face = observation.face
    return {
        "face": "found",
        "landmarks": face.landmark_count,
        "eye_state": sanitize(face.eye_state) if face.eye_state is not None else "not measured",
        "head_pose": sanitize(face.head_pose) if face.head_pose is not None else "not measured",
        "gaze": sanitize(face.gaze) if face.gaze is not None else "not measured (eye closed or iris unusable)",
    }


#: Debug-state keys that carry the hand-phone association and the handling cue.
_HAND_PHONE_KEYS = ("phone_detected", "phone_confidence", "hand_overlaps_phone", "phone_hold", "phone_handling")


def module_evidence(debug_state: Optional[Mapping[str, Any]]) -> dict:
    """Driver association + hand-phone evidence from a module's get_debug_state()."""
    if not debug_state:
        return {"driver_association": "not reported by this module", "hand_phone": "not reported by this module"}
    engine = debug_state.get("engine") if isinstance(debug_state.get("engine"), Mapping) else {}
    hand_phone = {k: debug_state.get(k) for k in _HAND_PHONE_KEYS if k in debug_state}
    if isinstance(engine, Mapping) and engine.get("phone_evidence") is not None:
        hand_phone["engine_phone_evidence"] = engine.get("phone_evidence")
    return {
        "driver_association": sanitize(engine.get("driver")) if engine.get("driver") is not None
        else "no driver association this frame",
        "hand_phone": sanitize(hand_phone) or "not reported by this module",
        "state": debug_state.get("state"),
        "gaze_zone": debug_state.get("gaze"),
        "abstaining": sanitize(engine.get("abstaining")) if engine else None,
        "events_fired_this_frame": sanitize(debug_state.get("events_fired_this_frame")),
    }


def open_conditions(debug_state: Optional[Mapping[str, Any]], episodes: Iterable[Mapping[str, Any]],
                    now: float) -> list[dict]:
    """Every condition that is developing or confirmed, and how long it has run.

    Two sources: the alert policy's open episodes (`running_s` since the episode
    started, monotonic) and the module's own trackers (`<name>_state` DEVELOPING /
    CONFIRMED with `<name>_duration_s`)."""
    out: list[dict] = []
    for ep in episodes:
        started = ep.get("started_at")
        out.append({
            "source": f"alert_episode:{ep.get('source')}", "condition": ep.get("type"),
            "state": ep.get("state"), "category": ep.get("category"),
            "running_s": round(max(0.0, now - float(started)), 2) if isinstance(started, (int, float)) else None,
            "actuations": ep.get("actuations"),
        })
    for key, value in (debug_state or {}).items():
        if not key.endswith("_state") or value not in ("DEVELOPING", "CONFIRMED"):
            continue
        name = key[: -len("_state")]
        duration = (debug_state or {}).get(f"{name}_duration_s")
        out.append({"source": "module_tracker", "condition": name, "state": value.lower(),
                    "running_s": duration if isinstance(duration, (int, float)) else None})
    return out[:_MAX_ITEMS]


def frame_snapshot(*, observation: Any, debug_state: Optional[Mapping[str, Any]],
                   episodes: Iterable[Mapping[str, Any]], now: float, frame_age_ms: Optional[float],
                   module_name: str) -> dict:
    """What the inference thread captures for one analysed frame (bounded, no pixels)."""
    return {
        "module": module_name,
        "frame_id": getattr(observation, "frame_id", None),
        "frame_age_ms": None if frame_age_ms is None else round(frame_age_ms, 1),
        "simulated": bool(getattr(observation, "simulated", False)),
        "source": getattr(observation, "source", None),
        "detections": detections(observation),
        "face": face_evidence(observation),
        "hands": (None if getattr(observation, "hands", None) is None
                  else {"count": len(observation.hands), "ran": True}),
        "evidence": module_evidence(debug_state),
        "open_conditions": open_conditions(debug_state, episodes, now),
    }


_REGISTRY_PINS: Optional[dict] = None


def mediapipe_pins() -> dict:
    """The MediaPipe graph versions the model registry pins, and what is installed."""
    global _REGISTRY_PINS  # noqa: PLW0603 - read once per process
    if _REGISTRY_PINS is None:
        pins: dict = {}
        try:
            import yaml

            data = yaml.safe_load((Path(__file__).with_name("model_registry.yaml")).read_text()) or {}
            for card in data.get("models", []) or []:
                if str(card.get("id", "")).startswith("mediapipe_"):
                    pins[str(card["id"])] = str(card.get("version"))
        except Exception as exc:  # noqa: BLE001 - versions are reported, never required
            pins["error"] = f"model_registry.yaml unreadable: {exc}"[:_MAX_STR]
        try:
            from importlib.metadata import PackageNotFoundError, version

            try:
                pins["installed"] = f"mediapipe-{version('mediapipe')}"
            except PackageNotFoundError:
                pins["installed"] = None
        except ImportError:  # pragma: no cover - stdlib since 3.8
            pins["installed"] = None
        _REGISTRY_PINS = pins
    return dict(_REGISTRY_PINS)


def model_versions(*, identity: Optional[Mapping[str, Any]], accel_cfg: Optional[Mapping[str, Any]],
                   software: str, model_version: Optional[str]) -> dict:
    accel_cfg = accel_cfg or {}
    detector = (identity or {}).get("detector") if identity else None
    return {
        "software": software,
        "model_version": model_version,
        "detector": sanitize(detector) if detector else None,
        "detector_pin_sha256": accel_cfg.get("yolo_hef_sha256"),
        "decoder": accel_cfg.get("yolo_decoder", "legacy"),
        "mediapipe": mediapipe_pins(),
    }


def unknown_and_degraded(monitoring: Mapping[str, Any]) -> list[str]:
    """Every reason the device gives for not being fully able to monitor, plus each
    impaired / unknown capability's own reason, de-duplicated, in order."""
    reasons: list[str] = [str(r) for r in monitoring.get("reasons") or []]
    for name, cap in (monitoring.get("capabilities") or {}).items():
        if not isinstance(cap, Mapping):
            continue
        state = str(cap.get("state", ""))
        if state and state not in ("available", "ready", "ok", "not_required", "paused"):
            detail = cap.get("reason") or cap.get("detail") or state
            reasons.append(f"{name}: {detail}")
    seen: set[str] = set()
    return [r for r in reasons if not (r in seen or seen.add(r))][:_MAX_ITEMS]


def capabilities(monitoring: Mapping[str, Any]) -> dict:
    caps = monitoring.get("capabilities") or {}
    return {
        "state": monitoring.get("state"),
        "monitoring": monitoring.get("monitoring"),
        "available": list(monitoring.get("available") or []),
        "impaired": list(monitoring.get("impaired") or []),
        "paused": list(monitoring.get("paused") or []),
        "detail": {name: {"state": c.get("state"), "reason": c.get("reason")}
                   for name, c in caps.items() if isinstance(c, Mapping)},
    }


def payload(*, snapshot: Optional[Mapping[str, Any]], snapshot_age_s: Optional[float],
            monitoring: Mapping[str, Any], performance: Mapping[str, Any], versions: Mapping[str, Any]) -> dict:
    """The sd-diagnostics/1 body."""
    return {
        "schema": SCHEMA,
        "enabled": True,
        "note": "developer diagnostics: local, token-gated, never uploaded; no frames or secrets",
        "platform": performance.get("platform"),
        "snapshot_age_s": None if snapshot_age_s is None else round(snapshot_age_s, 2),
        "frame": sanitize(dict(snapshot)) if snapshot is not None else None,
        "warming_up": snapshot is None,
        "capabilities": capabilities(monitoring),
        "unknown_or_degraded": unknown_and_degraded(monitoring),
        "versions": sanitize(dict(versions)),
        "latency": sanitize(dict(performance)),
    }
