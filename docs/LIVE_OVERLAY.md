# Live AI overlay: sd-overlay/1

The device can send the **clean** camera frame together with a compact
description of what the AI actually saw on that frame. The app draws the
description itself, which gives a clean consumer view and a polished "AI view"
from the same stream. Implementation: `core/overlay.py`; tests:
`tests/test_m13_overlay.py`.

## Requesting it

`GET /stream` with `X-Stream-Overlay: 1` (or `?overlay=1`), authenticated as
today: an `X-SD-Session` (sd-local-v1) or the legacy token.

* The response header is `X-Stream-Overlay: sd-overlay/1`.
* Each analysed JPEG part carries `X-SD-Overlay: <base64url, no padding, of compact JSON>`.
  It sits beside the existing `X-Frame-Seq`, `X-Frame-Age-Ms`,
  `X-Frame-Source` and `X-Stream-Id` headers.
* The overlay describes **that** JPEG: the device builds both from the same
  camera frame.
* A camera fallback part (`X-Frame-Source: camera`, nothing analysed) has no
  `X-SD-Overlay`. The app must then show "not analysed" and draw nothing.
* Camera privacy is unchanged. While monitoring is stopped, a remote viewer
  gets no frames unless the owner has enabled camera-only.
* Viewers that do not ask get exactly the device-drawn stream they got before.
  The device encodes each variant only while someone is watching it.

## The document

A real example, made by `core/overlay.py` from a fabricated observation:

```json
{"schema":"sd-overlay/1","module":"driver_attention","mode":"driver_safety",
 "analysed":true,"sim":false,"source":"camera","w":640,"h":480,"frame_id":1812,
 "det":[{"label":"person","box":[0.1,0.1,0.5,1.0],"track":3,"role":"driver"},
        {"label":"phone","box":[0.3125,0.4167,0.375,0.5417],"track":9,"role":"driver"}],
 "det_age_s":0.0,"det_fresh":true,
 "face":{"visible":true,"contours":{"oval":[[0.41,0.12],...],"eye_l":[...],"eye_r":[...]},
         "eyes_closed":false,
         "head_pose":{"yaw_deg":34.2,"pitch_deg":-3.1,"calibrated":true,"direction":"Left","kind":"head_pose"}},
 "hands":[[0.2969,0.3958,0.3906,0.5625]],
 "driver":{"box":[0.1,0.1,0.5,1.0],"basis":"configured_seat","identity":"driver","people":1},
 "driver_region":{"box":[0.05,0.0,0.55,1.0],"source":"configured"},
 "behaviours":[{"type":"distraction","phase":"confirmed","for_s":3.4}],
 "state":"DISTRACTED","alert_active":true,
 "last_event":{"type":"distraction","signals":"head turned away 3.4 s","severity":"loud","age_s":0.4},
 "unavailable":[],
 "perf":{"fps":14.0,"object_detector_ms":11.8,"face_landmarks_ms":9.4,"hand_landmarks_ms":7.0,"frame_age_ms":83}}
```

| Field | Meaning | How to show it |
|---|---|---|
| `analysed` | false: no observation behind this frame | draw nothing; "Not analysed" |
| `sim` / `source` | `true` / `sim` or `replay`: a simulated or replayed source | a permanent **Simulated** badge on every frame |
| `w`, `h` | the analysed frame size in pixels | aspect for mapping the boxes |
| `det` | `null` = detection did not run or is unavailable ("could not look"); `[]` = looked, found nothing | `null`: say "detection unavailable", never show an empty scene as clear |
| `det[].box` | normalised `[x1,y1,x2,y2]`, 0–1 | |
| `det[].track` | the shared tracker's id | optional small "#3" |
| `det[].role` | `driver` / `other` (phones only), or absent | highlight the driver; a passenger's phone is "other" |
| `det_age_s`, `det_fresh` | the boxes are the detector's latest run, carried between runs | fade the boxes when not fresh |
| `face` | `null` = face model not run this frame; `{"visible":false}` = looked, no face | |
| `face.contours` | FaceMesh points in drawing order (oval 36, each eye 16), normalised | thin outline, not the full mesh |
| `face.eyes_closed` | absent until a calibrated threshold exists | "Eyes closed" only when `true` |
| `face.head_pose` | **head** orientation in degrees; `calibrated` = relative to this driver's neutral pose; `direction` = Left/Right/Up/Down/Forward. `pitch_deg` is always in ±90: an uncalibrated raw fit (upright ≈ ±180 in its Euler convention) is wrapped by 180° | label it "Head", **never** "gaze" or "looking at" |
| `hands` | hand boxes when the hand model ran this frame (it is scheduled, not every frame) | |
| `driver`, `driver_region` | the person the driver layer associated, how (`basis`), and the configured or calibrated seat region | outline the driver; dashed region |
| `behaviours[]` | each behaviour in progress: `phase` = `developing` (condition true, not yet sustained) or `confirmed` (the module's alert condition); `for_s` = how long | "Head turned away · 2.1 s" developing, then confirmed |
| `state` | the module's dominant state label | |
| `alert_active` | an alert condition is active now | |
| `last_event` | the newest event, at most 30 s old, with the text **the device wrote** (`signals`); `event_key` is the same key as that event's `/api/events` row | the "why" line, tied to its history row |
| `unavailable` | components not working (stale, failed, unavailable) | hide or label those capabilities |
| `perf.fps` | the processed-frame rate, measured on this device | "14 fps" |
| `perf.<component>_ms` | each model's own run-time EMA, measured on this device | "Detector 11.8 ms" |
| `perf.frame_age_ms` | capture to overlay build, on the device's monotonic clock | |

**Never shown, by design:**
* confidence percentages (a detector score is not a probability of being right);
* eye gaze;
* anything from a frame other than the one it rides with.

## The app side (M13)

The iPhone app always asks for it and draws it itself: `ios/Sources/LiveOverlay.swift`
(decoding and every word, tested by `ios/Tests/LiveOverlayTests.swift`),
`AIViewOverlay.swift` (the AI view and the driver panel), `PresentationMode.swift`.
The clean picture is its default view; "AI view" is a toggle. See
`docs/IOS_APP_STATUS.md`, Mission 13.

M13.1: the same AI view toggle and the **Present** key are on the paired-camera
screen (`LocalDeviceViews.swift`), which works with the account server down
over the phone's sd-local-v1 session. Presentation mode is fed by either screen
through `PresentationSource.swift`: over the direct connection the modes are the
camera's own `/api/presets` (limited to its role's `allowed_presets`), a switch
is the signed local command and a chip is selected only once the camera reports
that variant; the alert evidence is the camera's `/api/events` (last minute, by
`age_s`) and `alert_active`; `last_event.event_key` names the why line's row in
that history; Jupiter's rules are said as running from `policy.mode` but their
wording (kept by the account) is not shown. The Head chip shows `pitch_deg`
as the device's signed number beside its `direction` word.

## Cost

* About 1–2 KB per frame (`tests/test_m13_overlay.py` bounds the header below 4 KB).
* One JSON build and one base64 per streamed frame.
* No second camera read and no extra model run. When both an overlay viewer
  and a device-drawn viewer watch, each streamed frame is encoded twice.
