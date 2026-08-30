# SensorDyme performance budget

What this device must achieve, and why each number is that number.

Every figure below is one of three things, and is labelled as such:

* **MEASURED** — taken from a real run on real hardware. Provenance in [§1](#1-measurement-provenance).
* **DERIVED** — arithmetic from a measurement, or from a threshold `config.yaml`
  already ships. No judgement involved; the number falls out.
* **RECOMMENDATION** — a judgement call by the author of this document. Argued, but
  not established. Argue back.

The organising principle: **this product's requirements are already written down.**
They are the temporal thresholds in `config.yaml` and the `RuleSpec` values in the
modules. A fall needs `stillness_duration_s` of evidence; flailing needs
`flailing_confirm_s`; the event engine needs `min_frames` sightings on top of that.
Those are safety requirements, expressed in seconds and in frame counts at the same
time — which means they silently encode a **minimum frame rate**. That derivation is
[§2](#2-the-binding-constraint-minimum-frame-rate), and it is the only part of this
document that constrains a hardware decision. Everything after it is headroom.

---

## 1. Measurement provenance

Raspberry Pi 5 Model B Rev 1.1, 4 GB RAM, Debian 12, kernel 6.12.93. Hailo-8L
(13 TOPS) on PCIe, HailoRT 4.20.0. Camera Module 3 Wide NoIR (`imx708_wide_noir`)
at 640×480 RGB888. This is the shipping `profile: pi`.

| Measurement | CPU | Hailo | Notes |
|---|---|---|---|
| YOLO11n single-model inference, mean | 293.1 ms | 17.3 ms | **16.9×** |
| … p50 | 293.2 ms | 15.2 ms | |
| … p95 | 295.1 ms | 15.4 ms | |
| … implied single-model rate | 3.4 /s | 57.9 /s | |
| 600-image benchmark, wall clock | 365.3 s | 29.3 s | 12.5× (the gap is smaller than 16.9× because image I/O and pre/post-processing are on the host in both cases) |
| Live pipeline, 9.5 min sustained | — | 28.6 inferences/s | |
| Process CPU during that run | — | 36–37 % | |
| RSS during that run | — | 307 → 308 MB | +1 MB over 9.5 min |
| SoC temperature during that run | — | 56.5 – 59.3 °C | fan level 1 of 4 |
| Idle SoC temperature | 53.8 °C | | |
| Peak SoC temperature | 64.8 °C | | during the CPU-only benchmark |
| `vcgencmd get_throttled` | **0x50000** | 0x0 | see [§7](#7-power-the-under-voltage-finding) |
| Accelerator errors over 9.5 min | — | 0 | |
| MJPEG stream | — | 79 real JPEG frames in 8 s (~10 /s) | against `dashboard.stream_fps: 12` |

Two honesty notes on the table above.

**The Hailo mean (17.3 ms) is larger than its own p95 (15.4 ms).** That is only
possible if a small number of inferences are far slower than the 95th percentile —
almost certainly first-inference warm-up, and possibly `core/accel.py`'s
`HailoYOLO.predict()` re-entering the `InferVStreams` context and re-activating the
network group on every frame (flagged as suspicious in
`docs/HARDWARE_VERIFICATION.md` §4–6). **The tail is not characterised.** Budget
against p95 for throughput and against a measured p99/max for latency — and that
max has not been measured. This is the single largest gap in the numbers below.

**"28.6 inferences/sec" is ambiguous** in the figures handed over: it may be
pipeline frames per second, or YOLO calls per second. The shipped
`module: driver_attention` runs YOLO on a stride of `yolo_every_n_frames: 3`, so the
two readings differ by 3×. Both are carried through [§6](#6-hailo-utilisation)
where the difference matters; elsewhere the conservative reading is used.

---

## 2. The binding constraint: minimum frame rate

### 2.1 How the event engine actually gates

`core/event_engine.py` confirms a detection on the first frame where **both**
conditions hold (lines 140–149):

```
held >= rule.persist_s   and   track.frames >= rule.min_frames
```

`update()` is called once per **processed pipeline frame** — verified in
`modules/baby_monitoring.py::process` (which calls `_engine.update` for every rule
on every frame, reusing cached detections between stride ticks) and in
`modules/fall_detection.py`. So `frames` counts pipeline frames, not detection
ticks.

At a steady rate *f*, frames land at 0, 1/f, 2/f, … The condition confirms at frame
index *n* where

```
n = max( ceil(f · persist_s),  min_frames − 1 )
```

and the confirmation latency is *n/f*. The frame-count gate therefore becomes the
binding one — confirmation goes **late** — below

> **f_min = (min_frames − 1) / persist_s**

Below `f_min` the detection does not become impossible; it becomes *late*, firing at
`(min_frames − 1)/f` seconds instead of at `persist_s`. That distinction matters: a
fall confirming at 4 s instead of 1.5 s is a degraded product, not a broken one.
What *is* broken is [§2.2](#22-the-hard-floor-a-detector-that-cannot-answer-at-all).

**DERIVED — every rule in the product, ordered by the frame rate it demands:**

| Detection | Module | Severity | `persist_s` | `min_frames` | **f_min** |
|---|---|---|---|---|---|
| `flailing` | baby_monitoring | warning (**loud**) | 0.5 | 4 | **6.00 /s** |
| `jumping` | fall_detection | info | 0.2 | 2 | 5.00 /s |
| `running` | fall_detection | warning | 1.0 | 5 | 4.00 /s |
| `choking_hazard:{side}` | baby_monitoring | **critical** | 0.8 | 4 | **3.75 /s** |
| `object_approaching:{label}` | baby_monitoring | info | 0.3 | 2 | 3.33 /s |
| `activity_burst` | baby_monitoring | info | 0.5 | 2 | 2.00 /s |
| `holding_bottle` / `holding_object:*` / `object_in_hand:*` | baby_monitoring | info | 1.0 | 3 | 2.00 /s |
| AI-configured custom rule, worst case | core/rules.py | any | 0.5 | 2 | 2.00 /s |
| **`fall`** | fall_detection | **critical** | 1.5 | 3 | **1.33 /s** |
| `zone_violation:*` / `safe_zone_exit` | all | critical | 2.0 | 3 | 1.00 /s |
| `crowd` | occupancy | warning | 5.0 | 3 | 0.40 /s |
| `person_lingering` | occupancy | warning | 30.0 | 10 | 0.30 /s |
| `person_absent` | baby_monitoring | warning (**loud**) | 30.0 | 5 | 0.13 /s |
| `unusual_stillness` | baby_monitoring | warning (**loud**) | 120.0 | 10 | 0.075 /s |
| `unusual_stillness` | fall_detection | warning (**loud**) | 600.0 | 10 | 0.015 /s |

Sources: `modules/baby_monitoring.py:385–588`, `modules/fall_detection.py:145–209`,
`modules/occupancy.py:114–323`, `core/rules.py:210–218`,
`config.yaml` (`stillness_duration_s`, `flailing_confirm_s`,
`choking_hazard_confirm_s`, `still_warning_s`, `absence_warning_s`,
`unusual_stillness_s`, `zone_confirm_s`, `linger_s`, `crowd_confirm_s`).

The AI-rule row is worth its own sentence. `core/rules.py:216` computes
`min_frames = max(2, int(min(persist_s, 10.0)))` against a `persist_s` floored at
0.5. At `persist_s = 0.5` that is 2 frames in half a second — 2.0 /s. At any
`persist_s ≥ 2` the ratio is `int(persist_s)/persist_s ≤ 1`, so an AI-authored rule
never demands more than 2 /s. That formula is well behaved; nothing to fix.

**Binding value: 6.0 /s**, set by `flailing`. Note this is a *loud* type
(`config.yaml` `alerts.loud_types`), so it pushes to the phone — it is not a
log-only nicety.

### 2.2 The hard floor: a detector that cannot answer at all

`core/activity.py` is a different shape of constraint. It buffers pose samples in a
trailing window and refuses to classify below a sample count:

```python
WINDOW_S = 1.2
MIN_SAMPLES = 6
...
while self._samples and now - self._samples[0].at > self.window_s:   # prune
...
if len(samples) < MIN_SAMPLES:
    state.signals["reason"] = f"collecting motion ({len(samples)}/{MIN_SAMPLES} frames)"
```

At rate *f*, samples surviving the prune number `floor(1.2·f) + 1`. Requiring ≥ 6:

```
floor(1.2 f) ≥ 5   →   f ≥ 5 / 1.2 = 4.167 /s
```

**DERIVED: below 4.17 /s, `core/activity.py` never emits a verdict.** Not late —
*absent*. The recogniser sits permanently in "collecting motion". Everything built
on it goes with it:

* `flailing` (baby_monitoring, **loud**, pushes to the phone)
* `running`, `jumping` (fall_detection)
* the stillness / head-droop / gesture signals (`STILL_MAX_HIP_SPEED`,
  `HEAD_DROOP_MAX_CLEARANCE`, `WAVE_MIN_SWINGS`)

What survives below 4.17 /s: the `fall` detection itself. `modules/fall_detection.py`
computes its drop velocity and its lying-down posture from `core/posture.py` on a
two-frame difference, independent of `ActivityRecognizer` — verified at
`fall_detection.py:306–307, 333–336, 442–449`. That is a good architectural
property and worth preserving deliberately: **the most safety-critical detection in
the product has the loosest frame-rate requirement of any of them.**

### 2.3 The maximum gap between two frames

Separate from the *average* rate, `core/event_engine.py:129` resets an unconfirmed
track when the gap since the last sighting exceeds `occlusion_grace_s`:

```python
if track.state in (OBSERVED, DEVELOPING) and now - track.last_seen > rule.occlusion_grace_s:
    track.first_seen = now; track.frames = 0; track.confidence = 0.0
```

`occlusion_grace_s` defaults to **1.0 s** and is left at the default by `fall`,
every zone rule, `flailing`, `choking_hazard`, and every `holding_*` rule. So:

> **DERIVED (hard): no two consecutively processed frames may be more than 1.0 s
> apart.** A single 1.1-second stall — a GC pause, a `.hef` re-activation, a
> blocking alert-sound call, a page fault under memory pressure — discards all
> accumulated evidence for every in-progress detection and restarts them from zero.
> The mean frame rate can be excellent and this can still be violated.

Confirmed tracks are safe: `resolve_after_s` is only evaluated on frames where the
condition is *absent*, so a stall cannot spuriously resolve an active alert.

`unusual_stillness` in both modules explicitly overrides this to 3.0 s. That is a
deliberate widening, not an oversight — the comment at `baby_monitoring.py:400` and
`fall_detection.py:210` explains it — and it means the stillness rules tolerate a
3 s stall.

### 2.4 Strided signals

Two modules run their object detector on a stride while the engine ticks every
frame:

| Module | Setting | Value | Effective detection rate |
|---|---|---|---|
| `driver_attention` | `yolo_every_n_frames` | 3 | *f*/3 |
| `baby_monitoring` | `detect_every_n_frames` | 2 | *f*/2 |
| `baby_monitoring` | `pose_every_n_frames` | 1 | *f* |
| `fall_detection` | (no stride) | — | *f* |

A strided signal quantises every temporal gate that depends on it. `driver_attention`'s
`phone_use_seconds: 2.0` is measured with a sampling period of 3/*f* seconds:

| *f* | Sampling period | Quantisation error on a 2.0 s gate |
|---|---|---|
| 3.4 /s (CPU) | 0.88 s | **44 %** |
| 6.0 /s | 0.50 s | 25 % |
| 15 /s | 0.20 s | 10 % |
| 28.6 /s (measured) | 0.105 s | **5.2 %** |

**RECOMMENDATION: hold strided-gate quantisation at or below 10 % of the shortest
gate that depends on it.** For `driver_attention`'s 2.0 s gates (`phone_use_seconds`,
`drowsy_seconds`) that is `3/f ≤ 0.2` → **f ≥ 15 /s**. The 10 % figure is a
judgement, not a derivation; the arithmetic that produces 15 /s from it is not.

`driver_attention`'s `erratic_hands_seconds: 1.0` is tighter still, and it needs
more than a sampling argument. `erratic_hand_min_reversals: 3` is counted by
`core/tracking.py::reversals`, which needs `len(trail) ≥ 3` and, to see three
direction changes with a minimum amplitude on each leg, at least 5 trail points.
The trail is bounded by **count** (30 points, `tracking.py:141`), not by time, so
those 5 points may span more than one second and the sustain gate is what forces
recency. For the 1.0 s sustain window to be evaluated on more than a single sample:
`f/3 ≥ 2` → **f ≥ 6 /s**. For the whole 5-point trail to refresh inside one
episode: `f/3 ≥ 5` → **f ≥ 15 /s** (RECOMMENDATION — 15 /s again, from an
independent route).

### 2.5 Signal conditioning

`config.yaml`'s `signal_median_window: 5` puts a 5-frame rolling median in front of
every EAR / yaw / pitch threshold in `driver_attention`. A median of 5 recognises a
step edge once 3 of its 5 samples are new — a group delay of **2 frames**. So every
sustained driver gate fires late by 2/*f* seconds:

| *f* | Median lateness | `drowsy_seconds: 2.0` fires at | Error |
|---|---|---|---|
| 3.4 /s | 0.588 s | 2.59 s | +29 % |
| 6.0 /s | 0.333 s | 2.33 s | +17 % |
| 15 /s | 0.133 s | 2.13 s | +6.7 % |
| 28.6 /s | 0.070 s | 2.07 s | **+3.5 %** |

The same filter suppresses any excursion shorter than 3 frames = 3/*f* s. At 3.4 /s
that is 0.88 s — long enough to erase a real half-second event. At 28.6 /s it is
0.105 s, which lands neatly below the 100–400 ms blink range the `mac_max` profile
cites (`blink_ignore_max_s: 0.5`) and is exactly the behaviour the filter was
designed for.

`signal_release_s: 0.4` (a condition must read false for 0.4 s before its timer
resets) needs at least one frame inside 0.4 s to be observed at all: **f ≥ 2.5 /s**.
Not binding.

### 2.6 The frame-rate budget

| Bound | Value | Kind | Set by |
|---|---|---|---|
| Absolute floor — `core/activity.py` cannot answer below this | **4.17 /s** | DERIVED | `MIN_SAMPLES 6 / WINDOW_S 1.2` |
| Minimum for on-time confirmation of every rule | **6.00 /s** | DERIVED | `flailing`, `(4−1)/0.5` |
| Target — strided gates and the median filter within 10 % | **15 /s** | RECOMMENDATION | `yolo_every_n_frames 3` against 2.0 s gates |
| Measured, Hailo, 9.5 min sustained | **28.6 /s** | MEASURED | live pipeline |
| Camera capture rate | 30 /s | config | `camera.fps` |
| Maximum interval between any two processed frames | **< 1.0 s** | DERIVED | `occlusion_grace_s` default |

**Budget: 15 /s target, 6 /s floor, and no inter-frame gap over 1.0 s.** The
measured Hailo pipeline clears the target by 1.9× and the floor by 4.8×.

---

## 3. Latency

### 3.1 Inference

| Budget | Value | Kind |
|---|---|---|
| Inference latency ceiling, mean | **35 ms** | RECOMMENDATION |
| Inference latency ceiling, p95 | **30 ms** | RECOMMENDATION |
| Measured (Hailo) | 17.3 ms mean, 15.2 p50, 15.4 p95 | MEASURED |

35 ms is not a round number chosen for looks: it is the per-frame period at the
measured 28.6 /s operating point (1/28.6 = 34.97 ms). Setting the *inference*
ceiling at the whole frame period is deliberately generous, because it makes the
host-work budget fall out of the same arithmetic:

> **DERIVED: at 28.6 frames/s, the frame period is 35.0 ms and inference occupies
> 17.3 ms, so all non-inference host work — capture handoff through `FrameBus`,
> colour conversion, MediaPipe FaceMesh or Pose on the CPU, module reasoning, the
> event-engine update, overlay annotation — fits in 17.7 ms.** That is a real,
> measured envelope for everything that is not the NPU, and it is the number any
> new CPU-side stage has to be scored against.

**p99 and max inference latency are UNMEASURED.** Given the mean-above-p95 anomaly
in §1, they should be measured before the 35 ms ceiling is treated as verified.

### 3.2 End-to-end alert latency

"Camera frame → notification leaves the device", decomposed. Two paths, and they
have very different budgets.

**Path A — in-car alerts (`driver_attention`).** These fire on the local sinks
(`console`, `sound`, `gpio_buzzer`). The cloud is not in the loop.

| Stage | Best | Typical | Worst | Kind |
|---|---|---|---|---|
| Wait for next camera frame | 0 | 16.7 ms | 33.3 ms | DERIVED from `camera.fps: 30` |
| `FrameBus` staleness + inference | 15.2 ms | 17.3 ms | ≥ 35 ms | MEASURED (p50/mean); worst UNMEASURED |
| Median-filter group delay | — | 70 ms | 70 ms | DERIVED, §2.5 |
| Evidence window (`drowsy_seconds`) | 2000 ms | 2000 ms | 2000 ms | product requirement |
| Local sink dispatch | — | — | — | **UNMEASURED** |
| **Total** | **~2.02 s** | **~2.10 s** | **~2.14 s** | |

The local sink is the gap. `core/alerts.py::SoundSink._play_tone` shells out to
`simpleaudio` or `aplay`. If that call is synchronous on the pipeline thread, a
multi-hundred-millisecond tone is both an added alert latency *and* a candidate for
the >1.0 s inter-frame stall of §2.3. **Measure this**; it is one instrumented
timestamp either side of `AlertManager.notify`.

**Path B — remote alerts (`baby_monitoring`, `fall_detection`).** These have to
reach a phone. `core/uplink.py`'s worker appends the event to a queue and then
sleeps for a fixed `poll_interval_s` (`uplink.py:317`, `self._stop.wait(self._poll_interval_s)`).
There is **no wake-on-record path** — `record()` appends under a lock and returns
(`uplink.py:101–119`).

| Stage | Best | Typical | Worst | Kind |
|---|---|---|---|---|
| Wait for next camera frame | 0 | 16.7 ms | 33.3 ms | DERIVED |
| Inference | 15.2 ms | 17.3 ms | ≥ 35 ms | MEASURED |
| Evidence window (`fall`, `stillness_duration_s`) | 1500 ms | 1500 ms | 1500 ms | product requirement |
| **Uplink poll wait** | **0** | **1000 ms** | **2000 ms** | DERIVED from `cloud.poll_interval_s: 2` |
| HTTP POST to backend | < 5 ms | < 5 ms | — | MEASURED (`tests/test_cloud_backend.py`, cited in `config.yaml`) |
| Backend → APNs → handset | — | — | — | **UNMEASURED, outside the device** |
| **Total, frame → leaves the device** | **1.52 s** | **2.53 s** | **3.57 s** | |

> **DERIVED, and the most actionable number in this document: of the 2.07 s of
> non-evidence latency in the worst case, the uplink poll wait is 2.00 s — 97 % of
> it. Inference is 35 ms, 1.7 %.** Optimising the detector to reduce fall-alert
> latency would be optimising the wrong term by two orders of magnitude.

**Budget: 3.6 s worst case, frame to device egress, for a `critical`-severity remote
alert.** That is the measured/derived chain above, accepted as-is.

**RECOMMENDATION:** have `Uplink.record()` signal the worker's `threading.Event`
when `severity == "loud"` (or `RuleSpec.severity == "critical"`), so a fall flushes
immediately instead of waiting out the poll. Worst case drops from 3.57 s to
~1.55 s — a 56 % cut for a few lines, and it leaves the polling cadence untouched
for everything else. The `event_key` de-duplication already in `record()` makes an
extra flush safe.

---

## 4. CPU

| Budget | Value | Kind |
|---|---|---|
| Steady-state process CPU ceiling | **50 %** | RECOMMENDATION |
| Measured (Hailo, 9.5 min, streaming viewer attached) | 36–37 % | MEASURED |

The measured figure is process CPU as the standard tools report it, i.e. relative to
**one** core. The Pi 5 has four, so the ceiling is 400 % and the pipeline is using
roughly 9 % of the machine. **This reading is not confirmed** — if 36–37 % were
instead machine-relative, the pipeline would be using ~1.5 cores and the headroom
picture changes materially. Confirm it with `top -H` before relying on the margin.

50 % leaves 13 points of headroom over the measured band for work that was not
running, or not running hard, during the 9.5-minute session:

* the health monitor, every `health.check_interval_s: 2.0` seconds, across camera,
  inference, accelerator, dashboard, uplink, wifi, provisioning and hardware;
* the recovery ladder when it actually fires (retry → reopen → restart component);
* BLE provisioning, which opens whenever Wi-Fi drops (`grace_s: 0`);
* the annotate-and-JPEG-encode path, whose cost `config.yaml` explicitly warns about
  ("Annotating + JPEG-encoding competes with the detector for CPU on the Pi").

That last one is partly inside the measured band: the MJPEG check served 79 frames
in 8 s during the same session. But no separate CPU figure was recorded for the
streaming and non-streaming halves of the run, so 36–37 % is the best available
*combined* bound and not a decomposition. **Measure the two halves separately** —
it is the difference between "streaming is free" and "streaming is most of our
headroom".

---

## 5. Memory and storage

| Budget | Value | Kind |
|---|---|---|
| RSS ceiling | **600 MB** | RECOMMENDATION |
| Measured, start of 9.5 min run | 307 MB | MEASURED |
| Measured, end of 9.5 min run | 308 MB | MEASURED |
| `data/` growth ceiling | **2 GB/year** | RECOMMENDATION |
| Telemetry growth, as shipped | ~5.5 GB/year | DERIVED |

600 MB on a 4 GB device leaves room for Pi OS, the page cache the camera and model
loading both lean on, and a second model — `docs/ACCELERATOR.md` lists
`yolov8s_pose.hef` (26 MB) and `face_landmarks_lite.hef` (4 MB) as the obvious next
offloads, and their host-side buffers are not free.

**The +1 MB over 9.5 minutes is not yet a leak measurement, and must not be
extrapolated.** Linearly it is 0.105 MB/min = **151 MB/day**, which would cross
600 MB in under two days and be a serious defect. It is equally consistent with
one-time lazy-load and allocator-arena growth that has already plateaued. A 9.5-minute
window cannot distinguish those. `docs/HARDWARE_VERIFICATION.md` item 11 already
specifies the 48-hour soak that can; **run it before this row means anything.**

Storage is a genuine, arithmetic problem today. `logging.telemetry_interval_s: 0.5`
gives 2 rows/s; `docs/HARDWARE_VERIFICATION.md` measures a row at ~87 bytes;
`core/telemetry.py` flushes after every write and **rotates nothing**:

```
2 rows/s × 87 B = 174 B/s = 15.0 MB/day = 5.5 GB/year
```

That exceeds the 2 GB/year budget by 2.7×. **RECOMMENDATIONS**, in order of how
little they cost: raise `telemetry_interval_s` to 2.0 (→ 1.4 GB/yr, inside budget,
and still ample for the model-training purpose the file exists for); or add size-based
rotation to `TelemetryLogger` — it already owns the file handles and already has a
rename path for header mismatches, so the mechanism is half-built.

---

## 6. Hailo utilisation

**DERIVED**, both readings of the ambiguous 28.6 /s figure:

| Reading | YOLO calls/s | Duty cycle at 17.3 ms/inference |
|---|---|---|
| 28.6 = YOLO inferences/s | 28.6 | **49.5 %** |
| 28.6 = pipeline frames/s, `yolo_every_n_frames: 3` | 9.5 | **16.5 %** |

Either way the accelerator is at most half busy, against a measured standalone
ceiling of 57.9 inferences/s. **Budget: ≤ 60 % sustained duty cycle.**

The practical consequence: **the NPU is not the bottleneck and adding a second
`.hef` is affordable.** The bottleneck is the 17.7 ms/frame of host CPU work derived
in §3.1 — which is precisely where MediaPipe FaceMesh (every frame, CPU) and
MediaPipe Pose (every frame in `baby_monitoring` and `fall_detection`, CPU) sit.
`docs/ACCELERATOR.md` already identifies `face_landmarks_lite.hef` and
`yolov8s_pose.hef` as available prebuilt drop-ins, with the honest caveat that they
are landmark nets only and the tracking/post-processing glue is a real project.
That project is where the next real speedup is; another detector model is not.

Zero accelerator errors over 9.5 minutes. **Budget: 0 accelerator errors per hour;
any non-zero rate is a defect, not a tolerance**, because `core/accel.py` falls back
to CPU on failure and §8 shows what the CPU path costs.

---

## 7. Power: the under-voltage finding

`vcgencmd get_throttled` went **0x0 → 0x50000 during the CPU-only benchmark**, and
stayed 0x0 across every Hailo run.

Decoded against `core/thermal.py:101–105`:

| Bit | Mask | Meaning | Set? |
|---|---|---|---|
| 16 | `0x10000` | under-voltage **has occurred** | **yes** |
| 17 | `0x20000` | arm frequency capped has occurred | no |
| 18 | `0x40000` | throttling **has occurred** | **yes** |
| 0 / 2 | `0x1` / `0x4` | under-voltage now / throttled now | no |

Read carefully, this is a good result wearing bad clothes. The sticky bits fired,
the *live* bits did not, and 64.8 °C is 15.2 °C below the 80 °C soft-throttle point
in `core/thermal.py`. **The throttling was therefore caused by the supply sagging,
not by heat.** The Pi 5 requires a 5 V / 5 A (25 W) USB-PD supply for full
performance; a 5 V / 3 A supply is the classic producer of exactly this signature
under a sustained all-core load.

The immediately useful part: the event happened **only** on the CPU-only path.
Offloading to the Hailo does not merely make inference 16.9× faster, it takes the
device out of the load regime where the power supply becomes marginal. Peak
temperature fell 64.8 → 59.3 °C at the same time.

**Budget:**

| Item | Value | Kind |
|---|---|---|
| `get_throttled` in steady state | **0x0**, both live and sticky | RECOMMENDATION |
| Power supply | 5 V / 5 A USB-PD, official or equivalent | RECOMMENDATION |
| Under-voltage events per 24 h | 0 | RECOMMENDATION |

**RECOMMENDATIONS:**

1. Ship the 5 V/5 A supply, and treat the PSU as a specified part rather than
   whatever is in the box. Under-voltage is silent — clocks drop, frame rate drops,
   §2's floors get closer — and the only visible symptom is a register nobody reads.
2. Clear the sticky bits at boot and surface a *new* sticky bit as a health warning.
   `core/device_health.py:224` already reports `throttled not in (None, "0x0")` as
   WARNING, so the reporting exists; what is missing is a boot-time baseline so
   yesterday's event does not permanently pin the device to WARNING (and so today's
   event is distinguishable from it).
3. Note the whole finding is a rerun of a rejected hypothesis: `SOFT_THROTTLE_C` is
   80 °C and nothing came within 15 °C of it. Do not respond to 0x50000 with cooling.

---

## 8. Thermal

| Budget | Value | Kind |
|---|---|---|
| Sustained SoC temperature ceiling | **75 °C** | DERIVED |
| Absolute ceiling | 80 °C | firmware (`core/thermal.py::SOFT_THROTTLE_C`) |
| Measured, Hailo, sustained | 56.5 – 59.3 °C | MEASURED |
| Measured, idle | 53.8 °C | MEASURED |
| Measured peak, CPU-only | 64.8 °C | MEASURED |
| Fan level under Hailo load | 1 of 4 | MEASURED |

75 °C is not a judgement — it is `core/thermal.py::WARN_C`, the point the product's
own health check starts telling the user the device is warm. Budgeting above the
threshold at which you have decided to alarm the customer is incoherent, so the
budget is the alarm.

Headroom to that budget under Hailo load: **75 − 59.3 = 15.7 °C**. Assuming SoC
temperature tracks ambient roughly 1:1 (**RECOMMENDATION** — a commonly used
approximation, not measured here), the device tolerates a room about 15 °C hotter
than the bench before its own health check complains. For a nursery or a bedroom
that is comfortable margin. For a **car parked in the sun**, where cabin
temperatures of 50–60 °C are ordinary, it is not, and `driver_attention` is the
shipping module. **The in-vehicle thermal case is UNMEASURED and is the largest
untested environmental risk in this budget.** The fan running at 1 of 4 is the
encouraging part: three levels of cooling are in reserve.

---

## 9. Frame drops

`core/frame_bus.py` drops frames **by design**: a single-slot latest-frame handoff,
so a slow inference skips intermediate frames instead of growing a backlog. Drops
are therefore not errors and a drop *rate* is the wrong metric on its own.

| Budget | Value | Kind |
|---|---|---|
| Sustained processed rate | ≥ 15 /s target, ≥ 6 /s floor (§2.6) | DERIVED |
| Implied acceptable drop rate at 30 /s capture | ≤ 50 % target, ≤ 80 % floor | DERIVED |
| Measured drop rate (28.6 processed / 30 captured) | **4.7 %** | DERIVED from MEASURED |
| **Maximum consecutive gap between processed frames** | **< 1.0 s** | DERIVED (§2.3) |

The last row is the one that matters and the one nothing currently measures.
`core/device_health.py::_measure_fps` samples `FrameBus._seq` between health checks
and reports a *mean* rate — which cannot see a single 1.2-second stall inside an
otherwise healthy two-second window, and a single such stall silently resets every
in-progress detection.

**RECOMMENDATION:** track the maximum inter-frame interval alongside the mean, and
surface p99 and max in `/api/health`. It is one `max()` over the delta the FPS
sampler is already computing, and it converts §2.3 from an unverifiable design
constraint into a monitored one.

---

## 10. What the CPU-only path can and cannot meet

The measurement is 293.1 ms/inference, **3.4 inferences/s** — and it is a
*single-model* figure with no host work around it, so a real pipeline is slower, not
faster. Against §2:

**Cannot meet, structurally:**

* **3.4 < 4.17 /s.** `core/activity.py` never accumulates 6 samples in its 1.2 s
  window (`floor(1.2 × 3.4) + 1 = 5`). **`flailing` — a `loud`, push-notifying baby
  alert — cannot fire at all**, nor can `running` or `jumping`. This is not
  degradation; the recogniser reports "collecting motion (5/6 frames)" forever.
* `choking_hazard` (**critical**) needs 3.75 /s for on-time confirmation and is late
  at 3.4 /s.
* Every `driver_attention` gate is measured with a 0.88 s sampling period (§2.4) and
  fires 0.59 s late from the median filter alone (§2.5) — a 2.0 s drowsiness
  threshold that fires at 2.59 s, 29 % late.
* `erratic_hand_movement` samples hands at 1.13 /s; the 1.0 s sustain window
  contains one sample, and the 5-point reversal trail spans 4.4 s of real time
  rather than the ~1 s the feature is reasoning about.

**Can meet:**

* **`fall` — the single most safety-critical detection — needs 1.33 /s and gets 3.4.**
  It confirms on time. This is not luck; it is a consequence of `fall_detection.py`
  computing posture and drop velocity per-frame rather than through
  `ActivityRecognizer`.
* Every zone rule (1.0 /s), `person_absent` (0.13 /s), both `unusual_stillness`
  rules (0.075 and 0.015 /s), `crowd`, `person_lingering`.
* The 1.0 s inter-frame ceiling, with margin: 294 ms per frame.

**Verdict: the CPU-only path is a degraded-but-safe fallback, not a shipping
configuration.** Core presence, zone and fall detection survive; the entire
pose-derived activity layer does not, and the driver module's timing is off by a
quarter to a half.

**Superseded in part by Phase 4/5.** This analysis was written when a device
that lost its accelerator silently continued on CPU. It no longer can:
`requirements-device.txt` ships no `ultralytics`, so there *is* no CPU detector on
a device. `core/accel.py` still returns `None` on failure, but `get_yolo_model`
then raises instead of loading a CPU model, and the service boots **degraded** —
`/api/health` reports `monitoring_active: false` with a reason. The numbers in
this section remain the right description of a **developer machine** running
without an accelerator, and of what the pose layer costs in general; they no
longer describe any state a shipped device can reach.

**RECOMMENDATION (still open, and now narrower):** `monitoring_active: false`
says monitoring stopped; it does not say *which detections* are unavailable. Have
the health endpoint report the detections unavailable at the current measured
frame rate, computed from the table in §2.1 rather than hardcoded. "Degraded" on
a baby monitor should name what stopped working.

**RECOMMENDATION:** set `accelerator.required: true` for any deployment running
`baby_monitoring`. `config.yaml` ships it as `false`, which is right for
`driver_attention` (degraded driver alerts still beat none) and wrong for a product
whose loudest alert is structurally impossible without the NPU.

---

## 11. Summary

| Budget | Value | Kind |
|---|---|---|
| Detector frame rate, target | 15 /s | RECOMMENDATION (§2.4, §2.5) |
| Detector frame rate, floor | 6.0 /s | DERIVED (§2.1, `flailing`) |
| Detector frame rate, absolute floor | 4.17 /s | DERIVED (§2.2, `core/activity.py`) |
| Max interval between processed frames | < 1.0 s | DERIVED (§2.3, `occlusion_grace_s`) |
| Inference latency, mean / p95 | ≤ 35 ms / ≤ 30 ms | RECOMMENDATION (§3.1) |
| Non-inference host work per frame | ≤ 17.7 ms | DERIVED (§3.1) |
| End-to-end alert latency, in-car (local sinks) | ≤ 2.2 s incl. 2.0 s evidence | DERIVED (§3.2) |
| End-to-end alert latency, remote (`critical`) | ≤ 3.6 s incl. 1.5 s evidence | DERIVED (§3.2) |
| Process CPU | ≤ 50 % (single-core-relative) | RECOMMENDATION (§4) |
| RSS | ≤ 600 MB | RECOMMENDATION (§5) |
| `data/` growth | ≤ 2 GB/year | RECOMMENDATION (§5) |
| Hailo duty cycle | ≤ 60 % | RECOMMENDATION (§6) |
| Accelerator errors | 0 /hour | RECOMMENDATION (§6) |
| `vcgencmd get_throttled` | 0x0, live and sticky | RECOMMENDATION (§7) |
| SoC temperature, sustained | ≤ 75 °C | DERIVED (§8, `WARN_C`) |
| Frame drop rate at 30 /s capture | ≤ 50 % | DERIVED (§9) |

### Open measurements, in priority order

1. **p99 / max inference latency**, and the cause of mean > p95 (§1, §3.1). Everything
   about tail behaviour and the <1.0 s stall constraint rests on this.
2. **Maximum inter-frame interval** under real load, not the mean (§2.3, §9).
3. **48-hour RSS soak** — until then the +1 MB is uninterpretable (§5).
4. **`AlertManager.notify` wall time**, especially the sound sink (§3.2).
5. **Whether 36–37 % CPU is core-relative or machine-relative** (§4), and the
   streaming/non-streaming split.
6. **In-vehicle thermal behaviour** — a parked car in sun, `driver_attention`
   running (§8).
7. **Per-frame MediaPipe Pose landmark jitter at the operating frame rate.** See
   below; this one is a live correctness risk, not just a budget gap.

### A frame-rate-dependent correctness risk, flagged here because it is arithmetic

`modules/fall_detection.py:336` computes velocity as a two-frame finite difference:

```python
velocity = (avg_y - self._prev_y) / (now - self._prev_t)     # pixels/second
```

and compares it against thresholds expressed **per second**:
`drop_velocity_threshold: 0.6` and `stillness_velocity_threshold: 0.02`, each scaled
by frame height (480 px). Landmark jitter, however, is roughly constant **per
frame** in pixels — so the *jitter-derived velocity* scales linearly with frame
rate. Working it through for a 1-pixel per-frame jitter on the averaged torso
landmark:

| *f* | Δt | "Still" allows | 1 px jitter reads as |
|---|---|---|---|
| 3.4 /s | 294 ms | 2.82 px/frame | 3.4 px/s ≪ 9.6 px/s threshold — passes |
| 28.6 /s | 35 ms | **0.34 px/frame** | 28.6 px/s ≫ 9.6 px/s threshold — **fails** |

`fall_present = is_still and is_down` (`fall_detection.py:449`). If per-frame torso
jitter exceeds ~0.34 px at 28.6 /s, `is_still` is never true and **falls never
confirm at the shipping frame rate** — while the same code passes at the low frame
rate the thresholds were plausibly tuned at. The `drop` threshold moves the other
way: it gets easier to trip as frame rate rises, producing spurious pendings.

This is derived, not measured — the jitter figure is unknown. **Measure it first**:
log `avg_y` per frame from a static scene with a motionless subject and take the
per-frame delta distribution. If the p95 exceeds 0.34 px, the fix is to make both
thresholds frame-rate-invariant — evaluate displacement over a fixed ~0.25 s window
instead of over one inter-frame gap — rather than to retune the constants, which
would only move the problem to the next frame rate.

This is exactly the failure mode `docs/HARDWARE_VERIFICATION.md` item 7 warned
about: *"several are expressed in seconds but gated on frame counts, so they behave
differently at 5 fps than at 30."*
