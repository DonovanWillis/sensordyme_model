# M13: performance of the device's Python pipeline (desktop measurements, no Pi)

**Status (2026-11-16):** the Raspberry Pi 5 + Hailo-8L was not reachable (not on
the tailnet, no SSH alias, not on this LAN). **Every number here is a desktop
measurement** (this MacBook, or a fleet worker) and is labelled with its host.
None of it is Raspberry Pi performance. What only the Pi can tell is listed in §7.

Raw outputs: `~/sensordyme-audit/2026-11-10/m13-evidence/perf/` (JSON per run).
Code: branch `feat/m13-perf` (base `feat/m13-showstopper` 0a9ddc8).

## 1. Conditions

| | Mac (all pipeline / latency figures unless stated) | Fleet worker `oryn` (decoder on 1,585 images) |
|---|---|---|
| CPU | Apple M3, 8 cores (4P + 4E), 8 GB | AMD Ryzen 7 5700X, job capped at 4 CPUs, CPU only (no GPU) |
| OS / Python | macOS 26.6.2, CPython 3.12.12 | WSL Linux, CPython 3.12.16 |
| Libraries | OpenCV 4.10.0, numpy 1.26.4, mediapipe 0.10.14, torch 2.14.1, ultralytics 8.3.28 | OpenCV 5.0.0, numpy 2.5.4, onnxruntime 1.26.0 |
| Load | **shared and busy**: the iOS Simulator was up and Xcode `swift-frontend` builds and other agents ran concurrently; 1-minute load average between 6 and 113 during the runs (recorded per run in the JSON) | dedicated job |

Method on the Mac: every before/after is an **interleaved A/B on the same host in
the same session**: A = base commit 0a9ddc8 (a `git archive` copy), B = this
branch, run alternately (ABAB / ABBA) through the same benchmark script with
`--repo` selecting whose code runs. Each run reports per-frame p50 / p90; an
arm's figure is the median over its runs, and the per-run values are given to
show the spread. The claims use the quiet window (1-minute load 3.3-7.0); a
round taken under load 17-115 is kept as evidence but not used. A change is
claimed only where every interleaved run agrees.

## 2. Per-stage cost of the real inference loop (desktop)

`benchmarks/m13_pipeline_profile.py` builds the production `DeviceRuntime` with
`main.build_runtime` from config.yaml (as the virtual device does; real models,
real FrameBus, real dashboard), publishes a fixed 640x480 sequence (a public-
domain portrait, so FaceMesh tracks a real face) on the runtime's FrameBus from
a capture thread at 30 fps, and drives `_inference_pass` (the loop body,
unchanged). Each stage is timed by wrapping instance attributes in the
benchmark process; the runtime's own stage clock (core/stage_timing.py) supplies
detector / decode / FaceMesh / Hands / tracking / behaviour_reasoning /
frame_age / frame_to_decision.

Detector backends:
- **hailo-replay**: the device's own `core/accel.HailoYOLO.predict` (letterbox,
  BGR->RGB, head pairing, decode, NMS, frame mapping) on a **SIMULATED chip**
  that answers with recorded float heads in ~0 ms. Inference time is therefore
  NOT represented; everything around it is the device's code.
- **ultralytics**: what a dev Mac runs (yolo11n.pt, CPU torch). Not the device.

Runs: camera thread at 30 fps, 900 analysed frames (600 for ultralytics) after
1,800 untimed warm-up frames (so the 60 s eye baselines hold a production-size
window), driver_attention from config.yaml (`yolo_every_n_frames: 3`), two runs
per arm, interleaved ABBA, 1-minute load average 3.3-7.0 (the quiet round;
`pipeline/ab_tables.md`). A first round under heavy load (load average 17-115:
an Xcode build and the iOS Simulator running) is kept in
`pipeline/round1_heavy_load/`; its stage times were 1.5-3x higher and too
contended to compare arms, so it is not used for claims.

**Per-stage cost, this branch (M13), Mac M3, ms p50 / p90 (median over runs).**
Frames with each stage: detector 1 in 3; Hands on ~1 in 10 (driver_attention
runs it only when a phone is in play); annotate / JPEG at the stream cadence.

| stage | hailo-replay, no viewer | hailo-replay, both viewers (device-drawn + sd-overlay/1) | ultralytics (dev Mac), no viewer |
|---|---|---|---|
| capture pick: frame_age (camera publish -> loop picks it) | 0.04 / 0.08 | 0.03 / 1.0 | 0.04 / 18.8 |
| detector_total (one shared run) | 1.87 / 3.70 | 1.42 / 3.51 | 43.5 / 63.6 (CPU torch) |
| — hailo_preprocess (letterbox + BGR->RGB) | 0.28 / 0.52 | 0.22 / 0.49 | — |
| — hailo_inference | ~0 (SIMULATED chip) | ~0 (SIMULATED) | — |
| — decoder_cpu (legacy, conf 0.20, incl. NMS) | 1.51 / 3.21 | 1.15 / 2.98 | (inside ultralytics) |
| face_landmarks (FaceMesh, every frame) | 4.30 / 8.02 | 3.92 / 6.61 | 3.81 / 5.10 |
| hands (MediaPipe Hands, when run) | 21.7 / 29.7 | 20.9 / 32.5 | 19.9 / 28.7 |
| tracking (shared tracker, per detector run) | 0.09 / 0.59 | 0.08 / 0.54 | 0.02 / 0.02 |
| behaviour_reasoning (module logic, no models) | **2.00 / 3.16** | **1.76 / 2.88** | **1.68 / 2.40** |
| perception engine begin + end of frame | 0.12 / 0.23 | 0.11 / 0.20 | 0.04 / 0.06 |
| alert_dispatch (policy for the frame's events) | 0.02 / 0.04 | 0.02 / 0.04 | 0.02 / 0.03 |
| custom rules | 0.01 / 0.06 | 0.01 / 0.05 | 0.01 / 0.04 |
| monitoring nodes | 0.004 / 0.006 | 0.003 / 0.006 | 0.003 / 0.005 |
| telemetry snapshot + log | 0.01 / 0.02 | 0.01 / 0.02 | 0.01 / 0.02 |
| dashboard status push | — | 0.007 / 0.011 | — |
| annotate (driver_attention, device-drawn view) | — | 3.57 / 5.65 | — |
| JPEG encode, annotated (q 60, 640x480) | — | 0.81 / 1.25 | — |
| sd-overlay/1 build + header encode | — | 0.04 + 0.10 / 0.10 + 0.15 | — |
| JPEG encode, overlay clean frame | — | 0.85 / 1.37 | — |
| **whole pass, work only** (excl. waiting for the frame) | 8.16 / 26.2 | 8.38 / 33.4 | 6.31 / 64.0 |
| **frame_to_decision** (capture -> alert decision) | 8.44 / 26.6 | 7.16 / 27.2 | 17.9 / 63.9 |
| processed fps (camera 30) | 29.96, 29.74 | 29.67, 27.94 | 25.3, 28.1 |
| process CPU (% of one core) | 34.7, 34.2 | 39.4, 39.2 | 120.9, 118.5 |
| RSS peak (MiB) | 227, 235 | 228, 230 | 269, 276 |

**Before/after on the same host and load (base 0a9ddc8 vs M13, p50, two
interleaved runs per arm; per-run values in `pipeline/ab_tables.md`):**

| config | stage | base | M13 | change |
|---|---|---|---|---|
| hailo-replay, paced, no viewer | behaviour_reasoning | 2.69 (2.57, 2.81) | 2.00 (1.99, 2.01) | **-26 %**, every M13 run below every base run |
| | pass work | 8.69 | 8.16 | -6 % |
| | frame_to_decision p50 / p90 | 8.88 / 28.5 | 8.44 / 26.6 | -5 % / -6 % |
| | decoder_cpu | 1.55 | 1.51 | within noise in paced mode (cold caches between frames) |
| | CPU % of one core | 35.6, 36.4 | 34.7, 34.2 | -1.5 points |
| hailo-replay, paced, both viewers | behaviour_reasoning | 2.42 (2.27, 2.57) | 1.76 (1.72, 1.80) | **-27 %** |
| | pass work / frame_to_decision | 8.43 / 7.50 | 8.38 / 7.16 | -1 % / -5 % (annotate + encodes unchanged by design) |
| hailo-replay, stepped (every frame analysed, no waiting) | behaviour_reasoning | 2.02 (2.08, 1.96) | 1.24 (1.22, 1.26) | **-39 %** |
| | decoder_cpu | 0.95 (0.98, 0.93) | 0.71 (0.68, 0.74) | **-26 %** |
| | pass work | 6.18 | 4.77 | **-23 %** |
| | processed fps (throughput ceiling) | 117.6, 127.7 | 158.5, 152.0 | +27 % |
| ultralytics, paced, no viewer | behaviour_reasoning | 2.46 (2.52, 2.39) | 1.68 (1.81, 1.55) | **-32 %** |
| | frame_to_decision | 22.7 | 17.9 | noisy (torch detector dominates) |

Nothing got slower: no stage's median moved up by more than the run-to-run
spread, RSS is unchanged, and every frame was still analysed at camera rate in
the paced runs. In the stepped runs FaceMesh and Hands also read ~5 % lower on
M13, which is not attributable to these changes (they do not touch the graphs)
and is reported as noise.

Reading it:
- On the device code path (hailo-replay), the per-frame CPU work that is ours
  (behaviour reasoning, perception engine, rules, nodes, dispatch, decode) is a
  few milliseconds; FaceMesh dominates every frame, Hands dominates the frames
  it runs on (driver_attention gates it to phone-relevant detector frames).
- `annotate` + JPEG are paid only while a viewer wants the device-drawn picture,
  at `dashboard.stream_fps` (12), not per analysed frame; the sd-overlay/1 path
  (clean frame + header) skips `annotate` entirely.

## 3. Bottlenecks in our code: changes made (behaviour identical, tests pin it)

### 3.1 `AdaptiveBaseline.reference()` sorted the whole 60 s window, twice per frame
`core/eye_signal.py`. driver_attention keeps two baselines (EAR and iris
openness, `ear_baseline_window_s: 60`) and calls `observe()` + `reference()` on
each every frame; `reference()` sorted every sample in the window: 1,800 floats
at 30 fps, twice per frame. The window is now kept sorted as samples arrive and
leave (`bisect.insort_right` in, leftmost equal out = the oldest), so
`reference()` is an index. While a NaN is in the window it falls back to the
original sort. `tests/test_m13_perf.py` pins `reference()` to the frozen pre-M13
implementation on random streams with ties, signed zeros, ±inf, NaN, irregular
time steps and resets.

`benchmarks/m13_baseline_cost.py` (Mac, two baselines x observe+reference per
frame, a full 60 s window at 30 fps = 1,800 samples, 3,000 timed frames, 3
interleaved rounds per arm; `baseline_cost_*_r*.json`):

| per frame | base 0a9ddc8 (3 runs) | M13 (3 runs) |
|---|---|---|
| p50 | 0.64 / 0.60 / 0.43 ms | 0.0042 / 0.0036 / 0.0027 ms |
| p90 | 0.79 / 0.77 / 0.69 ms | 0.0046 / 0.0040 / 0.0029 ms |

In the real loop (§2) `behaviour_reasoning` dropped by 0.7-0.8 ms per frame
(-26 % to -39 %) in every configuration; it was the largest single item of our
own per-frame CPU work. (On the Cortex-A76 the sort is expected to cost more;
not measured.)

### 3.2 The raw-head decoders did a redundant full pass over the class scores
`core/accel.py` (the decoder default stays `legacy`; thresholds, NMS, cap,
anchor convention unchanged):
- **legacy**: `scores.max(axis=1)` after `scores.argmax(axis=1)` was a second
  pass over (anchors x 80) per scale; the best score is now gathered at the
  argmax (the same float32; NaN still propagates, argmax returns the first NaN).
- **reference**: `np.nonzero(scores >= conf)` built and scanned the whole 2-D
  (anchors x classes) mask; anchors are now prefiltered by their per-anchor max
  and only those rows are thresholded: the same pairs in the same row-major order.

Pinned bit for bit (boxes, float32 scores, order, refusal messages) against the
frozen pre-M13 bodies on random and planted heads, both decoders, with and
without the `{80: hand}` map and a class filter, threshold edges included; the
existing decoder suites (test_hailo_decoder, test_phase3_raw_decode,
test_m10_hand_class_decode, test_m5_hw_validation, test_m83_inference_path,
test_m101_hw_readiness) pass unchanged. On the fleet, the product decoders at
HEAD equal the independent `hwv_decode` twins on all 3,170 decoded frames
(1,585 images x 2 decoders, max |Δ| 3.05e-5), with identical detection counts
before and after (11,883 legacy / 11,510 reference).

Decode time per frame, the device's `HailoYOLO._decode_raw`, identical float head
tensors, driver_attention's classes at conf 0.20 (**float tensors, desktop**):

| host | decoder | base 0a9ddc8 p50 / p90 | M13 p50 / p90 | change |
|---|---|---|---|---|
| Mac M3, 16 frames x 100 repeats x 6 interleaved rounds per arm, load 3.5-4.3 (`decoder_latency/ab_summary.json`; median of the per-round p50 / p90) | legacy | 0.73 / 1.10 ms (round p50 0.66-0.76) | 0.61 / 1.05 ms (0.55-0.63) | p50 **-16 %**; p90 within noise (per-round p90 0.82-1.27 vs 0.94-1.29) |
| | reference | 3.06 / 3.71 ms (2.96-3.09) | 0.70 / 1.17 ms (0.61-0.72) | p50 **-77 %**, p90 -68 % |
| oryn x86 (Ryzen 7 5700X, 4 CPUs), 1,585 val images, A/B alternated per frame (jobs `20261116-023006-932455` base, `20261116-025908-be792a` M13) | legacy | 0.97 / 1.42 ms (p95 1.71) | 0.69 / 1.12 ms (p95 1.36) | -29 % |
| | reference | 1.81 / 2.53 ms (p95 2.78) | 0.91 / 1.36 ms (p95 1.60) | **-50 %** |

On the Mac the base reference decoder was dominated by the 2-D `np.nonzero`
(2.9 ms of the 80x80 scale alone, numpy 1.26 arm64); after M13 the two decoders
cost about the same, so the "reference costs more host CPU" argument against
candidate B (M10.1: +0.9 ms on x86) shrinks to +0.2 ms on x86 and ~+0.1 ms here.

### 3.3 An alert waited for the rest of the uplink poll interval
`core/uplink.py`. The uplink worker slept `poll_interval_s` (2 s) between syncs,
so a queued alert reached the backend (and from there the owner's phone) up to
2 s later, 1 s on average, for no reason. `push_event()` now wakes the worker;
the event goes up within one round trip. The reconnection backoff is never
shortened (after a failed sync only `close()` ends the wait, so an outage plus
an alert storm is not a retry storm); merged events (the same alert episode) do
not wake it; what is sent is unchanged. Measured in §5.

## 4. Measured, not changed (and why)

| Where | Cost (M3) | Why it was not changed here | The fix |
|---|---|---|---|
| `modules/driver_attention.py` `annotate()`: mediapipe `draw_landmarks` of the 2,556-edge FACEMESH_TESSELATION, one Python `cv2.line` + isinstance checks per edge | ~3.4 ms per streamed frame (only while a viewer wants the device-drawn view; 12 fps) | file owned by another workstream (constraint) | precompute the tesselation index arrays once and draw them with one `cv2.polylines` call (same pixels per segment); or let viewers use sd-overlay/1, which never calls `annotate()` |
| `driver_attention._engine_step`: frame-quality thumbnail `cv2.resize(frame, (32, 24), INTER_AREA)` on the full 640x480 frame, every frame | 0.40-0.44 ms per frame | same file; and no cheaper resize is bit-identical: a two-step INTER_AREA is 9x faster but differs on 1,720 pixels in 50 random frames (a numpy block mean is slower). The quality score feeds only `frame_quality` (an abstention input), but "identical" was the bar | an owner decision: accept the two-step thumbnail (re-baseline frame_quality) or keep it |
| `driver_attention`: BGR->RGB `cvtColor` done twice on detector frames that run Hands (once for Hands, once for FaceMesh) | 0.04 ms each | same file | convert once per frame and pass it to both graphs |
| `core/dashboard.py push_frame` / `push_overlay_frame`: `cv2.imencode` (q 60) on the inference thread | ~0.8 ms per streamed frame on this portrait | push_overlay_frame is the overlay path (constraint); moving the encode to the stream thread changes when a frame becomes visible | keep: at 12 fps it is small on the Mac; measure on the Pi (§7) |
| `core/uplink.py _spool_append`: `os.fsync` per alert event, on the inference thread, after the decision and the buzzer command | ~0.3 ms on APFS (macOS fsync does not flush the drive cache) | the durability guarantee of M8 (an alert survives a power cut) depends on it | measure on the Pi's SD card (§7); if it is tens of ms, append on the uplink worker instead |
| `core/tracking.ObjectTracker.tracks()` rebuilt on every frame in `end_frame` | ~10 µs at 1-3 tracks (0.15 ms at the ~50 tracks of busy Open Images scenes) | negligible for a cabin | cache the tuple until the next update |
| mediapipe `SolutionBase.process` builds a namedtuple class per call | ~0.09 ms per graph call | not our code | — |
| sd-overlay/1 `build` + header encode | 0.03 + 0.09 ms per streamed frame | `core/overlay.py` (constraint); already cheap | — |

**Redundant model execution:** none found. The perception engine already runs
the detector once per frame for the union of every consumer's classes at the
lowest conf asked (core/perception.SharedDetector), and FaceMesh/Hands once per
frame (SharedGraph). In the profile the detector ran on exactly 1 frame in 3
(`yolo_every_n_frames: 3`) and FaceMesh once per frame.

**Queues and stale frames (verified, unchanged):** the FrameBus is one slot
(inference always takes the newest frame; frames that arrive while a frame is
processed are skipped, measured as `published_frames` > `processed_frames` and
by `frame_age`); the live stream is one shared slot per stream (a slow viewer
skips frames, `max_pending_frames_per_client: 1`); the alert pattern player
holds at most 2 waiting patterns (least urgent dropped); the uplink queue is a
bounded deque (`queue_max`, evictions counted); the local event journal and
stage-timing rings are fixed-size. Nothing in the loop grows with uptime.

## 5. Event-to-alert latency in software (SIMULATED)

`benchmarks/m13_event_latency.py`. The virtual device (`sim.harness.SimStack`):
scripted perception, the simulated buzzer sink, the real runtime, alert policy,
dashboard and uplink, `cloud/app.py` in process. Wall clock (`perf_counter`,
which the SimClock does not touch) from the moment the inference loop picks up
the decisive frame. Seven alerts from the drowsy, phone and distracted scenarios.

Decision path (M13 branch, 7 alerts: drowsiness x2, fatigue_warning x2,
phone_use, eyes_down, distraction; `event_latency_decision.json`), ms from the
decisive frame's pick-up (**SIMULATED, desktop**):

| step | M13 p50 | p90 | max | base 0a9ddc8 p50 |
|---|---|---|---|---|
| (a) alert decision (`AlertManager.notify` returned) | 1.00 | 1.03 | 1.04 | 1.09 |
| (b) buzzer sink `play()` called (inside notify; simulated sink) | 1.00 | 1.03 | 1.04 | 1.08 |
| (c) event in the local journal / `/api/events` (`push_event` returned) | 1.25 | 1.29 | 1.31 | 1.33 |
| (c') one loopback HTTP `GET /api/events` (separate read; the event was present in 7/7) | 1.21 | 4.92 | 10.1 | 1.11 |
| (d) queued for the uplink (`push_event` returned, spool fsync'd) | 1.50 | 1.56 | 1.60 | 1.58 |
| (d') one `sync_once()` POST to the in-process backend (stored in 7/7) | 5.78 | 6.54 | 7.03 | 6.45 |

Both arms raised the same 7 alerts in the same order (behaviour identical end
to end on the virtual device); `event_latency_decision{,_base}.json`.

(a) is almost entirely the module's own `process()` of the decisive frame with
scripted perception (no model runs); on the device, add the detector (when the
frame is a stride frame), FaceMesh and, for phone frames, Hands (§2/§7). The
buzzer is commanded inside the decision, before any log, journal or network
work. On the Pi the GPIO pulse follows on the pattern-player thread
(`alert_output`, Pi only).

Uplink scheduling, real threads, real time, `cloud.poll_interval_s: 2.0`, 20
alerts pushed at random moments of the poll cycle, push -> backend 200
(`event_latency_uplink_{base,m13}.json`):

| | p50 | p90 | max |
|---|---|---|---|
| base 0a9ddc8 | **1,285 ms** | 1,726 ms | 1,978 ms |
| M13 (wake on push) | **10.7 ms** | 24.2 ms | 40.5 ms |

So, end to end in software on this host: frame picked -> buzzer commanded ~1 ms,
-> local `/api/events` ~1.3 ms, -> backend stored ~10-25 ms after M13 (was
~1.3 s median, up to 2 s, waiting for the next poll). What the app then takes
(push delivery, its own polling of `/api/events`) is outside the device.

**Found and removed:** the uplink scheduling delay (§3.3). Nothing else on this
path waits: the decision, the buzzer command, the local journal and the uplink
queue all happen in the same pass, in that order (the alert is never behind a
log write), within ~2 ms of the frame on this host.

## 6. Decoder comparison without the Pi (float tensors)

INT8 tensors from the Hailo do not exist yet. What was done instead, all on the
**float** heads of the shipping damoyolo_tinynasL20_T (the ONNX the .hef was
compiled from, sha256 02d7ef7f…), through the device's own `HailoYOLO` on a
simulated chip, with the call driver_attention makes (its 19 classes, conf 0.20):

- **(a) CPU decode latency, both decoders on identical inputs**: §3.2 table
  (Mac: 16 real-shaped frames = the 6 known-good images + 10 Open Images val
  frames with a person or phone; oryn: all 1,585 val images).
- **(b) Detection differences, legacy (A) vs reference (B), same tensors**,
  1,585 Open Images val images, product gates, IoU >= 0.5 same-class greedy
  matching (fleet jobs `20261116-023006-932455` at base and
  `20261116-025908-be792a` at M13: identical counts and matches, as the
  bit-identity of §3.2 requires). **Float tensors, x86 worker; NOT INT8, NOT in-cabin.**

| class @ gate | boxes A / B | matched | A matched / B matched | matched IoU p50 (p5) | images with the class: only A / only B / both | centre shift A-B at stride 8 / 16 / 32 (px) |
|---|---|---|---|---|---|---|
| person @ 0.50 | 2,866 / 2,852 | 2,804 | 97.8 % / 98.3 % | 0.79 (0.60) | 0 / 0 / 1,333 | +3.9 / +7.7 / +14.2 (x); +3.9 / +7.7 / +12.2 (y) |
| cell phone @ 0.20 | 539 / 604 | 478 | 88.7 % / 79.1 % | 0.76 (0.55) | **0 / 50 / 322** | +3.8 / +8.0 / +15.0 (x) |
| eating/drinking classes @ 0.20 | 2,310 / 2,638 | 1,765 | 76.4 % / 66.9 % | 0.67 (0.53) | — | +3.7 / +7.5 / +14.2 (x) |

  Reading it: the boxes the two decoders share sit **stride/2 apart** (4 / 8 /
  16 px; slightly less at stride 32 where frame clipping trims the shift): the
  anchor-offset difference, exactly as M10.1 predicted. Person presence is the
  same image for image; the reference decoder reports a phone on **50 more
  images** (multilabel: a phone that loses its anchor's argmax to a person or a
  cup), and never misses one legacy reports. Scores of matched boxes are the
  same (p95 |delta| 0). On the Mac's 16 frames: person 19/19 matched, phone 5 vs
  5 (4 matched), other classes 23 vs 31.

Accuracy against labels at the product gates is the M11 result (fleet job
`20261114-182049-0c1222`, Open Images val, float): reference vs legacy, person
@0.50 precision/recall 0.930/0.553 vs 0.902/0.536, phone @0.20 0.527/0.914 vs
0.526/0.805 (docs/research/M11_MODEL_DECISION.md §2). Not redone.

### 6.1 The isolated reference-decoder development configuration

- `deploy/dev/reference_decoder.overlay.yaml`: one key,
  `accelerator.yolo_decoder: reference`. Nothing installs or loads it;
  `config.yaml` keeps `legacy`.
- `scripts/hw_validation/dev_config.py` merges it onto a **copy** of a device's
  config, validates the result with `core/config_schema.py` (any error: nothing
  written), prints the exact difference and refuses an output inside the
  installed tree.
- `hwv_stage.py latency --latency-decoder installed|legacy|reference|both`
  (new) times the decoder(s) **in process only**; `both` runs legacy and
  reference in ABAB blocks on the same frames and checks each against the
  stride budget. The default (`installed`) is the pre-M13 stage.

### 6.2 Exact Pi commands (when the device is reachable)

On the Pi, from the hwval checkout of this branch (docs/MONDAY_HARDWARE_SESSION.md
§5 for the ground rules; nothing here writes under the install tree):

```bash
export U=<service user>  D=/home/$U/sensordyme  PY=$D/.venv/bin/python
V=~/hwval/sensordyme_model/scripts/hw_validation
$PY $V/hwv_stage.py pins --install-dir $D
sudo systemctl stop sensordyme.service                  # the stages never do this themselves
$PY $V/hwv_stage.py model --install-dir $D
$PY $V/hwv_stage.py parse-hef --install-dir $D
$PY $V/hwv_stage.py capture --install-dir $D --bundle ~/hwval/capture_kg
$PY $V/hwv_stage.py parity --bundle ~/hwval/capture_kg --decoder both
$PY $V/hwv_stage.py latency --install-dir $D --latency-decoder both   # M13: both decoders, ABAB
sudo systemctl start sensordyme.service && systemctl is-active sensordyme.service
```

Optional, a development run of the whole device on the reference decoder
(development unit only, owner approval, never a customer device; the service
stays stopped meanwhile and is started again afterwards):

```bash
$PY $V/dev_config.py --install-dir $D --out ~/hwval/dev/config.reference.yaml
sudo systemctl stop sensordyme.service
cd $D && $PY main.py --config ~/hwval/dev/config.reference.yaml --no-display   # Ctrl-C to end
sudo systemctl start sensordyme.service
```

Then, with the service running: `SENSORDYME_TOKEN=$TOKEN $PY $V/hwv_stage.py observe --minutes 10`
records the runtime's own per-stage timings (decoder_cpu, frame_to_decision) on
the Pi for whichever config is running.

## 7. What only the Pi can tell us

- **Hailo-8L inference time and fps** for this .hef, and the accelerator duty
  cycle (`latency`, `hailortcli monitor`); every desktop run here simulates the
  chip.
- **PCIe transfer** host<->chip (inside `hailo_inference`), and whether the
  uint8 letterbox copy is a measurable part of it.
- **INT8 decode behaviour and cost**: the number of above-threshold candidates
  on INT8 class heads (quantised scores sit on discrete steps, which can change
  how many anchors pass 0.20 and therefore NMS cost), the INT8-vs-float parity
  per decoder (`parity --decoder both`), and the decoders' Cortex-A76 time
  (`latency --latency-decoder both`).
- **FaceMesh and Hands on the Cortex-A76** (XNNPACK on 4 A76 cores at 2.4 GHz
  vs the M3 here), and therefore the real per-frame budget left at 30 fps.
- **Thermal throttling** over a 30-minute soak (`thermal`): whether processed
  fps holds, and whether any of the per-frame costs above grow with heat.
- **The SD card's fsync time** on the alert path (§4) and the JPEG encode cost
  at the stream rate on the Pi.
- **Camera->buzzer end to end** with real optics and a real buzzer
  (docs/HAILO_HARDWARE_VALIDATION.md step 11), which no simulation replaces.

## 8. Reproduce

```bash
PY=.venv/bin/python   # from the repo root
# per-stage profile (add --repo <other checkout> for an A/B)
$PY benchmarks/m13_pipeline_profile.py --detector hailo-replay \
    --tensors ~/sensordyme-fleet/runs/20261116-023006-932455/artifacts/tensors \
    --paced 30 --viewers both --frames 900 --warmup 1800 --out run.json
$PY benchmarks/m13_decoder_latency.py --tensors ~/sensordyme-fleet/runs/20261116-023006-932455/artifacts --repeats 100
$PY benchmarks/m13_baseline_cost.py
$PY benchmarks/m13_event_latency.py --out event_latency.json
# fleet: legacy vs reference on 1,585 val images + the tensors above
~/sensordyme-fleet/sdrun submit -t oryn --repo . --no-gpu --cpus 4 --mem 4G -- bash research/m13/oryn_decoder_compare.sh
```
