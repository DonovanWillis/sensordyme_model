# M10.1: hardware accuracy gates, deployment decision, in-cabin readiness

**Status (2026-11-14): no hardware measurement has been made.**
- The Pi 5 + AI HAT+ (Hailo-8L) + Camera Module 3 Wide NoIR arrives
  2026-11-15. Every Pi number below is pending.
- Every accuracy number below comes from the float model on the Open Images
  val split (evidence level L1). None is from Hailo INT8, none is from a
  vehicle, and none is a driving result.

## 1. Candidates (this mission's lettering; M10's in brackets)

| | Candidate | Person AP50 | Person R@P0.80 (thr) | Large-person R@P0.80 | Phone AP50 | Hand AP50 | Networks | Product change to run it |
|---|---|---|---|---|---|---|---|---|
| **A** | shipping `.hef`, `legacy` decoder [M10 "C legacy"] | 0.660 | 0.605 (0.378) | 0.703 | 0.728 | — | 1 | none: **installed** |
| **B** | shipping `.hef`, `reference` decoder [M10 "C reference"] | 0.718 | 0.661 (0.332) | 0.733 | 0.795 | — | 1 | `accelerator.yolo_decoder: reference` (opt-in, exists) |
| **C** | EXP-08 three-class, `reference` + `{80: hand}` [M10 "A"] | 0.697 | 0.630 (0.295) | 0.725 | 0.837 | 0.723 | 1 | a compiled `.hef`, which **does not exist** (compile procedure written, not run) |
| **D** | B for person + C for phone/hand [M10 "B"] | 0.718 | 0.661 (0.332) | 0.733 | 0.837 | 0.723 | 2 | C's `.hef` **and** two network groups on one VDevice in `core/accel.py`, which **does not exist** |

Source: fleet job `20261114-034527-625fa0`, 1,585 val images, the product's own
letterbox, decode and NMS on the simulated chip, each model at its own
precision-0.80 threshold. Full CIs are in `docs/research/M10_MODEL_CANDIDATES.md`.

**Desktop cost (not a Pi):**
- one network: 48.7 ms p50 at 4 threads on CPU, 6.28 ms on CUDA;
- host decode at 0.20: A 1.04 ms, B 1.95 ms, C 2.04 ms, D about 4 ms.
- On the Pi 5 + Hailo-8L the published reference point for A is 22.07 ms mean
  inference and 29.5–29.9 fps. **That reference point was measured on an
  earlier phase's device, not this unit.**

## 2. Deployment decision

**Installed: A, unchanged.** No candidate has INT8, latency, thermal or in-cabin
evidence. Nothing here changes the production decoder or weights.

| | Verdict | Why |
|---|---|---|
| A | keep installed | the known-good model; every gate below is measured against it |
| B | **first candidate**, promotable to a development unit only after Pi evidence | the same network and the same Hailo cost. +0.056 person R@P0.80 and +0.067 phone AP50 in float. The only added cost is host decode (+0.9 ms on x86; Pi unmeasured) |
| C | experimental | no compiled `.hef`, and the box-logit range is about 3× A's (an INT8 risk). Only candidate with hand. Needs the DFC compile with owner approval, then the same gates |
| D | experimental, **not implemented** | 2× network work by construction. Its extra person accuracy over C is 0.008 on large subjects (CI −0.003 to +0.020), under the 0.02 bar that would justify a second network. Needs both C's `.hef` and new two-network code |

**Promotion rule for B** (nothing switches automatically; owner approval still
required):
1. `parity --decoder both`: B's INT8-vs-float agreement on the known-good images
   is no worse than A's.
2. `latency`: B's decode p95 on the Pi keeps `predict()` p95 inside the stride
   budget (100 ms at 30 fps, `yolo_every_n_frames` 3) with < 1 % late frames.
3. `thermal` (30 min, service running): never throttled; fps at the end within
   10 % of the start. Run once with A and once with B (development unit).
4. In-cabin sessions (section 4): B's false alerts per hour are no worse than A's
   at matched operating points. Thresholds are re-derived on INT8 scores (the
   float thresholds do not transfer).
5. Then `accelerator.yolo_decoder: reference` on a development unit, never
   straight to customers.

**C and D** go through the same gates plus their own. C's INT8 must cost no more
than 0.02 person R@P0.80 or 0.03 hand AP50 vs float. D's two-network pipeline
must hold ≥ 29.5 fps.

## 3. What M10.1 made ready (and verified without hardware)

- **INT8 parity for both decoders:** `hwv_stage.py parity --decoder legacy|reference|both`
  (default `both`). The reference arm decodes the device's own INT8 tensors
  offline. The production decoder is never switched.
- **Reference expectations:** `scripts/hw_validation/known_good/expected_float_reference.json`
  (fleet job `20261114-142158-db510c`). The same job rebuilt the legacy
  expectations and reproduced the committed file exactly on all 6 images:
  identical input tensors, head statistics to 1e-16, and identical detections
  (score and box difference 0.0).
- **A defect found and fixed by verification.** The offline "reference" spec the
  tools used (`hwv_decode.REFERENCE`) is the upstream DAMO transcription:
  `score > conf`, NMS 0.70, 100 boxes. The product's reference decoder
  (`core/accel.py`) keeps `score >= conf`, uses NMS 0.50 and keeps 500 boxes.
  - The parity gate would have compared two different decoders.
  - New `hwv_decode.DEVICE_REFERENCE` is pinned equal to `core.accel` on random
    heads, with and without the hand map.
- **The gate tested on itself.** In job `20261114-143113-fc4b32`, capture then
  parity ran with the FLOAT model behind the simulated chip. Both decoders
  agreed exactly: 100 and 251 matched detections, score delta 0, IoU 1.0. On
  the Pi, any disagreement the gate reports therefore comes from the chip.
- **Hand class in the diagnostic tools:**
  - `hwv_decode.decode(extra_classes=)` reads a mapped channel exactly as
    `core.accel` does;
  - `compare_float_vs_int8.py --extra-classes 80:hand`;
  - `capture_hailo_outputs.py` reads the installed map, for both its offline
    counts and its production boxes;
  - `hwv_stage.py arch --second-extra-classes 80:hand`;
  - unchanged bit for bit without a map. This closes two M10 open items.
- **Threshold procedure:** `research/m8/analysis.r_at_p` returns the score of the
  last box it keeps, so the threshold is inclusive. `operating_point` and both
  device decoders also keep `score >= threshold`, compared in float32. A
  derived threshold reproduces its measured operating point; this is pinned in
  `tests/test_m101_hw_readiness.py`.

## 4. The exact next hardware test (H1: INT8 parity of the shipping `.hef`, both decoders)

On the Pi, after the Sunday install (`docs/SUNDAY_DEVICE_ACTIVATION.md`). These
are the commands of `docs/MONDAY_HARDWARE_SESSION.md` §5, in this order:

```bash
export U=<service user>  D=/home/$U/sensordyme  PY=$D/.venv/bin/python
V=~/hwval/sensordyme_model/scripts/hw_validation
$PY $V/hwv_stage.py pins --install-dir $D                       # any state
sudo systemctl stop sensordyme.service                          # the stages never do this themselves
$PY $V/hwv_stage.py model --install-dir $D
$PY $V/hwv_stage.py parse-hef --install-dir $D
$PY $V/hwv_stage.py capture --install-dir $D --bundle ~/hwval/capture_kg
$PY $V/hwv_stage.py parity --bundle ~/hwval/capture_kg --decoder both
$PY $V/hwv_stage.py latency --install-dir $D
sudo systemctl start sensordyme.service && systemctl is-active sensordyme.service
SENSORDYME_TOKEN=$TOKEN $PY $V/hwv_stage.py thermal --minutes 30   # service running
```

Read-only on the model: nothing writes under the install tree, and nothing
changes a threshold or the decoder. The result is either "B's float gain
survives INT8 on this chip" or "it does not", per decoder, with the numbers.

## 5. In-cabin validation, ready to run (parked vehicle only)

`scripts/hw_validation/cabin_session.py` + `evaluation/schemas/cued_session_v1.yaml`:

```bash
python3 $V/cabin_session.py plan --preset core --out ~/hwval/cabin/S01_day      # prints the operator sheet
python3 $V/cabin_session.py run --plan ~/hwval/cabin/S01_day --subject S01 \
    --session S01-2026-11-16-a --vehicle V01 --seat driver \
    --lighting day --eyewear none --occupancy driver_only --mount dash_centre
python3 $V/cabin_session.py score --session ~/hwval/cabin/S01_day --void <cues not performed>
```

**How it records:**
- Each cue is printed with its time. The tool records the device's own per-alert
  availability (`/api/health`) every second and its event log (`/api/history`)
  at the end.
- No frames, video or audio are written. The token is read from
  `SENSORDYME_TOKEN` and never written.

**How it scores:**
- Through `evaluation.event_metrics`, one clip per block:
  - caught events and latency from the cue (which includes reaction time);
  - false alerts, and false alerts per hour;
  - alerts on legitimate behaviour (mirror and cluster glances);
  - drowsiness on an eyes-open look into the lap, counted as false;
  - abstentions (the device said it could not evaluate).
- One session never meets the power check, so it gives diagnostics, never
  claims.

**Presets and blocks:**
- `core`: baseline exposure, mirror and cluster glances, prolonged off-road,
  look down vs closure, 2 s / 3 s closures, handheld and mounted phone, absence.
- `passenger` (a second consenting adult): passenger leaning in while the
  driver is turned away, passenger phone, driver closure and a slump toward the
  passenger (acted), passenger eyes shut.
- `displacement`: run it, nudge the mount without recalibrating, run it again,
  and compare.
- **Sunglasses and NoIR night:** the same presets with `--eyewear sunglasses_ir_pass|sunglasses_ir_block`
  and `--lighting night_ir`.
- **Model uncertainty:** the abstention counts, plus the device's
  `driver_identity` in telemetry.

**Rules:**
- The full safety block of `incabin_scenarios_v1.yaml` applies.
- No public-road testing. Sessions follow the protocol's consent rules, and
  gates G1–G6 in `docs/research/INCABIN_READINESS.md` are still open.
- `--fake` is a dry run, labelled SIMULATED.

## 6. What still blocks real-world accuracy statements

1. **No Pi measurement yet:** INT8 parity, Pi decode time, thermal, the
   eye_timing rate actually processed, and the buzzer heard by a person.
2. **No consented in-cabin data.** The protocol is DRAFT (G1–G6 open), and the
   minimums are ≥ 30 subjects per claimed cell (about 60 subjects for an
   evaluation-only val/test split).
3. **Thresholds re-derived on INT8 scores:** needs a labelled set run through
   the Pi. The Open Images val subset is on the fleet, and copying it to the
   Pi is an owner decision.
4. **NoIR at night:** the detector and FaceMesh were never measured on IR
   frames from this camera.
5. **C and D:** no compiled `.hef`, no DFC run, and no two-network runtime.
6. **Escalation and temporal thresholds:** synthetic evidence only (L1).
