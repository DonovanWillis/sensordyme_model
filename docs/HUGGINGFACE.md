# Hugging Face in the SensorDyme Stack — Evaluation & Decision

**Date:** 2026-08-29
**Scope:** Should Hugging Face become part of the SensorDyme AI development ecosystem?
**Verdict: PARTIALLY — see [Recommendation](#recommendation).**

> **SUPERSEDED IN PART — read this first.** This document is a snapshot of an
> evaluation made *before* the detector migration, and it is kept as the record of
> how that decision was reached. Its present-tense claims about what SensorDyme
> ships are no longer true:
>
> * **"You are shipping AGPL-3.0 in a commercial product right now" (§0.3, §4.5)
>   describes the pre-Phase-4 state.** The device now runs
>   `damoyolo_tinynasL20_T.hef` (Apache-2.0) on the Hailo-8L, and
>   `requirements-device.txt` ships neither `ultralytics` nor `torch`. Ultralytics
>   remains pinned in `requirements.txt` for developer machines and CI only.
> * **"Code never executed on hardware"** — it has since. See
>   `docs/PHASE5_DEMO_READINESS.md`.
> * **Ranked action #5 was carried out**, and it is what chose the shipping
>   detector.
>
> **Do not follow any step here that installs an Ultralytics-derived `.hef`
> (`yolov11n`, `yolov8n`, `yolov8s_pose`, `yolov8s_h8l`) onto a device** — those
> weights are AGPL-3.0 on hardware that is conveyed to a customer. The
> Hailo-8L/mAP research below is still accurate and still useful; the licence
> *status* of the product is not. `docs/LICENSING.md` governs what may ship.

Every metric below is either a live API response or a document fetched during this
evaluation; URLs are given inline. Where I could not verify something, it is marked
**UNVERIFIED**.

---

## 0. TL;DR

1. **Hugging Face is the wrong place to look for the models that will actually run on
   your NPU.** Hailo publishes *nothing* on HF (`?author=hailo` → 0 results). The
   Hailo Model Zoo is on GitHub + S3, and it is the single highest-value finding in
   this document — see §2.
2. **Two assumptions in the current audit are wrong and should be corrected.**
   - "MediaPipe has no Hailo build and stays on CPU permanently" — **false for the
     landmark nets.** Hailo ships a prebuilt `face_landmarks_lite.hef` that is
     Google's genuine FaceMesh network. I downloaded and inspected it (§2.2).
   - "Transformers/attention are largely NOT supported on Hailo-8L" — **too strong.**
     ViT, DETR, CLIP and SigLIP encoders all have prebuilt `_h8l` HEFs (§2.4). What
     genuinely does not work is *autoregressive decoding* (VLMs/LLMs).
3. **You are shipping AGPL-3.0 in a commercial product right now.** Ultralytics
   8.3.28 in `.venv` is AGPL-3.0, and YOLO11n weights inherit it. This is a
   product-blocking legal issue that is independent of Hugging Face, and it is the
   most urgent thing in this document (§4).
4. **HF is genuinely valuable for exactly three things here:** Apache-2.0 detector
   *architectures* to replace the AGPL ones, `optimum`/`transformers` as an
   ONNX export path, and `huggingface_hub` as a weights-distribution mechanism.
5. **HF is nearly useless for your dataset gap.** There is no URFD, Le2i, UP-Fall or
   CAUCAFall on the Hub, and no night/IR or infant-safety data at all (§5).

---

## 1. Method

- HF Hub REST API (`https://huggingface.co/api/models?search=…&sort=downloads`) for
  model discovery and license/file metadata.
- `huggingface_hub` 1.29.0 + `onnx` 1.22.0 installed into a throwaway venv at
  `…/scratchpad/hf/hfenv` (nothing was installed into the project `.venv`).
- Hailo Model Zoo docs fetched from
  `https://raw.githubusercontent.com/hailo-ai/hailo_model_zoo/master/docs/public_models/HAILO8L/`.
- `.hef` availability verified by HTTP HEAD against the Hailo S3 bucket (real
  `content-length`, not just a 200).

---

## 2. The Hailo-8L Model Zoo — the actionable finding

Hailo has **no Hugging Face presence**. Verified:

```
GET https://huggingface.co/api/models?author=hailo  →  count 0
```

The real source is the Hailo Model Zoo. The Hailo-8L (13 TOPS, your AI HAT+) docs
live at `docs/public_models/HAILO8L/*.rst` and cover **29 task categories**.
Compiled binaries are at:

```
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/<name>.hef
```

All FPS figures below are **Hailo's own measurements at batch size 1**, host
Intel i5-9400 / PCIe Gen3 x4. On a Pi 5 the PCIe link is narrower, so treat these
as an **upper bound**, not a promise — the NPU compute is the same but the host
transfer is not. Marked **UNVERIFIED** for Pi 5 specifically.

I confirmed these four downloads resolve with real payloads:

| File | HTTP | Size |
|---|---|---|
| `yolov11n.hef` | 200 | 11,712,608 B |
| `yolov8n.hef` | 200 | 12,576,181 B |
| `yolov8s_pose.hef` | 200 | 27,448,226 B |
| `scrfd_2.5g.hef` | 200 | 5,337,106 B |

**This is the difference between "the NPU is idle" and "the NPU works." There are
62 prebuilt object-detection HEFs alone. You do not need the Dataflow Compiler to
get started — you need a download.**

### 2.1 Object detection (62 HEFs; selected)

`mAP_hw` is the post-quantization number — the quantization cost is visible and real.

| Model | mAP float | mAP int8 | FPS b1 | Input | Params (M) | License |
|---|---|---|---|---|---|---|
| `yolov11n` | 39.0 | 37.5 | 157 | 640² | 2.6 | **AGPL-3.0** |
| `yolov11s` | 46.3 | 45.1 | 92.0 | 640² | 9.4 | **AGPL-3.0** |
| `yolov8n` | 37.0 | 36.4 | 202 | 640² | 3.2 | **AGPL-3.0** |
| `yolov8s` | 44.6 | 43.9 | 110 | 640² | 11.2 | **AGPL-3.0** |
| `yolov10n` | 38.5 | 36.6 | 150 | 640² | 2.3 | **AGPL-3.0** |
| `damoyolo_tinynasL20_T` | 42.8 | 42.2 | 124 | 640² | 11.35 | Apache-2.0 |
| `yolox_tiny` | 32.6 | 31.4 | 214 | 416² | 5.05 | Apache-2.0 |
| `yolox_s_leaky` | 38.1 | 37.3 | 109 | 640² | 8.96 | Apache-2.0 |
| `nanodet_repvgg` | 29.3 | 28.6 | **620** | 416² | 6.74 | Apache-2.0 |
| `yolov6n_0.2.1` | 35.2 | 34.2 | **560** | 640² | 4.33 | GPL-3.0 |
| `ssd_mobilenet_v2` | 24.2 | 23.0 | 131 | 300² | 4.46 | Apache-2.0 |
| `detr_resnet_v1_18_bn` | 33.9 | 31.5 | 23.4 | 800² | — | Apache-2.0 |

**Note `yolov11n` is already in the zoo.** Your shipped model has a prebuilt HEF at
157 FPS with only 1.5 mAP lost to quantization. The blocker to using it is not
technical — it is the AGPL license (§4).

**`damoyolo_tinynasL20_T` is the standout Apache-2.0 swap:** 42.2 mAP int8 beats
yolov11n's 37.5 by 4.7 points at 124 FPS. It is the best licensed-clean detector in
the zoo.

### 2.2 MediaPipe on Hailo — the assumption that needs retracting

The audit states MediaPipe "has no Hailo build and stays on CPU permanently." The
zoo contradicts this for the landmark models:

| Model | FPS b1 | Input | Params (M) | Source |
|---|---|---|---|---|
| `face_landmarks_lite` | **432** | 192×192×3 | 0.6 | google-ai-edge/mediapipe |
| `hand_landmark_lite` | **1652** | 224×224×3 | 1.01 | google/mediapipe |
| `tddfa_mobilenet_v1` | **6914** | 120×120×3 | 3.26 | cleardusk/3DDFA_V2 |

**I downloaded and inspected `face_landmarks_lite`** (1,122,654 B zip → a single
1,242,398 B `.tflite`). Embedded provenance strings:

```
facemesh_facecontours_faceflag_blaze_2018_10_10_v0
--output=conv2d_21 --output=conv2d_31
"Detects human face landmarks." / "Face landmarks detection"
```

This is Google's actual FaceMesh network — the same one MediaPipe runs — as a plain
BlazeFace-family CNN with two heads: `conv2d_21` (landmarks) and `conv2d_31` (the
"faceflag" presence score). No unsupported ops; that is why it compiles.

Three honest caveats:

1. **This is the landmark net only, not the MediaPipe pipeline.** MediaPipe is
   detector → landmark. You would pair it with a Hailo face detector (§2.3) and
   **reimplement the post-processing yourself** — landmark decoding, EAR, the
   solution-level smoothing and tracking. That is real work, not a drop-in.
2. `tddfa_mobilenet_v1` (3DDFA_V2) gives dense 3D face geometry at NME 3.68 float /
   4.04 hardware — a **genuine head-pose replacement** and a better-posed one than
   deriving pose from a 2D mesh.
3. The `faceflag` head is a properly trained face-presence output. It is the honest
   version of what the withdrawn openpilot `faceProb` was supposed to provide.
   Given the measured 38.4% face availability on domain-shifted in-car imagery,
   this is worth testing directly — but **it must be validated on driver-facing
   imagery**, which you do not currently have.

### 2.3 Face detection (5 HEFs)

`scrfd_500m`, `scrfd_2.5g`, `scrfd_10g`, `retinaface_mobilenet_v1`,
`lightface_slim`. SCRFD is the strong modern choice and `scrfd_2.5g.hef` is 5.3 MB.
Pairing `scrfd_2.5g` → `face_landmarks_lite` reconstructs the FaceMesh pipeline
entirely on the NPU.

### 2.4 Pose, video, and the transformer question

| Model | Metric | FPS b1 | Input | License |
|---|---|---|---|---|
| `yolov8s_pose` | AP 59.2 → **55.7** int8 | 107 | 640² | **AGPL-3.0** |
| `yolov8m_pose` | AP 64.3 → 62.2 int8 | 48.5 | 640² | **AGPL-3.0** |
| `centerpose_regnetx_800mf` | AP 44.1 → 43.0 | 85.0 | 512² | Apache-2.0 |
| `vit_pose_small` | single-person | — | — | Apache-2.0 |
| `mspn_regnetx_800mf` | single-person | — | — | Apache-2.0 |
| `r3d_18` | Kinetics top-1 48.9 → **49.3** | 28.6 | **112×112×48** | BSD (torchvision) |

Note `yolov8s_pose` loses **3.5 AP** to int8 — the largest quantization penalty in
this table. Pose keypoints degrade more than boxes under quantization; budget for it.

**`r3d_18` is the most interesting model in the zoo for SensorDyme.** Input
`112×112×48` = 16 RGB frames — a genuine **spatio-temporal** classifier running on
the NPU at 28.6 FPS. This is the one architecture here that could address the
documented "cannot detect falls along the camera axis" limitation, because a fall
toward the camera is unmistakable in *motion* even when torso angle in the camera
plane is uninformative. Kinetics-400 top-1 of 48.9 is not a fall detector out of the
box — it would need fine-tuning on fall data you do not have (§5). But the
deployment path exists and is prebuilt.

**On transformers:** the brief's claim that attention is largely unsupported is too
strong. `vit_pose_small`, `detr_resnet_v1_18_bn`, and full CLIP/SigLIP encoder
families all have Hailo-8L HEFs:

```
clip_vit_b_16_image_encoder.hef, siglip_b_16_image_encoder.hef,
tinyclip_vit_8m_16_text_3m_yfcc15m_image_encoder.hef, … (10 image + 10 text)
```

The accurate statement is: **encoder-only transformers compile; autoregressive
decoding does not.** There is no LLM or VLM decoder anywhere in the Hailo-8L zoo.
That distinction determines the whole VLM story in §3.3.

### 2.5 Other categories worth knowing about

- `zero_dce.hef` / `zero_dce_pp.hef` — **low-light enhancement**. Directly relevant
  to the total absence of night/IR capability. Cheap to test.
- `face_attr_resnet_v1_18.hef`, `person_attr_resnet_v1_18.hef` — attribute heads.
- Also present: depth estimation, stereo depth, person re-ID, instance segmentation,
  semantic segmentation, super-resolution, text detection/recognition, ASR.

---

## 3. Hugging Face models — what is actually there

### 3.1 The task-specific community models are not usable

I searched drowsiness, driver distraction, fall detection, phone detection and head
pose. The results are hobby repos: the top drowsiness model
(`chbh7051/driver-drowsiness-detection`, Apache-2.0) has **139 downloads**; most have
0–20 and no license, no model card, no eval protocol, and no statement of what data
they were trained or validated on.

The most-downloaded fall model, `melihuzunoglu/human-fall-detection` (809 downloads),
is **AGPL-3.0** — unusable in the product for the same reason as YOLO11n.

**Do not ship any of these.** They would import unknown training data and unknown
bias into a *safety* product, with no way to audit either. Their only legitimate use
is as a weak baseline to compare against.

`litert-community/6DRepNet-HeadPose-LiteRT` (60 downloads) is mildly interesting as
a head-pose option, but `tddfa_mobilenet_v1` is already compiled for your NPU and is
the better bet.

### 3.2 Where HF genuinely wins: Apache-2.0 detector architectures

This is the real value. All verified Apache-2.0 via the API:

| Repo | Downloads | License | Notes |
|---|---|---|---|
| `PekingU/rtdetr_r18vd` | 46,934 | Apache-2.0 | RT-DETR, NMS-free |
| `PekingU/rtdetr_v2_r18vd` | 378,878 | Apache-2.0 | v2, most-used |
| `ustc-community/dfine-nano-coco` | 7,644 | Apache-2.0 | D-FINE, current SOTA-ish real-time |
| `ustc-community/dfine-small-coco` | 5,238 | Apache-2.0 | |
| `Roboflow/rf-detr-nano` | 980 | Apache-2.0 | |
| `Roboflow/rf-detr-base` | 36,402 | Apache-2.0 | |
| `jameslahm/yolov10n` | 2,076 | **AGPL-3.0** | avoid |
| `onnx-community/yolov10n` | 839 | **AGPL-3.0** | avoid |

None of these ship `.onnx` in-repo except the `onnx-community` mirrors (which are the
AGPL ones). Export is via `optimum` — see §6.

**Important caveat on the DETR family:** only `detr_resnet_v1_18_bn` is in the Hailo
zoo, at 23.4 FPS on an 800² input, and it loses 2.4 mAP to quantization. RT-DETR,
D-FINE and RF-DETR are **not** in the Hailo zoo. Compiling them yourself is
**UNVERIFIED** and I would not assume it works — DETR-family decoders use ops that
are far more likely to fall outside the supported set than a YOLO backbone. Treat
these as **CPU/ONNX-path candidates or Hailo research**, not as a safe NPU plan.

### 3.3 Small VLMs for the reasoning layer

| Model | License | ONNX in repo |
|---|---|---|
| `HuggingFaceTB/SmolVLM-256M-Instruct` | Apache-2.0 | 24 files |
| `HuggingFaceTB/SmolVLM2-256M-Video-Instruct` | Apache-2.0 | 24 files |
| `vikhyatk/moondream2` | Apache-2.0 | 0 |
| `microsoft/Florence-2-base` | MIT | 0 |
| `google/paligemma-3b-mix-224` | **gemma** (custom, restrictive) | 0 |

SmolVLM-256M is the only one with a maintained ONNX export. Real file sizes from the
API:

```
vision_encoder_q4f16.onnx           55.0 MB
decoder_model_merged_q4f16.onnx     77.0 MB
embed_tokens_q4f16.onnx             56.8 MB   →  ~189 MB total
```

That fits in Pi 5 RAM. But: **there is no Hailo HEF for any VLM**, so this is
100% CPU, competing with MediaPipe for the same cores, at seconds per inference.
It is **not a per-frame path** and never will be on this hardware.

Note `SmolVLM2-256M-Video-Instruct` accepts video — conceptually attractive for fall
and baby-monitoring context. Same CPU-only constraint applies.

**Recommendation for the reasoning layer: leave `jupiter/` on Ollama + qwen2.5.**
It already works, it is already local, and a 256M VLM would be a downgrade in
reasoning quality. The only thing SmolVLM offers that qwen2.5-instruct does not is
*vision* — so the narrow case worth testing is a low-rate escalation path ("something
anomalous fired; describe the scene once") at ≤0.2 Hz, not a monitoring loop.
Avoid PaliGemma: the Gemma license carries use restrictions that need legal review.

---

## 4. The licensing problem (urgent, and bigger than HF)

Verified locally:

```
/Users/donov/sensordyne-models/.venv/…/ultralytics-8.3.28.dist-info/METADATA
  License: AGPL-3.0
```

and `github.com/ultralytics/ultralytics/main/LICENSE` → "GNU AFFERO GENERAL PUBLIC
LICENSE Version 3".

**SensorDyme currently ships YOLO11n via Ultralytics in a commercial product with a
FastAPI backend and an iOS app.** AGPL-3.0's network clause is the specific hazard:
serving inference over a network is a trigger. Ultralytics' own README explicitly
directs commercial embedders to buy an Enterprise License.

This affects, today: YOLO11n (person + phone detection). It would also affect
`yolov8s_pose`, `yolov8m_pose`, `yolov10n`, and `melihuzunoglu/human-fall-detection`
if adopted.

**Two paths, and the team must pick one deliberately:**
1. Buy the Ultralytics Enterprise License. Keeps `yolov11n.hef` and `yolov8s_pose.hef`
   — the best accuracy-per-FPS in the zoo.
2. Migrate to Apache-2.0: `damoyolo_tinynasL20_T` for detection (42.2 mAP int8,
   *better* than yolov11n's 37.5), `centerpose_regnetx_800mf` for pose (AP 43.0,
   worse than yolov8s_pose's 55.7 — this one genuinely costs you accuracy).

I am not a lawyer and this is not legal advice; it is a flag that the license text
says what it says. **This needs a decision before launch, not after.**

---

## 5. Datasets — HF does not solve your data gap

I searched falls, URFD, drowsiness, distracted driver, activity recognition,
infrared/thermal person, night driving, infant and baby monitoring.

**Falls — the highest-value gap — are essentially not on HF.**
`URFD` returns **zero results**. No Le2i, no UP-Fall, no CAUCAFall. What exists:

| Dataset | Downloads | Assessment |
|---|---|---|
| `bear7011/enhanced-fall-dataset` | 1,795 | no license listed; unknown provenance |
| `pat2echo/fall-detection-posture-classification` | 789 | no license listed |
| `MahedixHasan/pose-guided-fall-detection-icta2026` | 678 | <1K samples |
| `DeZan/fall-detection` | 398 | MIT, but <1K samples |

None is a credible substitute for a canonical benchmark. **The canonical fall
datasets are on university sites, not the Hub** — UR Fall Detection (AGH Kraków),
Le2i (Univ. Bourgogne), UP-Fall (UP México), CAUCAFall (Univ. del Cauca). Getting
them means going to those sources and honoring their per-dataset academic terms.
Several are academic-use-only, which is its own product problem — **UNVERIFIED**
which are commercially usable; that needs checking per dataset before any of it
trains a shipped model.

**Night/IR: nothing.** `infrared+person`, `thermal+person+detection` and
`night+driving` all returned zero HF datasets. Your night/IR gap cannot be closed
from the Hub. (The relevant public sets — FLIR ADAS, KAIST Multispectral, LLVIP —
live elsewhere and have their own terms.)

**Infant/baby: nothing usable, and this is appropriate.** `baby+monitoring` returned
zero. The infant results are audio cry-detection sets
(`AxonData/infant-cry-detection-dataset`, `DynamicSuperb/Infant_Crying_Classification_Dataset`)
and unrelated demographic tables. There is no infant *vision* safety dataset on the
Hub, and there is unlikely ever to be a good one: filming infants creates consent and
child-privacy obligations that a public scrape cannot satisfy. **Plan for baby
monitoring to be validated on consented, self-collected data.** Do not go looking for
a shortcut here.

**Driver data — marginal, you already have better.**
`gymprathap/Driver-Distracted-Dataset` (139 downloads, "cc" — ambiguous, likely a
State Farm mirror) and several small drowsiness sets, best being
`akahana/Driver-Drowsiness-Dataset` (141 downloads, 10K–100K, parquet) and
`Simuletic/Driver_Drowsiness_Monitoring_Detection_Dataset` (CC-BY-4.0). Your existing
44,848-image subject-disjoint State Farm split is stronger than any of these.

The one genuine dataset-side win: **HF `datasets` + parquet is a good *format* for
your own data**, even when the data does not come from the Hub. Private repos would
give you versioned, hash-addressed dataset storage — worth having regardless.

---

## 6. Tooling: what to install, what to refuse

The decisive constraint: **torch is not installed, and it is a ~2 GB commitment.**

| Package | Verdict | Reasoning |
|---|---|---|
| `huggingface_hub` | **Install now** | ~5 MB, no torch. Metadata queries, weight download, private dataset/model hosting. Tested working (§7). |
| `datasets` | **Install now-ish** | Parquet + versioning for *your own* data. No torch needed for basic use. |
| `transformers` | **Dev machine only** | Needed to load RT-DETR/D-FINE for export. Pulls torch. Never on the Pi. |
| `optimum[onnxruntime]` | **Dev machine only** | The actual reason to touch HF tooling — ONNX export. Pulls torch. |
| `timm` | Only if a backbone is needed | Pulls torch. |
| `accelerate`, `peft` | **No** | Multi-GPU orchestration and LLM LoRA. You are not training LLMs; `peft` is irrelevant to CNN fine-tuning. Pure bloat here. |
| `evaluate` | **No** | Thin metric wrappers. Your existing eval harness measured phone recall at 5 confidence thresholds — it is already better suited than `evaluate` for a detection product. |

**Hard rule: torch never goes on the Pi 5.** Export happens on a dev machine; the Pi
gets `.hef` files and, at most, `onnxruntime` (already installed, 1.27).

Keep the project `.venv` clean. If you adopt HF export tooling, put
`transformers` + `optimum` + `torch` in a **separate export environment** so the
runtime venv stays small and the Pi image stays deployable.

---

## 7. What I actually tested

**Test 1 — `huggingface_hub` API against real repos.** Working:

```
== HuggingFaceTB/SmolVLM-256M-Instruct license: apache-2.0
   onnx/vision_encoder_q4f16.onnx    55.0 MB
   onnx/decoder_model_merged_q4f16.onnx  77.0 MB
== onnx-community/yolov10n license: agpl-3.0
   onnx/model.onnx        9.4 MB
   onnx/model_int8.onnx   2.7 MB
```

Programmatic license + file-size interrogation before downloading weights works, and
is the right way to screen candidates. Note it caught the AGPL on `yolov10n`.

**Test 2 — downloaded and inspected `face_landmarks_lite`.** 1.1 MB zip → a single
1.24 MB `.tflite`, confirmed to be
`facemesh_facecontours_faceflag_blaze_2018_10_10_v0` with outputs `conv2d_21`
(landmarks) and `conv2d_31` (faceflag). This is the evidence behind §2.2.

**Test 3 — HEF availability.** HEAD requests confirmed real `content-length` on
`yolov11n.hef`, `yolov8n.hef`, `yolov8s_pose.hef`, `scrfd_2.5g.hef`.

**Not tested:** nothing was run on Hailo hardware, and no model was benchmarked on a
Pi 5. All FPS numbers are Hailo's, on an x86 host. Every Pi 5 performance claim in
this document is **UNVERIFIED**.

---

## 8. The HF → ONNX → Hailo → Pi pipeline

**For zoo models, the pipeline does not exist — and that is the good news.**
Download `.hef`, run it. No export, no compiler, no calibration. This is why §2 is
the recommendation and §3.2 is not.

**For a custom or HF-sourced model, the real path is:**

```
HF (torch) ─optimum→ ONNX ─DFC parse→ HAR ─optimize+calibrate(int8)→ ─compile→ .hef → Pi
```

Where it breaks, in order of likelihood:

1. **Unsupported ops at DFC parse.** The zoo's contents *are* the compatibility list.
   A YOLO/RepVGG/MobileNet backbone is safe. A DETR-family decoder is a coin flip
   and **UNVERIFIED**.
2. **The compiler does not run on the Pi.** The Hailo Dataflow Compiler is x86-64
   Linux. Your dev machine here is **darwin/arm64** — so compilation needs an x86
   Linux box, a VM, or CI. This is a real workflow cost that must be planned for, and
   it is the step most likely to stall the project.
3. **Quantization accuracy loss**, and you need a calibration set. The zoo shows the
   cost is real and uneven: 0.6 mAP for `yolov8n`, 1.9 for `yolov10n`, **3.5 AP for
   `yolov8s_pose`**. Your calibration data must match the deployment domain —
   calibrating a driver-facing model on passenger-side imagery would repeat exactly
   the domain-shift failure that killed the openpilot integration.
4. **Post-processing is yours.** Some HEFs include NMS (`*_postprocess`,
   `*_nms_core`), most do not. Landmark decoding for FaceMesh is entirely on you.

**Verdict: practical for zoo models today. A meaningful project for anything else,
gated on x86 Linux access.**

---

## 9. Recommendation

### **PARTIALLY — adopt Hugging Face as a narrow tool, not as the model source.**

**YES to:**
- `huggingface_hub` as a dependency — screening, and private hosting of your own
  weights and datasets.
- HF as the source of **Apache-2.0 detector architectures** (D-FINE, RT-DETR) to
  escape the AGPL exposure.
- `optimum[onnxruntime]` in a **separate export environment** on a dev machine.
- HF `datasets`/parquet as the storage format for data you collect yourself.

**NO to:**
- Every task-specific community model found (drowsiness, fall, phone). Unlicensed or
  AGPL, unaudited data, ≤809 downloads. Not acceptable in a safety product.
- `peft`, `accelerate`, `evaluate` — bloat for this workload.
- Torch anywhere near the Pi or the project `.venv`.
- PaliGemma (restrictive custom license).
- Treating HF as the fix for the fall/night/infant data gaps. It is not.

**The most important thing in this document is not Hugging Face — it is that the
Hailo Model Zoo has 62 prebuilt object-detection HEFs plus FaceMesh, hand landmarks,
3DDFA head pose, SCRFD, pose, and a spatio-temporal video classifier, all
downloadable today, and your NPU is idle.**

### Ranked next actions

| # | Action | Validating experiment | Why |
|---|---|---|---|
| 1 | **Resolve the Ultralytics AGPL question** | Legal/business decision, not an experiment | Blocks launch. Everything else is moot if the detector must be replaced. |
| 2 | ~~Download `yolov11n.hef` + run one frame on the Hailo~~ **DONE, but with `damoyolo_tinynasL20_T.hef` instead — do not do this as written** | Any output at all from the NPU | Converts "code never executed on hardware" into a working path. **`yolov11n.hef` is AGPL-3.0-derived and NMS-on-chip (0.200 floor, 25.4% phone recall); it is the wrong first win on both counts.** |
| 3 | `scrfd_2.5g.hef` → `face_landmarks_lite.hef` | Face-detection availability on **driver-facing** frames vs MediaPipe's measured 38.4% | Directly attacks the worst measured number in the audit, and frees the CPU. |
| 4 | `tddfa_mobilenet_v1.hef` for head pose | Compare yaw/pitch against current FaceMesh-derived pose | 6914 FPS, better-posed geometry, NME 4.04 after quantization. |
| 5 | `damoyolo_tinynasL20_T.hef` vs YOLO11n on your State Farm split | **Phone recall at conf 0.35** — the metric that is currently 12–17% | Apache-2.0 *and* +4.7 mAP int8. Could fix the license and the phone problem at once. |
| 6 | `yolov8s_pose.hef` (or `centerpose` if AGPL is refused) | Fall/flail detection vs current MediaPipe Pose | Moves pose to NPU; budget for the 3.5 AP quantization loss. |
| 7 | Acquire URFD + UP-Fall **from source universities**, check commercial terms | — | You have zero fall data. This gates any real fall work. |
| 8 | Fine-tune `r3d_18` on fall data once #7 lands | Can it catch a fall **along the camera axis**? | The only candidate that addresses the documented geometric blind spot. |
| 9 | `zero_dce.hef` on low-light frames | Does downstream detection improve at night? | Cheapest probe at the night/IR gap. |
| 10 | SmolVLM-256M ONNX on Pi 5 CPU | Wall-clock seconds/inference | Only to *bound* the escalation-path idea. Expect it to be slow; keep `jupiter/` on Ollama. |

### Install now vs later

**Now:** `huggingface_hub` (project venv, ~5 MB, no torch).
**Now, separately:** nothing else — go download HEFs first.
**When you commit to custom models:** `datasets` (project), and a *separate* export
env with `transformers` + `optimum[onnxruntime]` + `torch`, on an x86 Linux host that
can also run the Hailo Dataflow Compiler.
**Never:** torch on the Pi.

### How HF fits the architecture

```
Pi 5 runtime      : .hef on Hailo-8L  ←── the zoo, not HF
                    onnxruntime on CPU for anything unsupported
                    Ollama + qwen2.5 for jupiter/ reasoning  ← unchanged
Dev/export host   : HF transformers + optimum → ONNX → Hailo DFC → .hef   (x86 Linux)
Data + artifacts  : huggingface_hub / datasets — private repos, your own data
```

HF sits **behind** the runtime, in the build and data layers. It should never appear
in a request path on the device.

---

## 10. Open questions / UNVERIFIED

- Hailo's FPS figures are x86-host; **real Pi 5 throughput is unmeasured.**
- Whether RT-DETR / D-FINE / RF-DETR compile for Hailo-8L at all. I found no evidence
  either way; only `detr_resnet_v1_18_bn` is in the zoo.
- Commercial usability of URFD / Le2i / UP-Fall / CAUCAFall — must be checked per
  dataset before any of it trains a shipped model.
- Whether `face_landmarks_lite`'s `conv2d_31` faceflag holds up on driver-facing
  imagery. Untested — and the openpilot failure is a warning against assuming.
- Zoo version pinning: all URLs above are `v2.19.0` / DFC v2.19.0. Your installed DFC
  and HailoRT must match the HEF version, and I could not verify what the Pi has.
