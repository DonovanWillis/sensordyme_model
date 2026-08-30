# Detector licensing — provenance and commercial-release position

> **This is engineering documentation, not legal advice.** Nothing here is a
> determination of what is or is not legal. What follows is (a) what the license
> texts at the cited URLs actually say, (b) what the SensorDyme architecture
> actually does, and (c) which combinations of those two a lawyer needs to look
> at. Every license claim cites the URL it was read from. **Have counsel review
> before any commercial release.**
>
> Verified 2026-08-29. Licenses change (YOLOv5's did) — re-verify before shipping.

---

## Bottom line

**DAMO-YOLO is the only measured Hailo-8L candidate whose upstream terms are
permissive, and the Apache-2.0 claim survives inspection** — `tinyvision/DAMO-YOLO`
is Apache-2.0 with an Apache `NOTICE` file, its COCO weights carry no separate or
additional terms that could be found, and the one non-commercial file in the repo
(`tools/onnx_inference.py`, CC BY-NC 4.0) is a standalone demo script that is not
part of the model, the weights, or anything a `.hef` is compiled from — so it does
not contaminate `damoyolo_tinynasL20_T.hef` provided SensorDyme never vendors that
file. The exposure that actually blocks release is the current stack, not the
future one: `ultralytics==8.3.28` is AGPL-3.0 (confirmed in the installed package
metadata on this machine), it is imported at `core/detectors.py:50`, and the device
both **conveys** that software to customers on an SD card and **serves its
inference output over a network** via the MJPEG `/stream` endpoint at
`core/dashboard.py:191`. Ultralytics' own licensing page names
"Embedded deployments in hardware, edge devices, robotics, cameras, or appliances"
as requiring a paid Enterprise License. The best-measured model, `yolov5s`
(90.0% phone recall), does **not** escape this — Hailo compiled it from
`ultralytics/yolov5` at tag `v2.0`, which was GPL-3.0, still strong copyleft.
The clean path is: DAMO-YOLO HEF on the NPU **and** `ultralytics` removed from
`requirements.txt` — both, not either.

> **STATUS, 2026-08-30 — both halves of that clean path have been taken.** The
> paragraph above is preserved as written because it is the analysis that drove
> the change, and because the history explains why the design looks the way it
> does. What is true now:
>
> * The device runs `damoyolo_tinynasL20_T.hef` (Apache-2.0 upstream), Hailo Model
>   Zoo v2.15.0 hailo8l, sha256 `e69136d3…`, 18,554,430 bytes — pinned and
>   checksum-verified by `setup_pi.sh`, recorded in `config.yaml` as
>   `accelerator.yolo_hef_sha256`, and enforced by
>   `tests/test_phase5_demo_readiness.py`.
> * `ultralytics` and `torch` are absent from `requirements-device.txt`, from
>   `setup_pi.sh`, and from `make deploy`. They were also found **installed on the
>   physical device** in Phase 5 (a survivor of an earlier install) and removed.
> * `make deploy` excludes `*.pt` **and** `*.hef`, so no model can reach a device
>   from a working tree.
>
> **This does not close the legal questions.** §7 open items 1–8 all remain open,
> the "have counsel review before commercial release" instruction is unchanged,
> and the Apache-2.0 §4(d) attribution obligation §8 item 5 identifies is
> **still unmet** — see §10.3. See §10 for the full close-out.

---

## 1. What SensorDyme actually does (the facts the licenses attach to)

These are the architectural facts that decide which license clauses are live.
They are verified in this repo, not assumed.

> **Line citations in this section were re-verified 2026-08-30** and several had
> drifted. Where a fact itself has changed since Phase 4, the row says so; §§1–8
> otherwise stand as written, and §10 records what closed.

| Fact | Evidence in this repo |
|---|---|
| `ultralytics==8.3.28` was a shipped runtime dependency — **since Phase 4 it is dev/CI only** | `requirements.txt:35` (the file `setup_mac.sh` and CI install). The device installs `requirements-device.txt`, which pins neither it nor `torch`. |
| It is imported to construct the detector — **lazily, and only when there is no accelerator** | `core/detectors.py:59` — `from ultralytics import YOLO`, inside `get_yolo_model`, wrapped in `try`; the `ImportError` is re-raised as a `RuntimeError` naming both fixes |
| Default weights **were** Ultralytics YOLO11n | `config.yaml:521` — `model_weights: yolo11n.pt`, now annotated "CPU/dev checkpoint only". **The shipped detector is `accelerator.yolo_hef` (`config.yaml:55`) = `models/damoyolo_tinynasL20_T.hef`**; when that is set, `model_weights` is only a cache key |
| Annotated inference output is served over HTTP | `core/dashboard.py:191` (`/stream` route), `core/dashboard.py:194` — `multipart/x-mixed-replace; boundary=frame` |
| The device pushes events to a vendor-operated backend | `cloud/`, FastAPI backend |
| The product is a physical device sold to customers | Pi 5 + AI HAT+, `setup_pi.sh`, `sensordyme.service` |
| Weights are **not** committed to git | `.gitignore:45-46` — `*.pt`, `*.hef`; `yolo11n.pt` is untracked |

Two distinct triggers follow from this, and they are often conflated:

1. **Conveying** — SensorDyme ships a device with the AGPL software installed on
   it. This is distribution, and it is triggered by GPL-3.0 and AGPL-3.0 alike,
   whether or not anything was modified.
2. **Remote network interaction (AGPL §13)** — the device serves inference output
   to a browser/app over the LAN. §13 is AGPL-specific and GPL-3.0 does not have it.

---

## 2. AGPL-3.0 §13, precisely

Quoted verbatim from the AGPL-3.0 text shipped inside the installed package
(`.venv/lib/python3.12/site-packages/ultralytics-8.3.28.dist-info/LICENSE`):

> **13. Remote Network Interaction; Use with the GNU General Public License.**
>
> Notwithstanding any other provision of this License, **if you modify the
> Program**, your modified version must prominently offer all users interacting
> with it remotely through a computer network (if your version supports such
> interaction) an opportunity to receive the Corresponding Source of your version
> by providing access to the Corresponding Source from a network server at no
> charge, through some standard or customary means of facilitating copying of
> software.

Read carefully, **§13 is conditioned on modifying the Program.** SensorDyme does
not patch ultralytics' source; it imports it. So the sharper question for
SensorDyme is not §13 in isolation — it is:

- **Is `sensordyne-models` a "work based on the Program"?** If importing
  `ultralytics` into `core/detectors.py` creates a combined work, then the whole
  application is covered, the combined work *is* the modified Program for §13
  purposes, and §13 fires on `/stream`. The FSF's stated position is that
  importing a module into your program creates a combined work; this is disputed
  by others and has thin case law. **This is the crux and it is a legal question,
  not an engineering one. NEEDS LEGAL REVIEW.**
- **Independently of §13, §4/§6 fire on conveying.** Shipping an SD card image
  containing AGPL code to a paying customer is conveying. If the application is
  a covered work, Corresponding Source for it must be offered to that customer.

**Ultralytics' own stated position** — https://www.ultralytics.com/license,
quoted from that page:

> "If you use Ultralytics YOLO code, models, architectures, training pipelines,
> or trained/fine-tuned models, you must either: Open-source your entire project
> under AGPL-3.0, or Obtain an Ultralytics Enterprise License."

and the Enterprise License is described there as required for:

> "Internal business tools or private company applications," "Any commercial
> product or service," "Proprietary / closed-source software," and "Embedded
> deployments in hardware, edge devices, robotics, cameras, or appliances."

**Pricing: not published.** The page states "Our licensing is tailored to each
organization's size and specific use case" and routes to a quote request form.
Budget is therefore **UNVERIFIED** — settle it by requesting a quote.

Note that this stated position is *broader* than what §13 alone compels — it
extends to weights and to non-networked commercial use. Whether it is broader
than what the license text can support is exactly the kind of question that goes
to counsel, not to this document. As a practical matter, the copyright holder's
stated interpretation is what you would be litigating against.

**Relicense history (both dated from the GitHub commit API):**

| Repo | Old | New | Date | Commit |
|---|---|---|---|---|
| `ultralytics/ultralytics` (YOLOv8/11) | GPL-3.0 | AGPL-3.0 | **2023-04-14** | "Update LICENSE to AGPL-3.0 (#2031)" — `2c6fc0a` |
| `ultralytics/yolov5` | GPL-3.0 | AGPL-3.0 | **2023-04-14** | "Update LICENSE to AGPL-3.0 (#11359)" |

Source: `https://api.github.com/repos/ultralytics/ultralytics/commits?path=LICENSE`
and `https://api.github.com/repos/ultralytics/yolov5/commits?path=LICENSE`.

---

## 3. The crux: what license does a prebuilt Hailo `.hef` carry?

This is the question the whole "just use DAMO-YOLO" plan rests on, so it gets
resolved from sources rather than asserted.

**What Hailo licenses.** `hailo_model_zoo`'s LICENSE is the **MIT License**,
"Copyright (c) Hailo, Inc. and its affiliates" —
https://raw.githubusercontent.com/hailo-ai/hailo_model_zoo/master/LICENSE
(identical on the Hailo-8/8L branch `update-to-version-v2.19.0`). The README
states: *"The Hailo Model Zoo is released under the MIT license."* —
https://raw.githubusercontent.com/hailo-ai/hailo_model_zoo/master/README.rst

**What Hailo does *not* do.** The README says nothing about the license of the
pretrained models or of the compiled HEF binaries. There is no per-model license
field in the model config
(`hailo_model_zoo/cfg/networks/damoyolo_tinynasL20_T.yaml` has `network_name`,
`input_shape`, `full_precision_result`, a weights `url`, and supported
architectures — **no `license` key**). There is no third-party license manifest
in the repo (top level is only `CITATION.cff`, `LICENSE`, `README.rst`,
`SUPPORT.rst`, `setup.py`, plus `docs/`, `hailo_model_zoo/`, `hailo_models/`,
`training/`). The S3 objects are served as bare downloads with no click-through
terms.

**What Hailo *does* do, and why it settles the question.** Every model row in
`docs/public_models/HAILO8L/HAILO8L_object_detection.rst` carries an explicit
**Source** link to the upstream repository it was built from. Hailo is telling
you where the model came from. Verbatim source links:

| Hailo model | Source link Hailo publishes |
|---|---|
| `damoyolo_tinynasL20_T` | `https://github.com/tinyvision/DAMO-YOLO` |
| `yolov5s` | `https://github.com/ultralytics/yolov5/releases/tag/v2.0` |
| `yolov11n` | `https://github.com/ultralytics/ultralytics` |
| `yolox_tiny` | `https://github.com/Megvii-BaseDetection/YOLOX` |
| `nanodet_repvgg` | `https://github.com/RangiLyu/nanodet` |
| `yolov6n` | `https://github.com/meituan/YOLOv6/releases/tag/0.1.0` |
| `ssd_mobilenet_v2` | `https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md` |

Source: https://raw.githubusercontent.com/hailo-ai/hailo_model_zoo/master/docs/public_models/HAILO8L/HAILO8L_object_detection.rst

**Answer: both, in different layers — and Hailo's MIT does not launder anything.**
Hailo's MIT license covers *Hailo's own contribution*: the zoo tooling, the
compilation recipe, the packaging. It cannot and does not relicense Alibaba's,
Megvii's, or Ultralytics' weights — Hailo has no authority to grant rights in
someone else's copyrighted work. A compiled `.hef` is a quantized, recompiled
representation of upstream weights and architecture; the most defensible reading
is that **upstream terms travel with the model content, and MIT covers the
wrapper**. This reading is consistent with Hailo's own behavior of naming the
upstream source per model.

The practical consequence is the one that matters:

- `damoyolo_tinynasL20_T.hef` → upstream is Apache-2.0 → **permissive, survives.**
- `yolov11n.hef` → upstream is `ultralytics/ultralytics` → **AGPL-3.0. The HEF
  does not escape it.** Running YOLO11 on the NPU instead of the CPU changes
  nothing about the weights' license.
- `yolov5s.hef` → upstream is the `v2.0` tag → **GPL-3.0** (see §4).

Anyone claiming "it's from Hailo so it's MIT" is reading the wrapper's license
and ignoring the contents. Do not ship on that reading.

**UNVERIFIED:** whether Hailo has a separate written position on HEF licensing
in a customer agreement, Developer Zone EULA, or the Dataflow Compiler EULA (the
DFC is behind an account wall and was not readable here). **What would settle
it:** ask Hailo support/FAE directly, in writing, "what license governs the
prebuilt HEFs on your public S3 bucket, and does it differ per upstream model?"
Get the answer in email. This is a 10-minute email that removes the single
largest uncertainty in this document.

---

## 4. Per-model license findings (primary sources)

All read from the upstream repository's own LICENSE file on 2026-08-29.

### DAMO-YOLO — Apache-2.0 ✅ (leading candidate, examined in depth)

- **LICENSE:** "Apache License Version 2.0, January 2004", "Copyright 2022-2023
  Alibaba Group Holding Limited."
  — https://raw.githubusercontent.com/tinyvision/DAMO-YOLO/master/LICENSE
- **GitHub license metadata:** `spdx_id: Apache-2.0` —
  https://api.github.com/repos/tinyvision/DAMO-YOLO

**Weights, checked separately as instructed.** No separate or additional terms
for the pretrained COCO weights could be found. The README distributes them from
Alibaba OSS (`idstcv.oss-cn-zhangjiakou.aliyuncs.com`) and Google Drive with no
attached conditions, no click-through, and no model-card-style restriction. There
is no `MODEL_LICENSE` file. Hailo additionally rehosts the pretrained model at
`https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/damoyolo_tinynasL20_T/pretrained/2022-12-19/damoyolo_tinynasL20_T.zip`
— which is permitted redistribution under Apache-2.0 §4, and is itself consistent
with the weights being Apache-2.0.

**The `NOTICE` file — the one real finding, and it is survivable.** DAMO-YOLO
ships an Apache `NOTICE`
(https://raw.githubusercontent.com/tinyvision/DAMO-YOLO/master/NOTICE). It
attributes portions of `damo/` to Facebook (MIT) and OpenMMLab (Apache-2.0) —
both permissive, both fine — **and it lists exactly one file under the Creative
Commons Attribution-NonCommercial 4.0 International Public License:**

```
tools/onnx_inference.py
```

**Why this does not sink the Apache-2.0 claim.** `tools/onnx_inference.py` is a
standalone ONNX demo script. It is not imported by the model definition, is not
part of training, is not part of the exported ONNX/weights, and is not an input
to Hailo's compilation. A `.hef` compiled from the released weights contains
nothing from it. **Action required, and it is trivial: never copy, vendor, or
adapt `tools/onnx_inference.py` into SensorDyme.** SensorDyme's inference path
already goes through `core/accel.py`'s own HailoRT code, so this is a
"don't start" rather than a "stop doing".

**Apache-2.0 §4(d) obligation:** if SensorDyme redistributes DAMO-YOLO-derived
material (a `.hef` on the device image counts), the `NOTICE` contents must be
carried in the distributed attribution notices. Add DAMO-YOLO's NOTICE to a
third-party attributions file in the device image.

**Same license covers the larger variants** (`damoyolo_tinynasL25_S`,
`damoyolo_tinynasL35_M`), both of which have Hailo-8L HEFs — so there is a
higher-accuracy upgrade path with no new licensing work.

### YOLOv5 (Ultralytics) — GPL-3.0 at the compiled tag, AGPL-3.0 today ⚠️

This is the model with the best measured recall (90.0%), so the nuance matters.

- **Current `master`:** "GNU AFFERO GENERAL PUBLIC LICENSE Version 3, 19 November 2007"
  — https://raw.githubusercontent.com/ultralytics/yolov5/master/LICENSE
- **At tag `v2.0`, which is what Hailo compiled:** "GNU GENERAL PUBLIC LICENSE
  Version 3, 29 June 2007"
  — https://raw.githubusercontent.com/ultralytics/yolov5/v2.0/LICENSE

So the `yolov5s.hef` derives from weights released under **GPL-3.0**, before the
2023-04-14 relicense. GPL-3.0 has no §13, so the `/stream` network-interaction
argument does not apply to it — but GPL-3.0 §6 conveying obligations do, if the
`.hef` is a covered work. Whether a GPL grant received under v2.0 remains
available after the upstream relicense (GPL grants are generally framed as
irrevocable for copies already received) and whether *weights* are "the Program"
at all are both genuinely unsettled. **NEEDS LEGAL REVIEW. Do not ship yolov5s on
the strength of "it was GPL back then" without that review.** The recall number
is attractive enough that it is worth asking counsel rather than silently
discarding — but note §5 below: the recall advantage is reproducible without it.

### YOLOv8 / YOLO11 (Ultralytics) — AGPL-3.0 ❌

- "GNU AFFERO GENERAL PUBLIC LICENSE Version 3, 19 November 2007"
  — https://raw.githubusercontent.com/ultralytics/ultralytics/main/LICENSE
- Installed package metadata on this machine: `License: AGPL-3.0`, classifier
  "GNU Affero General Public License v3 or later (AGPLv3+)"
  — `.venv/lib/python3.12/site-packages/ultralytics-8.3.28.dist-info/METADATA`
- Was GPL-3.0 until **2023-04-14**.

### YOLOv6 (Meituan) — GPL-3.0 ❌

- "GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007"
  — https://raw.githubusercontent.com/meituan/YOLOv6/main/LICENSE
- At tag `0.1.0` (what Hailo compiled): also GPL-3.0
  — https://raw.githubusercontent.com/meituan/YOLOv6/0.1.0/LICENSE

Strong copyleft. No §13 network clause, but conveying obligations apply. Note
this is a common misconception — YOLOv6 is frequently assumed permissive because
Meituan is not Ultralytics. It is not.

### YOLOX (Megvii) — Apache-2.0 ✅

- "Apache License Version 2.0, January 2004", "Copyright (c) 2021-2022 Megvii Inc.
  All rights reserved."
  — https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/main/LICENSE

Clean. Viable fallback to DAMO-YOLO. `yolox_tiny` HEF exists for Hailo-8L, as do
`yolox_s_leaky` / `yolox_s_wide_leaky`.

### NanoDet — Apache-2.0 ✅

- "Apache License Version 2.0, January 2004", "Copyright 2020-2021 RangiLyu"
  — https://raw.githubusercontent.com/RangiLyu/nanodet/main/LICENSE

### SSD-MobileNet (TF Model Garden) — Apache-2.0 ✅

- "Apache License Version 2.0, January 2004", "Copyright 2022 Google LLC. All
  rights reserved."
  — https://raw.githubusercontent.com/tensorflow/models/master/LICENSE

Clean, but 24.2 mAP and 300×300 input make it the weakest detector on the list.

### RT-DETR — Apache-2.0 upstream ✅, **AGPL-3.0 as used today** ❌

- Upstream `lyuwenyu/RT-DETR`: "Apache License, Version 2.0, January 2004"
  — https://raw.githubusercontent.com/lyuwenyu/RT-DETR/main/LICENSE

**Trap, and it is live in this repo.** `core/detectors.py:30` notes that RT-DETR
loads through the same `ultralytics.YOLO(...)` loader, and `README.md:101`
mentions `rtdetr-l.pt` for A/B testing. **RT-DETR loaded via `ultralytics` is
governed by ultralytics' AGPL-3.0, not by lyuwenyu's Apache-2.0** — you are using
Ultralytics' reimplementation and Ultralytics' weights. Getting RT-DETR under
Apache-2.0 means taking the code *and* the weights from `lyuwenyu/RT-DETR`
directly. No Hailo-8L HEF exists for RT-DETR.

### D-FINE — Apache-2.0 code ✅, **weights split by dataset** ⚠️

- `spdx_id: Apache-2.0` — https://api.github.com/repos/Peterande/D-FINE
- "Apache License Version 2.0, January 2004"
  — https://raw.githubusercontent.com/Peterande/D-FINE/master/LICENSE

The repo states, verbatim:

> "License note: Checkpoints trained or pretrained on Objects365 (*_obj365,
> *_obj2coco) may be subject to the Objects365 dataset terms and should not be
> assumed to be commercially cleared under the D-FINE license."

— https://github.com/Peterande/D-FINE

So: **COCO-only checkpoints (D-FINE-N…X) are the Apache-2.0 ones; `*_obj365` and
`*_obj2coco` are not cleared.** This is a textbook example of exactly the
code-vs-weights split this review was asked to check for. No Hailo-8L HEF.

### RF-DETR (Roboflow) — split licensing ⚠️

- `spdx_id: Apache-2.0`, default branch `develop` —
  https://api.github.com/repos/roboflow/rf-detr
- LICENSE: "Apache License Version 2.0, January 2004", "Copyright 2025 Roboflow, Inc."
  — https://raw.githubusercontent.com/roboflow/rf-detr/develop/LICENSE

The README states, verbatim:

> "Licensing is split by component:
> - The open-source `rfdetr` package and Apache-designated model weights are
>   licensed under Apache License 2.0.
> - Plus components, including the `rfdetr_plus` extension and RF-DETR-XL /
>   RF-DETR-2XL detection models, are licensed under PML 1.0."

— https://github.com/roboflow/rf-detr

Base package and Apache-designated weights: permissive. **RF-DETR-XL / 2XL and
`rfdetr_plus` are under PML 1.0, a separate Roboflow license whose terms were not
read here — UNVERIFIED, and NEEDS LEGAL REVIEW if those variants are ever
considered.** No Hailo-8L HEF. Note `raw.githubusercontent.com/.../main/LICENSE`
404s — the default branch is `develop`, not `main`.

---

## 5. Decision table

Recall figures are the measured phone-recall numbers from the hardware run;
"NMS threshold" is whether the score threshold is controllable at runtime.

| Model | Upstream license | Source URL | Hailo-8L HEF | NMS / threshold | Commercially shippable without a paid license? |
|---|---|---|---|---|---|
| **DAMO-YOLO** `tinynasL20_T` | **Apache-2.0** | https://raw.githubusercontent.com/tinyvision/DAMO-YOLO/master/LICENSE | ✅ verified reachable | **No on-chip NMS** — 6 raw tensors, threshold fully controllable | **YES** (carry NOTICE; do not vendor `tools/onnx_inference.py`) |
| YOLOX `tiny` | Apache-2.0 | https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/main/LICENSE | ✅ | on-chip NMS, fixed 0.200 | **YES** (but threshold locked) |
| NanoDet `repvgg` | Apache-2.0 | https://raw.githubusercontent.com/RangiLyu/nanodet/main/LICENSE | ✅ | on-chip NMS, fixed 0.200 | **YES** (but threshold locked) |
| SSD-MobileNet v2 | Apache-2.0 | https://raw.githubusercontent.com/tensorflow/models/master/LICENSE | ✅ | no NMS | **YES** (weakest accuracy) |
| **YOLOv5s** | **GPL-3.0** at compiled tag `v2.0`; AGPL-3.0 on `master` today | https://raw.githubusercontent.com/ultralytics/yolov5/v2.0/LICENSE | ✅ | on-chip NMS, 0.001 — **best measured, 90.0%** | **NEEDS LEGAL REVIEW** (see §4) |
| **YOLOv11n** (*was* the current detector when this table was written; replaced in Phase 4) | **AGPL-3.0** | https://raw.githubusercontent.com/ultralytics/ultralytics/main/LICENSE | ✅ | on-chip NMS, 0.200 — 25.4% | **NO** — Enterprise License or open-source the product |
| YOLOv8 | AGPL-3.0 | same | ✅ | on-chip NMS | **NO** — same as above |
| YOLOv6n | GPL-3.0 | https://raw.githubusercontent.com/meituan/YOLOv6/main/LICENSE | ✅ | on-chip NMS, 0.200 | **NO** (strong copyleft; conveying obligations) |
| RT-DETR (upstream) | Apache-2.0 | https://raw.githubusercontent.com/lyuwenyu/RT-DETR/main/LICENSE | ❌ none | n/a | YES **only** from lyuwenyu directly — **NO** via `ultralytics` |
| D-FINE (COCO ckpts) | Apache-2.0 | https://raw.githubusercontent.com/Peterande/D-FINE/master/LICENSE | ❌ none | n/a | **YES** for COCO-only; **NO/REVIEW** for `*_obj365` / `*_obj2coco` |
| RF-DETR (base) | Apache-2.0 | https://raw.githubusercontent.com/roboflow/rf-detr/develop/LICENSE | ❌ none | n/a | YES for base; **NEEDS LEGAL REVIEW** for XL/2XL (PML 1.0) |

All seven HEF URLs above were confirmed reachable on 2026-08-29 (HTTP 206 on a
ranged request) at
`https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/<name>.hef`.

**The licensing and the engineering point the same way.** We measured that the
NMS score threshold — not the architecture — dominates small-object recall:
`yolov5s` scored 90.0% at threshold 0.001 while `yolov11n` scored 25.4% at 0.200.
`damoyolo_tinynasL20_T` is the **only** Apache-2.0 candidate with **no on-chip
NMS**, which means its threshold is fully controllable — it is the one model that
can reproduce the yolov5s recall advantage without inheriting yolov5s' copyleft.
The permissively-licensed choice is also the technically correct one. That is a
fortunate alignment, not a coincidence to be relied on twice.

---

## 6. Cross-cutting items

**Removing the AGPL dependency is a small, bounded change — that is the good
news.** `ultralytics` appears in exactly one runtime call site:

- `core/detectors.py:59` — `from ultralytics import YOLO` (the only import)
- `requirements.txt:35` — `ultralytics==8.3.28`
- `config.yaml:521` — default `model_weights: yolo11n.pt`

*(Line numbers re-verified 2026-08-30; they were `:50`, `:4` and `:423` when this
section was written. The call sites are unchanged — only their positions moved.
All three have since been addressed; see §10.)*

`core/accel.py` already deliberately avoids the AGPL stack: its `_nms` docstring
states it is *"Written out rather than pulled from torchvision/ultralytics: this
runs on the Pi inside the no-NMS decode path, where the whole point is not to
depend on a heavyweight (and AGPL) inference stack"*, and it implements its own
letterbox, decode, and ultralytics-shaped result shims. **UNVERIFIED but
important:** those shims are *shaped like* ultralytics' API (`.boxes`, `xyxy[0]`,
`cls[0]`, `conf[0]`). Reimplementing an API surface is generally not copying
expression, but **confirm the shim code was independently written and not
copy-pasted from ultralytics source**, and note for counsel that the API shape is
intentionally mirrored for drop-in compatibility.

**Weights are not in git.** `.gitignore:45-46` excludes `*.pt` and `*.hef`, and
`yolo11n.pt` is untracked. The *repository* therefore does not distribute
AGPL-covered weights. The *device image* would, which is the distribution channel
that matters.

**HailoRT (the runtime SensorDyme links against) is fine.** The README states
HailoRT uses two licenses: *"libhailort, pyhailort & hailortcli"* distributed
under the **MIT license**, and *"hailonet"* (the GStreamer plugin) under
**LGPL-2.1-or-later** —
https://raw.githubusercontent.com/hailo-ai/hailort/master/README.md
SensorDyme uses `hailo_platform` (pyhailort) per `core/accel.py:88`, so **MIT**.
The GStreamer plugin is not used; if that ever changes, LGPL-2.1 dynamic-linking
obligations attach. Note the repo has no top-level LICENSE file and GitHub's
license metadata is `null`, so the README is the authority here — **UNVERIFIED**
whether the Hailo firmware blob and PCIe driver shipped in Raspberry Pi OS's
`hailo-all` package carry separate terms. **What would settle it:**
`dpkg -L hailo-all` on the Pi and read the shipped copyright files.

**Training-dataset terms apply equally to every candidate.** Every model in this
table is COCO-pretrained, and most use ImageNet-pretrained backbones. COCO
annotations are CC BY 4.0 and the images carry Flickr terms; ImageNet's terms are
non-commercial for the *images*. Whether model weights trained on a restricted
dataset are themselves encumbered is **genuinely unsettled law and UNVERIFIED**.
It is worth flagging to counsel that this applies **identically to DAMO-YOLO, to
YOLO11, and to everything else** — it is not a reason to prefer one candidate
over another, and it is not a reason to stay on the AGPL option. D-FINE's
Objects365 disclaimer (§4) shows that at least some upstreams take this
seriously enough to say so in writing.

---

## 7. Open items (UNVERIFIED) and what would settle each

| # | Unverified | What would settle it |
|---|---|---|
| 1 | Whether Hailo has written terms for prebuilt HEFs beyond the repo's MIT | **Email Hailo support/FAE and get it in writing.** Highest value, lowest effort item here. |
| 2 | Ultralytics Enterprise License cost | Request a quote at https://www.ultralytics.com/license — needed even just to price the alternative |
| 3 | Whether importing `ultralytics` makes SensorDyme a combined work | Counsel. This is the crux of §2 and cannot be resolved by engineering |
| 4 | Whether GPL-3.0-era yolov5 v2.0 weights remain usable under GPL-3.0 post-relicense | Counsel |
| 5 | Roboflow PML 1.0 terms (RF-DETR XL/2XL) | Read the PML 1.0 text; only needed if those variants are considered |
| 6 | Whether `core/accel.py` shims contain any copied ultralytics expression | Code provenance review by their author |
| 7 | `hailo-all` firmware/driver licensing | `dpkg -L hailo-all` + read `/usr/share/doc/*/copyright` on the Pi |
| 8 | Dataset-terms encumbrance of COCO/ImageNet-trained weights | Counsel; affects all candidates equally |

---

## 8. Recommended engineering actions

Ordered. Items 1–3 are the release blocker; the rest are hygiene.

1. **Adopt `damoyolo_tinynasL20_T.hef` as the shipping detector.** Apache-2.0
   upstream, HEF verified reachable, and its lack of on-chip NMS is what lets us
   set the score threshold that the recall measurements showed to be decisive.
2. **Remove `ultralytics` from `requirements.txt` and from `core/detectors.py:50`
   on the shipping image.** This is the actual blocker, and it is *not* solved by
   switching models alone — as long as the AGPL package is installed on a device
   sold to a customer, it is being conveyed. Keep it behind a dev-only extra if
   it is still wanted for bench A/B work that never ships.
3. **Change the `config.yaml:423` default off `yolo11n.pt`** so no shipped
   configuration path pulls Ultralytics weights.
4. **Never vendor `tools/onnx_inference.py` from DAMO-YOLO** (CC BY-NC 4.0).
   Add it to a denylist in review notes so it cannot be pulled in casually later.
5. **Ship a third-party attributions file** in the device image carrying the
   DAMO-YOLO `NOTICE` (Apache-2.0 §4(d)), the Hailo Model Zoo MIT notice, and the
   HailoRT MIT notice.
6. **Do not treat RT-DETR-via-ultralytics as an Apache-2.0 option** — fix the
   implication at `core/detectors.py:30` and `README.md:106` / `README.md:186`
   (cited as `README.md:101` when this was written) if either is ever read as a
   licensing statement. Both README references are inside the `mac_max` section,
   which is a developer-machine profile that never ships.
7. **Send the Hailo email (open item #1) this week** — it is the cheapest possible
   reduction in uncertainty and it gates the confidence of §3.

---

> **Again: this is engineering documentation, not legal advice.** It records what
> the cited license texts say and what this codebase and product architecture
> actually do. It does not determine what is legal, and it is not a substitute
> for counsel. **Have a lawyer review this before commercial release**, and put
> open items 3, 4, and 8 in front of them specifically — those are legal
> questions that no amount of source-reading here can close.

---

## 9. Runtime removal — what was actually done (Phase 4)

*Appended after the analysis above; §§1–8 are unchanged. This section records the
engineering change only. It does not revise any licensing conclusion, and the
"have a lawyer review this" note at the end of §8 still stands in full.*

### 9.1 What the audit found

§6 said `ultralytics` "appears in exactly one runtime call site". That is correct
about the *import*, and it understated the exposure in three ways that the removal
had to address:

| # | Finding | Where |
|---|---|---|
| 1 | The import was already **lazy** — inside `get_yolo_model`, not at module scope. So `import core.detectors` never touched ultralytics even before this change; the package was a *dependency* of the CPU path, not of the import graph. | `core/detectors.py:50` |
| 2 | `setup_pi.sh` and `make deploy` both installed `requirements.txt` on the device. Fixing one and not the other would have re-installed the AGPL package on the **next update** after a clean first install. | `setup_pi.sh:40`, `Makefile` `deploy:` |
| 3 | **`make deploy` rsync'd `*.pt` weights onto the device.** `yolo11n.pt` is gitignored, but rsync copies the working tree, not the index, and ultralytics auto-downloads that file into the repo root on any dev run. A deploy from a developer's Mac therefore placed an AGPL-3.0 **model weight** on customer hardware regardless of what was in the requirements file. | `Makefile` `deploy:` rsync flags |

Finding 3 is the one that would have survived a requirements-only fix. Per the
§4 tables, `yolo11n` traces to `ultralytics/ultralytics` and is AGPL-3.0; the
weights carry the licence just as the code does.

Also checked and found clean: no indirect import path reaches ultralytics
(`modules/__init__.py` imports every module eagerly, and none of them import it);
nothing in the repo imports `scipy`, `pandas`, `seaborn`, `PIL`, `tqdm` or
`psutil`, all of which were arriving only as ultralytics dependencies; the single
`import torch` (`modules/driver_attention_max.py:78`) is inside `require_mps()`
on the Mac-only profile; and **no test in `tests/` requires ultralytics** — the
suite's fakes (`tests/test_detectors.py`, `tests/test_driver_hand_motion.py`,
`tests/test_accel.py`) all reproduce the call surface rather than importing it.

### 9.2 The change

Split by **destination**, not by role:

| File | Contents | Installed by |
|---|---|---|
| `requirements-device.txt` | the shipping set — **no ultralytics, no torch** | `setup_pi.sh`, `make deploy` |
| `requirements.txt` | `-r requirements-device.txt` + `ultralytics==8.3.28` | `setup_mac.sh`, CI |
| `requirements-dev.txt` | ruff, pyright, coverage, … (unchanged) | `make setup-dev` |

The include means the shared pins exist once, so the device cannot end up on a
different `opencv` than CI tested against.

`core/detectors.py` now wraps the import: when there is neither a usable
accelerator nor an installed `ultralytics`, `get_yolo_model` raises a
`RuntimeError` naming the weights, both fixes (seat the HAT and set
`accelerator.yolo_hef`, *or* install the developer set), and the reason the AGPL
package is absent. It does **not** return a stub. A detector that silently finds
nothing would leave a baby monitor serving a live dashboard while detecting
nothing at all, which is a worse failure than not booting.

`core/detectors.py` also gains `require_backend(weights, device)` — a startup
preflight, because the two module families differ in *when* they load:
`occupancy.py` loads in `__init__` (fails at boot, loudly), while
`driver_attention.py` loads lazily on the first stride tick, inside the loop
where `services/runtime.py` catches per-frame exceptions and only escalates after
several consecutive failures. Calling `require_backend` once after
`accel.configure` collapses both into the same immediate, readable failure.

### 9.3 What still uses ultralytics, and why that is acceptable

All of it is off-device, where nothing is conveyed to anyone and §13 has no
network-interaction to attach to:

- `modules/driver_attention_max.py` — the `mac_max` developer profile (MPS).
- `benchmarks/accuracy_state_farm.py` — measures the CPU baseline that the Hailo
  accuracy numbers in §4 are compared *against*. Removing it would remove the
  control from the measurement.
- `scripts/prepare_state_farm.py` — produces an ultralytics-native training
  layout; training happens on a workstation.
- CI's runtime-smoke job, which installs `requirements.txt`.

### 9.4 What this does NOT close

- **§8 item 1** (adopt `damoyolo_tinynasL20_T.hef` as the shipping detector) is
  untouched. This change removes the AGPL *package* and the AGPL *weights file*
  from the device; it does not choose which `.hef` the device runs. A device
  pointed at an AGPL-derived `yolo11n.hef` is still in §3's unresolved territory.
- **§8 item 3** (`config.yaml` defaults still name `yolo11n.pt`) is untouched —
  `config.yaml` is device state and out of scope for this change.
- **§7 open items 1–8** are all still open, including item 3 (whether importing
  ultralytics creates a combined work) and item 6 (whether `core/accel.py`'s
  shims contain copied ultralytics expression). Item 3's practical urgency drops
  once the package is not on the device, but it still governs the dev machines
  and CI, and it is still a question for counsel.

`tests/test_phase4_no_agpl_runtime.py` enforces the mechanical parts of this
section: the import graph, the requirements exclusions, both install paths, and
the rsync exclude.

---

## 10. Shipping-image audit — Phase 6 (2026-08-30)

*Appended after §9. §§1–9 are unchanged except for line-number corrections, which
are marked where they occur. This section audits **what a fresh device install
actually contains**, which is a different question from "what does the repo pin".*

### 10.1 Close-out of §8

| §8 item | Status |
|---|---|
| 1. Adopt `damoyolo_tinynasL20_T.hef` as the shipping detector | **DONE.** `config.yaml:55`, provisioned + sha256-verified by `setup_pi.sh`. |
| 2. Remove `ultralytics` from the shipping image | **DONE** (Phase 4 for the repo; Phase 5 removed a surviving install from the physical device). Kept in `requirements.txt` for dev/CI — see §9.3. |
| 3. Change the `config.yaml` default off `yolo11n.pt` | **PARTIAL.** `accelerator.yolo_hef` now names the DAMO `.hef`, so the shipped detector is not Ultralytics. The `model_weights: yolo11n.pt` strings remain at `config.yaml:340, 480, 498, 521, 584` as **cache keys**, annotated as such. No weights file is fetched on a device (`make deploy` excludes `*.pt`), so nothing AGPL is pulled — but the strings still *read* as an Ultralytics default to anyone skimming the config. |
| 4. Never vendor `tools/onnx_inference.py` from DAMO-YOLO | **HOLDING.** Verified absent: `find . -name 'onnx_inference*'` returns nothing outside `.venv`. |
| 5. Ship a third-party attributions file in the device image | **NOT DONE — this is the open obligation.** See §10.3. |
| 6. Do not treat RT-DETR-via-ultralytics as Apache-2.0 | **HOLDING.** Both references are in the Mac-only `mac_max` section. |
| 7. Send the Hailo email (open item 1) | **UNVERIFIED** — no evidence in this repo that it was sent. |

### 10.2 The device dependency closure, resolved rather than assumed

`requirements-device.txt` names 11 packages. Resolving it (`uv pip compile`,
Python 3.12, 2026-08-30) yields **53**. Licence for each read from PyPI package
metadata, not from memory.

**No AGPL, GPL or LGPL package appears anywhere in the closure.** The strongest
copyleft present is `certifi` (**MPL-2.0**, file-level, unmodified — carry the
licence text, no source-disclosure trigger). Everything else is
MIT / BSD / Apache-2.0 / PSF / 0BSD / MIT-CMU / MIT-0.

Four things the file does not say, all of which reach a device:

* **`jax`, `jaxlib`, `scipy`, `ml-dtypes` are installed on the Pi.**
  `mediapipe==0.10.14` requires `jax` and `jaxlib` unconditionally. All four are
  permissive (Apache-2.0 / BSD), so this is not a licence problem — but it is a
  large, unpinned surface nobody chose, on a board where §9.2's stated reason for
  splitting the requirements files was to keep ~800 MB of framework off it.
  §9.1 recorded that "nothing in the repo imports `scipy`"; that is still true of
  *our* code, and `scipy` is on the device regardless.
* **`opencv-contrib-python` is installed alongside the pinned `opencv-python`.**
  `mediapipe` requires `opencv-contrib-python` (unpinned → 4.11.0.86), while
  `requirements-device.txt` pins `opencv-python==4.10.0.84`. **Both provide the
  same `cv2` package**, so one silently overwrites the other and the device's real
  OpenCV version depends on pip's install order. Licence is Apache-2.0 either way;
  the correctness cost is that the pin does not do what the file says it does.
* **`anthropic` (+ `pydantic`, `httpx`, `httpcore`, `h11`, `jiter`, `distro`,
  `anyio`, `sniffio`, `docstring-parser`, `typing-inspection`) reaches the
  device** for the `mac_max` VLM judge, which never runs there. All MIT/BSD/
  Apache-2.0. `requirements-device.txt` calls this out as "harmless (unused) on
  the Pi", which is accurate about licence and understates the dependency count.
* **`pytest` (+ `pluggy`, `iniconfig`) reaches the device.** No shipping module
  imports it (`grep` over `core/ modules/ services/ main.py scripts/` finds
  nothing). MIT, so no licence exposure.

**The real licence risk in the closure is not a package — it is the absence of a
lock.** Only the 11 direct pins are exact; the other 42 resolve fresh on every
install, and `setup_pi.sh`/`make deploy` install without hashes or a lockfile.
The concrete illustration is `chardet`, pulled in by `reportlab`:

| `chardet` version | Licence |
|---|---|
| ≤ 5.2.0 | **LGPL-2.1-or-later** |
| 7.6.0 (what resolves today) | 0BSD |

A device that resolved an older `chardet` — an offline wheelhouse, a mirror, a
pinned-back `reportlab` — would have LGPL-2.1 code in the image, and nothing in
the repo would notice. `tests/test_phase4_no_agpl_runtime.py` reads the
*requirements text*, so it cannot see this class of problem at all. **Pinning the
transitive closure is the single highest-value remaining licence control**, and
it is mechanical: `uv pip compile requirements-device.txt --generate-hashes`.

### 10.3 NOTICE and attribution — the open obligation

§8 item 5 called for a third-party attributions file in the device image. **There
is none.** The only licence text anywhere in the repo is
`third_party/DashSentinel/LICENSE` (MIT, Copyright (c) 2026 Alec Ramdhan —
correctly preserved for the vendored comparison module). There is no `NOTICE`, no
`LICENSE`, no `THIRD_PARTY_NOTICES` at the repo root or in the device image.

What that leaves unmet, concretely:

* **DAMO-YOLO, Apache-2.0 §4(d).** DAMO-YOLO ships a `NOTICE` file. §4(d) requires
  that a derivative work's distributed attribution notices carry the contents of
  that `NOTICE`. `damoyolo_tinynasL20_T.hef` is on the device image and the device
  is conveyed to customers, so this obligation is live and unmet. **This is the
  one attribution gap with a specific, named licence clause behind it.**
* **Apache-2.0 §4(a)/(b) generally** — a copy of the licence and retained
  attribution notices — for `mediapipe`, `opencv-python`, `opencv-contrib-python`,
  `requests`, `absl-py`, `flatbuffers`, `jax`, `jaxlib`, `ml-dtypes`, `distro`,
  and the Hailo Model Zoo compile recipe (MIT) and HailoRT/pyhailort (MIT).
* **BSD/MIT attribution** — `numpy`, `scipy`, `protobuf`, `httpx`, `httpcore`,
  `pycparser`, `idna`, `reportlab`, `contourpy`, `cycler`, `kiwisolver`, and the
  rest of the closure. These clauses are short and the obligation is the same
  shape: reproduce the notice.
* **`certifi` (MPL-2.0)** — carry the licence text; unmodified, so no source
  offer is triggered.
* **`matplotlib` (matplotlib/PSF-style)** and **`typing-extensions` (PSF-2.0)** —
  both require the notice be retained.

`docs/THIRD_PARTY_NOTICES.md` is the start of this file. **It is not complete and
must not be represented as satisfying these obligations** — the per-package
licence *texts* still have to be gathered from the installed distributions, which
can only be done on a machine that has them installed. See the header of that
file for exactly what is missing and the command that produces it.

### 10.4 Recommended next actions

Ordered, and all outside `docs/` — they need an owner who can edit code.

1. **Generate `docs/THIRD_PARTY_NOTICES.md` from real installed metadata** on a
   provisioned Pi, and include DAMO-YOLO's upstream `NOTICE` verbatim. This is the
   only item with a named licence clause (Apache-2.0 §4(d)) currently unsatisfied.
2. **Lock the transitive closure** — `uv pip compile requirements-device.txt
   --generate-hashes` — and install from the lock in `setup_pi.sh` and
   `make deploy`. Closes the `chardet`-shaped hole permanently.
3. **Resolve the `opencv-python` / `opencv-contrib-python` collision.** Two `cv2`
   providers install over each other; the pin currently does not bind.
4. **Decide whether `anthropic` and `pytest` belong on the device.** Neither is a
   licence problem. Both are install surface that the file's own stated principle
   ("split by destination") argues against.

