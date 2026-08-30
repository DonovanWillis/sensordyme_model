# Third-party notices — SensorDyme device image

> **STATUS: INCOMPLETE. Do not represent this file as satisfying SensorDyme's
> attribution obligations yet.** It is the start of the third-party attributions
> file that `docs/LICENSING.md` §8 item 5 called for and §10.3 records as still
> unmet. What is here is accurate; what is missing is stated explicitly below.
>
> **This is engineering documentation, not legal advice.** Every licence
> identification below comes from package metadata or an upstream file at the URL
> cited, read on the date given — not from memory. Have counsel review before a
> commercial release.

**Scope.** This file covers what a fresh install puts on a **customer device**:
`requirements-device.txt` as installed by `setup_pi.sh` and `make deploy`, plus
the detection model and the Hailo runtime. It does **not** cover the developer /
CI install (`requirements.txt`, which additionally pins the AGPL-3.0
`ultralytics`) or the tooling set (`requirements-dev.txt`) — neither reaches a
device. See `docs/LICENSING.md` for why that split exists.

## What is still missing

1. **The licence *texts* themselves.** Apache-2.0 §4(a) and the BSD/MIT clauses
   require reproducing the notice, not merely naming it. The tables below name
   the licence for every package; they do not yet carry each package's `LICENSE`
   file. Those must be collected from the installed distributions on a
   provisioned device, because that is the only place the exact shipped versions
   exist:

   ```bash
   # on the Pi, in ~/sensordyme
   .venv/bin/python - <<'PY'
   from importlib.metadata import distributions
   for d in sorted(distributions(), key=lambda d: d.metadata["Name"].lower()):
       name = d.metadata["Name"]
       for f in (d.files or []):
           if f.name.upper().startswith(("LICENSE", "NOTICE", "COPYING", "AUTHORS")):
               print(f"\n===== {name} {d.version} :: {f.name} =====")
               print(d.read_text(str(f)))
   PY
   ```

2. **System (apt) packages.** `setup_pi.sh` installs `python3-picamera2`,
   `python3-libcamera`, `python3-dbus`, `python3-gi`, `bluez`, `avahi-daemon`,
   `libatlas-base-dev`, `libopenjp2-7`, `alsa-utils` and `hailo-all` from Pi OS.
   **Several of these are LGPL or GPL** (`bluez` is GPL-2.0; `python3-gi`/GLib is
   LGPL-2.1; `libcamera` is LGPL-2.1). They are pre-existing OS components rather
   than things SensorDyme redistributes, but if SensorDyme ships a **prepared SD
   card image** it is redistributing them, and their obligations attach —
   including LGPL's relinking/source-offer terms. **UNVERIFIED and NEEDS LEGAL
   REVIEW: whether SensorDyme's distribution model conveys the OS image.** This
   is a materially different question from the Python closure below and it is not
   answered here. `docs/LICENSING.md` §7 open item 7 covers `hailo-all`
   specifically; settle it with `dpkg -L hailo-all` and
   `/usr/share/doc/*/copyright` on the device.

3. **`simpleaudio`** is MIT, but on Linux it links ALSA (`libasound`), which is
   **LGPL-2.1**. Dynamic linking, so the LGPL relinking obligation is the live
   one. **UNVERIFIED** — confirm the linkage on the device.

4. **The `.venv` is built with `--system-site-packages`**, so system Python
   packages are importable from it. Item 2 therefore cannot be skipped.

---

## 1. Detection model — DAMO-YOLO (Apache-2.0)

The device runs `models/damoyolo_tinynasL20_T.hef`: Hailo Model Zoo **v2.15.0**,
`hailo8l`, sha256
`e69136d3ebf3cc8ebaf3da5417a59962b139f3d3d67d715e9a9b1ae10963960e`,
18,554,430 bytes, provisioned and checksum-verified by `setup_pi.sh`.

Upstream is `tinyvision/DAMO-YOLO`, Apache-2.0
(https://raw.githubusercontent.com/tinyvision/DAMO-YOLO/master/LICENSE).
Apache-2.0 **§4(d)** requires that the contents of the upstream `NOTICE` file be
carried in the attribution notices of a distributed derivative work. The `.hef`
is on a device conveyed to customers, so the following is reproduced verbatim
from
https://raw.githubusercontent.com/tinyvision/DAMO-YOLO/master/NOTICE
(retrieved 2026-08-30, 7,276 bytes).

Two notes on reading it, neither of which changes SensorDyme's obligation to
carry it:

* The `tools/onnx_inference.py` entry (CC BY-NC 4.0) refers to a standalone demo
  script in DAMO-YOLO's repository. It is not part of the model, the weights, or
  anything a `.hef` is compiled from, and **SensorDyme must never vendor it** —
  `docs/LICENSING.md` §4 and §10.1 item 4. Verified absent from this repo.
* The trailing "Packages installed by codes or PIP" section lists DAMO-YOLO's own
  *training environment* (PyTorch, Pillow, `easydict` under LGPL-3.0, …). Those
  are inputs to Alibaba's training, not components of the compiled `.hef`, and
  SensorDyme ships none of them. It is reproduced because §4(d) says to reproduce
  the NOTICE, not to edit it.

```
Copyright (c) 2021-2022 Alibaba Group Holding Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
===========================================================


MIT License
-----------------------------

damo/utils/timer.py
damo/utils/imports.py
damo/dataset/transforms/tta_aug.py
damo/dataset/build.py
damo/dataset/collate_batch.py
damo/dataset/datasets/concat_dataset.py
damo/dataset/datasets/evaluation/coco/coco_eval.py
damo/dataset/samplers/distributed.py
damo/dataset/samplers/grouped_batch_sampler.py
damo/structures/bounding_box.py
damo/structures/boxlist_ops.py
damo/structures/image_list.py
Copyright (c) Facebook, Inc. and its affiliates.
All Rights Reserved.


Apache License 2.0 License
-----------------------------

damo/core/anchor.py
damo/core/sampler.py
damo/core/utils.py
damo/core/weight_init.py
damo/core/backbone/darknet.py
damo/core/ota_assigner.py
damo/core/atss_assigner.py
damo/base_models/losses/losses.py
damo/base_models/backbones/darknet.py
damo/base_models/necks/pafpn.py
opyright (c) OpenMMLab.
All rights reserved.


Creative Commons Attribution-NonCommercial 4.0 International Public License
-----------------------------

tools/onnx_inference.py

============================================================
Packages installed by codes or PIP.
============================
PyTorch's BSD-style license
============================
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions by Cruise LLC:
Copyright (c) 2022 Cruise LLC.
All rights reserved.

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

============================
Pillow's HPND License
============================
The Python Imaging Library (PIL) is

    Copyright © 1997-2011 by Secret Labs AB
    Copyright © 1995-2011 by Fredrik Lundh

Pillow is the friendly PIL fork. It is

    Copyright © 2010-2022 by Alex Clark and contributors

Like PIL, Pillow is licensed under the open source HPND License:

By obtaining, using, and/or copying this software and/or its associated
documentation, you agree that you have read, understood, and will comply
with the following terms and conditions:

Permission to use, copy, modify, and distribute this software and its
associated documentation for any purpose and without fee is hereby granted,
provided that the above copyright notice appears in all copies, and that
both that copyright notice and this permission notice appear in supporting
documentation, and that the name of Secret Labs AB or the author not be
used in advertising or publicity pertaining to distribution of the software
without specific, written prior permission.

SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR BE LIABLE FOR ANY SPECIAL,
INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.

============================
BSD License
============================
numpy==1.21.5
omegaconf
scikit-image


============================
Apache Software License
============================
Cython==0.29.24

============================
Apache License 2.0 License
============================
onnx==1.8.1
onnx-simplifier==0.3.5
ninja
timm

============================
MIT License
============================
tabulate==0.8.9
thop==0.0.31
opencv-python==4.5.4.60
thop
tensorboard
onnxruntime==1.8.0
loguru
scikit-image
tqdm

============================
LGPL-3.0 License
============================
easydict
```

---

## 2. Accelerator runtime — Hailo

| Component | Licence | Source |
|---|---|---|
| Hailo Model Zoo (the compile recipe and packaging around the `.hef`) | **MIT**, "Copyright (c) Hailo, Inc. and its affiliates" | https://raw.githubusercontent.com/hailo-ai/hailo_model_zoo/master/LICENSE |
| HailoRT — `libhailort`, `pyhailort`, `hailortcli` (what `core/accel.py` imports as `hailo_platform`, `core/accel.py:88`) | **MIT** | https://raw.githubusercontent.com/hailo-ai/hailort/master/README.md |
| HailoRT — `hailonet` GStreamer plugin | LGPL-2.1-or-later | same | 

SensorDyme uses `pyhailort` only; the GStreamer plugin is **not** used, so the
LGPL-2.1 obligations do not currently attach. If that changes, they do.

Hailo's MIT licence covers Hailo's own contribution and **does not relicense the
upstream model weights** — see `docs/LICENSING.md` §3. That is why this file
attributes DAMO-YOLO separately above.

**UNVERIFIED:** the Hailo firmware blob and PCIe driver shipped in Pi OS's
`hailo-all` apt package may carry separate terms. `docs/LICENSING.md` §7 open
item 7.

---

## 3. openpilot driver-monitoring model

`dmonitoring_model.onnx` (16,132,780 bytes) ships in the repo and is deliberately
**not** excluded by `make deploy` — `modules/driver_attention_openpilot.py` runs
it on the device via `onnxruntime`.

**Provenance is recorded** — `modules/driver_attention_openpilot.py:1-10` states
the model and the surrounding decision logic are vendored verbatim from
**openpilot v0.9.7**, `github.com/commaai/openpilot`.

**openpilot is MIT**, "Copyright (c) 2018, Comma.ai, Inc.", verified at both
https://raw.githubusercontent.com/commaai/openpilot/master/LICENSE and the
`v0.9.7` tag
(https://raw.githubusercontent.com/commaai/openpilot/v0.9.7/LICENSE), retrieved
2026-08-30. GitHub's licence metadata agrees (`spdx_id: MIT`). MIT requires the
copyright and permission notice be included in distributions — **carry that
notice**, which this file does by citing it here and which the missing-item 1
collection above must reproduce in full.

Two things remain open, and neither is answered here:

* **UNVERIFIED — whether the MIT repository licence governs the released
  *model artifact* as well as the code.** Some projects licence weights
  separately from source. No separate model-licence file was found for openpilot,
  but absence of a file is not a grant. This is the same code-vs-weights question
  `docs/LICENSING.md` §3 works through for the `.hef`.
* **The file is not pinned.** Unlike the `.hef`, `dmonitoring_model.onnx` has no
  URL and no checksum recorded anywhere in the repo — it is simply committed.
  **Recommendation:** provision and checksum-verify it the way `setup_pi.sh`
  provisions the `.hef`, so the shipped binary is traceable to a release.

---

## 4. Vendored source — DashSentinel

`third_party/DashSentinel/` is a full vendored copy of
`github.com/alec-kr/DashSentinel` (commit `3874e05b`), used by
`modules/driver_attention_dashsentinel.py` for head-to-head comparison. It is
**MIT**, "Copyright (c) 2026 Alec Ramdhan", and its `LICENSE` file is preserved
in place at `third_party/DashSentinel/LICENSE` — which is what MIT requires.

It reaches the device: `make deploy` does not exclude `third_party/`. Whether it
*should* is a packaging question, not a licence one.

---

## 5. Python packages on the device

Resolved from `requirements-device.txt` with
`uv pip compile --python-version 3.12` on **2026-08-30**: 11 direct pins expand
to **53 packages**. Licence for each read from that version's PyPI metadata
(`license_expression`, else the trove classifier, else the `license` field).

**No AGPL-, GPL- or LGPL-licensed Python package appears in this closure.** The
strongest copyleft present is `certifi` (MPL-2.0 — file-level, unmodified, so
carrying the licence text is the whole obligation).

**This table is a snapshot, not a lock.** Only the 11 direct pins are exact; the
other 42 are resolved fresh on every install, and neither `setup_pi.sh` nor
`make deploy` installs from a lockfile or with hashes. A different resolution can
therefore produce different licences — `chardet` is the live example: 0BSD at
7.6.0 below, but **LGPL-2.1-or-later at 5.2.0 and earlier**. See
`docs/LICENSING.md` §10.2. Re-generate this table whenever the closure moves, and
prefer fixing the underlying problem by locking the closure.

| Package | Version | Licence (PyPI metadata) | Why it is on the device | Project |
|---|---|---|---|---|
| `absl-py` | 2.5.0 | Apache-2.0 | via `mediapipe` | https://github.com/abseil/abseil-py |
| `annotated-types` | 0.8.0 | MIT | via `pydantic` | https://github.com/annotated-types/annotated-types |
| `anthropic` | 0.116.0 | MIT License | **direct pin** | https://github.com/anthropics/anthropic-sdk-python |
| `anyio` | 4.14.2 | MIT | via `anthropic`, `httpx` |  |
| `attrs` | 26.1.0 | MIT | via `mediapipe` |  |
| `certifi` | 2026.7.22 | Mozilla Public License 2.0 (MPL 2.0) | via `httpcore`, `httpx`, `requests` | https://github.com/certifi/python-certifi |
| `cffi` | 2.1.1 | MIT-0 | via `sounddevice` |  |
| `chardet` | 7.6.0 | 0BSD | via `reportlab` | https://github.com/chardet/chardet |
| `charset-normalizer` | 3.5.1 | MIT | via `requests` |  |
| `contourpy` | 1.3.3 | BSD License | via `matplotlib` | https://github.com/contourpy/contourpy |
| `cycler` | 0.12.1 | BSD License | via `matplotlib` |  |
| `distro` | 1.9.0 | Apache Software License | via `anthropic` | https://github.com/python-distro/distro |
| `docstring-parser` | 0.18.0 | MIT License | via `anthropic` |  |
| `flatbuffers` | 25.12.19 | Apache Software License | via `mediapipe`, `onnxruntime` | https://google.github.io/flatbuffers/ |
| `fonttools` | 4.63.0 | MIT | via `matplotlib` | http://github.com/fonttools/fonttools |
| `h11` | 0.16.0 | MIT License | via `httpcore` | https://github.com/python-hyper/h11 |
| `httpcore` | 1.0.9 | BSD-3-Clause | via `httpx` | https://www.encode.io/httpcore/ |
| `httpx` | 0.28.1 | BSD License | via `anthropic` | https://github.com/encode/httpx |
| `idna` | 3.19 | BSD-3-Clause | via `anyio`, `httpx`, `requests` | https://github.com/kjd/idna |
| `iniconfig` | 2.3.0 | MIT | via `pytest` | https://github.com/pytest-dev/iniconfig |
| `jax` | 0.7.1 | Apache-2.0 | via `mediapipe` | https://github.com/jax-ml/jax |
| `jaxlib` | 0.7.1 | Apache-2.0 | via `jax`, `mediapipe` | https://github.com/jax-ml/jax |
| `jiter` | 0.16.0 | MIT | via `anthropic` | https://github.com/pydantic/jiter/ |
| `kiwisolver` | 1.5.1 | BSD License | via `matplotlib` |  |
| `matplotlib` | 3.9.2 | Python Software Foundation License | **direct pin** | https://matplotlib.org |
| `mediapipe` | 0.10.14 | Apache Software License | **direct pin** | https://github.com/google/mediapipe |
| `ml-dtypes` | 0.5.4 | Apache-2.0 | via `jax`, `jaxlib` |  |
| `numpy` | 1.26.4 | BSD License | **direct pin** | https://numpy.org |
| `onnxruntime` | 1.27.0 | MIT License | **direct pin** | https://onnxruntime.ai |
| `opencv-contrib-python` | 4.11.0.86 | Apache Software License | via `mediapipe` | https://github.com/opencv/opencv-python |
| `opencv-python` | 4.10.0.84 | Apache Software License | **direct pin** | https://github.com/opencv/opencv-python |
| `opt-einsum` | 3.4.0 | MIT License | via `jax` |  |
| `packaging` | 26.3 | Apache-2.0 OR BSD-2-Clause | via `matplotlib`, `onnxruntime`, `pytest` | https://github.com/pypa/packaging |
| `pillow` | 12.3.0 | MIT-CMU | via `matplotlib`, `reportlab` | https://python-pillow.github.io |
| `pluggy` | 1.6.0 | MIT License | via `pytest` |  |
| `protobuf` | 4.25.9 | 3-Clause BSD License | via `mediapipe`, `onnxruntime` | https://developers.google.com/protocol-buffers/ |
| `pycparser` | 3.0 | BSD-3-Clause | via `cffi` | https://github.com/eliben/pycparser |
| `pydantic-core` | 2.46.5 | MIT | via `pydantic` | https://github.com/pydantic/pydantic |
| `pydantic` | 2.13.5 | MIT | via `anthropic` | https://github.com/pydantic/pydantic |
| `pyparsing` | 3.3.2 | MIT | via `matplotlib` | https://github.com/pyparsing/pyparsing/ |
| `pytest` | 8.3.3 | MIT License | **direct pin** | https://docs.pytest.org/en/latest/ |
| `python-dateutil` | 2.9.0.post0 | Apache Software License | via `matplotlib` | https://github.com/dateutil/dateutil |
| `pyyaml` | 6.0.2 | MIT License | **direct pin** | https://pyyaml.org/ |
| `reportlab` | 4.2.5 | BSD License | **direct pin** | https://www.reportlab.com/ |
| `requests` | 2.34.2 | Apache Software License | **direct pin** | https://github.com/psf/requests |
| `scipy` | 1.17.1 | BSD License | via `jax`, `jaxlib` |  |
| `simpleaudio` | 1.0.4 | MIT License | **direct pin** | https://github.com/hamiltron/py-simple-audio |
| `six` | 1.17.0 | MIT License | via `python-dateutil` | https://github.com/benjaminp/six |
| `sniffio` | 1.3.1 | Apache Software License | via `anthropic` | https://github.com/python-trio/sniffio |
| `sounddevice` | 0.5.6 | MIT | via `mediapipe` |  |
| `typing-extensions` | 4.16.0 | PSF-2.0 | via `anthropic`, `anyio`, `pydantic`, `pydantic-core`, `typing-inspection` |  |
| `typing-inspection` | 0.4.4 | MIT | via `pydantic` | https://github.com/pydantic/typing-inspection |
| `urllib3` | 2.7.0 | MIT | via `requests` |  |

### Notes on individual rows

* **`chardet` 7.6.0 — 0BSD.** Pulled in by `reportlab`, unpinned. Versions ≤ 5.2.0
  are **LGPL-2.1-or-later**. This row is the reason §10.2 of `docs/LICENSING.md`
  recommends locking the transitive closure.
* **`certifi` — MPL-2.0.** Weak, file-level copyleft. Unmodified here, so the
  obligation is to carry the licence text; no source offer is triggered.
* **`opencv-python` 4.10.0.84 and `opencv-contrib-python` 4.11.0.86 are both
  installed.** The first is a direct pin; the second is required by `mediapipe`.
  Both provide the same `cv2` package, so one overwrites the other and the
  device's effective OpenCV version depends on install order. Apache-2.0 either
  way — this is a correctness problem, not a licence one, but it means the pinned
  version is not necessarily the running one.
* **`jax`, `jaxlib`, `scipy`, `ml-dtypes`** are on the device solely because
  `mediapipe==0.10.14` requires `jax` and `jaxlib`. All permissive.
* **`anthropic` and its dependency chain** (`pydantic`, `pydantic-core`, `httpx`,
  `httpcore`, `h11`, `anyio`, `sniffio`, `jiter`, `distro`, `docstring-parser`,
  `typing-inspection`, `annotated-types`) reach the device for the `mac_max` VLM
  judge, which never runs there. All MIT / BSD / Apache-2.0.
* **`pytest`, `pluggy`, `iniconfig`** reach the device; no shipping module imports
  `pytest`. MIT.
* **`matplotlib`** carries the matplotlib licence (PSF-derived, BSD-compatible);
  **`typing-extensions`** is PSF-2.0. Both require the notice be retained.

---

## 6. Not on the device (recorded so the boundary is explicit)

| Component | Licence | Where it lives |
|---|---|---|
| `ultralytics==8.3.28` | **AGPL-3.0** | `requirements.txt` — developer machines and CI **only**. Removed from the device in Phase 4; a surviving install was removed from the physical device in Phase 5. `tests/test_phase4_no_agpl_runtime.py` enforces the exclusion. |
| `torch`, `torchvision` | BSD-3-Clause / Apache-2.0 / MIT | dependencies of `ultralytics`; never on the device |
| `yolo11n.pt` | **AGPL-3.0 weights** (`ultralytics/ultralytics`) | untracked in git, auto-downloaded by ultralytics on a dev machine. `make deploy` excludes `*.pt`. |
| Any `.hef` in a working tree | varies by upstream model | `make deploy` excludes `*.hef`; the device's model comes only from `setup_pi.sh`'s pinned, checksum-verified URL |
| `requirements-dev.txt` (ruff, pyright, pytest-cov, mutmut, pre-commit, pip-audit) | permissive | developer machines only |

---

*Generated 2026-08-30. Regenerate whenever `requirements-device.txt`, the pinned
`.hef`, or the vendored `third_party/` contents change.*
