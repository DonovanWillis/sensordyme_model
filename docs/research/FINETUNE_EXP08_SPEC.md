# EXP-08: DAMO-YOLO-T three-class fine-tune (person, phone, hand)

**Status:** spec written before any run.
- The 1–2 epoch **pilot (EXP-08a)** is authorised (Mission 4 continuation §5) within the envelope below.
- The **full campaign (EXP-08b)** needs separate approval.

## Why this experiment

Baselines are from `research/detector_bench`, Open Images val, manifest `b878e8429d221f3f` (2026-11-11).

1. **Shipping model, decoded as the device decodes it:** phone AP50 0.728, person AP50 0.656.
   - At the shipped 0.20 threshold, 59.5% of verified phone-free images get a false phone box.
2. **Decode fixes, no training:** anchor offset 0 plus multilabel scoring give phone 0.795 and person 0.713. That is the cheapest improvement, and it comes first (it needs Pi validation).
3. **No COCO model can detect hands.** The audit found the product's phone-use rule lacks a hand requirement, and hands currently come only from MediaPipe on the Pi's CPU.
   - A detector with a `hand` class on the Hailo-8L is the capability that fine-tuning, and only fine-tuning, can add.
4. Phone **precision** at useful recall is the product problem: false phone boxes on phone-free images. Fine-tuning with hard negatives targets it directly.

## Specification

| Field | EXP-08a pilot | EXP-08b full campaign (needs approval) |
|---|---|---|
| Hypothesis | The DAMO-YOLO training stack runs on the fleet; we can measure epoch time, peak memory and loss stability for a 3-class head | A 3-class DAMO-YOLO-T, fine-tuned with hard negatives, beats the decode-fixed baseline on phone precision at matched recall, adds hand AP50 ≥ 0.6, and does not lose person AP50 by more than 2 points |
| Baseline | Decode-fixed DAMO-T (`damo_t.offset0_multilabel`) on the same val split | Same, plus DAMO-S |
| Dataset | `oi7-sd-v1` (manifest `b878e8429d221f3f`): the **train** split (author-grouped, 60%), exported by `research/openimages/to_coco.py`. Val is used for monitoring. **Test is never touched** | Same, plus an Open Images *train*-split extension for phones and hands (human-verified boxes only), plus in-cabin data once consented |
| Data rights | Open Images annotations CC BY 4.0; images individually CC BY 2.0 (verified per file); attribution kept in the manifest | Same |
| Checkpoint | `damoyolo_tinynasL20_T_420.pth` (DAMO-YOLO release, Apache-2.0, COCO-trained), sha256 recorded at download | Same |
| Architecture | DAMO-YOLO-T (TinyNAS-L20 backbone, GiraffeNeck, ZeroHead), `num_classes = 3`, new class head, all other weights from the checkpoint | Same; DAMO-S as the alternative if T underfits |
| Loss and optimiser | The repo's own: quality focal + DFL + GIoU; SGD, momentum 0.9, weight decay 5e-4, cosine LR, warm-up | Same; LR sweep (3 points) |
| Hyperparameters | Batch 8 at 640, LR scaled linearly from the config, 2 epochs maximum, the repo's augmentations (mosaic, random affine) | Batch ~16 (gradient accumulation if memory-bound), 60–100 epochs with early stopping on val phone AP50 at matched recall |
| Class balance | Measured on the train split at export (person ≫ hand ≫ phone) | Phone oversampling (repeat factor) |
| Hard negatives | None in the pilot | Images where the baseline fires a phone with no phone verified, plus verified phone-free images, mined from **train** only |
| Label noise | Images where a class was never human-verified may hide unboxed instances. The export reports the count; the pilot includes them | `--require-verified`, or a per-class ignore mask |
| Seeds | 0 | 0, 1, 2 (variance) |
| Evaluation | Loss curve plus the repo's COCO eval on val. A learning signal only, **not comparable** to the Open Images-protocol baselines | ONNX export → `research/detector_bench` (Open Images protocol, same val), then once, frozen, on test |
| Success | Runs without OOM or instability; loss falls; epoch time and peak VRAM measured | Phone precision at recall 0.90 improves by ≥ 0.05 over the decode-fixed baseline with a non-overlapping bootstrap CI; hand AP50 ≥ 0.6; person AP50 within −2 points |
| Failure | Cannot run on torch 2.x without invasive patches → record it, and propose YOLOX-Tiny (Apache, Hailo-8L zoo) instead | Phone precision gain inside the CI; or an INT8 Hailo drop > 1.5 AP50 |
| Conversion | n/a | ONNX export → Hailo DFC (v3.x for hailo8l), calibration on ≥ 1,024 NoIR frames from the device, compile; check host-decoder parity; validate on the Pi |

## EXP-08a resource envelope (fixed before running)

| Resource | Limit |
|---|---|
| Worker | oryn only, through `sdrun` (one job) |
| GPU | `--vram-mb 4500`, below the fleet cap of 6000, because another process already holds about 2 GB of the 8 GB card. `--temp-c 80` (the job stops at 80 °C) |
| CPU and RAM | `--cpus 6 --mem 8G` (VM total 9.9 GB). Dataloader workers 4 |
| Time | `--timeout 3600` (hard). At most 2 epochs |
| Disk | Under `$SDJOB_DATA`: the venv (about 0.5 GB), the checkpoint (34 MB), the COCO export (symlinks plus JSON). No new image downloads |
| Network | GitHub (DAMO-YOLO source at a pinned commit, Apache-2.0), PyPI (pure-Python dependencies plus opencv-headless and pycocotools), the Aliyun OSS checkpoint URL |
| Not touched | The production `.hef`, the device, the test split, the frozen fleet configuration |
