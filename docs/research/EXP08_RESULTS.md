# EXP-08 results: DAMO-YOLO-T, 3 classes (person, phone, hand), short pilots

All runs: oryn RTX 4060, 2 epochs, batch 8, 640, from DAMO-YOLO-T 41.8 (Drive mirror,
sha256 `a5ee6382…`, Apache-2.0); Open Images subset `oi7-sd-v1` (manifest `b878e8429d221f3f`),
train split only; scored on **val** under the Open Images protocol
(`research/detector_bench`). The frozen test split was not used. Results are **2-epoch
pilots**: they describe training behaviour, not a deployable model.

## Label audit (research/openimages/label_audit.py, train split, 4,883 images)

| Class | Verified present | Verified absent | **Never verified** |
|---|---|---|---|
| person | 4,426 | 99 | 358 |
| phone | 260 | 137 | **4,486 (92%)** |
| hand | 2,819 | 158 | **1,906 (39%)** |

No box sits on a verified-absent class; 73/7/31 images are verified-present without a box
(person/phone/hand); 49 hand boxes are mostly a phone box. Stock COCO training treats every
never-verified class as background.

## Variants (per-class loss weight on never-verified classes)

`research/pilot/partial_labels_patch.py` threads each image's unverified classes through
DAMO-YOLO's dataset, mosaic/mixup and head; `partial_labels_sanity.py` proves on the patched
code, before every run, that the masked class gets no gradient and that masks survive
augmentation. `SD_UNVERIFIED_WEIGHT` = 1 is stock, 0 ignores, 0.3 down-weights.

| Run (job) | Weight | Seed | Phone AP50 | Person AP50 | Hand AP50 | Phone-free images with a false phone @0.5 |
|---|---|---|---|---|---|---|
| exp08a (015825-e57970) | 1 (stock) | 1234 | 0.664 | 0.628 | 0.466 | 0.24 |
| exp08a_rerun (040426-e7b2ef) | 1 | 1234 | 0.721 | 0.638 | 0.471 | 0.33 |
| exp08a_s7 (034605-2b0e79) | 1 | 7 | 0.569 | 0.627 | 0.461 | 0.19 |
| exp08c (033424-5b3015) | 0 (ignore) | 1234 | 0.646 | 0.629 | 0.439 | 0.40 |
| exp08c_s7 (035502-846360) | 0 | 7 | 0.706 | 0.618 | 0.440 | 0.50 |
| exp08d_w03 (043507-688dfc) | 0.3 | 1234 | 0.700 | 0.611 | 0.459 | 0.38 |
| exp08d_w03_s7 (044417-17bcc0) | 0.3 | 7 | 0.698 | 0.634 | 0.451 | 0.26 |
| **mean, stock** | | | 0.651 | **0.631** | **0.466** | **0.25** |
| mean, ignore | | | 0.676 | 0.624 | 0.440 | 0.45 |
| mean, soft 0.3 | | | **0.699** | 0.623 | 0.455 | 0.32 |

Reference (no training): the shipping DAMO-T with the reference decoder scores phone 0.795,
person 0.713 on the same val split (COCO classes; no hand).

**Noise floor.** The same seed run twice gave phone AP50 0.664 vs 0.721 (GPU non-determinism):
phone AP at 2 epochs moves about ±0.08 between runs; false-phone rates rest on 42 verified
phone-free images (±0.15). Hand AP (2,113 boxes) is stable to ±0.005.

## Findings

1. **Binary ignore masks hurt precision.** Hand AP drops 0.026 on both seeds (recall unchanged,
   precision lower at every threshold); false phones on phone-free images rise to 0.40–0.50.
   Removing the loss on 92% of images removes almost all phone *negative* supervision.
2. **Soft weighting (0.3)** recovers most of the hand loss and gives the most consistent phone
   AP, but its phone edge is inside the noise floor and its false-phone rate is not better.
3. **Stock training is the most balanced at this scale**: best hand and person, fewest false
   phones. Partial-label handling is principled but NOT justified by this evidence; the binding
   constraints are evaluation power (42 negatives, 128 phones) and training data (325 phone
   boxes), not label weighting.
4. After 2 epochs no variant reaches the pretrained COCO model on phone/person; the only new
   capability is hand (AP50 ≈ 0.46).
5. Throughput (GPU free of the owner's game): 0.25 s/iteration, ~3.2 min per epoch including
   evaluation; 2.3 GB PyTorch memory; peak card 4.9 GB; host RAM ≤ 6.6 GB. With Fortnite
   running, 0.75 s/iteration (3x slower).

## Proposed full run (needs approval; not started)

```
cd ~/sensordyme-fleet && ./sdrun submit -t oryn --repo ~/sensordyme/sensordyme_model \
  --ref <commit with this file> --include research --include evaluation --include core \
  --name exp08-full-60ep --gpu --vram-mb 4500 --temp-c 80 --cpus 6 --mem 8G --timeout 21600 \
  -- bash -c 'EXP=exp08_full60 PARTIAL_LABELS=0 EPOCHS=60 ALLOW_LONG_RUN=1 bash research/pilot/exp08.sh'
```

- Stock semantics (weight 1), 60 epochs, seed 1234, the repo's DAMO-YOLO augmentation; the
  script's eval-every-epoch keeps the best checkpoint; val-only scoring at the end.
- Expected cost: about 3.5 h with the GPU free (60 x ~3.2 min), up to ~10 h if the game runs
  concurrently: schedule with the machine's owner. 6 h timeout as a hard stop.
- Success: hand AP50 ≥ 0.60; phone precision at recall 0.70 ≥ the decoder-fixed pretrained
  model's on val; person AP50 within 2 points of 0.713. Failure: any of those missed, or the
  false-phone rate on phone-free images above the pretrained model's.
- Before it is worth more than one run: enlarge phones/hands and verified negatives (Open Images
  *train*-split extension, ~3–5 GB, needs approval) so differences exceed the noise floor.

## Checkpoint hashes (sha256, first 16 hex; files stay on oryn under $SDJOB_DATA)

exp08c `7349a6608ad11b76` · exp08a_s7 `35bc2f2728ab70a6` · exp08c_s7 `0aa750d0e4c75f0f` ·
exp08a_rerun `2ce383d04871a7e9` · exp08d_w03 `25f6db739c508ef1` · exp08d_w03_s7 `01adfcd48ff7c5e1`
(the original exp08a pilot predates checkpoint hashing). None is a production candidate.
