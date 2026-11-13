# M8.4 research package: edge detection for driver monitoring on a Pi 5 + Hailo-8L

Section 9 of Mission 8.4 (BETA stream). This is a research *plan and record*, written so that a
reviewer can check every number against a fleet job and rerun it. It is not a paper and makes no
claim of novelty. Companion: `docs/research/M84_MODEL_AUDIT.md` (all measurements),
`research/registry/experiments.jsonl` (the experiment registry), figures and their data in
`docs/research/m84_figures/` (made by `research/m84/figures.py` and `research/m84/summarise_b2.py`).

**Scope of evidence.** Every accuracy number is image-level, on Open Images V7 val (`oi7-sd-v1`,
manifest `b878e8429d221f3f`). None is in-cabin accuracy, NIR accuracy, distracted-driving *event*
accuracy, or a Hailo measurement. The frozen test split has never been used.

## 1. Research questions

| ID | Question | Status |
|---|---|---|
| RQ1 | Does fine-tuning DAMO-YOLO-T to {person, phone, hand} lose person accuracy because the fine-tune loader re-initialises the class-agnostic box regressor, rather than because of class interference? | tested by EXP-09 (§6 of the audit) |
| RQ2 | How large is the accuracy cost of 8-bit post-training quantisation of this detector, and where does it come from (weights/activations vs the uint8 outputs the `.hef` returns)? | L2 host simulation done (audit §5.5); L3/L4 pending |
| RQ3 | Can a detector-predicted hand box gate "phone present" into "phone held" without losing in-hand phones? | M8.1: yes on OI (false-phone images 33% → 2.4%, in-hand recall unchanged at 0.81; 21 in-hand phones) |
| RQ4 | Which failure modes dominate (size, occlusion, darkness proxy, look-alikes, localisation)? | audit §5 |
| RQ5 | Does an uncertainty-aware, temporally filtered decision layer reduce false alerts at a fixed miss rate on real in-cabin video? | **not testable yet**: no in-cabin video exists |

## 2. Baselines

1. Shipping: DAMO-YOLO-T COCO `.hef` (zoo v2.15.0), host decoder `legacy` (person AP50 0.656,
   phone 0.728; L1 float surrogate of the `.hef`; device path reproduced in M8.3).
2. Same weights, `reference` decoder (0.713 / 0.795).
3. EXP-08: 3-class fine-tune with the stock loader (person 0.686, phone 0.807, hand 0.725 at NMS 0.5).
4. Host-simulated INT8 of (1)/(2) (audit §5.5).
No external SOTA comparison is made: the task (image-level person/phone/hand on OI) has no
published benchmark protocol we match, and none was run.

## 3. Experimental models

| Model | Init | Trained | Classes | Evidence level |
|---|---|---|---|---|
| DAMO-T COCO (zoo) | COCO | — | 80 | L1, L2; L4 latency only |
| EXP-08 | COCO backbone+neck, random head | 60 ep OI train | 3 | L1 (+ L2 if exported, §8) |
| EXP-09 | COCO backbone+neck+box regressor+scales; person/phone class rows from COCO | 60 ep OI train, seed applied | 3 | L1 |

## 4. Dataset provenance

- Source: Open Images V7, human-verified image-level labels and boxes for person (incl. man, woman,
  boy, girl), mobile phone, human hand; images CC BY 2.0 per file with attribution in the manifest;
  annotations CC BY 4.0. Built by `research/openimages/oi.py` (Mission 4).
- Splits: photographer-grouped 60/20/20 (train 4,883 / val 1,585 / test 1,584 frozen); near
  duplicates grouped. Leakage check (`research/m84/leakage_check.py`): 0 shared ids, groups or file
  hashes across splits.
- Not used, and why: State Farm (competition-only), DMD and other driver datasets (research-only /
  NC-ND), WIDER-derived face sets (research-only).

## 5. Methodology

- **Metrics.** Open Images protocol (`research/detector_bench/oid_eval.py`): class scored only where
  verified; group-of rule; AP50 all-point; AP50:95 for localisation; operating points: recall at
  precision 0.80 (R@P0.80), precision at recall 0.70, false-box rate on verified-absent images.
- **Uncertainty.** Photographer-group bootstrap (the split unit), paired between models on the same
  resamples (`research.m8.analysis paired`, 300–1,000 resamples). Reported as Δ with a 95% percentile
  interval and p(Δ ≤ 0). No multiple-comparison correction is applied; the pre-registered gates are
  few and stated before the run.
- **Pre-registration.** Criteria are written into the registry before the job starts
  (`preregistered_at` < `started_at`, enforced by `research/registry/registry.py validate`).
- **Selection.** Checkpoints are selected on val COCO AP50:95 by the trainer (as EXP-08); the same
  val then scores them. This is optimistic for both EXP-08 and EXP-09 equally; the test split is
  reserved for a single final measurement of a chosen model.

## 6. Error analysis (summary; detail in the audit §5)

Small objects dominate misses (person recall 0.73 large → 0.18 small); person errors are mostly
mislocalisation; false phones are mostly on verified phone-free images and co-fire with "remote";
cross-class confusion among person/phone/hand is rare (≤ 7 person FPs on hand boxes); no measurable
degradation in the darkest quartile of web photos (a proxy only).

## 7. Ablation candidates

| Ablation | Isolates | Cost |
|---|---|---|
| EXP-09 vs EXP-08 | head initialisation (+ seed) | 1 run (done) |
| EXP-09a: copy `gfl_reg`/scales only, class rows random | box-regressor vs class-row init | 3.1 h |
| EXP-09b: freeze `gfl_reg` | whether re-learning box regression helps or hurts | 3.1 h |
| Seeds × 3 for the winner | run-to-run variance (unknown at 60 epochs) | 9 h |
| Hard negatives (136 verified train images) | look-alike suppression | 3.1 h; needs more val phones to measure |
| Partial-label weight 0.3 at 60 epochs | label noise handling | 3.1 h (2-epoch pilots inconclusive) |

## 8. Latency evaluation

| What | Host (measured) | Pi 5 + Hailo-8L |
|---|---|---|
| DAMO-T heads, float ONNX | x86 CPU ORT 62 ms p50 (quantcompute, 8 threads, `085405-4d4ad4`); 64.8 ms (oryn, M8.3); CUDA RTX 4060 6.6 ms (full zoo graph, `20261111-090242-3ab280`) | 22.07 ms mean accelerator time (PHASE5, shipping `.hef`) |
| Same, QDQ INT8 on x86 | 45 ms p50 (quantcompute) | — (host QDQ timing says nothing about the NPU) |
| 3-class EXP-08 ONNX | 5.8 ms CUDA / 47.8 ms CPU (oryn) | expected ≈ shipping (same backbone, 3 vs 80 class channels): **unmeasured** |
| Decode (legacy / reference, conf 0.2) | 1.66 / 2.73 ms p50 x86 (M8.3) | pending (`hwv_stage.py latency`) |
| Device path predict() with float stand-in, 3-class heads81 | audit §8 (x86; not Hailo) | pending |

Rule: a host number is never quoted as a Pi or Hailo number; the ordering of stage costs is what
transfers (M8.3).

## 9. False-positive analysis

Audit §5.3 and `b2_top_fps.csv`: per model and class, FPs split into localisation / duplicate /
cross-class / verified-absent / background, with the co-firing COCO look-alike. For the product, the
relevant false positive is a *false phone-use alert*, which needs the temporal layer and video;
image-level false phones are its upstream input only.

## 10. Quantisation evaluation plan

| Level | Done? | How |
|---|---|---|
| L2 host simulation | **yes** (shipping heads; 3-class candidate in audit §8) | `research/m84/hostquant.py`: ORT QDQ, 256 train calibration images, per-channel/per-tensor/percentile, uint8 outputs |
| L3 compiled artifact | no: no DFC on the fleet (oryn/quantcompute probed: no `hailo_sdk_client`, `hailo_platform`, `hailo_model_zoo`) | DFC on an x86 Linux host: parse `<cand>.heads81.onnx` with the six end nodes, optimise with ≥ 1,024 in-domain NoIR calibration frames, compile for hailo8l; `hailortcli parse-hef` must show 6 outputs 68/81 at 80/40/20; run the DFC INT8 emulator on val and compare with L1/L2 |
| L4 runtime | no | Pi stages (`scripts/hw_validation/hwv_stage.py`): parity on the known-good and OI-val float fixtures from audit §8, latency, thermal; thresholds re-derived on INT8 scores at matched precision |

Acceptance (proposed, from M8): INT8 vs float AP50 drop ≤ 0.015 per class on OI val **at L3/L4**; the
L2 result (person −0.032) says this gate may fail with naive PTQ and that the DFC's optimisation
level matters.

## 11. Reproducibility

- Code at a committed ref; fleet jobs record commit, environment lock hash, GPU driver, limits and
  usage (`runs/<id>/state.json`, copied into the evidence folder).
- Data: manifest id + per-file sha256; predictions hashed (first 16 hex in every report).
- Models: checkpoint and ONNX sha256 (audit §9); large files stay on the workers.
- Seeds: EXP-09 applies seed 1234 (the stock DAMO trainer does not; EXP-08 was unseeded). GPU
  non-determinism remains: a rerun is a new draw, not a bit-identical replay.
- Registry: `python -m research.registry.registry validate|show`.

## 12. Limitations and threats to validity

1. **Domain.** OI web photos ≠ in-cabin NoIR video (viewpoint, IR illumination, motion blur, posture,
   occlusion by the wheel). No in-cabin number exists.
2. **Power.** 128 val phones, 42 phone-free images, 21 in-hand phones: phone and gate effects below
   ~0.08 AP50 / ~15 points of rate are unresolvable.
3. **Label noise.** OI hands are under-annotated (39% of train images never verified); hand
   "false positives" include unlabelled hands; phone negatives are rare (92% never verified).
4. **Single runs.** EXP-08 and EXP-09 are one draw each; 60-epoch variance is unknown.
5. **Val reuse.** Checkpoint selection and evaluation share val.
6. **Inference-path mismatch.** Trained models were trained with stretch resize; the device
   letterboxes. Audit §8 measures the device path for the candidate; PyTorch-path numbers are not
   device numbers.
7. **Quantisation.** L2 ≠ L3/L4 (§10).
8. **Proxy darkness.** Mean luma of web photos is not night-time NIR.

## 13. The candidate contribution, assessed honestly

**Hypothesis (not a claim):** *uncertainty-aware, temporally robust distracted-driving detection on
resource-constrained edge hardware.*

What exists today is **engineering integration**: a zoo detector on a Hailo-8L, a corrected host
decoder, a measured hand-held gate on still images, calibration maps, a device-path validation
harness and an INT8 sensitivity estimate. Each is useful product work; none is a research result.

What would make it research, and the evidence each part needs:

| Component | Novel? | Evidence needed |
|---|---|---|
| Calibrated per-frame probabilities (Platt per model × decoder × INT8) | standard technique | show calibration *survives INT8 on the device* and across lighting (L4 + in-cabin) |
| Hand-gated phone "held" signal from one 3-class network | a modest design result if it holds | in-cabin clips with event labels; recall of phone-use events at a fixed false-alert rate vs a no-gate baseline; > 21 in-hand examples |
| Temporal robustness (k-of-n, track-consistent hand–phone motion) | standard ideas; novelty only in a careful edge evaluation | event-level metrics on video (precision/recall per event, time-to-alert, false alerts per hour), ablated against frame-level rules |
| Uncertainty-aware alerting (abstain / defer when calibrated confidence is low) | possibly novel in this setting | risk–coverage curves on in-cabin video; a demonstration that abstention lowers false alerts per hour without raising missed events |
| "Resource-constrained" | engineering | Pi-side latency, power and thermal under the full pipeline (L4) |

**Verdict:** a plausible applied-research direction, currently at hypothesis stage. The decisive
missing ingredient is consented in-cabin video with event labels (INCABIN_PILOT_PROTOCOL, gates
G1–G6); without it no statement about distracted-driving detection, temporal robustness or alert
quality can be supported. No citation, result or significance claim beyond the measured image-level
numbers is made here.
