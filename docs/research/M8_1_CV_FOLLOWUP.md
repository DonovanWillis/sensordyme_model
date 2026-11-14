# M8.1 CV follow-up: the oryn analyses that M8 could not run

- **Job:** `20261112-152527-afe36f` on oryn (CPU only, 76 s, peak 288 MB). It ran `research/m8/oryn_m8.sh` at commit `bc7fabd` on predictions already stored on oryn. No checkpoint was loaded, nothing was trained, and the frozen test split was not touched.
- **Inputs** (sha256, first 16 hex chars):

  | Predictions | Hash |
  |---|---|
  | pretrained reference | `725d8b2b9a998a28` |
  | pretrained legacy | `0f55bf8dfd40ceec` |
  | EXP-08 at NMS 0.5 | `ba6b0ad1b8952867` |
  | EXP-08 at NMS 0.7 | `b585b41d7016f09f` |

- **Data:** Open Images val, manifest `b878e8429d221f3f`. Bootstrap over photographer groups, 500 resamples.
- **Artifacts:** checksums verified, copied to `~/sensordyme-audit/2026-11-10/m81-evidence/cv/`.

All numbers are image-level Open Images results. **They are not driving-event accuracy.**

## 1. Paired statistics: pretrained (reference decoder) vs EXP-08 (NMS 0.5)

| Person (4,008 boxes, 1,060 groups) | Pretrained | EXP-08 | Δ (pretrained − EXP-08) | 95% CI |
|---|---|---|---|---|
| AP50 | 0.713 | 0.686 | +0.027 | 0.015 – 0.040 |
| Recall at precision 0.80 | 0.661 | 0.627 | +0.033 | 0.018 – 0.052 |
| Precision at recall 0.70 | 0.695 | 0.548 | +0.147 | 0.067 – 0.205 |
| False positives at recall 0.70 | 1,230 | 2,312 | −1,082 | −2,210 – −389 |

The person regression is **statistically real**: every interval excludes 0, with p(Δ ≤ 0) = 0 in all 500 resamples. M8's estimate, that the 0.034 gap was "probably real", is confirmed: the measured gap is 0.033.

| Phone (128 boxes, 121 groups) | Pretrained | EXP-08 | Δ | 95% CI |
|---|---|---|---|---|
| AP50 | 0.795 | 0.807 | −0.012 | −0.114 – 0.066 |
| Recall at precision 0.80 | 0.594 | 0.789 | −0.195 (EXP-08 better) | −0.481 – 0.045 |

The phone gain at a product operating point is large but **not significant at 95%** (p = 0.064 for EXP-08 being better). There are too few phones to tell.

**EXP-08 vs the shipping legacy decoder:**
- Person recall at precision 0.80 is +0.023 (CI 0.006–0.043), so EXP-08 is better than what ships today.
- Phone AP50 is +0.079 (CI −0.010–0.187).

## 2. Large people (the driver case): recall at precision 0.80, by box attribute

| Model | threshold | large (n=2,649) | medium (712) | small (322) | occluded (1,898) | truncated (1,505) |
|---|---|---|---|---|---|---|
| pretrained, reference | 0.332 | **0.733** | 0.594 | 0.177 | 0.580 | 0.714 |
| pretrained, legacy (ships) | 0.379 | 0.702 | 0.469 | 0.034 | 0.498 | 0.677 |
| EXP-08, NMS 0.5 | 0.300 | **0.714** | 0.510 | 0.165 | 0.531 | 0.690 |

For large people, EXP-08 trails the pretrained model by **0.019**, against 0.033 overall. Most of the regression is in medium and occluded people. EXP-08 beats the shipping legacy decoder on large people by 0.012.

M8 set two conditions for adopting Arch B:
1. a large-person gap above 0.02;
2. EXP-09 failing.

The measured gap is 0.019, so condition 1 is **not met** (a point estimate; the analysis has no per-attribute CI).

## 3. Phone held vs present, with the 3-class model's own predicted hands

All results at phone threshold 0.4, val split. "In hand" means the 21 phones a human verified in a hand.

| Gate | Precision | Recall (all phones) | In-hand recall | Not-in-hand recall | Phone-free images with a false phone |
|---|---|---|---|---|---|
| none (EXP-08) | 0.783 | 0.789 | 0.810 | 0.880 | 33.3% (28 false phones) |
| predicted hand overlap (hand ≥ 0.3) | **0.950** | 0.148 | **0.810** | 0.040 | **2.4%** (1) |
| predicted hand overlap (hand ≥ 0.5) | 0.941 | 0.125 | 0.762 | 0.000 | 2.4% (1) |
| human-hand overlap (oracle) | 0.944 | 0.133 | 0.810 | 0.000 | 2.4% (1) |
| shipping legacy phones + EXP-08 hands | 0.800 | 0.188 | **0.952** | 0.040 | 7.1% (6), down from 45.2% |

- **The model's own predicted hands work as well as human hand labels as a "phone is held" gate.**
  - No in-hand phone is lost: 0.810 before and after.
  - False phones on phone-free images drop from 28 to 1.
- **Overall recall falls** because most Open Images phones are not in a hand, which is the intended effect of the gate.
- **This removes M8's main caveat.** M8's hand-gate numbers used human hand boxes as an upper bound; a deployable hand source now matches that bound on this data. The samples are small (21 in-hand phones, 42 phone-free images), so the intervals are wide.
- **A hybrid also works.** Shipping phones gated by EXP-08 hands keep 95% of in-hand phones and cut false-phone images from 45% to 7%.
- The Arch B persons change nothing for phone gating: identical rows except the person-box rule.

## 4. Is further training justified?

- **Before Monday: no.**
  - The shipping `.hef` and the `legacy` decoder stay.
  - The binding unknowns are hardware unknowns: INT8 parity and Pi CPU cost.
  - A cabin-domain gap also remains: there is no in-cabin NoIR data.
- **EXP-09 (head-preserving fine-tune, about 3.1 h on oryn): justified as the next bounded experiment, with your approval.**
  - The person regression is now proven (CI 0.018–0.052 at recall/precision 0.80).
  - Its likely cause, the re-initialised box regressor, is testable in one run.
  - Success criterion: person recall at precision 0.80 within 0.01 of pretrained, while keeping phone AP50 ≥ 0.79 and hand AP50 ≥ 0.70.
- **Arch B (two detectors): not justified.** The large-person gap (0.019) is under M8's 0.02 bar, and it would double accelerator time.
- **More 3-class training beyond EXP-09: not justified by this evidence.** The phone gain is not significant (128 phones). The next accuracy gain is likelier to come from consented in-cabin data (with the hand gate already working) than from more Open Images epochs.
- **Hailo compatibility (unchanged from M8):**
  - The 3-class model must compile with the Dataflow Compiler (x86, not on the fleet) using at least 1,024 NoIR calibration frames.
  - Its thresholds must be re-derived on INT8 scores.
  - The reference decoder stays opt-in (`experiment-only`) until the H1 INT8 parity report.
