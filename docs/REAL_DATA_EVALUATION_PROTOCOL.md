# Real-data evaluation protocol (M11): from a stationary session to an accuracy claim

**Status:** ready to run on the hardware; **no real data exists yet.**
- Consent, privacy, data path and annotation rules live in `docs/research/INCABIN_PILOT_PROTOCOL.md` and `docs/research/INCABIN_READINESS.md`. Gates G1–G6 there are still **open**.
- This document adds the device side: what is measured, how labels are made independent and synchronised, and when a number may be called accuracy.

## 1. The rule

| Evidence | What it may be called | Tool |
|---|---|---|
| Simulator, `--fake`, the virtual device | SIMULATED. Never evidence | any `--fake` |
| One stationary session scored against the **cue schedule** | **diagnostics**: finds defects, never accuracy | `cabin_session.py score` → `evidence_level: diagnostic_cue_schedule`; every power-checked claim is withheld |
| Sessions scored against **independently verified labels** | **diagnostics per session**. An **accuracy estimate** once the scenario matrix's power check passes (≥ 30 subjects per claimed cell, events and hours per `incabin_scenarios_v1.yaml`) | `cabin_session.py score --labels …` → `evidence_level: verified_labels`, `accuracy_claim_permitted` |

**Why a cue is not truth:**
- A participant may not perform a cue: late, partial, eyes not fully shut, a glance instead of a hold.
- A participant may do something un-cued: an extra closure, a phone pickup.

Scored against cues, the first looks like a missed alert and the second like a false alert. Both
errors appear in `tests/test_m11_verified_labels.py`, on a session where one cue was skipped and one
closure was un-cued.

## 2. Independent labels: two supported methods

| Method | How | Clock | Blinding | Privacy |
|---|---|---|---|---|
| **Live observer** (no video) | A second person, **not** the one reading the cues, watches the participant and presses a key at the start and end of each behaviour: `cabin_session.py observe --independent` | `--clock same_host`: the observer's terminal is a second SSH session **on the Pi**, so labels and device events share one clock (uncertainty 0.15 s, which is key-press timing) | **not blinded**: the observer can hear the buzzer. Recorded, and flagged in every score | nothing recorded but timestamps and pseudonyms |
| **Video review** | A **separate**, consented camera (never the SensorDyme device) records the session. An annotator who never sees the device's alerts labels it afterwards. Export to `sensordyme.incabin.verified_labels/1` | `video_clock_overlay`: film `scripts/hw_validation/ms_clock.html` running on a screen in view at the start and end, and convert frame times to UTC. Measured `offset_s` and `uncertainty_s` ≤ 0.5 s | **blinded** if the video has no audio and the device's LEDs and screen are out of frame | the full `INCABIN_PILOT_PROTOCOL.md` regime: consent opt-ins, encrypted storage, retention, deletion by pseudonym |

**Required by the tool** (`cabin_session.load_labels`, which refuses otherwise):
- a pseudonymous labeller who is **not** the cue schedule or the operator;
- `independent_of_operator: true`;
- a method from the table;
- `clock.synchronized: true`, with an uncertainty of at most 0.5 s;
- a `consent_ref` (the consent record's id, **never a name**);
- annotation-v2 event types with valid UTC spans.

**For claims:** double annotation of 100 % of clips, with agreement reported per `incabin_v2.json` `agreement` (event IoU and F1, onset p90) and adjudication by a third person (protocol §8). One labeller is enough for diagnostics.

## 3. What is measured, and by which tool

| Measure | Source | Notes |
|---|---|---|
| Event recall, latency (labelled onset → alert) | `cabin_session.py score --labels`, per event type | latency includes the labeller's timing uncertainty, which is stated |
| Event precision; false alerts per hour | the same, per alert | exposure is labelled attentive time |
| Unknown / abstention frequency | the device's own `/api/health` per-alert availability, sampled each second by `run` | an abstention is never counted as a catch |
| Driver/passenger confusion | `passenger` preset: alerts during labelled `passenger_phone_use` and passenger closures count as false; the device's `driver_identity` in telemetry | `docs/DRIVER_ASSOCIATION.md` §8 lists the known limits |
| Phone-hand association | `phone_use` recall on labelled in-hand use, plus false `phone_use` on looks at a **mounted** phone (`phone_mounted` block) and on passenger phones | per-frame hand-phone association needs video review (states `phone_in_hand`) |
| Frame rate, processing latency, temperature, memory | `collect_evidence.py` during the session (`--minutes` = session length) and `hwv_stage.py observe` | the same run, so accuracy and load are on one record |
| Model and decoder provenance | the session records the device's `versions` and `model`; `collect_evidence.py` records the `.hef` sha256 and the decoder | a score is meaningless without them |

## 4. Session procedure (stationary only)

1. **Before:**
   - the consent form is signed and its id recorded;
   - the participant is a consenting adult;
   - the vehicle is parked with the brake on;
   - the observer and the operator are two different people.
2. **On the Pi:** `collect_evidence.py` (baseline), then `cabin_session.py plan --preset core`.
3. **Two SSH sessions on the Pi:**
   - the operator runs `cabin_session.py run …`;
   - the observer runs `cabin_session.py observe --clock same_host --independent --annotator O1 --consent-ref <id>`.
4. **Afterwards:**
   - `cabin_session.py score --labels labels_O1.json`;
   - `collect_evidence.py --minutes 1` (the post-session state);
   - share the evidence bundle's sha256.
5. **Never:**
   - drive;
   - record video with the SensorDyme device;
   - put a participant's name anywhere;
   - tune a threshold during or after a session without the owner's explicit decision and a re-run on held-out sessions.

## 5. Reporting

- Per session: diagnostics, labelled as such.
- Across sessions: `evaluation.event_metrics.summarise`, behind the power check. Report per subject with counts, and cluster-bootstrap intervals by subject.
- No pooled number appears before the minimums are met. Pilot subjects are validation-only (protocol §9).
