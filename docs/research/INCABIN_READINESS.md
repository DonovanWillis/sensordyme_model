# In-cabin collection readiness: checklist, consent workflow, transfer

**Status:** READY TO REVIEW. Nothing has been recorded.
- The protocol is `INCABIN_PILOT_PROTOCOL.md`.
- The label vocabulary is `evaluation/schemas/incabin_v1.json`, enforced by `evaluation/annotations.py` (`known_types`).
- Hardware calibration and validation live in the hardware package (`docs/HAILO_HARDWARE_VALIDATION.md`, branch `feat/m5-hw-validation`).

## Gates (all required before the first participant)

| # | Gate | Owner | Status |
|---|---|---|---|
| G1 | Counsel review of the consent form (below) for participants' jurisdictions (BIPA/CUBI/CCPA; GDPR if any EU/UK participant) | Founder + counsel | open |
| G2 | Written approval of the protocol, including stationary-vehicle-only scenarios | Founder | open |
| G3 | Pi camera use approved (it touches the device). The recording uses `rpicam-vid` only; SensorDyme software is not changed | Founder | open |
| G4 | Encryption at rest verified on the destination worker volume (BitLocker on the Windows host backing `$SDJOB_DATA`) | Fleet owner | open; nothing assumed |
| G5 | Transfer route approved (below) | Founder | open |
| G6 | Annotators named. A data processing agreement is needed if anyone outside the company labels | Founder | open |
| G7 | Camera intrinsics captured (checkerboard) for each mount used | Operator | ready (procedure in the hardware package) |

## Consent workflow

1. **Invitation.** Send the plain-language information sheet at least 24 h before the session. No session on the day of first contact.
2. **At the session:**
   - Read the form together and answer questions.
   - The participant ticks each opt-in separately (all default to NO): (a) evaluation, (b) training, (c) named external annotators, (d) stills in documents.
   - The participant signs and gets a copy.
3. **Pseudonym.** Assign it at signing (`S01`, `S02`, ...). The pseudonym-to-name key goes on paper or in an offline file held by the data protection lead, **never** on the worker or in git.
4. **Per-clip scope.** Every clip's metadata carries the opt-in flags. The manifest builder refuses training use without (b).
5. **Withdrawal (any time):**
   - Email or tell the operator.
   - Within 30 days: delete raw clips, crops, annotations and caches for that pseudonym; rebuild the manifests; confirm in writing.
   - Models already trained are disclosed as not retractable.
6. **Retention.** The manifest's `consent.retention_until` is the earliest of: the end of the pilot analysis + 12 months, or 3 years. `evaluation.manifest.validate` refuses expired data.

### Consent form: draft headings for counsel (not legal text)

- **Who:** SensorDyme (controller), the data protection lead's contact.
- **What and why:** develop and evaluate in-vehicle monitoring that SensorDyme sells; which behaviours are staged; that the vehicle is parked.
- **What is recorded:** visible and infrared video of face, upper body and hands. No audio. Optional self-reported demographics, stored separately.
- **Biometric notice:** face landmarks and geometry are computed. Purpose, and the retention schedule.
- **Uses, each optional:** (a) evaluation; (b) training; (c) named external annotators under a DPA; (d) stills in documents (default no).
- **Storage and security:** encrypted company compute storage; access limited to named staff; no cloud, no publication.
- **Risks:** fatigue; discomfort from IR light (eye-safe distance stated); stop any time without penalty; compensation is not conditional.
- **Rights:** access, deletion, withdrawal; how to exercise them; the limitation for models already trained.
- **Signature, date, the participant's copy.**

## Session checklist (operator)

**Before:**
- [ ] Gates G1–G6 closed; the signed form is in hand.
- [ ] The Pi clock is synced (chrony): `timedatectl` shows synchronized.
- [ ] Camera Module 3 **Wide NoIR** detected: `rpicam-hello --list-cameras` shows imx708_wide_noir.
- [ ] Encrypted SSD mounted; free space ≥ 50 GB.
- [ ] Mounts fixed (dash centre, plus mirror base if two cameras). Mount ID and angles photographed **without the participant**.
- [ ] IR illuminator at its rated distance; power logged.
- [ ] Prop phone blank (no personal content); props laid out.
- [ ] Vehicle parked, parking brake on, engine off or accessory power.

**During, per condition (day, then night IR):**
- [ ] Checkerboard sequence for each mount: 15 poses, filling the frame corners.
- [ ] Clapperboard: a stopwatch shown to every camera at the start.
- [ ] Run the script blocks in order (protocol §4). The cue logger is tapped at each start and end.
- [ ] After each block, check the clip plays and its size is non-zero.
- [ ] Clapperboard at the end.

**After:**
- [ ] `sha256sum` every clip into `SHA256SUMS` on the SSD.
- [ ] Session metadata CSV: pseudonym, clip, condition, mount, illuminator, eyewear, opt-ins, operator, start and end UTC.
- [ ] SSD unmounted and stored locked. Nothing copied to laptops or phones.

## Secure transfer (proposal; needs G4 and G5)

- **Option A (preferred):** attach the encrypted SSD to the worker host. Copy into `$SDJOB_DATA/incabin-pilot/<session>/`, run `sha256sum -c SHA256SUMS` there, then wipe the SSD copy after verification and a backup decision.
- **Option B:** stream from the Pi to the worker over SSH without the MacBook storing anything:

  ```
  ssh pi 'tar cf - <session>' | ssh <worker-admin> 'tar xf - -C <dest>'
  ```

  Then verify the checksums on the worker.
- **Then, on the worker:**
  - Build the manifest with `licence.class: own_collection`, `consent.protocol: <form version>`, `consent.retention_until: <date>`, and subjects and sessions as pseudonyms.
  - `python -m evaluation.manifest verify --near-duplicates`.
- **Annotation exports** come back as JSON per clip and are validated against `incabin_v1`. Only reports and metrics ever reach the Mac.

## What the first pilot can and cannot show

**Can:**
- head-pose error against known angles (rig);
- whether the glance and phone rules fire on staged behaviour;
- IR-versus-day failure modes;
- label definitions and annotator agreement.

**Cannot:** population-level accuracy. 3–5 people is a debugging set. Per-subject results only.
