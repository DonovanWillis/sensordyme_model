"""Reproducible real-video evaluation (Mission 4, Checkpoint B).

  manifest     what a dataset is: files, hashes, licence class, consent, splits
  annotations  what really happened in each clip: event intervals, don't-care spans
  replay       feeds a clip through a real VisionModule on the clip's own clock
  metrics      event- and frame-level scoring, confidence intervals, strata
  run          the CLI that ties them together and records provenance

Mission 10 (in-cabin, event level; docs/research/M10_EVAL_READINESS.md):
  scenarios       the scenario coverage matrix (schemas/incabin_scenarios_v1.yaml)
  annotations_v2  clip identity, scenario tags, annotator, uncertain spans; v1 migration
  agreement       double-annotation agreement (event IoU / onset, state kappa)
  leakage         subject-group splits, leakage refusals, the hash-stamped test lock
  event_metrics   alert scoring through the matrix, abstention, latency, power check
  report          Markdown + JSON report; NO DATA / INSUFFICIENT DATA, full provenance

Nothing here uploads anything. Datasets live on the compute workers
($SDJOB_DATA); the Mac only ever sees manifests and reports.
"""
