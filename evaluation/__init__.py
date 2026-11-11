"""Reproducible real-video evaluation (Mission 4, Checkpoint B).

  manifest     what a dataset is: files, hashes, licence class, consent, splits
  annotations  what really happened in each clip: event intervals, don't-care spans
  replay       feeds a clip through a real VisionModule on the clip's own clock
  metrics      event- and frame-level scoring, confidence intervals, strata
  run          the CLI that ties them together and records provenance

Nothing here uploads anything. Datasets live on the compute workers
($SDJOB_DATA); the Mac only ever sees manifests and reports.
"""
