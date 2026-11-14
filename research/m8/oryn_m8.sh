#!/usr/bin/env bash
# M8 CV on oryn, CPU only, no training: the M8 analyses (research/m8/analysis.py) on the val
# predictions that Missions 5-7 already wrote to $SDJOB_DATA. Checkpoints are not loaded.
#   sdrun submit -t oryn ... --no-gpu -- bash research/m8/oryn_m8.sh
set -u
D="$SDJOB_DATA"; O="$SDJOB_OUT"
P="$D/venvs/damo-pilot/bin/python"; export PYTHONDONTWRITEBYTECODE=1
OI="$D/openimages/oi7-sd-v1"; M="$D/m7_detector"
PRE="$D/openimages/bench_preds/m5.t.reference@native_val.jsonl"   # shipping weights, reference decoder
LEG="$D/openimages/bench_preds/m5.t.legacy@native_val.jsonl"      # shipping weights, shipping decoder
F60="$M/preds_full60_nms05.jsonl"                                # EXP-08 best, re-suppressed at NMS 0.5
F60RAW="$D/exp08_full60/preds_val.jsonl"                         # EXP-08 best, DAMO postprocess (NMS 0.7)
ls -l "$PRE" "$LEG" "$F60" "$F60RAW" || exit 2
sha256sum "$PRE" "$LEG" "$F60" "$F60RAW" | cut -c1-16
A="$P -m research.m8.analysis"
$A ops --data "$OI" --split val --out "$O" --tag oryn --preds "pretrained_reference=$PRE" --preds "pretrained_legacy=$LEG" \
  --preds "exp08_nms05=$F60" --preds "exp08_nms07=$F60RAW"; echo "ops rc=$?"
$A paired --data "$OI" --split val --out "$O" --tag pre_vs_exp08 --a "pretrained_reference=$PRE" --b "exp08_nms05=$F60" --n 500; echo "paired rc=$?"
$A paired --data "$OI" --split val --out "$O" --tag exp08_vs_legacy --a "exp08_nms05=$F60" --b "pretrained_legacy=$LEG" --n 500; echo "paired2 rc=$?"
# Phone interaction with realistic (predicted) hands from the 3-class model.
$A phone --data "$OI" --split val --out "$O" --tag exp08_own_hands --preds "exp08_nms05=$F60" --hands "exp08_nms05=$F60" --hand-thr 0.3; echo "phone1 rc=$?"
$A phone --data "$OI" --split val --out "$O" --tag exp08_own_hands_h05 --preds "exp08_nms05=$F60" --hands "exp08_nms05=$F60" --hand-thr 0.5; echo "phone2 rc=$?"
# Architecture B: pretrained person (reference), EXP-08 phone and hands.
$A phone --data "$OI" --split val --out "$O" --tag archB --preds "exp08_nms05=$F60" --hands "exp08_nms05=$F60" --persons "pretrained_reference=$PRE"; echo "phone3 rc=$?"
# Shipping config with an EXP-08 hand gate (architecture B used only as a hand source).
$A phone --data "$OI" --split val --out "$O" --tag legacy_with_exp08_hands --preds "pretrained_legacy=$LEG" --hands "exp08_nms05=$F60"; echo "phone4 rc=$?"
ls -l "$O"
