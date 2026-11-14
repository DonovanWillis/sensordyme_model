#!/usr/bin/env bash
# EXP-09 val analysis on oryn, CPU only: re-suppress at NMS 0.5 (as EXP-08 was, M7), score under
# the Open Images protocol, paired photographer-group bootstrap vs the pretrained model (reference
# decoder) and vs EXP-08, operating points / attributes / calibration, phone gate with own hands.
# Called at the end of research/m84/exp09.sh; can be rerun alone:
#   sdrun submit -t oryn --no-gpu ... -- bash research/m84/exp09_analysis.sh
set -u
D="$SDJOB_DATA"; O="$SDJOB_OUT"; SRC="$SDJOB_SRC"
P="$D/venvs/damo-pilot/bin/python"; export PYTHONDONTWRITEBYTECODE=1
OI="$D/openimages/oi7-sd-v1"; mkdir -p "$O/exp09"; A="$O/exp09"
PRE="$D/openimages/bench_preds/m5.t.reference@native_val.jsonl"   # shipping weights, reference decoder
LEG="$D/openimages/bench_preds/m5.t.legacy@native_val.jsonl"      # shipping weights, shipping decoder
E08="$D/m7_detector/preds_full60_nms05.jsonl"                     # EXP-08 best at NMS 0.5
RAW="$D/exp09_headpres/preds_val.jsonl"; E09="$D/exp09_headpres/preds_val_nms05.jsonl"
cd "$SRC"
ls -l "$PRE" "$LEG" "$E08" "$RAW" || exit 2
"$P" -m research.pilot.person_regression renms --preds "$RAW" --iou 0.5 --out "$E09"
sha256sum "$PRE" "$LEG" "$E08" "$RAW" "$E09" | tee "$A/preds.sha256"
"$P" -m research.detector_bench.bench score --data "$OI" --models "$D/models" --preds "$E09" --split val --name exp09_nms05 --out "$A"
"$P" -m research.detector_bench.bench score --data "$OI" --models "$D/models" --preds "$RAW" --split val --name exp09_nms07 --out "$A"
M="$P -m research.m8.analysis"
$M ops --data "$OI" --split val --out "$A" --tag exp09 --preds "pretrained_reference=$PRE" --preds "pretrained_legacy=$LEG" \
  --preds "exp08_nms05=$E08" --preds "exp09_nms05=$E09"; echo "ops rc=$?"
$M paired --data "$OI" --split val --out "$A" --tag pre_vs_exp09 --a "pretrained_reference=$PRE" --b "exp09_nms05=$E09" --n 1000; echo "paired1 rc=$?"
$M paired --data "$OI" --split val --out "$A" --tag exp09_vs_exp08 --a "exp09_nms05=$E09" --b "exp08_nms05=$E08" --n 1000; echo "paired2 rc=$?"
$M paired --data "$OI" --split val --out "$A" --tag exp09_vs_legacy --a "exp09_nms05=$E09" --b "pretrained_legacy=$LEG" --n 1000; echo "paired3 rc=$?"
$M phone --data "$OI" --split val --out "$A" --tag exp09_own_hands --preds "exp09_nms05=$E09" --hands "exp09_nms05=$E09" --hand-thr 0.3; echo "phone rc=$?"
"$P" -m research.pilot.person_regression analyze --data "$OI" --preds "pretrained_reference=$PRE" \
  --preds "exp09_nms05=$E09" --preds "exp08_nms05=$E08" --out "$A" > "$A/analyze_stdout.txt" 2>&1; echo "analyze rc=$?"
ls -l "$A"
