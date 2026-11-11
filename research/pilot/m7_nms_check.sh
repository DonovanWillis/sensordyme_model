#!/usr/bin/env bash
# M7 Checkpoint G follow-up (no training, CPU only): the pretrained reference predictions were
# suppressed at NMS IoU 0.5 (bench default) and the EXP-08 ones at 0.7 (DAMO postprocess). Put the
# EXP-08 best checkpoint and the merged architecture B on NMS 0.5, then score and analyse val.
#   sdrun submit -t oryn ... --no-gpu -- bash research/pilot/m7_nms_check.sh
set -u
D="$SDJOB_DATA"; O="$SDJOB_OUT"
P="$D/venvs/damo-pilot/bin/python"; export PYTHONDONTWRITEBYTECODE=1
OI="$D/openimages/oi7-sd-v1"; M="$D/m7_detector"; mkdir -p "$O/scores"
PRE="$D/openimages/bench_preds/m5.t.reference@native_val.jsonl"
"$P" -m research.pilot.person_regression renms --preds "$D/exp08_full60/preds_val.jsonl" --iou 0.5 --out "$M/preds_full60_nms05.jsonl"
"$P" -m research.pilot.person_regression merge --out "$M/preds_archB_nms05.jsonl" --take "person=$PRE" --take "phone,hand=$M/preds_full60_nms05.jsonl"
for N in full60_nms05 archB_nms05; do
  "$P" -m research.detector_bench.bench score --data "$OI" --models "$D/models" --preds "$M/preds_$N.jsonl" --split val --name "$N" --out "$O/scores" | grep -v "^ *$"
done
"$P" -m research.pilot.person_regression analyze --data "$OI" --preds "pretrained_corrected=$PRE" \
  --preds "full60_nms05=$M/preds_full60_nms05.jsonl" --preds "archB_nms05=$M/preds_archB_nms05.jsonl" --out "$O" > "$O/analyze_stdout.txt" 2>&1
echo "analyze rc=$?"
