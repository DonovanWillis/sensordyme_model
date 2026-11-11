#!/usr/bin/env bash
# M7 Checkpoint G on oryn (no training): score EXP-08 full60 epoch checkpoints on val, score
# architecture B (pretrained-corrected person+phone, 3-class hand) and run the person-regression
# analysis (research/pilot/person_regression.py). Val split only; checkpoints are read, not changed.
#   sdrun submit -t oryn ... --gpu --vram-mb 3000 -- bash research/pilot/m7_detector_analysis.sh
set -u
D="$SDJOB_DATA"; O="$SDJOB_OUT"; SRC="$SDJOB_SRC"
P="$D/venvs/damo-pilot/bin/python"; export PYTHONDONTWRITEBYTECODE=1
OI="$D/openimages/oi7-sd-v1"; W="$D/exp08_full60"; WD="$W/workdirs/exp08_full60"
R="$D/src/exp08_full60-DAMO-YOLO-319572eef367340267ab6ab8ae253527a71d7c3a"
M="$D/m7_detector"; mkdir -p "$M" "$O/scores"
PRE="$D/openimages/bench_preds/m5.t.reference@native_val.jsonl"
ls -l "$PRE" "$W/preds_val.jsonl"; ls "$WD" | grep -c "epoch_.*_ckpt.pth"
sha256sum "$WD/best_ckpt.pth" | cut -c1-16
cd "$SRC"
for E in 10 20 30 40 45 50 55 58 60; do
  CK="$WD/epoch_${E}_ckpt.pth"; [ -s "$CK" ] || { echo "missing $CK"; continue; }
  [ -s "$M/preds_e$E.jsonl" ] || "$P" research/pilot/predict_val.py --damo "$R" --config configs/exp08_full60.py --ckpt "$CK" --data "$OI" --out "$M/preds_e$E.jsonl" 2>&1 | grep predicted
  (cd "$SRC" && "$P" -m research.detector_bench.bench score --data "$OI" --models "$D/models" --preds "$M/preds_e$E.jsonl" --split val --name "e$E" --out "$O/scores") | sed "s/^/  /"
done
"$P" -m research.pilot.person_regression merge --out "$M/preds_archB.jsonl" --take "person,phone=$PRE" --take "hand=$W/preds_val.jsonl"
"$P" -m research.detector_bench.bench score --data "$OI" --models "$D/models" --preds "$M/preds_archB.jsonl" --split val --name archB --out "$O/scores"
"$P" -m research.detector_bench.bench score --data "$OI" --models "$D/models" --preds "$PRE" --split val --name pretrained_corrected --out "$O/scores"
ARGS=(--preds "pretrained_corrected=$PRE" --preds "full60_best=$W/preds_val.jsonl" --preds "archB=$M/preds_archB.jsonl")
for E in 10 20 30 40 45 50 55 58; do ARGS+=(--preds "e$E=$M/preds_e$E.jsonl"); done
"$P" -m research.pilot.person_regression analyze --data "$OI" "${ARGS[@]}" --out "$O" > "$O/analyze_stdout.txt" 2>&1; echo "analyze rc=$?"; tail -3 "$O/analyze_stdout.txt"
ls -l "$O"
