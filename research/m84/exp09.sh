#!/usr/bin/env bash
# EXP-09 head-preserving fine-tune (docs/research/M84_MODEL_AUDIT.md, registry EXP-09), oryn, ONE run.
# Same data, schedule and settings as exp08_full60 (job 20261111-055506-564ff3); the only intended
# change is the start of the head: gfl_reg + scales copied from the COCO checkpoint, gfl_cls
# person <- COCO row 0, phone <- COCO row 67, hand new (research/m84/head_preserve_patch.py).
# The seed is applied (stock DAMO ignores it; EXP-08 was therefore unseeded).
#   sdrun submit -t oryn --vram-mb 4500 --cpus 6 --mem 8G --timeout 12600 -- bash research/m84/exp09.sh
# Gates before any GPU work: (1) GPU not busy with someone else's work, (2) no split leakage.
set -uo pipefail
D="$SDJOB_DATA"; O="$SDJOB_OUT"; SRC="$SDJOB_SRC"
P="$D/venvs/damo-pilot/bin/python"; export PYTHONDONTWRITEBYTECODE=1
OI="$D/openimages/oi7-sd-v1"; SMI=/usr/lib/wsl/lib/nvidia-smi
EXP=exp09_headpres

echo "== gate 1: GPU contention (3 samples over 10 s; skip if util > 30% or > 2500 MiB in use before we start)"
busy=0
for _ in 1 2 3; do
  line=$($SMI --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1)
  echo "$line" | tee -a "$O/gpu_gate.txt"
  u=$(echo "$line" | cut -d, -f1 | tr -d ' '); m=$(echo "$line" | cut -d, -f2 | tr -d ' ')
  { [ "$u" -gt 30 ] || [ "$m" -gt 2500 ]; } && busy=$((busy+1))
  sleep 5
done
if [ "$busy" -ge 2 ]; then echo "GPU busy with other work ($busy/3 samples): EXP-09 not started" | tee "$O/skipped.txt"; exit 3; fi

echo "== gate 2: split leakage (manifest ids, photographer groups, file sha256)"
(cd "$SRC" && "$P" -m research.m84.leakage_check --data "$OI" --out "$O/leakage_manifest.json") || { echo "LEAK: refusing to train"; exit 3; }

echo "== train (research/pilot/exp08.sh, exp08_full60 settings + HEAD_PRESERVE=1)"
EXP=$EXP HEAD_PRESERVE=1 ALLOW_LONG_RUN=1 EPOCHS=60 WARMUP_EPOCHS=2 NO_AUG_EPOCHS=5 EARLY_STOP_PATIENCE=10 \
  SEED=1234 SD_UNVERIFIED_WEIGHT=1 PARTIAL_LABELS=0 TRAIN_BUDGET_S="${TRAIN_BUDGET_S:-11900}" \
  bash "$SRC/research/pilot/exp08.sh"
rc=$?
echo "exp08.sh rc=$rc"
[ "$rc" -eq 0 ] || exit "$rc"

echo "== leakage of the export the trainer actually read"
(cd "$SRC" && "$P" -m research.m84.leakage_check --data "$OI" --coco "$D/$EXP/coco" --out "$O/leakage_export.json") || echo "LEAK in export (see leakage_export.json)"

echo "== analysis (CPU; val only)"
bash "$SRC/research/m84/exp09_analysis.sh"
exit 0
