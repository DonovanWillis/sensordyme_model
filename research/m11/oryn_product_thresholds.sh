#!/usr/bin/env bash
# M11 Phase D on oryn, CPU only, val split only, no training, no new inference:
# re-score the stored M10 device-path predictions (job 20261114-034527-625fa0) at the product's raw
# thresholds and at matched precision (research/m11/product_thresholds.py).
#   sdrun submit -t oryn --no-gpu --cpus 4 --mem 4G --timeout 1800 -- bash research/m11/oryn_product_thresholds.sh
set -u
D="$SDJOB_DATA"; O="$SDJOB_OUT"; SRC="$SDJOB_SRC"
P="$D/venvs/sensordyme-a779b4d/bin/python"; export PYTHONDONTWRITEBYTECODE=1
OI="$D/openimages/oi7-sd-v1"; PR="$D/m10/20261114-034527-625fa0"
cd "$SRC" || exit 2
ls -l "$PR/zoo_legacy_val.jsonl" "$PR/zoo_reference_val.jsonl" "$PR/exp08_reference_val.jsonl" || exit 2
sha256sum "$PR/zoo_legacy_val.jsonl" "$PR/zoo_reference_val.jsonl" "$PR/exp08_reference_val.jsonl" | tee "$O/inputs.sha256"
"$P" -u -m research.m11.product_thresholds --data "$OI" --out "$O" \
  --preds "C_legacy=$PR/zoo_legacy_val.jsonl" --preds "C_reference=$PR/zoo_reference_val.jsonl" \
  --preds "A_exp08_reference=$PR/exp08_reference_val.jsonl"
echo "rc=$?"
