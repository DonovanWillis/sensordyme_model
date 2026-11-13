#!/usr/bin/env bash
# M8.4 B2 on oryn, CPU only, val only: luminance statistics of the val images, then slices and error
# anatomy for the stored predictions (pretrained reference/legacy, EXP-08 NMS 0.5, EXP-09 if present).
#   sdrun submit -t oryn --no-gpu --cpus 1 --mem 1G -- bash research/m84/oryn_b2.sh
set -u
D="$SDJOB_DATA"; O="$SDJOB_OUT"
P="$D/venvs/damo-pilot/bin/python"; export PYTHONDONTWRITEBYTECODE=1
OI="$D/openimages/oi7-sd-v1"; W="$D/m84"; mkdir -p "$W"
ST="$W/stats_val.csv"
[ -s "$ST" ] || $P -m research.m84.characterise imgstats --data "$OI" --split val --out "$ST"
cp "$ST" "$O/"
ARGS=(--preds "pretrained_reference=$D/openimages/bench_preds/m5.t.reference@native_val.jsonl"
      --preds "pretrained_legacy=$D/openimages/bench_preds/m5.t.legacy@native_val.jsonl"
      --preds "exp08_nms05=$D/m7_detector/preds_full60_nms05.jsonl")
E09="$D/exp09_headpres/preds_val_nms05.jsonl"
[ -s "$E09" ] && ARGS+=(--preds "exp09_nms05=$E09")
$P -m research.m84.characterise slices --data "$OI" --split val --stats "$ST" "${ARGS[@]}" --out "$O/b2_slices_val.json"; echo "slices rc=$?"
$P -m research.m84.characterise errors --data "$OI" --split val --stats "$ST" "${ARGS[@]}" --out "$O/b2_errors_val.json"; echo "errors rc=$?"
ls -l "$O"
