#!/usr/bin/env bash
# M10 Task 3 (a)+(b) on oryn, CPU only, val split only, no training:
#   1. every candidate through the device path (research/m10/candidates_val.py): shipping zoo heads
#      (legacy/reference) and the EXP-08 device-layout export with the M10 class map {80: hand};
#   2. the A / B / C table at precision-0.80 operating points + paired group bootstraps
#      (research/m10/compare.py), including the stored M5 / M7 prediction files as reproduction checks.
#   sdrun submit -t oryn --no-gpu --cpus 4 --mem 4G -- bash research/m10/oryn_m10.sh [LIMIT]
# A LIMIT argument makes it a smoke run (its numbers are coverage artefacts and must not be cited).
set -u
D="$SDJOB_DATA"; O="$SDJOB_OUT"; SRC="$SDJOB_SRC"
P="$D/venvs/sensordyme-a779b4d/bin/python"; export PYTHONDONTWRITEBYTECODE=1
OI="$D/openimages/oi7-sd-v1"; PR="$D/m10/$SDJOB_ID"; mkdir -p "$PR"
ZH="$D/models/damoyolo_tinynasL20_T.heads.onnx"          # zoo cut used by M8.3 (de27bd7c...)
H81="$D/m84/export_smoke/exp08.heads81.onnx"              # M8.4 device-layout export (a0327a25...)
M5L="$D/openimages/bench_preds/m5.t.legacy@native_val.jsonl"
M5R="$D/openimages/bench_preds/m5.t.reference@native_val.jsonl"
E08="$D/m7_detector/preds_full60_nms05.jsonl"            # EXP-08 via DAMO's own pipeline, re-NMS 0.5 (M7)
cd "$SRC" || exit 2
ls -l "$ZH" "$H81" "$M5L" "$M5R" "$E08" || exit 2
sha256sum "$ZH" "$H81" "$M5L" "$M5R" "$E08" | tee "$O/inputs.sha256"
LIM=()
[ -n "${1:-}" ] && LIM=(--limit "$1")
"$P" -u -m research.m10.candidates_val --data "$OI" --zoo-heads "$ZH" --exp08-heads81 "$H81" \
  --preds "$PR" --out "$O" --threads 4 "${LIM[@]}" 2>&1 | grep -v -E 'onnxruntime:Default|Warning'
echo "candidates rc=${PIPESTATUS[0]}"
sha256sum "$PR"/*.jsonl | tee "$O/preds.sha256"
N=1000; [ -n "${1:-}" ] && N=20
"$P" -u -m research.m10.compare --data "$OI" --out "$O" --n "$N" \
  --preds "C_legacy=$PR/zoo_legacy_val.jsonl" --preds "C_reference=$PR/zoo_reference_val.jsonl" \
  --preds "A_exp08_reference=$PR/exp08_reference_val.jsonl" --preds "A_exp08_legacy=$PR/exp08_legacy_val.jsonl" \
  --preds "A_exp08_reference_stretch=$PR/exp08_reference_stretch_val.jsonl" \
  --preds "A_exp08_damo_nms05_M7=$E08" --preds "C_legacy_M5=$M5L" --preds "C_reference_M5=$M5R" \
  --fuse "B=C_reference:person+A_exp08_reference:phone,hand" \
  --fuse "B_legacy_person=C_legacy:person+A_exp08_reference:phone,hand" \
  --pair "B:A_exp08_reference:person" --pair "A_exp08_reference:C_legacy:person" \
  --pair "C_reference:C_legacy:person" --pair "B:C_legacy:person" \
  --pair "A_exp08_reference:C_reference:phone" --pair "A_exp08_reference:C_legacy:phone" \
  --pair "A_exp08_reference:A_exp08_damo_nms05_M7:person" --pair "A_exp08_reference:A_exp08_damo_nms05_M7:phone" \
  --pair "A_exp08_reference:A_exp08_damo_nms05_M7:hand" \
  --pair "A_exp08_reference:A_exp08_reference_stretch:hand" --pair "A_exp08_reference:A_exp08_legacy:hand"
echo "compare rc=$?"
ls -l "$O"
