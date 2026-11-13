#!/usr/bin/env bash
# M8.4 B4 on oryn, CPU only: device-layout export + float parity fixtures + device-path val accuracy
# (research/m84/device_export.py), then host-simulated INT8 of the candidate's 3-class heads
# (research/m84/hostquant.py --heads). Val only. Never touches a .hef.
#   CAND=exp09_headpres|exp08_full60 sdrun submit -t oryn --no-gpu --cpus 4 --mem 4G -- bash research/m84/oryn_b4.sh
set -u
D="$SDJOB_DATA"; O="$SDJOB_OUT"
P="$D/venvs/damo-pilot/bin/python"; export PYTHONDONTWRITEBYTECODE=1
C=319572eef367340267ab6ab8ae253527a71d7c3a
CAND="${CAND:-exp09_headpres}"
OI="$D/openimages/oi7-sd-v1"; W="$D/m84/export_$CAND"; mkdir -p "$W" "$O/hostquant"
CK="$D/$CAND/workdirs/$CAND/best_ckpt.pth"
ls -l "$CK" || exit 2
$P -u -m research.m84.device_export --damo "$D/src/$CAND-DAMO-YOLO-$C" --config "configs/$CAND.py" --ckpt "$CK" \
  --zoo "$D/models/damoyolo_tinynasL20_T.onnx" --data "$OI" --work "$W" --out "$O" --name "$CAND" --threads 4 2>&1 \
  | grep -v -E "TracerWarning|^  (if|image_sizes)"
echo "device_export rc=${PIPESTATUS[0]}"
cp "$W/$CAND.heads81.onnx" "$O/" 2>/dev/null && sha256sum "$O/$CAND.heads81.onnx"   # ~32 MB: the compile input
$P -u -m research.m84.hostquant --data "$OI" --models "$D/models" --work "$W/hostquant" --out "$O/hostquant" \
  --heads "$W/$CAND.heads3.onnx" --labels person,phone,hand --threads 4
echo "hostquant rc=$?"
for V in float float_o8 pc_minmax pc_minmax_o8 pt_minmax_o8 pc_pct_o8; do
  $P -m research.detector_bench.bench score --data "$OI" --models "$D/models" --preds "$W/hostquant/preds_${V}_reference_val.jsonl" \
    --split val --name "hq_${CAND}_${V}_reference" --out "$O/hostquant/scores" | tail -2
done
for V in float_o8 pc_minmax_o8 pt_minmax_o8 pc_pct_o8; do
  $P -m research.m8.analysis paired --data "$OI" --split val --out "$O/hostquant/paired" --tag "float_vs_${V}" \
    --a "float=$W/hostquant/preds_float_reference_val.jsonl" --b "$V=$W/hostquant/preds_${V}_reference_val.jsonl" --n 300
done
ls -l "$O"
