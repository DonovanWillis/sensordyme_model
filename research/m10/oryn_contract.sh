#!/usr/bin/env bash
# M10 Task 2 on oryn, CPU only: the export / deployment contract (research/m10/export_contract.py) for the
# shipping zoo ONNX and the EXP-08 device-layout export, with the DAMO tree's config and the swap-R/B
# sensitivity of EXP-08 on the full val split. Run after research/m10/oryn_latency.sh (it makes the zoo cut).
#   sdrun submit -t oryn --no-gpu --cpus 4 --mem 4G -- bash research/m10/oryn_contract.sh
set -u
D="$SDJOB_DATA"; O="$SDJOB_OUT"; export PYTHONDONTWRITEBYTECODE=1
C=319572eef367340267ab6ab8ae253527a71d7c3a
cd "$SDJOB_SRC" || exit 2
P=""
for cand in "$D/venvs/damo-pilot/bin/python" "$D/venvs/sensordyme-a779b4d/bin/python" python; do
  if "$cand" -c "import onnx, onnxruntime, cv2, PIL, yaml, numpy" 2>/dev/null; then P="$cand"; break; fi
  echo "skip $cand: $("$cand" -c 'import onnx, onnxruntime, cv2, PIL, yaml' 2>&1 | tail -1)"
done
[ -n "$P" ] || { echo "no python with onnx+onnxruntime+cv2+PIL+yaml"; exit 3; }
echo "python: $P"
ARGS=(--onnx "zoo=$D/models/damoyolo_tinynasL20_T.onnx" --onnx "exp08=$D/m84/export_smoke/exp08.onnx"
      --onnx "exp08_heads81=$D/m84/export_smoke/exp08.heads81.onnx")
[ -s "$D/m10/latency/zoo.heads.extracted.onnx" ] && ARGS+=(--onnx "zoo_heads=$D/m10/latency/zoo.heads.extracted.onnx")
"$P" -u -m research.m10.export_contract --out "$O" "${ARGS[@]}" \
  --damo "$D/src/exp08_full60-DAMO-YOLO-$C" --damo-config configs/exp08_full60.py \
  --data "$D/openimages/oi7-sd-v1" --colour-images "${1:-1585}" --threads 4 2>&1 | grep -v -E 'onnxruntime:Default|Warning'
exit "${PIPESTATUS[0]}"
