#!/usr/bin/env bash
# M13 on oryn, CPU only, read-only, val split only, no training: legacy vs reference decoder on
# identical float head tensors + the tensors for a Mac latency run (research/m13/decoder_compare.py).
#   sdrun submit -t oryn --no-gpu --cpus 4 --mem 4G --timeout 3600 -- bash research/m13/oryn_decoder_compare.sh
set -u
D="$SDJOB_DATA"; O="$SDJOB_OUT"; export PYTHONDONTWRITEBYTECODE=1
cd "$SDJOB_SRC" || exit 2
P=""
for cand in "$D/venvs/sensordyme-a779b4d/bin/python" "$D/venvs/damo-pilot/bin/python" python; do
  if "$cand" -c "import onnx, onnxruntime, cv2, PIL, yaml, numpy" 2>/dev/null; then P="$cand"; break; fi
  echo "skip $cand"
done
[ -n "$P" ] || { echo "no python with onnx+onnxruntime+cv2+PIL+yaml"; exit 3; }
echo "python: $P"
H="$D/m101/zoo.heads.onnx"
[ -s "$H" ] || { echo "missing $H (made by research/m101/oryn_known_good.sh)"; exit 3; }
sha256sum "$D/models/damoyolo_tinynasL20_T.onnx" "$H" | tee "$O/inputs.sha256"
lscpu | grep -E 'Model name|^CPU\(s\)' | tee "$O/cpu.txt"
"$P" -u -m research.m13.decoder_compare --data "$D/openimages/oi7-sd-v1" \
  --onnx "$D/models/damoyolo_tinynasL20_T.onnx" --heads-onnx "$H" --out "$O" --threads 4 --save-n 10 \
  2>&1 | grep -v -E 'onnxruntime:Default|Warning'
exit "${PIPESTATUS[0]}"
