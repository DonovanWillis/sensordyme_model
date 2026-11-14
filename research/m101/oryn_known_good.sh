#!/usr/bin/env bash
# M10.1 on oryn, CPU only, read-only: known-good float expectations under both decoders + the
# repeatability check of the committed legacy file (research/m101/known_good_decoders.py).
#   sdrun submit -t oryn --no-gpu --cpus 4 --mem 4G -- bash research/m101/oryn_known_good.sh
set -u
D="$SDJOB_DATA"; O="$SDJOB_OUT"; export PYTHONDONTWRITEBYTECODE=1
cd "$SDJOB_SRC" || exit 2
P=""
for cand in "$D/venvs/damo-pilot/bin/python" "$D/venvs/sensordyme-a779b4d/bin/python" python; do
  if "$cand" -c "import onnx, onnxruntime, cv2, PIL, yaml, numpy" 2>/dev/null; then P="$cand"; break; fi
  echo "skip $cand"
done
[ -n "$P" ] || { echo "no python with onnx+onnxruntime+cv2+PIL+yaml"; exit 3; }
echo "python: $P"
W="$D/m101"; mkdir -p "$W"; H="$W/zoo.heads.onnx"
if [ ! -s "$H" ]; then
  "$P" -c "
import sys; from pathlib import Path
from research.m84.device_export import cut
from research.detector_bench.models import DamoYolo
cut(Path(sys.argv[1]), list(DamoYolo.HEAD_NODES), Path(sys.argv[2]))" "$D/models/damoyolo_tinynasL20_T.onnx" "$H" || exit 3
fi
sha256sum "$D/models/damoyolo_tinynasL20_T.onnx" "$H" | tee "$O/inputs.sha256"
"$P" -u -m research.m101.known_good_decoders --onnx "$D/models/damoyolo_tinynasL20_T.onnx" --heads-onnx "$H" \
  --out "$O" --threads 4 2>&1 | grep -v -E 'onnxruntime:Default|Warning'
exit "${PIPESTATUS[0]}"
