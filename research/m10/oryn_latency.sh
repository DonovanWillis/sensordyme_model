#!/usr/bin/env bash
# M10 Task 3(c) on oryn (sdgpu env: torch + onnxruntime-gpu): desktop latency of the two head graphs.
# NOT a Pi and NOT Hailo. The zoo graph is cut at the zoo's own end nodes (research/m84/device_export.cut,
# the same cut M8.4 used) so both networks are timed like for like (heads only, no post-processing).
#   sdrun submit -t oryn --gpu --vram-mb 2000 --cpus 4 -- bash research/m10/oryn_latency.sh
set -u
D="$SDJOB_DATA"; O="$SDJOB_OUT"; W="$D/m10/latency"; mkdir -p "$W"
cd "$SDJOB_SRC" || exit 2
ZC="$W/zoo.heads.extracted.onnx"
if [ ! -s "$ZC" ]; then
  python -c "
import sys; from pathlib import Path
from research.m84.device_export import cut
from research.detector_bench.models import DamoYolo
cut(Path(sys.argv[1]), list(DamoYolo.HEAD_NODES), Path(sys.argv[2]))" "$D/models/damoyolo_tinynasL20_T.onnx" "$ZC" || exit 3
fi
sha256sum "$D/models/damoyolo_tinynasL20_T.onnx" "$ZC" "$D/m84/export_smoke/exp08.heads81.onnx" | tee "$O/latency_inputs.sha256"
python -u -m research.m10.latency --onnx "zoo=$ZC" --onnx "exp08=$D/m84/export_smoke/exp08.heads81.onnx" \
  --out "$O" --providers cpu1,cpu4,cuda --runs 200 2>&1 | grep -v -E 'onnxruntime:Default|Warning'
exit "${PIPESTATUS[0]}"
