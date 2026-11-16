#!/usr/bin/env bash
# M10.1 on oryn, CPU only, read-only: the parity gate with float tensors in place of INT8
# (research/m101/parity_gate_selftest.py). Run after oryn_known_good.sh (it makes the heads cut).
#   sdrun submit -t oryn --no-gpu --cpus 4 --mem 4G -- bash research/m101/oryn_parity_selftest.sh
set -u
D="$SDJOB_DATA"; O="$SDJOB_OUT"; export PYTHONDONTWRITEBYTECODE=1
cd "$SDJOB_SRC" || exit 2
P=""
for cand in "$D/venvs/damo-pilot/bin/python" "$D/venvs/sensordyme-a779b4d/bin/python" python; do
  if "$cand" -c "import onnx, onnxruntime, cv2, PIL, yaml, numpy" 2>/dev/null; then P="$cand"; break; fi
done
[ -n "$P" ] || { echo "no python with onnx+onnxruntime+cv2+PIL+yaml"; exit 3; }
H="$D/m101/zoo.heads.onnx"
[ -s "$H" ] || { echo "$H missing: run research/m101/oryn_known_good.sh first"; exit 3; }
sha256sum "$H" | tee "$O/inputs.sha256"
"$P" -u -m research.m101.parity_gate_selftest --heads-onnx "$H" --out "$O" 2>&1 | grep -v -E 'onnxruntime:Default|Warning'
rc="${PIPESTATUS[0]}"
rm -rf "$O/bundle_float"  # tensors stay on the worker; only the report comes back
exit "$rc"
