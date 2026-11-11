#!/usr/bin/env bash
# EXP-08a pilot (docs/research/FINETUNE_EXP08_SPEC.md): 2 epochs of DAMO-YOLO-T, 3 classes, on oryn.
# Run through sdrun with the envelope in the spec. Writes small artifacts to $SDJOB_OUT;
# everything large stays in $SDJOB_DATA.
set -euo pipefail
O="$SDJOB_OUT"; DATA="$SDJOB_DATA"; SRC="$SDJOB_SRC"
UV=/home/sdjob/.local/bin/uv; BASE=/home/sdjob/envs/sdgpu/.venv/bin/python
export UV_PYTHON_PREFERENCE=only-managed UV_CACHE_DIR=/home/sdjob/.cache/uv PYTHONDONTWRITEBYTECODE=1
COMMIT=319572eef367340267ab6ab8ae253527a71d7c3a
CKPT_URL=https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/release_model/clean_model_0317/damoyolo_tinynasL20_T_420.pth
OI="$DATA/openimages/oi7-sd-v1"; WORK="$DATA/exp08a"; DAMO="$DATA/src/DAMO-YOLO-$COMMIT"; VENV="$DATA/venvs/damo-pilot"
mkdir -p "$WORK" "$DATA/src" "$DATA/models"

echo "== source (Apache-2.0) at $COMMIT"
[ -d "$DAMO" ] || curl -sSL --max-time 300 "https://codeload.github.com/tinyvision/DAMO-YOLO/tar.gz/$COMMIT" | tar -xz -C "$DATA/src"
# torch >= 2.6 defaults to weights_only=True; say so explicitly. If the release checkpoint needs
# full unpickling this fails loudly, by design: no arbitrary-code unpickling of a downloaded file.
sed -i "s/torch.load(pretrain_model, map_location='cpu')/torch.load(pretrain_model, map_location='cpu', weights_only=True)/" \
  "$DAMO/damo/detectors/detector.py"

echo "== venv (system site packages: the fleet's torch $("$BASE" -c 'import torch;print(torch.__version__)'))"
[ -x "$VENV/bin/python" ] || $UV venv -q -p "$BASE" "$VENV"
# sdgpu is itself a venv, so --system-site-packages would expose the bare interpreter, not
# sdgpu's torch. Layer sdgpu's site-packages underneath instead (read-only use; sdgpu unchanged).
"$BASE" -c 'import site; print(site.getsitepackages()[0])' > "$("$VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')/sdgpu.pth"
$UV pip freeze --python "$BASE" > "$O/base_freeze.txt"
$UV pip install -q --python "$VENV/bin/python" -c "$O/base_freeze.txt" \
  omegaconf loguru tqdm tabulate easydict opencv-python-headless pycocotools scikit-image pyyaml huggingface_hub safetensors setuptools
$UV pip install -q --python "$VENV/bin/python" --no-deps timm thop
$UV pip freeze --python "$VENV/bin/python" > "$O/pilot_freeze.txt"
P="$VENV/bin/python"
"$P" -c "import torch, timm, cv2, pycocotools; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'timm', timm.__version__)"

echo "== checkpoint"
[ -f "$DATA/models/damoyolo_tinynasL20_T_420.pth" ] || curl -sSL --max-time 600 -o "$DATA/models/damoyolo_tinynasL20_T_420.pth" "$CKPT_URL"
sha256sum "$DATA/models/damoyolo_tinynasL20_T_420.pth" | tee "$O/checkpoint.sha256"

echo "== data (train/val only; test is never exported)"
(cd "$SRC" && "$P" -m research.openimages.to_coco --data "$OI" --out "$WORK/coco") | tee "$O/coco_summary.json"

echo "== config"
"$P" - "$DAMO" "$OI" "$WORK" <<'PY'
import re, sys
damo, oi, work = sys.argv[1:]
cat = f"{damo}/damo/config/paths_catalog.py"
s = open(cat).read()
if "oi3_train_coco" not in s:
    s = s.replace("DATASETS = {", "DATASETS = {\n"
        f"        'oi3_train_coco': {{'img_dir': '{oi}', 'ann_file': '{work}/coco/train.json'}},\n"
        f"        'oi3_val_coco': {{'img_dir': '{oi}', 'ann_file': '{work}/coco/val.json'}},", 1)
    open(cat, "w").write(s)
c = open(f"{damo}/configs/damoyolo_tinynasL20_T.py").read()
c = c.replace("'num_classes': 80", "'num_classes': 3")
c = re.sub(r"self\.dataset\.class_names = \[.*?\]", "self.dataset.class_names = ['person', 'phone', 'hand']", c, flags=re.S)
c = c.replace("('coco_2017_train', )", "('oi3_train_coco', )").replace("('coco_2017_val', )", "('oi3_val_coco', )")
c = c.replace("self.train.batch_size = 256", "self.train.batch_size = 8")
c = c.replace("self.train.no_aug_epochs = 16", "self.train.no_aug_epochs = 0")
c = c.replace("self.train.warmup_epochs = 5", "self.train.warmup_epochs = 1")
c = c.replace("self.miscs.eval_interval_epochs = 10", "self.miscs.eval_interval_epochs = 1")
c = c.replace("self.miscs.ckpt_interval_epochs = 10", "self.miscs.ckpt_interval_epochs = 1")
c = c.replace("        # optimizer\n", f"        self.train.total_epochs = 2\n        self.train.finetune_path = '{work}/../models/damoyolo_tinynasL20_T_420.pth'\n"
              f"        self.miscs.output_dir = '{work}/workdirs'\n        self.test.batch_size = 8\n        # optimizer\n", 1)
open(f"{damo}/configs/exp08a.py", "w").write(c)
print(c[c.index("exp_name"):c.index("# augment")])
PY

echo "== train (2 epochs maximum; GPU telemetry every 5 s)"
( while true; do /usr/lib/wsl/lib/nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader >> "$O/gpu.csv" 2>/dev/null; sleep 5; done ) &
MON=$!
cd "$DAMO"
t0=$(date +%s)
set +e
PYTHONPATH="$DAMO" "$P" -m torch.distributed.run --nproc_per_node=1 --master_port 29517 tools/train.py -f configs/exp08a.py > "$O/train.log" 2>&1
rc=$?
set -e
kill $MON 2>/dev/null || true
echo "train rc=$rc in $(( $(date +%s)-t0 ))s"
grep -E "Error|error|Traceback" "$O/train.log" | tail -15 || true
grep -E "epoch|iter_time|AP" "$O/train.log" | tail -25 || true
[ "$rc" -eq 0 ] || exit $rc

echo "== checkpoint and validation scoring (val only)"
CK=$(find "$WORK/workdirs" -name 'latest_ckpt.pth' | head -1)
ls -l "$CK" | tee "$O/checkpoint_size.txt"
( while true; do /usr/lib/wsl/lib/nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader >> "$O/gpu_predict.csv" 2>/dev/null; sleep 5; done ) &
MON=$!
"$P" "$SRC/research/pilot/predict_val.py" --damo "$DAMO" --config configs/exp08a.py --ckpt "$CK" --data "$OI" --out "$WORK/preds_val.jsonl"
kill $MON 2>/dev/null || true
(cd "$SRC" && "$P" -m research.detector_bench.bench score --data "$OI" --models "$DATA/models" --preds "$WORK/preds_val.jsonl" --split val --name exp08a --out "$O")
"$P" - "$O/train.log" "$O/gpu.csv" <<'PY'
import re, statistics, sys
log = open(sys.argv[1]).read()
it = [float(x) for x in re.findall(r"iter_time:\s*([0-9.]+)", log)]
dt = [float(x) for x in re.findall(r"data_time:\s*([0-9.]+)", log)]
mem = [int(x) for x in re.findall(r"mem:\s*([0-9]+)", log)]
loss = re.findall(r"(?:total_loss|loss):\s*([0-9.]+)", log)
print({"iter_time_median_s": statistics.median(it) if it else None, "data_time_median_s": statistics.median(dt) if dt else None,
       "torch_mem_max_mb": max(mem) if mem else None, "loss_first": loss[:3], "loss_last": loss[-3:], "loss_points": len(loss)})
rows = [r.split(",") for r in open(sys.argv[2]).read().splitlines() if r.count(",") >= 3]
used = [int(r[1].split()[0]) for r in rows]; util = [int(r[2].split()[0]) for r in rows]; temp = [int(r[3]) for r in rows]
print({"gpu_samples": len(rows), "gpu_mem_used_max_mb": max(used, default=None), "gpu_util_mean": round(statistics.mean(util), 1) if util else None,
       "gpu_util_max": max(util, default=None), "gpu_temp_max": max(temp, default=None)})
PY
exit 0
