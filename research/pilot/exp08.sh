#!/usr/bin/env bash
# EXP-08 short pilots (docs/research/FINETUNE_EXP08_SPEC.md): DAMO-YOLO-T, 3 classes, on oryn.
#   EXP=exp08a PARTIAL_LABELS=0  bash research/pilot/exp08.sh   # stock COCO semantics (the 08a pilot)
#   EXP=exp08c PARTIAL_LABELS=1  bash research/pilot/exp08.sh   # per-class ignore masks for unverified classes
# Optional: SEED (default 1234), EPOCHS (default 2; the approved pilot scope is 1-2),
#   WARMUP_EPOCHS / NO_AUG_EPOCHS (defaults 1 / 0, the pilot schedule),
#   EARLY_STOP_PATIENCE (val evaluations without AP gain before stopping; default 0 = off),
#   TRAIN_BUDGET_S (training ends at the last epoch boundary that fits this many seconds after
#   the script starts, leaving time for scoring inside the job timeout; default 0 = no budget),
#   RESUME=1 (continue $EXP from its newest epoch checkpoint instead of the release weights).
# Every run carries research/pilot/early_stop_patch.py: per-epoch val metrics in sd_epochs.jsonl,
# best_ckpt.pth on val AP, and the patience/budget stops above.
# Run through sdrun with the envelope in the spec. Small artifacts go to $SDJOB_OUT; everything
# large stays in $SDJOB_DATA.
set -euo pipefail
O="$SDJOB_OUT"; DATA="$SDJOB_DATA"; SRC="$SDJOB_SRC"
UV=/home/sdjob/.local/bin/uv; BASE=/home/sdjob/envs/sdgpu/.venv/bin/python
export UV_PYTHON_PREFERENCE=only-managed UV_CACHE_DIR=/home/sdjob/.cache/uv PYTHONDONTWRITEBYTECODE=1
COMMIT=319572eef367340267ab6ab8ae253527a71d7c3a
# The README's Alibaba OSS links are dead (NoSuchBucket, checked 2026-11-11). The same README's
# Google Drive mirror of DAMO-YOLO-T (41.8 mAP release, Apache-2.0) is used, pinned by sha256.
CKPT_URL="https://drive.usercontent.google.com/download?id=1-9NzCRKJZs3ea_n35seEYSpq3M_RkhcT&export=download&confirm=t"
CKPT_SHA=a5ee63820e0e02a8a0779704c3ba477aefd66927e2e38900bb6cec0168005f95
CKPT="$DATA/models/drive/damoyolo_tinynasL20_T_418.pth"
EXP="${EXP:-exp08a}"; PARTIAL="${PARTIAL_LABELS:-0}"; SEED="${SEED:-1234}"; EPOCHS="${EPOCHS:-2}"
export SD_UNVERIFIED_WEIGHT="${SD_UNVERIFIED_WEIGHT:-0}"
WARMUP="${WARMUP_EPOCHS:-1}"; NOAUG="${NO_AUG_EPOCHS:-0}"; RESUME="${RESUME:-0}"
export SD_EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-0}"
T_START=$(date +%s); BUDGET="${TRAIN_BUDGET_S:-0}"
# More than 2 epochs is a multi-hour campaign: it needs explicit owner approval, recorded by
# setting ALLOW_LONG_RUN=1 for that one job (docs/research/EXP08_RESULTS.md, "Proposed full run").
[ "$EPOCHS" -le 2 ] || [ "${ALLOW_LONG_RUN:-0}" = "1" ] || {
  echo "EPOCHS=$EPOCHS exceeds the approved 1-2 epoch pilot scope (set ALLOW_LONG_RUN=1 only with approval)"; exit 2; }
OI="$DATA/openimages/oi7-sd-v1"; WORK="$DATA/$EXP"; DAMO="$DATA/src/DAMO-YOLO-$COMMIT"; VENV="$DATA/venvs/damo-pilot"
echo "experiment $EXP partial_labels=$PARTIAL unverified_weight=$SD_UNVERIFIED_WEIGHT seed=$SEED epochs=$EPOCHS" \
  "warmup=$WARMUP no_aug=$NOAUG patience=$SD_EARLY_STOP_PATIENCE budget_s=$BUDGET resume=$RESUME" | tee "$O/experiment.txt"
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

echo "== checkpoint (must be the pinned zip, loadable with weights_only=True)"
mkdir -p "$(dirname "$CKPT")"
[ -s "$CKPT" ] || curl -sSL --max-time 600 -o "$CKPT" "$CKPT_URL"
echo "$CKPT_SHA  $CKPT" | sha256sum -c - | tee "$O/checkpoint.sha256"
"$P" -c "import sys, zipfile, torch; f=sys.argv[1]; assert zipfile.is_zipfile(f), 'not a checkpoint archive'; ck=torch.load(f, map_location='cpu', weights_only=True); print('checkpoint ok:', len(ck['model']), 'tensors')" "$CKPT"

# Each experiment trains from its own copy of the pinned source; the partial-label patch is
# applied only to its copy and proven by the sanity script before any training.
RUN="$DATA/src/$EXP-DAMO-YOLO-$COMMIT"
rm -rf "$RUN" && cp -a "$DAMO" "$RUN"
# Resuming reads our own checkpoint (model + optimizer state): tensors only, so weights_only=True.
sed -i "s/ckpt = torch.load(ckpt_file_path, map_location=self.device)/ckpt = torch.load(ckpt_file_path, map_location=self.device, weights_only=True)/" \
  "$RUN/damo/apis/detector_trainer.py"
grep -q "weights_only=True" "$RUN/damo/apis/detector_trainer.py"
"$P" "$SRC/research/pilot/early_stop_patch.py" "$RUN" | tee "$O/patch.txt"
RESUME_CK=""
if [ "$RESUME" = "1" ]; then
  for f in "$WORK/workdirs/$EXP"/epoch_*_ckpt.pth; do
    [ -e "$f" ] || continue; n=${f##*/epoch_}; echo "${n%_ckpt.pth} $f"
  done | sort -n | tail -1 | cut -d' ' -f2- > "$O/resume_from.txt"
  RESUME_CK=$(cat "$O/resume_from.txt")
  [ -n "$RESUME_CK" ] || { echo "RESUME=1 but no epoch checkpoint under $WORK/workdirs/$EXP"; exit 2; }
  echo "resume from $RESUME_CK" | tee -a "$O/experiment.txt"
fi
if [ "$PARTIAL" = "1" ]; then
  "$P" "$SRC/research/pilot/partial_labels_patch.py" "$RUN" | tee "$O/patch.txt"
fi
# EXP-09 (docs/research/M84_MODEL_AUDIT.md): keep the pretrained head instead of re-initialising it.
# HEAD_PRESERVE=1 copies gfl_reg/scales and the gfl_cls rows in HEAD_ROWS (default person<-0,
# phone<-67, hand new); the loader verifies the copy and refuses to train if it is not exact.
# SEED_FIX=1 (default with HEAD_PRESERVE) applies cfg.miscs.seed, which stock DAMO never does.
if [ "${HEAD_PRESERVE:-0}" = "1" ]; then
  "$P" "$SRC/research/m84/head_preserve_patch.py" "$RUN" | tee -a "$O/patch.txt"
  export SD_HEAD_PRESERVE=1 SD_HEAD_ROWS="${HEAD_ROWS:-0,67,-1}" SD_HEAD_PRESERVE_REPORT="$O/head_preserve.json"
  export SD_SEED_FIX="${SEED_FIX:-1}"
  echo "head_preserve rows=$SD_HEAD_ROWS seed_fix=$SD_SEED_FIX" | tee -a "$O/experiment.txt"
fi
DAMO="$RUN"

echo "== data (train/val only; test is never exported)"
(cd "$SRC" && "$P" -m research.openimages.to_coco --data "$OI" --out "$WORK/coco") | tee "$O/coco_summary.json"

echo "== config"
"$P" - "$DAMO" "$OI" "$WORK" "$EXP" "$SEED" "$EPOCHS" "$WARMUP" "$NOAUG" "$RESUME_CK" <<'PY'
import re, sys
damo, oi, work, exp, seed, epochs, warmup, noaug, resume_ck = sys.argv[1:]
start = (f"        self.train.finetune_path = None\n        self.train.resume_path = '{resume_ck}'\n" if resume_ck
         else f"        self.train.finetune_path = '{work}/../models/drive/damoyolo_tinynasL20_T_418.pth'\n")
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
for old, new in [("self.train.no_aug_epochs = 16", f"self.train.no_aug_epochs = {int(noaug)}"),
                 ("self.train.warmup_epochs = 5", f"self.train.warmup_epochs = {int(warmup)}"),
                 ("self.miscs.eval_interval_epochs = 10", "self.miscs.eval_interval_epochs = 1"),
                 ("self.miscs.ckpt_interval_epochs = 10", "self.miscs.ckpt_interval_epochs = 1")]:
    assert c.count(old) == 1, old
    c = c.replace(old, new)
c = c.replace("        # optimizer\n", f"        self.train.total_epochs = {epochs}\n        self.miscs.seed = {seed}\n{start}"
              f"        self.miscs.output_dir = '{work}/workdirs'\n        self.test.batch_size = 8\n        # optimizer\n", 1)
open(f"{damo}/configs/{exp}.py", "w").write(c)
print(c[c.index("exp_name"):c.index("# augment")])
PY

if [ "$PARTIAL" = "1" ]; then
  echo "== partial-label sanity (must pass before training)"
  "$P" "$SRC/research/pilot/partial_labels_sanity.py" "$DAMO" "$WORK/coco/train.json" "$OI" | tee "$O/sanity.txt"
fi

echo "== train ($EPOCHS epochs; GPU telemetry every 5 s)"
if [ "$BUDGET" -gt 0 ]; then
  export SD_TRAIN_DEADLINE=$(( T_START + BUDGET ))
  echo "training stops at the last epoch boundary before $(date -d "@$SD_TRAIN_DEADLINE" -u +%FT%TZ)"
fi
( while true; do /usr/lib/wsl/lib/nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader >> "$O/gpu.csv" 2>/dev/null; sleep 5; done ) &
MON=$!
cd "$DAMO"
t0=$(date +%s)
set +e
PYTHONPATH="$DAMO" "$P" -m torch.distributed.run --nproc_per_node=1 --master_port 29517 tools/train.py -f "configs/$EXP.py" > "$O/train.log" 2>&1
rc=$?
set -e
kill $MON 2>/dev/null || true
echo "train rc=$rc in $(( $(date +%s)-t0 ))s"
grep -E "Error|error|Traceback" "$O/train.log" | tail -15 || true
grep -E "epoch|iter_time|AP" "$O/train.log" | tail -25 || true
[ "$rc" -eq 0 ] || exit $rc
if [ "${HEAD_PRESERVE:-0}" = "1" ]; then
  "$P" -c "import json,sys; r=json.load(open(sys.argv[1])); print('head_preserve ok:', r['ok']); sys.exit(0 if r['ok'] else 4)" "$O/head_preserve.json"
fi

echo "== checkpoint and validation scoring (val only)"
EPOCH_LOG="$WORK/workdirs/$EXP/sd_epochs.jsonl"
[ -s "$EPOCH_LOG" ] && cp "$EPOCH_LOG" "$O/sd_epochs.jsonl" && tail -3 "$EPOCH_LOG"
# Score the best val-AP checkpoint (early-stop patch); latest is hashed too for the record.
CK="$WORK/workdirs/$EXP/best_ckpt.pth"
[ -s "$CK" ] || CK=$(find "$WORK/workdirs" -name 'latest_ckpt.pth' | head -1)
ls -l "$CK" | tee "$O/checkpoint_size.txt"
sha256sum "$CK" | tee "$O/output_checkpoint.sha256"
LATEST="$WORK/workdirs/$EXP/latest_ckpt.pth"
[ ! -s "$LATEST" ] || [ "$LATEST" = "$CK" ] || sha256sum "$LATEST" | tee "$O/latest_checkpoint.sha256"
( while true; do /usr/lib/wsl/lib/nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader >> "$O/gpu_predict.csv" 2>/dev/null; sleep 5; done ) &
MON=$!
"$P" "$SRC/research/pilot/predict_val.py" --damo "$DAMO" --config "configs/$EXP.py" --ckpt "$CK" --data "$OI" --out "$WORK/preds_val.jsonl"
kill $MON 2>/dev/null || true
(cd "$SRC" && "$P" -m research.detector_bench.bench score --data "$OI" --models "$DATA/models" --preds "$WORK/preds_val.jsonl" --split val --name "$EXP" --out "$O")
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
