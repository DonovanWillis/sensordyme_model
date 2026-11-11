"""Add early stopping, a best checkpoint and an epoch log to DAMO-YOLO's trainer (pinned 319572e).

    python research/pilot/early_stop_patch.py <DAMO-YOLO source dir>

Stock DAMO-YOLO evaluates on val every `eval_interval_epochs` but keeps no best
checkpoint and never stops early. After each evaluation this patch:
  * records the val metrics (COCO AP, AP50, ...) to <workdir>/<exp>/sd_epochs.jsonl,
    one line per evaluation, appended atomically per line;
  * saves best_ckpt.pth when val AP improves;
  * stops training when AP has not improved for SD_EARLY_STOP_PATIENCE evaluations
    (0 or unset = never stop early);
  * stops training when the next epoch would end after SD_TRAIN_DEADLINE (unix time;
    unset = no deadline), so a run inside a hard job timeout ends at an epoch boundary
    with its checkpoints and scoring intact instead of being killed mid-epoch;
  * on resume, restores best AP and the patience counter from sd_epochs.jsonl, so
    a resumed run neither forgets its best epoch nor restarts its patience.
Exact-string edits with assertions; refuses an unexpected source tree; idempotent.
"""

# ruff: noqa: ISC004 - the edit table pairs (old, new) source fragments built from adjacent literals

from __future__ import annotations

import sys
from pathlib import Path

MARK = "# sensordyme-early-stop"
FILE = "damo/apis/detector_trainer.py"

EDITS = [
    (
        "            inference(\n                evalmodel,\n                data_loader_val,\n                dataset_name,\n"
        "                device=self.device,\n                output_folder=output_folder,\n            )\n",
        "            _sd_out = inference(  " + MARK + "\n                evalmodel,\n                data_loader_val,\n"
        "                dataset_name,\n                device=self.device,\n                output_folder=output_folder,\n"
        "            )\n"
        "            try:\n"
        "                self._sd_last_metrics = {k: float(v) for k, v in _sd_out[0].results['bbox'].items()}\n"
        "            except Exception:  # noqa: BLE001 - a missing metric must not stop training\n"
        "                self._sd_last_metrics = None\n",
    ),
    (
        "                self.evaluate(local_rank, self.cfg.dataset.val_ann)\n                self.model.train()\n",
        "                self.evaluate(local_rank, self.cfg.dataset.val_ann)\n                self.model.train()\n"
        "                if self._sd_after_eval(local_rank):  " + MARK + "\n"
        "                    break\n",
    ),
    (
        "    def save_ckpt(self, ckpt_name, local_rank, update_best_ckpt=False):\n",
        "    def _sd_after_eval(self, local_rank):  " + MARK + "\n"
        "        import json as _json\n"
        "        log_path = os.path.join(self.file_name, 'sd_epochs.jsonl')\n"
        "        if not hasattr(self, '_sd_best_ap'):\n"
        "            self._sd_best_ap, self._sd_best_epoch, self._sd_bad = -1.0, None, 0\n"
        "            if os.path.exists(log_path):  # resumed run: keep its best and its patience\n"
        "                with open(log_path) as f:\n"
        "                    for line in f:\n"
        "                        try:\n"
        "                            rec = _json.loads(line)\n"
        "                        except ValueError:\n"
        "                            continue\n"
        "                        self._sd_best_ap = rec.get('best_ap', self._sd_best_ap)\n"
        "                        self._sd_best_epoch = rec.get('best_epoch', self._sd_best_epoch)\n"
        "                        self._sd_bad = rec.get('bad_evals', self._sd_bad)\n"
        "        metrics = getattr(self, '_sd_last_metrics', None) or {}\n"
        "        ap = float(metrics.get('AP', -1.0))\n"
        "        epoch = self.epoch + 1\n"
        "        if ap > self._sd_best_ap:\n"
        "            self._sd_best_ap, self._sd_best_epoch, self._sd_bad = ap, epoch, 0\n"
        "            self.save_ckpt('best', local_rank=local_rank)\n"
        "        else:\n"
        "            self._sd_bad += 1\n"
        "        patience = int(os.environ.get('SD_EARLY_STOP_PATIENCE', '0') or 0)\n"
        "        stop = bool(patience) and self._sd_bad >= patience\n"
        "        reason = 'patience' if stop else None\n"
        "        now, prev = time.time(), getattr(self, '_sd_prev_eval_t', None)\n"
        "        self._sd_prev_eval_t = now\n"
        "        deadline = float(os.environ.get('SD_TRAIN_DEADLINE', '0') or 0)\n"
        "        if not stop and deadline and now + 1.1 * (now - prev if prev else 0.0) > deadline:\n"
        "            stop, reason = True, 'deadline'\n"
        "        if local_rank == 0:\n"
        "            rec = {'epoch': epoch, 'metrics': metrics, 'best_ap': self._sd_best_ap,\n"
        "                   'best_epoch': self._sd_best_epoch, 'bad_evals': self._sd_bad,\n"
        "                   'early_stop': stop, 'stop_reason': reason, 'time': now}\n"
        "            os.makedirs(self.file_name, exist_ok=True)\n"
        "            with open(log_path, 'a') as f:\n"
        "                f.write(_json.dumps(rec) + '\\n')\n"
        "        if stop:\n"
        "            logger.info('SensorDyme early stop ({}): patience {}, best val AP {:.4f} at epoch {}'.format(\n"
        "                reason, patience, self._sd_best_ap, self._sd_best_epoch))\n"
        "        return stop\n\n"
        "    def save_ckpt(self, ckpt_name, local_rank, update_best_ckpt=False):\n",
    ),
]


def apply(root: Path) -> str:
    path = root / FILE
    text = path.read_text()
    if MARK in text:
        return f"{FILE}: already patched"
    for old, new in EDITS:
        if text.count(old) != 1:
            raise SystemExit(f"{FILE}: expected exactly one match, found {text.count(old)}: {old[:70]!r}. "
                             "Refusing to patch an unexpected source tree.")
        text = text.replace(old, new)
    path.write_text(text)
    return f"{FILE}: {len(EDITS)} edits"


if __name__ == "__main__":
    print(apply(Path(sys.argv[1])))
