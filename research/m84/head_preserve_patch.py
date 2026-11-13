"""EXP-09: make DAMO-YOLO's fine-tune loader keep the pretrained head (pinned 319572e).

    python research/m84/head_preserve_patch.py <DAMO-YOLO source dir>

Why. Stock `Detector.load_pretrain_detector` (damo/detectors/detector.py) copies every
tensor from the checkpoint EXCEPT those whose name contains 'head': the whole ZeroHead
(`gfl_cls`, the class-agnostic `gfl_reg` box-distribution convs and the per-level
`scales`) keeps its fresh random init. EXP-08 therefore re-learned box regression from
4,883 images. M7/M8 hypothesised that this, not class competition, caused its person
regression (box quality fell for every class: docs/research/M8_CV_RESEARCH.md A3).

What this patch changes, only when SD_HEAD_PRESERVE=1 (otherwise the loader is stock):
  * every head tensor whose shape matches the checkpoint (gfl_reg.*, scales.*,
    integral buffers) is copied from the checkpoint;
  * gfl_cls weight/bias rows are copied row by row from the checkpoint rows listed in
    SD_HEAD_ROWS (comma separated, one per new class; -1 keeps DAMO's own init for that
    class). EXP-09: "0,67,-1" = person <- COCO person, phone <- COCO cell phone, hand new;
  * after load_state_dict it re-reads the model and checks, tensor by tensor, that every
    copied tensor/row equals the checkpoint and every -1 row does not; it writes the
    result to SD_HEAD_PRESERVE_REPORT (JSON) and raises if any check fails, so training
    can never start from a half-applied init.

It also (SD_SEED_FIX=1) restores the trainer's commented-out `set_seed(cfg.miscs.seed)`:
EXP-08's "seed 1234" was recorded in the config but never applied (stock trainer line 98).

Exact-string edits with assertions; refuses an unexpected source tree; idempotent.
"""

from __future__ import annotations

import sys
from pathlib import Path

MARK = "# sensordyme-head-preserve"
DETECTOR = "damo/detectors/detector.py"
TRAINER = "damo/apis/detector_trainer.py"

OLD_LOOP = (
    "        new_state_dict = {}\n"
    "        for k, v in self.state_dict().items():\n"
    "            k = k.replace('module.', '')\n"
    "            if 'head' in k:\n"
    "                new_state_dict[k] = self.state_dict()[k]\n"
    "                continue\n"
    "            new_state_dict[k] = state_dict[k]\n"
    "\n"
    "        self.load_state_dict(new_state_dict, strict=True)\n"
)

NEW_LOOP = (
    "        import json as _json, os as _os  " + MARK + "\n"
    "        _on = _os.environ.get('SD_HEAD_PRESERVE') == '1'\n"
    "        _rows = [int(r) for r in _os.environ.get('SD_HEAD_ROWS', '').split(',') if r.strip()]\n"
    "        _plan = {}\n"
    "        new_state_dict = {}\n"
    "        for k, v in self.state_dict().items():\n"
    "            k = k.replace('module.', '')\n"
    "            if 'head' in k:\n"
    "                cur = self.state_dict()[k]\n"
    "                src = state_dict.get(k) if _on else None\n"
    "                if src is not None and tuple(src.shape) == tuple(cur.shape):\n"
    "                    new_state_dict[k] = src.clone()\n"
    "                    _plan[k] = {'action': 'copied', 'shape': list(cur.shape)}\n"
    "                    continue\n"
    "                if (src is not None and '.gfl_cls.' in k and len(_rows) == cur.shape[0]\n"
    "                        and tuple(src.shape[1:]) == tuple(cur.shape[1:])):\n"
    "                    t = cur.clone()\n"
    "                    for i, r in enumerate(_rows):\n"
    "                        if r >= 0:\n"
    "                            t[i] = src[r]\n"
    "                    new_state_dict[k] = t\n"
    "                    _plan[k] = {'action': 'rows', 'rows': _rows, 'src_shape': list(src.shape),\n"
    "                                'shape': list(cur.shape)}\n"
    "                    continue\n"
    "                new_state_dict[k] = cur\n"
    "                _plan[k] = {'action': 'fresh_init', 'shape': list(cur.shape),\n"
    "                            'src_shape': None if src is None else list(src.shape)}\n"
    "                continue\n"
    "            new_state_dict[k] = state_dict[k]\n"
    "\n"
    "        self.load_state_dict(new_state_dict, strict=True)\n"
    "        if _on:\n"
    "            after = {kk.replace('module.', ''): vv for kk, vv in self.state_dict().items()}\n"
    "            fails = []\n"
    "            for k, p in _plan.items():\n"
    "                a, src = after[k].detach().cpu(), state_dict.get(k)\n"
    "                if p['action'] == 'copied':\n"
    "                    p['equal'] = bool(torch.equal(a, src.to(a.dtype)))\n"
    "                    if not p['equal']:\n"
    "                        fails.append(k)\n"
    "                elif p['action'] == 'rows':\n"
    "                    p['row_equal'] = [bool(torch.equal(a[i], src[r].to(a.dtype))) if r >= 0\n"
    "                                      else not bool(torch.equal(a[i], src[0].to(a.dtype)))\n"
    "                                      for i, r in enumerate(_rows)]\n"
    "                    if not all(p['row_equal']):\n"
    "                        fails.append(k)\n"
    "            reg = [k for k in _plan if '.gfl_reg.' in k]\n"
    "            if not reg or any(_plan[k]['action'] != 'copied' for k in reg):\n"
    "                fails.append('gfl_reg not fully copied')\n"
    "            cls = [k for k in _plan if '.gfl_cls.' in k]\n"
    "            if not cls or any(_plan[k]['action'] != 'rows' for k in cls):\n"
    "                fails.append('gfl_cls rows not applied')\n"
    "            rep = {'checkpoint': pretrain_model, 'rows': _rows, 'tensors': _plan, 'failures': fails,\n"
    "                   'ok': not fails}\n"
    "            out = _os.environ.get('SD_HEAD_PRESERVE_REPORT')\n"
    "            if out:\n"
    "                with open(out, 'w') as f:\n"
    "                    _json.dump(rep, f, indent=1)\n"
    "            logger.info('SensorDyme head-preserve: {} head tensors, failures: {}'.format(len(_plan), fails))\n"
    "            if fails:\n"
    "                raise RuntimeError('head-preserving init failed its checks: {}'.format(fails))\n"
)

OLD_SEED = "        # set_seed(cfg.miscs.seed)\n"
NEW_SEED = (
    "        if os.environ.get('SD_SEED_FIX') == '1':  " + MARK + "\n"
    "            set_seed(cfg.miscs.seed)\n"
)


def _edit(root: Path, rel: str, old: str, new: str) -> str:
    path = root / rel
    text = path.read_text()
    if MARK in text:
        return f"{rel}: already patched"
    if text.count(old) != 1:
        raise SystemExit(f"{rel}: expected exactly one match, found {text.count(old)}: {old[:70]!r}. "
                         "Refusing to patch an unexpected source tree.")
    path.write_text(text.replace(old, new))
    return f"{rel}: 1 edit"


def apply(root: Path) -> list[str]:
    return [_edit(root, DETECTOR, OLD_LOOP, NEW_LOOP), _edit(root, TRAINER, OLD_SEED, NEW_SEED)]


if __name__ == "__main__":
    for line in apply(Path(sys.argv[1])):
        print(line)
