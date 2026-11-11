"""The early-stop method the EXP-08 patch injects into DAMO-YOLO's trainer, run on a stub trainer."""

from __future__ import annotations

import json
import types

import pytest

from research.pilot import early_stop_patch as esp

_SAVE_DEF = "    def save_ckpt(self, ckpt_name, local_rank, update_best_ckpt=False):\n"


def _trainer_class():
    method = esp.EDITS[2][1].replace(_SAVE_DEF, "")
    src = ("import os, time\n"
           "logger = types.SimpleNamespace(info=lambda *a: None)\n"
           "class Trainer:\n"
           "    def __init__(self, file_name):\n"
           "        self.file_name, self.epoch, self.saved = file_name, 0, []\n"
           + method + _SAVE_DEF + "        self.saved.append((ckpt_name, self.epoch + 1))\n")
    ns = {"types": types}
    exec(compile(src, "patched_trainer", "exec"), ns)  # noqa: S102 - our own patch text
    return ns["Trainer"]


def _run(trainer, aps):
    stops = []
    for ap in aps:
        trainer._sd_last_metrics = {"AP": ap, "AP50": ap + 0.2}
        stops.append(trainer._sd_after_eval(local_rank=0))
        trainer.epoch += 1
        if stops[-1]:
            break
    return stops


def test_patience_stops_after_n_evaluations_without_improvement(tmp_path, monkeypatch):
    monkeypatch.setenv("SD_EARLY_STOP_PATIENCE", "3")
    monkeypatch.delenv("SD_TRAIN_DEADLINE", raising=False)
    t = _trainer_class()(str(tmp_path))
    stops = _run(t, [0.10, 0.20, 0.19, 0.20, 0.18, 0.30])
    assert stops == [False, False, False, False, True], "an equal AP is not an improvement"
    assert t.saved == [("best", 1), ("best", 2)]
    log = [json.loads(line) for line in (tmp_path / "sd_epochs.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in log] == [1, 2, 3, 4, 5]
    assert log[-1]["best_epoch"] == 2 and log[-1]["stop_reason"] == "patience"
    assert log[0]["metrics"]["AP50"] == pytest.approx(0.3)


def test_patience_zero_never_stops(tmp_path, monkeypatch):
    monkeypatch.delenv("SD_EARLY_STOP_PATIENCE", raising=False)
    monkeypatch.delenv("SD_TRAIN_DEADLINE", raising=False)
    t = _trainer_class()(str(tmp_path))
    assert not any(_run(t, [0.3] + [0.1] * 20))


def test_a_resumed_run_keeps_its_best_and_its_patience(tmp_path, monkeypatch):
    monkeypatch.setenv("SD_EARLY_STOP_PATIENCE", "3")
    monkeypatch.delenv("SD_TRAIN_DEADLINE", raising=False)
    cls = _trainer_class()
    first = cls(str(tmp_path))
    _run(first, [0.25, 0.20, 0.21])  # best 0.25 at epoch 1, two bad evaluations
    resumed = cls(str(tmp_path))
    resumed.epoch = 3
    assert _run(resumed, [0.24]) == [True], "the third bad evaluation spans the resume"
    assert resumed.saved == [] and resumed._sd_best_epoch == 1


def test_a_missing_metric_never_counts_as_best(tmp_path, monkeypatch):
    monkeypatch.setenv("SD_EARLY_STOP_PATIENCE", "2")
    monkeypatch.delenv("SD_TRAIN_DEADLINE", raising=False)
    t = _trainer_class()(str(tmp_path))
    t._sd_last_metrics = None
    t._sd_after_eval(0)
    t.epoch += 1
    assert _run(t, [0.1]) == [False] and t.saved == [("best", 2)]


def test_the_deadline_stops_before_an_epoch_that_would_not_finish(tmp_path, monkeypatch):
    monkeypatch.delenv("SD_EARLY_STOP_PATIENCE", raising=False)
    clock = iter([1000.0, 1100.0, 1200.0])
    t = _trainer_class()(str(tmp_path))
    fake_time = types.SimpleNamespace(time=lambda: next(clock))
    t._sd_after_eval.__func__.__globals__["time"] = fake_time
    monkeypatch.setenv("SD_TRAIN_DEADLINE", "1305")  # 100 s epochs: 1200 + 110 > 1305
    assert _run(t, [0.1, 0.2, 0.3]) == [False, False, True]
    assert json.loads((tmp_path / "sd_epochs.jsonl").read_text().splitlines()[-1])["stop_reason"] == "deadline"


def test_the_patch_refuses_an_unexpected_tree(tmp_path):
    target = tmp_path / esp.FILE
    target.parent.mkdir(parents=True)
    target.write_text("class Trainer:\n    pass\n")
    with pytest.raises(SystemExit, match="Refusing"):
        esp.apply(tmp_path)
