import fcntl
import importlib.util
import os
from pathlib import Path

import pytest
import torch

from cvlab import tiled_search
from cvlab.common import digest
from test_tiled_inference import SimpleDetector, competition


@pytest.mark.parametrize("full_score,improved", [(0.3, True), (0.05, False)])
def test_search_uses_full_holdout_and_real_tiled_export(
    tmp_path, monkeypatch, full_score, improved
):
    from cvlab import engine, metrics, prepare, submit_cpu

    config, records = competition(tmp_path)
    for record in records:
        record["sequence_id"] = "heldout"
    config["runtime"]["device"] = "cpu"
    config["model"]["name"] = "rtdetr"
    config["evaluation"] = {"max_detections": [100, 500]}
    frozen = tmp_path / "score_improvement/frozen"
    frozen.mkdir(parents=True)
    monkeypatch.setattr(
        tiled_search,
        "freeze_run",
        lambda *a: (frozen, config, "weights-sha", {"train"}),
    )
    monkeypatch.setattr(engine, "setup", lambda config: torch.device("cpu"))
    monkeypatch.setattr(engine, "load_trained", lambda *a: SimpleDetector())
    monkeypatch.setattr(submit_cpu, "_cuda_budget", lambda *a: None)
    monkeypatch.setattr(submit_cpu, "_available_ram_mib", lambda: 8192)
    monkeypatch.setattr(prepare, "select_records", lambda config, split: records)

    def baseline(model, config, rs, device):
        return [
            {
                "image_id": r["id"],
                "category_id": 1,
                "score": 0.9,
                "bbox": [10, 10, 5, 5],
            }
            for r in rs
        ], {}

    def evaluate(rs, predictions, caps):
        value = (
            0.1
            if len(predictions) <= len(rs)
            else (0.3 if len(rs) == 1 else full_score)
        )
        return {"coco100": {"mAP": value}}

    def full_baseline(config, path, size, rs):
        return {
            "image_size": size,
            "metrics": {"coco100": {"mAP": 0.1}},
            "record_ids": [r["id"] for r in rs],
            "data_fingerprint": digest(rs),
        }

    monkeypatch.setattr(engine, "predict", baseline)
    monkeypatch.setattr(metrics, "evaluate", evaluate)
    monkeypatch.setattr(tiled_search, "full_validation", full_baseline)
    with (tmp_path / ".local_gpu_queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = tiled_search.run(
            tmp_path, ["rtdetr"], [64], [0.5], 1, 6, 0, os.dup(lock.fileno())
        )
    entry = result["models"][0]
    assert entry["subset_winner"]["strategy"] == "tiles"
    assert entry["full_candidate"]["record_ids"] == [0, 1]
    if improved:
        assert Path(entry["submission"]).is_file()
        assert "Full-validation improvement" in entry["outcome"]
    else:
        assert "submission" not in entry
        assert "did not generalize" in entry["outcome"]


def test_tiling_launcher_inherits_gpu_lease_and_rejects_duplicates(
    tmp_path, monkeypatch
):
    import sys

    script = Path(__file__).parents[1] / "scripts/start_tiled_search.py"
    spec = importlib.util.spec_from_file_location("tiled_launcher", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "runs").mkdir()
    monkeypatch.setattr(sys, "argv", ["start", "--work", str(tmp_path)])
    calls = []

    def execute(binary, args):
        fd = int(args[args.index("--lock-fd") + 1])
        assert os.get_inheritable(fd)
        assert os.fstat(fd).st_ino == (tmp_path / ".local_gpu_queue.lock").stat().st_ino
        calls.append(args)

    monkeypatch.setattr(module.os, "execv", execute)
    monkeypatch.setattr(module.os, "chdir", lambda *a: None)
    module.main()
    assert calls[0][2:5] == ["-m", "cvlab", "evaluate-tiles"]
    calls.clear()
    with (tmp_path / ".tiled_search.launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        module.main()
    assert calls == []
