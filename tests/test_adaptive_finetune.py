import fcntl
import importlib.util
import os
import shutil
from pathlib import Path

import pytest
import torch

from cvlab import adaptive_finetune as adaptive, engine
from cvlab.common import read_json, write_json
from test_crop_finetune import tiny_run  # noqa: F401
from test_pipeline import competition  # noqa: F401


@pytest.mark.parametrize(
    "fold,official,accepted",
    [
        (0.32, 0.21, True),
        (0.29, 0.21, False),
        (0.32, 0.19, False),
        (0.32, None, False),
        (0.3005, 0.2005, False),
    ],
)
def test_promotion_requires_both_independent_gains(fold, official, accepted):
    result = adaptive.decision(0.3, fold, 0.2, official, 1e-5)
    assert result["accepted"] is accepted
    assert result["next_lr"] == (1e-5 if accepted else 5e-6)


def test_warm_optimizer_cache_keeps_moments_and_skips_duplicate_validation(
    tiny_run,  # noqa: F811
    tmp_path,
    monkeypatch,
):  # noqa: F811
    source = Path(tiny_run["output"]) / "best.pt"
    original = torch.load(source, weights_only=False)
    config = adaptive.attempt_config(tiny_run, tmp_path / "cache_run", 1e-5)
    config["loader"]["workers"] = 0
    cache = read_json(source.parent / "best_metrics.json")
    calls = []
    original_evaluate = engine.evaluate

    def evaluate(*args):
        calls.append(1)
        return original_evaluate(*args)

    monkeypatch.setattr(engine, "evaluate", evaluate)
    engine.train(
        config,
        init_checkpoint=source,
        warm_optimizer=True,
        initial_validation_cache=cache,
        yield_after_epoch=True,
        round_batches=1,
    )
    output = Path(config["output"])
    initial = torch.load(output / "best.pt", weights_only=False)
    last = torch.load(output / "last.pt", weights_only=False)
    old_step = next(iter(original["optimizer"]["state"].values()))["step"]
    new_step = next(iter(last["optimizer"]["state"].values()))["step"]
    assert len(calls) == 1 and new_step == old_step + 1
    torch.testing.assert_close(initial["model"]["weight"], original["model"]["weight"])
    torch.testing.assert_close(
        initial["rng"]["torch"], original["rng"]["torch"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        initial["rng"]["loader"], original["rng"]["loader"], rtol=0, atol=0
    )
    assert initial["rng"]["python"] == original["rng"]["python"]
    assert initial["optimizer"]["param_groups"][1]["lr"] == 1e-5
    assert initial["initialization"]["optimizer_restored_with_new_lr_and_budget"]
    assert read_json(output / "initial_validation.json")["timing"]["reused_from_source"]


@pytest.mark.parametrize("change", ["epoch", "split", "precision"])
def test_invalid_initial_metric_reuse_is_rejected(tiny_run, tmp_path, change):  # noqa: F811
    source = Path(tiny_run["output"]) / "best.pt"
    cache = read_json(source.parent / "best_metrics.json")
    config = adaptive.attempt_config(tiny_run, tmp_path / "bad", 1e-5)
    config["loader"]["workers"] = 0
    if change == "epoch":
        cache["epoch"] += 1
    elif change == "split":
        config["data"]["split"] = "official"
    else:
        payload = torch.load(source, weights_only=False)
        payload["config"]["runtime"]["precision"] = "bf16"
        torch.save(payload, source)
    with pytest.raises(ValueError, match="cache does not match"):
        engine.train(config, init_checkpoint=source, initial_validation_cache=cache)


def test_rejected_trial_restores_champion_and_halves_lr(
    tiny_run,  # noqa: F811
    tmp_path,
    monkeypatch,
):  # noqa: F811
    from cvlab import score_improvement, validation_pipeline

    work = tmp_path / "work"
    original = work / "runs/rtdetr_fold0_local"
    original.parent.mkdir(parents=True)
    shutil.copytree(tiny_run["output"], original)
    config = read_json(original / "config.json")
    config["model"]["name"] = "rtdetr"
    write_json(original / "config.json", config)
    payload = torch.load(original / "best.pt", weights_only=False)
    payload["config"] = config
    torch.save(payload, original / "best.pt")
    original_bytes = (original / "best.pt").read_bytes()
    real_hash = score_improvement.file_hash
    monkeypatch.setattr(
        score_improvement,
        "file_hash",
        lambda path: "a7cd89fb349d" + "0" * 52
        if path == original / "best.pt"
        else real_hash(path),
    )
    real_config = adaptive.attempt_config

    def cpu_config(*args):
        candidate = real_config(*args)
        candidate["loader"]["workers"] = 0
        return candidate

    monkeypatch.setattr(adaptive, "attempt_config", cpu_config)
    scores = iter([0.35, 0.34, 0.37])
    monkeypatch.setattr(
        validation_pipeline,
        "evaluate",
        lambda *args: {"coco100": {"mAP": next(scores)}},
    )

    interrupted = {"gate": False}

    def official(work, run, sha):
        if run.name == "attempt_001" and not interrupted["gate"]:
            interrupted["gate"] = True
            raise RuntimeError("simulated confirmation interruption")
        return {"rtdetr_fold0_local": 0.2, "attempt_001": 0.21, "attempt_003": 0.23}[
            run.name
        ]

    monkeypatch.setattr(adaptive, "official_score", official)
    with (work / ".local_gpu_queue.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        with pytest.raises(RuntimeError, match="confirmation interruption"):
            adaptive.run(work, 3, 1, os.dup(lease.fileno()))
        pending = read_json(work / "adaptive_training/state.json")
        assert pending["pending"]["number"] == 1 and pending["attempts"] == []
        state = adaptive.run(work, 3, 1, os.dup(lease.fileno()))
    assert [attempt["accepted"] for attempt in state["attempts"]] == [True, False, True]
    assert state["attempts"][1]["candidate_fold"] == 0.34
    assert state["lr"] == 5e-6 and state["champion"].endswith("attempt_003")
    directory = work / "runs/rtdetr_adaptive_finetune"
    steps = []
    for i in range(1, 4):
        row = torch.load(directory / f"attempt_{i:03d}/last.pt", weights_only=False)
        steps.append(next(iter(row["optimizer"]["state"].values()))["step"].item())
    assert steps == [2, 3, 3]  # Rejected attempt's optimizer step is not inherited.
    init3 = read_json(directory / "attempt_003/initialization.json")
    assert init3["checkpoint"].endswith("attempt_001/best.pt")
    assert (original / "best.pt").read_bytes() == original_bytes
    (directory / "STOP").touch()
    with (work / ".local_gpu_queue.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        reset = adaptive.run(work, 1, 1, os.dup(lease.fileno()), reset_to_original=True)
    assert reset["champion"] == str(original) and reset["manual_resets"] == 1
    assert reset["lr"] == 2.5e-6 and len(reset["attempts"]) == 3


def test_official_confirmation_rejects_seen_clips(tiny_run, tmp_path):  # noqa: F811
    source = Path(tiny_run["output"])
    split = read_json(source / "split.json")
    split["train_sequences"].append("d")
    write_json(source / "split.json", split)
    with pytest.raises(ValueError, match="source-training"):
        adaptive.official_score(tmp_path, source, "testsha")


def test_adaptive_launcher_cli_leases_and_duplicate_protection(tmp_path, monkeypatch):
    import sys
    from cvlab.cli import main

    script = Path(__file__).parents[1] / "scripts/start_adaptive_finetune.py"
    spec = importlib.util.spec_from_file_location("adaptive_launcher", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "runs/rtdetr_fold0_local/best.pt"
    source.parent.mkdir(parents=True)
    source.touch()
    argv = ["start", "--work", str(tmp_path), "--attempts", "2", "--export-tiles"]
    monkeypatch.setattr(sys, "argv", argv)
    calls = []

    def execute(binary, args):
        fd = int(args[args.index("--lock-fd") + 1])
        assert os.get_inheritable(fd)
        assert os.fstat(fd).st_ino == (tmp_path / ".local_gpu_queue.lock").stat().st_ino
        monkeypatch.setattr(sys, "argv", ["cvlab", *args[4:]])
        main()

    monkeypatch.setattr(module.os, "execv", execute)
    monkeypatch.setattr(module.os, "chdir", lambda *a: None)
    monkeypatch.setattr(adaptive, "run", lambda *args: calls.append(args))
    module.main()
    assert calls[0][:3] == (str(tmp_path), 2, 6000) and calls[0][-1] is True
    with (tmp_path / ".adaptive_finetune.launcher.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        monkeypatch.setattr(sys, "argv", argv)
        module.main()
    assert len(calls) == 1
