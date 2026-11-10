import fcntl
import importlib.util
import os
from pathlib import Path

import torch
import pytest

from cvlab import engine, iterative_finetune
from cvlab.common import read_json
from test_crop_finetune import tiny_run  # noqa: F401
from test_pipeline import competition  # noqa: F401


@pytest.mark.parametrize("fail", [False, True])
def test_tiling_allocator_limit_is_restored_before_more_training(monkeypatch, fail):
    from cvlab import score_improvement

    state = {"fraction": 0.95}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_per_process_memory_fraction", lambda device: state["fraction"]
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, device: state.update(fraction=fraction),
    )
    monkeypatch.setattr(score_improvement, "release_gpu", lambda: None)

    def action():
        state["fraction"] = 0.75  # Six-GiB inference cap on an eight-GiB device.
        if fail:
            raise RuntimeError("candidate failed")
        return "exported"

    if fail:
        with pytest.raises(RuntimeError, match="candidate failed"):
            iterative_finetune.candidate_with_restored_budget(action)
    else:
        assert iterative_finetune.candidate_with_restored_budget(action) == "exported"
    assert state["fraction"] == 0.95


def test_frozen_bn_keeps_statistics_and_affine_gradients():
    model = torch.nn.Sequential(torch.nn.BatchNorm2d(3), torch.nn.Conv2d(3, 2, 1))
    before = {key: value.clone() for key, value in model[0].state_dict().items()}
    engine.training_mode(model, {"freeze_batch_norm": True})
    model(torch.randn(2, 3, 8, 8)).square().mean().backward()
    assert model.training and model[1].training and not model[0].training
    assert model[0].weight.grad.abs().sum() > 0
    for key, value in model[0].state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_round_yield_resumes_optimizer_without_restarting_weights(tiny_run, tmp_path):  # noqa: F811
    from cvlab.crop_finetune import recipe
    from cvlab.validation_pipeline import ValidationPipeline

    config = recipe(tiny_run, tmp_path, epochs=10, lr=1e-5, full_frames=True)
    config["runtime"].update(device="cpu", precision="fp32", cpu_threads=1)
    config["loader"].update(workers=0, pin_memory=False)
    config["train"]["early_stopping_patience"] = 0
    source = Path(tiny_run["output"]) / "best.pt"
    output = Path(config["output"])
    for round_index, batch_count in enumerate([1, 2]):
        with ValidationPipeline(0) as pipeline:
            engine.train(
                config,
                resume=output / "last.pt" if round_index else None,
                init_checkpoint=source if not round_index else None,
                validation_pipeline=pipeline,
                yield_after_epoch=True,
                round_batches=batch_count,
            )
        last = torch.load(output / "last.pt", weights_only=False)
        assert last["epoch"] == round_index
        assert last["execution"]["round_batches"] == batch_count
        step = next(iter(last["optimizer"]["state"].values()))["step"]
        assert step == round_index + 1  # accumulation=2: one update per round.
    rows = iterative_finetune.history(output)
    assert [row["train_batches"] for row in rows] == [1, 2]
    assert len(rows) == 2


def test_controller_round_limits_resume_and_stop_preserve_source(
    tiny_run,  # noqa: F811
    tmp_path,
    monkeypatch,  # noqa: F811
):  # noqa: F811
    import shutil

    work = tmp_path / "work"
    source = work / "runs/rtdetr_fullframe_finetune"
    source.parent.mkdir(parents=True)
    shutil.copytree(tiny_run["output"], source)
    before = (source / "best.pt").read_bytes()
    real_recipe = iterative_finetune.recipe

    def cpu_recipe(*args):
        cfg = real_recipe(*args)
        cfg["runtime"].update(device="cpu", precision="fp32", cpu_threads=1)
        cfg["loader"].update(workers=0, pin_memory=False)
        cfg["train"]["early_stopping_patience"] = 0
        return cfg

    monkeypatch.setattr(iterative_finetune, "recipe", cpu_recipe)
    with (work / ".local_gpu_queue.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        first = iterative_finetune.run(work, 2, 1, os.dup(lease.fileno()))
        second = iterative_finetune.run(work, 1, 2, os.dup(lease.fileno()))
        assert len(first["rounds"]) == 2 and len(second["rounds"]) == 3
        output = work / "runs/rtdetr_iterative_finetune"
        (output / "STOP").touch()
        stopped = iterative_finetune.run(work, 3, 2, os.dup(lease.fileno()))
    assert len(stopped["rounds"]) == 3
    assert (source / "best.pt").read_bytes() == before
    cfg = read_json(output / "config.json")
    assert cfg["train"]["freeze_batch_norm"]
    assert cfg["data"]["crop_probability"] == 0
    assert cfg["train"]["lr"] == 1e-5


def test_iterative_launcher_to_cli_passes_rounds_and_lease(tmp_path, monkeypatch):
    import sys
    from cvlab.cli import main

    script = Path(__file__).parents[1] / "scripts/start_iterative_finetune.py"
    spec = importlib.util.spec_from_file_location("iterative_launcher", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "runs").mkdir()
    argv = ["start", "--work", str(tmp_path), "--batches", "8000", "--export-tiles"]
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
    monkeypatch.setattr(iterative_finetune, "run", lambda *args: calls.append(args))
    module.main()
    assert calls[0][:3] == (str(tmp_path), 3, 8000) and calls[0][-1] is True
    with (tmp_path / ".iterative_finetune.launcher.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        monkeypatch.setattr(sys, "argv", argv)
        module.main()
    assert len(calls) == 1


def test_candidate_export_between_rounds_keeps_optimizer_progress(
    tiny_run,  # noqa: F811
    tmp_path,
    monkeypatch,
):  # noqa: F811
    import shutil
    from cvlab import tiled_search, validation_pipeline
    from test_crop_finetune import TinyDetector

    work = tmp_path / "export_work"
    source = work / "runs/rtdetr_fullframe_finetune"
    source.parent.mkdir(parents=True)
    shutil.copytree(tiny_run["output"], source)
    real_recipe = iterative_finetune.recipe

    def cpu_recipe(*args):
        cfg = real_recipe(*args)
        cfg["runtime"].update(device="cpu", precision="fp32", cpu_threads=1)
        cfg["loader"].update(workers=0, pin_memory=False)
        return cfg

    monkeypatch.setattr(iterative_finetune, "recipe", cpu_recipe)
    scores = iter([0.1, 0.12, 0.13])  # initial validation, then two trained rounds.

    def score(*args):
        return {"coco100": {"mAP": next(scores)}}

    monkeypatch.setattr(engine, "evaluate", score)
    monkeypatch.setattr(validation_pipeline, "evaluate", score)
    events = []
    original_loss = TinyDetector.loss

    def loss(*args):
        events.append("train")
        return original_loss(*args)

    def export(*args, **kwargs):
        # Duplicated lease remains held throughout export and following training.
        fd = args[7]
        assert os.fstat(fd).st_ino == (work / ".local_gpu_queue.lock").stat().st_ino
        assert kwargs["runs"] == [work / "runs/rtdetr_iterative_finetune"]
        last = torch.load(kwargs["runs"][0] / "last.pt", weights_only=False)
        step = next(iter(last["optimizer"]["state"].values()))["step"]
        assert step == events.count("train")
        events.append("export")
        os.close(fd)  # Same ownership convention as tiled_search.run.
        return {"models": [{"outcome": "checked"}]}

    monkeypatch.setattr(TinyDetector, "loss", loss)
    monkeypatch.setattr(tiled_search, "run", export)
    with (work / ".local_gpu_queue.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        report = iterative_finetune.run(work, 2, 1, os.dup(lease.fileno()), True)
    assert events == ["train", "export", "train", "export"]
    assert len(report["tile_attempts"]) == 2
