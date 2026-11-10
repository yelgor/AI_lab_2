import copy
import fcntl
import importlib.util
import json
import os
import signal
from pathlib import Path

import pytest
import torch

from cvlab import engine
from cvlab.common import write_json
from cvlab.crop_finetune import recipe
from cvlab.prepare import select_records
from test_crop_finetune import tiny_run  # noqa: F401
from test_pipeline import competition  # noqa: F401


def launcher():
    path = Path(__file__).parents[1] / "scripts/start_fullframe_finetune.py"
    spec = importlib.util.spec_from_file_location("fullframe_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_frame_recipe_uses_all_frames_without_crops(tiny_run, tmp_path):  # noqa: F811
    before = copy.deepcopy(tiny_run)
    config = recipe(tiny_run, tmp_path, lr=1e-5, full_frames=True)
    assert tiny_run == before
    assert config["model"] == tiny_run["model"]
    assert config["data"]["split"] == "official"
    assert config["data"]["frame_step"] == 1
    assert config["data"]["crop_probability"] == 0
    assert len(select_records(config, "train")) == 6
    assert len(select_records(config, "val")) == 2
    assert config["train"]["lr"] == 1e-5
    assert config["train"]["early_stopping_patience"] == 2
    assert Path(config["output"]).name == "fcos_fullframe_finetune"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_early_stopping_preserves_best_weights_and_resume(
    tiny_run,  # noqa: F811
    tmp_path,
    asynchronous,  # noqa: F811
):
    from cvlab.validation_pipeline import ValidationPipeline

    source = Path(tiny_run["output"]) / "best.pt"
    source_bytes = source.read_bytes()
    config = recipe(tiny_run, tmp_path, epochs=10, lr=1e-5, full_frames=True)
    config["runtime"].update(device="cpu", precision="fp32", cpu_threads=1)
    config["loader"].update(workers=0, pin_memory=False)
    config["train"]["max_batches_per_epoch"] = 1
    if asynchronous:
        with ValidationPipeline(0) as pipeline:
            engine.train(config, init_checkpoint=source, validation_pipeline=pipeline)
    else:
        engine.train(config, init_checkpoint=source)
    output = Path(config["output"])
    history = (output / "history.jsonl").read_text()
    # At most one additional in-flight epoch while previous CPU metrics commit.
    assert 2 <= len(history.splitlines()) <= 3
    assert torch.load(output / "best.pt", weights_only=False)["epoch"] == -1
    last = torch.load(output / "last.pt", weights_only=False)
    assert last["best"] == 0.1
    assert source.read_bytes() == source_bytes
    engine.train(config, resume=output / "last.pt")
    assert (output / "history.jsonl").read_text() == history


def test_early_stopping_uses_significant_improvements_and_unique_epochs(tmp_path):
    config = {
        "train": {"early_stopping_patience": 2, "early_stopping_min_delta": 0.001}
    }
    write_json(
        tmp_path / "initial_validation.json", {"metrics": {"coco100": {"mAP": 0.2}}}
    )

    def history(values):
        (tmp_path / "history.jsonl").write_text(
            "".join(
                json.dumps({"epoch": epoch, "metrics": {"coco100": {"mAP": value}}})
                + "\n"
                for epoch, value in values
            )
        )

    history([(0, 0.199), (1, 0.2)])
    assert engine.early_stopping_reached(config, tmp_path)
    history([(0, 0.199), (1, 0.205), (1, 0.205), (2, 0.204)])
    assert not engine.early_stopping_reached(config, tmp_path)
    history([(0, 0.199), (1, 0.205), (2, 0.204), (3, 0.204)])
    assert engine.early_stopping_reached(config, tmp_path)


def test_pending_validation_is_completed_even_when_early_stopping_is_reached(
    tiny_run, tmp_path, monkeypatch  # noqa: F811
):  # noqa: F811
    source = Path(tiny_run["output"]) / "best.pt"
    config = recipe(tiny_run, tmp_path, epochs=1, lr=1e-5, full_frames=True)
    config["runtime"].update(device="cpu", precision="fp32", cpu_threads=1)
    config["loader"].update(workers=0, pin_memory=False)
    config["train"]["max_batches_per_epoch"] = 1
    engine.train(config, init_checkpoint=source)
    output = Path(config["output"])
    payload = torch.load(output / "last.pt", weights_only=False)
    payload["validation_pending"] = {
        "loss_sum": 1,
        "observed": 1,
        "batches": 1,
        "stop": True,
    }
    torch.save(payload, output / "last.pt")
    monkeypatch.setattr(engine, "early_stopping_reached", lambda *a: True)
    from test_crop_finetune import TinyDetector

    monkeypatch.setattr(
        TinyDetector,
        "loss",
        lambda *a: pytest.fail("Completed training must not repeat"),
    )
    engine.train(config, resume=output / "last.pt")
    resumed = torch.load(output / "last.pt", weights_only=False)
    assert resumed["validation_pending"] is None
    torch.testing.assert_close(payload["model"]["weight"], resumed["model"]["weight"])


@pytest.mark.parametrize(
    "args,expected",
    [
        (["python", "-u", "-m", "cvlab", "evaluate-bn"], True),
        (["python", "-u", "scripts/start_bn_search.py"], True),
        (["python", "-u", "-m", "cvlab", "train-fullframes"], False),
        (["python", "-u", "-m", "cvlab", "train"], False),
        (["systemd-inhibit", "python", "-m", "cvlab", "evaluate-bn"], False),
        (["python", "-c", "code", "-m", "cvlab", "evaluate-bn"], False),
    ],
)
def test_replace_targets_only_bn_python_for_same_work(tmp_path, args, expected):
    module = launcher()
    assert module.is_bn_job([*args, "--work", str(tmp_path)], tmp_path) is expected
    assert not module.is_bn_job([*args, "--work", str(tmp_path / "other")], tmp_path)


def test_bn_stop_pins_identity_and_never_signals_training(tmp_path, monkeypatch):
    module = launcher()
    for pid, command in [(11, "evaluate-bn"), (12, "train")]:
        process = tmp_path / str(pid)
        process.mkdir()
        (process / "cmdline").write_bytes(
            (
                "\0".join(["python", "-m", "cvlab", command, "--work", str(tmp_path)])
                + "\0"
            ).encode()
        )
    opened, signaled, closed = [], [], []
    monkeypatch.setattr(os, "pidfd_open", lambda pid: opened.append(pid) or 99)
    monkeypatch.setattr(
        signal, "pidfd_send_signal", lambda fd, sig: signaled.append((fd, sig))
    )
    monkeypatch.setattr(os, "close", lambda fd: closed.append(fd))
    assert module.stop_bn(tmp_path, tmp_path) == [11]
    assert opened == [11] and closed == [99]
    assert signaled == [(99, signal.SIGTERM)]


def test_full_frame_launcher_inherits_lease_and_cli_config(tmp_path, monkeypatch):
    import sys
    from cvlab import crop_finetune
    from cvlab.cli import main

    module = launcher()
    source = tmp_path / "runs/rtdetr_fold0_local"
    source.mkdir(parents=True)
    for file in ("best.pt", "config.json", "split.json"):
        (source / file).touch()
    argv = ["start", "--work", str(tmp_path), "--replace-bn"]
    monkeypatch.setattr(sys, "argv", argv)
    stopped, calls = [], []
    monkeypatch.setattr(module, "stop_bn", lambda work: stopped.append(work))

    def execute(binary, args):
        fd = int(args[args.index("--lock-fd") + 1])
        assert os.get_inheritable(fd)
        assert os.fstat(fd).st_ino == (tmp_path / ".local_gpu_queue.lock").stat().st_ino
        monkeypatch.setattr(sys, "argv", ["cvlab", *args[4:]])
        main()

    monkeypatch.setattr(module.os, "execv", execute)
    monkeypatch.setattr(module.os, "chdir", lambda *a: None)
    monkeypatch.setattr(
        crop_finetune, "run", lambda *args, **kw: calls.append((args, kw))
    )
    module.main()
    assert stopped == [tmp_path]
    assert calls[0][0][:5] == (str(tmp_path), ["rtdetr"], 2, 4, 1e-5)
    assert calls[0][1] == {"full_frames": True}
    with (tmp_path / ".fullframe_finetune.launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        monkeypatch.setattr(sys, "argv", argv)
        module.main()
    assert len(stopped) == len(calls) == 1
