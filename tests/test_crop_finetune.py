import numpy as np
import pytest
import torch
from PIL import Image
import copy
import hashlib
from pathlib import Path

from cvlab.data import Frames, training_crop
from cvlab.common import load_config, read_json
from test_pipeline import competition  # noqa: F401


class TinyDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))

    def loss(self, images, targets):
        return {"loss": (self.weight - 2).square()}

    def export_metadata(self):
        return None


@pytest.fixture
def tiny_run(competition, tmp_path, monkeypatch):  # noqa: F811
    from cvlab import engine, validation_pipeline

    root, prepared = competition
    config = load_config(Path(__file__).parents[1] / "configs/fcos.yaml")
    config["output"] = str(tmp_path / "source")
    config["data"].update(root=str(root), prepared=str(prepared))
    config["model"].update(pretrained=False, image_size=128)
    config["runtime"].update(device="cpu", precision="fp32", cpu_threads=1)
    config["loader"].update(workers=0, batch_size=2, pin_memory=False)
    config["train"].update(epochs=1, max_batches_per_epoch=1, max_hours=1)
    monkeypatch.setattr(engine, "build_model", lambda *a, **kw: TinyDetector())
    monkeypatch.setattr(engine, "environment", lambda: {})
    monkeypatch.setattr(engine, "predict", lambda *a, **kw: ([], {}))

    def scores(records, *args):
        # Distinct split scores expose accidental reuse of the source fold best.
        return {"coco100": {"mAP": 0.1 if records[0]["clip"] == "d" else 0.3}}

    monkeypatch.setattr(engine, "evaluate", scores)
    monkeypatch.setattr(validation_pipeline, "evaluate", scores)
    engine.train(config)
    return config


def warm_config(source, tmp_path):
    config = copy.deepcopy(source)
    config["output"] = str(tmp_path / "warm")
    config["data"].update(
        split="official", frame_step=2, crop_probability=0.5, crop_size=64
    )
    config["train"]["lr"] = 5e-5
    return config


@pytest.mark.parametrize("asynchronous", [False, True])
def test_warm_start_uses_new_holdout_fresh_optimizer_and_resumes(
    tiny_run, tmp_path, asynchronous
):
    from cvlab import engine
    from cvlab.validation_pipeline import ValidationPipeline

    source = Path(tiny_run["output"]) / "best.pt"
    original = source.read_bytes()
    weights = torch.load(source, weights_only=False)["model"]
    config = warm_config(tiny_run, tmp_path)
    output = Path(config["output"])
    if asynchronous:
        with ValidationPipeline(0) as pipeline:
            engine.train(config, init_checkpoint=source, validation_pipeline=pipeline)
    else:
        engine.train(config, init_checkpoint=source)
    initial = torch.load(output / "best.pt", weights_only=False)
    final = torch.load(output / "last.pt", weights_only=False)
    assert initial["epoch"] == -1  # Tied/worse training cannot replace initialization.
    assert initial["best"] == 0.1 and final["best"] == 0.1
    assert initial["optimizer"]["state"] == {}
    assert initial["optimizer"]["param_groups"][1]["lr"] == 5e-5
    torch.testing.assert_close(initial["model"]["weight"], weights["weight"])
    assert final["model"]["weight"] > weights["weight"]
    assert initial["initialization"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert initial["initialization"]["source_best_score"] == 0.3
    assert source.read_bytes() == original
    assert set(final["seen_training_sequences"]).isdisjoint(
        read_json(output / "split.json")["val_sequences"]
    )
    history = (output / "history.jsonl").read_text()
    source.unlink()  # Resume is self-contained, without the old source checkpoint.
    engine.train(config, resume=output / "last.pt")
    assert (output / "history.jsonl").read_text() == history
    assert (
        read_json(output / "initial_validation.json")["metrics"]["coco100"]["mAP"]
        == 0.1
    )


def test_interrupted_initial_validation_recovers_without_training_repeat(
    tiny_run, tmp_path, monkeypatch
):
    from cvlab import engine

    source = Path(tiny_run["output"]) / "best.pt"
    config = warm_config(tiny_run, tmp_path)
    real_predict = engine.predict

    def fail(*args, **kwargs):
        raise RuntimeError("interrupted initial validation")

    monkeypatch.setattr(engine, "predict", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        engine.train(config, init_checkpoint=source)
    output = Path(config["output"])
    assert torch.load(output / "last.pt", weights_only=False)["epoch"] == -1
    monkeypatch.setattr(engine, "predict", real_predict)
    engine.train(config, resume=output / "last.pt")
    assert len((output / "history.jsonl").read_text().splitlines()) == 1
    assert read_json(output / "initialization.json")["source_epoch"] == 0


@pytest.mark.parametrize("legacy", [False, True])
def test_warm_start_rejects_seen_validation_clips(tiny_run, tmp_path, legacy):
    from cvlab import engine
    from cvlab.prepare import select_records

    source = Path(tiny_run["output"]) / "best.pt"
    if legacy:
        payload = torch.load(source, weights_only=False)
        del payload["seen_training_sequences"]
        torch.save(payload, source)
    config = warm_config(tiny_run, tmp_path)
    config["data"].update(split="fold", fold=1)
    payload = torch.load(source, weights_only=False)
    seen = (
        payload.get("seen_training_sequences")
        or read_json(source.parent / "split.json")["train_sequences"]
    )
    if not set(seen) & {r["sequence_id"] for r in select_records(config, "val")}:
        config["data"]["fold"] = 2
    with pytest.raises(ValueError, match="leak"):
        engine.train(config, init_checkpoint=source)


def test_legacy_warm_start_accepts_verified_split(tiny_run, tmp_path):
    from cvlab import engine

    source = Path(tiny_run["output"]) / "best.pt"
    payload = torch.load(source, weights_only=False)
    del payload["seen_training_sequences"]
    torch.save(payload, source)
    config = warm_config(tiny_run, tmp_path)
    engine.train(config, init_checkpoint=source)
    assert (
        read_json(Path(config["output"]) / "initialization.json")["source_epoch"] == 0
    )
    payload["data_fingerprint"] = "wrong split"
    torch.save(payload, source)
    config["output"] = str(tmp_path / "bad")
    with pytest.raises(ValueError, match="fingerprint"):
        engine.train(config, init_checkpoint=source)


def test_warm_start_rejects_model_change_and_resume_combination(tiny_run, tmp_path):
    from cvlab import engine

    source = Path(tiny_run["output"]) / "best.pt"
    config = warm_config(tiny_run, tmp_path)
    with pytest.raises(ValueError, match="not both"):
        engine.train(config, resume=source, init_checkpoint=source)
    config["model"]["image_size"] = 256
    with pytest.raises(ValueError, match="model configuration"):
        engine.train(config, init_checkpoint=source)


def test_crop_recipe_uses_all_train_clips_without_changing_source(tiny_run, tmp_path):
    from cvlab.crop_finetune import recipe
    from cvlab.prepare import select_records

    before = copy.deepcopy(tiny_run)
    config = recipe(tiny_run, tmp_path)
    assert tiny_run == before and config["model"] == tiny_run["model"]
    assert len(select_records(config, "train")) == 3
    assert len(select_records(config, "val")) == 2
    assert {r["clip"] for r in select_records(config, "train")} == {"a", "b", "c"}
    assert config["data"]["crop_probability"] == 0.5
    assert config["train"]["lr"] == 5e-5
    assert config["loader"]["batch_size"] == 1


def test_controller_runs_engine_and_resumes_without_exporting_tied_score(
    tiny_run, tmp_path, monkeypatch
):
    import fcntl
    import os
    import shutil
    from cvlab import crop_finetune, submit_cpu

    work = tmp_path / "work"
    source = work / "runs/fcos_fold0_local"
    source.parent.mkdir(parents=True)
    shutil.copytree(tiny_run["output"], source)
    real_recipe = crop_finetune.recipe

    def cpu_recipe(*args):
        config = real_recipe(*args)
        config["runtime"].update(device="cpu", precision="fp32", cpu_threads=1)
        config["loader"].update(workers=0, pin_memory=False)
        config["train"].update(max_batches_per_epoch=1)
        return config

    monkeypatch.setattr(crop_finetune, "recipe", cpu_recipe)
    monkeypatch.setattr(
        submit_cpu,
        "create_submission",
        lambda *a, **kw: pytest.fail("Tied score must not export"),
    )
    with (work / ".local_gpu_queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        first = crop_finetune.run(work, ["fcos"], 1, 1, 5e-5, 64, os.dup(lock.fileno()))
        second = crop_finetune.run(
            work, ["fcos"], 1, 1, 5e-5, 64, os.dup(lock.fileno())
        )
    assert first == second and first[0]["gain"] == 0
    assert first[0]["submission"] is None
    assert (work / "crop_finetune_summary.json").is_file()


def test_launcher_waits_for_tiling_and_passes_inherited_gpu_lease(
    tmp_path, monkeypatch
):
    import fcntl
    import importlib.util
    import os
    import sys

    script = Path(__file__).parents[1] / "scripts/start_crop_finetune.py"
    spec = importlib.util.spec_from_file_location("crop_launcher", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "runs").mkdir()
    monkeypatch.setattr(sys, "argv", ["start", "--work", str(tmp_path)])
    calls, lock_order = [], []
    original_flock = fcntl.flock

    def observed_flock(file, operation):
        if operation == fcntl.LOCK_EX:
            lock_order.append(Path(file.name).name)
        return original_flock(file, operation)

    def execute(binary, args):
        fd = int(args[args.index("--lock-fd") + 1])
        assert os.get_inheritable(fd)
        assert os.fstat(fd).st_ino == (tmp_path / ".local_gpu_queue.lock").stat().st_ino
        with (tmp_path / ".local_gpu_queue.lock").open("a") as competing:
            with pytest.raises(BlockingIOError):
                original_flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)
        calls.append(args)

    monkeypatch.setattr(module.fcntl, "flock", observed_flock)
    monkeypatch.setattr(module.os, "execv", execute)
    monkeypatch.setattr(module.os, "chdir", lambda *a: None)
    module.main()
    assert lock_order == [".tiled_search.launcher.lock", ".local_gpu_queue.lock"]
    assert calls[0][2:5] == ["-m", "cvlab", "train-crops"]
    assert calls[0][calls[0].index("--lr") + 1] == "5e-05"
    calls.clear()
    with (tmp_path / ".crop_finetune.launcher.lock").open("a") as lock:
        original_flock(lock, fcntl.LOCK_EX)
        module.main()
    assert calls == []


def test_native_crop_clips_boxes_and_keeps_labels_aligned(monkeypatch):
    from cvlab import data

    monkeypatch.setattr(data.random, "choice", lambda choices: choices[0])
    monkeypatch.setattr(data.random, "uniform", lambda *args: 0)
    image, boxes, labels = training_crop(
        Image.new("RGB", (1000, 800)),
        [[10, 10, 100, 100], [800, 600, 990, 790]],
        [0, 7],
        400,
    )
    assert image.size == (400, 400) and labels == [0]
    np.testing.assert_array_equal(boxes, [[10, 10, 100, 100]])
    image, boxes, labels = training_crop(Image.new("RGB", (100, 80)), [], [], 400)
    assert image.size == (100, 80) and boxes.shape == (0, 4) and labels == []


def test_crop_is_training_only_and_legacy_spawn_dataset_still_works(tmp_path):
    path = tmp_path / "image.jpg"
    Image.new("RGB", (200, 100), "white").save(path)
    records = [
        {
            "id": 1,
            "sequence_id": "clip",
            "file_name": "image.jpg",
            "width": 200,
            "height": 100,
            "boxes": [[10, 10, 40, 40]],
            "labels": [0],
            "ignore_boxes": [],
        }
    ]
    normal = Frames(tmp_path, records, 128)
    cropping = Frames(tmp_path, records, 128, crop_probability=1, crop_size=64)
    torch.testing.assert_close(normal[0][0], cropping[0][0])
    torch.testing.assert_close(normal[0][1]["boxes"], cropping[0][1]["boxes"])
    legacy = Frames(tmp_path, records, 128, training=True, flip=0)
    del legacy.crop_probability, legacy.crop_size
    assert legacy[0][1]["labels"].tolist() == [1]
