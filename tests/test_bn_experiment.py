import copy
import fcntl
import importlib.util
import os
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from cvlab import bn_experiment as bn
from cvlab.common import digest, write_json


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)
        self.norm = nn.BatchNorm2d(3)
        self.dropout = nn.Dropout()
        self.layer_norm = nn.LayerNorm(3)

    def predict(self, images):
        assert not self.training and not self.dropout.training
        assert self.norm.training
        assert not torch.is_grad_enabled()
        return self.norm(self.conv(torch.stack(images)))


def inputs(tmp_path, clip="train"):
    Image.new("RGB", (100, 60), (180, 50, 30)).save(tmp_path / "image.jpg")
    return [
        {
            "id": 1,
            "sequence_id": clip,
            "clip": clip,
            "frame_id": 1,
            "file_name": "image.jpg",
            "width": 100,
            "height": 60,
            "boxes": [],
            "labels": [],
            "ignore_boxes": [],
        }
    ]


def config(tmp_path):
    return {
        "seed": 42,
        "model": {"name": "rtdetr", "image_size": 128},
        "data": {
            "root": str(tmp_path),
            "split": "official",
            "crop_probability": 1,
            "crop_size": 64,
        },
        "loader": {"workers": 0, "batch_size": 1, "pin_memory": False},
        "runtime": {"device": "cpu", "precision": "fp32"},
        "evaluation": {"max_detections": [100, 500]},
    }


def test_calibration_changes_only_running_statistics_and_restores_modes(tmp_path):
    model = Detector()
    before = copy.deepcopy(model.state_dict())
    momentum = model.norm.momentum
    result = bn.recalibrate(
        model, config(tmp_path), inputs(tmp_path), torch.device("cpu")
    )
    assert result == {"layers": ["norm"], "frames": 1}
    assert model.training and model.dropout.training
    assert model.norm.momentum == momentum
    changed = {
        key
        for key, value in model.state_dict().items()
        if not torch.equal(before[key], value)
    }
    assert changed == {
        "norm.running_mean",
        "norm.running_var",
        "norm.num_batches_tracked",
    }
    assert all(parameter.grad is None for parameter in model.parameters())


def test_calibration_restores_modes_and_momentum_on_failure(tmp_path, monkeypatch):
    model = Detector()
    model.dropout.eval()
    original_modes = [module.training for module in model.modules()]

    def fail(*args):
        raise RuntimeError("simulated forward failure")

    monkeypatch.setattr(model, "predict", fail)
    with pytest.raises(RuntimeError, match="forward failure"):
        bn.recalibrate(model, config(tmp_path), inputs(tmp_path), torch.device("cpu"))
    assert [module.training for module in model.modules()] == original_modes
    assert model.norm.momentum == 0.1


def test_statistic_transfer_preserves_all_learned_weights():
    model = Detector()
    before = copy.deepcopy(model.state_dict())
    donor = copy.deepcopy(before)
    for value in donor.values():
        value.fill_(3)
    assert bn.restore_statistics(model, donor) == ["norm"]
    for key, value in model.state_dict().items():
        if key in {"norm.running_mean", "norm.running_var", "norm.num_batches_tracked"}:
            assert torch.all(value == 3)
        else:
            torch.testing.assert_close(before[key], value, rtol=0, atol=0)


def test_real_rtdetr_calibration_preserves_parameters(tmp_path):
    from cvlab.models import build_model

    torch.set_num_threads(2)
    cfg = config(tmp_path)
    cfg["model"].update(
        pretrained=False, num_queries=20, num_denoising=0, max_detections=20
    )
    model = build_model(cfg["model"])
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    result = bn.recalibrate(model, cfg, inputs(tmp_path), torch.device("cpu"))
    assert len(result["layers"]) > 3
    assert all("backbone" not in name for name in result["layers"])
    for key, value in model.named_parameters():
        torch.testing.assert_close(before[key], value, rtol=0, atol=0)


def test_train_and_calibration_leakage_are_rejected():
    val = [{"sequence_id": "heldout"}]
    bn.check_isolation(["train"], [{"sequence_id": "train2"}], val)
    with pytest.raises(ValueError, match="overlaps"):
        bn.check_isolation(["heldout"], [], val)
    with pytest.raises(ValueError, match="overlaps"):
        bn.check_isolation(["train"], val, val)


@pytest.mark.parametrize(
    "full_score,exported", [(0.14, False), (0.155, False), (0.17, True)]
)
def test_controller_confirms_full_gain_over_existing_tiled_model(
    tmp_path, monkeypatch, full_score, exported
):
    from cvlab import engine, prepare, score_improvement, submit_cpu

    cfg = config(tmp_path)
    train_records = inputs(tmp_path)
    val = [{**train_records[0], "sequence_id": "val", "id": n} for n in range(2)]
    source = tmp_path / "runs/rtdetr_fold0_local"
    crop = tmp_path / "runs/rtdetr_crop_finetune"
    original = Detector()
    payload = {
        "config": cfg,
        "model": original.state_dict(),
        "adapter": None,
        "data_fingerprint": "verified",
        "epoch": 0,
        "optimizer": {"huge": True},
    }
    for directory, filename in [(source, "best.pt"), (crop, "last.pt")]:
        directory.mkdir(parents=True)
        torch.save(payload, directory / filename)
        write_json(
            directory / "split.json",
            {"fingerprint": "verified", "train_sequences": ["train"]},
        )
    original_bytes = (source / "best.pt").read_bytes()
    _, sha = bn.read_source(source / "best.pt")
    write_json(
        tmp_path / "score_improvement/tiling_summary.json",
        {
            "models": [
                {
                    "checkpoint_sha256": sha,
                    "full_candidate": {
                        "checkpoint_sha256": sha,
                        "record_ids": [0, 1],
                        "data_fingerprint": digest(val),
                        "metrics": {"coco100": {"mAP": 0.15}},
                    },
                }
            ]
        },
    )
    monkeypatch.setattr(
        prepare,
        "select_records",
        lambda c, split: train_records if split == "train" else val,
    )
    monkeypatch.setattr(engine, "setup", lambda c: torch.device("cpu"))
    monkeypatch.setattr(engine, "amp_context", lambda *a: nullcontext())

    def load(path, c, device):
        model = Detector()
        model.load_state_dict(torch.load(path, weights_only=False)["model"])
        return model

    monkeypatch.setattr(engine, "load_trained", load)
    monkeypatch.setattr(submit_cpu, "_cuda_budget", lambda *a: None)
    monkeypatch.setattr(submit_cpu, "_available_ram_mib", lambda: 8192)
    calls = []

    def validation(c, path, size, rs):
        calls.append((path.parent.name, len(rs)))
        scores = {
            "original_unchanged": 0.1,
            "original_recalibrated": 0.2,
            "crop_unchanged": 0.05,
            "crop_original_statistics": 0.3,
            "crop_recalibrated": 0.1,
        }
        value = (
            scores[path.parent.name]
            if len(rs) == 1
            else (0.1 if path.parent.name == "original_unchanged" else full_score)
        )
        return {
            "metrics": {"coco100": {"mAP": value}},
            "record_ids": [r["id"] for r in rs],
            "data_fingerprint": digest(rs),
        }

    exports = []
    monkeypatch.setattr(score_improvement, "full_validation", validation)
    monkeypatch.setattr(score_improvement, "export", lambda *args: exports.append(args))
    with (tmp_path / ".local_gpu_queue.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        result = bn.run(tmp_path, 1, 1, 6, os.dup(lease.fileno()))
    assert result["winner"] == "crop_original_statistics"
    assert result["reference"] == {"kind": "existing tiled", "mAP": 0.15}
    assert bool(exports) == exported
    assert calls[-2:] == [("original_unchanged", 2), ("crop_original_statistics", 2)]
    assert (source / "best.pt").read_bytes() == original_bytes
    derived = torch.load(
        Path(result["cases"][3]["run"]) / "best.pt", weights_only=False
    )
    assert derived["inference_only"] and "optimizer" not in derived


def test_launcher_inherits_lease_and_cli_forwards_parameters(tmp_path, monkeypatch):
    import sys
    from cvlab.cli import main

    spec = importlib.util.spec_from_file_location(
        "bn_launcher", Path(__file__).parents[1] / "scripts/start_bn_search.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "runs").mkdir()
    monkeypatch.setattr(sys, "argv", ["start", "--work", str(tmp_path)])
    calls = []

    def execute(binary, args):
        fd = int(args[args.index("--lock-fd") + 1])
        assert os.get_inheritable(fd)
        assert os.fstat(fd).st_ino == (tmp_path / ".local_gpu_queue.lock").stat().st_ino
        monkeypatch.setattr(sys, "argv", ["cvlab", *args[4:]])
        main()

    monkeypatch.setattr(module.os, "execv", execute)
    monkeypatch.setattr(module.os, "chdir", lambda *a: None)
    monkeypatch.setattr(bn, "run", lambda *a: calls.append(a))
    module.main()
    assert calls[0][:4] == (str(tmp_path), 192, 256, 6)
