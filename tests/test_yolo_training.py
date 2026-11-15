import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cvlab.common import CLASSES, read_json, write_json
from cvlab.yolo_training import (
    EpochObserver,
    common_config,
    prepare_dataset,
    train_challenger,
)
from test_pipeline import competition  # noqa: F401


def test_evaluation_does_not_reset_training_random_streams():
    import random
    from cvlab.yolo_training import isolated_evaluation

    py, numpy, tensor = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    with isolated_evaluation():
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        torch.set_num_threads(1)
    assert random.getstate() == py
    restored = np.random.get_state()
    assert restored[0] == numpy[0] and np.array_equal(restored[1], numpy[1]) and restored[2:] == numpy[2:]
    assert torch.equal(torch.get_rng_state(), tensor)
    assert torch.get_num_threads() == threads


def test_yolo_conversion_preserves_data_and_refuses_drift(competition, tmp_path):  # noqa: F811
    root, prepared = competition
    destination = tmp_path / "yolo"
    data = prepare_dataset(root, prepared, destination)
    assert data.is_file()
    report = read_json(destination / "prepared.json")
    assert report["frames"] == {"train": 6, "val": 2}
    assert not set(report["families"]["train"]) & set(report["families"]["val"])
    images = sorted((destination / "images/train").glob("*.jpg"))
    assert len(images) == 6 and all(
        p.is_symlink() and p.resolve().is_relative_to(root) for p in images
    )
    rows = (
        (destination / "labels/train" / (images[0].stem + ".txt"))
        .read_text()
        .splitlines()
    )
    assert [int(row.split()[0]) for row in rows] == [0, 7]
    np.testing.assert_allclose(
        [float(x) for x in rows[0].split()[1:]], [0.25, 0.5, 0.1, 0.2]
    )
    assert prepare_dataset(root, prepared, destination) == data
    manifest = read_json(prepared / "train.json")
    manifest["images"][0]["labels"][0] = 1
    write_json(prepared / "train.json", manifest)
    with pytest.raises(ValueError, match="changed"):
        prepare_dataset(root, prepared, destination)


def test_yolo_mask_restores_targets_and_predictor_maps_classes(monkeypatch):
    pytest.importorskip("ultralytics")
    from ultralytics.data import YOLODataset
    from cvlab import yolo_backend as backend

    obj = object.__new__(backend.MaskedDataset)
    obj.ignored = {"frame.jpg": [[0, 0, 0.75, 0.75]]}
    obj.im_files = ["frame.jpg"]
    obj.augment = True
    obj.labels = [{"bboxes": np.array([[0.25, 0.25, 0.1, 0.1]], dtype=np.float32)}]
    monkeypatch.setattr(
        YOLODataset,
        "load_image",
        lambda *a: (np.full((100, 100, 3), 255, np.uint8), (100, 100), (100, 100)),
    )
    pixels, _, _ = obj.load_image(0)
    assert pixels[5, 5].tolist() == [114, 114, 114]
    assert pixels[25, 25].tolist() == [255, 255, 255]

    class Core(torch.nn.Module):
        names = dict(enumerate(CLASSES))

        def forward(self, images):
            return images

    outputs = torch.tensor(
        [[10.0, 20.0, 30.0, 40.0, 0.9, 0.0], [12.0, 23.0, 40.0, 50.0, 0.8, 7.0]]
    )
    monkeypatch.setattr(backend, "non_max_suppression", lambda *a, **kw: [outputs])
    config = common_config("root", "prepared", "run", device="cpu")
    result = backend.Predictor(Core(), config).predict([torch.ones(3, 64, 64)])[0]
    assert result["labels"].tolist() == [1, 8]
    assert torch.equal(result["boxes"], outputs[:, :4])


def test_observer_waits_multiple_epochs_and_keeps_best(monkeypatch, tmp_path):
    from cvlab import yolo_training as module

    run = tmp_path / "run"
    (run / "weights").mkdir(parents=True)
    last = run / "weights/last.pt"
    last.write_bytes(b"weights")
    scores = iter([0.20, 0.21, 0.209, 0.208, 0.207])
    monkeypatch.setattr(
        module,
        "evaluate_checkpoint",
        lambda *a: {
            "metrics": {"coco100": {"mAP": next(scores)}},
            "checkpoint_sha256": module.file_sha(last),
        },
    )
    trainer = SimpleNamespace(
        epoch=0,
        last=last,
        tloss=torch.ones(3),
        stop=False,
        optimizer=SimpleNamespace(param_groups=[{"lr": 0.001}]),
    )
    observer = EpochObserver(run, {}, 1)
    for epoch in range(5):
        trainer.epoch = epoch
        observer(trainer)
        assert trainer.stop == (epoch == 4)
    assert (
        read_json(run / "common_best_metrics.json")["metrics"]["coco100"]["mAP"] == 0.21
    )
    assert read_json(run / "status.json")["status"] == "full_validation_plateau"
    assert len((run / "common_history.jsonl").read_text().splitlines()) == 5


def test_observer_serializes_nonfinite_meter_as_null_before_promoting(monkeypatch, tmp_path):
    from cvlab import yolo_training as module

    run = tmp_path / "run"
    weights = run / "weights"
    weights.mkdir(parents=True)
    last = weights / "last.pt"
    last.write_bytes(b"new weights")
    best = weights / "common_best.pt"
    best.write_bytes(b"old weights")
    (run / "common_best_metrics.json").write_text('{"old":true}')
    monkeypatch.setattr(
        module,
        "evaluate_checkpoint",
        lambda *a: {
            "metrics": {"coco100": {"mAP": 0.21}},
            "checkpoint_sha256": module.file_sha(last),
        },
    )
    trainer = SimpleNamespace(
        epoch=0,
        last=last,
        tloss=torch.tensor([1.0, float("nan"), float("inf")]),
        stop=False,
        optimizer=SimpleNamespace(param_groups=[{"lr": 0.001}]),
    )
    EpochObserver(run, {}, 1)(trainer)
    row = json.loads((run / "common_history.jsonl").read_text())
    assert row["train_loss"] == [1.0, None, None]
    assert row["numeric_diagnostics"]["loss_meter_finite"] is False
    assert row["numeric_diagnostics"]["batch_loss_finite"] is None
    assert best.read_bytes() == b"new weights"
    assert read_json(run / "common_best_metrics.json")["train_loss"] == [1.0, None, None]

    last.write_bytes(b"candidate with unserializable receipt")
    old_weights = best.read_bytes()
    old_receipt = (run / "common_best_metrics.json").read_bytes()
    monkeypatch.setattr(
        module,
        "evaluate_checkpoint",
        lambda *a: {
            "metrics": {"coco100": {"mAP": 0.22}, "bad": float("nan")},
            "checkpoint_sha256": module.file_sha(last),
        },
    )
    trainer.epoch = 1
    with pytest.raises(ValueError, match="JSON compliant"):
        EpochObserver(run, {}, 1)(trainer)
    assert best.read_bytes() == old_weights
    assert (run / "common_best_metrics.json").read_bytes() == old_receipt


def test_stock_yolo_cpu_training_saved_validation_and_resume(
    competition, tmp_path, monkeypatch  # noqa: F811
):  # noqa: F811
    pytest.importorskip("ultralytics")
    from ultralytics.utils import checks
    from cvlab.yolo_training import evaluate_checkpoint, export_checkpoint

    # Real forward/backward/checkpoint/common-COCO path; never downloads pretrained weights.
    monkeypatch.setattr(checks, "check_font", lambda *a, **kw: None)
    monkeypatch.setattr("ultralytics.data.utils.check_font", lambda *a, **kw: None)
    root, prepared = competition
    work = tmp_path / "work"
    work.mkdir()
    torch.set_num_threads(1)
    run = train_challenger(
        work,
        root,
        prepared,
        name="cpu_yolo",
        epochs=2,
        hours=1e-9,
        batch=2,
        image_size=64,
        workers=0,
        device="cpu",
        initial_model="yolo11n.yaml",
    )
    history = [
        json.loads(line)
        for line in (run / "common_history.jsonl").read_text().splitlines()
    ]
    assert len(history) == 1 and history[0]["frames"] == 2
    assert history[0]["metrics"]["coco100"]["mAP"] is not None
    last = run / "weights/last.pt"
    payload = torch.load(last, map_location="cpu", weights_only=False)
    assert payload["epoch"] == 0 and payload["optimizer"] is not None
    assert payload["updates"] > 0 and payload["ema"].names == dict(enumerate(CLASSES))
    assert (run / "weights/epoch0.pt").is_file()
    report = evaluate_checkpoint(
        run / "weights/common_best.pt", read_json(run / "config.json")
    )
    assert report["data_fingerprint"] == history[0]["data_fingerprint"]
    assert report["metrics"] == history[0]["metrics"]
    config = read_json(run / "config.json")
    config["model"]["score_threshold"] = (
        0  # Untrained tiny model's priors otherwise yield an empty export.
    )
    checked = export_checkpoint(
        run / "weights/common_best.pt", config, work / "submissions/cpu.csv"
    )
    assert (
        checked["valid"]
        and checked["matches_template_order"]
        and checked["images"] == 1
    )
    assert not checked["all_predictions_empty"]
    train_challenger(
        work,
        root,
        prepared,
        name="cpu_yolo",
        epochs=2,
        hours=1,
        batch=2,
        image_size=64,
        workers=0,
        device="cpu",
        resume=True,
        initial_model="yolo11n.yaml",
    )
    resumed = torch.load(last, map_location="cpu", weights_only=False)
    assert resumed["epoch"] == 1 and resumed["updates"] > payload["updates"]
    assert len((run / "common_history.jsonl").read_text().splitlines()) == 2
    with pytest.raises(ValueError, match="same recorded recipe"):
        train_challenger(
            work,
            root,
            prepared,
            name="cpu_yolo",
            epochs=3,
            hours=1,
            batch=2,
            image_size=64,
            workers=0,
            device="cpu",
            resume=True,
            initial_model="yolo11n.yaml",
        )
