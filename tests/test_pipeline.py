import copy
import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from cvlab.common import read_json
from cvlab.data import Frames, letterbox, mask_ignored, restore_boxes
from cvlab.metrics import evaluate
from cvlab.prepare import parse_boxes, prepare, select_records
from cvlab.submission import write_submission


@pytest.fixture
def competition(tmp_path):
    root = tmp_path / "data"
    for split, clips in [("train", ["a", "b", "c"]), ("val", ["d"])]:
        for clip in clips:
            for folder in ("images", "labels", "ignore"):
                (root / split / folder / clip).mkdir(parents=True, exist_ok=True)
            for frame in range(1, 3):
                Image.new("RGB", (173, 97), "white").save(
                    root / split / "images" / clip / f"{frame:07}.jpg"
                )
                (root / split / "labels" / clip / f"{frame:07}.txt").write_text(
                    "0 0.25 0.5 0.1 0.2\n7 0.75 0.5 0.2 0.2\n" if frame == 1 else ""
                )
                (root / split / "ignore" / clip / f"{frame:07}.txt").write_text(
                    "0.5 0.9 0.4 0.1\n"
                )
    path = root / "test/images/z/0000001.jpg"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (173, 97), "white").save(path)
    (root / "test_images.csv").write_text(
        "image_id,file_name,clip,frame,width,height\nz_0000001,test/images/z/0000001.jpg,z,1,173,97\n"
    )
    (root / "sample_submission.csv").write_text(
        "image_id,PredictionString\nz_0000001,\n"
    )
    output = tmp_path / "prepared"
    prepare(root, output, workers=2, folds=3)
    yield root, output
    # Real-model checkpoints are large; do not retain GBs across pytest runs.
    for checkpoint in tmp_path.rglob("*.pt"):
        checkpoint.unlink()


def test_prepare_folds_and_class_mapping(competition):
    root, output = competition
    records = read_json(output / "train.json")["images"]
    assert len(records) == 6
    assert records[0]["labels"] == [0, 7]
    dataset = Frames(root, records, 128, training=True, flip=0)
    image, target, _ = dataset[0]
    assert image.shape == (3, 128, 128)
    assert target["labels"].tolist() == [1, 8]
    _, empty, _ = dataset[1]
    assert empty["boxes"].shape == (0, 4)
    covered = set()
    for fold in range(3):
        config = {
            "data": {
                "prepared": str(output),
                "split": "fold",
                "fold": fold,
                "frame_step": 1,
            }
        }
        train, val = select_records(config, "train"), select_records(config, "val")
        assert not {r["clip"] for r in train} & {r["clip"] for r in val}
        assert not covered & {r["id"] for r in val}
        covered.update(r["id"] for r in val)
    assert covered == {r["id"] for r in records}


def test_round_trip_odd_letterbox():
    boxes = [[0, 0, 173, 97], [31, 9, 42, 37]]
    image = Image.new("RGB", (173, 97))
    _, transformed, geometry = letterbox(image, boxes, 128)
    restored = restore_boxes(torch.tensor(transformed), geometry)
    np.testing.assert_allclose(restored.numpy(), boxes, atol=2e-5)


def test_ignore_mask_preserves_targets():
    image = Image.new("RGB", (30, 30), "white")
    masked = mask_ignored(image, [[0, 0, 20, 20]], [[5, 5, 10, 10]])
    assert masked.getpixel((2, 2)) == (114, 114, 114)
    assert masked.getpixel((7, 7)) == (255, 255, 255)


def test_bad_labels_fail(tmp_path):
    file = tmp_path / "bad.txt"
    for line in ["8 .5 .5 .1 .1", "0 nan .5 .1 .1", "0 .5 .5 0 .1", "0 .5 .5 .1"]:
        file.write_text(line)
        with pytest.raises(ValueError):
            parse_boxes(file, 100, 100)


def test_coco_crowd_and_empty_predictions():
    record = {
        "id": 1,
        "width": 200,
        "height": 200,
        "boxes": [[10, 10, 30, 30]],
        "labels": [0],
        "ignore_boxes": [[100, 100, 180, 180]],
    }
    predictions = [
        {"image_id": 1, "category_id": 1, "bbox": [110, 110, 20, 20], "score": 0.99},
        {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},
    ]
    assert evaluate([record], predictions)["coco100"]["mAP"] == pytest.approx(1)
    assert evaluate([record], [])["coco500"]["mAP"] == 0
    bad = copy.deepcopy(predictions)
    bad[0]["bbox"] = [50, 50, 20, 20]
    assert evaluate([record], bad)["coco100"]["mAP"] < 1


def test_oof_requires_complete_disjoint_folds(competition, tmp_path, monkeypatch):
    import sys
    from cvlab.cli import main
    from cvlab.common import digest, write_json

    _, prepared = competition
    files = []
    for fold in range(3):
        config = {"data": {"prepared": str(prepared), "split": "fold", "fold": fold}}
        records = select_records(config, "val")
        predictions = []
        for record in records:
            for box, label in zip(record["boxes"], record["labels"]):
                x1, y1, x2, y2 = box
                predictions.append(
                    {
                        "image_id": record["id"],
                        "category_id": label + 1,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": 0.99,
                    }
                )
        path = tmp_path / f"fold{fold}.json"
        write_json(
            path,
            {
                "split": "fold",
                "fold": fold,
                "model_config": {"name": "test"},
                "protocol_fingerprint": "same-protocol",
                "record_ids": [r["id"] for r in records],
                "data_fingerprint": digest(records),
                "predictions": predictions,
            },
        )
        files.append(str(path))
    output = tmp_path / "oof.json"
    prefix = [
        "cvlab",
        "oof",
        "--manifest",
        str(prepared / "train.json"),
        "--out",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", prefix + files)
    main()
    assert read_json(output)["metrics"]["coco100"]["mAP"] == pytest.approx(1.0)
    monkeypatch.setattr(sys, "argv", prefix + files[:-1])
    with pytest.raises(ValueError, match="cover"):
        main()
    monkeypatch.setattr(sys, "argv", prefix + files + files[:1])
    with pytest.raises(ValueError, match="distinct"):
        main()


def test_submission_mapping_clipping_and_order(tmp_path):
    records = [
        {"id": 1, "submission_id": "a", "width": 100, "height": 50},
        {"id": 2, "submission_id": "b", "width": 100, "height": 50},
    ]
    sample, out = tmp_path / "sample.csv", tmp_path / "out.csv"
    sample.write_text("image_id,PredictionString\nb,\na,\n")
    predictions = [
        {"image_id": 1, "category_id": 8, "bbox": [-5, 5, 20, 10], "score": 0.9}
    ]
    write_submission(records, predictions, sample, out)
    with out.open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0] == {"image_id": "b", "PredictionString": ""}
    parts = rows[1]["PredictionString"].split()
    assert int(parts[0]) == 7
    assert list(map(float, parts[2:])) == [0, 5, 15, 10]
    many = predictions * 501
    write_submission(records, many, sample, out)
    with out.open() as stream:
        assert (
            len(list(csv.DictReader(stream))[1]["PredictionString"].split()) == 500 * 6
        )


def test_submission_without_template_uses_test_manifest_order(tmp_path):
    records = [
        {"id": 2, "submission_id": "b", "width": 100, "height": 50},
        {"id": 1, "submission_id": "a", "width": 100, "height": 50},
    ]
    output = tmp_path / "submission.csv"
    write_submission(records, [], tmp_path / "missing-template.csv", output)
    with output.open() as stream:
        rows = list(csv.DictReader(stream))
    assert [r["image_id"] for r in rows] == ["b", "a"]
    assert all(r["PredictionString"] == "" for r in rows)


def test_training_checkpoint_evaluation_submission(competition, tmp_path):
    from cvlab.common import load_config
    from cvlab.engine import load_trained, predict, setup, train

    root, prepared = competition
    config = load_config(Path(__file__).parents[1] / "configs/fcos.yaml")
    config["output"] = str(tmp_path / "run")
    config["data"].update(root=str(root), prepared=str(prepared), split="official")
    config["model"].update(pretrained=False, image_size=128, max_detections=20)
    config["runtime"].update(device="cpu", precision="fp32", cpu_threads=2)
    config["loader"].update(workers=0, batch_size=2, pin_memory=False)
    config["evaluation"]["warmup_batches"] = 0
    config["train"].update(epochs=1, max_batches_per_epoch=2, max_hours=1)
    train(config)
    output = Path(config["output"])
    assert (output / "best.pt").is_file()
    assert read_json(output / "best_metrics.json")["train_batches"] == 2
    device = setup(config)
    model = load_trained(output / "best.pt", config, device)
    test = select_records(config, "test")
    predictions, timing = predict(model, config, test, device)
    assert timing["images"] == 1
    write_submission(
        test, predictions, root / "sample_submission.csv", output / "submission.csv"
    )
    train(config, resume=output / "last.pt")  # Completed run: no extra epochs appended.
    assert len((output / "history.jsonl").read_text().splitlines()) == 1
    changed = copy.deepcopy(config)
    changed["data"]["horizontal_flip"] = 0
    with pytest.raises(ValueError, match="protocol"):
        train(changed, resume=output / "last.pt")
    assert read_json(output / "config.json")["data"]["horizontal_flip"] == 0.5


def test_resume_matches_uninterrupted_training(competition, tmp_path, monkeypatch):
    from cvlab.common import load_config
    from cvlab import engine

    root, prepared = competition
    config = load_config(Path(__file__).parents[1] / "configs/fcos.yaml")
    config["data"].update(root=str(root), prepared=str(prepared), split="official")
    config["model"].update(pretrained=False, image_size=128, max_detections=20)
    config["runtime"].update(device="cpu", precision="fp32", cpu_threads=2)
    config["loader"].update(workers=0, batch_size=2, pin_memory=False)
    config["evaluation"]["warmup_batches"] = 0
    config["train"].update(epochs=2, max_batches_per_epoch=1, max_hours=1)
    config["output"] = str(tmp_path / "uninterrupted")
    engine.train(config)
    full = torch.load(
        Path(config["output"]) / "last.pt", map_location="cpu", weights_only=False
    )
    config["output"] = str(tmp_path / "interrupted")
    real_predict, calls = engine.predict, []

    def interrupt_second_validation(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("simulated runtime disconnection")
        return real_predict(*args, **kwargs)

    monkeypatch.setattr(engine, "predict", interrupt_second_validation)
    with pytest.raises(RuntimeError, match="simulated"):
        engine.train(config)
    monkeypatch.setattr(engine, "predict", real_predict)
    engine.train(config, resume=Path(config["output"]) / "last.pt")
    resumed = torch.load(
        Path(config["output"]) / "last.pt", map_location="cpu", weights_only=False
    )
    for key in full["model"]:
        torch.testing.assert_close(
            full["model"][key], resumed["model"][key], rtol=0, atol=0
        )
    assert full["scheduler"] == resumed["scheduler"]
    assert full["epoch"] == resumed["epoch"] == 1


@pytest.mark.parametrize(
    "name,loss",
    [
        ("fasterrcnn", None),
        ("retinanet", "l1"),
        ("retinanet", "giou"),
        ("fcos", "giou"),
        ("fcos", "giou_l1"),
        ("rtdetr", "giou"),
        ("rtdetr", "giou_l1"),
    ],
)
def test_real_models_forward_backward_and_reload(name, loss):
    from cvlab.models import build_model

    torch.set_num_threads(2)
    torch.manual_seed(42)
    config = {
        "name": name,
        "pretrained": False,
        "image_size": 128,
        "max_detections": 20,
        "score_threshold": 0.001,
        "num_queries": 20,
        "num_denoising": 0,
    }
    if loss:
        config["box_loss"] = loss
    model = build_model(config)
    images = [torch.rand(3, 128, 128), torch.rand(3, 128, 128)]
    targets = [
        {
            "boxes": torch.tensor([[20.0, 30.0, 45.0, 65.0]]),
            "labels": torch.tensor([1]),
        },
        {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.int64)},
    ]
    model.train()
    losses = model.loss(images, targets)
    loss_tensor = sum(losses.values())
    assert torch.isfinite(loss_tensor)
    loss_tensor.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    model.eval()
    with torch.inference_mode():
        first = model.predict(images)
    state, metadata = model.state_dict(), model.export_metadata()
    restored = build_model(config, metadata)
    restored.load_state_dict(state)
    restored.eval()
    with torch.inference_mode():
        second = restored.predict(images)
    for a, b in zip(first, second):
        assert ((a["labels"] >= 1) & (a["labels"] <= 8)).all()
        torch.testing.assert_close(a["boxes"], b["boxes"])
        torch.testing.assert_close(a["scores"], b["scores"])


@pytest.mark.parametrize(
    "name,factory,weights",
    [
        ("fasterrcnn", "fasterrcnn_resnet50_fpn", "FasterRCNN_ResNet50_FPN_Weights"),
        ("retinanet", "retinanet_resnet50_fpn", "RetinaNet_ResNet50_FPN_Weights"),
        ("fcos", "fcos_resnet50_fpn", "FCOS_ResNet50_FPN_Weights"),
    ],
)
def test_pretrained_topology_restored_without_download(
    name, factory, weights, monkeypatch
):
    from torchvision.models import detection
    from torchvision.ops import FrozenBatchNorm2d
    from cvlab.models import build_model

    torch.set_num_threads(2)
    # Mock only downloaded weight values; exercise the real pretrained factory's
    # topology/frozen parameter behavior and the independent offline restore path.
    source = getattr(detection, factory)(
        weights=None, weights_backbone=None
    ).state_dict()
    monkeypatch.setattr(
        getattr(detection, weights), "get_state_dict", lambda *args, **kwargs: source
    )
    config = {"name": name, "pretrained": True, "image_size": 128, "max_detections": 20}
    initial = build_model(config).eval()
    restored = build_model(config, initialize_pretrained=False).eval()
    restored.load_state_dict(initial.state_dict())
    assert isinstance(restored.core.backbone.body.bn1, FrozenBatchNorm2d)
    assert {n for n, p in initial.named_parameters() if p.requires_grad} == {
        n for n, p in restored.named_parameters() if p.requires_grad
    }
    images = [torch.rand(3, 128, 128)]
    with torch.inference_mode():
        first, second = initial.predict(images), restored.predict(images)
    for a, b in zip(first, second):
        torch.testing.assert_close(a["boxes"], b["boxes"])
        torch.testing.assert_close(a["scores"], b["scores"])
