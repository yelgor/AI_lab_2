import copy
import gzip
import json

import pytest
import torch

from cvlab import class_ensemble as ensemble
from cvlab.common import CLASSES, digest, read_json, write_json
from cvlab.yolo_training import common_config
from test_pipeline import competition  # noqa: F401


def make_cache(
    path,
    records,
    backend,
    predictions,
    training="same_train",
    pipeline=None,
    checkpoint=None,
):
    path.mkdir(parents=True, exist_ok=True)
    with gzip.open(path / "frames.jsonl.gz", "wt") as stream:
        for record in records:
            stream.write(
                json.dumps(
                    {"image_id": record["id"], "predictions": predictions(record)}
                )
                + "\n"
            )
    value = {
        "version": 1,
        "classes": CLASSES,
        "frames": len(records),
        "identity": {
            "checkpoint_sha256": checkpoint
            or (ensemble.C1_SHA if backend == "c1" else "yolo_sha"),
            "data_fingerprint": digest(records),
            "training_fingerprint": training,
            "pipeline": pipeline or {"backend": backend},
            "split": "test" if "submission_id" in records[0] else "val",
        },
        "predictions_sha256": ensemble.sha(path / "frames.jsonl.gz"),
    }
    write_json(path / "cache.json", value)
    return value


def predictions(record, classes):
    return [
        {
            "image_id": record["id"],
            "category_id": i + 1,
            "score": 0.9,
            "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
        }
        for i, box in enumerate(record["boxes"])
        if i + 1 in classes
    ]


def test_real_common_metrics_show_complementary_classes_and_export(
    competition, tmp_path, monkeypatch  # noqa: F811
):  # noqa: F811
    root, prepared = competition
    config = common_config(
        root, prepared, tmp_path / "run", image_size=64, device="cpu", workers=0
    )
    original = read_json(prepared / "val.json")["images"][0]
    validation = []
    for i in range(7):
        record = copy.deepcopy(original)
        record.update(
            id=100 + i,
            sequence_id=f"heldout{i}",
            labels=list(range(8)),
            boxes=[[1 + j * 20, 20, 11 + j * 20, 30] for j in range(8)],
            ignore_boxes=[],
        )
        validation.append(record)
    write_json(prepared / "val.json", {"images": validation})
    c1, yolo = tmp_path / "c1_val", tmp_path / "yolo_val"
    make_cache(c1, validation, "c1", lambda r: predictions(r, {5, 7}))
    make_cache(yolo, validation, "yolo", lambda r: predictions(r, {1, 2, 3, 4, 6, 8}))
    report_path = tmp_path / "comparison.json"
    report = ensemble.compare(c1, yolo, config, report_path)
    assert report["accepted"] and report["records"] == 7
    aps = {
        key: value["metrics"]["coco100"]["mAP"]
        for key, value in report["cases"].items()
    }
    assert aps["class_route"] > 0.99 and aps["class_route"] > max(
        aps["c1"], aps["yolo"]
    )
    assert report["route"]["truck"] == report["route"]["bus"] == "c1"
    assert report["route"]["tricycle"] == report["route"]["motor"] == "yolo"
    with monkeypatch.context() as patch:
        patch.setattr("cvlab.metrics.evaluate", lambda *a, **kw: pytest.fail("Completed CPU comparisons must be reused"))
        assert ensemble.compare(c1, yolo, config, report_path) == report
    test = read_json(prepared / "test.json")["images"]

    def detections(r):
        return [
            {
                "image_id": r["id"],
                "category_id": c,
                "score": 0.8,
                "bbox": [5.0, 5.0, 10.0, 10.0],
            }
            for c in range(1, 9)
        ]

    c1_test, yolo_test = tmp_path / "c1_test", tmp_path / "yolo_test"
    make_cache(c1_test, test, "c1", detections)
    make_cache(yolo_test, test, "yolo", detections)
    csv_path = tmp_path / "ensemble.csv"
    checked = ensemble.export(c1_test, yolo_test, config, report_path, csv_path)
    assert checked["valid"] and checked["matches_template_order"]
    assert checked["images"] == 1 and checked["detections"] == 8
    assert read_json(csv_path.with_suffix(".csv.metadata.json"))["public_score"] is None
    wrong = copy.deepcopy(report)
    wrong["route"]["bus"] = "yolo"
    write_json(tmp_path / "changed_route.json", wrong)
    with pytest.raises(ValueError, match="Routing/evaluator changed"):
        ensemble.export(c1_test, yolo_test, config, tmp_path / "changed_route.json", tmp_path / "wrong_route.csv")
    meta = read_json(yolo_test / "cache.json")
    meta["identity"]["checkpoint_sha256"] = "another_epoch"
    write_json(yolo_test / "cache.json", meta)
    with pytest.raises(ValueError, match="Test weights/inference"):
        ensemble.export(c1_test, yolo_test, config, report_path, tmp_path / "wrong.csv")


def test_cache_stream_rejects_corruption_missing_frames_and_changed_cohort(tmp_path):
    records = [{"id": i, "width": 100, "height": 100} for i in (1, 2)]
    directory = tmp_path / "cache"
    make_cache(directory, records, "c1", lambda r: [])
    assert list(ensemble.stream_frames(directory, records)) == [[], []]
    with pytest.raises(ValueError, match="IDs/order"):
        list(ensemble.stream_frames(directory, records[::-1]))
    with gzip.open(directory / "frames.jsonl.gz", "wt") as stream:
        stream.write(json.dumps({"image_id": 1, "predictions": []}) + "\n")
    with pytest.raises(ValueError, match="checksum"):
        ensemble.metadata(directory)
    with pytest.raises(ValueError, match="Incomplete"):
        list(ensemble.stream_frames(directory, records))


def test_review_retains_class_size_gain_and_evidence_guards():
    decisions = {key: {"pooled_gain": .01, "reasons": ["family_lower_bound_not_positive"]}
                 for key in ("c1", "yolo")}
    assert ensemble.review_eligible(decisions)
    for reason in ("class_regression", "small_object_regression", "too_few_families", "missing_or_incompatible_class_evidence"):
        bad = copy.deepcopy(decisions)
        bad["yolo"]["reasons"].append(reason)
        assert not ensemble.review_eligible(bad)
    decisions["c1"]["pooled_gain"] = .0005
    assert not ensemble.review_eligible(decisions)


def test_snapshot_keeps_source_and_independent_frozen_bytes(tmp_path):
    source = tmp_path / "best.pt"
    source.write_bytes(b"learned_weights")
    frozen = ensemble.snapshot(source, tmp_path / "frozen")
    source.write_bytes(b"new_epoch")
    assert (
        frozen.read_bytes() == b"learned_weights"
        and source.read_bytes() == b"new_epoch"
    )


def test_cache_inference_and_reuse_do_not_repeat_model_work(
    competition, tmp_path, monkeypatch  # noqa: F811
):  # noqa: F811
    from cvlab import engine

    class Detector(torch.nn.Module):
        def predict(self, images):
            return [
                {
                    "boxes": torch.tensor([[0.0, 0.0, 64.0, 64.0]]),
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([5]),
                }
                for _ in images
            ]

    root, prepared = competition
    config = common_config(
        root, prepared, tmp_path / "run", image_size=64, device="cpu", workers=0
    )
    config["model"]["name"] = "rtdetr"
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"test_control")
    monkeypatch.setattr(ensemble, "C1_SHA", ensemble.sha(checkpoint))
    monkeypatch.setattr(engine, "load_trained", lambda *a: Detector())
    out = tmp_path / "cache"
    result = ensemble.cache_predictions(checkpoint, config, out, "c1")
    assert result["frames"] == 2
    assert (
        len(list(ensemble.stream_frames(out, ensemble.records_for(config, "val")))) == 2
    )
    monkeypatch.setattr(
        engine,
        "setup",
        lambda *a: pytest.fail("Cache hit must not initialize another model"),
    )
    assert (
        ensemble.cache_predictions(checkpoint, config, out, "c1")["identity"]
        == result["identity"]
    )
    changed = copy.deepcopy(config)
    changed["model"]["image_size"] = 128
    with pytest.raises(ValueError, match="different checkpoint/cohort/pipeline"):
        ensemble.cache_predictions(checkpoint, changed, out, "c1")


def test_route_caps_after_combining_and_rejected_report_cannot_export(tmp_path):
    def items(category):
        return [{"category_id": category, "score": 0.9 - i / 10000} for i in range(500)]

    left, right = items(5), items(1)
    old = copy.deepcopy((left, right))
    combined = ensemble.route_frame(left, right, "class_route")
    assert len(combined) == 500 and (left, right) == old
    path = tmp_path / "report.json"
    write_json(path, {"protocol": "fixed-truck-bus-route-v1", "accepted": False})
    with pytest.raises(ValueError, match="No accepted"):
        ensemble.export("unused", "unused", {}, path, tmp_path / "bad.csv")
    with pytest.raises(ValueError, match="No accepted"):
        ensemble.export("unused", "unused", {}, path, tmp_path / "bad_review.csv", manual_review=True)
