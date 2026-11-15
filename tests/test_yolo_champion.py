import hashlib
from unittest.mock import patch

import pytest

from cvlab import yolo_champion


def _view(checkpoint, split="val", *, flip=False):
    pipeline = {
        "backend": "yolo",
        "model": {"image_size": 1280, "nms_threshold": 0.6, "max_detections": 500},
    }
    if flip:
        pipeline["tta"] = {
            "adapter": yolo_champion.FLIP_ADAPTER,
            "adapter_sha256": hashlib.sha256(
                (
                    yolo_champion.Path(__file__).parents[1] / "cvlab/yolo_flip.py"
                ).read_bytes()
            ).hexdigest(),
            "nms_iou": 0.6,
        }
    return {
        "frames": 1,
        "identity": {
            "checkpoint_sha256": checkpoint,
            "data_fingerprint": "data",
            "training_fingerprint": "train",
            "pipeline": pipeline,
            "split": split,
        },
    }


def test_rejects_old_parent_report_protocol():
    with (
        patch.object(
            yolo_champion,
            "read_json",
            return_value={"protocol": "fixed-truck-bus-route-v1"},
        ),
        patch.object(
            yolo_champion, "sha", return_value=yolo_champion.CHAMPION_REPORT_SHA
        ),
        pytest.raises(ValueError, match="recorded champion report"),
    ):
        yolo_champion._parent("old-report.json", "old.csv")


def test_rejects_changed_validation_cohort_before_comparison():
    records = [{"id": 1}]
    parent = {"data_fingerprint": "different-cohort"}
    with (
        patch.object(yolo_champion, "_parent", return_value=(parent, {})),
        patch.object(yolo_champion, "records_for", return_value=records),
        pytest.raises(ValueError, match="cohort differs"),
    ):
        yolo_champion.compare_champion(
            "c1",
            "candidate-native",
            "candidate-flip",
            "champion-report",
            "champion.csv",
            {"data": {"split": "official"}},
            "out.json",
        )


def test_rejects_native_flip_and_val_test_checkpoint_mismatches():
    record = {"id": 1}
    native = _view("candidate-a")
    flipped = _view("candidate-b", flip=True)
    with patch.object(yolo_champion, "metadata", side_effect=[native, flipped]):
        with pytest.raises(ValueError, match="one full pipeline"):
            yolo_champion._yolo_views("native", "flip", [record], "val")

    val_view = _view("candidate-a")
    test_view = _view("candidate-b", split="test")
    assert not yolo_champion._same_checkpoint_pipeline(val_view, test_view)
