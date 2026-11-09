import contextlib
import copy
import io
from collections import defaultdict

import numpy as np
import pytest
from pycocotools.cocoeval import COCOeval

from cvlab import metrics


def dense_example():
    records, predictions = [], []
    for image_id in [7, 2, 9, 1, 5, 8]:
        category = image_id % 4
        record = {
            "id": image_id,
            "width": 1024,
            "height": 768,
            "boxes": [[10, 10, 25, 25], [50, 50, 100, 100], [200, 200, 350, 350]],
            "labels": [category] * 3,
            "ignore_boxes": [[400, 400, 600, 600]],
        }
        records.append(record)
        for box in record["boxes"]:
            x1, y1, x2, y2 = box
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": category + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": 0.5,
                }
            )
        predictions.extend(
            [
                {
                    "image_id": image_id,
                    "category_id": category + 1,
                    "bbox": [10, 10, 15, 15],
                    "score": 0.5,
                },
                {
                    "image_id": image_id,
                    "category_id": 8,
                    "bbox": [450, 450, 15, 15],
                    "score": 0.9,
                },
            ]
        )
    # True positives beyond rank 100 distinguish the two detection caps.
    for index in range(150):
        predictions.append(
            {"image_id": 7, "category_id": 4, "bbox": [650, 650, 10, 10], "score": 0.8}
        )
    return records, predictions


@pytest.mark.parametrize("batch_size", [1, 3, 32])
def test_batched_precision_equals_official_coco_at_each_cap(batch_size):
    records, predictions = dense_example()
    index = defaultdict(list)
    for prediction in predictions:
        index[(prediction["category_id"], prediction["image_id"])].append(prediction)
    with contextlib.redirect_stdout(io.StringIO()):
        classes = []
        for category in range(1, 9):
            precision, caps, _ = metrics._class_precision(
                sorted(records, key=lambda record: record["id"]),
                index,
                category,
                [100, 500],
                batch_size,
            )
            classes.append(precision)
        actual = np.concatenate(classes, axis=2)
        for cap in (100, 500):
            gt = metrics.coco_ground_truth(records)
            detections = gt.loadRes(copy.deepcopy(predictions))
            reference = COCOeval(gt, detections, "bbox")
            reference.params.maxDets = [1, 10, cap]
            reference.evaluate()
            reference.accumulate()
            np.testing.assert_array_equal(
                actual[..., caps.index(cap)], reference.eval["precision"][..., -1]
            )


def test_evaluation_preserves_predictions_and_bounds_coco_batches(monkeypatch):
    records, predictions = dense_example()
    original = copy.deepcopy(predictions)
    real = metrics.coco_ground_truth
    calls = []

    def limited(batch, category_ids=None):
        calls.append(len(batch))
        assert len(batch) <= 2
        assert len(category_ids) == 1
        return real(batch, category_ids)

    monkeypatch.setattr(metrics, "coco_ground_truth", limited)
    result = metrics.evaluate(records, predictions, image_batch_size=2)
    assert predictions == original
    assert len(calls) == 8 * 3
    assert result["coco500"]["mAP"] > result["coco100"]["mAP"]
    assert result["coco500"]["per_class"]["motor"] is None


def test_batched_empty_predictions_and_parameter_errors():
    records, _ = dense_example()
    result = metrics.evaluate(records, [], image_batch_size=1)
    assert result["coco100"]["mAP"] == result["coco500"]["mAP"] == 0
    with pytest.raises(ValueError, match="image_batch_size"):
        metrics.evaluate(records, [], image_batch_size=0)
