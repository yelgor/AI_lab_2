import contextlib
import io

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .common import CLASSES


def coco_ground_truth(records):
    annotations, images = [], []
    for record in records:
        images.append(
            {"id": record["id"], "width": record["width"], "height": record["height"]}
        )
        for box, label in zip(record["boxes"], record["labels"]):
            x1, y1, x2, y2 = box
            annotations.append(
                {
                    "image_id": record["id"],
                    "category_id": label + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                }
            )
        # Ignore files have no class. COCO crowds are class-specific, so represent
        # each region in every class. No target objects are discarded beforehand.
        for box in record["ignore_boxes"]:
            x1, y1, x2, y2 = box
            for label in range(1, 9):
                annotations.append(
                    {
                        "image_id": record["id"],
                        "category_id": label,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": (x2 - x1) * (y2 - y1),
                        "iscrowd": 1,
                    }
                )
    for i, annotation in enumerate(annotations, 1):
        annotation["id"] = i
    dataset = {
        "info": {},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i + 1, "name": name} for i, name in enumerate(CLASSES)],
    }
    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()
    return coco


def evaluate(records, predictions, max_detections=(100, 500)):
    """Local COCO crowd protocol. Competition scorer must be checked separately."""
    if not records:
        raise ValueError("Cannot evaluate empty validation set")
    record_ids = {r["id"] for r in records}
    if any(p["image_id"] not in record_ids for p in predictions):
        raise ValueError("Prediction refers to an unknown image")
    result = {}
    with contextlib.redirect_stdout(io.StringIO()):
        gt = coco_ground_truth(records)
        if predictions:
            detections = gt.loadRes(predictions)
        else:
            detections = COCO()
            detections.dataset = {**gt.dataset, "annotations": []}
            detections.createIndex()
        for cap in sorted(set(max_detections)):
            if cap < 10:
                raise ValueError("COCO max_detections must be >=10")
            evaluator = COCOeval(gt, detections, "bbox")
            evaluator.params.maxDets = [1, 10, cap]
            evaluator.evaluate()
            evaluator.accumulate()
            # COCOeval.summarize hardcodes 100 for its first AP statistic.
            # Read the precision tensor directly so dense500 is computed correctly.
            precision = evaluator.eval["precision"][:, :, :, :, -1]

            def average(array):
                valid = array[array > -1]
                return float(valid.mean()) if valid.size else None

            ious = evaluator.params.iouThrs
            result[f"coco{cap}"] = {
                "mAP": average(precision[:, :, :, 0]),
                "AP50": average(precision[np.isclose(ious, 0.5), :, :, 0]),
                "AP75": average(precision[np.isclose(ious, 0.75), :, :, 0]),
                "AP_small": average(precision[:, :, :, 1]),
                "AP_medium": average(precision[:, :, :, 2]),
                "AP_large": average(precision[:, :, :, 3]),
                "per_class": {
                    name: average(precision[:, :, i, 0])
                    for i, name in enumerate(CLASSES)
                },
            }
    return result
