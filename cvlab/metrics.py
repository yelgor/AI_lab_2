import contextlib
import copy
import io
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .common import CLASSES


def coco_ground_truth(records, category_ids=None):
    category_ids = (
        list(range(1, len(CLASSES) + 1)) if category_ids is None else category_ids
    )
    annotations, images = [], []
    for record in records:
        images.append(
            {"id": record["id"], "width": record["width"], "height": record["height"]}
        )
        for box, label in zip(record["boxes"], record["labels"]):
            if label + 1 not in category_ids:
                continue
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
            for label in category_ids:
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
        "categories": [{"id": i, "name": CLASSES[i - 1]} for i in category_ids],
    }
    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()
    return coco


def _class_precision(records, by_image, category_id, caps, batch_size):
    """Keep one class's compact matches; bound COCO objects/IoUs by image batch."""
    accumulator = COCOeval(iouType="bbox")
    accumulator.params.catIds = [category_id]
    accumulator.params.imgIds = [record["id"] for record in records]
    accumulator.params.maxDets = sorted({1, 10, *caps})
    by_area = [[] for _ in accumulator.params.areaRng]
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        gt = coco_ground_truth(batch, [category_id])
        # loadRes adds fields to its input. Copy only this small batch, leaving
        # the full prediction list unchanged and without its COCO-only fields.
        selected = [
            dict(p)
            for record in batch
            for p in by_image.get((category_id, record["id"]), ())
        ]
        if selected:
            detections = gt.loadRes(selected)
        else:
            detections = COCO()
            detections.dataset = {**gt.dataset, "annotations": []}
            detections.createIndex()
        matcher = COCOeval(gt, detections, "bbox")
        matcher.params.catIds = [category_id]
        matcher.params.maxDets = accumulator.params.maxDets
        matcher.evaluate()
        for area, entries in enumerate(by_area):
            for match in matcher.evalImgs[area * len(batch) : (area + 1) * len(batch)]:
                if match is None:
                    entries.append(None)
                else:
                    # accumulate uses match presence, not annotation IDs.
                    entries.append(
                        {
                            "dtScores": np.asarray(match["dtScores"], dtype=np.float64),
                            "dtMatches": match["dtMatches"] != 0,
                            "dtIgnore": match["dtIgnore"].astype(bool, copy=True),
                            "gtIgnore": np.asarray(match["gtIgnore"], dtype=bool),
                        }
                    )
        del matcher, detections, gt, selected
    accumulator.evalImgs = [entry for entries in by_area for entry in entries]
    accumulator._paramsEval = copy.deepcopy(accumulator.params)
    accumulator.accumulate()
    return (
        accumulator.eval["precision"],
        accumulator.params.maxDets,
        accumulator.params.iouThrs,
    )


def evaluate(records, predictions, max_detections=(100, 500), *, image_batch_size=32):
    """Exact local COCO crowd AP with classwise, batched matching.

    Competition scorer must still be checked separately. AP is accumulated
    globally over all images, never averaged over image batches.
    """
    if not records:
        raise ValueError("Cannot evaluate empty validation set")
    record_ids = {r["id"] for r in records}
    if any(p["image_id"] not in record_ids for p in predictions):
        raise ValueError("Prediction refers to an unknown image")
    caps = sorted(set(max_detections))
    if not caps or any(cap < 10 for cap in caps):
        raise ValueError("COCO max_detections must be >=10")
    if image_batch_size < 1:
        raise ValueError("image_batch_size must be >=1")
    records = sorted(records, key=lambda record: record["id"])
    by_image = defaultdict(list)
    for prediction in predictions:
        by_image[(prediction["category_id"], prediction["image_id"])].append(prediction)
    result = {}
    with contextlib.redirect_stdout(io.StringIO()):
        classes = []
        for category_id in range(1, len(CLASSES) + 1):
            precision, evaluated_caps, ious = _class_precision(
                records, by_image, category_id, caps, image_batch_size
            )
            classes.append(precision)
        all_precision = np.concatenate(classes, axis=2)
        for cap in caps:
            # COCOeval.summarize hardcodes 100 for its first AP statistic.
            # Read the precision tensor directly so dense500 is computed correctly.
            precision = all_precision[:, :, :, :, evaluated_caps.index(cap)]

            def average(array):
                valid = array[array > -1]
                return float(valid.mean()) if valid.size else None

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
