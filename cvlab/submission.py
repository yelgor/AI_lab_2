import csv
import math
from collections import defaultdict
from pathlib import Path


def write_submission(
    records, predictions, sample, output, threshold=0.001, max_detections=500
):
    if not 0 <= threshold <= 1 or not 1 <= max_detections <= 500:
        raise ValueError("Invalid submission threshold or detection cap")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate internal image IDs")
    by_id = {r["id"]: r for r in records}
    by_submission = {r["submission_id"]: r for r in records}
    if len(by_submission) != len(records):
        raise ValueError("Duplicate submission IDs")
    grouped = defaultdict(list)
    for prediction in predictions:
        record = by_id[prediction["image_id"]]
        score, category = float(prediction["score"]), int(prediction["category_id"]) - 1
        x, y, width, height = map(float, prediction["bbox"])
        if (
            not all(map(math.isfinite, [score, x, y, width, height]))
            or not 0 <= score <= 1
            or not 0 <= category < 8
        ):
            raise ValueError("Non-finite prediction or invalid class/score")
        x2, y2 = min(record["width"], x + width), min(record["height"], y + height)
        x, y = max(0.0, x), max(0.0, y)
        width, height = x2 - x, y2 - y
        if score >= threshold and width > 0 and height > 0:
            grouped[record["submission_id"]].append(
                (category, score, x, y, width, height)
            )
    if sample is not None and Path(sample).is_file():
        with Path(sample).open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["image_id", "PredictionString"]:
                raise ValueError("Unexpected sample_submission.csv columns")
            ordered_ids = [r["image_id"] for r in reader]
    else:
        # Test manifests retain the order and IDs from test_images.csv.
        ordered_ids = [r["submission_id"] for r in records]
    if len(set(ordered_ids)) != len(ordered_ids) or set(ordered_ids) != set(
        by_submission
    ):
        raise ValueError("Sample submission IDs do not exactly match test images")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id", "PredictionString"])
        for key in ordered_ids:
            detections = sorted(grouped[key], key=lambda p: p[1], reverse=True)[
                :max_detections
            ]
            formatted = [
                f"{c} {s:.7f} {x:.4f} {y:.4f} {w:.4f} {h:.4f}"
                for c, s, x, y, w, h in detections
            ]
            writer.writerow([key, " ".join(formatted)])
