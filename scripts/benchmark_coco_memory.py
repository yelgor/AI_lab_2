"""Compare full and batched COCO evaluation in separate Linux processes.

python scripts/benchmark_coco_memory.py previous
python scripts/benchmark_coco_memory.py batched
"""

import argparse
import contextlib
import io
import json
import resource
import sys
import time
from pathlib import Path

from pycocotools.cocoeval import COCOeval

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cvlab.metrics import coco_ground_truth, evaluate  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["previous", "batched"])
    parser.add_argument("--images", type=int, default=256)
    args = parser.parse_args()
    records, predictions = [], []
    for image_id in range(1, args.images + 1):
        boxes = [
            [(j % 8) * 40, (j // 8) * 40, (j % 8) * 40 + 12, (j // 8) * 40 + 20]
            for j in range(32)
        ]
        records.append(
            {
                "id": image_id,
                "width": 4096,
                "height": 3072,
                "boxes": boxes,
                "labels": [j % 8 for j in range(32)],
                "ignore_boxes": [[2000, 2000, 2100, 2100]],
            }
        )
        for j in range(500):
            box = boxes[j] if j < 32 else [1000 + j * 2, 1000, 1012 + j * 2, 1020]
            x1, y1, x2, y2 = box
            category = j % 8 + 1 if j < 32 or j % 5 == 0 else 3
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": category,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": 0.9 - j / 1000,
                }
            )
    started = time.perf_counter()
    scores = {}
    with contextlib.redirect_stdout(io.StringIO()):
        if args.mode == "previous":
            gt = coco_ground_truth(records)
            detections = gt.loadRes(predictions)
            for cap in (100, 500):
                evaluator = COCOeval(gt, detections, "bbox")
                evaluator.params.maxDets = [1, 10, cap]
                evaluator.evaluate()
                evaluator.accumulate()
                precision = evaluator.eval["precision"][:, :, :, 0, -1]
                scores[cap] = float(precision[precision > -1].mean())
        else:
            result = evaluate(records, predictions)
            scores = {cap: result[f"coco{cap}"]["mAP"] for cap in (100, 500)}
    print(
        json.dumps(
            {
                "mode": args.mode,
                "images": len(records),
                "detections": len(predictions),
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024,
                "seconds": time.perf_counter() - started,
                "mAP": scores,
            }
        )
    )


if __name__ == "__main__":
    main()
