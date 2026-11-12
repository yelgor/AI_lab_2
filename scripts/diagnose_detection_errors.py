"""Compare native-resolution detection errors on the same official validation set."""

import argparse
import copy
import csv
import fcntl
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from cvlab.common import CLASSES, read_json
from cvlab.postprocess_experiments import stream_array


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def ious(box, boxes):
    if not boxes:
        return np.empty(0, dtype=np.float64)
    a = np.asarray(box, dtype=np.float64)
    b = np.asarray(boxes, dtype=np.float64)
    inter_wh = np.maximum(0, np.minimum(a[2:], b[:, 2:]) - np.maximum(a[:2], b[:, :2]))
    inter = inter_wh[:, 0] * inter_wh[:, 1]
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-12)


def load_predictions(path):
    by_image = defaultdict(list)
    metadata = {}
    for prediction in stream_array(path, "predictions", metadata):
        # Tuples keep the two large prediction sets compact in RAM.
        by_image[prediction["image_id"]].append(
            (int(prediction["category_id"]), float(prediction["score"]), prediction["bbox"])
        )
    return by_image, metadata


def evaluate_errors(records, predictions, threshold):
    counts = defaultdict(lambda: defaultdict(float))
    for record in records:
        image_id = record["id"]
        family = record["sequence_id"].split("_")[0]
        gt_by_class = defaultdict(list)
        for box, label in zip(record["boxes"], record["labels"]):
            gt_by_class[label + 1].append(box)
        matched = {label: set() for label in gt_by_class}
        ignored = record.get("ignore_boxes", [])
        dets = [p for p in predictions.get(image_id, ()) if p[1] >= threshold]
        dets.sort(key=lambda p: p[1], reverse=True)
        for cls, _score, raw_box in dets:
            box = xywh_to_xyxy(raw_box)
            same = gt_by_class.get(cls, [])
            overlap = ious(box, same)
            best_same = float(overlap.max()) if overlap.size else 0.0
            unmatched = [i for i in range(len(same)) if i not in matched.get(cls, set())]
            best_unmatched = max((float(overlap[i]) for i in unmatched), default=0.0)
            key = ("all", "all", cls)
            family_key = ("family", family, cls)
            if best_unmatched >= 0.5:
                idx = max(unmatched, key=lambda i: overlap[i])
                matched.setdefault(cls, set()).add(idx)
                for bucket in (key, family_key):
                    counts[bucket]["tp"] += 1
                    counts[bucket]["matched_iou_sum"] += float(overlap[idx])
                continue
            pred_area = max(1e-12, box[2] - box[0]) * max(1e-12, box[3] - box[1])
            ignored_overlap = 0.0
            for ign in ignored:
                inter = (max(0.0, min(box[2], ign[2]) - max(box[0], ign[0])) *
                         max(0.0, min(box[3], ign[3]) - max(box[1], ign[1])))
                ignored_overlap = max(ignored_overlap, inter / pred_area)
            if ignored_overlap >= 0.5:
                continue
            other_best = max((float(ious(box, boxes).max()) for label, boxes in gt_by_class.items() if label != cls and boxes), default=0.0)
            if best_same >= 0.5:
                reason = "duplicate"
            elif other_best >= 0.5:
                reason = "wrong_class"
            elif best_same >= 0.1:
                reason = "localization"
            else:
                reason = "background"
            for bucket in (key, family_key):
                counts[bucket]["fp"] += 1
                counts[bucket][reason] += 1

        for cls, boxes in gt_by_class.items():
            missed = len(boxes) - len(matched.get(cls, set()))
            for bucket in (("all", "all", cls), ("family", family, cls)):
                counts[bucket]["gt"] += len(boxes)
                counts[bucket]["fn"] += missed
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--infer-source", action="store_true")
    args = parser.parse_args()
    work, source, candidate, output = map(Path.resolve, (args.work, args.source, args.candidate, args.output))
    source_predictions = work / "diagnostics" / "c1_source_native_predictions.json"
    if args.infer_source and not source_predictions.exists():
        with (work / ".local_gpu_queue.lock").open("a") as lease:
            print("Waiting for shared GPU lease before source inference", flush=True)
            fcntl.flock(lease, fcntl.LOCK_EX)
            import torch
            from cvlab.engine import load_trained, predict, setup
            from cvlab.prepare import select_records
            cfg = read_json(source / "config.json")
            cfg = copy.deepcopy(cfg)
            cfg["data"]["split"] = "official"
            cfg["loader"].update(batch_size=1, workers=0, pin_memory=False, persistent_workers=False)
            cfg["runtime"].update(device="cuda", precision="fp16", cpu_threads=2)
            records = select_records(cfg, "val")
            device = setup(cfg)
            model = load_trained(source / "best.pt", cfg, device)
            pred, timing = predict(model, cfg, records, device)
            source_predictions.parent.mkdir(parents=True, exist_ok=True)
            temp = source_predictions.with_suffix(".tmp")
            with temp.open("w") as stream:
                json.dump({"record_ids": [r["id"] for r in records], "timing": timing,
                           "predictions": pred}, stream, separators=(",", ":"), allow_nan=False)
            temp.replace(source_predictions)
            del pred, model
            torch.cuda.empty_cache()
            print(f"Saved source predictions: {source_predictions}; {timing}", flush=True)
    if not source_predictions.is_file():
        raise FileNotFoundError(f"Source predictions required: {source_predictions}")

    prepared = read_json(source / "config.json")["data"]["prepared"]
    records = read_json(Path(prepared) / "val.json")["images"]
    source_pred, source_meta = load_predictions(source_predictions)
    candidate_pred, candidate_meta = load_predictions(candidate)
    if source_meta.get("record_ids") != candidate_meta.get("record_ids"):
        raise ValueError("Source/candidate prediction image IDs differ")
    if source_meta["record_ids"] != [r["id"] for r in records]:
        raise ValueError("Predictions do not cover the prepared official validation in order")

    rows = []
    for threshold in (0.05, 0.25):
        old = evaluate_errors(records, source_pred, threshold)
        new = evaluate_errors(records, candidate_pred, threshold)
        for cls, name in enumerate(CLASSES, 1):
            a, b = old[("all", "all", cls)], new[("all", "all", cls)]
            row = {"scope": "overall", "family": "all", "class": name, "score_threshold": threshold}
            for metric in ("gt", "tp", "fn", "fp", "duplicate", "wrong_class", "localization", "background"):
                row[f"source_{metric}"] = int(a.get(metric, 0))
                row[f"candidate_{metric}"] = int(b.get(metric, 0))
                row[f"delta_{metric}"] = int(b.get(metric, 0) - a.get(metric, 0))
            row["source_matched_iou"] = a.get("matched_iou_sum", 0) / max(a.get("tp", 0), 1)
            row["candidate_matched_iou"] = b.get("matched_iou_sum", 0) / max(b.get("tp", 0), 1)
            rows.append(row)
            for family in sorted({k[1] for k in old if k[0] == "family"}):
                x, y = old[("family", family, cls)], new[("family", family, cls)]
                if x.get("gt", 0) or y.get("gt", 0):
                    rows.append({"scope": "family", "family": family, "class": name,
                                 "score_threshold": threshold,
                                 "source_gt": int(x.get("gt", 0)), "candidate_gt": int(y.get("gt", 0)),
                                 "source_tp": int(x.get("tp", 0)), "candidate_tp": int(y.get("tp", 0)),
                                 "source_fn": int(x.get("fn", 0)), "candidate_fn": int(y.get("fn", 0)),
                                 "source_fp": int(x.get("fp", 0)), "candidate_fp": int(y.get("fp", 0)),
                                 "source_duplicate": int(x.get("duplicate", 0)), "candidate_duplicate": int(y.get("duplicate", 0)),
                                 "source_wrong_class": int(x.get("wrong_class", 0)), "candidate_wrong_class": int(y.get("wrong_class", 0)),
                                 "source_localization": int(x.get("localization", 0)), "candidate_localization": int(y.get("localization", 0)),
                                 "source_background": int(x.get("background", 0)), "candidate_background": int(y.get("background", 0))})
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} diagnostic rows: {output}")


if __name__ == "__main__":
    main()
