"""CPU-only NMS experiments using streamed, previously saved validation predictions."""

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import digest, read_json, write_json
from .inference_experiments import balanced_subset
from .metrics import evaluate


def stream_array(path, key, metadata=None, chunk_size=65536):
    """Read one top-level JSON array incrementally; retain only small other fields."""
    metadata = {} if metadata is None else metadata
    decoder = json.JSONDecoder()
    buffer, position, eof = "", 0, False
    with Path(path).open(encoding="utf-8") as stream:

        def fill():
            nonlocal buffer, position, eof
            buffer = buffer[position:]
            position = 0
            chunk = stream.read(chunk_size)
            eof = not chunk
            buffer += chunk
            if len(buffer) > 8 * 1024**2:
                raise ValueError("A single JSON item exceeds the 8 MiB streaming limit")

        def peek():
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    return buffer[position]
                if eof:
                    return ""
                fill()

        def consume(expected):
            nonlocal position
            if peek() != expected:
                raise ValueError(f"Expected {expected!r} in {path}")
            position += 1

        def value():
            nonlocal position
            peek()
            while True:
                try:
                    item, end = decoder.raw_decode(buffer, position)
                    # A number might end at a chunk boundary before more digits.
                    incomplete_number = (
                        isinstance(item, (int, float))
                        and not isinstance(item, bool)
                        and end < len(buffer)
                        and buffer[end] not in " \r\n\t,]}:"
                    )
                    if (end == len(buffer) and not eof) or incomplete_number:
                        if eof:
                            raise ValueError(f"Invalid JSON number in {path}")
                        fill()
                        continue
                    position = end
                    return item
                except json.JSONDecodeError:
                    if eof:
                        raise ValueError(
                            f"Incomplete or invalid JSON in {path}"
                        ) from None
                    fill()

        consume("{")
        found, fields = False, set()
        if peek() != "}":
            while True:
                field = value()
                if not isinstance(field, str) or field in fields:
                    raise ValueError("Invalid or duplicate top-level JSON key")
                fields.add(field)
                consume(":")
                if field == key:
                    found = True
                    consume("[")
                    if peek() != "]":
                        while True:
                            yield value()
                            if peek() == "]":
                                break
                            consume(",")
                    consume("]")
                else:
                    metadata[field] = value()
                if peek() == "}":
                    break
                consume(",")
        consume("}")
        if peek() or not found:
            raise ValueError(
                f"Missing {key!r} array or trailing JSON content in {path}"
            )


def validation_subset(config, limit):
    prepared = Path(config["data"]["prepared"])
    mode = config["data"]["split"]
    if mode == "fold":
        manifest = prepared / "train.json"
        fold = read_json(prepared / "folds.json")["folds"][config["data"]["fold"]]
        sequences = set(fold["val_sequences"])
    elif mode == "official":
        manifest, sequences = prepared / "val.json", None
    else:
        raise ValueError("Expected fold or official validation")
    encoder = json.JSONEncoder(sort_keys=True)
    fingerprint = hashlib.sha256(b"[")
    candidates = []
    for record in stream_array(manifest, "images"):
        if sequences is not None and record["sequence_id"] not in sequences:
            continue
        if candidates:
            fingerprint.update(b", ")
        for chunk in encoder.iterencode(record):
            fingerprint.update(chunk.encode())
        candidates.append({"id": record["id"], "sequence_id": record["sequence_id"]})
    fingerprint.update(b"]")
    if not candidates:
        raise ValueError("Empty validation split")
    ids = {record["id"] for record in candidates}
    if len(ids) != len(candidates):
        raise ValueError("Duplicate validation image IDs")
    selected = {record["id"] for record in balanced_subset(candidates, limit)}
    records = [r for r in stream_array(manifest, "images") if r["id"] in selected]
    return records, ids, fingerprint.hexdigest()


def extra_nms(predictions, threshold):
    """Class-specific greedy NMS on xywh boxes; stable score ordering."""
    groups, result = defaultdict(list), []
    for prediction in predictions:
        groups[prediction["image_id"], prediction["category_id"]].append(prediction)
    for group in groups.values():
        boxes = np.asarray([p["bbox"] for p in group], dtype=np.float64)
        boxes[:, 2:] += boxes[:, :2]
        areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        order = np.argsort([-p["score"] for p in group], kind="stable")
        while order.size:
            chosen, rest = order[0], order[1:]
            result.append(group[chosen])
            intersection = np.maximum(
                0,
                np.minimum(boxes[chosen, 2:], boxes[rest, 2:])
                - np.maximum(boxes[chosen, :2], boxes[rest, :2]),
            ).prod(axis=1)
            union = areas[chosen] + areas[rest] - intersection
            order = rest[intersection / union <= threshold]
    return result


def run(prediction_file, config, output, max_images=192, thresholds=(0.3, 0.4, 0.5)):
    if max_images < 1 or any(not 0 < t <= 1 for t in thresholds):
        raise ValueError("Positive image limit and NMS thresholds in (0, 1] required")
    print("Selecting held-out frames from the streamed manifest", flush=True)
    records, validation_ids, full_fingerprint = validation_subset(config, max_images)
    selected = {r["id"] for r in records}
    predictions, metadata, counts = [], {}, defaultdict(int)
    print("Streaming saved predictions; GPU is not used", flush=True)
    for prediction in stream_array(prediction_file, "predictions", metadata):
        if prediction["image_id"] not in validation_ids:
            raise ValueError("Prediction contains an image outside validation")
        if prediction["image_id"] in selected:
            counts[prediction["image_id"]] += 1
            if counts[prediction["image_id"]] > 500:
                raise ValueError("More than 500 saved detections per image")
            values = [prediction["score"], *prediction["bbox"]]
            if len(values) != 5 or not all(math.isfinite(v) for v in values):
                raise ValueError("Invalid prediction values")
            if (
                min(prediction["bbox"][2:]) <= 0
                or not 1 <= prediction["category_id"] <= 8
                or not 0 <= prediction["score"] <= 1
            ):
                raise ValueError("Invalid box or internal category ID")
            predictions.append(prediction)
    if (
        len(metadata["record_ids"]) != len(validation_ids)
        or set(metadata["record_ids"]) != validation_ids
        or metadata["data_fingerprint"] != full_fingerprint
        or metadata["model_config"] != config["model"]
    ):
        raise ValueError(
            "Saved predictions do not match the validation data/model config"
        )
    report = {
        "source": str(Path(prediction_file).resolve()),
        "source_data_fingerprint": full_fingerprint,
        "subset_fingerprint": digest(records),
        "model_config": config["model"],
        "record_ids": [r["id"] for r in records],
        "full_validation_images": len(validation_ids),
        "subset": len(records) < len(validation_ids),
        "limitation": "Additional NMS cannot restore boxes discarded by original inference.",
        "experiments": [],
    }
    source_hash = hashlib.sha256()
    with Path(prediction_file).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            source_hash.update(chunk)
    report["source_sha256"] = source_hash.hexdigest()
    report["source_protocol_fingerprint"] = metadata.get("protocol_fingerprint")
    for threshold in (None, *thresholds):
        filtered = (
            predictions if threshold is None else extra_nms(predictions, threshold)
        )
        scores = evaluate(records, filtered, config["evaluation"]["max_detections"])
        case = {
            "additional_nms_iou": threshold,
            "detections": len(filtered),
            "metrics": scores,
        }
        report["experiments"].append(case)
        write_json(output, report)
        print(
            f"NMS={threshold}: mAP={scores['coco100']['mAP']}; AP_small={scores['coco100']['AP_small']}",
            flush=True,
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-images", default=192, type=int)
    parser.add_argument("--nms", nargs="+", type=float, default=[0.3, 0.4, 0.5])
    args = parser.parse_args()
    # Use the saved run config and suppress any environment override of its data root.
    config = read_json(args.run / "config.json")
    if os.environ.get("CVLAB_DATA_ROOT"):
        print("Using saved run config; CVLAB_DATA_ROOT is ignored", flush=True)
    run(args.run / "val_predictions.json", config, args.out, args.max_images, args.nms)


if __name__ == "__main__":
    main()
