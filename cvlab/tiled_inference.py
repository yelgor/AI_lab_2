"""Full-frame plus overlapping-tile inference with bounded per-frame merging."""

import csv
import fcntl
import math
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import batched_nms

from .common import digest, read_json, write_json
from .data import letterbox, restore_boxes
from .engine import amp_context, synchronize
from .prepare import safe_path
from .submission import write_submission


@dataclass(frozen=True)
class TileOptions:
    tile_side: int | None = 768
    overlap: float = 0.2
    nms_iou: float = 0.5
    edge_margin: float = 2
    workers: int = 2

    def validate(self):
        if self.tile_side is not None and (
            not isinstance(self.tile_side, int)
            or isinstance(self.tile_side, bool)
            or self.tile_side < 64
        ):
            raise ValueError("Tile side must be >=64")
        if not 0 <= self.overlap <= 0.5 or not 0 < self.nms_iou <= 1:
            raise ValueError("Overlap must be 0..0.5 and NMS IoU in (0,1]")
        if self.edge_margin < 0 or not 0 <= self.workers <= 4:
            raise ValueError("Nonnegative edge margin and 0..4 workers required")


def axis_starts(length, side, overlap):
    if length <= side:
        return [0]
    count = math.ceil((length - side) / max(1, side * (1 - overlap))) + 1
    return [round(index * (length - side) / (count - 1)) for index in range(count)]


def regions(width, height, options):
    options.validate()
    result = [(0, 0, width, height)]
    if options.tile_side is None:
        return result
    side = options.tile_side
    for y in axis_starts(height, side, options.overlap):
        for x in axis_starts(width, side, options.overlap):
            region = (x, y, min(x + side, width), min(y + side, height))
            if region not in result:
                result.append(region)
                if len(result) > 64:
                    raise ValueError("More than 64 views per image; increase tile side")
    return result


class TileFrames(Dataset):
    def __init__(self, root, records, size, options):
        self.root, self.size, self.options = str(root), size, options
        # Workers never receive the full annotation lists.
        self.records = [
            {
                key: record[key]
                for key in ("id", "file_name", "width", "height", "submission_id")
                if key in record
            }
            for record in records
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(safe_path(self.root, record["file_name"])) as opened:
            image = opened.convert("RGB")
        if image.size != (record["width"], record["height"]):
            raise ValueError("Image size differs from manifest")
        tensors, metadata = [], []
        for rect in regions(*image.size, self.options):
            patch, _, geometry = letterbox(image.crop(rect), [], self.size)
            tensors.append(torch.from_numpy(np.array(patch).transpose(2, 0, 1).copy()))
            metadata.append(
                {
                    "rect": rect,
                    "geometry": geometry,
                    "global": rect == (0, 0, *image.size),
                }
            )
        return tensors, metadata, record


def single_frame(batch):
    return batch[0]


def decode_view(output, view, record, margin, threshold):
    boxes = restore_boxes(output["boxes"].detach().float().cpu(), view["geometry"])
    scores, labels = (
        output["scores"].detach().float().cpu(),
        output["labels"].detach().cpu(),
    )
    left, top, right, bottom = view["rect"]
    width, height = right - left, bottom - top
    predictions = []
    for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
        if (
            not all(map(math.isfinite, box + [score]))
            or not 0 <= score <= 1
            or not 1 <= label <= 8
        ):
            raise ValueError("Invalid tiled prediction")
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1 or score < threshold:
            continue
        # Interior cut boxes are covered by an overlapping view; keep detections
        # touching the original image boundary and all full-frame detections.
        if not view["global"] and (
            (left > 0 and x1 <= margin)
            or (top > 0 and y1 <= margin)
            or (right < record["width"] and x2 >= width - margin)
            or (bottom < record["height"] and y2 >= height - margin)
        ):
            continue
        predictions.append(
            {
                "image_id": record["id"],
                "category_id": int(label),
                "score": score,
                "bbox": [x1 + left, y1 + top, x2 - x1, y2 - y1],
            }
        )
    return predictions


def merge_frame(predictions, iou, cap=500):
    if not predictions:
        return []
    boxes = torch.tensor(
        [p["bbox"] for p in predictions], dtype=torch.float32, device="cpu"
    )
    boxes[:, 2:] += boxes[:, :2]
    scores = torch.tensor(
        [p["score"] for p in predictions], dtype=torch.float32, device="cpu"
    )
    labels = torch.tensor(
        [p["category_id"] for p in predictions], dtype=torch.int64, device="cpu"
    )
    keep = batched_nms(boxes, scores, labels, iou)[:cap].tolist()
    return [predictions[index] for index in keep]


def frame_predictions(model, config, records, device, options):
    options.validate()
    model.eval()
    dataset = TileFrames(
        config["data"]["root"], records, config["model"]["image_size"], options
    )
    kwargs = {
        "batch_size": 1,
        "num_workers": options.workers,
        "collate_fn": single_frame,
        "pin_memory": device.type == "cuda",
    }
    if options.workers:
        kwargs.update(prefetch_factor=1, multiprocessing_context="spawn")
    loader = DataLoader(dataset, **kwargs)
    threshold = config["model"].get("score_threshold", 0.001)
    for tensors, views, record in loader:
        raw = []
        for tensor, view in zip(tensors, views):
            image = tensor.to(
                device=device, dtype=torch.float32, non_blocking=True
            ).div_(255)
            # The context must be inside the generator, around each forward.
            with torch.inference_mode(), amp_context(config, device):
                output = model.predict([image])[0]
            raw.extend(
                decode_view(output, view, record, options.edge_margin, threshold)
            )
            del image, output
        yield record, raw, len(views)


def predict_tiled(model, config, records, device, options, ious=None):
    ious = [options.nms_iou] if ious is None else list(ious)
    if not ious or any(not 0 < iou <= 1 for iou in ious):
        raise ValueError("NMS IoUs must be in (0,1]")
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started, views, outputs = time.perf_counter(), 0, {iou: [] for iou in ious}
    for index, (_, raw, count) in enumerate(
        frame_predictions(model, config, records, device, options), 1
    ):
        views += count
        for iou in ious:
            outputs[iou].extend(
                merge_frame(raw, iou, config["model"]["max_detections"])
            )
        if index % 16 == 0 or index == len(records):
            rate = index / max(time.perf_counter() - started, 1e-9)
            print(
                f"Tiled validation {index}/{len(records)}; views={views}; {rate:.2f} frames/s",
                flush=True,
            )
    synchronize(device)
    elapsed = time.perf_counter() - started
    return outputs, {
        "images": len(records),
        "views": views,
        "pipeline_seconds": elapsed,
        "pipeline_images_per_second": len(records) / elapsed,
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else None,
        "scope": "Original frames/s includes decode, tile resizing, transfer, inference and per-frame merge; excludes metrics/serialization.",
    }


def export_tiled(model, config, records, device, options, output, checkpoint_sha):
    from scripts.check_lab import validate_submission

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(config["data"]["root"])
    sample = root / "sample_submission.csv"
    if sample.is_file():
        with sample.open(newline="", encoding="utf-8-sig") as stream:
            ids = [row["image_id"] for row in csv.DictReader(stream)]
        index = {r["submission_id"]: r for r in records}
        if len(ids) != len(set(ids)) or set(ids) != set(index):
            raise ValueError("Sample and test IDs differ")
        records = [index[key] for key in ids]
    with output.with_suffix(output.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists():
            metadata = read_json(output.with_suffix(".csv.metadata.json"))
            if (
                metadata["checkpoint_sha256"] != checkpoint_sha
                or metadata["options"] != asdict(options)
                or metadata["image_size"] != config["model"]["image_size"]
            ):
                raise ValueError("Existing tiled export has different provenance")
            return validate_submission(output, root / "test_images.csv", sample)
        started, views = time.perf_counter(), 0
        temporary = output.with_suffix(output.suffix + ".tmp")
        with tempfile.TemporaryDirectory(
            prefix="tile-csv-", dir=output.parent
        ) as scratch:
            part = Path(scratch) / "part.csv"
            with temporary.open("w", newline="", encoding="utf-8") as destination:
                writer = csv.writer(destination)
                writer.writerow(["image_id", "PredictionString"])
                for count, (record, raw, nviews) in enumerate(
                    frame_predictions(model, config, records, device, options), 1
                ):
                    predictions = merge_frame(
                        raw, options.nms_iou, config["model"]["max_detections"]
                    )
                    write_submission(
                        [record],
                        predictions,
                        None,
                        part,
                        config["model"].get("score_threshold", 0.001),
                        config["model"]["max_detections"],
                    )
                    with part.open(newline="", encoding="utf-8") as stream:
                        reader = csv.reader(stream)
                        next(reader)
                        writer.writerows(reader)
                    views += nviews
                    if count % 16 == 0 or count == len(records):
                        rate = count / max(time.perf_counter() - started, 1e-9)
                        print(
                            f"Tiled test={count}/{len(records)}; views={views}; ETA={(len(records) - count) / rate / 60:.1f} min",
                            flush=True,
                        )
        checked = validate_submission(temporary, root / "test_images.csv", sample)
        if checked["all_predictions_empty"]:
            raise ValueError("All tiled predictions are empty; CSV not published")
        write_json(
            output.with_suffix(".csv.metadata.json"),
            {
                "checkpoint_sha256": checkpoint_sha,
                "options": asdict(options),
                "image_size": config["model"]["image_size"],
                "test_fingerprint": digest(records),
                "validation": checked,
                "views": views,
                "export_seconds": time.perf_counter() - started,
            },
        )
        temporary.replace(output)
        print(f"TILED CANDIDATE READY FOR KAGGLE: {output}", flush=True)
        return checked
