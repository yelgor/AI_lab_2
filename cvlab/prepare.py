"""Validate the competition's YOLO files and build a portable manifest."""

import csv
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .common import CLASSES, read_json, write_json


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes data root: {relative}")
    return path


def parse_boxes(path, width, height, ignored=False):
    boxes, labels = [], []
    if not path.exists():
        if ignored:
            return boxes, labels
        raise FileNotFoundError(
            f"Missing label file (empty frames need empty .txt): {path}"
        )
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        values = [float(x) for x in line.split()]
        if len(values) != (4 if ignored else 5) or not np.isfinite(values).all():
            raise ValueError(f"Invalid annotation: {path}:{number}")
        if not ignored:
            category = values.pop(0)
            if not category.is_integer() or not 0 <= category < len(CLASSES):
                raise ValueError(f"Invalid class: {path}:{number}")
            labels.append(int(category))
        cx, cy, bw, bh = values
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            raise ValueError(f"Invalid normalized box: {path}:{number}")
        x1, y1 = max(0, (cx - bw / 2) * width), max(0, (cy - bh / 2) * height)
        x2, y2 = min(width, (cx + bw / 2) * width), min(height, (cy + bh / 2) * height)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Degenerate box: {path}:{number}")
        boxes.append([x1, y1, x2, y2])
    return boxes, labels


def inspect_frame(args):
    root, split, relative = args
    path = safe_path(root, relative)
    with Image.open(path) as image:
        width, height = image.size
        image.verify()
    clip, frame = path.parent.name, path.stem
    label = root / split / "labels" / clip / f"{frame}.txt"
    ignore = root / split / "ignore" / clip / f"{frame}.txt"
    boxes, labels = parse_boxes(label, width, height)
    ignored, _ = parse_boxes(ignore, width, height, ignored=True)
    return {
        "file_name": relative,
        "clip": clip,
        "sequence_id": clip,
        "frame_id": int(frame),
        "width": width,
        "height": height,
        "boxes": boxes,
        "labels": labels,
        "ignore_boxes": ignored,
    }


def group_folds(records, folds, seed):
    from .sequential_policy import video_family

    groups = defaultdict(list)
    sequences = defaultdict(set)
    for record in records:
        family = video_family(record["sequence_id"])
        groups[family].append(record["id"])
        sequences[family].add(record["sequence_id"])
    if not 2 <= folds <= len(groups):
        raise ValueError("Need 2 <= folds <= number of independent flight families")
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    # Never split related excerpts of a flight between training and validation.
    keys.sort(key=lambda key: len(groups[key]), reverse=True)
    bins, loads = [[] for _ in range(folds)], [0] * folds
    for key in keys:
        selected = min(range(folds), key=lambda i: loads[i])
        bins[selected].extend(sorted(sequences[key]))
        loads[selected] += len(groups[key])
    return {
        "seed": seed,
        "grouping": "flight_family_v1",
        "folds": [{"fold": i, "val_sequences": b} for i, b in enumerate(bins)],
    }


def prepare(root, output, workers=8, folds=3, seed=42):
    root, output = Path(root).resolve(), Path(output)
    manifests = {}
    for split in ("train", "val"):
        paths = sorted((root / split / "images").glob("*/*.jpg"))
        if not paths:
            raise ValueError(f"No frames found: {root / split / 'images'}")
        tasks = [(root, split, p.relative_to(root).as_posix()) for p in paths]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(inspect_frame, tasks))
        for i, record in enumerate(records, 1):
            record["id"] = i
        manifests[split] = {"version": 1, "classes": CLASSES, "images": records}
    from .sequential_policy import video_family

    train_clips = {video_family(r["sequence_id"]) for r in manifests["train"]["images"]}
    val_clips = {video_family(r["sequence_id"]) for r in manifests["val"]["images"]}
    if train_clips & val_clips:
        raise ValueError(
            "Official train/val share flight families. Confirm grouping before proceeding."
        )
    for split, manifest in manifests.items():
        write_json(output / f"{split}.json", manifest)
        eda(root, manifest, output / f"eda_{split}")
    write_json(
        output / "folds.json", group_folds(manifests["train"]["images"], folds, seed)
    )
    test = test_manifest(root)
    write_json(output / "test.json", test)
    print(
        f"Prepared train={len(manifests['train']['images'])}, val={len(manifests['val']['images'])}, test={len(test['images'])}"
    )


def test_manifest(root):
    with (root / "test_images.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Empty test_images.csv or duplicate image_id")
    records = []
    for i, row in enumerate(rows, 1):
        path = safe_path(root, row["file_name"])
        if not path.is_file():
            raise FileNotFoundError(path)
        width, height = int(row["width"]), int(row["height"])
        if min(width, height) <= 0:
            raise ValueError("Invalid test dimensions")
        records.append(
            {
                "id": i,
                "submission_id": row["image_id"],
                "file_name": row["file_name"],
                "clip": row["clip"],
                "sequence_id": row["clip"],
                "frame_id": int(row["frame"]),
                "width": width,
                "height": height,
                "boxes": [],
                "labels": [],
                "ignore_boxes": [],
            }
        )
    sample = root / "sample_submission.csv"
    if sample.exists():
        with sample.open(newline="") as stream:
            sample_ids = [row["image_id"] for row in csv.DictReader(stream)]
        if len(sample_ids) != len(rows) or set(sample_ids) != {
            r["image_id"] for r in rows
        }:
            raise ValueError("Sample submission and test CSV have different IDs")
    return {"version": 1, "classes": CLASSES, "images": records}


def eda(root, manifest, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    counts, areas, density, by_clip = Counter(), [], [], defaultdict(list)
    for record in manifest["images"]:
        counts.update(record["labels"])
        density.append(len(record["boxes"]))
        by_clip[record["clip"]].append(record)
        areas.extend((b[2] - b[0]) * (b[3] - b[1]) for b in record["boxes"])
    summary = {
        "frames": len(density),
        "objects": sum(density),
        "class_counts": {name: counts[i] for i, name in enumerate(CLASSES)},
        "area_quantiles_px2": np.quantile(areas, [0, 0.25, 0.5, 0.75, 1]).tolist()
        if areas
        else [],
        "objects_per_frame_quantiles": np.quantile(density, [0, 0.5, 0.9, 1]).tolist(),
        "sequences": {
            clip: {
                "frames": len(rs),
                "objects": sum(len(r["boxes"]) for r in rs),
                "class_counts": dict(Counter(c for r in rs for c in r["labels"])),
            }
            for clip, rs in by_clip.items()
        },
    }
    write_json(output / "summary.json", summary)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(CLASSES, [counts[i] for i in range(8)])
    axes[0].tick_params(axis="x", rotation=60)
    axes[0].set_title("Class counts")
    axes[1].hist(np.sqrt(areas), bins=60)
    axes[1].set_title("sqrt(box area), original pixels")
    axes[2].hist(density, bins=40)
    axes[2].set_title("Objects per frame")
    fig.tight_layout()
    fig.savefig(output / "distributions.png")
    plt.close(fig)
    # Distributed across the manifest, not just adjacent frames.
    for index in np.linspace(
        0, len(manifest["images"]) - 1, min(6, len(manifest["images"])), dtype=int
    ):
        record = manifest["images"][index]
        image = Image.open(safe_path(root, record["file_name"])).convert("RGB")
        draw = ImageDraw.Draw(image)
        for box in record["ignore_boxes"]:
            draw.rectangle(box, outline="orange", width=2)
        for box, category in zip(record["boxes"], record["labels"]):
            draw.rectangle(box, outline="lime", width=2)
            draw.text((box[0], box[1]), CLASSES[category], fill="lime")
        image.thumbnail((1280, 800))
        image.save(output / f"gt_{record['id']:06}.jpg")


def select_records(config, split):
    prepared = Path(config["data"]["prepared"])
    if split == "test":
        return read_json(prepared / "test.json")["images"]
    mode = config["data"]["split"]
    if mode == "official":
        records = read_json(prepared / f"{split}.json")["images"]
    elif mode == "full_train_val":
        if split != "train":
            raise ValueError("Final train+val fit has no independent validation")
        records = read_json(config["data"]["training_manifest"])["images"]
    elif mode == "fold":
        records = read_json(prepared / "train.json")["images"]
        folds_file = prepared / config["data"].get("folds_file", "folds.json")
        fold = read_json(folds_file)["folds"][config["data"]["fold"]]
        validation = set(fold["val_sequences"])
        records = [
            r for r in records if (r["sequence_id"] in validation) == (split == "val")
        ]
    else:
        raise ValueError("data.split must be 'official' or 'fold'")
    if split == "train":
        step = config["data"].get("frame_step", 1)
        if step < 1:
            raise ValueError("frame_step must be positive")
        grouped = defaultdict(list)
        for record in records:
            grouped[record["sequence_id"]].append(record)
        records = [
            r
            for key in sorted(grouped)
            for r in sorted(grouped[key], key=lambda r: r["frame_id"])[::step]
        ]
    if not records:
        raise ValueError(f"Empty {split} split")
    return records
