import math
import random

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader, Dataset

from .common import seed_worker
from .prepare import safe_path


def letterbox(image, boxes, size):
    width, height = image.size
    ratio = min(size / width, size / height)
    nw, nh = max(1, round(width * ratio)), max(1, round(height * ratio))
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(image.resize((nw, nh), Image.Resampling.BILINEAR), (left, top))
    result = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    result[:, [0, 2]] = result[:, [0, 2]] * (nw / width) + left
    result[:, [1, 3]] = result[:, [1, 3]] * (nh / height) + top
    return (
        canvas,
        result,
        {
            "scale_x": nw / width,
            "scale_y": nh / height,
            "left": left,
            "top": top,
            "width": width,
            "height": height,
        },
    )


def mask_ignored(image, ignored, positives):
    # Explicit competition-permitted baseline: remove unknown visual content.
    # Restore all labelled object rectangles so masking never erases known targets.
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    for box in ignored:
        draw.rectangle(
            [
                math.floor(box[0]),
                math.floor(box[1]),
                math.ceil(box[2]),
                math.ceil(box[3]),
            ],
            fill=255,
        )
    for box in positives:
        draw.rectangle(
            [
                math.floor(box[0]),
                math.floor(box[1]),
                math.ceil(box[2]),
                math.ceil(box[3]),
            ],
            fill=0,
        )
    return Image.composite(Image.new("RGB", image.size, (114, 114, 114)), image, mask)


def training_crop(image, boxes, labels, size):
    """Crop at native scale, selecting a present class/object uniformly."""
    width, height = image.size
    crop_width, crop_height = min(width, size), min(height, size)
    if labels:
        category = random.choice(sorted(set(labels)))
        selected = random.choice(
            [i for i, label in enumerate(labels) if label == category]
        )
        x1, y1, x2, y2 = boxes[selected]
        left = round(
            (x1 + x2) / 2
            - crop_width / 2
            + random.uniform(-crop_width / 4, crop_width / 4)
        )
        top = round(
            (y1 + y2) / 2
            - crop_height / 2
            + random.uniform(-crop_height / 4, crop_height / 4)
        )
        left, top = (
            min(max(left, 0), width - crop_width),
            min(max(top, 0), height - crop_height),
        )
    else:
        left, top = (
            random.randint(0, width - crop_width),
            random.randint(0, height - crop_height),
        )
    shifted = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    shifted[:, [0, 2]] = np.clip(shifted[:, [0, 2]] - left, 0, crop_width)
    shifted[:, [1, 3]] = np.clip(shifted[:, [1, 3]] - top, 0, crop_height)
    keep = (shifted[:, 2] > shifted[:, 0]) & (shifted[:, 3] > shifted[:, 1])
    return (
        image.crop((left, top, left + crop_width, top + crop_height)),
        shifted[keep],
        [label for label, valid in zip(labels, keep) if valid],
    )


class Frames(Dataset):
    def __init__(
        self,
        root,
        records,
        size,
        training=False,
        flip=0.5,
        ignore_policy="mask",
        crop_probability=0.0,
        crop_size=None,
    ):
        self.root, self.records, self.size = root, records, size
        self.training, self.flip, self.ignore_policy = training, flip, ignore_policy
        self.crop_probability, self.crop_size = crop_probability, crop_size or size
        if not 0 <= crop_probability <= 1 or self.crop_size < 1:
            raise ValueError("Invalid training crop configuration")
        if ignore_policy not in ("mask", "none"):
            raise ValueError("ignore_policy must be mask or none")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(safe_path(self.root, record["file_name"])) as opened:
            image = opened.convert("RGB")
        if image.size != (record["width"], record["height"]):
            raise ValueError(
                f"Image dimensions disagree with manifest: {record['file_name']}"
            )
        if self.training and self.ignore_policy == "mask":
            image = mask_ignored(image, record["ignore_boxes"], record["boxes"])
        boxes, labels = record["boxes"], record["labels"]
        # getattr keeps already-running spawn-loader datasets compatible with
        # this additive option when workers import a newer source file.
        probability = getattr(self, "crop_probability", 0)
        if self.training and probability and random.random() < probability:
            image, boxes, labels = training_crop(image, boxes, labels, self.crop_size)
        image, boxes, geometry = letterbox(image, boxes, self.size)
        if self.training and random.random() < self.flip:
            image = ImageOps.mirror(image)
            boxes[:, [0, 2]] = self.size - boxes[:, [2, 0]]
        tensor = (
            torch.from_numpy(np.array(image).transpose(2, 0, 1).copy())
            .float()
            .div_(255)
        )
        target = {
            "boxes": torch.from_numpy(boxes),
            # Internal torchvision-compatible convention: background=0, targets=1..8.
            "labels": torch.tensor(labels, dtype=torch.int64) + 1,
        }
        return tensor, target, {"record": record, "geometry": geometry}


def collate(batch):
    return tuple(map(list, zip(*batch)))


def make_loader(config, records, training, generator=None, worker_context=None):
    cfg = config["loader"]
    workers = cfg["workers"]
    data = Frames(
        config["data"]["root"],
        records,
        config["model"]["image_size"],
        training,
        config["data"].get("horizontal_flip", 0.5),
        config["data"].get("ignore_policy", "mask"),
        crop_probability=config["data"].get("crop_probability", 0.0),
        crop_size=config["data"].get("crop_size"),
    )
    options = {
        "num_workers": workers,
        "pin_memory": cfg["pin_memory"],
        "collate_fn": collate,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if workers:
        if worker_context is not None:
            options["multiprocessing_context"] = worker_context
        options["prefetch_factor"] = cfg["prefetch_factor"]
        # Keep false by default: recreating workers gives reproducible epoch-boundary resume.
        options["persistent_workers"] = cfg.get("persistent_workers", False)
    sampling = cfg.get("sampling", "shuffle")
    if sampling not in ("shuffle", "flight_time"):
        raise ValueError("loader.sampling must be shuffle or flight_time")
    sampler = None
    if training and sampling == "flight_time":
        from .sampling import FlightTimeSampler

        sampler = FlightTimeSampler(records, generator, cfg.get("temporal_bins", 8))
    return DataLoader(
        data, batch_size=cfg["batch_size"], shuffle=training and sampler is None,
        sampler=sampler, drop_last=False, **options
    )


def restore_boxes(boxes, geometry):
    boxes = boxes.clone()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - geometry["left"]) / geometry["scale_x"]
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - geometry["top"]) / geometry["scale_y"]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, geometry["width"])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, geometry["height"])
    return boxes
