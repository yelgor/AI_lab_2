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


class Frames(Dataset):
    def __init__(
        self, root, records, size, training=False, flip=0.5, ignore_policy="mask"
    ):
        self.root, self.records, self.size = root, records, size
        self.training, self.flip, self.ignore_policy = training, flip, ignore_policy
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
        image, boxes, geometry = letterbox(image, record["boxes"], self.size)
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
            "labels": torch.tensor(record["labels"], dtype=torch.int64) + 1,
        }
        return tensor, target, {"record": record, "geometry": geometry}


def collate(batch):
    return tuple(map(list, zip(*batch)))


def make_loader(config, records, training, generator=None):
    cfg = config["loader"]
    workers = cfg["workers"]
    data = Frames(
        config["data"]["root"],
        records,
        config["model"]["image_size"],
        training,
        config["data"].get("horizontal_flip", 0.5),
        config["data"].get("ignore_policy", "mask"),
    )
    options = {
        "num_workers": workers,
        "pin_memory": cfg["pin_memory"],
        "collate_fn": collate,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if workers:
        options["prefetch_factor"] = cfg["prefetch_factor"]
        # Keep false by default: recreating workers gives reproducible epoch-boundary resume.
        options["persistent_workers"] = cfg.get("persistent_workers", False)
    return DataLoader(
        data, batch_size=cfg["batch_size"], shuffle=training, drop_last=False, **options
    )


def restore_boxes(boxes, geometry):
    boxes = boxes.clone()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - geometry["left"]) / geometry["scale_x"]
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - geometry["top"]) / geometry["scale_y"]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, geometry["width"])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, geometry["height"])
    return boxes
