"""Optional Ultralytics backend: stock trainer, ignored-region masking and common predictions."""

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from ultralytics.data import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.torch_utils import unwrap_model

from .common import CLASSES, read_json
from .data import mask_ignored


class MaskedDataset(YOLODataset):
    def __init__(self, *args, ignore_path=None, **kwargs):
        self.ignored = read_json(ignore_path) if ignore_path else {}
        super().__init__(*args, **kwargs)

    def load_image(self, i, rect_mode=True):
        image, original_shape, resized_shape = super().load_image(i, rect_mode)
        ignored = self.ignored.get(Path(self.im_files[i]).name, [])
        if self.augment and ignored:
            height, width = image.shape[:2]
            boxes = np.asarray(self.labels[i]["bboxes"], dtype=np.float32).copy()
            xyxy = boxes.copy()
            xyxy[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
            xyxy[:, 2:] = boxes[:, :2] + boxes[:, 2:] / 2
            xyxy *= np.array([width, height, width, height])
            regions = np.asarray(ignored) * np.array([width, height, width, height])
            # OpenCV's BGR channel order is preserved; the fill is gray in either order.
            image = np.array(mask_ignored(Image.fromarray(image), regions, xyxy))
        return image, original_shape, resized_shape


class FullDataTrainer(DetectionTrainer):
    ignore_path = None

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if mode == "train":
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        # Stock setup constructs a validation loader even when val=False. Avoid
        # spawning four idle workers; common validation owns its own small loader.
        workers = self.args.workers
        try:
            self.args.workers = 0
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        finally:
            self.args.workers = workers

    def validate(self):
        # Stock metrics treat ignored regions as negatives. The saved checkpoint
        # is evaluated by EpochObserver through our authoritative common protocol.
        return {}, 0.0

    def final_eval(self):
        # Do not strip optimizer/scaler from last.pt, or call epoch callbacks twice.
        # Full common validation has already run on the exact saved EMA weights.
        pass

    def build_dataset(self, img_path, mode="train", batch=None):
        cfg = self.args
        stride = max(
            int(unwrap_model(self.model).stride.max() if self.model else 0), 32
        )
        return MaskedDataset(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=cfg,
            rect=cfg.rect or mode == "val",
            cache=None,
            single_cls=cfg.single_cls or False,
            stride=stride,
            pad=0.0 if mode == "train" else 0.5,
            prefix=f"{mode}: ",
            task=cfg.task,
            classes=cfg.classes,
            data=self.data,
            fraction=1.0,
            ignore_path=self.ignore_path if mode == "train" else None,
        )


class Predictor(torch.nn.Module):
    def __init__(self, core, config):
        super().__init__()
        if list(core.names.values()) != CLASSES:
            raise ValueError(
                "YOLO checkpoint must use the exact eight competition classes"
            )
        self.core, self.config = core, config

    def predict(self, images):
        raw = self.core(torch.stack(images))
        result = non_max_suppression(
            raw,
            conf_thres=self.config["model"]["score_threshold"],
            iou_thres=self.config["model"]["nms_threshold"],
            nc=8,
            max_det=500,
        )
        return [
            {"boxes": p[:, :4], "scores": p[:, 4], "labels": p[:, 5].long() + 1}
            for p in result
        ]
