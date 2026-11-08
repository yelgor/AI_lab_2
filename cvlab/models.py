"""Four model adapters with a shared loss/predict interface.

Pinned APIs: torch 2.6, torchvision 0.21, transformers 4.51.3.
"""

import types

import torch
from torch import nn
from torchvision.models import detection
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.fcos import FCOSClassificationHead
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.ops import FrozenBatchNorm2d
from torchvision.ops.feature_pyramid_network import LastLevelP6P7


class TVDetector(nn.Module):
    def __init__(self, config, initialize_pretrained=True):
        super().__init__()
        name = config["name"]
        wants_pretrained_topology = config.get("pretrained", True)
        pretrained = wants_pretrained_topology and initialize_pretrained
        size = config["image_size"]
        kwargs = dict(
            min_size=size,
            max_size=size,
            box_detections_per_img=config["max_detections"],
        )
        if name != "fasterrcnn":
            kwargs = dict(
                min_size=size,
                max_size=size,
                detections_per_img=config["max_detections"],
                score_thresh=config.get("score_threshold", 0.001),
                nms_thresh=config.get("nms_threshold", 0.5),
            )
        else:
            kwargs.update(
                box_score_thresh=config.get("score_threshold", 0.001),
                box_nms_thresh=config.get("nms_threshold", 0.5),
            )
        if name == "fasterrcnn":
            weights = (
                detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
                if pretrained
                else None
            )
            self.core = detection.fasterrcnn_resnet50_fpn(
                weights=weights, weights_backbone=None, **kwargs
            )
            features = self.core.roi_heads.box_predictor.cls_score.in_features
            self.core.roi_heads.box_predictor = FastRCNNPredictor(features, 9)
            if "rpn_proposals" in config:
                value = config["rpn_proposals"]
                self.core.rpn._post_nms_top_n = {"training": value, "testing": value}
            if "roi_positive_fraction" in config:
                self.core.roi_heads.fg_bg_sampler.positive_fraction = config[
                    "roi_positive_fraction"
                ]
        elif name == "retinanet":
            weights = (
                detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT if pretrained else None
            )
            self.core = detection.retinanet_resnet50_fpn(
                weights=weights, weights_backbone=None, **kwargs
            )
            anchors = self.core.anchor_generator.num_anchors_per_location()[0]
            old_conv = self.core.head.classification_head.conv
            self.core.head.classification_head = RetinaNetClassificationHead(
                256, anchors, 9
            )
            self.core.head.classification_head.conv = old_conv
            loss = config.get("box_loss", "l1")
            if loss not in ("l1", "giou"):
                raise ValueError("RetinaNet box_loss supports l1 or giou")
            self.core.head.regression_head._loss_type = loss
        elif name == "fcos":
            weights = (
                detection.FCOS_ResNet50_FPN_Weights.DEFAULT if pretrained else None
            )
            self.core = detection.fcos_resnet50_fpn(
                weights=weights, weights_backbone=None, **kwargs
            )
            old_conv = self.core.head.classification_head.conv
            self.core.head.classification_head = FCOSClassificationHead(256, 1, 9)
            self.core.head.classification_head.conv = old_conv
            self.core.center_sampling_radius = config.get("center_sampling_radius", 1.5)
            loss = config.get("box_loss", "giou")
            if loss == "giou_l1":
                # A real loss-modification experiment; keep the native assignment and
                # classification/centerness, and add normalized L1 to positive boxes.
                original = self.core.head.compute_loss
                weight = config.get("extra_l1_weight", 0.5)

                def compute_loss(head, targets, head_outputs, anchors, matched_idxs):
                    losses = original(targets, head_outputs, anchors, matched_idxs)
                    extra, positives = head_outputs["bbox_regression"].sum() * 0, 0
                    for target, regression, anchor, matches in zip(
                        targets, head_outputs["bbox_regression"], anchors, matched_idxs
                    ):
                        keep = matches >= 0
                        if keep.any():
                            predicted = head.box_coder.decode(
                                regression[keep], anchor[keep]
                            )
                            truth = target["boxes"][matches[keep]]
                            extra = extra + torch.nn.functional.l1_loss(
                                predicted / size, truth / size, reduction="sum"
                            )
                            positives += int(keep.sum())
                    losses["bbox_regression"] = losses[
                        "bbox_regression"
                    ] + weight * extra / max(1, positives)
                    return losses

                self.core.head.compute_loss = types.MethodType(
                    compute_loss, self.core.head
                )
            elif loss != "giou":
                raise ValueError("FCOS box_loss supports giou or giou_l1")
        else:
            raise ValueError(f"Unknown torchvision detector {name}")
        if wants_pretrained_topology and not initialize_pretrained:
            # weights=None in torchvision factories otherwise changes FrozenBN
            # to BatchNorm AND unfreezes extra backbone layers. Reconstruct the
            # exact pretrained topology without downloading weights on restore.
            options = (
                {}
                if name == "fasterrcnn"
                else {
                    "returned_layers": [2, 3, 4],
                    "extra_blocks": LastLevelP6P7(256, 256),
                }
            )
            self.core.backbone = resnet_fpn_backbone(
                backbone_name="resnet50",
                weights=None,
                norm_layer=FrozenBatchNorm2d,
                trainable_layers=3,
                **options,
            )
            if name in ("fasterrcnn", "retinanet"):
                # Their pinned COCO_V1 factories use eps=0; eps is not saved in
                # state_dict, so restore it explicitly alongside FrozenBN.
                for layer in self.core.backbone.modules():
                    if isinstance(layer, FrozenBatchNorm2d):
                        layer.eps = 0.0
        if "anchor_sizes" in config:
            if name == "fcos":
                raise ValueError(
                    "Do not change FCOS anchor sizes; this implementation uses them as strides"
                )
            sizes = tuple(tuple(level) for level in config["anchor_sizes"])
            ratios = tuple(config.get("anchor_ratios", [0.5, 1.0, 2.0]))
            new = AnchorGenerator(sizes, (ratios,) * len(sizes))
            old_counts = self.core.anchor_generator.num_anchors_per_location()
            if new.num_anchors_per_location() != old_counts:
                raise ValueError(
                    "Anchor counts must match the pretrained head; vary sizes/ratios, not counts"
                )
            self.core.anchor_generator = new
        self.backbone_frozen = config.get("freeze_backbone", False)
        if self.backbone_frozen:
            for parameter in self.core.backbone.parameters():
                parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.backbone_frozen:
            self.core.backbone.eval()
        return self

    def loss(self, images, targets):
        return self.core(images, targets)

    def predict(self, images):
        outputs = self.core(images)
        # The spare index 0 is not a target class; exclude it for dense heads too.
        return [{k: v[out["labels"] > 0] for k, v in out.items()} for out in outputs]

    def export_metadata(self):
        return None


class QueryDetector(nn.Module):
    def __init__(self, config, metadata=None):
        super().__init__()
        from transformers import (
            RTDetrConfig,
            RTDetrForObjectDetection,
            RTDetrImageProcessor,
        )

        kwargs = {
            "num_labels": 8,
            "num_queries": config.get("num_queries", 500),
            "num_denoising": config.get("num_denoising", 100),
            "eval_size": None,
            "anchor_image_size": None,
            "disable_custom_kernels": True,
        }
        for key in ("matcher_class_cost", "matcher_bbox_cost", "matcher_giou_cost"):
            if key in config:
                kwargs[key] = config[key]
        if metadata:
            self.core = RTDetrForObjectDetection(
                RTDetrConfig.from_dict(metadata["hf_config"])
            )
            self.processor = RTDetrImageProcessor.from_dict(metadata["processor"])
        elif config.get("pretrained", True):
            identifier = config.get("weights", "PekingU/rtdetr_r50vd")
            self.core = RTDetrForObjectDetection.from_pretrained(
                identifier, ignore_mismatched_sizes=True, **kwargs
            )
            self.processor = RTDetrImageProcessor.from_pretrained(identifier)
        else:
            self.core = RTDetrForObjectDetection(RTDetrConfig(**kwargs))
            self.processor = RTDetrImageProcessor()
        self.max_detections = config["max_detections"]
        self.score_threshold = config.get("score_threshold", 0.001)
        self.box_loss = config.get("box_loss", "giou")
        if self.box_loss not in ("giou", "giou_l1"):
            raise ValueError("RT-DETR box_loss supports giou or giou_l1")
        if self.box_loss == "giou_l1":
            # Native RT-DETR already combines L1 and GIoU; vary their weighting.
            self.core.config.weight_loss_bbox = config.get("weight_loss_bbox", 10.0)
        self.backbone_frozen = config.get("freeze_backbone", False)
        if self.backbone_frozen:
            for parameter in self.core.model.backbone.parameters():
                parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.backbone_frozen:
            self.core.model.backbone.eval()
        return self

    def pixels(self, images):
        # Loader already returns rescaled [0,1], letterboxed tensors. Respect the
        # checkpoint processor's normalization rather than normalizing twice.
        if self.processor.do_normalize:
            mean = images[0].new_tensor(self.processor.image_mean)[None, :, None, None]
            std = images[0].new_tensor(self.processor.image_std)[None, :, None, None]
            return (torch.stack(images) - mean) / std
        return torch.stack(images)

    def export_metadata(self):
        return {
            "hf_config": self.core.config.to_dict(),
            "processor": self.processor.to_dict(),
        }

    def loss(self, images, targets):
        labels = []
        for image, target in zip(images, targets):
            height, width = image.shape[-2:]
            boxes = target["boxes"]
            center = (boxes[:, :2] + boxes[:, 2:]) / 2
            dimensions = boxes[:, 2:] - boxes[:, :2]
            boxes = torch.cat([center, dimensions], dim=1) / boxes.new_tensor(
                [width, height, width, height]
            )
            labels.append({"class_labels": target["labels"] - 1, "boxes": boxes})
        output = self.core(pixel_values=self.pixels(images), labels=labels)
        return {"total": output.loss}

    def predict(self, images):
        output = self.core(pixel_values=self.pixels(images))
        results = []
        for logits, boxes, image in zip(output.logits, output.pred_boxes, images):
            scores = logits.sigmoid().flatten()
            scores, indices = scores.topk(min(self.max_detections, scores.numel()))
            keep = scores >= self.score_threshold
            scores, indices = scores[keep], indices[keep]
            labels, queries = indices % 8, indices // 8
            selected = boxes[queries]
            xyxy = torch.cat(
                [
                    selected[:, :2] - selected[:, 2:] / 2,
                    selected[:, :2] + selected[:, 2:] / 2,
                ],
                dim=1,
            )
            height, width = image.shape[-2:]
            xyxy = xyxy * xyxy.new_tensor([width, height, width, height])
            results.append({"boxes": xyxy, "labels": labels + 1, "scores": scores})
        return results


def build_model(config, metadata=None, initialize_pretrained=True):
    if config["name"] == "rtdetr":
        return QueryDetector(config, metadata)
    return TVDetector(config, initialize_pretrained)
