import pytest
import torch
from PIL import Image

from cvlab.class_ensemble import pipeline_identity
from cvlab.common import CLASSES
from cvlab.data import letterbox, restore_boxes
from cvlab.tiled_inference import TileOptions, merge_frame
from cvlab.yolo_flip import HorizontalFlipOnlyPredictor, unflip_canvas_boxes


def test_horizontal_flip_unflips_canvas_boxes_before_restoring_odd_padding():
    # This aspect ratio produces an odd horizontal padding split on a 1280 canvas.
    image = Image.new("RGB", (101, 173), "white")
    _, scaled, geometry = letterbox(image, [[10, 20, 50, 90]], 1280)
    assert (1280 - round(101 * min(1280 / 101, 1280 / 173))) % 2 == 1
    mirrored_box = scaled.copy()
    mirrored_box[:, [0, 2]] = 1280 - scaled[:, [2, 0]]

    class FakePredictor:
        def eval(self):
            return self

        def predict(self, images):
            assert torch.equal(images[0], torch.flip(original, dims=(-1,)))
            return [
                {
                    "boxes": torch.tensor(mirrored_box, dtype=torch.float32),
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([3]),
                }
            ]

    original = torch.arange(3 * 1280 * 1280).reshape(3, 1280, 1280)
    output = HorizontalFlipOnlyPredictor(FakePredictor()).predict([original])[0]
    restored = restore_boxes(output["boxes"], geometry)
    assert restored[0].tolist() == pytest.approx([10, 20, 50, 90], abs=0.02)
    assert output["labels"].tolist() == [3]


def test_flip_cache_identity_is_explicit_without_changing_native_identity():
    config = {
        "model": {"name": "yolo11s", "image_size": 1280},
        "runtime": {"device": "cuda:0", "precision": "fp16"},
        "seed": 42,
    }
    options = TileOptions(tile_side=None, nms_iou=0.6, workers=2)
    native = pipeline_identity("yolo", config, options)
    flipped = pipeline_identity("yolo", config, options, "horizontal_flip")
    assert "tta" not in native
    assert flipped["tta"]["adapter"] == "yolo-horizontal-flip-only-canvas-unflip-v1"
    assert flipped["tta"]["adapter_sha256"]
    assert {k: v for k, v in flipped.items() if k != "tta"} == native
    with pytest.raises(ValueError, match="only for YOLO"):
        pipeline_identity("c1", config, options, "horizontal_flip")


def test_flip_box_untransform_validates_shape_and_preserves_classes_after_merge():
    boxes = torch.tensor([[101.0, 4.0, 123.0, 24.0]])
    assert unflip_canvas_boxes(boxes, 256).tolist() == [[133.0, 4.0, 155.0, 24.0]]
    with pytest.raises(ValueError, match="Nx4"):
        unflip_canvas_boxes(torch.ones(2, 5), 256)

    predictions = [
        {"image_id": 1, "category_id": 1, "score": 0.95, "bbox": [10, 10, 20, 20]},
        {"image_id": 1, "category_id": 1, "score": 0.80, "bbox": [10, 10, 20, 20]},
        {"image_id": 1, "category_id": 2, "score": 0.70, "bbox": [10, 10, 20, 20]},
    ]
    merged = merge_frame(predictions, 0.6, 500)
    assert [(p["category_id"], p["score"]) for p in merged] == [(1, 0.95), (2, 0.70)]

    dense = [
        {
            "image_id": 1,
            "category_id": (i % len(CLASSES)) + 1,
            "score": 1 - i / 2000,
            "bbox": [i * 10, 2, 4, 4],
        }
        for i in range(800)
    ]
    assert len(merge_frame(dense, 0.6, 500)) == 500
