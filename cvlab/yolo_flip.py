"""Horizontal-flip-only inference adapter for an existing YOLO prediction cache."""

import torch


def unflip_canvas_boxes(boxes, canvas_width):
    """Map xyxy boxes from the mirrored tensor canvas back to its original canvas."""
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("Expected an Nx4 xyxy box tensor")
    if not isinstance(canvas_width, (int, float)) or canvas_width <= 0:
        raise ValueError("Canvas width must be positive")
    boxes = boxes.clone()
    x1 = boxes[:, 0].clone()
    x2 = boxes[:, 2].clone()
    boxes[:, 0] = canvas_width - x2
    boxes[:, 2] = canvas_width - x1
    return boxes


class HorizontalFlipOnlyPredictor:
    """Predict only the missing flipped view; merge with native cache on CPU later."""

    def __init__(self, predictor):
        self.predictor = predictor

    def eval(self):
        self.predictor.eval()
        return self

    def predict(self, images):
        mirrored = [torch.flip(image, dims=(-1,)) for image in images]
        outputs = self.predictor.predict(mirrored)
        if len(outputs) != len(images):
            raise ValueError("YOLO returned a different number of flipped predictions")
        return [
            {
                **output,
                "boxes": unflip_canvas_boxes(output["boxes"], image.shape[-1]),
            }
            for image, output in zip(images, outputs, strict=True)
        ]
