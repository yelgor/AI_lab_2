from dataclasses import replace

import pytest
import torch
from PIL import Image

from cvlab.common import read_json
from cvlab.data import letterbox
from cvlab.tiled_inference import (
    TileFrames,
    TileOptions,
    decode_view,
    export_tiled,
    merge_frame,
    predict_tiled,
    regions,
)


def test_tile_grid_covers_odd_original_dimensions_and_avoids_global_duplicate():
    options = TileOptions(64, workers=0)
    boxes = regions(173, 97, options)
    assert boxes[0] == (0, 0, 173, 97)
    assert len(set(boxes)) == len(boxes)
    cover = torch.zeros(97, 173, dtype=torch.bool)
    for x1, y1, x2, y2 in boxes[1:]:
        assert 0 <= x1 < x2 <= 173 and 0 <= y1 < y2 <= 97
        cover[y1:y2, x1:x2] = True
    assert cover.all()
    assert regions(31, 27, options) == [(0, 0, 31, 27)]
    with pytest.raises(ValueError, match="64 views"):
        regions(2000, 2000, options)


def test_tile_box_restoration_and_interior_boundary_filter():
    image = Image.new("RGB", (80, 80))
    _, scaled, geometry = letterbox(image, [[5, 15, 35, 45]], 128)
    output = {
        "boxes": torch.tensor(scaled),
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([8]),
    }
    view = {"rect": (100, 20, 180, 100), "geometry": geometry, "global": False}
    record = {"id": 3, "width": 200, "height": 120}
    result = decode_view(output, view, record, 2, 0.001)
    assert result[0]["bbox"] == pytest.approx([105, 35, 30, 30])
    assert result[0]["category_id"] == 8 and result[0]["image_id"] == 3
    output["boxes"][0, 0] = 0
    assert decode_view(output, view, record, 2, 0.001) == []
    view["rect"] = (0, 20, 80, 100)
    assert decode_view(
        output, view, record, 2, 0.001
    )  # Original left boundary is valid.


def test_tile_merge_removes_duplicates_by_class_and_caps_original_image():
    predictions = [
        {"image_id": 1, "category_id": label, "score": score, "bbox": [1, 2, 10, 10]}
        for label, score in [(1, 0.9), (1, 0.8), (2, 0.7)]
    ]
    assert merge_frame(predictions, 0.5) == [predictions[0], predictions[2]]
    dense = [
        {
            "image_id": 1,
            "category_id": 1,
            "score": 1 - i / 2000,
            "bbox": [i * 10, 1, 3, 3],
        }
        for i in range(1000)
    ]
    assert len(merge_frame(dense, 0.5)) == 500
    assert dense[0]["bbox"] == [0, 1, 3, 3]


class SimpleDetector(torch.nn.Module):
    def predict(self, images):
        assert not torch.is_grad_enabled(), (
            "Inference context must be active during generator iteration"
        )
        return [
            {
                "boxes": image.new_tensor([[32, 32, 64, 64], [32, 32, 64, 64]]),
                "scores": image.new_tensor([0.9, 0.8]),
                "labels": torch.tensor([1, 8], device=image.device),
            }
            for image in images
        ]


def competition(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    records = []
    for index in range(2):
        Image.new("RGB", (173, 97), "white").save(data / f"{index}.jpg")
        records.append(
            {
                "id": index,
                "file_name": f"{index}.jpg",
                "width": 173,
                "height": 97,
                "submission_id": f"frame_{index}",
            }
        )
    (data / "test_images.csv").write_text(
        "image_id,width,height\nframe_0,173,97\nframe_1,173,97\n"
    )
    (data / "sample_submission.csv").write_text(
        "image_id,PredictionString\nframe_1,\nframe_0,\n"
    )
    config = {
        "data": {"root": str(data)},
        "model": {"image_size": 128, "score_threshold": 0.001, "max_detections": 500},
        "runtime": {"precision": "fp32"},
    }
    return config, records


@pytest.mark.parametrize("workers", [0, 1])
def test_tiled_prediction_streams_original_coordinates_and_inference_mode(
    tmp_path, workers
):
    config, records = competition(tmp_path)
    options = TileOptions(64, workers=workers)
    frames = TileFrames(config["data"]["root"], records, 128, options)
    tensors, _, row = frames[0]
    assert len(tensors) > 1 and tensors[0].dtype == torch.uint8
    assert "boxes" not in row
    outputs, timing = predict_tiled(
        SimpleDetector(), config, records, torch.device("cpu"), options, [0.5, 0.6]
    )
    assert timing["images"] == 2 and timing["views"] > 2
    for predictions in outputs.values():
        assert {p["image_id"] for p in predictions} == {0, 1}
        for prediction in predictions:
            x, y, width, height = prediction["bbox"]
            assert x >= 0 and y >= 0 and x + width <= 173.001 and y + height <= 97.001


def test_real_tiled_export_preserves_template_order_ids_and_provenance(tmp_path):
    config, records = competition(tmp_path)
    output = tmp_path / "tiled.csv"
    options = TileOptions(64, workers=0)
    result = export_tiled(
        SimpleDetector(),
        config,
        records,
        torch.device("cpu"),
        options,
        output,
        "frozen-sha",
    )
    assert result["valid"] and result["images"] == 2 and result["empty_images"] == 0
    assert result["matches_template_order"]
    metadata = read_json(output.with_suffix(".csv.metadata.json"))
    assert metadata["checkpoint_sha256"] == "frozen-sha"
    assert metadata["views"] > 2
    assert export_tiled(
        SimpleDetector(),
        config,
        records,
        torch.device("cpu"),
        options,
        output,
        "frozen-sha",
    )["valid"]
    with pytest.raises(ValueError, match="different provenance"):
        export_tiled(
            SimpleDetector(),
            config,
            records,
            torch.device("cpu"),
            replace(options, nms_iou=0.6),
            output,
            "frozen-sha",
        )


def test_tiled_case_selection_prefers_cheaper_strategy_on_ties():
    from cvlab.tiled_search import choose

    baseline = {"strategy": "global", "metrics": {"coco100": {"mAP": 0.2}}}
    tile = {
        "strategy": "tiles",
        "options": {"tile_side": 768},
        "metrics": {"coco100": {"mAP": 0.2}},
    }
    assert choose([tile, baseline]) is baseline


def test_real_rtdetr_forward_on_global_and_tiles(tmp_path):
    from cvlab.models import build_model

    torch.set_num_threads(2)
    torch.manual_seed(42)
    config, records = competition(tmp_path)
    config["model"].update(
        name="rtdetr",
        pretrained=False,
        num_queries=20,
        num_denoising=0,
        max_detections=20,
        score_threshold=0,
    )
    model = build_model(config["model"])
    outputs, timing = predict_tiled(
        model, config, records[:1], torch.device("cpu"), TileOptions(128, workers=0)
    )
    assert timing["views"] == 3
    assert 0 < len(outputs[0.5]) <= 20
    for prediction in outputs[0.5]:
        x, y, width, height = prediction["bbox"]
        assert 0 <= x < x + width <= 173.001
        assert 0 <= y < y + height <= 97.001
        assert 1 <= prediction["category_id"] <= 8
