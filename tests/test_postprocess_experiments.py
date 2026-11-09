import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cvlab.common import digest, write_json
from cvlab.postprocess_experiments import extra_nms, run, stream_array


@pytest.mark.parametrize("chunk_size", [1, 7, 65536])
def test_streaming_array_handles_chunk_boundaries_and_metadata(tmp_path, chunk_size):
    path = tmp_path / "predictions.json"
    data = {
        "before": [12345, 'українська \\"'],
        "predictions": [{"x": 123.456}, {}, {"s": "a\\b"}],
        "after": {"ok": True},
    }
    write_json(path, data)
    metadata = {}
    assert (
        list(stream_array(path, "predictions", metadata, chunk_size))
        == data["predictions"]
    )
    assert metadata == {
        key: value for key, value in data.items() if key != "predictions"
    }


@pytest.mark.parametrize("chunk_size", [1, 3, 8])
def test_streaming_numbers_split_at_decimal_or_exponent(tmp_path, chunk_size):
    path = tmp_path / "numbers.json"
    path.write_text('{"predictions":[123.456,1e-12,-123.4E+12,0]}')
    assert list(stream_array(path, "predictions", chunk_size=chunk_size)) == [
        123.456,
        1e-12,
        -123.4e12,
        0,
    ]


def test_streaming_memory_does_not_grow_with_prediction_count(tmp_path):
    import tracemalloc

    path = tmp_path / "many.json"
    item = '{"image_id":1,"category_id":1,"bbox":[1,2,3,4],"score":0.5}'
    with path.open("w") as stream:
        stream.write('{"predictions":[')
        for index in range(30000):
            stream.write(("," if index else "") + item)
        stream.write("]}")
    tracemalloc.start()
    try:
        assert sum(1 for _ in stream_array(path, "predictions")) == 30000
        assert tracemalloc.get_traced_memory()[1] < 2 * 1024**2
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize(
    "content",
    [
        '{"predictions":[{}',
        '{"other":[]}',
        '{"predictions":[]} trailing',
        '{"predictions":[],"predictions":[]}',
    ],
)
def test_streaming_rejects_invalid_or_missing_arrays(tmp_path, content):
    path = tmp_path / "invalid.json"
    path.write_text(content)
    with pytest.raises(ValueError):
        list(stream_array(path, "predictions", chunk_size=3))


def test_nms_uses_class_and_image_groups_and_does_not_mutate():
    predictions = [
        {"image_id": 1, "category_id": 1, "bbox": [10, 20, 10, 10], "score": 0.9},
        {"image_id": 1, "category_id": 1, "bbox": [11, 20, 10, 10], "score": 0.8},
        {"image_id": 1, "category_id": 2, "bbox": [11, 20, 10, 10], "score": 0.8},
        {"image_id": 2, "category_id": 1, "bbox": [11, 20, 10, 10], "score": 0.8},
    ]
    original = copy.deepcopy(predictions)
    assert extra_nms(predictions, 0.5) == [
        predictions[0],
        predictions[2],
        predictions[3],
    ]
    assert predictions == original


def test_cpu_postprocessing_streams_and_checks_dataset_fingerprint(tmp_path):
    records = [
        {
            "id": index,
            "sequence_id": "clip",
            "width": 100,
            "height": 100,
            "boxes": [[10, 20, 20, 30]],
            "labels": [0],
            "ignore_boxes": [],
        }
        for index in range(3)
    ]
    prepared = tmp_path / "prepared"
    write_json(prepared / "val.json", {"images": records})
    config = {
        "data": {"prepared": str(prepared), "split": "official"},
        "model": {"name": "retinanet"},
        "evaluation": {"max_detections": [100, 500]},
    }
    predictions = [
        {
            "image_id": record["id"],
            "category_id": 1,
            "bbox": [10, 20, 10, 10],
            "score": 0.9,
        }
        for record in records
    ]
    # A higher-scoring false positive overlaps the perfect box at IoU 0.43.
    # Aggressive NMS removes the true positive, making this a measurable ablation.
    predictions += [
        {
            "image_id": record["id"],
            "category_id": 1,
            "bbox": [14, 20, 10, 10],
            "score": 0.95,
        }
        for record in records
    ]
    source = tmp_path / "source.json"
    metadata = {
        "record_ids": [r["id"] for r in records],
        "data_fingerprint": digest(records),
        "model_config": config["model"],
    }
    write_json(source, {"predictions": predictions, **metadata})
    result = run(
        source, config, tmp_path / "sweep.json", max_images=1, thresholds=(0.3, 0.5)
    )
    assert result["record_ids"] == [1]
    assert result["subset"] and result["full_validation_images"] == 3
    assert [
        case["metrics"]["coco100"]["mAP"] for case in result["experiments"]
    ] == pytest.approx([0.5, 0, 0.5])
    metadata["data_fingerprint"] = "different"
    write_json(source, {"predictions": predictions, **metadata})
    with pytest.raises(ValueError, match="do not match"):
        run(source, config, tmp_path / "bad.json")


def test_postprocess_module_never_imports_torch():
    project = Path(__file__).resolve().parents[1]
    code = 'import sys; import cvlab.postprocess_experiments; assert "torch" not in sys.modules'
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project)
    subprocess.run([sys.executable, "-c", code], cwd=project, env=env, check=True)


def test_fold_subset_excludes_training_sequences(tmp_path):
    from cvlab.postprocess_experiments import validation_subset

    records = [
        {
            "id": index,
            "sequence_id": sequence,
            "boxes": [[1, 2, 3, 4]],
            "labels": [0],
            "ignore_boxes": [],
        }
        for index, sequence in enumerate(["train", "val", "train", "val"])
    ]
    write_json(tmp_path / "train.json", {"images": records})
    write_json(tmp_path / "folds.json", {"folds": [{"val_sequences": ["val"]}]})
    config = {"data": {"prepared": str(tmp_path), "split": "fold", "fold": 0}}
    selected, ids, fingerprint = validation_subset(config, 1)
    assert ids == {1, 3} and [r["id"] for r in selected] == [1]
    assert fingerprint == digest([records[1], records[3]])
