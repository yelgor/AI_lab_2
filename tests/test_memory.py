import hashlib
import json
import tracemalloc

import pytest

from cvlab.common import digest, read_json, write_json


def prediction_output():
    prediction = {
        "image_id": 1,
        "category_id": 2,
        "bbox": [123.5, 456.25, 18.75, 27.125],
        "score": 0.987654321,
    }
    # Share input records to measure serialization overhead, not input storage.
    return {"predictions": [prediction] * 10_000, "note": "Детекції"}


def test_large_json_write_has_bounded_extra_memory(tmp_path):
    value = prediction_output()
    path = tmp_path / "predictions.json"
    tracemalloc.start()
    try:
        write_json(path, value)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 2 * 1024**2
    assert read_json(path) == value
    assert not path.with_suffix(".json.tmp").exists()


def test_json_failure_preserves_previous_complete_file(tmp_path):
    path = tmp_path / "predictions.json"
    original = {"predictions": [], "epoch": 0}
    write_json(path, original)
    with pytest.raises(ValueError):
        write_json(path, {"predictions": [{"score": float("nan")}]})
    assert read_json(path) == original
    replacement = {"predictions": [], "epoch": 1}
    write_json(path, replacement)
    assert read_json(path) == replacement


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"z": "Україна", "a": [True, None, -0.0, 1e-20, 1e20]},
        prediction_output(),
    ],
)
def test_digest_preserves_existing_checkpoint_fingerprints(value):
    expected = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    assert digest(value) == expected


def test_large_digest_has_bounded_extra_memory():
    value = prediction_output()
    tracemalloc.start()
    try:
        digest(value)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 2 * 1024**2
