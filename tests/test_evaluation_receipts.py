import copy
from pathlib import Path

from cvlab import engine, efficient_search
from cvlab.common import load_config
from cvlab.evaluation_receipts import identity, load_receipt, save_training_evaluation
from test_crop_finetune import tiny_run  # noqa: F401
from test_pipeline import competition  # noqa: F401


def test_training_receipt_avoids_confirmation_gpu_forward(tmp_path, monkeypatch):
    config = load_config(Path(__file__).parents[1] / "configs/rtdetr.yaml")
    config["loader"]["batch_size"] = 1
    config["evaluation"]["cache_directory"] = str(tmp_path / "cache")
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"immutable test weights")
    records = [
        {"id": 1, "sequence_id": "flight1", "boxes": [[1, 2, 4, 8]], "labels": [0]}
    ]
    metrics = {"coco100": {"mAP": 0.2}, "per_video_mAP": {"flight1": 0.2}}
    save_training_evaluation(config, records, metrics, {}, [checkpoint])
    monkeypatch.setattr(
        engine,
        "setup",
        lambda *a: (_ for _ in ()).throw(AssertionError("GPU must not be initialized")),
    )
    adjusted = efficient_search.full_config(config, official=True)
    # Scheduling and source split do not change deterministic inference on these exact records.
    result = efficient_search._score(tmp_path, checkpoint, adjusted, records)
    assert result["cache_hit"] and result["metrics"] == metrics
    larger_batch = copy.deepcopy(adjusted)
    larger_batch["loader"]["batch_size"] = 2
    assert load_receipt(tmp_path / "cache", identity(checkpoint, larger_batch, records)) is None
    larger_batch["evaluation"]["validation_batch_size"] = 1
    assert load_receipt(tmp_path / "cache", identity(checkpoint, larger_batch, records)) is not None
    for section, key, value in [
        ("model", "image_size", 1280),
        ("runtime", "precision", "fp32"),
    ]:
        altered = copy.deepcopy(adjusted)
        altered[section][key] = value
        assert (
            load_receipt(tmp_path / "cache", identity(checkpoint, altered, records))
            is None
        )
    changed = copy.deepcopy(records)
    changed[0]["boxes"][0][0] = 2
    assert (
        load_receipt(tmp_path / "cache", identity(checkpoint, adjusted, changed))
        is None
    )
    assert (
        load_receipt(
            tmp_path / "cache",
            identity(checkpoint, adjusted, records, options={"tile": 960}),
        )
        is None
    )
    checkpoint.write_bytes(b"different weights")
    assert (
        load_receipt(tmp_path / "cache", identity(checkpoint, adjusted, records))
        is None
    )


def test_actual_warm_start_validation_binds_original_and_saved_weights(
    tiny_run, tmp_path, monkeypatch  # noqa: F811
):  # noqa: F811
    from cvlab.prepare import select_records

    config = copy.deepcopy(tiny_run)
    config["output"] = str(tmp_path / "continued")
    config["data"]["split"] = "official"
    config["evaluation"].update(
        per_video_metrics=True, cache_directory=str(tmp_path / "cache")
    )
    source = Path(tiny_run["output"]) / "best.pt"
    engine.train(
        config, init_checkpoint=source, warm_optimizer=True, yield_after_epoch=True
    )
    records = select_records(config, "val")
    candidate = Path(config["output"]) / "best.pt"
    monkeypatch.setattr(
        engine,
        "setup",
        lambda *a: (_ for _ in ()).throw(AssertionError("No repeat forward")),
    )
    assert efficient_search._score(tmp_path, source, config, records)["cache_hit"]
    assert efficient_search._score(tmp_path, candidate, config, records)["cache_hit"]
