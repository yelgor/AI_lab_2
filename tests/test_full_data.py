import json
from pathlib import Path

import pytest
import torch

from cvlab.common import read_json, write_json
from cvlab.full_data import joint_manifest, manifest_summary, recipe, sha256
from test_crop_finetune import tiny_run  # noqa: F401
from test_pipeline import competition  # noqa: F401


def test_joint_manifest_rekeys_colliding_ids_without_mutating_sources(tmp_path):
    records = [{"id": 1, "sequence_id": "uav1_a", "frame_id": 1, "file_name": "train/a.jpg"}]
    validation = [{"id": 1, "sequence_id": "uav2_a", "frame_id": 1, "file_name": "val/b.jpg"}]
    write_json(tmp_path / "train.json", {"images": records})
    write_json(tmp_path / "val.json", {"images": validation})
    source_sha = sha256(tmp_path / "train.json")
    out = tmp_path / "joint/trainval.json"
    summary = joint_manifest(tmp_path, out)
    assert summary["frames"] == 2 and summary["source_counts"] == {"train": 1, "val": 1}
    merged = read_json(out)["images"]
    assert [r["id"] for r in merged] == [1, 2]
    assert [r["origin_id"] for r in merged] == [1, 1]
    assert [r["origin_split"] for r in merged] == ["train", "val"]
    assert sha256(tmp_path / "train.json") == source_sha
    assert joint_manifest(tmp_path, out) == summary
    assert manifest_summary(tmp_path)["train"]["frames"] == 1
    validation[0]["sequence_id"] = "uav1_b"
    write_json(tmp_path / "val.json", {"images": validation})
    with pytest.raises(ValueError, match="families overlap"):
        manifest_summary(tmp_path)


def test_final_fit_trains_all_labels_without_inventing_validation_AP(tiny_run, tmp_path, monkeypatch):  # noqa: F811
    from cvlab import engine
    from cvlab.prepare import select_records

    source = Path(tiny_run["output"]) / "best.pt"
    merged = tmp_path / "joint.json"
    summary = joint_manifest(tiny_run["data"]["prepared"], merged)
    report = tmp_path / "selection_report.json"
    write_json(report, {"full_heldout_validation": True, "selected_checkpoint_sha256": sha256(source)})
    config = recipe(tiny_run, tmp_path / "final", epochs=1, final=True, manifest=merged, report=report)
    config["loader"].update(workers=0, pin_memory=False)
    monkeypatch.setattr(engine, "predict", lambda *a, **kw: pytest.fail("Final fit must not evaluate seen validation"))
    engine.train(config, init_checkpoint=source, warm_optimizer=True)
    output = Path(config["output"])
    checkpoint = torch.load(output / "final.pt", weights_only=False)
    assert checkpoint["best"] is None
    assert checkpoint["evaluation_status"] == "no_independent_validation_final_fit"
    assert not (output / "best.pt").exists()
    row = json.loads((output / "history.jsonl").read_text())
    assert "metrics" not in row and "mAP" not in row
    assert row["train_batches"] == summary["frames"] == 8
    assert checkpoint["training_coverage"]["unique_frames"] == 8
    assert checkpoint["validation_pending"] is None
    with pytest.raises(ValueError, match="no independent validation"):
        select_records(config, "val")
    report.write_text("{}")
    with pytest.raises(ValueError, match="report changed"):
        engine.train(config, resume=output / "last.pt")


def test_full_data_controller_runs_both_real_training_phases_and_does_not_repeat(tiny_run, tmp_path, monkeypatch):  # noqa: F811
    from cvlab import efficient_search, tiled_inference
    from cvlab.full_data import run

    work = tmp_path / "work"
    work.mkdir()
    source = Path(tiny_run["output"])
    original_sha = sha256(source / "best.pt")

    def score(work, checkpoint, config, records, **kwargs):
        assert len(records) == 2  # Full official validation, no fold/probe.
        value = 0.2 if checkpoint.parent == source else 0.21
        return {"metrics": {"coco100": {"mAP": value, "per_class": {"person": value}},
                            "per_video_mAP": {str(i): value for i in range(4)}}}

    def export(model, config, records, device, options, path, checkpoint_sha):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("image_id,PredictionString\nz_0000001,0 0.9 1 1 5 5\n")
        write_json(path.with_suffix(".csv.metadata.json"), {
            "checkpoint_sha256": checkpoint_sha,
            "validation": {"sha256": sha256(path), "valid": True},
        })

    monkeypatch.setattr(efficient_search, "_score", score)
    monkeypatch.setattr(tiled_inference, "export_tiled", export)
    # Keep the tiny fixture's no-worker CPU configuration for both recipes.
    from cvlab import full_data
    real_recipe = full_data.recipe
    def tiny_recipe(*args, **kwargs):
        config = real_recipe(*args, **kwargs)
        config["loader"].update(workers=0, pin_memory=False)
        config["evaluation"]["robust_selection"] = False
        return config
    monkeypatch.setattr(full_data, "recipe", tiny_recipe)
    state = run(work, source, selection_epochs=1, final_epochs=1)
    assert state["status"] == "READY_FOR_KAGGLE"
    selection = work / "runs/rtdetr_full_train_selection"
    final = work / "runs/rtdetr_full_data_final"
    assert read_json(selection / "coverage_summary.json")["dataset_size"] == 6
    assert read_json(final / "coverage_summary.json")["dataset_size"] == 8
    report = read_json(work / "full_data/selection_report.json")
    assert report["full_heldout_validation"] and report["frames"] == 2
    checkpoint = torch.load(final / "final.pt", weights_only=False)
    assert checkpoint["best"] is None
    assert sha256(source / "best.pt") == original_sha
    assert run(work, source, selection_epochs=1, final_epochs=1) == state


def test_recorded_public_regression_is_rejected_before_training(tmp_path, monkeypatch):
    from cvlab import full_data

    monkeypatch.setattr(full_data, "sha256", lambda p: "224c8cc6f9877a4cb4196c2ec514527964cf18c4f9a8d85031fc09cc234c8f9d")
    with pytest.raises(ValueError, match="rejected training source"):
        full_data.run(tmp_path, tmp_path / "rejected_source")


def test_measured_physical_batch_keeps_full_data_update_size(tiny_run, tmp_path):  # noqa: F811
    config = recipe(tiny_run, tmp_path / "batch2", epochs=1, batch_size=2)
    assert config["loader"]["batch_size"] == 2
    assert config["train"]["accumulation"] == 1
    assert config["evaluation"]["validation_batch_size"] == 1
    assert tiny_run["train"]["accumulation"] == 2  # Source config stays immutable.
    with pytest.raises(ValueError, match="divide"):
        recipe(tiny_run, tmp_path / "batch3", epochs=1, batch_size=3)


def test_full_epoch_coverage_counts_frames_instead_of_physical_batches(tmp_path, monkeypatch):
    import time
    from cvlab import engine, full_data

    output = tmp_path / "run"
    config = {"output": str(output), "loader": {"batch_size": 2}, "evaluation": {},
              "data": {"split": "official"}, "train": {"epochs": 1}}
    def train(*args, **kwargs):
        write_json(output / "coverage_epoch_0000.json", {"unique_frames": 5, "presentations": 5})
        (output / "history.jsonl").write_text(json.dumps({"epoch": 0, "train_batches": 3}) + "\n")
    monkeypatch.setattr(engine, "train", train)
    assert full_data.full_epochs(config, tmp_path / "source.pt", 5, time.time() + 60)
    summary = read_json(output / "coverage_summary.json")
    assert summary["actual_samples"] == 5 and summary["completed_full_epochs"] == 1


def test_harmful_update_stops_before_another_full_epoch(tmp_path, monkeypatch):
    from cvlab import engine, full_data
    import time

    output = tmp_path / "selection"
    config = {"output": str(output), "data": {"split": "official"},
              "train": {"epochs": 2}, "evaluation": {}}
    calls = []
    def train(*args, **kwargs):
        calls.append(kwargs)
        write_json(output / "coverage_epoch_0000.json", {"unique_frames": 10})
        (output / "history.jsonl").write_text(json.dumps({
            "epoch": 0, "train_batches": 10,
            "selection_decision": {"accepted": False, "reasons": ["class_regression"]},
        }) + "\n")
    monkeypatch.setattr(engine, "train", train)
    assert not full_data.full_epochs(config, tmp_path / "source.pt", 10, time.time() + 60)
    assert len(calls) == 1 and (output / "selection_halt.json").is_file()
    assert not full_data.full_epochs(config, tmp_path / "source.pt", 10, time.time() + 60)
    assert len(calls) == 1  # An automatic restart cannot silently continue the rejected recipe.
