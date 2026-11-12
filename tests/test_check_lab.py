import csv
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


@pytest.fixture
def checker():
    path = Path(__file__).parents[1] / "scripts/check_lab.py"
    spec = importlib.util.spec_from_file_location("check_lab", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def csv_file(path, header, rows):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    return path


@pytest.fixture
def test_images(tmp_path):
    return csv_file(
        tmp_path / "test_images.csv",
        ["image_id", "width", "height"],
        [["a", 100, 50], ["b", 200, 100]],
    )


def test_valid_submission_and_empty_submission_are_distinct(
    checker, test_images, tmp_path
):
    path = csv_file(
        tmp_path / "submission.csv",
        ["image_id", "PredictionString"],
        [["a", "2 0.9 10 5 20 10"], ["b", ""]],
    )
    result = checker.validate_submission(path, test_images)
    assert result["valid"] and result["images"] == 2 and result["detections"] == 1
    assert not result["all_predictions_empty"]
    assert len(result["sha256"]) == 64
    csv_file(path, ["image_id", "PredictionString"], [["a", ""], ["b", ""]])
    result = checker.validate_submission(path, test_images)
    assert result["valid"] and result["all_predictions_empty"]


@pytest.mark.parametrize(
    "prediction",
    [
        "8 0.9 1 1 2 2",
        "0 nan 1 1 2 2",
        "0 1.1 1 1 2 2",
        "0 0.9 -1 1 2 2",
        "0 0.9 1 1 0 2",
        "0 0.9 99 1 2 2",
        "0 0.9 1 1 2",
        " ".join(["0 0.9 1 1 2 2"] * 501),
    ],
)
def test_invalid_predictions_rejected(checker, test_images, tmp_path, prediction):
    path = csv_file(
        tmp_path / "bad.csv",
        ["image_id", "PredictionString"],
        [["a", prediction], ["b", ""]],
    )
    with pytest.raises(ValueError):
        checker.validate_submission(path, test_images)


@pytest.mark.parametrize(
    "rows", [[["a", ""]], [["a", ""], ["a", ""]], [["a", ""], ["c", ""]]]
)
def test_missing_duplicate_unknown_ids(checker, test_images, tmp_path, rows):
    path = csv_file(tmp_path / "bad.csv", ["image_id", "PredictionString"], rows)
    with pytest.raises(ValueError):
        checker.validate_submission(path, test_images)


def test_pending_checkpoint_is_not_a_validated_model(checker, tmp_path):
    run = tmp_path / "retinanet_fold0_local"
    run.mkdir()
    (run / "config.json").write_text(
        json.dumps(
            {
                "model": {"name": "retinanet"},
                "data": {"split": "fold", "fold": 0},
                "train": {"max_batches_per_epoch": None},
            }
        )
    )
    with zipfile.ZipFile(run / "last.pt", "w") as archive:
        archive.writestr("last/data.pkl", b"test container, not tensor validation")
    (run / "val_predictions.json").write_text("not JSON; must not be read")
    result = checker.inspect_run(run)
    assert not result["validated_result"]
    assert result["checkpoint_zip_headers"]["last.pt"]
    assert result["checkpoint_phase"] == "not_loaded"
    assert result["validation_artifact_present"]


def test_completed_smoke_is_not_counted_as_full_model(checker, tmp_path, monkeypatch):
    monkeypatch.setattr(
        checker, "memory_status", lambda: {"gpu_snapshot": "unavailable"}
    )
    run = tmp_path / "runs/fcos_smoke"
    run.mkdir(parents=True)
    (run / "config.json").write_text(
        json.dumps(
            {
                "model": {"name": "fcos"},
                "data": {"split": "fold", "fold": 0},
                "train": {"max_batches_per_epoch": 2},
            }
        )
    )
    (run / "best_metrics.json").write_text(
        json.dumps({"metrics": {"coco100": {"mAP": 0.2}}})
    )
    with zipfile.ZipFile(run / "best.pt", "w") as archive:
        archive.writestr("best/data.pkl", b"test")
    result = checker.audit(tmp_path)
    assert result["runs"][0]["validated_result"]
    assert result["models"]["fcos"]["validated_runs"] == 0
    assert result["ready_for_grading"] == "manual_review_required"


def test_import_does_not_load_torch_or_numpy():
    script = Path(__file__).parents[1] / "scripts/check_lab.py"
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy,sys; runpy.run_path(sys.argv[1]); "
            "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules",
            str(script),
        ],
        check=True,
    )


def test_related_clips_and_inherited_training_do_not_count_as_independent(checker, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model": {"name": "rtdetr"}}))
    (tmp_path / "best_metrics.json").write_text(json.dumps({"metrics": {"coco100": {"mAP": 0.25}}}))
    (tmp_path / "split.json").write_text(json.dumps({
        "train_sequences": ["uav1_a"], "seen_training_sequences": ["uav2_a"],
        "val_sequences": ["uav2_b"],
    }))
    with zipfile.ZipFile(tmp_path / "best.pt", "w") as archive:
        archive.writestr("best/data.pkl", b"test")
    value = checker.inspect_run(tmp_path)
    assert not value["sequence_leakage"] and value["flight_family_leakage"]
    assert not value["validated_result"] and value["mAP"] == 0.25


def test_activity_is_current_local_evidence_and_does_not_claim_worker_count(checker, tmp_path):
    proc, work = tmp_path / "proc", tmp_path / "work"
    (proc / "123").mkdir(parents=True)
    (proc / "123/cmdline").write_bytes(f"python\0-u\0-m\0cvlab\0train\0--config\0{work}/config.yaml\0".encode())
    (proc / "124").mkdir()
    (proc / "124/cmdline").write_bytes(b"python\0unrelated.py\0")
    value = checker.activity(work, proc)
    assert [p["pid"] for p in value["active_processes"]] == [123]
    assert value["plans"] == [] and "workers" in value["limits"]
