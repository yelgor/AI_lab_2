import copy
import fcntl
import importlib.util
import os
from pathlib import Path

import pytest
import torch

from cvlab import score_improvement as score
from cvlab.common import read_json, write_json


def case(size, value):
    return {"image_size": size, "metrics": {"coco100": {"mAP": value}}}


def test_select_case_prefers_smaller_size_on_ties_and_rejects_nan():
    assert (
        score.select_case([case(1280, 0.2), case(960, 0.2), case(1536, float("nan"))])[
            "image_size"
        ]
        == 960
    )
    with pytest.raises(ValueError):
        score.select_case([case(960, None)])


def test_frozen_checkpoint_is_immutable_and_official_split_is_separate(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    config = {
        "model": {"name": "rtdetr", "image_size": 960},
        "data": {"split": "fold"},
        "runtime": {},
        "loader": {},
    }
    write_json(source / "config.json", config)
    write_json(
        source / "split.json",
        {"fingerprint": "data-hash", "train_sequences": ["train"]},
    )
    checkpoint = {
        "config": config,
        "data_fingerprint": "data-hash",
        "model": {"w": torch.ones(1)},
    }
    torch.save(checkpoint, source / "best.pt")
    monkeypatch.setattr(score, "release_gpu", lambda: None)
    frozen, adjusted, sha, seen = score.freeze_run(source, tmp_path / "experiments")
    assert seen == {"train"} and adjusted["data"]["split"] == "official"
    assert read_json(source / "config.json")["data"]["split"] == "fold"
    assert score.file_hash(frozen / "best.pt") == sha
    torch.save({"different": True}, source / "best.pt")
    assert score.file_hash(frozen / "best.pt") == sha


@pytest.mark.parametrize("candidate_score,selected", [(0.5, 1280), (0.3, 960)])
def test_full_validation_controls_selection_and_native_csvs_are_exported_first(
    tmp_path, monkeypatch, candidate_score, selected
):
    from cvlab import inference_experiments, prepare

    work = tmp_path
    records = [{"id": 1, "sequence_id": "heldout"}, {"id": 2, "sequence_id": "heldout"}]
    config = {
        "model": {"name": "rtdetr", "image_size": 960},
        "data": {"split": "official"},
    }
    events = []

    def freeze(source, directory):
        frozen = directory / source.name
        frozen.mkdir(parents=True, exist_ok=True)
        adjusted = copy.deepcopy(config)
        adjusted["model"]["name"] = source.name.split("_")[0]
        return frozen, adjusted, "checksum", {"train"}

    def export(frozen, config, sha, size, output, budget):
        events.append(("export", config["model"]["name"], size))

    def probe(config, checkpoint, path, sizes, max_images):
        size = sizes[0]
        events.append(("probe", config["model"]["name"], size))
        if size == 1536:
            raise torch.cuda.OutOfMemoryError("synthetic budget limit")
        return {"experiments": [case(size, 0.1 if size == 960 else 0.9)]}

    def full(config, checkpoint, size, actual_records):
        assert actual_records == records
        events.append(("full", config["model"]["name"], size))
        result = case(size, 0.4 if size == 960 else candidate_score)
        result.update(
            record_ids=[r["id"] for r in records],
            data_fingerprint=score.digest(records),
        )
        return result

    monkeypatch.setattr(score, "freeze_run", freeze)
    monkeypatch.setattr(score, "export", export)
    monkeypatch.setattr(score, "full_validation", full)
    monkeypatch.setattr(score, "release_gpu", lambda: None)
    monkeypatch.setattr(prepare, "select_records", lambda *args: records)
    monkeypatch.setattr(inference_experiments, "compare_resolutions", probe)
    with (work / ".local_gpu_queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = score.run(
            work,
            ["rtdetr", "fasterrcnn"],
            [960, 1280, 1536],
            1,
            6,
            os.dup(lock.fileno()),
        )
    assert events[:2] == [("export", "rtdetr", 960), ("export", "fasterrcnn", 960)]
    assert all(entry["selected_image_size"] == selected for entry in result["models"])
    assert all(entry["skipped_oom_sizes"] == [1536] for entry in result["models"])
    assert result["target_kaggle_score"] == 0.2216324580
    assert result["measured_kaggle_score"] is None
    full_calls = len([event for event in events if event[0] == "full"])
    with (work / ".local_gpu_queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        score.run(
            work,
            ["rtdetr", "fasterrcnn"],
            [960, 1280, 1536],
            1,
            6,
            os.dup(lock.fileno()),
        )
    assert len([event for event in events if event[0] == "full"]) == full_calls


def test_official_validation_leak_is_rejected_before_submission(tmp_path, monkeypatch):
    from cvlab import prepare

    config = {"model": {"image_size": 960}}
    monkeypatch.setattr(
        score, "freeze_run", lambda *a: (tmp_path, config, "sha", {"seen"})
    )
    monkeypatch.setattr(prepare, "select_records", lambda *a: [{"sequence_id": "seen"}])
    with (tmp_path / ".local_gpu_queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(ValueError, match="training sequences"):
            score.run(tmp_path, ["rtdetr"], [960], 1, 6, os.dup(lock.fileno()))


def test_wrong_inherited_gpu_lease_is_rejected(tmp_path):
    (tmp_path / ".local_gpu_queue.lock").touch()
    with (tmp_path / "unrelated").open("a") as other:
        with pytest.raises(ValueError, match="another work directory"):
            score.run(tmp_path, ["rtdetr"], [960], 1, 6, other.fileno())


def test_launcher_inherits_gpu_lease_without_starting_duplicate(tmp_path, monkeypatch):
    import sys

    script = Path(__file__).parents[1] / "scripts/start_score_improvement.py"
    spec = importlib.util.spec_from_file_location("score_launcher", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "runs").mkdir()
    monkeypatch.setattr(sys, "argv", ["start", "--work", str(tmp_path)])
    executed = []

    def execute(binary, args):
        fd = int(args[args.index("--lock-fd") + 1])
        assert os.get_inheritable(fd)
        assert os.fstat(fd).st_ino == (tmp_path / ".local_gpu_queue.lock").stat().st_ino
        executed.append(args)

    monkeypatch.setattr(module.os, "execv", execute)
    monkeypatch.setattr(module.os, "chdir", lambda path: None)
    module.main()
    assert executed[0][2:5] == ["-m", "cvlab", "evaluate-score"]
    executed.clear()
    with (tmp_path / ".score_improvement.launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        module.main()
    assert not executed
