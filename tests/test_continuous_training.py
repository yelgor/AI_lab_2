import copy
import json
from pathlib import Path
import time

import pytest
import torch

from cvlab.common import read_json, write_json
from cvlab.full_data import adopt_continuous_policy, full_epochs, recipe
from test_crop_finetune import tiny_run  # noqa: F401
from test_pipeline import competition  # noqa: F401


def test_continuous_course_keeps_training_after_class_diagnostic(tmp_path, monkeypatch):
    from cvlab import engine

    output = tmp_path / "run"
    config = {
        "output": str(output),
        "loader": {"batch_size": 1},
        "data": {"split": "official"},
        "evaluation": {"robust_selection": False},
        "train": {
            "epochs": 3,
            "training_policy": "continuous",
            "early_stopping_patience": 3,
        },
    }
    write_json(
        output / "initial_validation.json", {"metrics": {"coco100": {"mAP": 0.2}}}
    )
    calls = []

    def train(cfg, **kwargs):
        epoch = len(calls)
        calls.append(kwargs)
        write_json(
            output / f"coverage_epoch_{epoch:04d}.json",
            {"unique_frames": 5, "presentations": 5},
        )
        with (output / "history.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_batches": 5,
                        "metrics": {"coco100": {"mAP": 0.21 + epoch * 0.01}},
                        "selection_decision": {
                            "accepted": False,
                            "reasons": ["class_regression"],
                        },
                    }
                )
                + "\n"
            )
        temporary = output / "last.tmp"
        temporary.write_bytes(f"learned_epoch_{epoch}".encode())
        temporary.replace(
            output / "last.pt"
        )  # Match the production atomic checkpoint writer.

    monkeypatch.setattr(engine, "train", train)
    assert full_epochs(config, tmp_path / "source.pt", 5, time.time() + 60)
    assert len(calls) == 3
    assert calls[0]["init_checkpoint"] == tmp_path / "source.pt"
    assert all(
        call["resume"] == output / "last.pt" and call["init_checkpoint"] is None
        for call in calls[1:]
    )
    assert not (output / "selection_halt.json").exists()
    assert [p.read_bytes() for p in sorted((output / "epochs").glob("epoch_*.pt"))] == [
        f"learned_epoch_{i}".encode() for i in range(3)
    ]


def test_adoption_preserves_halt_checkpoint_and_user_stop(tmp_path):
    write_json(tmp_path / "selection_halt.json", {"reasons": ["class_regression"]})
    (tmp_path / "last.pt").write_bytes(b"last learned state")
    (tmp_path / "STOP").touch()
    adopt_continuous_policy(tmp_path)
    assert not (tmp_path / "selection_halt.json").exists()
    assert read_json(next(tmp_path.glob("selection_halt.strict_*.json")))[
        "reasons"
    ] == ["class_regression"]
    assert (
        next((tmp_path / "epochs").glob("adopted_last_*.pt")).read_bytes()
        == b"last learned state"
    )
    assert (tmp_path / "STOP").exists() and (
        tmp_path / "last.pt"
    ).read_bytes() == b"last learned state"


def test_explicit_policy_migration_preserves_optimizer_schedule_and_training_best(
    tiny_run, tmp_path, monkeypatch  # noqa: F811
):  # noqa: F811
    from cvlab import engine

    source = Path(tiny_run["output"]) / "best.pt"
    config = recipe(tiny_run, tmp_path / "course", epochs=3, training_policy="strict")
    config["loader"].update(workers=0, pin_memory=False)
    values = iter((0.1, 0.25, 0.2))
    monkeypatch.setattr(
        engine, "evaluate", lambda *a, **kw: {"coco100": {"mAP": next(values)}}
    )
    engine.train(
        config, init_checkpoint=source, warm_optimizer=True, yield_after_epoch=True
    )
    output = Path(config["output"])
    before = torch.load(output / "last.pt", weights_only=False)
    assert before["best"] == 0.1  # Strict publication guards refused the epoch.
    changed = copy.deepcopy(config)
    changed["train"].update(
        training_policy="continuous",
        early_stopping_patience=3,
        early_stopping_min_delta=0.001,
    )
    changed["evaluation"]["robust_selection"] = False
    bad = copy.deepcopy(changed)
    bad["train"]["lr"] *= 0.5
    with pytest.raises(ValueError, match="configuration differs in train"):
        engine.train(bad, resume=output / "last.pt", allow_quality_policy_change=True)
    engine.train(
        changed,
        resume=output / "last.pt",
        yield_after_epoch=True,
        allow_quality_policy_change=True,
    )
    after = torch.load(output / "last.pt", weights_only=False)
    assert after["epoch"] == 1 and after["best"] == 0.25
    assert after["scheduler"]["last_epoch"] == before["scheduler"]["last_epoch"] + 1
    key = next(iter(before["optimizer"]["state"]))
    assert (
        after["optimizer"]["state"][key]["step"]
        > before["optimizer"]["state"][key]["step"]
    )
    assert read_json(output / "best_metrics.json")["epoch"] == 0
    assert (output / "quality_policy_transition.json").is_file()


def test_patience_stops_a_plateau_without_launching_a_new_recipe(tmp_path, monkeypatch):
    from cvlab import engine

    config = {
        "output": str(tmp_path),
        "train": {
            "epochs": 5,
            "training_policy": "continuous",
            "early_stopping_patience": 3,
            "early_stopping_min_delta": 0.001,
        },
        "evaluation": {},
    }
    write_json(
        tmp_path / "initial_validation.json", {"metrics": {"coco100": {"mAP": 0.2}}}
    )
    (tmp_path / "history.jsonl").write_text(
        "".join(
            json.dumps({"epoch": i, "metrics": {"coco100": {"mAP": 0.2}}}) + "\n"
            for i in range(3)
        )
    )
    monkeypatch.setattr(
        engine,
        "train",
        lambda *a, **kw: pytest.fail(
            "Plateau must not trigger another training recipe"
        ),
    )
    assert full_epochs(config, tmp_path / "source.pt", 5, time.time() + 60)
