import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from cvlab import efficient_search as search
from cvlab.common import load_config, read_json, write_json
from cvlab.sequential_policy import (
    local_rates,
    paired_evidence,
    per_video_scores,
    promotable,
    survivors,
    validation_records,
)
from test_crop_finetune import tiny_run  # noqa: F401
from test_pipeline import competition  # noqa: F401


def test_paired_evidence_and_cost_allocation():
    base = {str(i): 0.2 for i in range(8)}
    good = paired_evidence(base, {k: v + 0.02 for k, v in base.items()})
    assert good["lower"] == pytest.approx(0.02)
    assert good["positive_fraction"] == 1
    bad = paired_evidence(base, {k: v - 0.02 for k, v in base.items()})
    observations = [
        {"id": i, "batches": 6000, "seconds_per_batch": cost, "evidence": ev}
        for i, cost, ev in [(0, 2, good), (1, 1, good), (2, 0.1, bad)]
    ]
    assert [x["id"] for x in survivors(observations, 2, 12000)] == [1, 0]
    assert promotable({"evidence": good, "global_gain": 0.002})
    assert not promotable({"evidence": good, "global_gain": -0.001})
    mixed = paired_evidence(
        base, {k: v + (-1) ** int(k) * 0.05 for k, v in base.items()}
    )
    assert mixed["lower"] < 0 < mixed["upper"]
    assert not promotable({"evidence": mixed, "global_gain": 0.02})
    with pytest.raises(ValueError, match="same"):
        paired_evidence(base, {"other": 0.3})
    np.testing.assert_allclose(
        local_rates(1e-5), [1e-5 / 10**0.5, 1e-5, 1e-5 * 10**0.5]
    )


def test_allocation_tracks_new_champion_instead_of_old_source():
    base = {str(i): 0.20 for i in range(5)}
    candidate = {str(i): 0.22 for i in range(5)}
    row = {
        "evidence": paired_evidence(base, candidate),
        "per_video_mAP": candidate,
        "probe_mAP": 0.22,
        "global_gain": 0.02,
    }
    assert promotable(row)
    search.rebase_observations(
        [row],
        {"per_video_mAP": {str(i): 0.23 for i in range(5)}, "coco100": {"mAP": 0.23}},
    )
    assert row["source_evidence"]["mean_gain"] > 0
    assert row["evidence"]["upper"] < 0 and not promotable(row)


def test_family_scores_combine_correlated_excerpts():
    records = [
        {
            "id": i,
            "sequence_id": sequence,
            "width": 64,
            "height": 64,
            "boxes": [[0, 0, 10, 10]],
            "labels": [0],
            "ignore_boxes": [],
        }
        for i, sequence in enumerate(
            ["uav0000013_00000_v", "uav0000013_01000_v", "uav0000014_00000_v"]
        )
    ]
    predictions = [
        {"image_id": i, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9}
        for i in (0, 2)
    ]
    scores = per_video_scores(records, predictions)
    assert set(scores) == {"uav0000013", "uav0000014"}
    assert 0.49 < scores["uav0000013"] < 0.51
    assert scores["uav0000014"] == pytest.approx(1)


def test_probe_excludes_other_excerpts_of_training_flights(monkeypatch):
    from cvlab import prepare

    records = [
        {"id": i, "sequence_id": s}
        for i, s in enumerate(["uav1_a", "uav2_a", "uav2_b", "uav3_a"])
    ]
    monkeypatch.setattr(prepare, "select_records", lambda *a: records)
    config = {
        "evaluation": {"independent_video_families": True, "validation_images": 2}
    }
    result = validation_records(config, [{"sequence_id": "uav1_b"}])
    assert len(result) == 2 and all(r["sequence_id"] != "uav1_a" for r in result)
    assert records == validation_records({"evaluation": {}}, [])


@pytest.mark.parametrize("asynchronous", [False, True])
def test_real_engine_probe_resume_preserves_optimizer_and_full_training(
    tiny_run,  # noqa: F811
    tmp_path,
    monkeypatch,
    asynchronous,  # noqa: F811
):
    from cvlab import engine, validation_pipeline
    from cvlab.prepare import select_records

    config = copy.deepcopy(tiny_run)
    config["output"] = str(tmp_path / "probe")
    config["evaluation"].update(validation_images=1, per_video_metrics=True)
    config["train"].update(epochs=100, max_batches_per_epoch=None)
    config["loader"]["batch_size"] = 1
    sizes = []

    def predict(model, cfg, records, *a, **kw):
        sizes.append(len(records))
        return [], {}

    monkeypatch.setattr(engine, "predict", predict)
    source = Path(tiny_run["output"]) / "best.pt"
    output = Path(config["output"])
    if asynchronous:
        with validation_pipeline.ValidationPipeline(0) as pipeline:
            engine.train(
                config,
                init_checkpoint=source,
                warm_optimizer=True,
                yield_after_epoch=True,
                round_batches=1,
                validation_pipeline=pipeline,
            )
    else:
        engine.train(
            config,
            init_checkpoint=source,
            warm_optimizer=True,
            yield_after_epoch=True,
            round_batches=1,
        )
    split = read_json(output / "split.json")
    assert len(split["val_ids"]) == 1
    assert len(split["train_ids"]) == len(select_records(config, "train"))
    assert sizes == [1, 1]
    first = torch.load(output / "last.pt", weights_only=False)
    assert "per_video_mAP" in read_json(output / "initial_validation.json")["metrics"]
    row = json.loads((output / "history.jsonl").read_text().splitlines()[0])
    assert "per_video_mAP" in row["metrics"]
    engine.train(
        config, resume=output / "last.pt", yield_after_epoch=True, round_batches=1
    )
    second = torch.load(output / "last.pt", weights_only=False)
    assert (
        second["optimizer"]["state"][0]["step"]
        == first["optimizer"]["state"][0]["step"] + 1
    )


def test_race_recovers_measured_stage_without_retraining_and_reuses_center(
    tmp_path, monkeypatch
):
    from cvlab import engine, prepare, score_improvement

    original = tmp_path / "runs/rtdetr_fold0_local"
    original.mkdir(parents=True)
    source = original / "best.pt"
    source.write_bytes(b"public-best")
    config = load_config(Path(__file__).parents[1] / "configs/rtdetr.yaml")
    write_json(original / "config.json", config)
    real_hash = score_improvement.file_hash
    monkeypatch.setattr(
        score_improvement,
        "file_hash",
        lambda p: "a7cd89fb349d" + "0" * 52 if Path(p) == source else real_hash(p),
    )
    monkeypatch.setattr(score_improvement, "release_gpu", lambda: None)
    records = [{"id": i, "sequence_id": f"clip{i}"} for i in range(5)]
    monkeypatch.setattr(
        prepare,
        "select_records",
        lambda cfg, split: records
        if split == "val"
        else [{"id": 99, "sequence_id": "train"}],
    )
    calls = []
    interrupt = [True]

    def fake_train(cfg, **kwargs):
        output = Path(cfg["output"])
        output.mkdir(parents=True, exist_ok=True)
        history = output / "history.jsonl"
        rows = (
            [json.loads(s) for s in history.read_text().splitlines()]
            if history.exists()
            else []
        )
        epoch = len(rows)
        calls.append((str(output), epoch, kwargs["round_batches"], kwargs))
        if not rows:
            assert kwargs["init_checkpoint"] == source and kwargs["warm_optimizer"]
        else:
            assert kwargs["resume"] == output / "last.pt"
        gain = -0.04 if cfg["train"]["lr"] > 2e-5 else 0.02 + epoch * 0.001
        base = {
            "coco100": {"mAP": 0.2},
            "per_video_mAP": {str(i): 0.2 for i in range(5)},
        }
        write_json(
            output / "initial_validation.json", {"metrics": base, "elapsed_seconds": 1}
        )
        write_json(output / "config.json", cfg)
        write_json(output / "split.json", {"train_sequences": ["train"]})
        row = {
            "epoch": epoch,
            "train_batches": kwargs["round_batches"],
            "elapsed_seconds": 1 + (epoch + 1) * 60,
            "metrics": {
                "coco100": {"mAP": 0.2 + gain},
                "per_video_mAP": {str(i): 0.2 + gain for i in range(5)},
            },
        }
        rows.append(row)
        history.write_text("".join(json.dumps(r) + "\n" for r in rows))
        (output / "last.pt").write_bytes(f"{output}:{epoch}".encode())
        if interrupt[0] and len(calls) == 2:
            interrupt[0] = False
            raise RuntimeError("power loss after validation")

    monkeypatch.setattr(engine, "train", fake_train)
    monkeypatch.setattr(
        search, "confirm", lambda *a: {"accepted": False, "native_gain": -0.001}
    )

    def execute(**kwargs):
        lease = (tmp_path / ".local_gpu_queue.lock").open("a")
        fd = __import__("os").dup(lease.fileno())
        lease.close()
        return search.run(tmp_path, fd, **kwargs)

    with pytest.raises(RuntimeError, match="power loss"):
        execute()
    state = execute()
    assert state["cycles"][0]["complete"]
    assert len(calls) == 6 and sum(c[2] for c in calls) == 42000
    assert state["champion"] == str(original)
    assert source.read_bytes() == b"public-best"
    frozen = Path(state["cycles"][0]["observations"][0]["run"]) / "best.pt"
    checksum = hashlib.sha256(frozen.read_bytes()).hexdigest()
    execute()
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == checksum
    # The center LR is unchanged after a failed cycle: all its existing rungs are reused.
    assert sum(1 for c in calls if c[0].endswith("cycle_001/trial_1")) <= 3


def test_efficient_launcher_preserves_gpu_lease_and_options(tmp_path, monkeypatch):
    from cvlab.cli import main

    script = Path(__file__).parents[1] / "scripts/start_efficient_search.py"
    spec = importlib.util.spec_from_file_location("efficient_launcher", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "runs/rtdetr_fold0_local/best.pt"
    source.parent.mkdir(parents=True)
    source.touch()
    argv = ["start", "--work", str(tmp_path), "--hours", "5", "--export-tiles"]
    monkeypatch.setattr(sys, "argv", argv)
    calls = []
    monkeypatch.setattr(search, "run", lambda *a, **kw: calls.append((a, kw)))

    def execute(binary, args):
        monkeypatch.setattr(sys, "argv", ["cvlab", *args[4:]])
        main()

    monkeypatch.setattr(module.os, "execv", execute)
    monkeypatch.setattr(module.os, "chdir", lambda *a: None)
    module.main()
    assert calls[0][1]["hours"] == 5 and calls[0][1]["export_tiles"]
    assert calls[0][1]["budgets"] == [6000, 12000, 24000]
    with (tmp_path / ".efficient_search.launcher.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        monkeypatch.setattr(sys, "argv", argv)
        module.main()
    assert len(calls) == 1


@pytest.mark.parametrize(
    "native,tiled,accepted,calls_expected",
    [
        (0.209, 0.25, False, 2),
        (0.22, 0.19, False, 4),
        (0.22, 0.202, True, 4),
    ],
)
def test_full_confirmation_requires_native_and_tiled_gains(
    tmp_path,
    monkeypatch,
    native,
    tiled,
    accepted,
    calls_expected,
):
    from cvlab import prepare

    original = tmp_path / "runs/rtdetr_fold0_local"
    candidate = tmp_path / "candidate"
    config = load_config(Path(__file__).parents[1] / "configs/rtdetr.yaml")
    for folder in (original, candidate):
        write_json(folder / "config.json", config)
        write_json(folder / "split.json", {"train_sequences": ["training_flight"]})
    monkeypatch.setattr(
        prepare, "select_records", lambda *a: [
            {"id": i, "sequence_id": f"held_out_{i}"} for i in range(4)
        ]
    )
    calls = []

    def score(work, checkpoint, cfg, records, *, tiled=False):
        assert "validation_images" not in cfg["evaluation"]
        calls.append(tiled)
        value = (
            (0.20 if tiled else 0.21)
            if checkpoint.parent == original
            else (tiled_score if tiled else native)
        )
        return {"metrics": {"coco100": {"mAP": value}}}

    tiled_score = tiled
    monkeypatch.setattr(search, "_score", score)
    result = search.confirm(tmp_path, candidate, original, 0.001, False)
    assert result["accepted"] == accepted
    assert len(calls) == calls_expected


def test_confirmation_filters_related_and_inherited_families(tmp_path, monkeypatch):
    from cvlab import prepare

    original = tmp_path / "runs/rtdetr_fold0_local"
    candidate = tmp_path / "candidate"
    config = load_config(Path(__file__).parents[1] / "configs/rtdetr.yaml")
    for folder in (original, candidate):
        write_json(folder / "config.json", config)
        write_json(folder / "split.json", {
            "train_sequences": ["uav1_a"],
            "seen_training_sequences": ["uav2_a"],
        })
    records = [{"id": i, "sequence_id": f"uav{i}_b"} for i in range(1, 7)]
    monkeypatch.setattr(prepare, "select_records", lambda *a: records)
    calls = []

    def score(work, checkpoint, cfg, selected, **kwargs):
        calls.append([r["sequence_id"] for r in selected])
        return {"metrics": {"coco100": {"mAP": 0.2}}}

    monkeypatch.setattr(search, "_score", score)
    result = search.confirm(tmp_path, candidate, original, 0.001, False)
    assert calls == [[f"uav{i}_b" for i in range(3, 7)]] * 2
    assert result["native_frames"] == 4
    assert result["native_excluded_related_frames"] == 2
    assert not result["accepted"]
    monkeypatch.setattr(prepare, "select_records", lambda *a: records[:4])
    with pytest.raises(ValueError, match="four unseen"):
        search.confirm(tmp_path, candidate, original, 0.001, False)
    assert len(calls) == 2  # Rejected before spending any scoring compute.


def test_wall_budget_commits_partial_training_instead_of_killing_process(
    tiny_run,  # noqa: F811
    tmp_path,
):
    from cvlab import engine

    config = copy.deepcopy(tiny_run)
    config["output"] = str(tmp_path / "bounded")
    config["loader"]["batch_size"] = 1
    config["train"].update(epochs=100, accumulation=1, max_batches_per_epoch=None)
    source = Path(tiny_run["output"]) / "best.pt"
    engine.train(
        config,
        init_checkpoint=source,
        warm_optimizer=True,
        round_batches=10,
        round_seconds=1e-9,
        yield_after_epoch=True,
    )
    checkpoint = torch.load(Path(config["output"]) / "last.pt", weights_only=False)
    assert checkpoint["validation_pending"] is None
    rows = (Path(config["output"]) / "history.jsonl").read_text().splitlines()
    assert len(rows) == 1 and json.loads(rows[0])["train_batches"] == 1


def test_confirmation_cache_is_bound_to_weights_records_and_inference(
    tmp_path, monkeypatch
):
    from cvlab import score_improvement

    original = tmp_path / "runs/rtdetr_fold0_local"
    original.mkdir(parents=True)
    checkpoint = original / "best.pt"
    checkpoint.write_bytes(b"weights")
    config = load_config(Path(__file__).parents[1] / "configs/rtdetr.yaml")
    write_json(original / "config.json", config)
    calls = []

    def validate(*args):
        calls.append(args)
        return {"metrics": {"coco100": {"mAP": 0.2}}}

    monkeypatch.setattr(score_improvement, "full_validation", validate)
    records = [{"id": 1, "sequence_id": "held_out"}]
    search._score(tmp_path, checkpoint, config, records)
    search._score(tmp_path, checkpoint, config, records)
    assert len(calls) == 1
    checkpoint.write_bytes(b"new weights")
    search._score(tmp_path, checkpoint, config, records)
    config["model"]["score_threshold"] = 0.02
    search._score(tmp_path, checkpoint, config, records)
    search._score(tmp_path, checkpoint, config, [{"id": 2, "sequence_id": "held_out"}])
    assert len(calls) == 4
