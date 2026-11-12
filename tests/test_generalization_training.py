import json
import copy
from pathlib import Path

import pytest
import torch

from test_crop_finetune import tiny_run  # noqa: F401
from test_pipeline import competition  # noqa: F401

from cvlab.common import load_config, read_json, write_json
from cvlab.generalization_training import next_block, run


def test_larger_budget_visits_full_pool_between_initial_and_final_partial_blocks():
    completed, blocks = 0, []
    while completed < 72000:
        block = next_block(completed, 16121, 72000)
        blocks.append(block)
        completed += block
    assert blocks == [6000, 16121, 16121, 16121, 16121, 1516]
    assert next_block(72000, 16121, 72000) == 0


@pytest.mark.parametrize("degrade,interrupt,legacy", [
    (False, False, False), (True, False, False),
    (False, True, False), (True, False, True),
])
def test_continuation_preserves_completed_blocks_and_stops_on_validation(tmp_path, monkeypatch, degrade, interrupt, legacy):
    from cvlab import engine, efficient_search, prepare, score_improvement

    source = tmp_path / "runs/rtdetr_fold0_local"
    output = tmp_path / "runs/rtdetr_sampling_flight_time"
    source.mkdir(parents=True)
    (source / "best.pt").write_bytes(b"original learned checkpoint")
    config = load_config(Path(__file__).parents[1] / "configs/rtdetr.yaml")
    write_json(source / "config.json", config)
    def metrics(score):
        return {"coco100": {"mAP": score},
                "per_video_mAP": {f"uav{i}": score for i in range(4)}}

    write_json(source / "best_metrics.json", {"metrics": metrics(0.2)})
    write_json(source / "split.json", {"val_ids": [99]})
    write_json(tmp_path / "efficient_search/state.json", {"champion": str(source)})
    monkeypatch.setattr(prepare, "select_records", lambda *a: [None] * 10000)
    monkeypatch.setattr(score_improvement, "release_gpu", lambda: None)
    calls = []

    def train(cfg, **kwargs):
        output.mkdir(parents=True, exist_ok=True)
        epoch = len(calls)
        assert (kwargs["resume"] is not None) == bool(epoch)
        assert kwargs["warm_optimizer"] == (epoch == 0)
        assert kwargs["round_batches"] <= 10000
        calls.append(kwargs["round_batches"])
        write_json(output / "config.json", cfg)
        write_json(output / "split.json", {"val_ids": [1, 2, 3, 4]})
        write_json(output / "initial_validation.json", {"metrics": metrics(0.2)})
        (output / "last.pt").write_bytes(f"learned block {epoch}".encode())
        score = 0.21 - 0.01 * epoch if degrade else 0.21 + 0.01 * epoch
        row = {"epoch": epoch, "train_batches": kwargs["round_batches"], "metrics": metrics(score)}
        with (output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if interrupt and epoch == 0:
            raise RuntimeError("simulated exit after committed metrics")

    monkeypatch.setattr(engine, "train", train)
    confirmations = []
    def confirm(*args):
        confirmations.append(args)
        return {"accepted": not degrade, "native_gain": 0.01}
    monkeypatch.setattr(efficient_search, "confirm", confirm)
    maximum = 72000 if degrade else 12000
    if legacy:
        cfg = efficient_search.probe_config(config, output, 10 ** -5.5, 512)
        cfg["loader"].update(sampling="flight_time", temporal_bins=8)
        train(cfg, resume=None, warm_optimizer=True, round_batches=6000)
    if interrupt:
        with pytest.raises(RuntimeError, match="simulated exit"):
            run(tmp_path, maximum=maximum)
    plan = run(tmp_path, maximum=maximum)
    assert plan["completed_frames"] == sum(calls)
    assert plan["target_kaggle_score"] == 0.28
    assert plan["observed_kaggle_score"] is None
    assert len(confirmations) == 1
    if degrade:
        assert calls == [6000, 10000, 10000, 10000]
        assert plan["training_stop_reason"].startswith("three_validation")
        assert plan["status"] == "rejected_by_full_validation"
        assert read_json(tmp_path / "efficient_search/state.json")["champion"] == str(source)
    else:
        assert calls == [6000, 6000]
        assert plan["status"] == "accepted"
        assert read_json(tmp_path / "efficient_search/state.json")["champion"] == str(output / "rung_2")
    assert (source / "best.pt").read_bytes() == b"original learned checkpoint"
    for i in range(len(calls)):
        assert (output / f"rung_{i+1}/best.pt").read_bytes() == f"learned block {i}".encode()
    run(tmp_path, maximum=maximum)
    assert len(calls) == (4 if degrade else 2)


def test_real_engine_flight_sampler_resume_matches_uninterrupted_weights(tiny_run, tmp_path):  # noqa: F811
    from cvlab import engine
    from cvlab.prepare import select_records

    config = copy.deepcopy(tiny_run)
    config["loader"].update(sampling="flight_time", temporal_bins=2, batch_size=1)
    config["train"].update(epochs=2, max_batches_per_epoch=None)
    source = Path(tiny_run["output"]) / "best.pt"
    pool = len(select_records(config, "train"))
    config["output"] = str(tmp_path / "continuous")
    engine.train(config, init_checkpoint=source, warm_optimizer=True)
    continuous = torch.load(Path(config["output"]) / "last.pt", weights_only=False)
    config["output"] = str(tmp_path / "resumed")
    engine.train(config, init_checkpoint=source, warm_optimizer=True,
                 yield_after_epoch=True, round_batches=pool)
    last = Path(config["output"]) / "last.pt"
    engine.train(config, resume=last, yield_after_epoch=True, round_batches=pool)
    resumed = torch.load(last, weights_only=False)
    for name, value in continuous["model"].items():
        torch.testing.assert_close(value, resumed["model"][name], rtol=0, atol=0)
    assert torch.equal(continuous["rng"]["loader"], resumed["rng"]["loader"])
    coverage = read_json(Path(config["output"]) / "coverage_epoch_0001.json")
    assert coverage["unique_frames"] == pool
