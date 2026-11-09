import threading

import pytest
import torch

from cvlab.validation_pipeline import ValidationPipeline, cpu_snapshot


def test_cpu_snapshot_is_detached_and_does_not_alias_live_weights():
    parameter = torch.tensor([1.0], requires_grad=True)
    original = {"model": {"w": parameter}, "optimizer": [{"x": (parameter,)}]}
    saved = cpu_snapshot(original)
    with torch.no_grad():
        parameter.add_(10)
    assert saved["model"]["w"].item() == 1
    assert not saved["model"]["w"].requires_grad
    assert saved["optimizer"][0]["x"][0].device.type == "cpu"


def test_pipeline_prevents_two_prediction_sets_and_propagates_errors(monkeypatch):
    from cvlab import validation_pipeline

    started, release = threading.Event(), threading.Event()

    def blocked():
        started.set()
        assert release.wait(10)
        return 1

    monkeypatch.setattr(validation_pipeline, "finish_validation", blocked)
    with ValidationPipeline() as pipeline:
        job = pipeline.submit()
        assert started.wait(5)
        try:
            with pytest.raises(RuntimeError, match="previous CPU validation"):
                pipeline.submit()
        finally:
            release.set()
        pipeline.wait()
        assert job.result() == 1

    def failed():
        raise RuntimeError("CPU validation failed")

    monkeypatch.setattr(validation_pipeline, "finish_validation", failed)
    with pytest.raises(RuntimeError, match="CPU validation failed"):
        with ValidationPipeline() as pipeline:
            pipeline.submit()


def test_next_model_training_starts_before_previous_metrics_finish(
    tmp_path, monkeypatch
):
    import yaml
    from cvlab import engine, overlap_queue, validation_pipeline
    from scripts import local_gpu_queue

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/retinanet_fold0_local.yaml").write_text("{}")
    original_pipeline = validation_pipeline.ValidationPipeline
    monkeypatch.setattr(
        validation_pipeline, "ValidationPipeline", lambda: original_pipeline(0)
    )
    started, next_model = threading.Event(), threading.Event()

    def metric_job():
        started.set()
        assert next_model.wait(10), "Next GPU model was blocked by CPU metrics"

    def config(project, work, reference, name):
        path = work / "configs" / f"{name}.yaml"
        path.write_text(yaml.safe_dump({"model": {"name": name}}))
        return path, work / name

    def training(config, resume, validation_pipeline):
        if config["model"]["name"] == "fcos":
            validation_pipeline.submit()
            assert started.wait(5)
        else:
            next_model.set()

    monkeypatch.setattr(local_gpu_queue, "model_config", config)
    monkeypatch.setattr(engine, "train", training)
    monkeypatch.setattr(validation_pipeline, "finish_validation", metric_job)
    overlap_queue.run(tmp_path, ["fcos", "fasterrcnn"])
    assert next_model.is_set()
