import copy
import importlib.util
from pathlib import Path

import pytest
import torch

from cvlab.common import read_json
from cvlab.evaluation_receipts import file_sha
from cvlab.training_runtime import LossTotal, make_optimizer, train_batches
from test_crop_finetune import tiny_run  # noqa: F401
from test_pipeline import competition  # noqa: F401


class Quadratic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))

    def loss(self, images, targets):
        return {"loss": (self.weight - images[0]).square().sum()}


def configuration(device):
    return {
        "train": {
            "accumulation": 2,
            "lr": 0.01,
            "backbone_lr_factor": 0.1,
            "weight_decay": 0.001,
            "clip_grad": 1.0,
        },
        "runtime": {"precision": "fp32", "device": str(device)},
    }


def batches(values):
    for i, value in enumerate(values):
        yield [torch.tensor([value])], [{}], [{"record": {"id": i}}]


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
def test_update_readback_preserves_parameters_moments_and_partial_group(device):
    device = torch.device(device)
    cfg = configuration(device)
    results = []
    for mode in ("per_batch", "per_update"):
        model = Quadratic().to(device)
        optimizer = make_optimizer(model, cfg["train"])
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        result = train_batches(
            model,
            batches([2.0, 3.0, 4.0, 1.0, 2.0]),
            optimizer,
            scaler,
            cfg,
            device,
            5,
            loss_sync=mode,
            log_every=0,
        )
        results.append(
            (
                model.weight.detach().clone(),
                copy.deepcopy(optimizer.state_dict()),
                result,
            )
        )
    torch.testing.assert_close(results[0][0], results[1][0], atol=0, rtol=0)
    for parameter, state in results[0][1]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(
                value, results[1][1]["state"][parameter][name], atol=0, rtol=0
            )
    assert results[0][2]["loss_sum"] == results[1][2]["loss_sum"]
    old, new = results[0][2]["train_timing"], results[1][2]["train_timing"]
    assert new["images"] == old["images"] == 5
    assert new["optimizer_update_boundaries"] == old["optimizer_update_boundaries"] == 3
    assert old["loss_scalar_reads"] == 5 and new["loss_scalar_reads"] == 1
    assert old["loss_finite_checks"] == 5 and new["loss_finite_checks"] == 3


@pytest.mark.parametrize("mode", ["per_batch", "per_update"])
def test_nonfinite_accumulation_never_steps_weights_or_moments(mode):
    device = torch.device("cpu")
    cfg = configuration(device)
    model = Quadratic()
    optimizer = make_optimizer(model, cfg["train"])
    with pytest.raises(RuntimeError, match="Non-finite loss"):
        train_batches(
            model,
            batches([2.0, float("nan")]),
            optimizer,
            torch.amp.GradScaler("cuda", enabled=False),
            cfg,
            device,
            2,
            loss_sync=mode,
        )
    assert model.weight.item() == 0.5 and not optimizer.state


def test_budget_stop_is_on_update_boundary_and_consumes_no_extra_batch():
    device = torch.device("cpu")
    cfg, model = configuration(device), Quadratic()
    seen = []
    result = train_batches(
        model,
        batches([2.0, 3.0, 4.0, 1.0]),
        make_optimizer(model, cfg["train"]),
        torch.amp.GradScaler("cuda", enabled=False),
        cfg,
        device,
        4,
        on_sample=lambda metadata: seen.extend(m["record"]["id"] for m in metadata),
        on_update=lambda n, total: True,
    )
    assert result["stop"] and result["observed"] == 2 and seen == [0, 1]


def test_statistics_do_not_retain_autograd_graphs():
    meter = LossTotal(torch.device("cpu"))
    for _ in range(10):
        meter.add(torch.tensor(3.0, requires_grad=True).square())
    assert not meter.total.requires_grad and meter.total.grad_fn is None
    assert meter.value() == 90


def benchmark_module():
    script = Path(__file__).parents[1] / "scripts/benchmark_training.py"
    spec = importlib.util.spec_from_file_location("training_benchmark", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validation_batch_override_does_not_mutate_training_batch(competition):  # noqa: F811
    from cvlab import engine
    from cvlab.common import load_config
    from cvlab.prepare import select_records

    root, prepared = competition
    config = load_config(Path(__file__).parents[1] / "configs/fcos.yaml")
    config["data"].update(root=str(root), prepared=str(prepared), split="official")
    config["loader"].update(batch_size=2, workers=0, pin_memory=False)
    config["model"]["image_size"] = 128
    config["runtime"].update(device="cpu", precision="fp32")
    config["evaluation"]["validation_batch_size"] = 1

    class Empty(torch.nn.Module):
        def predict(self, images):
            assert len(images) == 1
            return [
                {
                    "boxes": torch.empty(0, 4),
                    "labels": torch.empty(0, dtype=torch.long),
                    "scores": torch.empty(0),
                }
                for image in images
            ]

    _, timing = engine.predict(
        Empty(), config, select_records(config, "val"), torch.device("cpu")
    )
    assert timing["batch_size"] == 1 and config["loader"]["batch_size"] == 2


def test_real_benchmark_loop_preserves_source_and_writes_only_measurements(
    tiny_run,  # noqa: F811
    tmp_path,
):  # noqa: F811
    source = Path(tiny_run["output"]) / "best.pt"
    fingerprint = file_sha(source)
    module = benchmark_module()
    assert (
        module.main(
            [
                "--work",
                str(tmp_path),
                "--source",
                str(source),
                "--device",
                "cpu",
                "--updates",
                "1",
                "--warmup-updates",
                "0",
                "--repeats",
                "1",
            ]
        )
        == 0
    )
    report = read_json(next((tmp_path / "benchmarks").glob("*.json")))
    assert file_sha(source) == fingerprint
    assert len(report["cases"]) == 4 and not report["models_saved"]
    assert all(
        case["train_timing"]["images"] == report["effective_batch"]
        for case in report["cases"]
    )
    assert report["summary"]["fastest_measured"]["median_images_per_second"] > 0
    assert report["summary"]["recommendation"] is None
    assert report["summary"]["quality_status"].startswith("not_evaluated")


def test_throughput_summary_does_not_recommend_noise_or_failed_memory_case():
    module = benchmark_module()
    cases = [{"status": "measured", "batch_size": 1, "loss_sync": mode,
              "train_timing": {"images_per_second": speed}}
             for mode, speed in [("per_batch", 10), ("per_update", 10.1)] for _ in range(2)]
    cases.append({"status": "out_of_memory", "batch_size": 2, "loss_sync": "per_update"})
    assert module.summarize(cases, 2)["recommendation"] is None
    cases.extend({"status": "measured", "batch_size": 2, "loss_sync": "per_update",
                  "train_timing": {"images_per_second": speed}} for speed in (12, 12.2))
    assert module.summarize(cases, 2)["recommendation"]["batch_size"] == 2
