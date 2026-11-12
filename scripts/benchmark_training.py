"""Short source-preserving training-throughput comparison under the shared GPU lease."""

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import math
from pathlib import Path
import statistics
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cvlab.common import load_config, read_json, write_json
from cvlab.data import make_loader
from cvlab.engine import amp_context, build_model, restore_rng, setup, training_mode
from cvlab.evaluation_receipts import file_sha
from cvlab.prepare import select_records
from cvlab.score_improvement import release_gpu
from cvlab.selection_policy import require_training_source
from cvlab.training_runtime import make_optimizer, train_batches
from cvlab.training_watchdog import gpu_info, command


def available_mib():
    path = Path("/proc/meminfo")
    if not path.is_file():
        return None
    values = dict(line.split(":", 1) for line in path.read_text().splitlines())
    return int(values["MemAvailable"].split()[0]) / 1024


def gpu_snapshot():
    code, output, error = command(
        [
            "nvidia-smi",
            "--query-gpu=name,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    return {"available": code == 0, "output": output if code == 0 else error}


def summarize(cases, effective_batch):
    groups = {}
    for case in cases:
        if case.get("status") == "measured":
            key = (case["batch_size"], case["loss_sync"])
            groups.setdefault(key, []).append(case["train_timing"]["images_per_second"])
    rows = [
        {
            "batch_size": bs,
            "accumulation": effective_batch // bs,
            "loss_sync": mode,
            "repeats": len(values),
            "median_images_per_second": statistics.median(values),
        }
        for (bs, mode), values in groups.items()
    ]
    # Only optimized cases are recommendations; reference mode exists for measurement.
    optimized = [row for row in rows if row["loss_sync"] == "per_update"]
    baseline = next(
        (
            row
            for row in rows
            if row["batch_size"] == 1 and row["loss_sync"] == "per_batch"
        ),
        None,
    )
    best = (
        max(optimized, key=lambda r: r["median_images_per_second"])
        if optimized
        else None
    )
    recommendation = None
    if (
        baseline
        and best
        and baseline["repeats"] >= 2
        and best["repeats"] >= 2
        and best["median_images_per_second"]
        > baseline["median_images_per_second"] * 1.03
    ):
        recommendation = best
    return {
        "measurements": rows,
        "fastest_measured": max(rows, key=lambda r: r["median_images_per_second"])
        if rows
        else None,
        "reference_batch_one": baseline,
        "recommendation": recommendation,
        "recommendation_status": "measured_faster; needs quality validation"
        if recommendation
        else "no_recommendation; insufficient repeats or <3% measured gain",
        "quality_status": "not_evaluated; physical-batch changes require robust validation in a new plan",
    }


def run_case(
    checkpoint,
    config,
    device,
    batch_size,
    mode,
    updates,
    warmup_updates,
    effective_batch,
):
    config = copy.deepcopy(config)
    config["loader"]["batch_size"] = batch_size
    config["train"]["accumulation"] = effective_batch // batch_size
    model = build_model(
        config["model"], checkpoint.get("adapter"), initialize_pretrained=False
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    optimizer = make_optimizer(model, config["train"])
    # load_state_dict may retain CPU step tensors: each case needs its OWN moments/counters.
    optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and config["runtime"]["precision"] == "fp16",
    )
    scaler.load_state_dict(checkpoint["scaler"])
    generator = torch.Generator().manual_seed(config["seed"])
    restore_rng(checkpoint["rng"], generator)
    records = select_records(config, "train")
    loader = make_loader(config, records, True, generator)
    measured_batches = updates * config["train"]["accumulation"]
    if len(loader) < max(
        measured_batches, warmup_updates * config["train"]["accumulation"]
    ):
        raise ValueError("Not enough training frames for requested benchmark")
    training_mode(model, config["train"])
    # RT-DETR denoising memory can grow with GT density. Do not recommend a
    # batch size after testing only easy, sparse frames from a random prefix.
    dense = sorted(records, key=lambda r: len(r["labels"]), reverse=True)[:batch_size]
    stress_loader = make_loader(config, dense, False)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_batches(
        model, stress_loader, optimizer, scaler, config, device, 1, log_every=0
    )
    stress_peak = (
        torch.cuda.max_memory_allocated(device) / 2**20
        if device.type == "cuda"
        else None
    )
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
    scaler.load_state_dict(checkpoint["scaler"])
    restore_rng(checkpoint["rng"], generator)
    del stress_loader
    if warmup_updates:
        train_batches(
            model,
            loader,
            optimizer,
            scaler,
            config,
            device,
            warmup_updates * config["train"]["accumulation"],
            log_every=0,
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    progress = train_batches(
        model,
        loader,
        optimizer,
        scaler,
        config,
        device,
        measured_batches,
        log_every=0,
        loss_sync=mode,
    )
    result = {
        "status": "measured",
        "batch_size": batch_size,
        "accumulation": config["train"]["accumulation"],
        "loss_sync": mode,
        "train_timing": progress["train_timing"],
        "mean_loss": progress["loss_sum"] / progress["observed"],
        "dense_stress": {
            "frame_ids": [r["id"] for r in dense],
            "objects_per_frame": [len(r["labels"]) for r in dense],
            "peak_allocated_mib": stress_peak,
        },
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20
        if device.type == "cuda"
        else None,
    }
    del model, optimizer, scaler, loader
    release_gpu()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument(
        "--source", type=Path, required=True, help="Immutable learned checkpoint"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Actual active recipe JSON/YAML; model must match source",
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--warmup-updates", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--vram-gib", type=float, default=6)
    parser.add_argument("--min-ram-mib", type=int, default=4096)
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait behind healthy GPU work; default exits if busy",
    )
    args = parser.parse_args(argv)
    if (
        args.updates < 1
        or args.warmup_updates < 0
        or args.repeats < 1
        or not math.isfinite(args.vram_gib)
        or args.vram_gib <= 0
        or args.min_ram_mib < 0
    ):
        parser.error("Positive budgets required")
    work, source = args.work.resolve(), args.source.resolve()
    if not work.is_dir() or not source.is_file():
        parser.error("Work directory/source checkpoint not found")
    with (work / ".local_gpu_queue.lock").open("a") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | (0 if args.wait else fcntl.LOCK_NB))
        except BlockingIOError:
            print(
                "GPU lease busy; current work preserved. Run after it finishes or use --wait.",
                flush=True,
            )
            return 1
        if args.device == "cuda" and gpu_info(0).get("compute_pids"):
            print(
                "Another GPU compute process is active; benchmark did not start.",
                flush=True,
            )
            return 1
        free = available_mib()
        if free is not None and free < args.min_ram_mib:
            print(
                f"Insufficient available RAM for benchmark: {free:.0f} MiB", flush=True
            )
            return 1
        source_sha = file_sha(source)
        require_training_source(source_sha)
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        config = copy.deepcopy(checkpoint["config"])
        if args.config:
            config = (
                read_json(args.config)
                if args.config.suffix == ".json"
                else load_config(args.config)
            )
        else:
            from cvlab.full_data import recipe

            config = recipe(config, work / "benchmark_recipe_only", epochs=1)
        if config["model"] != checkpoint["config"]["model"]:
            raise ValueError("Benchmark must preserve checkpoint model configuration")
        effective_batch = (
            config["loader"]["batch_size"] * config["train"]["accumulation"]
        )
        if any(bs < 1 or effective_batch % bs for bs in args.batch_sizes):
            raise ValueError("Batch sizes must divide the recipe's effective batch")
        config["runtime"]["device"] = args.device
        if args.device == "cpu":
            config["runtime"]["precision"] = "fp32"
            config["loader"].update(workers=0, pin_memory=False)
        device = setup(config)
        with amp_context(config, device):
            pass
        if device.type == "cuda":
            total = torch.cuda.get_device_properties(device).total_memory
            torch.cuda.set_per_process_memory_fraction(
                min(0.9, args.vram_gib * 2**30 / total), device
            )
        output = (
            work
            / "benchmarks"
            / (
                datetime.now(timezone.utc).strftime("training_%Y%m%dT%H%M%S_")
                + uuid4().hex[:8]
                + ".json"
            )
        )
        report = {
            "source_sha256": source_sha,
            "device": str(device),
            "effective_batch": effective_batch,
            "config": config,
            "cases": [],
            "models_saved": False,
            "gpu_before": gpu_snapshot() if device.type == "cuda" else None,
            "limits": "Throughput only; no quality metric or submission. Batch-size changes can change loss normalization/denoising despite equal effective batch. Alternating order mitigates but does not eliminate cache/thermal bias.",
        }
        cases = [
            (bs, mode)
            for bs in dict.fromkeys(args.batch_sizes)
            for mode in ("per_batch", "per_update")
        ]
        oom_sizes = set()
        for repeat in range(args.repeats):
            for bs, mode in cases if repeat % 2 == 0 else list(reversed(cases)):
                print(f"BENCH repeat={repeat + 1} batch={bs} sync={mode}", flush=True)
                free = available_mib()
                if free is not None and free < args.min_ram_mib:
                    report["status"] = "stopped_for_ram"
                    write_json(output, report)
                    return 1
                try:
                    result = (
                        {
                            "status": "skipped_after_oom",
                            "batch_size": bs,
                            "loss_sync": mode,
                        }
                        if bs in oom_sizes
                        else run_case(
                            checkpoint,
                            config,
                            device,
                            bs,
                            mode,
                            args.updates,
                            args.warmup_updates,
                            effective_batch,
                        )
                    )
                except torch.cuda.OutOfMemoryError:
                    oom_sizes.add(bs)
                    result = {
                        "status": "out_of_memory",
                        "batch_size": bs,
                        "loss_sync": mode,
                    }
                release_gpu()
                report["cases"].append({"repeat": repeat + 1, **result})
                report["summary"] = summarize(report["cases"], effective_batch)
                write_json(output, report)
                print(result, flush=True)
        if file_sha(source) != source_sha:
            raise ValueError("Source checkpoint changed during benchmark")
        report["gpu_after"] = gpu_snapshot() if device.type == "cuda" else None
        report["status"] = "complete"
        write_json(output, report)
        print(f"Training benchmark saved: {output}", flush=True)
        print(report["summary"], flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
