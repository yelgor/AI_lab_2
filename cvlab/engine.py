import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from .common import digest, protocol_fingerprint, seed_all, write_json
from .data import make_loader, restore_boxes
from .metrics import evaluate
from .models import build_model
from .prepare import select_records


def resolve_device(config):
    requested = config["runtime"]["device"]
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    return device


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def amp_context(config, device):
    precision = config["runtime"]["precision"]
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("fp16/bf16 are GPU options here; set precision: fp32 for CPU")
    if precision not in ("fp16", "bf16"):
        raise ValueError("precision must be fp32, fp16 or bf16")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bf16; use fp16")
    return torch.autocast(
        device_type="cuda",
        dtype=torch.float16 if precision == "fp16" else torch.bfloat16,
    )


def setup(config):
    seed_all(config["seed"], config["runtime"].get("deterministic", False))
    torch.set_num_threads(config["runtime"].get("cpu_threads", 4))
    device = resolve_device(config)
    # Validate precision before downloading model weights.
    with amp_context(config, device):
        pass
    torch.backends.cuda.matmul.allow_tf32 = config["runtime"].get("tf32", False)
    torch.backends.cudnn.allow_tf32 = config["runtime"].get("tf32", False)
    return device


def environment():
    packages = [
        "torch",
        "torchvision",
        "transformers",
        "numpy",
        "Pillow",
        "pycocotools",
        "timm",
    ]
    try:
        freeze = subprocess.check_output(
            [__import__("sys").executable, "-m", "pip", "freeze"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except subprocess.CalledProcessError:
        freeze = []
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: importlib.metadata.version(p) for p in packages},
        "pip_freeze": freeze,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "devices": [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "vram_gib": torch.cuda.get_device_properties(i).total_memory / 2**30,
            }
            for i in range(torch.cuda.device_count())
        ],
    }


def move(images, targets, device):
    return (
        [image.to(device, non_blocking=True) for image in images],
        [
            {key: value.to(device, non_blocking=True) for key, value in target.items()}
            for target in targets
        ],
    )


@torch.inference_mode()
def predict(model, config, records, device, worker_context=None):
    model.eval()
    loader = make_loader(config, records, False, worker_context=worker_context)
    warmup = config["evaluation"].get("warmup_batches", 2)
    if warmup:
        images, targets, _ = next(iter(loader))
        images, _ = move(images, targets, device)
        for _ in range(warmup):
            with amp_context(config, device):
                model.predict(images)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    predictions, model_times, pipeline_times = [], [], []
    started = time.perf_counter()
    iterator = iter(loader)
    processed = 0
    while True:
        batch_started = time.perf_counter()
        try:
            images, targets, metadata = next(iterator)
        except StopIteration:
            break
        images, _ = move(images, targets, device)
        synchronize(device)
        model_started = time.perf_counter()
        with amp_context(config, device):
            outputs = model.predict(images)
        synchronize(device)
        model_times.append(time.perf_counter() - model_started)
        for output, meta in zip(outputs, metadata):
            boxes = restore_boxes(output["boxes"].float().cpu(), meta["geometry"])
            labels, scores = output["labels"].cpu(), output["scores"].float().cpu()
            for box, label, score in zip(
                boxes.tolist(), labels.tolist(), scores.tolist()
            ):
                x1, y1, x2, y2 = box
                if not all(map(math.isfinite, box + [score])):
                    raise RuntimeError("Non-finite inference output")
                if x2 > x1 and y2 > y1:
                    predictions.append(
                        {
                            "image_id": meta["record"]["id"],
                            "category_id": label,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": score,
                        }
                    )
        processed += len(images)
        pipeline_times.append(time.perf_counter() - batch_started)
    elapsed = time.perf_counter() - started
    timing = {
        "images": processed,
        "pipeline_seconds": elapsed,
        "pipeline_images_per_second": processed / elapsed,
        "model_images_per_second": processed / sum(model_times),
        "model_batch_latency_p50_ms": float(np.percentile(model_times, 50) * 1000),
        "model_batch_latency_p95_ms": float(np.percentile(model_times, 95) * 1000),
        "pipeline_batch_latency_p95_ms": float(
            np.percentile(pipeline_times, 95) * 1000
        ),
        "batch_size": config["loader"]["batch_size"],
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20
        if device.type == "cuda"
        else None,
        "scope": "Warmup excluded; pipeline includes loader startup, reads, preprocessing, transfer, model and box restoration; excludes JSON/CSV serialization and metric computation.",
    }
    return predictions, timing


def rng_state(generator):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader": generator.get_state(),
    }


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    generator.set_state(state["loader"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_checkpoint(path, checkpoint):
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def early_stopping_reached(config, output):
    patience = config["train"].get("early_stopping_patience", 0)
    if not patience:
        return False
    delta = config["train"].get("early_stopping_min_delta", 0.0)
    if patience < 1 or not math.isfinite(delta) or delta < 0:
        raise ValueError("Invalid early-stopping configuration")
    output = Path(output)
    initial = output / "initial_validation.json"
    best = (
        json.loads(initial.read_text())["metrics"]["coco100"]["mAP"]
        if initial.is_file()
        else -math.inf
    )
    history = output / "history.jsonl"
    if not history.is_file():
        return False
    rows = {
        row["epoch"]: row
        for row in (
            json.loads(line)
            for line in history.read_text().splitlines()
            if line.strip()
        )
    }
    stale = 0
    for _, row in sorted(rows.items()):
        score = row["metrics"]["coco100"]["mAP"]
        if score > best + delta:
            best, stale = score, 0
        else:
            stale += 1
    return stale >= patience


def train(config, resume=None, validation_pipeline=None, init_checkpoint=None):
    if resume and init_checkpoint:
        raise ValueError("Use resume or init_checkpoint, not both")
    device = setup(config)
    output = Path(config["output"])
    if resume and Path(resume).resolve() != (output / "last.pt").resolve():
        raise ValueError(
            "Resume with last.pt inside output; transfer the whole run directory between environments"
        )
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(
            f"Run directory already exists: {output}. Choose a new output or use --resume."
        )
    output.mkdir(parents=True, exist_ok=True)
    training, validation = (
        select_records(config, "train"),
        select_records(config, "val"),
    )
    if {r["sequence_id"] for r in training} & {r["sequence_id"] for r in validation}:
        raise ValueError("Sequence leakage between train and validation")
    data_hash = digest({"train": training, "val": validation})
    construction = dict(config["model"])
    checkpoint = None
    initialization = None
    seen_sequences = set()
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        initialization = checkpoint.get("initialization")
        seen_sequences = set(
            checkpoint.get(
                "seen_training_sequences", [r["sequence_id"] for r in training]
            )
        )
    elif init_checkpoint:
        with Path(init_checkpoint).open("rb") as stream:
            source_hash = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                source_hash.update(chunk)
            stream.seek(0)
            checkpoint = torch.load(stream, map_location="cpu", weights_only=False)
        if checkpoint["config"]["model"]["name"] != config["model"]["name"]:
            raise ValueError("Initialization must use the same detector family")
        if checkpoint["config"]["model"] != config["model"]:
            raise ValueError(
                "Warm initialization keeps model configuration unchanged; vary data/train settings"
            )
        if "seen_training_sequences" in checkpoint:
            seen_sequences = set(checkpoint["seen_training_sequences"])
        else:
            source_split = Path(init_checkpoint).parent / "split.json"
            if not source_split.is_file():
                raise ValueError(
                    "Initialization needs source split.json to verify sequence leakage"
                )
            source_split = json.loads(source_split.read_text())
            if source_split["fingerprint"] != checkpoint["data_fingerprint"]:
                raise ValueError(
                    "Source split fingerprint differs from initialization checkpoint"
                )
            seen_sequences = set(source_split["train_sequences"])
        if seen_sequences & {record["sequence_id"] for record in validation}:
            raise ValueError(
                "Initialization would leak supervised training sequences into validation"
            )
        initialization = {
            "checkpoint": str(Path(init_checkpoint).resolve()),
            "sha256": source_hash.hexdigest(),
            "source_epoch": checkpoint["epoch"],
            "source_best_score": checkpoint["best"],
            "fresh_optimizer_schedule_budget": True,
        }
    model = build_model(
        construction,
        checkpoint.get("adapter") if checkpoint else None,
        initialize_pretrained=not bool(resume or init_checkpoint),
    ).to(device)
    if init_checkpoint:
        model.load_state_dict(checkpoint["model"])
        checkpoint = None
    backbone, heads = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (backbone if "backbone" in name else heads).append(parameter)
    tc = config["train"]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone, "lr": tc["lr"] * tc["backbone_lr_factor"]},
            {"params": heads, "lr": tc["lr"]},
        ],
        weight_decay=tc["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tc["epochs"]
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and config["runtime"]["precision"] == "fp16",
    )
    generator = torch.Generator().manual_seed(config["seed"])
    start_epoch, best, elapsed_before = 0, -1.0, 0.0
    pending_validation = None
    if resume:
        # Only load checkpoints you created; torch checkpoints are trusted Python artifacts.
        old = checkpoint["config"]
        for section in ("model", "train", "seed", "evaluation", "runtime", "loader"):
            if old[section] != config[section]:
                raise ValueError(
                    f"Resume configuration differs in {section}; start a separate experiment"
                )
        if checkpoint["data_fingerprint"] != data_hash:
            raise ValueError("Resume data/splits differ from checkpoint")
        if protocol_fingerprint(old) != protocol_fingerprint(config):
            raise ValueError("Resume experiment protocol differs from checkpoint")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng"], generator)
        pending_validation = checkpoint.get("validation_pending")
        start_epoch, best, elapsed_before = (
            checkpoint["epoch"] + (pending_validation is None),
            checkpoint["best"],
            checkpoint["elapsed_seconds"],
        )
        # State has been restored to the model/optimizer. Release the CPU copy
        # loaded from last.pt before training and validation allocate more RAM.
        checkpoint = None
    if validation_pipeline is not None:
        write_json(
            output / "execution.json",
            {
                "validation_mode": "CPU metrics overlapped with GPU training",
                "max_cpu_validation_jobs": 1,
                "wall_clock_budget_includes_overlap": True,
            },
        )
    if initialization:
        write_json(output / "initialization.json", initialization)
    write_json(output / "config.json", config)
    write_json(output / "environment.json", environment())
    write_json(
        output / "split.json",
        {
            "fingerprint": data_hash,
            "train_ids": [r["id"] for r in training],
            "val_ids": [r["id"] for r in validation],
            "train_sequences": sorted({r["sequence_id"] for r in training}),
            "val_sequences": sorted({r["sequence_id"] for r in validation}),
        },
    )
    worker_context = "spawn" if validation_pipeline is not None else None
    loader = make_loader(
        config, training, True, generator, worker_context=worker_context
    )
    started = time.perf_counter()
    accumulation = tc["accumulation"]
    if accumulation < 1:
        raise ValueError("accumulation must be >= 1")

    def snapshot(epoch, progress=None):
        return {
            "config": config,
            "model": model.state_dict(),
            "adapter": model.export_metadata(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": rng_state(generator),
            "epoch": epoch,
            "best": best,
            "elapsed_seconds": elapsed_before + time.perf_counter() - started,
            "data_fingerprint": data_hash,
            "validation_pending": progress,
            "initialization": initialization,
            "seen_training_sequences": sorted(seen_sequences),
        }

    if (
        initialization
        and start_epoch == 0
        and (
            not (output / "initial_validation.json").is_file()
            or not (output / "best.pt").is_file()
        )
    ):
        atomic_checkpoint(output / "last.pt", snapshot(-1))
        print(
            "Evaluating initialization on the new validation split before fine-tuning",
            flush=True,
        )
        initial_predictions, initial_timing = predict(
            model, config, validation, device, worker_context=worker_context
        )
        initial_scores = evaluate(
            validation, initial_predictions, config["evaluation"]["max_detections"]
        )
        initial_score = initial_scores["coco100"]["mAP"]
        if initial_score is None:
            raise ValueError("Initial validation has no evaluable targets")
        best = initial_score
        initial_log = {
            "epoch": -1,
            "train_loss": None,
            "train_batches": 0,
            "epoch_complete": True,
            "elapsed_seconds": elapsed_before + time.perf_counter() - started,
            "metrics": initial_scores,
            "timing": initial_timing,
            "lr": [group["lr"] for group in optimizer.param_groups],
        }
        initial_payload = snapshot(-1)
        atomic_checkpoint(output / "last.pt", initial_payload)
        atomic_checkpoint(output / "best.pt", initial_payload)
        write_json(output / "best_metrics.json", initial_log)
        write_json(output / "initial_validation.json", initial_log)
        del initial_predictions, initial_payload
        print(f"Initial official-validation mAP={best:.4f}", flush=True)

    validation_job = None
    for epoch in range(start_epoch, tc["epochs"]):
        if pending_validation is None and early_stopping_reached(config, output):
            print(
                "EARLY STOP: validation improvement below patience threshold",
                flush=True,
            )
            break
        if (
            pending_validation is None
            and elapsed_before + time.perf_counter() - started >= tc["max_hours"] * 3600
        ):
            break
        if pending_validation is None:
            if validation_pipeline is not None:
                validation_pipeline.memory_gate()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            batches = min(len(loader), tc.get("max_batches_per_epoch") or len(loader))
            total_loss, observed, stop = 0.0, 0, False
            print(f"epoch={epoch} training; up to {batches} batches", flush=True)
            for index, (images, targets, metadata) in enumerate(loader):
                if index >= batches:
                    break
                images, targets = move(images, targets, device)
                group_start = (index // accumulation) * accumulation
                group_size = min(accumulation, batches - group_start)
                with amp_context(config, device):
                    losses = model.loss(images, targets)
                    loss = sum(losses.values())
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss at epoch={epoch}, batch={index}: {losses}"
                    )
                scaler.scale(loss / group_size).backward()
                total_loss += loss.item()
                observed += 1
                seen_sequences.update(
                    meta["record"]["sequence_id"] for meta in metadata
                )
                if observed % 250 == 0:
                    print(
                        f"epoch={epoch} batch={observed}/{batches} mean_loss={total_loss / observed:.4f}",
                        flush=True,
                    )
                boundary = (index + 1) % accumulation == 0 or index + 1 == batches
                if boundary:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tc["clip_grad"])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    # Read only committed metrics; with overlap a previous epoch
                    # can finish while the next one is already using the GPU.
                    if observed % 250 == 0 or index + 1 == batches:
                        if early_stopping_reached(config, output):
                            print(
                                "EARLY STOP: committing current progress; best weights preserved",
                                flush=True,
                            )
                            stop = True
                            break
                    if (
                        elapsed_before + time.perf_counter() - started
                        >= tc["max_hours"] * 3600
                    ):
                        stop = True
                        break
            if not observed:
                break
            if validation_pipeline is not None:
                validation_pipeline.wait()
                if validation_job is not None:
                    best = validation_job.result()
                    validation_job = None
            scheduler.step()
            pending_validation = {
                "loss_sum": total_loss,
                "observed": observed,
                "batches": batches,
                "stop": stop,
            }
            atomic_checkpoint(output / "last.pt", snapshot(epoch, pending_validation))
            print(
                f"epoch={epoch} training complete; checkpoint saved; running validation",
                flush=True,
            )
        else:
            total_loss, observed, batches, stop = (
                pending_validation[key]
                for key in ("loss_sum", "observed", "batches", "stop")
            )
            print(
                f"epoch={epoch} resuming saved validation; no training batches repeated",
                flush=True,
            )
        pending_validation = None
        if validation_pipeline is not None:
            # Bound host memory: never collect a second validation prediction
            # set while CPU metrics still retain the previous set.
            validation_pipeline.wait()
        predictions, timing = predict(
            model, config, validation, device, worker_context=worker_context
        )
        print(f"epoch={epoch} inference complete; computing COCO metrics", flush=True)
        if validation_pipeline is not None:
            from .validation_pipeline import cpu_snapshot

            log = {
                "epoch": epoch,
                "train_loss": total_loss / observed,
                "train_batches": observed,
                "epoch_complete": observed == batches,
                "timing": timing,
                "lr": [group["lr"] for group in optimizer.param_groups],
            }
            payload = cpu_snapshot(snapshot(epoch))
            validation_job = validation_pipeline.submit(
                output,
                config,
                validation,
                predictions,
                timing,
                payload,
                log,
                elapsed_before,
                started,
            )
            del predictions, payload
            if stop:
                break
            continue
        scores = evaluate(
            validation, predictions, config["evaluation"]["max_detections"]
        )
        score = scores["coco100"]["mAP"]
        if score is None:
            raise ValueError("Validation has no evaluable target objects")
        improved = score > best
        best = max(best, score)
        elapsed = elapsed_before + time.perf_counter() - started
        log = {
            "epoch": epoch,
            "train_loss": total_loss / observed,
            "train_batches": observed,
            "epoch_complete": observed == batches,
            "elapsed_seconds": elapsed,
            "metrics": scores,
            "timing": timing,
            "lr": [g["lr"] for g in optimizer.param_groups],
        }
        with (output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(log, allow_nan=False) + "\n")
        payload = snapshot(epoch)
        atomic_checkpoint(output / "last.pt", payload)
        if improved:
            atomic_checkpoint(output / "best.pt", payload)
            write_json(output / "best_metrics.json", log)
            write_json(
                output / "val_predictions.json",
                {
                    "record_ids": [r["id"] for r in validation],
                    "predictions": predictions,
                    "timing": timing,
                    "metrics": scores,
                    "model_config": config["model"],
                    "split": config["data"]["split"],
                    "protocol_fingerprint": protocol_fingerprint(config),
                    "fold": config["data"].get("fold"),
                    "data_fingerprint": digest(validation),
                },
            )
        # Do not retain one full validation output while collecting the next.
        del predictions
        print(
            f"epoch={epoch} loss={total_loss / observed:.4f} mAP={score:.4f} best={best:.4f}",
            flush=True,
        )
        if stop:
            break
    if validation_pipeline is None:
        print(f"Run artifacts: {output.resolve()}")
    else:
        print(
            f"GPU phases complete: {output.resolve()}; CPU metrics may still be running",
            flush=True,
        )
    return validation_job


def load_trained(checkpoint_path, config, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Architecture and losses must match; precision/batch size may be benchmarked separately.
    inference_only = {"score_threshold", "nms_threshold", "max_detections"}
    if checkpoint["config"]["model"]["name"] in {"fasterrcnn", "retinanet", "fcos"}:
        # These CNN detectors have no learned parameters tied to input size.
        # The new size is used by both letterbox and the detector transform.
        inference_only.add("image_size")
    elif checkpoint["config"]["model"]["name"] == "rtdetr":
        hf_config = (checkpoint.get("adapter") or {}).get("hf_config", {})
        # Pinned RT-DETR builds positional encodings and anchors from feature
        # shapes when both settings are explicitly dynamic. Learned weights
        # and query count are independent of input resolution in this case.
        if (
            "eval_size" in hf_config
            and hf_config["eval_size"] is None
            and "anchor_image_size" in hf_config
            and hf_config["anchor_image_size"] is None
        ):
            inference_only.add("image_size")
    trained_architecture = {
        k: v
        for k, v in checkpoint["config"]["model"].items()
        if k not in inference_only
    }
    requested_architecture = {
        k: v for k, v in config["model"].items() if k not in inference_only
    }
    if trained_architecture != requested_architecture:
        raise ValueError("Model configuration differs from checkpoint")
    construction = dict(config["model"])
    model = build_model(
        construction, checkpoint.get("adapter"), initialize_pretrained=False
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    return model


def summarize(paths, output):
    rows = []
    for path in paths:
        path = Path(path)
        config = json.loads((path / "config.json").read_text())
        result = json.loads((path / "best_metrics.json").read_text())
        scores = result["metrics"]["coco100"]
        rows.append(
            {
                "run": str(path),
                "model": config["model"]["name"],
                "split": config["data"]["split"],
                "fold": config["data"].get("fold"),
                "image_size": config["model"]["image_size"],
                "epoch": result["epoch"],
                **{k: v for k, v in scores.items() if k != "per_class"},
                "pipeline_fps": result["timing"]["pipeline_images_per_second"],
                "peak_mb": result["timing"]["peak_allocated_mb"],
            }
        )
    if not rows:
        raise ValueError("No runs to summarize")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Path(output).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
