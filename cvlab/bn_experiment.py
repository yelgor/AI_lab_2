"""Test BatchNorm statistics using train-only calibration and fixed learned weights."""

import copy
import fcntl
import math
import os
import random
from pathlib import Path

import torch
from torch import nn

from .common import digest, read_json, write_json


def batch_norms(model):
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
        and module.track_running_stats
    }


@torch.no_grad()
def restore_statistics(model, state):
    """Transfer only running moments/counters, never affine or learned parameters."""
    norms = batch_norms(model)
    if not norms:
        raise ValueError("No mutable BatchNorm statistics found")
    for name, module in norms.items():
        for key in ("running_mean", "running_var", "num_batches_tracked"):
            full_key = f"{name}.{key}" if name else key
            value = state[full_key]
            target = getattr(module, key)
            if value.shape != target.shape:
                raise ValueError(f"BatchNorm shape mismatch: {full_key}")
            target.copy_(value)
    return sorted(norms)


@torch.no_grad()
def recalibrate(model, config, records, device):
    from .data import make_loader
    from .engine import amp_context

    norms = batch_norms(model)
    if not norms or not records:
        raise ValueError("Need mutable BatchNorm layers and calibration frames")
    original_modes = {module: module.training for module in model.modules()}
    momenta = {module: module.momentum for module in norms.values()}
    ordered = list(records)
    random.Random(config["seed"]).shuffle(ordered)
    loader_config = copy.deepcopy(config)
    loader_config["loader"].update(batch_size=1, workers=0)
    processed = 0
    try:
        model.eval()  # No dropout, denoising queries, losses or optimizer steps.
        for module in norms.values():
            module.reset_running_stats()
            module.momentum = None  # Cumulative average over all calibration batches.
            module.train()
        # training=False: whole-frame inference preprocessing, no crops/flips.
        for images, _, _ in make_loader(loader_config, ordered, False):
            with amp_context(config, device):
                model.predict([image.to(device) for image in images])
            processed += len(images)
            if processed % 64 == 0 or processed == len(ordered):
                print(
                    f"BN calibration {processed}/{len(ordered)} train frames",
                    flush=True,
                )
    finally:
        for module, momentum in momenta.items():
            module.momentum = momentum
        for module, training in original_modes.items():
            module.training = training
    return {"layers": sorted(norms), "frames": processed}


def read_source(path):
    import hashlib

    path = Path(path)
    with path.open("rb") as stream:
        hasher = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
        stream.seek(0)
        checkpoint = torch.load(stream, map_location="cpu", weights_only=False)
    split = read_json(path.parent / "split.json")
    if checkpoint["data_fingerprint"] != split["fingerprint"]:
        raise ValueError("Source split fingerprint differs")
    seen = checkpoint.get("seen_training_sequences", split["train_sequences"])
    # Drop optimizer, scheduler and RNG: these derived artifacts are inference-only.
    payload = {
        key: checkpoint[key]
        for key in ("config", "model", "adapter", "epoch", "data_fingerprint")
    }
    payload["seen_training_sequences"] = sorted(seen)
    payload["inference_only"] = True
    return payload, hasher.hexdigest()


def check_isolation(seen, calibration, validation):
    heldout = {record["sequence_id"] for record in validation}
    if heldout & (set(seen) | {record["sequence_id"] for record in calibration}):
        raise ValueError("Calibration or source training overlaps validation sequences")


def score_value(result):
    value = result["metrics"]["coco100"]["mAP"]
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Validation has no finite mAP")
    return value


def existing_reference(work, source_sha, records, native):
    """Include the existing tiled candidate in the export hurdle, with provenance."""
    path = Path(work) / "score_improvement/tiling_summary.json"
    reference = {"kind": "native", "mAP": score_value(native)}
    if path.is_file():
        for entry in read_json(path)["models"]:
            candidate = entry.get("full_candidate")
            if (
                entry.get("checkpoint_sha256") == source_sha
                and candidate
                and candidate.get("checkpoint_sha256") == source_sha
                and candidate.get("data_fingerprint") == digest(records)
                and candidate.get("record_ids") == [r["id"] for r in records]
                and score_value(candidate) > reference["mAP"]
            ):
                reference = {"kind": "existing tiled", "mAP": score_value(candidate)}
    return reference


def run(work, max_images=192, calibration_images=256, vram_gib=6, lock_fd=None):
    from .engine import atomic_checkpoint, load_trained, setup
    from .inference_experiments import balanced_subset
    from .prepare import select_records
    from .score_improvement import export, file_hash, full_validation, release_gpu
    from .submit_cpu import _available_ram_mib, _cuda_budget

    work = Path(work).resolve()
    if (
        max_images < 1
        or calibration_images < 1
        or not math.isfinite(vram_gib)
        or vram_gib <= 0
    ):
        raise ValueError("Positive subset, calibration size and GPU budget required")
    expected, actual = (work / ".local_gpu_queue.lock").stat(), os.fstat(lock_fd)
    if (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino):
        raise ValueError("GPU lease belongs to another work directory")
    with os.fdopen(lock_fd, "a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        available = _available_ram_mib()
        if available is not None and available < 4096:
            raise RuntimeError("Need 4 GiB available RAM; close heavy applications")
        original, source_sha = read_source(work / "runs/rtdetr_fold0_local/best.pt")
        config = copy.deepcopy(original["config"])
        config["data"].update(split="official", frame_step=1)
        config["loader"].update(batch_size=1, workers=0, persistent_workers=False)
        config["runtime"].update(device="cuda", precision="fp16", cpu_threads=2)
        records = select_records(config, "val")
        subset = balanced_subset(records, max_images)
        calibration = balanced_subset(
            select_records(config, "train"), calibration_images
        )
        sources = {"original": (original, source_sha)}
        crop_path = work / "runs/rtdetr_crop_finetune/last.pt"
        if crop_path.is_file():
            sources["crop"] = read_source(crop_path)
        for payload, _ in sources.values():
            if payload["config"]["model"] != original["config"]["model"]:
                raise ValueError(
                    "BN comparison requires identical model configurations"
                )
            check_isolation(payload["seen_training_sequences"], calibration, records)
        settings = {
            "version": 1,
            "source_sha256": {k: v[1] for k, v in sources.items()},
            "calibration_fingerprint": digest(calibration),
            "validation_fingerprint": digest(records),
            "subset_fingerprint": digest(subset),
            "config": config,
        }
        directory = work / "bn_diagnostic" / digest(settings)[:12]
        directory.mkdir(parents=True, exist_ok=True)
        report = {
            "settings": settings,
            "calibration_ids": [r["id"] for r in calibration],
            "cases": [],
            "submission": None,
        }
        cases = [("original", "unchanged"), ("original", "recalibrated")]
        if "crop" in sources:
            cases += [
                ("crop", "unchanged"),
                ("crop", "original_statistics"),
                ("crop", "recalibrated"),
            ]
        size = config["model"]["image_size"]
        device = setup(config)
        _cuda_budget(device, vram_gib)
        for source, operation in cases:
            label = f"{source}_{operation}"
            print(
                f"BN evaluating case={label}; subset={len(subset)} frames", flush=True
            )
            case_dir = directory / label
            case_dir.mkdir(exist_ok=True)
            path = case_dir / "best.pt"
            payload, sha = sources[source]
            if not path.is_file():
                temporary_source = case_dir / "source.pt"
                atomic_checkpoint(temporary_source, payload)
                model = load_trained(temporary_source, config, device)
                temporary_source.unlink()
                details = {}
                if operation == "original_statistics":
                    details["layers"] = restore_statistics(model, original["model"])
                elif operation == "recalibrated":
                    details = recalibrate(model, config, calibration, device)
                derived = dict(payload)
                derived["model"] = {
                    k: value.detach().cpu().clone()
                    for k, value in model.state_dict().items()
                }
                derived["bn_experiment"] = {
                    "operation": operation,
                    "source_sha256": sha,
                    "settings": settings,
                    **details,
                }
                if operation == "recalibrated":
                    derived["seen_training_sequences"] = sorted(
                        set(payload["seen_training_sequences"])
                        | {r["sequence_id"] for r in calibration}
                    )
                atomic_checkpoint(path, derived)
                del model, derived
                release_gpu()
            write_json(case_dir / "config.json", config)
            cache = case_dir / "subset.json"
            if not cache.is_file():
                write_json(cache, full_validation(config, path, size, subset))
            result = read_json(cache)
            if result["data_fingerprint"] != digest(subset):
                raise ValueError("Cached subset differs")
            entry = {"case": label, "run": str(case_dir), "subset": result}
            report["cases"].append(entry)
            print(f"BN case={label}; subset mAP={score_value(result):.6f}", flush=True)
            write_json(directory / "summary.json", report)
        # Prefer the untouched reference on ties.
        winner = max(report["cases"], key=lambda case: score_value(case["subset"]))
        report["winner"] = winner["case"]
        if winner is report["cases"][0]:
            report["outcome"] = (
                "No subset improvement; learned weights and original artifacts preserved"
            )
        else:
            for entry in (report["cases"][0], winner):
                case_dir = Path(entry["run"])
                cache = case_dir / "full.json"
                if not cache.is_file():
                    print(
                        f"BN FULL evaluating case={entry['case']}; images={len(records)}",
                        flush=True,
                    )
                    write_json(
                        cache,
                        full_validation(config, case_dir / "best.pt", size, records),
                    )
                entry["full"] = read_json(cache)
                if entry["full"]["data_fingerprint"] != digest(records):
                    raise ValueError("Cached full validation differs")
                print(
                    f"BN FULL case={entry['case']}; mAP={score_value(entry['full']):.6f}",
                    flush=True,
                )
            reference = existing_reference(
                work, source_sha, records, report["cases"][0]["full"]
            )
            gain = score_value(winner["full"]) - reference["mAP"]
            report.update(reference=reference, full_gain=gain, minimum_export_gain=0.01)
            if gain + 1e-12 >= 0.01:
                case_dir = Path(winner["run"])
                checkpoint_sha = file_hash(case_dir / "best.pt")
                destination = (
                    work / "submissions" / f"rtdetr_bn_{checkpoint_sha[:12]}_{size}.csv"
                )
                export(case_dir, config, checkpoint_sha, size, destination, vram_gib)
                report["submission"] = str(destination)
                report["outcome"] = "Full-validation improvement confirmed; CSV ready"
                print(f"BN CANDIDATE READY FOR KAGGLE: {destination}", flush=True)
            else:
                report["outcome"] = (
                    f"Gain over {reference['kind']}={gain:.6f}; below export threshold"
                )
        write_json(directory / "summary.json", report)
        write_json(work / "bn_diagnostic/summary.json", report)
        print(report["outcome"], flush=True)
        return report
