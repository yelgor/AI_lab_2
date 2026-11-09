"""Paired official-validation resolution search and immutable submission candidates."""

import copy
import fcntl
import gc
import hashlib
import math
import os
import shutil
from pathlib import Path

from .common import digest, read_json, write_json

TARGET = 0.2216324580


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def release_gpu():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def freeze_run(source, directory):
    import torch

    checkpoint_path = source / "best.pt"
    checkpoint_hash = file_hash(checkpoint_path)
    frozen = directory / f"{source.name}_{checkpoint_hash[:12]}"
    frozen.mkdir(parents=True, exist_ok=True)
    destination = frozen / "best.pt"
    if not destination.is_file():
        temporary = destination.with_suffix(".tmp")
        shutil.copyfile(checkpoint_path, temporary)
        temporary.replace(destination)
    if file_hash(destination) != checkpoint_hash:
        raise ValueError("Frozen checkpoint checksum differs")
    config = read_json(source / "config.json")
    split = read_json(source / "split.json")
    checkpoint = torch.load(destination, map_location="cpu", weights_only=False)
    if checkpoint["data_fingerprint"] != split["fingerprint"]:
        raise ValueError("Source split does not match checkpoint")
    if config["model"] != checkpoint["config"]["model"]:
        raise ValueError("Saved model configuration does not match checkpoint")
    seen = checkpoint.get("seen_training_sequences", split["train_sequences"])
    config["data"].update(split="official")
    config["loader"].update(
        batch_size=1, workers=0, pin_memory=False, persistent_workers=False
    )
    config["runtime"].update(device="cuda", precision="fp16", cpu_threads=2)
    write_json(frozen / "config.json", config)
    write_json(
        frozen / "source.json",
        {
            "run": str(source),
            "sha256": checkpoint_hash,
            "seen_training_sequences": sorted(seen),
        },
    )
    del checkpoint
    release_gpu()
    return frozen, config, checkpoint_hash, set(seen)


def export(frozen, config, checkpoint_hash, size, destination, vram_gib):
    from .submit_cpu import create_submission
    from scripts.check_lab import validate_submission

    if destination.is_file():
        metadata = read_json(
            destination.with_suffix(destination.suffix + ".metadata.json")
        )
        if (
            metadata["checkpoint_sha256"] != checkpoint_hash
            or metadata["inference_config"]["model"]["image_size"] != size
        ):
            raise ValueError("Existing submission uses another checkpoint or size")
        root = Path(config["data"]["root"])
        validate_submission(
            destination, root / "test_images.csv", root / "sample_submission.csv"
        )
        print(f"Existing candidate verified: {destination}", flush=True)
        return
    release_gpu()
    create_submission(
        frozen,
        destination,
        threads=2,
        chunk_size=16,
        device_name="cuda",
        vram_gib=vram_gib,
        image_size=size,
    )
    release_gpu()


def full_validation(config, checkpoint, size, records):
    from .engine import load_trained, predict, setup
    from .metrics import evaluate

    adjusted = copy.deepcopy(config)
    adjusted["model"]["image_size"] = size
    device = setup(adjusted)
    model = load_trained(checkpoint, adjusted, device)
    predictions, timing = predict(model, adjusted, records, device)
    scores = evaluate(records, predictions, adjusted["evaluation"]["max_detections"])
    del predictions, model
    release_gpu()
    return {
        "image_size": size,
        "metrics": scores,
        "timing": timing,
        "record_ids": [r["id"] for r in records],
        "data_fingerprint": digest(records),
    }


def select_case(cases):
    valid = [
        case
        for case in cases
        if isinstance(case["metrics"]["coco100"]["mAP"], (int, float))
        and math.isfinite(case["metrics"]["coco100"]["mAP"])
    ]
    if not valid:
        raise ValueError("No evaluable resolution cases")
    return max(
        valid, key=lambda case: (case["metrics"]["coco100"]["mAP"], -case["image_size"])
    )


def run(work, models, sizes, max_images, vram_gib, lock_fd):
    import torch
    from .inference_experiments import balanced_subset, compare_resolutions
    from .prepare import select_records

    work = Path(work).resolve()
    if (
        len(models) != len(set(models))
        or not models
        or max_images < 1
        or set(models) - {"rtdetr", "fasterrcnn", "fcos", "retinanet"}
    ):
        raise ValueError("Need unique model names and positive validation subset size")
    if not sizes or any(size < 64 or size % 32 for size in sizes):
        raise ValueError("Inference sizes must be multiples of 32, >=64")
    if not math.isfinite(vram_gib) or vram_gib <= 0:
        raise ValueError("Need a finite positive GPU memory budget")
    expected = (work / ".local_gpu_queue.lock").stat()
    inherited = os.fstat(lock_fd)
    if (expected.st_dev, expected.st_ino) != (inherited.st_dev, inherited.st_ino):
        raise ValueError("Inherited GPU lease is for another work directory")
    with os.fdopen(lock_fd, "a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory = work / "score_improvement"
        directory.mkdir(exist_ok=True)
        report = {
            "target_kaggle_score": TARGET,
            "measured_kaggle_score": None,
            "validation_split": "official",
            "models": [],
        }
        report_path = directory / "summary.json"
        frozen_runs = []
        # Produce useful native-resolution CSVs before the validation search.
        for name in models:
            source = work / "runs" / f"{name}_fold0_local"
            frozen, config, sha, seen = freeze_run(source, directory)
            records = select_records(config, "val")
            if seen & {r["sequence_id"] for r in records}:
                raise ValueError("Official validation contains training sequences")
            native = config["model"]["image_size"]
            destination = work / "submissions" / f"{name}_{sha[:12]}_{native}.csv"
            export(frozen, config, sha, native, destination, vram_gib)
            entry = {
                "model": name,
                "checkpoint_sha256": sha,
                "native_image_size": native,
                "native_submission": str(destination),
                "probes": [],
                "selected_image_size": native,
                "selected_submission": str(destination),
            }
            report["models"].append(entry)
            write_json(report_path, report)
            frozen_runs.append((frozen, config, sha, records, entry))
            print(f"FIRST CANDIDATE READY FOR KAGGLE: {destination}", flush=True)
        for frozen, config, sha, records, entry in frozen_runs:
            name, native = entry["model"], entry["native_image_size"]
            subset = balanced_subset(records, max_images)
            subset_ids = [record["id"] for record in subset]
            subset_fingerprint = digest(subset)
            for size in sorted({native, *sizes}):
                path = frozen / f"probe_{size}.json"
                try:
                    release_gpu()
                    if path.is_file():
                        probe = read_json(path)
                        if (
                            probe["checkpoint_sha256"] != sha
                            or probe["record_ids"] != subset_ids
                            or probe["data_fingerprint"] != subset_fingerprint
                        ):
                            raise ValueError(
                                "Cached resolution probe checkpoint or validation subset differs"
                            )
                    else:
                        probe = compare_resolutions(
                            config,
                            frozen / "best.pt",
                            path,
                            sizes=(size,),
                            max_images=max_images,
                        )
                    entry["probes"].append(probe["experiments"][0])
                except torch.cuda.OutOfMemoryError:
                    entry.setdefault("skipped_oom_sizes", []).append(size)
                    release_gpu()
                    print(
                        f"Skipping {name} size={size}: GPU memory budget exceeded",
                        flush=True,
                    )
                write_json(report_path, report)
            selected = select_case(entry["probes"])["image_size"]
            if selected == native:
                entry["selection_reason"] = (
                    "Native resolution wins the paired subset comparison"
                )
                write_json(report_path, report)
                continue
            # Both sizes are measured on ALL official validation frames. Never
            # compare a subset score or an old fold score against this holdout.
            confirmed = []
            for size in (native, selected):
                path = frozen / f"official_full_{size}.json"
                try:
                    if path.is_file():
                        result = read_json(path)
                        if (
                            result["image_size"] != size
                            or result["data_fingerprint"] != digest(records)
                            or result["record_ids"] != [r["id"] for r in records]
                        ):
                            raise ValueError("Cached full validation data differs")
                    else:
                        result = full_validation(
                            config, frozen / "best.pt", size, records
                        )
                        write_json(path, result)
                    confirmed.append(result)
                    print(
                        f"FULL official validation {name} size={size}: mAP={result['metrics']['coco100']['mAP']}",
                        flush=True,
                    )
                except torch.cuda.OutOfMemoryError:
                    entry.setdefault("full_validation_oom_sizes", []).append(size)
                    release_gpu()
                    break
            entry["full_validation"] = confirmed
            if len(confirmed) == 2:
                winner = select_case(confirmed)
                selected = winner["image_size"]
                destination = work / "submissions" / f"{name}_{sha[:12]}_{selected}.csv"
                export(frozen, config, sha, selected, destination, vram_gib)
                entry.update(
                    selected_image_size=selected,
                    selected_submission=str(destination),
                    selection_reason="Paired full official validation",
                )
            write_json(report_path, report)
        print(f"SCORE CANDIDATES COMPLETE: {report_path}", flush=True)
        for entry in report["models"]:
            print(f"UPLOAD TO KAGGLE: {entry['selected_submission']}", flush=True)
        return report
