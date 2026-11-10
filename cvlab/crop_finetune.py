"""Warm fine-tuning on all train clips with native crops and an untouched holdout."""

import copy
import fcntl
import math
import os
from pathlib import Path

import yaml

from .common import read_json, write_json


def recipe(source, work, hours=2, epochs=4, lr=5e-5, crop_size=768, full_frames=False):
    if (
        not math.isfinite(hours)
        or hours <= 0
        or epochs < 1
        or not math.isfinite(lr)
        or lr <= 0
        or crop_size < 64
    ):
        raise ValueError(
            "Positive budget, epochs, learning rate and crop size >=64 required"
        )
    name = source["model"]["name"]
    config = copy.deepcopy(source)
    mode = "fullframe" if full_frames else "crop"
    config["output"] = str(Path(work) / "runs" / f"{name}_{mode}_finetune")
    config["data"].update(
        split="official",
        frame_step=1 if full_frames else 2,
        crop_probability=0.0 if full_frames else 0.5,
        crop_size=crop_size,
    )
    config["loader"].update(
        batch_size=1,
        workers=2,
        prefetch_factor=1,
        persistent_workers=False,
        pin_memory=True,
    )
    config["train"].update(
        epochs=epochs,
        max_hours=hours,
        lr=lr,
        backbone_lr_factor=0.1,
        accumulation=4,
        max_batches_per_epoch=None,
    )
    config["runtime"].update(device="cuda", precision="fp16", cpu_threads=4)
    if full_frames:
        config["train"].update(
            early_stopping_patience=2, early_stopping_min_delta=0.001
        )
    return config


def run(work, models, hours, epochs, lr, crop_size, lock_fd, full_frames=False):
    from .engine import train
    from .score_improvement import release_gpu, file_hash
    from .submit_cpu import create_submission
    from .validation_pipeline import ValidationPipeline

    work = Path(work).resolve()
    if (
        not models
        or len(models) != len(set(models))
        or set(models) - {"rtdetr", "fasterrcnn", "fcos", "retinanet"}
    ):
        raise ValueError("Unique supported detector names required")
    expected, actual = (work / ".local_gpu_queue.lock").stat(), os.fstat(lock_fd)
    if (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino):
        raise ValueError("GPU lease belongs to another work directory")
    with os.fdopen(lock_fd, "a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        summaries = []
        for name in models:
            source = work / "runs" / f"{name}_fold0_local"
            arguments = [
                read_json(source / "config.json"),
                work,
                hours,
                epochs,
                lr,
                crop_size,
            ]
            config = (
                recipe(*arguments, full_frames=True)
                if full_frames
                else recipe(*arguments)
            )
            output = Path(config["output"])
            config_path = work / "configs" / f"{output.name}.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            if (
                config_path.is_file()
                and yaml.safe_load(config_path.read_text()) != config
            ):
                raise ValueError(
                    "Existing fine-tuning configuration differs; preserved"
                )
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            resume = output / "last.pt"
            print(
                f"FINE-TUNING {name}: source best.pt; all train clips; frame_step={config['data']['frame_step']}; crop_probability={config['data']['crop_probability']}; lr={lr}; budget={hours}h",
                flush=True,
            )
            with ValidationPipeline() as pipeline:
                train(
                    config,
                    resume=resume if resume.is_file() else None,
                    init_checkpoint=None if resume.is_file() else source / "best.pt",
                    validation_pipeline=pipeline,
                )
            release_gpu()
            initial = read_json(output / "initial_validation.json")
            best = read_json(output / "best_metrics.json")
            gain = (
                best["metrics"]["coco100"]["mAP"] - initial["metrics"]["coco100"]["mAP"]
            )
            entry = {
                "model": name,
                "validation_split": "official",
                "training_mode": "full frames" if full_frames else "native crops",
                "initial_validation": initial,
                "best_validation": best,
                "gain": gain,
                "run": str(output),
                "submission": None,
            }
            if gain + 1e-12 >= 0.01:
                sha = file_hash(output / "best.pt")
                destination = (
                    work
                    / "submissions"
                    / f"{name}_{'fullframe' if full_frames else 'crops'}_{sha[:12]}_{config['model']['image_size']}.csv"
                )
                if not destination.is_file():
                    create_submission(
                        output, destination, threads=2, device_name="cuda", vram_gib=6
                    )
                entry["submission"] = str(destination)
                print(
                    f"{'FULLFRAME' if full_frames else 'CROP'} CANDIDATE READY FOR KAGGLE: {destination}",
                    flush=True,
                )
            print(
                f"{name}: initial official mAP={initial['metrics']['coco100']['mAP']:.4f}; best={best['metrics']['coco100']['mAP']:.4f}; gain={gain:.4f}",
                flush=True,
            )
            write_json(output / "finetune_summary.json", entry)
            summaries.append(entry)
            release_gpu()
        mode = "fullframe" if full_frames else "crop"
        write_json(work / f"{mode}_finetune_summary.json", {"models": summaries})
        return summaries
