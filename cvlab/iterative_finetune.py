"""Continue saved weights in short rounds; publish validated candidates between rounds."""

import fcntl
import json
import os
from pathlib import Path

import yaml

from .common import read_json, write_json


def recipe(source, work):
    from .crop_finetune import recipe as warm_recipe

    config = warm_recipe(
        source, work, hours=12, epochs=10000, lr=1e-5, full_frames=True
    )
    config["output"] = str(Path(work) / "runs/rtdetr_iterative_finetune")
    config["train"].update(
        freeze_batch_norm=True,
        early_stopping_patience=5,
        early_stopping_min_delta=0.0005,
    )
    return config


def history(output):
    path = output / "history.jsonl"
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.is_file()
        else []
    )


def candidate_with_restored_budget(action):
    import torch
    from .score_improvement import release_gpu

    device = torch.cuda.current_device() if torch.cuda.is_available() else None
    fraction = (
        torch.cuda.get_per_process_memory_fraction(device)
        if device is not None
        else None
    )
    try:
        return action()
    finally:
        release_gpu()
        if fraction is not None:
            torch.cuda.set_per_process_memory_fraction(fraction, device)


def run(work, rounds, batches, lock_fd, export_tiles=False):
    from .engine import train
    from .score_improvement import file_hash, release_gpu
    from .validation_pipeline import ValidationPipeline

    work = Path(work).resolve()
    if rounds < 1 or batches < 1:
        raise ValueError("Positive rounds and batches required")
    expected, actual = (work / ".local_gpu_queue.lock").stat(), os.fstat(lock_fd)
    if (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino):
        raise ValueError("GPU lease belongs to another work directory")
    with os.fdopen(lock_fd, "a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = work / "runs/rtdetr_fullframe_finetune"
        config_path = work / "configs/rtdetr_iterative_finetune.yaml"
        if config_path.is_file():
            config = yaml.safe_load(config_path.read_text())
        else:
            config = recipe(read_json(source / "config.json"), work)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        output = Path(config["output"])
        report_path = work / "iterative_finetune_summary.json"
        report = (
            read_json(report_path)
            if report_path.is_file()
            else {"rounds": [], "tile_attempts": []}
        )
        for index in range(rounds):
            if (output / "STOP").exists():
                print("STOP requested; latest checkpoints preserved", flush=True)
                break
            previous = history(output)
            resume = output / "last.pt"
            print(
                f"ROUND {index + 1}/{rounds}: at most {batches} full training frames; learned weights preserved; optimizer restored on resume",
                flush=True,
            )
            with ValidationPipeline() as pipeline:
                train(
                    config,
                    resume=resume if resume.is_file() else None,
                    init_checkpoint=None if resume.is_file() else source / "best.pt",
                    validation_pipeline=pipeline,
                    yield_after_epoch=True,
                    round_batches=batches,
                )
            release_gpu()
            current = history(output)
            if len(current) <= len(previous):
                print(
                    "No new round: early stopping or cumulative training budget reached",
                    flush=True,
                )
                break
            initial = read_json(output / "initial_validation.json")["metrics"][
                "coco100"
            ]["mAP"]
            best = read_json(output / "best_metrics.json")["metrics"]["coco100"]["mAP"]
            entry = {
                "epoch": current[-1]["epoch"],
                "batches": current[-1]["train_batches"],
                "last_mAP": current[-1]["metrics"]["coco100"]["mAP"],
                "best_mAP": best,
                "initial_mAP": initial,
            }
            report["rounds"].append(entry)
            write_json(report_path, report)
            print(
                f"ROUND COMPLETE: mAP={entry['last_mAP']:.6f}; best={best:.6f}; initial={initial:.6f}; checkpoint={output / 'best.pt'}",
                flush=True,
            )
            if (output / "STOP").exists():
                print("STOP requested; skipping candidate export", flush=True)
                break
            if export_tiles and best - initial >= 0.002:
                sha = file_hash(output / "best.pt")
                attempted = {
                    row["checkpoint_sha256"] for row in report["tile_attempts"]
                }
                if sha not in attempted:
                    from .tiled_search import run as tiled_run

                    print(
                        "Checking fixed tiles=960, NMS=0.6 for the new best checkpoint",
                        flush=True,
                    )
                    result = candidate_with_restored_budget(
                        lambda: tiled_run(
                            work,
                            ["rtdetr"],
                            [960],
                            [0.6],
                            192,
                            6,
                            2,
                            os.dup(lease.fileno()),
                            min_gain=0.001,
                            runs=[output],
                        )
                    )
                    report["tile_attempts"].append(
                        {"checkpoint_sha256": sha, "result": result}
                    )
                    write_json(report_path, report)
                    print(
                        "Candidate check complete; next training round can continue while a ready CSV is uploaded",
                        flush=True,
                    )
        return report
