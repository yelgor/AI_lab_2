"""Verified chunked submissions; the legacy module entrypoint uses CPU."""

import argparse
import copy
import csv
import fcntl
import hashlib
import math
import shutil
import tempfile
import time
from pathlib import Path

from .common import digest, read_json, write_json


def create_submission(
    run,
    output,
    threads=4,
    chunk_size=16,
    device_name="cpu",
    vram_gib=None,
    image_size=None,
):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(output.suffix + ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                f"Submission generation already running: {output}"
            ) from None
        return _create_submission(
            run, output, threads, chunk_size, device_name, vram_gib, image_size
        )


def _available_ram_mib():
    memory = Path("/proc/meminfo")
    if not memory.is_file():
        return None
    fields = dict(line.split(":", 1) for line in memory.read_text().splitlines())
    return int(fields["MemAvailable"].split()[0]) / 1024


def _cuda_budget(device, vram_gib):
    import torch

    if vram_gib is None or not math.isfinite(vram_gib) or vram_gib <= 0:
        raise ValueError("CUDA export requires a finite positive VRAM budget")
    available = _available_ram_mib()
    if available is not None and available < 2048:
        raise RuntimeError("Less than 2 GiB RAM available; export not started")
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError("CUDA budget requires a CUDA device")
    # torch 2.6 set_per_process_memory_fraction rejects torch.device('cuda')
    # unless an explicit device index is supplied.
    index = (
        resolved.index if resolved.index is not None else torch.cuda.current_device()
    )
    free, total = torch.cuda.mem_get_info(index)
    budget = int(vram_gib * 1024**3)
    if budget >= total or free < budget + 512 * 1024**2:
        raise RuntimeError(
            "Insufficient free GPU memory for export budget plus 512 MiB margin"
        )
    torch.cuda.set_per_process_memory_fraction(budget / total, index)
    print(
        f"PyTorch GPU budget={budget / 1024**2:.0f} MiB; free={free / 1024**2:.0f} MiB",
        flush=True,
    )


def _create_submission(
    run, output, threads, chunk_size, device_name, vram_gib, image_size=None
):
    if threads < 1 or chunk_size < 1:
        raise ValueError("threads and chunk_size must be positive")
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    run, output = Path(run), Path(output)
    if output.exists():
        raise FileExistsError(f"Submission already exists; preserved: {output}")
    from .engine import load_trained, predict, setup
    from .prepare import select_records
    from .submission import write_submission
    from scripts.check_lab import validate_submission

    config = copy.deepcopy(read_json(run / "config.json"))
    if image_size is not None:
        if (
            not isinstance(image_size, int)
            or isinstance(image_size, bool)
            or image_size < 64
            or image_size % 32
        ):
            raise ValueError("image_size must be an integer multiple of 32, >=64")
        config["model"]["image_size"] = image_size
    config["runtime"].update(
        device=device_name,
        precision="fp16" if device_name == "cuda" else "fp32",
        cpu_threads=threads,
    )
    config["loader"].update(
        batch_size=1, workers=0, pin_memory=False, persistent_workers=False
    )
    config["evaluation"]["warmup_batches"] = 0
    records = select_records(config, "test")
    root = Path(config["data"]["root"])
    sample = root / "sample_submission.csv"
    if sample.is_file():
        with sample.open(newline="", encoding="utf-8-sig") as stream:
            ids = [row["image_id"] for row in csv.DictReader(stream)]
        index = {r["submission_id"]: r for r in records}
        if len(ids) != len(set(ids)) or set(ids) != set(index):
            raise ValueError("Sample submission and test manifest IDs differ")
        records = [index[key] for key in ids]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    device = setup(config)
    if device_name == "cuda":
        _cuda_budget(device, vram_gib)
    print(
        f"{device_name.upper()} submission: {len(records)} frames; threads={threads}; chunk={chunk_size}",
        flush=True,
    )
    # Freeze the checkpoint before the trainer can atomically replace best.pt.
    # Keeping this copy on the work volume avoids consuming root filesystem space.
    with tempfile.TemporaryDirectory(
        prefix="submission-", dir=output.parent
    ) as scratch:
        checkpoint = Path(scratch) / "best.pt"
        shutil.copyfile(run / "best.pt", checkpoint)
        hasher = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        model = load_trained(checkpoint, config, device)
        print(f"Checkpoint frozen: {hasher.hexdigest()}", flush=True)
        started = time.perf_counter()
        with temporary.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.writer(destination)
            writer.writerow(["image_id", "PredictionString"])
            for start in range(0, len(records), chunk_size):
                batch = records[start : start + chunk_size]
                predictions, _ = predict(model, config, batch, device)
                part = Path(scratch) / "chunk.csv"
                write_submission(
                    batch,
                    predictions,
                    None,
                    part,
                    config["model"]["score_threshold"],
                    config["model"]["max_detections"],
                )
                with part.open(newline="", encoding="utf-8") as stream:
                    reader = csv.reader(stream)
                    next(reader)
                    writer.writerows(reader)
                del predictions
                count = start + len(batch)
                rate = count / max(time.perf_counter() - started, 1e-9)
                eta_minutes = (len(records) - count) / rate / 60
                print(
                    f"test={count}/{len(records)}; {rate:.2f} images/s; ETA={eta_minutes:.1f} min",
                    flush=True,
                )
        checked = validate_submission(temporary, root / "test_images.csv", sample)
        if checked["all_predictions_empty"]:
            raise ValueError(
                "All predictions are empty; partial file retained for inspection"
            )
        write_json(
            output.with_suffix(output.suffix + ".metadata.json"),
            {
                "run": str(run.resolve()),
                "checkpoint_sha256": hasher.hexdigest(),
                "test_fingerprint": digest(records),
                "inference_config": config,
                "inference_seconds": time.perf_counter() - started,
                "device": device_name,
                "pytorch_vram_budget_gib": vram_gib if device_name == "cuda" else None,
                "validation": checked,
            },
        )
    temporary.replace(output)
    print(f"READY FOR KAGGLE: {output.resolve()}", flush=True)
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=16)
    args = parser.parse_args()
    create_submission(args.run, args.out, args.threads, args.chunk_size)


if __name__ == "__main__":
    main()
