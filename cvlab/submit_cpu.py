"""Create a verified submission on CPU alongside the GPU training queue."""

import argparse
import copy
import csv
import fcntl
import hashlib
import shutil
import tempfile
import time
from pathlib import Path

from .common import digest, read_json, write_json


def create_submission(run, output, threads=4, chunk_size=16):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(output.suffix + ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                f"Submission generation already running: {output}"
            ) from None
        return _create_submission(run, output, threads, chunk_size)


def _create_submission(run, output, threads, chunk_size):
    if threads < 1 or chunk_size < 1:
        raise ValueError("threads and chunk_size must be positive")
    run, output = Path(run), Path(output)
    if output.exists():
        raise FileExistsError(f"Submission already exists; preserved: {output}")
    from .engine import load_trained, predict, setup
    from .prepare import select_records
    from .submission import write_submission
    from scripts.check_lab import validate_submission

    config = copy.deepcopy(read_json(run / "config.json"))
    config["runtime"].update(device="cpu", precision="fp32", cpu_threads=threads)
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
    print(
        f"CPU submission: {len(records)} frames; threads={threads}; chunk={chunk_size}",
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
                "cpu_inference_seconds": time.perf_counter() - started,
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
