"""Reusable evaluation receipts bound to exact weights, labeled records and inference."""

import hashlib
import importlib.metadata
from pathlib import Path
import time

from .common import digest, read_json, write_json


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(checkpoint, config, records, *, options=None):
    runtime = dict(config["runtime"])
    runtime.setdefault("tf32", False)
    runtime.setdefault("deterministic", False)
    # Workers/pinning only schedule the same deterministic validation transforms.
    # Semantic data changes remain covered by the complete labeled-record digest.
    return {
        "version": 1,
        "checkpoint_sha256": file_sha(checkpoint),
        "data_fingerprint": digest(records),
        "model": config["model"],
        "runtime": runtime,
        "seed": config["seed"],
        "ignore_policy": config["data"].get("ignore_policy"),
        "max_detections": config["evaluation"]["max_detections"],
        "batch_size": 1 if options is not None else config["evaluation"].get("validation_batch_size", config["loader"]["batch_size"]),
        "options": options,
        "implementation": digest(
            [
                (name, (Path(__file__).parent / name).read_text())
                for name in (
                    "engine.py",
                    "common.py",
                    "prepare.py",
                    "data.py",
                    "metrics.py",
                    "models.py",
                    "tiled_inference.py",
                    "sequential_policy.py",
                )
            ]
        ),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "transformers", "pycocotools", "numpy", "Pillow")
        },
    }


def load_receipt(directory, expected):
    path = Path(directory) / f"{digest(expected)}.json"
    if not path.is_file():
        return None
    value = read_json(path)
    if value.get("identity") != expected:
        raise ValueError("Evaluation receipt identity mismatch")
    # Older pooled-only scores cannot satisfy the new selection protocol.
    if not value.get("metrics", {}).get("per_video_mAP"):
        return None
    return value


def save_receipt(directory, expected, metrics, timing, *, origin):
    value = {
        "identity": expected,
        "metrics": metrics,
        "timing": timing,
        "origin": origin,
        "created": time.time(),
    }
    write_json(Path(directory) / f"{digest(expected)}.json", value)
    return value


def save_training_evaluation(config, records, metrics, timing, checkpoints):
    directory = config["evaluation"].get("cache_directory")
    if directory and metrics.get("per_video_mAP"):
        for checkpoint in checkpoints:
            save_receipt(
                directory,
                identity(checkpoint, config, records),
                metrics,
                timing,
                origin="committed_training_validation",
            )
