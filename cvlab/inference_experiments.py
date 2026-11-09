"""Compare inference resolutions on a fixed held-out subset without retraining."""

import copy
import hashlib
import os
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path


def balanced_subset(records, limit):
    if limit is None or limit >= len(records):
        return records
    if limit < 1:
        raise ValueError("max_images must be positive")
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[record["sequence_id"]].append(index)
    groups = [groups[key] for key in sorted(groups)]
    counts = [0] * len(groups)
    for index in range(limit):
        available = [i for i, group in enumerate(groups) if counts[i] < len(group)]
        selected = min(available, key=lambda i: counts[i])
        counts[selected] += 1
    indices = []
    for group, count in zip(groups, counts):
        if count == 1:
            indices.append(group[(len(group) - 1) // 2])
        elif count > 1:
            indices.extend(
                group[round(i * (len(group) - 1) / (count - 1))] for i in range(count)
            )
    return [records[index] for index in sorted(indices)]


def wait_for_idle(poll_seconds=5):
    announced = False
    while True:
        result = subprocess.run(
            ["pgrep", "-f", r"[p]ython.*-m cvlab (train|evaluate|predict|benchmark)"],
            capture_output=True,
            text=True,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr)
        others = [int(pid) for pid in result.stdout.split() if int(pid) != os.getpid()]
        if not others:
            return
        if not announced:
            print(
                "Waiting for current CVLab jobs; the model queue will yield to this experiment.",
                flush=True,
            )
            announced = True
        time.sleep(poll_seconds)


def compare_resolutions(
    config, checkpoint, output, sizes=(960, 1280), max_images=192, wait_gpu=False
):
    # Delay Torch imports and model construction until existing GPU work ends.
    import fcntl

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(tempfile.gettempdir()) / f"cvlab-resolution-{os.getuid()}.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "A resolution experiment is already queued for this user"
            ) from None
        if wait_gpu:
            wait_for_idle()
        return _compare(config, checkpoint, output, sizes, max_images)


def _compare(config, checkpoint, output, sizes, max_images):
    import gc
    import torch
    from .common import digest, write_json
    from .engine import load_trained, predict, setup
    from .metrics import evaluate
    from .prepare import select_records

    if config["model"]["name"] not in {"retinanet", "fcos", "fasterrcnn"}:
        raise ValueError(
            "Inference resolution override is supported for TorchVision CNN detectors"
        )
    if not sizes or any(size < 64 or size % 32 for size in sizes):
        raise ValueError("All resolutions must be multiples of 32, >=64")
    records = select_records(config, "val")
    full_count = len(records)
    records = balanced_subset(records, max_images)
    checkpoint_hash = hashlib.sha256()
    with Path(checkpoint).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checkpoint_hash.update(chunk)
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash.hexdigest(),
        "record_ids": [record["id"] for record in records],
        "data_fingerprint": digest(records),
        "full_validation_images": full_count,
        "subset": len(records) < full_count,
        "config": config,
        "experiments": [],
    }
    for size in sizes:
        inference_config = copy.deepcopy(config)
        inference_config["model"]["image_size"] = size
        inference_config["loader"]["batch_size"] = 1
        device = setup(inference_config)
        print(
            f"Resolution={size}; validation images={len(records)}; fixed checkpoint",
            flush=True,
        )
        model = load_trained(checkpoint, inference_config, device)
        predictions, timing = predict(model, inference_config, records, device)
        scores = evaluate(records, predictions, config["evaluation"]["max_detections"])
        report["experiments"].append(
            {"image_size": size, "metrics": scores, "timing": timing}
        )
        write_json(output, report)
        print(
            f"Resolution={size}: mAP={scores['coco100']['mAP']}; AP_small={scores['coco100']['AP_small']}",
            flush=True,
        )
        del predictions, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return report
