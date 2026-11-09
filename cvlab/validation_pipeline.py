"""Overlap one CPU validation task with GPU training, without sharing live weights."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from .common import digest, protocol_fingerprint, write_json
from .metrics import evaluate


def cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(item) for item in value)
    return value


def finish_validation(
    output, config, records, predictions, timing, payload, log, elapsed_before, started
):
    from .engine import atomic_checkpoint

    epoch = payload["epoch"]
    print(f"epoch={epoch} CPU metrics running; GPU may continue training", flush=True)
    scores = evaluate(records, predictions, config["evaluation"]["max_detections"])
    score = scores["coco100"]["mAP"]
    if score is None:
        raise ValueError("Validation has no evaluable target objects")
    improved = score > payload["best"]
    payload["best"] = max(payload["best"], score)
    payload["validation_pending"] = None
    payload["elapsed_seconds"] = elapsed_before + time.perf_counter() - started
    log.update(metrics=scores, elapsed_seconds=payload["elapsed_seconds"])
    output = Path(output)
    atomic_checkpoint(output / "last.pt", payload)
    if improved:
        atomic_checkpoint(output / "best.pt", payload)
        write_json(output / "best_metrics.json", log)
        write_json(
            output / "val_predictions.json",
            {
                "record_ids": [record["id"] for record in records],
                "predictions": predictions,
                "timing": timing,
                "metrics": scores,
                "model_config": config["model"],
                "split": config["data"]["split"],
                "protocol_fingerprint": protocol_fingerprint(config),
                "fold": config["data"].get("fold"),
                "data_fingerprint": digest(records),
            },
        )
    history = output / "history.jsonl"
    rows = []
    if history.is_file():
        rows = [
            json.loads(line)
            for line in history.read_text().splitlines()
            if line.strip()
        ]
    # Replace any uncommitted legacy row left by an earlier checkpoint failure.
    rows = [row for row in rows if row["epoch"] != epoch] + [log]
    temporary = history.with_suffix(".jsonl.tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    temporary.replace(history)
    print(
        f"{output.name}: epoch={epoch} loss={log['train_loss']:.4f} mAP={score:.4f} best={payload['best']:.4f}",
        flush=True,
    )
    return payload["best"]


class ValidationPipeline:
    def __init__(self, minimum_available_mib=4096):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="coco-cpu")
        self.jobs = []
        self.minimum_available_mib = minimum_available_mib

    def wait(self):
        for job in self.jobs:
            job.result()
        self.jobs.clear()

    def memory_gate(self):
        memory = Path("/proc/meminfo")
        if not self.jobs or not memory.is_file():
            return
        fields = dict(line.split(":", 1) for line in memory.read_text().splitlines())
        available = int(fields["MemAvailable"].split()[0]) / 1024
        if available < self.minimum_available_mib:
            print(
                f"Available RAM={available:.0f} MiB; finishing CPU validation before more GPU work",
                flush=True,
            )
            self.wait()

    def submit(self, *args):
        if any(not job.done() for job in self.jobs):
            raise RuntimeError(
                "Finish previous CPU validation before collecting another prediction set"
            )
        self.wait()
        job = self.executor.submit(finish_validation, *args)
        self.jobs.append(job)
        return job

    def __enter__(self):
        return self

    def __exit__(self, *error):
        try:
            self.wait()
        finally:
            self.executor.shutdown(wait=True)
