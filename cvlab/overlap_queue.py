"""One GPU trainer and one CPU validation worker shared across model runs."""

import fcntl
import gc
from pathlib import Path

import yaml


def run(work, models):
    import torch
    from .engine import train
    from .validation_pipeline import ValidationPipeline
    from scripts.local_gpu_queue import model_config

    work = Path(work).resolve()
    project = Path(__file__).resolve().parents[1]
    reference = yaml.safe_load(
        (work / "configs/retinanet_fold0_local.yaml").read_text()
    )
    with (work / ".local_gpu_queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "Another GPU queue holds the work directory lock"
            ) from None
        with ValidationPipeline() as pipeline:
            for name in models:
                pipeline.memory_gate()
                config_path, output = model_config(project, work, reference, name)
                config = yaml.safe_load(config_path.read_text())
                resume = output / "last.pt"
                print(
                    f"GPU starting {name}; previous CPU metrics may run concurrently",
                    flush=True,
                )
                train(
                    config,
                    resume=resume if resume.is_file() else None,
                    validation_pipeline=pipeline,
                )
                # Validation jobs retain CPU snapshots, never the live model.
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        print("All GPU runs and CPU metrics complete", flush=True)
