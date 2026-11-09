"""Queue crop-aware fine-tuning after the already queued tiling experiment."""

import argparse
import fcntl
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["rtdetr"],
        choices=["rtdetr", "fasterrcnn", "fcos", "retinanet"],
    )
    parser.add_argument("--hours", type=float, default=2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--crop-size", type=int, default=768)
    args = parser.parse_args()
    work = args.work.resolve()
    if not (work / "runs").is_dir():
        raise FileNotFoundError(work / "runs")
    with (work / ".crop_finetune.launcher.lock").open("a") as startup:
        try:
            fcntl.flock(startup, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Crop fine-tuning already queued or running", flush=True)
            return
        print("Waiting for the queued tiling experiment before fine-tuning", flush=True)
        with (work / ".tiled_search.launcher.lock").open("a") as dependency:
            fcntl.flock(dependency, fcntl.LOCK_EX)
        with (work / ".local_gpu_queue.lock").open("a") as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            for file in (startup, lease):
                os.set_inheritable(file.fileno(), True)
            os.chdir(Path(__file__).resolve().parents[1])
            os.environ.pop("PYTHONPATH", None)
            os.environ.pop("CVLAB_DATA_ROOT", None)
            os.execv(
                sys.executable,
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "cvlab",
                    "train-crops",
                    "--work",
                    str(work),
                    "--lock-fd",
                    str(lease.fileno()),
                    "--models",
                    *args.models,
                    "--hours",
                    str(args.hours),
                    "--epochs",
                    str(args.epochs),
                    "--lr",
                    str(args.lr),
                    "--crop-size",
                    str(args.crop_size),
                ],
            )


if __name__ == "__main__":
    main()
