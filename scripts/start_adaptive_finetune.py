"""Queue short champion/challenger attempts without replacing a champion on rejection."""

import argparse
import fcntl
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--batches", type=int, default=6000)
    parser.add_argument("--export-tiles", action="store_true")
    parser.add_argument("--reset-to-original", action="store_true")
    args = parser.parse_args()
    if args.attempts < 1 or args.batches < 1:
        raise ValueError("Positive attempts and batches required")
    work = args.work.resolve()
    if not (work / "runs/rtdetr_fold0_local/best.pt").is_file():
        raise FileNotFoundError("Original public-best checkpoint is missing")
    with (work / ".adaptive_finetune.launcher.lock").open("a") as startup:
        try:
            fcntl.flock(startup, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Adaptive training already queued or running", flush=True)
            return
        with (work / ".local_gpu_queue.lock").open("a") as lease:
            print(
                "Waiting for GPU lease; champion/challenger mode starts next",
                flush=True,
            )
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
                    "train-adaptive",
                    "--work",
                    str(work),
                    "--lock-fd",
                    str(lease.fileno()),
                    "--attempts",
                    str(args.attempts),
                    "--batches",
                    str(args.batches),
                    *(["--export-tiles"] if args.export_tiles else []),
                    *(["--reset-to-original"] if args.reset_to_original else []),
                ],
            )


if __name__ == "__main__":
    main()
