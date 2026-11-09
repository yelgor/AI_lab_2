"""Queue score experiments after active GPU work; acquire the lease before advertising a job."""

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
        default=["rtdetr", "fasterrcnn"],
        choices=["rtdetr", "fasterrcnn", "fcos", "retinanet"],
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[960, 1280, 1536])
    parser.add_argument("--max-images", type=int, default=192)
    parser.add_argument("--vram-gib", type=float, default=6)
    args = parser.parse_args()
    work = args.work.resolve()
    if not (work / "runs").is_dir():
        raise FileNotFoundError(work / "runs")
    with (work / ".score_improvement.launcher.lock").open("a") as startup:
        try:
            fcntl.flock(startup, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Score improvement is already queued or running", flush=True)
            return
        with (work / ".local_gpu_queue.lock").open("a") as lease:
            print(
                "Waiting for active GPU queue to finish; current training continues",
                flush=True,
            )
            fcntl.flock(lease, fcntl.LOCK_EX)
            os.set_inheritable(startup.fileno(), True)
            os.set_inheritable(lease.fileno(), True)
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
                    "evaluate-score",
                    "--work",
                    str(work),
                    "--lock-fd",
                    str(lease.fileno()),
                    "--models",
                    *args.models,
                    "--sizes",
                    *map(str, args.sizes),
                    "--max-images",
                    str(args.max_images),
                    "--vram-gib",
                    str(args.vram_gib),
                ],
            )


if __name__ == "__main__":
    main()
