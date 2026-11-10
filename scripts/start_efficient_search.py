"""Queue a bounded warm-start search after existing GPU work has finished."""

import argparse
import fcntl
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--hours", type=float, default=4)
    parser.add_argument("--reserve-hours", type=float, default=1)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--budgets", type=int, nargs=3, default=[6000, 12000, 24000])
    parser.add_argument("--probe-images", type=int, default=512)
    parser.add_argument("--export-tiles", action="store_true")
    parser.add_argument("--reset-to-original", action="store_true")
    args = parser.parse_args()
    work = args.work.resolve()
    if not (work / "runs/rtdetr_fold0_local/best.pt").is_file():
        raise FileNotFoundError("Original public-best checkpoint is missing")
    with (work / ".efficient_search.launcher.lock").open("a") as startup:
        try:
            fcntl.flock(startup, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Efficient search already queued or running", flush=True)
            return
        with (work / ".local_gpu_queue.lock").open("a") as lease:
            print("Waiting for GPU lease; existing training is preserved", flush=True)
            fcntl.flock(lease, fcntl.LOCK_EX)
            for stream in (startup, lease):
                os.set_inheritable(stream.fileno(), True)
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
                    "train-efficient",
                    "--work",
                    str(work),
                    "--lock-fd",
                    str(lease.fileno()),
                    "--hours",
                    str(args.hours),
                    "--reserve-hours",
                    str(args.reserve_hours),
                    "--cycles",
                    str(args.cycles),
                    "--probe-images",
                    str(args.probe_images),
                    "--budgets",
                    *map(str, args.budgets),
                    *(["--export-tiles"] if args.export_tiles else []),
                    *(["--reset-to-original"] if args.reset_to_original else []),
                ],
            )


if __name__ == "__main__":
    main()
