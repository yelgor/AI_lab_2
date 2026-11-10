"""Queue saved-weight training rounds with periodic validation and submissions."""

import argparse
import fcntl
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batches", type=int, default=6000)
    parser.add_argument("--export-tiles", action="store_true")
    parser.add_argument("--replace-bn", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1 or args.batches < 1:
        raise ValueError("Positive rounds and batches required")
    work = args.work.resolve()
    if not (work / "runs").is_dir():
        raise FileNotFoundError(work / "runs")
    with (work / ".iterative_finetune.launcher.lock").open("a") as startup:
        try:
            fcntl.flock(startup, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Iterative fine-tuning already queued or running", flush=True)
            return
        if args.replace_bn:
            from start_fullframe_finetune import stop_bn

            stop_bn(work)
        with (work / ".local_gpu_queue.lock").open("a") as lease:
            print(
                "Waiting for current checkpoint/metrics to finish; then continuing saved best weights",
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
                    "train-iterate",
                    "--work",
                    str(work),
                    "--lock-fd",
                    str(lease.fileno()),
                    "--rounds",
                    str(args.rounds),
                    "--batches",
                    str(args.batches),
                    *(["--export-tiles"] if args.export_tiles else []),
                ],
            )


if __name__ == "__main__":
    main()
