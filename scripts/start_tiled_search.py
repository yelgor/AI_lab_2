"""Queue tiled detection after the current GPU search, preserving its work."""

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
    parser.add_argument("--tile-sides", nargs="+", type=int, default=[768, 960])
    parser.add_argument("--ious", nargs="+", type=float, default=[0.5, 0.6])
    parser.add_argument("--max-images", type=int, default=192)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--vram-gib", type=float, default=6)
    parser.add_argument("--min-gain", type=float, default=0.01)
    args = parser.parse_args()
    work = args.work.resolve()
    if not (work / "runs").is_dir():
        raise FileNotFoundError(work / "runs")
    with (work / ".tiled_search.launcher.lock").open("a") as startup:
        try:
            fcntl.flock(startup, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Tiling search already queued or running", flush=True)
            return
        with (work / ".local_gpu_queue.lock").open("a") as lease:
            print("Waiting for current GPU work; tiling runs next", flush=True)
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
                    "evaluate-tiles",
                    "--work",
                    str(work),
                    "--lock-fd",
                    str(lease.fileno()),
                    "--models",
                    *args.models,
                    "--tile-sides",
                    *map(str, args.tile_sides),
                    "--ious",
                    *map(str, args.ious),
                    "--max-images",
                    str(args.max_images),
                    "--workers",
                    str(args.workers),
                    "--vram-gib",
                    str(args.vram_gib),
                    "--min-gain",
                    str(args.min_gain),
                ],
            )


if __name__ == "__main__":
    main()
