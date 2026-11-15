"""Cache detectors under the shared GPU lease, then compare/export the fixed route on CPU."""

import argparse
import copy
import fcntl
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def require_ram():
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    if int(memory["MemAvailable"].split()[0]) < 4 * 1024**2:
        raise RuntimeError("Need at least 4 GiB available RAM")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    cache = sub.add_parser("cache")
    cache.add_argument("--backend", choices=["c1", "yolo"], required=True)
    cache.add_argument("--checkpoint", required=True, type=Path)
    cache.add_argument("--output", required=True, type=Path)
    cache.add_argument("--split", choices=["val", "test"], default="val")
    cache.add_argument(
        "--tile-side", type=int, help="Default: c1=960; YOLO=global only"
    )
    cache.add_argument("--device", choices=["cuda:0", "cpu"], default="cuda:0")
    for command in ("compare", "export"):
        item = sub.add_parser(command)
        item.add_argument("--c1", required=True, type=Path)
        item.add_argument("--yolo", required=True, type=Path)
        item.add_argument("--output", required=True, type=Path)
        if command == "export":
            item.add_argument("--report", required=True, type=Path)
            item.add_argument("--manual-review", action="store_true",
                              help="Permit explicit family uncertainty after pooled/class/size guards pass against both singles; never submits")
    args = parser.parse_args()
    os.environ.pop("CVLAB_DATA_ROOT", None)
    if args.command != "cache":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from cvlab.common import load_config
    from cvlab.class_ensemble import cache_predictions, compare, export, snapshot

    work = args.work.resolve()
    if not work.is_dir():
        raise FileNotFoundError("Work volume is not mounted")
    config = copy.deepcopy(load_config(args.config))
    config["data"].update(split="official", frame_step=1)
    if args.command == "compare":
        import torch

        require_ram()
        torch.set_num_threads(1)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.with_suffix(args.output.suffix+".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            compare(args.c1, args.yolo, config, args.output)
        return
    if args.command == "export":
        import torch

        torch.set_num_threads(1)
        export(args.c1, args.yolo, config, args.report, args.output, manual_review=args.manual_review)
        return
    tile = (
        args.tile_side
        if args.tile_side is not None
        else (960 if args.backend == "c1" else None)
    )
    if tile == 0:
        tile = None
    config["loader"].update(
        batch_size=1,
        workers=2,
        prefetch_factor=1,
        persistent_workers=False,
        pin_memory=args.device != "cpu",
    )
    config["runtime"].update(
        device=args.device,
        precision="fp32" if args.device == "cpu" else "fp16",
        cpu_threads=2,
    )
    config["model"].update(score_threshold=0.001, max_detections=500)
    if (
        cache_predictions(
            args.checkpoint,
            config,
            args.output,
            args.backend,
            args.split,
            tile,
            existing_only=True,
        )
        is not None
    ):
        return
    if shutil.disk_usage(work).free < 4 * 2**30:
        raise RuntimeError("Need 4 GiB disk headroom for snapshots and prediction caches")
    # Snapshot best, never live last.pt (Ultralytics overwrites last in place).
    checkpoint = snapshot(args.checkpoint, work / "ensemble/snapshots")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "cache.lock").open("a") as own:
        try:
            fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("This cache is already queued/running", flush=True)
            return
        print("Waiting for shared GPU lease; current training is preserved", flush=True)
        with (work / ".local_gpu_queue.lock").open("a") as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            require_ram()
            if args.device != "cpu":
                from cvlab.training_watchdog import gpu_info

                gpu = gpu_info(0)
                if not gpu["ready"] or gpu["compute_pids"]:
                    raise RuntimeError(
                        "GPU unavailable or owned outside the shared lease"
                    )
            if args.backend == "yolo":
                from cvlab.yolo_training import configure_optional_backend

                configure_optional_backend(work)
            cache_predictions(
                checkpoint, config, args.output, args.backend, args.split, tile
            )


if __name__ == "__main__":
    main()
