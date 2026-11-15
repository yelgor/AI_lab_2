"""Prepare or queue one YOLO11s course using the shared local GPU lease."""

import argparse
import fcntl
import math
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Existing config supplies actual root/prepared paths only",
    )
    parser.add_argument("--name", default="yolo11s_1280")
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument(
        "--hours",
        default=10,
        type=float,
        help="Soft session budget; finish and validate the current epoch",
    )
    parser.add_argument("--batch", default=2, type=int)
    parser.add_argument("--imgsz", default=1280, type=int)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--device", default="0", choices=["0", "cpu"])
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Use full precision while resuming a numerically unstable AMP course",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export after training, for review; never submits",
    )
    parser.add_argument(
        "--export-every",
        default=0,
        type=int,
        help="Also export the validated training best every N full epochs, for review",
    )
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    if (
        args.epochs < 1
        or args.batch < 1
        or args.imgsz < 32
        or args.imgsz % 32
        or not math.isfinite(args.hours)
        or args.hours <= 0
        or not 0 <= args.workers <= 4
        or args.export_every < 0
        or not args.name
        or Path(args.name).name != args.name
        or args.name in (".", "..")
    ):
        parser.error("Invalid recipe or run name")
    if args.prepare_only and (
        args.resume or args.export or args.export_only or args.export_every
    ):
        parser.error("--prepare-only cannot train/resume/export")
    from cvlab.common import load_config, read_json, write_json
    from cvlab.yolo_training import (
        prepare_dataset,
        configure_optional_backend,
        train_challenger,
        export_checkpoint,
        file_sha,
    )

    work = args.work.resolve()
    if not work.is_dir():
        raise FileNotFoundError("Work volume must already be mounted and writable")
    os.environ.pop("CVLAB_DATA_ROOT", None)
    source = load_config(args.config)
    root, prepared = (
        Path(source["data"][key]).resolve() for key in ("root", "prepared")
    )
    if args.prepare_only:
        print(
            "Prepared CPU-only:",
            prepare_dataset(root, prepared, work / "yolo_data"),
            flush=True,
        )
        return
    with (work / ".yolo.launcher.lock").open("a") as own:
        try:
            fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("YOLO already queued/running; duplicate not started", flush=True)
            return
        print("Waiting for shared GPU lease; active training is preserved", flush=True)
        with (work / ".local_gpu_queue.lock").open("a") as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            memory = dict(
                line.split(":", 1)
                for line in Path("/proc/meminfo").read_text().splitlines()
            )
            if int(memory["MemAvailable"].split()[0]) < 4 * 1024**2:
                raise RuntimeError(
                    "Less than 4 GiB available RAM; release other applications first"
                )
            if shutil.disk_usage(work).free < 4 * 2**30:
                raise RuntimeError(
                    "Less than 4 GiB disk headroom; preserve artifacts and free space"
                )
            if args.device != "cpu":
                from cvlab.training_watchdog import gpu_info

                gpu = gpu_info(0)
                if gpu.get("compute_pids"):
                    raise RuntimeError(
                        "GPU compute process outside the shared lease; inspect it before launch"
                    )
            run = work / "runs" / args.name
            if not args.export_only:
                run = train_challenger(
                    work,
                    root,
                    prepared,
                    name=args.name,
                    epochs=args.epochs,
                    hours=args.hours,
                    batch=args.batch,
                    image_size=args.imgsz,
                    workers=args.workers,
                    device=args.device,
                    resume=args.resume,
                    amp=False if args.no_amp else None,
                    export_every=args.export_every,
                )
            if args.export or args.export_only:
                configure_optional_backend(work)
                config = read_json(run / "config.json")
                best = run / "weights" / "common_best.pt"
                report = read_json(run / "common_best_metrics.json")
                if report["checkpoint_sha256"] != file_sha(best):
                    raise ValueError(
                        "Common validation receipt does not match exported weights"
                    )
                destination = (
                    work / "submissions" / f"{args.name}_{file_sha(best)[:12]}.csv"
                )
                result = export_checkpoint(best, config, destination)
                write_json(
                    run / "export_review.json",
                    {
                        "status": "QUALITY_REVIEW_REQUIRED",
                        "submission": str(destination),
                        "validation_report": report,
                        "csv_validation": result,
                        "reference_comparison": None,
                        "public_score": None,
                        "note": "Compare the full held-out pipeline with c1 before promotion; no auto-submit",
                    },
                )
                print("Export review:", run / "export_review.json", flush=True)


if __name__ == "__main__":
    main()
