"""Queue full train selection, then an explicitly unvalidated train+val fit."""
import argparse
import fcntl
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--selection-epochs", type=int, default=5)
    parser.add_argument("--final-epochs", type=int, default=1)
    parser.add_argument("--plan-name", default="default")
    parser.add_argument("--batch-size", type=int, default=1, help="Measured physical batch; keeps effective batch and requires a new plan when changed")
    parser.add_argument("--weight-loss-vfl", type=float, help="RT-DETR classification/VFL loss coefficient; requires a new plan")
    parser.add_argument("--training-policy", choices=["continuous", "strict"],
                        help="New plans default continuous; explicitly adopt an existing plan at its saved boundary")
    args = parser.parse_args()
    work = args.work.resolve()
    with (work / ".full_data.launcher.lock").open("a") as own:
        try:
            fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Full-data plan already queued or running", flush=True)
            return
        print("Waiting for shared GPU lease; current training/validation preserved", flush=True)
        with (work / ".local_gpu_queue.lock").open("a") as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
            if int(memory["MemAvailable"].split()[0]) < 4 * 1024**2:
                raise RuntimeError("Less than 4 GiB RAM; defer full-data plan")
            os.environ.pop("PYTHONPATH", None)
            os.environ.pop("CVLAB_DATA_ROOT", None)
            from cvlab.full_data import run

            model_overrides = ({"weight_loss_vfl": args.weight_loss_vfl}
                               if args.weight_loss_vfl is not None else {})
            run(work, args.source, hours=args.hours,
                selection_epochs=args.selection_epochs, final_epochs=args.final_epochs,
                plan_name=args.plan_name, batch_size=args.batch_size,
                model_overrides=model_overrides, training_policy=args.training_policy)


if __name__ == "__main__":
    main()
