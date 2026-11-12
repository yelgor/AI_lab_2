"""Queue a sampling ablation and larger bounded continuation behind the GPU lease."""
import argparse
import fcntl
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=72000)
    parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--reserve-hours", type=float, default=2)
    args = parser.parse_args()
    work = args.work.resolve()
    with (work / ".sampling_experiment.lock").open("a") as own:
        try:
            fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Sampling experiment already queued", flush=True)
            return
        print("Waiting for shared GPU lease; current experiment preserved", flush=True)
        with (work / ".local_gpu_queue.lock").open("a") as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
            if int(mem["MemAvailable"].split()[0]) < 4 * 1024**2:
                raise RuntimeError("Less than 4 GiB RAM available; defer experiment")
            os.environ.pop("PYTHONPATH", None)
            os.environ.pop("CVLAB_DATA_ROOT", None)
            from cvlab.generalization_training import run

            run(work, args.frames, args.hours, args.reserve_hours)


if __name__ == "__main__":
    main()
