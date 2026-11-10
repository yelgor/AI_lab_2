"""Queue full-frame RT-DETR fine-tuning, optionally replacing BN diagnostics."""

import argparse
import fcntl
import math
import os
import signal
import sys
from pathlib import Path


def is_bn_job(argv, work):
    if not argv or not Path(argv[0]).name.startswith("python") or "-c" in argv:
        return False
    if "--work" not in argv:
        return False
    position = argv.index("--work")
    if position + 1 >= len(argv) or Path(argv[position + 1]).resolve() != work:
        return False
    if "-m" in argv:
        position = argv.index("-m")
        return argv[position + 1 : position + 3] == ["cvlab", "evaluate-bn"]
    # Only the actual script argument, not a shell/inhibitor command string.
    positional = [value for value in argv[1:] if not value.startswith("-")]
    return bool(positional) and Path(positional[0]).name == "start_bn_search.py"


def stop_bn(work, proc=Path("/proc")):
    stopped = []
    for process in proc.iterdir():
        if not process.name.isdigit():
            continue
        try:
            command = (process / "cmdline").read_bytes()
            argv = command.decode().rstrip("\0").split("\0")
            if not is_bn_job(argv, work):
                continue
            # Pin the Linux process identity before rechecking the command,
            # so a recycled PID cannot signal an unrelated process.
            fd = os.pidfd_open(int(process.name))
            try:
                if (process / "cmdline").read_bytes() != command:
                    continue
                signal.pidfd_send_signal(fd, signal.SIGTERM)
                stopped.append(int(process.name))
                print(
                    f"Stopped BN diagnostic PID {process.name}; source checkpoints preserved",
                    flush=True,
                )
            finally:
                os.close(fd)
        except (OSError, UnicodeError):
            continue
    return stopped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--hours", type=float, default=2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--replace-bn", action="store_true")
    args = parser.parse_args()
    if (
        not math.isfinite(args.hours)
        or args.hours <= 0
        or args.epochs < 1
        or not math.isfinite(args.lr)
        or args.lr <= 0
    ):
        raise ValueError("Positive finite budget, epochs and learning rate required")
    work = args.work.resolve()
    source = work / "runs/rtdetr_fold0_local"
    for filename in ("best.pt", "config.json", "split.json"):
        if not (source / filename).is_file():
            raise FileNotFoundError(source / filename)
    with (work / ".fullframe_finetune.launcher.lock").open("a") as startup:
        try:
            fcntl.flock(startup, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Full-frame fine-tuning already queued or running", flush=True)
            return
        if args.replace_bn:
            stop_bn(work)
        with (work / ".local_gpu_queue.lock").open("a") as lease:
            print(
                "Waiting for the GPU lease; existing training is preserved", flush=True
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
                    "train-fullframes",
                    "--work",
                    str(work),
                    "--lock-fd",
                    str(lease.fileno()),
                    "--hours",
                    str(args.hours),
                    "--epochs",
                    str(args.epochs),
                    "--lr",
                    str(args.lr),
                ],
            )


if __name__ == "__main__":
    main()
