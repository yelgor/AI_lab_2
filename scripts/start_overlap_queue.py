"""Hand off the legacy launcher, preserving its active training child."""

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml


def legacy_launchers(work, proc=Path("/proc")):
    result = []
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            argv = (directory / "cmdline").read_bytes().decode().strip("\0").split("\0")
            if not argv or not Path(argv[0]).name.startswith("python"):
                continue
            if not any(Path(arg).name == "local_gpu_queue.py" for arg in argv):
                continue
            if "--work" not in argv:
                continue
            if Path(argv[argv.index("--work") + 1]).resolve() == work:
                result.append(int(directory.name))
        except (OSError, UnicodeError, IndexError):
            continue
    return result


def training_parents(run, proc=Path("/proc")):
    matches = {}
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            argv = (directory / "cmdline").read_bytes().decode().strip("\0").split("\0")
            if directory.stat().st_uid != os.getuid() or not Path(
                argv[0]
            ).name.startswith("python"):
                continue
            index = argv.index("-m")
            if argv[index + 1 : index + 3] != ["cvlab", "train"]:
                continue
            config = yaml.safe_load(Path(argv[argv.index("--config") + 1]).read_text())
            if Path(config["output"]).resolve() != run:
                continue
            status = dict(
                line.split(":", 1)
                for line in (directory / "status").read_text().splitlines()
            )
            matches[int(directory.name)] = int(status["PPid"])
        except (OSError, ValueError, IndexError, KeyError, UnicodeError):
            continue
    return [pid for pid, parent in matches.items() if parent not in matches]


def latest_metric_line(path):
    with Path(path).open("rb") as stream:
        stream.seek(max(0, stream.seek(0, 2) - 16384))
        lines = stream.read().decode(errors="replace").splitlines()
    matches = [line for line in lines if re.match(r"^epoch=\d+ loss=.* mAP=", line)]
    return matches[-1] if matches else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["fcos", "fasterrcnn", "rtdetr"],
        choices=["fcos", "fasterrcnn", "retinanet", "rtdetr"],
    )
    parser.add_argument("--take-over", action="store_true")
    parser.add_argument("--handoff-run", type=Path)
    parser.add_argument("--handoff-log", type=Path)
    args = parser.parse_args()
    work = args.work.resolve()
    if bool(args.handoff_run) != bool(args.handoff_log):
        parser.error("Use --handoff-run and --handoff-log together")
    handoff_run = args.handoff_run.resolve() if args.handoff_run else None
    marker = latest_metric_line(args.handoff_log) if args.handoff_log else None
    if not (work / "configs/retinanet_fold0_local.yaml").is_file():
        raise FileNotFoundError(work / "configs/retinanet_fold0_local.yaml")
    if args.take_over:
        for pid in legacy_launchers(work):
            print(
                f"Stopping legacy queue launcher {pid}; active training child is preserved",
                flush=True,
            )
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    announced = False
    while True:
        if handoff_run and latest_metric_line(args.handoff_log) != marker:
            for pid in training_parents(handoff_run):
                print(
                    f"Validation checkpoint complete; stopping trainer {pid} before handoff",
                    flush=True,
                )
                try:
                    # Background jobs can inherit SIGINT=ignored. SIGTERM is
                    # issued only after the completed-checkpoint log marker.
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            handoff_run = None
        check = subprocess.run(
            ["pgrep", "-f", r"[p]ython.*-m cvlab (train|evaluate|predict|benchmark)"],
            capture_output=True,
            text=True,
        )
        if check.returncode not in (0, 1):
            raise RuntimeError(check.stderr)
        if not check.stdout.strip() and not legacy_launchers(work):
            break
        if not announced:
            print(
                "Waiting for the current validation checkpoint / run completion",
                flush=True,
            )
            announced = True
        time.sleep(0.2 if handoff_run else 2)
    project = Path(__file__).resolve().parents[1]
    os.chdir(project)
    os.environ.pop("CVLAB_DATA_ROOT", None)
    os.environ.pop("PYTHONPATH", None)
    # Active work is advertised by the standard train-prefix process name.
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-u",
            "-m",
            "cvlab",
            "train-overlap",
            "--work",
            str(work),
            "--models",
            *args.models,
        ],
    )


if __name__ == "__main__":
    main()
