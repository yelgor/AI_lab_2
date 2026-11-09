"""Run independent models sequentially on one Linux GPU, reusing prepared data."""

import argparse
import copy
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


def process_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def wait_for_gpu(wait_pid=None, poll_seconds=15):
    if wait_pid:
        print(f"Waiting for existing launcher PID {wait_pid}", flush=True)
        while process_alive(wait_pid):
            time.sleep(poll_seconds)
    while (
        subprocess.run(
            ["pgrep", "-f", r"[p]ython.*-m cvlab (train|evaluate|predict|benchmark)"],
            stdout=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    ):
        time.sleep(poll_seconds)


def run_cli(project, arguments):
    env = os.environ.copy()
    env.pop("CVLAB_DATA_ROOT", None)
    env.pop("PYTHONPATH", None)
    subprocess.run(
        [sys.executable, "-u", "-m", "cvlab", *map(str, arguments)],
        cwd=project,
        env=env,
        stdin=subprocess.DEVNULL,
        check=True,
    )


def model_config(project, work, reference, model):
    config = yaml.safe_load((project / "configs" / f"{model}.yaml").read_text())
    for section in ("data", "loader", "runtime"):
        config[section] = copy.deepcopy(reference[section])
    config["seed"] = reference["seed"]
    config["model"]["image_size"] = reference["model"]["image_size"]
    for key in ("epochs", "max_hours", "accumulation", "max_batches_per_epoch"):
        config["train"][key] = reference["train"][key]
    name = f"{model}_fold{config['data']['fold']}_local"
    config["output"] = str(work / "runs" / name)
    path = work / "configs" / f"{name}.yaml"
    if path.is_file() and yaml.safe_load(path.read_text()) != config:
        raise ValueError(
            f"Existing configuration differs; preserved without changes: {path}"
        )
    temporary = path.with_suffix(".yaml.tmp")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False))
    temporary.replace(path)
    return path, Path(config["output"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument(
        "--models",
        nargs="*",
        default=["fcos", "fasterrcnn", "rtdetr"],
        choices=["fcos", "fasterrcnn", "retinanet", "rtdetr"],
    )
    parser.add_argument(
        "--predict-retina",
        action="store_true",
        help="Generate an initial RetinaNet submission before queued training",
    )
    args = parser.parse_args(argv)
    if args.wait_pid is not None and args.wait_pid <= 0:
        parser.error("--wait-pid must be a positive process ID")
    work = args.work.resolve()
    project = Path(__file__).resolve().parents[1]
    reference_path = work / "configs/retinanet_fold0_local.yaml"
    reference = yaml.safe_load(reference_path.read_text())
    with (work / ".local_gpu_queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("A local queue is already running; no duplicate started.", flush=True)
            return
        wait_for_gpu(args.wait_pid)
        if args.predict_retina:
            checkpoint = work / "runs/retinanet_fold0_local/best.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            run_cli(
                project,
                [
                    "predict",
                    "--config",
                    reference_path,
                    "--checkpoint",
                    checkpoint,
                    "--out",
                    work / "submissions/retinanet_fold0.csv",
                ],
            )
        for model in args.models:
            wait_for_gpu()
            config, output = model_config(project, work, reference, model)
            arguments = ["train", "--config", config]
            if (output / "last.pt").is_file():
                arguments += ["--resume", output / "last.pt"]
            print(f"Starting {model}: {output}", flush=True)
            run_cli(project, arguments)
            if (output / "val_predictions.json.tmp").exists() or not (
                output / "val_predictions.json"
            ).is_file():
                run_cli(
                    project,
                    [
                        "evaluate",
                        "--config",
                        config,
                        "--checkpoint",
                        output / "best.pt",
                        "--out",
                        output / "val_predictions.json",
                    ],
                )


if __name__ == "__main__":
    main()
