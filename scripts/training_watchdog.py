"""Read-only training supervision; run periodically with a systemd user timer."""
import argparse
import csv
import fcntl
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import shutil
import subprocess
import time


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=15)
        return {"code": p.returncode, "output": p.stdout.strip(), "error": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"code": -1, "error": str(exc), "output": ""}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, help="Actual active job log; otherwise use the newest local log")
    parser.add_argument("--state", type=Path, help="Actual active plan state; otherwise use the newest plan")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "watchdog.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    logger = logging.getLogger("watchdog")
    logger.setLevel(logging.INFO)
    logger.addHandler(RotatingFileHandler(args.output / "watchdog.log", maxBytes=512000, backupCount=3))
    previous_path = args.output / "status.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    now = time.time()
    alerts = []
    processes = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            argv = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            if str(args.work) in argv and any(token in argv for token in ("-m cvlab train", "scripts/start_", "scripts/export_", "scripts/line_search_")):
                processes.append({"pid": int(proc.name), "argv": argv, "io": (proc / "io").read_text()})
        except (OSError, ProcessLookupError):
            pass
    mem = {k: int(v.split()[0]) for k, v in (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())}
    if mem["MemAvailable"] < 1500000:
        alerts.append("Available RAM below 1.5 GB")
    gpu = command(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"])
    if gpu["code"]:
        alerts.append("nvidia-smi failed; do not restart CUDA repeatedly")
    mount = command(["findmnt", "-n", "-o", "SOURCE,FSTYPE,OPTIONS", "--target", str(args.work)])
    if not args.work.is_dir() or "/dev/nvme0n1p5" not in mount["output"]:
        alerts.append("Expected work volume is not mounted")
    disk = shutil.disk_usage(args.work if args.work.exists() else args.output)
    if disk.free < 10 * 1024**3:
        alerts.append("Work disk has less than 10 GiB free")
    logs = list((args.work / "logs").glob("*.log"))
    log = args.log or (max(logs, key=lambda p: p.stat().st_mtime) if logs else args.work / "logs/efficient_search.log")
    tail = ""
    age = None
    if log.exists():
        age = now - log.stat().st_mtime
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 6000))
            tail = stream.read().decode(errors="replace")
        if processes and age > 3600:
            alerts.append("Log quiet for over 60 minutes: inspect CPU evaluation; no automatic kill")
        if any(term in tail for term in ("Traceback (most recent call last)", "CUDA error:", "OutOfMemoryError")):
            alerts.append("Failure signature in recent training log")
    if previous.get("processes") and not processes:
        alerts.append("Training exited; inspect final log (exit code unavailable for pre-existing process)")
    state = {}
    try:
        paths = [p for p in [args.work / "efficient_search/state.json",
                             *list((args.work / "full_data").glob("**/state.json"))] if p.is_file()]
        state_path = args.state or (max(paths, key=lambda p: p.stat().st_mtime) if paths else args.work / "efficient_search/state.json")
        state = json.loads(state_path.read_text())
    except (OSError, ValueError) as exc:
        alerts.append("Cannot read search state: " + str(exc))
    observations = []
    confirmations = []
    for cycle in state.get("cycles", []):
        for item in cycle.get("observations", []):
            observations.append({"cycle": cycle["number"], "trial": item.get("index"), "rung": item["rung"], "frames": item["batches"], "probe_mAP": item["probe_mAP"], "global_gain": item["global_gain"], "checkpoint": str(Path(item["run"]) / "best.pt")})
        confirmations.extend(cycle.get("confirmations", []))
    if observations:
        with (args.output / "progress.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(observations[0]))
            writer.writeheader()
            writer.writerows(observations)
    submissions = [{"path": str(p), "bytes": p.stat().st_size} for p in sorted((args.work / "submissions").glob("*.csv"))]
    status = {"time": now, "processes": processes, "gpu": gpu, "memory_kib": {k: mem[k] for k in ("MemAvailable", "SwapTotal", "SwapFree")}, "mount": mount, "disk_free_bytes": disk.free, "log": str(log), "plan_status": state.get("status"), "log_age_seconds": age, "log_tail": tail, "champion": state.get("champion"), "observations": observations, "confirmations": confirmations, "submissions": submissions, "alerts": alerts}
    temporary = previous_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2))
    temporary.replace(previous_path)
    changed = len(observations) != len(previous.get("observations", [])) or confirmations != previous.get("confirmations", []) or submissions != previous.get("submissions", [])
    new_alerts = [a for a in alerts if a not in previous.get("alerts", [])]
    message = "; ".join(new_alerts)
    if changed:
        message += f"; completed blocks={len(observations)}, confirmations={len(confirmations)}, CSVs={len(submissions)}"
    if message:
        command(["notify-send", "RT-DETR watchdog", message])
    logger.info(json.dumps({"time": now, "pids": [p["pid"] for p in processes], "gpu": gpu["output"], "available_kib": mem["MemAvailable"], "blocks": len(observations), "alerts": alerts}))


if __name__ == "__main__":
    main()
