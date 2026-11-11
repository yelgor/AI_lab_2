"""Local, bounded training supervision. Standard library only; no LLM/API calls."""

import argparse
import csv
import fcntl
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid


def read_json(path, default=None):
    try:
        with Path(path).open() as stream:
            return json.load(stream)
    except FileNotFoundError:
        return default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
    temporary.replace(path)


def command(argv, timeout=10):
    try:
        result = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return -1, "", type(error).__name__


def tail(path, size=32768):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(max(0, stream.seek(0, 2) - size))
            return stream.read(size).decode(errors="replace")
    except OSError:
        return ""


def process_info(pid, proc=Path("/proc")):
    try:
        folder = proc / str(pid)
        fields = (folder / "stat").read_text().rsplit(") ", 1)[1].split()
        info = {
            "pid": int(pid),
            "ppid": int(fields[1]),
            "start": fields[19],
            "cpu_ticks": int(fields[11]) + int(fields[12]),
            "io_bytes": 0,
            "state": fields[0],
            "rss_mib": int(fields[21]) * os.sysconf("SC_PAGE_SIZE") / 2**20,
        }
        for line in (folder / "io").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("rchar", "wchar"):
                info["io_bytes"] += int(value)
        return info
    except (OSError, ValueError, IndexError):
        return None


def alive(pid, start):
    info = process_info(pid)
    return info is not None and info["state"] != "Z" and info["start"] == start


def cvlab_processes(project, work, proc=Path("/proc")):
    """Inspect actual Python argv, exclude shell wrappers and collapse loader workers."""
    matches = {}
    for folder in proc.iterdir():
        if not folder.name.isdigit():
            continue
        try:
            if folder.stat().st_uid != os.getuid() or int(folder.name) == os.getpid():
                continue
            argv = (folder / "cmdline").read_bytes().decode().strip("\0").split("\0")
            if not argv or not Path(argv[0]).name.startswith("python"):
                continue
            module_job = any(
                argv[i : i + 2] == ["-m", "cvlab"] for i in range(len(argv) - 1)
            )
            script_job = any(
                Path(a).name
                in {
                    "start_efficient_search.py",
                    "start_adaptive_finetune.py",
                    "local_gpu_queue.py",
                    "start_overlap_queue.py",
                    "start_tiled_search.py",
                    "start_iterative_finetune.py",
                    "start_fullframe_finetune.py",
                    "start_crop_finetune.py",
                    "start_score_improvement.py",
                }
                for a in argv[1:]
            )
            if not (module_job or script_job):
                continue
            cwd = (folder / "cwd").resolve()
            related = cwd == project or any(
                a == str(work) or a.startswith(str(work) + "/") for a in argv
            )
            if related and (info := process_info(folder.name, proc)):
                matches[info["pid"]] = info
        except (OSError, UnicodeError):
            continue
    roots = [p for p in matches.values() if p["ppid"] not in matches]
    return (
        roots,
        sum(p["cpu_ticks"] for p in matches.values()),
        sum(p["io_bytes"] for p in matches.values()),
    )


def gpu_info(index):
    code, output, _ = command(
        [
            "nvidia-smi",
            "-i",
            str(index),
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if code:
        return {"ready": False, "compute_pids": []}
    try:
        util, used, total, temperature = next(csv.reader(output.splitlines()))
        info = dict(
            ready=True,
            utilization=int(util),
            used_mib=int(used),
            total_mib=int(total),
            temperature=int(temperature),
        )
        code, output, _ = command(
            [
                "nvidia-smi",
                "-i",
                str(index),
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ]
        )
        info["compute_pids"] = (
            [
                int(line.strip())
                for line in output.splitlines()
                if line.strip().isdigit()
            ]
            if not code
            else [-1]
        )
        return info
    except (ValueError, StopIteration):
        return {"ready": False, "compute_pids": []}


def controller_summary(path):
    value = read_json(path, {})
    cycles = value.get("cycles", [])
    submissions = sorted(
        {
            c["submission"]
            for cycle in cycles
            for c in cycle.get("confirmations", [])
            if c.get("submission")
        }
    )
    return {
        "complete_cycles": sum(bool(c.get("complete")) for c in cycles),
        "cycles": len(cycles),
        "observations": sum(len(c.get("observations", [])) for c in cycles),
        "rung": cycles[-1].get("rung") if cycles else None,
        "champion": value.get("champion_sha256"),
        "submissions": submissions,
    }


def snapshot(config, state):
    work, project = Path(config["work"]), Path(config["project"])
    mount = Path(config["mount"]) if config.get("mount") else None
    mounted = mount is None or os.path.ismount(mount)
    writable = mounted and work.is_dir() and os.access(work, os.W_OK)
    if mounted and mount is not None and work.is_dir():
        writable = writable and work.stat().st_dev == mount.stat().st_dev
    memory = dict(
        line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
    )
    roots, ticks, io_bytes = cvlab_processes(project, work)
    result = {
        "mounted": mounted,
        "writable": writable,
        "available_mib": int(memory["MemAvailable"].split()[0]) // 1024,
        "swap_free_mib": int(memory.get("SwapFree", "0 kB").split()[0]) // 1024,
        "roots": roots,
        "cpu_ticks": ticks,
        "io_bytes": io_bytes,
        "gpu": gpu_info(config["gpu"]),
        "work_free_gib": shutil.disk_usage(work).free / 2**30 if writable else 0,
        "state_free_mib": shutil.disk_usage(config["state_dir"]).free / 2**20,
        "controller": controller_summary(work / "efficient_search/state.json")
        if writable
        else {},
        "stop": (Path(config["state_dir"]) / "PAUSE").exists()
        or (mounted and (work / "runs/rtdetr_efficient_search/STOP").exists()),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    }
    if writable:
        logs = list((work / "logs").glob("*.log"))
        if logs:
            latest = max(logs, key=lambda path: path.stat().st_mtime)
            lines = tail(latest, 8192).replace("\r", "\n").splitlines()
            progress = [
                line
                for line in lines
                if line.startswith(
                    (
                        "epoch=",
                        "RACE ",
                        "PROBE ",
                        "CONFIRM ",
                        "Tiled ",
                        "READY FOR KAGGLE",
                        "ACCEPT",
                        "ROLLBACK",
                    )
                )
            ]
            result.update(log=str(latest), progress=progress[-1] if progress else None)
    if state.get("active_job"):
        job = read_json(state["active_job"], {})
        result["job"] = job
        result["job_alive"] = bool(
            job.get("pid")
            and job.get("boot_id") == result["boot_id"]
            and alive(job["pid"], job.get("start"))
        )
    return result


def failure_kind(text, code):
    lower = text.lower()
    if any(
        s in lower
        for s in (
            "unspecified launch failure",
            "device-side assert",
            "cuda unknown error",
            "cuda unavailable",
            "cuda недоступна",
        )
    ):
        return "cuda_fault"
    if "input/output error" in lower:
        return "storage_io_fault"
    if any(
        s in lower
        for s in (
            "read-only file system",
            "no space left on device",
            "volume is dirty",
            "hibernat",
        )
    ):
        return "filesystem_fault"
    if any(
        s in lower
        for s in (
            "valueerror:",
            "filenotfounderror:",
            "modulenotfounderror:",
            "importerror:",
            "typeerror:",
            "unpicklingerror:",
            "non-finite loss",
        )
    ):
        return "configuration_fault"
    if "cuda out of memory" in lower:
        return "gpu_memory"
    return "killed_or_oom" if code in (-9, 137) else "transient_exit"


def decide(config, state, sample, now):
    """Pure scheduling policy. Never signals/kills a live training process."""
    if sample.get("job_alive") or sample["roots"]:
        return "monitor", "Training or validation process is active"
    if sample["stop"]:
        return "paused", "Checkpoint-boundary pause requested"
    if not sample["mounted"] or not sample["writable"]:
        return (
            "waiting_mount",
            "Waiting for the configured data volume with write access",
        )
    if (
        sample["work_free_gib"] < config["min_disk_gib"]
        or sample["state_free_mib"] < 256
    ):
        return "waiting_disk", "Insufficient free space for checkpoints and exports"
    if state.get("blocked"):
        blocked = state["blocked"]
        if blocked["kind"] != "filesystem_fault" and (
            blocked["kind"] != "cuda_fault"
            or blocked.get("boot_id") == sample["boot_id"]
        ):
            return "blocked", blocked["kind"]
    deadline = state.get("deadline")
    if deadline and now >= deadline - 300:
        return "finished", "Configured wall-time budget exhausted"
    completed = sample["controller"].get("complete_cycles", 0) - state.get(
        "baseline_cycles", sample["controller"].get("complete_cycles", 0)
    )
    if completed >= config["max_cycles"]:
        return "finished", "Configured search cycles completed"
    if now < state.get("next_start", 0):
        return "cooldown", "Bounded retry backoff"
    if sample["available_mib"] < config["min_ram_mib"]:
        return "waiting_ram", "Waiting for available system RAM"
    if not sample["gpu"]["ready"]:
        return "waiting_gpu", "NVIDIA driver is unavailable"
    if sample["gpu"]["compute_pids"]:
        return "waiting_gpu", "Another GPU compute process is active"
    return "start", "Continue the saved efficient-search state"


def make_job(config, state, sample, now):
    if not state.get("deadline"):
        state["deadline"] = now + config["total_hours"] * 3600
        state["baseline_cycles"] = sample["controller"].get("complete_cycles", 0)
    hours = min(config["session_hours"], (state["deadline"] - now) / 3600)
    return {
        "status": "queued",
        "created": now,
        "config": config,
        "hours": hours,
        "reserve_hours": min(config["reserve_hours"], hours / 3),
        "before": sample["controller"],
    }


def run_worker(path):
    job = read_json(path)
    config = job["config"]
    # One durable receipt per job: a daemon restart cannot start the same job twice.
    with Path(str(path) + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if job["status"] in ("finished", "running"):
            return
        info = process_info(os.getpid())
        job.update(
            status="running",
            pid=os.getpid(),
            start=info["start"],
            started=time.time(),
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        )
        write_json(path, job)
        log = Path(config["work"]) / "logs/watchdog_training.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(log, maxBytes=5 * 1024**2, backupCount=2)
        logger = logging.getLogger("cvlab-job-" + str(os.getpid()))
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.propagate = False
        logger.info("WATCHDOG JOB %s", Path(path).stem)
        env = os.environ.copy()
        for key in ("PYTHONPATH", "CVLAB_DATA_ROOT"):
            env.pop(key, None)
        env.update(
            CUDA_VISIBLE_DEVICES=str(config["gpu"]),
            TORCH_HOME=str(Path(config["work"]) / "cache/torch"),
            HF_HOME=str(Path(config["work"]) / "cache/huggingface"),
        )
        python = str(Path(config["project"]) / ".venv/bin/python")
        argv = [
            python,
            "-u",
            "scripts/start_efficient_search.py",
            "--work",
            config["work"],
            "--hours",
            str(job["hours"]),
            "--reserve-hours",
            str(job["reserve_hours"]),
            "--cycles",
            "1",
            "--export-tiles",
        ]
        if config.get("inhibit", True):
            argv = [
                "systemd-inhibit",
                "--what=sleep:idle",
                "--mode=block",
                "--why=CVLab supervised training",
                "--",
                *argv,
            ]
        try:
            smoke = subprocess.run(
                [
                    python,
                    "-c",
                    'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; x=torch.ones(32,32,device="cuda"); y=x@x; torch.cuda.synchronize()',
                ],
                env=env,
                cwd=config["project"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=90,
            )
            if smoke.returncode:
                logger.error("CUDA preflight failed: %s", smoke.stderr[-8192:])
                code, reason = (
                    smoke.returncode,
                    failure_kind(smoke.stderr, smoke.returncode),
                )
            else:
                recent_output = ""
                with subprocess.Popen(
                    argv,
                    cwd=config["project"],
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                ) as process:
                    for chunk in iter(lambda: process.stdout.readline(8192), b""):
                        decoded = chunk.decode(errors="replace")
                        recent_output = (recent_output + decoded)[-32768:]
                        logger.info(decoded.rstrip("\n"))
                    code = process.wait()
                reason = failure_kind(recent_output, code) if code else None
        except (OSError, subprocess.TimeoutExpired) as error:
            logger.error("Worker failed: %s", str(error))
            code, reason = (
                -1,
                "cuda_fault"
                if isinstance(error, subprocess.TimeoutExpired)
                else "configuration_fault",
            )
        finally:
            handler.close()
            logger.removeHandler(handler)
        job.update(
            status="finished", finished=time.time(), returncode=code, reason=reason
        )
        write_json(path, job)


def worker_entry(path):
    try:
        run_worker(path)
    except BlockingIOError:
        return  # Another wrapper owns this exact durable job.
    except (OSError, ValueError) as error:
        job = read_json(path, {})
        if job.get("pid") != os.getpid():
            raise
        reason = failure_kind(str(error), -1)
        job.update(
            status="finished",
            finished=time.time(),
            returncode=-1,
            reason="configuration_fault" if reason == "transient_exit" else reason,
            error=str(error)[-1024:],
        )
        write_json(path, job)


def consume_job(state, job, sample, config, now):
    """Commit a durable worker result once; failures get a finite retry budget."""
    state["active_job"] = None
    state["last_job"] = {
        k: job.get(k) for k in ("started", "finished", "returncode", "reason", "error")
    }
    if job["returncode"] == 0:
        if job["before"] == sample["controller"] and not sample["stop"]:
            state["blocked"] = {"kind": "no_progress", "boot_id": sample["boot_id"]}
        state["next_start"] = now + config["interval"]
        return
    failures = state.get("failures", 0) + 1
    state["failures"] = failures
    kind = job.get("reason") or "transient_exit"
    if (
        kind
        in ("cuda_fault", "configuration_fault", "filesystem_fault", "storage_io_fault")
        or failures > config["max_retries"]
    ):
        state["blocked"] = {
            "kind": kind if failures <= config["max_retries"] else "retry_limit",
            "boot_id": sample["boot_id"],
        }
    state["next_start"] = now + min(900, 60 * 2 ** min(failures, 4))


class Watchdog:
    def __init__(self, config):
        self.config = config
        self.directory = Path(config["state_dir"])
        self.directory.mkdir(parents=True, exist_ok=True)
        self.state_path = self.directory / "state.json"
        self.state = read_json(
            self.state_path, {"version": 1, "failures": 0, "active_job": None}
        )
        self.child = None
        self.logger = logging.getLogger("cvlab-watchdog-" + str(os.getpid()))
        self.logger.setLevel(logging.INFO)
        self.handler = RotatingFileHandler(
            self.directory / "events.log", maxBytes=1024**2, backupCount=2
        )
        self.logger.addHandler(self.handler)
        self.logger.propagate = False

    def event(self, key, message):
        self.logger.info("%s %s %s", time.strftime("%Y-%m-%dT%H:%M:%S%z"), key, message)
        self.handler.flush()
        previous = self.state.setdefault("notifications", {}).get(key, 0)
        if time.time() - previous >= 900:
            self.state["notifications"][key] = time.time()
            if shutil.which("notify-send"):
                command(
                    [
                        "notify-send",
                        "--app-name=CVLab watchdog",
                        "CVLab training",
                        message,
                    ],
                    timeout=3,
                )

    def tick(self):
        config, state, now = self.config, self.state, time.time()
        if self.child is not None and self.child.poll() is not None:
            self.child = None
        try:
            sample = snapshot(config, state)
        except (OSError, ValueError) as error:
            self.event(
                "sensor_error", f"Local status read failed: {type(error).__name__}"
            )
            state.update(
                action="sensor_error", reason=type(error).__name__, updated=now
            )
            write_json(self.state_path, state)
            write_json(
                self.directory / "status.json",
                {
                    "action": "sensor_error",
                    "reason": type(error).__name__,
                    "updated": now,
                },
            )
            return
        job = sample.get("job", {})
        if job.get("status") == "finished":
            consume_job(state, job, sample, config, now)
            self.event(
                "job_exit",
                f"Training session exit={job['returncode']}; reason={job.get('reason') or 'completed checkpoint boundary'}",
            )
            sample.pop("job", None)
        elif state.get("active_job") and not sample.get("job_alive"):
            # Queued/running worker lost after a crash; never duplicate a surviving child.
            if (
                not sample["roots"]
                and now - job.get("created", state.get("active_created", 0)) > 120
            ):
                lost = {**job, "returncode": -1, "reason": "transient_exit"}
                consume_job(state, lost, sample, config, now)
            else:
                sample["job_alive"] = True
        action, reason = decide(config, state, sample, now)
        if action == "start":
            checkpoint = Path(config["work"]) / "runs/rtdetr_fold0_local/best.pt"
            if (
                not checkpoint.is_file()
                or not (Path(config["project"]) / ".venv/bin/python").is_file()
            ):
                action, reason = (
                    "blocked",
                    "Original checkpoint or project interpreter is missing",
                )
            else:
                with (Path(config["work"]) / ".local_gpu_queue.lock").open(
                    "a"
                ) as lease:
                    try:
                        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        fcntl.flock(lease, fcntl.LOCK_UN)
                    except BlockingIOError:
                        action, reason = (
                            "waiting_gpu",
                            "Existing GPU queue owns the lease",
                        )
            if action == "start":
                state.pop("blocked", None)
                job_path = self.directory / "jobs" / f"{uuid.uuid4().hex}.json"
                write_json(job_path, make_job(config, state, sample, now))
                state["active_job"] = str(job_path)
                state["active_created"] = now
                write_json(self.state_path, state)
                self.child = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "cvlab.training_watchdog",
                        "--worker",
                        str(job_path),
                    ],
                    cwd=config["project"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                action, reason = (
                    "monitor",
                    "Started a bounded resumable training session",
                )
        if (
            action == "waiting_mount"
            and not sample["mounted"]
            and config.get("device")
            and now - state.get("last_mount_attempt", 0) >= 900
        ):
            state["last_mount_attempt"] = now
            code, _, error = command(
                [
                    "udisksctl",
                    "mount",
                    "--no-user-interaction",
                    "--block-device",
                    config["device"],
                    "--filesystem-type",
                    "ntfs",
                ]
            )
            if code:
                if error[-1000:] != state.get("mount_error"):
                    self.event(
                        "mount_failed", "Normal volume mount failed: " + error[-1000:]
                    )
                state["mount_error"] = error[-1000:]
        signature = [
            sample["cpu_ticks"],
            sample["io_bytes"],
            sample.get("controller"),
            os.path.getmtime(Path(config["work"]) / "logs/watchdog_training.log")
            if sample["writable"]
            and (Path(config["work"]) / "logs/watchdog_training.log").exists()
            else None,
        ]
        activity = hashlib.sha256(
            json.dumps(signature, sort_keys=True).encode()
        ).hexdigest()
        if (
            activity != state.get("activity")
            or sample["gpu"].get("utilization", 0) >= 10
        ):
            state.update(activity=activity, last_activity=now)
        if (
            action == "monitor"
            and now - state.get("last_activity", now) >= config["stall_minutes"] * 60
        ):
            self.event(
                "quiet_process",
                "No observed CPU, I/O, GPU or log progress; inspect the live process (it was preserved)",
            )
        for submission in sample.get("controller", {}).get("submissions", []):
            if submission not in state.setdefault("seen_submissions", []):
                state["seen_submissions"].append(submission)
                self.event(
                    "submission_" + Path(submission).stem,
                    f"READY FOR KAGGLE: {submission}",
                )
        if sample["gpu"].get("temperature", 0) >= 88:
            self.event("temperature", "GPU temperature >=88 C; check ventilation")
        if sample["available_mib"] < 1024:
            self.event(
                "memory_pressure",
                "Available system RAM is below 1 GiB; checkpoints and retry limits are preserved",
            )
        if action == "monitor" and not sample["writable"]:
            self.event(
                "volume_lost",
                "The work volume is unavailable or read-only while a process is active",
            )
        if action != state.get("action"):
            self.event(action, reason)
        state.update(action=action, reason=reason, updated=now)
        write_json(self.state_path, state)
        write_json(
            self.directory / "status.json",
            {
                "action": action,
                "reason": reason,
                "updated": now,
                "deadline": state.get("deadline"),
                "failures": state.get("failures", 0),
                "sample": sample,
            },
        )


def validate_config(config):
    for key in ("project", "work", "state_dir"):
        if not Path(config[key]).is_absolute():
            raise ValueError(f"{key} must be absolute")
    state, work = Path(config["state_dir"]), Path(config["work"])
    if state == work or work in state.parents:
        raise ValueError("Watchdog state must be outside the data volume work tree")
    if config.get("mount"):
        mount = Path(config["mount"])
        if (
            not mount.is_absolute()
            or mount not in work.parents
            or state == mount
            or mount in state.parents
        ):
            raise ValueError(
                "Work must be inside its mount; watchdog state must be outside it"
            )
    for key in (
        "total_hours",
        "session_hours",
        "reserve_hours",
        "interval",
        "min_ram_mib",
        "min_disk_gib",
        "stall_minutes",
    ):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"Positive finite {key} required")
    if (
        config["reserve_hours"] >= config["session_hours"]
        or config["max_cycles"] < 1
        or config["max_retries"] < 0
        or config["gpu"] < 0
    ):
        raise ValueError("Invalid search/retry budget")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--pause", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rearm", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker_entry(args.worker)
        return
    if not args.config:
        parser.error("--config is required")
    config = read_json(args.config)
    validate_config(config)
    directory = Path(config["state_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    if args.status:
        status = read_json(directory / "status.json", {"action": "not_started"})
        print(status["action"] + ": " + status.get("reason", ""))
        sample = status.get("sample", {})
        gpu = sample.get("gpu", {})
        if gpu.get("ready"):
            print(
                f"GPU: {gpu.get('utilization')}%; VRAM={gpu.get('used_mib')}/{gpu.get('total_mib')} MiB; temperature={gpu.get('temperature')} C"
            )
        print(
            f"Available RAM: {sample.get('available_mib', '?')} MiB; retries used: {status.get('failures', 0)}"
        )
        if sample.get("progress"):
            print(sample["progress"])
        if status.get("deadline"):
            print(
                "Budget ends:",
                time.strftime(
                    "%Y-%m-%d %H:%M:%S %z", time.localtime(status["deadline"])
                ),
            )
        print("Full local status:", directory / "status.json")
        return
    if args.pause or args.resume:
        marker = directory / "PAUSE"
        if args.pause:
            marker.touch()
            work = Path(config["work"])
            if work.is_dir() and (
                not config.get("mount") or os.path.ismount(config["mount"])
            ):
                target = work / "runs/rtdetr_efficient_search"
                target.mkdir(parents=True, exist_ok=True)
                (target / "STOP").touch()
        else:
            marker.unlink(missing_ok=True)
            work_marker = Path(config["work"]) / "runs/rtdetr_efficient_search/STOP"
            if not config.get("mount") or os.path.ismount(config["mount"]):
                work_marker.unlink(missing_ok=True)
        return
    if args.rearm:
        with (directory / "supervisor.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state = read_json(directory / "state.json", {})
            sample = snapshot(config, state)
            if sample.get("job_alive") or sample["roots"]:
                raise RuntimeError(
                    "Rearm only after active work finishes; the live process was preserved"
                )
            write_json(
                directory / "state.json",
                {"version": 1, "failures": 0, "active_job": None},
            )
        return
    with (directory / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        watchdog = Watchdog(config)
        stop = False

        def shutdown(*_):
            nonlocal stop
            stop = True

        signal.signal(signal.SIGTERM, shutdown)
        signal.signal(signal.SIGINT, shutdown)
        while not stop:
            watchdog.tick()
            if args.once:
                break
            for _ in range(max(1, int(config["interval"]))):
                if stop:
                    break
                time.sleep(1)
        watchdog.handler.close()


if __name__ == "__main__":
    main()
