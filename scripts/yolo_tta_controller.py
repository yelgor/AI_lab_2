"""Monitor a stable YOLO course and run frozen-champion TTA checks every 2 epochs."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _notify(title, message):
    print(f"NOTIFY {title}: {message}", flush=True)
    if shutil.which("notify-send"):
        subprocess.run(["notify-send", title, message], check=False)


def _run(args):
    print("RUN", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True)


def _active(unit):
    result = subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", unit], check=False
    )
    return result.returncode == 0


def _history(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _write_status(path, **fields):
    value = {"updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **fields}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False, indent=2) + "\n")
    temporary.replace(path)


def _health(work, run, unit, log):
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        fields = value.split()
        if fields and fields[0].isdigit():
            memory[key] = int(fields[0])
    disk_free = shutil.disk_usage(work).free
    gpu = None
    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
            if len(parts) == 5:
                gpu = {
                    "utilization_percent": int(parts[0]),
                    "memory_used_mib": int(parts[1]),
                    "memory_total_mib": int(parts[2]),
                    "temperature_c": int(parts[3]),
                    "power_w": float(parts[4]),
                }
    progress = None
    if log.exists():
        import re

        tail = log.read_bytes()[-8192:].decode("utf-8", errors="replace")
        matches = list(re.finditer(r"(\d+)/24201\s+([0-9.]+)it/s", tail))
        if matches:
            progress = {
                "batch": int(matches[-1].group(1)),
                "dataset_size": 24201,
                "batches_per_second": float(matches[-1].group(2)),
            }
    return {
        "sampled_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "training_unit_active": _active(unit),
        "mem_available_mib": memory.get("MemAvailable", 0) // 1024,
        "swap_free_mib": memory.get("SwapFree", 0) // 1024,
        "disk_free_bytes": disk_free,
        "gpu": gpu,
        "progress": progress,
        "numeric_failure_present": (run / "numeric_failure.json").is_file(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-name", default="yolo11s_fp32_continuation")
    parser.add_argument("--first-unit", required=True)
    parser.add_argument("--initial-checkpoint", required=True, type=Path)
    parser.add_argument("--champion-report", required=True, type=Path)
    parser.add_argument("--champion-csv", required=True, type=Path)
    parser.add_argument("--c1-val", required=True, type=Path)
    parser.add_argument("--c1-test", required=True, type=Path)
    parser.add_argument("--session-hours", type=float, default=4)
    parser.add_argument("--max-hours", type=float, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--poll-seconds", type=float, default=0.2)
    args = parser.parse_args()

    work = args.work.resolve()
    run = work / "runs" / args.run_name
    weights = run / "weights"
    history_path = run / "common_history.jsonl"
    status_path = run / "status.json"
    controller_status = run / "tta_controller_status.json"
    log = work / "logs" / f"{args.run_name}.log"
    init = args.initial_checkpoint.resolve()
    started = time.monotonic()
    unit = args.first_unit
    # The first segment is already running under first-unit; all segments this
    # controller launches continue that same fixed 20-epoch schedule.
    resume = True
    cycle = 0
    desired_pair = 2
    next_health_check = 0.0
    last_candidate = None
    alerted = set()

    def launch_training():
        nonlocal unit, resume, cycle
        if shutil.disk_usage(work).free < 4 * 2**30:
            _write_status(
                controller_status,
                phase="paused_low_disk",
                free_bytes=shutil.disk_usage(work).free,
            )
            _notify(
                "CVLab YOLO paused",
                "Disk headroom is below the 4 GiB safety floor; checkpoints preserved.",
            )
            return False
        cycle += 1
        unit = f"cvlab-yolo-fp32-r{cycle:02d}.service"
        command = [
            "systemd-run",
            "--user",
            f"--unit={unit.removesuffix('.service')}",
            "--collect",
            "--property=Type=exec",
            f"--property=WorkingDirectory={ROOT}",
            f"--property=StandardOutput=append:{log}",
            f"--property=StandardError=append:{log}",
            "/usr/bin/systemd-inhibit",
            "--what=sleep:idle",
            "--mode=block",
            "--why=YOLO FP32 continuation course",
            sys.executable,
            "-u",
            ROOT / "scripts/start_yolo_training.py",
            "--work",
            work,
            "--config",
            args.config.resolve(),
            "--name",
            args.run_name,
            "--init-checkpoint",
            init,
            "--epochs",
            args.epochs,
            "--hours",
            args.session_hours,
            "--batch",
            1,
            "--imgsz",
            1280,
            "--workers",
            2,
            "--no-amp",
            "--lr",
            "0.0001",
            "--freeze-batch-norm",
        ]
        if resume:
            command.append("--resume")
        _run(command)
        resume = True
        return True

    def evaluate_epoch(row):
        nonlocal last_candidate
        checkpoint = weights / f"epoch{row['epoch']}.pt"
        from cvlab.yolo_training import file_sha
        from cvlab.class_ensemble import snapshot

        if not checkpoint.is_file() or file_sha(checkpoint) != row["checkpoint_sha256"]:
            raise RuntimeError(
                "Saved epoch checkpoint does not match full-validation receipt"
            )
        frozen = snapshot(checkpoint, work / "ensemble" / "snapshots")
        checkpoint_id = row["checkpoint_sha256"]
        base = work / "ensemble" / "yolo_fp32_candidates" / checkpoint_id
        native_val, flip_val = base / "native_val", base / "horizontal_flip_val"
        common = [
            sys.executable,
            ROOT / "scripts/class_ensemble.py",
            "--work",
            work,
            "--config",
            run / "config.json",
        ]
        for output, view in ((native_val, "native"), (flip_val, "horizontal_flip")):
            _run(
                common
                + [
                    "cache",
                    "--backend",
                    "yolo",
                    "--checkpoint",
                    frozen,
                    "--output",
                    output,
                    "--split",
                    "val",
                    "--view",
                    view,
                ]
            )
        report = base / "comparison_vs_scored_champion.json"
        _run(
            common
            + [
                "compare-champion",
                "--c1",
                args.c1_val,
                "--candidate-native",
                native_val,
                "--candidate-flip",
                flip_val,
                "--champion-report",
                args.champion_report,
                "--champion-csv",
                args.champion_csv,
                "--output",
                report,
            ]
        )
        comparison = json.loads(report.read_text())
        eligible = []
        for route in ("truck_bus_c1", "all_yolo"):
            decision = comparison["decisions_vs_champion"][route]
            if decision["accepted"] or comparison["review_eligible"][route]:
                eligible.append(
                    (
                        comparison["cases"][route]["metrics"]["coco100"]["mAP"],
                        route,
                        not decision["accepted"],
                    )
                )
        csv_path = None
        if eligible:
            _, route, manual = max(eligible)
            native_test, flip_test = base / "native_test", base / "horizontal_flip_test"
            for output, view in (
                (native_test, "native"),
                (flip_test, "horizontal_flip"),
            ):
                _run(
                    common
                    + [
                        "cache",
                        "--backend",
                        "yolo",
                        "--checkpoint",
                        frozen,
                        "--output",
                        output,
                        "--split",
                        "test",
                        "--view",
                        view,
                    ]
                )
            csv_path = (
                work
                / "submissions"
                / f"yolo_fp32_epoch_{row['epoch'] + 1:02d}_{checkpoint_id[:12]}_{route}.csv"
            )
            export = common + [
                "export-champion",
                "--c1-val",
                args.c1_val,
                "--c1-test",
                args.c1_test,
                "--candidate-native-val",
                native_val,
                "--candidate-flip-val",
                flip_val,
                "--candidate-native-test",
                native_test,
                "--candidate-flip-test",
                flip_test,
                "--champion-report",
                args.champion_report,
                "--champion-csv",
                args.champion_csv,
                "--report",
                report,
                "--output",
                csv_path,
                "--route",
                route,
            ]
            if manual:
                export.append("--manual-review")
            _run(export)
        summary = {
            "phase": "candidate_checked",
            "epoch": row["epoch"],
            "checkpoint_sha256": checkpoint_id,
            "comparison": str(report.relative_to(work)),
            "decisions_vs_champion": comparison["decisions_vs_champion"],
            "csv": str(csv_path.relative_to(work)) if csv_path else None,
            "csv_sha256": None,
        }
        if csv_path:
            summary["csv_sha256"] = file_sha(csv_path)
        last_candidate = summary
        _write_status(controller_status, **summary)
        for route in ("truck_bus_c1", "all_yolo"):
            metrics = comparison["cases"][route]["metrics"]
            print(
                f"CANDIDATE epoch={row['epoch'] + 1} route={route} "
                f"AP100={metrics['coco100']['mAP']:.6f} "
                f"AP500={metrics['coco500']['mAP']:.6f} "
                f"APsmall={metrics['coco100']['AP_small']:.6f} "
                f"gain={comparison['decisions_vs_champion'][route].get('pooled_gain')} "
                f"families={comparison['decisions_vs_champion'][route].get('per_family_gain')}",
                flush=True,
            )
        message = f"Epoch {row['epoch'] + 1} full TTA check complete. " + (
            f"Candidate CSV: {csv_path.name}"
            if csv_path
            else "Champion retained; no candidate passed."
        )
        _notify("CVLab YOLO candidate checked", message)

    while time.monotonic() - started < args.max_hours * 3600:
        rows = _history(history_path)
        now = time.monotonic()
        if now >= next_health_check:
            health = _health(work, run, unit, log)
            _write_status(
                controller_status,
                phase="training",
                completed_epochs=len(rows),
                target_epochs=desired_pair,
                training_unit=unit,
                health=health,
                last_candidate=last_candidate,
            )
            warnings = []
            if health["mem_available_mib"] < 4096:
                warnings.append("available RAM below 4 GiB")
            if health["disk_free_bytes"] < 4 * 2**30:
                warnings.append("disk headroom below 4 GiB")
            if health["gpu"] is None:
                warnings.append("CUDA telemetry unavailable")
            if health["numeric_failure_present"]:
                _notify(
                    "CVLab YOLO numeric failure",
                    "Training numeric-failure receipt detected; no automatic resume.",
                )
                return 2
            warning_key = "; ".join(warnings)
            if warning_key and warning_key not in alerted:
                _notify("CVLab YOLO resource warning", warning_key)
                alerted.add(warning_key)
            next_health_check = now + 60
        if len(rows) >= desired_pair:
            row = rows[desired_pair - 1]
            if _active(unit):
                (run / "STOP").touch()
                print(
                    f"STOP requested after {desired_pair} full epochs; waiting for saved boundary",
                    flush=True,
                )
                while _active(unit):
                    time.sleep(1)
            evaluate_epoch(row)
            if row["epoch"] + 1 >= args.epochs:
                break
            (run / "STOP").unlink(missing_ok=True)
            desired_pair += 2
            if not launch_training():
                return 0
            continue

        if not _active(unit):
            if rows and rows[-1]["epoch"] + 1 >= args.epochs:
                break
            if not launch_training():
                return 0
        elif len(rows) >= desired_pair - 1:
            current = rows[desired_pair - 2]
            try:
                status = json.loads(status_path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                status = {}
            # Request STOP only after the first epoch's callback has completed
            # its own STOP check; the following saved boundary is epoch two.
            if (
                status.get("last_epoch") == current["epoch"]
                and status.get("status") == "training"
            ):
                (run / "STOP").touch()
                print(
                    f"Pair boundary armed after epoch {current['epoch'] + 1}",
                    flush=True,
                )

        time.sleep(args.poll_seconds)

    if _active(unit):
        (run / "STOP").touch()
        print(
            "Overall course budget reached; requesting the next safe epoch boundary",
            flush=True,
        )
        while _active(unit):
            time.sleep(1)
    final_rows = _history(history_path)
    _write_status(
        controller_status,
        phase="course_budget_complete",
        completed_epochs=len(final_rows),
        last_epoch=final_rows[-1]["epoch"] if final_rows else None,
    )
    _notify(
        "CVLab YOLO course paused",
        f"Stable continuation reached controller budget with {len(final_rows)} validated epochs; checkpoints remain saved.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
