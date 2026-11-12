"""Small CPU-only checks of local artifacts and submission CSV; no tensor loading."""

import argparse
import csv
import hashlib
import json
import math
import subprocess
import zipfile
from pathlib import Path

MODELS = ("retinanet", "fcos", "fasterrcnn", "rtdetr")


def small_json(path):
    if path.stat().st_size > 8 * 1024**2:
        raise ValueError(f"Metadata is unexpectedly large; not loaded: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Metadata must be a JSON object: {path}")
    return result


def inspect_run(path):
    result = {"path": str(path), "name": path.name}
    for filename in ("config.json", "best_metrics.json", "split.json"):
        file = path / filename
        if file.is_file():
            try:
                result[filename] = small_json(file)
            except (ValueError, OSError) as error:
                result.setdefault("errors", []).append(str(error))
    config = result.pop("config.json", {})
    metrics = result.pop("best_metrics.json", {})
    split = result.pop("split.json", {})
    result.update(
        model=config.get("model", {}).get("name"),
        fold=config.get("data", {}).get("fold"),
    )
    result["split"] = config.get("data", {}).get("split")
    result["limited_training"] = (
        config.get("train", {}).get("max_batches_per_epoch") is not None
    )
    result["checkpoints"] = {
        name: (path / name).stat().st_size if (path / name).is_file() else 0
        for name in ("last.pt", "best.pt")
    }
    result["checkpoint_zip_headers"] = {
        name: zipfile.is_zipfile(path / name) if size else False
        for name, size in result["checkpoints"].items()
    }
    result["mAP"] = metrics.get("metrics", {}).get("coco100", {}).get("mAP")
    score = result["mAP"]
    score_ok = (
        isinstance(score, (int, float))
        and not isinstance(score, bool)
        and math.isfinite(score)
        and 0 <= score <= 1
    )
    if score is not None and not score_ok:
        result.setdefault("errors", []).append("Invalid mAP value")
        result["mAP"] = None
    result["validated_result"] = bool(
        result["checkpoint_zip_headers"]["best.pt"] and score_ok
    )
    result["sequence_leakage"] = bool(
        set(split.get("train_sequences", ())) & set(split.get("val_sequences", ()))
    )
    def family(sequence):
        return sequence.split("_", 1)[0] if sequence.startswith("uav") else sequence
    seen = set(split.get("train_sequences", ())) | set(split.get("seen_training_sequences", ()))
    result["flight_family_leakage"] = bool(
        {family(s) for s in seen} & {family(s) for s in split.get("val_sequences", ())}
    )
    if result["flight_family_leakage"]:
        result.setdefault("errors", []).append("Train/inherited validation flight families overlap")
    if result["sequence_leakage"]:
        result.setdefault("errors", []).append("Train/validation sequences overlap")
    result["validated_result"] &= not result.get("errors", False)
    result["history_rows"] = 0
    history = path / "history.jsonl"
    if history.is_file() and history.stat().st_size <= 8 * 1024**2:
        for line in history.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                result["history_incomplete_line"] = True
                continue
            if not isinstance(row, dict):
                result["history_incomplete_line"] = True
                continue
            result["history_rows"] += 1
            result["last_measured_epoch"] = row.get("epoch")
            result["elapsed_seconds"] = row.get("elapsed_seconds")
    # Presence is evidence only. Never unpickle or load multi-GB predictions.
    result["validation_artifact_present"] = (path / "val_predictions.json").is_file()
    result["checkpoint_phase"] = "not_loaded"
    result["quality_metrics"] = metrics.get("metrics", {}).get("coco100", {})
    result["timing"] = metrics.get("timing", {})
    result["selection_decision"] = metrics.get("selection_decision")
    return result


def validate_submission(path, test_csv, sample=None):
    images = {}
    with Path(test_csv).open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            key = row["image_id"]
            if not key or key in images:
                raise ValueError("Duplicate or empty image_id in test_images.csv")
            size = int(row["width"]), int(row["height"])
            if min(size) <= 0:
                raise ValueError("Invalid test image dimensions")
            images[key] = size
    if not images:
        raise ValueError("Empty test_images.csv")
    expected_order = list(images)
    if sample is not None and Path(sample).is_file():
        with Path(sample).open(newline="", encoding="utf-8-sig") as stream:
            expected_order = [row["image_id"] for row in csv.DictReader(stream)]
        if len(expected_order) != len(images) or set(expected_order) != set(images):
            raise ValueError("Sample submission IDs differ from test images")
    seen, actual_order, detections, empty = set(), [], 0, 0
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["image_id", "PredictionString"]:
            raise ValueError("Expected columns: image_id,PredictionString")
        for number, row in enumerate(reader, 2):
            key, prediction = row["image_id"], row["PredictionString"]
            if None in row or prediction is None:
                raise ValueError(f"Row {number}: wrong CSV field count")
            if key not in images or key in seen:
                raise ValueError(f"Row {number}: unknown or duplicate image_id {key}")
            seen.add(key)
            actual_order.append(key)
            values = prediction.split()
            if len(values) % 6 or len(values) > 500 * 6:
                raise ValueError(
                    f"Row {number}: expected <=500 groups of class score x y width height"
                )
            width, height = images[key]
            empty += not values
            for offset in range(0, len(values), 6):
                category = int(values[offset])
                score, x, y, w, h = map(float, values[offset + 1 : offset + 6])
                if not 0 <= category <= 7 or not all(
                    map(math.isfinite, (score, x, y, w, h))
                ):
                    raise ValueError(f"Row {number}: invalid class or non-finite value")
                if not 0 <= score <= 1 or x < 0 or y < 0 or w <= 0 or h <= 0:
                    raise ValueError(f"Row {number}: invalid score or box")
                if x + w > width + 0.001 or y + h > height + 0.001:
                    raise ValueError(
                        f"Row {number}: box exceeds original image dimensions"
                    )
                detections += 1
    if seen != set(images):
        raise ValueError(f"Missing {len(set(images) - seen)} test images")
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return {
        "valid": True,
        "images": len(seen),
        "detections": detections,
        "empty_images": empty,
        "all_predictions_empty": detections == 0,
        "matches_template_order": actual_order == expected_order,
        "sha256": sha.hexdigest(),
    }


def memory_status():
    result = {}
    memory = Path("/proc/meminfo")
    if memory.is_file():
        fields = dict(line.split(":", 1) for line in memory.read_text().splitlines())
        result["available_ram_mib"] = int(fields["MemAvailable"].split()[0]) / 1024
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        result["gpu_snapshot"] = (
            gpu.stdout.strip() if gpu.returncode == 0 else "unavailable"
        )
    except (OSError, subprocess.TimeoutExpired):
        result["gpu_snapshot"] = "unavailable"
    return result


def activity(work, proc=Path("/proc")):
    """Read current local evidence, without launching/stopping any process or loading tensors."""
    work = Path(work).resolve()
    jobs = []
    for process in proc.glob("[0-9]*"):
        try:
            argv = (process / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            if str(work) in argv and ("-m cvlab " in argv or ".py " in argv):
                if any(token in argv for token in ("train", "start_", "export_", "evaluate", "predict", "line_search")):
                    jobs.append({"pid": int(process.name), "argv": argv.strip()})
        except (OSError, ProcessLookupError):
            continue
    plans = []
    candidates = [work / "efficient_search/state.json", work / "generalization_training/continuation.json"]
    candidates.extend((work / "full_data").glob("**/state.json"))
    for path in candidates:
        if path.is_file():
            try:
                state = small_json(path)
                plans.append({"path": str(path), "mtime": path.stat().st_mtime,
                              **{k: state[k] for k in ("status", "finished", "source_sha256", "champion_sha256", "submission") if k in state}})
            except (OSError, ValueError):
                continue
    return {"active_processes": jobs, "plans": sorted(plans, key=lambda p: p["mtime"], reverse=True),
            "limits": "Process argv does not establish completed training or distinguish DataLoader workers. Plan status may be stale."}


def audit(work, submission=None, data_root=None):
    work = Path(work)
    prepared = work / "prepared"
    report = {"work": str(work), "memory": memory_status(), "prepared": {}, "runs": []}
    report["activity"] = activity(work)
    for name in ("train.json", "val.json", "test.json", "folds.json"):
        report["prepared"][name] = (prepared / name).is_file()
    folds_path = prepared / "folds.json"
    report["folds_prepared"] = (
        len(small_json(folds_path)["folds"]) if folds_path.is_file() else 0
    )
    for folder in sorted((work / "runs").glob("*")):
        if folder.is_dir():
            report["runs"].append(inspect_run(folder))
    report["models"] = {}
    for model in MODELS:
        measured = [
            r
            for r in report["runs"]
            if r["model"] == model
            and r["validated_result"]
            and not r["limited_training"]
        ]
        report["models"][model] = {
            "measured_runs_including_historical": sum(
                r["model"] == model and r["mAP"] is not None for r in report["runs"]
            ),
            "validated_runs": len(measured),
            "measured_folds": sorted(
                {
                    r["fold"]
                    for r in measured
                    if r["split"] == "fold" and r["fold"] is not None
                }
            ),
            "cv_protocol_consistency": "not_verified",
            "oof_evaluation": "not_verified",
        }
    report["submission"] = {"status": "not_checked"}
    if submission is not None:
        root = Path(data_root) if data_root else work / "dataset"
        try:
            report["submission"] = validate_submission(
                submission, root / "test_images.csv", root / "sample_submission.csv"
            )
        except (ValueError, OSError, KeyError, csv.Error) as error:
            report["submission"] = {"valid": False, "error": str(error)}
    report["manual_checks"] = [
        "Kaggle score exceeds the competition baseline",
        "Full CV/OOF uses consistent protocols",
        "Tuning/loss modifications and failure-case analysis",
        "Final pipeline experiments and report in the course template",
        "GitHub Pages, LMS submission and defense preparation",
    ]
    report["ready_for_grading"] = "manual_review_required"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--submission", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if not args.work.is_dir():
        parser.error("--work directory does not exist")
    report = audit(args.work, args.submission, args.data_root)
    print("RAM available, MiB:", report["memory"].get("available_ram_mib", "unknown"))
    print(
        "GPU index, free MiB, total MiB, utilization %:",
        report["memory"]["gpu_snapshot"],
    )
    print("Prepared files:", report["prepared"])
    print("Current local processes (workers may share argv):", report["activity"]["active_processes"])
    print("Recorded plans (may be stale):", report["activity"]["plans"])
    for run in report["runs"]:
        print(
            f"{run['name']}: mAP={run['mAP']}, validated={run['validated_result']}, "
            f"limited={run['limited_training']}, last.pt bytes={run['checkpoints']['last.pt']}"
        )
    for name, status in report["models"].items():
        print(f"{name}: measured folds {status['measured_folds']}; OOF not verified")
    print("Submission:", report["submission"])
    print("Kaggle baseline and report quality require separate verification.")
    output = args.out or args.work / "checks/status.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    temporary.replace(output)
    print("Saved:", output)
    return 1 if report["submission"].get("valid") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
