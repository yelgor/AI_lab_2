"""Validate full-frame/NMS/tiling ablations before exporting a new candidate."""

import copy
import fcntl
import math
import os
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from .common import digest, read_json, write_json
from .inference_experiments import balanced_subset
from .score_improvement import freeze_run, full_validation, release_gpu
from .tiled_inference import TileOptions, export_tiled, merge_frame, predict_tiled


def choose(cases):
    def key(case):
        cost = {"global": 0, "global_nms": 1, "tiles": 2}[case["strategy"]]
        return (
            case["metrics"]["coco100"]["mAP"],
            -cost,
            (case.get("options") or {}).get("tile_side") or 10**9,
        )

    valid = [
        case
        for case in cases
        if isinstance(case["metrics"]["coco100"]["mAP"], (int, float))
        and math.isfinite(case["metrics"]["coco100"]["mAP"])
    ]
    if not valid:
        raise ValueError("No evaluable tiling cases")
    return max(valid, key=key)


def comparison_entry(strategy, options, scores, timing=None):
    return {
        "strategy": strategy,
        "options": asdict(options) if options else None,
        "metrics": scores,
        "timing": timing,
    }


def validate_cache(value, sha, records):
    if (
        value["checkpoint_sha256"] != sha
        or value["data_fingerprint"] != digest(records)
        or value["record_ids"] != [r["id"] for r in records]
    ):
        raise ValueError("Tiling cache checkpoint/data differs")


def selected_size(work, name, sha, native):
    path = work / "score_improvement/summary.json"
    if path.is_file():
        for entry in read_json(path)["models"]:
            if entry["model"] == name and entry["checkpoint_sha256"] == sha:
                return entry["selected_image_size"]
    return native


def run(work, models, sides, ious, max_images, vram_gib, workers, lock_fd):
    import torch
    from .engine import load_trained, predict, setup
    from .metrics import evaluate
    from .prepare import select_records
    from .submit_cpu import _available_ram_mib, _cuda_budget

    if (
        not models
        or len(set(models)) != len(models)
        or max_images < 1
        or set(models) - {"rtdetr", "fasterrcnn", "fcos", "retinanet"}
    ):
        raise ValueError("Unique models and positive validation subset required")
    if not sides or not ious or any(not 0 < iou <= 1 for iou in ious):
        raise ValueError("Valid NMS IoUs required")
    for side in sides:
        TileOptions(side, nms_iou=ious[0], workers=workers).validate()
    work = Path(work).resolve()
    expected, actual = (work / ".local_gpu_queue.lock").stat(), os.fstat(lock_fd)
    if (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino):
        raise ValueError("GPU lease belongs to another work directory")
    with os.fdopen(lock_fd, "a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = {
            "validation_split": "official",
            "target_kaggle_score": 0.2216324580,
            "models": [],
        }
        for name in models:
            source = work / "runs" / f"{name}_fold0_local"
            frozen, config, sha, seen = freeze_run(source, work / "score_improvement")
            config = copy.deepcopy(config)
            config["model"]["image_size"] = selected_size(
                work, name, sha, config["model"]["image_size"]
            )
            size = config["model"]["image_size"]
            records = select_records(config, "val")
            if seen & {r["sequence_id"] for r in records}:
                raise ValueError("Official validation contains training sequences")
            subset = balanced_subset(records, max_images)
            directory = frozen / "tiles"
            directory.mkdir(exist_ok=True)
            settings = {
                "sides": sides,
                "ious": ious,
                "image_size": size,
                "workers": workers,
            }
            path = directory / f"subset_{digest([settings, subset])[:12]}.json"
            entry = {
                "model": name,
                "checkpoint_sha256": sha,
                "image_size": size,
                "subset_images": len(subset),
                "settings": settings,
                "record_ids": [r["id"] for r in subset],
                "data_fingerprint": digest(subset),
                "cases": [],
                "subset_complete": False,
            }
            print(
                f"TILING SEARCH model={name}; input={size}; checkpoint={sha[:12]}",
                flush=True,
            )
            model = None

            def get_model():
                nonlocal model
                if model is None:
                    release_gpu()
                    device = setup(config)
                    _cuda_budget(device, vram_gib)
                    model = load_trained(frozen / "best.pt", config, device)
                return model

            device = torch.device(config["runtime"]["device"])
            if path.is_file() and read_json(path).get("subset_complete"):
                entry = read_json(path)
                validate_cache(entry, sha, subset)
            else:
                raw, timing = predict(get_model(), config, subset, device)
                scores = evaluate(subset, raw, config["evaluation"]["max_detections"])
                entry["cases"].append(comparison_entry("global", None, scores, timing))
                by_image = defaultdict(list)
                for prediction in raw:
                    by_image[prediction["image_id"]].append(prediction)
                for iou in ious:
                    options = TileOptions(None, nms_iou=iou, workers=workers)
                    merged = [
                        p
                        for r in subset
                        for p in merge_frame(
                            by_image[r["id"]], iou, config["model"]["max_detections"]
                        )
                    ]
                    scores = evaluate(
                        subset, merged, config["evaluation"]["max_detections"]
                    )
                    entry["cases"].append(
                        comparison_entry("global_nms", options, scores)
                    )
                del raw, by_image, merged
                for side in sides:
                    options = TileOptions(side, workers=workers)
                    outputs, timing = predict_tiled(
                        get_model(), config, subset, device, options, ious
                    )
                    for iou, predictions in outputs.items():
                        options = TileOptions(side, nms_iou=iou, workers=workers)
                        scores = evaluate(
                            subset, predictions, config["evaluation"]["max_detections"]
                        )
                        entry["cases"].append(
                            comparison_entry("tiles", options, scores, timing)
                        )
                        print(
                            f"{name}: tile={side} NMS={iou} subset mAP={scores['coco100']['mAP']}",
                            flush=True,
                        )
                    del outputs, predictions
                    write_json(path, entry)
                entry["subset_complete"] = True
                write_json(path, entry)
            winner = choose(entry["cases"])
            entry["subset_winner"] = winner
            model = None
            release_gpu()
            if winner["strategy"] == "global":
                entry["outcome"] = (
                    "No subset improvement; keeping existing global prediction"
                )
            else:
                available = _available_ram_mib()
                if available is not None and available < 4096:
                    raise RuntimeError(
                        "Need 4 GiB available RAM for full validation; close heavy applications"
                    )
                baseline_path = frozen / f"official_full_{size}.json"
                if baseline_path.is_file():
                    baseline = read_json(baseline_path)
                    if baseline["data_fingerprint"] != digest(records) or baseline[
                        "record_ids"
                    ] != [r["id"] for r in records]:
                        raise ValueError("Global full-validation cache differs")
                else:
                    baseline = full_validation(
                        config, frozen / "best.pt", size, records
                    )
                    write_json(baseline_path, baseline)
                options = TileOptions(**winner["options"])
                signature = digest([asdict(options), size, digest(records)])[:12]
                candidate_path = directory / f"full_{signature}.json"
                if candidate_path.is_file():
                    candidate = read_json(candidate_path)
                    validate_cache(candidate, sha, records)
                else:
                    outputs, timing = predict_tiled(
                        get_model(), config, records, device, options
                    )
                    scores = evaluate(
                        records,
                        outputs[options.nms_iou],
                        config["evaluation"]["max_detections"],
                    )
                    candidate = comparison_entry(
                        winner["strategy"], options, scores, timing
                    )
                    candidate.update(
                        checkpoint_sha256=sha,
                        record_ids=[r["id"] for r in records],
                        data_fingerprint=digest(records),
                    )
                    del outputs
                    write_json(candidate_path, candidate)
                entry["full_global"] = baseline
                entry["full_candidate"] = candidate
                if (
                    candidate["metrics"]["coco100"]["mAP"]
                    > baseline["metrics"]["coco100"]["mAP"] + 1e-5
                ):
                    destination = (
                        work
                        / "submissions"
                        / f"{name}_{sha[:12]}_{size}_tiles_{signature}.csv"
                    )
                    export_tiled(
                        get_model(),
                        config,
                        select_records(config, "test"),
                        device,
                        options,
                        destination,
                        sha,
                    )
                    entry.update(
                        outcome="Full-validation improvement; CSV ready",
                        submission=str(destination),
                    )
                else:
                    entry["outcome"] = (
                        "Subset improvement did not generalize; keeping existing global prediction"
                    )
            model = None
            release_gpu()
            report["models"].append(entry)
            write_json(work / "score_improvement/tiling_summary.json", report)
            print(f"{name}: {entry['outcome']}", flush=True)
        return report
