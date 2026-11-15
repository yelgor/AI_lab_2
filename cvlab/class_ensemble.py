"""A predeclared two-detector class route, with reusable full-frame prediction caches."""

from dataclasses import asdict
import gc
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import tempfile
import time

from .common import CLASSES, digest, read_json, write_json
from .prepare import select_records
from .sequential_policy import assert_family_isolation, per_video_scores
from .selection_policy import assess_metrics
from .tiled_inference import TileOptions, frame_predictions, merge_frame

C1_SHA = "c1d7fcc83dde045578514e64c85e04b215fca988c804985f1a5a313ebe6daab4"
ROUTE = {CLASSES.index(name) + 1 for name in ("truck", "bus")}
CASES = ("c1", "yolo", "class_route")
FAMILY_REVIEW_REASONS = {
    "family_lower_bound_not_positive", "too_few_improving_families",
    "family_regression", "gain_depends_on_one_family",
}


def review_eligible(decisions):
    """Keep pooled/class/size/evidence guards; family uncertainty needs manual review."""
    return set(decisions) == {"c1", "yolo"} and all(
        d.get("pooled_gain", -1) >= 0.001
        and set(d.get("reasons", [])) <= FAMILY_REVIEW_REASONS
        for d in decisions.values()
    )


def routing():
    return {name: "c1" if i+1 in ROUTE else "yolo" for i, name in enumerate(CLASSES)}


def evaluator_identity():
    return digest([(name, (Path(__file__).parent / name).read_text())
                   for name in ("metrics.py", "sequential_policy.py", "selection_policy.py")])


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def records_for(config, split):
    if config["data"]["split"] != "official":
        raise ValueError("Require official train/val with full held-out coverage")
    records = select_records(config, split)
    if split == "val":
        assert_family_isolation(select_records(config, "train"), records)
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate manifest IDs")
    return records


def snapshot(source, directory):
    """Copy an immutable source before waiting for the GPU; live best may advance."""
    source, directory = Path(source), Path(directory)
    expected = sha(source)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{expected}.pt"
    if not target.exists():
        with tempfile.NamedTemporaryFile(
            dir=directory, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
        try:
            shutil.copyfile(source, temporary)
            if sha(temporary) != expected or sha(source) != expected:
                raise ValueError(
                    "Source changed during snapshot; retry at its saved boundary"
                )
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    if sha(target) != expected:
        raise ValueError("Frozen checkpoint checksum differs")
    return target


def pipeline_identity(backend, config, options):
    return {
        "backend": backend,
        "model": config["model"],
        "runtime": config["runtime"],
        "seed": config["seed"],
        "options": asdict(options),
        "classes": CLASSES,
        "implementation": digest(
            [
                (name, (Path(__file__).parent / name).read_text())
                for name in (
                    "engine.py",
                    "common.py",
                    "models.py",
                    "yolo_backend.py",
                    "data.py",
                    "tiled_inference.py",
                )
            ]
        ),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "torchvision",
                "numpy",
                "Pillow",
                "ultralytics" if backend == "yolo" else "transformers",
            )
        },
    }


def metadata(directory):
    directory = Path(directory)
    value = read_json(directory / "cache.json")
    if value.get("version") != 1 or value.get("classes") != CLASSES:
        raise ValueError("Unsupported prediction cache/classes")
    if sha(directory / "frames.jsonl.gz") != value["predictions_sha256"]:
        raise ValueError("Prediction cache checksum differs")
    return value


def stream_frames(directory, records):
    """Require one explicit row per frame, including frames with no detections."""
    with gzip.open(Path(directory) / "frames.jsonl.gz", "rt") as stream:
        for record in records:
            line = stream.readline()
            if not line:
                raise ValueError("Incomplete prediction cache")
            frame = json.loads(line)
            if frame["image_id"] != record["id"]:
                raise ValueError("Prediction IDs/order differ from manifest")
            predictions = frame["predictions"]
            if len(predictions) > 500:
                raise ValueError("Prediction cache exceeds the per-frame cap")
            for p in predictions:
                values = [p["score"], *p["bbox"]]
                if (
                    p["image_id"] != record["id"]
                    or len(values) != 5
                    or not all(math.isfinite(v) for v in values)
                    or not 1 <= p["category_id"] <= 8
                    or not 0 <= p["score"] <= 1
                ):
                    raise ValueError("Invalid cached prediction")
                x, y, w, h = p["bbox"]
                if (
                    x < -1e-3
                    or y < -1e-3
                    or w <= 0
                    or h <= 0
                    or x + w > record["width"] + 1e-3
                    or y + h > record["height"] + 1e-3
                ):
                    raise ValueError("Cached box lies outside the original frame")
            yield predictions
        if stream.readline():
            raise ValueError("Unexpected extra prediction frames")


def cache_predictions(
    checkpoint,
    config,
    directory,
    backend,
    split="val",
    tile_side=None,
    *,
    existing_only=False,
):
    """GPU lease belongs to the caller; each checkpoint/cohort is inferred once."""
    import torch
    from .engine import load_trained, setup

    checkpoint, directory = Path(checkpoint), Path(directory)
    if backend not in ("c1", "yolo"):
        raise ValueError("Backend must be c1 or yolo")
    if backend == "c1" and sha(checkpoint) != C1_SHA:
        raise ValueError("Control must be the preserved public-best c1 checkpoint")
    records = records_for(config, split)
    options = TileOptions(
        tile_side,
        overlap=0.2,
        nms_iou=0.6,
        edge_margin=2,
        workers=config["loader"]["workers"],
    )
    expected = {
        "checkpoint_sha256": sha(checkpoint),
        "data_fingerprint": digest(records),
        "training_fingerprint": digest(select_records(config, "train")),
        "pipeline": pipeline_identity(backend, config, options),
        "split": split,
    }
    if (directory / "cache.json").exists():
        old = metadata(directory)
        if old["identity"] != expected:
            raise ValueError(
                "Cache belongs to a different checkpoint/cohort/pipeline; use a new directory"
            )
        print(f"PREDICTION CACHE HIT: {directory}", flush=True)
        return old
    if existing_only:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    device = setup(config)
    if backend == "yolo":
        from ultralytics import YOLO
        from .yolo_backend import Predictor

        model = Predictor(YOLO(str(checkpoint)).model.to(device), config)
    else:
        model = load_trained(checkpoint, config, device)
    temporary = directory / "frames.jsonl.gz.tmp"
    started = time.perf_counter()
    with gzip.open(temporary, "wt", compresslevel=1) as stream:
        for count, (record, raw, views) in enumerate(
            frame_predictions(model, config, records, device, options), 1
        ):
            predictions = merge_frame(raw, options.nms_iou, 500)
            stream.write(
                json.dumps(
                    {"image_id": record["id"], "predictions": predictions},
                    allow_nan=False,
                )
                + "\n"
            )
            if count % 64 == 0 or count == len(records):
                print(
                    f"CACHE {backend} {split}={count}/{len(records)}; "
                    f"{count / max(time.perf_counter() - started, 1e-6):.2f} frames/s",
                    flush=True,
                )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    temporary.replace(directory / "frames.jsonl.gz")
    result = {
        "version": 1,
        "classes": CLASSES,
        "identity": expected,
        "frames": len(records),
        "predictions_sha256": sha(directory / "frames.jsonl.gz"),
        "seconds": time.perf_counter() - started,
    }
    write_json(directory / "cache.json", result)
    return result


def compatible_pair(c1, yolo, records, split):
    values = [metadata(path) for path in (c1, yolo)]
    for value, backend in zip(values, ("c1", "yolo"), strict=True):
        ident = value["identity"]
        if (
            ident["split"] != split
            or ident["data_fingerprint"] != digest(records)
            or value["frames"] != len(records)
            or ident["pipeline"]["backend"] != backend
        ):
            raise ValueError(
                "Prediction cache does not match the complete cohort/backend"
            )
        if backend == "c1" and ident["checkpoint_sha256"] != C1_SHA:
            raise ValueError(
                "Control cache does not belong to preserved public-best c1"
            )
    if (
        values[0]["identity"]["training_fingerprint"]
        != values[1]["identity"]["training_fingerprint"]
    ):
        raise ValueError("The detectors must use the same declared training manifest")
    return values


def route_frame(c1, yolo, case):
    if case == "c1":
        return c1
    if case == "yolo":
        return yolo
    if case != "class_route":
        raise ValueError("Unknown predeclared ensemble case")
    selected = [p for p in c1 if p["category_id"] in ROUTE]
    selected += [p for p in yolo if p["category_id"] not in ROUTE]
    # Classes come from exactly one detector; no cross-model score averaging,
    # extra NMS or box-coordinate interpolation. Only the common global cap applies.
    return sorted(selected, key=lambda p: p["score"], reverse=True)[:500]


def compare(c1, yolo, config, output):
    from .metrics import evaluate

    records = records_for(config, "val")
    inputs = compatible_pair(c1, yolo, records, "val")
    report = {
        "protocol": "fixed-truck-bus-route-v1",
        "records": len(records),
        "data_fingerprint": digest(records),
        "inputs": inputs,
        "route": routing(),
        "evaluator_identity": evaluator_identity(),
        "cases": {},
        "public_score": None,
        "accepted": False,
        "limitation": "Official val has been used for repeated model selection; this is not new independent CV/OOF.",
    }
    if Path(output).exists():
        prior = read_json(output)
        for key in ("protocol", "data_fingerprint", "inputs", "route", "evaluator_identity"):
            if prior.get(key) != report[key]:
                raise ValueError("Existing comparison has a different identity; preserve it and use a new report")
        report = prior
    for case in CASES:
        if case in report["cases"]:
            print(f"CPU METRICS CACHE HIT: {case}", flush=True)
            continue
        predictions = []
        for left, right in zip(
            stream_frames(c1, records), stream_frames(yolo, records), strict=True
        ):
            predictions.extend(route_frame(left, right, case))
        scores = evaluate(records, predictions, [100, 500])
        scores["per_video_mAP"] = per_video_scores(records, predictions)
        report["cases"][case] = {"metrics": scores, "detections": len(predictions)}
        del predictions
        gc.collect()
        write_json(output, report)
        print(
            f"FULL VAL {case}: AP={scores['coco100']['mAP']}; small={scores['coco100']['AP_small']}",
            flush=True,
        )
    candidate = report["cases"]["class_route"]["metrics"]
    report["decisions"] = {
        case: assess_metrics(report["cases"][case]["metrics"], candidate)
        for case in ("c1", "yolo")
    }
    report["accepted"] = all(d["accepted"] for d in report["decisions"].values())
    report["review_eligible"] = review_eligible(report["decisions"])
    report["status"] = (
        "FULL_VALIDATION_ACCEPTED"
        if report["accepted"]
        else "FAMILY_UNCERTAINTY_MANUAL_REVIEW" if report["review_eligible"] else "REJECTED_KEEP_SINGLE_MODELS"
    )
    write_json(output, report)
    return report


def export(c1, yolo, config, report_path, output, *, manual_review=False):
    """CPU-only, streaming test export with identities matched to the accepted val pair."""
    from .submission import write_submission
    from scripts.check_lab import validate_submission
    import csv

    report = read_json(report_path)
    if (
        report.get("protocol") != "fixed-truck-bus-route-v1"
        or not (report.get("accepted") or manual_review and report.get("review_eligible"))
        or set(report.get("decisions", {})) != {"c1", "yolo"}
    ):
        raise ValueError(
            "No accepted full-validation comparison; ensemble export refused"
        )
    actual = {
        case: assess_metrics(
            report["cases"][case]["metrics"], report["cases"]["class_route"]["metrics"]
        )
        for case in ("c1", "yolo")
    }
    if (actual != report["decisions"] or not (
            all(d["accepted"] for d in actual.values()) or manual_review and review_eligible(actual))):
        raise ValueError("Selection decisions do not match the recorded full metrics")
    if report.get("route") != routing() or report.get("evaluator_identity") != evaluator_identity():
        raise ValueError("Routing/evaluator changed since full validation")
    validation = records_for(config, "val")
    if report["data_fingerprint"] != digest(validation) or report["records"] != len(
        validation
    ):
        raise ValueError("Accepted validation cohort changed")
    records = records_for(config, "test")
    inputs = compatible_pair(c1, yolo, records, "test")
    for expected, measured in zip(report["inputs"], inputs, strict=True):
        a, b = expected["identity"], measured["identity"]
        if any(
            a[key] != b[key]
            for key in ("checkpoint_sha256", "pipeline", "training_fingerprint")
        ):
            raise ValueError(
                "Test weights/inference differ from the validated pipeline"
            )
    output = Path(output)
    if output.exists():
        raise ValueError("Preserve an existing export; use a new destination")
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(config["data"]["root"])
    sample_path = root / "sample_submission.csv"
    if sample_path.is_file():
        with sample_path.open(newline="") as stream:
            order = [r["image_id"] for r in csv.DictReader(stream)]
    else:
        # Some Kaggle data packages provide only test_images.csv. Its order is
        # the canonical template order in that case.
        with (root / "test_images.csv").open(newline="") as stream:
            order = [r["image_id"] for r in csv.DictReader(stream)]
    if len(order) != len(set(order)) or set(order) != {
        r["submission_id"] for r in records
    }:
        raise ValueError("Test/template IDs differ")
    # Tiny per-frame CSV parts keep memory bounded and allow arbitrary template order.
    temporary = output.with_suffix(".csv.tmp")
    with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
        parts = {}
        for record, left, right in zip(
            records,
            stream_frames(c1, records),
            stream_frames(yolo, records),
            strict=True,
        ):
            part = Path(scratch) / f"{record['id']}.csv"
            write_submission(
                [record], route_frame(left, right, "class_route"), None, part
            )
            parts[record["submission_id"]] = part
        with temporary.open("w", newline="") as destination:
            writer = csv.writer(destination)
            writer.writerow(["image_id", "PredictionString"])
            for key in order:
                with parts[key].open(newline="") as stream:
                    rows = csv.reader(stream)
                    next(rows)
                    writer.writerow(next(rows))
    checked = validate_submission(
        temporary, root / "test_images.csv", sample_path if sample_path.is_file() else None
    )
    if (
        not checked["valid"]
        or not checked["matches_template_order"]
        or checked["all_predictions_empty"]
    ):
        raise ValueError("Ensemble CSV validation failed")
    write_json(
        output.with_suffix(".csv.metadata.json"),
        {
            "protocol": report["protocol"],
            "selection_report_sha256": sha(report_path),
            "inputs": inputs,
            "validation": checked,
            "public_score": None,
            "status": "READY_FOR_MANUAL_SUBMISSION_REVIEW",
            "strict_promotion_accepted": report["accepted"],
            "manual_review": manual_review,
            "family_diagnostics": report["decisions"],
            "data_fingerprint": digest(records),
        },
    )
    temporary.replace(output)
    print(f"ENSEMBLE READY FOR MANUAL REVIEW: {output}", flush=True)
    return checked
