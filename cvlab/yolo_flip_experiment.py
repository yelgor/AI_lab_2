"""Full-validation A/B and export for horizontal-flip YOLO plus the frozen c1 route."""

import csv
import gc
import tempfile
import time
from pathlib import Path

from .class_ensemble import (
    C1_SHA,
    FAMILY_REVIEW_REASONS,
    compatible_pair,
    evaluator_identity,
    metadata,
    records_for,
    route_frame,
    sha,
    stream_frames,
)
from .common import digest, read_json, write_json
from .selection_policy import assess_metrics
from .tiled_inference import merge_frame

YOLO_SHA = "42e4c916c45a262d1e29863dc7edb919679528fa673241eef4641a25606285a8"
PARENT_CSV_SHA = "1d4ac45ca8f1b121d40c65ddea376a498b845ab1249cd0f00063eb2f8ddf2562"
PARENT_PROTOCOL = "fixed-truck-bus-route-v1"


def _validate_flip_cache(flip_path, yolo_meta, records, split):
    flip = metadata(flip_path)
    identity = flip["identity"]
    native = yolo_meta["identity"]
    pipeline = dict(identity["pipeline"])
    tta = pipeline.pop("tta", None)
    if (
        identity["split"] != split
        or identity["checkpoint_sha256"] != YOLO_SHA
        or native["checkpoint_sha256"] != YOLO_SHA
        or identity["data_fingerprint"] != digest(records)
        or identity["training_fingerprint"] != native["training_fingerprint"]
        or flip["frames"] != len(records)
        or identity["pipeline"]["backend"] != "yolo"
        or pipeline != native["pipeline"]
        or not tta
        or tta["adapter"] != "yolo-horizontal-flip-only-canvas-unflip-v1"
        or tta["nms_iou"] != 0.6
    ):
        raise ValueError("Horizontal-flip cache is not compatible with frozen YOLO")
    return flip


def _iter_routed(c1_path, yolo_path, flip_path, records):
    for c1, native, flipped in zip(
        stream_frames(c1_path, records),
        stream_frames(yolo_path, records),
        stream_frames(flip_path, records),
        strict=True,
    ):
        # Each view has already had its own class-aware NMS. Merge across views
        # on CPU, again class-aware, then apply the class route and global cap.
        tta = merge_frame(native + flipped, 0.6, 500)
        yield tta, route_frame(c1, tta, "class_route")


def _manual_review_eligible(decision):
    return (
        decision.get("pooled_gain", -1) >= 0.001
        and set(decision.get("reasons", [])) <= FAMILY_REVIEW_REASONS
    )


def compare_flip(
    c1_path,
    yolo_path,
    flip_path,
    parent_report_path,
    parent_csv_path,
    config,
    output,
):
    """Compare one TTA route against the exact current public-best route report."""
    from .metrics import evaluate
    from .sequential_policy import per_video_scores

    c1_path, yolo_path, flip_path = map(Path, (c1_path, yolo_path, flip_path))
    parent_report_path, parent_csv_path, output = map(
        Path, (parent_report_path, parent_csv_path, output)
    )
    if output.exists():
        raise ValueError("Preserve existing A/B report; choose a new output path")
    parent = read_json(parent_report_path)
    if (
        parent.get("protocol") != PARENT_PROTOCOL
        or parent.get("records") != 2846
        or parent.get("route", {}).get("truck") != "c1"
        or parent.get("route", {}).get("bus") != "c1"
        or parent.get("accepted") is not False
        or parent.get("review_eligible") is not True
    ):
        raise ValueError(
            "Provided comparison is not the recorded parent public-best route"
        )
    if sha(parent_csv_path) != PARENT_CSV_SHA:
        raise ValueError("Public-best control CSV SHA differs")
    parent_meta = read_json(parent_csv_path.with_suffix(".csv.metadata.json"))
    if (
        parent_meta.get("validation", {}).get("sha256") != PARENT_CSV_SHA
        or parent_meta.get("strict_promotion_accepted") is not False
        or parent_meta.get("inputs", [])[0]["identity"]["checkpoint_sha256"] != C1_SHA
        or parent_meta.get("inputs", [])[1]["identity"]["checkpoint_sha256"] != YOLO_SHA
    ):
        raise ValueError("Public-best CSV metadata does not match its frozen sources")

    records = records_for(config, "val")
    if len(records) != 2846 or digest(records) != parent.get("data_fingerprint"):
        raise ValueError(
            "Current official-val cohort differs from the public-best report"
        )
    native_inputs = compatible_pair(c1_path, yolo_path, records, "val")
    if parent.get("inputs") != native_inputs:
        raise ValueError("Native cache identities differ from the exact parent report")
    flip_meta = _validate_flip_cache(flip_path, native_inputs[1], records, "val")

    baseline_metrics = parent["cases"]["class_route"]["metrics"]
    if output.exists():
        raise ValueError("A/B report output already exists")
    candidate_predictions = []
    comparison_started = time.perf_counter()
    for _, routed in _iter_routed(c1_path, yolo_path, flip_path, records):
        candidate_predictions.extend(routed)
    candidate_metrics = evaluate(records, candidate_predictions, [100, 500])
    candidate_metrics["per_video_mAP"] = per_video_scores(
        records, candidate_predictions
    )
    del candidate_predictions
    gc.collect()

    decision = assess_metrics(baseline_metrics, candidate_metrics)
    review_eligible = decision["accepted"] or _manual_review_eligible(decision)
    report = {
        "protocol": "public-best-route-vs-yolo-horizontal-flip-v1",
        "parent_protocol": parent["protocol"],
        "parent_comparison_sha256": sha(parent_report_path),
        "parent_csv_sha256": PARENT_CSV_SHA,
        "records": len(records),
        "data_fingerprint": digest(records),
        "evaluator_identity": evaluator_identity(),
        "route": parent["route"],
        "inference": {
            "image_size": 1280,
            "yolo_views": ["native_reused", "horizontal_flip_new"],
            "flip_box_transform": "canvas_width_minus_x2_x1_before_geometry_restore",
            "class_aware_nms_iou": 0.6,
            "global_max_detections": 500,
        },
        "inputs": {
            "c1": native_inputs[0],
            "yolo_native": native_inputs[1],
            "yolo_horizontal_flip": flip_meta,
        },
        "cases": {
            "public_best_route": {"metrics": baseline_metrics},
            "flip_route": {"metrics": candidate_metrics},
        },
        "decision_vs_public_best_route": decision,
        "accepted": decision["accepted"],
        "review_eligible": review_eligible,
        "status": (
            "FULL_VALIDATION_ACCEPTED"
            if decision["accepted"]
            else "FAMILY_UNCERTAINTY_MANUAL_REVIEW"
            if review_eligible
            else "REJECTED_KEEP_PUBLIC_BEST"
        ),
        "control_public_score": 0.2176009680,
        "comparison_seconds": time.perf_counter() - comparison_started,
        "public_score": None,
        "limitation": (
            "Repeatedly selected official validation is not new independent "
            "CV/OOF or a Kaggle-score forecast."
        ),
    }
    write_json(output, report)
    return report


def export_flip(
    val_c1_path,
    val_yolo_path,
    val_flip_path,
    test_c1_path,
    test_yolo_path,
    test_flip_path,
    parent_report_path,
    report_path,
    config,
    output,
    *,
    manual_review=False,
):
    """Export the eligible flip route using exact frozen val/test pipeline identities."""
    from .submission import write_submission
    from scripts.check_lab import validate_submission

    report = read_json(report_path)
    parent = read_json(parent_report_path)
    validation_records = records_for(config, "val")
    candidate_metrics = report.get("cases", {}).get("flip_route", {}).get("metrics", {})
    decision = assess_metrics(
        parent.get("cases", {}).get("class_route", {}).get("metrics", {}),
        candidate_metrics,
    )
    review_eligible = decision["accepted"] or _manual_review_eligible(decision)
    if (
        report.get("protocol") != "public-best-route-vs-yolo-horizontal-flip-v1"
        or not (
            report.get("accepted") or manual_review and report.get("review_eligible")
        )
        or report.get("records") != 2846
        or report.get("data_fingerprint") != digest(validation_records)
        or parent.get("records") != 2846
        or report.get("route") != parent.get("route")
        or report.get("cases", {}).get("public_best_route", {}).get("metrics")
        != parent.get("cases", {}).get("class_route", {}).get("metrics")
        or report.get("decision_vs_public_best_route") != decision
        or report.get("accepted") != decision["accepted"]
        or report.get("review_eligible") != review_eligible
        or report.get("parent_comparison_sha256") != sha(parent_report_path)
        or report.get("parent_csv_sha256") != PARENT_CSV_SHA
        or report.get("evaluator_identity") != evaluator_identity()
    ):
        raise ValueError("No eligible A/B report for manual-review export")
    if output.exists():
        raise ValueError("Preserve an existing CSV; choose a new output path")

    val_records = records_for(config, "val")
    val_c1, val_yolo = compatible_pair(val_c1_path, val_yolo_path, val_records, "val")
    val_flip = _validate_flip_cache(val_flip_path, val_yolo, val_records, "val")
    recorded = report["inputs"]
    if (
        recorded["c1"] != val_c1
        or recorded["yolo_native"] != val_yolo
        or recorded["yolo_horizontal_flip"] != val_flip
    ):
        raise ValueError("A/B report validation cache identities changed")

    records = records_for(config, "test")
    # Test caches live separately, and c1 remains byte-for-byte the validated pipeline.
    test_c1_meta, test_yolo_meta = compatible_pair(
        test_c1_path, test_yolo_path, records, "test"
    )
    test_flip = _validate_flip_cache(test_flip_path, test_yolo_meta, records, "test")
    for key, val_meta, test_meta in (
        ("c1", val_c1, test_c1_meta),
        ("yolo_native", val_yolo, test_yolo_meta),
        ("yolo_horizontal_flip", val_flip, test_flip),
    ):
        expected, actual = val_meta["identity"], test_meta["identity"]
        if any(
            expected[k] != actual[k]
            for k in ("checkpoint_sha256", "pipeline", "training_fingerprint")
        ):
            raise ValueError(f"Val/test pipeline differs for {key}")
    if (
        test_c1_meta["identity"]["checkpoint_sha256"] != C1_SHA
        or test_yolo_meta["identity"]["checkpoint_sha256"] != YOLO_SHA
    ):
        raise ValueError("Test caches use different frozen checkpoint sources")
    if test_flip["identity"]["data_fingerprint"] != digest(records):
        raise ValueError("Flipped test cache cohort differs from official test records")

    root = Path(config["data"]["root"])
    sample = root / "sample_submission.csv"
    if sample.is_file():
        with sample.open(newline="", encoding="utf-8-sig") as stream:
            order = [row["image_id"] for row in csv.DictReader(stream)]
    else:
        with (root / "test_images.csv").open(
            newline="", encoding="utf-8-sig"
        ) as stream:
            order = [row["image_id"] for row in csv.DictReader(stream)]
    if len(order) != len(set(order)) or set(order) != {
        r["submission_id"] for r in records
    }:
        raise ValueError("Test template IDs differ from prediction caches")

    output.parent.mkdir(parents=True, exist_ok=True)
    parts = {}
    with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
        for record, (_, routed) in zip(
            records,
            _iter_routed(test_c1_path, test_yolo_path, test_flip_path, records),
            strict=True,
        ):
            part = Path(scratch) / f"{record['id']}.csv"
            write_submission([record], routed, None, part, max_detections=500)
            parts[record["submission_id"]] = part
        temporary = output.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image_id", "PredictionString"])
            for image_id in order:
                with parts[image_id].open(newline="", encoding="utf-8") as source:
                    rows = csv.reader(source)
                    next(rows)
                    writer.writerow(next(rows))
    checked = validate_submission(
        temporary,
        root / "test_images.csv",
        sample if sample.is_file() else None,
    )
    if (
        not checked["valid"]
        or not checked["matches_template_order"]
        or checked["all_predictions_empty"]
    ):
        temporary.unlink(missing_ok=True)
        raise ValueError("Flip-route CSV failed structural checks")
    from .common import write_json

    meta = {
        "protocol": report["protocol"],
        "comparison_report_sha256": sha(report_path),
        "parent_comparison_sha256": report["parent_comparison_sha256"],
        "inputs": {
            "c1_test": test_c1_meta,
            "yolo_native_test": test_yolo_meta,
            "yolo_flip_test": test_flip,
        },
        "validation": checked,
        "strict_promotion_accepted": report["accepted"],
        "manual_review": manual_review,
        "family_decision": report["decision_vs_public_best_route"],
        "public_score": None,
        "status": "READY_FOR_MANUAL_SUBMISSION_REVIEW",
        "data_fingerprint": digest(records),
    }
    write_json(output.with_suffix(".csv.metadata.json"), meta)
    temporary.replace(output)
    return checked
