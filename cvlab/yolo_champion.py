"""Compare later YOLO checkpoints with the scored flip-TTA champion."""

import csv
import gc
import tempfile
import time
from pathlib import Path

from .class_ensemble import (
    C1_SHA,
    FAMILY_REVIEW_REASONS,
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

CHAMPION_PROTOCOL = "public-best-route-vs-yolo-horizontal-flip-v1"
CHAMPION_SCORE = 0.2237335225
CHAMPION_CSV_SHA = "2dee65d0f257ba61569d48ba94ad95e74e740802b7e430cf10556a1fe2ebdce0"
FLIP_ADAPTER = "yolo-horizontal-flip-only-canvas-unflip-v1"
CHAMPION_REPORT_SHA = "19d37b024c47257e8afd819ac163deeaea0544d360b89680323ec7655fafba54"


def _parent(report_path, csv_path):
    report_path, csv_path = Path(report_path), Path(csv_path)
    report = read_json(report_path)
    if (
        report.get("protocol") != CHAMPION_PROTOCOL
        or sha(report_path) != CHAMPION_REPORT_SHA
        or report.get("records") != 2846
        or report.get("accepted") is not True
        or report.get("status") != "FULL_VALIDATION_ACCEPTED"
        or report.get("parent_csv_sha256")
        != "1d4ac45ca8f1b121d40c65ddea376a498b845ab1249cd0f00063eb2f8ddf2562"
    ):
        raise ValueError("Comparison is not the recorded champion report")
    if sha(csv_path) != CHAMPION_CSV_SHA:
        raise ValueError("CSV is not the measured public-best champion")
    meta = read_json(csv_path.with_suffix(".csv.metadata.json"))
    if (
        meta.get("validation", {}).get("sha256") != CHAMPION_CSV_SHA
        or meta.get("comparison_report_sha256") != sha(report_path)
        or meta.get("strict_promotion_accepted") is not True
        or meta.get("public_score") is not None
    ):
        raise ValueError("Champion metadata does not match report and CSV")
    inputs = report.get("inputs", {})
    if (
        inputs.get("c1", {}).get("identity", {}).get("checkpoint_sha256") != C1_SHA
        or inputs.get("yolo_native", {}).get("identity", {}).get("checkpoint_sha256")
        != "42e4c916c45a262d1e29863dc7edb919679528fa673241eef4641a25606285a8"
        or inputs.get("yolo_horizontal_flip", {})
        .get("identity", {})
        .get("checkpoint_sha256")
        != "42e4c916c45a262d1e29863dc7edb919679528fa673241eef4641a25606285a8"
    ):
        raise ValueError("Champion report has different frozen model sources")
    return report, meta


def _yolo_views(native_path, flip_path, records, split):
    native, flip = metadata(native_path), metadata(flip_path)
    n_id, f_id = native["identity"], flip["identity"]
    pipeline = dict(f_id["pipeline"])
    tta = pipeline.pop("tta", None)
    if (
        native["frames"] != len(records)
        or flip["frames"] != len(records)
        or any(x["split"] != split for x in (n_id, f_id))
        or any(x["data_fingerprint"] != digest(records) for x in (n_id, f_id))
        or n_id["pipeline"]["backend"] != "yolo"
        or f_id["pipeline"]["backend"] != "yolo"
        or n_id["checkpoint_sha256"] != f_id["checkpoint_sha256"]
        or n_id["training_fingerprint"] != f_id["training_fingerprint"]
        or pipeline != n_id["pipeline"]
        or not tta
        or tta.get("adapter") != FLIP_ADAPTER
        or tta.get("adapter_sha256") != sha(Path(__file__).with_name("yolo_flip.py"))
        or tta.get("nms_iou") != 0.6
        or n_id["pipeline"]["model"].get("image_size") != 1280
        or n_id["pipeline"]["model"].get("nms_threshold") != 0.6
        or n_id["pipeline"]["model"].get("max_detections") != 500
    ):
        raise ValueError("Native/flip YOLO caches do not describe one full pipeline")
    return native, flip


def _same_checkpoint_pipeline(val_view, test_view):
    before, after = val_view["identity"], test_view["identity"]
    return all(
        before[key] == after[key]
        for key in ("checkpoint_sha256", "pipeline", "training_fingerprint")
    )


def _combined(c1_path, native_path, flip_path, records):
    for c1, native, flipped in zip(
        stream_frames(c1_path, records),
        stream_frames(native_path, records),
        stream_frames(flip_path, records),
        strict=True,
    ):
        merged = merge_frame(native + flipped, 0.6, 500)
        yield c1, merged


def _family_only_manual(decision):
    return (
        decision.get("pooled_gain", -1) >= 0.001
        and set(decision.get("reasons", [])) <= FAMILY_REVIEW_REASONS
    )


def compare_champion(
    c1_path,
    candidate_native_path,
    candidate_flip_path,
    champion_report_path,
    champion_csv_path,
    config,
    output,
):
    """Full official-val comparison for both class route and all-YOLO route."""
    from .metrics import evaluate
    from .sequential_policy import per_video_scores

    output = Path(output)
    if output.exists():
        raise ValueError("Preserve an existing comparison; choose a new path")
    parent, champion_meta = _parent(champion_report_path, champion_csv_path)
    records = records_for(config, "val")
    if len(records) != 2846 or digest(records) != parent.get("data_fingerprint"):
        raise ValueError("Validation cohort differs from current champion")
    c1_meta = metadata(c1_path)
    if c1_meta != parent["inputs"]["c1"]:
        raise ValueError("c1 cache is not the champion's exact cached source")
    candidate_native, candidate_flip = _yolo_views(
        candidate_native_path, candidate_flip_path, records, "val"
    )
    if (
        candidate_native["identity"]["training_fingerprint"]
        != c1_meta["identity"]["training_fingerprint"]
    ):
        raise ValueError("Candidate and champion use different train manifests")
    baseline = parent["cases"]["flip_route"]["metrics"]
    started = time.perf_counter()
    cases = {}
    decisions = {}
    for case in ("truck_bus_c1", "all_yolo"):
        predictions = []
        for c1, yolo in _combined(
            c1_path, candidate_native_path, candidate_flip_path, records
        ):
            predictions.extend(
                route_frame(c1, yolo, "class_route") if case == "truck_bus_c1" else yolo
            )
        metrics = evaluate(records, predictions, [100, 500])
        metrics["per_video_mAP"] = per_video_scores(records, predictions)
        cases[case] = {"metrics": metrics, "detections": len(predictions)}
        decisions[case] = assess_metrics(baseline, metrics)
        del predictions
        gc.collect()
    accepted = {key: value["accepted"] for key, value in decisions.items()}
    review = {
        key: decision["accepted"] or _family_only_manual(decision)
        for key, decision in decisions.items()
    }
    report = {
        "protocol": "candidate-vs-scored-yolo-flip-champion-v1",
        "champion_protocol": CHAMPION_PROTOCOL,
        "champion_comparison_sha256": sha(champion_report_path),
        "champion_csv_sha256": CHAMPION_CSV_SHA,
        "champion_public_score": CHAMPION_SCORE,
        "records": len(records),
        "data_fingerprint": digest(records),
        "evaluator_identity": evaluator_identity(),
        "candidate_checkpoint_sha256": candidate_native["identity"][
            "checkpoint_sha256"
        ],
        "inputs": {
            "c1": c1_meta,
            "candidate_yolo_native": candidate_native,
            "candidate_yolo_horizontal_flip": candidate_flip,
        },
        "inference": {
            "image_size": 1280,
            "views": ["native", "horizontal_flip"],
            "class_aware_nms_iou": 0.6,
            "global_max_detections": 500,
            "c1_reused_without_inference": True,
        },
        "baseline": {"route": "scored_champion_flip_route", "metrics": baseline},
        "cases": cases,
        "decisions_vs_champion": decisions,
        "accepted": accepted,
        "review_eligible": review,
        "status": {
            key: "FULL_VALIDATION_ACCEPTED"
            if decision["accepted"]
            else "FAMILY_UNCERTAINTY_MANUAL_REVIEW"
            if review[key]
            else "REJECTED_KEEP_CHAMPION"
            for key, decision in decisions.items()
        },
        "comparison_seconds": time.perf_counter() - started,
        "public_score": None,
        "limitation": "Local official-val AP is not a Kaggle score or independent OOF estimate.",
    }
    write_json(output, report)
    return report


def export_candidate(
    c1_val_path,
    c1_test_path,
    native_val_path,
    flip_val_path,
    native_test_path,
    flip_test_path,
    champion_report_path,
    champion_csv_path,
    report_path,
    config,
    output,
    *,
    route="truck_bus_c1",
    manual_review=False,
):
    """Export only a recorded accepted candidate through its exact val/test views."""
    from .submission import write_submission
    from scripts.check_lab import validate_submission
    from .class_ensemble import compatible_pair

    report = read_json(report_path)
    champion, champion_meta = _parent(champion_report_path, champion_csv_path)
    val_records = records_for(config, "val")
    if (
        report.get("protocol") != "candidate-vs-scored-yolo-flip-champion-v1"
        or report.get("champion_comparison_sha256") != sha(champion_report_path)
        or report.get("champion_csv_sha256") != CHAMPION_CSV_SHA
        or report.get("champion_public_score") != CHAMPION_SCORE
        or report.get("data_fingerprint") != digest(val_records)
        or report.get("records") != len(val_records)
        or report.get("evaluator_identity") != evaluator_identity()
        or report.get("candidate_checkpoint_sha256")
        != report.get("inputs", {})
        .get("candidate_yolo_native", {})
        .get("identity", {})
        .get("checkpoint_sha256")
    ):
        raise ValueError("Candidate report is not bound to the current champion")
    decision = report.get("decisions_vs_champion", {}).get(route)
    baseline = champion["cases"]["flip_route"]["metrics"]
    measured_baseline = report.get("baseline", {}).get("metrics")
    recomputed = assess_metrics(
        baseline, report.get("cases", {}).get(route, {}).get("metrics", {})
    )
    if not decision or not (
        decision.get("accepted")
        or manual_review
        and report.get("review_eligible", {}).get(route)
    ):
        raise ValueError("Candidate did not pass full validation guards")
    if (
        measured_baseline != baseline
        or recomputed != decision
        or report.get("accepted", {}).get(route) != decision.get("accepted")
        or report.get("review_eligible", {}).get(route)
        != (decision.get("accepted") or _family_only_manual(decision))
        or report.get("inputs", {}).get("c1") != champion.get("inputs", {}).get("c1")
    ):
        raise ValueError("Candidate report metrics/guards differ from its champion")
    if route not in ("truck_bus_c1", "all_yolo"):
        raise ValueError("Unknown candidate route")
    val_c1 = metadata(c1_val_path)
    if val_c1 != champion["inputs"]["c1"]:
        raise ValueError("Validation c1 cache differs from scored champion")
    val_native, val_flip = _yolo_views(
        native_val_path, flip_val_path, val_records, "val"
    )
    if (
        val_native != report["inputs"]["candidate_yolo_native"]
        or val_flip != report["inputs"]["candidate_yolo_horizontal_flip"]
    ):
        raise ValueError("Candidate validation views changed after comparison")

    test_records = records_for(config, "test")
    test_c1, _ = compatible_pair(c1_test_path, native_test_path, test_records, "test")
    test_native, test_flip = _yolo_views(
        native_test_path, flip_test_path, test_records, "test"
    )
    champion_test = champion_meta.get("inputs", {}).get("c1_test")
    if champion_test is not None and test_c1 != champion_test:
        raise ValueError("Test c1 predictions differ from scored champion")
    for val_view, test_view in ((val_native, test_native), (val_flip, test_flip)):
        if not _same_checkpoint_pipeline(val_view, test_view):
            raise ValueError("Candidate test inference differs from validated pipeline")
    if Path(output).exists():
        raise ValueError("Preserve existing CSV; choose a new output path")
    root = Path(config["data"]["root"])
    sample_path = root / "sample_submission.csv"
    manifest = sample_path if sample_path.is_file() else root / "test_images.csv"
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        order = [row["image_id"] for row in csv.DictReader(stream)]
    if len(order) != len(set(order)) or set(order) != {
        r["submission_id"] for r in test_records
    }:
        raise ValueError("Test manifest/template differs from prediction caches")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    parts = {}
    with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
        for record, (c1, yolo) in zip(
            test_records,
            _combined(c1_test_path, native_test_path, flip_test_path, test_records),
            strict=True,
        ):
            predictions = (
                route_frame(c1, yolo, "class_route")
                if route == "truck_bus_c1"
                else yolo
            )
            part = Path(scratch) / f"{record['id']}.csv"
            write_submission([record], predictions, None, part, max_detections=500)
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
        sample_path if sample_path.is_file() else None,
    )
    if (
        not checked["valid"]
        or not checked["matches_template_order"]
        or checked["all_predictions_empty"]
    ):
        temporary.unlink(missing_ok=True)
        raise ValueError("Candidate CSV failed structural checks")
    write_json(
        output.with_suffix(".csv.metadata.json"),
        {
            "protocol": report["protocol"],
            "champion_comparison_sha256": report["champion_comparison_sha256"],
            "candidate_comparison_sha256": sha(report_path),
            "champion_csv_sha256": CHAMPION_CSV_SHA,
            "route": route,
            "inputs": {
                "c1_test": test_c1,
                "candidate_yolo_native_test": test_native,
                "candidate_yolo_flip_test": test_flip,
            },
            "validation": checked,
            "strict_promotion_accepted": decision["accepted"],
            "manual_review": manual_review,
            "family_decision": decision,
            "public_score": None,
            "status": "READY_FOR_MANUAL_SUBMISSION_REVIEW",
            "data_fingerprint": digest(test_records),
        },
    )
    temporary.replace(output)
    return checked
