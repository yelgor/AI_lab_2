"""Validate and export a line-search point between c1 source and continuation."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cvlab.common import read_json, write_json
from cvlab.engine import predict, setup
from cvlab.full_data import sha256
from cvlab.metrics import evaluate
from cvlab.models import build_model
from cvlab.prepare import select_records
from cvlab.tiled_inference import TileOptions, export_tiled, predict_tiled


def interpolate(source, target, alpha):
    if source.keys() != target.keys():
        raise ValueError("Checkpoint parameter sets differ")
    result = {}
    for key, left in source.items():
        right = target[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"Checkpoint tensor mismatch: {key}")
        result[key] = (
            left.mul(1 - alpha).add(right, alpha=alpha)
            if left.is_floating_point()
            else (left if alpha < 1 else right)
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 < args.alpha < 1:
        raise ValueError("Interpolation alpha must be strictly between 0 and 1")
    work, source_path, target_path = map(
        lambda p: p.resolve(), (args.work, args.source, args.candidate)
    )
    output_dir = work / "runs" / f"rtdetr_c1_path_alpha_{args.alpha:.3f}"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    checkpoint_path = output_dir / "interpolated.pt"

    lease = (work / ".local_gpu_queue.lock").open("a")
    print("Waiting for the shared GPU lease", flush=True)
    fcntl.flock(lease, fcntl.LOCK_EX)

    source_sha, target_sha = sha256(source_path), sha256(target_path)
    source_ckpt = torch.load(source_path, map_location="cpu", weights_only=False)
    target_ckpt = torch.load(target_path, map_location="cpu", weights_only=False)
    if source_ckpt["config"]["model"] != target_ckpt["config"]["model"]:
        raise ValueError("Checkpoint model configurations differ")
    if source_ckpt.get("adapter") != target_ckpt.get("adapter"):
        raise ValueError("Checkpoint adapter metadata differs")
    adapter = target_ckpt.get("adapter")
    checkpoint = {
        "model": interpolate(source_ckpt["model"], target_ckpt["model"], args.alpha),
        "config": target_ckpt["config"],
        "adapter": target_ckpt.get("adapter"),
        "interpolation": {
            "source_sha256": source_sha,
            "target_sha256": target_sha,
            "alpha": args.alpha,
        },
    }
    torch.save(checkpoint, checkpoint_path)
    del source_ckpt, target_ckpt, checkpoint
    interpolated_sha = sha256(checkpoint_path)

    config = read_json(target_path.parent / "config.json")
    config["data"]["split"] = "official"
    config["loader"].update(
        batch_size=1, workers=2, pin_memory=True, persistent_workers=False,
        prefetch_factor=1,
    )
    config["runtime"].update(device="cuda", precision="fp16", cpu_threads=2)
    device = setup(config)
    model = build_model(config["model"], adapter, initialize_pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"])

    records = select_records(config, "val")
    baseline = read_json(work / "full_data" / "c1_continuation" / "selection_report.json")
    report = {
        "method": "linear interpolation along the observed source-to-candidate update vector",
        "alpha": args.alpha,
        "source_checkpoint_sha256": source_sha,
        "target_checkpoint_sha256": target_sha,
        "interpolated_checkpoint_sha256": interpolated_sha,
        "frames": len(records),
        "data_fingerprint": baseline["data_fingerprint"],
        "baseline_native_AP": baseline["baseline_native_AP"],
        "baseline_tiled_AP": baseline["baseline_tiled_AP"],
        "native": None,
        "tiled": None,
        "csv": None,
    }

    native_predictions, native_timing = predict(model, config, records, device)
    native_metrics = evaluate(records, native_predictions, [100, 500])
    report["native"] = {
        "metrics": native_metrics,
        "timing": native_timing,
        "gain": native_metrics["coco100"]["mAP"] - report["baseline_native_AP"],
    }
    write_json(report_path, report)
    print(json.dumps({"phase": "native", "mAP": native_metrics["coco100"]["mAP"],
                      "gain": report["native"]["gain"]}), flush=True)
    del native_predictions, native_metrics

    options = TileOptions(960, overlap=0.2, nms_iou=0.6, edge_margin=2, workers=2)
    tiled, tiled_timing = predict_tiled(model, config, records, device, options)
    tiled_metrics = evaluate(records, tiled[0.6], [100, 500])
    report["tiled"] = {
        "metrics": tiled_metrics,
        "timing": tiled_timing,
        "gain": tiled_metrics["coco100"]["mAP"] - report["baseline_tiled_AP"],
        "options": {"tile_side": 960, "overlap": 0.2, "nms_iou": 0.6, "edge_margin": 2},
    }
    write_json(report_path, report)
    print(json.dumps({"phase": "tiled", "mAP": tiled_metrics["coco100"]["mAP"],
                      "gain": report["tiled"]["gain"]}), flush=True)
    del tiled

    if report["native"]["gain"] < 0.001 or report["tiled"]["gain"] < 0.001:
        report["decision"] = "rejected_by_full_validation_gates"
        write_json(report_path, report)
        print(f"REJECTED: full validation gains below gate; see {report_path}", flush=True)
        return

    test_records = select_records(config, "test")
    csv_path = work / "submissions" / (
        f"rtdetr_{interpolated_sha[:12]}_c1_path_alpha{int(args.alpha * 100):02d}_tiles.csv"
    )
    result = export_tiled(
        model, config, test_records, device, options, csv_path, interpolated_sha
    )
    if not result.get("valid") or not result.get("matches_template_order") or result.get("all_predictions_empty"):
        raise RuntimeError(f"Interpolated candidate CSV failed checks: {result}")
    metadata_path = csv_path.with_suffix(".csv.metadata.json")
    metadata = read_json(metadata_path)
    metadata.update(
        training_route="warm_start_weight_vector_interpolation",
        source_checkpoint_sha256=source_sha,
        target_checkpoint_sha256=target_sha,
        interpolation_alpha=args.alpha,
        validation_report=str(report_path),
        full_validation_native_gain=report["native"]["gain"],
        full_validation_tiled_gain=report["tiled"]["gain"],
    )
    write_json(metadata_path, metadata)
    report["decision"] = "ready_for_manual_kaggle_evaluation"
    report["csv"] = str(csv_path)
    report["csv_sha256"] = result["sha256"]
    report["csv_detections"] = result["detections"]
    write_json(report_path, report)
    print(f"READY_FOR_KAGGLE: {csv_path}", flush=True)
    print(json.dumps({"checkpoint_sha256": interpolated_sha, **result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
