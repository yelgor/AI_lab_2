"""Validate and export a line-search point between c1 source and continuation."""

import argparse
import fcntl
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cvlab.common import read_json, write_json
from cvlab.engine import setup
from cvlab.full_data import sha256
from cvlab.models import build_model
from cvlab.prepare import select_records
from cvlab.tiled_inference import TileOptions, export_tiled


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

    lease = (work / ".local_gpu_queue.lock").open("a")
    print("Waiting for the shared GPU lease", flush=True)
    fcntl.flock(lease, fcntl.LOCK_EX)

    from cvlab.common import digest
    from cvlab.efficient_search import _score, full_config
    from cvlab.selection_policy import PROTOCOL, assess_metrics, require_training_source

    source_sha, target_sha = sha256(source_path), sha256(target_path)
    require_training_source(source_sha)
    source_ckpt = torch.load(source_path, map_location="cpu", weights_only=False)
    target_ckpt = torch.load(target_path, map_location="cpu", weights_only=False)
    if source_ckpt["config"]["model"] != target_ckpt["config"]["model"]:
        raise ValueError("Checkpoint model configurations differ")
    if source_ckpt.get("adapter") != target_ckpt.get("adapter"):
        raise ValueError("Checkpoint adapter metadata differs")
    from cvlab.sequential_policy import assert_family_isolation

    config = full_config(target_ckpt["config"], official=True)
    records = select_records(config, "val")
    seen = set()
    for saved, path in ((source_ckpt, source_path), (target_ckpt, target_path)):
        exposure = saved.get("seen_training_sequences")
        if exposure is None:
            split_path = path.parent / "split.json"
            split = read_json(split_path)
            if not saved.get("data_fingerprint") or split.get("fingerprint") != saved["data_fingerprint"]:
                raise ValueError("Cannot verify inherited training exposure")
            exposure = set(split["train_sequences"]) | set(split.get("seen_training_sequences", []))
        seen.update(exposure)
    assert_family_isolation([], records, seen)
    provenance = {"source_sha256": source_sha, "target_sha256": target_sha, "alpha": args.alpha}
    # Different endpoints cannot overwrite the same alpha-named checkpoint/report.
    output_dir = work / "runs" / f"rtdetr_c1_path_{digest(provenance)[:12]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path, checkpoint_path = output_dir / "report.json", output_dir / "interpolated.pt"
    adapter = target_ckpt.get("adapter")
    if checkpoint_path.exists():
        existing = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if existing.get("interpolation") != provenance:
            raise ValueError("Existing interpolated checkpoint has different provenance")
        del existing
    else:
        checkpoint = {
            "model": interpolate(source_ckpt["model"], target_ckpt["model"], args.alpha),
            "config": target_ckpt["config"], "adapter": adapter, "interpolation": provenance,
            "seen_training_sequences": sorted(seen),
        }
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(checkpoint_path)
        del checkpoint
    del source_ckpt, target_ckpt
    interpolated_sha = sha256(checkpoint_path)
    require_training_source(interpolated_sha)
    report = {
        "selection_protocol": PROTOCOL, "alpha": args.alpha,
        "source_checkpoint_sha256": source_sha, "target_checkpoint_sha256": target_sha,
        "interpolated_checkpoint_sha256": interpolated_sha,
        "frames": len(records), "data_fingerprint": digest(records), "csv": None,
    }
    for tiled, phase in ((False, "native"), (True, "tiled")):
        baseline = _score(work, source_path, config, records, tiled=tiled)
        measured = _score(work, checkpoint_path, config, records, tiled=tiled)
        decision = assess_metrics(baseline["metrics"], measured["metrics"])
        report[phase] = {"baseline": baseline, "candidate": measured, "decision": decision}
        write_json(report_path, report)
        print(json.dumps({"phase": phase, **decision}), flush=True)
        if not decision["accepted"]:
            report["decision"] = "rejected_by_robust_validation"
            write_json(report_path, report)
            print(f"REJECTED before further GPU work: {decision['reasons']}", flush=True)
            return
    device = setup(config)
    model = build_model(config["model"], adapter, initialize_pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"])
    options = TileOptions(960, overlap=0.2, nms_iou=0.6, edge_margin=2, workers=2)

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
        full_validation_native_gain=report["native"]["decision"]["pooled_gain"],
        full_validation_tiled_gain=report["tiled"]["decision"]["pooled_gain"],
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
