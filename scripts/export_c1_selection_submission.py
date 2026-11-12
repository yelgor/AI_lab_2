"""Export the fully confirmed c1 full-train selection checkpoint for Kaggle."""

import argparse
import fcntl
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvlab.common import read_json, write_json
from cvlab.full_data import sha256
from cvlab.selection_policy import PROTOCOL, require_training_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--plan-name", default="c1_continuation")
    args = parser.parse_args()
    work, source = args.work.resolve(), args.source.resolve()
    run = work / "runs" / f"rtdetr_{args.plan_name}_train_selection"
    report_path = work / "full_data" / args.plan_name / "selection_report.json"

    with (work / ".local_gpu_queue.lock").open("a") as lease:
        print("Waiting for shared GPU lease; active validation is preserved", flush=True)
        fcntl.flock(lease, fcntl.LOCK_EX)
        if not report_path.is_file():
            raise FileNotFoundError(f"Full confirmation report not found: {report_path}")
        report = read_json(report_path)
        candidate = run / "best.pt"
        candidate_sha = sha256(candidate)
        require_training_source(candidate_sha)
        if (report.get("selection_protocol") != PROTOCOL or not report.get("accepted")
                or not report.get("native_decision", {}).get("accepted")
                or not report.get("tiled_decision", {}).get("accepted")
                or report.get("source_checkpoint_sha256") != sha256(source / "best.pt")):
            raise RuntimeError("A source-bound robust confirmation is required; legacy AP-only gates cannot export")
        if (report.get("candidate_checkpoint_sha256") != candidate_sha
                or report.get("selected_checkpoint_sha256") != candidate_sha
                or report.get("native_gain", -1) < 0.001
                or report.get("tiled_gain", -1) < 0.001):
            raise RuntimeError("Candidate did not pass both full-validation gates; no CSV exported")

        source_config = read_json(source / "config.json")
        config = json.loads(json.dumps(source_config))
        config["data"]["split"] = "official"
        config["loader"].update(batch_size=1, workers=2, pin_memory=True,
                                persistent_workers=False, prefetch_factor=1)
        config["runtime"].update(device="cuda", precision="fp16", cpu_threads=2)
        from cvlab.engine import load_trained, setup
        from cvlab.prepare import select_records
        from cvlab.tiled_inference import TileOptions, export_tiled
        from cvlab.common import digest
        import torch

        if digest(select_records(config, "val")) != report["data_fingerprint"]:
            raise RuntimeError("Validation records differ from confirmation; no export")
        records = select_records(config, "test")
        device = setup(config)
        model = load_trained(candidate, config, device)
        output = work / "submissions" / f"rtdetr_{candidate_sha[:12]}_c1_epoch1_selection_tiles.csv"
        options = TileOptions(960, overlap=0.2, nms_iou=0.6, edge_margin=2, workers=2)
        result = export_tiled(model, config, records, device, options, output, candidate_sha)
        if not result.get("valid") or not result.get("matches_template_order") or result.get("all_predictions_empty"):
            raise RuntimeError(f"Export failed final CSV validation: {result}")
        metadata_path = output.with_suffix(".csv.metadata.json")
        metadata = read_json(metadata_path)
        metadata.update(
            training_route="full_train_official_val_selection",
            selection_report=str(report_path),
            selection_report_sha256=sha256(report_path),
            independent_validation={
                "frames": report["frames"],
                "data_fingerprint": report["data_fingerprint"],
                "source_checkpoint_sha256": report["source_checkpoint_sha256"],
                "native_AP_gain": report["native_gain"],
                "tiled_AP_gain": report["tiled_gain"],
                "candidate_native_AP": report["candidate_native_AP"],
                "candidate_tiled_AP": report["candidate_tiled_AP"],
            },
        )
        write_json(metadata_path, metadata)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"READY_FOR_KAGGLE: {output}", flush=True)
        print(json.dumps({"checkpoint_sha256": candidate_sha,
                          "csv_sha256": result["sha256"], **result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
