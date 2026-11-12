"""Measure validation response along a saved checkpoint update vector."""

import argparse
import csv
import fcntl
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cvlab.common import CLASSES, read_json, write_json
from cvlab.engine import predict, setup
from cvlab.metrics import evaluate
from cvlab.models import build_model
from cvlab.prepare import select_records
from cvlab.inference_experiments import balanced_subset
from cvlab.sequential_policy import video_family


def checkpoint_state(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = {key: value.detach().clone() for key, value in checkpoint["model"].items()}
    adapter = checkpoint.get("adapter")
    config = checkpoint["config"]
    return state, adapter, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    args = parser.parse_args()
    work, source, candidate = map(Path.resolve, (args.work, args.source, args.candidate))
    output = work / "diagnostics" / "c1_weight_line_search"
    output.mkdir(parents=True, exist_ok=True)
    alphas = [float(value) for value in args.alphas.split(",")]
    if not alphas or any(not 0 <= alpha <= 1 for alpha in alphas):
        raise ValueError("All interpolation alphas must be in [0,1]")

    with (work / ".local_gpu_queue.lock").open("a") as lease:
        print("Waiting for shared GPU lease before weight-direction evaluation", flush=True)
        fcntl.flock(lease, fcntl.LOCK_EX)
        source_state, source_adapter, source_config = checkpoint_state(source)
        candidate_state, candidate_adapter, candidate_config = checkpoint_state(candidate)
        if source_config["model"] != candidate_config["model"]:
            raise ValueError("Source and candidate model configurations differ")
        if source_adapter != candidate_adapter:
            raise ValueError("Source and candidate RT-DETR adapter metadata differ")
        if source_state.keys() != candidate_state.keys():
            raise ValueError("Source and candidate parameter keys differ")

        config = read_json(candidate.parent / "config.json")
        config["data"]["split"] = "official"
        config["loader"].update(batch_size=1, workers=0, pin_memory=False,
                                persistent_workers=False)
        config["runtime"].update(device="cuda", precision="fp16", cpu_threads=2)
        full_records = select_records(config, "val")
        records = balanced_subset(full_records, min(512, len(full_records)))
        device = setup(config)
        model = build_model(config["model"], candidate_adapter,
                            initialize_pretrained=False).to(device)

        result = {
            "description": "Linear interpolation in model-weight space from source to candidate; inference-only, no optimizer update.",
            "source_checkpoint": str(source),
            "candidate_checkpoint": str(candidate),
            "validation_frames": len(records),
            "full_validation_frames": len(full_records),
            "subset_policy": "fixed temporally-spread subset balanced across all official-validation families",
            "validation_families": sorted({video_family(r["sequence_id"]) for r in records}),
            "alphas": [],
        }
        rows = []
        for alpha in alphas:
            interpolated = {}
            for key, src in source_state.items():
                dst = candidate_state[key]
                interpolated[key] = (
                    src.mul(1.0 - alpha).add(dst, alpha=alpha)
                    if src.is_floating_point()
                    else (src if alpha == 0 else dst)
                )
            model.load_state_dict(interpolated, strict=True)
            del interpolated
            predictions, timing = predict(model, config, records, device)
            scores = evaluate(records, predictions, [100, 500])
            family_scores = {}
            for family in result["validation_families"]:
                family_records = [r for r in records if video_family(r["sequence_id"]) == family]
                family_ids = {r["id"] for r in family_records}
                family_predictions = [p for p in predictions if p["image_id"] in family_ids]
                family_scores[family] = evaluate(family_records, family_predictions, [100])["coco100"]["mAP"]
            macro_family_ap = sum(family_scores.values()) / len(family_scores)
            row = {
                "alpha": alpha,
                "coco100_mAP": scores["coco100"]["mAP"],
                "coco100_AP50": scores["coco100"]["AP50"],
                "coco100_AP75": scores["coco100"]["AP75"],
                "macro_family_AP": macro_family_ap,
                "timing_seconds": timing["pipeline_seconds"],
                "family_AP": family_scores,
                "per_class_AP": scores["coco100"]["per_class"],
            }
            result["alphas"].append(row)
            flat = {key: value for key, value in row.items() if key not in ("family_AP", "per_class_AP")}
            flat.update({f"family_{key}": value for key, value in family_scores.items()})
            flat.update({f"class_{key}": value for key, value in row["per_class_AP"].items()})
            rows.append(flat)
            write_json(output / "report.json", result)
            with (output / "metrics.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(json.dumps({"alpha": alpha, "mAP": row["coco100_mAP"],
                              "macro_family_AP": macro_family_ap,
                              "per_class_AP": row["per_class_AP"],
                              "family_AP": family_scores,
                              "seconds": timing["pipeline_seconds"]}, sort_keys=True), flush=True)
            del predictions, scores, family_scores, family_records, family_predictions
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        result["interpretation_limits"] = [
            "This probes one observed weight-update vector only; it does not estimate a general gradient-to-Kaggle-score law.",
            "Validation is the same seven-family official cohort used to select the endpoint; treat line-search maxima as exploratory, not independent confirmation.",
            "Kaggle public score remains external and cannot be inferred from this AP curve.",
        ]
        write_json(output / "report.json", result)
        print(f"Saved direction study: {output}", flush=True)


if __name__ == "__main__":
    main()
