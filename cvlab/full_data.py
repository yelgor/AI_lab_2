"""Full official train selection and explicitly unvalidated train+val final fit."""
import copy
import hashlib
import json
import math
from pathlib import Path
import time

from .common import read_json, write_json, digest
from .postprocess_experiments import stream_array


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def manifest_summary(prepared):
    result = {}
    for split in ("train", "val"):
        keys, families = set(), set()
        for record in stream_array(Path(prepared) / f"{split}.json", "images"):
            key = (record["sequence_id"], record["frame_id"])
            if key in keys:
                raise ValueError(f"Duplicate sequence/frame in {split}")
            keys.add(key)
            from .sequential_policy import video_family
            families.add(video_family(record["sequence_id"]))
        result[split] = {"frames": len(keys), "families": sorted(families)}
    if set(result["train"]["families"]) & set(result["val"]["families"]):
        raise ValueError("Official train/val flight families overlap")
    return result


def joint_manifest(prepared, destination):
    """Stream a new manifest with unique IDs; never overwrite either input."""
    prepared, destination = Path(prepared), Path(destination)
    if destination.resolve() in {(prepared / f"{s}.json").resolve() for s in ("train", "val")}:
        raise ValueError("Refusing to overwrite a source manifest")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    seen, files, counts = set(), set(), {}
    with temporary.open("w") as out:
        out.write('{"version":1,"images":[')
        count = 0
        for split in ("train", "val"):
            counts[split] = 0
            for record in stream_array(prepared / f"{split}.json", "images"):
                key = (split, record["sequence_id"], record["frame_id"])
                if key in seen or record["file_name"] in files:
                    raise ValueError("Duplicate full-data source key or image path")
                seen.add(key)
                files.add(record["file_name"])
                record["origin_split"], record["origin_id"] = split, record["id"]
                count += 1
                record["id"] = count
                if count > 1:
                    out.write(",")
                json.dump(record, out, sort_keys=True, allow_nan=False)
                counts[split] += 1
        out.write('],"source_counts":')
        json.dump(counts, out, sort_keys=True)
        out.write("}")
    actual_sha = sha256(temporary)
    if destination.exists():
        if sha256(destination) != actual_sha:
            raise ValueError("Existing joint manifest differs; preserve both versions")
        temporary.unlink()
    else:
        temporary.replace(destination)
    return {"frames": count, "source_counts": counts, "sha256": actual_sha}


def recipe(source_config, output, *, epochs, final=False, manifest=None, report=None):
    config = copy.deepcopy(source_config)
    config["output"] = str(output)
    config["data"].update(split="official", frame_step=1)
    for key in ("folds_file", "training_manifest"):
        config["data"].pop(key, None)
    config["loader"].update(batch_size=1, workers=2, prefetch_factor=1,
                            persistent_workers=False, sampling="flight_time", temporal_bins=8)
    # Continue the selected candidate at its tuned LR. Silently replacing a
    # rung's rate here can invalidate its optimizer-moment continuation.
    config["train"].update(epochs=epochs, max_hours=48, max_batches_per_epoch=None,
                           lr=source_config["train"]["lr"], freeze_batch_norm=True,
                           early_stopping_patience=0)
    for key in ("validation_images", "mode", "selection_report", "selection_report_sha256"):
        config["evaluation"].pop(key, None)
    config["evaluation"].update(independent_video_families=True, per_video_metrics=True)
    if final:
        config["data"].update(split="full_train_val", training_manifest=str(manifest))
        config["evaluation"].update(mode="full_data_final_fit", selection_report=str(report),
                                    selection_report_sha256=sha256(report))
    return config


def rows(output):
    path = Path(output) / "history.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def full_epochs(config, source, count, deadline):
    from .engine import train
    from .score_improvement import release_gpu

    output = Path(config["output"])
    while len(rows(output)) < config["train"]["epochs"]:
        if time.time() >= deadline:
            return False
        if (output / "STOP").exists():
            return False
        before = rows(output)
        last = output / "last.pt"
        print(f"FULL DATA train={count}; validation={'none (final fit)' if config['data']['split']=='full_train_val' else 'full official'}; epoch={len(before)+1}/{config['train']['epochs']}; full pass, no batch cap", flush=True)
        # No round_batches/round_seconds: evaluation boundaries follow full
        # epochs, not a new random prefix. An interrupted unsaved epoch can repeat.
        train(config, resume=last if last.exists() else None,
              init_checkpoint=None if last.exists() else Path(source),
              warm_optimizer=not last.exists(), yield_after_epoch=True)
        release_gpu()
        after = rows(output)
        if len(after) <= len(before):
            raise RuntimeError("Full-data epoch made no committed progress")
        row = after[-1]
        coverage = read_json(output / f"coverage_epoch_{row['epoch']:04d}.json")
        if row["train_batches"] != count or coverage["unique_frames"] != count:
            raise RuntimeError("Epoch was not a complete unique-frame pass; preserve and inspect")
        write_json(output / "coverage_summary.json", {
            "dataset_size": count, "completed_full_epochs": len(after),
            "actual_samples": sum(r["train_batches"] for r in after),
            "last_epoch_unique_frames": coverage["unique_frames"],
            "resume_granularity": "full_epoch_boundary; unfinished unsaved epoch may repeat",
        })
    return True


def run(work, source_run, *, hours=8, selection_epochs=2, final_epochs=1,
        plan_name="default"):
    from .efficient_search import _score, full_config
    from .engine import load_trained, setup
    from .prepare import select_records
    from .score_improvement import release_gpu
    from .tiled_inference import TileOptions, export_tiled

    if (not plan_name.replace("_", "").isalnum()
            or not math.isfinite(hours) or hours <= 0
            or selection_epochs < 1 or final_epochs < 1):
        raise ValueError("Require positive finite hours and positive epoch counts")
    work, source_run = Path(work).resolve(), Path(source_run).resolve()
    root = work / "full_data" if plan_name == "default" else work / "full_data" / plan_name
    state_path = root / "state.json"
    source = source_run / "best.pt"
    source_sha = sha256(source)
    state = read_json(state_path) if state_path.exists() else {
        "source_sha256": source_sha, "deadline": time.time() + hours * 3600,
        "selection_epochs": selection_epochs, "final_epochs": final_epochs,
        "target_public_score": 0.28,
    }
    if state["source_sha256"] != source_sha:
        raise ValueError("Full-data source changed")
    if state.get("finished"):
        return state
    base = read_json(source_run / "config.json")
    prepared = Path(base["data"]["prepared"])
    counts = manifest_summary(prepared)
    state["data"] = counts
    selection = (work / "runs/rtdetr_full_train_selection" if plan_name == "default"
                 else work / "runs" / f"rtdetr_{plan_name}_train_selection")
    config = recipe(base, selection, epochs=state["selection_epochs"])
    state["status"] = "full_train_selection"
    write_json(state_path, state)
    print(f"Full-data phase A: train={counts['train']['frames']}, val={counts['val']['frames']}; source={source_sha[:12]}", flush=True)
    if not full_epochs(config, source, counts["train"]["frames"], state["deadline"]):
        return state
    report_path = root / "selection_report.json"
    if report_path.exists():
        report = read_json(report_path)
    else:
        evaluation = full_config(config, official=True)
        validation = select_records(evaluation, "val")
        candidate = selection / "best.pt"
        baseline = _score(work, source, evaluation, validation)
        measured = _score(work, candidate, evaluation, validation)
        native_gain = measured["metrics"]["coco100"]["mAP"] - baseline["metrics"]["coco100"]["mAP"]
        old_tiled = _score(work, source, evaluation, validation, tiled=True)
        new_tiled = _score(work, candidate, evaluation, validation, tiled=True)
        tiled_gain = new_tiled["metrics"]["coco100"]["mAP"] - old_tiled["metrics"]["coco100"]["mAP"]
        selected = candidate if native_gain >= 0.001 and tiled_gain >= 0.001 else source
        report = {
            "full_heldout_validation": True, "route": "full_train_official_val_selection",
            "frames": len(validation), "data_fingerprint": digest(validation),
            "cap": 100, "image_size": config["model"]["image_size"], "tile_side": 960, "overlap": 0.2,
            "nms_iou": 0.6, "edge_margin": 2,
            "source_checkpoint_sha256": source_sha, "candidate_checkpoint_sha256": sha256(candidate),
            "selected_checkpoint": str(selected), "selected_checkpoint_sha256": sha256(selected),
            "baseline_native_AP": baseline["metrics"]["coco100"]["mAP"],
            "candidate_native_AP": measured["metrics"]["coco100"]["mAP"],
            "baseline_tiled_AP": old_tiled["metrics"]["coco100"]["mAP"],
            "candidate_tiled_AP": new_tiled["metrics"]["coco100"]["mAP"],
            "native_gain": native_gain, "tiled_gain": tiled_gain,
            "selected_lr": config["train"]["lr"], "selected_final_epochs": final_epochs,
            "final_budget_reason": "One predeclared complete train+val pass; no tuning on seen validation",
        }
        write_json(report_path, report)
    combined = root / "trainval.json"
    joint = joint_manifest(prepared, combined)
    final_run = (work / "runs/rtdetr_full_data_final" if plan_name == "default"
                 else work / "runs" / f"rtdetr_{plan_name}_final")
    final_config = recipe(base, final_run, epochs=state["final_epochs"], final=True,
                          manifest=combined, report=report_path)
    state.update(status="full_data_final_fit", joint_manifest=joint,
                 selection_report_sha256=sha256(report_path))
    write_json(state_path, state)
    print(f"Full-data phase B: train+val={joint['frames']}; independent validation AP unavailable", flush=True)
    if not full_epochs(final_config, report["selected_checkpoint"], joint["frames"], state["deadline"]):
        return state
    checkpoint = final_run / "final.pt"
    sha = sha256(checkpoint)
    destination = work / "submissions" / f"rtdetr_{sha[:12]}_full_data_final_tiles.csv"
    device = setup(final_config)
    model = load_trained(checkpoint, final_config, device)
    export_tiled(model, final_config, select_records(final_config, "test"), device,
                 TileOptions(960, overlap=0.2, nms_iou=0.6, edge_margin=2, workers=2), destination, sha)
    del model
    release_gpu()
    metadata_path = destination.with_suffix(".csv.metadata.json")
    metadata = read_json(metadata_path)
    metadata.update(training_route="full_data_final_fit", independent_validation_AP=None,
                    training_manifest_sha256=joint["sha256"],
                    selection_report_sha256=sha256(report_path))
    write_json(metadata_path, metadata)
    state.update(status="READY_FOR_KAGGLE", finished=True, submission=str(destination),
                 checkpoint_sha256=sha, csv_sha256=metadata["validation"]["sha256"])
    write_json(state_path, state)
    print(f"READY_FOR_KAGGLE full_data_final_fit: {destination}", flush=True)
    return state
