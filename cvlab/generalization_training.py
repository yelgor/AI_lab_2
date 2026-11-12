"""Bounded continuation with full training passes and independent validation."""
import copy
import json
from pathlib import Path
import time

from .common import read_json, write_json


def history(output):
    path = Path(output) / "history.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len({r["epoch"] for r in rows}) != len(rows):
        raise ValueError("Duplicate training epochs; inspect history before continuation")
    return sorted(rows, key=lambda row: row["epoch"])


def next_block(completed, pool, maximum):
    if pool < 1 or maximum < 1 or completed < 0:
        raise ValueError("Invalid training frame budget")
    return min(6000 if completed == 0 else pool, max(0, maximum - completed))


def run(work, maximum=72000, hours=8, reserve_hours=2):
    """Caller must hold the shared GPU lease throughout training/confirmation."""
    from .engine import train, early_stopping_reached
    from .efficient_search import confirm, frozen_rung, probe_config
    from .prepare import select_records
    from .score_improvement import file_hash, release_gpu
    from .sequential_policy import paired_evidence, promotable

    if not 0 < reserve_hours < hours or maximum < 6000:
        raise ValueError("Require hours > reserve > 0 and maximum >= 6000")
    work = Path(work)
    output = work / "runs/rtdetr_sampling_flight_time"
    source = work / "runs/rtdetr_fold0_local"
    config = probe_config(read_json(source / "config.json"), output, 10 ** -5.5, 512)
    config["loader"].update(sampling="flight_time", temporal_bins=8)
    if (output / "config.json").exists() and read_json(output / "config.json") != config:
        raise ValueError("Existing sampling configuration differs; preserve the run")
    # Keep controller receipts outside the model directory: engine.train
    # deliberately rejects nonempty new run directories without last.pt.
    plan_path = work / "generalization_training/continuation.json"
    if plan_path.exists():
        plan = read_json(plan_path)
        if plan["maximum_frames"] != maximum or plan["source_sha256"] != file_hash(source / "best.pt"):
            raise ValueError("Continuation settings/source changed; preserve the saved plan")
        if plan.get("finished"):
            print("Continuation already finished; no repeated training", flush=True)
            return plan
    else:
        plan = {
            "maximum_frames": maximum, "source_sha256": file_hash(source / "best.pt"),
            "deadline": time.time() + hours * 3600, "reserve_seconds": reserve_hours * 3600,
            "target_kaggle_score": 0.28, "observed_kaggle_score": None,
            "control": str(work / "runs/rtdetr_efficient_search/cycle_001/trial_0/rung_1"),
            "note": "First 6000-frame block is the sampling ablation; later blocks add training exposure",
        }
    existing = history(output)
    for row in existing:
        if not (output / f"rung_{row['epoch'] + 1}/best.pt").exists():
            if row["epoch"] != existing[-1]["epoch"]:
                raise RuntimeError("An earlier measured rung is missing; inspect before more training")
            # The legacy one-block sampler did not freeze its rung. Preserve
            # those weights before another block can replace last.pt.
            frozen_rung(output, row["epoch"], row)
    records = select_records(config, "train")
    pool = len(records)
    del records
    plan["eligible_training_frames"] = pool
    write_json(plan_path, plan)
    stopping = copy.deepcopy(config)
    stopping["train"].update(early_stopping_patience=3, early_stopping_min_delta=0.001)
    while True:
        rows = history(output)
        # Recover a completed validation if the controller exited before freezing
        # its rung/committing the receipt. Do not repeat its training block.
        pending_epoch = plan.get("pending_epoch")
        committed = next((row for row in rows if row["epoch"] == pending_epoch), None)
        if committed is not None:
            if committed["epoch"] != rows[-1]["epoch"]:
                raise ValueError("Stale pending receipt; inspect checkpoint lineage")
            frozen_rung(output, committed["epoch"], committed)
            plan.pop("pending_batches", None)
            plan.pop("pending_epoch", None)
            write_json(plan_path, plan)
        completed = sum(row["train_batches"] for row in rows)
        plan["completed_frames"] = completed
        if completed >= maximum:
            plan["training_stop_reason"] = "frame_budget"
            break
        if (output / "STOP").exists():
            plan["status"] = "paused_at_checkpoint_boundary"
            write_json(plan_path, plan)
            return plan
        if time.time() >= plan["deadline"] - plan["reserve_seconds"]:
            plan["training_stop_reason"] = "wall_budget"
            break
        if early_stopping_reached(stopping, output):
            plan["training_stop_reason"] = "three_validation_checks_without_significant_AP_gain"
            break
        batches = plan.get("pending_batches") or next_block(completed, pool, maximum)
        plan.update(status="training", pending_batches=batches,
                    pending_epoch=rows[-1]["epoch"] + 1 if rows else 0)
        write_json(plan_path, plan)
        print(f"COVERAGE continuation={completed}/{maximum}; next={batches}; eligible={pool}", flush=True)
        last = output / "last.pt"
        train(config, resume=last if last.exists() else None,
              init_checkpoint=None if last.exists() else source / "best.pt",
              warm_optimizer=not last.exists(), yield_after_epoch=True,
              round_batches=batches,
              round_seconds=max(1, plan["deadline"] - plan["reserve_seconds"] - time.time()))
        release_gpu()
        after = history(output)
        if len(after) <= len(rows):
            raise RuntimeError("Training returned without a new measured block; no restart loop")
        row = after[-1]
        frozen_rung(output, row["epoch"], row)
        plan.pop("pending_batches", None)
        plan.pop("pending_epoch", None)
        plan["completed_frames"] = sum(row["train_batches"] for row in after)
        write_json(plan_path, plan)
    rows = history(output)
    if rows:
        # Materialize a legacy completed first block too, if it predates this controller.
        best = max(rows, key=lambda row: row["metrics"]["coco100"]["mAP"])
        candidate = output / f"rung_{best['epoch'] + 1}"
        if not (candidate / "best.pt").exists():
            # Only last.pt can be frozen as the last measured row; never mislabel it.
            if best["epoch"] != rows[-1]["epoch"]:
                raise RuntimeError("Best historical rung is missing; preserve weights for inspection")
            candidate = frozen_rung(output, best["epoch"], best)
        central_path = work / "efficient_search/state.json"
        central = read_json(central_path)
        champion = Path(central["champion"])
        baseline = read_json(champion / "best_metrics.json")["metrics"]
        # Comparisons require the exact same probe, not just matching family names.
        candidate_split = read_json(candidate / "split.json")
        champion_split = read_json(champion / "split.json")
        matching_probe = candidate_split["val_ids"] == champion_split["val_ids"]
        if file_hash(champion / "best.pt") == plan["source_sha256"]:
            baseline = read_json(output / "initial_validation.json")["metrics"]
            matching_probe = True
        if not matching_probe:
            plan.update(status="needs_matching_probe_confirmation", candidate=str(candidate))
        else:
            evidence = paired_evidence(baseline["per_video_mAP"], best["metrics"]["per_video_mAP"])
            gain = best["metrics"]["coco100"]["mAP"] - baseline["coco100"]["mAP"]
            plan.update(candidate=str(candidate), probe_gain=gain, evidence=evidence)
            if promotable({"evidence": evidence, "global_gain": gain}):
                plan["status"] = "confirming"
                write_json(plan_path, plan)
                checked = confirm(work, candidate, champion, 0.001, True)
                plan["confirmation"] = checked
                if checked["accepted"]:
                    sha = file_hash(candidate / "best.pt")
                    central.setdefault("external_promotions", []).append({
                        **checked, "run": str(candidate), "checkpoint_sha256": sha,
                        "completed_training_frames": plan["completed_frames"],
                    })
                    central.update(champion=str(candidate), champion_sha256=sha,
                                   center_lr=config["train"]["lr"])
                    write_json(central_path, central)
            else:
                plan["status"] = "not_promoted_by_probe"
    if plan.get("status") == "confirming":
        plan["status"] = "accepted" if plan["confirmation"]["accepted"] else "rejected_by_full_validation"
    elif plan.get("status") == "training":
        plan["status"] = "finished_without_candidate"
    plan.update(finished=True, status=plan.get("status", "finished"))
    write_json(plan_path, plan)
    print(f"CONTINUATION FINISHED: {plan_path}; frames={plan['completed_frames']}; {plan.get('training_stop_reason')}", flush=True)
    return plan
