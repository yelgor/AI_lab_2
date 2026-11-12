"""Warm-start local LR racing with paired video evidence and full confirmation."""

import copy
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import time

from .common import digest, read_json, write_json
from .sequential_policy import (
    local_rates,
    paired_evidence,
    promotable,
    survivors,
    validation_records,
    video_family,
)

CONFIRMATION_PROTOCOL = "paired-family-class-v3"


def probe_config(source, output, lr, probe_images):
    config = copy.deepcopy(source)
    config["output"] = str(output)
    config["train"].update(
        lr=lr,
        epochs=10000,
        max_hours=48,
        max_batches_per_epoch=None,
        freeze_batch_norm=True,
        early_stopping_patience=0,
    )
    config["evaluation"].update(
        validation_images=probe_images,
        independent_video_families=True,
        per_video_metrics=True,
    )
    config["loader"].update(
        batch_size=1, workers=2, prefetch_factor=1, persistent_workers=False
    )
    config["runtime"]["cpu_threads"] = 4
    return config


def full_config(source, *, official=False):
    config = copy.deepcopy(source)
    for key in ("validation_images", "independent_video_families", "per_video_metrics"):
        config["evaluation"].pop(key, None)
    if official:
        config["data"]["split"] = "official"
    config["loader"].update(batch_size=1, workers=0, persistent_workers=False)
    for key in ("sampling", "temporal_bins"):
        config["loader"].pop(key, None)
    return config


def frozen_rung(trial, rung, row):
    """Preserve every measured state, even if a later training stage regresses."""
    destination = trial / f"rung_{rung + 1}"
    destination.mkdir(exist_ok=True)
    if not all(
        (destination / name).is_file()
        for name in ("best.pt", "config.json", "split.json", "best_metrics.json")
    ):
        temporary = destination / "best.tmp"
        shutil.copyfile(trial / "last.pt", temporary)
        for filename in ("config.json", "split.json"):
            shutil.copyfile(trial / filename, destination / filename)
        write_json(destination / "best_metrics.json", row)
        temporary.replace(destination / "best.pt")
    return destination


def _score(work, checkpoint, config, records, *, tiled=False):
    from dataclasses import asdict
    from .engine import load_trained, predict, setup
    from .evaluation_receipts import identity, load_receipt, save_receipt
    from .metrics import evaluate
    from .score_improvement import release_gpu
    from .sequential_policy import per_video_scores
    from .tiled_inference import TileOptions, predict_tiled

    options = TileOptions(960, overlap=0.2, nms_iou=0.6, edge_margin=2, workers=2)
    expected = identity(checkpoint, config, records,
                        options=asdict(options) if tiled else None)
    directory = config["evaluation"].get("cache_directory", str(Path(work) / "evaluation_cache"))
    cached = load_receipt(directory, expected)
    if cached is not None:
        print(f"EVALUATION CACHE HIT: {expected['checkpoint_sha256'][:12]}; tiled={tiled}", flush=True)
        return {**cached, "cache_hit": True}
    started = time.perf_counter()
    print(f"CONFIRM {'tiled' if tiled else 'native'}: sha={expected['checkpoint_sha256'][:12]}; images={len(records)}", flush=True)
    device = setup(config)
    model = load_trained(checkpoint, config, device)
    if tiled:
        outputs, timing = predict_tiled(model, config, records, device, options)
        predictions = outputs[0.6]
    else:
        predictions, timing = predict(model, config, records, device)
    metrics = evaluate(records, predictions, config["evaluation"]["max_detections"])
    metrics["per_video_mAP"] = per_video_scores(records, predictions)
    timing = {**timing, "evaluation_wall_seconds": time.perf_counter() - started}
    result = save_receipt(directory, expected, metrics, timing, origin="full_confirmation")
    del model, predictions
    if tiled:
        del outputs
    release_gpu()
    print(f"CONFIRM mAP={metrics['coco100']['mAP']:.6f}; seconds={timing['evaluation_wall_seconds']:.1f}", flush=True)
    return {**result, "cache_hit": False}


def confirm(work, candidate, champion, min_gain, export_tiles):
    from .engine import load_trained, setup
    from .prepare import select_records
    from .score_improvement import file_hash, release_gpu
    from .tiled_inference import TileOptions, export_tiled
    from .selection_policy import assess_metrics, public_rejection

    candidate, champion = Path(candidate), Path(champion)
    rejected = public_rejection(file_hash(candidate / "best.pt"))
    if rejected:
        return {"accepted": False, "confirmation_protocol": CONFIRMATION_PROTOCOL,
                "public_rejection": rejected}
    config = full_config(read_json(candidate / "config.json"))
    original = read_json(work / "runs/rtdetr_fold0_local/config.json")
    # A clip-disjoint historical fold can still contain related flight excerpts.
    # Include inherited training exposure recorded by newer engine versions.
    seen = set()
    for run in {candidate, champion, work / "runs/rtdetr_fold0_local"}:
        split = read_json(run / "split.json")
        seen.update(video_family(s) for s in split["train_sequences"])
        seen.update(video_family(s) for s in split.get("seen_training_sequences", []))
    raw_records = select_records(full_config(original), "val")
    records = [r for r in raw_records if video_family(r["sequence_id"]) not in seen]
    families = {video_family(r["sequence_id"]) for r in records}
    if len(families) < 4:
        raise ValueError("Full confirmation needs at least four unseen flight families")
    old = _score(work, champion / "best.pt", config, records)
    new = _score(work, candidate / "best.pt", config, records)
    native_gain = new["metrics"]["coco100"]["mAP"] - old["metrics"]["coco100"]["mAP"]
    result = {
        "accepted": False, "native_gain": native_gain,
        "confirmation_protocol": CONFIRMATION_PROTOCOL,
        "native_frames": len(records), "native_families": len(families),
        "native_excluded_related_frames": len(raw_records) - len(records),
        "native_data_fingerprint": digest(records),
        "native_per_class_gain": {
            name: value - old["metrics"]["coco100"].get("per_class", {}).get(name, value)
            for name, value in new["metrics"]["coco100"].get("per_class", {}).items()
            if value is not None
            and old["metrics"]["coco100"].get("per_class", {}).get(name) is not None
        },
    }
    result["native_decision"] = assess_metrics(old["metrics"], new["metrics"], min_gain=min_gain)
    if not result["native_decision"]["accepted"]:
        return result
    official = full_config(config, official=True)
    records = select_records(official, "val")
    if seen & {video_family(r["sequence_id"]) for r in records}:
        raise ValueError("Official confirmation includes training video families")
    old = _score(work, champion / "best.pt", official, records, tiled=True)
    new = _score(work, candidate / "best.pt", official, records, tiled=True)
    tiled_gain = new["metrics"]["coco100"]["mAP"] - old["metrics"]["coco100"]["mAP"]
    result["tiled_decision"] = assess_metrics(old["metrics"], new["metrics"], min_gain=min_gain)
    result.update(tiled_gain=tiled_gain, accepted=result["tiled_decision"]["accepted"])
    result["tiled_per_class_gain"] = {
        name: value - old["metrics"]["coco100"].get("per_class", {}).get(name, value)
        for name, value in new["metrics"]["coco100"].get("per_class", {}).items()
        if value is not None
        and old["metrics"]["coco100"].get("per_class", {}).get(name) is not None
    }
    if result["accepted"] and export_tiles:
        sha = file_hash(candidate / "best.pt")
        destination = work / "submissions" / f"rtdetr_{sha[:12]}_efficient_tiles.csv"
        device = setup(official)
        model = load_trained(candidate / "best.pt", official, device)
        options = TileOptions(960, overlap=0.2, nms_iou=0.6, edge_margin=2, workers=2)
        export_tiled(
            model,
            official,
            select_records(official, "test"),
            device,
            options,
            destination,
            sha,
        )
        del model
        release_gpu()
        result["submission"] = str(destination)
        print(f"READY FOR KAGGLE: {destination}", flush=True)
    return result


def observe(trial, rung):
    from .score_improvement import file_hash

    rows = [
        json.loads(line)
        for line in (trial / "history.jsonl").read_text().splitlines()
        if line.strip()
    ]
    row = next(row for row in rows if row["epoch"] == rung)
    initial = read_json(trial / "initial_validation.json")
    base = initial["metrics"]
    from .selection_policy import assess_metrics
    destination = frozen_rung(trial, rung, row)
    cumulative = sum(r["train_batches"] for r in rows if r["epoch"] <= rung)
    prior_elapsed = (
        initial["elapsed_seconds"]
        if rung == 0
        else next(r["elapsed_seconds"] for r in rows if r["epoch"] == rung - 1)
    )
    return {
        "trial": str(trial),
        "run": str(destination),
        "rung": rung,
        "checkpoint_sha256": file_hash(destination / "best.pt"),
        "batches": cumulative,
        "seconds_per_batch": max(0.001, row["elapsed_seconds"] - prior_elapsed)
        / max(1, row["train_batches"]),
        "global_gain": row["metrics"]["coco100"]["mAP"] - base["coco100"]["mAP"],
        "probe_mAP": row["metrics"]["coco100"]["mAP"],
        "per_video_mAP": row["metrics"]["per_video_mAP"],
        "metrics": row["metrics"],
        "probe_decision": assess_metrics(base, row["metrics"]),
        "evidence": paired_evidence(
            base["per_video_mAP"], row["metrics"]["per_video_mAP"]
        ),
    }


def rebase_observations(observations, champion_metrics):
    """After promotion, allocate against the new champion rather than an obsolete source."""
    for observation in observations:
        from .selection_policy import assess_metrics

        observation["probe_decision"] = assess_metrics(champion_metrics, observation.get("metrics", {}))
        observation.setdefault(
            "source_evidence", copy.deepcopy(observation["evidence"])
        )
        observation["evidence"] = paired_evidence(
            champion_metrics["per_video_mAP"], observation["per_video_mAP"]
        )
        observation["global_gain"] = (
            observation["probe_mAP"] - champion_metrics["coco100"]["mAP"]
        )


def run(
    work,
    lock_fd,
    *,
    hours=4,
    reserve_hours=1,
    cycles=1,
    budgets=(6000, 12000, 24000),
    probe_images=512,
    export_tiles=False,
    reset_to_original=False,
):
    from .engine import train
    from .prepare import select_records
    from .score_improvement import file_hash, release_gpu

    if (
        not math.isfinite(hours)
        or not math.isfinite(reserve_hours)
        or not 0 < reserve_hours < hours
        or cycles < 1
        or probe_images < 4
        or len(budgets) != 3
        or list(budgets) != sorted(set(budgets))
        or budgets[0] < 1
    ):
        raise ValueError(
            "Require finite hours > reserve > 0, positive cycles, 3 increasing budgets and a probe >=4"
        )
    work = Path(work).resolve()
    expected, actual = (work / ".local_gpu_queue.lock").stat(), os.fstat(lock_fd)
    if (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino):
        raise ValueError("GPU lease belongs to another work directory")
    deadline = time.monotonic() + hours * 3600
    search_deadline = deadline - reserve_hours * 3600
    state_path = work / "efficient_search/state.json"
    directory = work / "runs/rtdetr_efficient_search"
    settings = {"budgets": list(budgets), "probe_images": probe_images}
    with os.fdopen(lock_fd, "a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = work / "runs/rtdetr_fold0_local"
        original_sha = file_hash(original / "best.pt")
        if not original_sha.startswith("a7cd89fb349d"):
            raise ValueError("Verified public-best a7cd89fb349d checkpoint required")
        original_config = read_json(original / "config.json")
        if (
            original_config["data"]["split"] != "fold"
            or original_config["model"]["name"] != "rtdetr"
        ):
            raise ValueError("Use the original RT-DETR fold checkpoint")
        state = (
            read_json(state_path)
            if state_path.is_file()
            else {
                "version": 1,
                "settings": settings,
                "champion": str(original),
                "champion_sha256": original_sha,
                "center_lr": 1e-5,
                "radius": 0.5,
                "cycles": [],
            }
        )
        if state["version"] != 1 or state["settings"] != settings:
            raise ValueError(
                "Search settings changed; preserve the existing search state"
            )
        if reset_to_original:
            if state["cycles"] and not state["cycles"][-1].get("complete"):
                state["cycles"][-1].update(complete=True, stopped_by_user=True)
            state.update(
                champion=str(original),
                champion_sha256=original_sha,
                center_lr=1e-5,
                radius=max(0.125, state["radius"] / 2),
            )
            write_json(state_path, state)
            print(
                "Returned to public-best weights; previous trials preserved", flush=True
            )
        for _ in range(cycles):
            if time.monotonic() >= search_deadline or (directory / "STOP").exists():
                break
            cycle = (
                state["cycles"][-1]
                if state["cycles"] and not state["cycles"][-1].get("complete")
                else None
            )
            if cycle is None:
                source = Path(state["champion"])
                source_config = read_json(source / "config.json")
                cycle = {
                    "number": len(state["cycles"]) + 1,
                    "source": str(source),
                    "source_sha256": state["champion_sha256"],
                    "rung": 0,
                    "trials": [],
                    "active": [0, 1, 2],
                    "observations": [],
                    "confirmations": [],
                }
                for i, lr in enumerate(
                    local_rates(state["center_lr"], state["radius"])
                ):
                    output = directory / f"cycle_{cycle['number']:03d}" / f"trial_{i}"
                    config = probe_config(source_config, output, lr, probe_images)
                    identity = digest(
                        {k: v for k, v in config.items() if k != "output"}
                    )
                    for prior_cycle in state["cycles"]:
                        if prior_cycle["source_sha256"] == cycle["source_sha256"]:
                            for prior in prior_cycle["trials"]:
                                if prior.get("identity") == identity:
                                    config = prior["config"]
                    cycle["trials"].append(
                        {"lr": lr, "identity": identity, "config": config}
                    )
                training = select_records(cycle["trials"][0]["config"], "train")
                records = validation_records(cycle["trials"][0]["config"], training)
                if len({video_family(r["sequence_id"]) for r in records}) < 4:
                    raise ValueError(
                        "Need at least four unseen video families; no training started"
                    )
                cycle["probe_fingerprint"] = digest(records)
                del training, records
                state["cycles"].append(cycle)
                write_json(state_path, state)
            if file_hash(Path(cycle["source"]) / "best.pt") != cycle["source_sha256"]:
                raise ValueError("Source checkpoint changed")
            from .selection_policy import require_training_source

            require_training_source(cycle["source_sha256"])
            probe = cycle["trials"][0]["config"]
            training = select_records(probe, "train")
            records = validation_records(probe, training)
            if digest(records) != cycle["probe_fingerprint"]:
                raise ValueError("Held-out probe data changed")
            del training, records
            while cycle["rung"] < 3:
                rung = cycle["rung"]
                current = []
                for index in cycle["active"]:
                    previous = next(
                        (
                            o
                            for o in cycle["observations"]
                            if o["index"] == index and o["rung"] == rung
                        ),
                        None,
                    )
                    if previous:
                        from .selection_policy import assess_metrics

                        if "metrics" not in previous:
                            previous["metrics"] = read_json(Path(previous["run"]) / "best_metrics.json")["metrics"]
                        reference = read_json(Path(previous["trial"]) / "initial_validation.json")["metrics"]
                        if state["champion"] != cycle["source"]:
                            reference = read_json(Path(state["champion"]) / "best_metrics.json")["metrics"]
                        previous["probe_decision"] = assess_metrics(reference, previous["metrics"])
                        current.append(previous)
                        continue
                    if (
                        time.monotonic() >= search_deadline
                        or (directory / "STOP").exists()
                    ):
                        write_json(state_path, state)
                        print(
                            "Search paused at checkpoint boundary; champion preserved",
                            flush=True,
                        )
                        return state
                    config = cycle["trials"][index]["config"]
                    trial = Path(config["output"])
                    last = trial / "last.pt"
                    history = trial / "history.jsonl"
                    rows = (
                        [
                            json.loads(s)
                            for s in history.read_text().splitlines()
                            if s.strip()
                        ]
                        if history.is_file()
                        else []
                    )
                    completed = sum(
                        row["train_batches"] for row in rows if row["epoch"] < rung
                    )
                    extra = budgets[rung] - completed
                    print(
                        f"RACE cycle={cycle['number']} stage={rung + 1}/3 trial={index}; LR={config['train']['lr']:.3g}; cumulative budget={budgets[rung]} frames",
                        flush=True,
                    )
                    if not any(row["epoch"] == rung for row in rows):
                        if (
                            trial.exists()
                            and any(trial.iterdir())
                            and not last.is_file()
                        ):
                            import uuid

                            trial.rename(
                                trial.with_name(
                                    trial.name + "_incomplete_" + uuid.uuid4().hex[:6]
                                )
                            )
                        train(
                            config,
                            resume=last if last.is_file() else None,
                            init_checkpoint=None
                            if last.is_file()
                            else Path(cycle["source"]) / "best.pt",
                            warm_optimizer=not last.is_file(),
                            yield_after_epoch=True,
                            round_batches=extra,
                            round_seconds=max(1, search_deadline - time.monotonic()),
                        )
                    release_gpu()
                    observation = observe(trial, rung)
                    observation["index"] = index
                    cycle["observations"].append(observation)
                    current.append(observation)
                    write_json(state_path, state)
                    ev = observation["evidence"]
                    print(
                        f"PROBE trial={index} frames={observation['batches']}; global gain={observation['global_gain']:+.5f}; family gain={ev['mean_gain']:+.5f}; approximate 90% interval=[{ev['lower']:+.5f},{ev['upper']:+.5f}]",
                        flush=True,
                    )
                # Confirm at most the strongest eligible checkpoint per stage, allowing early CSVs.
                if (directory / "STOP").exists():
                    print(
                        "Search stopped at committed boundary; champion preserved",
                        flush=True,
                    )
                    return state
                if state["champion_sha256"] != cycle["source_sha256"]:
                    rebase_observations(
                        cycle["observations"],
                        read_json(Path(state["champion"]) / "best_metrics.json")[
                            "metrics"
                        ],
                    )
                confirmed = {
                    c["checkpoint_sha256"] for c in cycle["confirmations"]
                    if c.get("confirmation_protocol") == CONFIRMATION_PROTOCOL
                }
                eligible = sorted(
                    (
                        o
                        for o in cycle["observations"]
                        if promotable(o) and o["checkpoint_sha256"] not in confirmed
                    ),
                    key=lambda o: o["evidence"]["lower"],
                    reverse=True,
                )
                if eligible and deadline - time.monotonic() < reserve_hours * 3600:
                    print(
                        "Confirmation pending; reserve exhausted, resume this search to confirm saved weights",
                        flush=True,
                    )
                    return state
                if eligible:
                    candidate = eligible[0]
                    checked = next(
                        (
                            c
                            for c in cycle["confirmations"]
                            if c["checkpoint_sha256"] == candidate["checkpoint_sha256"]
                            and c.get("confirmation_protocol") == CONFIRMATION_PROTOCOL
                        ),
                        None,
                    )
                    if checked is None:
                        result = confirm(
                            work,
                            candidate["run"],
                            state["champion"],
                            0.001,
                            export_tiles,
                        )
                        checked = {
                            **result,
                            "checkpoint_sha256": candidate["checkpoint_sha256"],
                            "run": candidate["run"],
                            "lr": cycle["trials"][candidate["index"]]["lr"],
                        }
                        cycle["confirmations"].append(checked)
                        if checked["accepted"]:
                            state.update(
                                champion=checked["run"],
                                champion_sha256=checked["checkpoint_sha256"],
                                center_lr=checked["lr"],
                            )
                            rebase_observations(
                                cycle["observations"],
                                read_json(
                                    Path(state["champion"]) / "best_metrics.json"
                                )["metrics"],
                            )
                        write_json(state_path, state)
                    print(
                        f"{'ACCEPT' if checked['accepted'] else 'ROLLBACK'} full confirmation: {checked}",
                        flush=True,
                    )
                if rung < 2:
                    selected = survivors(current, 2 - rung, budgets[rung + 1])
                    cycle["active"] = [o["index"] for o in selected]
                cycle["rung"] += 1
                if not cycle["active"]:
                    cycle["rung"] = 3
                write_json(state_path, state)
            accepted = [c for c in cycle["confirmations"] if c["accepted"]]
            state["radius"] = (
                min(1.0, state["radius"] * 1.2)
                if accepted
                else max(0.125, state["radius"] / 2)
            )
            cycle["complete"] = True
            write_json(state_path, state)
        print(
            f"SEARCH FINISHED/PAUSED; champion={state['champion_sha256'][:12]}; state={state_path}",
            flush=True,
        )
        return state
