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
    from .score_improvement import file_hash, full_validation, release_gpu
    from .tiled_inference import TileOptions, predict_tiled

    sha = file_hash(checkpoint)
    options = TileOptions(960, overlap=0.2, nms_iou=0.6, edge_margin=2, workers=2)
    identity = {
        "checkpoint_sha256": sha,
        "data_fingerprint": digest(records),
        "model": config["model"],
        "runtime": config["runtime"],
        "data": {
            k: v for k, v in config["data"].items() if k not in ("root", "prepared")
        },
        "loader": config["loader"],
        "max_detections": config["evaluation"]["max_detections"],
        "options": asdict(options) if tiled else None,
    }
    path = work / "efficient_search/cache" / f"{digest(identity)}.json"
    if path.is_file():
        result = read_json(path)
        if result["identity"] != identity:
            raise ValueError("Confirmation cache identity differs")
        return result
    # Reuse the immutable original tiled result only with identical weights/data/options.
    original = work / "score_improvement/tiling_summary.json"
    source_config = read_json(work / "runs/rtdetr_fold0_local/config.json")
    compatible_original = (
        source_config["model"] == config["model"]
        and source_config["runtime"]["precision"] == config["runtime"]["precision"]
        and source_config["runtime"].get("tf32", False)
        == config["runtime"].get("tf32", False)
        and source_config["data"].get("ignore_policy")
        == config["data"].get("ignore_policy")
    )
    if tiled and compatible_original and original.is_file():
        for entry in read_json(original)["models"]:
            value = entry.get("full_candidate") or {}
            if (
                entry.get("checkpoint_sha256") == sha
                and value.get("checkpoint_sha256") == sha
                and value.get("data_fingerprint") == digest(records)
                and value.get("record_ids") == [r["id"] for r in records]
                and value.get("options") == asdict(options)
                and entry.get("image_size") == config["model"]["image_size"]
            ):
                result = {
                    "identity": identity,
                    "metrics": value["metrics"],
                    "reused": str(original),
                }
                write_json(path, result)
                return result
    print(
        f"CONFIRM {'tiled' if tiled else 'native'}: sha={sha[:12]}; images={len(records)}",
        flush=True,
    )
    if not tiled:
        result = full_validation(
            config, checkpoint, config["model"]["image_size"], records
        )
    else:
        from .engine import load_trained, setup
        from .metrics import evaluate

        device = setup(config)
        model = load_trained(checkpoint, config, device)
        outputs, timing = predict_tiled(model, config, records, device, options)
        result = {
            "metrics": evaluate(
                records, outputs[0.6], config["evaluation"]["max_detections"]
            ),
            "timing": timing,
        }
        del model, outputs
        release_gpu()
    result["identity"] = identity
    write_json(path, result)
    print(f"CONFIRM mAP={result['metrics']['coco100']['mAP']:.6f}", flush=True)
    return result


def confirm(work, candidate, champion, min_gain, export_tiles):
    from .engine import load_trained, setup
    from .prepare import select_records
    from .score_improvement import file_hash, release_gpu
    from .tiled_inference import TileOptions, export_tiled

    candidate, champion = Path(candidate), Path(champion)
    config = full_config(read_json(candidate / "config.json"))
    original = read_json(work / "runs/rtdetr_fold0_local/config.json")
    # Both contenders are evaluated with the same complete original holdout and precision.
    records = select_records(full_config(original), "val")
    old = _score(work, champion / "best.pt", config, records)
    new = _score(work, candidate / "best.pt", config, records)
    native_gain = new["metrics"]["coco100"]["mAP"] - old["metrics"]["coco100"]["mAP"]
    result = {"accepted": False, "native_gain": native_gain}
    if native_gain < min_gain:
        return result
    official = full_config(config, official=True)
    records = select_records(official, "val")
    seen = {
        video_family(s) for s in read_json(candidate / "split.json")["train_sequences"]
    }
    if seen & {video_family(r["sequence_id"]) for r in records}:
        raise ValueError("Official confirmation includes training video families")
    old = _score(work, champion / "best.pt", official, records, tiled=True)
    new = _score(work, candidate / "best.pt", official, records, tiled=True)
    tiled_gain = new["metrics"]["coco100"]["mAP"] - old["metrics"]["coco100"]["mAP"]
    result.update(tiled_gain=tiled_gain, accepted=tiled_gain >= min_gain)
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
        "evidence": paired_evidence(
            base["per_video_mAP"], row["metrics"]["per_video_mAP"]
        ),
    }


def rebase_observations(observations, champion_metrics):
    """After promotion, allocate against the new champion rather than an obsolete source."""
    for observation in observations:
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
                confirmed = {c["checkpoint_sha256"] for c in cycle["confirmations"]}
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
