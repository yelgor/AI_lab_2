"""Champion/challenger training: reject bad rounds, restore state, reduce the LR."""

import copy
import fcntl
import json
import os
from pathlib import Path

from .common import digest, read_json, write_json


def decision(base_fold, new_fold, base_official, new_official, lr, min_gain=0.001):
    gains = (
        new_fold - base_fold,
        None if new_official is None else new_official - base_official,
    )
    accepted = gains[1] is not None and min(gains) >= min_gain
    return {
        "accepted": accepted,
        "fold_gain": gains[0],
        "official_gain": gains[1],
        "next_lr": lr if accepted else lr * 0.5,
    }


def attempt_config(source, output, lr):
    config = copy.deepcopy(source)
    # Keep the original training split: added fold-validation clips must remain held out.
    config["output"] = str(output)
    config["train"].update(
        lr=lr,
        epochs=10000,
        max_hours=0.5,
        max_batches_per_epoch=None,
        freeze_batch_norm=True,
        early_stopping_patience=0,
    )
    config["runtime"]["cpu_threads"] = 4
    config["loader"].update(workers=2, prefetch_factor=1, persistent_workers=False)
    return config


def official_score(work, run, sha, limit=192):
    from .inference_experiments import balanced_subset
    from .prepare import select_records
    from .score_improvement import full_validation

    config = read_json(run / "config.json")
    config["data"]["split"] = "official"
    config["loader"].update(batch_size=1, workers=0, persistent_workers=False)
    records = balanced_subset(select_records(config, "val"), limit)
    seen = set(read_json(run / "split.json")["train_sequences"])
    if seen & {record["sequence_id"] for record in records}:
        raise ValueError("Official confirmation contains source-training sequences")
    path = (
        work / "adaptive_training" / f"official_{sha[:12]}_{digest(records)[:12]}.json"
    )
    if not path.is_file():
        value = full_validation(
            config, run / "best.pt", config["model"]["image_size"], records
        )
        value["checkpoint_sha256"] = sha
        write_json(path, value)
    result = read_json(path)
    if result["checkpoint_sha256"] != sha or result["data_fingerprint"] != digest(
        records
    ):
        raise ValueError("Official confirmation cache differs")
    return result["metrics"]["coco100"]["mAP"]


def run(work, attempts, batches, lock_fd, export_tiles=False, reset_to_original=False):
    from .engine import train
    from .score_improvement import file_hash, release_gpu
    from .validation_pipeline import ValidationPipeline

    work = Path(work).resolve()
    if attempts < 1 or batches < 1:
        raise ValueError("Positive attempts and batches required")
    expected, actual = (work / ".local_gpu_queue.lock").stat(), os.fstat(lock_fd)
    if (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino):
        raise ValueError("GPU lease belongs to another work directory")
    with os.fdopen(lock_fd, "a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory = work / "runs/rtdetr_adaptive_finetune"
        directory.mkdir(parents=True, exist_ok=True)
        state_path = work / "adaptive_training/state.json"
        if state_path.is_file():
            state = read_json(state_path)
        else:
            original = work / "runs/rtdetr_fold0_local"
            sha = file_hash(original / "best.pt")
            if not sha.startswith("a7cd89fb349d"):
                raise ValueError(
                    "Original public-best weights differ; preserve/restore the verified a7cd89fb349d run"
                )
            config = read_json(original / "config.json")
            if config["data"]["split"] != "fold" or config["model"]["name"] != "rtdetr":
                raise ValueError("Use the original RT-DETR fold checkpoint")
            state = {
                "champion": str(original),
                "champion_sha256": sha,
                "lr": 1e-5,
                "next_attempt": 1,
                "pending": None,
                "attempts": [],
            }
            write_json(state_path, state)
        if reset_to_original:
            original = work / "runs/rtdetr_fold0_local"
            sha = file_hash(original / "best.pt")
            if not sha.startswith("a7cd89fb349d"):
                raise ValueError("Original public-best checkpoint changed")
            if state["pending"]:
                state["next_attempt"] = max(
                    state["next_attempt"], state["pending"]["number"] + 1
                )
            state.update(
                champion=str(original),
                champion_sha256=sha,
                lr=max(1e-6, state["lr"] * 0.5),
                pending=None,
            )
            state["manual_resets"] = state.get("manual_resets", 0) + 1
            write_json(state_path, state)
            print(
                "Champion reset to verified original weights; all trial artifacts preserved",
                flush=True,
            )
        for _ in range(attempts):
            if (directory / "STOP").exists() or state["lr"] < 1e-6:
                print("Adaptive continuation stopped; champion preserved", flush=True)
                break
            champion = Path(state["champion"])
            if file_hash(champion / "best.pt") != state["champion_sha256"]:
                raise ValueError("Champion checkpoint changed")
            base_log = read_json(champion / "best_metrics.json")
            base_fold = base_log["metrics"]["coco100"]["mAP"]
            base_official = official_score(work, champion, state["champion_sha256"])
            pending = state["pending"] or {
                "number": state["next_attempt"],
                "batches": batches,
            }
            state["pending"] = pending
            write_json(state_path, state)
            output = directory / f"attempt_{pending['number']:03d}"
            config = attempt_config(
                read_json(champion / "config.json"), output, state["lr"]
            )
            last = output / "last.pt"
            completed = output / "history.jsonl"
            print(
                f"ATTEMPT {pending['number']}: source={state['champion_sha256'][:12]}; LR={state['lr']}; frames<={pending['batches']}; original fold held out",
                flush=True,
            )
            if not completed.is_file():
                if output.is_dir() and any(output.iterdir()) and not last.is_file():
                    import uuid

                    output.rename(
                        output.with_name(
                            output.name + "_incomplete_" + uuid.uuid4().hex[:6]
                        )
                    )
                with ValidationPipeline() as pipeline:
                    train(
                        config,
                        resume=last if last.is_file() else None,
                        init_checkpoint=None
                        if last.is_file()
                        else champion / "best.pt",
                        validation_pipeline=pipeline,
                        yield_after_epoch=True,
                        round_batches=pending["batches"],
                        warm_optimizer=not last.is_file(),
                        initial_validation_cache=None if last.is_file() else base_log,
                    )
            release_gpu()
            rows = [
                json.loads(line)
                for line in completed.read_text().splitlines()
                if line.strip()
            ]
            new_log = rows[-1]
            new_fold = new_log["metrics"]["coco100"]["mAP"]
            candidate_path = output / ("best.pt" if new_fold > base_fold else "last.pt")
            candidate_sha = file_hash(candidate_path)
            new_official = (
                official_score(work, output, candidate_sha)
                if new_fold - base_fold >= 0.001
                else None
            )
            result = decision(
                base_fold, new_fold, base_official, new_official, state["lr"]
            )
            result.update(
                number=pending["number"],
                source_sha256=state["champion_sha256"],
                candidate_sha256=candidate_sha,
                candidate_checkpoint=str(candidate_path),
                run=str(output),
                base_fold=base_fold,
                candidate_fold=new_fold,
                base_official=base_official,
                candidate_official=new_official,
            )
            state["attempts"].append(result)
            state["lr"] = result["next_lr"]
            state["pending"] = None
            state["next_attempt"] = pending["number"] + 1
            if result["accepted"]:
                state["champion"] = str(output)
                state["champion_sha256"] = candidate_sha
            write_json(state_path, state)
            official_gain_text = (
                "not checked"
                if result["official_gain"] is None
                else f"{result['official_gain']:.6f}"
            )
            print(
                f"{'ACCEPT' if result['accepted'] else 'ROLLBACK'}: fold gain={result['fold_gain']:.6f}; official gain={official_gain_text}; next LR={state['lr']}; champion={state['champion_sha256'][:12]}",
                flush=True,
            )
            if (
                result["accepted"]
                and export_tiles
                and not (directory / "STOP").exists()
            ):
                from .iterative_finetune import candidate_with_restored_budget
                from .tiled_search import run as tiled_run

                result["tile_check"] = candidate_with_restored_budget(
                    lambda: tiled_run(
                        work,
                        ["rtdetr"],
                        [960],
                        [0.6],
                        192,
                        6,
                        2,
                        os.dup(lease.fileno()),
                        min_gain=0.001,
                        runs=[output],
                    )
                )
                write_json(state_path, state)
        return state
