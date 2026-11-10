"""Video-level paired evidence and budgeted successive halving (no GPU required)."""

from collections import defaultdict
import math

import numpy as np


def video_family(sequence):
    # VisDrone excerpts uav0000013_00000_v / uav0000013_01000_v share a flight.
    return sequence.split("_", 1)[0] if sequence.startswith("uav") else sequence


def validation_records(config, training):
    from .prepare import select_records
    from .inference_experiments import balanced_subset

    records = select_records(config, "val")
    evaluation = config["evaluation"]
    if evaluation.get("independent_video_families", False):
        seen = {video_family(r["sequence_id"]) for r in training}
        records = [r for r in records if video_family(r["sequence_id"]) not in seen]
        if not records:
            raise ValueError("No unseen video families remain for validation")
    limit = evaluation.get("validation_images")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("validation_images must be a positive integer")
    return balanced_subset(records, limit)


def per_video_scores(records, predictions):
    from .metrics import evaluate

    groups, outputs = defaultdict(list), defaultdict(list)
    for record in records:
        groups[video_family(record["sequence_id"])].append(record)
    lookup = {r["id"]: video_family(r["sequence_id"]) for r in records}
    for prediction in predictions:
        outputs[lookup[prediction["image_id"]]].append(prediction)
    result = {}
    for family, group in sorted(groups.items()):
        score = evaluate(group, outputs[family], [100])["coco100"]["mAP"]
        if score is not None:
            result[family] = score
    return result


def paired_evidence(base, candidate, *, alpha=0.1, draws=2000, seed=42):
    """Approximate paired family-bootstrap interval for macro family AP gain.

    This is an allocation heuristic, not a guarantee on global COCO/Kaggle AP.
    Repeated looks and hyperparameter selection invalidate a nominal test claim.
    """
    if set(base) != set(candidate) or not base:
        raise ValueError("Paired evidence needs exactly the same nonempty families")
    if not 0 < alpha < 1 or draws < 100:
        raise ValueError("Invalid bootstrap settings")
    differences = np.array([candidate[k] - base[k] for k in sorted(base)])
    if not np.isfinite(differences).all():
        raise ValueError("Non-finite family scores")
    rng = np.random.default_rng(seed)
    means = differences[
        rng.integers(len(differences), size=(draws, len(differences)))
    ].mean(axis=1)
    lower, upper = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {
        "families": len(base),
        "mean_gain": float(differences.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "alpha": alpha,
        "positive_fraction": float((differences > 0).mean()),
        "auxiliary_estimand": "equal-weight mean of per-video-family COCO100 AP gains",
    }


def local_rates(center, radius=0.5):
    if not math.isfinite(center) or center <= 0 or not 0 < radius <= 1:
        raise ValueError("Positive LR and log10 radius in (0,1] required")
    return [center * 10**offset for offset in (-radius, 0, radius)]


def survivors(observations, keep, next_budget, min_gain=0.001):
    """Prune demonstrably harmful trials, then rank optimistic gain per added second."""
    eligible = [
        item
        for item in observations
        if item["evidence"]["families"] >= 4 and item["evidence"]["upper"] > -min_gain
    ]

    def utility(item):
        extra_batches = max(1, next_budget - item["batches"])
        seconds = extra_batches * item["seconds_per_batch"] + item.get(
            "validation_seconds", 0
        )
        return max(0.0, item["evidence"]["upper"] - min_gain) / max(seconds, 1e-9)

    return sorted(
        eligible, key=lambda x: (utility(x), x["evidence"]["mean_gain"]), reverse=True
    )[:keep]


def promotable(observation, min_gain=0.001):
    return (
        observation["evidence"]["families"] >= 4
        and observation["evidence"]["lower"] > 0
        and observation["global_gain"] >= min_gain
    )
