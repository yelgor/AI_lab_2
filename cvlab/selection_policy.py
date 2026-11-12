"""Conservative candidate selection; pooled AP or weight direction alone is insufficient."""

import math
from pathlib import Path

from .common import read_json
from .sequential_policy import paired_evidence

PROTOCOL = "paired-family-class-v1"
DEFAULTS = {
    "min_families": 4,
    "min_win_fraction": 2 / 3,
    "max_family_drop": 0.01,
    "max_class_drop": 0.005,
    "max_small_drop": 0.003,
    "bootstrap_alpha": 0.1,
}


def finite_score(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def assess_metrics(base, candidate, *, min_gain=0.001):
    """Predeclared allocation guards, not a calibrated repeated-look significance test.

    Family AP is an equal-family auxiliary estimand, distinct from pooled COCO AP.
    Whole-flight bootstrap and leave-one-family-out means avoid frame pseudo-replication.
    """
    result = {
        "protocol": PROTOCOL,
        "accepted": False,
        "reasons": [],
        "thresholds": {**DEFAULTS, "min_gain": min_gain},
    }
    reasons = result["reasons"]
    old, new = base.get("coco100", {}), candidate.get("coco100", {})
    if not all(finite_score(m.get("mAP")) for m in (old, new)):
        reasons.append("missing_or_invalid_pooled_AP")
        return result
    result["pooled_gain"] = new["mAP"] - old["mAP"]
    if result["pooled_gain"] < min_gain:
        reasons.append("pooled_gain_below_minimum")

    before, after = old.get("per_class", {}), new.get("per_class", {})
    classes = {k for k, v in before.items() if v is not None}
    if (
        not classes
        or classes != {k for k, v in after.items() if v is not None}
        or any(not finite_score(m.get(k)) for m in (before, after) for k in classes)
    ):
        reasons.append("missing_or_incompatible_class_evidence")
    else:
        deltas = {k: after[k] - before[k] for k in sorted(classes)}
        result["per_class_gain"] = deltas
        if min(deltas.values()) < -DEFAULTS["max_class_drop"] - 1e-12:
            reasons.append("class_regression")
    if old.get("AP_small") is not None or new.get("AP_small") is not None:
        if not all(finite_score(m.get("AP_small")) for m in (old, new)):
            reasons.append("incompatible_small_object_evidence")
        else:
            result["small_gain"] = new["AP_small"] - old["AP_small"]
            if result["small_gain"] < -DEFAULTS["max_small_drop"] - 1e-12:
                reasons.append("small_object_regression")

    before, after = base.get("per_video_mAP", {}), candidate.get("per_video_mAP", {})
    if (
        not before
        or set(before) != set(after)
        or any(not finite_score(v) for m in (before, after) for v in m.values())
    ):
        reasons.append("missing_or_incompatible_family_evidence")
        return result
    differences = {k: after[k] - before[k] for k in sorted(before)}
    evidence = paired_evidence(before, after, alpha=DEFAULTS["bootstrap_alpha"])
    result.update(family_evidence=evidence, per_family_gain=differences)
    if len(differences) < DEFAULTS["min_families"]:
        reasons.append("too_few_families")
    if evidence["lower"] <= 0:
        reasons.append("family_lower_bound_not_positive")
    if evidence["positive_fraction"] + 1e-12 < DEFAULTS["min_win_fraction"]:
        reasons.append("too_few_improving_families")
    if min(differences.values()) < -DEFAULTS["max_family_drop"] - 1e-12:
        reasons.append("family_regression")
    if len(differences) > 1:
        total = sum(differences.values())
        result["leave_one_family_out_min_gain"] = min(
            (total - d) / (len(differences) - 1) for d in differences.values()
        )
        if result["leave_one_family_out_min_gain"] <= 0:
            reasons.append("gain_depends_on_one_family")
    result["accepted"] = not reasons
    return result


def public_rejection(checkpoint_sha, *, ledger=None):
    """Honor recorded negative feedback without inventing scores or submitting anything."""
    path = (
        Path(ledger)
        if ledger
        else Path(__file__).parents[1] / "experiments/leaderboard_results.json"
    )
    entries = read_json(path).get("observations", []) if path.is_file() else []
    for entry in entries:
        key = entry.get("checkpoint_sha256") or entry.get("checkpoint_sha256_prefix")
        if not key or not checkpoint_sha.startswith(key) or len(key) < 12:
            continue
        if (
            entry.get("blocked_training_source")
            or entry.get("regression_vs_best_public", 0) > 0
            or entry.get("approx_regression_vs_c1_public_best", 0) > 0
        ):
            return {
                "reason": "recorded_public_regression",
                "checkpoint_sha256": checkpoint_sha,
                "public_score": entry.get("public_score"),
                "file": entry.get("file"),
            }
    return None


def require_training_source(checkpoint_sha):
    rejection = public_rejection(checkpoint_sha)
    if rejection:
        raise ValueError(f"Refusing rejected training source: {rejection}")


def choose_checkpoint(config, output, metrics, previous_best):
    """Select robust weights during training, not only after pooled-AP best overwrites them."""
    score = metrics["coco100"]["mAP"]
    if not config["evaluation"].get("robust_selection", False):
        return score > previous_best, None
    output = Path(output)
    baseline = read_json(output / "initial_validation.json")["metrics"]
    decision = assess_metrics(baseline, metrics)
    if not decision["accepted"]:
        return False, decision
    prior = read_json(output / "best_metrics.json")["metrics"]
    old = assess_metrics(baseline, prior)

    def utility(item):
        return (item["family_evidence"]["lower"], item["pooled_gain"])

    return not old["accepted"] or utility(decision) > utility(old), decision
