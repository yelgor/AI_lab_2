import copy

import pytest

from cvlab.common import write_json
from cvlab.selection_policy import assess_metrics, public_rejection, choose_checkpoint


def scores(gain=0, families=7):
    return {
        "coco100": {
            "mAP": 0.2 + gain,
            "AP_small": 0.08 + gain,
            "per_class": {
                "person": 0.3 + gain,
                "car": 0.4 + gain,
                "truck": 0.2 + gain,
                "bus": 0.04 + gain,
            },
        },
        "per_video_mAP": {str(i): 0.2 + gain for i in range(families)},
    }


def test_broad_class_and_flight_improvement_passes():
    result = assess_metrics(scores(), scores(0.01))
    assert result["accepted"] and result["family_evidence"]["lower"] == pytest.approx(
        0.01
    )


def test_pooled_gain_cannot_hide_person_car_regression():
    # Synthetic replay of the observed pattern, not reconstructed raw predictions.
    candidate = scores(0.014)
    candidate["coco100"]["per_class"].update(
        person=0.2946, car=0.3907, truck=0.2543, bus=0.0960
    )
    result = assess_metrics(scores(), candidate)
    assert not result["accepted"] and "class_regression" in result["reasons"]


def test_one_flight_gain_cannot_promote_pooled_AP():
    candidate = scores(0.01)
    candidate["per_video_mAP"] = {str(i): 0.3 if i == 0 else 0.2 for i in range(7)}
    result = assess_metrics(scores(), candidate)
    assert not result["accepted"]
    assert "too_few_improving_families" in result["reasons"]
    assert "gain_depends_on_one_family" in result["reasons"]


@pytest.mark.parametrize(
    "change,reason",
    [
        (lambda m: m.pop("per_video_mAP"), "missing_or_incompatible_family_evidence"),
        (
            lambda m: m["coco100"].pop("per_class"),
            "missing_or_incompatible_class_evidence",
        ),
        (
            lambda m: m["coco100"].update(mAP=float("nan")),
            "missing_or_invalid_pooled_AP",
        ),
        (lambda m: m["coco100"].update(AP_small=0.07), "small_object_regression"),
        (lambda m: m["per_video_mAP"].update({"0": 0.18}), "family_regression"),
    ],
)
def test_incomplete_or_harmful_evidence_fails_closed(change, reason):
    candidate = scores(0.01)
    change(candidate)
    result = assess_metrics(scores(), candidate)
    assert not result["accepted"] and reason in result["reasons"]


def test_public_outcome_is_checked_by_identity_not_filename(tmp_path):
    ledger = tmp_path / "outcomes.json"
    write_json(
        ledger,
        {
            "observations": [
                {
                    "checkpoint_sha256_prefix": "123456789abc",
                    "public_score": 0.19,
                    "blocked_training_source": True,
                }
            ]
        },
    )
    assert public_rejection("123456789abc" + "d" * 52, ledger=ledger)
    assert public_rejection("e" * 64, ledger=ledger) is None


def test_training_keeps_broad_improvement_instead_of_later_pooled_maximum(tmp_path):
    config = {"evaluation": {"robust_selection": True}}
    write_json(tmp_path / "initial_validation.json", {"metrics": scores()})
    write_json(tmp_path / "best_metrics.json", {"metrics": scores(0.005)})
    bad = scores(0.04)
    bad["coco100"]["per_class"]["car"] = 0.38
    assert not choose_checkpoint(config, tmp_path, bad, 0.205)[0]
    good = scores(0.01)
    assert choose_checkpoint(config, tmp_path, good, 0.205)[0]
    assert not choose_checkpoint(config, tmp_path, copy.deepcopy(scores()), 0.205)[0]
