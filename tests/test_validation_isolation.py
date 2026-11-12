import pytest

from cvlab.prepare import group_folds
from cvlab.sequential_policy import assert_family_isolation, video_family


def test_folds_keep_all_related_excerpts_together():
    records = [
        {"id": i, "sequence_id": seq}
        for i, seq in enumerate([
            "uav1_a", "uav1_b", "uav2_a", "uav2_b", "uav3_a", "uav4_a"
        ])
    ]
    result = group_folds(records, 3, 42)
    assert result == group_folds(records, 3, 42)
    sets = [{video_family(s) for s in f["val_sequences"]} for f in result["folds"]]
    assert not sets[0] & sets[1] and not sets[1] & sets[2] and not sets[0] & sets[2]
    assert set.union(*sets) == {"uav1", "uav2", "uav3", "uav4"}
    assert sum(len(f["val_sequences"]) for f in result["folds"]) == 6


def test_too_few_families_is_rejected_even_with_many_clips():
    records = [{"id": i, "sequence_id": f"uav1_{i}"} for i in range(9)]
    with pytest.raises(ValueError, match="independent flight"):
        group_folds(records, 3, 42)


def test_inherited_training_exposure_cannot_be_used_as_validation():
    train = [{"sequence_id": "uav1_a"}]
    val = [{"sequence_id": "uav2_b"}]
    assert_family_isolation(train, val)
    with pytest.raises(ValueError, match="inherited weights"):
        assert_family_isolation(train, val, ["uav2_a"])
    with pytest.raises(ValueError, match="flight-family leakage"):
        assert_family_isolation(train, [{"sequence_id": "uav1_b"}])


def test_alternate_folds_do_not_replace_legacy_assignments(tmp_path):
    from cvlab.common import write_json
    from cvlab.prepare import select_records

    records = [{"id": i, "sequence_id": seq, "frame_id": 1}
               for i, seq in enumerate(["uav1_a", "uav1_b", "uav2_a"])]
    write_json(tmp_path / "train.json", {"images": records})
    write_json(tmp_path / "folds.json", {"folds": [{"val_sequences": ["uav1_a"]}]})
    write_json(tmp_path / "family.json", {"folds": [{"val_sequences": ["uav1_a", "uav1_b"]}]})
    config = {"data": {"prepared": str(tmp_path), "split": "fold", "fold": 0,
                       "folds_file": "family.json"}}
    assert [r["id"] for r in select_records(config, "val")] == [0, 1]
    assert [r["id"] for r in select_records(config, "train")] == [2]
    config["data"].pop("folds_file")
    assert [r["id"] for r in select_records(config, "val")] == [0]
