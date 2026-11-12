from cvlab.training_coverage import TrainingCoverage


def test_actual_frame_repeats_and_family_grouping():
    coverage = TrainingCoverage()
    a = {"record": {"id": 7, "sequence_id": "uav0000013_00000_v", "labels": [0, 0, 2]}}
    b = {"record": {"id": 9, "sequence_id": "uav0000013_01000_v", "labels": []}}
    coverage.update([a, b, a])
    report = coverage.report(10)
    assert report["presentations"] == 3
    assert report["unique_frames"] == 2
    assert report["repeated_presentations"] == 1
    assert report["coverage_fraction"] == 0.2
    assert report["family_presentations"] == {"uav0000013": 3}
    assert report["source_class_frame_presentations"] == {"0": 2, "2": 2}
    assert report["frame_ids"] == [7, 9]
