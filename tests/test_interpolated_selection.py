import importlib.util
from pathlib import Path
import sys

import torch

from cvlab import efficient_search
from cvlab.common import load_config, read_json
from cvlab.full_data import sha256


def test_interpolation_uses_actual_baseline_and_preserves_endpoint_identity(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "scripts/export_c1_interpolated_candidate.py"
    spec = importlib.util.spec_from_file_location("interpolation_export", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = load_config(Path(__file__).parents[1] / "configs/rtdetr.yaml")
    source, target = tmp_path / "source.pt", tmp_path / "target.pt"
    for checkpoint, weight in ((source, 1.0), (target, 2.0)):
        torch.save({"model": {"weight": torch.tensor([weight])}, "config": config,
                    "seen_training_sequences": ["training"]}, checkpoint)
    records = [{"id": i, "sequence_id": str(i)} for i in range(4)]
    monkeypatch.setattr(module, "select_records", lambda *a: records)
    monkeypatch.setattr(module, "setup", lambda *a: (_ for _ in ()).throw(AssertionError("Rejected native must skip more GPU work")))
    calls = []
    def score(work, checkpoint, cfg, data, *, tiled=False):
        assert not tiled and data == records
        calls.append(checkpoint)
        value = 0.2 if checkpoint == source else 0.21
        return {"identity": {"checkpoint_sha256": sha256(checkpoint)}, "metrics": {
            "coco100": {"mAP": value, "per_class": {"car": 0.4 if checkpoint == source else 0.38}},
            "per_video_mAP": {str(i): value for i in range(4)},
        }}
    monkeypatch.setattr(efficient_search, "_score", score)
    monkeypatch.setattr(sys, "argv", ["export", "--work", str(tmp_path), "--source", str(source),
                                      "--candidate", str(target)])
    module.main()
    checkpoint = next((tmp_path / "runs").glob("*/interpolated.pt"))
    fingerprint = sha256(checkpoint)
    report = read_json(checkpoint.parent / "report.json")
    assert report["native"]["baseline"]["identity"]["checkpoint_sha256"] == sha256(source)
    assert "class_regression" in report["native"]["decision"]["reasons"]
    assert report["csv"] is None and len(calls) == 2
    module.main()
    assert sha256(checkpoint) == fingerprint
    torch.save({"model": {"weight": torch.tensor([3.0])}, "config": config,
                "seen_training_sequences": ["training"]}, target)
    module.main()
    assert len(list((tmp_path / "runs").glob("*/interpolated.pt"))) == 2
    assert sha256(checkpoint) == fingerprint
