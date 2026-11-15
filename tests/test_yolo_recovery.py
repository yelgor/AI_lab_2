import pytest
import torch

from cvlab.common import read_json
from cvlab.yolo_training import train_challenger
from test_pipeline import competition  # noqa: F401


def test_fp32_checkpoint_resumes_raw_weights_moments_and_explicit_precision(
    competition, tmp_path, monkeypatch  # noqa: F811
):
    pytest.importorskip("ultralytics")
    from cvlab.yolo_recovery import RecoveryTrainer

    monkeypatch.setattr("ultralytics.data.utils.check_font", lambda *a, **kw: None)
    root, prepared = competition
    work = tmp_path / "work"
    work.mkdir()
    torch.set_num_threads(1)
    saved_trainers = []
    save_model = RecoveryTrainer.save_model

    def capture_save(self):
        save_model(self)
        saved_trainers.append(self)

    monkeypatch.setattr(RecoveryTrainer, "save_model", capture_save)
    options = dict(name="precision", epochs=2, batch=2, image_size=64, workers=0,
                   device="cpu", amp=False, initial_model="yolo11n.yaml", freeze_batch_norm=True)
    run = train_challenger(work, root, prepared, hours=1e-9, **options)
    path = run / "weights/resume_fp32.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["cvlab_resume_format"] == "raw-model-fp32-v1"
    assert saved["epoch"] == 0
    raw, ema = saved["model"].state_dict(), saved["ema"].state_dict()
    assert any(not torch.equal(raw[k], ema[k]) for k in raw)
    floating = [t for state in saved["optimizer"]["state"].values() for t in state.values()
                if isinstance(t, torch.Tensor) and t.is_floating_point()]
    assert floating and all(t.dtype == torch.float32 for t in floating)
    assert any(not torch.equal(t, t.half().float()) for t in floating)
    previous_bytes = path.read_bytes()
    # A bad EMA must not destroy the last valid full-precision recovery state.
    ema_tensor = next(saved_trainers[-1].ema.ema.parameters())
    old_value = ema_tensor.reshape(-1)[0].clone()
    ema_tensor.reshape(-1)[0] = float("nan")
    with pytest.raises(RuntimeError, match="previous FP32 resume checkpoint preserved"):
        save_model(saved_trainers[-1])
    assert path.read_bytes() == previous_bytes
    ema_tensor.reshape(-1)[0] = old_value
    # Simulate an old checkpoint requesting AMP: explicit caller FP32 must win
    # even though the upstream resume allowlist does not include this argument.
    saved["train_args"]["amp"] = True
    torch.save(saved, path)
    original = RecoveryTrainer.resume_training
    seen = []

    def checked_resume(self, checkpoint):
        original(self, checkpoint)
        if checkpoint and checkpoint.get("cvlab_resume_format"):
            assert self.args.amp is False
            assert all(torch.equal(value.cpu(), raw[key]) for key, value in self.model.state_dict().items())
            for key, state in self.optimizer.state_dict()["state"].items():
                for name, value in state.items():
                    if isinstance(value, torch.Tensor):
                        assert torch.equal(value.cpu(), saved["optimizer"]["state"][key][name])
            seen.append(True)

    monkeypatch.setattr(RecoveryTrainer, "resume_training", checked_resume)
    train_challenger(work, root, prepared, hours=1, resume=True, **options)
    assert seen
    continued = torch.load(path, map_location="cpu", weights_only=False)
    assert continued["epoch"] == 1 and continued["updates"] > saved["updates"]
    assert continued["train_args"]["amp"] is False
    assert len((run / "common_history.jsonl").read_text().splitlines()) == 2
    assert read_json(run / "yolo_plan.json")["freeze_batch_norm"] is True

    # A new recipe reuses all learned weights and explicitly starts a new optimizer.
    monkeypatch.setattr(RecoveryTrainer, "resume_training", original)
    warm = train_challenger(work, root, prepared, name="warm", epochs=2, hours=1e-9,
                            batch=1, image_size=64, workers=0, device="cpu", amp=False,
                            init_checkpoint=run / "weights/common_best.pt", lr=1e-4,
                            freeze_batch_norm=True)
    assert read_json(warm / "initialization.json")["optimizer_restored"] is False
    assert read_json(warm / "initial_validation.json")["frames"] == 2
    assert read_json(warm / "yolo_plan.json")["initial_model"] == "learned_eight_class_checkpoint"
    assert read_json(warm / "yolo_plan.json")["arguments"]["lr0"] == 1e-4
    from cvlab.common import write_json
    provenance = work / "yolo_data/prepared.json"
    info = read_json(provenance)
    info["families"]["train"] += info["families"]["val"]
    write_json(provenance, info)
    with pytest.raises(ValueError, match="trained on held-out"):
        train_challenger(work, root, prepared, name="leaky", epochs=2, hours=1,
                         batch=1, image_size=64, workers=0, device="cpu", amp=False,
                         init_checkpoint=run / "weights/common_best.pt", lr=1e-4)


def test_precision_policy_audit_does_not_break_a_second_resume(competition, tmp_path, monkeypatch):  # noqa: F811
    """Metadata from a prior precision transition is not a recipe difference."""
    pytest.importorskip("ultralytics")
    from cvlab.yolo_recovery import RecoveryTrainer
    from cvlab.common import write_json

    monkeypatch.setattr("ultralytics.data.utils.check_font", lambda *a, **kw: None)
    root, prepared = competition
    work = tmp_path / "work"
    work.mkdir()
    torch.set_num_threads(1)
    options = dict(name="audit", epochs=2, batch=2, image_size=64, workers=0,
                   device="cpu", amp=False, initial_model="yolo11n.yaml")
    run = train_challenger(work, root, prepared, hours=1e-9, **options)
    plan = read_json(run / "yolo_plan.json")
    plan["resume_precision_changes"] = [{"field": "amp", "from": True, "to": False}]
    write_json(run / "yolo_plan.json", plan)
    train_challenger(work, root, prepared, hours=1, resume=True, **options)
    assert read_json(run / "yolo_plan.json")["resume_precision_changes"] == plan["resume_precision_changes"]
    assert (run / "weights/resume_fp32.pt").is_file()
    assert RecoveryTrainer.freeze_batch_norm is False
