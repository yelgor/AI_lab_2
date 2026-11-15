"""Explicit precision control and full-precision raw-model resume for YOLO training."""

from copy import deepcopy
import math
from pathlib import Path

import torch
from ultralytics.utils.torch_utils import unwrap_model

from .yolo_backend import FullDataTrainer


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(item) for item in value)
    return deepcopy(value)


def require_finite(value):
    if isinstance(value, torch.nn.Module):
        require_finite(value.state_dict())
    elif isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise RuntimeError("Invalid numeric state; previous FP32 resume checkpoint preserved")
    elif isinstance(value, dict):
        for item in value.values():
            require_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            require_finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError("Invalid numeric state; previous FP32 resume checkpoint preserved")


class RecoveryTrainer(FullDataTrainer):
    freeze_batch_norm = False

    def check_resume(self, overrides):
        super().check_resume(overrides)
        # Upstream 8.3.250 does not forward amp in its resume override allowlist.
        if "amp" in overrides:
            self.args.amp = bool(overrides["amp"])

    def _model_train(self):
        super()._model_train()
        if self.freeze_batch_norm:
            for layer in self.model.modules():
                if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                    layer.eval()

    def setup_model(self):
        checkpoint = super().setup_model()
        if self.resume and checkpoint.get("cvlab_resume_format") == "raw-model-fp32-v1":
            # Stock loading prefers EMA; optimizer moments belong to the raw model.
            self.model.load_state_dict(checkpoint["model"].state_dict(), strict=True)
        return checkpoint

    def save_model(self):
        super().save_model()
        # Inference/epoch artifacts retain the original format. Only one extra
        # latest resume file retains raw weights, EMA and unquantized moments.
        payload = torch.load(self.last, map_location="cpu", weights_only=False)
        payload.update(
            model=deepcopy(unwrap_model(self.model)).cpu().float(),
            ema=deepcopy(unwrap_model(self.ema.ema)).cpu().float(),
            optimizer=cpu_state(self.optimizer.state_dict()),
            scaler=cpu_state(self.scaler.state_dict()),
            cvlab_resume_format="raw-model-fp32-v1",
        )
        for key in ("model", "ema", "optimizer", "scaler"):
            require_finite(payload[key])
        if payload["scaler"].get("scale", 1.0) <= 0:
            raise RuntimeError("Invalid scaler; previous FP32 resume checkpoint preserved")
        target = Path(self.wdir) / "resume_fp32.pt"
        temporary = target.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(target)
