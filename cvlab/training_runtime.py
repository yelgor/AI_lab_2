"""Shared training loop with bounded loss readback and observable pipeline cost."""

from itertools import islice
import time

import torch


class LossTotal:
    """Keep detached scalar statistics on-device without retaining autograd graphs."""

    def __init__(self, device):
        self.total = torch.zeros((), device=device, dtype=torch.float64)
        self.scalar_reads = 0
        self.finite_checks = 0

    def add(self, loss):
        self.total.add_(loss.detach().to(dtype=torch.float64))

    def check(self):
        self.finite_checks += 1
        if not bool(torch.isfinite(self.total)):
            raise RuntimeError(
                "Non-finite loss before optimizer update; weights were not stepped"
            )

    def value(self):
        self.scalar_reads += 1
        return self.total.item()


def train_batches(
    model,
    loader,
    optimizer,
    scaler,
    config,
    device,
    batches,
    *,
    on_sample=None,
    on_update=None,
    prefix="",
    log_every=250,
    loss_sync=None,
):
    """Perform the same normalized accumulation, clipping and AMP updates as train().

    per_batch is the previous readback policy, retained for direct A/B measurement.
    per_update checks detached loss BEFORE each optimizer update, including a final
    short accumulation group. GPU-to-host logging is otherwise limited to reports.
    """
    from .engine import amp_context, move, synchronize

    mode = loss_sync or config["runtime"].get("loss_sync", "per_update")
    if mode not in {"per_batch", "per_update"}:
        raise ValueError("loss_sync must be per_batch or per_update")
    accumulation = config["train"]["accumulation"]
    if accumulation < 1 or batches < 1:
        raise ValueError("Positive accumulation and batch limit required")
    loss_total = LossTotal(device)
    old_total = 0.0
    observed, images_seen, updates, stop = 0, 0, 0, False
    loader_wait = 0.0
    optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    started = previous_end = time.perf_counter()
    for index, (images, targets, metadata) in enumerate(islice(loader, batches)):
        loader_wait += time.perf_counter() - previous_end
        images, targets = move(images, targets, device)
        group_start = (index // accumulation) * accumulation
        group_size = min(accumulation, batches - group_start)
        with amp_context(config, device):
            losses = model.loss(images, targets)
            loss = sum(losses.values())
        if mode == "per_batch":
            loss_total.finite_checks += 1
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite loss before backward")
        scaler.scale(loss / group_size).backward()
        if mode == "per_batch":
            old_total += loss.item()
            loss_total.scalar_reads += 1
        else:
            loss_total.add(loss)
        observed += 1
        images_seen += len(images)
        if on_sample:
            on_sample(metadata)
        if log_every and observed % log_every == 0:
            mean = (old_total if mode == "per_batch" else loss_total.value()) / observed
            print(
                f"{prefix}batch={observed}/{batches} mean_loss={mean:.4f}", flush=True
            )
        if (index + 1) % accumulation == 0 or index + 1 == batches:
            if mode == "per_update":
                loss_total.check()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["train"]["clip_grad"]
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            if on_update and on_update(observed, batches):
                stop = True
                break
        previous_end = time.perf_counter()
    synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "loss_sum": old_total if mode == "per_batch" else loss_total.value(),
        "observed": observed,
        "batches": batches,
        "stop": stop,
        "train_timing": {
            "pipeline_seconds": elapsed,
            "images": images_seen,
            "images_per_second": images_seen / elapsed if elapsed else None,
            "loader_wait_seconds": loader_wait,
            "optimizer_update_boundaries": updates,
            "loss_scalar_reads": loss_total.scalar_reads,
            "loss_finite_checks": loss_total.finite_checks,
            "loss_sync": mode,
            "scope": "Full training pipeline including loader startup/wait, transfer, loss, backward, clipping and optimizer; excludes validation/checkpoint serialization. Loader wait may overlap GPU work; AMP may skip an update on gradient overflow.",
        },
    }


def make_optimizer(model, train_config):
    backbone, heads = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (backbone if "backbone" in name else heads).append(parameter)
    return torch.optim.AdamW(
        [
            {
                "params": backbone,
                "lr": train_config["lr"] * train_config["backbone_lr_factor"],
            },
            {"params": heads, "lr": train_config["lr"]},
        ],
        weight_decay=train_config["weight_decay"],
    )
