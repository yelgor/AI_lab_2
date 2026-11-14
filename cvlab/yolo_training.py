"""A fixed YOLO challenger recipe with streamed dataset preparation and common evaluation."""

from contextlib import contextmanager
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

import yaml

from .common import CLASSES, digest, read_json, write_json
from .postprocess_experiments import stream_array
from .prepare import safe_path
from .sequential_policy import video_family


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_dataset(root, prepared, destination):
    """Use image symlinks and small YOLO labels; never duplicate the image archive."""
    root, prepared, destination = map(
        lambda p: Path(p).resolve(), (root, prepared, destination)
    )
    identity = {
        "root": str(root),
        "manifests": {s: file_sha(prepared / f"{s}.json") for s in ("train", "val")},
    }
    ready = destination / "prepared.json"
    if ready.exists():
        if read_json(ready)["identity"] != identity:
            raise ValueError(
                "Prepared YOLO dataset changed; preserve it and use a new destination"
            )
        return destination / "data.yaml"
    counts, families, ignored, sequences = {}, {}, {}, {}
    for split in ("train", "val"):
        images, labels = destination / "images" / split, destination / "labels" / split
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        ids, families[split], sequences[split] = set(), set(), set()
        for record in stream_array(prepared / f"{split}.json", "images"):
            key = record["id"]
            if key in ids:
                raise ValueError("Duplicate image ID in source manifest")
            ids.add(key)
            source = safe_path(root, record["file_name"]).resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            link = images / f"{key:08d}{source.suffix.lower()}"
            if link.is_symlink():
                if link.resolve() != source:
                    raise ValueError("Existing image link has different source")
            elif link.exists():
                raise ValueError("Refusing to overwrite a materialized dataset image")
            else:
                link.symlink_to(source)
            width, height = record["width"], record["height"]
            lines = []
            for box, label in zip(record["boxes"], record["labels"], strict=True):
                x1, y1, x2, y2 = box
                if (
                    not 0 <= label < len(CLASSES)
                    or not all(math.isfinite(v) for v in box)
                    or not 0 <= x1 < x2 <= width
                    or not 0 <= y1 < y2 <= height
                ):
                    raise ValueError("Invalid prepared target box/class")
                lines.append(
                    f"{label} {(x1 + x2) / (2 * width):.9g} {(y1 + y2) / (2 * height):.9g} {(x2 - x1) / width:.9g} {(y2 - y1) / height:.9g}"
                )
            text = "\n".join(lines) + ("\n" if lines else "")
            label_path = labels / f"{key:08d}.txt"
            if label_path.exists() and label_path.read_text() != text:
                raise ValueError(
                    "Existing label contents differ; preserve the dataset version"
                )
            label_path.write_text(text)
            sequences[split].add(record["sequence_id"])
            families[split].add(video_family(record["sequence_id"]))
            if split == "train" and record["ignore_boxes"]:
                ignored[link.name] = [
                    [x1 / width, y1 / height, x2 / width, y2 / height]
                    for x1, y1, x2, y2 in record["ignore_boxes"]
                ]
        counts[split] = len(ids)
    if not all(counts.values()) or families["train"] & families["val"]:
        raise ValueError("Require nonempty, flight-disjoint train and validation")
    write_json(destination / "ignored_train.json", ignored)
    data = {
        "path": str(destination),
        "train": "images/train",
        "val": "images/val",
        "names": CLASSES,
    }
    (destination / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    write_json(
        ready,
        {
            "identity": identity,
            "frames": counts,
            "families": {s: sorted(v) for s, v in families.items()},
            "sequences": {s: sorted(v) for s, v in sequences.items()},
        },
    )
    return destination / "data.yaml"


def common_config(root, prepared, run, image_size=1280, device="cuda:0", workers=2):
    return {
        "seed": 42,
        "output": str(run),
        "data": {
            "root": str(root),
            "prepared": str(prepared),
            "split": "official",
            "frame_step": 1,
            "ignore_policy": "mask",
        },
        "model": {
            "name": "yolo11s",
            "image_size": image_size,
            "max_detections": 500,
            "score_threshold": 0.001,
            "nms_threshold": 0.6,
        },
        "loader": {
            "batch_size": 1,
            "workers": workers,
            "prefetch_factor": 1,
            "persistent_workers": False,
            "pin_memory": device != "cpu",
        },
        "runtime": {
            "device": device,
            "precision": "fp32" if device == "cpu" else "fp16",
            "cpu_threads": 2,
            "tf32": False,
        },
        "evaluation": {"max_detections": [100, 500], "warmup_batches": 0},
        "train": {"max_batches_per_epoch": None},
    }


def evaluate_checkpoint(checkpoint, config):
    """Evaluate serialized weights through the same loader/ignore rules/COCO evaluator."""
    from ultralytics import YOLO
    from .engine import predict, setup
    from .metrics import evaluate
    from .prepare import select_records
    from .yolo_backend import Predictor
    from .sequential_policy import per_video_scores

    device = setup(config)
    core = YOLO(str(checkpoint)).model.to(device)
    records = select_records(config, "val")
    predictions, timing = predict(
        Predictor(core, config),
        config,
        records,
        device,
        worker_context="spawn" if config["loader"]["workers"] else None,
    )
    metrics = evaluate(records, predictions, [100, 500])
    metrics["per_video_mAP"] = per_video_scores(records, predictions)
    del core, predictions
    return {
        "checkpoint_sha256": file_sha(checkpoint),
        "frames": len(records),
        "metrics": metrics,
        "timing": timing,
        "data_fingerprint": digest(records),
    }


def export_checkpoint(checkpoint, config, destination):
    from ultralytics import YOLO
    from .engine import setup
    from .prepare import select_records
    from .tiled_inference import export_tiled, TileOptions
    from .yolo_backend import Predictor

    device = setup(config)
    model = Predictor(YOLO(str(checkpoint)).model.to(device), config)
    return export_tiled(
        model,
        config,
        select_records(config, "test"),
        device,
        TileOptions(tile_side=None, nms_iou=0.6, workers=config["loader"]["workers"]),
        destination,
        file_sha(checkpoint),
        announce_ready=False,
    )


def configure_optional_backend(work):
    """Keep optional settings/caches local and avoid credential-driven logger uploads."""
    cache = Path(work) / "cache" / "ultralytics"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(cache)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    from ultralytics import settings

    settings.update(
        {
            key: False
            for key in (
                "sync",
                "hub",
                "wandb",
                "mlflow",
                "comet",
                "clearml",
                "neptune",
                "dvc",
                "tensorboard",
            )
            if key in settings
        }
    )


def atomic_copy(source, target):
    target = Path(target)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)


@contextmanager
def isolated_evaluation():
    """Evaluation/export setup must not reseed or retune the ongoing trainer."""
    import torch
    from .engine import rng_state, restore_rng

    generator = torch.Generator()
    state, threads = rng_state(generator), torch.get_num_threads()
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    deterministic = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    try:
        yield
    finally:
        restore_rng(state, generator)
        torch.set_num_threads(threads)
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
        torch.use_deterministic_algorithms(deterministic[0], warn_only=deterministic[1])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = (
            deterministic[2:]
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class EpochObserver:
    """Observe saved EMA weights without resetting the training recipe after one check."""

    def __init__(
        self,
        run,
        config,
        hours,
        *,
        minimum_epochs=5,
        patience=3,
        min_delta=0.001,
        export_every=0,
    ):
        self.run, self.config = Path(run), config
        self.hours, self.started = hours, time.monotonic()
        self.minimum_epochs, self.patience, self.min_delta = (
            minimum_epochs,
            patience,
            min_delta,
        )
        self.export_every = export_every
        history = self.run / "common_history.jsonl"
        self.rows = (
            [json.loads(line) for line in history.read_text().splitlines()]
            if history.exists()
            else []
        )

    def __call__(self, trainer):
        epoch = trainer.epoch
        if self.rows and epoch <= self.rows[-1]["epoch"]:
            return
        print(f"YOLO epoch={epoch}: saved; full common validation starts", flush=True)
        with isolated_evaluation():
            row = evaluate_checkpoint(trainer.last, self.config)
        score = row["metrics"]["coco100"]["mAP"]
        if score is None or not math.isfinite(score):
            raise RuntimeError("Common validation returned no finite AP")
        row.update(
            epoch=epoch,
            elapsed_seconds=time.monotonic() - self.started,
            train_loss=[float(x) for x in trainer.tloss.cpu()]
            if trainer.tloss is not None
            else None,
            lr={
                str(i): group["lr"]
                for i, group in enumerate(trainer.optimizer.param_groups)
            },
        )
        best = max((r["metrics"]["coco100"]["mAP"] for r in self.rows), default=-1)
        if score > best:
            atomic_copy(trainer.last, self.run / "weights" / "common_best.pt")
            write_json(self.run / "common_best_metrics.json", row)
        self.rows.append(row)
        with (self.run / "common_history.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if self.export_every and (epoch + 1) % self.export_every == 0:
            best_path = self.run / "weights/common_best.pt"
            report = read_json(self.run / "common_best_metrics.json")
            destination = (
                self.run.parent.parent
                / "submissions"
                / f"{self.run.name}_{file_sha(best_path)[:12]}.csv"
            )
            try:
                if report["checkpoint_sha256"] != file_sha(best_path):
                    raise ValueError("Best validation receipt and exported weights differ")
                with isolated_evaluation():
                    checked = export_checkpoint(best_path, self.config, destination)
                write_json(
                    self.run / "export_review.json",
                    {
                        "status": "QUALITY_REVIEW_REQUIRED",
                        "submission": str(destination),
                        "validation_report": report,
                        "csv_validation": checked,
                        "reference_comparison": None,
                        "public_score": None,
                    },
                )
            except Exception as error:
                # An exporter failure must not throw away a healthy training course.
                write_json(
                    self.run / "export_error.json",
                    {"epoch": epoch, "error": str(error)},
                )
                print(
                    f"YOLO export deferred: {error}; training state retained",
                    flush=True,
                )
        anchor, stale = -1, 0
        for old in self.rows:
            ap = old["metrics"]["coco100"]["mAP"]
            if ap > anchor + self.min_delta:
                anchor, stale = ap, 0
            else:
                stale += 1
        reason = None
        if (self.run / "STOP").exists():
            reason = "user_stop_at_saved_boundary"
        elif time.monotonic() - self.started >= self.hours * 3600:
            reason = "session_budget_at_saved_boundary"
        elif epoch + 1 >= self.minimum_epochs and stale >= self.patience:
            reason = "full_validation_plateau"
        elif epoch + 1 >= getattr(trainer, "epochs", math.inf):
            reason = "scheduled_course_complete"
        if reason:
            trainer.stop = True
        write_json(
            self.run / "status.json",
            {
                "status": reason or "training",
                "last_epoch": epoch,
                "last_ap": score,
                "best_ap": max(best, score),
                "stale_checks": stale,
                "checkpoint_sha256": row["checkpoint_sha256"],
                "public_score": None,
                "training_best_is_not_public_best": True,
            },
        )
        print(
            f"YOLO epoch={epoch}: common AP={score:.4f}; best={max(best, score):.4f}; "
            f"next={reason or 'continue same optimizer/schedule'}",
            flush=True,
        )


def train_challenger(
    work,
    root,
    prepared,
    *,
    name="yolo11s_1280",
    epochs=20,
    hours=10,
    batch=2,
    image_size=1280,
    workers=2,
    device="0",
    resume=False,
    initial_model=None,
    export_every=0,
):
    """One fixed course, official COCO initialization, full data every epoch."""
    configure_optional_backend(work)
    from ultralytics import YOLO, __version__
    from .yolo_backend import FullDataTrainer

    work = Path(work).resolve()
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Run name must be a single directory name")
    if (
        epochs < 1
        or not math.isfinite(hours)
        or hours <= 0
        or batch < 1
        or image_size % 32
    ):
        raise ValueError(
            "Positive epoch/batch/time and stride-aligned image size required"
        )
    run = work / "runs" / name
    if (run / "STOP").exists():
        raise RuntimeError(
            "User STOP exists; preserve it until continuation is explicitly requested"
        )
    data = prepare_dataset(root, prepared, work / "yolo_data")
    config = common_config(
        root,
        prepared,
        run,
        image_size,
        "cpu" if device == "cpu" else f"cuda:{device}",
        workers,
    )
    arguments = dict(
        data=str(data),
        project=str(work / "runs"),
        name=name,
        exist_ok=True,
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        workers=workers,
        device=device,
        cache=False,
        save=True,
        save_period=1,
        plots=False,
        val=False,
        patience=0,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.05,
        cos_lr=True,
        warmup_epochs=1.0,
        nbs=16,
        mosaic=0.5,
        scale=0.3,
        close_mosaic=3,
        seed=42,
        max_det=500,
        conf=0.001,
        iou=0.6,
        amp=device != "cpu",
    )
    plan = {
        "backend": f"ultralytics=={__version__}",
        "initial_model": initial_model or "yolo11s.pt",
        "arguments": arguments,
        "dataset": read_json(data.parent / "prepared.json"),
    }
    if resume:
        old = read_json(run / "yolo_plan.json")
        if old != plan:
            raise ValueError(
                "Resume requires the same recorded recipe/data; do not reset the schedule"
            )
        checkpoint = run / "weights" / "last.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = YOLO(str(checkpoint))
    else:
        if (run / "yolo_plan.json").exists():
            if (
                read_json(run / "yolo_plan.json") != plan
                or (run / "weights" / "last.pt").exists()
            ):
                raise ValueError(
                    "Run already exists; use --resume with the same recipe"
                )
        run.mkdir(parents=True, exist_ok=True)
        write_json(run / "yolo_plan.json", plan)
        write_json(run / "config.json", config)
        # The only production initialization is the official COCO asset.
        source = initial_model or str(work / "cache" / "ultralytics" / "yolo11s.pt")
        model = YOLO(source)
        if Path(source).is_file():
            write_json(
                run / "initialization.json",
                {
                    "source": "official COCO yolo11s.pt"
                    if initial_model is None
                    else initial_model,
                    "sha256": file_sha(source),
                },
            )
    FullDataTrainer.ignore_path = data.parent / "ignored_train.json"
    observer = EpochObserver(run, config, hours, export_every=export_every)
    if resume and model.ckpt["epoch"] > (
        observer.rows[-1]["epoch"] if observer.rows else -1
    ):
        # A crash during metrics can leave a complete saved epoch without its
        # common receipt. Validate it before advancing, never skip that evidence.
        from types import SimpleNamespace

        saved = model.ckpt
        previous = SimpleNamespace(
            epoch=saved["epoch"],
            epochs=epochs,
            last=checkpoint,
            tloss=None,
            optimizer=SimpleNamespace(param_groups=saved["optimizer"]["param_groups"]),
            stop=False,
        )
        observer(previous)
        if previous.stop:
            return run
    model.add_callback("on_fit_epoch_end", observer)
    model.train(trainer=FullDataTrainer, resume=resume, **arguments)
    return run
