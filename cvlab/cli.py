import argparse
import json
from pathlib import Path

import yaml

from .common import digest, load_config, protocol_fingerprint, read_json, write_json


def config_for(args):
    config = load_config(args.config)
    for override in args.overrides:
        key, value = override.split("=", 1)
        parts = key.split(".")
        current = config
        for part in parts[:-1]:
            current = current[part]
        current[parts[-1]] = yaml.safe_load(value)
    if config["model"]["image_size"] % 32 or config["model"]["image_size"] < 64:
        raise ValueError("image_size must be a multiple of 32, >=64")
    if 100 not in config["evaluation"]["max_detections"]:
        raise ValueError("Keep standard COCO100 in evaluation.max_detections")
    if not 1 <= config["model"]["max_detections"] <= 500:
        raise ValueError("model.max_detections must be 1..500")
    return config


def main():
    parser = argparse.ArgumentParser(description="Drone detection experiment runner")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Show hardware and package versions")
    doctor.add_argument("--json", action="store_true")
    prep = commands.add_parser(
        "prepare", help="Validate competition data, create manifests, EDA and folds"
    )
    prep.add_argument("--root", required=True)
    prep.add_argument("--out", default="prepared")
    prep.add_argument("--workers", type=int, default=8)
    prep.add_argument("--folds", type=int, default=3)
    prep.add_argument("--seed", type=int, default=42)
    overlap = commands.add_parser(
        "train-overlap", help="Overlap CPU metrics with GPU training across model runs"
    )
    overlap.add_argument("--work", required=True)
    overlap.add_argument(
        "--models",
        nargs="+",
        default=["fcos", "fasterrcnn", "rtdetr"],
        choices=["fcos", "fasterrcnn", "retinanet", "rtdetr"],
    )
    export = commands.add_parser(
        "predict-submission", help="Generate a verified submission in bounded chunks"
    )
    export.add_argument("--run", required=True)
    export.add_argument("--out", required=True)
    export.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    export.add_argument("--threads", type=int, default=2)
    export.add_argument("--chunk-size", type=int, default=16)
    export.add_argument("--vram-gib", type=float, default=1.5)
    resolutions = commands.add_parser(
        "evaluate-resolutions",
        help="Compare inference sizes with a fixed checkpoint and validation subset",
    )
    resolutions.add_argument("--config", required=True)
    resolutions.add_argument("--checkpoint", required=True)
    resolutions.add_argument("--out", required=True)
    resolutions.add_argument("--sizes", type=int, nargs="+", default=[960, 1280])
    resolutions.add_argument("--max-images", type=int, default=192)
    resolutions.add_argument("--wait-gpu", action="store_true")
    resolutions.add_argument("--set", dest="overrides", action="append", default=[])
    for name in ("train", "evaluate", "predict", "benchmark"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=True)
        sub.add_argument(
            "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE"
        )
        if name == "train":
            sub.add_argument(
                "--resume",
                help="Resume from your own last.pt; epoch-boundary continuation",
            )
        else:
            sub.add_argument("--checkpoint", required=True)
            sub.add_argument("--out", required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("runs", nargs="+")
    summary.add_argument("--out", default="comparison.csv")
    oof = commands.add_parser(
        "oof", help="Check and pool complete, disjoint fold validation predictions"
    )
    oof.add_argument("files", nargs="+")
    oof.add_argument("--manifest", required=True, help="prepared/train.json")
    oof.add_argument("--out", default="oof.json")
    args = parser.parse_args()
    if args.command == "prepare":
        from .prepare import prepare

        prepare(args.root, args.out, args.workers, args.folds, args.seed)
    elif args.command == "doctor":
        from .engine import environment

        print(json.dumps(environment(), indent=2))
    elif args.command == "summarize":
        from .engine import summarize

        summarize(args.runs, args.out)
    elif args.command == "oof":
        from .metrics import evaluate

        records = read_json(args.manifest)["images"]
        by_id = {r["id"]: r for r in records}
        seen, predictions, model_config, folds, protocol = set(), [], None, set(), None
        for path in args.files:
            result = read_json(path)
            if result["split"] != "fold" or result["fold"] in folds:
                raise ValueError(
                    "OOF needs distinct fold runs, not official-validation runs"
                )
            folds.add(result["fold"])
            if model_config is None:
                model_config = result["model_config"]
                protocol = result["protocol_fingerprint"]
            if model_config != result["model_config"]:
                raise ValueError("OOF runs use different model configurations")
            if protocol != result["protocol_fingerprint"]:
                raise ValueError("OOF runs use different training/evaluation protocols")
            ids = result["record_ids"]
            if len(set(ids)) != len(ids) or seen.intersection(ids):
                raise ValueError("Overlapping OOF image IDs")
            subset = [by_id[i] for i in ids]
            if digest(subset) != result["data_fingerprint"]:
                raise ValueError("OOF manifest differs from prediction manifest")
            if any(p["image_id"] not in set(ids) for p in result["predictions"]):
                raise ValueError(
                    "Prediction contains an image outside its held-out fold"
                )
            seen.update(ids)
            predictions.extend(result["predictions"])
        if seen != set(by_id):
            raise ValueError(
                "OOF predictions do not cover every training image exactly once"
            )
        sequence_fold = {}
        for index, path in enumerate(args.files):
            for image_id in read_json(path)["record_ids"]:
                sequence = by_id[image_id]["sequence_id"]
                if sequence in sequence_fold and sequence_fold[sequence] != index:
                    raise ValueError("A sequence is split across OOF folds")
                sequence_fold[sequence] = index
        write_json(
            args.out,
            {
                "metrics": evaluate(records, predictions),
                "record_ids": sorted(seen),
                "predictions": predictions,
                "model_config": model_config,
            },
        )
    elif args.command == "predict-submission":
        from .submit_cpu import create_submission

        create_submission(
            args.run,
            args.out,
            args.threads,
            args.chunk_size,
            device_name=args.device,
            vram_gib=args.vram_gib,
        )
    elif args.command == "train-overlap":
        from .overlap_queue import run

        run(args.work, args.models)
    elif args.command == "evaluate-resolutions":
        from .inference_experiments import compare_resolutions

        compare_resolutions(
            config_for(args),
            args.checkpoint,
            args.out,
            args.sizes,
            args.max_images,
            args.wait_gpu,
        )
    else:
        from .engine import load_trained, predict, setup, train
        from .metrics import evaluate
        from .prepare import select_records
        from .submission import write_submission

        config = config_for(args)
        if args.command == "train":
            train(config, args.resume)
            return
        device = setup(config)
        records = select_records(config, "test" if args.command == "predict" else "val")
        model = load_trained(args.checkpoint, config, device)
        predictions, timing = predict(model, config, records, device)
        result = {
            "record_ids": [r["id"] for r in records],
            "predictions": predictions,
            "timing": timing,
            "model_config": config["model"],
            "split": config["data"]["split"],
            "fold": config["data"].get("fold"),
            "data_fingerprint": digest(records),
            "protocol_fingerprint": protocol_fingerprint(config),
            "config": config,
        }
        if args.command in ("evaluate", "benchmark"):
            result["metrics"] = evaluate(
                records, predictions, config["evaluation"]["max_detections"]
            )
            write_json(args.out, result)
        else:
            write_submission(
                records,
                predictions,
                Path(config["data"]["root"]) / "sample_submission.csv",
                args.out,
                config["model"]["score_threshold"],
                config["model"]["max_detections"],
            )
            write_json(str(args.out) + ".predictions.json", result)
        print(
            json.dumps(
                {
                    key: value
                    for key, value in result.items()
                    if key in ("timing", "metrics")
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
