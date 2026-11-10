"""Read-only comparison of tiled CSV provenance, cached AP and class distributions."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


OLD = "rtdetr_a7cd89fb349d_960_tiles_b57dbbdfb3d4.csv"
NEW = "rtdetr_cc3d87b1034f_960_tiles_b57dbbdfb3d4.csv"


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def metric_summary(value):
    if not value:
        return None
    return {
        key: {
            metric: scores.get(metric)
            for metric in ("mAP", "AP50", "AP_small", "per_class")
        }
        for key, scores in value.get("metrics", {}).items()
    }


def csv_stats(path):
    counts, confident, score_sums, area_sums = (
        defaultdict(int),
        defaultdict(int),
        defaultdict(float),
        defaultdict(float),
    )
    rows, empty = 0, 0
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            rows += 1
            tokens = row["PredictionString"].split()
            if len(tokens) % 6:
                raise ValueError(f"Malformed predictions: {row['image_id']}")
            empty += not tokens
            for index in range(0, len(tokens), 6):
                category = int(tokens[index])
                score = float(tokens[index + 1])
                counts[category] += 1
                confident[category] += score >= 0.1
                score_sums[category] += score
                area_sums[category] += float(tokens[index + 4]) * float(
                    tokens[index + 5]
                )
    return {
        "images": rows,
        "empty_images": empty,
        "detections": sum(counts.values()),
        "classes": {
            str(key): {
                "detections": counts[key],
                "score_at_least_0.1": confident[key],
                "mean_score": score_sums[key] / counts[key],
                "mean_area_px": area_sums[key] / counts[key],
            }
            for key in sorted(counts)
        },
    }


def audit_one(work, filename):
    path = work / "submissions" / filename
    if not path.is_file():
        return {"file": filename, "missing": True}
    metadata = read(path.with_suffix(".csv.metadata.json"))
    if not metadata:
        return {"file": filename, "missing_metadata": True, "csv": csv_stats(path)}
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    sha = metadata["checkpoint_sha256"]
    result = {
        "file": filename,
        "checkpoint_sha256": sha,
        "image_size": metadata.get("image_size"),
        "options": metadata.get("options"),
        "test_fingerprint": metadata.get("test_fingerprint"),
        "csv_matches_export_checksum": hasher.hexdigest()
        == metadata.get("validation", {}).get("sha256"),
        "csv": csv_stats(path),
        "frozen_runs": [],
        "selection_reports": [],
    }
    signature = path.stem.rsplit("_tiles_", 1)[-1]
    for frozen in sorted((work / "score_improvement").glob(f"*_{sha[:12]}")):
        source = read(frozen / "source.json")
        native = read(frozen / f"official_full_{metadata['image_size']}.json")
        tiled = read(frozen / "tiles" / f"full_{signature}.json")
        result["frozen_runs"].append(
            {
                "directory": str(frozen),
                "source": source,
                "native_full": metric_summary(native),
                "tiled_full": metric_summary(tiled),
                "native_validation_fingerprint": (native or {}).get("data_fingerprint"),
                "tiled_validation_fingerprint": (tiled or {}).get("data_fingerprint"),
                "native_images": len((native or {}).get("record_ids", [])),
                "tiled_images": len((tiled or {}).get("record_ids", [])),
            }
        )
    for report_name in ("tiling_summary.json", "tiling_finetune_summary.json"):
        report = read(work / "score_improvement" / report_name)
        for entry in (report or {}).get("models", []):
            if entry.get("checkpoint_sha256") == sha:
                result["selection_reports"].append(
                    {
                        key: entry.get(key)
                        for key in (
                            "source_run",
                            "outcome",
                            "export_reference",
                            "full_validation_gain",
                            "minimum_full_validation_gain",
                            "submission",
                        )
                    }
                )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--old", default=OLD)
    parser.add_argument("--new", default=NEW)
    args = parser.parse_args()
    result = {
        "old": audit_one(args.work, args.old),
        "new": audit_one(args.work, args.new),
    }
    result["same_test_fingerprint"] = bool(
        result["old"].get("test_fingerprint")
    ) and result["old"].get("test_fingerprint") == result["new"].get("test_fingerprint")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
