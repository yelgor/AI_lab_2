"""Write a separate family-grouped split, preserving all existing fold artifacts."""
import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from cvlab.common import digest, read_json, write_json
    from cvlab.postprocess_experiments import stream_array
    from cvlab.prepare import group_folds
    from cvlab.sequential_policy import video_family

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.resolve() == (args.prepared / "folds.json").resolve():
        raise ValueError("Refusing to replace the active legacy folds.json")
    records = [
        {"id": r["id"], "sequence_id": r["sequence_id"]}
        for r in stream_array(args.prepared / "train.json", "images")
    ]
    result = group_folds(records, 3, 42)
    result["source_assignment_fingerprint"] = digest(records)
    result["usage"] = "New fold experiments only; do not resume old fine-tuned weights on changed held-out families"
    summaries = []
    for fold in result["folds"]:
        sequences = set(fold["val_sequences"])
        validation = [r for r in records if r["sequence_id"] in sequences]
        families = Counter(video_family(r["sequence_id"]) for r in validation)
        summaries.append({"fold": fold["fold"], "validation_frames": len(validation),
                          "validation_families": len(families), "family_frames": dict(sorted(families.items()))})
    result["summary"] = summaries
    if args.out.exists() and read_json(args.out) != result:
        raise ValueError("Existing split differs; choose a new versioned output")
    write_json(args.out, result)
    print(args.out)
    for row in summaries:
        print(f"fold={row['fold']} validation_frames={row['validation_frames']} families={row['validation_families']}")


if __name__ == "__main__":
    main()
