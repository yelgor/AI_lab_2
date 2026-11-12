"""Count actual optimizer-input coverage without consuming random numbers."""
from collections import Counter


class TrainingCoverage:
    def __init__(self):
        self.ids = set()
        self.presentations = 0
        self.families = Counter()
        self.classes = Counter()

    def update(self, metadata):
        for meta in metadata:
            record = meta["record"]
            self.ids.add(record["id"])
            self.presentations += 1
            self.families[record["sequence_id"].split("_", 1)[0]] += 1
            # These describe source annotations, before any crop augmentation.
            self.classes.update(str(label) for label in set(record["labels"]))

    def report(self, eligible):
        return {
            "presentations": self.presentations,
            "unique_frames": len(self.ids),
            "repeated_presentations": self.presentations - len(self.ids),
            "eligible_frames": eligible,
            "coverage_fraction": len(self.ids) / eligible if eligible else 0,
            "frame_ids": sorted(self.ids),
            "family_presentations": dict(sorted(self.families.items())),
            "source_class_frame_presentations": dict(sorted(self.classes.items())),
            "scope": "This epoch only; source annotations before augmentation; historical runs without this artifact are unknown",
        }
