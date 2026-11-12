"""Randomized, flight- and time-spread permutations for short training blocks."""
from collections import defaultdict, deque

import torch
from torch.utils.data import Sampler


class FlightTimeSampler(Sampler):
    """Visit each eligible frame once per epoch, interleaving flights/time bins.

    Short prefixes intentionally favor coverage over the frame-frequency
    distribution. No oversampling, labels, or validation records are involved.
    The loader's checkpointed generator provides epoch-boundary reproducibility.
    """

    def __init__(self, records, generator, bins=8):
        if not isinstance(bins, int) or bins < 1:
            raise ValueError("temporal_bins must be a positive integer")
        self.generator = generator
        self.size = len(records)
        families = defaultdict(list)
        for index, record in enumerate(records):
            sequence = record["sequence_id"]
            family = sequence.split("_", 1)[0] if sequence.startswith("uav") else sequence
            families[family].append(index)
        self.groups = []
        for family in sorted(families):
            # Sequence ordering keeps excerpts separate; frame IDs order time
            # within each excerpt even if the input manifest is shuffled.
            indices = sorted(families[family], key=lambda i: (
                records[i]["sequence_id"], records[i].get("frame_id", records[i]["id"])
            ))
            count = min(bins, len(indices))
            self.groups.append([
                indices[j * len(indices) // count:(j + 1) * len(indices) // count]
                for j in range(count)
            ])

    def __len__(self):
        return self.size

    def _order(self, size):
        return torch.randperm(size, generator=self.generator).tolist()

    def __iter__(self):
        queues = []
        for groups in self.groups:
            bins = [deque(group[i] for i in self._order(len(group))) for group in groups]
            family = deque()
            while bins:
                for i in self._order(len(bins)):
                    family.append(bins[i].popleft())
                bins = [group for group in bins if group]
            queues.append(family)
        while queues:
            for i in self._order(len(queues)):
                yield queues[i].popleft()
            queues = [group for group in queues if group]
