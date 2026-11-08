import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import yaml

CLASSES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    override = os.environ.get("CVLAB_DATA_ROOT")
    if override:
        config["data"]["root"] = override
    return config


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def protocol_fingerprint(config):
    data = {
        key: value
        for key, value in config["data"].items()
        if key not in ("root", "prepared", "fold")
    }
    return digest(
        {
            "data": data,
            **{
                section: config[section]
                for section in (
                    "model",
                    "train",
                    "loader",
                    "evaluation",
                    "runtime",
                    "seed",
                )
            },
        }
    )


def seed_all(seed, deterministic=False):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id):
    import torch

    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
