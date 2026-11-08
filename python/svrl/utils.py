"""Shared helpers: device selection with a VRAM cap, seeding, JSONL logging."""
from __future__ import annotations

import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np


def device(max_vram_gb: float = 6.0) -> str:
    import torch

    if torch.cuda.is_available() and os.environ.get("SVRL_CPU") != "1":
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, max_vram_gb / total))
        return "cuda"
    return "cpu"


def seed_all(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()

    def log(self, **kw) -> None:
        kw.setdefault("time", round(time.time() - self.t0, 3))
        with self.path.open("a") as f:
            f.write(json.dumps(kw, default=float) + "\n")
