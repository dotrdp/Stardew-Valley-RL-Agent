"""Reader for ``furrow-traj/2`` datasets (memory-mapped ``.npy`` shards) and samplers.

Usage::

    ds = Dataset("data/expert")
    batch = ds.sample(4096, rng)          # episode-balanced step sampling
    batch["obs"]      # f32 [B, G]
    batch["mask"]     # bool [B, A]
    batch["label"]    # i64 [B]   clean expert label (falls back to executed action)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

COLUMNS = (
    "obs_global", "action", "expert_action", "actor", "action_mask", "reward", "terminated", "truncated",
    "day_ends", "ticks", "outcome", "episode", "day", "minute", "progress", "expert_intent",
)


@dataclass
class Shard:
    name: str
    cols: dict  # name -> np.memmap


class Dataset:
    """A furrow-traj/2 dataset directory (or several, concatenated)."""

    def __init__(self, *dirs: str | Path, load_obs: bool = True):
        self.dirs = [Path(d) for d in dirs]
        self.meta = json.loads((self.dirs[0] / "meta.json").read_text())
        self.num_actions = int(self.meta["num_actions"])
        self.obs_dim = int(self.meta["obs_global_dim"])
        self.episodes: list[dict] = []
        self._shards: dict[tuple[int, str], Shard] = {}
        offset = 0
        for di, d in enumerate(self.dirs):
            m = json.loads((d / "meta.json").read_text())
            if m["obs_global_dim"] != self.obs_dim or m["num_actions"] != self.num_actions:
                raise ValueError(f"{d}: incompatible dataset (obs/action dims differ)")
            for line in (d / "episodes.jsonl").read_text().splitlines():
                if not line.strip():
                    continue
                e = json.loads(line)
                e["_dir"] = di
                e["_gid"] = offset
                offset += 1
                self.episodes.append(e)
            for sd in sorted((d / "shards").iterdir()):
                if sd.name.startswith(".tmp"):
                    continue
                cols = {}
                for c in COLUMNS:
                    p = sd / f"{c}.npy"
                    if p.exists() and (load_obs or c != "obs_global"):
                        cols[c] = np.load(p, mmap_mode="r")
                self._shards[(di, sd.name)] = Shard(sd.name, cols)
        self.lengths = np.array([e["length"] for e in self.episodes], dtype=np.int64)
        self.days = np.array([e["days"] for e in self.episodes], dtype=np.int64)
        self.success = np.array([e["success"] for e in self.episodes], dtype=bool)

    def __len__(self) -> int:
        return len(self.episodes)

    @property
    def num_steps(self) -> int:
        return int(self.lengths.sum())

    def episode(self, i: int) -> dict:
        """All columns of episode ``i`` (as numpy arrays, obs converted to f32, masks unpacked)."""
        e = self.episodes[i]
        sh = self._shards[(e["_dir"], e["shard"])]
        s, n = e["start"], e["length"]
        out = {c: np.asarray(v[s : s + n]) for c, v in sh.cols.items()}
        if "obs_global" in out:
            out["obs_global"] = out["obs_global"].astype(np.float32)
        out["mask"] = np.unpackbits(out.pop("action_mask"), axis=-1, bitorder="little", count=self.num_actions).astype(bool)
        return out

    def episode_indices(self, success_only: bool = False, fastest_frac: float | None = None, max_episodes: int | None = None) -> np.ndarray:
        idx = np.arange(len(self))
        if success_only:
            idx = idx[self.success[idx]]
        if fastest_frac is not None:
            k = max(1, int(round(len(idx) * fastest_frac)))
            idx = idx[np.argsort(self.days[idx], kind="stable")[:k]]
        if max_episodes is not None:
            idx = idx[:max_episodes]
        return idx

    def materialize(self, episodes: np.ndarray | None = None, label: str = "expert_action") -> dict:
        """Load the selected episodes fully into RAM (obs f16→f32 kept as f16 to save memory)."""
        if episodes is None:
            episodes = np.arange(len(self))
        obs, mask, lab, ep, dte, day, de, rew, term, trunc = [], [], [], [], [], [], [], [], [], []
        for i in episodes:
            e = self.episodes[int(i)]
            sh = self._shards[(e["_dir"], e["shard"])]
            s, n = e["start"], e["length"]
            obs.append(np.asarray(sh.cols["obs_global"][s : s + n]))
            mask.append(np.asarray(sh.cols["action_mask"][s : s + n]))
            l = np.asarray(sh.cols[label][s : s + n]).astype(np.int64)
            if label == "expert_action":
                bad = l < 0
                if bad.any():
                    l[bad] = np.asarray(sh.cols["action"][s : s + n])[bad]
            lab.append(l)
            ep.append(np.full(n, int(i), dtype=np.int64))
            d = np.asarray(sh.cols["day"][s : s + n]).astype(np.float32)
            # day of the *decision* = day after the previous step (first step: day 0)
            d0 = np.concatenate([[0.0], d[:-1]])
            day.append(d0)
            dte.append((e["days"] - d0) if e["success"] else np.full(n, np.nan, dtype=np.float32))
            de.append(np.asarray(sh.cols["day_ends"][s : s + n]))
            rew.append(np.asarray(sh.cols["reward"][s : s + n]))
            term.append(np.asarray(sh.cols["terminated"][s : s + n]))
            trunc.append(np.asarray(sh.cols["truncated"][s : s + n]))
        return {
            "obs": np.concatenate(obs),
            "mask_packed": np.concatenate(mask),
            "label": np.concatenate(lab),
            "episode": np.concatenate(ep),
            "day": np.concatenate(day).astype(np.float32),
            "days_to_go": np.concatenate(dte).astype(np.float32),
            "day_ends": np.concatenate(de),
            "reward": np.concatenate(rew),
            "terminated": np.concatenate(term),
            "truncated": np.concatenate(trunc),
            "num_actions": self.num_actions,
        }


def unpack_masks(packed: np.ndarray, num_actions: int) -> np.ndarray:
    return np.unpackbits(packed, axis=-1, bitorder="little", count=num_actions).astype(bool)


def episode_balanced_weights(episode: np.ndarray) -> np.ndarray:
    """Per-step sampling weights so every episode has equal total weight."""
    _, inv, counts = np.unique(episode, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv]
    return w / w.sum()
