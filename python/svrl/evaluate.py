"""Evaluation protocol (DESIGN.md §6): censored completion day on paired held-out seeds.

* ``C`` = completion day if the objective is finished, else ``max_days`` (censored).
* Reported: success rate, mean / median / IQM of ``C`` with stratified-bootstrap 95% CIs,
  speed score ``S = (max_days - mean C) / (max_days - mean C_expert)``, and the per-seed paired
  difference to the expert (same seeds ⇒ same weather, merchant stock and forage).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

import furrow

TEST_SEEDS = (2**33, 2**33 + 1000)
VAL_SEEDS = (2**32, 2**32 + 1000)


def run_policy(policy: Callable[[dict, "furrow.VecEnv"], np.ndarray], seeds: list[int], max_days: int = 192,
               num_threads: int | None = None, **cfg) -> dict:
    """Run one episode per seed with ``policy(obs, vec) -> actions``; returns per-seed arrays."""
    vec = furrow.VecEnv(len(seeds), seeds=list(seeds), num_threads=num_threads, max_days=max_days, **cfg)
    obs, _ = vec.reset()
    n = len(seeds)
    days = np.full(n, max_days, dtype=np.int64)
    success = np.zeros(n, dtype=bool)
    ret = np.zeros(n, dtype=np.float64)
    decisions = np.zeros(n, dtype=np.int64)
    requests = np.zeros(n, dtype=np.int64)
    finished = np.zeros(n, dtype=bool)
    while not finished.all():
        acts = policy(obs, vec)
        obs, rew, term, trunc, info = vec.step(acts)
        ret[~finished] += rew[~finished]
        newly = info["_final_obs"] & ~finished
        if newly.any():
            success[newly] = info["success"][newly]
            days[newly] = np.where(info["success"][newly], info["episode_days"][newly], max_days)
            decisions[newly] = info["decisions"][newly]
            requests[newly] = info["requests_done"][newly]
            finished |= newly
    return {"seeds": np.asarray(seeds), "C": days, "success": success, "return": ret, "decisions": decisions, "requests": requests}


def masked_random_policy(seed: int = 0):
    rng = np.random.default_rng(seed)

    def pol(obs, vec):
        m = obs["action_mask"]
        g = rng.random(m.shape) * m
        return g.argmax(-1)

    return pol


def expert_policy(style: int = 0):
    def pol(obs, vec):
        return vec.expert_actions([style])

    return pol


def torch_policy(model, device: str, greedy: bool = True, temperature: float = 1.0, target_days: float | None = None):
    import torch

    def pol(obs, vec):
        o = torch.from_numpy(np.asarray(obs["global_"], dtype=np.float16)).to(device)  # match f16 training inputs
        m = torch.from_numpy(np.asarray(obs["action_mask"])).to(device)
        dtg = None
        if getattr(model, "cond", False):
            day = o[:, 0:1].float()  # not used; days_to_go from calendar feature below
            elapsed = o[:, 14].float() * 384.0  # 'days elapsed / 384' feature
            dtg = torch.clamp(torch.as_tensor(target_days, device=device) - elapsed, min=1.0)
            del day
        a = model.act(o, m, greedy=greedy, temperature=temperature, dtg=dtg)
        return a.cpu().numpy()

    return pol


def iqm(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(x[lo:hi].mean()) if hi > lo else float(x.mean())


def bootstrap_ci(x: np.ndarray, stat=np.mean, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    bs = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def summarize(r: dict, max_days: int = 192, expert: dict | None = None) -> dict:
    C = r["C"].astype(np.float64)
    out = {
        "n": int(len(C)),
        "success_rate": float(r["success"].mean()),
        "success_ci": bootstrap_ci(r["success"].astype(float)),
        "mean_C": float(C.mean()),
        "mean_C_ci": bootstrap_ci(C),
        "median_C": float(np.median(C)),
        "iqm_C": iqm(C),
        "iqm_C_ci": bootstrap_ci(C, stat=iqm, n_boot=1000),
        "mean_return": float(r["return"].mean()),
        "mean_decisions": float(r["decisions"].mean()),
        "mean_requests": float(r["requests"].mean()),
        "C_hist": np.bincount(np.minimum(C.astype(int), max_days), minlength=max_days + 1).tolist(),
    }
    if expert is not None:
        ce = expert["C"].astype(np.float64)
        out["speed_score"] = float((max_days - C.mean()) / max(1e-9, (max_days - ce.mean())))
        d = C - ce
        out["paired_diff_vs_expert"] = float(d.mean())
        out["paired_diff_ci"] = bootstrap_ci(d)
        out["frac_faster_than_expert"] = float((d < 0).mean())
    return out
