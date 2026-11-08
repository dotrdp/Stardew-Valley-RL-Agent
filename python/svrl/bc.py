"""Behaviour cloning (optionally time-conditioned on days-to-go) from furrow-traj datasets."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import Dataset, episode_balanced_weights, unpack_masks
from .models import Policy
from .utils import JsonlLogger, device as pick_device, seed_all


def masked_ce(logits: torch.Tensor, mask: torch.Tensor, label: torch.Tensor, smoothing: float = 0.0) -> torch.Tensor:
    """Cross-entropy with label smoothing spread over *valid* actions only."""
    logp = torch.log_softmax(logits, -1)
    nll = -logp.gather(-1, label.unsqueeze(-1)).squeeze(-1)
    if smoothing <= 0:
        return nll.mean()
    m = mask.float()
    smooth = -(logp.masked_fill(~mask, 0.0) * m).sum(-1) / m.sum(-1).clamp_min(1.0)
    return ((1 - smoothing) * nll + smoothing * smooth).mean()


def load_tensors(datasets: list[str], dev: str, success_only: bool = False, fastest_frac: float | None = None,
                 max_episodes: int | None = None, label: str = "expert_action") -> dict:
    ds = Dataset(*datasets)
    idx = ds.episode_indices(success_only=success_only, fastest_frac=fastest_frac, max_episodes=max_episodes)
    m = ds.materialize(idx, label=label)
    t = {
        "obs": torch.from_numpy(m["obs"]).to(dev),  # f16
        "mask": torch.from_numpy(unpack_masks(m["mask_packed"], ds.num_actions)).to(dev),
        "label": torch.from_numpy(m["label"]).to(dev),
        "dtg": torch.from_numpy(np.nan_to_num(m["days_to_go"], nan=200.0)).to(dev),
        "w": torch.from_numpy(episode_balanced_weights(m["episode"])).to(dev),
        "episode": m["episode"],
    }
    t["obs_dim"], t["num_actions"], t["episodes"] = ds.obs_dim, ds.num_actions, len(idx)
    return t


def train_bc(datasets: list[str], out: str, steps: int = 20000, batch: int = 4096, lr: float = 3e-4, hidden: int = 512,
             layers: int = 3, label_smoothing: float = 0.02, cond: bool = False, seed: int = 0, max_vram_gb: float = 6.0,
             success_only: bool = False, fastest_frac: float | None = None, max_episodes: int | None = None,
             init: str | None = None, data: dict | None = None, log_every: int = 500) -> dict:
    """Train a (masked) BC policy; returns final metrics. Saves ``out`` (torch checkpoint)."""
    seed_all(seed)
    dev = pick_device(max_vram_gb)
    t = data if data is not None else load_tensors(datasets, dev, success_only, fastest_frac, max_episodes)
    n = t["obs"].shape[0]
    if init:
        pol = Policy.load(init, dev)
    else:
        pol = Policy(t["obs_dim"], t["num_actions"], hidden, layers, cond=cond).to(dev)
        sample = t["obs"][torch.randperm(n, device=dev)[: min(n, 200_000)]]
        pol.norm.fit(sample)
    opt = torch.optim.AdamW(pol.actor.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    log = JsonlLogger(Path(out).with_suffix(".log.jsonl"))
    # hold out 5% of episodes for validation accuracy
    eps = np.unique(t["episode"])
    rng = np.random.default_rng(seed)
    val_eps = set(rng.choice(eps, size=max(1, len(eps) // 20), replace=False).tolist())
    is_val = torch.from_numpy(np.isin(t["episode"], list(val_eps))).to(dev)
    w = t["w"].clone()
    w[is_val] = 0
    w = w / w.sum()
    val_idx = torch.nonzero(is_val).squeeze(-1)
    metrics = {}
    for step in range(steps):
        idx = torch.multinomial(w, batch, replacement=True)
        lg = pol.logits(t["obs"][idx], t["mask"][idx], t["dtg"][idx] if cond else None)
        loss = masked_ce(lg, t["mask"][idx], t["label"][idx], label_smoothing)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.actor.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                vi = val_idx[torch.randint(0, len(val_idx), (min(len(val_idx), 20000),), device=dev)]
                vlg = pol.logits(t["obs"][vi], t["mask"][vi], t["dtg"][vi] if cond else None)
                acc = (vlg.argmax(-1) == t["label"][vi]).float().mean().item()
                vloss = F.cross_entropy(vlg, t["label"][vi]).item()
            metrics = dict(step=step, loss=loss.item(), val_loss=vloss, val_acc=acc)
            log.log(**metrics)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pol.save(out, metrics=metrics, datasets=datasets, cond=cond)
    return metrics
