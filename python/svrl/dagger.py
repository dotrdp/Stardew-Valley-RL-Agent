"""DAgger with the Rust expert as the interactive oracle (DESIGN.md §4, §6).

Each round rolls out a β-mixture of expert and learner on training seeds in a ``furrow.VecEnv``,
labels every visited state with the expert's clean action (``vec.expert_actions()``), aggregates
all rounds, and retrains the learner. β_i = 0.5^(i+1) after an initial BC round.
"""
from __future__ import annotations

import numpy as np
import torch

import furrow

from .bc import load_tensors, train_bc
from .data import episode_balanced_weights
from .evaluate import VAL_SEEDS, run_policy, summarize, torch_policy
from .models import Policy
from .utils import JsonlLogger, device as pick_device


def collect_round(policy: Policy | None, beta: float, num_envs: int, steps: int, seed: int, dev: str) -> dict:
    vec = furrow.VecEnv(num_envs, seed=seed, seed_range=(0, 2**32))
    obs, _ = vec.reset()
    rng = np.random.default_rng(seed)
    O, M, L, E = [], [], [], []
    ep_id = np.arange(num_envs, dtype=np.int64) + seed * 1_000_000
    next_id = int(ep_id.max()) + 1
    for _ in range(steps):
        label = vec.expert_actions()
        O.append(np.asarray(obs["global_"], dtype=np.float16))
        M.append(np.asarray(obs["action_mask"]))
        L.append(label.copy())
        E.append(ep_id.copy())
        if policy is None or beta >= 1.0:
            act = label
        else:
            with torch.no_grad():
                o = torch.from_numpy(O[-1]).to(dev)
                m = torch.from_numpy(M[-1]).to(dev)
                learner = policy.act(o, m, greedy=False).cpu().numpy()
            use_exp = rng.random(num_envs) < beta
            act = np.where(use_exp, label, learner)
        obs, _, _, _, info = vec.step(act)
        done = info["_final_obs"]
        if done.any():
            k = int(done.sum())
            ep_id[done] = np.arange(next_id, next_id + k)
            next_id += k
    return {"obs": np.concatenate(O), "mask": np.concatenate(M), "label": np.concatenate(L), "episode": np.concatenate(E)}


def run_dagger(init_datasets: list[str], out_dir: str, rounds: int = 6, num_envs: int = 64, steps_per_round: int = 1500,
               bc_steps: int = 8000, init_max_episodes: int | None = 100, eval_seeds: int = 64, max_vram_gb: float = 6.0,
               seed: int = 0) -> list[dict]:
    dev = pick_device(max_vram_gb)
    log = JsonlLogger(f"{out_dir}/dagger.log.jsonl")
    base = load_tensors(init_datasets, dev, max_episodes=init_max_episodes)
    data = {k: base[k] for k in ("obs", "mask", "label", "dtg")}
    episode = base["episode"]
    ckpt = f"{out_dir}/dagger_r0.pt"
    train_bc([], ckpt, steps=bc_steps, data={**data, "w": base["w"], "episode": episode, "obs_dim": base["obs_dim"], "num_actions": base["num_actions"]},
             max_vram_gb=max_vram_gb, seed=seed)
    seeds_eval = list(range(VAL_SEEDS[0], VAL_SEEDS[0] + eval_seeds))
    hist = []
    for r in range(rounds + 1):
        pol = Policy.load(ckpt, dev).eval()
        res = summarize(run_policy(torch_policy(pol, dev, greedy=True), seeds_eval))
        rec = {"round": r, "samples": int(data["obs"].shape[0]), "success_rate": res["success_rate"], "mean_C": res["mean_C"], "iqm_C": res["iqm_C"], "ckpt": ckpt}
        hist.append(rec)
        log.log(**rec)
        if r == rounds:
            break
        beta = 0.5 ** (r + 1)
        new = collect_round(pol, beta, num_envs, steps_per_round, seed=seed + 1000 * (r + 1), dev=dev)
        data["obs"] = torch.cat([data["obs"], torch.from_numpy(new["obs"]).to(dev)])
        data["mask"] = torch.cat([data["mask"], torch.from_numpy(new["mask"]).to(dev)])
        data["label"] = torch.cat([data["label"], torch.from_numpy(new["label"]).to(dev)])
        data["dtg"] = torch.cat([data["dtg"], torch.full((new["obs"].shape[0],), 200.0, device=dev)])
        episode = np.concatenate([episode, new["episode"]])
        w = torch.from_numpy(episode_balanced_weights(episode)).to(dev)
        ckpt = f"{out_dir}/dagger_r{r + 1}.pt"
        train_bc([], ckpt, steps=bc_steps, data={**data, "w": w, "episode": episode, "obs_dim": base["obs_dim"], "num_actions": base["num_actions"]},
                 max_vram_gb=max_vram_gb, seed=seed + r + 1)
    return hist
