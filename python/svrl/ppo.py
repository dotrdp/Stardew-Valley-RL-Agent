"""PPO finetuning from an imitation prior (DESIGN.md §6 'Finetuning protocol').

* frozen observation normaliser from BC; separate actor/critic trunks;
* critic warm-up on rollouts of the frozen BC policy before any actor update;
* masks applied to every log-prob/entropy/KL; KL(π‖π_BC) penalty with linear annealing;
* GAE with per-day discounting ``γ_day ** day_ends`` and bootstrapping from ``final_obs`` on
  truncation (never across episode boundaries); fixed reward scale (unit = days).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

import furrow

from .evaluate import VAL_SEEDS, run_policy, summarize, torch_policy
from .models import Policy
from .utils import JsonlLogger, device as pick_device, seed_all


def _t(x, dev, dtype=None):
    return torch.as_tensor(np.asarray(x), device=dev, dtype=dtype)


def finetune(init: str, out: str, iters: int = 60, num_envs: int = 128, horizon: int = 128, gamma_day: float = 0.99,
             lam: float = 0.98, lr: float = 1e-5, clip: float = 0.2, epochs: int = 3, minibatch: int = 4096,
             kl_coef: float = 0.2, kl_final: float = 0.02, ent_coef: float = 0.0, warmup_iters: int = 10, value_scale: float = 100.0,
             eval_every: int = 10, eval_seeds: int = 64, seed: int = 0, max_vram_gb: float = 6.0) -> list[dict]:
    seed_all(seed)
    dev = pick_device(max_vram_gb)
    pol = Policy.load(init, dev)
    prior = copy.deepcopy(pol).eval()
    for p in prior.parameters():
        p.requires_grad_(False)
    opt_a = torch.optim.Adam(pol.actor.parameters(), lr=lr)
    opt_c = torch.optim.Adam(pol.critic.parameters(), lr=3e-4)
    vec = furrow.VecEnv(num_envs, seed=seed + 7, seed_range=(0, 2**32))
    obs, _ = vec.reset()
    log = JsonlLogger(out.replace(".pt", ".log.jsonl"))
    hist = []
    G = vec.obs_global_dim
    ep_ret = np.zeros(num_envs)
    done_days, done_succ = [], []

    def evaluate(it):
        res = summarize(run_policy(torch_policy(pol.eval(), dev, greedy=True), list(range(VAL_SEEDS[0], VAL_SEEDS[0] + eval_seeds))))
        pol.train()
        rec = {"iter": it, "val_success": res["success_rate"], "val_mean_C": res["mean_C"], "val_iqm_C": res["iqm_C"]}
        log.log(**rec)
        hist.append(rec)
        return rec

    evaluate(0)
    for it in range(1, iters + 1):
        # ---------------- rollout
        B_obs = torch.zeros(horizon, num_envs, G, device=dev, dtype=torch.float16)
        B_mask = torch.zeros(horizon, num_envs, vec.num_actions, device=dev, dtype=torch.bool)
        B_act = torch.zeros(horizon, num_envs, device=dev, dtype=torch.long)
        B_logp = torch.zeros(horizon, num_envs, device=dev)
        B_val = torch.zeros(horizon, num_envs, device=dev)
        B_rew = torch.zeros(horizon, num_envs, device=dev)
        B_disc = torch.zeros(horizon, num_envs, device=dev)  # γ^day_ends
        B_term = torch.zeros(horizon, num_envs, device=dev)
        B_trunc = torch.zeros(horizon, num_envs, device=dev)
        B_fval = torch.zeros(horizon, num_envs, device=dev)  # V(final_obs) for truncated rows
        frozen = it <= warmup_iters
        for t in range(horizon):
            o = _t(obs["global_"], dev, torch.float16)
            m = _t(obs["action_mask"], dev, torch.bool)
            with torch.no_grad():
                actor = prior if frozen else pol
                lg = actor.logits(o, m)
                dist = torch.distributions.Categorical(logits=lg)
                a = dist.sample()
                B_logp[t] = torch.distributions.Categorical(logits=pol.logits(o, m)).log_prob(a)
                B_val[t] = pol.value(o)
            obs, rew, term, trunc, info = vec.step(a.cpu().numpy())
            B_obs[t], B_mask[t], B_act[t] = o, m, a
            B_rew[t] = _t(rew, dev, torch.float32)
            B_disc[t] = gamma_day ** _t(info["day_ends"], dev, torch.float32)
            B_term[t] = _t(term, dev, torch.float32)
            B_trunc[t] = _t(trunc, dev, torch.float32)
            if trunc.any():
                fo = _t(info["final_obs"]["global_"], dev, torch.float16)
                with torch.no_grad():
                    B_fval[t] = pol.value(fo) * value_scale
            ep_ret += rew
            fin = info["_final_obs"]
            for i in np.nonzero(fin)[0]:
                done_days.append(int(info["episode_days"][i]))
                done_succ.append(bool(info["success"][i]))
                ep_ret[i] = 0
        with torch.no_grad():
            last_v = pol.value(_t(obs["global_"], dev, torch.float16)) * value_scale
        # ---------------- GAE (per-day discount; no bootstrapping across episode ends)
        V = B_val * value_scale
        adv = torch.zeros_like(B_rew)
        gae = torch.zeros(num_envs, device=dev)
        for t in reversed(range(horizon)):
            ended = torch.clamp(B_term[t] + B_trunc[t], max=1.0)
            next_v = last_v if t == horizon - 1 else V[t + 1]
            next_v = torch.where(B_trunc[t] > 0, B_fval[t], next_v * (1 - ended))
            delta = B_rew[t] + B_disc[t] * next_v - V[t]
            gae = delta + B_disc[t] * lam * (1 - ended) * gae
            adv[t] = gae
        ret = (adv + V) / value_scale
        # ---------------- update
        N = horizon * num_envs
        f_obs, f_mask, f_act = B_obs.reshape(N, G), B_mask.reshape(N, -1), B_act.reshape(N)
        f_logp, f_adv, f_ret = B_logp.reshape(N), adv.reshape(N), ret.reshape(N)
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)
        kl_now = kl_coef + (kl_final - kl_coef) * min(1.0, max(0, it - warmup_iters) / max(1, iters - warmup_iters))
        stats = {}
        for _ in range(epochs):
            perm = torch.randperm(N, device=dev)
            for s in range(0, N, minibatch):
                idx = perm[s : s + minibatch]
                v = pol.value(f_obs[idx])
                v_loss = F.mse_loss(v, f_ret[idx])
                opt_c.zero_grad(set_to_none=True)
                v_loss.backward()
                opt_c.step()
                if frozen:
                    continue
                lg = pol.logits(f_obs[idx], f_mask[idx])
                dist = torch.distributions.Categorical(logits=lg)
                logp = dist.log_prob(f_act[idx])
                ratio = (logp - f_logp[idx]).exp()
                pg = -torch.min(ratio * f_adv[idx], ratio.clamp(1 - clip, 1 + clip) * f_adv[idx]).mean()
                with torch.no_grad():
                    plg = prior.logits(f_obs[idx], f_mask[idx])
                p_logp = torch.log_softmax(lg, -1)
                kl = (p_logp.exp() * (p_logp - torch.log_softmax(plg, -1))).masked_fill(~f_mask[idx], 0).sum(-1).mean()
                ent = dist.entropy().mean()
                loss = pg + kl_now * kl - ent_coef * ent
                opt_a.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pol.actor.parameters(), 0.5)
                opt_a.step()
                stats = {"pg": pg.item(), "kl": kl.item(), "ent": ent.item()}
        ev = 1 - ((f_ret - pol.value(f_obs).detach()).var() / (f_ret.var() + 1e-8)).item()
        rec = {"iter": it, "frozen": frozen, "v_loss": v_loss.item(), "explained_var": ev, "kl_coef": kl_now, **stats}
        if done_days:
            rec["train_mean_days"] = float(np.mean(done_days[-256:]))
            rec["train_success"] = float(np.mean(done_succ[-256:]))
        log.log(**rec)
        if it % eval_every == 0 or it == iters:
            evaluate(it)
    pol.save(out, history=hist)
    return hist
