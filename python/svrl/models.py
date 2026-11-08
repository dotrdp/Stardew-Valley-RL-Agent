"""Policies: masked-logit MLP actor with a separate critic trunk (DESIGN.md §6)."""
from __future__ import annotations

import torch
import torch.nn as nn

NEG = -1e9


class ObsNorm(nn.Module):
    """Frozen per-feature standardisation estimated from a dataset (never updated online)."""

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> "ObsNorm":
        x = x.float()
        self.mean.copy_(x.mean(0))
        self.std.copy_(x.std(0).clamp_min(1e-3))
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x.float() - self.mean) / self.std).clamp(-10, 10)


def mlp(inp: int, hidden: int, layers: int, out: int) -> nn.Sequential:
    mods: list[nn.Module] = []
    d = inp
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU()]
        d = hidden
    mods.append(nn.Linear(d, out))
    return nn.Sequential(*mods)


class Policy(nn.Module):
    """Actor (masked logits) + critic; optional days-to-go conditioning input (time-conditioned BC)."""

    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 512, layers: int = 3, cond: bool = False):
        super().__init__()
        self.obs_dim, self.num_actions, self.cond = obs_dim, num_actions, cond
        self.cfg = dict(obs_dim=obs_dim, num_actions=num_actions, hidden=hidden, layers=layers, cond=cond)
        self.norm = ObsNorm(obs_dim)
        extra = 1 if cond else 0
        self.actor = mlp(obs_dim + extra, hidden, layers, num_actions)
        self.critic = mlp(obs_dim + extra, hidden, max(2, layers - 1), 1)

    def _inp(self, obs: torch.Tensor, dtg: torch.Tensor | None) -> torch.Tensor:
        x = self.norm(obs)
        if self.cond:
            if dtg is None:
                raise ValueError("conditioned policy needs days_to_go")
            x = torch.cat([x, (dtg.float() / 96.0).unsqueeze(-1)], -1)
        return x

    def logits(self, obs: torch.Tensor, mask: torch.Tensor, dtg: torch.Tensor | None = None) -> torch.Tensor:
        lg = self.actor(self._inp(obs, dtg))
        return lg.masked_fill(~mask, NEG)

    def value(self, obs: torch.Tensor, dtg: torch.Tensor | None = None) -> torch.Tensor:
        return self.critic(self._inp(obs, dtg)).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, mask: torch.Tensor, greedy: bool = True, temperature: float = 1.0, dtg=None) -> torch.Tensor:
        lg = self.logits(obs, mask, dtg)
        if greedy:
            return lg.argmax(-1)
        return torch.distributions.Categorical(logits=lg / temperature).sample()

    def save(self, path: str, **extra) -> None:
        torch.save({"cfg": self.cfg, "state": self.state_dict(), **extra}, path)

    @staticmethod
    def load(path: str, device: str = "cpu") -> "Policy":
        ck = torch.load(path, map_location=device, weights_only=False)
        p = Policy(**ck["cfg"])
        p.load_state_dict(ck["state"])
        return p.to(device)
