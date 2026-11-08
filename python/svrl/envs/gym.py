"""Gymnasium (≥1.0) environments over the native ``furrow`` module.

* :class:`FurrowEnv` — ``gymnasium.Env`` with a ``Dict`` observation
  ``{"global": Box(G), "action_mask": MultiBinary(A)}`` (plus ``"grid"`` in primitive mode)
  and a ``Discrete(A)`` action space. Info carries ``day_ends``, ``ticks`` and ``outcome`` so
  learners can discount per in-game day (``gamma_day ** day_ends``).
* :class:`FurrowVecEnv` — ``gymnasium.vector.VectorEnv`` backed by the Rust ``VecEnv``
  (parallel Rust threads, GIL released). Declares ``AutoresetMode.SAME_STEP`` and reports the
  terminal observation of finished sub-envs in ``info["final_obs"]`` / ``info["_final_obs"]``.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, VectorEnv

import furrow


def _obs_space(num_actions: int, g: int, grid_shape) -> spaces.Dict:
    d = {
        "global": spaces.Box(low=-1.0, high=2.0, shape=(g,), dtype=np.float32),
        "action_mask": spaces.MultiBinary(num_actions),
    }
    if grid_shape is not None:
        d["grid"] = spaces.Box(low=0, high=255, shape=tuple(grid_shape), dtype=np.uint8)
    return spaces.Dict(d)


def _conv_obs(o: dict) -> dict:
    out = {"global": np.asarray(o["global_"], dtype=np.float32), "action_mask": np.asarray(o["action_mask"], dtype=np.int8)}
    if "grid" in o:
        out["grid"] = np.asarray(o["grid"], dtype=np.uint8)
    return out


class FurrowEnv(gym.Env):
    """A single Furrow environment (macro actions by default)."""

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None, **cfg: Any):
        super().__init__()
        self.render_mode = render_mode
        self._cfg = dict(cfg)
        self._seed = int(self._cfg.pop("seed", 0))
        self.env = furrow.Env(seed=self._seed, **self._cfg)
        self.action_space = spaces.Discrete(self.env.num_actions)
        self.observation_space = _obs_space(self.env.num_actions, self.env.obs_global_dim, self.env.grid_shape)
        self.action_names = self.env.action_names()
        self.gamma_day = self.env.gamma_day

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(0, 2**32))
        obs, info = self.env.reset(seed=int(seed))
        return _conv_obs(obs), dict(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(int(action) if not isinstance(action, str) else action)
        return _conv_obs(obs), float(reward), bool(terminated), bool(truncated), dict(info)

    def action_masks(self) -> np.ndarray:
        """For sb3-contrib's MaskablePPO."""
        return np.asarray(self.env.action_mask(), dtype=bool)

    def render(self):
        if self.render_mode == "ansi":
            return self.env.render()
        return None

    def describe(self) -> str:
        return self.env.describe()

    def expert_action(self, style: int = 0) -> int:
        return int(self.env.expert_action(style))


class FurrowVecEnv(VectorEnv):
    """Vectorised Furrow (Rust threads) with Gymnasium SAME_STEP autoreset semantics."""

    metadata = {"autoreset_mode": AutoresetMode.SAME_STEP, "render_modes": []}

    def __init__(self, num_envs: int, seed: int = 0, seed_range=(0, 2**32), num_threads: int | None = None, **cfg: Any):
        super().__init__()
        self.vec = furrow.VecEnv(num_envs, seed=seed, seed_range=tuple(seed_range), num_threads=num_threads, **cfg)
        self.num_envs = num_envs
        a = self.vec.num_actions
        self.single_action_space = spaces.Discrete(a)
        self.action_space = spaces.MultiDiscrete(np.full(num_envs, a))
        self.single_observation_space = _obs_space(a, self.vec.obs_global_dim, self.vec.grid_shape)
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, num_envs)
        self.action_names = self.vec.action_names()

    def reset(self, *, seed=None, options=None):
        obs, info = self.vec.reset()
        return _conv_obs(obs), dict(info)

    def step(self, actions):
        obs, rew, term, trunc, info = self.vec.step(np.asarray(actions, dtype=np.int64))
        info = dict(info)
        info["final_obs"] = _conv_obs(info["final_obs"])
        return _conv_obs(obs), rew, term, trunc, info

    def expert_actions(self, styles=None) -> np.ndarray:
        return self.vec.expert_actions(styles)

    def close_extras(self, **kwargs):
        pass


def register_envs() -> None:
    """Register ``Furrow-Macro-v1``, ``Furrow-Primitive-v1`` and ``Furrow-Remixed-v1``."""
    for env_id, kw in [
        ("Furrow-Macro-v1", {}),
        ("Furrow-Primitive-v1", {"action_mode": "primitive"}),
        ("Furrow-Remixed-v1", {"restoration": "remixed"}),
    ]:
        if env_id not in gym.registry:
            gym.register(id=env_id, entry_point="svrl.envs.gym:FurrowEnv", kwargs=kw, disable_env_checker=False)
