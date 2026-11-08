"""Furrow — an original farming-sim RL environment (Rust core, Python bindings).

Quick start::

    import furrow
    env = furrow.Env(seed=0)                      # macro-action mode (default)
    obs, info = env.reset()
    print(env.describe())                          # text state
    print(env.valid_actions()[:10])                # names of valid macros
    obs, reward, terminated, truncated, info = env.step("Sleep")   # id or name
    a = env.expert_action()                        # scripted speedrun expert's choice

    vec = furrow.VecEnv(64, seed=0)               # 64 envs stepped in parallel (Rust threads)
    obs, info = vec.reset()
    obs, rew, term, trunc, info = vec.step(vec.expert_actions())

    furrow.generate_demos("data/expert", episodes=100, noise=0.05)  # furrow-traj/2 dataset

Gymnasium wrappers live in ``svrl.envs`` (``gymnasium.make("Furrow-Macro-v1")``).
"""
from ._furrow import (  # noqa: F401
    SIM_VERSION,
    __version__,
    content_hash,
    content_sources,
    generate_demos,
    item_keys,
    macro_action_names,
    obs_layout,
)
from ._furrow import PyEnv as Env  # noqa: F401
from ._furrow import PyVecEnv as VecEnv  # noqa: F401

__all__ = [
    "Env",
    "VecEnv",
    "generate_demos",
    "content_hash",
    "content_sources",
    "obs_layout",
    "macro_action_names",
    "item_keys",
    "SIM_VERSION",
    "__version__",
]
