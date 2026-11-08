"""Gymnasium wrappers for Furrow.

>>> import gymnasium as gym, svrl.envs
>>> env = gym.make("Furrow-Macro-v1")
>>> obs, info = env.reset(seed=0)
"""
from .gym import FurrowEnv, FurrowVecEnv, register_envs

register_envs()

__all__ = ["FurrowEnv", "FurrowVecEnv", "register_envs"]
