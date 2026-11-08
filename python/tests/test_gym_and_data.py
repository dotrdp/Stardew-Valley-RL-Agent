import json

import numpy as np
import pytest

furrow = pytest.importorskip("furrow")
gym = pytest.importorskip("gymnasium")


def test_gymnasium_env_checker():
    import svrl.envs  # noqa: F401
    from gymnasium.utils.env_checker import check_env

    env = gym.make("Furrow-Macro-v1")
    check_env(env.unwrapped, skip_render_check=True)


def test_vector_env_same_step():
    from gymnasium.vector import AutoresetMode

    from svrl.envs import FurrowVecEnv

    v = FurrowVecEnv(3, seed=2)
    assert v.metadata["autoreset_mode"] == AutoresetMode.SAME_STEP
    obs, _ = v.reset()
    obs, r, te, tr, info = v.step(v.expert_actions())
    assert obs["global"].shape == (3, v.single_observation_space["global"].shape[0])
    assert "final_obs" in info and "_final_obs" in info


def test_generate_and_load_dataset(tmp_path):
    from svrl.data import Dataset

    s = furrow.generate_demos(str(tmp_path / "ds"), episodes=3, seed=0, noise=0.1, num_threads=2)
    assert s["episodes"] == 3 and s["steps"] > 100
    ds = Dataset(tmp_path / "ds")
    assert len(ds) == 3 and ds.num_steps == s["steps"]
    ep = ds.episode(0)
    assert ep["obs_global"].dtype == np.float32 and ep["mask"].shape[1] == ds.num_actions
    # every executed action was valid under the recorded mask
    assert ep["mask"][np.arange(len(ep["action"])), ep["action"]].all()
    m = ds.materialize()
    assert m["obs"].shape[0] == s["steps"]
    meta = json.loads((tmp_path / "ds" / "meta.json").read_text())
    assert meta["format"] == "furrow-traj/2" and len(meta["action_names"]) == ds.num_actions


def test_bc_and_evaluate_smoke(tmp_path):
    torch = pytest.importorskip("torch")
    from svrl import evaluate as ev
    from svrl.bc import train_bc
    from svrl.models import Policy

    furrow.generate_demos(str(tmp_path / "ds"), episodes=4, seed=1, num_threads=2)
    m = train_bc([str(tmp_path / "ds")], str(tmp_path / "bc.pt"), steps=200, batch=256, max_vram_gb=1.0, log_every=100)
    assert m["val_acc"] > 0.2
    pol = Policy.load(str(tmp_path / "bc.pt"), "cpu").eval()
    r = ev.run_policy(ev.torch_policy(pol, "cpu"), [2**33, 2**33 + 1], max_days=30)
    s = ev.summarize(r, max_days=30, expert=r)
    assert s["n"] == 2 and 0.0 <= s["success_rate"] <= 1.0
    del torch
