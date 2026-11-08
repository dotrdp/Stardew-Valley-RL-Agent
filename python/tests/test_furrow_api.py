import numpy as np
import pytest

furrow = pytest.importorskip("furrow")


def test_env_reset_step_by_name_and_id():
    env = furrow.Env(seed=1)
    obs, info = env.reset()
    assert obs["global_"].shape == (env.obs_global_dim,)
    assert obs["action_mask"].dtype == np.bool_ and obs["action_mask"].shape == (env.num_actions,)
    names = env.action_names()
    assert names[0] == "Wait30" and len(names) == env.num_actions
    valid = env.valid_actions()
    assert valid and all(obs["action_mask"][env.action_id(n)] for n in valid)
    obs, r, term, trunc, info = env.step("Wait30")
    assert info["outcome"] == "done" and info["ticks"] >= 5
    obs, r, term, trunc, info = env.step(env.action_id("Sleep"))
    assert info["day"] == 1 and info["day_ends"] == 1
    with pytest.raises(KeyError):
        env.step("NotAnAction")


def test_determinism_and_clone():
    a, b = furrow.Env(seed=7), furrow.Env(seed=7)
    for _ in range(50):
        x = a.expert_action()
        assert x == b.expert_action()
        a.step(x)
        b.step(x)
    assert a.stable_hash() == b.stable_hash()
    c = a.clone()
    x = a.expert_action()
    a.step(x)
    c.step(x)
    assert a.stable_hash() == c.stable_hash()


def test_expert_completes_an_episode():
    env = furrow.Env(seed=3)
    env.reset()
    info = {}
    while not env.is_done():
        _, _, _, _, info = env.step(env.expert_action())
    assert info["success"] and info["requests_done"] == info["total_requests"]


def test_vecenv_autoreset_reports_final_obs():
    vec = furrow.VecEnv(4, seed=0)
    obs, info = vec.reset()
    assert obs["global_"].shape == (4, vec.obs_global_dim)
    finished = 0
    for _ in range(5000):
        obs, rew, term, trunc, info = vec.step(vec.expert_actions())
        fin = info["_final_obs"]
        if fin.any():
            assert (info["episode_days"][fin] > 0).all()
            assert info["final_obs"]["global_"].shape == (4, vec.obs_global_dim)
            finished += int(fin.sum())
        if finished >= 2:
            break
    assert finished >= 2


def test_eval_mode_seeds_do_not_reset():
    vec = furrow.VecEnv(2, seeds=[11, 12])
    vec.reset()
    for _ in range(3000):
        _, _, _, _, info = vec.step(vec.expert_actions())
        if info["done_mask"].all():
            break
    assert vec.done_mask().all()
