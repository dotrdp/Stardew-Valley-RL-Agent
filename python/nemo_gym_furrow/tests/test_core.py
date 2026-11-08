"""Sessions, tools and the verify reward (no network, no NeMo Gym)."""

from __future__ import annotations

import random
import threading

import pytest

furrow = pytest.importorskip("furrow")

from nemo_gym_furrow.core import (  # noqa: E402
    FAILURE_CAP,
    FurrowSession,
    SessionClosedError,
    SessionStore,
    TaskSpec,
    expand_actions,
    expert_reference,
    group_actions,
    partial_progress,
    request_fractions,
    segment_reward,
    start_env,
)

SPEC = TaskSpec(seed=5, start_day=20, horizon_days=7, max_tool_calls=400)


def expert_name(session: FurrowSession) -> str:
    return session.all_names[session.env.expert_action()]


# ---- action grouping -------------------------------------------------------------------------


def test_group_expand_roundtrip_all_names_and_valid_sets():
    env = furrow.Env(seed=3)
    env.reset(seed=3)
    names = list(env.action_names())
    assert sorted(expand_actions(group_actions(names))) == sorted(names)
    for _ in range(300):  # valid sets along an expert run
        valid = list(env.valid_actions())
        grouped = group_actions(valid)
        assert sorted(expand_actions(grouped)) == sorted(valid)
        assert len(grouped) < len(", ".join(valid)) + 1
        env.step(env.expert_action())


# ---- start states and determinism ------------------------------------------------------------


def test_start_state_and_expert_reference_are_deterministic():
    a, b = FurrowSession(SPEC), FurrowSession(SPEC)
    assert a.env.stable_hash() == b.env.stable_hash()
    assert a.start_describe == b.start_describe
    assert a.expert_days == b.expert_days is not None
    # first state with day >= start_day (one Sleep can end two days)
    assert SPEC.start_day <= a.start_info["day"] <= SPEC.start_day + 2
    # the reference rollout runs on a clone: the session env is untouched
    assert a.env.stable_hash() == start_env(SPEC).stable_hash()
    assert expert_reference(a.env, SPEC.horizon_days) == a.expert_days


def test_same_actions_same_outcome():
    rng = random.Random(7)
    a, b = FurrowSession(SPEC), FurrowSession(SPEC)
    for _ in range(40):
        if a.done:
            break
        name = rng.choice(expand_actions(a.list_actions()["actions"]))
        b.list_actions()
        assert a.act([name]) == b.act([name])
    assert a.env.stable_hash() == b.env.stable_hash()
    assert a.result() == b.result()


def test_start_day_beyond_the_run_is_rejected():
    with pytest.raises(ValueError):
        start_env(TaskSpec(seed=5, start_day=5000))


# ---- tools -----------------------------------------------------------------------------------


def test_observe_status_list_actions():
    s = FurrowSession(SPEC)
    obs = s.observe()
    assert obs["state"] == s.start_describe and obs["progress"]["tool_calls_used"] == 1
    st = s.status()
    assert st["state"].splitlines()[0] == s.start_describe.splitlines()[0]
    assert "The Old Mill:" in st["state"] and st["progress"]["day_ends_left"] == SPEC.horizon_days
    la = s.list_actions()
    assert sorted(expand_actions(la["actions"])) == sorted(s.env.valid_actions())
    assert la["count"] == len(s.env.valid_actions())
    assert s.tool_calls == 3


def test_unknown_and_masked_actions_are_errors_with_close_matches():
    s = FurrowSession(SPEC)
    before = s.env.stable_hash()
    out = s.act("Fish(woods_rivr)")
    assert out["errors"][0]["error"] == "unknown action name"
    assert "Fish(woods_river)" in out["errors"][0]["did_you_mean"]
    assert out["executed"] == [] and s.invalid_actions == 1
    assert s.env.stable_hash() == before  # nothing ran
    masked = next(n for n in s.all_names if n not in s.env.valid_actions())
    out = s.act([masked, "Sleep"])
    assert out["errors"][0]["error"].startswith("not valid") and out["not_run"] == ["Sleep"]
    assert s.invalid_actions == 2 and s.env.stable_hash() == before


def test_day_plan_runs_in_order_and_stops_at_day_end():
    s = FurrowSession(SPEC)
    day = s.env.info()["day"]
    out = s.act(["Wait30", "Sleep", "Wait30"])
    assert [r["action"] for r in out["executed"]] == ["Wait30", "Sleep"]
    assert out["not_run"] == ["Wait30"] and "day ended" in out["note"]
    assert s.env.info()["day"] > day and s.macros_executed == 2


def test_plan_length_cap_and_empty_plan():
    s = FurrowSession(TaskSpec(seed=5, start_day=20, max_plan_length=3))
    assert "at most 3" in s.act(["Wait30"] * 4)["error"]
    assert "at least one" in s.act([])["error"]
    assert s.macros_executed == 0


def test_tool_budget_ends_the_segment():
    s = FurrowSession(TaskSpec(seed=5, start_day=20, max_tool_calls=3))
    for _ in range(3):
        assert "error" not in s.status()
    out = s.act(["Wait30"])
    assert out["done"] and out["end_reason"] == "tool_budget" and s.macros_executed == 0
    assert s.result()["end_reason"] == "tool_budget"


def test_horizon_ends_the_segment_and_further_acts_are_refused():
    s = FurrowSession(TaskSpec(seed=5, start_day=20, horizon_days=2, max_tool_calls=50))
    s.act(["Sleep"])
    out = s.act(["Sleep"])
    assert out["progress"]["done"] and out["progress"]["end_reason"] == "horizon"
    out = s.act(["Wait30"])
    assert "segment is over" in out["error"] and out["not_run"] == ["Wait30"]
    r = s.result()
    assert not r["success"] and r["days_played"] == pytest.approx(2.0, abs=0.01) and r["reward"] <= FAILURE_CAP


def test_expert_as_agent_matches_the_reference_exactly():
    s = FurrowSession(SPEC)
    while not s.done:
        s.act([expert_name(s)])
    r = s.result()
    assert r["success"] and r["end_reason"] == "success"
    assert r["days_to_success"] == pytest.approx(s.expert_days)
    assert r["reward"] == pytest.approx(0.75) and r["speed_ratio"] == pytest.approx(1.0)
    assert r["requests_gained"] == 1


def test_concurrent_calls_on_one_session_are_serialized():
    s = FurrowSession(TaskSpec(seed=5, start_day=20, max_tool_calls=10_000))
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for _ in range(25):
                s.status()
                s.act(["Wait30"])
        except BaseException as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and s.tool_calls == 200


# ---- reward ----------------------------------------------------------------------------------


def test_reward_is_lexicographic():
    rng = random.Random(0)
    successes, failures = [], []
    for _ in range(2000):
        h = rng.randint(1, 14)
        e = rng.choice([None, rng.uniform(0, h)])
        successes.append(segment_reward(True, rng.uniform(0, h), e, h, 1.0))
        failures.append(segment_reward(False, None, e, h, rng.uniform(-1, 2)))
    assert min(successes) > 0.5 > FAILURE_CAP >= max(failures) >= 0.0
    assert max(successes) < 1.0


def test_reward_prefers_faster_success_relative_to_expert():
    r = [segment_reward(True, a, 2.0, 7, 1.0) for a in (0.0, 0.5, 1.0, 2.0, 4.0, 7.0)]
    assert r == sorted(r, reverse=True) and len(set(r)) == len(r)
    assert segment_reward(True, 2.0, 2.0, 7, 1.0) == pytest.approx(0.75)
    # same agent time: an easier start state (smaller expert time) scores lower
    assert segment_reward(True, 3.0, 1.0, 7, 1.0) < segment_reward(True, 3.0, 5.0, 7, 1.0)
    # an expert failure is scored as if the expert took the whole horizon
    assert segment_reward(True, 3.0, None, 7, 1.0) == segment_reward(True, 3.0, 7.0, 7, 1.0)
    with pytest.raises(ValueError):
        segment_reward(True, None, 2.0, 7, 1.0)


def test_partial_progress_from_describe():
    start = (
        "You are at Homestead (5,4). Energy 100/200. Money 1000. Bag 16/24 slots free.\n"
        "  * River Catch — 1/3 of: minnow x3 (have 0), ✓brown_trout x1 (have 0), perch x1 (have 0)\n"
        "  * The Ledger — pay 8000 coins\n"
    )
    end = (
        "You are at Homestead (5,4). Energy 100/200. Money 4000. Bag 16/24 slots free.\n"
        "  * River Catch — 2/3 of: ✓minnow x3 (have 0), ✓brown_trout x1 (have 0), perch x1 (have 0)\n"
        "  * The Ledger — pay 8000 coins\n"
    )
    fr = request_fractions(start)
    assert fr["River Catch"] == pytest.approx(1 / 3) and fr["The Ledger"] == pytest.approx(0.5 * 1000 / 8000)
    assert partial_progress(start, end) == pytest.approx(1 / 3)
    assert partial_progress(end, start) == 0.0
    # items held toward an open slot count half
    held = start.replace("minnow x3 (have 0)", "minnow x3 (have 3)")
    assert partial_progress(start, held) == pytest.approx(0.5 / 3)


def test_failed_segment_gets_partial_credit_below_success():
    s = FurrowSession(TaskSpec(seed=5, start_day=20, horizon_days=1, max_tool_calls=50))
    s.act(["Sleep"])
    r = s.result()
    assert not r["success"] and 0.0 <= r["reward"] <= FAILURE_CAP
    assert r["reward"] == pytest.approx(FAILURE_CAP * r["progress"])


# ---- session store ---------------------------------------------------------------------------


def test_store_is_idempotent_closes_and_caches_results():
    store = SessionStore()
    a = store.create("s1", SPEC)
    assert store.create("s1", SPEC) is a
    with pytest.raises(ValueError):
        store.create("s1", TaskSpec(seed=6, start_day=1))
    a.act(["Sleep"])
    r1 = store.finish("s1")
    assert store.get("s1") is None and len(store) == 0
    assert store.finish("s1") == r1  # retried verify gets the same answer
    assert store.finish("never") is None
    store.close("s2")
    with pytest.raises(SessionClosedError):
        store.create("s2", SPEC)


def test_store_evicts_idle_and_excess_sessions():
    store = SessionStore(ttl_seconds=3600, max_sessions=2)
    for i in range(3):
        store.create(f"s{i}", TaskSpec(seed=5, start_day=i))
    assert len(store) == 2 and store.get("s0") is None
    store.ttl_seconds = -1.0
    store.create("s9", TaskSpec(seed=5, start_day=1))
    assert len(store) == 1 and store.get("s9") is not None


def test_task_spec_from_row_and_validation():
    spec = TaskSpec.from_mapping({"seed": 1, "start_day": 2, "horizon_days": None}, horizon_days=3)
    assert spec == TaskSpec(seed=1, start_day=2, horizon_days=3)
    with pytest.raises(ValueError):
        TaskSpec.from_mapping({"seed": 1})
    with pytest.raises(ValueError):
        TaskSpec(seed=1, start_day=0, horizon_days=0)
