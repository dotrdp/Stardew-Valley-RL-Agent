"""The NeMo Gym resources server in process (FastAPI TestClient; no sockets, no other servers)."""

from __future__ import annotations

import pytest

pytest.importorskip("nemo_gym")
pytest.importorskip("furrow")

from fastapi.testclient import TestClient  # noqa: E402
from nemo_gym.base_resources_server import ResourcesSeedSessionRequest  # noqa: E402
from nemo_gym.episode_types import EpisodeId, TaskId  # noqa: E402
from nemo_gym.testing.session_conformance import check_resources_session_contract  # noqa: E402

from nemo_gym_furrow.core import FurrowSession, TaskSpec, expand_actions  # noqa: E402
from nemo_gym_furrow.fake_openai import responses_reply  # noqa: E402
from nemo_gym_furrow.prompt import TOOLS  # noqa: E402


def verify_body(row: dict) -> dict:
    """A verify request as the agent sends it: the row plus the final Responses-API object."""
    return {**row, "response": responses_reply({"input": row["responses_create_params"]["input"]})}


def test_tool_routes_match_the_tool_schemas(make_server):
    paths = {r.path for r in make_server().setup_webserver().routes}
    for tool in TOOLS:
        assert f"/{tool['name']}" in paths
    assert {"/seed_session", "/verify", "/close_session", "/aggregate_metrics"} <= paths


def test_rollout_through_routes(make_server, example_rows):
    row = example_rows[0]
    client = TestClient(make_server().setup_webserver())
    assert client.post("/seed_session", json=row).status_code == 200
    obs = client.post("/observe", json={}).json()
    spec = TaskSpec.from_mapping(row)
    assert obs["state"] == FurrowSession(spec).start_describe
    assert obs["state"] in row["responses_create_params"]["input"][1]["content"]  # prompt shows the same state
    actions = expand_actions(client.post("/list_actions", json={}).json()["actions"])
    out = client.post("/act", json={"actions": [actions[0]]}).json()
    assert out["executed"][0]["action"] == actions[0]
    out = client.post("/act", json={"actions": "Bogus"}).json()  # a bare string is accepted too
    assert out["errors"][0]["error"] == "unknown action name"
    st = client.post("/status", json={}).json()
    assert st["progress"]["tool_calls_used"] == 5 and st["progress"]["invalid_actions"] == 1
    v = client.post("/verify", json=verify_body(row)).json()
    assert v["mask_sample"] is False and 0.0 <= v["reward"] < 0.5
    assert v["tool_calls"] == 5 and v["invalid_actions"] == 1 and v["macros_executed"] == 1
    assert v["task_id"] == row["task_id"] and v["expert_days"] == pytest.approx(row["expert_days"], abs=1e-6)
    # the session is gone; a retried verify returns the cached result
    assert client.post("/observe", json={}).status_code == 404
    assert client.post("/verify", json=verify_body(row)).json()["reward"] == v["reward"]


def test_sessions_are_per_cookie(make_server, example_rows):
    app = make_server().setup_webserver()
    a, b = TestClient(app), TestClient(app)
    a.post("/seed_session", json=example_rows[0])
    b.post("/seed_session", json=example_rows[1])
    a.post("/act", json={"actions": ["Sleep"]})
    sa = a.post("/status", json={}).json()["progress"]
    sb = b.post("/status", json={}).json()["progress"]
    assert sa["tool_calls_used"] == 2 and sb["tool_calls_used"] == 1
    assert sa["now"] != sb["now"]
    assert TestClient(app).post("/status", json={}).status_code == 404  # no session for a fresh cookie


def test_verify_without_session_is_masked(make_server, example_rows):
    client = TestClient(make_server().setup_webserver())
    v = client.post("/verify", json=verify_body(example_rows[0])).json()
    assert v["reward"] == 0.0 and v["mask_sample"] is True and v["failure_kind"] == "session_lost"


def test_bad_task_rows_are_rejected(make_server, example_rows):
    client = TestClient(make_server().setup_webserver())
    row = {**example_rows[0], "start_day": 100_000}
    assert client.post("/seed_session", json=row).status_code == 422
    row = {k: v for k, v in example_rows[0].items() if k != "seed"}
    assert client.post("/seed_session", json=row).status_code == 422


def test_server_defaults_fill_missing_row_fields(make_server, example_rows):
    row = {k: v for k, v in example_rows[0].items() if k not in ("horizon_days", "max_tool_calls")}
    client = TestClient(make_server(default_horizon_days=2, default_max_tool_calls=4).setup_webserver())
    client.post("/seed_session", json=row)
    p = client.post("/status", json={}).json()["progress"]
    assert p["day_ends_left"] == 2 and p["tool_calls_left"] == 3


def seed_request(sid: str, row: dict) -> ResourcesSeedSessionRequest:
    return ResourcesSeedSessionRequest(
        resources_session_id=sid,
        episode_id=EpisodeId(rollout_id=f"rollout-{sid}", attempt=0),
        task_id=TaskId(taskset="furrow", task_id=row["task_id"]),
        task_data={k: row[k] for k in ("seed", "start_day", "horizon_days", "max_tool_calls")},
    )


def test_environment_server_session_contract(make_server, example_rows):
    check_resources_session_contract(
        make_server().setup_webserver(), seed_request("contract", example_rows[0]), keeps_state=True
    )


def test_environment_server_seed_reaches_the_tools(make_server, example_rows):
    client = TestClient(make_server().setup_webserver())
    res = client.post("/seed_session", json=seed_request("env-1", example_rows[2]).model_dump(mode="json"))
    assert res.status_code == 200 and res.json()["resources_session_id"] == "env-1"
    assert client.post("/act", json={"actions": ["Wait30"]}).json()["progress"]["tool_calls_used"] == 1
    close = {"resources_session_id": "env-1", "episode_id": {"rollout_id": "rollout-env-1", "attempt": 0}}
    assert client.post("/close_session", json=close).status_code == 200
    assert client.post("/status", json={}).status_code == 404


def test_aggregate_metrics(make_server, example_rows):
    app = make_server().setup_webserver()
    results = []
    for i, row in enumerate(example_rows[:3]):
        c = TestClient(app)
        c.post("/seed_session", json=row)
        c.post("/act", json={"actions": ["Sleep"]})
        results.append({**c.post("/verify", json=verify_body(row)).json(), "_ng_task_index": i, "_ng_rollout_index": 0})
    for r in results:
        r.pop("responses_create_params")
    agg = TestClient(app).post("/aggregate_metrics", json={"verify_responses": results}).json()
    assert agg["agent_metrics"]["success_rate"] == 0.0
    assert agg["key_metrics"]["mean/reward"] == pytest.approx(sum(r["reward"] for r in results) / 3)
