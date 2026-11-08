"""tasks.py (dataset generation), prompt.py and the fake OpenAI-compatible model (no network)."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("furrow")

from nemo_gym_furrow import fake_openai, tasks  # noqa: E402
from nemo_gym_furrow.core import FurrowSession, TaskSpec  # noqa: E402
from nemo_gym_furrow.prompt import SYSTEM_PROMPT, TOOLS, user_message  # noqa: E402


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    out = tmp_path_factory.mktemp("tasks")
    tasks.main(["--out", str(out), "--train", "4", "--val", "2", "--test", "2", "--days-per-seed", "2", "--workers", "2"])
    return out


def test_splits_use_disjoint_seed_ranges(generated):
    seeds = {}
    for split in ("train", "val", "test"):
        rows = read_jsonl(generated / f"{split}.jsonl")
        assert rows and all(r["split"] == split for r in rows)
        lo, hi = tasks.SEED_RANGES[split]
        assert all(lo <= r["seed"] < hi for r in rows)
        seeds[split] = {r["seed"] for r in rows}
    assert tasks.SEED_RANGES["train"][1] <= 2**32 == tasks.SEED_RANGES["val"][0]
    assert tasks.SEED_RANGES["val"][1] <= 2**33 == tasks.SEED_RANGES["test"][0]
    assert not (seeds["train"] & seeds["val"] or seeds["train"] & seeds["test"] or seeds["val"] & seeds["test"])
    manifest = json.loads((generated / "manifest.json").read_text())
    assert manifest["files"]["train"]["seeds"] == 4


def test_rows_match_the_server_start_state(generated):
    rows = read_jsonl(generated / "train.jsonl") + read_jsonl(generated / "test.jsonl")
    for r in rows:
        assert set(r) >= {"responses_create_params", "task_id", "seed", "start_day", "horizon_days", "agent_ref"}
        s = FurrowSession(TaskSpec.from_mapping(r))
        assert s.start_info["day"] == r["start_day"]
        assert s.expert_days == pytest.approx(r["expert_days"], abs=1e-6)  # expert finishes within the horizon
        msgs = r["responses_create_params"]["input"]
        assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert msgs[1]["content"] == user_message(s.start_describe, r["start_day"], r["horizon_days"], r["max_tool_calls"])
        assert r["responses_create_params"]["tools"] == TOOLS


def test_generation_is_reproducible(generated, tmp_path):
    tasks.generate(tmp_path, {"val": 2}, days_per_seed=2)
    assert (tmp_path / "val.jsonl").read_text() == (generated / "val.jsonl").read_text()


def test_min_expert_days_filter(tmp_path):
    tasks.generate(tmp_path, {"train": 3}, days_per_seed=3, min_expert_days=1.0)
    assert all(r["expert_days"] >= 1.0 for r in read_jsonl(tmp_path / "train.jsonl"))


def test_system_prompt_is_short_and_names_every_tool():
    words = len(SYSTEM_PROMPT.split())
    # no tokenizer offline: ~4 chars/token and ~1.6 tokens/word are both generous upper estimates here
    assert len(SYSTEM_PROMPT) / 4 <= 800 and words * 1.6 <= 800
    for tool in TOOLS:
        assert tool["name"] + "(" in SYSTEM_PROMPT


# ---- fake model ------------------------------------------------------------------------------


def test_fake_policy_cycle():
    assert fake_openai.choose("u", []) == {"name": "observe", "arguments": {}}
    assert fake_openai.choose("u", [("observe", '{"state": "x"}')])["name"] == "list_actions"
    listed = json.dumps({"actions": "Sleep Wait30\nGoTo(woods|coast)", "count": 4})
    picks = set()
    for k in range(40):
        nxt = fake_openai.choose(f"u{k}", [("observe", "{}"), ("list_actions", listed)])
        assert nxt["name"] == "act"
        picks.update(nxt["arguments"]["actions"])
    assert picks <= {"Sleep", "Wait30", "GoTo(woods)", "GoTo(coast)"} and len(picks) >= 3
    done = json.dumps({"executed": [], "progress": {"done": True}})
    assert fake_openai.choose("u", [("act", done)]) is None
    assert fake_openai.choose("u", [("act", '{"error": "budget", "done": true}')]) is None


def test_fake_replies_are_valid_openai_objects():
    from nemo_gym.openai_utils import NeMoGymChatCompletion, NeMoGymResponse

    r = fake_openai.responses_reply({"input": [{"role": "user", "content": "hi"}], "model": "m"})
    resp = NeMoGymResponse.model_validate(r)
    assert resp.output[0].type == "function_call" and resp.output[0].name == "observe"
    c = fake_openai.chat_reply({"messages": [{"role": "user", "content": "hi"}]})
    chat = NeMoGymChatCompletion.model_validate(c)
    assert chat.choices[0].message.tool_calls[0].function.name == "observe"
    # chat-format history is parsed like the Responses format
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "observe"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "{}"},
    ]
    assert json.loads(fake_openai.chat_reply({"messages": msgs})["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]) == {}


def test_fake_model_plays_a_session_to_the_end():
    """The fake policy against a session in process: a full offline rollout without HTTP."""
    s = FurrowSession(TaskSpec(seed=5, start_day=20, horizon_days=2, max_tool_calls=200))
    outputs: list[tuple[str, str]] = []
    while True:
        nxt = fake_openai.choose("task", outputs)
        if nxt is None:
            break
        tool = getattr(s, nxt["name"])
        out = tool(**nxt["arguments"]) if nxt["arguments"] else tool()
        outputs.append((nxt["name"], json.dumps(out)))
    assert s.done and s.macros_executed > 0
    assert s.result()["end_reason"] in {"horizon", "success", "tool_budget"}
