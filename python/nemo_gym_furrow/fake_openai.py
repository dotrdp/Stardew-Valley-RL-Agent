"""A tiny fake OpenAI-compatible model server for offline end-to-end runs.

Serves ``POST /v1/responses`` (used by Gym's ``openai_model`` server), ``POST /v1/chat/completions``
(used by Gym's ``vllm_model`` server) and ``GET /v1/models``. It never generates text: it reads the
conversation and replies with the next tool call of a fixed random policy:

    observe -> list_actions -> act([random valid action]) -> list_actions -> act(...) -> ...

with a short random day plan (2-3 names, possibly invalid later in the plan) 20% of the time. It
stops with a plain assistant message once a tool output reports ``"done": true``. Choices are seeded
from the conversation, so a rollout is reproducible.

    python -m nemo_gym_furrow.fake_openai --port 8765
"""

import argparse
import hashlib
import json
import random
import time
import uuid
from typing import Any, Optional

from nemo_gym_furrow.core import expand_actions

MODEL_NAME = "furrow-fake-random"


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return ""


def _parse_responses_input(items: Any) -> tuple[str, list[tuple[str, str]]]:
    """(first user text, [(tool name, tool output), ...]) from a Responses API ``input``."""
    if isinstance(items, str):
        return items, []
    first_user, calls, outputs = "", {}, []
    for it in items or []:
        kind = it.get("type", "message")
        if kind == "message" and it.get("role") == "user" and not first_user:
            first_user = _text_of(it.get("content"))
        elif kind == "function_call":
            calls[it.get("call_id")] = it.get("name", "")
        elif kind == "function_call_output":
            outputs.append((calls.get(it.get("call_id"), ""), str(it.get("output", ""))))
    return first_user, outputs


def _parse_chat_messages(messages: list[dict]) -> tuple[str, list[tuple[str, str]]]:
    first_user, calls, outputs = "", {}, []
    for m in messages or []:
        role = m.get("role")
        if role == "user" and not first_user:
            first_user = _text_of(m.get("content"))
        elif role == "assistant":
            for tc in m.get("tool_calls") or []:
                calls[tc.get("id")] = (tc.get("function") or {}).get("name", "")
        elif role == "tool":
            outputs.append((calls.get(m.get("tool_call_id"), ""), _text_of(m.get("content"))))
    return first_user, outputs


def _json(text: str) -> dict:
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _is_done(payload: dict) -> bool:
    progress = payload.get("progress") if isinstance(payload.get("progress"), dict) else {}
    return bool(payload.get("done") or progress.get("done"))


def choose(first_user: str, outputs: list[tuple[str, str]], seed: int = 0, max_calls: int = 400) -> Optional[dict]:
    """Next tool call as ``{"name": ..., "arguments": {...}}``, or ``None`` to stop."""
    if len(outputs) >= max_calls:
        return None
    if not outputs:
        return {"name": "observe", "arguments": {}}
    last_name, last_text = outputs[-1]
    last = _json(last_text)
    if _is_done(last):
        return None
    if last_name != "list_actions" or "actions" not in last:
        return {"name": "list_actions", "arguments": {}}
    digest = hashlib.sha256(f"{seed}|{len(outputs)}|{first_user}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "little"))
    valid = expand_actions(last["actions"]) or ["Sleep"]
    if rng.random() < 0.2 and len(valid) > 1:
        plan = [rng.choice(valid) for _ in range(rng.randint(2, 3))]
    else:
        plan = [rng.choice(valid)]
    return {"name": "act", "arguments": {"actions": plan}}


def responses_reply(body: dict, seed: int = 0, max_calls: int = 400) -> dict:
    first_user, outputs = _parse_responses_input(body.get("input"))
    nxt = choose(first_user, outputs, seed, max_calls)
    if nxt is None:
        item: dict[str, Any] = {
            "type": "message",
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Segment finished.", "annotations": []}],
        }
    else:
        item = {
            "type": "function_call",
            "id": f"fc_{uuid.uuid4().hex[:24]}",
            "call_id": f"call_{uuid.uuid4().hex[:24]}",
            "name": nxt["name"],
            "arguments": json.dumps(nxt["arguments"]),
            "status": "completed",
        }
    return {
        "id": f"resp_{uuid.uuid4().hex[:24]}",
        "object": "response",
        "created_at": time.time(),
        "model": body.get("model") or MODEL_NAME,
        "status": "completed",
        "output": [item],
        "parallel_tool_calls": False,
        "tool_choice": body.get("tool_choice") or "auto",
        "tools": body.get("tools") or [],
        "usage": {
            "input_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 1,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 1,
        },
    }


def chat_reply(body: dict, seed: int = 0, max_calls: int = 400) -> dict:
    first_user, outputs = _parse_chat_messages(body.get("messages") or [])
    nxt = choose(first_user, outputs, seed, max_calls)
    if nxt is None:
        message: dict[str, Any] = {"role": "assistant", "content": "Segment finished."}
        finish = "stop"
    else:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {"name": nxt["name"], "arguments": json.dumps(nxt["arguments"])},
                }
            ],
        }
        finish = "tool_calls"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model") or MODEL_NAME,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 1, "total_tokens": 1},
    }


def create_app(seed: int = 0, max_calls: int = 400):
    # No `from __future__ import annotations` in this module: FastAPI must see `Request` as a class.
    from fastapi import FastAPI, Request

    app = FastAPI(title="furrow fake OpenAI-compatible model")

    @app.get("/v1/models")
    async def models() -> dict:
        return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "furrow"}]}

    @app.post("/v1/responses")
    async def responses(request: Request) -> dict:
        return responses_reply(await request.json(), seed, max_calls)

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> dict:
        return chat_reply(await request.json(), seed, max_calls)

    return app


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-calls", type=int, default=400, help="stop after this many tool outputs")
    a = p.parse_args(argv)
    import uvicorn

    uvicorn.run(create_app(a.seed, a.max_calls), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
