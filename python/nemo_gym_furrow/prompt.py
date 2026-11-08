"""System prompt, tool schemas and the dataset-row builder for Furrow segment tasks."""

from __future__ import annotations

from typing import Any, Optional

SYSTEM_PROMPT = """\
You are playing Furrow, a farming life-sim, through tools. The farm belongs to an old mill that \
the town wants restored. The Old Mill has 6 wings and 24 restoration requests. Each request lists \
item options ("0/3 of: radish x5, pea x5, ...": deliver any 3 of the options in the listed \
quantities) or a coin payment. Items come from farming (buy seeds, plant, water, harvest; crops \
grow for several days and only in their season), foraging, fishing, mining, chopping wood, \
crafting machines that process goods, and the shops. Deliver hands in held items that fill open \
request slots (a slot needs its full quantity); coin requests are paid with PayCoins(...).

Time: a day runs from 06:00 until you Sleep; staying up past 02:00 makes you pass out. Every \
action costs in-game minutes and most cost energy; sleeping restores energy. Seasons last 24 days.

Your task is one segment of a longer run. Complete ANY ONE new restoration request as fast as \
possible, in in-game time, before the segment's deadline (usually 7 days). Finishing sooner gives \
a higher score. A failed segment earns only small partial credit for moving an open request \
forward. Tool calls and day-plan lengths are capped, so plan whole days at once.

Tools:
- observe(): the full state: date, weather, location, energy, money, bag, crops, machines and every \
open request with how many of each item you already have. Also returns segment progress.
- list_actions(): the macro actions valid right now, grouped. "GoTo(woods|coast)" means \
GoTo(woods) or GoTo(coast); "Buy(pea_seed,1|5)" means Buy(pea_seed,1) or Buy(pea_seed,5).
- act(actions): run one macro, e.g. ["Fish(woods_river)"], or a day plan, e.g. ["GoTo(woods)", \
"Forage(woods)", "GoTo(homestead)", "Deliver", "Sleep"]. The plan runs in order and stops at the \
first invalid name (you get close matches), when the day ends, or when the segment ends. Returns \
what ran, the new compact state and progress.
- status(): compact state and progress (time, day-ends and tool calls left).

Macros are composite: GoTo walks there; Forage/Fish/MineFloor/MineDescend/ChopWood work for up to \
two hours; Plant(crop,field) tills and plants all held seeds of that crop; WaterCrops, Harvest, LoadMachines, \
CollectMachines, ShipSurplus (sell unneeded items), Craft(...), UpgradeTool(...), Eat, Wait30, \
Sleep. An action can only run where and when it makes sense, so check list_actions when unsure.

Work efficiently: read the open requests, pick the one you can finish soonest with what you hold \
and what is reachable this season, gather exactly that, then Deliver. When the segment is over, \
reply with a one-line summary and no tool call."""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "observe",
        "description": "Full text state (date, location, energy, money, bag, crops, machines, open requests "
        "with items held) plus segment progress.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        "strict": True,
    },
    {
        "type": "function",
        "name": "list_actions",
        "description": "Macro actions valid right now, grouped: Verb(a|b) means Verb(a) or Verb(b); "
        "Buy(x,1|5) means Buy(x,1) or Buy(x,5).",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        "strict": True,
    },
    {
        "type": "function",
        "name": "act",
        "description": "Run one macro or a day plan (list of macros run in order; stops at the first invalid "
        "name, when the day ends, or when the segment ends). Returns results, the new compact state and progress.",
        "parameters": {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": 'Macro names, e.g. ["GoTo(woods)", "Forage(woods)", "Sleep"].',
                }
            },
            "required": ["actions"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "status",
        "description": "Compact state and segment progress (time, day-ends left, tool calls left, invalid actions).",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        "strict": True,
    },
]


def user_message(start_describe: str, start_day: int, horizon_days: int, max_tool_calls: int) -> str:
    return (
        f"Segment start: Day {start_day + 1}. Goal: complete any one new restoration request before Day "
        f"{start_day + horizon_days + 1} begins ({horizon_days} day-ends at most), using at most "
        f"{max_tool_calls} tool calls.\n\nCurrent state:\n{start_describe}"
    )


def build_row(
    *,
    task_id: str,
    split: str,
    seed: int,
    start_day: int,
    horizon_days: int,
    max_tool_calls: int,
    start_describe: str,
    expert_days: Optional[float],
    agent_name: Optional[str] = "furrow_simple_agent",
    env_config: Optional[dict] = None,
    temperature: Optional[float] = None,
) -> dict[str, Any]:
    """One NeMo Gym dataset row: ``responses_create_params`` (prompt + tools) plus the task fields the
    resources server reads in ``/seed_session`` (``seed``, ``start_day``, ``horizon_days``, ...)."""
    params: dict[str, Any] = {
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message(start_describe, start_day, horizon_days, max_tool_calls)},
        ],
        "tools": TOOLS,
        "parallel_tool_calls": False,
    }
    if temperature is not None:
        params["temperature"] = temperature
    row: dict[str, Any] = {
        "responses_create_params": params,
        "task_id": task_id,
        "split": split,
        "seed": seed,
        "start_day": start_day,
        "horizon_days": horizon_days,
        "max_tool_calls": max_tool_calls,
        "expert_days": expert_days,
    }
    if env_config:
        row["env_config"] = env_config
    if agent_name:
        row["agent_ref"] = {"type": "responses_api_agents", "name": agent_name}
    return row
