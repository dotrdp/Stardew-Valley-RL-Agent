"""Furrow segment tasks for LLM agents: start states, tools, the verify reward and a session store.

Nothing in this module imports NeMo Gym, so it is unit-testable on its own. The NeMo Gym resources
server (``resources_servers/furrow/app.py``) is a thin HTTP layer over :class:`SessionStore` and
:class:`FurrowSession`.

A task is a *segment* of a full run (DESIGN.md §9): replay the scripted expert on ``seed`` until the
day counter reaches ``start_day``, then the LLM plays for at most ``horizon_days`` in-game days. The
segment succeeds as soon as any new restoration request is completed.
"""

from __future__ import annotations

import difflib
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import furrow

MINUTES_PER_DAY = 1440
DEFAULT_HORIZON_DAYS = 7
DEFAULT_MAX_TOOL_CALLS = 60
DEFAULT_MAX_PLAN_LENGTH = 40
# Reward constants (see segment_reward).
SUCCESS_FLOOR = 0.5
FAILURE_CAP = 0.45
SPEED_SMOOTHING_DAYS = 0.25


# --------------------------------------------------------------------------------------------
# Task spec and start states
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskSpec:
    """What a dataset row asks for. ``start_day`` is Furrow's 0-based ``info["day"]``
    (``describe()`` prints it as ``Day start_day + 1``)."""

    seed: int
    start_day: int
    horizon_days: int = DEFAULT_HORIZON_DAYS
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    max_plan_length: int = DEFAULT_MAX_PLAN_LENGTH
    env_config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.seed < 0 or self.start_day < 0:
            raise ValueError("seed and start_day must be >= 0")
        if self.horizon_days < 1 or self.max_tool_calls < 1 or self.max_plan_length < 1:
            raise ValueError("horizon_days, max_tool_calls and max_plan_length must be >= 1")

    @classmethod
    def from_mapping(cls, row: dict[str, Any], **defaults: Any) -> "TaskSpec":
        """Build from a dataset row / task_data dict; ``defaults`` fill keys the row leaves out."""

        def pick(key: str, fallback: Any) -> Any:
            value = row.get(key)
            return fallback if value is None else value

        if row.get("seed") is None or row.get("start_day") is None:
            raise ValueError("task row needs integer 'seed' and 'start_day'")
        return cls(
            seed=int(row["seed"]),
            start_day=int(row["start_day"]),
            horizon_days=int(pick("horizon_days", defaults.get("horizon_days", DEFAULT_HORIZON_DAYS))),
            max_tool_calls=int(pick("max_tool_calls", defaults.get("max_tool_calls", DEFAULT_MAX_TOOL_CALLS))),
            max_plan_length=int(pick("max_plan_length", defaults.get("max_plan_length", DEFAULT_MAX_PLAN_LENGTH))),
            env_config=dict(pick("env_config", {})),
        )


def start_env(spec: TaskSpec) -> "furrow.Env":
    """Replay the expert on ``spec.seed`` until ``info["day"] >= spec.start_day``. Deterministic."""
    env = furrow.Env(seed=spec.seed, **spec.env_config)
    env.reset(seed=spec.seed)
    while env.info()["day"] < spec.start_day:
        if env.is_done():
            raise ValueError(f"expert run on seed {spec.seed} ended before day {spec.start_day}")
        env.step(env.expert_action())
    if env.is_done():
        raise ValueError(f"expert run on seed {spec.seed} is already finished at day {spec.start_day}")
    return env


def elapsed_days(start_info: dict, info: dict) -> float:
    return (info["abs_minute"] - start_info["abs_minute"]) / MINUTES_PER_DAY


def expert_reference(env: "furrow.Env", horizon_days: int) -> Optional[float]:
    """Days the expert needs from ``env``'s state (on a clone) to complete one more request,
    or ``None`` if it does not within ``horizon_days``."""
    sim = env.clone()
    start = sim.info()
    deadline_day = start["day"] + horizon_days
    while not sim.is_done():
        _, _, _, _, info = sim.step(sim.expert_action())
        if info["requests_done"] > start["requests_done"]:
            return elapsed_days(start, info)
        if info["day"] >= deadline_day:
            break
    return None


# --------------------------------------------------------------------------------------------
# Compact action lists
# --------------------------------------------------------------------------------------------

_NAME_RE = re.compile(r"^(?P<verb>[A-Za-z0-9_]+)(?:\((?P<args>[^()]*)\))?$")


def group_actions(names: Iterable[str]) -> str:
    """Group macro names compactly, one line per verb (verbs in first-appearance order).

    ``GoTo(woods|coast)`` means ``GoTo(woods)`` or ``GoTo(coast)``; for several arguments the last
    one varies: ``Buy(pea_seed,1|5; corn_seed,1)``. Argument-free macros share the first line.
    :func:`expand_actions` recovers exactly the same names (grouped order).
    """
    plain: list[str] = []
    verbs: "OrderedDict[str, OrderedDict[str, list[str]]]" = OrderedDict()
    for name in names:
        m = _NAME_RE.match(name)
        if m is None or m.group("args") is None:
            plain.append(name)
            continue
        args = m.group("args")
        prefix, _, last = args.rpartition(",")
        verbs.setdefault(m.group("verb"), OrderedDict()).setdefault(prefix, []).append(last)
    lines = [" ".join(plain)] if plain else []
    for verb, groups in verbs.items():
        parts = [(f"{prefix}," if prefix else "") + "|".join(lasts) for prefix, lasts in groups.items()]
        lines.append(f"{verb}({'; '.join(parts)})")
    return "\n".join(lines)


def expand_actions(grouped: str) -> list[str]:
    """Inverse of :func:`group_actions` (same multiset of names)."""
    out: list[str] = []
    for line in grouped.splitlines():
        line = line.strip()
        if not line:
            continue
        if "(" not in line:
            out.extend(line.split())
            continue
        verb, _, rest = line.partition("(")
        for part in rest.rstrip(")").split("; "):
            prefix, sep, lasts = part.rpartition(",")
            for last in lasts.split("|"):
                out.append(f"{verb}({prefix}{sep}{last})")
    return out


# --------------------------------------------------------------------------------------------
# Partial progress (failures only)
# --------------------------------------------------------------------------------------------

_REQ_SLOTS_RE = re.compile(r"^\s*\*\s+(?P<name>.+?) — (?P<k>\d+)/(?P<n>\d+) of: (?P<opts>.*)$")
_REQ_COINS_RE = re.compile(r"^\s*\*\s+(?P<name>.+?) — pay (?P<cost>\d+) coins")
_OPT_RE = re.compile(r"(?P<done>✓)?(?P<item>\S+) x(?P<qty>\d+) \(have (?P<have>\d+)\)")
_MONEY_RE = re.compile(r"Money (\d+)\.")


def request_fractions(describe_text: str) -> dict[str, float]:
    """Per open request, a progress fraction in [0, 1]: delivered slots count 1, items held toward
    an open slot count up to 0.5 (best options first), coins held toward a payment count up to 0.5."""
    money_m = _MONEY_RE.search(describe_text)
    money = int(money_m.group(1)) if money_m else 0
    out: dict[str, float] = {}
    for line in describe_text.splitlines():
        m = _REQ_SLOTS_RE.match(line)
        if m:
            k, n = int(m.group("k")), int(m.group("n"))
            held = sorted(
                (
                    min(1.0, int(o.group("have")) / max(1, int(o.group("qty"))))
                    for o in _OPT_RE.finditer(m.group("opts"))
                    if not o.group("done")
                ),
                reverse=True,
            )
            out[m.group("name")] = (k + 0.5 * sum(held[: max(0, n - k)])) / max(1, n)
            continue
        m = _REQ_COINS_RE.match(line)
        if m:
            out[m.group("name")] = 0.5 * min(1.0, money / max(1, int(m.group("cost"))))
    return out


def partial_progress(start_describe: str, end_describe: str) -> float:
    """Best gain on any request still open at both ends, clipped to [0, 1]."""
    start, end = request_fractions(start_describe), request_fractions(end_describe)
    gains = [end[name] - start[name] for name in start.keys() & end.keys()]
    return max(0.0, min(1.0, max(gains, default=0.0)))


# --------------------------------------------------------------------------------------------
# Reward
# --------------------------------------------------------------------------------------------


def segment_reward(
    success: bool,
    agent_days: Optional[float],
    expert_days: Optional[float],
    horizon_days: int,
    progress: float,
) -> float:
    """Lexicographic segment reward in [0, 1).

    * success: ``0.5 + 0.5 * (E + c) / ((E + c) + (A + c))`` with A = the agent's days to complete the
      request, E = the expert's days from the same state (``horizon_days`` if the expert fails), c =
      0.25 days. Strictly decreasing in A, equal to 0.75 when A == E, and always > 0.5 because A <= H.
    * failure: ``0.45 * progress`` (partial credit), so every failure scores below every success.
    """
    if success:
        if agent_days is None:
            raise ValueError("a successful segment needs agent_days")
        e = (horizon_days if expert_days is None else expert_days) + SPEED_SMOOTHING_DAYS
        a = max(0.0, agent_days) + SPEED_SMOOTHING_DAYS
        return SUCCESS_FLOOR + (1.0 - SUCCESS_FLOOR) * e / (e + a)
    return FAILURE_CAP * max(0.0, min(1.0, progress))


# --------------------------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------------------------


def _clock(info: dict) -> str:
    minute = int(info["minute"])
    return f"Day {int(info['day']) + 1} {minute // 60 % 24:02d}:{minute % 60:02d}"


class FurrowSession:
    """One rollout: one ``furrow.Env`` plus segment bookkeeping. All public methods are
    serialized by ``self.lock`` so concurrent tool calls on one session cannot interleave."""

    def __init__(self, spec: TaskSpec):
        self.spec = spec
        self.lock = threading.Lock()
        self.env = start_env(spec)
        self.start_info = self.env.info()
        self.start_describe = self.env.describe()
        self.expert_days = expert_reference(self.env, spec.horizon_days)
        self.all_names: list[str] = list(self.env.action_names())
        self._name_set = frozenset(self.all_names)
        self.deadline_day = int(self.start_info["day"]) + spec.horizon_days
        self.tool_calls = 0
        self.invalid_actions = 0
        self.macros_executed = 0
        self.success_info: Optional[dict] = None
        self.end_reason: Optional[str] = None
        self.last_used = time.monotonic()

    # ---- state helpers (call with the lock held) ----

    @property
    def done(self) -> bool:
        return self.end_reason is not None

    def _update_end(self, info: dict) -> None:
        if self.end_reason is not None:
            return
        if info["requests_done"] > self.start_info["requests_done"]:
            self.success_info = dict(info)
            self.end_reason = "success"
        elif self.env.is_done():
            self.end_reason = "env_finished"
        elif info["day"] >= self.deadline_day:
            self.end_reason = "horizon"

    def _progress(self) -> dict[str, Any]:
        info = self.env.info()
        return {
            "now": _clock(info),
            "goal": "complete any one new restoration request",
            "success": self.success_info is not None,
            "done": self.done,
            "end_reason": self.end_reason,
            "days_used": round(elapsed_days(self.start_info, info), 3),
            "day_ends_left": max(0, self.deadline_day - int(info["day"])),
            "deadline": f"segment ends when Day {self.deadline_day + 1} begins",
            "requests_completed_this_segment": int(info["requests_done"] - self.start_info["requests_done"]),
            "tool_calls_used": self.tool_calls,
            "tool_calls_left": max(0, self.spec.max_tool_calls - self.tool_calls),
            "invalid_actions": self.invalid_actions,
        }

    def _compact_state(self) -> str:
        lines = self.env.describe().splitlines()
        keep = lines[:2] + [ln for ln in lines if ln.startswith(("Bag:", "The Old Mill:"))]
        return "\n".join(keep)

    def _begin_call(self) -> Optional[dict]:
        """Count a tool call; return an error payload if the budget is exhausted."""
        self.last_used = time.monotonic()
        self.tool_calls += 1
        if self.tool_calls > self.spec.max_tool_calls:
            self.end_reason = self.end_reason or "tool_budget"
            return {"error": "tool-call budget exhausted; the segment is over", **self._progress()}
        return None

    # ---- tools ----

    def observe(self) -> dict:
        with self.lock:
            err = self._begin_call()
            if err:
                return err
            return {"state": self.env.describe(), "progress": self._progress()}

    def status(self) -> dict:
        with self.lock:
            err = self._begin_call()
            if err:
                return err
            return {"state": self._compact_state(), "progress": self._progress()}

    def list_actions(self) -> dict:
        with self.lock:
            err = self._begin_call()
            if err:
                return err
            if self.done:
                return {"actions": "", "count": 0, "progress": self._progress()}
            valid = list(self.env.valid_actions())
            return {"actions": group_actions(valid), "count": len(valid), "progress": self._progress()}

    def act(self, actions: Any) -> dict:
        with self.lock:
            err = self._begin_call()
            if err:
                return err
            names = [actions] if isinstance(actions, str) else list(actions or [])
            names = [str(n).strip() for n in names]
            if not names:
                return {"error": "give at least one action name", **self._payload([], [], [])}
            if len(names) > self.spec.max_plan_length:
                return {
                    "error": f"a plan may hold at most {self.spec.max_plan_length} actions",
                    **self._payload([], [], names),
                }
            if self.done:
                return {"error": "the segment is over; no more actions run", **self._payload([], [], names)}
            results: list[dict] = []
            errors: list[dict] = []
            plan_day = int(self.env.info()["day"])
            for i, name in enumerate(names):
                if self.done:
                    break
                if name not in self._name_set:
                    self.invalid_actions += 1
                    errors.append(
                        {
                            "index": i,
                            "action": name,
                            "error": "unknown action name",
                            "did_you_mean": difflib.get_close_matches(name, self.all_names, n=3, cutoff=0.5),
                        }
                    )
                    break
                valid = self.env.valid_actions()
                if name not in valid:
                    self.invalid_actions += 1
                    errors.append(
                        {
                            "index": i,
                            "action": name,
                            "error": "not valid in the current state (see list_actions)",
                            "did_you_mean": difflib.get_close_matches(name, list(valid), n=3, cutoff=0.5),
                        }
                    )
                    break
                _, _, _, _, info = self.env.step(name)
                self.macros_executed += 1
                results.append({"action": name, "outcome": info["outcome"], "now": _clock(info)})
                self._update_end(info)
                if int(info["day"]) != plan_day:
                    break
            not_run = names[len(results) + len(errors) :]
            note = None
            if not_run and not errors:
                note = "the segment ended" if self.done else "the day ended, so the rest of the plan was not run"
            return self._payload(results, errors, not_run, note)

    def _payload(self, results: list, errors: list, not_run: list, note: Optional[str] = None) -> dict:
        out: dict[str, Any] = {"executed": results}
        if errors:
            out["errors"] = errors
        if not_run:
            out["not_run"] = not_run
        if note:
            out["note"] = note
        out["state"] = self._compact_state()
        out["progress"] = self._progress()
        return out

    # ---- verify ----

    def result(self) -> dict[str, Any]:
        """Reward plus metrics for the segment as it stands now (idempotent)."""
        with self.lock:
            info = self.env.info()
            success = self.success_info is not None
            agent_days = elapsed_days(self.start_info, self.success_info) if success else None
            progress = 1.0 if success else partial_progress(self.start_describe, self.env.describe())
            reward = segment_reward(success, agent_days, self.expert_days, self.spec.horizon_days, progress)
            return {
                "reward": reward,
                "success": success,
                "end_reason": self.end_reason or "unfinished",
                "days_to_success": agent_days,
                "expert_days": self.expert_days,
                "expert_success": self.expert_days is not None,
                "speed_ratio": (
                    (self.expert_days if self.expert_days is not None else self.spec.horizon_days) / max(agent_days, 1e-3)
                    if success
                    else None
                ),
                "progress": progress,
                "days_played": elapsed_days(self.start_info, info),
                "requests_gained": int(info["requests_done"] - self.start_info["requests_done"]),
                "slots_gained": int(info["slots_filled"] - self.start_info["slots_filled"]),
                "tool_calls": self.tool_calls,
                "invalid_actions": self.invalid_actions,
                "macros_executed": self.macros_executed,
                "seed": self.spec.seed,
                "start_day": self.spec.start_day,
                "horizon_days": self.spec.horizon_days,
            }


class SessionClosedError(KeyError):
    pass


class SessionStore:
    """Thread-safe map ``session id -> FurrowSession`` with idle-TTL and size-cap eviction.

    Closed ids are remembered (bounded) so a late seed cannot recreate a closed session, and
    verify results are cached (bounded) so a retried verify returns the same answer.
    """

    def __init__(self, ttl_seconds: float = 3600.0, max_sessions: int = 4096, max_remembered: int = 100_000):
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self.max_remembered = max_remembered
        self._lock = threading.Lock()
        self._sessions: "OrderedDict[str, FurrowSession]" = OrderedDict()
        self._closed: "OrderedDict[str, None]" = OrderedDict()
        self._results: "OrderedDict[str, dict]" = OrderedDict()

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _remember(self, table: OrderedDict, key: str, value: Any) -> None:
        table[key] = value
        table.move_to_end(key)
        while len(table) > self.max_remembered:
            table.popitem(last=False)

    def _evict_locked(self, reserve: int = 0) -> None:
        """Drop idle sessions, then the least recently used ones until ``reserve`` more fit."""
        now = time.monotonic()
        for sid in [s for s, sess in self._sessions.items() if now - sess.last_used > self.ttl_seconds]:
            del self._sessions[sid]
        while self._sessions and len(self._sessions) + reserve > self.max_sessions:
            self._sessions.popitem(last=False)

    def create(self, session_id: str, spec: TaskSpec) -> FurrowSession:
        """Idempotent: an existing session with the same spec is returned as is."""
        with self._lock:
            if session_id in self._closed:
                raise SessionClosedError(session_id)
            existing = self._sessions.get(session_id)
        if existing is not None:
            if existing.spec != spec:
                raise ValueError(f"session {session_id!r} is already bound to another task")
            return existing
        session = FurrowSession(spec)  # expensive: expert replay; done outside the store lock
        with self._lock:
            if session_id in self._closed:
                raise SessionClosedError(session_id)
            existing = self._sessions.get(session_id)
            if existing is not None:
                return existing
            self._evict_locked(reserve=1)
            self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Optional[FurrowSession]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                self._sessions.move_to_end(session_id)
            return session

    def finish(self, session_id: str) -> Optional[dict]:
        """Score and drop the session; a repeated call returns the cached result."""
        with self._lock:
            cached = self._results.get(session_id)
            session = self._sessions.get(session_id)
        if cached is not None:
            return cached
        if session is None:
            return None
        result = session.result()
        with self._lock:
            self._remember(self._results, session_id, result)
            self._sessions.pop(session_id, None)
        return result

    def close(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            self._remember(self._closed, session_id, None)
