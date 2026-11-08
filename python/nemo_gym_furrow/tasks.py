"""Generate Furrow segment-task datasets (NeMo Gym JSONL) for train / val / test.

Seed ranges are disjoint per split: train ``[0, 2^32)``, val ``[2^32, 2^33)``, test ``[2^33, 2^34)``
(``example`` rows are drawn from the train range). For each seed the scripted expert plays the full
run once; start days are sampled from that trajectory among the days from which the expert completes
its next request within ``--horizon-days`` (so every task is feasible), and the row stores the
expert's days to that request (``expert_days``) for reference. The resources server recomputes the
reference from the same state at seed time; both use the same deterministic expert.

    python -m nemo_gym_furrow.tasks --out data/nemo_gym_furrow --train 2000 --val 200 --test 200
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import furrow

from nemo_gym_furrow.core import DEFAULT_HORIZON_DAYS, DEFAULT_MAX_TOOL_CALLS, MINUTES_PER_DAY
from nemo_gym_furrow.prompt import build_row

SEED_RANGES: dict[str, tuple[int, int]] = {
    "train": (0, 1 << 32),
    "val": (1 << 32, 1 << 33),
    "test": (1 << 33, 1 << 34),
    "example": (0, 1 << 32),
}


@dataclass
class StartCandidate:
    day: int
    abs_minute: int
    requests_done: int
    describe: str
    expert_days: Optional[float] = None


def expert_start_candidates(seed: int, horizon_days: int, env_config: Optional[dict] = None) -> list[StartCandidate]:
    """Play the expert on ``seed`` once; return each day's start state (the first state with
    ``info["day"] == d``) with the expert's days to its next request, if within ``horizon_days``."""
    env = furrow.Env(seed=seed, **(env_config or {}))
    _, info = env.reset(seed=seed)
    candidates = [StartCandidate(info["day"], info["abs_minute"], info["requests_done"], env.describe())]
    completions: list[tuple[int, int, int]] = []  # (abs_minute, day, requests_done) at each completion
    last_requests, last_day = info["requests_done"], info["day"]
    while not env.is_done():
        _, _, _, _, info = env.step(env.expert_action())
        if info["requests_done"] > last_requests:
            completions.append((info["abs_minute"], info["day"], info["requests_done"]))
            last_requests = info["requests_done"]
        if info["day"] > last_day and not env.is_done():
            candidates.append(StartCandidate(info["day"], info["abs_minute"], info["requests_done"], env.describe()))
        last_day = info["day"]
    for cand in candidates:
        nxt = next((c for c in completions if c[2] > cand.requests_done), None)
        if nxt is not None and nxt[1] < cand.day + horizon_days:
            cand.expert_days = (nxt[0] - cand.abs_minute) / MINUTES_PER_DAY
    return candidates


def _seed_rows(job: dict[str, Any]) -> list[dict[str, Any]]:
    seed = job["seed"]
    rng = random.Random(f"{job['base_seed']}:{job['split']}:{seed}")
    feasible = [
        c
        for c in expert_start_candidates(seed, job["horizon_days"], job["env_config"])
        if c.expert_days is not None
        and c.expert_days >= job["min_expert_days"]
        and job["min_start_day"] <= c.day <= job["max_start_day"]
    ]
    picked = sorted(rng.sample(feasible, min(job["days_per_seed"], len(feasible))), key=lambda c: c.day)
    return [
        build_row(
            task_id=f"furrow-{job['split']}-{seed}-d{c.day}",
            split=job["split"],
            seed=seed,
            start_day=c.day,
            horizon_days=job["horizon_days"],
            max_tool_calls=job["max_tool_calls"],
            start_describe=c.describe,
            expert_days=round(c.expert_days, 6),
            agent_name=job["agent_name"],
            env_config=job["env_config"] or None,
            temperature=job["temperature"],
        )
        for c in picked
    ]


def split_seeds(split: str, n: int, base_seed: int) -> list[int]:
    lo, hi = SEED_RANGES[split]
    rng = random.Random(f"furrow-nemo-tasks:{base_seed}:{split}")
    seeds: list[int] = []
    seen: set[int] = set()
    while len(seeds) < n:
        s = rng.randrange(lo, hi)
        if s not in seen:
            seen.add(s)
            seeds.append(s)
    return seeds


def generate(
    out_dir: Path,
    counts: dict[str, int],
    *,
    days_per_seed: int = 2,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    base_seed: int = 0,
    agent_name: Optional[str] = "furrow_simple_agent",
    env_config: Optional[dict] = None,
    min_start_day: int = 0,
    max_start_day: int = 10_000,
    min_expert_days: float = 0.0,
    temperature: Optional[float] = None,
    workers: int = 1,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "furrow_version": furrow.__version__,
        "sim_version": furrow.SIM_VERSION,
        "content_hash": furrow.content_hash(),
        "seed_ranges": {k: list(v) for k, v in SEED_RANGES.items() if k in counts},
        "days_per_seed": days_per_seed,
        "horizon_days": horizon_days,
        "max_tool_calls": max_tool_calls,
        "base_seed": base_seed,
        "env_config": env_config or {},
        "min_expert_days": min_expert_days,
        "files": {},
    }
    for split, n_seeds in counts.items():
        jobs = [
            dict(
                seed=s,
                split=split,
                base_seed=base_seed,
                horizon_days=horizon_days,
                max_tool_calls=max_tool_calls,
                days_per_seed=days_per_seed,
                agent_name=agent_name,
                env_config=env_config or {},
                min_start_day=min_start_day,
                max_start_day=max_start_day,
                min_expert_days=min_expert_days,
                temperature=temperature,
            )
            for s in split_seeds(split, n_seeds, base_seed)
        ]
        if workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                per_seed = list(pool.map(_seed_rows, jobs, chunksize=4))
        else:
            per_seed = [_seed_rows(j) for j in jobs]
        rows = [r for rs in per_seed for r in rs]
        path = out_dir / f"{split}.jsonl"
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        manifest["files"][split] = {"path": path.name, "seeds": n_seeds, "rows": len(rows)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, type=Path, help="output directory for <split>.jsonl and manifest.json")
    p.add_argument("--train", type=int, default=0, help="number of train seeds")
    p.add_argument("--val", type=int, default=0, help="number of validation seeds")
    p.add_argument("--test", type=int, default=0, help="number of test seeds")
    p.add_argument("--example", type=int, default=0, help="number of example seeds (smoke-test rows)")
    p.add_argument("--days-per-seed", type=int, default=2, help="start days sampled per seed")
    p.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS)
    p.add_argument("--max-tool-calls", type=int, default=DEFAULT_MAX_TOOL_CALLS)
    p.add_argument("--min-start-day", type=int, default=0)
    p.add_argument("--max-start-day", type=int, default=10_000)
    p.add_argument(
        "--min-expert-days", type=float, default=0.0, help="drop starts the expert finishes faster (e.g. 0.25)"
    )
    p.add_argument("--seed", type=int, default=0, help="base seed for seed and start-day sampling")
    p.add_argument("--agent-name", default="furrow_simple_agent", help="agent_ref.name per row ('' to omit)")
    p.add_argument("--env-config", default="{}", help='JSON furrow.Env kwargs, e.g. \'{"restoration": "remixed"}\'')
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--workers", type=int, default=1, help="processes (one expert run per seed)")
    a = p.parse_args(argv)
    counts = {k: v for k, v in (("train", a.train), ("val", a.val), ("test", a.test), ("example", a.example)) if v}
    if not counts:
        p.error("ask for at least one of --train/--val/--test/--example")
    manifest = generate(
        a.out,
        counts,
        days_per_seed=a.days_per_seed,
        horizon_days=a.horizon_days,
        max_tool_calls=a.max_tool_calls,
        base_seed=a.seed,
        agent_name=a.agent_name or None,
        env_config=json.loads(a.env_config),
        min_start_day=a.min_start_day,
        max_start_day=a.max_start_day,
        min_expert_days=a.min_expert_days,
        temperature=a.temperature,
        workers=a.workers,
    )
    print(json.dumps(manifest["files"], indent=2))


if __name__ == "__main__":
    main()
