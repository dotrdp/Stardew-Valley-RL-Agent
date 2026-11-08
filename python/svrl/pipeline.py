"""Resumable experiment pipeline (DESIGN.md §8).

Stages (each skipped if its output exists, unless ``--force``):
``collect`` → ``bc`` (incl. data-scaling, DART, filtered and time-conditioned BC) → ``dagger`` →
``ppo`` → ``evaluate`` (all policies + random + expert on paired held-out test seeds) → ``report``
(``results.json``, figures, ``docs/RESULTS.md``, Typst tables for the papers).

Usage::

    python -m svrl.pipeline --config python/configs/pipeline.yaml [--stages collect,bc] [--force]
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import yaml

import furrow

from . import evaluate as ev
from .bc import train_bc
from .utils import JsonlLogger, git_commit

STAGES = ["collect", "bc", "dagger", "ppo", "evaluate", "report"]
REPO = Path(__file__).resolve().parents[2]


def _done(p: Path, force: bool) -> bool:
    return p.exists() and not force


def stage_collect(cfg: dict, force: bool, log: JsonlLogger) -> None:
    for name, d in cfg["datasets"].items():
        out = Path(cfg["data_dir"]) / f"{cfg['name']}-{name}"
        if _done(out / "episodes.jsonl", force):
            continue
        if out.exists():
            import shutil

            shutil.rmtree(out)
        t = time.time()
        s = furrow.generate_demos(str(out), episodes=d["episodes"], seed=d.get("seed", 0), seed_range=(0, 2**32),
                                  noise=d.get("noise", 0.0), styles=d.get("styles"))
        log.log(stage="collect", dataset=name, seconds=round(time.time() - t, 2), **s)
        print(f"[collect] {name}: {s} in {time.time() - t:.1f}s")


def stage_bc(cfg: dict, force: bool, log: JsonlLogger) -> None:
    rd, dd, n = Path(cfg["run_dir"]), Path(cfg["data_dir"]), cfg["name"]
    b = cfg["bc"]
    exp, dart, sty = str(dd / f"{n}-expert"), str(dd / f"{n}-dart"), str(dd / f"{n}-styles")
    jobs = []
    for k in b["scaling"]:
        jobs.append((f"bc_expert_{k}", dict(datasets=[exp], max_episodes=k)))
    jobs += [
        ("bc_dart", dict(datasets=[exp, dart])),
        ("bc_mixed", dict(datasets=[exp, dart, sty])),
        ("bc_filtered25", dict(datasets=[exp, sty], fastest_frac=0.25)),
        ("bc_cond", dict(datasets=[exp, dart, sty], cond=True, success_only=True)),
    ]
    for name, kw in jobs:
        out = rd / f"{name}.pt"
        if _done(out, force):
            continue
        t = time.time()
        m = train_bc(out=str(out), steps=b["steps"], batch=b["batch"], max_vram_gb=cfg["max_vram_gb"], seed=cfg["seed"], **kw)
        log.log(stage="bc", model=name, seconds=round(time.time() - t, 1), **m)
        print(f"[bc] {name}: {m} ({time.time() - t:.0f}s)")


def stage_dagger(cfg: dict, force: bool, log: JsonlLogger) -> None:
    from .dagger import run_dagger

    rd = Path(cfg["run_dir"])
    if _done(rd / "dagger" / "history.json", force):
        return
    d = cfg["dagger"]
    t = time.time()
    hist = run_dagger([str(Path(cfg["data_dir"]) / f"{cfg['name']}-expert")], str(rd / "dagger"), rounds=d["rounds"], num_envs=d["num_envs"],
                      steps_per_round=d["steps_per_round"], bc_steps=d["bc_steps"], init_max_episodes=d["init_episodes"],
                      max_vram_gb=cfg["max_vram_gb"], seed=cfg["seed"])
    (rd / "dagger" / "history.json").write_text(json.dumps(hist, indent=1))
    log.log(stage="dagger", seconds=round(time.time() - t, 1), history=hist)
    print(f"[dagger] {hist}")


def stage_ppo(cfg: dict, force: bool, log: JsonlLogger) -> None:
    from .ppo import finetune

    rd = Path(cfg["run_dir"])
    out = rd / "ppo_from_bc.pt"
    if _done(out, force):
        return
    p = cfg["ppo"]
    t = time.time()
    hist = finetune(str(rd / "bc_dart.pt"), str(out), iters=p["iters"], num_envs=p["num_envs"], horizon=p["horizon"],
                    warmup_iters=p["warmup_iters"], lr=p["lr"], kl_coef=p["kl_coef"], kl_final=p["kl_final"],
                    max_vram_gb=cfg["max_vram_gb"], seed=cfg["seed"])
    log.log(stage="ppo", seconds=round(time.time() - t, 1), history=hist)
    print(f"[ppo] {hist}")


def _policies(cfg: dict):
    import torch

    from .models import Policy
    from .utils import device

    rd = Path(cfg["run_dir"])
    dev = device(cfg["max_vram_gb"])
    out = [("random", ev.masked_random_policy(0)), ("expert", ev.expert_policy(0))]
    names = [f"bc_expert_{k}" for k in cfg["bc"]["scaling"]] + ["bc_dart", "bc_mixed", "bc_filtered25"]
    for nme in names:
        p = rd / f"{nme}.pt"
        if p.exists():
            out.append((nme, ev.torch_policy(Policy.load(str(p), dev).eval(), dev)))
    if (rd / "bc_cond.pt").exists():
        pol = Policy.load(str(rd / "bc_cond.pt"), dev).eval()
        for tgt in cfg["eval"]["conditioned_targets"]:
            out.append((f"bc_cond@{tgt}", ev.torch_policy(pol, dev, target_days=float(tgt))))
    dh = rd / "dagger" / "history.json"
    if dh.exists():
        hist = json.loads(dh.read_text())
        best = max(hist, key=lambda h: (h["success_rate"], -h["mean_C"]))
        out.append(("dagger", ev.torch_policy(Policy.load(best["ckpt"], dev).eval(), dev)))
    if (rd / "ppo_from_bc.pt").exists():
        out.append(("ppo_from_bc", ev.torch_policy(Policy.load(str(rd / "ppo_from_bc.pt"), dev).eval(), dev)))
    del torch
    return out


def stage_evaluate(cfg: dict, force: bool, log: JsonlLogger) -> None:
    rd = Path(cfg["run_dir"])
    out = rd / "results.json"
    if _done(out, force):
        return
    n = cfg["eval"]["test_seeds"]
    seeds = list(range(ev.TEST_SEEDS[0], ev.TEST_SEEDS[0] + n))
    raw = {}
    for name, pol in _policies(cfg):
        t = time.time()
        raw[name] = ev.run_policy(pol, seeds)
        print(f"[eval] {name}: success {raw[name]['success'].mean():.3f} mean C {raw[name]['C'].mean():.1f} ({time.time() - t:.0f}s)")
    res = {k: ev.summarize(v, expert=raw["expert"]) for k, v in raw.items()}
    for k, v in raw.items():
        res[k]["per_seed_C"] = v["C"].tolist()
    meta = {"git_commit": git_commit(), "sim_version": furrow.SIM_VERSION, "content_hash": f"{furrow.content_hash():016x}",
            "test_seeds": [seeds[0], seeds[-1]], "max_days": 192, "host": platform.node(), "python": platform.python_version()}
    out.write_text(json.dumps({"meta": meta, "results": res}, indent=1))
    log.log(stage="evaluate", results={k: {"success": v["success_rate"], "mean_C": v["mean_C"]} for k, v in res.items()})


def stage_report(cfg: dict, force: bool, log: JsonlLogger) -> None:
    from .report import write_report

    write_report(cfg, REPO)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "python/configs/pipeline.yaml"))
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text())
    Path(cfg["run_dir"]).mkdir(parents=True, exist_ok=True)
    log = JsonlLogger(Path(cfg["run_dir"]) / "pipeline.log.jsonl")
    log.log(stage="start", config=cfg, git_commit=git_commit(), sim_version=furrow.SIM_VERSION)
    for s in a.stages.split(","):
        s = s.strip()
        t = time.time()
        globals()[f"stage_{s}"](cfg, a.force, log)
        print(f"== stage {s} done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
