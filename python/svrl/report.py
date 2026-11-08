"""Report stage: ``results.json`` → figures, ``docs/RESULTS.md`` and Typst tables for the papers.

Numbers in the papers come only from here (never hand-typed).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import furrow

PRETTY = {
    "random": "Masked random",
    "expert": "Scripted expert",
    "bc_expert_25": "BC (25 ep)",
    "bc_expert_100": "BC (100 ep)",
    "bc_expert_400": "BC (400 ep)",
    "bc_expert_1000": "BC (1000 ep)",
    "bc_dart": "BC + DART",
    "bc_mixed": "BC (mixed styles)",
    "bc_filtered25": "Filtered BC (fastest 25%)",
    "dagger": "DAgger",
    "ppo_from_bc": "PPO finetune (from BC+DART)",
}


def pretty(k: str) -> str:
    if k.startswith("bc_cond@"):
        return f"Time-cond. BC (target {k.split('@')[1]} d)"
    return PRETTY.get(k, k)


def throughput(seconds: float = 3.0) -> list[dict]:
    """Measure VecEnv macro-step throughput from Python (random valid actions and expert labels)."""
    rows = []
    for n in (16, 64, 256):
        for mode in ("random", "expert"):
            vec = furrow.VecEnv(n, seed=0)
            obs, _ = vec.reset()
            rng = np.random.default_rng(0)
            steps, t0 = 0, time.time()
            while time.time() - t0 < seconds:
                if mode == "random":
                    m = obs["action_mask"]
                    a = (rng.random(m.shape) * m).argmax(-1)
                else:
                    a = vec.expert_actions()
                obs, *_ = vec.step(a)
                steps += n
            rows.append({"num_envs": n, "policy": mode, "steps_per_s": steps / (time.time() - t0)})
    return rows


def _fmt_ci(v, ci, digits=1):
    return f"{v:.{digits}f} [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def write_report(cfg: dict, repo: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rd = Path(cfg["run_dir"])
    data = json.loads((rd / "results.json").read_text())
    res, meta = data["results"], data["meta"]
    figdir = repo / "docs" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    gen = {p: repo / "papers" / p / "generated" for p in ("furrow", "speedrun")}
    for g in gen.values():
        g.mkdir(parents=True, exist_ok=True)
    order = [k for k in ["random", "expert", "bc_expert_25", "bc_expert_100", "bc_expert_400", "bc_expert_1000", "bc_dart",
                         "bc_mixed", "bc_filtered25"] if k in res]
    order += sorted([k for k in res if k.startswith("bc_cond@")])
    order += [k for k in ["dagger", "ppo_from_bc"] if k in res]
    max_days = meta.get("max_days", 192)

    # ---------------- main table (markdown + typst)
    md = ["| Method | Success rate | Mean completion day C (95% CI) | IQM C | Median C | Speed score | Δ vs expert (paired, days) |",
          "|---|---|---|---|---|---|---|"]
    ty = ["#table(columns: 7, align: (left, right, right, right, right, right, right), stroke: none,",
          "  table.hline(), [*Method*], [*Success*], [*Mean C (95% CI)*], [*IQM C*], [*Median C*], [*Speed*], [*Δ vs expert*], table.hline(),"]
    for k in order:
        r = res[k]
        sp = f"{r.get('speed_score', float('nan')):.2f}"
        dd = _fmt_ci(r["paired_diff_vs_expert"], r["paired_diff_ci"]) if "paired_diff_vs_expert" in r else "–"
        md.append(f"| {pretty(k)} | {r['success_rate']:.3f} | {_fmt_ci(r['mean_C'], r['mean_C_ci'])} | {r['iqm_C']:.1f} | {r['median_C']:.0f} | {sp} | {dd} |")
        ty.append(f"  [{pretty(k)}], [{r['success_rate']:.3f}], [{_fmt_ci(r['mean_C'], r['mean_C_ci'])}], [{r['iqm_C']:.1f}], [{r['median_C']:.0f}], [{sp}], [{dd}],")
    ty.append("  table.hline())")
    (gen["speedrun"] / "main_table.typ").write_text("\n".join(ty) + "\n")

    # ---------------- CDF of completion day
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for k in [x for x in ["expert", "bc_expert_1000", "bc_dart", "dagger", "ppo_from_bc", "bc_expert_100", "random"] if x in res]:
        C = np.sort(np.asarray(res[k]["per_seed_C"]))
        succ = C < max_days
        y = np.arange(1, len(C) + 1) / len(C)
        ax.step(np.where(succ, C, np.nan), y, where="post", label=pretty(k))
    ax.set_xlabel("in-game day of completion")
    ax.set_ylabel("fraction of test seeds completed")
    ax.set_xlim(40, max_days)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    for p in (figdir / "completion_cdf.png", gen["speedrun"] / "completion_cdf.png"):
        fig.savefig(p, dpi=160)
    plt.close(fig)

    # ---------------- data scaling
    sc = [(int(k.split("_")[-1]), res[k]) for k in res if k.startswith("bc_expert_")]
    sc.sort()
    if sc:
        fig, ax1 = plt.subplots(figsize=(5.5, 3.6))
        xs = [s[0] for s in sc]
        ax1.plot(xs, [s[1]["success_rate"] for s in sc], "o-", color="C0", label="success rate")
        ax1.set_xscale("log")
        ax1.set_xlabel("expert episodes in the BC dataset")
        ax1.set_ylabel("success rate", color="C0")
        ax1.set_ylim(0, 1.02)
        ax2 = ax1.twinx()
        ax2.plot(xs, [s[1]["mean_C"] for s in sc], "s--", color="C1", label="mean C")
        ax2.axhline(res["expert"]["mean_C"], color="C1", lw=0.8, ls=":")
        ax2.set_ylabel("mean censored completion day", color="C1")
        ax1.grid(alpha=0.3)
        fig.tight_layout()
        for p in (figdir / "bc_scaling.png", gen["speedrun"] / "bc_scaling.png"):
            fig.savefig(p, dpi=160)
        plt.close(fig)

    # ---------------- DAgger rounds / PPO curve
    dh = rd / "dagger" / "history.json"
    dag = json.loads(dh.read_text()) if dh.exists() else []
    ppo_log = rd / "ppo_from_bc.log.jsonl"
    ppo = [json.loads(l) for l in ppo_log.read_text().splitlines()] if ppo_log.exists() else []
    ppo_val = [r for r in ppo if "val_success" in r]
    if dag or ppo_val:
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
        if dag:
            axs[0].plot([h["round"] for h in dag], [h["mean_C"] for h in dag], "o-")
            axs[0].set_xlabel("DAgger round")
            axs[0].set_ylabel("validation mean C")
            axs[0].grid(alpha=0.3)
        if ppo_val:
            axs[1].plot([h["iter"] for h in ppo_val], [h["val_mean_C"] for h in ppo_val], "o-", color="C2")
            axs[1].set_xlabel("PPO iteration")
            axs[1].set_ylabel("validation mean C")
            axs[1].grid(alpha=0.3)
        fig.tight_layout()
        for p in (figdir / "dagger_ppo.png", gen["speedrun"] / "dagger_ppo.png"):
            fig.savefig(p, dpi=160)
        plt.close(fig)

    # ---------------- throughput (furrow paper)
    tp = throughput()
    (rd / "throughput.json").write_text(json.dumps(tp, indent=1))
    tty = ["#table(columns: 3, align: (right, left, right), stroke: none, table.hline(),",
           "  [*Envs*], [*Policy*], [*Macro steps / s*], table.hline(),"]
    for r in tp:
        tty.append(f"  [{r['num_envs']}], [{r['policy']}], [{r['steps_per_s']:,.0f}],")
    tty.append("  table.hline())")
    (gen["furrow"] / "throughput_table.typ").write_text("\n".join(tty) + "\n")
    # key numbers for inline use in the papers
    keys = {
        "expert_mean_C": res["expert"]["mean_C"], "expert_success": res["expert"]["success_rate"],
        "random_success": res["random"]["success_rate"], "n_test": res["expert"]["n"],
        "best_learned": min((k for k in order if k not in ("expert", "random")), key=lambda k: (-res[k]["success_rate"], res[k]["mean_C"]), default=None),
        "throughput_256_random": next((r["steps_per_s"] for r in tp if r["num_envs"] == 256 and r["policy"] == "random"), None),
        "throughput_256_expert": next((r["steps_per_s"] for r in tp if r["num_envs"] == 256 and r["policy"] == "expert"), None),
    }
    if keys["best_learned"]:
        b = res[keys["best_learned"]]
        keys.update(best_name=pretty(keys["best_learned"]), best_success=b["success_rate"], best_mean_C=b["mean_C"], best_speed=b.get("speed_score"))
    (rd / "keys.json").write_text(json.dumps(keys, indent=1))
    kv = ["// generated by svrl.report — do not edit", "#let results = ("]
    for k, v in keys.items():
        if isinstance(v, str):
            kv.append(f'  {k}: "{v}",')
        elif v is not None:
            kv.append(f"  {k}: {v:.4g},")
    kv.append(")")
    for g in gen.values():
        (g / "keys.typ").write_text("\n".join(kv) + "\n")

    # ---------------- RESULTS.md
    lines = [
        "# Results",
        "",
        f"Generated by `python -m svrl.pipeline --stages report` from `{rd}/results.json` "
        f"(git `{meta['git_commit']}`, SIM_VERSION {meta['sim_version']}, content `{meta['content_hash']}`).",
        "",
        f"Evaluation: {res['expert']['n']} held-out **test seeds** ({meta['test_seeds'][0]}…{meta['test_seeds'][1]}), one episode each, "
        f"same seeds for every method (paired: identical weather, merchant stock and forage). "
        f"**C** = in-game day on which all 24 restoration requests are complete, censored at {max_days} days for failures. "
        "Speed score S = (max_days − mean C) / (max_days − mean C_expert); Δ = per-seed C − C_expert. "
        "CIs: 95% bootstrap over seeds. Greedy decoding for learned policies.",
        "",
        *md,
        "",
        "![completion CDF](figures/completion_cdf.png)",
        "",
        "![BC data scaling](figures/bc_scaling.png)",
        "",
    ]
    if dag:
        lines += ["## DAgger", "", "| Round | Samples | Val. success | Val. mean C |", "|---|---|---|---|"]
        lines += [f"| {h['round']} | {h['samples']:,} | {h['success_rate']:.3f} | {h['mean_C']:.1f} |" for h in dag]
        lines += ["", "![DAgger and PPO](figures/dagger_ppo.png)", ""]
    if ppo_val:
        lines += ["## PPO finetuning (validation seeds)", "", "| Iter | Val. success | Val. mean C |", "|---|---|---|"]
        lines += [f"| {h['iter']} | {h['val_success']:.3f} | {h['val_mean_C']:.1f} |" for h in ppo_val]
        lines += [""]
    lines += ["## Simulator throughput (Python → Rust VecEnv, macro actions, this machine)", "", "| Envs | Policy | Macro steps/s |", "|---|---|---|"]
    lines += [f"| {r['num_envs']} | {r['policy']} | {r['steps_per_s']:,.0f} |" for r in tp]
    lines += ["", f"Host: `{meta['host']}`, Python {meta['python']}."]
    (repo / "docs" / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("[report] wrote docs/RESULTS.md, figures, papers/*/generated")
