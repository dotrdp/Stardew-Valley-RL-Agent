# Research notes compiled from the agent runs (2026-11-08)

This folder collects what the design/research agents produced while the project was being
planned, so the reasoning behind `docs/DESIGN.md` is auditable. Several larger agent fan-outs
were cut short by account rate limits; only outputs that actually completed are kept here.

| File | What it is | Status | How it was used |
|---|---|---|---|
| [`reviews/2026-11-08-rl-critique.md`](reviews/2026-11-08-rl-critique.md) | Independent RL-research critique of the v1 contract: 19 amendments (11 "must") | complete | Folded into DESIGN v2: per-day discounting with potential-based shaping, expert information contract, item-centric observations, macro semantics (time boxes, ≥5-tick cost, stable ids, `Buy(item,q)`), trajectory columns (clean expert labels, terminated/truncated, day_ends), BC→PPO finetuning protocol, evaluation protocol (paired held-out seeds, censored completion day, IQM + bootstrap CIs), canonical datasets and baseline list, segment tasks for LLM agents |
| [`reviews/2026-11-08-systems-critique.md`](reviews/2026-11-08-systems-critique.md) | Independent Rust-systems critique with measured prototypes (pyo3 0.30 zero-copy + `py.detach`, rayon load-imbalance benchmark, npz vs npy mmap timings) | complete | Folded into DESIGN v2.1: `.npy` shard directories instead of `.npz` (mmap), GIL-released output filling, explicit seed schedule for auto-reset, every action advances the clock + `max_steps`, explicit content ids, `SIM_VERSION` + stable state hash, per-VecEnv rayon pool with `with_max_len(1)`, episode-parallel demo generation with ordered commit, `_furrow` module name |
| [`baselines/baselines-unverified.md`](baselines/baselines-unverified.md) | Raw findings of five research angles (human Community-Center speedruns, prior AI work on Stardew Valley, learning-from-trajectories methods, evaluation methodology, simulator throughput) | research done; verification partial | Background for the papers' related work; **unverified claims are not cited** |
| [`baselines/baselines-verification-partial.md`](baselines/baselines-verification-partial.md) | Adversarial fact-check verdicts for the findings above (41 confirmed of the batches that finished) | partial | Only confirmed items are cited in `papers/` (e.g. StarDojo, EnvPool, Sample Factory, PufferLib, Craftax, NLE, Crafter, VPT, NetHack datasets, rliable, BALROG) |

## Key takeaways that shaped the build

1. **Reward/discount alignment** (RL critique #1): with a per-decision discount and raw progress
   rewards, ranking by return is not ranking by completion day. Furrow therefore uses −1 per day,
   potential-based progress shaping and γ applied per *in-game day*.
2. **Learnable expert labels** (RL critique #2): the expert must act on information the learner
   observes, and must be stateless so DAgger can query it anywhere. Implemented as a stateless
   priority-rule planner; strict observation-only decoding is future work.
3. **Memory-mappable data** (systems critique #1): numpy cannot memory-map `.npz`; datasets are
   directories of aligned `.npy` files.
4. **Throughput** (systems critique #9, throughput research): stragglers from variable macro
   cost dominate lockstep efficiency; demo generation runs whole episodes per task instead
   (measured ≈88k recorded macro steps/s on 16 threads).
5. **Prior work** (baseline research): the only dedicated Stardew Valley agent benchmark found is
   StarDojo (LLM agents, short tasks, real-time); long-horizon imitation in NetHack/Minecraft
   recovers only part of the demonstrator's competence from BC alone, motivating DAgger and RL
   finetuning.

Unfinished (rate-limited) work that was **not** re-run at the user's request: a 50-agent
architecture-proposal fan-out, SMAPI API research, and citation verification for every angle.
