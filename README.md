# Stardew Valley RL Agent — speedrunning the Community Center from trajectories

> **Goal.** Teach a model to *speedrun* a long-horizon farming-sim objective — finish every
> restoration request of a multi-wing community building (the "Community Center" objective) in
> as few in-game days as possible — learned **primarily from trajectories** (demonstrations),
> then improved with reinforcement learning, and evaluated both in a fast original simulator and
> in the real game through a mod.

This repository was rebuilt in November 2026 around that objective. The first iteration (an
HTTP/`curl` reflection bridge driven from Python) is kept under [`legacy/`](legacy/) for
reference; everything else is new. `master` is the main branch.

## Why two environments

Training a speedrunner needs millions of episodes; the real game runs in real time and cannot be
redistributed. So the project has two environments that share one agent interface:

| | **Furrow** — original Rust farming-sim | **RLBridge** — SMAPI mod + Rust client |
|---|---|---|
| What it is | A from-scratch, *original* farming/life-sim in the same genre (seasons, crops, foraging, fishing, mining, shops, crafting, a multi-wing restoration objective). No Stardew code, data, names or maps. | A C# SMAPI mod that turns the player's **own installed copy** of Stardew Valley into a lockstep RL environment, plus a Rust client. Uses only the public modding API at runtime; ships no game code or data. |
| Speed | 10⁴–10⁶ steps/s, vectorised, deterministic, headless | real-time bound (~10²–10³ ticks/s) |
| Objective | Restore the *Old Mill* (original content, fixed or *remixed* request lists) | Complete the Community Center bundles (read from the live game) |
| Used for | Large-scale demonstration generation, imitation + RL training, benchmarking, LLM agents | Recording human play, evaluating/finetuning in the real game |

The game is re-implemented as an **original** environment on purpose: decompiled game source is
not used anywhere in this repository, and the real-game bridge only observes and acts through
SMAPI on a licensed copy.

## Architecture

```
                 ┌──────────────────────────── Python: svrl ─────────────────────────────┐
                 │  data (mmap'd trajectories) → BC / time-conditioned transformer / DAgger │
                 │  / IQL / PPO-finetune (KL to the imitation prior) → evaluate → report    │
                 └───────────────▲──────────────────────────────────────▲─────────────────┘
      furrow-traj v2 datasets    │ furrow (PyO3, abi3 wheel)             │ Gymnasium / NeMo Gym
   (.npy shards, action logs)    │                                      │  (python/svrl/envs, python/nemo_gym_furrow)
┌────────────────────────────────┴───────────┐        ┌─────────────────┴──────────────────────┐
│ Rust workspace                              │        │ Real game                               │
│  furrow        sim core, Env, VecEnv        │        │  mod/RLBridge   C# SMAPI mod: lockstep  │
│  furrow-expert scripted speedrun planner    │        │                 stepping, JSON snapshots,│
│  furrow-gen    episode-parallel runner      │        │                 input + macro actions,   │
│  furrow-traj   dataset format (no deps)     │◀──────▶│                 human-play recorder      │
│  furrow-cli    play / bench / demos / replay│        │  sdv-bridge     Rust client, SdvEnv,     │
│  furrow-py     Python bindings              │        │                 recorder → furrow-traj   │
└─────────────────────────────────────────────┘        └─────────────────────────────────────────┘
```

**Learning pipeline** (`python -m svrl.pipeline`): collect expert demonstrations at several noise
levels (clean labels kept, DART-style) → behaviour cloning (entity-centric action head) →
time-conditioned sequence policy (conditions on *days-to-go*, Decision-Transformer style) →
DAgger with the Rust expert → offline RL (IQL) → PPO finetuning from the imitation prior with a
frozen normaliser, critic warm-up and KL/kickstarting → evaluation on held-out seeds (censored
completion day, IQM + bootstrap CIs, paired differences vs. the expert) → figures and paper
tables.

**Key design decisions** (all in [`docs/DESIGN.md`](docs/DESIGN.md), the contract every crate
is built against):
- **Macro actions** (`Plant(seed)`, `Fish(spot)`, `Buy(shop,item,q)`, `Deliver`, `Sleep`, …)
  executed by deterministic controllers — 15–40 decisions per in-game day instead of ~10⁵
  primitive inputs per episode. Primitive mode exists as a secondary track.
- **Reward = speed only.** A per-day cost with potential-based progress shaping, discounted *per
  in-game day* (never per decision), so the optimum is provably the fastest completion.
- **Item-centric observations** (what is still needed, how much is held, how far away every
  point of interest is) whose meaning is identical across fixed and remixed objectives.
- **Bit-identical determinism**: in-crate RNG, exogenous randomness keyed by `(seed, day,
  purpose)` so every policy faces the same world on a seed (paired evaluation), stable state
  hashing, golden trajectories, zero allocations on the hot path.
- **Trajectory format** `furrow-traj/2`: memory-mappable `.npy` shard directories with clean
  expert labels next to executed actions; for Furrow the action log is the source of truth and
  observations are a regenerable cache.
- A **state-feedback expert** that reads only the observation (so DAgger labels are learnable)
  and recovers from off-plan states.
- **LLM agents** via NVIDIA NeMo Gym (segment tasks, lexicographic speed reward) and standard
  **Gymnasium** wrappers.

## Repository layout

```
Cargo.toml, rust-toolchain.toml   Rust workspace (pinned stable)
crates/
  furrow/            simulation core + RL env; content tables in crates/furrow/data/*.toml
  furrow-expert/     scripted speedrun planner (expert policy, DAgger oracle)
  furrow-gen/        episode-parallel runner: generate / relabel / materialize / evaluate
  furrow-traj/       trajectory dataset format (writer, memmap reader)
  furrow-py/         PyO3 extension `furrow._furrow` + python shim (`import furrow`)
  furrow-cli/        `furrow` binary
  sdv-bridge/        Rust client for the RLBridge mod (+ mock server)
mod/RLBridge/        C# SMAPI mod for the real game
python/
  svrl/              training package (data, models, bc, dt, dagger, iql, ppo, evaluate, pipeline, envs)
  furrow_mock/       pure-Python stand-in with the native module's surface (tests, offline dev)
  nemo_gym_furrow/   NeMo Gym resources server + recipes
  configs/, tests/
papers/              Typst sources: the Furrow benchmark paper and the speedrun-learning paper
docs/                DESIGN.md (contract), CONTENT.md, TRAJECTORY_FORMAT.md, BRIDGE_PROTOCOL.md,
                     SMAPI_NOTES.md, NEMO_GYM_*.md, BASELINES.md, RESULTS.md, reviews/
legacy/              the previous HTTP/curl agent (superseded)
```

## Status

| Component | State |
|---|---|
| Design contract (`docs/DESIGN.md` v2.1, two independent critiques folded in) | done |
| Original content tables (`crates/furrow/data`, `docs/CONTENT.md`) | in progress |
| Foundations: RNG/hash/grid/pathfinding, `furrow-traj` v2, `svrl.data` | in progress |
| Simulation core, macro controllers, observations, `VecEnv` | next |
| Expert planner | next |
| Python bindings, Gymnasium wrappers, NeMo Gym server | in progress (mock backend first) |
| RLBridge SMAPI mod + `sdv-bridge` client | research → build |
| Training pipeline (BC → DT → DAgger → IQL → PPO), evaluation, results | after the sim |
| Papers (Typst) | scaffolding |

Results will be reported in `docs/RESULTS.md` only from the pipeline's `results.json`.

## Quickstart (development)

```bash
# Rust (toolchain pinned by rust-toolchain.toml)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
cargo build --workspace && cargo test --workspace

# Python
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python maturin pytest numpy torch gymnasium pyyaml tqdm
.venv/bin/maturin develop --release -m crates/furrow-py/Cargo.toml   # builds `furrow`
uv pip install --python .venv/bin/python -e python                     # installs `svrl`
.venv/bin/python -m pytest -q python/tests

# Papers
typst compile papers/furrow/main.typ
```

The real-game bridge requires your own copy of Stardew Valley 1.6 with SMAPI 4; see
`mod/RLBridge/README.md` once it lands.

## Acknowledgements

[SMAPI](https://smapi.io) (modding API), [Gymnasium](https://gymnasium.farama.org),
[NVIDIA NeMo Gym](https://github.com/NVIDIA-NeMo/Gym). Stardew Valley is © ConcernedApe; this
project is not affiliated with or endorsed by ConcernedApe, and redistributes none of the game.

## License

MIT — see [LICENSE](LICENSE).
