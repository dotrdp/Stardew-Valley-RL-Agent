# Stardew Valley RL Agent — speedrunning the Community Center from trajectories

> **Goal.** Teach a model to *speedrun* a long-horizon farming-sim objective: complete every
> request of a multi-wing community building (the "Community Center" objective) in as few in-game
> days as possible. The model learns **primarily from trajectories** (demonstrations), is then
> improved with reinforcement learning, and is evaluated in a fast original simulator and,
> through a mod, in the real game.

This repository was rebuilt in November 2026 around that objective. The first iteration, an
HTTP/`curl` reflection bridge driven from Python, is kept under [`legacy/`](legacy/) for
reference. `master` is the main branch.

**Headline results** (200 held-out test seeds, see [`docs/RESULTS.md`](docs/RESULTS.md)):

| Policy | Success | Mean completion day | Δ vs expert (paired, 95% CI) |
|---|---|---|---|
| Masked random | 0.5% | 192.0 | +114.1 [+110.4, +117.6] |
| Scripted expert (demonstrator) | 98.5% | 77.9 | +0.0 [+0.0, +0.0] |
| BC, 100 expert episodes | 68.0% | 135.8 | +57.9 [+50.3, +65.2] |
| BC, 1000 expert episodes | 83.0% | 103.8 | +25.9 [+19.0, +32.8] |
| BC + DART (noise-injected demos, clean labels) | 93.0% | 86.9 | +9.0 [+3.7, +14.5] |
| DAgger (from 100 episodes, 6 rounds) | 91.5% | 97.8 | +19.9 [+13.7, +26.1] |
| **PPO finetune from BC + DART** | 99.0% | 76.4 | -1.5 [-5.4, +2.2] |

Behaviour cloning alone saturates below the demonstrator. Recovery data (DART, DAgger) closes most of
the gap, and PPO finetuning with a KL anchor to the imitation prior reaches the expert's level
(the difference is not significant) in under a minute of training.

## Why two environments

Training a speedrunner needs millions of episodes, but the real game runs in real time and cannot
be redistributed. So the project has two environments that share one agent interface:

| | **Furrow**: original Rust farming sim | **RLBridge**: SMAPI mod + Rust client |
|---|---|---|
| What it is | An *original* farming/life sim in the same genre: seasons, 13 crops, foraging, 14 fish, a 60-floor mine, shops, a wandering merchant, crafting and machines, skills, tool upgrades, and a 6-wing / 24-request restoration objective. It contains no Stardew Valley code, data, names or maps. | A C# SMAPI mod that turns your **own installed copy** of Stardew Valley into a lockstep RL environment, with Community Center bundle state in every observation. It uses only the public modding API at runtime and ships no game code or data. |
| Speed | ~50–65k macro steps/s from Python on one 8-core desktop (each macro is about a hundred game minutes of primitive steps) | real-time bound |
| Objective | Restore the *Old Mill*: fixed requests, or *remixed* requests with held-out item pools | Complete the Community Center |
| Used for | Demonstrations, imitation and RL training, benchmarking, LLM agents | Recording human play, evaluation and finetuning in the real game |

The game is re-implemented as an **original** environment on purpose: no decompiled game source is
used anywhere in this repository, and the real-game bridge only observes and acts through SMAPI on a
licensed copy.

## Quick start

```bash
# Rust (toolchain pinned by rust-toolchain.toml)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
cargo test --workspace

# Python env + native module + training package
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python maturin pytest numpy torch gymnasium pyyaml tqdm matplotlib
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release -m crates/furrow-py/Cargo.toml   # `import furrow`
uv pip install --python .venv/bin/python -e python                                          # `import svrl`
.venv/bin/python -m pytest -q python/tests
```

### Play with the environment

```python
import furrow

env = furrow.Env(seed=0)                        # macro-action mode, Fixed objective
obs, info = env.reset()
print(env.describe())                           # compact text state (for humans and LLMs)
print(env.valid_actions()[:8])                  # ['Wait30', 'Sleep', 'GoTo(village)', ...]
obs, reward, terminated, truncated, info = env.step("Buy(radish_seed,5)")   # by name or by id
obs, reward, terminated, truncated, info = env.step(env.expert_action())    # the scripted speedrunner
print(env.render())                             # ASCII map

vec = furrow.VecEnv(64, seed=0)                 # 64 envs stepped by Rust threads (GIL released)
obs, info = vec.reset()
obs, rew, term, trunc, info = vec.step(vec.expert_actions())   # info["final_obs"]/["_final_obs"] on episode ends

furrow.generate_demos("data/expert", episodes=1000, noise=0.0)   # furrow-traj/2 dataset, ~90k steps/s
```

Gymnasium (`passes gymnasium.utils.env_checker`):

```python
import gymnasium as gym, svrl.envs
env = gym.make("Furrow-Macro-v1")               # also Furrow-Primitive-v1, Furrow-Remixed-v1
vec = svrl.envs.FurrowVecEnv(128, seed=0)       # gymnasium.vector.VectorEnv, AutoresetMode.SAME_STEP
```

Terminal: `cargo run --release -p furrow-cli -- play --seed 0` (also `render`, `expert`, `demos`, `bench`).

### Reproduce the experiments and papers

```bash
.venv/bin/python -m svrl.pipeline --config python/configs/pipeline.yaml   # collect → bc → dagger → ppo → evaluate → report
typst compile --root papers papers/furrow/main.typ && typst compile --root papers papers/speedrun/main.typ
```

Every number in `docs/RESULTS.md` and in the papers is generated from `runs/v1/results.json`.

### LLM agents (NVIDIA NeMo Gym)

`python/nemo_gym_furrow/` is a NeMo Gym resources server that exposes Furrow to LLM agents as tools
(`observe`, `list_actions`, `act`, `status`). Tasks are speedrun *segments*, and the verify reward is
lexicographic: any success outranks any failure, and faster is better relative to the expert. See
[`python/nemo_gym_furrow/README.md`](python/nemo_gym_furrow/README.md) and
[`docs/NEMO_GYM.md`](docs/NEMO_GYM.md).

### The real game

`mod/RLBridge` (C# SMAPI mod) and `crates/sdv-bridge` (Rust client: `SdvClient`, `SdvEnv` with the
Community Center objective, plus a mock server for tests) implement the protocol in
[`docs/BRIDGE_PROTOCOL.md`](docs/BRIDGE_PROTOCOL.md). You need your own copy of Stardew Valley 1.6
with SMAPI 4. See [`mod/RLBridge/README.md`](mod/RLBridge/README.md) for status and caveats.

## How it works

```
                 ┌──────────────────────────── Python: svrl ─────────────────────────────┐
                 │ data (mmap'd .npy shards) → BC / time-conditioned BC / DAgger / PPO     │
                 │ finetune (KL to the imitation prior) → paired-seed evaluation → report  │
                 └───────────────▲──────────────────────────────────────▲─────────────────┘
      furrow-traj/2 datasets     │ furrow (PyO3 abi3 wheel)              │ Gymnasium / NeMo Gym
┌────────────────────────────────┴───────────┐        ┌─────────────────┴──────────────────────┐
│ Rust workspace                              │        │ Real game                               │
│  furrow        sim, Env, VecEnv, obs, text  │        │  mod/RLBridge   SMAPI mod: lockstep,    │
│  furrow-expert stateless speedrun planner   │        │                 snapshots incl. bundles,│
│  furrow-gen    episode-parallel demos       │        │                 input + macros, recorder│
│  furrow-traj   dataset format (npy shards)  │        │  sdv-bridge     Rust client + SdvEnv    │
│  furrow-cli    play / bench / demos         │        │                 + mock server           │
│  furrow-py     Python bindings              │        └─────────────────────────────────────────┘
└─────────────────────────────────────────────┘
```

Key design decisions are in [`docs/DESIGN.md`](docs/DESIGN.md), the contract the code is built
against. Two independent critiques shaped it; see [`research/`](research/).

- **Macro actions.** 192 actions such as `Plant(crop,field)`, `Fish(spot)`,
  `Buy(item,1|5|25)`, `Deliver` and `Sleep`. Deterministic controllers with BFS path finding
  execute them, time-boxed and masked by cheap preconditions. That gives 15–40 decisions per
  in-game day instead of about 10⁵ keyboard inputs per episode. A primitive interface (67 actions
  plus an egocentric grid) also exists.
- **The reward is speed only.** It is −1 per in-game day plus potential-based progress shaping,
  discounted per *day* (never per decision), so the return-optimal policy is the fastest one.
- **Item-centric observations** (777 floats): what is still needed, what is held, prices,
  travel times to every point of interest, crop and machine status, calendar, weather and forecast.
- **Bit-identical determinism.** An in-crate PCG32/SplitMix64 RNG; exogenous randomness keyed by
  `(seed, day, purpose)` so every policy faces the same world on a seed (paired evaluation);
  stable state hashing; vectorised stepping identical to serial stepping; dataset generation
  identical across thread counts.
- **A stateless expert.** A priority-rule planner with a seasonal cost model. It is a function of
  the current state only, so it labels learner-visited states (DAgger) and supports noise
  injection with clean labels (DART).
- **Evaluation.** Censored completion day `C` on 200 held-out test seeds, with success rate,
  mean/IQM/median and bootstrap CIs, a speed score, and per-seed paired differences to the expert.

## Repository layout

```
Cargo.toml, rust-toolchain.toml   Rust workspace (pinned stable)
crates/
  furrow/            simulation core + RL env; original content in crates/furrow/data/*.toml
  furrow-expert/     scripted speedrun planner (expert policy, DAgger oracle)
  furrow-gen/        episode-parallel runner: expert demonstrations
  furrow-traj/       trajectory dataset format (npy shard writer)
  furrow-py/         PyO3 extension `furrow._furrow` + python shim (`import furrow`)
  furrow-cli/        `furrow` binary: play, render, expert, demos, bench
  sdv-bridge/        Rust client for the RLBridge mod (+ mock server)
mod/RLBridge/        C# SMAPI mod for the real game
python/
  svrl/              envs (Gymnasium), data, models, bc, dagger, ppo, evaluate, pipeline, report
  nemo_gym_furrow/   NeMo Gym resources server + recipes
  configs/, tests/
papers/              Typst sources: Furrow benchmark paper, speedrun-learning paper (+ PDFs)
research/            compiled design critiques and baseline research from the planning agents
docs/                DESIGN.md, RESULTS.md, BRIDGE_PROTOCOL.md, NEMO_GYM.md, figures/
legacy/              the previous HTTP/curl agent (superseded)
```

## Status

| Component | State |
|---|---|
| Furrow simulator, content, macro/primitive actions, observations, reward | done, tested (determinism, VecEnv ≡ serial, clock monotonicity) |
| Scripted expert | done: 98.5% success on Fixed test seeds; 66% on held-out Remixed pools |
| Python bindings, Gymnasium envs, CLI | done, tested |
| Trajectory format + parallel demo generation | done, tested (thread-count independent) |
| BC / time-conditioned BC / DAgger / PPO finetune + evaluation + report | done; results in `docs/RESULTS.md` |
| NeMo Gym resources server | see `python/nemo_gym_furrow/README.md` |
| SMAPI mod (C#) | written against the documented API; **not compiled here** (needs .NET + the game) |
| `sdv-bridge` Rust client | done, tested against the mock server |
| Papers (Typst → PDF) | `papers/furrow/main.pdf`, `papers/speedrun/main.pdf` |
| Not yet built | entity-centric action head, sequence-model policies, IQL, golden-trajectory files, lints from DESIGN §2.5, rollout-improved expert |

## Acknowledgements

[SMAPI](https://smapi.io), [Gymnasium](https://gymnasium.farama.org),
[NVIDIA NeMo Gym](https://github.com/NVIDIA-NeMo/Gym). Stardew Valley is © ConcernedApe. This
project is not affiliated with or endorsed by ConcernedApe and redistributes none of the game.

## License

MIT, see [LICENSE](LICENSE).
