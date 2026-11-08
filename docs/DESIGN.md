# Design contract — speedrun agents learned from trajectories

> **Implementation status (2026-11-08).** Implemented: content (`crates/furrow/data`), sim core,
> macro + primitive actions and masks, observations (777 dims), per-day discounted shaped reward,
> `Env`/`VecEnv` (rayon, same-step auto-reset, deterministic seed schedule), `describe()`/ASCII
> render, the scripted expert (100% success on test seeds), `furrow-traj/2` writer, the
> episode-parallel generator, PyO3 bindings, Gymnasium wrappers, BC / time-conditioned BC /
> DAgger / PPO finetuning, the evaluation protocol and the report stage. Deviations from the
> contract below, all deliberate for v1:
> - Content is parsed once at runtime (`OnceLock`), not compiled by `build.rs`;
>   `OBS_GLOBAL_DIM` is `furrow::obs::obs_global_dim()`. Ids are still explicit and dense.
> - The Python `Env.step` returns the Gymnasium 5-tuple `(obs, reward, terminated, truncated,
>   info)` and accepts action ids **or names**; `VecEnv.step` likewise (info carries
>   `final_obs`/`_final_obs`, `day_ends`, `ticks`, `outcome`, episode stats).
> - The expert computes its features from `GameState` accessors that mirror observation
>   features (no `ExpertView` decoder yet); `cost_to_go`/rollout-improved expert not yet built.
> - Datasets store observations directly (replay/materialize tools from action logs are future
>   work); columns are as written by `furrow-traj` (see its crate docs).
> - Learners use a flat masked head (entity-centric head, sequence transformer and IQL are
>   future work). Lints (`clippy.toml`), golden-trajectory files and the allocation test are
>   future work; determinism is covered by tests.
> - The SMAPI bridge (`mod/RLBridge`, `crates/sdv-bridge`) is designed (§7) but not implemented.

Status: **v2.1 contract** = v1 + the RL critique (`research/reviews/2026-11-08-rl-critique.md`) +
the Rust-systems critique (`research/reviews/2026-11-08-systems-critique.md`). Every crate/package in
this repo is implemented against this document. If an implementation needs to deviate, update
this file in the same change and say why.

## 0. Goal and ground rules

**Objective.** Train a model, primarily from trajectories (demonstrations), to *speedrun a
long-horizon restoration objective*: complete every delivery request of a multi-wing building
restoration project (the analogue of finishing a farming game's "community center") in as few
in-game days as possible.

Two environments share one agent interface:

| | **Furrow** (`crates/furrow*`) | **SDV bridge** (`mod/RLBridge`, `crates/sdv-bridge`) |
|---|---|---|
| What | An *original* farming/life-sim written from scratch in Rust | A SMAPI mod + Rust client that drives the user's own installed copy of Stardew Valley |
| Speed | 10⁴–10⁶ steps/s (vectorised, headless) | real-time-bound (lockstep, ~10²–10³ ticks/s) |
| Objective | Restore the Old Mill (original content) | Complete the Community Center (read from the live game) |
| Use | Scalable data generation, imitation + RL training, benchmarking | Recording human trajectories, evaluating/finetuning in the real game |

**Originality rule (hard).** Furrow is an original game in the farming-sim genre. It MUST NOT
contain Stardew Valley code, decompiled code, data, maps, art, character/NPC/place/item names
specific to Stardew Valley, or its specific numeric tables. Genre mechanics (seasons, crops,
tools, foraging, fishing, mining, shops, crafting, a collection-based restoration goal) are fine;
all names, numbers, maps, and request lists are designed fresh (see `crates/furrow/data/`,
`docs/CONTENT.md`). The SDV bridge never ships game code or data: it reads/writes the running
game through SMAPI's public modding API at runtime, on the user's own licensed copy.

**Primary track.** Macro-action mode (§3.4) is the primary research track (15–40 decisions per
day, 1–6k per episode). Primitive mode (§3.3) is a secondary track kept for hierarchical work.

## 1. Repository layout and crate boundaries

```
Cargo.toml                 # Rust workspace (all crates below)
rust-toolchain.toml        # pinned stable (1.99.1), bumped deliberately
crates/
  furrow/                  # GameState, Env, VecEnv, compiled content (build.rs), rayon behind `parallel` feature
    data/*.toml            # original content tables (compiled by build.rs into consts + static tables)
    build.rs               # parses + validates data/*.toml, emits $OUT_DIR/content.rs
  furrow-expert/           # scripted speedrun planner (macro-level expert policy) + ExpertView
  furrow-traj/             # trajectory dataset format (schema, .npy writer/reader, memmap2, half); NO in-tree deps
  furrow-gen/              # the one episode-parallel runner: generate / relabel / materialize / evaluate
  furrow-py/               # PyO3 extension `furrow._furrow` ([lib] name = "_furrow") + python/furrow shim + .pyi
  furrow-cli/              # `furrow` binary: play / render / bench / demos / replay / traj …  (calls furrow-gen)
  sdv-bridge/              # Rust client for the RLBridge SMAPI mod (+ mock server); depends only on furrow-traj
mod/RLBridge/              # C# SMAPI mod for the real game
python/                    # `svrl` training package (pyproject.toml, pure Python + torch)
  svrl/ data.py models.py bc.py dt.py dagger.py iql.py ppo.py evaluate.py pipeline.py envs/ backend.py
  furrow_mock/             # pure-Python stand-in with the §3.7 surface (tests, NeMo Gym dev)
  nemo_gym_furrow/         # NeMo Gym resources server (§9)
  configs/*.yaml
  tests/
papers/                    # Typst sources (§10)
docs/                      # this file, CONTENT.md, TRAJECTORY_FORMAT.md, BRIDGE_PROTOCOL.md,
                           # SMAPI_NOTES.md, NEMO_GYM_*.md, BASELINES.md, RESULTS.md, reviews/
legacy/                    # the previous HTTP/curl Python agent, kept for reference
```

Boundaries: `furrow-traj` has no in-tree dependencies so real-game recordings share the format;
`furrow-gen` owns the parallel episode loop and the seed scheme — `furrow-cli` and `furrow-py`
call it and never reimplement it; `sdv-bridge` uses one OS thread per game connection (blocking
I/O never runs on rayon workers). Toolchain: Rust 1.99.1 (edition 2021), Python ≥3.11 with
`uv`, torch ≥2.4. Workspace deps (declared once): serde, serde_json, toml (build-dep), rayon,
thiserror, anyhow, clap, flate2, memmap2, half, pyo3 0.30 (`abi3-py311`, no
`extension-module` feature — maturin sets `PYO3_BUILD_EXTENSION_MODULE`), numpy 0.30, tempfile,
proptest (dev), criterion (dev). No `zip`, no `rand`.

## 2. Furrow — the simulation (`crates/furrow`)

### 2.1 Time, calendar, weather
- 1 **tick = 1 game minute**. Day runs 06:00 (minute 360) to 02:00 (minute 1560): 1200 ticks.
  The absolute clock (`abs_minute: u32`) also counts the skipped night, so elapsed time is
  well defined across day boundaries. Time is **event-driven**: timers are absolute minute
  stamps (`ready_at`, `upgrade_done_at`); `advance(minutes)` handles only events due in the
  interval (shop open/close, machine ready, 02:00 pass-out, 06:00 spawns) so per-tick work is
  O(1); overnight processing is one linear pass over the soil planes.
- 4 seasons (Spring, Summer, Autumn, Winter) × **24 days** = 96-day year. Day index `d ≥ 0`
  counts from Spring 1, Year 1. Weekday = `d % 7` (shops may close on a weekday).
- Weather per day ∈ {Sunny, Rainy, Stormy}(stormy counts as rainy), plus Snowy in Winter
  (counts as sunny for crops). Tomorrow's weather is rolled at day start and is observable
  (forecast). Rain waters all outdoor soil.
- At 02:00 the farmer *passes out*: day ends, wakes with 50% energy and pays a 10% money
  penalty (capped at 1000). Sleeping in bed at any time ends the day normally (full energy).

### 2.2 World and state layout
Locations are tile grids with terrain, objects, and (farm/greenhouse only) soil. All maps are
**original and procedurally laid out from a seed** on top of a fixed topology (the set of
locations, warps and POIs is fixed by content; only their positions move):

| id | Location | Size (w×h) | Contents |
|---|---|---|---|
| 0 | Homestead (farm) | 48×36 | bed, shipping crate, well, tillable field, machine pads, debris, small pond |
| 1 | Village | 44×32 | shops (general store, smithy, fishmonger, carpenter), Old Mill board, merchant stall |
| 2 | Woods | 40×32 | forage spawns, trees, hardwood stumps, river (fishing) |
| 3 | Coast | 40×24 | beach forage, tide pools, sea (fishing) |
| 4 | Quarry | 32×24 | mine entrance (+ elevator), rock nodes, mountain lake (fishing) |
| 5 | Mines | 24×24 per floor | floors 1..=60, procedurally generated per (seed, day, floor) |
| 6 | Greenhouse | 16×12 | locked until unlocked by a restoration reward; plant any season |

(Exact names/sizes may be revised by `docs/CONTENT.md`; the content tables are authoritative.)
Locations connect through edge/door **warps** (a world graph). Each location exposes named
**points of interest** (POIs, e.g. `bed`, `crate`, `store_counter`, `mill_board`, `river_spot`)
used by macro actions and by the observation (§3.5 ETAs).

Terrain (u8): `Void, Grass, Dirt(tillable), Path, Sand, Floor, Water, Wall, Bridge`.
Objects (u8 kind + u16 data): `None, Debris{Weeds|Twig|Stone}, Tree{stage}, Stump, Boulder,
Rock{kind}, Forage{item}, Building, Counter{shop}, Bed, Crate, Well, MillBoard, Ladder,
Elevator, MachinePad, Machine{handle}, Warp`.
Soil (farm/greenhouse): `tilled, watered, fertilizer, crop_stage, crop: Option<handle>`.

**Memory layout.** `GameState` keeps every location's tiles plus the current mine floor in one
contiguous per-env arena of SoA planes (`terrain u8, obj_kind u8, obj_data u16, soil u8,
crop_stage u8`) with an incrementally maintained passability bitset. Crops, machines and forage
live in fixed-capacity slabs sized by content consts and addressed by `u16` handles; backpack,
skills and tools are fixed arrays. Nothing grows after `new()`: `reset` regenerates in place
and `Env: Clone` is about one arena memcpy (≈40–60 KB/env).

### 2.3 Player
- Position (location, x, y), facing (N/E/S/W), energy (max 200; upgradable), money (start 500).
- Backpack: **24 slots** of `(item_id u16, count u16)`; stacks ≤ 999; selected slot index.
- Tools (always owned, not in backpack): Hoe, Watering Can, Axe, Pickaxe, Rod — each with a
  level 0..=3. Upgrades at the smithy cost money + bars and take **2 days** (tool unusable).
  Watering-can capacity 40/60/80/120 by level; refill at well/water.
- Skills: Farming, Foraging, Fishing, Mining — XP → level 0..=10 (thresholds in
  `data/skills.toml`). Levels reduce tool energy cost, raise yields/catch rates, unlock recipes.

### 2.4 Mechanics (MUST)
- **Farming**: Hoe tills Dirt (2 energy). Planting a seed on tilled soil. Watering. Overnight, a
  watered (or rained-on) crop advances one growth day; a crop not in season (outside the
  greenhouse) dies at season change. Harvest when grown; regrowing crops reset to their
  regrow stage. Yields per `data/crops.toml`.
- **Debris/trees**: Weeds (any tool / free), Twig (axe), Stone (pickaxe), Stump (axe lvl ≥1),
  Boulder (pickaxe lvl ≥2). Trees take N axe hits → wood (+sap). Hits needed shrink with level.
- **Foraging**: each day at 06:00 every forage location spawns items from its seasonal table
  up to a cap; picking up = Interact on the tile; skill can double.
- **Fishing**: Use the rod facing water. A cast consumes `t ~ U[a,b]` minutes (shorter with
  level/rod) and yields a fish drawn from the location/season/weather/time table subject to
  difficulty vs. skill+rod (else junk/nothing). No reflex minigame.
- **Mining**: Quarry elevator gives access to floors reached in multiples of 5. Rocks need
  pickaxe hits by hardness; drop stone/ore/gems by depth table; breaking rocks may reveal the
  ladder (Interact → next floor). Leaving the mines returns to the Quarry.
- **Shops**: open hours + closed weekday per shop; stock may depend on season. Buying requires
  money and backpack space. The wandering **merchant** (Village, 2 fixed weekdays) sells 4 random
  items (including off-season items) per visit — routing RNG, rolled per (seed, day) (§2.5).
- **Shipping**: items placed in the crate are paid at their sell price overnight.
- **Crafting**: recipes (`data/recipes.toml`) turn inputs into items/machines; unlocked by
  skill levels or rewards. Machines are placed on farm **machine pads**: load input → output
  after a duration (minutes or days) → collect.
- **Eating**: edible items restore energy.
- **Restoration (the objective)**: the Old Mill has **wings**; each wing has **requests**; each
  request lists candidate slots `(item, qty)` and needs `k` of them, or a coin amount. Items are
  delivered at the Mill board. Completing a wing grants a reward (item, money, or **feature
  unlock**: greenhouse, minecart fast-travel, etc.). **Episode success = all wings complete.**
  Request lists are fixed in `data/restoration.toml` (`RestorationMode::Fixed`); the
  `Remixed` mode resamples **slot item identities only** (never wing/request/slot counts or `k`)
  from per-request pools using the episode seed. Pools are partitioned
  (`RemixPartition::{Train, Test}`) for OOD evaluation. Content validation rejects/resamples a
  Remixed list whose static critical-path lower bound (season windows, growth days, unlock
  dependencies) exceeds `max_days − 24`.
- **Probabilities** are integers in parts-per-million (`chance_ppm(u32)`); no float decides a
  game outcome.

### 2.5 Content, determinism and performance
- **Compiled content.** `crates/furrow/build.rs` (serde + toml as build-dependencies) parses
  `data/*.toml`, validates it (dangling ids, unobtainable requested items without circular
  dependence on the item's own wing reward, missing recipe inputs, Remixed feasibility, POIs in
  bounds) — a violation fails the build — and emits `$OUT_DIR/content.rs` with
  `pub const NUM_ITEMS, NUM_CROPS, NUM_LOCATIONS, NUM_POIS, NUM_MACROS, NUM_PRIMITIVES,
  OBS_GLOBAL_DIM, OBS_GRID_CHANNELS: usize`, static tables, typed ids (`ItemId(u16)`, …),
  `pub const CONTENT_HASH: u64` (FNV-1a 64 over the raw TOML bytes in a fixed file order) and
  the JSON string returned by `furrow.content_json()`. Every entity carries an explicit,
  immutable `id = N` inside an array of tables; ids are dense, unique, append-only across content
  versions and never derived from table-key order. Macro ids are enumerated from entity ids in a
  documented order. Grid payload ids are a dense build-time "tile-visible id" (build fails > 255).
- **Determinism key.** `pub const SIM_VERSION: u32` is bumped by hand in any change to
  dynamics, macro controllers, pathfinding, RNG draw order or obs encoding (goldens fail until
  re-blessed together with a bump). Given `(SIM_VERSION, CONTENT_HASH, EnvConfig, seed,
  actions)`, states, rewards, masks and step flags are **bit-identical across runs, thread
  counts and platforms**. Both values are recorded in dataset `meta.json`/`episodes.jsonl`;
  replay refuses a mismatch. State identity is `GameState::stable_hash() -> u64`: explicit
  little-endian field serialization (no `usize`, no `#[derive(Hash)]`, no std `DefaultHasher`)
  through an in-crate FNV-1a-64/xxh64 with a checked-in test vector.
- **RNG.** In-crate (`furrow::rng`: SplitMix64 seeding + PCG32 XSH-RR with `below(n)`,
  `range`, `chance_ppm`); no `rand`. **Exogenous randomness** (layout, weather, forage spawns,
  merchant stock, mine floors, daily fish tables) never comes from a long-lived stream: each
  event seeds a fresh PCG32 from `mix(episode_seed, DOMAIN_TAG, day[, location | floor])`, so it
  never depends on the policy's actions and two policies on one seed face identical worlds
  (paired evaluation, §6). Only outcomes of agent actions (catches, drops, yields, skill
  doubling) use the per-env **dynamics** stream, which lives in `GameState` and is covered by
  `stable_hash`. **Expert randomness** (noise, style draws) is counter-based, keyed by
  `(expert seed, episode seed, step index)`, so labels are pure functions of (state, keys).
- **No HashMap/HashSet, no nondeterministic float.** `clippy.toml` in `furrow` and
  `furrow-expert` sets `disallowed-types = [HashMap, HashSet]` and `disallowed-methods` for
  `f32/f64::{ln, ln_1p, log2, log10, exp, exp_m1, powf, powi, sin, cos, tan, hypot, cbrt}` and
  `Instant::now`/`SystemTime::now` (allowed in build.rs, benches, furrow-cli). Observation
  log-scalings use build-time lookup tables (`LOG1P_COUNT[1000]`, log-money by integer
  magnitude). Use `Vec`/`BTreeMap` where iteration order matters.
- **Pathfinding.** Routing between locations uses a next-hop warp table computed once per
  content version (static topology). Within a location: 4-neighbour A* with a Manhattan
  heuristic and a binary heap over per-env scratch buffers with generation stamps (no clearing,
  no allocation). Multi-target macros pick the next target with one early-exit BFS from the
  player or a boustrophedon sweep over rectangular plots — never one A* per candidate. Per-env
  distance tables (location: warps ↔ POIs) are built lazily on first entry and invalidated by a
  per-location `walk_version`; nothing is precomputed in `reset`; no cache is shared across envs.
- **Allocation-free hot paths** (`tests/alloc_free.rs` with a counting global allocator asserts
  0 allocations over 10k `step` + `write_obs` + mask calls after warm-up, both action modes).
- **Targets** (dev box: AMD Ryzen 7 9700X, 8C/16T, 32 MB L3, 16 pool threads;
  `profile.release`: `lto = "fat"`, `codegen-units = 1`, `panic = "unwind"` so PyO3 turns
  panics into exceptions): primitive ≥ 2M steps/s aggregate at N=1024 with random valid actions
  including obs+mask writes; macro ≥ 20k steps/s aggregate replaying golden expert action logs
  (the expert's state distribution) with per-macro-kind p99 ≤ 1 ms; `reset` p99 ≤ 50 µs.
  `furrow bench` reports steps/s at 1/2/4/8/16 threads, p50/p99, per-macro-kind histograms,
  RSS/env and the Python-loop rate; method: fixed seeds, 2 s warm-up, ≥ 5 runs × 10 s, median
  with min/max, system load recorded; absolute numbers are never a CI gate. criterion 0.8
  micro-benches cover step per macro kind, write_obs, mask, A*, reset. `[profile.profiling]`
  inherits release with `debug = "line-tables-only"`.

## 3. Furrow — the RL environment API

### 3.1 Config
```rust
pub struct EnvConfig {
    pub seed: u64,
    pub action_mode: ActionMode,          // Primitive | Macro      (default Macro)
    pub restoration: RestorationMode,     // Fixed | Remixed        (default Fixed)
    pub remix_partition: RemixPartition,  // Train | Test           (default Train)
    pub max_days: u16,                    // compute cap (truncation), default 192; NOT part of the objective
    pub max_steps: u32,                   // compute cap: 100_000 (Primitive) / 20_000 (Macro) by default
    pub max_decisions_per_day: u16,       // default 64; afterwards only Sleep is unmasked (macro mode)
    pub macro_max_minutes: u16,           // default 120: time box; a macro hitting it returns Partial (no penalty)
    pub grid: bool,                       // include the egocentric grid in obs (default: true in Primitive, false in Macro)
    pub grid_radius: u8,                  // default 6 → 13×13
    pub reward: RewardConfig,
    pub start_money: u32,                 // default 500
}
/// Speed-only objective + potential-based progress shaping, discounted PER IN-GAME DAY.
pub struct RewardConfig {
    pub per_day: f32,        // default -1.0: charged at every day end (Sleep or pass-out). Return unit = days.
    pub completion: f32,     // default 0.0 terminal bonus
    pub phi_slot: f32,       // potential weights: Φ(s) = phi_slot·slots + phi_request·requests + phi_wing·wings,
    pub phi_request: f32,    //   scaled so Φ_max ≈ expert mean completion days (defaults set from content;
    pub phi_wing: f32,       //   e.g. 0.25 / 0.5 / 2.0 for ~100 slots / 28 requests / 6 wings)
    pub shaping: bool,       // default true: r += γ_day^day_ends · Φ(s') − Φ(s)   (Φ is NOT zeroed at termination)
    pub gamma_day: f32,      // default 0.99 (never < 0.97); learners MUST use this value
}
```
Raw per-slot/request/wing rewards and costs per simulated tick are **not allowed**: the
expected discounted return is then an increasing affine function of E[γ_day^T] (T = completion
day), so the optimum is pure speed, and shaping is potential-based (Ng et al. 1999).

### 3.2 Core API (Rust, `furrow::Env`, `furrow::VecEnv`)
```rust
impl Env {
    pub fn new(cfg: EnvConfig) -> Env;
    pub fn reset(&mut self, seed: u64);
    pub fn step(&mut self, action: u32) -> Step;   // in cfg.action_mode
    pub fn step_primitive(&mut self, a: Primitive) -> Step; // always available (macros use it)
    pub fn num_actions(&self) -> usize;
    pub fn action_mask(&self) -> &ActionMask;      // bitset [u64; ⌈A/64⌉], cached at end of reset/step;
                                                   //   .write_bools(&mut [bool]) / .write_packed_le(&mut [u8])
    pub fn write_obs(&self, global: &mut [f32], grid: Option<&mut [u8]>);
    pub fn state(&self) -> &GameState;            // read-only full state (macro controllers, tests)
    pub fn stable_hash(&self) -> u64;
    pub fn render_ascii(&self) -> String;
    pub fn describe(&self) -> String;              // compact deterministic text state (LLM agents)
}
impl Clone for Env {}                              // bit-identical snapshot (one arena memcpy)
#[derive(Clone, Copy)]
pub struct Step {
    pub reward: f32,
    pub terminated: bool,     // all wings complete
    pub truncated: bool,      // max_days or max_steps reached
    pub day_ends: u8,         // day boundaries crossed by this transition (0 within a day)
    pub ticks: u32,           // absolute in-game minutes elapsed, including skipped night
    pub outcome: MacroOutcome,// Done | Partial (time box hit, may be re-issued) | Failed | Masked
}
pub const SIM_VERSION: u32; pub const CONTENT_HASH: u64; pub const OBS_GLOBAL_DIM: usize; // from build.rs
```
**VecEnv** (normative). `VecEnv::new(cfg, num_envs, num_threads, seed_range: (u64, u64))`;
`reset(&mut self, out: &mut VecOut)`; `step(&mut self, actions: &[u32], out: &mut VecOut) ->
Result<(), StepError>` (Err on wrong length or action ≥ num_actions); `envs()`/`envs_mut()`.
Each VecEnv owns a `rayon::ThreadPool` (default `available_parallelism`) entered via
`pool.install`, never the global pool; macro mode uses `par_iter_mut()…with_max_len(1)`,
primitive mode `with_min_len(max(1, N/(8·threads)))`. `VecOut` holds caller-owned slices:
`obs_global [N·G] f32`, `obs_grid Option<[N·C·H·W] u8>`, `mask [N·A] bool`, and per env
`reward f32, terminated, truncated bool, day_ends u8, ticks u32, outcome u8`; for finished
episodes `final_global, final_grid, final_mask` (the complete terminal observation),
`final_day_ends`, `ep_return f32, ep_days u16, ep_ticks u32, ep_success bool, ep_step_cap bool,
ep_seed u64, ep_slots/requests/wings u16, ep_decisions u32`. **Same-step auto-reset**: for env
`i` with `terminated|truncated`, `reward/terminated/truncated/day_ends/ticks/ep_*` describe the
finished episode's last transition, `final_*` its terminal observation, and `obs_*`/`mask` the
first observation of the next episode; for other envs `final_*`/`ep_*` are stale. **Seeds**:
env `i`'s `k`-th episode uses `seed(i,k) = lo + splitmix64(splitmix64(cfg.seed) ^ ((i as u64)
<< 32 | k as u64)) % (hi − lo)` over `seed_range = (lo, hi)` — independent of thread count and
completion order, reported in `ep_seed`. **Eval mode** (`VecEnv::with_seeds(cfg, seeds,
num_threads)`) steps an explicit seed list without auto-reset and exposes `done_mask()`.
Workloads with no neural policy in the loop (expert demos, DAgger relabel, expert/random
baselines, obs materialization, benchmarks) do **not** use lockstep VecEnv: `furrow-gen` runs
one `(Env, Expert)` episode per rayon task to completion into a thread-local buffer and commits
results in episode-id order through a bounded reorder buffer (byte-identical output across
thread counts).

### 3.3 Primitive action space (Discrete, `furrow::action::Primitive`) — secondary track
| ids | action | ticks |
|---|---|---|
| 0 | Wait | 10 |
| 1–4 | Move N/E/S/W (turn to face; step if passable) | 1 |
| 5 | UseTool (selected tool/item on faced tile: till, water, chop, mine, fish, plant seed, place machine) | per content (≥1) |
| 6 | Interact (faced tile: pick up forage, ship selected stack at crate, open shop at counter, deliver selected at mill board, sleep at bed, ladder, warp, machine load/collect, refill can at well) | per content (≥1) |
| 7–30 | Select backpack slot 0..23 | 1 |
| 31–35 | Select tool Hoe/Can/Axe/Pickaxe/Rod | 1 |
| 36 | Eat selected | 1 |
| 37 | Open crafting menu | 1 |
| 38–53 | Menu option 0..15 (shop: buy stock row k; crafting: craft recipe k; smithy: upgrade) | 1 |
| 54 | Close menu | 1 |
**Every step advances the absolute clock by ≥ 1 tick.** Actions invalid in the current state are
masked; stepping a masked action is Wait(1) with `outcome = Masked`. In primitive mode the
observation includes the item id and price of each visible shop-menu row.

### 3.4 Macro action space (Discrete, enumerated at build time from content) — primary track
Every macro is executed by a deterministic controller that issues primitive steps until done,
time-boxed (`macro_max_minutes`), the day ends, or it fails. Normative semantics:
1. **Ids** are a pure function of the content version, sorted by `(kind, entity id)`, listed in
   dataset `meta.json`. They never depend on seed, `RestorationMode`, season or current stock.
2. `Buy(shop, item, q)` with `q ∈ {1, 5, 25}` packs, enumerated over the union of all stock
   tables including the full merchant pool; masked when not stocked today, unaffordable, or no
   space. Never indexed by menu row.
3. `Plant(seed, Farm | Greenhouse)` tills as needed and plants all held seeds of that kind in a
   fixed tile order (time-boxed).
4. `Craft(recipe)` crafts one; machines are placed automatically on the next free machine pad.
5. `ClearField`, `ChopWood`, `ChopHardwood`, `Forage(location)`, `Fish(spot)`, `MineFloor`,
   `MineDescend` are **time-boxed** (≤ 60–120 min, per content) and stop early when done.
6. `Deliver` assigns held items to open slots by a deterministic max-matching: maximize
   completed requests, prefer items that are candidates for fewer other open slots. Coin
   requests are paid only through `PayCoins(request)`, masked unless affordable.
7. `LoadMachines`, `ShipSurplus` and `Eat` never consume items that the shared **reservation
   function** (items needed by open requests, as seeds for planned crops, or as recipe inputs)
   marks as reserved.
8. Other macros: `Wait30`, `Sleep`, `GoTo(location)`, `WaterCrops`, `Harvest`,
   `CollectMachines`, `UpgradeTool(tool)`.
9. Macros cost **≥ 5 ticks even when they fail** (`outcome = Failed`, elapsed time charged;
   `Partial` when the time box was hit, no penalty, may be re-issued). A day accepts at most
   `max_decisions_per_day` macro decisions; afterwards only `Sleep` is unmasked. Stepping a
   masked macro = Wait(10 min) with `outcome = Masked`.
10. **Mask rule**: masked ⇒ never executed; unmasked ⇒ advances ≥ 1 tick, possibly returning
    `Failed`. Predicates are conservative O(1) checks on maintained summaries (per-location
    forage counts, per-shop "reachable before close" from the warp table + ETAs, held ∩
    open-slot counts, affordable, space, inputs held); budget ≤ 2 µs per env.
`furrow::macro_action::MacroAction` has `fn name(&self) -> String` and a table
`macro_actions() -> &'static [MacroAction]`.

### 3.5 Observations
`write_obs(global, grid)`:
- `global: [f32; OBS_GLOBAL_DIM]` — fixed layout exported as
  `furrow::obs::global_layout() -> &'static [(name, offset, len)]`. **Shapes depend only on
  content; no feature depends on `EnvConfig`; identical meaning in Fixed and Remixed.** Scaling
  uses content constants and lookup tables only. Groups:
  - calendar: day-of-season/24, season one-hot, year/4, weekday one-hot, minute-of-day/1560,
    **days elapsed / 384**, days left in season/24; weather today + forecast one-hots;
  - player: energy/max, `log1p(money)/log1p(1e6)` and `min(money,1e4)/1e4` (table-based),
    location one-hot, x/w, y/h, free backpack slots/24, tool levels/3 and upgrade timers/2, skill
    levels/10, current mine floor/60, unlocked-feature flags, unlocked-recipe flags,
    `last_outcome` one-hot;
  - **items** (per item id): `held[i] = LOG1P_COUNT[count]/8`, `need[i]` = units still required
    by open requests (a k-of-N request counts each candidate until k are filled; /16, clipped),
    `cover[i] = min(held, need)/need`, `n_open_slots[i]/8` (scarcity), `price[i]` (today's best
    buy price, table-log-scaled, 0 if not purchasable today);
  - **requests/wings**: per request `filled/k`, done flag, remaining coin amount/1e4; per wing done;
  - **ETAs**: `eta[poi]` = travel minutes from the player to every macro-target POI (one BFS
    from the player per step + cached per-location warp↔POI tables + warp next-hop table) /600;
    `opens_on_arrival[shop]`;
  - farm: free tilled and tillable tiles per field (farm, greenhouse) /max; per crop: planted,
    watered, ready counts (/max tiles), min days-to-ready/24, count maturing before season end;
    machine summary per machine type (idle/working/ready); pending crate value (table-log);
  - world: forage available per location/cap, merchant stock multi-hot, shop-open flags,
    active-menu one-hot (primitive mode), menu rows' item ids/prices (primitive mode).
- `grid: [u8; C×(2r+1)×(2r+1)]` (only when `cfg.grid`), channel-major, egocentric around the
  player: terrain, object kind, tile-visible id, soil flags (tilled|watered<<1|fertilized<<2),
  crop growth stage, is-player. Out-of-map = Void.
- `action_mask` for the active action mode.

### 3.6 Reward, discounting and termination
Per `RewardConfig` (§3.1). **Discounting is per in-game day, never per decision.** A transition
that crosses `day_ends` day boundaries is discounted by `γ_day^day_ends` (= 1 within a day).
GAE, IQL/TD targets, DT returns and critic warm-up all use this. `terminated` when all wings
complete; `truncated` at `max_days` or `max_steps` (compute caps, not part of the objective;
PPO may train with a shorter cap, bootstrapping from `final_obs`). Info per finished episode as
in §3.2 (`ep_step_cap` marks step-cap truncations).

### 3.7 Python native module `furrow` (crates/furrow-py) — the contract `svrl` codes against
Package `furrow` = `python/furrow/__init__.py` shim re-exporting the abi3 extension
`furrow._furrow` (pyo3 0.30 — GIL released via `py.detach`; `.pyi` stubs + `py.typed`).
```python
import furrow
furrow.__version__; furrow.SIM_VERSION: int; furrow.content_hash() -> int; furrow.content_json() -> str
cfg = dict(action_mode="macro"|"primitive", restoration="fixed"|"remixed", remix_partition="train"|"test",
           max_days=192, max_steps=None, max_decisions_per_day=64, macro_max_minutes=120, grid=None,
           grid_radius=6, start_money=500,
           reward=dict(per_day=-1.0, completion=0.0, phi_slot=..., phi_request=..., phi_wing=...,
                       shaping=True, gamma_day=0.99))                              # all optional

env = furrow.Env(seed=0, **cfg)
env.reset(seed: int) -> None
env.step(action: int) -> dict(reward=float, terminated=bool, truncated=bool, day_ends=int, ticks=int,
                              outcome="done"|"partial"|"failed"|"masked")
env.obs() -> dict(global_=f32[G], action_mask=bool[A], grid=u8[C,H,W] if cfg.grid)   # key "global_"
env.num_actions: int; env.obs_global_dim: int; env.grid_shape: tuple[int,int,int] | None
env.action_names() -> list[str]; env.global_layout() -> list[tuple[str,int,int]]
env.describe() -> str; env.render() -> str; env.stable_hash() -> int
env.info() -> dict(day=..., minute=..., abs_minute=..., money=..., energy=..., slots_filled=..., total_slots=...,
                   requests_done=..., total_requests=..., wings_done=..., total_wings=..., decisions=...,
                   decisions_today=..., success=bool, episode_return=float, episode_seed=int)
env.clone() -> Env                                            # bit-identical snapshot
env.expert_action(style: int = 0) -> int                      # clean expert label (macro mode)
env.noisy_expert_action(noise: float, style: int = 0) -> tuple[int, int]   # (executed, clean)
env.expert_cost_to_go(action: int, n_dyn_seeds: int = 4) -> float          # mean completion day (§4)

vec = furrow.VecEnv(num_envs: int, seed_range: tuple[int,int] = (0, 2**32), num_threads: int | None = None, **cfg)
vec = furrow.VecEnv.with_seeds(seeds: list[int], num_threads=None, **cfg)   # eval mode: no auto-reset
vec.reset() -> obs dict with leading N axis
vec.step(actions: np.ndarray[int64 | int32, (N,)]) -> (obs: dict, reward: f32[N], terminated: bool[N],
         truncated: bool[N], day_ends: u8[N], ticks: u32[N],
         info: dict(outcome=u8[N],
                    done_idx=i64[k],                                      # rows that finished this step
                    final_obs=dict(global_=f32[k,G], action_mask=bool[k,A], grid=u8[k,C,H,W] if cfg.grid),
                    final_day_ends=u8[k], episode_seed=u64[k], episode_return=f32[k], episode_days=i32[k],
                    episode_ticks=i64[k], success=bool[k], step_cap=bool[k], slots_filled=i32[k],
                    requests_done=i32[k], wings_done=i32[k], decisions=i32[k]))
vec.step_into(actions, out: dict) -> info   # writes into caller arrays (e.g. rollout["global_"][t+1]); checks
                                            # dtype/shape/C-contiguity, raising TypeError/ValueError
vec.expert_actions(styles: np.ndarray[int32,(N,)] | None = None) -> i64[N]                      # clean labels
vec.noisy_expert_actions(noise: float, styles=None) -> tuple[i64[N], i64[N]]                     # (executed, clean)
vec.num_envs / num_actions / obs_global_dim / grid_shape; vec.action_names(); vec.done_mask() -> bool[N]  # eval mode

furrow.generate_demos(out_dir: str, episodes: int, seed_range: tuple[int,int] = (0, 2**32), noise: float = 0.0,
                      styles: list[int] | None = None, record_grid: bool = False, record_obs: bool = True,
                      num_threads: int | None = None, **cfg) -> dict        # writes a furrow-traj dataset (§5)
furrow.materialize(dataset_dir: str, record_grid=False, num_threads=None) -> dict  # rebuild obs cache by replay
furrow.relabel(dataset_dir: str, num_threads=None) -> dict                          # fill expert_action by replay
furrow.sdv.SdvEnv(addr: str = "127.0.0.1:7777", **cfg)  # same obs()/step()/action_names() surface
```
Mechanics: `vec.step` allocates its output arrays **under the GIL** (`PyArray::new` filled
through `&mut [MaybeUninit<T>]`, or `zeros`) and fills them in place inside one
`py.detach(|| pool.install(...))` — it never steps into internal buffers and then copies out
under the GIL. Actions are accepted as `PyArrayLike1<i64, AllowTypeChange>` and range-checked
before detaching. Concurrent calls on one `VecEnv` raise `RuntimeError: Already borrowed`.
Long calls (`generate_demos`, `materialize`, `relabel`) run in chunks and call
`py.check_signals()` between chunks so Ctrl-C works. A panic inside `VecEnv.step` becomes an
exception carrying a repro bundle (env index, episode seed, SIM_VERSION, the episode's action
log) and poisons the VecEnv until `reset()`. Arrays returned are fresh; callers may keep them.
Buffers are indexed by env id so an EnvPool-style `send/recv` can be added later.

## 4. Expert planner (`crates/furrow-expert`)
A hand-written, **state-feedback** speedrun policy over macro actions:
`Expert::new(ExpertConfig { style, noise, expert_seed }) ; fn act(&mut self, env: &Env) -> u32`.
- **Information contract.** Decision rules read only the §3.5 observation (f32, before any f16
  quantization) and static content tables, through an `ExpertView` built from `write_obs`
  output. Anything else the expert needs is added to the observation first. Full `GameState`
  access is allowed only inside macro controllers on the env side. Test: perturbing
  unobservable state (future-day RNG keys, unvisited mine floors, weather beyond the forecast)
  leaves `act()` unchanged.
- **Statelessness.** `Expert::new(cfg).act(env)` returns the same action as a long-running
  instance on any state (tested on ≥1e5 states from noise=0.3 and DAgger rollouts). Expert
  randomness is counter-based (§2.5), so repeated queries never shift a stream.
- **Recovery.** Expert takeover from states at uniformly random days of noise=0.3 rollouts
  succeeds ≥90% of the time (report extra days).
- **Success bar.** With `noise=0` completes the objective on ≥95% of seeds in Fixed *and*
  Remixed; content validation guarantees feasibility (§2.4).
- **Noise** (`noise ∈ [0,1]`, probability of a random valid macro); `noisy_act` returns
  `(executed, clean)`. `style` perturbs priorities so demos have diverse speeds; style 0 is
  canonical.
- **Cost.** `act ≤ 100 µs` p50 in release builds; `vec.expert_actions` labels envs in parallel.
- `cost_to_go(env, action, n_dyn_seeds) -> f32`: mean completion day when taking `action` then
  following the expert, using the same dynamics seeds for every compared action (needs
  `Env: Clone`). Used for the rollout-improved expert `π′(s) = argmin_a Q^E(s,a)` (headroom
  reference) and cost-sensitive labels (AggreVaTe-style) at sampled decision points.
- Strategy: maintain the set of still-needed items; each day prioritise (1) harvest/water,
  (2) deliver, (3) time-gated sources (weather/time-window fish, seasonal forage, merchant),
  (4) planting needed crops early enough to mature in-season, (5) machines, (6) tool upgrades
  that unlock needed items, (7) money for coin requests and seeds via the best profit crop,
  (8) sleep before passing out. Exposes `intent() -> u8` (which priority rule fired).
- DAgger protocol: `β_i = 0.5^i`, ≥ 8 rounds, aggregate all rounds, report score every round.

## 5. Trajectory format (`crates/furrow-traj`, `docs/TRAJECTORY_FORMAT.md`)
For `env = furrow` the **action log is the source of truth**: per episode `SIM_VERSION`,
`CONTENT_HASH`, `EnvConfig`, seed, `action u16[T]`, `expert_action i16[T]`, flags and
`final_state_hash` (≈5 B/step, in `actions/` .npy files + `episodes.jsonl`). Observation arrays
are a derived cache keyed by `obs_layout_hash`, rebuilt by episode-parallel replay
(`furrow traj materialize`, asserting every `final_state_hash`); `furrow traj relabel` fills
`expert_action` for learner-generated (DAgger) logs. Human SDV data cannot be replayed and
always keeps full observations. A dataset is a directory:
- `meta.json` — `{format: "furrow-traj/2", env: "furrow"|"sdv", sim_version, content_hash,
  env_config, reward_config, gamma_day, seed_range, seed_scheme, action_mode, num_actions,
  obs_global_dim, obs_global_dtype: "f16", obs_layout_hash, grid_shape|null, global_layout,
  action_names, generator: {...}}`
- `episodes.jsonl` — one line per episode: `{episode, seed, style, shard, start, length,
  return, days, success, step_cap, final_state_hash, generator, actor_mix}`.
- `final_obs/global.npy` `f32[E, G]` (+ `final_obs/action_mask.npy`, `final_obs/grid.npy`),
  aligned with `episodes.jsonl`.
- `shards/<writer>-<seq>/`: **one raw `.npy` per array** — little-endian, C-order,
  uncompressed, npy v1.0 header padded so data starts 64-byte aligned; shards cut only at
  episode boundaries and sized by bytes (≤ 512 MiB); writers stream rows into
  `shards/.tmp-<id>/`, patch header shapes on close, fsync, rename atomically, then append the
  shard's lines to `episodes.jsonl` (a crash leaves only an ignored temp dir); concurrent
  writers use distinct `<writer>` prefixes; zstd of whole shard dirs is for archival only.
  Arrays: `obs_global f16[T,G]` (cache), `obs_grid u8[T,C,H,W]` (optional), `action i32[T]`,
  `expert_action i32[T]` (clean expert label at this state; −1 if unavailable), `actor u8[T]`
  (0 expert, 1 injected noise, 2 learner/DAgger, 3 human), `action_mask u8[T, ceil(A/8)]`
  (bit-packed, little-endian bit order — `np.unpackbits(bitorder='little', count=A)`),
  `reward f32[T]`, `terminated u8[T]`, `truncated u8[T]`, `day_ends u8[T]`, `ticks u16[T]`,
  `outcome u8[T]`, `episode u32[T]`, `day u16[T]`, `minute u16[T]`, `progress u16[T,3]` (slots,
  requests, wings after the step), optional `expert_intent u8[T]`, optional
  `prim_offset u32[T]`/`prim_len u16[T]`. Row `t` = `(obs_t, mask_t, action_t, reward_t,
  flags_t)`; the terminal observation is not a row (it lives in `final_obs/`). `days_to_go` is
  derived by consumers (`completion_day − day`, NaN for failures). `svrl` rounds live
  observations through f16 at inference so training and serving inputs match.
Readers: Rust (`furrow_traj::Reader`, memmap2) and Python (`svrl.data`,
`np.load(mmap_mode='r')`). The SDV recorder produces the same layout (`env: "sdv"`).

## 6. Learning (`python/svrl`)
- **Canonical datasets** (from the train seed range, pinned by `content_hash` + a manifest of
  `(seed, generator)`): `fixed-expert` (noise 0, style 0, 1,000 ep); `fixed-styles` (noise 0,
  8 styles, 1,000 ep); `fixed-dart` (noise ∈ {0.05, 0.1, 0.2} with clean labels, 1,000 ep each);
  `fixed-mixed` (union + noise 0.3 and 0.5; for DT/offline RL); `remixed-expert` (5,000 ep);
  nested subsets of 25/100/400 episodes for data-scaling curves.
- **Sampling**: episode-balanced (pick an episode uniformly, then a step). Single-style data by
  default; mixed styles only with style/`days_to_go` conditioning or history models.
- **Models**: `MLPPolicy` (obs → masked logits, value) with an **entity-centric head** for
  item-parameterised macros (shared MLP over `[h_state, e_item(i), need[i], cover[i], held[i],
  price[i], q]`; flat head kept as an ablation); `SpeedrunTransformer` — causal transformer over
  the last K steps with tokens `[days_to_go, obs, prev_action]` (prev_action ablated).
- **Time-conditioned imitation**: condition on successful episodes only (or a success token
  with censored target `max_days − day + 96` for failures); also store
  `days_to_go_expected` (mean over 4 dynamics re-runs) and compare; test-time target = the
  q-quantile of training completion days, q ∈ {0.05, 0.25, 0.5}, decremented per day end,
  clipped ≥ 1; scale by /384.
- **Finetuning protocol (BC → PPO)**: (1) observation normalizer from BC, frozen — never
  gymnasium `NormalizeObservation`/`NormalizeReward` or online statistics; (2) separate actor
  and critic trunks (or stop critic gradients at the shared trunk); (3) critic warm-up: regress
  on day-discounted dataset returns, then train only the critic on frozen-BC rollouts until
  explained variance ≥ 0.8 or 20 iterations; (4) masks stored in the buffer and applied to every
  log-prob/ratio/entropy/KL; (5) KL(π_θ‖π_BC) on visited states with an annealed coefficient
  and/or decaying kickstarting cross-entropy toward `vec.expert_actions`; BC trained with label
  smoothing ≈ 0.02 or temperature calibrated on validation; (6) GAE
  `δ_t = r_t + γ_day^{day_ends_t} V(s_{t+1})(1 − terminated_t) − V(s_t)`, `s_{t+1} = final_obs`
  on truncation, recursion cut at episode ends, λ ≈ 0.98; (7) fixed reward scale (unit = days,
  value targets /100), no running reward normalization; (8) evaluation logged before the first
  actor update and every iteration (a >10% drop at iteration 1 = protocol bug).
- **Baselines (required rows)** with budgets — Track O (offline: canonical datasets, fixed
  gradient steps), Track I (interactive: ≤ 1e6 expert labels), Track R (RL: ≤ 2e8 decisions on
  train seeds): masked-uniform random; expert (canonical style; best style); rollout-improved
  expert (headroom); BC on expert data; naive BC on noisy executed labels vs. BC+DART clean
  labels; DAgger; filtered BC (fastest 10%/25%); time-conditioned BC/DT; IQL (discrete, per-day
  discount); PPO from scratch; PPO from BC and from DAgger (headline).
- **Evaluation protocol** (`evaluate.py`): seeds — train `[0, 2^32)`, validation
  `[2^32, 2^32+1000)`, test `[2^33, 2^33+1000)`; model selection on validation only. Primary
  metric: **censored completion day** `C` (= completion day if successful, else `max_days`);
  report mean, median, IQM, CDF over 1,000 test seeds, success rate, speed score
  `S = (max_days − mean C_agent)/(max_days − mean C_expert)`, per-seed paired difference vs.
  the expert with 95% bootstrap CI (same seeds, same world draws), ≥ 5 training seeds per method
  with rliable-style CIs, greedy and sampled decoding, missed-seasonal-window counts and
  wings-done-vs-day curves. Tiers: (1) Fixed held-out seeds, (2) Remixed held-out seeds,
  (3) Remixed-OOD (`RemixPartition::Test`). Writes `results.json` and `docs/RESULTS.md`.
- GPU etiquette: respect `--max-vram-gb` (default 6) via `torch.cuda.set_per_process_memory_fraction`.

## 7. Real game: SDV bridge
- **Mod** (`mod/RLBridge`, C#, SMAPI 4 / game 1.6): TCP server (default `127.0.0.1:7777`),
  newline-delimited JSON. **Lockstep**: when an agent is attached the mod gates the game update
  loop and advances exactly the requested number of ticks per `step`; human-play recording mode
  runs free. Observation = structured JSON snapshot (calendar, weather, player, inventory,
  location + local tile grid, Community Center bundle state, money, skills, tools). Actions:
  primitive input frames (move/use/interact/select) and macros (`goto`, `harvest`, `water`,
  `buy`, `deliver_bundle`, `sleep`…) executed with the game's own path-finding and APIs where
  possible. Recorder: writes JSONL.gz of `(snapshot, input, events)` for human runs. Verified
  API notes: `docs/SMAPI_NOTES.md`; protocol: `docs/BRIDGE_PROTOCOL.md`.
- **Rust client** (`crates/sdv-bridge`): typed protocol, connection pool for several game
  instances (one OS thread each), an `SdvEnv` with the same reset/step/obs shape as Furrow, a
  JSONL.gz → furrow-traj converter, and an in-process mock server used by the tests.
- **Human data** (`collect_human`): segments primitive human input into the SDV macro
  vocabulary using game events (warp, purchase, harvest, ship, bundle delivery); writes
  `expert_action` = inferred macro with a `label_confidence`, none for unsegmentable spans.
- Furrow and SDV share the API shape, **not** the meaning of observations/actions: v1 claims no
  cross-environment transfer. SDV finetuning is offline-to-online (BC/IQL init, then a small
  number of online episodes); PPO-scale finetuning in the real game is out of scope.

## 8. Data-collection + training pipeline (`python/svrl/pipeline.py`, `python/configs/pipeline.yaml`)
One reproducible, resumable runner (`python -m svrl.pipeline --config ... [--stages ...]`) whose
stages write into `runs/<name>/` (each stage skips if its outputs exist, `--force` re-runs):
1. `collect_expert` — the canonical datasets of §6 via `furrow-gen` (action logs + obs cache).
2. `collect_human` — (real game) convert RLBridge recordings (`*.jsonl.gz`) → furrow-traj.
3. `train_bc` → 4. `train_dt` (time-conditioned) → 5. `dagger` (≥ 8 rounds; learner logs
   relabelled by `furrow traj relabel`) → 6. `train_iql` → 7. `train_ppo` (from scratch, from
   BC, from DAgger; protocol of §6)
8. `evaluate` — protocol of §6 for all policies + expert + random baselines → `results.json`
9. `report` — figures (matplotlib) + `docs/RESULTS.md` + paper tables (`papers/*/generated/`).
Every stage logs config, git commit, SIM_VERSION, content hash, seeds, wall-clock and metrics.

## 9. Gym integrations (`python/svrl/envs/`, `python/nemo_gym_furrow/`)
- **Gymnasium** (Farama, ≥1.0): `svrl.envs.FurrowEnv` (`gymnasium.Env`, Dict obs `{global,
  action_mask[, grid]}`, `Discrete` actions, info carries `day_ends`, `ticks`, `outcome`)
  registered as `Furrow-Macro-v1` / `Furrow-Primitive-v1`; `svrl.envs.FurrowVecEnv`
  (`gymnasium.vector.VectorEnv` over the native `VecEnv`) **declares
  `metadata["autoreset_mode"] = gymnasium.vector.AutoresetMode.SAME_STEP`** and maps the
  native `final_obs` to `info["final_obs"]`/`info["_final_obs"]` per the Gymnasium 1.x
  convention; `SdvEnv` with the same spaces for the real game. `FurrowEnv` passes
  `gymnasium.utils.env_checker`.
- **NVIDIA NeMo Gym** (`python/nemo_gym_furrow/`): a NeMo Gym *resources server* exposing the
  macro-level game to LLM agents through tools (observe / list_actions / act / status), one
  Furrow env per rollout session. **Tasks are segments, not full runs**: start states are
  sampled from expert trajectories (seed + day); the goal is to complete the next request or
  wing, or to play N = 7 days, with a cap on tool calls. `act` accepts a single macro or a
  **day plan** (list of macros run until the day ends). **Verify is lexicographic**: any success
  scores above any failure; among successes, faster is better, measured against the expert's
  days from the same state; partial credit applies only to failures and stays below the lowest
  success score. Includes task-dataset generation, an agent config, a fake OpenAI-compatible
  model for offline tests, and a recipe for rollout collection and GRPO training with NeMo RL.
  Verified notes: `docs/NEMO_GYM_NOTES.md`, `docs/NEMO_GYM_IMPL_BRIEF.md`.

## 10. Papers (`papers/`, Typst → PDF)
Written in Typst (`typst compile`), figures from the pipeline's `report` stage, numbers only
from `results.json` (never hand-typed):
- `papers/furrow/` — *Furrow: a fast, original farming-sim benchmark for long-horizon
  speedrunning from trajectories* (environment, content, determinism, throughput, baselines).
- `papers/speedrun/` — *Learning to speedrun from trajectories* (BC vs. time-conditioned
  transformer vs. DAgger vs. IQL vs. PPO finetuning; ablations; real-game bridge; LLM agents).

## 11. Quality bar and test suites
CI (`.github/workflows/ci.yml`) runs exactly: `cargo fmt --all --check`;
`PYO3_PYTHON=$(which python3) cargo clippy --workspace --all-targets -- -D warnings` (lints
furrow-py too; `-D warnings` scoped to clippy, not global `RUSTFLAGS`); `cargo test --workspace`;
the Python job (3.11 and 3.13) builds the wheel with maturin and runs `pytest python/tests`; a
`macos-14` job runs only the golden/determinism tests in release (cross-platform bit-identity).
`nightly.yml` (release): expert ≥ 95% on 200 seeds, 100k proptest cases, bench smoke run.
Suites:
- (a) **Goldens** `crates/furrow/tests/golden/*.json`: SIM_VERSION, CONTENT_HASH, cfg, seed,
  actions, `stable_hash` every 256 steps, final hash and return; a failure reports the first
  diverging checkpoint; `FURROW_BLESS=1` rewrites them and the test fails if goldens changed
  without a SIM_VERSION bump.
- (b) **proptest** over seeds × random valid action sequences (≤ 2k steps, both modes): energy
  ∈ [0, max]; money ≥ 0; stacks ≤ 999; ≤ 24 slots; player on a passable tile; time strictly
  increases each step; item-conservation ledger; a masked primitive advances exactly 1 tick;
  clone-then-step equals step (by hash); `reset(s)` equals `new(cfg{seed: s})`.
- (c) **VecEnv vs serial**: N=37 envs × 1k steps at 1 and 16 threads produce byte-identical
  outputs to N independent Envs; auto-reset obs equals a fresh `reset(seed(i,k))`; `final_*`
  equals the pre-reset `write_obs`.
- (d) **Macro expansion**: the primitives a macro issues, replayed in Primitive mode, give an
  identical hash; re-executing the expert's macro ids reproduces its trajectory bit-identically;
  ≥ 99% of unmasked macros on noisy-rollout states advance beyond the clock, the rest return
  `Failed`/`Partial`; the canonical expert averages ≤ 40 decisions/day.
- (e) Test vectors for SplitMix64, PCG32, `below()`, `stable_hash`, CONTENT_HASH.
- (f) `alloc_free.rs`; (g) world draws independent of actions (two policies, same seed ⇒ same
  weather/merchant/forage); content validation at build time; expert statelessness, recovery,
  success rate (Fixed and Remixed), unobservable-perturbation invariance; GAE targets on a
  scripted rollout match hand-computed values for terminated and truncated endings across an
  auto-reset boundary.
- (h) **pytest**: dtype/shape/contiguity errors; fresh-array semantics; GIL release (a background
  thread progresses during a long step); the concurrent-call `RuntimeError`; an RSS-stable
  1e5-step leak check; gymnasium `env_checker` plus SAME_STEP `final_obs`; the mock backend and
  the native module pass one shared contract test suite.
