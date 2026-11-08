# Design contract — speedrun agents learned from trajectories

Status: **v2 contract** (v1 + the RL critique in `docs/reviews/2026-11-08-rl-critique.md`).
Every crate/package in this repo is implemented against this document. If an implementation
needs to deviate, update this file in the same change and say why.

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

## 1. Repository layout

```
Cargo.toml                 # Rust workspace (all crates below)
rust-toolchain.toml        # stable
crates/
  furrow/                  # sim core + RL env (lib, no I/O besides content loading)
    data/*.toml            # original content tables (embedded with include_str!)
  furrow-expert/           # scripted speedrun planner (macro-level expert policy)
  furrow-traj/             # trajectory dataset format: writer + reader + replay (Rust)
  furrow-py/               # PyO3 extension module `furrow` (built with maturin)
  furrow-cli/              # `furrow` binary: play / render / bench / demos / replay
  sdv-bridge/              # Rust client for the RLBridge SMAPI mod (+ mock server for tests)
mod/RLBridge/              # C# SMAPI mod for the real game
python/                    # `svrl` training package (pyproject.toml, pure Python + torch)
  svrl/ data.py models.py bc.py dt.py dagger.py iql.py ppo.py evaluate.py pipeline.py envs/ ...
  furrow_mock/             # pure-Python stand-in with the §3.7 surface (tests, NeMo Gym dev)
  nemo_gym_furrow/         # NeMo Gym resources server (§9)
  configs/*.yaml
  tests/
papers/                    # Typst sources (§10)
docs/                      # this file, CONTENT.md, TRAJECTORY_FORMAT.md, BRIDGE_PROTOCOL.md,
                           # SMAPI_NOTES.md, NEMO_GYM_*.md, BASELINES.md, RESULTS.md, reviews/
legacy/                    # the previous HTTP/curl Python agent, kept for reference
```

Toolchain: Rust stable (edition 2021), Python ≥3.11 with `uv`, torch ≥2.4. Rust deps are
declared once in `[workspace.dependencies]`; crates use `dep.workspace = true`.

## 2. Furrow — the simulation (`crates/furrow`)

### 2.1 Time, calendar, weather
- 1 **tick = 1 game minute**. Day runs 06:00 (minute 360) to 02:00 (minute 1560): 1200 ticks.
  The absolute clock (`abs_minute: u32`) also counts the skipped night, so elapsed time is
  well defined across day boundaries.
- 4 seasons (Spring, Summer, Autumn, Winter) × **24 days** = 96-day year. Day index `d ≥ 0`
  counts from Spring 1, Year 1. Weekday = `d % 7` (shops may close on a weekday).
- Weather per day ∈ {Sunny, Rainy, Stormy}(stormy counts as rainy), plus Snowy in Winter
  (counts as sunny for crops). Tomorrow's weather is rolled at day start and is observable
  (forecast). Rain waters all outdoor soil.
- At 02:00 the farmer *passes out*: day ends, wakes with 50% energy and pays a 10% money
  penalty (capped at 1000). Sleeping in bed at any time ends the day normally (full energy).

### 2.2 World
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
Objects (u8 kind + optional payload): `None, Debris{Weeds|Twig|Stone}, Tree{stage}, Stump,
Boulder, Rock{kind}, Forage{item}, Building, Counter{shop}, Bed, Crate, Well, MillBoard,
Ladder, Elevator, MachinePad, Machine{instance}, Warp`.
Soil (farm/greenhouse): `tilled, watered, fertilizer, crop: Option<CropInstance>`.

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
  `Remixed` mode resamples candidate lists from per-request pools using the episode seed. Pools
  are partitioned (`RemixPartition::{Train, Test}`) so a held-out partition exists for OOD
  evaluation. Content validation rejects/resamples a Remixed list whose static critical-path
  lower bound (season windows, growth days, unlock dependencies) exceeds `max_days − 24`.

### 2.5 Determinism and performance
- All randomness comes from per-env PRNGs implemented **in-crate** (`furrow::rng`: SplitMix64
  seeding + PCG32 (XSH-RR) with our own `below(n)`/`range`/`chance(p)` helpers — no `rand`
  crate, because its sampling algorithms change across versions and would silently break
  recorded trajectories). **World randomness** (layout, weather, forage spawns, merchant stock,
  mine floors, fish/forage tables' daily rolls) is drawn from **counter-based streams keyed by
  `(seed, day, purpose[, location, floor])`**, independent of the policy's actions, so every
  policy faces the same world on a given seed (paired evaluation, §6). **Dynamics randomness**
  (fish catch, drops, yields) uses a separate stream. **Expert noise** uses a third, private
  stream. Given `(content version, EnvConfig, seed, action sequence)` the trajectory is
  bit-identical across runs and platforms. No `HashMap` iteration order may affect outcomes
  (use `Vec`/`BTreeMap`). Floating point never decides game outcomes except through the RNG
  helpers (integers/fixed-point for game logic).
- `Env: Clone` — a cheap snapshot whose continuation is bit-identical (tested).
- Hot paths allocate nothing per step. Targets (16-core desktop, release build):
  primitive steps ≥ 2M/s aggregate, macro steps ≥ 20k/s aggregate, `reset` ≤ 50 µs.

## 3. Furrow — the RL environment API

### 3.1 Config
```rust
pub struct EnvConfig {
    pub seed: u64,
    pub action_mode: ActionMode,          // Primitive | Macro      (default Macro)
    pub restoration: RestorationMode,     // Fixed | Remixed        (default Fixed)
    pub remix_partition: RemixPartition,  // Train | Test           (default Train)
    pub max_days: u16,                    // compute cap (truncation), default 192; NOT part of the objective
    pub max_decisions_per_day: u16,       // default 64; afterwards only Sleep is unmasked (macro mode)
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

### 3.2 Core API (Rust, `furrow::Env`)
```rust
impl Env {
    pub fn new(cfg: EnvConfig) -> Env;
    pub fn reset(&mut self, seed: u64);
    pub fn step(&mut self, action: u32) -> Step;   // in cfg.action_mode
    pub fn step_primitive(&mut self, a: Primitive) -> Step; // always available (macros use it)
    pub fn num_actions(&self) -> usize;
    pub fn action_mask(&self, out: &mut [bool]);
    pub fn write_obs(&self, global: &mut [f32], grid: Option<&mut [u8]>);
    pub fn state(&self) -> &GameState;            // read-only full state (macro controllers, tests)
    pub fn render_ascii(&self) -> String;
    pub fn describe(&self) -> String;              // compact deterministic text state (LLM agents)
    pub fn content() -> &'static Content;          // parsed data tables
}
#[derive(Clone, Copy)]
pub struct Step {
    pub reward: f32,
    pub terminated: bool,     // all wings complete
    pub truncated: bool,      // max_days reached
    pub day_ends: u8,         // day boundaries crossed by this transition (0 within a day)
    pub ticks: u32,           // absolute in-game minutes elapsed, including skipped night
    pub macro_failed: bool,   // unmasked macro that failed part-way (its elapsed time is charged)
    pub invalid_action: bool, // a masked action was stepped (treated as Wait 10 min)
}
pub const OBS_GLOBAL_DIM: usize; // see §3.5; a pure function of content
```
`VecEnv::new(n, cfg, seed_range: (u64, u64))` holds N envs, steps them in parallel with rayon
into preallocated buffers, and **auto-resets** a finished env in the same call (the returned
obs is the new episode's first obs). Auto-reset seeds are drawn **only from `seed_range`**.
Per finished episode it reports: `final_obs` (the full observation dict after the last action:
global, grid if enabled, action_mask), `final_day_ends`, `episode_return, episode_days,
episode_ticks, success, slots_filled, requests_done, wings_done, decisions`. **Eval mode**
(`VecEnv::with_seeds(cfg, seeds: &[u64])`) steps an explicit seed list and does not auto-reset.

### 3.3 Primitive action space (Discrete, `furrow::action::Primitive`) — secondary track
| ids | action |
|---|---|
| 0 | Wait (10 minutes) |
| 1–4 | Move N/E/S/W (turn to face; step if passable; 1 tick) |
| 5 | UseTool (selected tool/item on faced tile: till, water, chop, mine, fish, plant seed, place machine) |
| 6 | Interact (faced tile: pick up forage, ship selected stack at crate, open shop at counter, deliver selected at mill board, sleep at bed, ladder, warp, machine load/collect, refill can at well) |
| 7–30 | Select backpack slot 0..23 |
| 31–35 | Select tool Hoe/Can/Axe/Pickaxe/Rod |
| 36 | Eat selected |
| 37 | Open crafting menu |
| 38–53 | Menu option 0..15 (shop: buy stock row k; crafting: craft recipe k; smithy: upgrade) |
| 54 | Close menu |
**Every step advances the absolute clock**: primitives cost ≥1 tick, including Select, Eat,
Menu and Close. Actions invalid in the current state are masked; stepping a masked action is
Wait(1) with `invalid_action = true`. In primitive mode the observation includes the item id
and price of each visible shop-menu row.

### 3.4 Macro action space (Discrete, enumerated at startup from content) — primary track
Every macro is executed by a deterministic controller that issues primitive steps (with
A* over cached per-POI distance fields across the world graph) until done, time-boxed, the day
ends, or it fails. Normative semantics:
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
9. Macros cost **≥ 5 ticks even when they fail** (`macro_failed = true`, elapsed time charged).
   A day accepts at most `max_decisions_per_day` macro decisions; afterwards only `Sleep` is
   unmasked. Stepping a masked macro = Wait(10 min) with `invalid_action = true`. Masks are
   cheap preconditions computed with ETAs (shop open on arrival, affordable, space, inputs held).
`furrow::macro_action::MacroAction` has `fn name(&self) -> String` and a table
`macro_actions() -> &'static [MacroAction]`.

### 3.5 Observations
`write_obs(global, grid)`:
- `global: [f32; OBS_GLOBAL_DIM]` — fixed, documented layout exported as
  `furrow::obs::global_layout() -> Vec<(name, offset, len)>`. **Shapes depend only on content;
  no feature depends on `EnvConfig`; identical meaning in Fixed and Remixed.** Scaling uses
  content constants only. Groups:
  - calendar: day-of-season/24, season one-hot, year/4, weekday one-hot, minute-of-day/1560,
    **days elapsed / 384**, days left in season/24; weather today + forecast one-hots;
  - player: energy/max, `log1p(money)/log1p(1e6)` and `min(money,1e4)/1e4`, location one-hot,
    x/w, y/h, free backpack slots/24, tool levels/3 and upgrade timers/2, skill levels/10,
    current mine floor/60, unlocked-feature flags, unlocked-recipe flags, `last_macro_failed`;
  - **items** (per item id): `held[i] = log1p(count)/8`, `need[i]` = units still required by open
    requests (a k-of-N request counts each candidate until k are filled; /16, clipped),
    `cover[i] = min(held, need)/need`, `n_open_slots[i]/8` (scarcity), `price[i]` (today's best
    buy price, log-scaled, 0 if not purchasable today);
  - **requests/wings**: per request `filled/k`, done flag, remaining coin amount/1e4; per wing done;
  - **ETAs**: `eta[poi]` = travel minutes from the player to every macro-target POI (cached
    per-POI distance fields) /600; `opens_on_arrival[shop]`;
  - farm: free tilled and tillable tiles per field (farm, greenhouse) /max; per crop: planted,
    watered, ready counts (/max tiles), min days-to-ready/24, count maturing before season end;
    machine summary per machine type (idle/working/ready); pending crate value (log);
  - world: forage available per location/cap, merchant stock multi-hot, shop-open flags,
    active-menu one-hot (primitive mode), menu rows' item ids/prices (primitive mode).
- `grid: [u8; C×(2r+1)×(2r+1)]` (only when `cfg.grid`), channel-major, egocentric around the
  player: terrain, object kind, object payload (item/crop id low byte), soil flags
  (tilled|watered<<1|fertilized<<2), crop growth stage, is-player. Out-of-map = Void.
- `action_mask` for the active action mode.

### 3.6 Reward, discounting and termination
Per `RewardConfig` (§3.1). **Discounting is per in-game day, never per decision.** A transition
that crosses `day_ends` day boundaries is discounted by `γ_day^day_ends` (= 1 within a day).
GAE, IQL/TD targets, DT returns and critic warm-up all use this. `terminated` when all wings
complete; `truncated` at `max_days` (a compute cap, not part of the objective; PPO may train
with a shorter cap, bootstrapping from `final_obs`). Info per finished episode as in §3.2.

### 3.7 Python native module `furrow` (crates/furrow-py) — the contract `svrl` codes against
```python
import furrow                       # abi3 wheel built with maturin (pyo3 0.30: GIL released via py.detach)
furrow.__version__; furrow.content_hash() -> int; furrow.content_json() -> str   # all tables as JSON
cfg = dict(action_mode="macro"|"primitive", restoration="fixed"|"remixed", remix_partition="train"|"test",
           max_days=192, max_decisions_per_day=64, grid=None, grid_radius=6, start_money=500,
           reward=dict(per_day=-1.0, completion=0.0, phi_slot=..., phi_request=..., phi_wing=...,
                       shaping=True, gamma_day=0.99))                              # all optional

env = furrow.Env(seed=0, **cfg)
env.reset(seed: int) -> None
env.step(action: int) -> dict(reward=float, terminated=bool, truncated=bool, day_ends=int, ticks=int,
                              macro_failed=bool, invalid_action=bool)
env.obs() -> dict(global_=f32[G], action_mask=bool[A], grid=u8[C,H,W] if cfg.grid)   # key "global_"
env.num_actions: int; env.obs_global_dim: int; env.grid_shape: tuple[int,int,int] | None
env.action_names() -> list[str]; env.global_layout() -> list[tuple[str,int,int]]
env.describe() -> str; env.render() -> str
env.info() -> dict(day=..., minute=..., abs_minute=..., money=..., energy=..., slots_filled=..., total_slots=...,
                   requests_done=..., total_requests=..., wings_done=..., total_wings=..., decisions=...,
                   decisions_today=..., success=bool, episode_return=float)
env.clone() -> Env                                            # bit-identical snapshot
env.expert_action(style: int = 0) -> int                      # clean expert label (macro mode)
env.noisy_expert_action(noise: float, style: int = 0) -> tuple[int, int]   # (executed, clean)
env.expert_cost_to_go(action: int, n_dyn_seeds: int = 4) -> float          # mean completion day (§4)

vec = furrow.VecEnv(num_envs: int, seed_range: tuple[int,int] = (0, 2**32), num_threads: int | None = None, **cfg)
vec = furrow.VecEnv.with_seeds(seeds: list[int], num_threads=None, **cfg)   # eval mode: no auto-reset
vec.reset() -> obs dict with leading N axis
vec.step(actions: np.ndarray[int64|int32, (N,)]) -> (obs: dict, reward: f32[N], terminated: bool[N],
         truncated: bool[N], day_ends: u8[N], ticks: u32[N],
         info: dict(macro_failed=bool[N], invalid_action=bool[N],
                    # rows below valid where terminated|truncated:
                    final_obs=dict(global_=f32[N,G], action_mask=bool[N,A], grid=u8[N,C,H,W] if cfg.grid),
                    final_day_ends=u8[N], episode_return=f32[N], episode_days=i32[N], episode_ticks=i64[N],
                    success=bool[N], slots_filled=i32[N], requests_done=i32[N], wings_done=i32[N], decisions=i32[N]))
vec.expert_actions(styles: np.ndarray[int32,(N,)] | None = None) -> i64[N]                      # clean labels
vec.noisy_expert_actions(noise: float, styles=None) -> tuple[i64[N], i64[N]]                     # (executed, clean)
vec.num_envs / num_actions / obs_global_dim / grid_shape; vec.action_names(); vec.done_mask() -> bool[N]  # eval mode

furrow.generate_demos(out_dir: str, episodes: int, seed_range: tuple[int,int] = (0, 2**32), noise: float = 0.0,
                      styles: list[int] | None = None, record_grid: bool = False, record_obs: bool = True,
                      num_threads: int | None = None, **cfg) -> dict        # writes a furrow-traj dataset (§5)
furrow.replay(dataset_dir: str, out_dir: str, record_grid=False, num_threads=None) -> dict   # regenerate obs cache
furrow.sdv.SdvEnv(addr: str = "127.0.0.1:7777", **cfg)  # same obs()/step()/action_names() surface
```
Auto-reset semantics are those of §3.2. Arrays returned by `VecEnv` are fresh numpy arrays
(copied out of the Rust buffers after the GIL is re-acquired), so callers may keep them.

## 4. Expert planner (`crates/furrow-expert`)
A hand-written, **state-feedback** speedrun policy over macro actions:
`Expert::new(ExpertConfig { style, noise, noise_seed }) ; fn act(&mut self, env: &Env) -> u32`.
- **Information contract.** Decision rules read only the §3.5 observation (f32, before any f16
  quantization) and static content tables, through an `ExpertView` built from `write_obs`
  output. Anything else the expert needs is added to the observation first. Full `GameState`
  access is allowed only inside macro controllers on the env side.
- **Statelessness.** `Expert::new(cfg).act(env)` returns the same action as a long-running
  instance on any state (tested on ≥1e5 states from noise=0.3 and DAgger rollouts).
- **Recovery.** Expert takeover from states at uniformly random days of noise=0.3 rollouts
  succeeds ≥90% of the time (report extra days).
- **Success bar.** With `noise=0` completes the objective on ≥95% of seeds in Fixed *and*
  Remixed; content validation guarantees feasibility (§2.4).
- **Noise** (`noise ∈ [0,1]`, probability of a random valid macro) uses the private noise RNG
  stream; `noisy_act` returns `(executed, clean)`. `style` perturbs priorities so demos have
  diverse speeds; style 0 is canonical.
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
A Furrow dataset is **canonically** `(content_hash, EnvConfig, seed, action sequence)` per
episode; observation arrays are a regenerable cache (`furrow replay --obs-layout <v>`). Human
SDV data cannot be replayed and always keeps full observations. A dataset is a directory:
- `meta.json` — `{format: "furrow-traj/2", env: "furrow"|"sdv", env_version, content_hash,
  env_config, reward_config, gamma_day, seed_range, action_mode, num_actions, obs_global_dim,
  obs_layout_version, grid_shape|null, global_layout, action_names, generator: {...}}`
- `episodes.jsonl` — one line per episode: `{episode, seed, style, shard, start, length,
  return, days, success, generator, actor_mix}`.
- `final_obs/global.npy` `f32[E, G]` (+ `final_obs/action_mask.npy`, `final_obs/grid.npy`),
  aligned with `episodes.jsonl`.
- `shard_00000/` … (~2^20 steps each): **separate uncompressed `.npy` files** (memory-mappable):
  `obs_global f16[T,G]` (optional cache), `obs_grid u8[T,C,H,W]` (optional), `action i32[T]`,
  `expert_action i32[T]` (clean expert label at this state; −1 if unavailable), `actor u8[T]`
  (0 expert, 1 injected noise, 2 learner/DAgger, 3 human), `action_mask u8[T, ceil(A/8)]`
  (bit-packed, little-endian bit order), `reward f32[T]`, `terminated u8[T]`, `truncated u8[T]`,
  `day_ends u8[T]`, `ticks u16[T]`, `macro_failed u8[T]`, `episode u32[T]`, `day u16[T]`,
  `minute u16[T]`, `progress u16[T,3]` (slots, requests, wings after the step),
  `days_to_go f32[T]` (completion_day − day for successful episodes, **NaN** for failures),
  optional `expert_intent u8[T]`, optional `prim_offset u32[T]`/`prim_len u16[T]` (primitive
  expansion of each macro, for hierarchical work).
Readers: Rust (`furrow_traj::Reader`) and Python (`svrl.data`). The SDV recorder produces the
same layout (with `env: "sdv"` and its own layout/action names).

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
  instances, an `SdvEnv` with the same reset/step/obs shape as Furrow, a JSONL.gz →
  furrow-traj converter, and an in-process mock server used by the tests.
- **Human data** (`collect_human`): segments primitive human input into the SDV macro
  vocabulary using game events (warp, purchase, harvest, ship, bundle delivery); writes
  `expert_action` = inferred macro with a `label_confidence`, none for unsegmentable spans.
- Furrow and SDV share the API shape, **not** the meaning of observations/actions: v1 claims no
  cross-environment transfer. SDV finetuning is offline-to-online (BC/IQL init, then a small
  number of online episodes); PPO-scale finetuning in the real game is out of scope.

## 8. Data-collection + training pipeline (`python/svrl/pipeline.py`, `python/configs/pipeline.yaml`)
One reproducible, resumable runner (`python -m svrl.pipeline --config ... [--stages ...]`) whose
stages write into `runs/<name>/` (each stage skips if its outputs exist, `--force` re-runs):
1. `collect_expert` — the canonical datasets of §6 (clean + noisy with clean labels).
2. `collect_human` — (real game) convert RLBridge recordings (`*.jsonl.gz`) → furrow-traj.
3. `train_bc` → 4. `train_dt` (time-conditioned) → 5. `dagger` (≥ 8 rounds with the Rust expert)
   → 6. `train_iql` → 7. `train_ppo` (from scratch, from BC, from DAgger; protocol of §6)
8. `evaluate` — protocol of §6 for all policies + expert + random baselines → `results.json`
9. `report` — figures (matplotlib) + `docs/RESULTS.md` + paper tables (`papers/*/generated/`).
Every stage logs config, git commit, content hash, seeds, wall-clock and metrics (JSONL).

## 9. Gym integrations (`python/svrl/envs/`, `python/nemo_gym_furrow/`)
- **Gymnasium** (Farama, ≥1.0): `svrl.envs.FurrowEnv` (`gymnasium.Env`, Dict obs `{global,
  action_mask[, grid]}`, `Discrete` actions, info carries `day_ends`, `ticks`, `macro_failed`)
  registered as `Furrow-Macro-v1` / `Furrow-Primitive-v1`; `svrl.envs.FurrowVecEnv`
  (`gymnasium.vector.VectorEnv` over the native `VecEnv`) **declares
  `metadata["autoreset_mode"] = gymnasium.vector.AutoresetMode.SAME_STEP`** and exposes
  `info["final_obs"]`/`info["_final_obs"]` per the Gymnasium 1.x convention; `SdvEnv` with the
  same spaces for the real game. `FurrowEnv` passes `gymnasium.utils.env_checker`.
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

## 11. Quality bar
- `cargo test --workspace`, `cargo clippy --workspace -- -D warnings`, `cargo fmt --check`,
  `pytest python/tests` all pass. CI in `.github/workflows/ci.yml`.
- Furrow tests include: determinism (same seed+actions ⇒ same state hash; `Clone` snapshot
  continuation identical); world draws independent of actions (two different policies see the
  same weather/merchant/forage on a seed); content validation (every requested item obtainable
  without circular dependence on its own wing's reward; every recipe input exists; Remixed
  lists feasible); macro semantics (≥ 99% of unmasked macros on noisy-rollout states make
  progress beyond the clock, the rest set `macro_failed`; re-executing the expert's macro ids
  through `Env::step` reproduces its trajectory bit-identically; the canonical expert averages
  ≤ 40 decisions/day); every step advances the clock; expert statelessness, recovery and success
  rate (Fixed and Remixed); GAE targets on a scripted rollout match hand-computed values for
  terminated and truncated endings across an auto-reset boundary; throughput benches.
