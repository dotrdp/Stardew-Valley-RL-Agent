# Design contract — speedrun agents learned from trajectories

Status: **v1 contract**. Every crate/package in this repo is implemented against this document.
If an implementation needs to deviate, update this file in the same change and say why.

## 0. Goal and ground rules

**Objective.** Train a model, primarily from trajectories (demonstrations), to *speedrun a
long-horizon restoration objective*: complete every delivery request of a multi-wing building
restoration project (the analogue of finishing a farming game's "community center") in as few
in-game days as possible.

Two environments share one agent interface:

| | **Furrow** (`crates/furrow*`) | **SDV bridge** (`mod/RLBridge`, `crates/sdv-bridge`) |
|---|---|---|
| What | An *original* farming/life-sim written from scratch in Rust | A SMAPI mod + Rust client that drives the user's own installed copy of Stardew Valley |
| Speed | 10⁵–10⁶ steps/s (vectorised, headless) | real-time-bound (lockstep, ~10²–10³ ticks/s) |
| Objective | Restore the Old Mill (original content) | Complete the Community Center (read from the live game) |
| Use | Scalable data generation, imitation + RL training, benchmarking | Recording human trajectories, evaluating/finetuning in the real game |

**Originality rule (hard).** Furrow is an original game in the farming-sim genre. It MUST NOT
contain Stardew Valley code, decompiled code, data, maps, art, character/NPC/place/item names
specific to Stardew Valley, or its specific numeric tables. Genre mechanics (seasons, crops,
tools, foraging, fishing, mining, shops, crafting, a collection-based restoration goal) are fine;
all names, numbers, maps, and request lists are designed fresh (see `crates/furrow/data/`).
The SDV bridge never ships game code or data: it reads/writes the running game through SMAPI's
public modding API at runtime, on the user's own licensed copy.

## 1. Repository layout

```
Cargo.toml                 # Rust workspace (all crates below)
rust-toolchain.toml        # stable
crates/
  furrow/                  # sim core + RL env (lib, no I/O besides content loading)
    data/*.toml            # original content tables (embedded with include_str!)
  furrow-expert/           # scripted speedrun planner (macro-level expert policy)
  furrow-traj/             # trajectory dataset format: writer + reader (Rust)
  furrow-py/               # PyO3 extension module `furrow` (built with maturin)
  furrow-cli/              # `furrow` binary: play / render / bench / demos / replay
  sdv-bridge/              # Rust client for the RLBridge SMAPI mod (+ mock server for tests)
mod/RLBridge/              # C# SMAPI mod for the real game
python/                    # `svrl` training package (pyproject.toml, pure Python + torch)
  svrl/ data.py models.py bc.py dt.py ppo.py evaluate.py envs/ ...
  configs/*.yaml
  tests/
docs/                      # this file, ENV.md, TRAJECTORY_FORMAT.md, BRIDGE_PROTOCOL.md, RESULTS.md
legacy/                    # the previous HTTP/curl Python agent, kept for reference
```

Toolchain: Rust stable (edition 2021), Python ≥3.11 with `uv`, torch ≥2.4. Rust deps are
declared once in `[workspace.dependencies]`; crates use `dep.workspace = true`.

## 2. Furrow — the simulation (`crates/furrow`)

### 2.1 Time, calendar, weather
- 1 **tick = 1 game minute**. Day runs 06:00 (minute 360) to 02:00 (minute 1560): 1200 ticks.
- 4 seasons (Spring, Summer, Autumn, Winter) × **24 days** = 96-day year. Day index `d ≥ 0`
  counts from Spring 1, Year 1. Weekday = `d % 7` (shops may close on a weekday).
- Weather per day ∈ {Sunny, Rainy, Stormy}(stormy counts as rainy), plus Snowy in Winter
  (counts as sunny for crops). Tomorrow's weather is rolled at day start and is observable
  (forecast). Rain waters all outdoor soil.
- At 02:00 the farmer *passes out*: day ends, wakes with 50% energy and pays a 10% money
  penalty (capped at 1000). Sleeping in bed at any time ends the day normally (full energy).

### 2.2 World
Locations are tile grids with terrain, objects, and (farm/greenhouse only) soil. All maps are
**original and procedurally laid out from a seed** on top of a fixed topology:

| id | Location | Size (w×h) | Contents |
|---|---|---|---|
| 0 | Homestead (farm) | 48×36 | bed, shipping crate, well, tillable field, debris, small pond |
| 1 | Village | 44×32 | shops (general store, smithy, fishmonger, carpenter), Old Mill board, merchant stall |
| 2 | Woods | 40×32 | forage spawns, trees, hardwood stumps, river (fishing) |
| 3 | Coast | 40×24 | beach forage, tide pools, sea (fishing) |
| 4 | Quarry | 32×24 | mine entrance (+ elevator), rock nodes, mountain lake (fishing) |
| 5 | Mines | 24×24 per floor | floors 1..=60, procedurally generated per (seed, day, floor) |
| 6 | Greenhouse | 16×12 | locked until unlocked by a restoration reward; plant any season |

Locations connect through edge/door **warps** (a world graph). Each location exposes named
**points of interest** (POIs, e.g. `bed`, `crate`, `store_counter`, `mill_board`, `river_spot`)
used by macro actions and by the observation.

Terrain (u8): `Void, Grass, Dirt(tillable), Path, Sand, Floor, Water, Wall, Bridge`.
Objects (u8 kind + optional payload): `None, Debris{Weeds|Twig|Stone}, Tree{stage}, Stump,
Boulder, Rock{kind}, Forage{item}, Building, Counter{shop}, Bed, Crate, Well, MillBoard,
Ladder, Elevator, Machine{instance}, Warp`.
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
  items (including off-season items) per visit — routing RNG.
- **Shipping**: items placed in the crate are paid at their sell price overnight.
- **Crafting**: recipes (`data/recipes.toml`) turn inputs into items/machines; unlocked by
  skill levels or rewards. Machines are placed on the farm: load input → output after a
  duration (minutes or days) → collect.
- **Eating**: edible items restore energy.
- **Restoration (the objective)**: the Old Mill has **wings**; each wing has **requests**; each
  request lists candidate slots `(item, qty)` and needs `k` of them (or a coin amount). Items are
  delivered at the Mill board. Completing a wing grants a reward (item, money, or **feature
  unlock**: greenhouse, minecart fast-travel, etc.). **Episode success = all wings complete.**
  Request lists are fixed in `data/restoration.toml` (`RestorationMode::Fixed`); the
  `Remixed` mode resamples candidate lists from per-request pools using the episode seed.

### 2.5 Determinism and performance
- All randomness comes from per-env PRNGs implemented **in-crate** (`furrow::rng`: SplitMix64
  seeding + PCG32 (XSH-RR) with our own `below(n)`/`range`/`chance(p)` helpers — no `rand`
  crate, because its sampling algorithms change across versions and would silently break
  recorded trajectories). Streams: *world* (layout, merchant stock, weather) and *dynamics*,
  both derived from the episode seed. Given `(content version, EnvConfig, seed, action
  sequence)` the trajectory is bit-identical across runs and platforms. No `HashMap` iteration
  order may affect outcomes (use `Vec`/`BTreeMap`). Floating point never decides game
  outcomes except through the RNG helpers (integers/fixed-point for game logic).
- Hot paths allocate nothing per step. Targets (16-core desktop, release build):
  primitive steps ≥ 2M/s aggregate, macro steps ≥ 20k/s aggregate, `reset` ≤ 50 µs.

## 3. Furrow — the RL environment API

### 3.1 Config
```rust
pub struct EnvConfig {
    pub seed: u64,
    pub action_mode: ActionMode,          // Primitive | Macro      (default Macro)
    pub restoration: RestorationMode,     // Fixed | Remixed        (default Fixed)
    pub max_days: u16,                    // truncation, default 192 (two years)
    pub grid_radius: u8,                  // egocentric view radius, default 6 → 13×13
    pub reward: RewardConfig,
    pub start_money: u32,                 // default 500
}
pub struct RewardConfig {
    pub per_slot: f32,        // default 1.0   (each request slot filled)
    pub per_request: f32,     // default 2.0
    pub per_wing: f32,        // default 5.0
    pub completion: f32,      // default 50.0
    pub per_day: f32,         // default -0.25 (speedrun pressure; applied at each day end)
    pub shaping: bool,        // default false (potential-based: needed items held)
}
```

### 3.2 Core API (Rust, `furrow::Env`)
```rust
impl Env {
    pub fn new(cfg: EnvConfig) -> Env;
    pub fn reset(&mut self, seed: u64);
    pub fn step(&mut self, action: u32) -> Step;   // in cfg.action_mode
    pub fn step_primitive(&mut self, a: Primitive) -> Step; // always available (macros use it)
    pub fn num_actions(&self) -> usize;
    pub fn action_mask(&self, out: &mut [bool]);
    pub fn write_obs(&self, global: &mut [f32], grid: &mut [u8]);
    pub fn state(&self) -> &GameState;            // read-only full state (expert, tests)
    pub fn render_ascii(&self) -> String;
    pub fn content() -> &'static Content;          // parsed data tables
}
pub struct Step { pub reward: f32, pub terminated: bool, pub truncated: bool, pub ticks: u32 }
pub const OBS_GLOBAL_DIM: usize; // see §3.5
```
`VecEnv` holds N envs, steps them in parallel with rayon into preallocated buffers, and
**auto-resets** a finished env in the same call (the returned obs is the new episode's first
obs; the finished episode's stats are reported in the info buffers, and its last global
observation in `final_global`).

### 3.3 Primitive action space (Discrete, `furrow::action::Primitive`)
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
Actions invalid in the current state are masked; stepping a masked action is a no-op Wait(1).

### 3.4 Macro action space (Discrete, enumerated at startup from content)
Every macro is executed by a deterministic controller that issues primitive steps (with
A* pathfinding across the world graph) until done, the day ends, or it fails. Macros that
cannot make progress are masked. The enumeration (ids stable for a content version) is:
`Wait30`, `Sleep`, `GoTo(location)`×7, `WaterCrops`, `Harvest`, `ClearField`,
`Plant(seed)`×|seeds|, `Forage(location)`×|forage locations|, `Fish(spot)`×|fishing spots|
(fish 60 min), `ChopWood`, `ChopHardwood`, `MineDescend` (mine ~2 h going deeper),
`MineFloor` (clear current floor), `Buy(shop_item)`×|stock rows| (buys one pack),
`ShipSurplus` (ship everything not reserved for open requests or needed as seeds/inputs),
`Craft(recipe)`×|recipes|, `LoadMachines`, `CollectMachines`, `UpgradeTool(tool)`×5,
`Deliver` (deliver every held item that fits an open slot), `Eat`.
`furrow::macro_action::MacroAction` has `fn name(&self) -> String` and a table
`macro_actions() -> &'static [MacroAction]`.

### 3.5 Observations
`write_obs(global, grid)`:
- `global: [f32; OBS_GLOBAL_DIM]` — a fixed, documented layout (exported as
  `furrow::obs::global_layout() -> Vec<(name, offset, len)>`), all features scaled to ~[0,1]:
  calendar (day-of-season, season one-hot, year, weekday one-hot, minute-of-day), weather today
  + forecast one-hots, energy, log-money, location one-hot, x/y, tool levels and upgrade
  timers, skill levels, backpack counts per item id (log1p/8), restoration state (per slot
  delivered flag, per request/wing done flags, request→item requirement multi-hot), farm summary
  per crop (planted / watered / ready / mean days-to-ready), machine summary, forage available
  per location, deepest mine floor, unlocked features, shop-open flags, merchant stock
  multi-hot, active-menu one-hot, days elapsed / max_days.
- `grid: [u8; C×(2r+1)×(2r+1)]`, channel-major, egocentric around the player, channels:
  terrain, object kind, object payload (item/crop id low byte), soil flags
  (tilled|watered<<1|fertilized<<2), crop growth stage, is-player. Out-of-map = Void.
- `action_mask` for the active action mode.

### 3.6 Reward and termination
Per `RewardConfig` (§3.1). `terminated` when all wings complete; `truncated` at `max_days`.
Info per finished episode: `episode_return, episode_days, episode_ticks, success,
slots_filled, requests_done, wings_done`.

## 4. Expert planner (`crates/furrow-expert`)
A hand-written, **state-feedback** speedrun policy over macro actions:
`Expert::new(ExpertConfig { seed, noise, style }) ; fn act(&mut self, env: &Env) -> u32`.
It reads `env.state()` (full information) and decides from the current state only (plus
small caches that are pure functions of state), so it can be **queried on learner-visited
states (DAgger)**. Strategy: maintain the set of still-needed items; each day prioritise
(1) harvest/water, (2) deliver, (3) time-gated sources (weather/time-window fish, seasonal
forage, merchant), (4) planting needed crops early enough to mature in-season, (5) machines,
(6) tool upgrades that unlock needed items, (7) money for coin requests and seeds via the best
profit crop, (8) sleep before passing out. `noise ∈ [0,1]` = probability of a random valid
macro; `style` perturbs priorities so demos have diverse speeds. Must complete the Fixed
objective with `noise=0` on ≥95% of seeds.

## 5. Trajectory format (`crates/furrow-traj`, `docs/TRAJECTORY_FORMAT.md`)
A dataset is a directory:
- `meta.json` — `{format: "furrow-traj/1", env: "furrow"|"sdv", env_version, content_hash,
  action_mode, num_actions, obs_global_dim, grid_shape|null, global_layout, action_names,
  generator: {...}}`
- `episodes.jsonl` — one line per episode: `{episode, seed, shard, start, length, return,
  days, success, generator}`.
- `shard_00000.npz`, … (~2^20 steps each) — arrays: `obs_global f16[T,G]`, `obs_grid u8[T,C,H,W]`
  (optional), `action i32[T]`, `action_mask u8[T, ceil(A/8)]` (bit-packed, little-endian bit
  order), `reward f32[T]`, `done u8[T]`, `episode u32[T]`, `day u16[T]`, `minute u16[T]`,
  `days_to_go f32[T]` (episode total days − current day; for time-conditioned imitation).
Readers: Rust (`furrow_traj::Reader`) and Python (`svrl.data`). The SDV recorder produces the
same layout (with `env: "sdv"` and its own layout/action names).

## 6. Learning (`python/svrl`)
- **Data**: memory-mapped shards; uniform step sampling (MLP) or windowed sequences (transformer).
- **Models**: `MLPPolicy` (obs → masked logits, value) and `SpeedrunTransformer` — a causal
  transformer over the last K steps with tokens `[days_to_go, obs, prev_action]`; masked action
  logits + value head. Conditioning on a small `days_to_go` at test time asks for a speedrun.
- **Algorithms**: BC (cross-entropy), time-conditioned BC/DT, DAgger with the Rust expert,
  PPO finetuning from the imitation policy with a KL penalty to it (`ppo.py`).
- **Evaluation** (`evaluate.py`): success rate and days-to-complete distribution on held-out
  seeds vs. expert and random baselines; writes `docs/RESULTS.md` tables.
- GPU etiquette: respect `--max-vram-gb` (default 6) via `torch.cuda.set_per_process_memory_fraction`.

## 7. Real game: SDV bridge
- **Mod** (`mod/RLBridge`, C#, SMAPI 4 / game 1.6): TCP server (default `127.0.0.1:7777`),
  newline-delimited JSON. **Lockstep**: when an agent is attached the mod gates the game update
  loop (Harmony prefix on the game's update) and advances exactly the requested number of ticks
  per `step`; human-play recording mode runs free. Observation = structured JSON snapshot
  (calendar, weather, player, inventory, location + local tile grid, Community Center bundle
  state, money, skills, tools). Actions: primitive input frames (move/use/interact/select)
  and macros (`goto`, `harvest`, `water`, `buy`, `deliver_bundle`, `sleep`…) executed with the
  game's own path-finding and APIs where possible. Recorder: writes JSONL.gz of
  `(snapshot, input, events)` for human runs. Full spec: `docs/BRIDGE_PROTOCOL.md`.
- **Rust client** (`crates/sdv-bridge`): typed protocol, connection pool for several game
  instances, an `SdvEnv` with the same reset/step/obs shape as Furrow, a JSONL.gz →
  furrow-traj converter, and an in-process mock server used by the tests.

## 8. Quality bar
- `cargo test --workspace`, `cargo clippy --workspace -- -D warnings`, `cargo fmt --check`,
  `pytest python/tests` all pass. CI in `.github/workflows/ci.yml`.
- Furrow tests include: determinism (same seed+actions ⇒ same hash of state), content
  validation (every requested item is obtainable; every recipe input exists), macro progress
  (each macro either changes state or is masked), expert success rate, throughput benches.
