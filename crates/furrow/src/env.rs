//! The RL environment: `EnvConfig`, `RewardConfig`, `Env`, `Step` (DESIGN.md §3).

use crate::action::{Primitive, NUM_PRIMITIVES};
use crate::content::content;
use crate::macro_action::{macro_actions, num_macros, run_macro, Macro, Outcome};
use crate::nav::NavTables;
use crate::obs::{obs_global_dim, GRID_CHANNELS};
use crate::state::GameState;

/// Action space mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ActionMode {
    Primitive,
    #[default]
    Macro,
}

/// Restoration list mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RestorationMode {
    #[default]
    Fixed,
    Remixed,
}

/// Remixed pool partition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RemixPartition {
    #[default]
    Train,
    Test,
}

/// Speed-only reward with potential-based progress shaping, discounted per in-game day.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RewardConfig {
    pub per_day: f32,
    pub completion: f32,
    pub phi_slot: f32,
    pub phi_request: f32,
    pub phi_wing: f32,
    pub shaping: bool,
    pub gamma_day: f32,
}

impl Default for RewardConfig {
    fn default() -> Self {
        RewardConfig {
            per_day: -1.0,
            completion: 0.0,
            phi_slot: 0.25,
            phi_request: 0.5,
            phi_wing: 2.0,
            shaping: true,
            gamma_day: 0.99,
        }
    }
}

/// Environment configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnvConfig {
    pub seed: u64,
    pub action_mode: ActionMode,
    pub restoration: RestorationMode,
    pub remix_partition: RemixPartition,
    pub max_days: u16,
    pub max_steps: u32,
    pub max_decisions_per_day: u16,
    pub macro_max_minutes: u16,
    pub grid: bool,
    pub grid_radius: u8,
    pub reward: RewardConfig,
    pub start_money: u32,
}

impl Default for EnvConfig {
    fn default() -> Self {
        EnvConfig {
            seed: 0,
            action_mode: ActionMode::Macro,
            restoration: RestorationMode::Fixed,
            remix_partition: RemixPartition::Train,
            max_days: 192,
            max_steps: 20_000,
            max_decisions_per_day: 64,
            macro_max_minutes: 120,
            grid: false,
            grid_radius: 6,
            reward: RewardConfig::default(),
            start_money: content().world.start_money,
        }
    }
}

impl EnvConfig {
    /// Default `max_steps` for a mode if the caller passes 0.
    pub fn normalized(mut self) -> Self {
        if self.max_steps == 0 {
            self.max_steps = match self.action_mode {
                ActionMode::Primitive => 100_000,
                ActionMode::Macro => 20_000,
            };
        }
        self
    }
}

/// Result of one environment step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Step {
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
    pub day_ends: u8,
    pub ticks: u32,
    pub outcome: Outcome,
}

/// Per-episode statistics.
#[derive(Clone, Copy, Debug, Default)]
pub struct EpisodeStats {
    pub episode_return: f32,
    pub decisions: u32,
    pub success: bool,
    pub step_cap: bool,
}

/// The environment.
#[derive(Clone, Debug)]
pub struct Env {
    cfg: EnvConfig,
    state: GameState,
    nav: NavTables,
    stats: EpisodeStats,
    done: bool,
    obs_cache: Vec<f32>,
    mask_cache: Vec<bool>,
    grid_cache: Vec<u8>,
    cache_valid: bool,
}

impl Env {
    /// Create an environment and reset it with `cfg.seed`.
    pub fn new(cfg: EnvConfig) -> Env {
        let cfg = cfg.normalized();
        let state = GameState::new(
            cfg.seed,
            cfg.restoration == RestorationMode::Remixed,
            cfg.remix_partition == RemixPartition::Test,
            cfg.start_money,
        );
        let nav = NavTables::build(&state);
        let side = (2 * cfg.grid_radius as usize + 1) * (2 * cfg.grid_radius as usize + 1);
        let mut env = Env {
            cfg,
            state,
            nav,
            stats: EpisodeStats::default(),
            done: false,
            obs_cache: vec![0.0; obs_global_dim()],
            mask_cache: vec![false; Self::actions_for(cfg.action_mode)],
            grid_cache: vec![0; GRID_CHANNELS * side],
            cache_valid: false,
        };
        env.refresh();
        env
    }
    fn actions_for(mode: ActionMode) -> usize {
        match mode {
            ActionMode::Primitive => NUM_PRIMITIVES,
            ActionMode::Macro => num_macros(),
        }
    }
    /// Configuration.
    pub fn cfg(&self) -> &EnvConfig {
        &self.cfg
    }
    /// Start a new episode.
    pub fn reset(&mut self, seed: u64) {
        self.cfg.seed = seed;
        self.state = GameState::new(
            seed,
            self.cfg.restoration == RestorationMode::Remixed,
            self.cfg.remix_partition == RemixPartition::Test,
            self.cfg.start_money,
        );
        self.nav = NavTables::build(&self.state);
        self.stats = EpisodeStats::default();
        self.done = false;
        self.refresh();
    }
    /// Number of actions in the configured mode.
    pub fn num_actions(&self) -> usize {
        self.mask_cache.len()
    }
    /// Read-only state.
    pub fn state(&self) -> &GameState {
        &self.state
    }
    /// Mutable state (tests/expert cost-to-go only).
    pub fn state_mut(&mut self) -> &mut GameState {
        self.cache_valid = false;
        &mut self.state
    }
    /// Navigation tables.
    pub fn nav(&self) -> &NavTables {
        &self.nav
    }
    /// Episode statistics so far.
    pub fn stats(&self) -> EpisodeStats {
        self.stats
    }
    /// Episode finished (terminated or truncated)?
    pub fn is_done(&self) -> bool {
        self.done
    }

    fn potential(&self) -> f32 {
        let r = &self.cfg.reward;
        let s = &self.state.restoration;
        r.phi_slot * s.slots_filled as f32 + r.phi_request * s.requests_done as f32 + r.phi_wing * s.wings_done as f32
    }

    fn refresh(&mut self) {
        let mut obs = std::mem::take(&mut self.obs_cache);
        self.state.write_global(&self.nav, &mut obs);
        self.obs_cache = obs;
        let mut mask = std::mem::take(&mut self.mask_cache);
        match self.cfg.action_mode {
            ActionMode::Macro => self.state.macro_mask(&self.nav, self.cfg.max_decisions_per_day, &mut mask),
            ActionMode::Primitive => {
                for (i, m) in mask.iter_mut().enumerate() {
                    *m = Primitive::from_id(i as u32).map(|p| self.state.primitive_valid(p)).unwrap_or(false);
                }
            }
        }
        if self.done {
            mask.iter_mut().for_each(|m| *m = false);
        }
        self.mask_cache = mask;
        if self.cfg.grid {
            let mut gr = std::mem::take(&mut self.grid_cache);
            self.state.write_grid(self.cfg.grid_radius, &mut gr);
            self.grid_cache = gr;
        }
        self.cache_valid = true;
    }

    /// Step with an action id in the configured mode.
    pub fn step(&mut self, action: u32) -> Step {
        if self.done {
            return Step {
                reward: 0.0,
                terminated: self.stats.success,
                truncated: !self.stats.success,
                day_ends: 0,
                ticks: 0,
                outcome: Outcome::Masked,
            };
        }
        let phi0 = self.potential();
        let abs0 = self.state.abs_minute;
        let day0 = self.state.day;
        let (_ticks, _day_ends, outcome) = match self.cfg.action_mode {
            ActionMode::Macro => {
                let table = macro_actions();
                let m: Macro = table.get(action as usize).copied().unwrap_or(Macro::Wait30);
                let r = run_macro(&mut self.state, &self.nav, m, self.cfg.macro_max_minutes as u32, self.cfg.max_decisions_per_day);
                (r.ticks, r.day_ends, r.outcome)
            }
            ActionMode::Primitive => {
                let p = Primitive::from_id(action).unwrap_or(Primitive::Wait);
                let valid = self.state.primitive_valid(p);
                let r = self.state.step_primitive(p);
                (r.ticks, r.day_ends, if !valid { Outcome::Masked } else if r.ok { Outcome::Done } else { Outcome::Failed })
            }
        };
        let ticks = self.state.abs_minute - abs0;
        let day_ends = (self.state.day - day0) as u8;
        self.stats.decisions += 1;
        let terminated = self.state.complete;
        let truncated = !terminated && (self.state.day >= self.cfg.max_days || self.stats.decisions >= self.cfg.max_steps);
        let rc = self.cfg.reward;
        let mut reward = rc.per_day * day_ends as f32;
        if terminated {
            reward += rc.completion;
        }
        if rc.shaping {
            let gamma = rc.gamma_day.powi(day_ends as i32);
            reward += gamma * self.potential() - phi0;
        }
        self.stats.episode_return += reward;
        if terminated || truncated {
            self.done = true;
            self.stats.success = terminated;
            self.stats.step_cap = !terminated && self.stats.decisions >= self.cfg.max_steps;
        }
        self.refresh();
        Step {
            reward,
            terminated,
            truncated,
            day_ends,
            ticks,
            outcome,
        }
    }

    /// Step a primitive regardless of mode (used by tests and the macro-expansion checks).
    pub fn step_primitive(&mut self, p: Primitive) -> Step {
        let r = self.state.step_primitive(p);
        self.refresh();
        Step {
            reward: 0.0,
            terminated: self.state.complete,
            truncated: false,
            day_ends: r.day_ends,
            ticks: r.ticks,
            outcome: if r.ok { Outcome::Done } else { Outcome::Failed },
        }
    }

    /// Current action mask.
    pub fn action_mask(&self) -> &[bool] {
        &self.mask_cache
    }
    /// Current global observation.
    pub fn obs_global(&self) -> &[f32] {
        &self.obs_cache
    }
    /// Current grid observation (empty if `cfg.grid` is false).
    pub fn obs_grid(&self) -> &[u8] {
        if self.cfg.grid {
            &self.grid_cache
        } else {
            &[]
        }
    }
    /// Grid shape `(C, H, W)`.
    pub fn grid_shape(&self) -> (usize, usize, usize) {
        let s = 2 * self.cfg.grid_radius as usize + 1;
        (GRID_CHANNELS, s, s)
    }
    /// Copy observations into caller buffers.
    pub fn write_obs(&self, global: &mut [f32], grid: Option<&mut [u8]>) {
        global[..self.obs_cache.len()].copy_from_slice(&self.obs_cache);
        if let Some(g) = grid {
            if self.cfg.grid {
                g[..self.grid_cache.len()].copy_from_slice(&self.grid_cache);
            }
        }
    }
    /// Stable state hash.
    pub fn stable_hash(&self) -> u64 {
        self.state.stable_hash()
    }
    /// Compact text description of the state (for LLM agents).
    pub fn describe(&self) -> String {
        self.state.describe()
    }
    /// ASCII map of the current location.
    pub fn render_ascii(&self) -> String {
        self.state.render_ascii()
    }
    /// Names of the currently valid actions.
    pub fn valid_action_names(&self) -> Vec<String> {
        let names = self.action_names();
        names.into_iter().zip(self.mask_cache.iter()).filter(|(_, m)| **m).map(|(n, _)| n).collect()
    }
    /// Action id by name (exact match).
    pub fn action_id(&self, name: &str) -> Option<u32> {
        self.action_names().iter().position(|n| n == name).map(|p| p as u32)
    }
    /// Action names in the configured mode.
    pub fn action_names(&self) -> Vec<String> {
        match self.cfg.action_mode {
            ActionMode::Macro => macro_actions().iter().map(|m| m.name()).collect(),
            ActionMode::Primitive => (0..NUM_PRIMITIVES as u32).map(|i| Primitive::from_id(i).unwrap().name()).collect(),
        }
    }
}
