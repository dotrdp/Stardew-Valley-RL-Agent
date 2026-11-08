//! Furrow — an original farming-sim reinforcement-learning environment.
//!
//! See `docs/DESIGN.md` for the contract. Module map:
//! - [`content`]: original data tables (embedded TOML) → dense id-indexed tables.
//! - [`rng`], [`hash`]: deterministic randomness and stable hashing.
//! - [`world`]: tile grids, procedural layout, pathfinding.
//! - [`state`]: the full game state and its systems (calendar, farming, fishing, mining, shops…).
//! - [`action`]: primitive actions; [`macro_action`]: macro actions and their controllers.
//! - [`obs`]: observation encoder; [`env`]: `Env`/`Step`/`EnvConfig`; [`vec_env`]: vectorised envs.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod content;
pub mod hash;
pub mod rng;

/// Simulation version. Bump on any change to dynamics, controllers, pathfinding, RNG draw order
/// or observation encoding (goldens must be re-blessed together with a bump).
pub const SIM_VERSION: u32 = 1;
