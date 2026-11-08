//! Vectorised environments (DESIGN.md §3.2): N `Env`s stepped in parallel with rayon on a pool
//! owned by the `VecEnv`, same-step auto-reset with a thread-count–independent seed schedule.

use crate::env::{Env, EnvConfig};
use crate::rng::splitmix64;

/// Output buffers for one `VecEnv` step (all `N`-major, caller-owned).
#[derive(Clone, Debug, Default)]
pub struct VecOut {
    /// `[N·G]` observations (after auto-reset for finished rows).
    pub obs: Vec<f32>,
    /// `[N·A]` masks.
    pub mask: Vec<bool>,
    /// `[N·C·H·W]` grids (empty unless `cfg.grid`).
    pub grid: Vec<u8>,
    /// Per-env step results.
    pub reward: Vec<f32>,
    pub terminated: Vec<bool>,
    pub truncated: Vec<bool>,
    pub day_ends: Vec<u8>,
    pub ticks: Vec<u32>,
    pub outcome: Vec<u8>,
    /// Terminal observation of finished rows (stale otherwise).
    pub final_obs: Vec<f32>,
    pub final_mask: Vec<bool>,
    /// Finished-episode statistics (valid where `terminated | truncated`).
    pub ep_return: Vec<f32>,
    pub ep_days: Vec<u16>,
    pub ep_success: Vec<bool>,
    pub ep_seed: Vec<u64>,
    pub ep_slots: Vec<u16>,
    pub ep_requests: Vec<u16>,
    pub ep_wings: Vec<u16>,
    pub ep_decisions: Vec<u32>,
}

impl VecOut {
    fn sized(n: usize, g: usize, a: usize, grid: usize) -> VecOut {
        VecOut {
            obs: vec![0.0; n * g],
            mask: vec![false; n * a],
            grid: vec![0; n * grid],
            reward: vec![0.0; n],
            terminated: vec![false; n],
            truncated: vec![false; n],
            day_ends: vec![0; n],
            ticks: vec![0; n],
            outcome: vec![0; n],
            final_obs: vec![0.0; n * g],
            final_mask: vec![false; n * a],
            ep_return: vec![0.0; n],
            ep_days: vec![0; n],
            ep_success: vec![false; n],
            ep_seed: vec![0; n],
            ep_slots: vec![0; n],
            ep_requests: vec![0; n],
            ep_wings: vec![0; n],
            ep_decisions: vec![0; n],
        }
    }
}

/// N environments with auto-reset.
pub struct VecEnv {
    envs: Vec<Env>,
    episode_idx: Vec<u32>,
    seeds: Vec<u64>,
    seed_range: (u64, u64),
    base_seed: u64,
    auto_reset: bool,
    #[cfg(feature = "parallel")]
    pool: rayon::ThreadPool,
    /// Last outputs.
    pub out: VecOut,
}

/// Seed of env `i`'s `k`-th episode within `range` (independent of thread count/order).
pub fn episode_seed(base: u64, i: usize, k: u32, range: (u64, u64)) -> u64 {
    let span = range.1.saturating_sub(range.0).max(1);
    range.0 + splitmix64(splitmix64(base) ^ ((i as u64) << 32 | k as u64)) % span
}

impl VecEnv {
    /// `n` envs whose episodes draw seeds from `seed_range` (auto-reset on).
    pub fn new(cfg: EnvConfig, n: usize, num_threads: usize, seed_range: (u64, u64)) -> VecEnv {
        let seeds: Vec<u64> = (0..n).map(|i| episode_seed(cfg.seed, i, 0, seed_range)).collect();
        Self::build(cfg, seeds, num_threads, seed_range, true)
    }
    /// Evaluation mode: one episode per given seed, no auto-reset.
    pub fn with_seeds(cfg: EnvConfig, seeds: &[u64], num_threads: usize) -> VecEnv {
        Self::build(cfg, seeds.to_vec(), num_threads, (0, u64::MAX), false)
    }
    fn build(cfg: EnvConfig, seeds: Vec<u64>, num_threads: usize, seed_range: (u64, u64), auto_reset: bool) -> VecEnv {
        let n = seeds.len();
        #[cfg(feature = "parallel")]
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(if num_threads == 0 { std::thread::available_parallelism().map(|x| x.get()).unwrap_or(4) } else { num_threads })
            .build()
            .expect("thread pool");
        #[cfg(not(feature = "parallel"))]
        let _ = num_threads;
        let mk = |s: u64| Env::new(EnvConfig { seed: s, ..cfg });
        #[cfg(feature = "parallel")]
        let envs: Vec<Env> = {
            use rayon::prelude::*;
            pool.install(|| seeds.par_iter().map(|s| mk(*s)).collect())
        };
        #[cfg(not(feature = "parallel"))]
        let envs: Vec<Env> = seeds.iter().map(|s| mk(*s)).collect();
        let g = crate::obs::obs_global_dim();
        let a = envs.first().map(|e| e.num_actions()).unwrap_or(0);
        let grid = envs.first().map(|e| e.obs_grid().len()).unwrap_or(0);
        let mut v = VecEnv {
            out: VecOut::sized(n, g, a, grid),
            episode_idx: vec![0; n],
            seeds,
            seed_range,
            base_seed: cfg.seed,
            auto_reset,
            envs,
            #[cfg(feature = "parallel")]
            pool,
        };
        v.write_all_obs();
        v
    }
    /// Number of envs.
    pub fn len(&self) -> usize {
        self.envs.len()
    }
    /// No envs?
    pub fn is_empty(&self) -> bool {
        self.envs.is_empty()
    }
    /// Envs (read-only).
    pub fn envs(&self) -> &[Env] {
        &self.envs
    }
    /// Current seeds per env.
    pub fn seeds(&self) -> &[u64] {
        &self.seeds
    }
    /// Map a function over all envs in parallel on the VecEnv's pool (e.g. expert labelling).
    pub fn map_envs<T: Send, F: Fn(usize, &Env) -> T + Sync + Send>(&self, f: F) -> Vec<T> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            self.pool.install(|| self.envs.par_iter().enumerate().map(|(i, e)| f(i, e)).collect())
        }
        #[cfg(not(feature = "parallel"))]
        {
            self.envs.iter().enumerate().map(|(i, e)| f(i, e)).collect()
        }
    }
    /// Done mask (eval mode).
    pub fn done_mask(&self) -> Vec<bool> {
        self.envs.iter().map(|e| e.is_done()).collect()
    }
    fn write_all_obs(&mut self) {
        let g = crate::obs::obs_global_dim();
        let a = self.envs.first().map(|e| e.num_actions()).unwrap_or(0);
        for (i, e) in self.envs.iter().enumerate() {
            self.out.obs[i * g..(i + 1) * g].copy_from_slice(e.obs_global());
            self.out.mask[i * a..(i + 1) * a].copy_from_slice(e.action_mask());
            let gl = e.obs_grid().len();
            if gl > 0 {
                self.out.grid[i * gl..(i + 1) * gl].copy_from_slice(e.obs_grid());
            }
        }
    }
    /// Reset all envs to fresh episodes (auto-reset mode: next seeds in the schedule).
    pub fn reset(&mut self) {
        for i in 0..self.envs.len() {
            if self.auto_reset {
                let s = episode_seed(self.base_seed, i, self.episode_idx[i], self.seed_range);
                self.seeds[i] = s;
            }
            let s = self.seeds[i];
            self.envs[i].reset(s);
        }
        self.write_all_obs();
    }

    /// Step all envs with `actions` (len N) and fill `self.out`.
    pub fn step(&mut self, actions: &[u32]) -> Result<(), String> {
        let n = self.envs.len();
        if actions.len() != n {
            return Err(format!("expected {n} actions, got {}", actions.len()));
        }
        let g = crate::obs::obs_global_dim();
        let a = self.envs.first().map(|e| e.num_actions()).unwrap_or(0);
        if let Some(bad) = actions.iter().find(|x| **x as usize >= a) {
            return Err(format!("action {bad} out of range (num_actions={a})"));
        }
        let gl = self.envs.first().map(|e| e.obs_grid().len()).unwrap_or(0);
        let auto = self.auto_reset;
        let base = self.base_seed;
        let range = self.seed_range;
        let out = &mut self.out;
        // split output buffers into per-env chunks
        let work = |(i, ((env, ep_idx), seed)): (usize, ((&mut Env, &mut u32), &mut u64)), o: RowOut<'_>| {
            if env.is_done() && !auto {
                // eval mode: finished envs are frozen
                o.obs.copy_from_slice(env.obs_global());
                o.mask.iter_mut().for_each(|m| *m = false);
                *o.reward = 0.0;
                *o.terminated = false;
                *o.truncated = false;
                *o.day_ends = 0;
                *o.ticks = 0;
                *o.outcome = 3;
                return;
            }
            let st = env.step(actions[i]);
            *o.reward = st.reward;
            *o.terminated = st.terminated;
            *o.truncated = st.truncated;
            *o.day_ends = st.day_ends;
            *o.ticks = st.ticks;
            *o.outcome = st.outcome as u8;
            if st.terminated || st.truncated {
                o.final_obs.copy_from_slice(env.obs_global());
                o.final_mask.copy_from_slice(env.action_mask());
                let stats = env.stats();
                let s = env.state();
                *o.ep_return = stats.episode_return;
                *o.ep_days = s.day;
                *o.ep_success = stats.success;
                *o.ep_seed = *seed;
                *o.ep_slots = s.restoration.slots_filled;
                *o.ep_requests = s.restoration.requests_done as u16;
                *o.ep_wings = s.restoration.wings_done as u16;
                *o.ep_decisions = stats.decisions;
                if auto {
                    *ep_idx += 1;
                    *seed = episode_seed(base, i, *ep_idx, range);
                    env.reset(*seed);
                }
            }
            o.obs.copy_from_slice(env.obs_global());
            o.mask.copy_from_slice(env.action_mask());
            if gl > 0 {
                o.grid.copy_from_slice(env.obs_grid());
            }
        };
        let rows = rows(out, n, g, a, gl);
        let items = self.envs.iter_mut().zip(self.episode_idx.iter_mut()).zip(self.seeds.iter_mut()).enumerate();
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let zipped: Vec<_> = items.zip(rows).collect();
            self.pool.install(|| zipped.into_par_iter().with_max_len(1).for_each(|(it, o)| work(it, o)));
        }
        #[cfg(not(feature = "parallel"))]
        for (it, o) in items.zip(rows) {
            work(it, o);
        }
        Ok(())
    }
}

/// Mutable views into one env's row of the output buffers.
struct RowOut<'a> {
    obs: &'a mut [f32],
    mask: &'a mut [bool],
    grid: &'a mut [u8],
    final_obs: &'a mut [f32],
    final_mask: &'a mut [bool],
    reward: &'a mut f32,
    terminated: &'a mut bool,
    truncated: &'a mut bool,
    day_ends: &'a mut u8,
    ticks: &'a mut u32,
    outcome: &'a mut u8,
    ep_return: &'a mut f32,
    ep_days: &'a mut u16,
    ep_success: &'a mut bool,
    ep_seed: &'a mut u64,
    ep_slots: &'a mut u16,
    ep_requests: &'a mut u16,
    ep_wings: &'a mut u16,
    ep_decisions: &'a mut u32,
}

fn rows(o: &mut VecOut, n: usize, g: usize, a: usize, gl: usize) -> Vec<RowOut<'_>> {
    let mut obs = o.obs.chunks_mut(g.max(1));
    let mut mask = o.mask.chunks_mut(a.max(1));
    let mut grid: Vec<&mut [u8]> = if gl > 0 { o.grid.chunks_mut(gl).collect() } else { (0..n).map(|_| <&mut [u8]>::default()).collect() };
    let mut fo = o.final_obs.chunks_mut(g.max(1));
    let mut fm = o.final_mask.chunks_mut(a.max(1));
    let mut r = o.reward.iter_mut();
    let mut te = o.terminated.iter_mut();
    let mut tr = o.truncated.iter_mut();
    let mut de = o.day_ends.iter_mut();
    let mut ti = o.ticks.iter_mut();
    let mut oc = o.outcome.iter_mut();
    let mut er = o.ep_return.iter_mut();
    let mut ed = o.ep_days.iter_mut();
    let mut es = o.ep_success.iter_mut();
    let mut esd = o.ep_seed.iter_mut();
    let mut esl = o.ep_slots.iter_mut();
    let mut erq = o.ep_requests.iter_mut();
    let mut ew = o.ep_wings.iter_mut();
    let mut edc = o.ep_decisions.iter_mut();
    let mut grid_it = grid.drain(..);
    (0..n)
        .map(|_| RowOut {
            obs: obs.next().unwrap(),
            mask: mask.next().unwrap(),
            grid: grid_it.next().unwrap(),
            final_obs: fo.next().unwrap(),
            final_mask: fm.next().unwrap(),
            reward: r.next().unwrap(),
            terminated: te.next().unwrap(),
            truncated: tr.next().unwrap(),
            day_ends: de.next().unwrap(),
            ticks: ti.next().unwrap(),
            outcome: oc.next().unwrap(),
            ep_return: er.next().unwrap(),
            ep_days: ed.next().unwrap(),
            ep_success: es.next().unwrap(),
            ep_seed: esd.next().unwrap(),
            ep_slots: esl.next().unwrap(),
            ep_requests: erq.next().unwrap(),
            ep_wings: ew.next().unwrap(),
            ep_decisions: edc.next().unwrap(),
        })
        .collect()
}
