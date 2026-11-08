//! Episode-parallel runner (DESIGN.md §1, §3.2): runs whole `(Env, Expert)` episodes on a rayon
//! pool and commits them in episode-id order, so outputs are byte-identical across thread counts.

use std::path::Path;

use furrow::macro_action::macro_actions;
use furrow::vec_env::episode_seed;
use furrow::{Env, EnvConfig};
use furrow_expert::{Expert, ExpertConfig};
use furrow_traj::{DatasetWriter, EpisodeBuf};
use rayon::prelude::*;

/// Generation request.
#[derive(Clone, Debug)]
pub struct GenConfig {
    pub env: EnvConfig,
    pub episodes: u64,
    pub seed_range: (u64, u64),
    /// Expert noise (ppm) — executed action random with this probability; label stays clean.
    pub noise_ppm: u32,
    /// Styles assigned round-robin to episodes.
    pub styles: Vec<u32>,
    pub num_threads: usize,
}

/// Summary of a generated dataset.
#[derive(Clone, Debug, Default)]
pub struct GenSummary {
    pub episodes: u64,
    pub steps: u64,
    pub successes: u64,
    pub mean_days: f64,
    pub mean_return: f64,
}

/// Run one expert episode and record it.
pub fn run_episode(cfg: &EnvConfig, seed: u64, style: u32, noise_ppm: u32) -> EpisodeBuf {
    let mut env = Env::new(EnvConfig { seed, ..*cfg });
    let ex = Expert::new(ExpertConfig {
        style,
        noise_ppm,
        seed: seed ^ 0xE4E4,
    });
    let g = env.obs_global().len();
    let a = env.num_actions();
    let mut ep = EpisodeBuf::new(g, a);
    while !env.is_done() {
        let (clean, intent) = ex.act_with_intent(&env);
        let (exec, _) = if noise_ppm > 0 {
            ex.noisy_act(&env)
        } else {
            (clean, clean)
        };
        let obs = env.obs_global().to_vec();
        let mask = env.action_mask().to_vec();
        let s = env.step(exec);
        let st = env.state();
        let r = &st.restoration;
        ep.push(
            &obs,
            &mask,
            exec as i32,
            clean as i32,
            if exec == clean { 0 } else { 1 },
            s.reward,
            s.terminated,
            s.truncated,
            s.day_ends,
            s.ticks,
            s.outcome as u8,
            st.day,
            st.minute,
            [r.slots_filled, r.requests_done as u16, r.wings_done as u16],
            intent as u8,
        );
    }
    let stats = env.stats();
    ep.final_obs = env.obs_global().to_vec();
    ep.seed = seed;
    ep.style = style;
    ep.noise_ppm = noise_ppm;
    ep.ret = stats.episode_return;
    ep.days = env.state().day;
    ep.success = stats.success;
    ep.step_cap = stats.step_cap;
    ep.final_state_hash = env.stable_hash();
    ep
}

/// Generate an expert dataset into `out_dir`.
pub fn generate(out_dir: &Path, g: &GenConfig) -> std::io::Result<GenSummary> {
    let probe = Env::new(g.env);
    let meta = serde_json::json!({
        "env": "furrow",
        "sim_version": furrow::SIM_VERSION,
        "content_hash": format!("{:016x}", furrow::content::content_hash()),
        "action_mode": format!("{:?}", g.env.action_mode),
        "restoration": format!("{:?}", g.env.restoration),
        "max_days": g.env.max_days,
        "gamma_day": g.env.reward.gamma_day,
        "reward_config": format!("{:?}", g.env.reward),
        "seed_range": [g.seed_range.0, g.seed_range.1],
        "seed_scheme": "episode_seed(base=env.seed, i=episode, k=0, range)",
        "global_layout": furrow::obs::global_layout().iter().map(|(n, o, l)| serde_json::json!([n, o, l])).collect::<Vec<_>>(),
        "action_names": macro_actions().iter().map(|m| m.name()).collect::<Vec<_>>(),
        "generator": {"kind": "expert", "noise_ppm": g.noise_ppm, "styles": g.styles},
    });
    let mut w = DatasetWriter::create(
        out_dir,
        &meta,
        probe.obs_global().len(),
        probe.num_actions(),
    )?;
    let threads = if g.num_threads == 0 {
        std::thread::available_parallelism()
            .map(|x| x.get())
            .unwrap_or(4)
    } else {
        g.num_threads
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("pool");
    let styles = if g.styles.is_empty() {
        vec![0]
    } else {
        g.styles.clone()
    };
    let mut sum = GenSummary::default();
    let chunk = (threads * 4) as u64;
    let mut start = 0u64;
    while start < g.episodes {
        let end = (start + chunk).min(g.episodes);
        let eps: Vec<EpisodeBuf> = pool.install(|| {
            (start..end)
                .into_par_iter()
                .map(|i| {
                    let seed = episode_seed(g.env.seed, i as usize, 0, g.seed_range);
                    run_episode(&g.env, seed, styles[i as usize % styles.len()], g.noise_ppm)
                })
                .collect()
        });
        for e in eps {
            sum.successes += e.success as u64;
            sum.mean_days += e.days as f64;
            sum.mean_return += e.ret as f64;
            w.push(e)?;
        }
        start = end;
    }
    let (eps, steps) = w.finish()?;
    sum.episodes = eps;
    sum.steps = steps;
    if eps > 0 {
        sum.mean_days /= eps as f64;
        sum.mean_return /= eps as f64;
    }
    Ok(sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_is_thread_count_independent() {
        let d1 = tempfile::tempdir().unwrap();
        let d2 = tempfile::tempdir().unwrap();
        let mk = |t| GenConfig {
            env: EnvConfig::default(),
            episodes: 3,
            seed_range: (0, 100),
            noise_ppm: 50_000,
            styles: vec![0, 1],
            num_threads: t,
        };
        let s1 = generate(d1.path(), &mk(1)).unwrap();
        let s2 = generate(d2.path(), &mk(3)).unwrap();
        assert_eq!(s1.steps, s2.steps);
        let a = std::fs::read(d1.path().join("episodes.jsonl")).unwrap();
        let b = std::fs::read(d2.path().join("episodes.jsonl")).unwrap();
        assert_eq!(a, b);
        let shard =
            |d: &std::path::Path| std::fs::read(d.join("shards/w0-00000/action.npy")).unwrap();
        assert_eq!(shard(d1.path()), shard(d2.path()));
    }
}
