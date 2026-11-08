#![allow(clippy::needless_range_loop)]
//! Run the expert on a range of seeds and print per-episode results + a per-day trace for one seed.
use furrow::{Env, EnvConfig};
use furrow_expert::{Expert, ExpertConfig};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5);
    let trace_seed: Option<u64> = args.get(2).and_then(|s| s.parse().ok());
    let c = furrow::content::content();
    let mut days = Vec::new();
    let mut succ = 0;
    for seed in 0..n {
        let mut env = Env::new(EnvConfig {
            seed,
            ..Default::default()
        });
        let ex = Expert::new(ExpertConfig::default());
        let mut intents = [0u32; 16];
        let mut last_day = 0;
        let mut fails = 0u32;
        let mut done_day = vec![0u16; c.requests.len()];
        let mut fail_by: std::collections::BTreeMap<String, u32> = Default::default();
        let names = env.action_names();
        while !env.is_done() {
            let (a, it) = ex.act_with_intent(&env);
            intents[it as usize] += 1;
            let s = env.step(a);
            if s.outcome == furrow::macro_action::Outcome::Failed {
                fails += 1;
                *fail_by.entry(names[a as usize].clone()).or_default() += 1;
            }
            for r in 0..c.requests.len() {
                if done_day[r] == 0 && env.state().restoration.request_done[r] {
                    done_day[r] = env.state().day;
                }
            }
            if Some(seed) == trace_seed && (env.state().day != last_day || env.is_done()) {
                let st = env.state();
                println!(
                    "day {:3} {:?} money {:6} energy {:4} slots {:3}/{} req {:2}/{} wings {}/{} deepest {:2} tools {:?} skills {:?}",
                    st.day, st.season(), st.player.money, st.player.energy, st.restoration.slots_filled, c.total_slots,
                    st.restoration.requests_done, c.requests.len(), st.restoration.wings_done, c.wings.len(), st.mine.deepest,
                    st.player.tool_level, [st.level(furrow::content::Skill::Farming), st.level(furrow::content::Skill::Foraging), st.level(furrow::content::Skill::Fishing), st.level(furrow::content::Skill::Mining)]
                );
                last_day = st.day;
            }
        }
        let st = env.state();
        let stats = env.stats();
        if stats.success {
            succ += 1;
        }
        days.push(st.day);
        let mut order: Vec<(u16, String)> = (0..c.requests.len())
            .map(|r| (done_day[r], c.requests[r].key.clone()))
            .collect();
        order.sort();
        let last: Vec<String> = order
            .iter()
            .rev()
            .take(4)
            .map(|(d, k)| format!("{k}@{d}"))
            .collect();
        let mut fb: Vec<(u32, String)> = fail_by.into_iter().map(|(k, v)| (v, k)).collect();
        fb.sort();
        fb.reverse();
        fb.truncate(3);
        println!("   last requests {:?} top failures {:?}", last, fb);
        let open: Vec<String> = furrow_expert::open_requests(st)
            .iter()
            .map(|r| c.requests[*r as usize].key.clone())
            .collect();
        println!(
            "seed {seed:3} success {} day {:3} decisions {:5} failed {:4} return {:8.2} reqs {:2}/{} open {:?} intents {:?}",
            stats.success, st.day, stats.decisions, fails, stats.episode_return, st.restoration.requests_done, c.requests.len(), open, intents
        );
    }
    let mean = days.iter().map(|d| *d as f64).sum::<f64>() / days.len() as f64;
    println!("success {succ}/{n}, mean final day {mean:.1}");
}
