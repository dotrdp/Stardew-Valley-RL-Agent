use furrow::{Env, EnvConfig};
use furrow_expert::{Expert, ExpertConfig};
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(60);
    let mut env = Env::new(EnvConfig { seed, ..Default::default() });
    let ex = Expert::new(ExpertConfig::default());
    let names = env.action_names();
    for _ in 0..n {
        if env.is_done() { break; }
        let st = env.state();
        let (a, it) = ex.act_with_intent(&env);
        let before = format!("d{} {:02}:{:02} loc{} ({},{}) e{} ${}", st.day, st.minute/60, st.minute%60, st.player.loc, st.player.x, st.player.y, st.player.energy, st.player.money);
        let s = env.step(a);
        let st = env.state();
        println!("{before} -> {:28} {:?} {:?} ticks {} -> loc{} ({},{})", names[a as usize], it, s.outcome, s.ticks, st.player.loc, st.player.x, st.player.y);
    }
}
