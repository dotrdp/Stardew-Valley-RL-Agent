use furrow::{Env, EnvConfig};
use furrow_expert::{Expert, ExpertConfig, Planner};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let seed: u64 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let day: u16 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(80);
    let mut env = Env::new(EnvConfig {
        seed,
        ..Default::default()
    });
    let ex = Expert::new(ExpertConfig::default());
    while !env.is_done() && env.state().day < day {
        let x = ex.act(&env);
        env.step(x);
    }
    let st = env.state();
    let c = furrow::content::content();
    let p = Planner::new(st, 0);
    println!(
        "day {} season {:?} money {}",
        st.day,
        st.season(),
        st.player.money
    );
    for r in &c.requests {
        if st.restoration.request_done[r.id as usize] {
            continue;
        }
        let range = st.slots_of(r.id);
        for (k, i) in range.clone().enumerate() {
            let it = st.restoration.slot_item[i];
            println!(
                "  {:16} slot {:14} x{:3} filled {} target {} held {} cost {}",
                r.key,
                c.items[it as usize].key,
                r.slots[k].count,
                st.restoration.slot_filled[i],
                p.targets.contains(&i),
                st.count(it),
                p.item_cost(it, r.slots[k].count as u32, 0)
            );
        }
    }
    for (i, it) in c.items.iter().enumerate() {
        if p.deficit(i as u16) > 0 {
            println!("  deficit {} {}", it.key, p.deficit(i as u16));
        }
    }
    let mask = env.action_mask();
    let names = env.action_names();
    let v: Vec<&String> = names
        .iter()
        .zip(mask)
        .filter(|(_, m)| **m)
        .map(|(n, _)| n)
        .collect();
    println!("valid: {:?}", v);
}
