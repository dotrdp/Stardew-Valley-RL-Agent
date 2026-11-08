//! Golden trajectories: full expert episodes must reproduce bit-identically.
//! If a change to dynamics/controllers/RNG is intended, bump `furrow::SIM_VERSION` and re-bless with
//! `FURROW_BLESS=1 cargo test -p furrow-expert --test golden -- --nocapture` (prints new values).
use furrow::{Env, EnvConfig};
use furrow_expert::{Expert, ExpertConfig};

fn run(seed: u64) -> (u64, u16, u32) {
    let mut env = Env::new(EnvConfig {
        seed,
        ..Default::default()
    });
    let ex = Expert::new(ExpertConfig::default());
    while !env.is_done() {
        let a = ex.act(&env);
        env.step(a);
    }
    (env.stable_hash(), env.state().day, env.stats().decisions)
}

const GOLDEN: [(u64, u64, u16, u32); 3] = [
    (0, 0x4d5d_cc01_207c_5bd5, 60, 1002),
    (1, 0x15f2_14f1_7d0e_40fa, 64, 1125),
    (2, 0x9986_0346_bd9e_899e, 60, 813),
];

#[test]
fn expert_episodes_are_golden() {
    assert_eq!(
        furrow::SIM_VERSION,
        1,
        "SIM_VERSION changed: re-bless goldens"
    );
    let bless = std::env::var("FURROW_BLESS").is_ok();
    for (seed, hash, day, decisions) in GOLDEN {
        let got = run(seed);
        if bless {
            println!("({seed}, 0x{:016x}, {}, {}),", got.0, got.1, got.2);
            continue;
        }
        assert_eq!(
            got,
            (hash, day, decisions),
            "golden mismatch for seed {seed}"
        );
    }
}
