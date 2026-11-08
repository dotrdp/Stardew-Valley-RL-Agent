use furrow::rng::Pcg32;
use furrow::{ActionMode, Env, EnvConfig};

fn random_valid(env: &Env, rng: &mut Pcg32) -> u32 {
    let mask = env.action_mask();
    let n = mask.iter().filter(|m| **m).count();
    assert!(n > 0, "empty mask at day {} minute {}", env.state().day, env.state().minute);
    let mut k = rng.below(n as u32) as usize;
    for (i, m) in mask.iter().enumerate() {
        if *m {
            if k == 0 {
                return i as u32;
            }
            k -= 1;
        }
    }
    unreachable!()
}

fn run(mode: ActionMode, seed: u64, steps: usize) -> (u64, u16, f32, u32) {
    let mut env = Env::new(EnvConfig { seed, action_mode: mode, ..Default::default() });
    let mut rng = Pcg32::from_seed(seed ^ 0x55);
    let mut last_abs = env.state().abs_minute;
    let mut ret = 0.0;
    let mut day_ends = 0u32;
    for _ in 0..steps {
        if env.is_done() {
            break;
        }
        let a = random_valid(&env, &mut rng);
        let s = env.step(a);
        assert!(s.ticks >= 1, "step consumed no time");
        let abs = env.state().abs_minute;
        assert!(abs > last_abs, "clock did not advance: {last_abs} -> {abs}");
        assert_eq!(abs - last_abs, s.ticks, "ticks mismatch");
        last_abs = abs;
        ret += s.reward;
        day_ends += s.day_ends as u32;
    }
    (env.stable_hash(), env.state().day, ret, day_ends)
}

#[test]
fn random_macros_are_deterministic_and_advance_days() {
    for seed in 0..3u64 {
        let a = run(ActionMode::Macro, seed, 400);
        let b = run(ActionMode::Macro, seed, 400);
        assert_eq!(a, b, "nondeterministic for seed {seed}");
        assert!(a.1 >= 5, "only reached day {} after 400 macros", a.1);
        assert_eq!(a.1 as u32, a.3, "day counter and day_ends disagree");
    }
}

#[test]
fn random_primitives_are_deterministic() {
    for seed in 0..2u64 {
        let a = run(ActionMode::Primitive, seed, 3000);
        let b = run(ActionMode::Primitive, seed, 3000);
        assert_eq!(a, b);
        assert!(a.1 >= 1);
    }
}

#[test]
fn obs_and_mask_have_expected_sizes() {
    let env = Env::new(EnvConfig::default());
    assert_eq!(env.obs_global().len(), furrow::obs::obs_global_dim());
    assert_eq!(env.action_mask().len(), env.num_actions());
    assert!(env.num_actions() > 100);
    let names = env.action_names();
    assert_eq!(names.len(), env.num_actions());
    assert_eq!(names[0], "Wait30");
}
