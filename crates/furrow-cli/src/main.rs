//! `furrow` — command-line tools: play interactively, run the expert, generate demos, benchmark.
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, Subcommand};
use furrow::{Env, EnvConfig, VecEnv};
use furrow_expert::{Expert, ExpertConfig};

#[derive(Parser)]
#[command(
    name = "furrow",
    version,
    about = "Furrow: an original farming-sim RL environment"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Play in the terminal with macro actions (type a number, a name, `e` for the expert's move, `q` to quit).
    Play {
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Print the ASCII map and text state for a seed.
    Render {
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Run the scripted expert on seeds [start, start+n) and print completion days.
    Expert {
        #[arg(long, default_value_t = 0)]
        start: u64,
        #[arg(short, long, default_value_t = 16)]
        n: u64,
        #[arg(long, default_value_t = 0)]
        style: u32,
    },
    /// Generate an expert-demonstration dataset (furrow-traj/2).
    Demos {
        out: PathBuf,
        #[arg(short, long, default_value_t = 100)]
        episodes: u64,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value_t = 0.0)]
        noise: f64,
        #[arg(long, default_value_t = 0)]
        threads: usize,
    },
    /// Measure VecEnv macro-step throughput with masked random actions.
    Bench {
        #[arg(long, default_value_t = 256)]
        envs: usize,
        #[arg(long, default_value_t = 5.0)]
        seconds: f64,
        #[arg(long, default_value_t = 0)]
        threads: usize,
    },
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().cmd {
        Cmd::Play { seed } => play(seed),
        Cmd::Render { seed } => {
            let env = Env::new(EnvConfig {
                seed,
                ..Default::default()
            });
            println!("{}\n{}", env.render_ascii(), env.describe());
            Ok(())
        }
        Cmd::Expert { start, n, style } => {
            let mut days = Vec::new();
            for seed in start..start + n {
                let mut env = Env::new(EnvConfig {
                    seed,
                    ..Default::default()
                });
                let ex = Expert::new(ExpertConfig {
                    style,
                    ..Default::default()
                });
                while !env.is_done() {
                    let a = ex.act(&env);
                    env.step(a);
                }
                let s = env.stats();
                println!(
                    "seed {seed}: success={} day={} decisions={}",
                    s.success,
                    env.state().day,
                    s.decisions
                );
                days.push(env.state().day as f64);
            }
            println!(
                "mean day {:.1}",
                days.iter().sum::<f64>() / days.len().max(1) as f64
            );
            Ok(())
        }
        Cmd::Demos {
            out,
            episodes,
            seed,
            noise,
            threads,
        } => {
            let g = furrow_gen::GenConfig {
                env: EnvConfig {
                    seed,
                    ..Default::default()
                },
                episodes,
                seed_range: (0, 1 << 32),
                noise_ppm: (noise.clamp(0.0, 1.0) * 1e6) as u32,
                styles: vec![0],
                num_threads: threads,
            };
            let t = Instant::now();
            let s = furrow_gen::generate(&out, &g)?;
            println!("{s:?} in {:.1}s", t.elapsed().as_secs_f64());
            Ok(())
        }
        Cmd::Bench {
            envs,
            seconds,
            threads,
        } => {
            let mut v = VecEnv::new(EnvConfig::default(), envs, threads, (0, 1 << 32));
            let a = v.envs()[0].num_actions();
            let mut rng = furrow::rng::Pcg32::from_seed(1);
            let mut acts = vec![0u32; envs];
            let t = Instant::now();
            let mut steps = 0u64;
            while t.elapsed().as_secs_f64() < seconds {
                for (i, x) in acts.iter_mut().enumerate() {
                    let m = &v.out.mask[i * a..(i + 1) * a];
                    let valid: Vec<u32> = (0..a as u32).filter(|k| m[*k as usize]).collect();
                    *x = valid[rng.below(valid.len() as u32) as usize];
                }
                v.step(&acts).map_err(anyhow::Error::msg)?;
                steps += envs as u64;
            }
            println!(
                "{envs} envs: {:.0} macro steps/s",
                steps as f64 / t.elapsed().as_secs_f64()
            );
            Ok(())
        }
    }
}

fn play(seed: u64) -> anyhow::Result<()> {
    let mut env = Env::new(EnvConfig {
        seed,
        ..Default::default()
    });
    let ex = Expert::new(ExpertConfig::default());
    let names = env.action_names();
    let stdin = io::stdin();
    loop {
        println!("\n{}", env.describe());
        let valid: Vec<(usize, &String)> = names
            .iter()
            .enumerate()
            .filter(|(i, _)| env.action_mask()[*i])
            .collect();
        for (k, (_, n)) in valid.iter().enumerate() {
            print!("[{k}] {n}  ");
        }
        print!("\n> ");
        io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            return Ok(());
        }
        let line = line.trim();
        let a = match line {
            "q" => return Ok(()),
            "m" => {
                println!("{}", env.render_ascii());
                continue;
            }
            "e" => ex.act(&env),
            s => match s.parse::<usize>() {
                Ok(k) if k < valid.len() => valid[k].0 as u32,
                _ => match env.action_id(s) {
                    Some(id) => id,
                    None => {
                        println!("unknown action {s:?}");
                        continue;
                    }
                },
            },
        };
        let s = env.step(a);
        println!(
            "-> {} ({:?}, {} min, reward {:.2})",
            names[a as usize], s.outcome, s.ticks, s.reward
        );
        if s.terminated || s.truncated {
            println!(
                "episode over: success={} day={}",
                env.stats().success,
                env.state().day
            );
            return Ok(());
        }
    }
}
