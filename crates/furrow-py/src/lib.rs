//! Python bindings for Furrow (`furrow._furrow`), built with maturin (abi3, Python ≥ 3.11).
//!
//! The API mirrors Gymnasium: `reset() -> (obs, info)`, `step(a) -> (obs, reward, terminated,
//! truncated, info)`, where `obs = {"global_": f32[G], "action_mask": bool[A] (, "grid": u8[C,H,W])}`.
//! Actions may be given as ids or names (e.g. `"Plant(radish,farm)"`). Long-running calls release
//! the GIL.

use numpy::{PyArray1, PyArrayLike1, PyArrayMethods, AllowTypeChange};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use furrow::{ActionMode, Env, EnvConfig, RemixPartition, RestorationMode, RewardConfig, VecEnv};
use furrow_expert::{Expert, ExpertConfig};

fn parse_cfg(seed: u64, kw: Option<&Bound<'_, PyDict>>) -> PyResult<EnvConfig> {
    let mut cfg = EnvConfig { seed, ..Default::default() };
    let Some(kw) = kw else { return Ok(cfg.normalized()) };
    for (k, v) in kw.iter() {
        let key: String = k.extract()?;
        match key.as_str() {
            "action_mode" => {
                cfg.action_mode = match v.extract::<String>()?.as_str() {
                    "macro" => ActionMode::Macro,
                    "primitive" => {
                        cfg.max_steps = 0;
                        cfg.grid = true;
                        ActionMode::Primitive
                    }
                    o => return Err(PyValueError::new_err(format!("action_mode must be 'macro' or 'primitive', got {o}"))),
                }
            }
            "restoration" => {
                cfg.restoration = match v.extract::<String>()?.as_str() {
                    "fixed" => RestorationMode::Fixed,
                    "remixed" => RestorationMode::Remixed,
                    o => return Err(PyValueError::new_err(format!("restoration must be 'fixed' or 'remixed', got {o}"))),
                }
            }
            "remix_partition" => {
                cfg.remix_partition = match v.extract::<String>()?.as_str() {
                    "train" => RemixPartition::Train,
                    "test" => RemixPartition::Test,
                    o => return Err(PyValueError::new_err(format!("remix_partition must be 'train' or 'test', got {o}"))),
                }
            }
            "max_days" => cfg.max_days = v.extract()?,
            "max_steps" => cfg.max_steps = v.extract::<Option<u32>>()?.unwrap_or(0),
            "max_decisions_per_day" => cfg.max_decisions_per_day = v.extract()?,
            "macro_max_minutes" => cfg.macro_max_minutes = v.extract()?,
            "grid" => {
                if let Some(g) = v.extract::<Option<bool>>()? {
                    cfg.grid = g
                }
            }
            "grid_radius" => cfg.grid_radius = v.extract()?,
            "start_money" => cfg.start_money = v.extract()?,
            "reward" => {
                let d = v.cast::<PyDict>()?;
                let mut r = RewardConfig::default();
                for (rk, rv) in d.iter() {
                    let rk: String = rk.extract()?;
                    match rk.as_str() {
                        "per_day" => r.per_day = rv.extract()?,
                        "completion" => r.completion = rv.extract()?,
                        "phi_slot" => r.phi_slot = rv.extract()?,
                        "phi_request" => r.phi_request = rv.extract()?,
                        "phi_wing" => r.phi_wing = rv.extract()?,
                        "shaping" => r.shaping = rv.extract()?,
                        "gamma_day" => r.gamma_day = rv.extract()?,
                        o => return Err(PyKeyError::new_err(format!("unknown reward key {o}"))),
                    }
                }
                cfg.reward = r;
            }
            "seed" => cfg.seed = v.extract()?,
            o => return Err(PyKeyError::new_err(format!("unknown config key {o}"))),
        }
    }
    Ok(cfg.normalized())
}

fn obs_dict<'py>(py: Python<'py>, env: &Env) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("global_", PyArray1::from_slice(py, env.obs_global()))?;
    d.set_item("action_mask", PyArray1::from_slice(py, env.action_mask()))?;
    if !env.obs_grid().is_empty() {
        let (c, h, w) = env.grid_shape();
        d.set_item("grid", PyArray1::from_slice(py, env.obs_grid()).reshape([c, h, w])?)?;
    }
    Ok(d)
}

fn info_dict<'py>(py: Python<'py>, env: &Env) -> PyResult<Bound<'py, PyDict>> {
    let st = env.state();
    let c = furrow::content::content();
    let stats = env.stats();
    let d = PyDict::new(py);
    d.set_item("day", st.day)?;
    d.set_item("minute", st.minute)?;
    d.set_item("abs_minute", st.abs_minute)?;
    d.set_item("season", st.season().name())?;
    d.set_item("location", &c.locations[st.player.loc as usize].key)?;
    d.set_item("money", st.player.money)?;
    d.set_item("energy", st.player.energy)?;
    d.set_item("slots_filled", st.restoration.slots_filled)?;
    d.set_item("total_slots", c.total_slots)?;
    d.set_item("requests_done", st.restoration.requests_done)?;
    d.set_item("total_requests", c.requests.len())?;
    d.set_item("wings_done", st.restoration.wings_done)?;
    d.set_item("total_wings", c.wings.len())?;
    d.set_item("decisions", stats.decisions)?;
    d.set_item("decisions_today", st.decisions_today)?;
    d.set_item("success", stats.success)?;
    d.set_item("episode_return", stats.episode_return)?;
    d.set_item("episode_seed", st.seed)?;
    Ok(d)
}

fn resolve_action(env: &Env, a: &Bound<'_, PyAny>) -> PyResult<u32> {
    if let Ok(i) = a.extract::<u32>() {
        if (i as usize) < env.num_actions() {
            return Ok(i);
        }
        return Err(PyValueError::new_err(format!("action {i} out of range (num_actions={})", env.num_actions())));
    }
    let name: String = a.extract()?;
    env.action_id(&name).ok_or_else(|| PyKeyError::new_err(format!("unknown action name {name:?}")))
}

/// A single Furrow environment.
#[pyclass(module = "furrow._furrow")]
pub struct PyEnv {
    env: Env,
}

#[pymethods]
impl PyEnv {
    #[new]
    #[pyo3(signature = (seed=0, **kwargs))]
    fn new(seed: u64, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Ok(PyEnv { env: Env::new(parse_cfg(seed, kwargs)?) })
    }

    /// Reset to a new episode. Returns `(obs, info)`.
    #[pyo3(signature = (seed=None))]
    fn reset<'py>(&mut self, py: Python<'py>, seed: Option<u64>) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
        let s = seed.unwrap_or(self.env.cfg().seed);
        self.env.reset(s);
        Ok((obs_dict(py, &self.env)?, info_dict(py, &self.env)?))
    }

    /// Step with an action id or name. Returns `(obs, reward, terminated, truncated, info)`.
    #[allow(clippy::type_complexity)]
    fn step<'py>(&mut self, py: Python<'py>, action: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyDict>, f32, bool, bool, Bound<'py, PyDict>)> {
        let a = resolve_action(&self.env, action)?;
        let s = self.env.step(a);
        let info = info_dict(py, &self.env)?;
        info.set_item("day_ends", s.day_ends)?;
        info.set_item("ticks", s.ticks)?;
        info.set_item("outcome", format!("{:?}", s.outcome).to_lowercase())?;
        Ok((obs_dict(py, &self.env)?, s.reward, s.terminated, s.truncated, info))
    }

    /// Current observation dict.
    fn obs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        obs_dict(py, &self.env)
    }
    /// Current info dict.
    fn info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        info_dict(py, &self.env)
    }
    /// Current action mask.
    fn action_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        PyArray1::from_slice(py, self.env.action_mask())
    }
    /// All action names (index = action id).
    fn action_names(&self) -> Vec<String> {
        self.env.action_names()
    }
    /// Names of the currently valid actions.
    fn valid_actions(&self) -> Vec<String> {
        self.env.valid_action_names()
    }
    /// Action id for a name.
    fn action_id(&self, name: &str) -> PyResult<u32> {
        self.env.action_id(name).ok_or_else(|| PyKeyError::new_err(format!("unknown action name {name:?}")))
    }
    /// Compact text description of the state (for LLM agents).
    fn describe(&self) -> String {
        self.env.describe()
    }
    /// ASCII map of the current location.
    fn render(&self) -> String {
        self.env.render_ascii()
    }
    /// Stable hash of the full state.
    fn stable_hash(&self) -> u64 {
        self.env.stable_hash()
    }
    /// Episode finished?
    fn is_done(&self) -> bool {
        self.env.is_done()
    }
    /// Bit-identical copy of the environment.
    fn clone(&self) -> PyEnv {
        PyEnv { env: self.env.clone() }
    }
    /// The scripted expert's (clean) action for the current state (macro mode).
    #[pyo3(signature = (style=0))]
    fn expert_action(&self, style: u32) -> u32 {
        Expert::new(ExpertConfig { style, ..Default::default() }).act(&self.env)
    }
    /// `(executed, clean)` noisy expert action (noise in [0, 1]).
    #[pyo3(signature = (noise, style=0, seed=0))]
    fn noisy_expert_action(&self, noise: f64, style: u32, seed: u64) -> (u32, u32) {
        let ppm = (noise.clamp(0.0, 1.0) * 1e6) as u32;
        Expert::new(ExpertConfig { style, noise_ppm: ppm, seed }).noisy_act(&self.env)
    }
    #[getter]
    fn num_actions(&self) -> usize {
        self.env.num_actions()
    }
    #[getter]
    fn obs_global_dim(&self) -> usize {
        self.env.obs_global().len()
    }
    #[getter]
    fn grid_shape(&self) -> Option<(usize, usize, usize)> {
        if self.env.obs_grid().is_empty() {
            None
        } else {
            Some(self.env.grid_shape())
        }
    }
    #[getter]
    fn action_mode(&self) -> &'static str {
        match self.env.cfg().action_mode {
            ActionMode::Macro => "macro",
            ActionMode::Primitive => "primitive",
        }
    }
    #[getter]
    fn gamma_day(&self) -> f32 {
        self.env.cfg().reward.gamma_day
    }
    fn __repr__(&self) -> String {
        let st = self.env.state();
        format!("furrow.Env(seed={}, day={}, minute={}, requests_done={})", st.seed, st.day, st.minute, st.restoration.requests_done)
    }
}

/// N environments stepped in parallel (Rust threads, GIL released), same-step auto-reset.
#[pyclass(module = "furrow._furrow")]
pub struct PyVecEnv {
    v: VecEnv,
    auto_reset: bool,
}

impl PyVecEnv {
    fn obs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let n = self.v.len();
        let g = furrow::obs::obs_global_dim();
        let a = self.v.envs().first().map(|e| e.num_actions()).unwrap_or(0);
        let d = PyDict::new(py);
        d.set_item("global_", PyArray1::from_slice(py, &self.v.out.obs).reshape([n, g])?)?;
        d.set_item("action_mask", PyArray1::from_slice(py, &self.v.out.mask).reshape([n, a])?)?;
        if !self.v.out.grid.is_empty() {
            let (c, h, w) = self.v.envs()[0].grid_shape();
            d.set_item("grid", PyArray1::from_slice(py, &self.v.out.grid).reshape([n, c, h, w])?)?;
        }
        Ok(d)
    }
}

#[pymethods]
impl PyVecEnv {
    #[new]
    #[pyo3(signature = (num_envs, seed=0, seed_range=(0, 1u64 << 32), num_threads=None, seeds=None, **kwargs))]
    fn new(
        num_envs: usize,
        seed: u64,
        seed_range: (u64, u64),
        num_threads: Option<usize>,
        seeds: Option<Vec<u64>>,
        kwargs: Option<&Bound<'_, PyDict>>,
        py: Python<'_>,
    ) -> PyResult<Self> {
        let cfg = parse_cfg(seed, kwargs)?;
        let threads = num_threads.unwrap_or(0);
        let (v, auto) = py.detach(|| match seeds {
            Some(s) => (VecEnv::with_seeds(cfg, &s, threads), false),
            None => (VecEnv::new(cfg, num_envs, threads, seed_range), true),
        });
        Ok(PyVecEnv { v, auto_reset: auto })
    }

    /// Reset all envs. Returns `(obs, info)`.
    fn reset<'py>(&mut self, py: Python<'py>) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
        py.detach(|| self.v.reset());
        let info = PyDict::new(py);
        info.set_item("episode_seed", PyArray1::from_slice(py, self.v.seeds()))?;
        Ok((self.obs(py)?, info))
    }

    /// Step all envs. `actions`: int array of length N. Returns
    /// `(obs, reward, terminated, truncated, info)`; `info["final_obs"]`/`info["_final_obs"]` follow
    /// Gymnasium's SAME_STEP convention, and per-episode stats are valid where `_final_obs` is True.
    #[allow(clippy::type_complexity)]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyArrayLike1<'py, i64, AllowTypeChange>,
    ) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<bool>>, Bound<'py, PyArray1<bool>>, Bound<'py, PyDict>)> {
        let acts: Vec<u32> = actions.as_array().iter().map(|x| *x as u32).collect();
        py.detach(|| self.v.step(&acts)).map_err(PyValueError::new_err)?;
        let o = &self.v.out;
        let n = self.v.len();
        let g = furrow::obs::obs_global_dim();
        let a = self.v.envs().first().map(|e| e.num_actions()).unwrap_or(0);
        let done: Vec<bool> = (0..n).map(|i| o.terminated[i] || o.truncated[i]).collect();
        let info = PyDict::new(py);
        info.set_item("day_ends", PyArray1::from_slice(py, &o.day_ends))?;
        info.set_item("ticks", PyArray1::from_slice(py, &o.ticks))?;
        info.set_item("outcome", PyArray1::from_slice(py, &o.outcome))?;
        info.set_item("_final_obs", PyArray1::from_slice(py, &done))?;
        let fo = PyDict::new(py);
        fo.set_item("global_", PyArray1::from_slice(py, &o.final_obs).reshape([n, g])?)?;
        fo.set_item("action_mask", PyArray1::from_slice(py, &o.final_mask).reshape([n, a])?)?;
        info.set_item("final_obs", fo)?;
        info.set_item("episode_return", PyArray1::from_slice(py, &o.ep_return))?;
        info.set_item("episode_days", PyArray1::from_slice(py, &o.ep_days))?;
        info.set_item("success", PyArray1::from_slice(py, &o.ep_success))?;
        info.set_item("episode_seed", PyArray1::from_slice(py, &o.ep_seed))?;
        info.set_item("slots_filled", PyArray1::from_slice(py, &o.ep_slots))?;
        info.set_item("requests_done", PyArray1::from_slice(py, &o.ep_requests))?;
        info.set_item("wings_done", PyArray1::from_slice(py, &o.ep_wings))?;
        info.set_item("decisions", PyArray1::from_slice(py, &o.ep_decisions))?;
        if !self.auto_reset {
            info.set_item("done_mask", PyArray1::from_vec(py, self.v.done_mask()))?;
        }
        Ok((
            self.obs(py)?,
            PyArray1::from_slice(py, &o.reward),
            PyArray1::from_slice(py, &o.terminated),
            PyArray1::from_slice(py, &o.truncated),
            info,
        ))
    }

    /// Clean expert labels for every env (computed in parallel).
    #[pyo3(signature = (styles=None))]
    fn expert_actions<'py>(&self, py: Python<'py>, styles: Option<Vec<u32>>) -> Bound<'py, PyArray1<i64>> {
        let v = py.detach(|| {
            self.v.map_envs(|i, e| {
                let style = styles.as_ref().map(|s| s[i % s.len()]).unwrap_or(0);
                if e.is_done() {
                    0
                } else {
                    Expert::new(ExpertConfig { style, ..Default::default() }).act(e) as i64
                }
            })
        });
        PyArray1::from_vec(py, v)
    }

    /// `(executed, clean)` noisy expert actions.
    #[pyo3(signature = (noise, styles=None, seed=0))]
    #[allow(clippy::type_complexity)]
    fn noisy_expert_actions<'py>(&self, py: Python<'py>, noise: f64, styles: Option<Vec<u32>>, seed: u64) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
        let ppm = (noise.clamp(0.0, 1.0) * 1e6) as u32;
        let v = py.detach(|| {
            self.v.map_envs(|i, e| {
                let style = styles.as_ref().map(|s| s[i % s.len()]).unwrap_or(0);
                if e.is_done() {
                    (0, 0)
                } else {
                    let (x, c) = Expert::new(ExpertConfig { style, noise_ppm: ppm, seed: seed ^ i as u64 }).noisy_act(e);
                    (x as i64, c as i64)
                }
            })
        });
        let (x, c): (Vec<i64>, Vec<i64>) = v.into_iter().unzip();
        (PyArray1::from_vec(py, x), PyArray1::from_vec(py, c))
    }

    /// Text descriptions of every env.
    fn describe(&self) -> Vec<String> {
        self.v.envs().iter().map(|e| e.describe()).collect()
    }
    /// Done mask (evaluation mode).
    fn done_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        PyArray1::from_vec(py, self.v.done_mask())
    }
    /// Current episode seeds.
    fn seeds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        PyArray1::from_slice(py, self.v.seeds())
    }
    /// Action names.
    fn action_names(&self) -> Vec<String> {
        self.v.envs().first().map(|e| e.action_names()).unwrap_or_default()
    }
    #[getter]
    fn num_envs(&self) -> usize {
        self.v.len()
    }
    #[getter]
    fn num_actions(&self) -> usize {
        self.v.envs().first().map(|e| e.num_actions()).unwrap_or(0)
    }
    #[getter]
    fn obs_global_dim(&self) -> usize {
        furrow::obs::obs_global_dim()
    }
    #[getter]
    fn grid_shape(&self) -> Option<(usize, usize, usize)> {
        self.v.envs().first().filter(|e| !e.obs_grid().is_empty()).map(|e| e.grid_shape())
    }
    fn __len__(&self) -> usize {
        self.v.len()
    }
}

/// Generate an expert-demonstration dataset (furrow-traj/2) in parallel. Returns a summary dict.
#[pyfunction]
#[pyo3(signature = (out_dir, episodes, seed=0, seed_range=(0, 1u64 << 32), noise=0.0, styles=None, num_threads=None, **kwargs))]
#[allow(clippy::too_many_arguments)]
fn generate_demos<'py>(
    py: Python<'py>,
    out_dir: String,
    episodes: u64,
    seed: u64,
    seed_range: (u64, u64),
    noise: f64,
    styles: Option<Vec<u32>>,
    num_threads: Option<usize>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let cfg = parse_cfg(seed, kwargs)?;
    let g = furrow_gen::GenConfig {
        env: cfg,
        episodes,
        seed_range,
        noise_ppm: (noise.clamp(0.0, 1.0) * 1e6) as u32,
        styles: styles.unwrap_or_default(),
        num_threads: num_threads.unwrap_or(0),
    };
    let path = std::path::PathBuf::from(out_dir);
    let s = py.detach(|| furrow_gen::generate(&path, &g)).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let d = PyDict::new(py);
    d.set_item("episodes", s.episodes)?;
    d.set_item("steps", s.steps)?;
    d.set_item("successes", s.successes)?;
    d.set_item("mean_days", s.mean_days)?;
    d.set_item("mean_return", s.mean_return)?;
    Ok(d)
}

/// FNV-1a 64 hash of the content tables.
#[pyfunction]
fn content_hash() -> u64 {
    furrow::content::content_hash()
}

/// The raw content TOML sources: `{file_name: text}`.
#[pyfunction]
fn content_sources<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (k, v) in furrow::content::SOURCES {
        d.set_item(k, v)?;
    }
    Ok(d)
}

/// Observation layout: list of `(name, offset, length)`.
#[pyfunction]
fn obs_layout<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
    let items: Vec<(&str, usize, usize)> = furrow::obs::global_layout().to_vec();
    PyList::new(py, items)
}

/// Macro action names (index = id).
#[pyfunction]
fn macro_action_names() -> Vec<String> {
    furrow::macro_action::macro_actions().iter().map(|m| m.name()).collect()
}

/// Item keys (index = item id).
#[pyfunction]
fn item_keys() -> Vec<String> {
    furrow::content::content().items.iter().map(|i| i.key.clone()).collect()
}

#[pymodule]
fn _furrow(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEnv>()?;
    m.add_class::<PyVecEnv>()?;
    m.add_function(wrap_pyfunction!(generate_demos, m)?)?;
    m.add_function(wrap_pyfunction!(content_hash, m)?)?;
    m.add_function(wrap_pyfunction!(content_sources, m)?)?;
    m.add_function(wrap_pyfunction!(obs_layout, m)?)?;
    m.add_function(wrap_pyfunction!(macro_action_names, m)?)?;
    m.add_function(wrap_pyfunction!(item_keys, m)?)?;
    m.add("SIM_VERSION", furrow::SIM_VERSION)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
