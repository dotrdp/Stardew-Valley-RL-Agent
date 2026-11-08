//! Rust client for the RLBridge SMAPI mod (see `docs/BRIDGE_PROTOCOL.md`).
//!
//! * [`SdvClient`] — newline-delimited JSON over TCP, request ids, timeouts.
//! * [`SdvEnv`] — reset/step/observe in the same shape as `furrow::Env`, with the Community Center
//!   completion as the objective.
//! * [`MockServer`] — an in-process fake mod for tests (no game required).

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Protocol version spoken by this client.
pub const PROTOCOL: u32 = 1;

/// Errors.
#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("mod error: {0}")]
    Remote(String),
    #[error("protocol: {0}")]
    Protocol(String),
}

/// One frame of held input.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct InputFrame {
    /// "up" | "down" | "left" | "right" | None
    #[serde(default)]
    pub r#move: Option<String>,
    #[serde(default)]
    pub use_tool: bool,
    #[serde(default)]
    pub action: bool,
    #[serde(default)]
    pub select_slot: Option<u8>,
    #[serde(default)]
    pub face: Option<u8>,
}

/// Inventory item.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ItemStack {
    pub slot: u32,
    pub id: String,
    pub name: String,
    pub stack: u32,
    #[serde(default)]
    pub quality: u8,
}

/// Player state.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct PlayerState {
    pub location: String,
    pub tile: [i32; 2],
    pub facing: u8,
    pub stamina: f32,
    pub max_stamina: u32,
    pub money: i64,
    pub current_slot: u32,
    #[serde(default)]
    pub items: Vec<ItemStack>,
    #[serde(default)]
    pub skills: std::collections::BTreeMap<String, u32>,
}

/// Local map window.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct LocationWindow {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub radius: u32,
    /// Row-major '0'/'1' passability around the player.
    pub passable: String,
}

/// One bundle item requirement.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct BundleItem {
    pub id: String,
    pub qty: u32,
    #[serde(default)]
    pub quality: u8,
    pub done: bool,
}

/// One Community Center bundle.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Bundle {
    pub key: String,
    pub name: String,
    pub area: String,
    pub items: Vec<BundleItem>,
    pub need: u32,
    pub complete: bool,
}

/// Community Center progress.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct CommunityCenter {
    pub complete: bool,
    pub areas_complete: Vec<bool>,
    pub bundles: Vec<Bundle>,
}

/// Full snapshot from the mod.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Snapshot {
    pub tick: u64,
    pub day: u32,
    pub season: String,
    pub year: u32,
    pub time: u32,
    pub weather: String,
    #[serde(default)]
    pub weather_tomorrow: String,
    pub player: PlayerState,
    pub location: LocationWindow,
    pub community_center: CommunityCenter,
    #[serde(default)]
    pub menu: Option<String>,
    #[serde(default)]
    pub event_active: bool,
}

impl Snapshot {
    /// Absolute in-game day index (0-based; 28-day seasons in the commercial game).
    pub fn day_index(&self) -> u32 {
        let s = match self.season.as_str() {
            "spring" => 0,
            "summer" => 1,
            "fall" | "autumn" => 2,
            _ => 3,
        };
        (self.year.saturating_sub(1)) * 112 + s * 28 + self.day.saturating_sub(1)
    }
    /// Number of completed bundles.
    pub fn bundles_complete(&self) -> usize {
        self.community_center
            .bundles
            .iter()
            .filter(|b| b.complete)
            .count()
    }
}

/// Response of a step/macro.
#[derive(Clone, Debug)]
pub struct StepResult {
    pub ok: bool,
    pub obs: Snapshot,
    pub events: Vec<Value>,
    pub detail: String,
}

/// Low-level client.
pub struct SdvClient {
    stream: TcpStream,
    reader: BufReader<TcpStream>,
    next_id: u64,
}

impl SdvClient {
    /// Connect and handshake.
    pub fn connect<A: ToSocketAddrs>(
        addr: A,
        timeout: Duration,
    ) -> Result<(Self, Value), BridgeError> {
        let stream = TcpStream::connect(addr)?;
        stream.set_read_timeout(Some(timeout))?;
        stream.set_nodelay(true)?;
        let reader = BufReader::new(stream.try_clone()?);
        let mut c = SdvClient {
            stream,
            reader,
            next_id: 1,
        };
        let hello = c.request(json!({"cmd": "hello"}))?;
        let proto = hello.get("protocol").and_then(Value::as_u64).unwrap_or(0);
        if proto != PROTOCOL as u64 {
            return Err(BridgeError::Protocol(format!(
                "mod speaks protocol {proto}, client {PROTOCOL}"
            )));
        }
        Ok((c, hello))
    }
    /// Send a request and wait for its response.
    pub fn request(&mut self, mut req: Value) -> Result<Value, BridgeError> {
        let id = self.next_id;
        self.next_id += 1;
        req["id"] = json!(id);
        let mut line = serde_json::to_string(&req)?;
        line.push('\n');
        self.stream.write_all(line.as_bytes())?;
        loop {
            let mut buf = String::new();
            if self.reader.read_line(&mut buf)? == 0 {
                return Err(BridgeError::Protocol("connection closed".into()));
            }
            let v: Value = serde_json::from_str(buf.trim())?;
            if v.get("id").and_then(Value::as_u64) != Some(id) {
                continue; // stale/unsolicited message
            }
            if v.get("ok").and_then(Value::as_bool) == Some(false) && v.get("obs").is_none() {
                return Err(BridgeError::Remote(
                    v.get("error")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown")
                        .to_string(),
                ));
            }
            return Ok(v);
        }
    }
    fn parse_step(v: Value) -> Result<StepResult, BridgeError> {
        Ok(StepResult {
            ok: v.get("ok").and_then(Value::as_bool).unwrap_or(false),
            obs: serde_json::from_value(v.get("obs").cloned().unwrap_or(Value::Null))?,
            events: v
                .get("events")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
            detail: v
                .get("detail")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
        })
    }
    /// Current snapshot.
    pub fn observe(&mut self) -> Result<Snapshot, BridgeError> {
        let v = self.request(json!({"cmd": "observe"}))?;
        Ok(serde_json::from_value(v["obs"].clone())?)
    }
    /// Hold `input` for `ticks` update ticks.
    pub fn step(&mut self, ticks: u32, input: &InputFrame) -> Result<StepResult, BridgeError> {
        Self::parse_step(self.request(json!({"cmd": "step", "ticks": ticks, "input": input}))?)
    }
    /// Run a macro.
    pub fn run_macro(&mut self, name: &str, args: Value) -> Result<StepResult, BridgeError> {
        Self::parse_step(self.request(json!({"cmd": "macro", "name": name, "args": args}))?)
    }
    /// Enable/disable lockstep.
    pub fn lockstep(&mut self, enabled: bool) -> Result<(), BridgeError> {
        self.request(json!({"cmd": "lockstep", "enabled": enabled}))
            .map(|_| ())
    }
    /// Load a save (episode reset).
    pub fn load_save(&mut self, name: &str) -> Result<Snapshot, BridgeError> {
        let v = self.request(json!({"cmd": "load_save", "save_name": name}))?;
        Ok(serde_json::from_value(v["obs"].clone())?)
    }
}

/// Episode-level environment over the bridge (objective: Community Center completion).
pub struct SdvEnv {
    client: SdvClient,
    save: String,
    start_day: u32,
    last: Snapshot,
    /// Ticks per primitive step.
    pub ticks_per_step: u32,
}

impl SdvEnv {
    /// Connect, enable lockstep and remember the save used for resets.
    pub fn connect(addr: &str, save: &str) -> Result<Self, BridgeError> {
        let (mut client, _) = SdvClient::connect(addr, Duration::from_secs(30))?;
        client.lockstep(true)?;
        let last = client.observe()?;
        Ok(SdvEnv {
            start_day: last.day_index(),
            client,
            save: save.to_string(),
            last,
            ticks_per_step: 15,
        })
    }
    /// Reload the start save.
    pub fn reset(&mut self) -> Result<&Snapshot, BridgeError> {
        self.last = self.client.load_save(&self.save)?;
        self.start_day = self.last.day_index();
        Ok(&self.last)
    }
    /// Primitive step; reward = −(days elapsed) + bundles completed (potential difference).
    pub fn step(&mut self, input: &InputFrame) -> Result<(f32, bool, &Snapshot), BridgeError> {
        let before = self.last.clone();
        let r = self.client.step(self.ticks_per_step, input)?;
        self.last = r.obs;
        Ok((
            self.reward(&before),
            self.last.community_center.complete,
            &self.last,
        ))
    }
    /// Macro step.
    pub fn step_macro(
        &mut self,
        name: &str,
        args: Value,
    ) -> Result<(f32, bool, &Snapshot), BridgeError> {
        let before = self.last.clone();
        let r = self.client.run_macro(name, args)?;
        self.last = r.obs;
        Ok((
            self.reward(&before),
            self.last.community_center.complete,
            &self.last,
        ))
    }
    fn reward(&self, before: &Snapshot) -> f32 {
        let days = self.last.day_index() as f32 - before.day_index() as f32;
        let bundles = self.last.bundles_complete() as f32 - before.bundles_complete() as f32;
        -days + 0.25 * bundles
    }
    /// Days since the episode start.
    pub fn days_elapsed(&self) -> u32 {
        self.last.day_index().saturating_sub(self.start_day)
    }
    /// Last snapshot.
    pub fn last(&self) -> &Snapshot {
        &self.last
    }
}

/// A fake RLBridge for tests: a tiny world where moving right fills bundles.
pub struct MockServer {
    pub addr: String,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl MockServer {
    /// Start on an ephemeral port; serves one client then exits.
    pub fn start() -> std::io::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let addr = listener.local_addr()?.to_string();
        let handle = std::thread::spawn(move || {
            let Ok((stream, _)) = listener.accept() else {
                return;
            };
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut out = stream;
            let mut snap = mock_snapshot();
            let mut line = String::new();
            while reader.read_line(&mut line).map(|n| n > 0).unwrap_or(false) {
                let req: Value = serde_json::from_str(line.trim()).unwrap_or(Value::Null);
                line.clear();
                let id = req.get("id").cloned().unwrap_or(json!(0));
                let resp = match req.get("cmd").and_then(Value::as_str).unwrap_or("") {
                    "hello" => {
                        json!({"id": id, "ok": true, "protocol": PROTOCOL, "game_version": "mock", "smapi_version": "mock", "mod_version": "mock"})
                    }
                    "observe" => json!({"id": id, "ok": true, "obs": snap}),
                    "lockstep" | "record" => json!({"id": id, "ok": true}),
                    "load_save" => {
                        snap = mock_snapshot();
                        json!({"id": id, "ok": true, "obs": snap})
                    }
                    "step" => {
                        let ticks = req.get("ticks").and_then(Value::as_u64).unwrap_or(1) as u32;
                        snap.tick += ticks as u64;
                        snap.time += 10 * (ticks / 7).max(1);
                        if req["input"]["move"] == "right" {
                            snap.player.tile[0] += 1;
                            if let Some(b) = snap
                                .community_center
                                .bundles
                                .iter_mut()
                                .find(|b| !b.complete)
                            {
                                b.complete = true;
                            }
                        }
                        if snap.time >= 2600 {
                            snap.time = 600;
                            snap.day += 1;
                        }
                        snap.community_center.complete =
                            snap.community_center.bundles.iter().all(|b| b.complete);
                        json!({"id": id, "ok": true, "obs": snap, "events": []})
                    }
                    "macro" => {
                        if req["name"] == "sleep" {
                            snap.day += 1;
                            snap.time = 600;
                        }
                        json!({"id": id, "ok": true, "obs": snap, "events": [{"kind": "day_end"}], "detail": "mock"})
                    }
                    other => {
                        json!({"id": id, "ok": false, "error": format!("unknown cmd {other}")})
                    }
                };
                let mut s = serde_json::to_string(&resp).unwrap();
                s.push('\n');
                if out.write_all(s.as_bytes()).is_err() {
                    break;
                }
            }
        });
        Ok(MockServer {
            addr,
            handle: Some(handle),
        })
    }
}

impl Drop for MockServer {
    fn drop(&mut self) {
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn mock_snapshot() -> Snapshot {
    Snapshot {
        tick: 0,
        day: 1,
        season: "spring".into(),
        year: 1,
        time: 600,
        weather: "sunny".into(),
        weather_tomorrow: "rain".into(),
        player: PlayerState {
            location: "Farm".into(),
            tile: [64, 15],
            facing: 2,
            stamina: 270.0,
            max_stamina: 270,
            money: 500,
            ..Default::default()
        },
        location: LocationWindow {
            name: "Farm".into(),
            width: 80,
            height: 65,
            radius: 1,
            passable: "111111111".into(),
        },
        community_center: CommunityCenter {
            complete: false,
            areas_complete: vec![false; 6],
            bundles: (0..3)
                .map(|i| Bundle {
                    key: format!("Pantry/{i}"),
                    name: format!("Mock {i}"),
                    area: "Pantry".into(),
                    need: 1,
                    ..Default::default()
                })
                .collect(),
        },
        menu: None,
        event_active: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_talks_to_mock() {
        let srv = MockServer::start().unwrap();
        let mut env = SdvEnv::connect(&srv.addr, "mock_save").unwrap();
        assert_eq!(env.last().player.money, 500);
        let right = InputFrame {
            r#move: Some("right".into()),
            ..Default::default()
        };
        let mut done = false;
        for _ in 0..3 {
            let (_, d, s) = env.step(&right).unwrap();
            done = d;
            assert!(s.tick > 0);
        }
        assert!(done, "three bundles should complete");
        let (r, _, s) = env.step_macro("sleep", json!({})).unwrap();
        assert_eq!(s.day, 2);
        assert!(r < 0.0);
        env.reset().unwrap();
        assert_eq!(env.last().day, 1);
        assert_eq!(env.days_elapsed(), 0);
    }

    #[test]
    fn snapshot_day_index_uses_28_day_seasons() {
        let mut s = mock_snapshot();
        s.season = "fall".into();
        s.day = 3;
        s.year = 2;
        assert_eq!(s.day_index(), 112 + 56 + 2);
    }
}
