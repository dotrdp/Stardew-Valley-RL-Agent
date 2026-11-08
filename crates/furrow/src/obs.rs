//! Observation encoder (DESIGN.md §3.5): a fixed, content-derived layout of scaled features,
//! identical in meaning across Fixed and Remixed objectives; no feature depends on `EnvConfig`.
//! Scaling uses integer arithmetic only (no transcendental float functions).

use std::sync::OnceLock;

use crate::content::{content, Feature, Tool};
use crate::nav::NavTables;
use crate::state::GameState;
use crate::world::{Obj, NO_CROP, SOIL_TILLED, SOIL_WATERED};

/// A named feature group: `(name, offset, len)`.
pub type LayoutEntry = (&'static str, usize, usize);

static LAYOUT: OnceLock<(Vec<LayoutEntry>, usize)> = OnceLock::new();

/// Grid observation channels.
pub const GRID_CHANNELS: usize = 7;

fn build_layout() -> (Vec<LayoutEntry>, usize) {
    let c = content();
    let mut v = Vec::new();
    let mut off = 0usize;
    let mut push = |name: &'static str, len: usize| {
        v.push((name, off, len));
        off += len;
    };
    push("calendar", 16);
    push("weather", 8);
    push("player", 51 + c.recipes.len());
    push("items", c.items.len() * 6);
    push("requests", c.requests.len() * 3);
    push("wings", c.wings.len());
    push("eta", c.pois.len());
    push("farm", 2 + c.crops.len() * 5);
    push("machines", c.machines.len() * 3);
    push("world", c.forage.len() + c.shops.len() + 2);
    (v, off)
}

/// The global observation layout.
pub fn global_layout() -> &'static [LayoutEntry] {
    &LAYOUT.get_or_init(build_layout).0
}

/// Global observation dimension (a pure function of content).
pub fn obs_global_dim() -> usize {
    LAYOUT.get_or_init(build_layout).1
}

#[inline]
fn frac(n: u32, d: u32) -> f32 {
    if d == 0 {
        0.0
    } else {
        (n.min(d) as f32) / (d as f32)
    }
}

impl GameState {
    /// Write the global observation. `eta` must hold `eta_all` output (length ≥ number of POIs).
    pub fn write_global(&mut self, nav: &NavTables, out: &mut [f32]) {
        let c = self.c();
        debug_assert!(out.len() >= obs_global_dim());
        out.iter_mut().for_each(|x| *x = 0.0);
        let layout = global_layout();
        let g = |name: &str| -> usize { layout.iter().find(|e| e.0 == name).map(|e| e.1).unwrap() };

        // calendar
        let mut o = g("calendar");
        out[o] = frac(self.day_of_season() as u32, c.world.season_days as u32 - 1);
        out[o + 1 + self.season() as usize] = 1.0;
        out[o + 5] = frac(self.year() as u32, 4);
        out[o + 6 + self.weekday() as usize] = 1.0;
        out[o + 13] = frac(self.minute as u32, c.world.day_end as u32);
        out[o + 14] = frac(self.day as u32, 384);
        out[o + 15] = frac(self.days_left_in_season() as u32, c.world.season_days as u32 - 1);
        // weather
        o = g("weather");
        out[o + self.weather as usize] = 1.0;
        out[o + 4 + self.forecast as usize] = 1.0;
        // player
        o = g("player");
        let p = &self.player;
        out[o] = frac(p.energy.max(0) as u32, p.energy_max as u32);
        out[o + 1] = frac(p.money, 10_000);
        out[o + 2] = frac(p.money, 100_000);
        out[o + 3] = frac(p.money.checked_ilog2().unwrap_or(0), 20);
        out[o + 4 + p.loc as usize] = 1.0; // 7 locations
        let gr = self.world.grid(p.loc);
        out[o + 11] = frac(p.x as u32, gr.w as u32);
        out[o + 12] = frac(p.y as u32, gr.h as u32);
        out[o + 13] = frac(self.free_slots() as u32, 24);
        for t in 0..5 {
            out[o + 14 + t] = frac(p.tool_level[t] as u32, 3);
        }
        if let Some((t, ready)) = p.upgrading {
            out[o + 19 + t as usize] = 1.0;
            out[o + 24] = frac(ready.saturating_sub(self.day) as u32, c.upgrade_days as u32);
        }
        for (i, s) in crate::content::Skill::ALL.iter().enumerate() {
            out[o + 25 + i] = frac(self.level(*s) as u32, 10);
        }
        out[o + 29] = frac(self.mine.floor as u32, c.mining.floors as u32);
        out[o + 30] = frac(self.mine.deepest as u32, c.mining.floors as u32);
        for (i, f) in Feature::ALL.iter().enumerate() {
            out[o + 31 + i] = self.has_feature(*f) as u8 as f32;
        }
        out[o + 34] = self.last.failed as u8 as f32;
        out[o + 35] = self.last.passed_out as u8 as f32;
        out[o + 36] = frac(p.can_water as u32, self.can_capacity() as u32);
        out[o + 37 + self.menu.index() as usize] = 1.0; // 6
        out[o + 43] = frac(self.decisions_today as u32, 64);
        // 44..51 reserved (zeros); recipes after 51
        for r in 0..c.recipes.len() {
            out[o + 51 + r] = self.recipe_unlocked(r as u8) as u8 as f32;
        }
        // items
        o = g("items");
        let season = self.season();
        let merchant_today = self.merchant_today();
        for (i, it) in c.items.iter().enumerate() {
            let b = o + i * 6;
            let held = self.count(it.id);
            let need = self.need(it.id);
            out[b] = frac(held, 200);
            out[b + 1] = (held > 0) as u8 as f32;
            out[b + 2] = frac(need, 16);
            out[b + 3] = if need > 0 { frac(held.min(need), need) } else { 0.0 };
            out[b + 4] = frac(self.open_slots_for(it.id), 8);
            let mut price = c.shop_row(it.id, season).map(|(s, row)| c.shops[s as usize].stock[row].price);
            if merchant_today && self.merchant_stock.contains(&it.id) {
                let mp = c.merchant_price(it.id);
                price = Some(price.map(|p| p.min(mp)).unwrap_or(mp));
            }
            out[b + 5] = price.map(|p| frac(p, 2500)).unwrap_or(0.0);
        }
        // requests / wings
        o = g("requests");
        for r in &c.requests {
            let b = o + r.id as usize * 3;
            out[b] = if r.need > 0 { frac(self.filled_in(r.id) as u32, r.need as u32) } else { 0.0 };
            out[b + 1] = self.restoration.request_done[r.id as usize] as u8 as f32;
            out[b + 2] = if r.coins > 0 && !self.restoration.request_done[r.id as usize] { frac(r.coins, 10_000) } else { 0.0 };
        }
        o = g("wings");
        for w in &c.wings {
            out[o + w.id as usize] = self.restoration.wing_done[w.id as usize] as u8 as f32;
        }
        // eta
        o = g("eta");
        let mut etas = vec![u32::MAX; c.pois.len()];
        self.eta_all(nav, &mut etas);
        for (i, e) in etas.iter().enumerate() {
            out[o + i] = if *e == u32::MAX { 1.0 } else { frac(*e, 600) };
        }
        // farm
        o = g("farm");
        let farm = c.world.start_location;
        let gh = c.loc_id("greenhouse");
        out[o] = frac(self.free_field_tiles(farm), 512);
        out[o + 1] = frac(self.free_field_tiles(gh), 108);
        let days_left = self.days_left_in_season() as u32;
        for cr in &c.crops {
            let b = o + 2 + cr.id as usize * 5;
            let mut planted = 0u32;
            let mut watered = 0u32;
            let mut ready = 0u32;
            let mut min_days = u32::MAX;
            let mut maturing = 0u32;
            for l in [farm, gh] {
                let gr = self.world.grid(l);
                for i in 0..gr.crop.len() {
                    if gr.crop[i] != cr.id {
                        continue;
                    }
                    planted += 1;
                    if gr.soil[i] & SOIL_WATERED != 0 {
                        watered += 1;
                    }
                    let to_go = cr.grow_days.saturating_sub(gr.crop_days[i]) as u32;
                    if to_go == 0 {
                        ready += 1;
                    } else {
                        min_days = min_days.min(to_go);
                        if l == gh || to_go <= days_left {
                            maturing += 1;
                        }
                    }
                }
            }
            out[b] = frac(planted, 64);
            out[b + 1] = if planted > 0 { frac(watered, planted) } else { 0.0 };
            out[b + 2] = frac(ready, 64);
            out[b + 3] = if min_days == u32::MAX { 1.0 } else { frac(min_days, 24) };
            out[b + 4] = frac(maturing, 64);
        }
        // machines
        o = g("machines");
        for m in &self.machines {
            let b = o + m.kind as usize * 3;
            if m.has_output {
                out[b + 2] += 0.25;
            } else if m.ready_at != 0 {
                out[b + 1] += 0.25;
            } else {
                out[b] += 0.25;
            }
        }
        for x in out[o..o + c.machines.len() * 3].iter_mut() {
            *x = x.min(1.0);
        }
        // world
        o = g("world");
        for (i, fl) in c.forage.iter().enumerate() {
            out[o + i] = frac(self.count_kind(fl.location, Obj::Forage), fl.daily_cap as u32);
        }
        let b = o + c.forage.len();
        for s in &c.shops {
            out[b + s.id as usize] = self.shop_open(s.id) as u8 as f32;
        }
        let b = b + c.shops.len();
        out[b] = merchant_today as u8 as f32;
        out[b + 1] = self.merchant_open() as u8 as f32;
    }

    /// Write the egocentric grid (`GRID_CHANNELS × (2r+1) × (2r+1)`, channel-major).
    pub fn write_grid(&self, radius: u8, out: &mut [u8]) {
        let r = radius as i32;
        let side = (2 * r + 1) as usize;
        let g = self.grid();
        let (px, py) = (self.player.x as i32, self.player.y as i32);
        for ch in 0..GRID_CHANNELS {
            for dy in -r..=r {
                for dx in -r..=r {
                    let x = px + dx;
                    let y = py + dy;
                    let oi = ch * side * side + (dy + r) as usize * side + (dx + r) as usize;
                    out[oi] = if !g.in_bounds(x, y) {
                        0
                    } else {
                        let i = g.idx(x as u8, y as u8);
                        match ch {
                            0 => g.terrain_plane()[i],
                            1 => g.obj_plane()[i],
                            2 => (g.data[i] & 0xff) as u8,
                            3 => g.soil[i] & (SOIL_TILLED | SOIL_WATERED | crate::world::SOIL_FERT),
                            4 => {
                                if g.crop[i] == NO_CROP {
                                    0
                                } else {
                                    g.crop[i] + 1
                                }
                            }
                            5 => g.crop_days[i],
                            _ => (dx == 0 && dy == 0) as u8,
                        }
                    };
                }
            }
        }
    }
}

/// Tool index helper for tests.
pub fn tool_index(t: Tool) -> usize {
    t as usize
}
