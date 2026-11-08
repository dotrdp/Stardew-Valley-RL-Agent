//! Text views of the state: `describe()` (compact, deterministic, for LLM agents) and an ASCII
//! renderer of the current location.

use std::fmt::Write;

use crate::content::{content, Category, Feature, Skill, Tool};
use crate::state::GameState;
use crate::world::{Obj, Terrain, NO_CROP, SOIL_TILLED, SOIL_WATERED};

fn hhmm(m: u16) -> String {
    format!("{:02}:{:02}", (m / 60) % 24, m % 60)
}

impl GameState {
    /// Compact, deterministic text description of the state (≈ 300–900 tokens).
    pub fn describe(&self) -> String {
        let c = content();
        let mut s = String::with_capacity(2048);
        let p = &self.player;
        let _ = writeln!(
            s,
            "Day {} ({} {}/{}, year {}), {}, weather {} (tomorrow {}).",
            self.day + 1,
            self.season().name(),
            self.day_of_season() + 1,
            c.world.season_days,
            self.year() + 1,
            hhmm(self.minute),
            self.weather.name(),
            self.forecast.name()
        );
        let _ = writeln!(
            s,
            "You are at {} ({},{}). Energy {}/{}. Money {}. Bag {}/{} slots free.",
            c.locations[p.loc as usize].name,
            p.x,
            p.y,
            p.energy.max(0),
            p.energy_max,
            p.money,
            self.free_slots(),
            c.world.backpack_slots
        );
        let tools: Vec<String> = Tool::ALL
            .iter()
            .map(|t| {
                let up = if p.upgrading.map(|(u, _)| u == *t).unwrap_or(false) { "*" } else { "" };
                format!("{}{}{}", t.name(), p.tool_level[*t as usize], up)
            })
            .collect();
        let skills: Vec<String> = Skill::ALL.iter().map(|k| format!("{} {}", k.name(), self.level(*k))).collect();
        let _ = writeln!(s, "Tools: {}. Skills: {}.", tools.join(" "), skills.join(", "));
        let feats: Vec<&str> = Feature::ALL.iter().filter(|f| self.has_feature(**f)).map(|f| f.name()).collect();
        if !feats.is_empty() {
            let _ = writeln!(s, "Unlocked: {}.", feats.join(", "));
        }
        // inventory (aggregated by item)
        let mut inv: Vec<(u16, u32)> = Vec::new();
        for (it, n) in p.bag.iter() {
            if *n == 0 {
                continue;
            }
            match inv.iter_mut().find(|x| x.0 == *it) {
                Some(x) => x.1 += *n as u32,
                None => inv.push((*it, *n as u32)),
            }
        }
        inv.sort();
        let inv_s: Vec<String> = inv.iter().map(|(i, n)| format!("{} x{}", c.items[*i as usize].key, n)).collect();
        let _ = writeln!(s, "Bag: {}.", if inv_s.is_empty() { "empty".to_string() } else { inv_s.join(", ") });
        // farm
        let mut crops: Vec<(u8, u32, u32, u32)> = Vec::new(); // crop, planted, ready, unwatered
        for l in [c.world.start_location, c.loc_id("greenhouse")] {
            let g = self.world.grid(l);
            for i in 0..g.crop.len() {
                let cr = g.crop[i];
                if cr == NO_CROP {
                    continue;
                }
                let ready = g.crop_days[i] >= c.crops[cr as usize].grow_days;
                let unwatered = g.soil[i] & SOIL_TILLED != 0 && g.soil[i] & SOIL_WATERED == 0;
                match crops.iter_mut().find(|x| x.0 == cr) {
                    Some(x) => {
                        x.1 += 1;
                        x.2 += ready as u32;
                        x.3 += unwatered as u32;
                    }
                    None => crops.push((cr, 1, ready as u32, unwatered as u32)),
                }
            }
        }
        if !crops.is_empty() {
            crops.sort();
            let cs: Vec<String> = crops
                .iter()
                .map(|(cr, n, r, u)| format!("{} {} ({} ready, {} dry)", c.crops[*cr as usize].key, n, r, u))
                .collect();
            let _ = writeln!(s, "Crops: {}.", cs.join(", "));
        }
        if !self.machines.is_empty() {
            let ms: Vec<String> = self
                .machines
                .iter()
                .map(|m| {
                    let k = &c.machines[m.kind as usize].key;
                    if m.has_output {
                        format!("{k}: {} ready", c.items[m.output as usize].key)
                    } else if m.ready_at != 0 {
                        format!("{k}: {} in {}m", c.items[m.output as usize].key, m.ready_at.saturating_sub(self.abs_minute))
                    } else {
                        format!("{k}: idle")
                    }
                })
                .collect();
            let _ = writeln!(s, "Machines: {}.", ms.join("; "));
        }
        if self.merchant_today() {
            let st: Vec<String> = self
                .merchant_stock
                .iter()
                .map(|i| format!("{} ({}g)", c.items[*i as usize].key, c.merchant_price(*i)))
                .collect();
            let _ = writeln!(s, "Merchant in Hollow Fork today: {}.", st.join(", "));
        }
        if self.player.loc == c.loc_id("mines") {
            let _ = writeln!(s, "Mine floor {} (deepest {}).", self.mine.floor, self.mine.deepest);
        } else if self.mine.deepest > 0 {
            let _ = writeln!(s, "Deepest mine floor reached: {}.", self.mine.deepest);
        }
        // restoration
        let r = &self.restoration;
        let _ = writeln!(
            s,
            "{}: {}/{} wings, {}/{} requests, {}/{} slots.",
            c.restoration_name,
            r.wings_done,
            c.wings.len(),
            r.requests_done,
            c.requests.len(),
            r.slots_filled,
            c.total_slots
        );
        for w in &c.wings {
            if r.wing_done[w.id as usize] {
                continue;
            }
            let _ = writeln!(s, "- {}:", w.name);
            for rq in &w.requests {
                let req = &c.requests[*rq as usize];
                if r.request_done[*rq as usize] {
                    continue;
                }
                if req.coins > 0 {
                    let _ = writeln!(s, "  * {} — pay {} coins", req.name, req.coins);
                    continue;
                }
                let range = self.slots_of(req.id);
                let items: Vec<String> = range
                    .clone()
                    .enumerate()
                    .map(|(k, i)| {
                        let it = r.slot_item[i];
                        let mark = if r.slot_filled[i] { "✓" } else { "" };
                        format!("{}{} x{} (have {})", mark, c.items[it as usize].key, req.slots[k].count, self.count(it))
                    })
                    .collect();
                let _ = writeln!(s, "  * {} — {}/{} of: {}", req.name, self.filled_in(req.id), req.need, items.join(", "));
            }
        }
        s
    }

    /// ASCII map of the current location (with the player as `@`).
    pub fn render_ascii(&self) -> String {
        let g = self.grid();
        let mut s = String::with_capacity((g.w as usize + 1) * g.h as usize + 64);
        for y in 0..g.h {
            for x in 0..g.w {
                if (x, y) == (self.player.x, self.player.y) {
                    s.push('@');
                    continue;
                }
                let i = g.idx(x, y);
                let ch = match g.obj(x, y) {
                    Obj::None => {
                        if g.crop[i] != NO_CROP {
                            if g.crop_days[i] >= content().crops[g.crop[i] as usize].grow_days {
                                '*'
                            } else {
                                '"'
                            }
                        } else if g.soil[i] & SOIL_TILLED != 0 {
                            if g.soil[i] & SOIL_WATERED != 0 {
                                '='
                            } else {
                                '-'
                            }
                        } else {
                            match g.terrain(x, y) {
                                Terrain::Void => ' ',
                                Terrain::Grass => '.',
                                Terrain::Dirt => ',',
                                Terrain::Path => ':',
                                Terrain::Sand => '_',
                                Terrain::Floor => ' ',
                                Terrain::Water => '~',
                                Terrain::Wall => '#',
                                Terrain::Bridge => '=',
                            }
                        }
                    }
                    Obj::Weeds => 'w',
                    Obj::Twig => 't',
                    Obj::StoneDebris => 's',
                    Obj::Tree => 'T',
                    Obj::Stump => 'U',
                    Obj::Boulder => 'O',
                    Obj::Rock => 'o',
                    Obj::Forage => 'f',
                    Obj::Building => 'B',
                    Obj::Counter => '$',
                    Obj::Bed => 'b',
                    Obj::Crate => 'c',
                    Obj::Well => 'W',
                    Obj::MillBoard => 'M',
                    Obj::Ladder => 'H',
                    Obj::Elevator => 'E',
                    Obj::MachinePad => 'p',
                    Obj::Machine => 'm',
                    Obj::Warp => '>',
                    Obj::Stall => 'S',
                    Obj::MineExit => '^',
                    Obj::MineEntrance => 'v',
                };
                s.push(ch);
            }
            s.push('\n');
        }
        s
    }
}

/// Category name helper for text views.
pub fn category_name(cat: Category) -> &'static str {
    cat.name()
}
