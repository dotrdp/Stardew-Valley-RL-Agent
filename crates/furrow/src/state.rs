//! Full game state and its systems: calendar/day-end, inventory, skills, machines, shipping,
//! restoration progress and rewards (DESIGN.md §2).

use crate::content::{
    content, Category, Content, Dir, Feature, ItemId, LocId, LocKind, MachineId, RequestId, Reward,
    Season, ShopId, Skill, Tool, Weather, WingId,
};
use crate::hash::StableHasher;
use crate::rng::{stream, Domain, Pcg32};
use crate::world::{self, Grid, Obj, PathScratch, Routes, World, NO_CROP, SOIL_WATERED};

/// Backpack capacity (fixed by content; mirrored here as a const for array sizing).
pub const BAG: usize = 24;

/// What the player currently has "in hand".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sel {
    /// A backpack slot.
    Slot(u8),
    /// A tool.
    Tool(Tool),
}

/// Open menu.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Menu {
    None,
    Shop(ShopId),
    Merchant,
    Craft,
    Board,
    Elevator,
}

impl Menu {
    /// Dense index for observations (0 = none).
    pub fn index(self) -> u8 {
        match self {
            Menu::None => 0,
            Menu::Shop(_) => 1,
            Menu::Merchant => 2,
            Menu::Craft => 3,
            Menu::Board => 4,
            Menu::Elevator => 5,
        }
    }
}

/// The farmer.
#[derive(Clone, Debug)]
pub struct Player {
    pub loc: LocId,
    pub x: u8,
    pub y: u8,
    pub facing: Dir,
    pub energy: i32,
    pub energy_max: u16,
    pub money: u32,
    /// `(item, count)`; `count == 0` means empty.
    pub bag: [(ItemId, u16); BAG],
    pub sel: Sel,
    pub tool_level: [u8; 5],
    /// `(tool, ready_day)` while an upgrade is in progress.
    pub upgrading: Option<(Tool, u16)>,
    pub xp: [u32; 4],
    pub can_water: u16,
}

/// A placed machine.
#[derive(Clone, Copy, Debug)]
pub struct MachineInst {
    pub kind: MachineId,
    pub loc: LocId,
    pub x: u8,
    pub y: u8,
    /// Output item when working/ready.
    pub output: ItemId,
    pub out_count: u16,
    /// Absolute minute at which the output is ready (0 = idle).
    pub ready_at: u32,
    /// Output is ready to collect.
    pub has_output: bool,
}

/// Restoration progress.
#[derive(Clone, Debug)]
pub struct Restoration {
    /// Item of each slot (flat over requests; remixed in Remixed mode).
    pub slot_item: Vec<ItemId>,
    pub slot_filled: Vec<bool>,
    /// First flat slot index of each request.
    pub slot_offset: Vec<u16>,
    pub request_done: Vec<bool>,
    pub wing_done: Vec<bool>,
    pub coins_paid: Vec<u32>,
    pub slots_filled: u16,
    pub requests_done: u8,
    pub wings_done: u8,
}

/// Mine state.
#[derive(Clone, Copy, Debug, Default)]
pub struct MineState {
    /// Current floor (0 = not in the mines).
    pub floor: u8,
    pub deepest: u8,
    /// Day the current floor grid was generated for.
    pub floor_day: u16,
}

/// Outcome flags of the most recent action, for observations.
#[derive(Clone, Copy, Debug, Default)]
pub struct LastEvent {
    pub failed: bool,
    pub passed_out: bool,
    pub caught: Option<ItemId>,
}

/// The whole game state.
#[derive(Clone, Debug)]
pub struct GameState {
    pub seed: u64,
    pub day: u16,
    /// Minute of day (`day_start..=day_end`).
    pub minute: u16,
    pub abs_minute: u32,
    pub weather: Weather,
    pub forecast: Weather,
    pub player: Player,
    pub world: World,
    pub machines: Vec<MachineInst>,
    pub crate_items: Vec<(ItemId, u16)>,
    pub restoration: Restoration,
    /// Unlocked features bitmask (bit = `Feature as u8`).
    pub features: u8,
    pub menu: Menu,
    pub merchant_stock: [ItemId; 4],
    pub mine: MineState,
    pub dyn_rng: Pcg32,
    pub steps: u32,
    pub decisions_today: u16,
    pub last: LastEvent,
    /// All wings complete.
    pub complete: bool,
    pub scratch: PathScratch,
    pub routes: Routes,
    /// Per-step cache of ETAs to every POI (see `eta_cached`).
    pub eta_cache: Vec<u32>,
    /// Cache key: (steps, loc, x, y, abs_minute).
    pub eta_key: (u32, u8, u8, u8, u32),
}

impl GameState {
    /// Fresh episode.
    pub fn new(seed: u64, remixed: bool, test_partition: bool, start_money: u32) -> GameState {
        let c = content();
        let world = World::generate(c, seed);
        let (sx, sy) = c.world.start_tile;
        let slot_item = if remixed {
            world::remix_slots(c, seed, test_partition)
        } else {
            c.requests
                .iter()
                .flat_map(|r| r.slots.iter().map(|s| s.item))
                .collect()
        };
        let mut slot_offset = Vec::with_capacity(c.requests.len());
        let mut acc = 0u16;
        for r in &c.requests {
            slot_offset.push(acc);
            acc += r.slots.len() as u16;
        }
        let mut st = GameState {
            seed,
            day: 0,
            minute: c.world.day_start,
            abs_minute: c.world.day_start as u32,
            weather: Weather::Sunny,
            forecast: Weather::Sunny,
            player: Player {
                loc: c.world.start_location,
                x: sx,
                y: sy,
                facing: Dir::S,
                energy: c.world.energy_max as i32,
                energy_max: c.world.energy_max,
                money: start_money,
                bag: [(0, 0); BAG],
                sel: Sel::Tool(Tool::Hoe),
                tool_level: [0; 5],
                upgrading: None,
                xp: [0; 4],
                can_water: c.tools.can_capacity[0],
            },
            world,
            machines: Vec::new(),
            crate_items: Vec::new(),
            restoration: Restoration {
                slot_filled: vec![false; slot_item.len()],
                slot_item,
                slot_offset,
                request_done: vec![false; c.requests.len()],
                wing_done: vec![false; c.wings.len()],
                coins_paid: vec![0; c.requests.len()],
                slots_filled: 0,
                requests_done: 0,
                wings_done: 0,
            },
            features: 0,
            menu: Menu::None,
            merchant_stock: [0; 4],
            mine: MineState::default(),
            dyn_rng: world::dynamics_rng(seed),
            steps: 0,
            decisions_today: 0,
            last: LastEvent::default(),
            complete: false,
            scratch: PathScratch::default(),
            routes: Routes::build(c),
            eta_cache: Vec::new(),
            eta_key: (u32::MAX, 0, 0, 0, 0),
        };
        st.start_day();
        st
    }

    /// Content shortcut.
    #[inline]
    pub fn c(&self) -> &'static Content {
        content()
    }

    // ------------------------------------------------------------------ calendar

    /// Season of a day index.
    pub fn season_of(day: u16) -> Season {
        let sd = content().world.season_days as u16;
        Season::from_index(((day / sd) % 4) as u8)
    }
    /// Current season.
    pub fn season(&self) -> Season {
        Self::season_of(self.day)
    }
    /// Day within the season (0-based).
    pub fn day_of_season(&self) -> u16 {
        self.day % self.c().world.season_days as u16
    }
    /// Days left in the current season after today.
    pub fn days_left_in_season(&self) -> u16 {
        self.c().world.season_days as u16 - 1 - self.day_of_season()
    }
    /// Weekday 0..7.
    pub fn weekday(&self) -> u8 {
        (self.day % 7) as u8
    }
    /// Year (0-based).
    pub fn year(&self) -> u16 {
        self.day / (self.c().world.season_days as u16 * 4)
    }
    /// Exogenous weather roll for a day.
    pub fn roll_weather(seed: u64, day: u16) -> Weather {
        let c = content();
        let mut r = stream(seed, Domain::Weather, day as u32, 0);
        let s = Self::season_of(day);
        match r.weighted(&c.world.weather_ppm[s as usize]) {
            0 => Weather::Sunny,
            1 => Weather::Rainy,
            2 => Weather::Stormy,
            _ => Weather::Snowy,
        }
    }
    /// Is the merchant in town today?
    pub fn merchant_today(&self) -> bool {
        self.has_feature(Feature::MerchantDaily)
            || self.c().merchant.weekdays.contains(&self.weekday())
    }
    /// Shop open now? (hours and weekday)
    pub fn shop_open(&self, shop: ShopId) -> bool {
        let s = &self.c().shops[shop as usize];
        self.shop_open_at(shop, self.minute) && s.closed_weekday != self.weekday() as i8
    }
    /// Shop open at a given minute today (ignores weekday).
    pub fn shop_open_at(&self, shop: ShopId, minute: u16) -> bool {
        let s = &self.c().shops[shop as usize];
        minute >= s.open && minute < s.close
    }
    /// Merchant open now?
    pub fn merchant_open(&self) -> bool {
        let m = &self.c().merchant;
        self.merchant_today() && self.minute >= m.open && self.minute < m.close
    }

    fn start_day(&mut self) {
        let c = self.c();
        self.weather = Self::roll_weather(self.seed, self.day);
        self.forecast = Self::roll_weather(self.seed, self.day + 1);
        self.decisions_today = 0;
        // rain waters outdoor soil
        if self.weather.is_rain() {
            let farm = c.world.start_location;
            let g = self.world.grid_mut(farm);
            for s in g.soil.iter_mut() {
                if *s & world::SOIL_TILLED != 0 {
                    *s |= SOIL_WATERED;
                }
            }
        }
        // forage
        let season = self.season();
        for fl in &c.forage {
            let g = self.world.grid_mut(fl.location);
            world::spawn_forage(c, self.seed, self.day as u32, season, g, fl);
        }
        // surface rocks
        let quarry = c.loc_id("quarry");
        world::respawn_surface_rocks(
            c,
            self.seed,
            self.day as u32,
            self.world.grid_mut(quarry),
            quarry,
        );
        // merchant stock
        if self.merchant_today() {
            let mut r = stream(self.seed, Domain::Merchant, self.day as u32, 0);
            let mut pool = c.merchant.pool.clone();
            r.shuffle(&mut pool);
            for i in 0..4 {
                self.merchant_stock[i] = pool[i % pool.len()];
            }
        }
        // tool upgrade completion
        if let Some((t, ready)) = self.player.upgrading {
            if self.day >= ready {
                self.player.tool_level[t as usize] =
                    (self.player.tool_level[t as usize] + 1).min(3);
                self.player.upgrading = None;
                if t == Tool::Can {
                    self.player.can_water =
                        c.tools.can_capacity[self.player.tool_level[Tool::Can as usize] as usize];
                }
            }
        }
    }

    /// Advance the clock by `minutes` within the day. Returns 1 if the day ended (pass-out).
    pub fn advance(&mut self, minutes: u32) -> u8 {
        let c = self.c();
        let m = minutes.min(10_000) as u16;
        self.minute = self.minute.saturating_add(m);
        self.abs_minute += minutes;
        if self.minute >= c.world.day_end {
            self.end_day(true);
            return 1;
        }
        0
    }

    /// End the day (sleep or pass-out). Returns the absolute minutes skipped.
    pub fn end_day(&mut self, passed_out: bool) -> u32 {
        let c = self.c();
        let before = self.abs_minute;
        // --- overnight pipeline
        self.grow_crops();
        self.pay_crate();
        self.menu = Menu::None;
        self.mine = MineState {
            floor: 0,
            deepest: self.mine.deepest,
            floor_day: 0,
        };
        if self.player.loc == c.loc_id("mines") {
            // kicked out to the quarry entrance
            let p = &c.pois[c.poi_id("mine_entrance") as usize];
            self.player.loc = p.location;
            self.player.x = p.stand.0;
            self.player.y = p.stand.1;
        }
        // wake up in bed
        let bed = &c.pois[c.poi_id("bed") as usize];
        self.player.loc = bed.location;
        self.player.x = bed.stand.0;
        self.player.y = bed.stand.1;
        self.player.facing = Dir::S;
        if passed_out {
            self.player.energy =
                (self.player.energy_max as u32 * c.world.passout_energy_ppm / 1_000_000) as i32;
            let pen =
                ((self.player.money as u64 * c.world.passout_money_ppm as u64) / 1_000_000) as u32;
            self.player.money -= pen.min(c.world.passout_money_cap);
            self.last.passed_out = true;
        } else {
            self.player.energy = self.player.energy_max as i32;
        }
        // next day
        let next_day = self.day + 1;
        let next_abs = next_day as u32 * 1440 + c.world.day_start as u32;
        self.day = next_day;
        self.minute = c.world.day_start;
        self.abs_minute = next_abs.max(before + 1);
        // season change kills out-of-season outdoor crops
        if self.day_of_season() == 0 {
            let season = self.season();
            let farm = c.world.start_location;
            let g = self.world.grid_mut(farm);
            for i in 0..g.crop.len() {
                let cr = g.crop[i];
                if cr != NO_CROP && !crate::content::season_in(c.crops[cr as usize].seasons, season)
                {
                    g.crop[i] = NO_CROP;
                    g.crop_days[i] = 0;
                }
            }
        }
        self.start_day();
        self.abs_minute - before
    }

    fn grow_crops(&mut self) {
        let c = self.c();
        for l in [c.world.start_location, c.loc_id("greenhouse")] {
            let gh = c.locations[l as usize].kind == LocKind::Greenhouse;
            let g = self.world.grid_mut(l);
            for i in 0..g.crop.len() {
                if g.crop[i] == NO_CROP {
                    g.soil[i] &= !SOIL_WATERED;
                    continue;
                }
                if gh || g.soil[i] & SOIL_WATERED != 0 {
                    g.crop_days[i] = g.crop_days[i].saturating_add(1);
                }
                g.soil[i] &= !SOIL_WATERED;
            }
        }
    }

    fn pay_crate(&mut self) {
        let c = self.c();
        let mut total = 0u32;
        for (item, n) in self.crate_items.drain(..) {
            total += c.items[item as usize].sell * n as u32;
        }
        self.player.money = self.player.money.saturating_add(total);
    }

    // ------------------------------------------------------------------ inventory

    /// Count of an item in the bag.
    pub fn count(&self, item: ItemId) -> u32 {
        self.player
            .bag
            .iter()
            .filter(|(i, n)| *i == item && *n > 0)
            .map(|(_, n)| *n as u32)
            .sum()
    }
    /// Number of empty slots.
    pub fn free_slots(&self) -> u8 {
        self.player.bag.iter().filter(|(_, n)| *n == 0).count() as u8
    }
    /// Can `n` of `item` be added?
    pub fn can_add(&self, item: ItemId, n: u16) -> bool {
        let max = self.c().world.stack_max;
        let mut room: u32 = self
            .player
            .bag
            .iter()
            .filter(|(i, c)| *i == item && *c > 0)
            .map(|(_, c)| (max - *c) as u32)
            .sum();
        room += self.free_slots() as u32 * max as u32;
        room >= n as u32
    }
    /// Add items; returns the leftover that did not fit.
    pub fn add_item(&mut self, item: ItemId, mut n: u16) -> u16 {
        let max = self.c().world.stack_max;
        for slot in self.player.bag.iter_mut() {
            if n == 0 {
                break;
            }
            if slot.0 == item && slot.1 > 0 && slot.1 < max {
                let take = (max - slot.1).min(n);
                slot.1 += take;
                n -= take;
            }
        }
        for slot in self.player.bag.iter_mut() {
            if n == 0 {
                break;
            }
            if slot.1 == 0 {
                let take = max.min(n);
                *slot = (item, take);
                n -= take;
            }
        }
        n
    }
    /// Remove up to `n` of `item`; returns how many were removed.
    pub fn remove_item(&mut self, item: ItemId, mut n: u16) -> u16 {
        let want = n;
        for slot in self.player.bag.iter_mut() {
            if n == 0 {
                break;
            }
            if slot.0 == item && slot.1 > 0 {
                let take = slot.1.min(n);
                slot.1 -= take;
                n -= take;
                if slot.1 == 0 {
                    *slot = (0, 0);
                }
            }
        }
        want - n
    }
    /// Slot index holding an item (first), if any.
    pub fn slot_of(&self, item: ItemId) -> Option<u8> {
        self.player
            .bag
            .iter()
            .position(|(i, n)| *i == item && *n > 0)
            .map(|p| p as u8)
    }
    /// Item in the selected slot, if the selection is a non-empty slot.
    pub fn selected_item(&self) -> Option<(ItemId, u16)> {
        match self.player.sel {
            Sel::Slot(i) => {
                let (it, n) = self.player.bag[i as usize];
                if n > 0 {
                    Some((it, n))
                } else {
                    None
                }
            }
            Sel::Tool(_) => None,
        }
    }

    // ------------------------------------------------------------------ skills / energy

    /// Skill level from XP.
    pub fn level(&self, s: Skill) -> u8 {
        let xp = self.player.xp[s as usize];
        self.c()
            .skills
            .thresholds
            .iter()
            .filter(|t| xp >= **t)
            .count() as u8
    }
    /// Add XP.
    pub fn add_xp(&mut self, s: Skill, xp: u16) {
        self.player.xp[s as usize] = self.player.xp[s as usize].saturating_add(xp as u32);
    }
    /// Energy cost of a base amount for a skill (level reduction, floor 60%).
    pub fn energy_cost(&self, base: u16, s: Skill) -> i32 {
        let red = (self.c().skills.energy_cost_reduction_ppm_per_level as u64
            * self.level(s) as u64)
            .min(400_000);
        let cost = base as u64 * (1_000_000 - red) / 1_000_000;
        (cost.max(1)) as i32
    }
    /// Spend energy; returns true if the player passed out as a result (caller must end the day).
    pub fn spend_energy(&mut self, cost: i32) -> bool {
        self.player.energy -= cost;
        self.player.energy <= 0
    }
    /// Feature unlocked?
    pub fn has_feature(&self, f: Feature) -> bool {
        self.features & (1 << (f as u8)) != 0
    }
    /// Recipe unlocked?
    pub fn recipe_unlocked(&self, r: crate::content::RecipeId) -> bool {
        match self.c().recipes[r as usize].unlock {
            crate::content::Unlock::Start => true,
            crate::content::Unlock::Skill(s, l) => self.level(s) >= l,
            crate::content::Unlock::Wing(w) => self.restoration.wing_done[w as usize],
        }
    }

    // ------------------------------------------------------------------ restoration

    /// Flat slot range of a request.
    pub fn slots_of(&self, r: RequestId) -> std::ops::Range<usize> {
        let c = self.c();
        let start = self.restoration.slot_offset[r as usize] as usize;
        start..start + c.requests[r as usize].slots.len()
    }
    /// Number of filled slots in a request.
    pub fn filled_in(&self, r: RequestId) -> u8 {
        self.slots_of(r)
            .filter(|i| self.restoration.slot_filled[*i])
            .count() as u8
    }
    /// Is a request still open?
    pub fn request_open(&self, r: RequestId) -> bool {
        !self.restoration.request_done[r as usize]
    }
    /// Units of `item` still useful for open requests (all unfilled candidate slots).
    pub fn need(&self, item: ItemId) -> u32 {
        let c = self.c();
        let mut n = 0u32;
        for r in &c.requests {
            if self.restoration.request_done[r.id as usize] {
                continue;
            }
            let rng = self.slots_of(r.id);
            for (k, i) in rng.enumerate() {
                if !self.restoration.slot_filled[i] && self.restoration.slot_item[i] == item {
                    n += r.slots[k].count as u32;
                }
            }
        }
        n
    }
    /// Deliver to a request: fills every unfilled slot whose item we hold enough of, until the
    /// request's `need` is met. Returns the number of slots filled.
    pub fn deliver(&mut self, r: RequestId) -> u8 {
        let c = self.c();
        if self.restoration.request_done[r as usize] {
            return 0;
        }
        let req = &c.requests[r as usize];
        let mut filled = 0u8;
        let range = self.slots_of(r);
        // prefer slots whose item is scarce (needed by fewer other open slots)
        let mut order: Vec<usize> = range.clone().collect();
        order.sort_by_key(|i| {
            let item = self.restoration.slot_item[*i];
            (self.open_slots_for(item), *i)
        });
        for i in order {
            if self.filled_in(r) >= req.need {
                break;
            }
            if self.restoration.slot_filled[i] {
                continue;
            }
            let k = i - range.start;
            let item = self.restoration.slot_item[i];
            let cnt = req.slots[k].count;
            if self.count(item) >= cnt as u32 {
                self.remove_item(item, cnt);
                self.restoration.slot_filled[i] = true;
                self.restoration.slots_filled += 1;
                filled += 1;
            }
        }
        if filled > 0 {
            self.check_request(r);
        }
        filled
    }
    /// Number of open (unfilled, in open requests) slots wanting an item.
    pub fn open_slots_for(&self, item: ItemId) -> u32 {
        let c = self.c();
        let mut n = 0;
        for r in &c.requests {
            if self.restoration.request_done[r.id as usize] {
                continue;
            }
            for i in self.slots_of(r.id) {
                if !self.restoration.slot_filled[i] && self.restoration.slot_item[i] == item {
                    n += 1;
                }
            }
        }
        n
    }
    /// Pay a coin request. Returns true if paid.
    pub fn pay_coins(&mut self, r: RequestId) -> bool {
        let c = self.c();
        let req = &c.requests[r as usize];
        if req.coins == 0
            || self.restoration.request_done[r as usize]
            || self.player.money < req.coins
        {
            return false;
        }
        self.player.money -= req.coins;
        self.restoration.coins_paid[r as usize] = req.coins;
        self.check_request(r);
        true
    }
    fn check_request(&mut self, r: RequestId) {
        let c = self.c();
        let req = &c.requests[r as usize];
        let done = if req.coins > 0 {
            self.restoration.coins_paid[r as usize] >= req.coins
        } else {
            self.filled_in(r) >= req.need
        };
        if done && !self.restoration.request_done[r as usize] {
            self.restoration.request_done[r as usize] = true;
            self.restoration.requests_done += 1;
            let w = req.wing;
            if c.wings[w as usize]
                .requests
                .iter()
                .all(|q| self.restoration.request_done[*q as usize])
            {
                self.restoration.wing_done[w as usize] = true;
                self.restoration.wings_done += 1;
                self.grant_reward(w);
                if self.restoration.wings_done as usize == c.wings.len() {
                    self.complete = true;
                }
            }
        }
    }
    fn grant_reward(&mut self, w: WingId) {
        let c = self.c();
        match &c.wings[w as usize].reward {
            Reward::Unlock(f) => self.features |= 1 << (*f as u8),
            Reward::Energy(e) => {
                self.player.energy_max += e;
                self.player.energy += *e as i32;
            }
            Reward::ToolLevel(tools) => {
                for t in tools {
                    let l = &mut self.player.tool_level[*t as usize];
                    *l = (*l + 1).min(3);
                }
            }
            Reward::Money(m) => self.player.money = self.player.money.saturating_add(*m),
        }
    }

    // ------------------------------------------------------------------ machines

    /// Machine at a tile.
    pub fn machine_at(&self, loc: LocId, x: u8, y: u8) -> Option<usize> {
        self.machines
            .iter()
            .position(|m| m.loc == loc && m.x == x && m.y == y)
    }
    /// Update machine readiness from the clock.
    pub fn tick_machines(&mut self) {
        let now = self.abs_minute;
        for m in self.machines.iter_mut() {
            if m.ready_at != 0 && !m.has_output && now >= m.ready_at {
                m.has_output = true;
            }
        }
    }

    // ------------------------------------------------------------------ geometry helpers

    /// Current grid.
    #[inline]
    pub fn grid(&self) -> &Grid {
        self.world.grid(self.player.loc)
    }
    /// Tile in front of the player (may be out of bounds).
    pub fn facing_tile(&self) -> (i32, i32) {
        let (dx, dy) = self.player.facing.delta();
        (self.player.x as i32 + dx, self.player.y as i32 + dy)
    }
    /// Object in front of the player.
    pub fn facing_obj(&self) -> Obj {
        let (fx, fy) = self.facing_tile();
        let g = self.grid();
        if g.in_bounds(fx, fy) {
            g.obj(fx as u8, fy as u8)
        } else {
            Obj::None
        }
    }
    /// Is the player in the farm or greenhouse?
    pub fn on_field_location(&self) -> bool {
        let k = self.c().locations[self.player.loc as usize].kind;
        matches!(k, LocKind::Farm | LocKind::Greenhouse)
    }
    /// Can a crop be planted here now (season check; greenhouse ignores season)?
    pub fn crop_plantable(&self, crop: crate::content::CropId) -> bool {
        let c = self.c();
        let k = c.locations[self.player.loc as usize].kind;
        match k {
            LocKind::Greenhouse => true,
            LocKind::Farm => {
                crate::content::season_in(c.crops[crop as usize].seasons, self.season())
            }
            _ => false,
        }
    }
    /// Items the shipping crate would pay for (items with sell > 0 that are not seeds/machines).
    pub fn shippable(&self, item: ItemId) -> bool {
        let it = &self.c().items[item as usize];
        it.sell > 0 && !matches!(it.category, Category::Seed | Category::Machine)
    }

    /// Stable hash of the whole state (DESIGN.md §2.5).
    pub fn stable_hash(&self) -> u64 {
        let mut h = StableHasher::new();
        h.u64(self.seed)
            .u16(self.day)
            .u16(self.minute)
            .u32(self.abs_minute);
        h.u8(self.weather as u8).u8(self.forecast as u8);
        let p = &self.player;
        h.u8(p.loc)
            .u8(p.x)
            .u8(p.y)
            .u8(p.facing as u8)
            .i32(p.energy)
            .u16(p.energy_max)
            .u32(p.money);
        for (i, n) in p.bag {
            h.u16(i).u16(n);
        }
        match p.sel {
            Sel::Slot(i) => h.u8(0).u8(i),
            Sel::Tool(t) => h.u8(1).u8(t as u8),
        };
        h.u8s(&p.tool_level);
        match p.upgrading {
            Some((t, d)) => h.u8(1).u8(t as u8).u16(d),
            None => h.u8(0),
        };
        for x in p.xp {
            h.u32(x);
        }
        h.u16(p.can_water);
        self.world.hash_into(&mut h);
        h.u32(self.machines.len() as u32);
        for m in &self.machines {
            h.u8(m.kind)
                .u8(m.loc)
                .u8(m.x)
                .u8(m.y)
                .u16(m.output)
                .u16(m.out_count)
                .u32(m.ready_at)
                .bool(m.has_output);
        }
        h.u32(self.crate_items.len() as u32);
        for (i, n) in &self.crate_items {
            h.u16(*i).u16(*n);
        }
        h.u16s(&self.restoration.slot_item);
        for f in &self.restoration.slot_filled {
            h.bool(*f);
        }
        for f in &self.restoration.request_done {
            h.bool(*f);
        }
        for f in &self.restoration.wing_done {
            h.bool(*f);
        }
        h.u32s(&self.restoration.coins_paid);
        h.u8(self.features).u8(self.menu.index());
        h.u16s(&self.merchant_stock);
        h.u8(self.mine.floor).u8(self.mine.deepest);
        let (a, b) = self.dyn_rng.state_words();
        h.u64(a).u64(b);
        h.u32(self.steps)
            .u16(self.decisions_today)
            .bool(self.complete);
        h.finish()
    }
}
