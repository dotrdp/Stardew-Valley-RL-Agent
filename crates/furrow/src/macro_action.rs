//! Macro actions (DESIGN.md §3.4): a stable enumeration derived from content, cheap masks, and
//! deterministic controllers that execute each macro through primitive steps.

use std::sync::OnceLock;

use crate::action::Primitive;
use crate::content::{
    content, Category, CropId, Dir, Feature, ItemId, LocId, LocKind, PoiId, PoiKind, RecipeId,
    RequestId, Tool,
};
use crate::nav::{hub, outdoor, NavTables};
use crate::state::{GameState, Menu, Sel};
use crate::world::{Grid, Obj, Rect, Terrain, NO_CROP, SOIL_TILLED, SOIL_WATERED, UNREACHABLE};

/// A macro action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Macro {
    Wait30,
    Sleep,
    GoTo(LocId),
    WaterCrops,
    Harvest,
    ClearField,
    /// Plant all held seeds of a crop on the farm (`false`) or in the greenhouse (`true`).
    Plant(CropId, bool),
    /// Forage at a forage location (index into `content().forage`).
    Forage(u8),
    /// Fish for up to ~60 minutes at a spot.
    Fish(PoiId),
    ChopWood,
    ChopHardwood,
    MineDescend,
    MineFloor,
    /// Buy `q` of an item from the cheapest seller available today.
    Buy(ItemId, u8),
    ShipSurplus,
    Craft(RecipeId),
    LoadMachines,
    CollectMachines,
    UpgradeTool(Tool),
    Deliver,
    PayCoins(RequestId),
    Eat,
}

impl Macro {
    /// Human-readable, stable name.
    pub fn name(&self) -> String {
        let c = content();
        match self {
            Macro::Wait30 => "Wait30".into(),
            Macro::Sleep => "Sleep".into(),
            Macro::GoTo(l) => format!("GoTo({})", c.locations[*l as usize].key),
            Macro::WaterCrops => "WaterCrops".into(),
            Macro::Harvest => "Harvest".into(),
            Macro::ClearField => "ClearField".into(),
            Macro::Plant(cr, gh) => format!(
                "Plant({},{})",
                c.crops[*cr as usize].key,
                if *gh { "greenhouse" } else { "farm" }
            ),
            Macro::Forage(f) => format!(
                "Forage({})",
                c.locations[c.forage[*f as usize].location as usize].key
            ),
            Macro::Fish(p) => format!("Fish({})", c.pois[*p as usize].key),
            Macro::ChopWood => "ChopWood".into(),
            Macro::ChopHardwood => "ChopHardwood".into(),
            Macro::MineDescend => "MineDescend".into(),
            Macro::MineFloor => "MineFloor".into(),
            Macro::Buy(i, q) => format!("Buy({},{})", c.items[*i as usize].key, q),
            Macro::ShipSurplus => "ShipSurplus".into(),
            Macro::Craft(r) => format!("Craft({})", c.recipes[*r as usize].key),
            Macro::LoadMachines => "LoadMachines".into(),
            Macro::CollectMachines => "CollectMachines".into(),
            Macro::UpgradeTool(t) => format!("UpgradeTool({})", t.name()),
            Macro::Deliver => "Deliver".into(),
            Macro::PayCoins(r) => format!("PayCoins({})", c.requests[*r as usize].key),
            Macro::Eat => "Eat".into(),
        }
    }
}

static TABLE: OnceLock<Vec<Macro>> = OnceLock::new();

/// The macro table (ids are indices), a pure function of content.
pub fn macro_actions() -> &'static [Macro] {
    TABLE.get_or_init(|| {
        let c = content();
        let mut v = vec![Macro::Wait30, Macro::Sleep];
        for l in &c.locations {
            if l.kind != LocKind::Mines {
                v.push(Macro::GoTo(l.id));
            }
        }
        v.extend([Macro::WaterCrops, Macro::Harvest, Macro::ClearField]);
        for cr in &c.crops {
            v.push(Macro::Plant(cr.id, false));
        }
        for cr in &c.crops {
            v.push(Macro::Plant(cr.id, true));
        }
        for f in &c.forage {
            v.push(Macro::Forage(f.id));
        }
        for s in &c.fishing_spots {
            v.push(Macro::Fish(*s));
        }
        v.extend([
            Macro::ChopWood,
            Macro::ChopHardwood,
            Macro::MineDescend,
            Macro::MineFloor,
        ]);
        for i in &c.buyable {
            v.push(Macro::Buy(*i, 1));
            v.push(Macro::Buy(*i, 5));
            v.push(Macro::Buy(*i, 25));
        }
        v.push(Macro::ShipSurplus);
        for r in &c.recipes {
            v.push(Macro::Craft(r.id));
        }
        v.extend([Macro::LoadMachines, Macro::CollectMachines]);
        for t in Tool::ALL {
            v.push(Macro::UpgradeTool(t));
        }
        v.push(Macro::Deliver);
        for r in &c.coin_requests {
            v.push(Macro::PayCoins(*r));
        }
        v.push(Macro::Eat);
        v
    })
}

/// Number of macro actions.
pub fn num_macros() -> usize {
    macro_actions().len()
}

/// Id of a macro (linear search; tables are small).
pub fn macro_id(m: Macro) -> Option<u32> {
    macro_actions()
        .iter()
        .position(|x| *x == m)
        .map(|p| p as u32)
}

/// Outcome of a macro.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Outcome {
    Done = 0,
    /// Time box hit; may be re-issued.
    Partial = 1,
    Failed = 2,
    /// A masked action was stepped.
    Masked = 3,
}

/// Result of a macro.
#[derive(Clone, Copy, Debug)]
pub struct MacroResult {
    pub ticks: u32,
    pub day_ends: u8,
    pub outcome: Outcome,
}

/// Minimum ticks a macro consumes.
pub const MIN_MACRO_TICKS: u32 = 5;
/// Travel allowance on top of the activity time box.
pub const TRAVEL_CAP: u32 = 480;

/// Controller execution context.
struct Ctl<'a> {
    st: &'a mut GameState,
    budget: u32,
    /// Activity time box (minutes) applied from `start_activity()`.
    box_minutes: u32,
    abs0: u32,
    day0: u16,
    ticks: u32,
    day_ends: u8,
    progressed: bool,
    path: Vec<Dir>,
}

impl Ctl<'_> {
    #[inline]
    fn c(&self) -> &'static crate::content::Content {
        content()
    }
    /// Step one primitive; returns `ok`.
    fn prim(&mut self, p: Primitive) -> bool {
        let r = self.st.step_primitive(p);
        self.sync();
        if r.ok
            && !matches!(
                p,
                Primitive::Move(_)
                    | Primitive::Face(_)
                    | Primitive::SelectSlot(_)
                    | Primitive::SelectTool(_)
                    | Primitive::CloseMenu
                    | Primitive::Wait
            )
        {
            self.progressed = true;
        }
        r.ok
    }
    /// Recompute elapsed ticks / day boundaries from the absolute clock.
    fn sync(&mut self) {
        self.ticks = self.st.abs_minute - self.abs0;
        self.day_ends = (self.st.day - self.day0) as u8;
    }
    /// Start the activity time box (called after travelling to the activity site).
    fn start_activity(&mut self) {
        self.budget = self.ticks + self.box_minutes;
    }
    /// Stop condition: the day ended or the time box is exhausted.
    fn over(&self) -> bool {
        self.day_ends > 0 || self.ticks >= self.budget || self.st.complete
    }
    fn select_tool(&mut self, t: Tool) -> bool {
        if self.st.player.sel == Sel::Tool(t) {
            return true;
        }
        self.prim(Primitive::SelectTool(t))
    }
    fn select_item(&mut self, item: ItemId) -> bool {
        if let Sel::Slot(i) = self.st.player.sel {
            if self.st.player.bag[i as usize] != (0, 0)
                && self.st.player.bag[i as usize].0 == item
                && self.st.player.bag[i as usize].1 > 0
            {
                return true;
            }
        }
        match self.st.slot_of(item) {
            Some(i) => self.prim(Primitive::SelectSlot(i)),
            None => false,
        }
    }
    fn close_menu(&mut self) {
        if self.st.menu != Menu::None {
            self.prim(Primitive::CloseMenu);
        }
    }
    /// Walk the current `self.path`; re-plans up to twice if blocked. Returns true if finished.
    fn walk(&mut self, target: (u8, u8), adjacent: bool) -> bool {
        for _attempt in 0..3 {
            {
                let st = &mut *self.st;
                let g = st.world.grid(st.player.loc);
                st.scratch.bfs(g, st.player.x, st.player.y);
                if !st.scratch.path(g, target.0, target.1, &mut self.path) {
                    return false;
                }
            }
            let n = self.path.len();
            if n == 0 {
                return true;
            }
            let steps = if adjacent { n - 1 } else { n };
            let mut blocked = false;
            let dirs: Vec<Dir> = self.path[..steps].to_vec();
            for d in dirs {
                if self.over() {
                    return false;
                }
                let before = (self.st.player.loc, self.st.player.x, self.st.player.y);
                self.prim(Primitive::Move(d));
                let after = (self.st.player.loc, self.st.player.x, self.st.player.y);
                if before == after {
                    blocked = true;
                    break;
                }
                if after.0 != before.0 {
                    return true; // warped / descended
                }
            }
            if blocked {
                continue;
            }
            if adjacent {
                let last = self.path[n - 1];
                if self.st.player.facing != last {
                    self.prim(Primitive::Face(last));
                }
            }
            return true;
        }
        false
    }
    /// Navigate to a location (its hub) if not already there.
    fn goto_loc(&mut self, loc: LocId) -> bool {
        let c = self.c();
        let mines = c.loc_id("mines");
        if self.st.player.loc == mines && loc != mines {
            let exit = &c.pois[c.poi_id("mine_exit") as usize];
            if !self.walk(exit.stand, false) {
                return false;
            }
            if self.st.player.facing != exit.face {
                self.prim(Primitive::Face(exit.face));
            }
            if !self.prim(Primitive::Interact) {
                return false;
            }
        }
        let mut hops = 0;
        while self.st.player.loc != loc && hops < 8 && !self.over() {
            hops += 1;
            let cur = self.st.player.loc;
            if self.st.has_feature(Feature::Minecart) && outdoor(cur) && outdoor(loc) {
                let h = hub(loc);
                let de = self.st.advance(c.world.minecart_minutes as u32);
                self.sync();
                if de == 0 {
                    self.st.player.loc = loc;
                    self.st.player.x = h.0;
                    self.st.player.y = h.1;
                }
                return de == 0;
            }
            let wid = self.st.routes.next[cur as usize][loc as usize];
            if wid == 255 {
                return false;
            }
            let w = &c.warps[wid as usize];
            if w.to == c.loc_id("greenhouse") && !self.st.has_feature(Feature::Greenhouse) {
                return false;
            }
            if !self.walk(w.tile, false) {
                return false;
            }
        }
        self.st.player.loc == loc
    }
    /// Go next to a tile in a location and face it.
    fn goto_adjacent(&mut self, loc: LocId, target: (u8, u8)) -> bool {
        if !self.goto_loc(loc) {
            return false;
        }
        if (self.st.player.x, self.st.player.y) == target {
            // standing on the target (e.g. a walkable crop tile): step off, then face back
            for d in Dir::ALL {
                let (dx, dy) = d.delta();
                let (nx, ny) = (target.0 as i32 + dx, target.1 as i32 + dy);
                let g = self.st.world.grid(self.st.player.loc);
                if g.passable(nx, ny)
                    && !matches!(g.obj(nx as u8, ny as u8), Obj::Warp | Obj::Ladder)
                {
                    self.prim(Primitive::Move(d));
                    let back = Dir::ALL[(d as usize + 2) % 4];
                    self.prim(Primitive::Face(back));
                    return (self.st.player.x as i32, self.st.player.y as i32) == (nx, ny);
                }
            }
            return false;
        }
        self.walk(target, true)
    }
    /// Go to a POI stand tile and face the POI.
    fn goto_poi(&mut self, poi: PoiId) -> bool {
        let p = &self.c().pois[poi as usize];
        if !self.goto_loc(p.location) {
            return false;
        }
        if !self.walk(p.stand, false) {
            return false;
        }
        if self.st.player.facing != p.face {
            self.prim(Primitive::Face(p.face));
        }
        true
    }
    /// Nearest tile in the current location satisfying `pred` (BFS order, ties by index).
    fn nearest<F: Fn(&Grid, u8, u8) -> bool>(&mut self, pred: F) -> Option<(u8, u8)> {
        let st = &mut *self.st;
        let g = st.world.grid(st.player.loc);
        st.scratch.bfs(g, st.player.x, st.player.y);
        let mut best: Option<(u8, u8, u16)> = None;
        for y in 0..g.h {
            for x in 0..g.w {
                if !pred(g, x, y) {
                    continue;
                }
                let d = st.scratch.dist(g, x, y);
                if d == UNREACHABLE {
                    continue;
                }
                if best.is_none_or(|b| d < b.2) {
                    best = Some((x, y, d));
                }
            }
        }
        best.map(|b| (b.0, b.1))
    }
    fn field_rects(&self, loc: LocId) -> Vec<Rect> {
        self.c()
            .regions_of(loc, "field")
            .map(|r| Rect {
                x0: r.rect[0],
                y0: r.rect[1],
                x1: r.rect[2],
                y1: r.rect[3],
            })
            .collect()
    }
    fn energy_ok(&self, base: u16) -> bool {
        self.st.player.energy > base as i32
    }
}

// ------------------------------------------------------------------------------------------ masks

impl GameState {
    /// Items reserved for open requests (never shipped/loaded/eaten); seeds are always reserved.
    pub fn reserved(&self, item: ItemId) -> u32 {
        let c = self.c();
        let mut r = self.need(item);
        if c.items[item as usize].category == Category::Seed {
            r += self.count(item);
        }
        r
    }
    /// Does any tile of the farm have an unwatered, planted, tilled soil?
    pub fn farm_needs_water(&self) -> bool {
        let g = self.world.grid(self.c().world.start_location);
        (0..g.soil.len()).any(|i| {
            g.crop[i] != NO_CROP && g.soil[i] & SOIL_TILLED != 0 && g.soil[i] & SOIL_WATERED == 0
        })
    }
    /// Any ready crop on farm or greenhouse?
    pub fn any_ready_crop(&self) -> bool {
        let c = self.c();
        [c.world.start_location, c.loc_id("greenhouse")]
            .iter()
            .any(|l| {
                let g = self.world.grid(*l);
                (0..g.crop.len()).any(|i| {
                    g.crop[i] != NO_CROP && g.crop_days[i] >= c.crops[g.crop[i] as usize].grow_days
                })
            })
    }
    /// Count tiles in the field rects of a location satisfying a predicate.
    pub fn count_field<F: Fn(&Grid, usize) -> bool>(&self, loc: LocId, f: F) -> u32 {
        let c = self.c();
        let g = self.world.grid(loc);
        let mut n = 0;
        for r in c.regions_of(loc, "field") {
            for y in r.rect[1]..=r.rect[3] {
                for x in r.rect[0]..=r.rect[2] {
                    if f(g, g.idx(x, y)) {
                        n += 1;
                    }
                }
            }
        }
        n
    }
    /// Debris count on the farm field.
    pub fn field_debris(&self) -> u32 {
        let farm = self.c().world.start_location;
        self.count_field(farm, |g, i| {
            matches!(
                Obj::from_plane(g.obj_plane()[i]),
                Obj::Weeds | Obj::Twig | Obj::StoneDebris | Obj::Tree
            )
        })
    }
    /// Free tillable tiles (dirt, no object, no crop) in a location's fields.
    pub fn free_field_tiles(&self, loc: LocId) -> u32 {
        self.count_field(loc, |g, i| {
            g.terrain_plane()[i] == Terrain::Dirt as u8
                && g.obj_plane()[i] == 0
                && g.crop[i] == NO_CROP
        })
    }
    /// Objects of a kind in a location.
    pub fn count_kind(&self, loc: LocId, o: Obj) -> u32 {
        self.world.grid(loc).count_obj(o)
    }
    fn tool_usable(&self, t: Tool) -> bool {
        self.player.upgrading.map(|(u, _)| u != t).unwrap_or(true)
    }
    /// Cheapest seller of an item today: `(price, Some(shop) | None=merchant)`.
    pub fn best_seller(&self, item: ItemId) -> Option<(u32, Option<crate::content::ShopId>)> {
        let c = self.c();
        let mut best: Option<(u32, Option<crate::content::ShopId>)> = None;
        if let Some((s, row)) = c.shop_row(item, self.season()) {
            let shop = &c.shops[s as usize];
            if shop.closed_weekday != self.weekday() as i8 {
                best = Some((shop.stock[row].price, Some(s)));
            }
        }
        if self.merchant_today() && self.merchant_stock.contains(&item) {
            let p = c.merchant_price(item);
            if best.is_none_or(|b| p < b.0) {
                best = Some((p, None));
            }
        }
        best
    }
    /// Any shippable stack with nothing reserved?
    pub fn any_surplus(&self) -> bool {
        self.player
            .bag
            .iter()
            .any(|(i, n)| *n > 0 && self.shippable(*i) && self.reserved(*i) == 0)
    }
    /// Best food slot to eat (unreserved edible with the most energy), if any.
    pub fn food_slot(&self) -> Option<u8> {
        let c = self.c();
        let mut best: Option<(u8, u16)> = None;
        for (i, (item, n)) in self.player.bag.iter().enumerate() {
            if *n == 0 {
                continue;
            }
            let e = c.items[*item as usize].energy;
            if e == 0 || self.reserved(*item) >= *n as u32 {
                continue;
            }
            if best.is_none_or(|b| e > b.1) {
                best = Some((i as u8, e));
            }
        }
        best.map(|b| b.0)
    }
    /// Is the macro valid now? (cheap preconditions; ETA-aware for shops)
    pub fn macro_valid(&mut self, nav: &NavTables, m: Macro, max_decisions_per_day: u16) -> bool {
        let c = self.c();
        if self.complete {
            return false;
        }
        if self.decisions_today >= max_decisions_per_day {
            return m == Macro::Sleep;
        }
        let in_mines = self.player.loc == c.loc_id("mines");
        match m {
            Macro::Wait30 | Macro::Sleep => true,
            Macro::GoTo(l) => {
                l != self.player.loc
                    && !in_mines
                    && (l != c.loc_id("greenhouse") || self.has_feature(Feature::Greenhouse))
            }
            Macro::WaterCrops => {
                !in_mines
                    && self.tool_usable(Tool::Can)
                    && self.farm_needs_water()
                    && self.player.energy > 2
            }
            Macro::Harvest => !in_mines && self.any_ready_crop() && self.free_slots() > 0,
            Macro::ClearField => {
                !in_mines
                    && self.field_debris() > 0
                    && self.player.energy > 4
                    && (self.tool_usable(Tool::Axe) || self.tool_usable(Tool::Pickaxe))
            }
            Macro::Plant(cr, gh) => {
                let crop = &c.crops[cr as usize];
                let loc = if gh {
                    c.loc_id("greenhouse")
                } else {
                    c.world.start_location
                };
                !in_mines
                    && (!gh || self.has_feature(Feature::Greenhouse))
                    && (gh || crate::content::season_in(crop.seasons, self.season()))
                    && self.count(crop.seed) > 0
                    && self.free_field_tiles(loc) > 0
                    && self.player.energy > 2
            }
            Macro::Forage(f) => {
                let fl = &c.forage[f as usize];
                !in_mines && self.count_kind(fl.location, Obj::Forage) > 0 && self.free_slots() > 0
            }
            Macro::Fish(spot) => {
                !in_mines
                    && self.tool_usable(Tool::Rod)
                    && self.player.energy > c.fishing.energy_per_cast as i32
                    && self.free_slots() > 0
                    && self.eta_poi(nav, spot) != u32::MAX
            }
            Macro::ChopWood => {
                !in_mines
                    && self.tool_usable(Tool::Axe)
                    && self.player.energy > 3
                    && (self.count_kind(c.loc_id("woods"), Obj::Tree) > 0
                        || self.count_kind(c.world.start_location, Obj::Tree) > 0)
            }
            Macro::ChopHardwood => {
                !in_mines
                    && self.tool_usable(Tool::Axe)
                    && self.player.tool_level[Tool::Axe as usize] >= 1
                    && self.player.energy > 3
                    && self.count_kind(c.loc_id("woods"), Obj::Stump) > 0
            }
            Macro::MineDescend => {
                self.tool_usable(Tool::Pickaxe)
                    && self.player.energy > 3
                    && self.mine.deepest < c.mining.floors
                    && (!in_mines || self.mine_progress_possible())
            }
            Macro::MineFloor => {
                self.tool_usable(Tool::Pickaxe)
                    && self.player.energy > 3
                    && (!in_mines || self.mine_progress_possible())
            }
            Macro::Buy(item, q) => {
                if in_mines {
                    return false;
                }
                let Some((price, seller)) = self.best_seller(item) else {
                    return false;
                };
                if self.player.money < price * q as u32 || !self.can_add(item, q as u16) {
                    return false;
                }
                let poi = match seller {
                    Some(s) => c.shops[s as usize].poi,
                    None => c.merchant.poi,
                };
                let eta = self.eta_poi(nav, poi);
                if eta == u32::MAX {
                    return false;
                }
                let arrive = self.minute as u32 + eta + 2;
                match seller {
                    Some(s) => {
                        let sh = &c.shops[s as usize];
                        arrive < sh.close as u32 && arrive + 120 >= sh.open as u32
                    }
                    None => arrive < c.merchant.close as u32,
                }
            }
            Macro::ShipSurplus => !in_mines && self.any_surplus(),
            Macro::Craft(r) => {
                self.can_craft(r)
                    && (c.items[c.recipes[r as usize].output as usize].category
                        != Category::Machine
                        || (!in_mines
                            && self.count_kind(c.world.start_location, Obj::MachinePad) > 0))
            }
            Macro::LoadMachines => {
                !in_mines
                    && (self
                        .machines
                        .iter()
                        .any(|m| m.ready_at == 0 && !m.has_output && self.any_input_for(m.kind))
                        || (self.held_machine().is_some()
                            && self.count_kind(c.world.start_location, Obj::MachinePad) > 0))
            }
            Macro::CollectMachines => {
                !in_mines && self.machines.iter().any(|m| m.has_output) && self.free_slots() > 0
            }
            Macro::UpgradeTool(t) => {
                if in_mines || !self.can_upgrade(t) {
                    return false;
                }
                let s = c
                    .shops
                    .iter()
                    .find(|s| s.key == "smithy")
                    .map(|s| s.id)
                    .unwrap_or(1);
                if c.shops[s as usize].closed_weekday == self.weekday() as i8 {
                    return false;
                }
                let eta = self.eta_poi(nav, c.shops[s as usize].poi);
                eta != u32::MAX && (self.minute as u32 + eta + 2) < c.shops[s as usize].close as u32
            }
            Macro::Deliver => {
                !in_mines
                    && (0..c.requests.len() as u8)
                        .any(|r| c.requests[r as usize].coins == 0 && self.can_deliver_any(r))
            }
            Macro::PayCoins(r) => !in_mines && self.can_deliver_any(r),
            Macro::Eat => {
                self.player.energy < self.player.energy_max as i32 && self.food_slot().is_some()
            }
        }
    }
    /// In the mines: is a breakable rock or a ladder reachable from the player?
    pub fn mine_progress_possible(&mut self) -> bool {
        let c = self.c();
        let pick = self.player.tool_level[Tool::Pickaxe as usize];
        let loc = self.player.loc;
        let (px, py) = (self.player.x, self.player.y);
        let g = self.world.grid(loc);
        self.scratch.bfs(g, px, py);
        for y in 0..g.h {
            for x in 0..g.w {
                let o = g.obj(x, y);
                let ok = match o {
                    Obj::Ladder => true,
                    Obj::Rock => c.rocks[(g.data[g.idx(x, y)] & 0xff) as usize].min_pick <= pick,
                    _ => false,
                };
                if ok && self.scratch.dist(g, x, y) != UNREACHABLE {
                    return true;
                }
            }
        }
        false
    }
    /// A machine item held in the bag (to be placed), if any.
    pub fn held_machine(&self) -> Option<ItemId> {
        let c = self.c();
        self.player
            .bag
            .iter()
            .find(|(i, n)| *n > 0 && c.items[*i as usize].category == Category::Machine)
            .map(|(i, _)| *i)
    }
    /// Any bag item satisfies some rule of a machine kind (with fuel)?
    pub fn any_input_for(&self, kind: crate::content::MachineId) -> bool {
        let c = self.c();
        let m = &c.machines[kind as usize];
        self.player.bag.iter().any(|(item, n)| {
            *n > 0
                && m.rules.iter().any(|r| {
                    let input_ok = match r.input {
                        crate::content::InputSel::Item(i) => i == *item,
                        crate::content::InputSel::Category(cat) => {
                            c.items[*item as usize].category == cat
                        }
                    } && *n >= r.input_count
                        && self.reserved(*item) + r.input_count as u32 <= *n as u32;
                    input_ok
                        && r.fuel
                            .map(|(f, fc)| self.count(f) >= fc as u32)
                            .unwrap_or(true)
                })
        })
    }
    /// Write the macro mask.
    pub fn macro_mask(&mut self, nav: &NavTables, max_decisions_per_day: u16, out: &mut [bool]) {
        let table = macro_actions();
        for (i, m) in table.iter().enumerate() {
            out[i] = self.macro_valid(nav, *m, max_decisions_per_day);
        }
    }
}

impl Obj {
    /// Decode a plane byte.
    pub fn from_plane(v: u8) -> Obj {
        // mirrors the private decoder in world.rs via a table lookup
        const ALL: [Obj; 23] = [
            Obj::None,
            Obj::Weeds,
            Obj::Twig,
            Obj::StoneDebris,
            Obj::Tree,
            Obj::Stump,
            Obj::Boulder,
            Obj::Rock,
            Obj::Forage,
            Obj::Building,
            Obj::Counter,
            Obj::Bed,
            Obj::Crate,
            Obj::Well,
            Obj::MillBoard,
            Obj::Ladder,
            Obj::Elevator,
            Obj::MachinePad,
            Obj::Machine,
            Obj::Warp,
            Obj::Stall,
            Obj::MineExit,
            Obj::MineEntrance,
        ];
        ALL.get(v as usize).copied().unwrap_or(Obj::None)
    }
}

// ------------------------------------------------------------------------------------ controllers

/// Execute a macro with a time box of `budget` minutes.
pub fn run_macro(
    st: &mut GameState,
    nav: &NavTables,
    m: Macro,
    budget: u32,
    max_decisions_per_day: u16,
) -> MacroResult {
    let abs0 = st.abs_minute;
    let day0 = st.day;
    if !st.macro_valid(nav, m, max_decisions_per_day) {
        st.advance(10);
        st.tick_machines();
        return MacroResult {
            ticks: st.abs_minute - abs0,
            day_ends: (st.day - day0) as u8,
            outcome: Outcome::Masked,
        };
    }
    st.decisions_today = st.decisions_today.saturating_add(1);
    let mut ctl = Ctl {
        st,
        budget: budget + TRAVEL_CAP,
        box_minutes: budget,
        abs0,
        day0,
        ticks: 0,
        day_ends: 0,
        progressed: false,
        path: Vec::new(),
    };
    let done = ctl.run(m);
    ctl.close_menu_if_needed(m);
    ctl.sync();
    let progressed = ctl.progressed;
    let timeboxed = ctl.ticks >= ctl.budget && !done;
    if ctl.ticks < MIN_MACRO_TICKS && ctl.day_ends == 0 {
        let pad = MIN_MACRO_TICKS - ctl.ticks;
        ctl.st.advance(pad);
        ctl.st.tick_machines();
        ctl.sync();
    }
    let ticks = ctl.ticks;
    let day_ends = ctl.day_ends;
    // a time-boxed macro that made progress is Partial (may be re-issued)
    let _ = timeboxed;
    let outcome = if done {
        Outcome::Done
    } else if progressed {
        Outcome::Partial
    } else {
        Outcome::Failed
    };
    MacroResult {
        ticks,
        day_ends,
        outcome,
    }
}

impl Ctl<'_> {
    fn close_menu_if_needed(&mut self, m: Macro) {
        if self.st.menu != Menu::None && self.day_ends == 0 && !matches!(m, Macro::Sleep) {
            self.prim(Primitive::CloseMenu);
        }
    }

    /// Returns true if the macro fully completed its goal.
    fn run(&mut self, m: Macro) -> bool {
        let c = self.c();
        match m {
            Macro::Wait30 => {
                for _ in 0..3 {
                    if self.over() {
                        break;
                    }
                    self.prim(Primitive::Wait);
                }
                self.progressed = true;
                true
            }
            Macro::Sleep => {
                let bed = c.poi_id("bed");
                if !self.goto_poi(bed) {
                    return false;
                }
                self.progressed = true;
                self.prim(Primitive::Interact)
            }
            Macro::GoTo(l) => {
                let ok = self.goto_loc(l);
                if ok {
                    self.progressed = true;
                }
                ok
            }
            Macro::WaterCrops => self.water_crops(),
            Macro::Harvest => self.harvest(),
            Macro::ClearField => self.clear_field(),
            Macro::Plant(cr, gh) => self.plant(cr, gh),
            Macro::Forage(f) => self.forage(f),
            Macro::Fish(spot) => self.fish(spot),
            Macro::ChopWood => self.chop(Obj::Tree),
            Macro::ChopHardwood => self.chop(Obj::Stump),
            Macro::MineDescend => self.mine(true),
            Macro::MineFloor => self.mine(false),
            Macro::Buy(item, q) => self.buy(item, q),
            Macro::ShipSurplus => self.ship_surplus(),
            Macro::Craft(r) => {
                if self.st.menu != Menu::Craft {
                    self.close_menu();
                    self.prim(Primitive::OpenCraft);
                }
                let ok = self.prim(Primitive::MenuOption(r));
                self.prim(Primitive::CloseMenu);
                if ok {
                    let out = c.recipes[r as usize].output;
                    if c.items[out as usize].category == Category::Machine {
                        self.place_machine(out);
                    }
                }
                ok
            }
            Macro::LoadMachines => self.load_machines(),
            Macro::CollectMachines => self.collect_machines(),
            Macro::UpgradeTool(t) => {
                let s = c
                    .shops
                    .iter()
                    .find(|s| s.key == "smithy")
                    .map(|s| s.id)
                    .unwrap_or(1);
                if !self.goto_poi(c.shops[s as usize].poi) {
                    return false;
                }
                if !self.prim(Primitive::Interact) {
                    return false;
                }
                let rows = self.st.shop_rows(s).len() as u8;
                let ok = self.prim(Primitive::MenuOption(rows + t as u8));
                self.prim(Primitive::CloseMenu);
                ok
            }
            Macro::Deliver => self.deliver(),
            Macro::PayCoins(r) => {
                if !self.goto_poi(c.poi_id("mill_board")) {
                    return false;
                }
                if !self.prim(Primitive::Interact) {
                    return false;
                }
                let ok = self.prim(Primitive::MenuOption(r));
                self.prim(Primitive::CloseMenu);
                ok
            }
            Macro::Eat => {
                let Some(slot) = self.st.food_slot() else {
                    return false;
                };
                self.prim(Primitive::SelectSlot(slot));
                self.prim(Primitive::Eat)
            }
        }
    }

    fn refill_can(&mut self) -> bool {
        let loc = self.st.player.loc;
        let Some(t) =
            self.nearest(|g, x, y| g.terrain(x, y) == Terrain::Water || g.obj(x, y) == Obj::Well)
        else {
            return false;
        };
        if !self.goto_adjacent(loc, t) {
            return false;
        }
        self.select_tool(Tool::Can);
        self.prim(Primitive::UseTool)
    }

    fn water_crops(&mut self) -> bool {
        let c = self.c();
        let farm = c.world.start_location;
        if !self.goto_loc(farm) {
            return false;
        }
        self.start_activity();
        self.select_tool(Tool::Can);
        loop {
            if self.over() || !self.energy_ok(1) {
                return !self.st.farm_needs_water();
            }
            if self.st.player.can_water == 0 && !self.refill_can() {
                return false;
            }
            let Some(t) = self.nearest(|g, x, y| {
                let i = g.idx(x, y);
                g.crop[i] != NO_CROP
                    && g.soil[i] & SOIL_TILLED != 0
                    && g.soil[i] & SOIL_WATERED == 0
            }) else {
                return true;
            };
            if !self.goto_adjacent(farm, t) {
                return false;
            }
            self.select_tool(Tool::Can);
            if !self.prim(Primitive::UseTool) {
                return false;
            }
        }
    }

    fn harvest(&mut self) -> bool {
        let c = self.c();
        let mut any = false;
        for loc in [c.world.start_location, c.loc_id("greenhouse")] {
            if loc == c.loc_id("greenhouse") && !self.st.has_feature(Feature::Greenhouse) {
                continue;
            }
            let has = {
                let g = self.st.world.grid(loc);
                (0..g.crop.len()).any(|i| {
                    g.crop[i] != NO_CROP && g.crop_days[i] >= c.crops[g.crop[i] as usize].grow_days
                })
            };
            if !has {
                continue;
            }
            if !self.goto_loc(loc) {
                return any;
            }
            loop {
                if self.over() || self.st.free_slots() == 0 {
                    return any;
                }
                let Some(t) = self.nearest(|g, x, y| {
                    let i = g.idx(x, y);
                    g.crop[i] != NO_CROP && g.crop_days[i] >= c.crops[g.crop[i] as usize].grow_days
                }) else {
                    break;
                };
                if !self.goto_adjacent(loc, t) {
                    return any;
                }
                if !self.prim(Primitive::Interact) {
                    return any;
                }
                any = true;
            }
        }
        true
    }

    fn clear_field(&mut self) -> bool {
        let c = self.c();
        let farm = c.world.start_location;
        if !self.goto_loc(farm) {
            return false;
        }
        self.start_activity();
        let rects = self.field_rects(farm);
        loop {
            if self.over() || !self.energy_ok(3) {
                return self.st.field_debris() == 0;
            }
            let rr = rects.clone();
            let Some(t) = self.nearest(|g, x, y| {
                rr.iter().any(|r| r.contains(x, y))
                    && matches!(
                        g.obj(x, y),
                        Obj::Weeds | Obj::Twig | Obj::StoneDebris | Obj::Tree
                    )
            }) else {
                return true;
            };
            let tool = match self.st.world.grid(farm).obj(t.0, t.1) {
                Obj::Weeds => Tool::Hoe,
                Obj::Twig | Obj::Tree => Tool::Axe,
                _ => Tool::Pickaxe,
            };
            if !self.st.tool_usable(tool) {
                return false;
            }
            if !self.goto_adjacent(farm, t) {
                return false;
            }
            self.select_tool(tool);
            // hit until gone
            let mut guard = 0;
            while self.st.world.grid(farm).obj(t.0, t.1) != Obj::None && guard < 20 {
                guard += 1;
                if self.over() || !self.energy_ok(3) {
                    return false;
                }
                if !self.prim(Primitive::UseTool) {
                    return false;
                }
            }
        }
    }

    fn plant(&mut self, cr: CropId, gh: bool) -> bool {
        let c = self.c();
        let crop = &c.crops[cr as usize];
        let loc = if gh {
            c.loc_id("greenhouse")
        } else {
            c.world.start_location
        };
        if !self.goto_loc(loc) {
            return false;
        }
        self.start_activity();
        let rects = self.field_rects(loc);
        loop {
            if self.st.count(crop.seed) == 0 {
                return true;
            }
            if self.over() || !self.energy_ok(2) {
                return false;
            }
            let rr = rects.clone();
            let tilled = self.nearest(|g, x, y| {
                let i = g.idx(x, y);
                rr.iter().any(|r| r.contains(x, y))
                    && g.soil[i] & SOIL_TILLED != 0
                    && g.crop[i] == NO_CROP
                    && g.obj(x, y) == Obj::None
            });
            if let Some(t) = tilled {
                if !self.goto_adjacent(loc, t) {
                    return false;
                }
                if !self.select_item(crop.seed) {
                    return false;
                }
                if !self.prim(Primitive::UseTool) {
                    return false;
                }
                continue;
            }
            if !self.st.tool_usable(Tool::Hoe) {
                return false;
            }
            let rr2 = rects.clone();
            let Some(t) = self.nearest(|g, x, y| {
                let i = g.idx(x, y);
                rr2.iter().any(|r| r.contains(x, y))
                    && g.terrain(x, y) == Terrain::Dirt
                    && g.soil[i] & SOIL_TILLED == 0
                    && g.crop[i] == NO_CROP
                    && g.obj(x, y) == Obj::None
            }) else {
                return false;
            };
            if !self.goto_adjacent(loc, t) {
                return false;
            }
            self.select_tool(Tool::Hoe);
            if !self.prim(Primitive::UseTool) {
                return false;
            }
        }
    }

    fn forage(&mut self, f: u8) -> bool {
        let fl = &self.c().forage[f as usize];
        let loc = fl.location;
        if !self.goto_loc(loc) {
            return false;
        }
        self.start_activity();
        loop {
            if self.over() || self.st.free_slots() == 0 {
                return self.st.count_kind(loc, Obj::Forage) == 0;
            }
            let Some(t) = self.nearest(|g, x, y| g.obj(x, y) == Obj::Forage) else {
                return self.st.count_kind(loc, Obj::Forage) == 0;
            };
            if !self.goto_adjacent(loc, t) {
                return false;
            }
            if !self.prim(Primitive::Interact) {
                return false;
            }
        }
    }

    fn fish(&mut self, spot: PoiId) -> bool {
        let c = self.c();
        if !self.goto_poi(spot) {
            return false;
        }
        self.select_tool(Tool::Rod);
        self.start_activity();
        let limit = self.ticks + 60.min(self.box_minutes);
        let mut casts = 0;
        while !self.over()
            && self.ticks < limit
            && self.energy_ok(c.fishing.energy_per_cast)
            && self.st.free_slots() > 0
        {
            if !self.prim(Primitive::UseTool) {
                break;
            }
            casts += 1;
        }
        casts > 0
    }

    fn chop(&mut self, kind: Obj) -> bool {
        let c = self.c();
        let farm = c.world.start_location;
        let woods = c.loc_id("woods");
        let loc = if kind == Obj::Tree && self.st.count_kind(self.st.player.loc, Obj::Tree) > 0 {
            self.st.player.loc
        } else if kind == Obj::Tree
            && self.st.count_kind(farm, Obj::Tree) > 0
            && self.st.player.loc == farm
        {
            farm
        } else {
            woods
        };
        if !self.goto_loc(loc) {
            return false;
        }
        self.start_activity();
        self.select_tool(Tool::Axe);
        let mut chopped = 0;
        loop {
            if self.over() || !self.energy_ok(3) {
                return chopped > 0;
            }
            let Some(t) = self.nearest(|g, x, y| g.obj(x, y) == kind) else {
                return chopped > 0;
            };
            if !self.goto_adjacent(loc, t) {
                return chopped > 0;
            }
            self.select_tool(Tool::Axe);
            let mut guard = 0;
            while self.st.world.grid(loc).obj(t.0, t.1) == kind && guard < 30 {
                guard += 1;
                if self.over() || !self.energy_ok(3) {
                    return chopped > 0;
                }
                if !self.prim(Primitive::UseTool) {
                    return chopped > 0;
                }
            }
            chopped += 1;
        }
    }

    fn enter_mines(&mut self) -> bool {
        let c = self.c();
        if self.st.player.loc == c.loc_id("mines") {
            return true;
        }
        let emax = self.st.elevator_max();
        if emax >= c.mining.elevator_step {
            if !self.goto_poi(c.poi_id("elevator")) {
                return false;
            }
            if !self.prim(Primitive::Interact) {
                return false;
            }
            return self.prim(Primitive::MenuOption(emax / c.mining.elevator_step));
        }
        if !self.goto_poi(c.poi_id("mine_entrance")) {
            return false;
        }
        self.prim(Primitive::Interact)
    }

    fn mine(&mut self, descend: bool) -> bool {
        let c = self.c();
        if !self.enter_mines() {
            return false;
        }
        self.start_activity();
        let mines = c.loc_id("mines");
        self.select_tool(Tool::Pickaxe);
        let pick = self.st.player.tool_level[Tool::Pickaxe as usize];
        let mut broke = 0;
        loop {
            if self.over() || !self.energy_ok(3) {
                return broke > 0;
            }
            if descend {
                if let Some(l) = self.nearest(|g, x, y| g.obj(x, y) == Obj::Ladder) {
                    let floor = self.st.mine.floor;
                    if !self.walk(l, false) {
                        return broke > 0;
                    }
                    if self.st.mine.floor != floor {
                        self.progressed = true;
                        if self.st.mine.floor >= c.mining.floors {
                            return true;
                        }
                        continue;
                    }
                }
            }
            let Some(t) = self.nearest(|g, x, y| {
                g.obj(x, y) == Obj::Rock
                    && c.rocks[(g.data[g.idx(x, y)] & 0xff) as usize].min_pick <= pick
            }) else {
                // floor exhausted: take a ladder if one is reachable
                if let Some(l) = self.nearest(|g, x, y| g.obj(x, y) == Obj::Ladder) {
                    let floor = self.st.mine.floor;
                    if self.walk(l, false) && self.st.mine.floor != floor {
                        self.progressed = true;
                        continue;
                    }
                }
                return broke > 0;
            };
            if !self.goto_adjacent(mines, t) {
                return broke > 0;
            }
            self.select_tool(Tool::Pickaxe);
            let mut guard = 0;
            while self.st.world.grid(mines).obj(t.0, t.1) == Obj::Rock && guard < 10 {
                guard += 1;
                if self.over() || !self.energy_ok(3) {
                    return broke > 0;
                }
                if !self.prim(Primitive::UseTool) {
                    return broke > 0;
                }
            }
            broke += 1;
        }
    }

    fn buy(&mut self, item: ItemId, q: u8) -> bool {
        let c = self.c();
        let Some((_, seller)) = self.st.best_seller(item) else {
            return false;
        };
        let poi = match seller {
            Some(s) => c.shops[s as usize].poi,
            None => c.merchant.poi,
        };
        if !self.goto_poi(poi) {
            return false;
        }
        // wait for opening if early
        let open_at = match seller {
            Some(s) => c.shops[s as usize].open,
            None => c.merchant.open,
        };
        let mut waits = 0;
        while self.st.minute < open_at && waits < 24 && self.day_ends == 0 {
            self.prim(Primitive::Wait);
            waits += 1;
        }
        if !self.prim(Primitive::Interact) {
            return false;
        }
        let row = match self.st.menu {
            Menu::Shop(s) => self.st.shop_rows(s).iter().position(|r| r.0 == item),
            Menu::Merchant => self.st.merchant_stock.iter().position(|i| *i == item),
            _ => None,
        };
        let Some(row) = row else {
            self.prim(Primitive::CloseMenu);
            return false;
        };
        let mut bought = 0;
        for _ in 0..q {
            if !self.prim(Primitive::MenuOption(row as u8)) {
                break;
            }
            bought += 1;
        }
        self.prim(Primitive::CloseMenu);
        bought == q
    }

    fn ship_surplus(&mut self) -> bool {
        let c = self.c();
        if !self.goto_poi(c.poi_id("crate")) {
            return false;
        }
        let mut shipped = 0;
        for i in 0..crate::state::BAG {
            if self.over() {
                break;
            }
            let (item, n) = self.st.player.bag[i];
            if n == 0 || !self.st.shippable(item) || self.st.reserved(item) > 0 {
                continue;
            }
            self.prim(Primitive::SelectSlot(i as u8));
            if self.prim(Primitive::Interact) {
                shipped += 1;
            }
        }
        shipped > 0
    }

    /// Place a held machine item on the nearest free machine pad of the farm.
    fn place_machine(&mut self, item: ItemId) -> bool {
        let farm = self.c().world.start_location;
        if !self.goto_loc(farm) {
            return false;
        }
        let Some(t) = self.nearest(|g, x, y| g.obj(x, y) == Obj::MachinePad) else {
            return false;
        };
        if !self.goto_adjacent(farm, t) {
            return false;
        }
        if !self.select_item(item) {
            return false;
        }
        self.prim(Primitive::UseTool)
    }

    fn load_machines(&mut self) -> bool {
        let c = self.c();
        let farm = c.world.start_location;
        if !self.goto_loc(farm) {
            return false;
        }
        let mut loaded = 0;
        while let Some(item) = self.st.held_machine() {
            if self.over() || !self.place_machine(item) {
                break;
            }
            loaded += 1;
        }
        let n = self.st.machines.len();
        for mi in 0..n {
            if self.over() {
                break;
            }
            let m = self.st.machines[mi];
            if m.ready_at != 0 || m.has_output || !self.st.any_input_for(m.kind) {
                continue;
            }
            // pick an input slot
            let mut slot = None;
            for (i, (item, cnt)) in self.st.player.bag.iter().enumerate() {
                if *cnt == 0 {
                    continue;
                }
                let ok = c.machines[m.kind as usize].rules.iter().any(|r| {
                    let input_ok = match r.input {
                        crate::content::InputSel::Item(x) => x == *item,
                        crate::content::InputSel::Category(cat) => {
                            c.items[*item as usize].category == cat
                        }
                    } && *cnt >= r.input_count
                        && self.st.reserved(*item) + r.input_count as u32 <= *cnt as u32;
                    input_ok
                        && r.fuel
                            .map(|(f, fc)| self.st.count(f) >= fc as u32)
                            .unwrap_or(true)
                });
                if ok {
                    slot = Some(i as u8);
                    break;
                }
            }
            let Some(slot) = slot else { continue };
            if !self.goto_adjacent(m.loc, (m.x, m.y)) {
                return loaded > 0;
            }
            self.prim(Primitive::SelectSlot(slot));
            if self.prim(Primitive::Interact) {
                loaded += 1;
            }
        }
        loaded > 0
    }

    fn collect_machines(&mut self) -> bool {
        let c = self.c();
        if !self.goto_loc(c.world.start_location) {
            return false;
        }
        let mut got = 0;
        let n = self.st.machines.len();
        for mi in 0..n {
            if self.over() || self.st.free_slots() == 0 {
                break;
            }
            let m = self.st.machines[mi];
            if !m.has_output {
                continue;
            }
            if !self.goto_adjacent(m.loc, (m.x, m.y)) {
                return got > 0;
            }
            if self.prim(Primitive::Interact) {
                got += 1;
            }
        }
        got > 0
    }

    fn deliver(&mut self) -> bool {
        let c = self.c();
        if !self.goto_poi(c.poi_id("mill_board")) {
            return false;
        }
        if !self.prim(Primitive::Interact) {
            return false;
        }
        // requests that this delivery completes first, then the rest
        let mut order: Vec<(bool, u8)> = Vec::new();
        for r in 0..c.requests.len() as u8 {
            let req = &c.requests[r as usize];
            if req.coins > 0 || !self.st.can_deliver_any(r) {
                continue;
            }
            let range = self.st.slots_of(r);
            let fillable = range
                .enumerate()
                .filter(|(k, i)| {
                    !self.st.restoration.slot_filled[*i]
                        && self.st.count(self.st.restoration.slot_item[*i])
                            >= req.slots[*k].count as u32
                })
                .count() as u8;
            let completes = self.st.filled_in(r) + fillable >= req.need;
            order.push((!completes, r));
        }
        order.sort();
        let mut any = false;
        for (_, r) in order {
            if self.over() {
                break;
            }
            if self.st.can_deliver_any(r) && self.prim(Primitive::MenuOption(r)) {
                any = true;
            }
        }
        self.prim(Primitive::CloseMenu);
        any
    }
}

/// Kind lookup for a POI by key (helper for tests/expert).
pub fn poi_kind(p: PoiId) -> PoiKind {
    content().pois[p as usize].kind
}
