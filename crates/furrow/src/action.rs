//! Primitive actions (DESIGN.md §3.3): every step advances the clock by ≥ 1 tick.

use crate::content::{Category, Dir, InputSel, PoiKind, Skill, Tool, WeatherReq};
use crate::state::{GameState, Menu, Sel};
use crate::world::{self, Obj, Terrain, NO_CROP, SOIL_FERT, SOIL_TILLED, SOIL_WATERED};

/// Number of menu option primitives.
pub const MENU_OPTIONS: u8 = 24;
/// Total number of primitive actions.
pub const NUM_PRIMITIVES: usize = 7 + 24 + 5 + 2 + MENU_OPTIONS as usize + 1 + 4;

/// A primitive action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Primitive {
    /// Wait 10 minutes.
    Wait,
    /// Face a direction and step if passable (1 tick).
    Move(Dir),
    /// Use the selected tool/item on the faced tile.
    UseTool,
    /// Interact with the faced tile.
    Interact,
    /// Select backpack slot.
    SelectSlot(u8),
    /// Select a tool.
    SelectTool(Tool),
    /// Eat the selected item.
    Eat,
    /// Open the crafting menu.
    OpenCraft,
    /// Choose a menu row.
    MenuOption(u8),
    /// Close any menu.
    CloseMenu,
    /// Turn to face a direction without moving (1 tick).
    Face(Dir),
}

impl Primitive {
    /// Decode from a dense id (None if out of range).
    pub fn from_id(id: u32) -> Option<Primitive> {
        Some(match id {
            0 => Primitive::Wait,
            1 => Primitive::Move(Dir::N),
            2 => Primitive::Move(Dir::E),
            3 => Primitive::Move(Dir::S),
            4 => Primitive::Move(Dir::W),
            5 => Primitive::UseTool,
            6 => Primitive::Interact,
            7..=30 => Primitive::SelectSlot((id - 7) as u8),
            31..=35 => Primitive::SelectTool(Tool::ALL[(id - 31) as usize]),
            36 => Primitive::Eat,
            37 => Primitive::OpenCraft,
            38..=61 => Primitive::MenuOption((id - 38) as u8),
            62 => Primitive::CloseMenu,
            63..=66 => Primitive::Face(Dir::ALL[(id - 63) as usize]),
            _ => return None,
        })
    }
    /// Dense id.
    pub fn id(self) -> u32 {
        match self {
            Primitive::Wait => 0,
            Primitive::Move(d) => 1 + d as u32,
            Primitive::UseTool => 5,
            Primitive::Interact => 6,
            Primitive::SelectSlot(i) => 7 + i as u32,
            Primitive::SelectTool(t) => 31 + t as u32,
            Primitive::Eat => 36,
            Primitive::OpenCraft => 37,
            Primitive::MenuOption(k) => 38 + k as u32,
            Primitive::CloseMenu => 62,
            Primitive::Face(d) => 63 + d as u32,
        }
    }
    /// Human-readable name.
    pub fn name(self) -> String {
        match self {
            Primitive::Wait => "Wait".into(),
            Primitive::Move(d) => format!("Move{d:?}"),
            Primitive::UseTool => "UseTool".into(),
            Primitive::Interact => "Interact".into(),
            Primitive::SelectSlot(i) => format!("SelectSlot{i}"),
            Primitive::SelectTool(t) => format!("Select{}", t.name()),
            Primitive::Eat => "Eat".into(),
            Primitive::OpenCraft => "OpenCraft".into(),
            Primitive::MenuOption(k) => format!("Menu{k}"),
            Primitive::CloseMenu => "CloseMenu".into(),
            Primitive::Face(d) => format!("Face{d:?}"),
        }
    }
}

/// Result of a primitive step.
#[derive(Clone, Copy, Debug, Default)]
pub struct PrimResult {
    /// Game minutes consumed (absolute clock, including a skipped night).
    pub ticks: u32,
    /// The action had its intended effect.
    pub ok: bool,
    /// Day boundaries crossed.
    pub day_ends: u8,
}

impl GameState {
    /// Rows visible in the current shop menu: `(item, price)` for stock in season, followed by
    /// tool-upgrade rows at the smithy.
    pub fn shop_rows(&self, shop: crate::content::ShopId) -> Vec<(crate::content::ItemId, u32)> {
        let c = self.c();
        let season = self.season();
        c.shops[shop as usize]
            .stock
            .iter()
            .filter(|r| crate::content::season_in(r.seasons, season))
            .map(|r| (r.item, r.price))
            .collect()
    }
    /// Is this the smithy (sells upgrades)?
    pub fn is_smithy(&self, shop: crate::content::ShopId) -> bool {
        self.c().shops[shop as usize].key == "smithy"
    }
    /// Can the player afford/upgrade a tool now at the smithy?
    pub fn can_upgrade(&self, t: Tool) -> bool {
        let c = self.c();
        let lvl = self.player.tool_level[t as usize];
        if lvl >= 3 || self.player.upgrading.is_some() {
            return false;
        }
        let u = &c.upgrades[lvl as usize];
        self.player.money >= u.coins && self.count(u.bar) >= u.bars as u32
    }
    /// Is the primitive valid in the current state (mask)?
    pub fn primitive_valid(&self, p: Primitive) -> bool {
        let c = self.c();
        let in_menu = self.menu != Menu::None;
        match p {
            Primitive::Wait => true,
            Primitive::Move(_) => !in_menu,
            Primitive::UseTool => {
                !in_menu && self.player.energy > 0 && self.use_tool_target().is_some()
            }
            Primitive::Interact => !in_menu && self.interact_target(),
            Primitive::SelectSlot(i) => {
                (i as usize) < crate::state::BAG && self.player.bag[i as usize].1 > 0
            }
            Primitive::SelectTool(t) => self.player.upgrading.map(|(u, _)| u != t).unwrap_or(true),
            Primitive::Eat => self
                .selected_item()
                .map(|(it, _)| {
                    c.items[it as usize].energy > 0
                        && self.player.energy < self.player.energy_max as i32
                })
                .unwrap_or(false),
            Primitive::OpenCraft => !in_menu,
            Primitive::MenuOption(k) => self.menu_option_valid(k),
            Primitive::CloseMenu => in_menu,
            Primitive::Face(_) => !in_menu,
        }
    }

    fn menu_option_valid(&self, k: u8) -> bool {
        let c = self.c();
        match self.menu {
            Menu::None => false,
            Menu::Shop(s) => {
                let rows = self.shop_rows(s);
                if (k as usize) < rows.len() {
                    let (item, price) = rows[k as usize];
                    self.player.money >= price && self.can_add(item, 1)
                } else if self.is_smithy(s) && (k as usize) < rows.len() + 5 {
                    self.can_upgrade(Tool::ALL[k as usize - rows.len()])
                } else {
                    false
                }
            }
            Menu::Merchant => {
                k < 4 && {
                    let item = self.merchant_stock[k as usize];
                    self.player.money >= c.merchant_price(item) && self.can_add(item, 1)
                }
            }
            Menu::Craft => (k as usize) < c.recipes.len() && self.can_craft(k),
            Menu::Board => (k as usize) < c.requests.len() && self.can_deliver_any(k),
            Menu::Elevator => {
                let max = self.elevator_max();
                (k as usize) <= (max / c.mining.elevator_step) as usize
                    && !(self.player.loc == c.loc_id("mines")
                        && k * c.mining.elevator_step == self.mine.floor)
            }
        }
    }
    /// Deepest elevator floor available (multiple of elevator_step).
    pub fn elevator_max(&self) -> u8 {
        let step = self.c().mining.elevator_step;
        (self.mine.deepest / step) * step
    }
    /// Can recipe `r` be crafted now?
    pub fn can_craft(&self, r: u8) -> bool {
        let c = self.c();
        let rc = &c.recipes[r as usize];
        self.recipe_unlocked(r)
            && rc.inputs.iter().all(|(i, n)| self.count(*i) >= *n as u32)
            && self.can_add(rc.output, rc.count)
    }
    /// Would delivering to request `r` fill at least one slot or pay its coins?
    pub fn can_deliver_any(&self, r: u8) -> bool {
        let c = self.c();
        let req = &c.requests[r as usize];
        if self.restoration.request_done[r as usize] {
            return false;
        }
        if req.coins > 0 {
            return self.player.money >= req.coins;
        }
        let range = self.slots_of(r);
        range.enumerate().any(|(k, i)| {
            !self.restoration.slot_filled[i]
                && self.count(self.restoration.slot_item[i]) >= req.slots[k].count as u32
        })
    }

    /// What UseTool would do on the faced tile, if anything.
    fn use_tool_target(&self) -> Option<ToolUse> {
        let c = self.c();
        let (fx, fy) = self.facing_tile();
        let g = self.grid();
        if !g.in_bounds(fx, fy) {
            return None;
        }
        let (x, y) = (fx as u8, fy as u8);
        let obj = g.obj(x, y);
        let terrain = g.terrain(x, y);
        let idx = g.idx(x, y);
        match self.player.sel {
            Sel::Tool(t) => {
                if self.player.upgrading.map(|(u, _)| u == t).unwrap_or(false) {
                    return None;
                }
                match t {
                    Tool::Hoe => {
                        if obj == Obj::Weeds {
                            return Some(ToolUse::ClearWeeds);
                        }
                        if terrain == Terrain::Dirt
                            && obj == Obj::None
                            && g.soil[idx] & SOIL_TILLED == 0
                        {
                            return Some(ToolUse::Till);
                        }
                        None
                    }
                    Tool::Can => {
                        if terrain == Terrain::Water || obj == Obj::Well {
                            return Some(ToolUse::Refill);
                        }
                        if g.soil[idx] & SOIL_TILLED != 0
                            && g.soil[idx] & SOIL_WATERED == 0
                            && self.player.can_water > 0
                        {
                            return Some(ToolUse::Water);
                        }
                        None
                    }
                    Tool::Axe => match obj {
                        Obj::Weeds => Some(ToolUse::ClearWeeds),
                        Obj::Twig => Some(ToolUse::Twig),
                        Obj::Tree => Some(ToolUse::Tree),
                        Obj::Stump if self.player.tool_level[Tool::Axe as usize] >= 1 => {
                            Some(ToolUse::Stump)
                        }
                        _ => None,
                    },
                    Tool::Pickaxe => match obj {
                        Obj::Weeds => Some(ToolUse::ClearWeeds),
                        Obj::StoneDebris => Some(ToolUse::StoneDebris),
                        Obj::Rock => {
                            let kind = (g.data[idx] & 0xff) as usize;
                            if self.player.tool_level[Tool::Pickaxe as usize]
                                >= c.rocks[kind].min_pick
                            {
                                Some(ToolUse::Rock)
                            } else {
                                None
                            }
                        }
                        Obj::Boulder if self.player.tool_level[Tool::Pickaxe as usize] >= 2 => {
                            Some(ToolUse::Boulder)
                        }
                        _ => None,
                    },
                    Tool::Rod => {
                        if terrain == Terrain::Water && self.fishing_spot_here().is_some() {
                            Some(ToolUse::Fish)
                        } else {
                            None
                        }
                    }
                }
            }
            Sel::Slot(_) => {
                let (item, _) = self.selected_item()?;
                let it = &c.items[item as usize];
                match it.category {
                    Category::Seed => {
                        let crop = c.crop_by_seed[item as usize]?;
                        if g.soil[idx] & SOIL_TILLED != 0
                            && g.crop[idx] == NO_CROP
                            && obj == Obj::None
                            && self.crop_plantable(crop)
                        {
                            Some(ToolUse::Plant)
                        } else {
                            None
                        }
                    }
                    Category::Machine => {
                        if obj == Obj::MachinePad {
                            Some(ToolUse::Place)
                        } else {
                            None
                        }
                    }
                    Category::Fertilizer => {
                        if g.soil[idx] & SOIL_TILLED != 0 && g.soil[idx] & SOIL_FERT == 0 {
                            Some(ToolUse::Fertilize)
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            }
        }
    }

    /// The fishing spot POI whose water tile is the faced tile, if the player stands at a spot.
    pub fn fishing_spot_here(&self) -> Option<crate::content::PoiId> {
        let c = self.c();
        let (fx, fy) = self.facing_tile();
        c.pois
            .iter()
            .find(|p| {
                p.kind == PoiKind::FishingSpot
                    && p.location == self.player.loc
                    && (p.tile.0 as i32, p.tile.1 as i32) == (fx, fy)
            })
            .map(|p| p.id)
    }

    fn interact_target(&self) -> bool {
        let (fx, fy) = self.facing_tile();
        let g = self.grid();
        if !g.in_bounds(fx, fy) {
            return false;
        }
        let (x, y) = (fx as u8, fy as u8);
        let idx = g.idx(x, y);
        if g.crop[idx] != NO_CROP && self.crop_ready_at(idx) {
            return true;
        }
        match g.obj(x, y) {
            Obj::Forage => self.free_slots() > 0 || self.can_add(g.data[idx], 1),
            Obj::Crate => self
                .selected_item()
                .map(|(i, _)| self.shippable(i))
                .unwrap_or(false),
            Obj::Counter => self.shop_open(g.data[idx] as u8),
            Obj::Stall => self.merchant_open(),
            Obj::MillBoard | Obj::Bed | Obj::Elevator | Obj::MineExit | Obj::Ladder => true,
            Obj::MineEntrance => true,
            Obj::Well => self.player.can_water < self.can_capacity(),
            Obj::Machine => {
                let m = &self.machines[g.data[idx] as usize];
                if m.has_output {
                    self.can_add(m.output, m.out_count)
                } else if m.ready_at == 0 {
                    self.machine_load_rule(m.kind).is_some()
                } else {
                    false
                }
            }
            _ => false,
        }
    }
    /// Watering-can capacity at the current level.
    pub fn can_capacity(&self) -> u16 {
        self.c().tools.can_capacity[self.player.tool_level[Tool::Can as usize] as usize]
    }
    /// Is the crop at a tile index of the current grid ready?
    pub fn crop_ready_at(&self, idx: usize) -> bool {
        let g = self.grid();
        let cr = g.crop[idx];
        cr != NO_CROP && g.crop_days[idx] >= self.c().crops[cr as usize].grow_days
    }
    /// Which rule of a machine kind the selected item (plus held fuel) satisfies.
    pub fn machine_load_rule(&self, kind: crate::content::MachineId) -> Option<usize> {
        let c = self.c();
        let (item, n) = self.selected_item()?;
        let m = &c.machines[kind as usize];
        m.rules.iter().position(|r| {
            let input_ok = match r.input {
                InputSel::Item(i) => i == item,
                InputSel::Category(cat) => c.items[item as usize].category == cat,
            } && n >= r.input_count;
            let fuel_ok = r
                .fuel
                .map(|(f, fc)| self.count(f) >= fc as u32)
                .unwrap_or(true);
            input_ok && fuel_ok
        })
    }

    /// Execute a primitive. Invalid primitives behave as `Wait(1)` with `ok = false`.
    /// `ticks` and `day_ends` are derived from the absolute clock, so they are exact even when
    /// the step crosses 02:00 (pass-out) or ends the day by sleeping.
    pub fn step_primitive(&mut self, p: Primitive) -> PrimResult {
        let abs0 = self.abs_minute;
        let day0 = self.day;
        let mut r = self.step_primitive_inner(p);
        r.ticks = self.abs_minute - abs0;
        r.day_ends = (self.day - day0) as u8;
        r
    }

    fn step_primitive_inner(&mut self, p: Primitive) -> PrimResult {
        self.steps = self.steps.wrapping_add(1);
        self.last = crate::state::LastEvent::default();
        if !self.primitive_valid(p) {
            self.last.failed = true;
            self.advance(1);
            self.tick_machines();
            return PrimResult::default();
        }
        let mut r = PrimResult {
            ticks: 1,
            ok: true,
            day_ends: 0,
        };
        let mut extra_ticks: u32 = 0;
        match p {
            Primitive::Wait => extra_ticks = 9,
            Primitive::Move(d) => self.do_move(d),
            Primitive::UseTool => {
                let (t, ok) = self.do_use_tool();
                extra_ticks = t;
                r.ok = ok;
            }
            Primitive::Interact => {
                // Sleep ends the day directly.
                if self.facing_obj() == Obj::Bed {
                    self.end_day(false);
                    self.tick_machines();
                    return PrimResult {
                        ticks: 0,
                        ok: true,
                        day_ends: 0,
                    };
                }
                r.ok = self.do_interact();
            }
            Primitive::SelectSlot(i) => self.player.sel = Sel::Slot(i),
            Primitive::SelectTool(t) => self.player.sel = Sel::Tool(t),
            Primitive::Eat => {
                if let Some((item, _)) = self.selected_item() {
                    let e = self.c().items[item as usize].energy as i32;
                    self.remove_item(item, 1);
                    self.player.energy =
                        (self.player.energy + e).min(self.player.energy_max as i32);
                }
            }
            Primitive::OpenCraft => self.menu = Menu::Craft,
            Primitive::MenuOption(k) => r.ok = self.do_menu(k),
            Primitive::CloseMenu => self.menu = Menu::None,
            Primitive::Face(d) => self.player.facing = d,
        }
        // energy exhaustion → pass out
        let exhausted = self.player.energy <= 0;
        let de = self.advance(1 + extra_ticks);
        if de == 0 && exhausted {
            self.end_day(true);
        }
        self.tick_machines();
        r
    }

    fn do_move(&mut self, d: Dir) {
        let c = self.c();
        self.player.facing = d;
        let (nx, ny) = self.facing_tile();
        let g = self.grid();
        if !g.passable(nx, ny) {
            self.last.failed = true;
            return;
        }
        let (x, y) = (nx as u8, ny as u8);
        match g.obj(x, y) {
            Obj::Warp => {
                let w = &c.warps[g.data[g.idx(x, y)] as usize];
                if w.to == c.loc_id("greenhouse")
                    && !self.has_feature(crate::content::Feature::Greenhouse)
                {
                    self.last.failed = true;
                    return;
                }
                self.player.loc = w.to;
                self.player.x = w.to_tile.0;
                self.player.y = w.to_tile.1;
            }
            Obj::Ladder => {
                self.descend(self.mine.floor + 1);
            }
            _ => {
                self.player.x = x;
                self.player.y = y;
            }
        }
    }

    /// Enter the mines at a floor (regenerating the floor grid).
    pub fn descend(&mut self, floor: u8) {
        let c = self.c();
        let floor = floor.clamp(1, c.mining.floors);
        let mines = c.loc_id("mines");
        let mut g = std::mem::replace(
            self.world.grid_mut(mines),
            world::Grid::filled(1, 1, Terrain::Void),
        );
        world::generate_mine_floor(c, self.seed, self.day as u32, floor, &mut g);
        *self.world.grid_mut(mines) = g;
        self.mine.floor = floor;
        self.mine.floor_day = self.day;
        self.mine.deepest = self.mine.deepest.max(floor);
        let exit = &c.pois[c.poi_id("mine_exit") as usize];
        self.player.loc = mines;
        self.player.x = exit.stand.0;
        self.player.y = exit.stand.1;
        self.player.facing = Dir::N;
    }

    /// Leave the mines to the quarry entrance.
    pub fn surface(&mut self) {
        let c = self.c();
        let p = &c.pois[c.poi_id("mine_entrance") as usize];
        self.player.loc = p.location;
        self.player.x = p.stand.0;
        self.player.y = p.stand.1;
        self.player.facing = Dir::S;
        self.mine.floor = 0;
    }

    /// Returns (extra ticks, ok).
    fn do_use_tool(&mut self) -> (u32, bool) {
        let c = self.c();
        let Some(u) = self.use_tool_target() else {
            return (0, false);
        };
        let (fx, fy) = self.facing_tile();
        let (x, y) = (fx as u8, fy as u8);
        let loc = self.player.loc;
        let idx = self.grid().idx(x, y);
        let axe_l = self.player.tool_level[Tool::Axe as usize] as usize;
        let pick_l = self.player.tool_level[Tool::Pickaxe as usize] as usize;
        match u {
            ToolUse::Till => {
                let cost = self.energy_cost(c.tools.hoe_energy, Skill::Farming);
                self.spend_energy(cost);
                self.world.grid_mut(loc).soil[idx] |= SOIL_TILLED;
                self.add_xp(Skill::Farming, c.xp.hoe_till);
                (0, true)
            }
            ToolUse::Water => {
                let cost = self.energy_cost(c.tools.can_energy_per_tile, Skill::Farming);
                self.spend_energy(cost);
                self.player.can_water -= 1;
                self.world.grid_mut(loc).soil[idx] |= SOIL_WATERED;
                self.add_xp(Skill::Farming, c.xp.water_tile);
                (0, true)
            }
            ToolUse::Refill => {
                self.player.can_water = self.can_capacity();
                (1, true)
            }
            ToolUse::ClearWeeds => {
                let cost = self.energy_cost(1, Skill::Foraging);
                self.spend_energy(cost);
                self.world.grid_mut(loc).clear_obj(x, y);
                self.add_item(c.item_id("fiber"), 1);
                (0, true)
            }
            ToolUse::Twig => {
                let cost = self.energy_cost(c.tools.axe_energy, Skill::Foraging);
                self.spend_energy(cost);
                self.world.grid_mut(loc).clear_obj(x, y);
                self.add_item(c.item_id("wood"), 2);
                (0, true)
            }
            ToolUse::StoneDebris => {
                let cost = self.energy_cost(c.tools.pickaxe_energy, Skill::Mining);
                self.spend_energy(cost);
                self.world.grid_mut(loc).clear_obj(x, y);
                self.add_item(c.item_id("stone"), 2);
                (0, true)
            }
            ToolUse::Tree => {
                let cost = self.energy_cost(c.tools.axe_energy, Skill::Foraging);
                self.spend_energy(cost);
                let g = self.world.grid_mut(loc);
                let swings = g.data[idx] + 1;
                if swings >= c.tools.tree_hits[axe_l] as u16 {
                    g.clear_obj(x, y);
                    let wood = c.tools.tree_wood[axe_l] as u16;
                    self.add_item(c.item_id("wood"), wood);
                    if self.dyn_rng.chance_ppm(c.tools.tree_resin_ppm) {
                        self.add_item(c.item_id("resin"), 1);
                    }
                    self.add_xp(Skill::Foraging, c.xp.chop_tree);
                } else {
                    g.data[idx] = swings;
                }
                (0, true)
            }
            ToolUse::Stump => {
                let cost = self.energy_cost(c.tools.axe_energy, Skill::Foraging);
                self.spend_energy(cost);
                let g = self.world.grid_mut(loc);
                let swings = g.data[idx] + 1;
                if swings >= c.tools.stump_hits[axe_l.max(1)] as u16 {
                    g.clear_obj(x, y);
                    self.add_item(c.item_id("hardwood"), c.tools.stump_hardwood as u16);
                    self.add_xp(Skill::Foraging, c.xp.chop_stump);
                } else {
                    g.data[idx] = swings;
                }
                (0, true)
            }
            ToolUse::Boulder => {
                let cost = self.energy_cost(c.tools.pickaxe_energy, Skill::Mining);
                self.spend_energy(cost);
                let g = self.world.grid_mut(loc);
                let swings = g.data[idx] + 1;
                if swings >= c.tools.boulder_hits[pick_l.max(2)] as u16 {
                    g.clear_obj(x, y);
                    self.add_item(c.item_id("stone"), c.tools.boulder_stone as u16);
                    self.add_xp(Skill::Mining, c.mining.xp_per_rock * 2);
                } else {
                    g.data[idx] = swings;
                }
                (0, true)
            }
            ToolUse::Rock => {
                let cost = self.energy_cost(c.mining.energy_per_swing, Skill::Mining);
                self.spend_energy(cost);
                let g = self.world.grid_mut(loc);
                let data = g.data[idx];
                let kind = (data & 0xff) as usize;
                let swings = ((data >> 8) & 0x7f) + 1;
                let ladder = data & (1 << 15) != 0;
                let needed = (c.rocks[kind].hardness as i32 - pick_l as i32).max(1) as u16;
                if swings >= needed {
                    if ladder && loc == c.loc_id("mines") {
                        g.set_obj(x, y, Obj::Ladder, 0);
                    } else {
                        g.clear_obj(x, y);
                    }
                    let lvl = self.level(Skill::Mining) as u32;
                    let extra_ppm = (c.skills.mining_extra_drop_ppm_per_level * lvl).min(900_000);
                    let drops = c.rocks[kind].drops.clone();
                    for d in drops {
                        if self.dyn_rng.chance_ppm(d.ppm) {
                            let mut n = self.dyn_rng.range(d.min as u32, d.max as u32) as u16;
                            if self.dyn_rng.chance_ppm(extra_ppm) {
                                n += 1;
                            }
                            self.add_item(d.item, n);
                        }
                    }
                    self.add_xp(Skill::Mining, c.mining.xp_per_rock);
                } else {
                    g.data[idx] = (data & 0x80ff) | (swings << 8);
                }
                (0, true)
            }
            ToolUse::Fish => {
                let cost = self.energy_cost(c.fishing.energy_per_cast, Skill::Fishing);
                self.spend_energy(cost);
                let lvl = self.level(Skill::Fishing) as u32;
                let rod = self.player.tool_level[Tool::Rod as usize] as u32;
                let minus = (c.fishing.cast_minus_per_level as u32 * lvl)
                    .min(c.fishing.cast_min as u32 - 3);
                let dur = self.dyn_rng.range(
                    c.fishing.cast_min as u32 - minus,
                    c.fishing.cast_max as u32 - minus,
                );
                let spot = self.fishing_spot_here().unwrap_or(0);
                let season = self.season();
                let is_rain = self.weather.is_rain();
                let minute = self.minute;
                let eligible: Vec<(usize, u32)> = c
                    .fish
                    .iter()
                    .enumerate()
                    .filter(|(_, f)| {
                        f.spots.contains(&spot)
                            && crate::content::season_in(f.seasons, season)
                            && match f.weather {
                                WeatherReq::Any => true,
                                WeatherReq::Rain => is_rain,
                                WeatherReq::Sun => !is_rain,
                            }
                            && minute >= f.time.0
                            && minute < f.time.1
                            && rod >= f.min_rod as u32
                    })
                    .map(|(i, f)| (i, f.weight))
                    .collect();
                if eligible.is_empty() {
                    self.last.failed = true;
                    return (dur, false);
                }
                let weights: Vec<u32> = eligible.iter().map(|e| e.1).collect();
                let pick = eligible[self.dyn_rng.weighted(&weights)].0;
                let f = &c.fish[pick];
                let p = (c.fishing.base_ppm as i64
                    + c.fishing.skill_ppm as i64 * lvl as i64
                    + c.fishing.rod_ppm as i64 * rod as i64
                    - c.fishing.diff_ppm as i64 * f.difficulty as i64)
                    .clamp(c.fishing.min_ppm as i64, c.fishing.max_ppm as i64)
                    as u32;
                if self.dyn_rng.chance_ppm(p) {
                    self.add_item(f.item, 1);
                    self.add_xp(Skill::Fishing, c.fishing.xp_per_catch);
                    self.last.caught = Some(f.item);
                } else if self.dyn_rng.chance_ppm(c.fishing.junk_ppm_on_miss) {
                    let j = c.junk[self.dyn_rng.below(c.junk.len() as u32) as usize];
                    self.add_item(j, 1);
                }
                (dur, true)
            }
            ToolUse::Plant => {
                let (item, _) = self.selected_item().unwrap();
                let crop = c.crop_by_seed[item as usize].unwrap();
                self.remove_item(item, 1);
                let g = self.world.grid_mut(loc);
                g.crop[idx] = crop;
                g.crop_days[idx] = 0;
                (0, true)
            }
            ToolUse::Fertilize => {
                let (item, _) = self.selected_item().unwrap();
                self.remove_item(item, 1);
                self.world.grid_mut(loc).soil[idx] |= SOIL_FERT;
                (0, true)
            }
            ToolUse::Place => {
                let (item, _) = self.selected_item().unwrap();
                let kind = c.machine_by_item[item as usize].unwrap();
                self.remove_item(item, 1);
                let id = self.machines.len() as u16;
                self.machines.push(crate::state::MachineInst {
                    kind,
                    loc,
                    x,
                    y,
                    output: 0,
                    out_count: 0,
                    ready_at: 0,
                    has_output: false,
                });
                self.world.grid_mut(loc).set_obj(x, y, Obj::Machine, id);
                (0, true)
            }
        }
    }

    fn do_interact(&mut self) -> bool {
        let c = self.c();
        let (fx, fy) = self.facing_tile();
        let (x, y) = (fx as u8, fy as u8);
        let loc = self.player.loc;
        let idx = self.grid().idx(x, y);
        // harvest
        if self.grid().crop[idx] != NO_CROP && self.crop_ready_at(idx) {
            let cr = self.grid().crop[idx];
            let crop = &c.crops[cr as usize];
            let mut n = self
                .dyn_rng
                .range(crop.yield_min as u32, crop.yield_max as u32) as u16;
            let lvl = self.level(Skill::Farming) as u32;
            let fert = self.grid().soil[idx] & SOIL_FERT != 0;
            let extra = (crop.extra_yield_ppm_per_level * lvl + if fert { 150_000 } else { 0 })
                .min(900_000);
            if self.dyn_rng.chance_ppm(extra) {
                n += 1;
            }
            let left = self.add_item(crop.produce, n);
            if left == n {
                self.last.failed = true;
                return false;
            }
            self.add_xp(Skill::Farming, crop.xp);
            let g = self.world.grid_mut(loc);
            if crop.regrow_days > 0 {
                g.crop_days[idx] = crop.grow_days - crop.regrow_days;
            } else {
                g.crop[idx] = NO_CROP;
                g.crop_days[idx] = 0;
            }
            return true;
        }
        match self.grid().obj(x, y) {
            Obj::Forage => {
                let item = self.grid().data[idx];
                let lvl = self.level(Skill::Foraging) as u32;
                let n = if self
                    .dyn_rng
                    .chance_ppm((c.skills.foraging_double_ppm_per_level * lvl).min(900_000))
                {
                    2
                } else {
                    1
                };
                if self.add_item(item, n) == n {
                    return false;
                }
                self.world.grid_mut(loc).clear_obj(x, y);
                self.add_xp(Skill::Foraging, c.xp.forage_pickup);
                true
            }
            Obj::Crate => {
                let Some((item, n)) = self.selected_item() else {
                    return false;
                };
                if !self.shippable(item) {
                    return false;
                }
                self.remove_item(item, n);
                self.crate_items.push((item, n));
                true
            }
            Obj::Counter => {
                let shop = self.grid().data[idx] as u8;
                if self.shop_open(shop) {
                    self.menu = Menu::Shop(shop);
                    true
                } else {
                    false
                }
            }
            Obj::Stall => {
                if self.merchant_open() {
                    self.menu = Menu::Merchant;
                    true
                } else {
                    false
                }
            }
            Obj::MillBoard => {
                self.menu = Menu::Board;
                true
            }
            Obj::Well => {
                self.player.can_water = self.can_capacity();
                true
            }
            Obj::Elevator => {
                self.menu = Menu::Elevator;
                true
            }
            Obj::MineEntrance => {
                self.descend(1);
                true
            }
            Obj::MineExit => {
                self.surface();
                true
            }
            Obj::Ladder => {
                self.descend(self.mine.floor + 1);
                true
            }
            Obj::Machine => {
                let mi = self.grid().data[idx] as usize;
                let m = self.machines[mi];
                if m.has_output {
                    if self.add_item(m.output, m.out_count) == 0 {
                        let mm = &mut self.machines[mi];
                        mm.has_output = false;
                        mm.ready_at = 0;
                        mm.output = 0;
                        mm.out_count = 0;
                        true
                    } else {
                        false
                    }
                } else if m.ready_at == 0 {
                    let Some(ri) = self.machine_load_rule(m.kind) else {
                        return false;
                    };
                    let rule = c.machines[m.kind as usize].rules[ri].clone();
                    let (item, _) = self.selected_item().unwrap();
                    self.remove_item(item, rule.input_count);
                    if let Some((f, fc)) = rule.fuel {
                        self.remove_item(f, fc);
                    }
                    let mm = &mut self.machines[mi];
                    mm.output = rule.output;
                    mm.out_count = rule.output_count;
                    mm.ready_at = self.abs_minute + rule.minutes;
                    mm.has_output = false;
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    fn do_menu(&mut self, k: u8) -> bool {
        let c = self.c();
        match self.menu {
            Menu::None => false,
            Menu::Shop(s) => {
                let rows = self.shop_rows(s);
                if (k as usize) < rows.len() {
                    let (item, price) = rows[k as usize];
                    self.player.money -= price;
                    self.add_item(item, 1);
                    true
                } else {
                    let t = Tool::ALL[k as usize - rows.len()];
                    let lvl = self.player.tool_level[t as usize];
                    let u = c.upgrades[lvl as usize].clone();
                    self.player.money -= u.coins;
                    self.remove_item(u.bar, u.bars);
                    self.player.upgrading = Some((t, self.day + c.upgrade_days as u16));
                    if self.player.sel == Sel::Tool(t) {
                        self.player.sel = Sel::Tool(Tool::Hoe);
                    }
                    true
                }
            }
            Menu::Merchant => {
                let item = self.merchant_stock[k as usize];
                self.player.money -= c.merchant_price(item);
                self.add_item(item, 1);
                true
            }
            Menu::Craft => {
                let rc = c.recipes[k as usize].clone();
                for (i, n) in &rc.inputs {
                    self.remove_item(*i, *n);
                }
                self.add_item(rc.output, rc.count);
                self.add_xp(Skill::Farming, c.xp.craft);
                true
            }
            Menu::Board => {
                let req = &c.requests[k as usize];
                if req.coins > 0 {
                    self.pay_coins(k)
                } else {
                    self.deliver(k) > 0
                }
            }
            Menu::Elevator => {
                let floor = k * c.mining.elevator_step;
                self.menu = Menu::None;
                if floor == 0 {
                    if self.player.loc == c.loc_id("mines") {
                        self.surface();
                    }
                } else {
                    self.descend(floor);
                }
                true
            }
        }
    }
}

/// What a `UseTool` would do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ToolUse {
    Till,
    Water,
    Refill,
    ClearWeeds,
    Twig,
    StoneDebris,
    Tree,
    Stump,
    Boulder,
    Rock,
    Fish,
    Plant,
    Fertilize,
    Place,
}
