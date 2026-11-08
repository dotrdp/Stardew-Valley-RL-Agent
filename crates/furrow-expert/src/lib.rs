//! Scripted speedrun expert for Furrow (DESIGN.md §4).
//!
//! A **stateless, state-feedback** priority policy over macro actions: every decision is a pure
//! function of the current environment state (plus counter-keyed noise), so it can label
//! learner-visited states for DAgger.
//!
//! Decision procedure:
//! 1. **Targets.** For every open request choose the `k − filled` unfilled slots with the lowest
//!    estimated acquisition cost ([`Planner::item_cost`]: seasonal crop windows, forage seasons,
//!    fish windows and rod gates, mine depth and pickaxe gates, shops, merchant, recipes and
//!    machines). Targets expand into intermediate needs (bars, machine inputs, tool upgrades,
//!    coins).
//! 2. **Rules.** Walk an ordered list of rules (sleep before pass-out, deliver completions, tend
//!    the farm, craft/load machines, plant, upgrade, buy, time-gated fishing, forage, mine, chop,
//!    ship, money farming) and return the first one whose macro is currently unmasked.
//!
//! Information use: the rules read inventory, calendar, weather/forecast, restoration progress,
//! tool/skill levels, machine status, field/crop aggregates and ETAs — all of which are part of
//! the observation (§3.5) — plus static content tables. Known gap (documented in RESULTS):
//! nothing else, but the derived quantities are computed from `GameState` directly rather than
//! decoded from the f32 observation vector.

use std::sync::OnceLock;

use furrow::content::{
    content, season_in, Category, Content, CropId, InputSel, ItemId, PoiKind, RequestId, Season,
    SeasonMask, Tool, WeatherReq,
};
use furrow::macro_action::{macro_actions, Macro};
use furrow::rng::{stream, Domain};
use furrow::state::GameState;
use furrow::world::{Obj, NO_CROP};
use furrow::Env;

/// Expert configuration.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExpertConfig {
    /// Priority perturbation style (0 = canonical).
    pub style: u32,
    /// Probability (ppm) of executing a uniformly random valid macro instead of the clean label.
    pub noise_ppm: u32,
    /// Key for the counter-based noise stream.
    pub seed: u64,
}

/// The expert policy.
#[derive(Clone, Debug, Default)]
pub struct Expert {
    /// Configuration.
    pub cfg: ExpertConfig,
}

/// Intent codes (which rule fired), for analysis and hierarchical imitation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Intent {
    Sleep = 0,
    Eat = 1,
    Deliver = 2,
    Farm = 3,
    Craft = 4,
    Machines = 5,
    Plant = 6,
    Upgrade = 7,
    Buy = 8,
    Fish = 9,
    Forage = 10,
    Mine = 11,
    Chop = 12,
    Ship = 13,
    Money = 14,
    Fallback = 15,
}

// --------------------------------------------------------------------------------- static sources

struct Sources {
    crop_of: Vec<Option<CropId>>,
    forage_seasons: Vec<SeasonMask>,
    /// item -> list of (forage location index, season mask)
    forage_locs: Vec<Vec<(u8, SeasonMask)>>,
    fish_of: Vec<Vec<u8>>,
    /// item -> (min floor, min pick) of the cheapest rock dropping it
    mine_of: Vec<Option<(u8, u8)>>,
    shop_seasons: Vec<SeasonMask>,
    merchant: Vec<bool>,
    recipe_of: Vec<Option<u8>>,
    /// item -> (machine kind, rule index)
    machine_of: Vec<Option<(u8, usize)>>,
}

fn sources() -> &'static Sources {
    static S: OnceLock<Sources> = OnceLock::new();
    S.get_or_init(|| {
        let c = content();
        let n = c.items.len();
        let mut s = Sources {
            crop_of: vec![None; n],
            forage_seasons: vec![0; n],
            forage_locs: vec![Vec::new(); n],
            fish_of: vec![Vec::new(); n],
            mine_of: vec![None; n],
            shop_seasons: vec![0; n],
            merchant: vec![false; n],
            recipe_of: vec![None; n],
            machine_of: vec![None; n],
        };
        for cr in &c.crops {
            s.crop_of[cr.produce as usize] = Some(cr.id);
        }
        for (fi, fl) in c.forage.iter().enumerate() {
            for (si, t) in fl.tables.iter().enumerate() {
                for (it, _) in t {
                    s.forage_seasons[*it as usize] |= 1 << si;
                    let e = &mut s.forage_locs[*it as usize];
                    if let Some(x) = e.iter_mut().find(|x| x.0 == fi as u8) {
                        x.1 |= 1 << si;
                    } else {
                        e.push((fi as u8, 1 << si));
                    }
                }
            }
        }
        for f in &c.fish {
            s.fish_of[f.item as usize].push(f.id);
        }
        for r in &c.rocks {
            let min_floor = c
                .bands
                .iter()
                .filter(|b| b.rocks.iter().any(|(k, w)| *k == r.id && *w > 0))
                .map(|b| b.floors.0)
                .min()
                .unwrap_or(1);
            for d in &r.drops {
                let e = &mut s.mine_of[d.item as usize];
                let cand = (min_floor, r.min_pick);
                if e.is_none_or(|x| (cand.0, cand.1) < (x.0, x.1)) {
                    *e = Some(cand);
                }
            }
        }
        for sh in &c.shops {
            for r in &sh.stock {
                s.shop_seasons[r.item as usize] |= r.seasons;
            }
        }
        for i in &c.merchant.pool {
            s.merchant[*i as usize] = true;
        }
        for r in &c.recipes {
            s.recipe_of[r.output as usize] = Some(r.id);
        }
        for m in &c.machines {
            for (ri, r) in m.rules.iter().enumerate() {
                if s.machine_of[r.output as usize].is_none() {
                    s.machine_of[r.output as usize] = Some((m.id, ri));
                }
            }
        }
        s
    })
}

fn macro_index() -> &'static std::collections::BTreeMap<String, u32> {
    static M: OnceLock<std::collections::BTreeMap<String, u32>> = OnceLock::new();
    M.get_or_init(|| {
        macro_actions()
            .iter()
            .enumerate()
            .map(|(i, m)| (m.name(), i as u32))
            .collect()
    })
}

/// Macro id of a macro.
pub fn mid(m: Macro) -> u32 {
    *macro_index().get(&m.name()).expect("macro in table")
}

// ------------------------------------------------------------------------------------- planner

/// Per-decision plan derived from the state.
pub struct Planner<'a> {
    st: &'a GameState,
    c: &'static Content,
    /// Units wanted per item (targets + intermediate needs).
    pub want: Vec<u32>,
    /// Chosen target slots (flat slot indices).
    pub targets: Vec<usize>,
    /// Tool levels needed.
    pub need_tool: [u8; 5],
    /// Coins needed for pending coin requests and upgrades.
    pub coins_needed: u32,
}

const FAR: u32 = 10_000;

impl<'a> Planner<'a> {
    /// Build the plan.
    pub fn new(st: &'a GameState, style: u32) -> Planner<'a> {
        let c = content();
        let mut p = Planner {
            st,
            c,
            want: vec![0; c.items.len()],
            targets: Vec::new(),
            need_tool: [0; 5],
            coins_needed: 0,
        };
        p.choose_targets(style);
        p.expand();
        p
    }

    fn days_until(&self, s: Season) -> u32 {
        let cur = self.st.season() as u32;
        if cur == s as u32 {
            0
        } else {
            ((s as u32 + 4 - cur) % 4) * self.c.world.season_days as u32
                - self.st.day_of_season() as u32
        }
    }
    /// Earliest wait (days) until a season in `mask` (0 if now).
    fn wait_mask(&self, mask: SeasonMask) -> u32 {
        Season::ALL
            .iter()
            .filter(|s| season_in(mask, **s))
            .map(|s| self.days_until(*s))
            .min()
            .unwrap_or(FAR)
    }
    /// Days remaining in the contiguous growing window of a crop starting today (0 if out of season).
    pub fn crop_window(&self, cr: CropId) -> u32 {
        let crop = &self.c.crops[cr as usize];
        if !season_in(crop.seasons, self.st.season()) {
            return 0;
        }
        let mut w = self.st.days_left_in_season() as u32;
        let mut s = self.st.season() as u8;
        for _ in 0..3 {
            s = (s + 1) % 4;
            if season_in(crop.seasons, Season::from_index(s)) {
                w += self.c.world.season_days as u32;
            } else {
                break;
            }
        }
        w
    }

    /// Estimated acquisition cost (≈ days) of `count` more units of `item`.
    pub fn item_cost(&self, item: ItemId, count: u32, depth: u8) -> u32 {
        if count == 0 {
            return 0;
        }
        let st = self.st;
        let c = self.c;
        let s = sources();
        let held = st.count(item);
        if held >= count {
            return 0;
        }
        let missing = count - held;
        let mut best = FAR;
        // shop now / later
        if s.shop_seasons[item as usize] != 0 {
            let w = self.wait_mask(s.shop_seasons[item as usize]);
            let price = c
                .shop_row(item, Season::from_index(st.season() as u8))
                .map(|(sh, r)| c.shops[sh as usize].stock[r].price)
                .unwrap_or(200);
            best = best.min(w + 1 + (price * missing) / 800);
        }
        if s.merchant[item as usize] && st.merchant_today() && st.merchant_stock.contains(&item) {
            best = best.min(1 + c.merchant_price(item) * missing / 800);
        }
        // crop
        if let Some(cr) = s.crop_of[item as usize] {
            let crop = &c.crops[cr as usize];
            let gh = st.has_feature(furrow::content::Feature::Greenhouse);
            let window = self.crop_window(cr);
            let now = if window >= crop.grow_days as u32 {
                crop.grow_days as u32 + 1
            } else {
                FAR
            };
            let later = self
                .wait_mask(crop.seasons & !(1 << st.season() as u8))
                .saturating_add(crop.grow_days as u32 + 1);
            let ghc = if gh { crop.grow_days as u32 + 1 } else { FAR };
            best = best.min(now.min(later).min(ghc));
        }
        // forage
        if s.forage_seasons[item as usize] != 0 {
            let w = self.wait_mask(s.forage_seasons[item as usize]);
            best = best.min(w + 1 + missing / 3);
        }
        // fish
        for f in &s.fish_of[item as usize] {
            let fish = &c.fish[*f as usize];
            let rod = st.player.tool_level[Tool::Rod as usize];
            let gate = if rod >= fish.min_rod {
                0
            } else {
                8 * (fish.min_rod - rod) as u32
            };
            let weather = match fish.weather {
                WeatherReq::Rain => 4,
                _ => 0,
            };
            let diff = fish.difficulty as u32 / 15;
            best = best.min(self.wait_mask(fish.seasons) + 2 + gate + weather + diff + missing);
        }
        // mining
        if let Some((floor, pick)) = s.mine_of[item as usize] {
            let pl = st.player.tool_level[Tool::Pickaxe as usize];
            let gate = if pl >= pick {
                0
            } else {
                6 * (pick - pl) as u32
            };
            let depth_cost = (floor as u32).saturating_sub(st.mine.deepest as u32) / 6;
            let rarity = match c.items[item as usize].category {
                Category::Gem => 2 + missing * 2,
                _ => 1 + missing / 15,
            };
            best = best.min(2 + gate + depth_cost + rarity);
        }
        // basic resources
        let key = c.items[item as usize].key.as_str();
        match key {
            "wood" => best = best.min(1 + missing / 40),
            "stone" => best = best.min(1 + missing / 40),
            "fiber" => best = best.min(1),
            "resin" => best = best.min(2 + missing / 3),
            "hardwood" => {
                let axe = st.player.tool_level[Tool::Axe as usize];
                best = best.min(if axe >= 1 {
                    2 + missing / 8
                } else {
                    9 + missing / 8
                });
            }
            _ => {}
        }
        if depth < 2 {
            // recipe
            if let Some(r) = s.recipe_of[item as usize] {
                let rc = &c.recipes[r as usize];
                let batches = missing.div_ceil(rc.count as u32);
                let mut cost = 1u32;
                for (i, n) in &rc.inputs {
                    cost = cost.saturating_add(self.item_cost(*i, *n as u32 * batches, depth + 1));
                }
                if !st.recipe_unlocked(r) {
                    cost = cost.saturating_add(6);
                }
                best = best.min(cost);
            }
            // machine
            if let Some((mk, ri)) = s.machine_of[item as usize] {
                let m = &c.machines[mk as usize];
                let rule = &m.rules[ri];
                let batches = missing.div_ceil(rule.output_count as u32);
                let input_cost = match rule.input {
                    InputSel::Item(i) => {
                        self.item_cost(i, rule.input_count as u32 * batches, depth + 1)
                    }
                    InputSel::Category(Category::Crop) => 4,
                    InputSel::Category(Category::Fish) => 3 + batches,
                    InputSel::Category(_) => 6,
                };
                let have_machine = st.machines.iter().any(|x| x.kind == mk) || st.count(m.item) > 0;
                let machine_cost = if have_machine {
                    0
                } else {
                    self.item_cost(m.item, 1, depth + 1)
                };
                let days = (rule.minutes / 1440) * batches.div_ceil(2) + 1;
                best = best.min(
                    1u32.saturating_add(input_cost)
                        .saturating_add(machine_cost)
                        .saturating_add(days),
                );
            }
        }
        best
    }

    fn choose_targets(&mut self, style: u32) {
        let st = self.st;
        let c = self.c;
        for r in &c.requests {
            if st.restoration.request_done[r.id as usize] || r.coins > 0 {
                if r.coins > 0 && !st.restoration.request_done[r.id as usize] {
                    self.coins_needed += r.coins;
                }
                continue;
            }
            let filled = st.filled_in(r.id);
            let left = r.need.saturating_sub(filled) as usize;
            if left == 0 {
                continue;
            }
            let range = st.slots_of(r.id);
            let mut cands: Vec<(u32, usize)> = range
                .clone()
                .enumerate()
                .filter(|(_, i)| !st.restoration.slot_filled[*i])
                .map(|(k, i)| {
                    let item = st.restoration.slot_item[i];
                    let mut cost = self.item_cost(item, r.slots[k].count as u32, 0);
                    if style & 1 != 0 {
                        cost = cost.saturating_add((i as u32 * 7 + style) % 3);
                    }
                    (cost, i)
                })
                .collect();
            cands.sort();
            for (_, i) in cands.into_iter().take(left) {
                self.targets.push(i);
                let k = i - range.start;
                let item = st.restoration.slot_item[i];
                self.want[item as usize] += r.slots[k].count as u32;
            }
        }
    }

    /// Expand targets into intermediate needs (machine inputs, bars, tools, coins).
    fn expand(&mut self) {
        let st = self.st;
        let c = self.c;
        let s = sources();
        // tool needs from targeted items still missing
        for i in 0..c.items.len() {
            let w = self.want[i];
            if w == 0 || st.count(i as ItemId) >= w {
                continue;
            }
            for f in &s.fish_of[i] {
                let fish = &c.fish[*f as usize];
                // only count fish that are the cheapest way to get the item
                self.need_tool[Tool::Rod as usize] =
                    self.need_tool[Tool::Rod as usize].max(fish.min_rod.min(1));
            }
            if let Some((_, pick)) = s.mine_of[i] {
                self.need_tool[Tool::Pickaxe as usize] =
                    self.need_tool[Tool::Pickaxe as usize].max(pick);
            }
            if c.items[i].key == "hardwood" {
                self.need_tool[Tool::Axe as usize] = 1;
            }
        }
        // fish that strictly require rod 1 (pike/char/...) — max over targeted fish
        for i in 0..c.items.len() {
            if self.want[i] == 0 || st.count(i as ItemId) >= self.want[i] {
                continue;
            }
            let min_rod = s.fish_of[i]
                .iter()
                .map(|f| c.fish[*f as usize].min_rod)
                .min();
            if let Some(mr) = min_rod {
                self.need_tool[Tool::Rod as usize] = self.need_tool[Tool::Rod as usize].max(mr);
            }
        }
        // upgrades → bars + coins
        for t in Tool::ALL {
            let lvl = st.player.tool_level[t as usize];
            let pending = st.player.upgrading.map(|(u, _)| u == t).unwrap_or(false);
            let mut l = lvl + pending as u8;
            while l < self.need_tool[t as usize] && l < 3 {
                let u = &c.upgrades[l as usize];
                self.want[u.bar as usize] += u.bars as u32;
                self.coins_needed += u.coins;
                l += 1;
            }
        }
        // machine outputs → inputs (two passes so bars → ore resolves)
        for _pass in 0..2 {
            for i in 0..c.items.len() {
                let deficit = self.want[i]
                    .saturating_sub(st.count(i as ItemId) + self.pending_output(i as ItemId));
                if deficit == 0 {
                    continue;
                }
                if let Some((mk, ri)) = s.machine_of[i] {
                    let rule = &c.machines[mk as usize].rules[ri];
                    if s.crop_of[i].is_some() || s.forage_seasons[i] != 0 {
                        continue;
                    }
                    let batches = deficit.div_ceil(rule.output_count as u32);
                    if let InputSel::Item(inp) = rule.input {
                        self.want[inp as usize] =
                            self.want[inp as usize].max(batches * rule.input_count as u32);
                    }
                    if let Some((f, fc)) = rule.fuel {
                        self.want[f as usize] = self.want[f as usize].max(batches * fc as u32);
                    }
                    // ensure a machine
                    let m = &c.machines[mk as usize];
                    if !st.machines.iter().any(|x| x.kind == mk) && st.count(m.item) == 0 {
                        self.want[m.item as usize] = self.want[m.item as usize].max(1);
                    }
                } else if let Some(r) = s.recipe_of[i] {
                    let rc = &c.recipes[r as usize];
                    if c.items[i].category == Category::Machine
                        && st
                            .machines
                            .iter()
                            .any(|x| c.machines[x.kind as usize].item == i as ItemId)
                    {
                        continue;
                    }
                    let batches = deficit.div_ceil(rc.count as u32);
                    // fertilizer is cheaper to buy
                    if s.shop_seasons[i] != 0 && c.items[i].category != Category::Machine {
                        continue;
                    }
                    for (inp, n) in &rc.inputs {
                        self.want[*inp as usize] =
                            self.want[*inp as usize].max(batches * *n as u32);
                    }
                }
            }
        }
    }

    /// Units of an item that machines are currently producing.
    pub fn pending_output(&self, item: ItemId) -> u32 {
        self.st
            .machines
            .iter()
            .filter(|m| m.output == item && (m.ready_at != 0 || m.has_output))
            .map(|m| m.out_count as u32)
            .sum()
    }
    /// Units still missing for an item (want − held − in machines).
    pub fn deficit(&self, item: ItemId) -> u32 {
        self.want[item as usize].saturating_sub(self.st.count(item) + self.pending_output(item))
    }
    /// Expected yield of a crop's currently planted tiles.
    pub fn planted_output(&self, cr: CropId) -> u32 {
        let c = self.c;
        let crop = &c.crops[cr as usize];
        let mut n = 0u32;
        for l in [c.world.start_location, c.loc_id("greenhouse")] {
            let g = self.st.world.grid(l);
            n += g.crop.iter().filter(|x| **x == cr).count() as u32;
        }
        let avg = (crop.yield_min as u32 + crop.yield_max as u32).div_ceil(2);
        n * avg
    }
}

// --------------------------------------------------------------------------------------- expert

impl Expert {
    /// New expert.
    pub fn new(cfg: ExpertConfig) -> Self {
        Expert { cfg }
    }

    /// Clean expert label for the current state.
    pub fn act(&self, env: &Env) -> u32 {
        self.act_with_intent(env).0
    }

    /// `(executed, clean)`: with probability `noise_ppm` the executed action is a uniformly random
    /// valid macro (counter-keyed by seed, episode seed, day, decision index — no hidden state).
    pub fn noisy_act(&self, env: &Env) -> (u32, u32) {
        let clean = self.act(env);
        if self.cfg.noise_ppm == 0 {
            return (clean, clean);
        }
        let st = env.state();
        let key = st.steps.wrapping_mul(2654435761) ^ (self.cfg.style << 20);
        let mut r = stream(
            self.cfg.seed ^ st.seed.rotate_left(17),
            Domain::Expert,
            st.day as u32,
            key ^ st.decisions_today as u32,
        );
        if !r.chance_ppm(self.cfg.noise_ppm) {
            return (clean, clean);
        }
        let mask = env.action_mask();
        let n = mask.iter().filter(|m| **m).count() as u32;
        if n == 0 {
            return (clean, clean);
        }
        let mut k = r.below(n);
        for (i, m) in mask.iter().enumerate() {
            if *m {
                if k == 0 {
                    return (i as u32, clean);
                }
                k -= 1;
            }
        }
        (clean, clean)
    }

    /// Clean label plus the rule that fired.
    pub fn act_with_intent(&self, env: &Env) -> (u32, Intent) {
        let st = env.state();
        let mask = env.action_mask();
        let c = content();
        let valid = |m: Macro| mask[mid(m) as usize];
        let style = self.cfg.style;
        let p = Planner::new(st, style);
        let in_mines = st.player.loc == c.loc_id("mines");

        // only Sleep allowed
        let n_valid = mask.iter().filter(|m| **m).count();
        if n_valid == 1 && valid(Macro::Sleep) {
            return (mid(Macro::Sleep), Intent::Sleep);
        }
        if n_valid == 0 {
            return (mid(Macro::Wait30), Intent::Fallback);
        }
        // ---- sleep before pass-out
        let eta_bed = eta(env, c.poi_id("bed"));
        let late = st.minute as u32 + eta_bed.min(300) + 110 >= c.world.day_end as u32;
        if late && valid(Macro::Sleep) {
            return (mid(Macro::Sleep), Intent::Sleep);
        }
        // ---- energy
        if st.player.energy < 30 && valid(Macro::Eat) {
            return (mid(Macro::Eat), Intent::Eat);
        }
        if st.player.energy < 12 && valid(Macro::Sleep) {
            return (mid(Macro::Sleep), Intent::Sleep);
        }
        // ---- keep bag space: ship unreserved stacks when nearly full
        if st.free_slots() <= 2 && valid(Macro::ShipSurplus) {
            return (mid(Macro::ShipSurplus), Intent::Ship);
        }
        // ---- deliver when it completes a request, or when we hold many deliverables
        if valid(Macro::Deliver) && !in_mines {
            let (completes, fillable) = deliverable(st, &p);
            let near_board = st.player.loc == c.loc_id("village");
            if completes || fillable >= 3 || (near_board && fillable > 0) {
                return (mid(Macro::Deliver), Intent::Deliver);
            }
        }
        // ---- coins
        for r in &c.coin_requests {
            let m = Macro::PayCoins(*r);
            if valid(m) && st.player.money >= c.requests[*r as usize].coins + upgrade_coins(st, &p)
            {
                return (mid(m), Intent::Deliver);
            }
        }
        // ---- farm upkeep
        if valid(Macro::CollectMachines) {
            return (mid(Macro::CollectMachines), Intent::Machines);
        }
        if valid(Macro::Harvest) {
            return (mid(Macro::Harvest), Intent::Farm);
        }
        if valid(Macro::WaterCrops) && !st.weather.is_rain() {
            return (mid(Macro::WaterCrops), Intent::Farm);
        }
        // ---- crafting needed items/machines
        for rc in &c.recipes {
            let m = Macro::Craft(rc.id);
            if !valid(m) {
                continue;
            }
            let out = rc.output;
            let is_machine = c.items[out as usize].category == Category::Machine;
            let wanted = p.deficit(out) > 0;
            // machines: build when its outputs are wanted
            let machine_useful = is_machine && machine_wanted(st, &p, out);
            if (wanted && !is_machine) || machine_useful {
                // do not consume inputs that are themselves targets (unless the output is targeted)
                if rc.inputs.iter().all(|(i, n)| {
                    st.count(*i)
                        >= *n as u32
                            + p.want[*i as usize].saturating_sub(if is_machine {
                                p.want[*i as usize]
                            } else {
                                0
                            })
                        || is_machine
                        || wanted
                }) {
                    return (mid(m), Intent::Craft);
                }
            }
        }
        if valid(Macro::LoadMachines) && load_useful(st, &p) {
            return (mid(Macro::LoadMachines), Intent::Machines);
        }
        // ---- at the village: buy everything first, then go plant
        if st.player.loc == c.loc_id("village") {
            if let Some(a) = self.buy_rule(st, &p, &valid) {
                return (a, Intent::Buy);
            }
            if st.player.money < p.coins_needed + 800 {
                if let Some(a) = self.money_crop_rule(st, &p, &valid) {
                    return (a, Intent::Money);
                }
            }
        }
        // ---- plant held seeds of targeted / money crops
        for cr in &c.crops {
            let held_seeds = st.count(cr.seed);
            if held_seeds == 0 {
                continue;
            }
            let farm = Macro::Plant(cr.id, false);
            let gh = Macro::Plant(cr.id, true);
            let window = p.crop_window(cr.id);
            if valid(farm) && window >= cr.grow_days as u32 {
                return (mid(farm), Intent::Plant);
            }
            if valid(gh) {
                return (mid(gh), Intent::Plant);
            }
        }
        // ---- upgrades
        for t in [Tool::Pickaxe, Tool::Rod, Tool::Axe] {
            let lvl = st.player.tool_level[t as usize];
            if lvl < p.need_tool[t as usize]
                && valid(Macro::UpgradeTool(t))
                && !upgrade_blocks_now(st, t)
            {
                return (mid(Macro::UpgradeTool(t)), Intent::Upgrade);
            }
        }
        // ---- buy (seeds for targeted crops, targeted shop/merchant items)
        if let Some(a) = self.buy_rule(st, &p, &valid) {
            return (a, Intent::Buy);
        }
        // ---- money pressure: keep a seed/upgrade budget flowing
        let money_goal = p.coins_needed + 800;
        if st.player.money < money_goal {
            if valid(Macro::ShipSurplus) && surplus_value(st) >= 150 {
                return (mid(Macro::ShipSurplus), Intent::Ship);
            }
            if let Some(a) = self.money_crop_rule(st, &p, &valid) {
                return (a, Intent::Money);
            }
        }
        // ---- time-gated fishing
        if let Some(a) = fish_rule(st, &p, &valid, true) {
            return (a, Intent::Fish);
        }
        // ---- seasonal sources before all-year ones: forage, then fish, then mining
        let fish_first = style & 2 != 0;
        if let Some(a) = forage_rule(st, &p, &valid) {
            return (a, Intent::Forage);
        }
        let fish = fish_rule(st, &p, &valid, false);
        if fish_first {
            if let Some(a) = fish {
                return (a, Intent::Fish);
            }
        }
        if let Some(a) = mine_rule(st, &p, &valid) {
            return (a, Intent::Mine);
        }
        if let Some(a) = fish {
            return (a, Intent::Fish);
        }
        // ---- wood / hardwood / resin / fiber
        let wood = c.item_id("wood");
        let hardwood = c.item_id("hardwood");
        let resin = c.item_id("resin");
        if p.deficit(hardwood) > 0 && valid(Macro::ChopHardwood) {
            return (mid(Macro::ChopHardwood), Intent::Chop);
        }
        if (p.deficit(wood) > 0 || p.deficit(resin) > 0) && valid(Macro::ChopWood) {
            return (mid(Macro::ChopWood), Intent::Chop);
        }
        let stone = c.item_id("stone");
        let fiber = c.item_id("fiber");
        if (p.deficit(stone) > 0 || p.deficit(fiber) > 0) && valid(Macro::ClearField) {
            return (mid(Macro::ClearField), Intent::Chop);
        }
        // ---- money (fallback)
        if st.player.money < money_goal {
            if valid(Macro::ShipSurplus) {
                return (mid(Macro::ShipSurplus), Intent::Ship);
            }
            if let Some(a) = self.money_crop_rule(st, &p, &valid) {
                return (a, Intent::Money);
            }
            // fish for money at the most productive spot
            if st.minute < 1260 {
                if let Some(a) = money_fish(st, &valid) {
                    return (a, Intent::Money);
                }
            }
        }
        if in_mines && valid(Macro::GoTo(c.world.start_location)) {
            return (mid(Macro::GoTo(c.world.start_location)), Intent::Fallback);
        }
        // nothing useful left today: end the day (time is the only resource that matters)
        if valid(Macro::Sleep) {
            return (mid(Macro::Sleep), Intent::Sleep);
        }
        (mid(Macro::Wait30), Intent::Fallback)
    }

    fn buy_rule(&self, st: &GameState, p: &Planner, valid: &dyn Fn(Macro) -> bool) -> Option<u32> {
        let c = content();
        // seeds for targeted crops
        for cr in &c.crops {
            let need = p.deficit(cr.produce);
            if need == 0 {
                continue;
            }
            let window = p.crop_window(cr.id);
            let gh = st.has_feature(furrow::content::Feature::Greenhouse);
            if window < cr.grow_days as u32 && !gh {
                continue;
            }
            let avg = (cr.yield_min as u32 + cr.yield_max as u32).div_ceil(2);
            let harvests = if cr.regrow_days > 0 {
                1 + window.saturating_sub(cr.grow_days as u32) / cr.regrow_days as u32
            } else {
                1
            };
            let per_seed = avg * harvests.max(1);
            let expect = p.planted_output(cr.id);
            let need_left = need.saturating_sub(expect);
            if need_left == 0 {
                continue;
            }
            let seeds = need_left.div_ceil(per_seed) + 1;
            let held = st.count(cr.seed);
            if held >= seeds {
                continue;
            }
            if let Some(a) = buy_q(st, cr.seed, seeds - held, valid) {
                return Some(a);
            }
        }
        // targeted items purchasable directly (bread, merchant specials, ores when cheap)
        for (i, it) in c.items.iter().enumerate() {
            let d = p.deficit(i as ItemId);
            if d == 0 || it.category == Category::Seed {
                continue;
            }
            let Some((price, seller)) = st.best_seller(i as ItemId) else {
                continue;
            };
            // merchant: buy targeted rarities; shops: buy only cheap items or food/fertilizer
            let worth = seller.is_none()
                || price <= 60
                || matches!(it.category, Category::Food | Category::Fertilizer);
            if !worth {
                continue;
            }
            if st.player.money < price.saturating_mul(d.min(25)) + 200 && seller.is_some() {
                continue;
            }
            if let Some(a) = buy_q(st, i as ItemId, d, valid) {
                return Some(a);
            }
        }
        None
    }

    fn money_crop_rule(
        &self,
        st: &GameState,
        p: &Planner,
        valid: &dyn Fn(Macro) -> bool,
    ) -> Option<u32> {
        let c = content();
        // planted tile budget for money crops
        let budget = 24 + (self.cfg.style % 3) * 8;
        let planted: u32 = {
            let g = st.world.grid(c.world.start_location);
            g.crop.iter().filter(|x| **x != NO_CROP).count() as u32
        };
        if planted >= budget {
            return None;
        }
        let mut best: Option<(i64, CropId)> = None;
        for cr in &c.crops {
            let window = p.crop_window(cr.id);
            if window < cr.grow_days as u32 {
                continue;
            }
            let Some((sh, row)) = c.shop_row(cr.seed, st.season()) else {
                continue;
            };
            let price = c.shops[sh as usize].stock[row].price as i64;
            let avg = (cr.yield_min as i64 + cr.yield_max as i64 + 1) / 2;
            let sell = c.items[cr.produce as usize].sell as i64;
            let harvests = if cr.regrow_days > 0 {
                1 + (window as i64 - cr.grow_days as i64) / cr.regrow_days as i64
            } else {
                1
            };
            let profit = (sell * avg * harvests - price) * 100 / (window.min(30) as i64).max(1);
            if best.is_none_or(|b| profit > b.0) {
                best = Some((profit, cr.id));
            }
        }
        let (_, cr) = best?;
        let crop = &c.crops[cr as usize];
        let held = st.count(crop.seed);
        let want = (budget - planted).min(25);
        if held >= want {
            return None;
        }
        let _ = p;
        buy_q(st, crop.seed, want - held, valid)
    }
}

/// Sell value of unreserved shippable items in the bag.
fn surplus_value(st: &GameState) -> u32 {
    let c = content();
    st.player
        .bag
        .iter()
        .filter(|(i, n)| *n > 0 && st.shippable(*i) && st.reserved(*i) == 0)
        .map(|(i, n)| c.items[*i as usize].sell * *n as u32)
        .sum()
}

fn buy_q(st: &GameState, item: ItemId, n: u32, valid: &dyn Fn(Macro) -> bool) -> Option<u32> {
    let _ = st;
    for q in [25u8, 5, 1] {
        if n >= q as u32 || q == 1 {
            let m = Macro::Buy(item, q);
            if valid(m) {
                return Some(mid(m));
            }
        }
    }
    None
}

fn eta(env: &Env, poi: u8) -> u32 {
    let layout = furrow::obs::global_layout();
    let (_, off, _) = layout.iter().find(|e| e.0 == "eta").copied().unwrap();
    let v = env.obs_global()[off + poi as usize];
    if v >= 1.0 {
        600
    } else {
        (v * 600.0) as u32
    }
}

/// (some request completes by delivering now, number of fillable target slots)
fn deliverable(st: &GameState, p: &Planner) -> (bool, u32) {
    let c = content();
    let mut completes = false;
    let mut fillable = 0;
    for r in &c.requests {
        if st.restoration.request_done[r.id as usize] || r.coins > 0 {
            continue;
        }
        let range = st.slots_of(r.id);
        let mut f = 0;
        for (k, i) in range.enumerate() {
            if !st.restoration.slot_filled[i]
                && st.count(st.restoration.slot_item[i]) >= r.slots[k].count as u32
            {
                f += 1;
                if p.targets.contains(&i) {
                    fillable += 1;
                }
            }
        }
        if f > 0 && st.filled_in(r.id) as u32 + f >= r.need as u32 {
            completes = true;
        }
    }
    (completes, fillable)
}

fn upgrade_coins(st: &GameState, p: &Planner) -> u32 {
    let c = content();
    let mut coins = 0;
    for t in Tool::ALL {
        let mut l = st.player.tool_level[t as usize];
        while l < p.need_tool[t as usize] && l < 3 {
            coins += c.upgrades[l as usize].coins;
            l += 1;
        }
    }
    coins
}

/// Avoid starting an upgrade that disables a tool needed for something time-critical today.
fn upgrade_blocks_now(st: &GameState, t: Tool) -> bool {
    // never take the rod away on a rainy day (rain-only fish), or the can when crops need water
    match t {
        Tool::Rod => st.weather.is_rain(),
        Tool::Can => st.farm_needs_water(),
        _ => false,
    }
}

fn machine_wanted(st: &GameState, p: &Planner, machine_item: ItemId) -> bool {
    let c = content();
    if st
        .machines
        .iter()
        .any(|m| c.machines[m.kind as usize].item == machine_item)
    {
        return false;
    }
    if p.want[machine_item as usize] > st.count(machine_item) {
        return true;
    }
    let Some(mk) = c.machine_by_item[machine_item as usize] else {
        return false;
    };
    c.machines[mk as usize]
        .rules
        .iter()
        .any(|r| p.deficit(r.output) > 0)
}

fn load_useful(st: &GameState, p: &Planner) -> bool {
    let c = content();
    st.machines.iter().any(|m| {
        m.ready_at == 0
            && !m.has_output
            && c.machines[m.kind as usize]
                .rules
                .iter()
                .any(|r| p.deficit(r.output) > 0)
    }) && st
        .machines
        .iter()
        .any(|m| m.ready_at == 0 && !m.has_output && st.any_input_for(m.kind))
}

fn fish_rule(
    st: &GameState,
    p: &Planner,
    valid: &dyn Fn(Macro) -> bool,
    gated_only: bool,
) -> Option<u32> {
    let c = content();
    let rod = st.player.tool_level[Tool::Rod as usize];
    if st
        .player
        .upgrading
        .map(|(u, _)| u == Tool::Rod)
        .unwrap_or(false)
    {
        return None;
    }
    let mut best: Option<(u32, u8)> = None;
    // dried fish wants generic fish
    let dried = c.item_id("dried_fish");
    let generic_fish = p.deficit(dried) * 3;
    for spot in &c.fishing_spots {
        let m = Macro::Fish(*spot);
        if !valid(m) {
            continue;
        }
        let mut score = 0u32;
        for f in &c.fish {
            if !f.spots.contains(spot) || rod < f.min_rod || !season_in(f.seasons, st.season()) {
                continue;
            }
            let weather_ok = match f.weather {
                WeatherReq::Any => true,
                WeatherReq::Rain => st.weather.is_rain(),
                WeatherReq::Sun => !st.weather.is_rain(),
            };
            // reachable within its window: allow if the window opens within 60 minutes
            let time_ok = st.minute + 60 >= f.time.0 && st.minute + 20 < f.time.1;
            if !weather_ok || !time_ok {
                continue;
            }
            let d = p.deficit(f.item);
            if d == 0 {
                continue;
            }
            let gated = f.weather != WeatherReq::Any
                || f.time.1 < c.world.day_end
                || f.seasons.count_ones() <= 1;
            if gated_only && !gated {
                continue;
            }
            score += 10 + if gated { 20 } else { 0 } + (100 - f.difficulty as u32) / 10;
        }
        let rack = c.item_id("drying_rack");
        let have_rack = st
            .machines
            .iter()
            .any(|m| c.machines[m.kind as usize].item == rack);
        if !gated_only && generic_fish > 0 && have_rack {
            score += 5;
        }
        if score > 0 && best.is_none_or(|b| score > b.0) {
            best = Some((score, *spot));
        }
    }
    best.map(|b| mid(Macro::Fish(b.1)))
}

fn money_fish(st: &GameState, valid: &dyn Fn(Macro) -> bool) -> Option<u32> {
    let c = content();
    // pick the spot whose eligible fish have the best expected value per cast
    let rod = st.player.tool_level[Tool::Rod as usize];
    let mut best: Option<(u32, u8)> = None;
    for spot in &c.fishing_spots {
        if !valid(Macro::Fish(*spot)) {
            continue;
        }
        let mut wsum = 0u32;
        let mut vsum = 0u32;
        for f in &c.fish {
            if f.spots.contains(spot)
                && rod >= f.min_rod
                && season_in(f.seasons, st.season())
                && st.minute >= f.time.0
                && st.minute < f.time.1
            {
                wsum += f.weight;
                vsum +=
                    f.weight * c.items[f.item as usize].sell * (100 - f.difficulty as u32) / 100;
            }
        }
        if wsum == 0 {
            continue;
        }
        let ev = vsum / wsum;
        if best.is_none_or(|b| ev > b.0) {
            best = Some((ev, *spot));
        }
    }
    best.map(|b| mid(Macro::Fish(b.1)))
}

fn forage_rule(st: &GameState, p: &Planner, valid: &dyn Fn(Macro) -> bool) -> Option<u32> {
    let c = content();
    let s = sources();
    let mut best: Option<(u32, u8)> = None;
    for (fi, fl) in c.forage.iter().enumerate() {
        let m = Macro::Forage(fi as u8);
        if !valid(m) {
            continue;
        }
        let table = &fl.tables[st.season() as usize];
        let mut score = 0;
        for (it, w) in table {
            if p.deficit(*it) > 0 {
                score += *w;
            }
        }
        // quarry forage (quartz) also via mining; fine
        let _ = s;
        if score > 0 && best.is_none_or(|b| score > b.0) {
            best = Some((score, fi as u8));
        }
    }
    best.map(|b| mid(Macro::Forage(b.1)))
}

fn mine_rule(st: &GameState, p: &Planner, valid: &dyn Fn(Macro) -> bool) -> Option<u32> {
    let c = content();
    let s = sources();
    let pick = st.player.tool_level[Tool::Pickaxe as usize];
    if st
        .player
        .upgrading
        .map(|(u, _)| u == Tool::Pickaxe)
        .unwrap_or(false)
    {
        return None;
    }
    let mut need_floor: Option<u8> = None;
    for (i, it) in c.items.iter().enumerate() {
        if p.deficit(i as ItemId) == 0 {
            continue;
        }
        // items also obtainable more cheaply elsewhere (crops/forage) are not mined for
        if s.crop_of[i].is_some() {
            continue;
        }
        let Some((floor, min_pick)) = s.mine_of[i] else {
            continue;
        };
        if min_pick > pick {
            continue;
        }
        if matches!(it.key.as_str(), "stone")
            && st.field_debris() > 20
            && st.player.loc != c.loc_id("mines")
        {
            // stone is also on the farm; still allow mining
        }
        need_floor = Some(need_floor.map_or(floor, |f| f.max(floor)));
    }
    let floor = need_floor?;
    let target = if floor > 1 { floor + 2 } else { 6 };
    if st.mine.deepest < target && valid(Macro::MineDescend) {
        return Some(mid(Macro::MineDescend));
    }
    if valid(Macro::MineFloor) {
        // at a floor with nothing breakable left → descend if possible
        let mines = c.loc_id("mines");
        if st.player.loc == mines {
            let g = st.world.grid(mines);
            let rocks = (0..g.obj_plane().len())
                .filter(|i| {
                    g.obj_plane()[*i] == Obj::Rock as u8
                        && c.rocks[(g.data[*i] & 0xff) as usize].min_pick <= pick
                })
                .count();
            if rocks == 0 && valid(Macro::MineDescend) {
                return Some(mid(Macro::MineDescend));
            }
        }
        return Some(mid(Macro::MineFloor));
    }
    if valid(Macro::MineDescend) {
        return Some(mid(Macro::MineDescend));
    }
    None
}

/// Helper used by tests/CLI: the request ids still open.
pub fn open_requests(st: &GameState) -> Vec<RequestId> {
    content()
        .requests
        .iter()
        .filter(|r| !st.restoration.request_done[r.id as usize])
        .map(|r| r.id)
        .collect()
}

/// Helper: does a POI kind exist (sanity for tests).
pub fn has_poi(kind: PoiKind) -> bool {
    content().pois.iter().any(|p| p.kind == kind)
}

#[cfg(test)]
mod tests {
    use super::*;
    use furrow::EnvConfig;

    #[test]
    fn every_macro_has_an_id() {
        for m in macro_actions() {
            let _ = mid(*m);
        }
    }

    #[test]
    fn expert_is_stateless_and_deterministic() {
        let mut env = Env::new(EnvConfig {
            seed: 3,
            ..Default::default()
        });
        let e = Expert::new(ExpertConfig::default());
        for _ in 0..200 {
            if env.is_done() {
                break;
            }
            let a = e.act(&env);
            let b = Expert::new(ExpertConfig::default()).act(&env);
            assert_eq!(a, b);
            env.step(a);
        }
    }
}
