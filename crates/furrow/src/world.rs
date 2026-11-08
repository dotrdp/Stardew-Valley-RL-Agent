//! Tile world: SoA grids per location, procedural layout from the layout stream, and BFS
//! pathfinding over per-env scratch buffers (DESIGN.md §2.2, §2.5).

use crate::content::{self, Content, Dir, ItemId, LocId, LocKind, PoiKind, RockId, Season, WingId};
use crate::hash::StableHasher;
use crate::rng::{stream, Domain, Pcg32};

/// Terrain kinds (grid channel 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Terrain {
    Void = 0,
    Grass = 1,
    Dirt = 2,
    Path = 3,
    Sand = 4,
    Floor = 5,
    Water = 6,
    Wall = 7,
    Bridge = 8,
}

impl Terrain {
    #[inline]
    fn from_u8(v: u8) -> Terrain {
        match v {
            1 => Terrain::Grass,
            2 => Terrain::Dirt,
            3 => Terrain::Path,
            4 => Terrain::Sand,
            5 => Terrain::Floor,
            6 => Terrain::Water,
            7 => Terrain::Wall,
            8 => Terrain::Bridge,
            _ => Terrain::Void,
        }
    }
    /// Can a player stand here (ignoring objects)?
    #[inline]
    pub fn walkable(self) -> bool {
        matches!(
            self,
            Terrain::Grass | Terrain::Dirt | Terrain::Path | Terrain::Sand | Terrain::Floor | Terrain::Bridge
        )
    }
}

/// Object kinds (grid channel 1). `obj_data` meaning per kind is documented on each variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Obj {
    None = 0,
    /// Debris: any tool / free. Yields fiber.
    Weeds = 1,
    /// Debris: axe. Yields wood.
    Twig = 2,
    /// Debris: pickaxe. Yields stone.
    StoneDebris = 3,
    /// Tree. data = hits remaining.
    Tree = 4,
    /// Hardwood stump. data = hits remaining (axe ≥ 1).
    Stump = 5,
    /// Boulder. data = hits remaining (pickaxe ≥ 2).
    Boulder = 6,
    /// Rock node. data = rock kind | hits remaining << 8 | ladder flag << 15.
    Rock = 7,
    /// Forage item. data = item id.
    Forage = 8,
    Building = 9,
    /// Shop counter. data = shop id.
    Counter = 10,
    Bed = 11,
    Crate = 12,
    Well = 13,
    MillBoard = 14,
    /// Revealed ladder down.
    Ladder = 15,
    Elevator = 16,
    /// Empty machine pad.
    MachinePad = 17,
    /// Placed machine. data = machine instance index.
    Machine = 18,
    /// Warp. data = warp id.
    Warp = 19,
    Stall = 20,
    MineExit = 21,
    MineEntrance = 22,
}

impl Obj {
    #[inline]
    fn from_u8(v: u8) -> Obj {
        // SAFETY-free: match on all discriminants.
        match v {
            1 => Obj::Weeds,
            2 => Obj::Twig,
            3 => Obj::StoneDebris,
            4 => Obj::Tree,
            5 => Obj::Stump,
            6 => Obj::Boulder,
            7 => Obj::Rock,
            8 => Obj::Forage,
            9 => Obj::Building,
            10 => Obj::Counter,
            11 => Obj::Bed,
            12 => Obj::Crate,
            13 => Obj::Well,
            14 => Obj::MillBoard,
            15 => Obj::Ladder,
            16 => Obj::Elevator,
            17 => Obj::MachinePad,
            18 => Obj::Machine,
            19 => Obj::Warp,
            20 => Obj::Stall,
            21 => Obj::MineExit,
            22 => Obj::MineEntrance,
            _ => Obj::None,
        }
    }
    /// Does this object block movement?
    #[inline]
    pub fn blocks(self) -> bool {
        !matches!(self, Obj::None | Obj::Warp | Obj::Ladder | Obj::MachinePad | Obj::Forage)
    }
    /// Is this a debris/tree/rock-like breakable?
    pub fn breakable(self) -> bool {
        matches!(
            self,
            Obj::Weeds | Obj::Twig | Obj::StoneDebris | Obj::Tree | Obj::Stump | Obj::Boulder | Obj::Rock
        )
    }
}

/// Soil flags (grid channel 3).
pub const SOIL_TILLED: u8 = 1;
/// Soil flag: watered today.
pub const SOIL_WATERED: u8 = 2;
/// Soil flag: fertilized.
pub const SOIL_FERT: u8 = 4;
/// "No crop" marker in the crop plane.
pub const NO_CROP: u8 = 255;

/// One location's tile planes (structure-of-arrays).
#[derive(Clone, Debug)]
pub struct Grid {
    /// Width in tiles.
    pub w: u8,
    /// Height in tiles.
    pub h: u8,
    terrain: Vec<u8>,
    obj: Vec<u8>,
    /// Per-object payload, see [`Obj`].
    pub data: Vec<u16>,
    /// Soil flags.
    pub soil: Vec<u8>,
    /// Crop id or [`NO_CROP`].
    pub crop: Vec<u8>,
    /// Days of growth accumulated.
    pub crop_days: Vec<u8>,
}

impl Grid {
    /// A grid filled with one terrain.
    pub fn filled(w: u8, h: u8, t: Terrain) -> Grid {
        let n = w as usize * h as usize;
        Grid {
            w,
            h,
            terrain: vec![t as u8; n],
            obj: vec![0; n],
            data: vec![0; n],
            soil: vec![0; n],
            crop: vec![NO_CROP; n],
            crop_days: vec![0; n],
        }
    }
    /// Flat index.
    #[inline]
    pub fn idx(&self, x: u8, y: u8) -> usize {
        y as usize * self.w as usize + x as usize
    }
    /// In bounds?
    #[inline]
    pub fn in_bounds(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && x < self.w as i32 && y < self.h as i32
    }
    /// Terrain at tile.
    #[inline]
    pub fn terrain(&self, x: u8, y: u8) -> Terrain {
        Terrain::from_u8(self.terrain[self.idx(x, y)])
    }
    /// Set terrain.
    #[inline]
    pub fn set_terrain(&mut self, x: u8, y: u8, t: Terrain) {
        let i = self.idx(x, y);
        self.terrain[i] = t as u8;
    }
    /// Object at tile.
    #[inline]
    pub fn obj(&self, x: u8, y: u8) -> Obj {
        Obj::from_u8(self.obj[self.idx(x, y)])
    }
    /// Set object + data.
    #[inline]
    pub fn set_obj(&mut self, x: u8, y: u8, o: Obj, data: u16) {
        let i = self.idx(x, y);
        self.obj[i] = o as u8;
        self.data[i] = data;
    }
    /// Clear object.
    #[inline]
    pub fn clear_obj(&mut self, x: u8, y: u8) {
        self.set_obj(x, y, Obj::None, 0);
    }
    /// Raw terrain plane.
    pub fn terrain_plane(&self) -> &[u8] {
        &self.terrain
    }
    /// Raw object plane.
    pub fn obj_plane(&self) -> &[u8] {
        &self.obj
    }
    /// Passable for the player?
    #[inline]
    pub fn passable(&self, x: i32, y: i32) -> bool {
        if !self.in_bounds(x, y) {
            return false;
        }
        let i = self.idx(x as u8, y as u8);
        Terrain::from_u8(self.terrain[i]).walkable() && !Obj::from_u8(self.obj[i]).blocks()
    }
    /// Count objects of a kind.
    pub fn count_obj(&self, o: Obj) -> u32 {
        self.obj.iter().filter(|v| **v == o as u8).count() as u32
    }
    /// Hash all planes.
    pub fn hash_into(&self, h: &mut StableHasher) {
        h.u8(self.w).u8(self.h);
        h.u8s(&self.terrain).u8s(&self.obj).u16s(&self.data).u8s(&self.soil).u8s(&self.crop).u8s(&self.crop_days);
    }
}

/// Rectangle helper (inclusive).
#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub x0: u8,
    pub y0: u8,
    pub x1: u8,
    pub y1: u8,
}

impl Rect {
    fn from_arr(r: [u8; 4]) -> Rect {
        Rect {
            x0: r[0],
            y0: r[1],
            x1: r[2],
            y1: r[3],
        }
    }
    /// Contains?
    pub fn contains(&self, x: u8, y: u8) -> bool {
        x >= self.x0 && x <= self.x1 && y >= self.y0 && y <= self.y1
    }
    /// Iterate tiles row-major.
    pub fn tiles(&self) -> impl Iterator<Item = (u8, u8)> + '_ {
        (self.y0..=self.y1).flat_map(move |y| (self.x0..=self.x1).map(move |x| (x, y)))
    }
    /// Number of tiles.
    pub fn area(&self) -> u32 {
        (self.x1 as u32 - self.x0 as u32 + 1) * (self.y1 as u32 - self.y0 as u32 + 1)
    }
}

/// Pathfinding scratch (BFS) reusable across calls without allocation.
#[derive(Clone, Debug, Default)]
pub struct PathScratch {
    dist: Vec<u16>,
    parent: Vec<u16>,
    queue: Vec<u16>,
    stamp: Vec<u32>,
    gen: u32,
    w: u8,
    h: u8,
    src: u16,
}

/// Unreachable marker.
pub const UNREACHABLE: u16 = u16::MAX;

impl PathScratch {
    fn ensure(&mut self, g: &Grid) {
        let n = g.w as usize * g.h as usize;
        if self.dist.len() < n {
            self.dist.resize(n, UNREACHABLE);
            self.parent.resize(n, UNREACHABLE);
            self.stamp.resize(n, 0);
            self.queue.resize(n, 0);
        }
        self.w = g.w;
        self.h = g.h;
    }
    /// BFS from `(sx, sy)` over passable tiles. Blocking tiles adjacent to a reachable tile get a
    /// distance too (so objects can be targeted) but are not expanded.
    pub fn bfs(&mut self, g: &Grid, sx: u8, sy: u8) {
        self.ensure(g);
        self.gen = self.gen.wrapping_add(1);
        if self.gen == 0 {
            self.gen = 1;
            self.stamp.iter_mut().for_each(|s| *s = 0);
        }
        let w = g.w as usize;
        let s = g.idx(sx, sy);
        self.src = s as u16;
        self.stamp[s] = self.gen;
        self.dist[s] = 0;
        self.parent[s] = UNREACHABLE;
        let mut head = 0usize;
        let mut tail = 0usize;
        self.queue[tail] = s as u16;
        tail += 1;
        while head < tail {
            let cur = self.queue[head] as usize;
            head += 1;
            let d = self.dist[cur];
            let cx = (cur % w) as i32;
            let cy = (cur / w) as i32;
            for dir in [Dir::N, Dir::E, Dir::S, Dir::W] {
                let (dx, dy) = dir.delta();
                let nx = cx + dx;
                let ny = cy + dy;
                if !g.in_bounds(nx, ny) {
                    continue;
                }
                let ni = g.idx(nx as u8, ny as u8);
                if self.stamp[ni] == self.gen {
                    continue;
                }
                self.stamp[ni] = self.gen;
                self.dist[ni] = d + 1;
                self.parent[ni] = cur as u16;
                // warps and ladders are destinations only: never path *through* them
                if g.passable(nx, ny) && !matches!(g.obj(nx as u8, ny as u8), Obj::Warp | Obj::Ladder) {
                    self.queue[tail] = ni as u16;
                    tail += 1;
                }
            }
        }
    }
    /// Distance from the BFS source to a tile (UNREACHABLE if none). For a blocking tile this is
    /// the distance to it through its passable neighbour.
    #[inline]
    pub fn dist(&self, g: &Grid, x: u8, y: u8) -> u16 {
        let i = g.idx(x, y);
        if i < self.stamp.len() && self.stamp[i] == self.gen {
            self.dist[i]
        } else {
            UNREACHABLE
        }
    }
    /// First step direction from the source toward `(x, y)` (None if unreachable or at source).
    pub fn first_step(&self, g: &Grid, x: u8, y: u8) -> Option<Dir> {
        let mut i = g.idx(x, y);
        if i >= self.stamp.len() || self.stamp[i] != self.gen || self.dist[i] == 0 {
            return None;
        }
        let w = g.w as usize;
        loop {
            let p = self.parent[i] as usize;
            if p as u16 == self.src {
                let (px, py) = ((p % w) as i32, (p / w) as i32);
                let (cx, cy) = ((i % w) as i32, (i / w) as i32);
                return Some(match (cx - px, cy - py) {
                    (0, -1) => Dir::N,
                    (1, 0) => Dir::E,
                    (0, 1) => Dir::S,
                    _ => Dir::W,
                });
            }
            i = p;
        }
    }
    /// Full path (sequence of directions) to a tile.
    pub fn path(&self, g: &Grid, x: u8, y: u8, out: &mut Vec<Dir>) -> bool {
        out.clear();
        let mut i = g.idx(x, y);
        if i >= self.stamp.len() || self.stamp[i] != self.gen {
            return false;
        }
        let w = g.w as usize;
        while i as u16 != self.src {
            let p = self.parent[i] as usize;
            let (px, py) = ((p % w) as i32, (p / w) as i32);
            let (cx, cy) = ((i % w) as i32, (i / w) as i32);
            out.push(match (cx - px, cy - py) {
                (0, -1) => Dir::N,
                (1, 0) => Dir::E,
                (0, 1) => Dir::S,
                _ => Dir::W,
            });
            i = p;
        }
        out.reverse();
        true
    }
    /// Nearest tile (by BFS distance) among `cands` satisfying `pred`; returns (x, y, dist).
    pub fn nearest<F: Fn(u8, u8) -> bool>(&self, g: &Grid, cands: impl Iterator<Item = (u8, u8)>, pred: F) -> Option<(u8, u8, u16)> {
        let mut best: Option<(u8, u8, u16)> = None;
        for (x, y) in cands {
            if !pred(x, y) {
                continue;
            }
            let d = self.dist(g, x, y);
            if d == UNREACHABLE {
                continue;
            }
            if best.is_none_or(|b| d < b.2) {
                best = Some((x, y, d));
            }
        }
        best
    }
}

/// Static routing between locations via warps.
#[derive(Clone, Debug)]
pub struct Routes {
    /// `next[from][to]` = warp id to take from `from` toward `to` (255 = none / same).
    pub next: Vec<Vec<u8>>,
    /// Number of warps on the route.
    pub hops: Vec<Vec<u8>>,
}

impl Routes {
    /// Build from content warps (BFS over the location graph).
    pub fn build(c: &Content) -> Routes {
        let n = c.locations.len();
        let mut next = vec![vec![255u8; n]; n];
        let mut hops = vec![vec![255u8; n]; n];
        for from in 0..n {
            // BFS
            let mut dist = vec![255u8; n];
            let mut first = vec![255u8; n];
            let mut q = std::collections::VecDeque::new();
            dist[from] = 0;
            q.push_back(from);
            while let Some(u) = q.pop_front() {
                for w in &c.warps {
                    if w.from as usize == u {
                        let v = w.to as usize;
                        if dist[v] == 255 {
                            dist[v] = dist[u] + 1;
                            first[v] = if u == from { w.id } else { first[u] };
                            q.push_back(v);
                        }
                    }
                }
            }
            for to in 0..n {
                next[from][to] = first[to];
                hops[from][to] = dist[to];
            }
        }
        Routes { next, hops }
    }
}

/// All locations' grids plus mine-floor state.
#[derive(Clone, Debug)]
pub struct World {
    /// Grids indexed by location id (the mines grid is the current floor).
    pub grids: Vec<Grid>,
    /// Current mine floor (0 = not generated).
    pub mine_floor: u8,
}

impl World {
    /// Build the world for an episode seed (procedural layout).
    pub fn generate(c: &Content, seed: u64) -> World {
        let grids = c.locations.iter().map(|l| layout_location(c, seed, l.id)).collect();
        World { grids, mine_floor: 0 }
    }
    /// Grid of a location.
    #[inline]
    pub fn grid(&self, l: LocId) -> &Grid {
        &self.grids[l as usize]
    }
    /// Mutable grid.
    #[inline]
    pub fn grid_mut(&mut self, l: LocId) -> &mut Grid {
        &mut self.grids[l as usize]
    }
    /// Hash all grids.
    pub fn hash_into(&self, h: &mut StableHasher) {
        for g in &self.grids {
            g.hash_into(h);
        }
        h.u8(self.mine_floor);
    }
}

fn base_terrain(kind: LocKind) -> Terrain {
    match kind {
        LocKind::Farm | LocKind::Wild => Terrain::Grass,
        LocKind::Town => Terrain::Path,
        LocKind::Mines | LocKind::Greenhouse => Terrain::Floor,
    }
}

/// Lay out one (non-mine) location from the layout stream.
pub fn layout_location(c: &Content, seed: u64, loc: LocId) -> Grid {
    let l = &c.locations[loc as usize];
    let mut g = Grid::filled(l.width, l.height, base_terrain(l.kind));
    let mut rng = stream(seed, Domain::Layout, 0, loc as u32);
    // border walls
    for x in 0..l.width {
        g.set_terrain(x, 0, Terrain::Wall);
        g.set_terrain(x, l.height - 1, Terrain::Wall);
    }
    for y in 0..l.height {
        g.set_terrain(0, y, Terrain::Wall);
        g.set_terrain(l.width - 1, y, Terrain::Wall);
    }
    if l.kind == LocKind::Mines {
        return g; // floors are generated on entry
    }
    // regions
    for r in c.regions.iter().filter(|r| r.location == loc) {
        let rect = Rect::from_arr(r.rect);
        match r.kind.as_str() {
            "house_floor" => rect.tiles().for_each(|(x, y)| g.set_terrain(x, y, Terrain::Floor)),
            "field" => rect.tiles().for_each(|(x, y)| g.set_terrain(x, y, Terrain::Dirt)),
            "pads" => rect.tiles().for_each(|(x, y)| {
                g.set_terrain(x, y, Terrain::Floor);
                g.set_obj(x, y, Obj::MachinePad, 0);
            }),
            "water" => rect.tiles().for_each(|(x, y)| g.set_terrain(x, y, Terrain::Water)),
            "sand" => rect.tiles().for_each(|(x, y)| g.set_terrain(x, y, Terrain::Sand)),
            "path" => rect.tiles().for_each(|(x, y)| g.set_terrain(x, y, Terrain::Path)),
            "bridge" => rect.tiles().for_each(|(x, y)| g.set_terrain(x, y, Terrain::Bridge)),
            _ => {}
        }
    }
    // trees
    for r in c.regions_of(loc, "trees") {
        for (x, y) in Rect::from_arr(r.rect).tiles() {
            if g.terrain(x, y) == Terrain::Grass && g.obj(x, y) == Obj::None && rng.chance_ppm(c.density.tree_region_ppm) {
                g.set_obj(x, y, Obj::Tree, c.tools.tree_hits[0] as u16);
            }
        }
    }
    // stumps
    for r in c.regions_of(loc, "stumps") {
        let rect = Rect::from_arr(r.rect);
        let mut placed = 0;
        let mut tries = 0;
        while placed < c.density.stump_count && tries < 200 {
            tries += 1;
            let x = rng.range(rect.x0 as u32, rect.x1 as u32) as u8;
            let y = rng.range(rect.y0 as u32, rect.y1 as u32) as u8;
            if g.terrain(x, y) == Terrain::Grass {
                g.set_obj(x, y, Obj::Stump, c.tools.stump_hits[1] as u16);
                placed += 1;
            }
        }
    }
    // boulders (quarry)
    for r in c.regions_of(loc, "boulders") {
        let rect = Rect::from_arr(r.rect);
        let mut placed = 0;
        let mut tries = 0;
        while placed < 4 && tries < 100 {
            tries += 1;
            let x = rng.range(rect.x0 as u32, rect.x1 as u32) as u8;
            let y = rng.range(rect.y0 as u32, rect.y1 as u32) as u8;
            if g.terrain(x, y) == Terrain::Grass && g.obj(x, y) == Obj::None {
                g.set_obj(x, y, Obj::Boulder, c.tools.boulder_hits[2] as u16);
                placed += 1;
            }
        }
    }
    // field debris
    for r in c.regions_of(loc, "field") {
        for (x, y) in Rect::from_arr(r.rect).tiles() {
            if g.obj(x, y) != Obj::None {
                continue;
            }
            let roll = rng.below(1_000_000);
            let d = &c.density;
            if roll < d.field_weeds_ppm {
                g.set_obj(x, y, Obj::Weeds, 0);
            } else if roll < d.field_weeds_ppm + d.field_twig_ppm {
                g.set_obj(x, y, Obj::Twig, 0);
            } else if roll < d.field_weeds_ppm + d.field_twig_ppm + d.field_stone_ppm {
                g.set_obj(x, y, Obj::StoneDebris, 0);
            } else if roll < d.field_weeds_ppm + d.field_twig_ppm + d.field_stone_ppm + d.field_tree_ppm {
                g.set_obj(x, y, Obj::Tree, c.tools.tree_hits[0] as u16);
            }
        }
    }
    // wild fiber
    for r in c.regions_of(loc, "forage") {
        for (x, y) in Rect::from_arr(r.rect).tiles() {
            if g.obj(x, y) == Obj::None && g.terrain(x, y).walkable() && rng.chance_ppm(c.density.wild_fiber_ppm) {
                g.set_obj(x, y, Obj::Weeds, 0);
            }
        }
    }
    // POIs
    for p in c.pois.iter().filter(|p| p.location == loc) {
        let (tx, ty) = p.tile;
        let (sx, sy) = p.stand;
        let obj = match p.kind {
            PoiKind::Bed => Obj::Bed,
            PoiKind::Crate => Obj::Crate,
            PoiKind::Well => Obj::Well,
            PoiKind::MillBoard => Obj::MillBoard,
            PoiKind::Counter => Obj::Counter,
            PoiKind::Stall => Obj::Stall,
            PoiKind::FishingSpot => Obj::None,
            PoiKind::MineEntrance => Obj::MineEntrance,
            PoiKind::Elevator => Obj::Elevator,
            PoiKind::MineExit => Obj::MineExit,
        };
        let data = match p.kind {
            PoiKind::Counter => c.shops.iter().find(|s| s.poi == p.id).map(|s| s.id as u16).unwrap_or(0),
            _ => p.id as u16,
        };
        if p.kind == PoiKind::FishingSpot {
            g.set_terrain(tx, ty, Terrain::Water);
        } else {
            if !g.terrain(tx, ty).walkable() {
                g.set_terrain(tx, ty, Terrain::Floor);
            }
            g.set_obj(tx, ty, obj, data);
        }
        if !g.terrain(sx, sy).walkable() {
            g.set_terrain(sx, sy, Terrain::Path);
        }
        g.clear_obj(sx, sy);
    }
    // warps
    for w in c.warps.iter().filter(|w| w.from == loc) {
        g.set_terrain(w.tile.0, w.tile.1, Terrain::Path);
        g.set_obj(w.tile.0, w.tile.1, Obj::Warp, w.id as u16);
    }
    for w in c.warps.iter().filter(|w| w.to == loc) {
        if !g.terrain(w.to_tile.0, w.to_tile.1).walkable() {
            g.set_terrain(w.to_tile.0, w.to_tile.1, Terrain::Path);
        }
        g.clear_obj(w.to_tile.0, w.to_tile.1);
    }
    if c.world.start_location == loc {
        let (x, y) = c.world.start_tile;
        g.set_terrain(x, y, Terrain::Floor);
        g.clear_obj(x, y);
    }
    // connectivity repair: carve L-paths from a hub to every POI stand and warp tile
    let hub = c
        .warps
        .iter()
        .find(|w| w.to == loc)
        .map(|w| w.to_tile)
        .or_else(|| c.pois.iter().find(|p| p.location == loc).map(|p| p.stand))
        .unwrap_or((1, 1));
    let mut targets: Vec<(u8, u8)> = c.pois.iter().filter(|p| p.location == loc).map(|p| p.stand).collect();
    targets.extend(c.warps.iter().filter(|w| w.from == loc).map(|w| w.tile));
    targets.extend(c.warps.iter().filter(|w| w.to == loc).map(|w| w.to_tile));
    if c.world.start_location == loc {
        targets.push(c.world.start_tile);
    }
    for r in c.regions_of(loc, "field") {
        targets.push((r.rect[0], r.rect[1]));
    }
    for r in c.regions_of(loc, "pads") {
        targets.push((r.rect[0], r.rect[1].saturating_add(1)));
    }
    let mut scratch = PathScratch::default();
    for t in targets {
        scratch.bfs(&g, hub.0, hub.1);
        if scratch.dist(&g, t.0, t.1) != UNREACHABLE && g.passable(t.0 as i32, t.1 as i32) {
            continue;
        }
        carve(&mut g, hub, t);
    }
    g
}

/// Clear blocking objects (and bridge water) along an L-shaped path from `a` to `b`.
fn carve(g: &mut Grid, a: (u8, u8), b: (u8, u8)) {
    let mut x = a.0 as i32;
    let mut y = a.1 as i32;
    let (bx, by) = (b.0 as i32, b.1 as i32);
    let step = |g: &mut Grid, x: i32, y: i32| {
        if !g.in_bounds(x, y) {
            return;
        }
        let (ux, uy) = (x as u8, y as u8);
        match g.terrain(ux, uy) {
            Terrain::Water => g.set_terrain(ux, uy, Terrain::Bridge),
            Terrain::Wall | Terrain::Void => g.set_terrain(ux, uy, Terrain::Path),
            _ => {}
        }
        if g.obj(ux, uy).blocks() && g.obj(ux, uy).breakable() {
            g.clear_obj(ux, uy);
        } else if matches!(g.obj(ux, uy), Obj::Building) {
            g.clear_obj(ux, uy);
        }
    };
    while x != bx {
        x += (bx - x).signum();
        if (x, y) != (bx, by) {
            step(g, x, y);
        }
    }
    while y != by {
        y += (by - y).signum();
        if (x, y) != (bx, by) {
            step(g, x, y);
        }
    }
}

/// Generate a mine floor from the (seed, day, floor) stream.
pub fn generate_mine_floor(c: &Content, seed: u64, day: u32, floor: u8, grid: &mut Grid) {
    let l = &c.locations[c.loc_id("mines") as usize];
    *grid = Grid::filled(l.width, l.height, Terrain::Floor);
    for x in 0..l.width {
        grid.set_terrain(x, 0, Terrain::Wall);
        grid.set_terrain(x, l.height - 1, Terrain::Wall);
    }
    for y in 0..l.height {
        grid.set_terrain(0, y, Terrain::Wall);
        grid.set_terrain(l.width - 1, y, Terrain::Wall);
    }
    // POIs (exit, elevator)
    for p in c.pois.iter().filter(|p| p.location == l.id) {
        let o = match p.kind {
            PoiKind::MineExit => Obj::MineExit,
            PoiKind::Elevator => Obj::Elevator,
            _ => Obj::None,
        };
        grid.set_obj(p.tile.0, p.tile.1, o, p.id as u16);
        grid.clear_obj(p.stand.0, p.stand.1);
    }
    let mut rng = stream(seed, Domain::Mine, day, floor as u32);
    let band = c.band_for_floor(floor).expect("band");
    let weights: Vec<u32> = band.rocks.iter().map(|r| r.1).collect();
    let n = rng.range(c.mining.rocks_per_floor_min as u32, c.mining.rocks_per_floor_max as u32);
    let mut placed: Vec<(u8, u8)> = Vec::new();
    let mut tries = 0;
    while (placed.len() as u32) < n && tries < 2000 {
        tries += 1;
        let x = rng.range(1, l.width as u32 - 2) as u8;
        let y = rng.range(1, l.height as u32 - 4) as u8; // keep the bottom rows (exit/elevator) clear
        if grid.obj(x, y) != Obj::None {
            continue;
        }
        let kind: RockId = band.rocks[rng.weighted(&weights)].0;
        let hardness = c.rocks[kind as usize].hardness as u16;
        grid.set_obj(x, y, Obj::Rock, kind as u16 | (hardness << 8));
        placed.push((x, y));
    }
    // ladder rocks
    for _ in 0..c.mining.ladder_rocks {
        if placed.is_empty() {
            break;
        }
        let i = rng.below(placed.len() as u32) as usize;
        let (x, y) = placed[i];
        let idx = grid.idx(x, y);
        grid.data[idx] |= 1 << 15;
    }
    for (x, y) in &placed {
        if rng.chance_ppm(c.mining.extra_ladder_ppm) {
            let idx = grid.idx(*x, *y);
            grid.data[idx] |= 1 << 15;
        }
    }
    // repair reachability of all rocks from the exit stand
    let exit = c.pois.iter().find(|p| p.location == l.id && p.kind == PoiKind::MineExit).map(|p| p.stand).unwrap_or((12, 22));
    let mut scratch = PathScratch::default();
    scratch.bfs(grid, exit.0, exit.1);
    for (x, y) in placed {
        if scratch.dist(grid, x, y) == UNREACHABLE {
            carve(grid, exit, (x, y));
            scratch.bfs(grid, exit.0, exit.1);
        }
    }
}

/// Daily surface rock respawn in the quarry (rocks region), keyed by (seed, day).
pub fn respawn_surface_rocks(c: &Content, seed: u64, day: u32, grid: &mut Grid, loc: LocId) {
    let mut rng = stream(seed, Domain::Surface, day, loc as u32);
    let weights: Vec<u32> = c.surface.quarry_rock_kinds.iter().map(|r| r.1).collect();
    for r in c.regions_of(loc, "rocks") {
        let rect = Rect::from_arr(r.rect);
        let existing = rect.tiles().filter(|(x, y)| grid.obj(*x, *y) == Obj::Rock).count() as u32;
        let mut need = (c.surface.quarry_rocks as u32).saturating_sub(existing);
        let mut tries = 0;
        while need > 0 && tries < 400 {
            tries += 1;
            let x = rng.range(rect.x0 as u32, rect.x1 as u32) as u8;
            let y = rng.range(rect.y0 as u32, rect.y1 as u32) as u8;
            if grid.obj(x, y) == Obj::None && grid.terrain(x, y) == Terrain::Grass {
                let kind = c.surface.quarry_rock_kinds[rng.weighted(&weights)].0;
                let hardness = c.rocks[kind as usize].hardness as u16;
                grid.set_obj(x, y, Obj::Rock, kind as u16 | (hardness << 8));
                need -= 1;
            }
        }
    }
}

/// Daily forage spawn: top up a location's forage to its cap from the seasonal table.
/// Returns the number of forage objects now present.
pub fn spawn_forage(c: &Content, seed: u64, day: u32, season: Season, grid: &mut Grid, fl: &content::ForageLocation) -> u32 {
    let mut rng = stream(seed, Domain::Forage, day, fl.location as u32);
    let table = &fl.tables[season as usize];
    if table.is_empty() {
        return grid.count_obj(Obj::Forage);
    }
    let weights: Vec<u32> = table.iter().map(|t| t.1).collect();
    let rects: Vec<Rect> = c.regions_of(fl.location, "forage").map(|r| Rect::from_arr(r.rect)).collect();
    if rects.is_empty() {
        return 0;
    }
    let mut count = grid.count_obj(Obj::Forage);
    // only spawn on tiles reachable from the location's entrance
    let hub = c.warps.iter().find(|w| w.to == fl.location).map(|w| w.to_tile).unwrap_or(c.world.start_tile);
    let mut scratch = PathScratch::default();
    scratch.bfs(grid, hub.0, hub.1);
    let mut tries = 0;
    while count < fl.daily_cap as u32 && tries < 500 {
        tries += 1;
        let r = &rects[rng.below(rects.len() as u32) as usize];
        let x = rng.range(r.x0 as u32, r.x1 as u32) as u8;
        let y = rng.range(r.y0 as u32, r.y1 as u32) as u8;
        if grid.obj(x, y) == Obj::None && grid.terrain(x, y).walkable() && scratch.dist(grid, x, y) != UNREACHABLE {
            let item: ItemId = table[rng.weighted(&weights)].0;
            grid.set_obj(x, y, Obj::Forage, item);
            count += 1;
        }
    }
    count
}

/// Remixed restoration: resample slot items per request from its pool (train/test partition).
pub fn remix_slots(c: &Content, seed: u64, test_partition: bool) -> Vec<ItemId> {
    let mut rng = stream(seed, Domain::Remix, 0, test_partition as u32);
    let mut out = Vec::with_capacity(c.total_slots as usize);
    for r in &c.requests {
        let pool = if test_partition { &r.pool_test } else { &r.pool_train };
        if r.slots.is_empty() {
            continue;
        }
        let mut p: Vec<ItemId> = pool.clone();
        rng.shuffle(&mut p);
        for (i, _) in r.slots.iter().enumerate() {
            out.push(if i < p.len() { p[i] } else { r.slots[i].item });
        }
    }
    out
}

/// Pick a deterministic hash-friendly wing index helper (used by rewards).
pub fn wing_index(w: WingId) -> usize {
    w as usize
}

/// Convenience: a fresh RNG for dynamics from the episode seed.
pub fn dynamics_rng(seed: u64) -> Pcg32 {
    stream(seed, Domain::Dynamics, 0, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content::content;

    #[test]
    fn layout_is_deterministic_and_connected() {
        let c = content();
        let a = World::generate(c, 1);
        let b = World::generate(c, 1);
        let mut ha = StableHasher::new();
        let mut hb = StableHasher::new();
        a.hash_into(&mut ha);
        b.hash_into(&mut hb);
        assert_eq!(ha.finish(), hb.finish());
        let d = World::generate(c, 2);
        let mut hd = StableHasher::new();
        d.hash_into(&mut hd);
        assert_ne!(ha.finish(), hd.finish());
        // every POI stand and warp tile reachable from the location's entry
        let mut s = PathScratch::default();
        for l in &c.locations {
            if l.kind == LocKind::Mines {
                continue;
            }
            let g = a.grid(l.id);
            let hub = c.warps.iter().find(|w| w.to == l.id).map(|w| w.to_tile).unwrap_or(c.world.start_tile);
            s.bfs(g, hub.0, hub.1);
            for p in c.pois.iter().filter(|p| p.location == l.id) {
                assert_ne!(s.dist(g, p.stand.0, p.stand.1), UNREACHABLE, "poi {} unreachable", p.key);
            }
            for w in c.warps.iter().filter(|w| w.from == l.id) {
                assert_ne!(s.dist(g, w.tile.0, w.tile.1), UNREACHABLE, "warp {} unreachable", w.id);
            }
        }
    }

    #[test]
    fn routes_connect_all_outdoor_locations() {
        let c = content();
        let r = Routes::build(c);
        let home = c.loc_id("homestead");
        for l in &c.locations {
            if l.kind == LocKind::Mines {
                continue;
            }
            assert_ne!(r.hops[home as usize][l.id as usize], 255, "no route to {}", l.key);
        }
    }

    #[test]
    fn mine_floor_generates_with_ladder() {
        let c = content();
        let l = &c.locations[c.loc_id("mines") as usize];
        let mut g = Grid::filled(l.width, l.height, Terrain::Floor);
        generate_mine_floor(c, 5, 3, 7, &mut g);
        let ladders = g.data.iter().zip(g.obj_plane()).filter(|(d, o)| **o == Obj::Rock as u8 && (**d & (1 << 15)) != 0).count();
        assert!(ladders >= 1);
        assert!(g.count_obj(Obj::Rock) >= c.mining.rocks_per_floor_min as u32);
    }
}
