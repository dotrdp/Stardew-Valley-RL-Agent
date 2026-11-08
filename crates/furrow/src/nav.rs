//! Navigation: travel-time estimates across locations (for ETA features and masks) and the
//! shared walking helper used by macro controllers.

use crate::content::{content, Feature, LocId, LocKind, PoiId};
use crate::state::GameState;
use crate::world::{Grid, PathScratch, UNREACHABLE};

/// Per-location distance planes from each arrival tile (warp destinations), built at reset.
#[derive(Clone, Debug, Default)]
pub struct NavTables {
    /// `planes[loc]` = list of `(arrival_tile, dist plane)`.
    planes: Vec<Vec<ArrivalPlane>>,
}

/// Distances from one arrival tile to every tile of a location.
type ArrivalPlane = ((u8, u8), Vec<u16>);

impl NavTables {
    /// Build from the current world grids.
    pub fn build(st: &GameState) -> NavTables {
        let c = content();
        let mut scratch = PathScratch::default();
        let mut planes = Vec::with_capacity(c.locations.len());
        for l in &c.locations {
            let g = st.world.grid(l.id);
            let mut arrivals: Vec<(u8, u8)> = c
                .warps
                .iter()
                .filter(|w| w.to == l.id)
                .map(|w| w.to_tile)
                .collect();
            if l.kind == LocKind::Mines {
                arrivals.clear();
            }
            if c.world.start_location == l.id {
                arrivals.push(c.world.start_tile);
            }
            if c.locations[l.id as usize].kind == LocKind::Wild && l.key == "quarry" {
                // mine exit arrival
                let p = &c.pois[c.poi_id("mine_entrance") as usize];
                arrivals.push(p.stand);
            }
            let mut v = Vec::new();
            for a in arrivals {
                scratch.bfs(g, a.0, a.1);
                let mut plane = vec![UNREACHABLE; g.w as usize * g.h as usize];
                for y in 0..g.h {
                    for x in 0..g.w {
                        plane[g.idx(x, y)] = scratch.dist(g, x, y);
                    }
                }
                v.push((a, plane));
            }
            planes.push(v);
        }
        NavTables { planes }
    }

    /// Distance within `loc` from an arrival tile to a target tile (UNREACHABLE if unknown).
    fn dist_from_arrival(&self, g: &Grid, loc: LocId, arrival: (u8, u8), target: (u8, u8)) -> u16 {
        for (a, plane) in &self.planes[loc as usize] {
            if *a == arrival {
                return plane[g.idx(target.0, target.1)];
            }
        }
        UNREACHABLE
    }
}

/// Hub (default arrival tile) of a location: the first warp arriving there, else the start tile.
pub fn hub(loc: LocId) -> (u8, u8) {
    let c = content();
    c.warps
        .iter()
        .find(|w| w.to == loc)
        .map(|w| w.to_tile)
        .unwrap_or(c.world.start_tile)
}

impl GameState {
    /// Estimated travel minutes from the player to `(loc, tile)`; `u32::MAX` if unreachable.
    /// Uses a live BFS in the current location and the static tables elsewhere.
    pub fn eta_to(&mut self, nav: &NavTables, loc: LocId, target: (u8, u8)) -> u32 {
        let c = self.c();
        let cur = self.player.loc;
        let (px, py) = (self.player.x, self.player.y);
        if cur == loc {
            let g = self.world.grid(cur);
            self.scratch.bfs(g, px, py);
            let d = self.scratch.dist(g, target.0, target.1);
            return if d == UNREACHABLE { u32::MAX } else { d as u32 };
        }
        if loc == c.loc_id("mines") || cur == c.loc_id("mines") {
            return u32::MAX;
        }
        // walk route
        let mut total: u32 = 0;
        let mut at = cur;
        let mut pos = (px, py);
        let mut first = true;
        let mut hops = 0;
        while at != loc && hops < 8 {
            hops += 1;
            let wid = self.routes.next[at as usize][loc as usize];
            if wid == 255 {
                return u32::MAX;
            }
            let w = &c.warps[wid as usize];
            let g = self.world.grid(at);
            let d = if first {
                self.scratch.bfs(g, pos.0, pos.1);
                self.scratch.dist(g, w.tile.0, w.tile.1)
            } else {
                nav.dist_from_arrival(g, at, pos, w.tile)
            };
            if d == UNREACHABLE {
                return u32::MAX;
            }
            total += d as u32;
            at = w.to;
            pos = w.to_tile;
            first = false;
        }
        let g = self.world.grid(loc);
        let d = nav.dist_from_arrival(g, loc, pos, target);
        if d == UNREACHABLE {
            return u32::MAX;
        }
        let walk = total + d as u32;
        // minecart shortcut between outdoor locations
        if self.has_feature(Feature::Minecart) && outdoor(cur) && outdoor(loc) {
            let h = hub(loc);
            let d2 = nav.dist_from_arrival(g, loc, h, target);
            if d2 != UNREACHABLE {
                return walk.min(c.world.minecart_minutes as u32 + d2 as u32);
            }
        }
        walk
    }

    /// ETA to every POI, computed at most once per state (cached by step/position/time).
    pub fn eta_cached(&mut self, nav: &NavTables) -> &[u32] {
        let key = (
            self.steps,
            self.player.loc,
            self.player.x,
            self.player.y,
            self.abs_minute,
        );
        if self.eta_key != key || self.eta_cache.len() != self.c().pois.len() {
            let mut v = std::mem::take(&mut self.eta_cache);
            v.resize(self.c().pois.len(), u32::MAX);
            self.eta_all(nav, &mut v);
            self.eta_cache = v;
            self.eta_key = key;
        }
        &self.eta_cache
    }

    /// ETA to a POI's stand tile.
    pub fn eta_poi(&mut self, nav: &NavTables, poi: PoiId) -> u32 {
        if (poi as usize) < self.c().pois.len() {
            return self.eta_cached(nav)[poi as usize];
        }
        let p = &self.c().pois[poi as usize];
        self.eta_to(nav, p.location, p.stand)
    }
}

impl GameState {
    /// ETAs (minutes) from the player to every POI stand, with a single BFS in the current
    /// location. `u32::MAX` marks unreachable POIs (e.g. inside the mines).
    pub fn eta_all(&mut self, nav: &NavTables, out: &mut [u32]) {
        let c = self.c();
        let cur = self.player.loc;
        let mines = c.loc_id("mines");
        {
            let g = self.world.grid(cur);
            self.scratch.bfs(g, self.player.x, self.player.y);
        }
        for (pi, p) in c.pois.iter().enumerate() {
            if pi >= out.len() {
                break;
            }
            if p.location == cur {
                let g = self.world.grid(cur);
                let d = self.scratch.dist(g, p.stand.0, p.stand.1);
                out[pi] = if d == UNREACHABLE { u32::MAX } else { d as u32 };
                continue;
            }
            if p.location == mines || cur == mines {
                out[pi] = u32::MAX;
                continue;
            }
            let mut total: u32 = 0;
            let mut at = cur;
            let mut pos = (self.player.x, self.player.y);
            let mut first = true;
            let mut ok = true;
            let mut hops = 0;
            while at != p.location && hops < 8 {
                hops += 1;
                let wid = self.routes.next[at as usize][p.location as usize];
                if wid == 255 {
                    ok = false;
                    break;
                }
                let w = &c.warps[wid as usize];
                let g = self.world.grid(at);
                let d = if first {
                    self.scratch.dist(g, w.tile.0, w.tile.1)
                } else {
                    nav.dist_from_arrival(g, at, pos, w.tile)
                };
                if d == UNREACHABLE {
                    ok = false;
                    break;
                }
                total += d as u32;
                at = w.to;
                pos = w.to_tile;
                first = false;
            }
            if !ok {
                out[pi] = u32::MAX;
                continue;
            }
            let g = self.world.grid(p.location);
            let d = nav.dist_from_arrival(g, p.location, pos, p.stand);
            if d == UNREACHABLE {
                out[pi] = u32::MAX;
                continue;
            }
            let mut walk = total + d as u32;
            if self.has_feature(Feature::Minecart) && outdoor(cur) && outdoor(p.location) {
                let h = hub(p.location);
                let d2 = nav.dist_from_arrival(g, p.location, h, p.stand);
                if d2 != UNREACHABLE {
                    walk = walk.min(c.world.minecart_minutes as u32 + d2 as u32);
                }
            }
            out[pi] = walk;
        }
    }
}

/// Outdoor location (minecart-reachable)?
pub fn outdoor(loc: LocId) -> bool {
    let c = content();
    matches!(
        c.locations[loc as usize].kind,
        LocKind::Farm | LocKind::Town | LocKind::Wild
    )
}
