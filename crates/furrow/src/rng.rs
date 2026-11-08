//! Deterministic randomness (DESIGN.md §2.5).
//!
//! No external RNG crate: the algorithms below are frozen so recorded trajectories replay
//! bit-identically across versions and platforms.
//!
//! * [`splitmix64`] — seed mixing (Steele, Lea & Flood 2014).
//! * [`Pcg32`] — PCG-XSH-RR 64/32 (O'Neill 2014), the standard `pcg32` variant.
//! * [`Domain`] + [`stream`] — *exogenous* randomness (layout, weather, forage, merchant, mine
//!   floors) is drawn from a fresh generator keyed by `(episode_seed, domain, day, extra)`, so it
//!   never depends on how many draws the policy's actions consumed.

/// SplitMix64 output function; also used as a mixing hash for stream keys.
#[inline]
pub const fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Purposes of exogenous random streams. The discriminant is part of the stream key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Domain {
    /// Procedural map layout (debris, trees, rocks) — keyed by location.
    Layout = 1,
    /// Daily weather roll.
    Weather = 2,
    /// Daily forage spawns — keyed by location.
    Forage = 3,
    /// Wandering merchant stock.
    Merchant = 4,
    /// Mine floor generation — keyed by floor.
    Mine = 5,
    /// Surface rock respawns.
    Surface = 6,
    /// Remixed restoration sampling.
    Remix = 7,
    /// Expert policy noise / style draws.
    Expert = 8,
    /// Per-env dynamics stream seed derivation.
    Dynamics = 9,
}

/// A fresh PCG32 keyed by `(episode_seed, domain, day, extra)`.
#[inline]
pub fn stream(episode_seed: u64, domain: Domain, day: u32, extra: u32) -> Pcg32 {
    let k = splitmix64(episode_seed ^ 0xA076_1D64_78BD_642F)
        ^ splitmix64(((domain as u64) << 48) | ((day as u64) << 16) | extra as u64);
    Pcg32::new(splitmix64(k), splitmix64(k ^ 0xDEAD_BEEF_CAFE_F00D))
}

/// PCG-XSH-RR 64/32 generator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

const PCG_MULT: u64 = 6_364_136_223_846_793_005;

impl Pcg32 {
    /// Seed with an initial state and a stream selector (as `pcg32_srandom_r`).
    pub fn new(init_state: u64, init_seq: u64) -> Self {
        let mut g = Pcg32 {
            state: 0,
            inc: (init_seq << 1) | 1,
        };
        g.next_u32();
        g.state = g.state.wrapping_add(init_state);
        g.next_u32();
        g
    }

    /// Seed from a single `u64` via SplitMix64 (state and stream derived).
    pub fn from_seed(seed: u64) -> Self {
        let a = splitmix64(seed);
        let b = splitmix64(a);
        Pcg32::new(a, b)
    }

    /// Next 32 random bits.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Next 64 random bits (two 32-bit outputs, high word first).
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let hi = self.next_u32() as u64;
        let lo = self.next_u32() as u64;
        (hi << 32) | lo
    }

    /// Uniform integer in `0..n` (unbiased, rejection sampling as in `pcg32_boundedrand_r`).
    /// `n == 0` returns 0.
    #[inline]
    pub fn below(&mut self, n: u32) -> u32 {
        if n <= 1 {
            return 0;
        }
        let threshold = n.wrapping_neg() % n;
        loop {
            let r = self.next_u32();
            if r >= threshold {
                return r % n;
            }
        }
    }

    /// Uniform integer in `lo..=hi` (`hi < lo` returns `lo`).
    #[inline]
    pub fn range(&mut self, lo: u32, hi: u32) -> u32 {
        if hi <= lo {
            return lo;
        }
        lo + self.below(hi - lo + 1)
    }

    /// Bernoulli trial with probability `ppm / 1_000_000`.
    #[inline]
    pub fn chance_ppm(&mut self, ppm: u32) -> bool {
        if ppm >= 1_000_000 {
            return true;
        }
        self.below(1_000_000) < ppm
    }

    /// Weighted choice: returns the index of the chosen weight (weights sum must be > 0, else 0).
    pub fn weighted(&mut self, weights: &[u32]) -> usize {
        let total: u32 = weights.iter().sum();
        if total == 0 {
            return 0;
        }
        let mut r = self.below(total);
        for (i, w) in weights.iter().enumerate() {
            if r < *w {
                return i;
            }
            r -= *w;
        }
        weights.len() - 1
    }

    /// In-place Fisher–Yates shuffle.
    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        let n = xs.len();
        for i in (1..n).rev() {
            let j = self.below(i as u32 + 1) as usize;
            xs.swap(i, j);
        }
    }

    /// Raw state, for hashing.
    pub fn state_words(&self) -> (u64, u64) {
        (self.state, self.inc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcg32_reference_vector() {
        // Reference: pcg32_srandom_r(&rng, 42u, 54u) then pcg32_random_r — first outputs from the
        // PCG minimal C library's `pcg32-demo` ("Round 1").
        let mut g = Pcg32::new(42, 54);
        let expected: [u32; 6] = [
            0xa15c_02b7, 0x7b47_f409, 0xba1d_3330, 0x83d2_f293, 0xbfa4_784b, 0xcbed_606e,
        ];
        for e in expected {
            assert_eq!(g.next_u32(), e);
        }
    }

    #[test]
    fn splitmix64_reference_vector() {
        // Reference: first output of SplitMix64 with seed 0 is 0xE220A8397B1DCDAF.
        assert_eq!(splitmix64(0), 0xE220_A839_7B1D_CDAF);
    }

    #[test]
    fn below_is_in_range_and_deterministic() {
        let mut a = Pcg32::from_seed(7);
        let mut b = Pcg32::from_seed(7);
        for _ in 0..10_000 {
            let x = a.below(13);
            assert!(x < 13);
            assert_eq!(x, b.below(13));
        }
    }

    #[test]
    fn streams_are_independent_of_draw_count() {
        let mut s1 = stream(99, Domain::Weather, 5, 0);
        let mut s2 = stream(99, Domain::Weather, 5, 0);
        let _ = Pcg32::from_seed(99).next_u32(); // unrelated draws
        assert_eq!(s1.next_u64(), s2.next_u64());
        let mut s3 = stream(99, Domain::Weather, 6, 0);
        assert_ne!(s1.next_u64(), s3.next_u64());
    }
}
