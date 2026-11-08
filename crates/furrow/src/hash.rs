//! Stable hashing (DESIGN.md §2.5): explicit little-endian serialization through FNV-1a 64.
//! Never `#[derive(Hash)]`, never `usize`, never `std::collections::hash_map::DefaultHasher`.

/// FNV-1a 64-bit streaming hasher.
#[derive(Clone, Debug)]
pub struct StableHasher(u64);

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

impl Default for StableHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl StableHasher {
    /// A fresh hasher at the FNV offset basis.
    pub const fn new() -> Self {
        StableHasher(FNV_OFFSET)
    }
    /// Feed raw bytes.
    #[inline]
    pub fn bytes(&mut self, b: &[u8]) -> &mut Self {
        let mut h = self.0;
        for &x in b {
            h ^= x as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
        self.0 = h;
        self
    }
    /// Feed a `u8`.
    #[inline]
    pub fn u8(&mut self, v: u8) -> &mut Self {
        self.bytes(&[v])
    }
    /// Feed a `bool` as one byte.
    #[inline]
    pub fn bool(&mut self, v: bool) -> &mut Self {
        self.u8(v as u8)
    }
    /// Feed a `u16` little-endian.
    #[inline]
    pub fn u16(&mut self, v: u16) -> &mut Self {
        self.bytes(&v.to_le_bytes())
    }
    /// Feed a `u32` little-endian.
    #[inline]
    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.bytes(&v.to_le_bytes())
    }
    /// Feed a `u64` little-endian.
    #[inline]
    pub fn u64(&mut self, v: u64) -> &mut Self {
        self.bytes(&v.to_le_bytes())
    }
    /// Feed an `i32` little-endian.
    #[inline]
    pub fn i32(&mut self, v: i32) -> &mut Self {
        self.bytes(&v.to_le_bytes())
    }
    /// Feed a slice of `u8` with a `u32` length prefix.
    pub fn u8s(&mut self, v: &[u8]) -> &mut Self {
        self.u32(v.len() as u32);
        self.bytes(v)
    }
    /// Feed a slice of `u16` with a `u32` length prefix.
    pub fn u16s(&mut self, v: &[u16]) -> &mut Self {
        self.u32(v.len() as u32);
        for x in v {
            self.u16(*x);
        }
        self
    }
    /// Feed a slice of `u32` with a `u32` length prefix.
    pub fn u32s(&mut self, v: &[u32]) -> &mut Self {
        self.u32(v.len() as u32);
        for x in v {
            self.u32(*x);
        }
        self
    }
    /// Finish.
    #[inline]
    pub fn finish(&self) -> u64 {
        self.0
    }
}

/// FNV-1a 64 of a byte string (used for `CONTENT_HASH`).
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = StableHasher::new();
    h.bytes(bytes);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fnv_vectors() {
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a64(b"foobar"), 0x85944171f73967e8);
    }
}
