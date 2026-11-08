//! `furrow-traj/2` trajectory datasets (DESIGN.md §5): memory-mappable `.npy` shard directories.
//!
//! Layout of a dataset directory:
//! ```text
//! meta.json                  dataset metadata (env, sim/content versions, layout, action names, ...)
//! episodes.jsonl             one line per episode (seed, shard, start, length, days, success, ...)
//! final_obs/global.npy       f32[E, G] terminal observation per episode (aligned with episodes.jsonl)
//! shards/<writer>-<seq>/     one raw .npy per column, little-endian, C-order, uncompressed,
//!                            header padded so data starts 64-byte aligned
//! ```
//! Shards are written to `shards/.tmp-<writer>-<seq>/` and renamed atomically; only then are
//! their episode lines appended, so a crash leaves at most an ignored temp directory.
//! Masks are bit-packed little-endian (`np.unpackbits(m, axis=-1, bitorder="little", count=A)`).

#![forbid(unsafe_code)]

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use half::f16;

/// Format identifier.
pub const FORMAT: &str = "furrow-traj/2";

/// numpy dtypes used by the format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F16,
    F32,
    U8,
    U16,
    U32,
    I32,
    I64,
    U64,
}

impl DType {
    /// numpy `descr` string.
    pub fn descr(self) -> &'static str {
        match self {
            DType::F16 => "<f2",
            DType::F32 => "<f4",
            DType::U8 => "|u1",
            DType::U16 => "<u2",
            DType::U32 => "<u4",
            DType::I32 => "<i4",
            DType::I64 => "<i8",
            DType::U64 => "<u8",
        }
    }
    /// Bytes per element.
    pub fn size(self) -> usize {
        match self {
            DType::U8 => 1,
            DType::F16 | DType::U16 => 2,
            DType::F32 | DType::U32 | DType::I32 => 4,
            DType::I64 | DType::U64 => 8,
        }
    }
}

/// An npy v1.0 header for `shape`, padded so that the data starts at a multiple of 64 bytes.
pub fn npy_header(dtype: DType, shape: &[usize]) -> Vec<u8> {
    let shape_s = match shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", shape[0]),
        _ => format!("({})", shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let dict = format!("{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}", dtype.descr(), shape_s);
    let pre = 10; // magic(6) + version(2) + header_len(2)
    let mut total = pre + dict.len() + 1;
    total = total.div_ceil(64) * 64;
    let hlen = total - pre;
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(hlen as u16).to_le_bytes());
    out.extend_from_slice(dict.as_bytes());
    while out.len() < total - 1 {
        out.push(b' ');
    }
    out.push(b'\n');
    out
}

/// Write a whole `.npy` file.
pub fn write_npy(path: &Path, dtype: DType, shape: &[usize], data: &[u8]) -> io::Result<()> {
    let expect: usize = shape.iter().product::<usize>() * dtype.size();
    if expect != data.len() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("npy {}: {} bytes for shape {:?} ({})", path.display(), data.len(), shape, expect)));
    }
    let mut f = BufWriter::new(File::create(path)?);
    f.write_all(&npy_header(dtype, shape))?;
    f.write_all(data)?;
    f.flush()?;
    f.into_inner().map_err(|e| e.into_error())?.sync_all()
}

/// Read an `.npy` file's header: `(descr, shape, data_offset)`.
pub fn read_npy_header(bytes: &[u8]) -> io::Result<(String, Vec<usize>, usize)> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "not an npy file"));
    }
    let (hlen, pre) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12),
        v => return Err(io::Error::new(io::ErrorKind::InvalidData, format!("npy version {v}"))),
    };
    let header = std::str::from_utf8(&bytes[pre..pre + hlen]).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let descr = header.split("'descr':").nth(1).and_then(|s| s.split('\'').nth(1)).unwrap_or("").to_string();
    let shape_s = header.split("'shape':").nth(1).and_then(|s| s.split('(').nth(1)).and_then(|s| s.split(')').next()).unwrap_or("");
    let shape = shape_s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
    Ok((descr, shape, pre + hlen))
}

/// Pack a bool mask little-endian into bytes (`ceil(len/8)` bytes).
pub fn pack_mask_le(mask: &[bool], out: &mut Vec<u8>) {
    for chunk in mask.chunks(8) {
        let mut b = 0u8;
        for (k, m) in chunk.iter().enumerate() {
            if *m {
                b |= 1 << k;
            }
        }
        out.push(b);
    }
}

/// Unpack a little-endian packed mask.
pub fn unpack_mask_le(bytes: &[u8], len: usize) -> Vec<bool> {
    (0..len).map(|i| bytes[i / 8] & (1 << (i % 8)) != 0).collect()
}

/// One episode's per-step columns.
#[derive(Clone, Debug, Default)]
pub struct EpisodeBuf {
    /// Observation dim G.
    pub g: usize,
    /// Number of actions A.
    pub a: usize,
    pub obs: Vec<u16>, // f16 bits, T·G
    pub action: Vec<i32>,
    pub expert_action: Vec<i32>,
    pub actor: Vec<u8>,
    pub mask: Vec<u8>, // T·ceil(A/8)
    pub reward: Vec<f32>,
    pub terminated: Vec<u8>,
    pub truncated: Vec<u8>,
    pub day_ends: Vec<u8>,
    pub ticks: Vec<u32>,
    pub outcome: Vec<u8>,
    pub day: Vec<u16>,
    pub minute: Vec<u16>,
    pub progress: Vec<u16>, // T·3
    pub intent: Vec<u8>,
    /// Terminal observation (f32, G).
    pub final_obs: Vec<f32>,
    /// Episode record fields.
    pub seed: u64,
    pub style: u32,
    pub noise_ppm: u32,
    pub ret: f32,
    pub days: u16,
    pub success: bool,
    pub step_cap: bool,
    pub final_state_hash: u64,
}

impl EpisodeBuf {
    /// New empty episode buffer.
    pub fn new(g: usize, a: usize) -> Self {
        EpisodeBuf { g, a, ..Default::default() }
    }
    /// Number of steps.
    pub fn len(&self) -> usize {
        self.action.len()
    }
    /// Empty?
    pub fn is_empty(&self) -> bool {
        self.action.is_empty()
    }
    /// Append one step.
    #[allow(clippy::too_many_arguments)]
    pub fn push(
        &mut self,
        obs: &[f32],
        mask: &[bool],
        action: i32,
        expert_action: i32,
        actor: u8,
        reward: f32,
        terminated: bool,
        truncated: bool,
        day_ends: u8,
        ticks: u32,
        outcome: u8,
        day: u16,
        minute: u16,
        progress: [u16; 3],
        intent: u8,
    ) {
        self.obs.extend(obs.iter().map(|x| f16::from_f32(*x).to_bits()));
        pack_mask_le(mask, &mut self.mask);
        self.action.push(action);
        self.expert_action.push(expert_action);
        self.actor.push(actor);
        self.reward.push(reward);
        self.terminated.push(terminated as u8);
        self.truncated.push(truncated as u8);
        self.day_ends.push(day_ends);
        self.ticks.push(ticks);
        self.outcome.push(outcome);
        self.day.push(day);
        self.minute.push(minute);
        self.progress.extend_from_slice(&progress);
        self.intent.push(intent);
    }
    /// Approximate bytes per step.
    pub fn bytes_per_step(&self) -> usize {
        self.g * 2 + self.a.div_ceil(8) + 40
    }
}

/// Explicit little-endian serialization of primitive columns.
trait LeBytes: Copy {
    fn put(self, out: &mut Vec<u8>);
}
macro_rules! le {
    ($($t:ty),*) => {$(impl LeBytes for $t { #[inline] fn put(self, out: &mut Vec<u8>) { out.extend_from_slice(&self.to_le_bytes()); } })*};
}
le!(u16, u32, i32, f32);

fn as_bytes<T: LeBytes>(v: &[T]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * std::mem::size_of::<T>());
    for x in v {
        x.put(&mut out);
    }
    out
}

/// Streaming dataset writer.
pub struct DatasetWriter {
    dir: PathBuf,
    writer: String,
    seq: u32,
    shard_cap: usize,
    pending: Vec<EpisodeBuf>,
    pending_bytes: usize,
    next_episode: u64,
    episodes_file: BufWriter<File>,
    final_obs: Vec<f32>,
    g: usize,
    a: usize,
    total_steps: u64,
}

impl DatasetWriter {
    /// Create a dataset (directory must be empty or absent). `meta` is written as `meta.json`.
    pub fn create(dir: &Path, meta: &serde_json::Value, g: usize, a: usize) -> io::Result<Self> {
        fs::create_dir_all(dir.join("shards"))?;
        fs::create_dir_all(dir.join("final_obs"))?;
        if dir.join("episodes.jsonl").exists() {
            return Err(io::Error::new(io::ErrorKind::AlreadyExists, format!("{} already holds a dataset", dir.display())));
        }
        let mut meta = meta.clone();
        if let Some(m) = meta.as_object_mut() {
            m.insert("format".into(), FORMAT.into());
            m.insert("obs_global_dim".into(), g.into());
            m.insert("num_actions".into(), a.into());
            m.insert("obs_global_dtype".into(), "f16".into());
            m.insert("mask_packing".into(), "little".into());
        }
        fs::write(dir.join("meta.json"), serde_json::to_vec_pretty(&meta)?)?;
        let episodes_file = BufWriter::new(OpenOptions::new().create(true).append(true).open(dir.join("episodes.jsonl"))?);
        Ok(DatasetWriter {
            dir: dir.to_path_buf(),
            writer: "w0".into(),
            seq: 0,
            shard_cap: 512 << 20,
            pending: Vec::new(),
            pending_bytes: 0,
            next_episode: 0,
            episodes_file,
            final_obs: Vec::new(),
            g,
            a,
            total_steps: 0,
        })
    }
    /// Set the shard size cap in bytes.
    pub fn with_shard_cap(mut self, bytes: usize) -> Self {
        self.shard_cap = bytes.max(1);
        self
    }
    /// Add a finished episode (shards are cut only at episode boundaries).
    pub fn push(&mut self, ep: EpisodeBuf) -> io::Result<()> {
        assert_eq!(ep.g, self.g);
        self.pending_bytes += ep.len() * ep.bytes_per_step();
        self.pending.push(ep);
        if self.pending_bytes >= self.shard_cap {
            self.flush_shard()?;
        }
        Ok(())
    }
    fn flush_shard(&mut self) -> io::Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let name = format!("{}-{:05}", self.writer, self.seq);
        self.seq += 1;
        let tmp = self.dir.join("shards").join(format!(".tmp-{name}"));
        let fin = self.dir.join("shards").join(&name);
        fs::create_dir_all(&tmp)?;
        let t: usize = self.pending.iter().map(|e| e.len()).sum();
        let g = self.g;
        let mb = self.a.div_ceil(8);
        macro_rules! col {
            ($field:ident, $ty:ty) => {{
                let mut v: Vec<$ty> = Vec::with_capacity(t);
                for e in &self.pending {
                    v.extend_from_slice(&e.$field);
                }
                v
            }};
        }
        let mut episode_col: Vec<u32> = Vec::with_capacity(t);
        let mut lines = Vec::new();
        let mut start = 0usize;
        for e in &self.pending {
            let id = self.next_episode;
            self.next_episode += 1;
            episode_col.extend(std::iter::repeat_n(id as u32, e.len()));
            lines.push(serde_json::json!({
                "episode": id, "seed": e.seed, "style": e.style, "noise_ppm": e.noise_ppm,
                "shard": name, "start": start, "length": e.len(), "return": e.ret, "days": e.days,
                "success": e.success, "step_cap": e.step_cap, "final_state_hash": format!("{:016x}", e.final_state_hash),
            }));
            start += e.len();
            self.final_obs.extend_from_slice(&e.final_obs);
        }
        write_npy(&tmp.join("obs_global.npy"), DType::F16, &[t, g], &as_bytes(&col!(obs, u16)))?;
        write_npy(&tmp.join("action.npy"), DType::I32, &[t], &as_bytes(&col!(action, i32)))?;
        write_npy(&tmp.join("expert_action.npy"), DType::I32, &[t], &as_bytes(&col!(expert_action, i32)))?;
        write_npy(&tmp.join("actor.npy"), DType::U8, &[t], &col!(actor, u8))?;
        write_npy(&tmp.join("action_mask.npy"), DType::U8, &[t, mb], &col!(mask, u8))?;
        write_npy(&tmp.join("reward.npy"), DType::F32, &[t], &as_bytes(&col!(reward, f32)))?;
        write_npy(&tmp.join("terminated.npy"), DType::U8, &[t], &col!(terminated, u8))?;
        write_npy(&tmp.join("truncated.npy"), DType::U8, &[t], &col!(truncated, u8))?;
        write_npy(&tmp.join("day_ends.npy"), DType::U8, &[t], &col!(day_ends, u8))?;
        write_npy(&tmp.join("ticks.npy"), DType::U32, &[t], &as_bytes(&col!(ticks, u32)))?;
        write_npy(&tmp.join("outcome.npy"), DType::U8, &[t], &col!(outcome, u8))?;
        write_npy(&tmp.join("episode.npy"), DType::U32, &[t], &as_bytes(&episode_col))?;
        write_npy(&tmp.join("day.npy"), DType::U16, &[t], &as_bytes(&col!(day, u16)))?;
        write_npy(&tmp.join("minute.npy"), DType::U16, &[t], &as_bytes(&col!(minute, u16)))?;
        write_npy(&tmp.join("progress.npy"), DType::U16, &[t, 3], &as_bytes(&col!(progress, u16)))?;
        write_npy(&tmp.join("expert_intent.npy"), DType::U8, &[t], &col!(intent, u8))?;
        fs::rename(&tmp, &fin)?;
        for l in lines {
            writeln!(self.episodes_file, "{l}")?;
        }
        self.episodes_file.flush()?;
        self.total_steps += t as u64;
        self.pending.clear();
        self.pending_bytes = 0;
        Ok(())
    }
    /// Flush remaining episodes and write `final_obs/global.npy`. Returns (episodes, steps).
    pub fn finish(mut self) -> io::Result<(u64, u64)> {
        self.flush_shard()?;
        let e = self.next_episode as usize;
        write_npy(&self.dir.join("final_obs").join("global.npy"), DType::F32, &[e, self.g], &as_bytes(&self.final_obs))?;
        Ok((self.next_episode, self.total_steps))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_is_64_aligned_and_parses() {
        for shape in [vec![3usize], vec![7, 777], vec![1, 2, 3]] {
            let h = npy_header(DType::F16, &shape);
            assert_eq!(h.len() % 64, 0);
            let (d, s, off) = read_npy_header(&h).unwrap();
            assert_eq!(d, "<f2");
            assert_eq!(s, shape);
            assert_eq!(off, h.len());
        }
    }

    #[test]
    fn mask_round_trip() {
        let m: Vec<bool> = (0..37).map(|i| i % 3 == 0).collect();
        let mut b = Vec::new();
        pack_mask_le(&m, &mut b);
        assert_eq!(b.len(), 5);
        assert_eq!(unpack_mask_le(&b, 37), m);
    }

    #[test]
    fn writer_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = DatasetWriter::create(dir.path(), &serde_json::json!({"env": "test"}), 4, 10).unwrap().with_shard_cap(200);
        for ep in 0..3 {
            let mut e = EpisodeBuf::new(4, 10);
            for t in 0..5 {
                let mask: Vec<bool> = (0..10).map(|i| (i + t) % 2 == 0).collect();
                e.push(&[0.5, 1.0, 0.25, t as f32], &mask, t as i32, t as i32, 0, -1.0, t == 4, false, 1, 10, 0, t as u16, 360, [1, 2, 3], 0);
            }
            e.final_obs = vec![ep as f32; 4];
            e.seed = ep;
            w.push(e).unwrap();
        }
        let (eps, steps) = w.finish().unwrap();
        assert_eq!((eps, steps), (3, 15));
        let lines = std::fs::read_to_string(dir.path().join("episodes.jsonl")).unwrap();
        assert_eq!(lines.lines().count(), 3);
        let shards: Vec<_> = std::fs::read_dir(dir.path().join("shards")).unwrap().collect();
        assert!(shards.len() >= 2, "shard cap should split");
        let bytes = std::fs::read(dir.path().join("final_obs/global.npy")).unwrap();
        let (_, shape, off) = read_npy_header(&bytes).unwrap();
        assert_eq!(shape, vec![3, 4]);
        assert_eq!(off % 64, 0);
    }
}
