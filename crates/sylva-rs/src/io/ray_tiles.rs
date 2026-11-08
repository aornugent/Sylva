// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Shots files -> tiled raycloudtools ray clouds, streamed.
//!
//! `rayextract trees` loads its whole cloud (~450 B per ray at peak), so a
//! plot goes through it tile by tile. This cuts the echoes of many
//! single-position Shots files into square tiles centred on multiples of the
//! tile width (the grid `rayextract trees --grid_width` assumes) with a
//! buffer on every side, binning each echo by its *end point* (raysplit
//! clips rays to the cell instead, so a tile holding a scanner keeps almost
//! every ray). Each tile is thinned to the first echo per voxel, in file
//! order, and written as an RCT ray cloud whose rays start at their scanner:
//! `x y z` (double), `time` = row index in the tile, `nx ny nz` = start -
//! end (float), grey RGB, alpha 255. Misses are not written: `rayextract`
//! ignores unbounded rays.
//!
//! Memory is bounded by one row group per file being read and one tile
//! being thinned; the binned echoes wait on disk.
//!
//! Beside each cloud, sidecars name every ray's echo exactly, in cloud order:
//! `scan.u16` (the file), `pulse.u32` (the pulse's row in that file) and
//! `echo.u8` (the echo's place in its pulse, from 0). So anything `rayextract`
//! writes per ray (`_labels.bin`) joins to the echo's character and its pulse.

use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::io::shots::ShotsFile;
use crate::Point;

/// How to cut and thin the tiles.
#[derive(Debug, Clone)]
pub struct TileOptions {
    /// Tile width (m); tile `(kx, ky)` covers `[(k - 1/2) w, (k + 1/2) w)`.
    pub width: f64,
    /// Overlap added on every side of a tile (m); must be below `width / 2`.
    pub buffer: f64,
    /// Thinning voxel for the tile clouds (m); 0 keeps every echo.
    pub voxel: f64,
    /// Thinning voxel for the terrain cloud, cut from each tile's own cell (m).
    pub terrain_voxel: f64,
    /// Keep only echoes with `xmin <= x <= xmax`, `ymin <= y <= ymax`.
    pub extent: Option<[f64; 4]>,
    /// Tiles with fewer rays after thinning are not written.
    pub min_rays: usize,
}

/// One written tile.
#[derive(Debug, Clone)]
pub struct TileInfo {
    pub kx: i64,
    pub ky: i64,
    /// Echoes binned into the tile, buffer included, before thinning.
    pub raw: usize,
    /// Rays written.
    pub rays: usize,
}

const REC: usize = 31; // x, y, z f64 + file index u16 + pulse row u32 + echo in pulse u8

/// Which echo a binned point is: its pulse's row in the file and its place in that pulse.
type EchoId = (u32, u8);

fn tile_name(kx: i64, ky: i64) -> String {
    format!("{kx}_{ky}")
}

/// Cells whose buffered extent holds `p` (at most 2 x 2 with buffer < width / 2).
fn cells(p: &Point, w: f64, b: f64, out: &mut Vec<(i64, i64)>) {
    out.clear();
    let (x0, x1) = (((p[0] - b) / w - 0.5).ceil() as i64, ((p[0] + b) / w + 0.5).floor() as i64);
    let (y0, y1) = (((p[1] - b) / w - 0.5).ceil() as i64, ((p[1] + b) / w + 0.5).floor() as i64);
    for kx in x0..=x1 {
        for ky in y0..=y1 {
            out.push((kx, ky));
        }
    }
}

/// Bin one file's echoes into `bins/<kx>_<ky>/<file>.bin`; return its scanner position.
fn bin_file(path: &Path, file_idx: u16, bins: &Path, opts: &TileOptions) -> Result<Point> {
    let f = ShotsFile::open(path)?;
    let mut origin: Option<Point> = None;
    let mut writers: BTreeMap<(i64, i64), BufWriter<File>> = BTreeMap::new();
    let mut hit = Vec::with_capacity(4);
    let mut rec = [0u8; REC];
    rec[24..26].copy_from_slice(&file_idx.to_le_bytes());
    let mut row0: usize = 0; // pulses in the groups before this one
    for g in 0..f.n_groups() {
        let shots = f.read_group(g)?;
        if row0 + shots.n_shots() > u32::MAX as usize {
            return Err(Error::File { path: path.into(), msg: "ray tiles: more than 2^32 pulses in one file".into() });
        }
        for s in 0..shots.n_shots() {
            let o = shots.origin[s];
            match origin {
                None => origin = Some(o),
                Some(p) if p != o => {
                    return Err(Error::File { path: path.into(), msg: "ray tiles need one scan position per file".into() })
                }
                _ => {}
            }
            let d = shots.direction[s];
            let e0 = shots.echo_start[s];
            rec[26..30].copy_from_slice(&((row0 + s) as u32).to_le_bytes());
            for e in e0..e0 + shots.echo_count[s] as usize {
                rec[30] = u8::try_from(e - e0).map_err(|_| Error::File { path: path.into(), msg: "ray tiles: more than 255 echoes in a pulse".into() })?;
                let r = shots.echo_range[e];
                let p = [o[0] + d[0] * r, o[1] + d[1] * r, o[2] + d[2] * r];
                if let Some([x0, y0, x1, y1]) = opts.extent {
                    if p[0] < x0 || p[0] > x1 || p[1] < y0 || p[1] > y1 {
                        continue;
                    }
                }
                for (k, v) in p.iter().enumerate() {
                    rec[k * 8..k * 8 + 8].copy_from_slice(&v.to_le_bytes());
                }
                cells(&p, opts.width, opts.buffer, &mut hit);
                for &c in &hit {
                    let w = match writers.get_mut(&c) {
                        Some(w) => w,
                        None => {
                            let dir = bins.join(tile_name(c.0, c.1));
                            fs::create_dir_all(&dir)?;
                            writers.entry(c).or_insert(BufWriter::with_capacity(1 << 20, File::create(dir.join(format!("{file_idx:05}.bin")))?))
                        }
                    };
                    w.write_all(&rec)?;
                }
            }
        }
        row0 += shots.n_shots();
    }
    for (_, mut w) in writers {
        w.flush()?;
    }
    origin.ok_or_else(|| Error::File { path: path.into(), msg: "no shots".into() })
}

/// Keep the first point per voxel, in order.
fn thin(pts: &[(Point, u16)], voxel: f64) -> Vec<usize> {
    if voxel <= 0.0 {
        return (0..pts.len()).collect();
    }
    let mut seen: HashSet<(i64, i64, i64)> = HashSet::with_capacity(pts.len() / 2);
    (0..pts.len())
        .filter(|&i| {
            let p = pts[i].0;
            seen.insert(((p[0] / voxel).floor() as i64, (p[1] / voxel).floor() as i64, (p[2] / voxel).floor() as i64))
        })
        .collect()
}

/// Write an RCT ray cloud: rays from `origins[file]` to each point.
pub fn write_rct_ray_cloud(path: &Path, pts: &[(Point, u16)], keep: &[usize], origins: &[Point]) -> Result<()> {
    let mut w = BufWriter::with_capacity(1 << 22, File::create(path)?);
    write!(
        w,
        "ply\nformat binary_little_endian 1.0\ncomment generated by sylva ray_tiles\nelement vertex {}\n\
         property double x\nproperty double y\nproperty double z\nproperty double time\n\
         property float nx\nproperty float ny\nproperty float nz\n\
         property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\nend_header\n",
        keep.len()
    )?;
    let mut rec = [0u8; 48];
    rec[44..48].copy_from_slice(&[127, 127, 127, 255]);
    for (row, &i) in keep.iter().enumerate() {
        let (p, f) = pts[i];
        let o = origins[f as usize];
        for k in 0..3 {
            rec[k * 8..k * 8 + 8].copy_from_slice(&p[k].to_le_bytes());
            rec[32 + k * 4..36 + k * 4].copy_from_slice(&((o[k] - p[k]) as f32).to_le_bytes());
        }
        rec[24..32].copy_from_slice(&(row as f64).to_le_bytes());
        w.write_all(&rec)?;
    }
    w.flush()?;
    Ok(())
}

/// A tile's binned echoes, and which echo each is (parallel arrays).
fn read_bins(dir: &Path) -> Result<(Vec<(Point, u16)>, Vec<EchoId>)> {
    let mut parts: Vec<PathBuf> = fs::read_dir(dir)?.map(|e| e.map(|e| e.path())).collect::<std::io::Result<_>>()?;
    parts.sort(); // file order, so thinning keeps the same echo every run
    let (mut out, mut ids) = (Vec::new(), Vec::new());
    let mut buf = Vec::new();
    for p in parts {
        buf.clear();
        BufReader::new(File::open(&p)?).read_to_end(&mut buf)?;
        for r in buf.chunks_exact(REC) {
            let f = |k: usize| f64::from_le_bytes(r[k * 8..k * 8 + 8].try_into().unwrap());
            out.push(([f(0), f(1), f(2)], u16::from_le_bytes([r[24], r[25]])));
            ids.push((u32::from_le_bytes(r[26..30].try_into().unwrap()), r[30]));
        }
    }
    Ok((out, ids))
}

/// Thin one tile and write `tiles/<kx>_<ky>/cloud.ply` and its sidecars `scan.u16`, `pulse.u32`, `echo.u8`;
/// return its info and its terrain points (its own cell, at `terrain_voxel`).
fn finish_tile(bin_dir: &Path, out: &Path, origins: &[Point], opts: &TileOptions) -> Result<(TileInfo, Vec<(Point, u16)>)> {
    let name = bin_dir.file_name().unwrap().to_string_lossy().to_string();
    let (kx, ky) = name.split_once('_').and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?))).ok_or_else(|| Error::invalid(format!("bad tile dir {name}")))?;
    let (pts, ids) = read_bins(bin_dir)?;
    fs::remove_dir_all(bin_dir)?;
    let keep = thin(&pts, opts.voxel);
    let info = TileInfo { kx, ky, raw: pts.len(), rays: keep.len() };
    if keep.len() < opts.min_rays.max(1) {
        return Ok((info, Vec::new()));
    }
    let dir = out.join("tiles").join(&name);
    fs::create_dir_all(&dir)?;
    write_rct_ray_cloud(&dir.join("cloud.ply"), &pts, &keep, origins)?;
    let mut scan = BufWriter::new(File::create(dir.join("scan.u16"))?);
    let mut pulse = BufWriter::new(File::create(dir.join("pulse.u32"))?);
    let mut echo = BufWriter::new(File::create(dir.join("echo.u8"))?);
    for &i in &keep {
        scan.write_all(&pts[i].1.to_le_bytes())?;
        pulse.write_all(&ids[i].0.to_le_bytes())?;
        echo.write_all(&[ids[i].1])?;
    }
    scan.flush()?;
    pulse.flush()?;
    echo.flush()?;
    let (w, cx, cy) = (opts.width, kx as f64 * opts.width, ky as f64 * opts.width);
    let cell: Vec<(Point, u16)> = keep
        .iter()
        .map(|&i| pts[i])
        .filter(|(p, _)| p[0] >= cx - w / 2.0 && p[0] < cx + w / 2.0 && p[1] >= cy - w / 2.0 && p[1] < cy + w / 2.0)
        .collect();
    let tk = thin(&cell, opts.terrain_voxel);
    Ok((info, tk.into_iter().map(|i| cell[i]).collect()))
}

/// Cut `files` (one scan position each) into ray-cloud tiles under `out`:
/// `tiles/<kx>_<ky>/{cloud.ply, scan.u16, pulse.u32, echo.u8}` and `terrain.ply`. Returns every
/// tile binned, written or not (`rays < min_rays` are skipped), in name order.
/// `threads` bounds how many tiles are thinned at once (each holds its echoes in memory).
pub fn tile_ray_clouds(files: &[PathBuf], out: &Path, opts: &TileOptions, threads: usize) -> Result<(Vec<TileInfo>, usize)> {
    if !(opts.buffer >= 0.0 && opts.buffer < opts.width / 2.0) {
        return Err(Error::invalid("ray tiles: buffer must be in [0, width / 2)"));
    }
    if files.len() > u16::MAX as usize {
        return Err(Error::invalid("ray tiles: at most 65535 files"));
    }
    let bins = out.join("bins");
    fs::create_dir_all(&bins)?;
    let origins: Vec<Point> = files
        .par_iter()
        .enumerate()
        .map(|(i, p)| bin_file(p, i as u16, &bins, opts))
        .collect::<Result<_>>()?;
    let mut dirs: Vec<PathBuf> = fs::read_dir(&bins)?.map(|e| e.map(|e| e.path())).collect::<std::io::Result<_>>()?;
    dirs.sort();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads.max(1)).build().map_err(|e| Error::invalid(e.to_string()))?;
    let done: Vec<(TileInfo, Vec<(Point, u16)>)> = pool.install(|| dirs.par_iter().map(|d| finish_tile(d, out, &origins, opts)).collect::<Result<_>>())?;
    fs::remove_dir(&bins)?;
    let terrain: Vec<(Point, u16)> = done.iter().flat_map(|(_, t)| t.iter().copied()).collect();
    let all: Vec<usize> = (0..terrain.len()).collect();
    write_rct_ray_cloud(&out.join("terrain.ply"), &terrain, &all, &origins)?;
    Ok((done.into_iter().map(|(i, _)| i).collect(), terrain.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cells_follow_the_grid_width_convention() {
        let mut out = Vec::new();
        cells(&[0.0, 0.0, 0.0], 25.0, 5.0, &mut out);
        assert_eq!(out, vec![(0, 0)]);
        cells(&[12.0, 14.0, 0.0], 25.0, 5.0, &mut out);
        assert_eq!(out, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
        cells(&[-12.6, 0.0, 0.0], 25.0, 5.0, &mut out);
        assert_eq!(out, vec![(-1, 0), (0, 0)]);
    }

    #[test]
    fn thin_keeps_the_first_point_per_voxel() {
        let pts = vec![([0.001, 0.0, 0.0], 0), ([0.004, 0.0, 0.0], 1), ([0.02, 0.0, 0.0], 0)];
        assert_eq!(thin(&pts, 0.01), vec![0, 2]);
        assert_eq!(thin(&pts, 0.0), vec![0, 1, 2]);
    }
}
