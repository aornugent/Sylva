# 3dtrees changes to Sylva

What the [3dtrees](https://github.com/aornugent/3dtrees) fork adds to
[uq-eorc/sylva](https://github.com/uq-eorc/sylva). We pull from upstream and never push to it.

- `main` mirrors upstream; `develop` is our trunk and merges upstream in.
- `git log main..develop` is everything that is ours.
- Each version below is a tag `3dtrees-N` on `develop`, newest first.
- New versions go on top. Older entries are rechunked now and then.

## 3dtrees-2: 2026-11-07, upstream v0.3.2

| Change | Effect for upstream users |
|---|---|
| Synced to upstream v0.3.2 | upstream's hemispherical photos, `Shots.scan_image`/`scan_lines`, corrected RXP miss reconstruction and `Shots.fill_missing_from_timing` |
| `Shots.from_scan` removed | back to upstream's API (see below) |

Removed `Shots.from_scan`: upstream's `fill_missing_from_timing` now owns miss recovery from pulse
timing. The rest of `from_scan` (sort a single-scan export by `gps_time`, then `from_pointcloud`)
lives in 3dtrees' importer. Everything else is as in 3dtrees-1.

## 3dtrees-1: 2026-11-05, upstream v0.3.1+4 (`f6a7e10`), at `92f3409`

All additions: no upstream function changes behaviour.

| Python | What it is |
|---|---|
| `sylva.qsm.evidence`, `ScanIndex`, `EvidenceParams`, `Reason`, `contradicted` | what the scan rays say about each cylinder of a QSM |
| `sylva.io.tile_ray_clouds` | Shots files → raycloudtools ray-cloud tiles, streamed |
| `sylva.qsm.surface_support`, `surface_distance` | how well points sit on a QSM's cylinder surfaces |
| `Shots.from_scan` | Shots from a single-scan LAS/LAZ/COPC export (removed in 3dtrees-2) |

### `qsm.evidence`: what the rays say about each cylinder

A QSM (for example from raycloudtools) claims a radius for every cylinder, but most of those radii
are a taper prior, not a measurement. `evidence(scans, a, b, r_claim=None, params=EvidenceParams())`
goes back to the scan rays and reports, per cylinder and independent of its claimed radius:

- `mu`, `tau`: the log radius its echoes measure, with a calibrated precision (beam footprint,
  per-scan floor, clutter). `tau == 0` means the rays don't fix the radius.
- the beam calliper, `(lo, hi)`: the bounds set by pulses that hit near the axis or passed close by.
- counts: echoes, crossings, and contradicting passes inside the claimed radius.
- `n_core`, `n_pen`, and the hits' median reflectance and deviation: how many pulses went through
  the core. A solid stops pulses at its surface; foliage lets many through. This tells leaf clumps
  fitted as fat twigs from wood (AUC 0.91 on a mulga plot).
- `reason` (a `Reason`): why `tau` is what it is: measured, no hits, or prior only (clutter,
  footprint, or thin wood seen only by echoes-only scans).
- `contradicted(ev)`: one audit rule, `n_contra >= max(2, 0.05 n_cross)`.

Each echo goes to the nearest cylinder surface whose window holds it, one hit per pulse. A tile's
cylinders are therefore judged together, and neighbours keep their own echoes.

`ScanIndex(path)` holds one Shots file's pulses binned by direction, at 16 B per pulse plus 8 per
echo. It reads only angles and ranges (`ShotsFile::read_rays`), so a 12.7M-pulse scan indexes in
0.7 s. A 61k-cylinder tile against five scans takes a few seconds.

`EvidenceParams` is a frozen dataclass holding every setting, with a `VERSION` (now 2) that a
cache can key on.

### `io.tile_ray_clouds`: streamed ray-cloud tiles

Cuts single-position Shots files into raycloudtools ray-cloud tiles:
- echoes are binned by end point into buffered cells on the grid `rayextract --grid_width` assumes;
- each tile is thinned to the first echo per voxel and written with its rays from the scanner;
- a terrain cloud is cut from each tile's own cell.

Memory stays bounded: one row group per file and a few tiles at a time, with binned echoes waiting
on disk.

### `qsm.surface_support` / `surface_distance`

- `surface_support` samples each cylinder's surface and tests the samples against a cloud.
- `surface_distance` gives each point its signed distance to its own tree's nearest cylinder.

Both are parallel and deterministic. 3dtrees has replaced `surface_support` with `evidence`, but
its viewer still uses `surface_distance`.
