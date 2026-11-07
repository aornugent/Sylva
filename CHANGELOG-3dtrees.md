# 3dtrees changes to Sylva

`main` mirrors upstream; `develop` is ours (`git log main..develop`). Newest first.

- `qsm.pairs` returns the raw table `qsm.evidence` reduces inside: every pulse near each cylinder that reached its disc, with its closest approach to the axis and all its echoes (range, reflectance, deviation). The joint wood, foliage and ray model (3dtrees research 74) is a per-pulse likelihood and needs that table, not evidence's summaries. `ScanIndex.origin` exposes the scanner position. Additive: no output changes.

- Synced to upstream v0.3.2 and dropped our `Shots.from_scan`. Upstream's `fill_missing_from_timing` now recovers misses from pulse timing, the one part of `from_scan` that needed Sylva; 3dtrees sorts the export itself and calls `from_pointcloud`.
- `qsm.evidence` checks each cylinder of a QSM against the scan rays: the radius its echoes measure and how precisely, the beam calliper, the pulses that pass inside the claimed radius, and how far pulses penetrate (with the hits' reflectance and deviation). RCT's radii are mostly a taper prior rather than measurements, and this says which ones the rays support, contradict or can't see, and tells leaf clumps fitted as fat twigs from wood. `qsm.ScanIndex` bins each scan's pulses by direction so a whole tile's cylinders can be checked in seconds.
- `io.tile_ray_clouds` cuts Shots files into buffered raycloudtools ray-cloud tiles, streamed with bounded memory. `rayextract` can't fit a whole plot in memory, and its input has to keep each echo's ray back to its scanner.
- `qsm.surface_support` and `qsm.surface_distance` measure how well points sit on a QSM's cylinder surfaces. They were the first screen for bad fits; `evidence` has since replaced `surface_support`, and the 3dtrees viewer still colours points with `surface_distance`.
