# 3dtrees changes to Sylva

`main` mirrors upstream; `develop` is ours (`git log main..develop`). Newest first.

- Synced to upstream v0.3.2; dropped our `Shots.from_scan` (upstream's `fill_missing_from_timing` covers it).
- `qsm.evidence`: what the scan rays say about each QSM cylinder (measured log radius and precision, beam calliper, passes inside the claim, penetration and echo character to tell leaf clumps from wood), indexed per scan by `qsm.ScanIndex`.
- `io.tile_ray_clouds`: Shots files → raycloudtools ray-cloud tiles, streamed.
- `qsm.surface_support`, `qsm.surface_distance`: how well points sit on a QSM's cylinders.
