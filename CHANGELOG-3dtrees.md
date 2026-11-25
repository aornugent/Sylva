# 3dtrees changes to Sylva

What our trunk adds over upstream (`git diff main...3dtrees/rebuild`), as it stands now, rewritten when it changes.
The trunk is `3dtrees/rebuild` until it is merged into `develop`. `main` mirrors upstream; the trunk is merged up to
upstream's 2.0.0. The rules behind each feature are in its module documentation, not here. Work since deleted, the
joint fit (`support_tree`, `ScanIndex`, `JointParams`, `contest`, `tree_context`) and the beam scan with solids, is at
the tag `archive/sylva-joint-2026-11-25`; earlier work such as `joint_tree` at `archive/sylva-develop-2026-11-12`.

- `io.tile_ray_clouds` cuts Shots files into buffered raycloudtools tiles, thinned side by side within a memory
  budget. Beside each cloud, sidecars name every ray's scan, pulse row and echo. Its terrain cloud leaves out
  MTA-uncertain echoes, whose ghosts can lie tens of metres underground and drag `rayextract terrain` down with them.
  3dtrees needs it because `rayextract` can't take a whole plot. Additive.
- The RXP reader's `keep_uncertain` keeps an MTA-uncertain echo in zone 1, on the pulse it was recorded after, where
  upstream keeps it at the learned model's best zone; confident zone calls stand. 3dtrees needs it because the model
  (trained on a VZ-600i) marks about a quarter of Alice Mulga's VZ-2000i echoes uncertain: dropped they leave false
  misses, kept at the model's zone they put 5-7 % of a scan's echoes in zone 2, on the wrong pulse. Changes the output
  only with `keep_uncertain=True`.
- The learned MTA model makes its keep/drop call without summing every tree where only that call matters (one zone
  hypothesis, as with `max_zone=1`): it stops once the remaining trees' leaf bounds can't carry the score across the
  threshold, and skips echoes the amplitude filters drop anyway. 3dtrees reads every scan this way; it cuts a read's
  CPU time by about an eighth. No output change.
- `ScanPosition.save_shots` reads a scan, moves it into the project frame and writes its Shots file without the
  pulses passing through Python, and the RXP-to-Shots path along the way stops copying whole scans (columns move
  rather than copy, misses are appended and the pulses transformed in place, the writer re-aims beams a row group at
  a time). 3dtrees needs it to read more scans at once: a 40 M-pulse scan peaks at 5.3 GB instead of 13.2 GB. No
  output change.
- Our bindings live in `crates/sylva-py/src/fork_py.rs`, reached from upstream's `lib.rs` by one `mod` line and one
  `register` line, so a merge from upstream meets them nowhere else. No outputs.
