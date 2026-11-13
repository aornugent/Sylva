# 3dtrees changes to Sylva

What our `develop` adds over upstream (`git diff main...develop`), as it stands now, rewritten when `develop`
changes. `main` mirrors upstream. Bench numbers and the history behind each rule are in 3dtrees' research notes and in
the module documentation, not here. Work since deleted, such as the per-tree fit `joint_tree`, is at the tag
`archive/sylva-develop-2026-11-12`; `qsm.rebase`, 3dtrees ticket 125's base stage, waits for review on the branch
`3dtrees/125-rebase`.

- `qsm.support_tree` measures one tree's wood from the rays. Each row is cut into pieces of at most 30 cm. Each
  scan's pulses near a piece measure its circle from their silhouette and front. Where no circle is determined they
  bound its radius instead (spans, passes and the ray veto), or a row's own stops give it a point circle. Pieces with
  several stems are fitted by the exact likelihood. One solve over the tree, under a prior along the branches, then
  gives each node its radius, held inside its bounds and capped by the site's D/H. Each node says how its radius is
  known (`basis`) and what a reviewer should check (`diag`), and gets `p_wood` from its existence evidence. Every stop
  is classified, and the tree's volume comes with an interval. 3dtrees needs it because this is its measurement: about
  half a CPU-second per tree, where the refit and `joint_tree` took minutes. Its outputs are only those 3dtrees reads,
  plus `pieces`. A new function: changes no existing output. The rules are in `support_tree.rs`'s module
  documentation.
- `qsm.ScanIndex` holds one scan's pulses binned by direction, with each echo's character and each pulse's row in its
  Shots file. `qsm.pairs` returns the pulses near each cylinder. 3dtrees needs them because every fit above reads
  pulses by cylinder, and the row joins a fit's echoes to the tile sidecars. Additive, with `ShotsFile::read_rays`.
- `qsm.joint` is the joint model's per-pulse likelihood, with exact derivatives. `qsm.joint_table` gives its pair
  table, and `joint_fit` (Rust) fits a table under a Gaussian Markov prior. 3dtrees needs them because they fit
  `support_tree`'s routed pieces. `_core.qsm_joint_loglik` exposes the likelihood for checking. Additive, with a
  `libm` dependency for erfc.
- `qsm.JointParams` holds the settings of `support_tree` and the likelihood. Each field is a real, written
  once in Rust (`params!`), and Python makes a frozen dataclass from the core's defaults. 3dtrees needs it because its
  fit cache keys on `dataclasses.asdict` of it and the Sylva checkout. Additive.
- `qsm.tree_context` gives each tree its neighbours' cylinders within a reach, in parallel. 3dtrees needs it because
  it built these in numpy, 5x slower; the contexts are identical. Additive.
- `qsm.contest` gives, of the points near each tree's cylinders, the share another tree holds and which tree holds
  most, in parallel. 3dtrees needs it because its numpy version took most of a plot's rebuild. Bit-identical, and
  additive.
- `qsm.evidence` checks each cylinder of a QSM against the rays. 3dtrees calls it nowhere today but may score with
  it, so it is kept apart in `evidence.rs` and `evidence.py`, to be deleted whole. Additive.
- `io.tile_ray_clouds` cuts Shots files into buffered raycloudtools tiles, with bounded memory. Beside each cloud,
  sidecars name every ray's scan, pulse row and echo. 3dtrees needs it because `rayextract` can't take a whole plot,
  and the sidecars join what it writes per ray back to the echoes. Additive.
- `synthetic.scan(..., solids=)` puts solid, opaque capped cylinders in a beam scan beside the points. 3dtrees needs it
  because wood drawn as surface points echoes off its back face and widens thin silhouettes, so its bench couldn't
  score wood under 2 cm. Scans without solids are unchanged.
- Parity cases for the likelihood and the pair table (`tests/parity/cases_joint.py`) are recorded against 3dtrees'
  numpy oracle, so a change to either shows. Tests only.
- Our bindings live in `crates/sylva-py/src/fork_py.rs`, reached from upstream's `lib.rs` by one `mod` line and one
  `register` line, so a merge from upstream meets them nowhere else. No outputs.
