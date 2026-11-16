// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! A scene of proposed elements (sticks and leaf patches) scored against scans' range images by the ray likelihood
//! (3dtrees research 130). Its inputs are pulses (`crate::io::shots`, `crate::shots::reconstruct`).
//!
//! - [`lattice`]: each pulse's native (line, col) cell from its time.
//! - [`range_image`]: one scan's observations on its lattice, and the signals read off them.
//! - [`moments`]: surfel moments per (voxel, scan), and the candidate elements read off them.
//! - [`settings`]: the instrument and site settings and the background.
//! - [`foliage`]: the foliage field, a residual extinction on voxels.
//! - [`scene`]: elements, their rendering, the scan alignment error and the [`scene::Scene`] that holds them.
//!
//! # Glossary
//!
//! One name per concept, the same in Rust, the bindings, 3dtrees' drivers and its notes.
//!
//! | concept | name | meaning |
//! |---|---|---|
//! | one scanner position's pulses | scan | its range image is the scan on its lattice |
//! | a lattice position | cell | holds a pulse's echoes, a miss (a pulse with no echo) or no pulse |
//! | a cell's nearest echo | first echo | the one the ray likelihood reads |
//! | the most echoes the instrument records per pulse | the target cap | `max_targets`; a pulse at it is capped (`RangeImage::capped`), and says nothing past its last echo |
//! | the range up to which a pulse shows clear space | free space | `free_to`: the first echo less the receiver's resolution |
//! | a proposed part of the scene | element | a stick (an axis and radius; wood) or a patch (a ball; leaf) |
//! | elements held against scans | scene | `Scene` |
//! | `−log` likelihood ratio against the empty scene | cost | also of a move: `add`, `remove`, `change`, `costs_to_add`, `costs_to_remove` |
//! | a cell's likelihood of its first echo | ray likelihood | the scene's data term |
//! | an element's coverage of a cell's beam | fill | the beam (a uniform disc) convolved with the section |
//! | what a cell's first echo says about an element | outcome | on it, empty (none), in front, behind (passed through) |
//! | the element's predicted echo in a cell | the element's echo | `front_echo`; [`scene::Outcome::of`] against it |
//! | the cost from pulses that passed through | evidence against | `ray_behind` (and `ray_empty`, pulses with no echo) |
//! | cells the element mostly fills but whose first echo isn't its | expected hits with no echo | |
//! | the weight a cell's evidence counts with | cell weight | `1 / (φ m)`, [`settings::Settings::cell_weight`] |
//! | φ, for correlation the model leaves out | evidence discount | `discount` |
//! | beams overlapping a spot | footprints per cell | `m`, `multiplicity` |
//! | an along-beam error per (scan, place) | scan alignment error | `align_*`; its value at an element, `align_shift` |
//! | the scan alignment error's mode and precision kept between refits, a move's change to them a rank-k correction | held alignment | `refit_alignment` re-solves it; `exact_cost` is the exact marginal |
//! | a section's departure from round, per (element, scan) | stem front shift | `front_shift_*` |
//! | an element's departure from its form | shape mismatch | `mismatch`, `mismatch_*` |
//! | the chance a pulse its section fills meets solid | opacity | `opacity` |
//! | an echo's reflectance against its return's energy share, placing a silhouette's edge within a cell | located edges | `refl_noise`, `refl`, `obs_refl` |
//! | an element's reflectance offset from its surface's, per (element, scan) | reflectance level | `refl_level`, `refl_level_sd` |
//! | a cell's surface continuity | excess; surface class | solid, mid, fragmented |
//! | wood-or-leaf evidence from photometry | class evidence | `class_*`, the row's `class` |
//! | what elements don't explain: extinction σ per voxel | foliage field | `Foliage`, `foliage()`, `refit_foliage`; held through moves, as the alignment is (`FieldCounts`) |
//! | what of a ray the foliage field is fitted from | the field's cut | a ray no element fronts counts whole; one an element fronts counts up to its front element's front surface (`cut`: its clear path and its echoes in front), never its echoes at or behind it; voxels within `margin` of an element's surface are left out (`foliage`'s module docs) |
//! | κ outside the field, and a ground plane | background | `Background` |
//! | the chance the pulse meets no foliage before a return | chance a pulse arrives | `arrives` |
//! | the cost's curvature in an element's geometry | information | how well a scan pins the element down; `expected` its average over the cells' outcomes |
//! | the variance of the cost's slope were the cells independent | gradient's variance | `gradient_variance`; with E, the sandwich `E⁻¹ J E⁻¹` |
//! | the arc of a stick's outline a scan's echoes on it span | arc | the support function's coverage, per (element, scan) and per candidate |
//! | the instrument and site values | settings | `Settings`; a site's settings file in research 130 |
//! | the epoch (scanning day) the alignment error's spread is stated for | reference epoch | `ref_epoch`; a scan of another epoch adds `align_epoch_*` |
//! | elements read off the moments | candidates | `candidates` |
//! | sticks and paths not seen, scored per scan | hidden links | |
//! | hypotheses whose cost together isn't the sum of their costs alone | interacting hypotheses | `interactions`: a shared cell or front element, or a shared alignment node |
//! | a measurement against the stand's truth or AM's held-out scans | check | |

pub mod foliage;
pub mod lattice;
pub mod settings;
pub mod moments;
pub mod range_image;
pub mod scene;
