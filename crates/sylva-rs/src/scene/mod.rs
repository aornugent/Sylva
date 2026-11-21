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
//! - [`instrument`]: the instrument that recorded a scan: its cap and detection floor, owned by its range image.
//! - [`settings`]: the site settings and the background.
//! - [`foliage`]: the foliage field, a residual extinction on voxels.
//! - [`dual`]: forward derivatives in any number of variables: second order for the own nodes' Newton, first order
//!   for the geometry's slope and Fisher information.
//! - [`render`]: elements and their layers in a view's cells, the k-buffer and its mixture.
//! - [`pyramid`]: one scan's pulses aggregated over its lattice level by level: carving.
//! - [`pulse`]: one cell's pulse against its owners and the one pass over it.
//! - [`term`]: an element's (or a group's) term per scan: its cells, its own nodes and its deposit.
//! - [`alignment`]: the scan alignment error, solved with the terms and held between refits.
//! - [`chain`]: chains grown by the likelihood beside a scene: the coupling along them, a chain's state, the grower.
//! - [`query`]: what the scene predicts and how well it pins elements down, without moving them.
//! - [`scene`]: the [`scene::Scene`] that holds elements against scans: its cost, moves and rows.
//!
//! # Glossary
//!
//! One name per concept, the same in Rust, the bindings, 3dtrees' drivers and its notes.
//!
//! | concept | name | meaning |
//! |---|---|---|
//! | one scanner position's pulses | scan | its range image is the scan on its lattice |
//! | a lattice position | cell | holds a pulse's echoes, a miss (a pulse with no echo) or no pulse |
//! | a cell's nearest echo | first echo | the one the outcome reads |
//! | a pulse's echoes after its first | later echoes | the ray likelihood reads the pulse whole, each echo behind the last (3dtrees research 130 slice 2c) |
//! | what a pulse meets that isn't a surface it resolves | occluders; the medium | `Occluder`: an element's layer or the ground, one list per pulse, nearest first; the foliage field, κ outside its box |
//! | the share of a return that merges with the medium behind its far side | merge share | `merge_share`, the site's calibrated map of the field's `1 − exp(−b ∫ σ (1 − p_resolve))`; a mixture inside the return |
//! | how far a merge moves an echo back from the nearer return's range | pull | `pull`, by separation and energy ratio, measured |
//! | a cell's pulse and what made its echoes | pulse; owners | `Pulse`: its echoes against its owners, its occluders and the medium (and outliers) |
//! | the receiver's reading of a pulse's occluders | receiver | `Receiver`: which returns merge, which are recorded, the cap; the expected number of returns it records (`predict`'s `returns`) |
//! | one reading of a pulse that gives every quantity the scene takes from it | the pass | `Pulse::read`: the likelihood with its derivatives, the receiver, the beam each owner leaves (`predict`'s `beam`), each echo's responsibilities (3dtrees research 130 slice-pulse) |
//! | the posterior over which owner made an echo | responsibility | per echo, over the members, the medium and an outlier, summing to one (`predict`'s `owned`); with a foliage field, what the elements take of each echo the field counts (soft attribution) |
//! | the most targets the scanner reports per pulse | the target cap | `max_targets`, the programme's; every target reported takes a slot, those the import drops too (`RangeImage::slots`); a pulse that used them all is censored (`RangeImage::capped`) and says nothing past its last echo |
//! | the range up to which a pulse shows clear space | free space | `free_to`: the first echo less the receiver's resolution; what a pulse shows clear along its ray, its echoes and the beam they leave, in one place ([`foliage::walked`]) |
//! | one scan's pulses aggregated over its lattice, level by level | depth pyramid | [`pyramid::Pyramid`]: per tile its first echoes' nearest and farthest, its pulses, misses, echoes and energy; carving a voxel ([`pyramid::carve`]: every pulse in its cover clear past it) read off it at the level its cover fits (3dtrees research 130 slice-pyramid) |
//! | a proposed part of the scene | element | a form (a stick, an axis and radius; or a patch, a ball) and a class |
//! | what an element is made of, wood or leaf | class | `Class`: its reflectance; a mark, not a shape |
//! | sticks joined end to end, scored as one | chain; group | `Element::group`, a key: a group's members are one term per scan, one set of nuisances (alignment shift, stem front shift, reflectance level) |
//! | a declared term over a chain's geometry alone | coupling | [`chain::Coupling`]: taper, form departures and bends (the bend sd by radius) as a Gaussian Markov process along the chain; it proposes and ranks the grower's continuations and gives a chain's row, beside the cost, never in it |
//! | a chain grown piece by piece by the likelihood | grower; a chain's state | [`scene::Scene::grow`]; [`chain::Chain`]: its term in information form, updated by each piece's cells alone (3dtrees research 130 slice-grower) |
//! | elements held against scans | scene | `Scene` |
//! | `−log` likelihood ratio against the empty scene | cost | also of a move: `add`, `remove`, `change`, `costs_to_add` |
//! | a cell's likelihood of its pulse | ray likelihood | the scene's data term |
//! | an element's coverage of a cell's beam | fill | the beam (a uniform disc) convolved with the section |
//! | what a cell's first echo says about an element | outcome | on it, empty (none), in front, behind (passed through) |
//! | the element's predicted echo in a cell | the element's echo | `front_echo`; [`scene::Outcome::of`] against it |
//! | the cost from pulses that passed through | evidence against | `ray_behind` (and `ray_empty`, pulses with no echo) |
//! | the weight a cell's evidence counts with | cell weight | `1 / (φ m)`, [`settings::Settings::cell_weight`] |
//! | the chance a layer is part of its cell | entry weight | a smooth step in its widest fill from `MIN_FILL` (0.2 %) to `FULL_FILL` (1 %); a layer at 1 % or more is wholly in its cell, one below partial; a cell's likelihood is the mixture over which partial layers are in it (3dtrees research 130 slice-trim) |
//! | the chance a layer is its cell's nearest | front share | how a cell's evidence is shared among its layers' terms; all the front layer's where that is wholly in it |
//! | φ, for correlation the model leaves out | evidence discount | `discount` |
//! | beams overlapping a spot | footprints per cell | `m`, `multiplicity`: per axis of the cell's lattice steps (`RangeImage::steps`: the zenith step along a line, the head's step × sin θ across lines), so an element's total weight is its area in footprints |
//! | an along-beam error per (scan, place) | scan alignment error | `align_*`; its value at an element, `align_shift` |
//! | the scan alignment error's mode and precision kept between refits, a move's change to them a rank-k correction | held alignment | `refit_alignment` re-solves it; `exact_cost` is the exact marginal |
//! | a section's departure from round, per (element, scan) | stem front shift | `front_shift_*` |
//! | an element's departure from its form | shape mismatch | `mismatch`, `mismatch_*` |
//! | the chance a pulse its section fills meets solid | opacity | `opacity` |
//! | an echo's reflectance against its return's energy share, placing a silhouette's edge within a cell | located edges | `refl_noise`, `refl`, `obs_refl` |
//! | an element's reflectance offset from its surface's, per (element, scan) | reflectance level | `refl_level`, `refl_level_sd` |
//! | a cell's surface continuity | excess; surface class | solid, mid, fragmented; per cell, the depth bias and noise of every echo in it; a term's, the median over the cells it mostly fills, its alignment spread |
//! | wood-or-leaf evidence from photometry | class evidence | the consumer's, from `predict`'s cells (3dtrees research 130 slice 9), beside the cost, never in it; not the scene's |
//! | what elements don't explain: extinction σ per voxel | foliage field | `Foliage`, `foliage()`; its counts are kept live through moves (`FieldCounts`) |
//! | what of a ray the foliage field counts | soft attribution | every ray's whole path; each echo a hit less what the elements take of it by their responsibilities, at each term's nodes and front share (a term's `deposit`; 3dtrees research 130 slice-soft): each echo's mass sums to one between the field and the elements |
//! | the foliage field's prior per voxel | the field's prior | a Gamma, `field_prior_mean` worth `field_prior_path` m of weighted ray; measured per site (the empty scene's type-II maximum) |
//! | the field's counts scored with σ integrated out | the field's term | `Σ ln NB` over the voxels (a Gamma-mixed Poisson marginal), live and exact through moves; its change against the empty scene's is the cost's `field_part`, the last column of `costs_to_add` in a scene with a field, and each row's `field`; the plug-in the cells' ratios already read of the hits the elements take is each term's own (its `plug`, in its rows' `ray_*`) (3dtrees research 130 formulation.md) |
//! | what a pulse meets of the field | the held field | σ's posterior mean, read by the ray likelihood (τ, the medium's echoes, the merge) and so by the responsibilities, fixed at the scene's build (one step of EM from κ) or held from another scene (`hold_foliage`) |
//! | κ outside the field, and a ground plane | background | `Background` |
//! | the chance the pulse meets no foliage before a return | chance a pulse arrives | `arrives`; `predict`'s `read`: an echo of the pulse on the element |
//! | the cost's expected curvature in an element's geometry | information | how well a scan pins the element down: `expected`, the Fisher information (the curvature's average over the cells' outcomes, from first derivatives), beside the cost's `slope`, both from one forward pass per cell; over several elements jointly (a pair's), the latents they share counted once |
//! | what one hypothesis is expected to cost against another were the other true | divergence | `divergence(p, q)`: `E_p[cost(q) − cost(p)]` from their predictions |
//! | the variance of the cost's slope were the cells independent | gradient's variance | `gradient_variance`; with E, the sandwich `E⁻¹ J E⁻¹` |
//! | the arc of a stick's outline a scan's echoes on it span | arc | the support function's coverage, per (element, scan) and per candidate |
//! | the instrument and site values | settings | `Settings`; a site's settings file in research 130 |
//! | the epoch (scanning day) the alignment error's spread is stated for | reference epoch | `ref_epoch`; a scan of another epoch adds `align_epoch_*` |
//! | elements read off the moments | candidates | `candidates` |

pub mod alignment;
pub mod chain;
pub mod dual;
#[cfg(test)]
mod fixtures;
pub mod foliage;
pub mod instrument;
pub mod lattice;
pub mod moments;
pub mod pulse;
pub mod pyramid;
pub mod query;
pub mod range_image;
pub mod render;
pub mod scene;
pub mod settings;
pub mod term;
