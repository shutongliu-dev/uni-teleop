Study seed assets
================

`seed_variants.py` authors immutable cases for `stationery_sorting`,
`book_stacking` and `book_shelving`. It accepts exactly integer seeds 0–9.
These are ordinary task initial cases, not learning splits or competitive games.

`prepare_asset(parent_release, seed, output)` returns `seed_manifest.json`.
Publication requires execution from a complete verified source snapshot and a
new directory below the shared `task_parts/seed_randomization/v1/study` root.
Original source assets, book artwork, font and license dependencies remain pinned.

`compose_case(parent_release, asset_manifest_path, configuration_id)` returns the
scene, task, robot initial state, frozen evaluator source texts, actual initial
body poses, geometric interaction references, collision envelopes and asset
hashes. The common publisher owns release registration and case selection.
Serialize dictionaries in their original insertion order: the historical study
rules generate their public target-condition list in that order.

Seed 0 preserves the original USD bytes and complete evaluation configuration.
Other seeds move each object by at most 12 mm in each horizontal axis and rotate
it by at most 6 degrees. Actual local mesh/primitive dimensions change uniformly:
pencil 1.01–1.04, other stationery and books 0.98–1.02. All books in a case share
one scale, preserving their size/thickness order. Pencil diameter stays at least
15 mm. Explicit mass, COM and inertia follow a density-preserving transformation
of the original engineering priors. New roughness textures preserve category
colors and existing labels. No material friction or solver parameter changes.

The task-owned fixed fixtures and shelf slots remain fixed. Book stack support
heights follow actual scaled book thickness. Current room, robot installation,
robot initial joints, controller, rule thresholds and budgets are preserved.
Private reset poses remain host metadata; public goals/rules contain only their
existing declared targets. Interaction points are geometric references, not
validated grasp plans. CPU authoring tests establish no native stability,
reachability, rendering quality or robot-manipulation qualification.
