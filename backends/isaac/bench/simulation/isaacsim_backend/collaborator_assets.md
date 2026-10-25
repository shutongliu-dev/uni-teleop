Collaborator rigid USD assets — native reference adapter

Source, unchanged:
  <开发存储>/assets/objects/hand_system0/object_usd_20261011
Derived catalog and independent convex-hull audit geometry:
  <开发存储>/assets/objects/hand_system0/bench_catalog/v1

The source contains 1,804 USD files, including 1,677 entry layers and 127
configuration layers. The 221 collaborator name/ID groups are NOT a claim of
221 unique physical meshes. selection_index.json retains every entry, exact
entry byte-hash groups, representative choice, and reasons for exclusions.
The current adapter accepts 203 group representatives. Eighteen groups need
an explicit nested-body/articulation or frame adapter and remain excluded.
A CPU-compatible entry is not automatically native-contact-qualified.

The scene contract is the existing object/fixture contract:
  {"id":"cube", "asset_id":"collab_cube_small",
   "asset_root":"<开发存储>/assets/objects/hand_system0/bench_catalog/v1",
   "position_m":[0.45,-0.10,0.781], "quaternion_wxyz":[1,0,0,0]}

Priority IDs: collab_cube_small, collab_blue_cup, collab_bench_spoon,
collab_peg, collab_peg_hole, collab_rubber_mallet, collab_apple.
The cube measures 32 x 40 x 60 mm and source mass is 50 g. The source rigid
root is the placement frame, not necessarily the object's center of mass.
Use bounds_min_m/max_m after the scene rotation when placing on a table.

Source USDs are referenced directly. Meshes, material bindings, textures,
layer composition, collision meshes and collision approximations are retained.
Accepted sources must have meter units, Z-up, identity root transform, exactly
one rigid body on that root, positive explicit mass, and enabled mesh colliders.
There is no silent scale/root-transform normalization, hull replacement, mass
rewrite, source pose playback, or source file modification. The fixture mode
uses an immovable kinematic body without per-step pose writes. GPU contact
filters address its rigid root, preserving compound decomposition contact
identity; raw static multi-shape collider filters are unsupported on this GPU
backend. scene_info records body_mode and contact_filter_paths explicitly.

Source physics materials take priority over a root fallback friction of 0.65.
An explicit scene friction overrides every collider. Contact/rest offsets are
1 mm / 0 mm; rigid iterations 16 / 4 and speculative CCD are explicit adapter
assumptions. Values are simulation assumptions, not measured materials.
Mass is preserved; missing inertia/COM is computed by PhysX. Native metadata
records resolved mass, full inertia tensor, and COM/principal-frame pose.

All local USD/texture dependencies are hashed. Built-in OmniPBR MDL shaders
remain runtime dependencies pinned by the simulator version. Authored convex
hull meshes are exported individually solely for independent audit. A source
convexDecomposition is NOT replaced by one convex hull: its cooked pieces are
not available to the offline cloth-clearance audit and therefore that check
must report unsupported/unchecked for scenes requiring those pieces. Rigid
solver-contact checks remain available. Cup-mouth or peg-hole cavity quality
requires a dedicated native insertion/clearance test; falling on a table is
not such a test.

Rebuild in a NEW version directory rather than replacing a frozen catalog:
  <开发存储>/assets/tools/usd/26.8/venv/bin/python \
    scripts/assets/prepare_collaborator_catalog.py --output <new-version-path>
Native selected-object qualification (independent of task scores):
  ISAAC_ALLOW_UNVALIDATED_DRIVER=1 scripts/isaac_python.sh \
    scripts/assets/probe_collaborator_objects.py --device cuda:1 --output <new-run>
CPU integrity regression:
  .venv/bin/python -m pytest tests/test_collaborator_usd_catalog.py -q

Keep source provenance and any later license grants alongside the shared
source. No standalone redistribution license was included; none is inferred.

2026-10-11 native qualification evidence:
  <开发存储>/datasets/fr3_sharpa_bench/asset_inspections/hand_system0_native_20261011/
  cpu_catalog_review.json: all 203 selected-entry dependency hashes verified.
  probe_03/result.json: native trial result, passed.
  probe_03_audit.json: independent raw-state/contact/hash audit, 19 checks passed.
  probe_03/all_seven_settle.png: actual RTX view after seven gravity drops.
  probe_03/metadata.json: source dependencies and resolved mass/inertia/COM.
  Seven priority rigid objects: 2 s, 100 control intervals, all native support
  contacts positive, final speed below 0.00025 m/s, worst reported object
  contact separation -0.179 mm. The separate immovable fixture load test ran
  1.2 s, measured positive source-fixture contact impulse 0.0811 Ns, and ended
  with all object speeds below 0.00034 m/s. These are diagnostic observations,
  not calibrated force measurements or task success scores.
  Earlier probe_01 (source quaternion precision) and probe_02 (unsupported
  raw-static compound contact filter) failures remain archived. probe_03 uses
  the fixed adapter and retained source hashes; no source assets were changed.
