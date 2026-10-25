# Local reconstruction assets

This directory is the default location for a private reconstructed scene:

- `lab_scan_collision.json` contains the scene alignment and box fallback.
- `lab_scan_collision_mesh_v2.json` references optional convex collision meshes.
- `meshes/` contains those local OBJ hulls.

The JSON and OBJ files are intentionally ignored by Git because they may encode a
real workspace. Point the runtime at another private directory with
`SONIC_RECONSTRUCTION_COLLISION` or `--scan-collision-manifest`.

The reconstruction scripts in `scripts/` regenerate these files from a local
Nerfstudio workspace. Tests use `tests/fixtures/lab_scan_collision.json`, which is
synthetic and contains no scanned geometry.
