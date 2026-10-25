# uni-teleop development

- Keep the common launcher independent of Isaac, MuJoCo and device SDK imports.
- `vendor/` contains pinned working-tree snapshots, not submodules. Keep source
  attribution and record local portability changes in `docs/integration.md`.
- FACTR is the leader arm; DYNAMIXEL names its motor/serial implementation.
- Keep simulator environments isolated. Do not change the original repositories.
- Never commit SDK binaries, license keys, operator calibrations, device serials,
  personal installation paths, model weights or session recordings.
- Physical feedback is opt-in. Do not change motor enablement as a side effect of
  listing, checking, configuring, or packaging the software.
- Distinguish CPU tests, native simulation, live device tests and task success.
- Tests for the unified launcher run with `python -m unittest discover -s tests`.
