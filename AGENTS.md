# uni-teleop development

- Keep the common launcher independent of Isaac, MuJoCo and device SDK imports.
- Own device acquisition in `uni_teleop/devices`, robot mapping in `mapping`,
  feedback in `feedback`, and presentation/recording in `video` / `recording`.
- `backends/` owns simulator engines, models and tasks. Do not reintroduce device
  implementations there or duplicate implementations for convenience.
- Add supported combinations to the component/recipe registry. Unsupported
  hardware/backend pairings must not silently select another pipeline.
- Preserve upstream attribution in `sources.lock.json` and `docs/provenance/`.
- FACTR is the leader arm; DYNAMIXEL names its motor/serial implementation.
- Keep simulator environments isolated. Do not change the original repositories.
- Never commit SDK binaries, license keys, operator calibrations, device serials,
  personal installation paths, model weights or session recordings.
- Physical feedback is opt-in. Do not change motor enablement as a side effect of
  listing, checking, configuring, or packaging the software.
- Distinguish CPU tests, native simulation, live device tests and task success.
- Tests for the unified launcher run with `python -m unittest discover -s tests`.
