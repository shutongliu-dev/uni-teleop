# DexBench data-v6 integration

Upstream: https://github.com/VVVVME50/dexbench, tag `data-v6`, commit
`08c4e48f19cc108d8dc283b0ba1e758b13d03217` (release source ZIP).
Imported on 2026-11-08. Original upstream file hashes are recorded in the
`dexbench-data-v6` entry in `sources.lock.json`; the older `dexbench` entry
continues to document retained legacy code and the teleoperation foundation.

The import updates `backends/isaac/{bench,scenes,scripts,agent,tests}`, package
metadata and the data manifest. Device adapters are excluded: acquisition
remains owned by `uni_teleop/devices`, using the existing module migration map.
Local teleoperation, arm motion, workbench presentation and replay extensions
are retained. The simulator integration combines v6 task/robot loading with
the CPU rigid profile, configurable physics/control timing, optional task
observers and grouped, on-demand camera rendering. V6 cloth and particle visual
updates are retained in that grouped rendering path.

The downloaded data is not source-controlled. This installation uses
`local/data/dexbench-data-v6`, selected by local configuration `DEXBENCH_DATA`
and the backend's ignored `runtime/data` symlink. The two archive parts passed
their supplied SHA-256 checks; extraction used zstd `--long=31` and retained
hardlinks. The official manifest check passed for all 74,392 files. Original
data-v3 assets and the pre-import code backup are retained locally; historical
episodes are not rewritten.

The catalog exposes 7 rooms and 51 current tasks, with original and L-mount
robot profiles. The teleop default stays `original`; `--robot-profile l_mount`
and `--robot-configuration` request explicit alternatives. Catalog availability
does not establish native simulation or teleoperation support for every task:
in particular, CPU rigid mode rejects cloth, particles and soft bodies, and
single-arm mappings do not establish multi-arm control support.

Validation on this machine:

- Unified launcher CPU suite: 53 tests passed.
- Focused backend CPU suite: 69 tests passed (arm motion, task bundles,
  rendering synchronization, CPU profile admission and contact safety).
  The upstream render test double was adapted to the grouped camera API;
  CPU admission coverage includes v6 named cloths. The broad upstream suite
  was stopped after repeated full-asset hashing; it was not completed and is
  not claimed as passing.
- MANUS recipe command construction: dry-run passed; no live hardware started.
- V6 kitchen/breakfast_serving, original robot: native headless capture passed,
  including headset preview, 3 stored states, 2 control steps at 40 ms and
  physics steps at 4 ms.
- Separate replay process: 3 states rendered to three nonuniform RGB camera
  streams, without advancing physics. This is a pipeline smoke test, not task
  success or validation of all 51 tasks.

Local evidence is under `local/downloads/dexbench-data-v6` and
`local/validation/data-v6`; these files are not committed.
