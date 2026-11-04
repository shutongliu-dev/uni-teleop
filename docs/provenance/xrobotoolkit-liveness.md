# XRoboToolkit connection liveness

Upstream: https://github.com/XR-Robotics/XRoboToolkit-PC-Service
Revision: `85bac4dbc1fd5cef42c74a160d9c30aa3491f122`.
The patch and upstream source hash are recorded in `sources.lock.json`.

On 2026-11-03 the PC service removed an active PICO connection after its
dedicated heartbeat was delayed. TCP tracking packets continued arriving,
but the service reported `device missing` and stopped forwarding poses to
the SDK. Restarting the SDK alone did not restore forwarding.

`xrobotoolkit-liveness.patch` updates the service's connection activity time
when a registered device supplies a complete, framing-validated packet.
The existing 20-second connection inactivity check remains enabled. Pose
freshness, controller validity, clutch and pause checks in uni-teleop are
unchanged; transport activity alone never authorizes robot motion.

The local build is isolated under `local/xr-service-fix/`. It contains source,
build script, native regression executable, provenance and the original
service wrapper. The wrapper at `local/pico-diagnostics/service-trace.sh`
prepends the patched library directory to `LD_LIBRARY_PATH`; neither the
upstream checkout nor its installed libraries are edited. Restore the saved
wrapper to roll back, then restart the service and its SDK reader.

Native regression: complete tracking refreshes liveness, partial/corrupt
frames do not, and an idle connection still expires. The original library
fails the complete-tracking case; the patched library passes all cases.
These are transport tests, not robot task-success tests.
