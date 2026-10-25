"""Versioned elementary manipulation tasks, separate from the frozen workcell."""
from pathlib import Path

from bench.paths import DATA_ROOT


# Current development release: v12 SHARPA (2026-10-15). It is v8_embodiment_fidelity_r2
# (gravity compensation, SHARPA vendor actuators, load-bearing pinch) with the hand turned
# 180 deg about the flange axis (flange_axis_centred_v2), so top-down grasps use the natural
# posture. Tasks, evaluator and physics are byte-identical to v8 r2; it is a separate stratum.
# Prepared scoring is distinct from native acceptance; v8 r2 and earlier stay frozen and
# selectable by path.
DEFAULT_TASK_RELEASE = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/simple_manipulation/'
                            'v12_mount_v2_r1/sharpa_w01_right')
