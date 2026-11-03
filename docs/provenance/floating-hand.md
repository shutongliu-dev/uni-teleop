# Standalone hand teleoperation

Design reference: `humanoid-wholebody-datasets`, commit `9eb55eb`,
https://github.com/loversinsnow/humanoid-wholebody-datasets .

Read-only review covered `hand_teleop/calibration.py`, `transforms.py`, `scene.py`,
`models.py`, `cli.py` and the README. No files or environments in that project
were changed, and it is not a runtime dependency. The integration is newly
implemented in uni-teleop; it does not vendor that project's source.

Adopted concepts: independent controller palm poses and glove fingers, relative
pose anchoring, a free hand root with six-axis impedance and gravity support,
and independent input-channel holds. Coordinate conversion is OpenXR to
X-forward/Y-left/Z-up, with spatial rotation increments applied at the anchor.

The integration reuses uni-teleop's PICO receiver, MANUS acquisition process and
official SHARPA optimizer boundary. SHARPA kinematics, visuals, collisions,
inertias and finger effort limits come from the installed DexBench robot asset.
This implementation currently supports one right hand, unlike the reference's
dual xHand/Inspire/Wuji2 setup. Both simulators run native contact dynamics.
