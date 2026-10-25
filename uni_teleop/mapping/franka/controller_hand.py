"""PICO trigger trajectory for SHARPA: open -> pinch prepose -> close.

Robot-only targets, independent of objects and task state. The close anchors
were fitted separately to each pinned hand's FK, moving thumb/index reference
points symmetrically from the prepose toward a 12 mm separation. This is not a
surface clearance or force command. Native contacts can stop measured closure.
The other fingers retain the prepose; this trajectory is a pinch, not a power grasp.
"""
from bench.control.hand_presets import PINCH_ANCHORS

VERSION = 'sharpa_trigger_pinch_v1'
CLOSED = {
    'left': dict(thumb_CMC_FE=.46297826, thumb_CMC_AA=.06130183,
                 thumb_MCP_FE=.58094366, thumb_MCP_AA=-.22797097, thumb_IP=.48210164,
                 index_MCP_FE=.51639623, index_MCP_AA=-.24232819,
                 index_PIP=.86473382, index_DIP=1.11418055),
    'right': dict(thumb_CMC_FE=.46297522, thumb_CMC_AA=.06130569,
                  thumb_MCP_FE=.580942, thumb_MCP_AA=-.22796757, thumb_IP=.48210173,
                  index_MCP_FE=.51640116, index_MCP_AA=-.24232772,
                  index_PIP=.86472922, index_DIP=1.11417514),
}


class ControllerHand:
    def __init__(self, model):
        side = 'left' if model.embodiment.side == 'left' else 'right'
        prefix = model.frame_id.removesuffix('_hand_flange') + '_'
        self.prepose = {n: PINCH_ANCHORS[side][side+'_'+n.removeprefix(prefix)] for n in model.hand.names}
        self.closed = self.prepose | {prefix+n: value for n, value in CLOSED[side].items()}
        for pose in (self.prepose, self.closed):
            if any(not j.lower <= pose[j.name] <= j.upper for j in model.hand.active):
                raise ValueError('Controller hand anchor is outside URDF limits')

    def targets(self, amount):
        if amount <= .5:
            return {name: 2*amount*value for name, value in self.prepose.items()}
        return {name: value + (2*amount-1)*(self.closed[name]-value)
                for name, value in self.prepose.items()}
