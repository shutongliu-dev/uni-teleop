"""Frozen SHARPA reference postures, evaluated against each side's actual URDF.

These are robot-only joint targets, not grasp primitives. No simulator, task
object, measured state, trajectory or controller lifecycle is read or changed.
The upstream model supplies the zero/open pose and joint semantics. The pinch
anchor is the *open* hand reference from the recorded pickup02 engineering plan;
no closed target, wrist pose, object dimension or task strategy is included.
That trial's left-hand success does not qualify these interpolated references,
the right hand, another object, or a motion between postures.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from bench.control.hand_posture import HandPosturePrior
from bench.geometry import matrix_quaternion
from bench.protocol.hand_presets import (
    HandPresetCatalog, HandPresetRequest, HandPresetResult, HandPresetSource, HandPresetValidation,
)

VERSION = 'sharpa_w01_preposes_v1'
UPSTREAM_REPOSITORY = 'https://github.com/sharpa-robotics/sharpa-urdf-usd-xml'
UPSTREAM_COMMIT = '0d19cac602f46456b819e4b6a2c09a74982c9a3e'
PINCH_EVIDENCE_SHA256 = '6a2e49900fe1905d478e4c57a1029f66c6806ea6227975b512e87cad85da2094'
PINCH_EVIDENCE_ID = 'card_room_reserve_access_20261017/pickup02/inputs/grasp_plan.json'
FINGERS = ('thumb', 'index', 'middle', 'ring', 'little')
URDF_SHA256 = {
    'left': '2b7677225d4ba33b44688c07ef7c9dcd48ba40a31f51f81bb5e8283403e2956a',
    'right': 'ff9ffc2af5069c09378d4ffc7870d53c507a50941849be79b7dfe7c5f9dcb6cc',
}
KINEMATICS_SHA256 = {
    'left': 'f811c41b9746054f7615f1a956b0e489aee86f8c99c5f8ce92488f20ec660800',
    'right': 'ed6eb049f6f132d9501d08511f3665da1cccf40b9021f1fd0a290cf48212e29b',
}
PINCH_ANCHOR_SHA256 = {
    'left': 'da2be8f716dfd84f4008680fa0dec1cb69e50c661ff08ba64a270bf3574d0fb3',
    'right': '2a7b7dac29bea69852cfde441549645e13642f0a0d362805fd00962904574582',
}

# Independent frozen source records, even where numeric values coincide. Each
# side's axes, origins, chain topology and FK are checked separately below.
PINCH_ANCHORS = {
    'left': {
        'left_thumb_CMC_FE': .4685350207873381, 'left_thumb_CMC_AA': .05492726365273376,
        'left_thumb_MCP_FE': .27201938620618454, 'left_thumb_MCP_AA': -.19032918312335542,
        'left_thumb_IP': .2602125965682699,
        'left_index_MCP_FE': .5178952782032245, 'left_index_MCP_AA': .21792102714412587,
        'left_index_PIP': .865789003358936, 'left_index_DIP': 1.1220944065416567,
        'left_middle_MCP_FE': .6, 'left_middle_MCP_AA': .3491,
        'left_middle_PIP': 1.5, 'left_middle_DIP': 1.2,
        'left_ring_MCP_FE': .6, 'left_ring_MCP_AA': .3491,
        'left_ring_PIP': 1.5, 'left_ring_DIP': 1.2,
        'left_pinky_MCP_FE': .6, 'left_pinky_MCP_AA': .3491,
        'left_pinky_PIP': 1.5, 'left_pinky_DIP': 1.2, 'left_pinky_CMC': 0.,
    },
    'right': {
        'right_thumb_CMC_FE': .4685350207873381, 'right_thumb_CMC_AA': .05492726365273376,
        'right_thumb_MCP_FE': .27201938620618454, 'right_thumb_MCP_AA': -.19032918312335542,
        'right_thumb_IP': .2602125965682699,
        'right_index_MCP_FE': .5178952782032245, 'right_index_MCP_AA': .21792102714412587,
        'right_index_PIP': .865789003358936, 'right_index_DIP': 1.1220944065416567,
        'right_middle_MCP_FE': .6, 'right_middle_MCP_AA': .3491,
        'right_middle_PIP': 1.5, 'right_middle_DIP': 1.2,
        'right_ring_MCP_FE': .6, 'right_ring_MCP_AA': .3491,
        'right_ring_PIP': 1.5, 'right_ring_DIP': 1.2,
        'right_pinky_MCP_FE': .6, 'right_pinky_MCP_AA': .3491,
        'right_pinky_PIP': 1.5, 'right_pinky_DIP': 1.2, 'right_pinky_CMC': 0.,
    },
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _kinematics_signature(hand):
    return _digest({'root': hand.root, 'names': hand.names, 'joints': [
        {'name': j.name, 'parent': j.parent, 'child': j.child, 'kind': j.kind,
         'origin': j.origin.tolist(), 'axis': j.axis.tolist(),
         'lower': j.lower, 'upper': j.upper, 'velocity': j.velocity}
        for j in hand.joints]})


def _side_model(geometry, side):
    if side not in ('left', 'right'):
        raise ValueError('Hand preset side must be left or right')
    model = geometry.sides.get(side) if hasattr(geometry, 'sides') else geometry
    embodiment = getattr(model, 'embodiment', None)
    if (not getattr(embodiment, 'sharpa_joint_semantics', False)
            or embodiment.side != side or model.frame_id != side + '_hand_flange'):
        raise ValueError('Requested SHARPA side is not present in this robot geometry')
    hand = model.hand
    by_name = {joint.name: joint for joint in hand.joints}
    try:
        actual_urdf_sha256 = hashlib.sha256(Path(model.hand_path).read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError('Pinned hand preset asset is unavailable') from error
    expected_tips = {finger: f'{side}_{"pinky" if finger == "little" else finger}_fingertip'
                     for finger in FINGERS}
    if (set(hand.names) != set(PINCH_ANCHORS[side])
            or [joint.name for joint in hand.active] != hand.names
            or any(joint is not by_name.get(joint.name) for joint in hand.active)
            or model.fingertip_links != expected_tips
            or model.tip_offsets
            or actual_urdf_sha256 != URDF_SHA256[side]
            or _kinematics_signature(hand) != KINEMATICS_SHA256[side]):
        raise ValueError('Hand preset requires its pinned side-specific SHARPA URDF and axes')
    if model.hand_posture_prior != HandPosturePrior(side=side):
        raise ValueError('Hand preset requires the pinned SHARPA DIP/PIP coupling')
    if _digest(PINCH_ANCHORS[side]) != PINCH_ANCHOR_SHA256[side]:
        raise ValueError('Hand preset anchor parameters changed without a version update')
    return model


def _check_posture(model, q):
    if set(q) != set(model.hand.names):
        raise ValueError('Preset must specify all actual hand joints')
    for joint in model.hand.active:
        if not np.isfinite(q[joint.name]) or not joint.lower <= q[joint.name] <= joint.upper:
            raise ValueError('Hand preset is outside its pinned joint limits')
    if any(model.hand_posture_prior.violation(finger, q) for finger in FINGERS):
        raise ValueError('Hand preset violates its pinned posture coupling')


def generate_hand_preset(geometry, request: HandPresetRequest | dict) -> HandPresetResult:
    """Return 22 hand targets and local FK; never execute or infer object capture.

    amount=0 is the all-zero/open URDF configuration, amount=1 is the selected
    anchor. Interpolation is in joint space, not metric grasp aperture. For open,
    the anchor is also all zero, so amount does not alter the returned posture.
    """
    request = HandPresetRequest.model_validate(request.model_dump() if isinstance(request, HandPresetRequest) else request)
    model = _side_model(geometry, request.side)
    opened = dict.fromkeys(model.hand.names, 0.)
    anchor = dict(PINCH_ANCHORS[request.side]) if request.preset == 'pinch_prepose' else opened
    # Bounds and DIP-PIP inequalities are linear. Checking both endpoints proves
    # their whole convex interpolation domain, rather than only a sampled grid.
    _check_posture(model, opened)
    _check_posture(model, anchor)
    q = {name: float(request.amount * anchor[name]) for name in model.hand.names}
    _check_posture(model, q)
    poses = model.hand.forward(q)
    tips = {}
    for finger, link in model.fingertip_links.items():
        pose = np.asarray(poses[link], float)
        if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0., 0., 0., 1.], atol=1e-12, rtol=0.)
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-10, rtol=0.)
                or abs(np.linalg.det(pose[:3, :3]) - 1.) > 1e-10):
            raise ValueError('Invalid side-specific hand preset FK')
        tips[finger] = {'frame_id': model.frame_id, 'position_m': pose[:3, 3].tolist(),
                        'quaternion_wxyz': matrix_quaternion(pose[:3, :3]).tolist()}
    points = [np.array(tips[f]['position_m']) for f in FINGERS]
    if min(np.linalg.norm(a - b) for i, a in enumerate(points) for b in points[:i]) < .0196:
        raise ValueError('Preset FK violates the frozen virtual-tip separation bound')
    project = Path(__file__).resolve().parents[2]
    files = ('bench/control/hand_presets.py', 'bench/protocol/hand_presets.py',
             'bench/control/kinematics.py', 'bench/control/hand_posture.py', 'bench/geometry.py')
    return HandPresetResult(preset=request.preset, side=request.side, amount=request.amount,
        frame_id=model.frame_id, hand_posture_rad=q, fingertip_poses=tips,
        thumb_index_distance_m=float(np.linalg.norm(points[0] - points[1])),
        source=HandPresetSource(upstream_repository=UPSTREAM_REPOSITORY, upstream_commit=UPSTREAM_COMMIT,
            hand_urdf_sha256=URDF_SHA256[request.side], hand_kinematics_sha256=KINEMATICS_SHA256[request.side],
            anchor_sha256=_digest(anchor),
            anchor_origin=PINCH_EVIDENCE_ID if request.preset == 'pinch_prepose' else 'upstream_urdf_zero_configuration',
            anchor_evidence_sha256=PINCH_EVIDENCE_SHA256 if request.preset == 'pinch_prepose' else URDF_SHA256[request.side],
            implementation_sha256={name: hashlib.sha256((project/name).read_bytes()).hexdigest() for name in files}),
        validation=HandPresetValidation())


def hand_preset_catalog(geometry) -> HandPresetCatalog:
    """Discover the verified subset for this robot; no object-specific presets."""
    sides, unavailable = [], {}
    for side in ('left', 'right'):
        try:
            _side_model(geometry, side)
        except (AttributeError, OSError, TypeError, ValueError):
            unavailable[side] = 'No matching pinned SHARPA side, geometry or coupling'
        else:
            sides.append(side)
    return HandPresetCatalog.model_validate({'protocol': 'bench_hand_preset_catalog_v1', 'library_version': VERSION,
        'available': bool(sides), 'sides': sides,
        'reason': None if sides else 'No matching pinned SHARPA hand geometry is available',
        'presets': [
            {'preset': 'open', 'meaning': 'URDF zero configuration; amount has no effect'},
            {'preset': 'pinch_prepose', 'meaning': 'Joint interpolation toward an open thumb/index pinch reference; not object-sized closure'},
        ] if sides else [],
        'amount': {'minimum': 0., 'maximum': 1., 'default': 1.,
                   'zero': 'open URDF zero configuration', 'unit': 'joint_space_interpolation_fraction'},
        'unavailable_sides': unavailable,
        'unavailable_presets': {'envelop_prepose': 'No reviewed frozen anchor with full-domain evidence'},
        'validation': HandPresetValidation().model_dump(mode='json'),
        'execution': 'Use hand_posture_rad in an ordinary move_geometry action; duration and motion are checked separately'})
