"""Static public kinematics of an FR3 + hand embodiment for the classical executor.

Pure Python; never imports a simulator. The default right SHARPA keeps its
pinned derived with_flange URDF exactly. Every other registered hand uses its
versioned FR3 assembly: the controller's hand tree is the assembly subtree below
the arm's tool flange (fr3_link8), including any adapter disc. Every mount
convention places each hand identically in its hand frame H (fingers +z, palm
+x, thumb +y), so the public hand frame keeps the name ``<side>_hand_flange`` and
is H: fr3_link8 itself for flange_axis_centred_v1 (identity wrist offset, the
original executor path), and fr3_link8 rotated 180 deg about the flange axis for
flange_axis_centred_v2 (for SHARPA H is the URDF link of that name in both).
Hand-relative quantities (fingertips in the hand frame) are therefore identical
across conventions; only the arm's joint-7 solution differs.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import io
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from bench.control.hand_posture import HandPosturePrior
from bench.control.kinematics import URDFKinematics
from bench.paths import load_json

ARM_URDF = 'robots/franka/fr3/fr3_no_hand.absolute.urdf'
SHARPA_HAND_URDF = 'robots/sharpa/wave_01/derived/{side}_sharpa_wave_with_flange.urdf'
SHARPA_FINGERS = ('thumb', 'index', 'middle', 'ring', 'little')
DEFAULT_HAND = 'sharpa_w01_right'


def urdf_subtree(source, root_link, stop_links=()):
    """A robot element with root_link and its descendants, not descending below stop_links."""
    root = source if isinstance(source, ET.Element) else ET.parse(source).getroot()
    links = {link.get('name'): link for link in root.findall('link')}
    if root_link not in links:
        raise ValueError(f'URDF has no link {root_link!r}')
    children = {}
    for joint in root.findall('joint'):
        children.setdefault(joint.find('parent').get('link'), []).append(joint)
    robot = ET.Element('robot', name=f"{root.get('name', 'robot')}__{root_link}")
    robot.append(deepcopy(links[root_link]))
    stack = [root_link]
    while stack:
        link = stack.pop()
        if link in stop_links:
            continue
        for joint in children.get(link, []):
            child = joint.find('child').get('link')
            robot.append(deepcopy(links[child]))
            robot.append(deepcopy(joint))
            stack.append(child)
    return robot


class CoupledKinematics(URDFKinematics):
    """URDF kinematics whose public coordinates exclude passive mimic followers.

    ``forward`` expands every follower from its leader before FK, so a public
    joint map fully determines the pose, exactly like the native mimic drive.
    """

    def __init__(self, source, mimic=None):
        # The kinematics algorithm is hash-pinned by recorded datasets; feed it a
        # serialized document rather than changing its constructor.
        if isinstance(source, ET.Element):
            source = io.BytesIO(ET.tostring(source))
        super().__init__(source)
        self.mimic = {name: {'joint': rule['joint'], 'multiplier': float(rule['multiplier']),
                             'offset': float(rule['offset'])} for name, rule in (mimic or {}).items()}
        known = {joint.name: joint for joint in self.active}
        self.followers = {}
        for follower, rule in self.mimic.items():
            if follower not in known or rule['joint'] not in known or rule['joint'] in self.mimic:
                raise ValueError(f'Invalid mimic coupling for {follower!r}')
            self.followers.setdefault(rule['joint'], []).append((known[follower], rule['multiplier']))
        self.active = [joint for joint in self.active if joint.name not in self.mimic]
        self.names = [joint.name for joint in self.active]

    def expand(self, positions):
        q = dict(positions)
        for follower, rule in self.mimic.items():
            q[follower] = float(q.get(rule['joint'], 0.)) * rule['multiplier'] + rule['offset']
        return q

    def forward(self, positions, base=None):
        return super().forward(self.expand(positions) if self.mimic else positions, base)


@dataclass
class Embodiment:
    """Everything the executor needs to know about one arm + hand, nothing native."""
    robot_id: str
    hand_id: str
    side: str
    arm: URDFKinematics
    arm_tip: str
    hand: URDFKinematics
    frame_id: str
    tips: dict
    tip_offsets: dict
    posture_prior: HandPosturePrior
    asset_paths: list
    mount: str
    expected_mount: dict
    sharpa_joint_semantics: bool
    arm_base: np.ndarray = field(default_factory=lambda: np.eye(4))
    # arm_tip -> public hand frame (identity for flange_axis_centred_v1).
    wrist_offset: np.ndarray = field(default_factory=lambda: np.eye(4))
    assembly_version: str = None
    base_joint_names: tuple = ()
    controller_tip_semantics: str = 'URDF virtual frame origins, not contact-patch centers'


def _fingers_partition(hand, tips, frame):
    """Per-finger independent chains to ``frame``; joints shared by all chains are base joints."""
    children = {joint.child: joint for joint in hand.joints}
    public = set(hand.names)
    chains = {}
    for finger, (tip, _) in tips.items():
        chain, child = [], tip
        if child not in children:
            raise ValueError(f'URDF fingertip frame is missing: {tip}')
        while child in children:
            joint = children[child]
            if joint.name in public:
                chain.append(joint.name)
            child = joint.parent
        if child != frame or not chain:
            raise ValueError(f'URDF fingertip is disconnected: {tip}')
        chains[finger] = list(reversed(chain))
    shared = set.intersection(*(set(chain) for chain in chains.values())) if len(chains) > 1 else set()
    return chains, tuple(name for name in hand.names if name in shared)


def hand_frame_offset(yaw):
    """fr3_link8 -> hand frame H: a pure rotation about the flange axis."""
    offset = np.eye(4)
    if yaw:
        offset[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    return offset


def rerooted_hand_tree(subtree, flange_link, frame_link, yaw):
    """The hand subtree hung from a bare hand-frame link H = flange_link * Rz(yaw).

    Joints leaving ``flange_link`` get origin Rz(-yaw) * origin, so every pose in
    H coordinates equals the identity-yaw assembly's pose in flange coordinates.
    """
    from bench.geometry import rpy_matrix
    from scipy.spatial.transform import Rotation
    tree = deepcopy(subtree)
    root = tree.find(f"link[@name='{flange_link}']")
    tree.remove(root)
    tree.insert(0, ET.Element('link', name=frame_link))
    inverse = np.linalg.inv(hand_frame_offset(yaw))
    for joint in tree.findall('joint'):
        if joint.find('parent').get('link') != flange_link:
            continue
        joint.find('parent').set('link', frame_link)
        origin = joint.find('origin')
        if origin is None:
            origin = ET.SubElement(joint, 'origin', xyz='0 0 0', rpy='0 0 0')
        T = np.eye(4)
        T[:3, :3] = rpy_matrix(np.fromstring(origin.get('rpy', '0 0 0'), sep=' '))
        T[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        T = inverse @ T
        origin.set('xyz', ' '.join(repr(float(x)) for x in T[:3, 3]))
        origin.set('rpy', ' '.join(repr(float(x)) for x in Rotation.from_matrix(T[:3, :3]).as_euler('xyz')))
    return tree


def load_embodiment(assets_root, hand_id=None, assembly_version=None):
    """Static executor embodiment for FR3 + one registered right hand."""
    from bench.simulation.isaacsim_backend.robot_registry import get_hand_spec
    assets = Path(assets_root)
    hand_id = hand_id or DEFAULT_HAND
    spec = get_hand_spec(hand_id, assembly_version)
    yaw = spec['hand_frame_yaw_rad']
    arm_path = assets / ARM_URDF
    arm = URDFKinematics(arm_path)
    if hand_id == DEFAULT_HAND:
        hand_path = assets / SHARPA_HAND_URDF.format(side='right')
        tips = {name: (f'right_{"pinky" if name == "little" else name}_fingertip', None) for name in SHARPA_FINGERS}
        if not yaw:
            return Embodiment(robot_id=hand_id, hand_id=hand_id, side='right', arm=arm, arm_tip='fr3_link8',
                hand=URDFKinematics(hand_path), frame_id='right_hand_flange', tips={k: v[0] for k, v in tips.items()},
                tip_offsets={}, posture_prior=HandPosturePrior(), asset_paths=[arm_path, hand_path],
                mount='fr3_link8_to_right_hand_flange_identity_uncalibrated_simulation_assumption',
                expected_mount={'parent': 'fr3_link8', 'child': 'right_hand_flange',
                                'translation_m': [0, 0, 0], 'quaternion_wxyz': [1, 0, 0, 0]},
                sharpa_joint_semantics=True)
        return Embodiment(robot_id=hand_id, hand_id=hand_id, side='right', arm=arm, arm_tip='fr3_link8',
            hand=URDFKinematics(hand_path), frame_id='right_hand_flange', tips={k: v[0] for k, v in tips.items()},
            tip_offsets={}, posture_prior=HandPosturePrior(), asset_paths=[arm_path, hand_path],
            mount=f"fr3_link8_to_right_hand_flange_{spec['mount_convention']}_uncalibrated_simulation_assumption",
            expected_mount={'parent': 'fr3_link8', 'child': 'right_hand_flange',
                            'translation_m': [0, 0, 0], 'quaternion_wxyz': spec['mount_quaternion_wxyz']},
            sharpa_joint_semantics=True, wrist_offset=hand_frame_offset(yaw), assembly_version=spec['assembly_version'])
    directory = assets / 'robots/assemblies' / ('fr3_' + hand_id) / spec['assembly_version']
    source = load_json(directory / 'SOURCE.json')
    assembly = Path(source['urdf_path'])
    if source['hand_spec']['mount_translation_m'] != spec['mount_translation_m'] or \
            source['hand_spec']['mount_rpy_rad'] != spec['mount_rpy_rad']:
        raise ValueError('Versioned assembly mount differs from the hand registry')
    subtree = urdf_subtree(assembly, 'fr3_link8')
    if yaw:
        subtree = rerooted_hand_tree(subtree, 'fr3_link8', 'right_hand_flange', yaw)
    hand = CoupledKinematics(subtree, spec['mimic_joints'])
    prefix = spec['prefix']
    frames = spec.get('controller_fingertips') or {f: [link, [0., 0., 0.]] for f, link in spec['fingertip_links'].items()}
    tips = {finger: (prefix + link, np.asarray(offset, dtype=float)) for finger, (link, offset) in frames.items()}
    mount = source['mount']
    return Embodiment(robot_id=hand_id, hand_id=hand_id, side='right', arm=arm, arm_tip='fr3_link8', hand=hand,
        frame_id='right_hand_flange', tips={k: v[0] for k, v in tips.items()},
        tip_offsets={k: v[1] for k, v in tips.items() if np.any(v[1] != 0)},
        posture_prior=HandPosturePrior.unconstrained(), asset_paths=[arm_path, assembly],
        mount=f"fr3_link8_to_{mount['child']}_{spec['mount_convention']}_uncalibrated_simulation_assumption",
        expected_mount={key: mount[key] for key in ('parent', 'child', 'translation_m', 'quaternion_wxyz')},
        sharpa_joint_semantics=False, wrist_offset=hand_frame_offset(yaw),
        assembly_version=spec['assembly_version'] if yaw else None,
        controller_tip_semantics=spec.get('controller_fingertip_semantics',
                                          'URDF virtual frame origins, not contact-patch centers'))


BIMANUAL_ROBOT = 'dual_fr3_sharpa_w01'


def load_bimanual_side(assets_root, side, assembly_version=None, *, arm_base_poses=None):
    """Static executor embodiment of one arm of the dual FR3 + SHARPA robot.

    The arm chain comes from that side's versioned assembly (names prefixed
    fr3_<side>_); the hand is the official <side>_sharpa_wave_with_flange URDF
    at that assembly's flange mount. ``arm_base`` is the arm's offset from the
    robot base frame.
    """
    from bench.simulation.isaacsim_backend.robot_registry import get_hand_spec
    from bench.geometry import quaternion_matrix, transform
    from bench.simulation.isaacsim_backend.robot_layout import arm_base_poses as resolve_arm_base_poses
    if side not in ('left', 'right'):
        raise ValueError('side must be left or right')
    assets = Path(assets_root)
    spec = get_hand_spec(BIMANUAL_ROBOT, assembly_version)
    yaw = spec['hand_frame_yaw_rad']
    directory = assets / 'robots/assemblies' / BIMANUAL_ROBOT / spec['assembly_version']
    source = load_json(directory / f'SOURCE_{side}.json')
    assembly = Path(source['urdf_path'])
    arm = URDFKinematics(io.BytesIO(ET.tostring(urdf_subtree(assembly, f'fr3_{side}_base', stop_links=(f'fr3_{side}_link8',)))))
    hand_path = assets / SHARPA_HAND_URDF.format(side=side)
    pose = resolve_arm_base_poses({'hand_id': BIMANUAL_ROBOT, 'assembly_version': assembly_version,
                                  **({'arm_base_poses': arm_base_poses} if arm_base_poses is not None else {})})[side]
    offset = transform(pose['position_m'], quaternion_matrix(pose['quaternion_wxyz']))
    mount = source['mount']
    return Embodiment(robot_id=f'{BIMANUAL_ROBOT}:{side}', hand_id=BIMANUAL_ROBOT, side=side, arm=arm,
        arm_tip=f'fr3_{side}_link8', hand=URDFKinematics(hand_path), frame_id=f'{side}_hand_flange',
        tips={name: f'{side}_{"pinky" if name == "little" else name}_fingertip' for name in SHARPA_FINGERS},
        tip_offsets={}, posture_prior=HandPosturePrior(side=side), asset_paths=[assembly, hand_path],
        mount=(f'fr3_{side}_link8_to_{side}_hand_flange_identity_uncalibrated_simulation_assumption' if not yaw else
               f"fr3_{side}_link8_to_{side}_hand_flange_{spec['mount_convention']}_uncalibrated_simulation_assumption"),
        expected_mount={key: mount[key] for key in ('parent', 'child', 'translation_m', 'quaternion_wxyz')},
        sharpa_joint_semantics=True, arm_base=offset, wrist_offset=hand_frame_offset(yaw),
        assembly_version=spec['assembly_version'] if yaw else None)
