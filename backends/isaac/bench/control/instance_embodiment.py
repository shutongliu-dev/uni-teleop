"""Kinematic identifier projection of an unchanged original right-hand robot."""
from dataclasses import dataclass, replace
import io
import xml.etree.ElementTree as ET

from bench.geometry import quaternion_matrix, transform
from bench.control.embodiment import load_embodiment
from bench.control.hand_posture import HandPosturePrior
from bench.control.kinematics import URDFKinematics
from bench.simulation.isaacsim_backend.instance_asset import namespaced_tree
from bench.simulation.isaacsim_backend.robot_instances import ROBOT_ID, checked_namespace, instance_name


@dataclass(frozen=True)
class InstanceHandPosturePrior(HandPosturePrior):
    """The source DIP/PIP inequality, with actuator names mapped bijectively."""
    namespace: str = 'member'

    def __post_init__(self):
        super().__post_init__()
        checked_namespace(self.namespace)
        if self.side != 'right':
            raise ValueError('Instance hand prior requires the original right-hand prior')

    @property
    def pairs(self):
        return {finger: tuple(instance_name(self.namespace, name) for name in pair)
                for finger, pair in super().pairs.items()}

    def describe(self):
        return {**super().describe(), 'profile_id': f'sharpa_dip_pip_{self.namespace}_source_v1'}


def load_instance(assets_root, namespace, configuration):
    """Names only are changed in memory; assets and numerical kinematics remain original."""
    source = load_embodiment(assets_root, configuration['hand_id'], configuration['assembly_version'])
    name = lambda original: instance_name(namespace, original)
    arm_path, hand_path = source.asset_paths
    arm = URDFKinematics(io.BytesIO(ET.tostring(namespaced_tree(ET.parse(arm_path).getroot(), namespace))))
    hand = URDFKinematics(io.BytesIO(ET.tostring(namespaced_tree(ET.parse(hand_path).getroot(), namespace))))
    pose = transform(configuration['base_position_m'], quaternion_matrix(configuration['base_quaternion_wxyz']))
    mount = {**source.expected_mount, 'parent': name(source.expected_mount['parent']),
             'child': name(source.expected_mount['child'])}
    return replace(source, robot_id=f'{ROBOT_ID}:{namespace}', hand_id=ROBOT_ID, side=namespace,
        arm=arm, hand=hand, arm_tip=name(source.arm_tip), frame_id=name(source.frame_id),
        tips={finger: name(link) for finger, link in source.tips.items()},
        posture_prior=InstanceHandPosturePrior(namespace=namespace), expected_mount=mount,
        arm_base=pose, mount=f'{namespace}:original_right:{source.mount}')
