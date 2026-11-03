"""Standalone right SHARPA subtree, shared by both simulation backends.

Extract into the run directory; never rewrite the installed asset bundle.
"""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

PALM = 'right_hand_C_MC'
INITIAL_POSITION = np.array([0., 0., 1.12])
# Hand +Z points along the fingers: start fingers down, palm facing +X.
INITIAL_QUATERNION = np.array([1., 0., 0., 0.])  # xyzw, Rx(pi)


@dataclass
class HandModel:
    urdf: Path
    names: tuple
    lower: np.ndarray
    upper: np.ndarray
    effort: np.ndarray

    @property
    def initial(self):
        return np.clip(np.zeros(len(self.names)), self.lower, self.upper)


def extract_hand(data, output):
    source = Path(data) / 'assets/robots/assemblies/fr3_sharpa/isaacsim_6_1_v2/fr3_sharpa.urdf'
    tree = ET.parse(source).getroot()
    links, joints, pending = {PALM}, [], [PALM]
    while pending:
        parent = pending.pop(0)
        for joint in tree.findall('joint'):
            if joint.find('parent').get('link') == parent:
                child = joint.find('child').get('link')
                links.add(child)
                pending.append(child)
                joints.append(deepcopy(joint))
    root = ET.Element('robot', name='standalone_sharpa')
    for link in tree.findall('link'):
        if link.get('name') in links:
            root.append(deepcopy(link))
    root.extend(joints)
    for mesh in root.findall('.//mesh'):
        path = (source.parent / mesh.get('filename')).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh.set('filename', str(path))
    active = [j for j in joints if j.get('type') != 'fixed']
    if len(active) != 22:
        raise ValueError('Expected SHARPA with 22 finger joints')
    path = Path(output) / 'sharpa.urdf'
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding='unicode', xml_declaration=True)
    limits = [j.find('limit') for j in active]
    return HandModel(path, tuple(j.get('name') for j in active),
                     *[np.array([float(l.get(key)) for l in limits])
                       for key in ('lower', 'upper', 'effort')])
