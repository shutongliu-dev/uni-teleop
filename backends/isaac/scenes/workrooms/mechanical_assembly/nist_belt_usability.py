"""Explicit Task7 handling revision; original NIST pulleys and belt stay metric.

The independent idler is project-authored, not source CAD. Its upper hub is
physical geometry accessible above the belt, and its lower bore has a lead-in.
This module does not author robot actions, task credit or calibrated tension.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scenes.workrooms.laboratory.object_geometry import cylinder

PROFILE = 'nist_manipulation_scale_v1'


def idler_parts():
    """One watertight annular solid; real bore and no hidden supporting body."""
    radial_section = np.asarray([
        [.007, -.013], [.009, -.013], [.015, -.007],
        [.015, .013], [.012, .013], [.012, .0355],
        [.0115, .036], [.0065, .036], [.006, .0355],
        [.006, -.010], [.007, -.013],
    ])
    mesh = trimesh.creation.revolve(radial_section, sections=96)
    mesh.fix_normals()
    if not mesh.is_watertight or mesh.volume <= 0:
        raise ValueError('Idler must retain one closed solid and a real axial bore')
    return [mesh]


def idler_metadata(original_parts, original_mass, parts):
    density = original_mass / sum(p.volume for p in original_parts)
    return {
        'profile': PROFILE, 'source': 'Project-authored independent idler; not NIST source CAD.',
        'belt_contact_diameter_m': .030, 'original_belt_contact_height_m': .026,
        'overall_height_m': .049, 'upper_grasp_hub_diameter_m': .024,
        'upper_grasp_hub_height_m': .023, 'through_bore_diameter_m': .012,
        'lower_entry_diameter_m': .014, 'lower_entry_depth_m': .003,
        'lower_outer_lead_diameter_m': .018, 'lower_outer_lead_height_m': .006,
        'straight_belt_contact_height_m': .020,
        'preserved_density_kg_m3': float(density),
        'original_mass_kg': float(original_mass),
        'recomputed_mass_kg': float(density * sum(p.volume for p in parts)),
        'collision': 'Single watertight SDF body with real through bore and lower lead-in.',
    }


ASSEMBLY_GUIDANCE = {
    'instruction_zh': '建议先安装两只带轮并套入皮带，再抓住张紧轮上方凸起轮毂，'
        '从上方装到独立轴上，最后松手。允许采用其他可行顺序。',
    'mandatory_sequence': False,
    'reason_zh': '皮带采用不可伸长的被动链节近似；三轮全部就位后再横向拉过法兰，长度余量不足。',
    'scope': 'Suggested physical assembly order; no new scorer sequence or precision-tension requirement.',
}


def idler_shaft_parts():
    """Capture the idler bore before its rolling surface reaches the belt.

    The 47.5 mm post tip remains 3 mm below the seated hub's top. It is a real
    stationary guide, never a joint or force that pulls the moving idler in.
    """
    nose = trimesh.creation.revolve(np.asarray([
        [0., .0425], [.0037, .0425], [.0023, .0475],
        [0., .0475], [0., .0425],
    ]), sections=48)
    nose.fix_normals()
    return [cylinder(.0037, .041, (0., 0., .022)), nose,
            cylinder(.010, .0015, (0., 0., .00075))]
