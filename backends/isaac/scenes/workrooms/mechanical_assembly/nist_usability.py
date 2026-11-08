"""Explicit physical size adaptation of NIST-derived manipulation tasks.

Original CAD files and their millimetre-to-metre conversion remain immutable.
The opt-in profile bakes any additional task scale into both render and collision
geometry before mass properties, passive joints and task goals are authored.
"""
from __future__ import annotations

PROFILE = 'nist_manipulation_scale_v1'


def configure(author, requested):
    if requested is None:
        return
    if requested != PROFILE:
        raise ValueError(f'Unknown NIST usability profile: {requested}')
    author.meta['usability_profile'] = PROFILE
    author.meta['usability_adaptation'] = {
        'profile': PROFILE,
        'source_cad_unchanged': True,
        'original_nist_dimensions_claimed': False,
        'room_and_workplate_scale': 1.,
        'task_geometry_baked_in_metres': True,
        'mass_basis': 'Preserve declared per-part material density when scaling; recompute mesh volume, COM and inertia.',
        'scope': 'Physical usability adaptation, not a NIST-certified board or hand-object qualification.',
        'changes': [],
    }


def enabled(author):
    return author.meta.get('usability_profile') == PROFILE


def scale_source(author, mesh, factor):
    """Scale the just-loaded source mesh, preserving its original provenance."""
    if factor == 1.:
        return mesh
    if not enabled(author) or factor <= 0:
        raise ValueError('Source scaling requires an explicit positive usability profile')
    before = mesh.bounds.tolist()
    mesh.apply_scale(factor)
    record = author.used[next(reversed(author.used))]
    record['task_adaptation_scale'] = factor
    record['task_bounds_before_adaptation_m'] = before
    record['task_bounds_after_adaptation_m'] = mesh.bounds.tolist()
    record['task_adaptation_note'] = 'Additional physical task scale, separate from original mm-to-m conversion; not original NIST dimensions.'
    return mesh


def capsule_mass_ratio(old_radius, new_radius, length):
    """Match the same source mesh-volume convention used by Author.body."""
    import trimesh
    def volume(radius):
        return abs(trimesh.creation.capsule(radius=radius, height=length, count=[8, 8]).volume)
    return volume(new_radius)/volume(old_radius)
