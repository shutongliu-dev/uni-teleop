"""Explicit pre-engaged scene diagnostic; never a loose-bolt task reset."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from bench.geometry import matrix_quaternion, quaternion_matrix
from bench.isaacsim.release import digest, task_inputs
from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset


# These immutable releases supplied native18/native19's same engaged poses.
# Their PGS acceptance is not transferred to the selected scene's solver.
CONTACT_RESET_MANIFESTS = {
    '924b9f4dec7eb1437b60d2292cb0ca7c454b243b5b26647086a4591642a06b5d': 'native18',
    '94c67e82adbb4c2cb35d7522ea58df665b833b1c821d21ed76780d1ad542983a': 'native19',
}
ROLES = {'screwdriver': 'objects', 'bolt': 'objects', 'threaded_fixture': 'fixtures'}


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def _actor(scene, role):
    matches = [row for row in scene[ROLES[role]] if row['id'] == role]
    if len(matches) != 1:
        raise ValueError('Contact diagnostic requires exactly one actor: '+role)
    return matches[0]


def _pose(actor):
    p = np.asarray(actor['position_m'], dtype=float)
    q = np.asarray(actor.get('quaternion_wxyz', [1., 0., 0., 0.]), dtype=float)
    if (p.shape != (3,) or q.shape != (4,) or not np.isfinite(np.r_[p, q]).all()
            or not np.isclose(np.linalg.norm(q), 1., atol=1e-6, rtol=0)):
        raise ValueError('Contact reset requires finite positions and unit quaternions')
    return p, q


def _physical_actor(actor):
    excluded = {'id', 'asset_root', 'position_m', 'quaternion_wxyz',
                'usd_physics_fallbacks_provenance'}
    return {k: v for k, v in actor.items() if k not in excluded}


def _physical_asset(actor):
    asset = load_task_asset(actor['asset_id'], actor['asset_root'], verify_hashes=True)
    excluded = {'asset_root', 'manifest_path', 'manifest_sha256', 'source_dependencies_sha256'}
    return {k: v for k, v in asset.items() if k not in excluded}


def prepare_contact_reset(scene, task, source_release=None):
    """Copy inputs and optionally map two pinned engaged poses before env creation.

    No source physics, robot reset, fixture, support, rule or threshold is copied.
    Only the two movable reset poses and their passive-check reference change.
    """
    selected, diagnostic_task = deepcopy(scene), deepcopy(task)
    if source_release is None:
        return selected, diagnostic_task, None
    root = Path(source_release).resolve()
    source_hash = digest(root/'release_manifest.json')
    if source_hash not in CONTACT_RESET_MANIFESTS:
        raise ValueError('Contact reset source must be the pinned native18/native19 release')
    _, source, source_task, _ = task_inputs(root, 'fastener_assembly')
    if task['evaluation'].get('reset_variant') != 'loose_bolt':
        raise ValueError('Contact reset option requires an explicit loose-bolt target')
    # Solver and timestep are deliberately NOT imported from the pose source.
    ignored = {'solver_type', 'physics_dt_s'}
    profile = lambda s: {k: v for k, v in s['isaacsim'].items() if k not in ignored}
    if profile(scene) != profile(source):
        raise ValueError('Contact reset physics profile differs beyond solver/timestep')
    if any(scene.get(k) != source.get(k) for k in ('physics', 'robot', 'constraints', 'cloth')):
        raise ValueError('Contact reset requires compatible gravity, robot and constraints')
    if task['initial_state'] != source_task['initial_state']:
        raise ValueError('Contact reset cannot import a different robot ready pose')
    references = {'initial_objects', 'fixture_position_m', 'fixture_quaternion_wxyz'}
    if any(task['evaluation'].get(k) != v for k, v in source_task['evaluation'].items()
           if k not in references):
        raise ValueError('Contact reset cannot change frozen physical thresholds or actor paths')
    assets = {}
    for role in ROLES:
        current_actor, old_actor = _actor(scene, role), _actor(source, role)
        if _physical_actor(current_actor) != _physical_actor(old_actor):
            raise ValueError('Contact reset actor physics differs: '+role)
        current_asset, old_asset = _physical_asset(current_actor), _physical_asset(old_actor)
        if current_asset != old_asset:
            raise ValueError('Contact reset rigid shape/mass/material source differs: '+role)
        assets[role] = {'physical_asset_sha256': _json_digest(current_asset),
                        'usd_path': current_asset['usd_path'],
                        'usd_sha256': digest(current_asset['usd_path'])}
    current_fixture, source_fixture = _actor(scene, 'threaded_fixture'), _actor(source, 'threaded_fixture')
    cp, cq = _pose(current_fixture)
    sp, sq = _pose(source_fixture)
    diagnostic_task['evaluation']['fixture_position_m'] = cp.tolist()
    diagnostic_task['evaluation']['fixture_quaternion_wxyz'] = cq.tolist()
    rotation = quaternion_matrix(cq) @ quaternion_matrix(sq).T
    unchanged_fixture = np.array_equal(cp, sp) and np.array_equal(cq, sq)
    poses = {}
    for role in ('screwdriver', 'bolt'):
        actor, old = _actor(selected, role), _actor(source, role)
        p, q = _pose(old)
        pose = {'position_m': (cp+rotation@(p-sp)).tolist(),
                'quaternion_wxyz': matrix_quaternion(rotation@quaternion_matrix(q)).tolist()}
        if unchanged_fixture:
            pose = {k: deepcopy(old[k]) for k in ('position_m', 'quaternion_wxyz')}
        poses[role] = {'canonical_pose': {k: deepcopy(actor[k]) for k in pose},
                       'source_pose': {k: deepcopy(old[k]) for k in pose},
                       'mapped_diagnostic_pose': deepcopy(pose)}
        actor.update(pose)
        diagnostic_task['evaluation']['initial_objects'][role] = deepcopy(pose)
    receipt = {
        'schema': 'mechanical_fastener_contact_reset.v1',
        'source_release': str(root), 'source_manifest_sha256': source_hash,
        'historical_pose_run': CONTACT_RESET_MANIFESTS[source_hash],
        'source_files_sha256': {name: digest(root/name) for name in
            ('scene.json', 'task.json', 'initial_state.json', 'physics_profile.json')},
        'canonical_scene_sha256': _json_digest(scene),
        'diagnostic_scene_sha256': _json_digest(selected),
        'source_fixture_pose': {'position_m': sp.tolist(), 'quaternion_wxyz': sq.tolist()},
        'current_fixture_pose': {'position_m': cp.tolist(), 'quaternion_wxyz': cq.tolist()},
        'object_poses': poses, 'compatible_rigid_assets': assets,
        'source_isaacsim_options': deepcopy(source['isaacsim']),
        'selected_isaacsim_options': deepcopy(selected['isaacsim']),
        'changed_scene_fields': [f'objects.{role}.{key}' for role in poses
                                 for key in ('position_m', 'quaternion_wxyz')],
        'passive_reference_fields': ['evaluation.initial_objects',
                                    'evaluation.fixture_position_m', 'evaluation.fixture_quaternion_wxyz'],
        'original_releases_modified': False, 'object_pose_velocity_writes_after_reset': 0,
        'physics_accepted': False, 'task_success_claimed': False,
        'robot_manipulation_qualified': False, 'prior_acceptance_transfers': False,
        'scope': 'Pre-engaged tool/bolt contact diagnostic only; does not test loose pickup, thread start or tool insertion.'}
    return selected, diagnostic_task, receipt


def passive_support_pairs(scene, config, paths, *, contact_reset=False):
    """Choose actual reset support explicitly, retaining old releases' pairs."""
    if contact_reset:
        return {'tool_bolt': (paths['screwdriver'], paths['bolt']),
                'bolt_fixture': (paths['bolt'], paths['fixture'])}
    tool = config.get('tool_rest_actor')
    bolt = config.get('supply_rest_actor')
    pairs = ({'tool_supply': (paths['screwdriver'], tool)} if tool else
             {'tool_bolt': (paths['screwdriver'], paths['bolt'])})
    layout = scene.get('mechanical_task_provenance', {}).get('bolt_supply_layout', {})
    if bolt and layout.get('vertical_support') == 'original_workroom_table':
        room = scene.get('workroom', {})
        if (room.get('family'), room.get('version')) != ('mechanical_assembly', 'mechanical_workshop_20261017_v3'):
            raise ValueError('Table support requires the pinned mechanical workroom')
        pairs['bolt_table'] = (paths['bolt'], '/World/Room/Geometry/worktop/Collider')
    elif bolt:
        pairs['bolt_supply'] = (paths['bolt'], bolt)
    else:
        pairs['bolt_fixture'] = (paths['bolt'], paths['fixture'])
    return pairs
