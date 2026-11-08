"""Derive the gear base world with the open-end wrench 50 mm closer to the robot.

Only the canonical table position of the free open-end wrench changes, from
task.ORIGINAL_OPEN_END_WRENCH_XY_M (.80, .29) to task.OPEN_END_WRENCH_XY_M (.75, .29):
at x .80 its pickup approach/precontact/lift anchors lay 14.6/36.6/28.5 mm beyond the
orientation-free fingertip reach of the original single-arm FR3. Its asset, orientation
and 0.2 mm table gap, the other five objects, fixtures, robot, room, desktop dressing,
rules, referee, physics profile and every asset byte are unchanged. The L5 reset audit
provenance is recomputed on the derived scene. Development evidence stays external.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

from bench.isaacsim.release import checked, digest, dump, validate
from bench.paths import load_json, OUTPUT_ROOT
from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
from .task import (OPEN_END_WRENCH_XY_M, ORIGINAL_OPEN_END_WRENCH_XY_M, TASK_ID,
                   reset_audit, tabletop_position)

CHANGED = ('scene.json', 'task.json', 'source_provenance.json')


def moved_scene(scene, task):
    """The derived scene/task: the wrench at its reach position and a fresh reset audit."""
    scene, task = deepcopy(scene), deepcopy(task)
    wrench = next(row for row in scene['objects'] if row['id'] == 'open_end_wrench')
    if wrench['position_m'][:2] != list(ORIGINAL_OPEN_END_WRENCH_XY_M):
        raise ValueError('The source release does not hold the wrench at its original position')
    asset = load_task_asset(wrench['asset_id'], wrench['asset_root'], verify_hashes=True)
    position = tabletop_position(asset, list(OPEN_END_WRENCH_XY_M), wrench['quaternion_wxyz'])
    if position[2] != wrench['position_m'][2]:
        raise ValueError('A planar wrench move must keep its exact table gap')
    wrench['position_m'] = position
    initial = task['evaluation']['initial_objects']['open_end_wrench']
    if initial != {'position_m': [*ORIGINAL_OPEN_END_WRENCH_XY_M, position[2]],
                   'quaternion_wxyz': wrench['quaternion_wxyz']}:
        raise ValueError('Referee initial pose differs from the source scene')
    initial['position_m'] = deepcopy(position)
    layout = task['supply_layout']
    audit = reset_audit(scene, minimum_table_edge_clearance_m=layout['minimum_table_edge_clearance_m'],
                        minimum_fixture_clearance_m=layout['minimum_fixture_clearance_m'])
    if not audit['passed']:
        raise ValueError('Derived wrench reset audit failed: '+json.dumps(audit['failures']))
    scene['mechanical_task_provenance']['reset_audit'] = audit
    return scene, task


def prepare_release(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    original = validate(source)
    if original['task_id'] != TASK_ID or 'configuration_id' in original:
        raise ValueError('The wrench reach revision derives the single-arm three-gear task release')
    if output.exists() or not output.is_relative_to(OUTPUT_ROOT):
        raise ValueError('Choose a new immutable shared release directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+output.name+'.preparing-', dir=output.parent))
    try:
        for relative, row in original['files'].items():
            checked(source/relative, row['sha256'])
            destination = staging/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source/relative, destination)
        for name in CHANGED:
            (staging/name).chmod(0o644)
        scene = load_json(staging/'scene.json')
        task = load_json(staging/'task.json')
        stored = scene['mechanical_task_provenance']['reset_audit']
        scene, task = moved_scene(scene, task)
        audit = scene['mechanical_task_provenance']['reset_audit']
        refreshed = sorted(name for name in audit['objects'] if audit['objects'][name] != stored['objects'][name])
        stale = {name: row for name, row in stored['objects'].items() if name != 'open_end_wrench'}
        for name, row in stale.items():
            # The stored idler_grip_v2 audit predates its taller grip hub (idler_grip_v1 asset):
            # only that upper bound may differ; every other unmoved row must be identical.
            expected = deepcopy(row)
            if name == 'idler_gear':
                expected['world_bounds_m'][1][2] = audit['objects'][name]['world_bounds_m'][1][2]
            if audit['objects'][name] != expected:
                raise ValueError('Recomputed reset audit changed an unmoved object: '+name)
        if {key: value for key, value in audit.items() if key != 'objects'} != {
                key: value for key, value in stored.items() if key != 'objects'}:
            raise ValueError('Recomputed reset audit changed its contract or verdict')
        dump(staging/'scene.json', scene)
        dump(staging/'task.json', task)
        provenance = load_json(staging/'source_provenance.json')
        revision = {'source_release': str(source), 'source_release_sha256': digest(source/'release_manifest.json'),
            'changed': 'Only the open-end wrench canonical table position: (0.80, 0.29) to (0.75, 0.29) m, '
                       '50 mm toward the robot; same asset, orientation and 0.2 mm table gap.',
            'reason': 'At x 0.80 the wrench pickup approach/precontact/lift anchors lay 14.6/36.6/28.5 mm beyond '
                      'the orientation-free fingertip reach of the original single-arm FR3.',
            'previous_position_xy_m': list(ORIGINAL_OPEN_END_WRENCH_XY_M),
            'position_xy_m': list(OPEN_END_WRENCH_XY_M),
            'reset_audit': 'Recomputed on the derived scene (mechanical_l5_reset_audit.v2, unchanged thresholds).',
            'reset_audit_refreshed_objects': refreshed,
            'reset_audit_note': 'Besides the moved wrench row, only the idler upper bound may differ from the stored '
                                'audit, which predates the taller idler grip hub (0.8202 -> 0.8302 m).',
            'preserved': ['assets', 'orientation', 'table gap', 'other five objects', 'fixtures', 'robot', 'room',
                          'desktop dressing', 'rules', 'referee', 'physics profile'],
            'scope': 'Input derivation only; validation records remain external and are not admission or scoring inputs.'}
        provenance['wrench_reach_derivation'] = revision
        dump(staging/'source_provenance.json', provenance)
        manifest = deepcopy(original)
        manifest.update(release_id=TASK_ID+'_'+output.name,
            prepared_at_utc=datetime.now(timezone.utc).isoformat(), wrench_reach_derivation=revision,
            qualification_scope='Open-end wrench 50 mm closer to the robot; no inherited native, transmission '
                                'or robot-manipulation qualification.')
        manifest['files'] = {name: {**row, 'sha256': digest(staging/name)} for name, row in original['files'].items()}
        manifest['files_sha256'] = {name: row['sha256'] for name, row in manifest['files'].items()}
        for entry in manifest['tasks']:
            entry['scene_sha256'] = digest(staging/entry['scene'])
            entry['task_sha256'] = digest(staging/entry['task'])
        if any(row['sha256'] != manifest['files'][name]['sha256']
               for name, row in original['files'].items() if name not in CHANGED):
            raise ValueError('Wrench derivation changed a protected frozen input')
        dump(staging/'release_manifest.json', manifest)
        validate(staging)
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
