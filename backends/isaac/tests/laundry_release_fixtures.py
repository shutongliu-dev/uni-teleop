"""Prepare CPU release fixtures from current physical inputs, without old campaigns."""
import json
from pathlib import Path

from bench.paths import load_json
from scenes.workrooms.laundry_room import release
from scenes.workrooms.laundry_room.composition import ROBOT_PHYSICS_OPTIONS


def unbound_source(source, output, *, table=True, visuals=True):
    source, output = Path(source), Path(output)
    if output.exists():
        return output
    scene, task = [load_json(source/name) for name in ('scene.json', 'task.json')]
    physics = {k: v for k, v in scene['isaacsim'].items() if k not in ROBOT_PHYSICS_OPTIONS}
    if not table:
        physics.pop('workroom_surface_materials', None)
    release.prepare_task_release(task['id'], output,
        asset_manifest=scene['laundry_task_provenance']['asset_candidate']['manifest_path'],
        physics_options=physics, rule_profile=task['evaluation']['rule_profile'],
        configuration_id=task['robot_configuration_id'],
        stand_manifest=scene['robot_configuration']['stand_manifest']['path'],
        public_material_contract=task.get('public_material_contract'),
        cloth_visual_profiles=({r['id']: r['visual_profile'] for r in scene['cloths']
                                if 'visual_profile' in r} if visuals else None))
    return output
