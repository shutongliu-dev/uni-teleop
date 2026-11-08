"""Fail closed unless a Task7 revision adds static labels and one camera only."""
from copy import deepcopy
import argparse
import json
from pathlib import Path

from bench.isaacsim.release import digest, dump
from bench.paths import load_json
from scripts.validation.kitchen_asset_identity import _property_graph, _json_digest


def compare(old_release, new_release):
    from pxr import Usd
    old_release, new_release = Path(old_release), Path(new_release)
    scenes = [load_json(p/'scene.json') for p in (old_release, new_release)]
    tasks = [load_json(p/'task.json') for p in (old_release, new_release)]
    assert all(t['id'] == 'surface_wiping' for t in tasks)
    usd_paths = [s['native_task_bundle']['usd_path'] for s in scenes]
    stages = [Usd.Stage.Open(p) for p in usd_paths]
    graphs = [_property_graph(s, set()) for s in stages]
    before, after = graphs
    assert all(after.get(p) == value for p, value in before.items()), 'An existing USD prim changed'
    added = sorted(set(after)-set(before))
    assert added, 'No visual addition found'
    roots = ('/World/KitchenTask/VisualLabels', '/World/KitchenTask/Materials/WipingLabels')
    for path in added:
        assert any(path == p or path.startswith(p+'/') for p in roots), path
        prim = stages[1].GetPrimAtPath(path)
        assert prim.GetTypeName() in ('', 'Xform', 'Scope', 'Mesh', 'Material', 'Shader'), path
        assert not any('Physics' in name or 'Physx' in name for name in prim.GetAppliedSchemas()), path
        assert not any(a.GetTimeSamples() for a in prim.GetAttributes()), path
        assert not any(a.GetName().startswith(('physics:', 'physx')) for a in prim.GetAttributes()), path
    normalized_tasks = deepcopy(tasks)
    labels = normalized_tasks[1]['asset_metadata'].pop('visual_labels')
    assert normalized_tasks[1]['evaluation']['metadata'].pop('visual_labels') == labels
    assert labels['collision'] is False and labels['scoring'] is False
    assert normalized_tasks[0] == normalized_tasks[1], 'Rules, metadata or task behavior changed'
    normalized_scenes = deepcopy(scenes)
    for s in normalized_scenes:
        s['cameras'].pop('task')
        s['kitchen_task_provenance'].pop('asset_candidate')
        b = s['native_task_bundle']
        b.pop('usd_path'); b.pop('usd_sha256')
        b['observer']['configuration']['asset_metadata'].pop('visual_labels', None)
        b['observer']['configuration']['evaluation']['metadata'].pop('visual_labels', None)
    assert normalized_scenes[0] == normalized_scenes[1], 'Scene changed beyond labels and task camera'
    fixed_files = ['initial_state.json', 'rules.json', 'physics_profile.json']
    fixed_files += ['evaluator/'+p.name for p in (old_release/'evaluator').glob('*.py')]
    hashes = {}
    for relative in fixed_files:
        left, right = digest(old_release/relative), digest(new_release/relative)
        assert left == right, relative
        hashes[relative] = left
    return {
        'schema': 'kitchen_wiping_visual_identity.v1', 'passed': True,
        'old_release': str(old_release), 'new_release': str(new_release),
        'old_release_manifest_sha256': digest(old_release/'release_manifest.json'),
        'new_release_manifest_sha256': digest(new_release/'release_manifest.json'),
        'old_usd': {'path': usd_paths[0], 'sha256': digest(usd_paths[0])},
        'new_usd': {'path': usd_paths[1], 'sha256': digest(usd_paths[1])},
        'unchanged_existing_prim_count': len(before), 'existing_property_graph_sha256': _json_digest(before),
        'added_static_visual_prims': added, 'unchanged_files_sha256': hashes,
        'old_task_camera': scenes[0]['cameras']['task'], 'new_task_camera': scenes[1]['cameras']['task'],
        'normalized_task_sha256': _json_digest(normalized_tasks[0]),
        'normalized_scene_sha256': _json_digest(normalized_scenes[0]),
        'allowed_changes': ['Six static noncollision tabletop labels and their appearance materials',
                            'Font/label metadata and asset reference digests', 'Task camera framing only'],
        'native_validation_performed': False,
        'scope': 'Exact authored equivalence of all pre-existing USD prim properties, physical materials, '
                 'body/reset/fixture data, rules/referee, original robot/controller, parent scene and '
                 'all other cameras. Old object-contact evidence remains limited to its original '
                 'engineering probe; this receipt does not qualify new pixels or robot manipulation.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-release', type=Path, required=True)
    parser.add_argument('--new-release', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = compare(args.old_release, args.new_release)
    dump(args.output, report)
    print(json.dumps({'passed': report['passed'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
