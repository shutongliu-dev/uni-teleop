"""Version task presentation, explicit clutter and direct-table supply resets."""
from copy import deepcopy
from datetime import datetime, timezone
import importlib
from pathlib import Path
import shutil
import tempfile

from bench.isaacsim.release import checked, digest, dump, validate
from bench.paths import OUTPUT_ROOT, load_json


TASK_IDS = ('parts_sorting', 'peg_insertion', 'shaft_spacer_assembly',
            'fastener_assembly', 'gear_train_assembly')


def parent_releases(*, parents=None, gear_parent=None, parent_map=None):
    if parent_map is not None and (parents is not None or gear_parent is not None):
        raise ValueError('Choose an explicit parent map or a historical parent directory')
    if parent_map is not None:
        selected = deepcopy(parent_map)
    elif parents is not None or gear_parent is not None:
        if parents is None or gear_parent is None:
            raise ValueError('Historical reproduction requires both parents and gear_parent')
        selected = {}
        for task_id in TASK_IDS:
            root = Path(gear_parent) if task_id == 'gear_train_assembly' else Path(parents)/task_id
            selected[task_id] = {'path': str(root.resolve()),
                                 'manifest_sha256': digest(root/'release_manifest.json')}
    else:
        selected = {}
        for task_id in TASK_IDS:
            definition = importlib.import_module(
                'scenes.workrooms.mechanical_assembly.tasks.'+task_id+'.definition').TASK
            if task_id == 'fastener_assembly':
                release = definition['design']['loose_bolt_candidate']['release']
            elif task_id == 'gear_train_assembly':
                release = definition['design']['supply_layout_candidate']['release']
            else:
                release = definition['source_release']
            selected[task_id] = {key: release[key] for key in ('path', 'manifest_sha256')}
    if set(selected) != set(TASK_IDS):
        raise ValueError('Presentation requires an explicit parent for all five tasks')
    for task_id, row in selected.items():
        root = Path(row['path']).resolve()
        checked(root/'release_manifest.json', row['manifest_sha256'])
        manifest = validate(root)
        if manifest['tasks'][0]['id'] != task_id:
            raise ValueError('Presentation parent task identity differs: '+task_id)
        row['path'] = str(root)
    return selected


def presentation_inputs(scene, task, *, remove_pads=False):
    from .presentation_reset import remove_supply_pads
    from .table_distractors import add_distractors
    if task['task_id'] == 'gear_train_assembly':
        from .tasks.gear_train_assembly.task import reset_audit
        if not reset_audit(scene)['passed']:
            raise ValueError('L5 supply layout lacks the required tabletop and fixture clearance')
    if remove_pads:
        scene, task, pads = remove_supply_pads(scene, task)
    else:
        pads = {'removed_fixture_ids': [], 'scope': 'Frozen parent supply supports and reset preserved.'}
    scene, task, clutter = add_distractors(scene, task)
    return scene, task, {'supply_reset': pads, 'distractors': clutter,
                         'remove_supply_pads_requested': bool(remove_pads)}


def check_parent_physics(source_scene, source_task, scene, task, report):
    expected_scene, expected_task, expected_report = presentation_inputs(
        source_scene, source_task, remove_pads=report.get('remove_supply_pads_requested', False))
    stripped = deepcopy(scene)
    stripped.pop('workroom_presentation', None)
    for row in stripped['objects']+stripped['fixtures']:
        row.pop('presentation', None)
    if stripped != expected_scene or task != expected_task:
        raise ValueError('Presentation changes parent physics, task rules or undeclared physical inputs')
    if any(report.get(key) != value for key, value in expected_report.items()):
        raise ValueError('Presentation physical revision receipt differs from the actual transformation')


def revise_presentation(parent, output, scene, task, report):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    if output.exists() or not output.is_relative_to(OUTPUT_ROOT):
        raise ValueError('Choose a fresh presentation release under shared datasets')
    original = validate(parent)
    source_scene = load_json(parent/'scene.json')
    source_task = load_json(parent/'task.json')
    if (scene['workroom'] != source_scene['workroom'] or scene['robot'] != source_scene['robot']
            or scene['isaacsim'] != source_scene['isaacsim']
            or scene['physics'] != source_scene['physics']
            or task['public_rules'] != source_task['public_rules']
            or task['id'] != source_task['id']):
        raise ValueError('Presentation must preserve parent room, robot, physics settings and task rules')
    check_parent_physics(source_scene, source_task, scene, task, report)
    from .release import asset_inventory
    if task['id'] == 'fastener_assembly':
        from .tasks.fastener_assembly.release import _asset_inventory
        assets = _asset_inventory(scene)
    else:
        assets = asset_inventory(scene)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.'+output.name+'-', dir=output.parent))
    try:
        for name, row in original['files'].items():
            target = temporary/name
            target.parent.mkdir(parents=True, exist_ok=True)
            # Frozen parents may be read-only. New release working copies must
            # be writable without ever chmod'ing or editing the parent files.
            shutil.copyfile(parent/name, target)
            checked(target, row['sha256'])
        dump(temporary/'scene.json', scene)
        dump(temporary/'task.json', task)
        dump(temporary/'asset_file_hashes.json', assets)
        dump(temporary/'presentation_audit.json', report)
        provenance = load_json(temporary/'source_provenance.json')
        provenance['presentation_revision'] = {
            'parent_release': str(parent), 'parent_manifest_sha256': digest(parent/'release_manifest.json'),
            'visual_layers': 'Runtime-loaded USD overrides, checked to preserve composed physics.',
            'physical_scene_changes': {key: report[key] for key in (
                'supply_reset', 'distractors', 'remove_supply_pads_requested')},
            'native_manipulation_qualification_claimed': False,
            'original_assets_and_releases_modified': False}
        dump(temporary/'source_provenance.json', provenance)
        files = {str(p.relative_to(temporary)): {'sha256': digest(p)}
                 for p in sorted(temporary.rglob('*')) if p.is_file()}
        result = deepcopy(original)
        result.update(release_id=task['id']+'_'+output.parent.parent.name,
            presentation_contract_version=2,
            prepared_at_utc=datetime.now(timezone.utc).isoformat(), files=files,
            files_sha256={p:r['sha256'] for p,r in files.items()},
            asset_files_sha256={p:r['sha256'] for p,r in assets.items()},
            presentation_parent_release={'path':str(parent),'manifest_sha256':digest(parent/'release_manifest.json')},
            qualification_scope='New material/detail and explicitly declared physical distractor/reset candidate; no inherited native manipulation qualification.')
        for name in ('scene','task','initial_state'):
            result['tasks'][0][name+'_sha256'] = files[name+'.json']['sha256']
        dump(temporary/'release_manifest.json', result)
        validate(temporary)
        for name, row in original['files'].items():
            checked(parent/name, row['sha256'])
        if output.exists():
            raise FileExistsError(output)
        temporary.rename(output)
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def author_presentation(stage, root_path, output, *, role='preserve', tool=None, style_root=None):
    """Capture shading into a standalone stronger layer; no base layer edits."""
    from pxr import Sdf, Usd
    from bench.simulation.isaacsim_backend.presentation import physical_fingerprint, apply_presentation, presentation_stage
    from .presentation_materials import build_materials, style_visible_meshes, required_texture_files
    from .visual_tools import author_tool_details
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    before = physical_fingerprint(stage, root_path)
    scratch = presentation_stage(stage)
    materials = build_materials(scratch, root_path+'/PresentationLooks')
    style = style_visible_meshes(scratch, style_root or root_path, role, materials)
    details = author_tool_details(scratch, root_path, tool, materials=materials) if tool else None
    if physical_fingerprint(scratch, root_path) != before:
        raise ValueError('Visual authoring changed base physical inputs')
    output.parent.mkdir(parents=True, exist_ok=True)
    layer = Sdf.Layer.CreateNew(str(output))
    Sdf.CopySpec(scratch.GetRootLayer(), root_path, layer, '/Asset')
    layer.Save()
    descriptor = {'schema':'asset_presentation.v1', 'usd_path':str(output), 'sha256':digest(output),
        'prim_path':'/Asset', 'dependencies_sha256':{str(p):digest(p) for p in required_texture_files()}}
    # Validate the exact runtime path, including target-path relocation.
    applied = apply_presentation(stage, root_path, {'presentation':descriptor})
    return descriptor, {'material':style,'tool_details':details, 'runtime_reload':applied,
        'physical_inputs_unchanged': physical_fingerprint(stage, root_path)==before}


def checked_texture_dependencies(source):
    from .presentation_materials import required_texture_files
    dependencies = source.get('dependencies_sha256')
    required = {str(path) for path in required_texture_files()}
    if not isinstance(dependencies, dict) or set(dependencies) != required:
        raise ValueError('Presentation source lacks the original texture dependency hashes')
    for path, expected in dependencies.items():
        checked(path, expected)
    return dict(dependencies)


def prepare_from_presentations(asset_root, parents=None, gear_parent=None, output=None, *, parent_map=None):
    """Freeze releases only from source-pinned authoring receipts."""
    if output is None:
        raise ValueError('A fresh output directory is required')
    asset_root, output = Path(asset_root).resolve(), Path(output).resolve()
    selected = parent_releases(parents=parents, gear_parent=gear_parent, parent_map=parent_map)
    source = load_json(asset_root/'SOURCE.json')
    if (source.get('schema') != 'mechanical_presentation_assets.v2'
            or set(source['tasks']) != set(TASK_IDS) or source.get('parent_releases') != selected
            or type(source.get('remove_supply_pads_requested')) is not bool):
        raise ValueError('A complete presentation inventory bound to the selected parents is required')
    for path, expected in source['files_sha256'].items():
        checked(path, expected)
    dependencies = checked_texture_dependencies(source)
    dependencies.update({str(asset_root/name): digest(asset_root/name) for name in ('SOURCE.json', 'LICENSE.txt')})

    def descriptor(path):
        return {'schema': 'asset_presentation.v1', 'usd_path': str(path),
                'sha256': source['files_sha256'][str(path)], 'prim_path': '/Asset',
                'dependencies_sha256': dict(dependencies)}

    releases = {}
    for task_id in TASK_IDS:
        parent = Path(selected[task_id]['path'])
        scene, task = (load_json(parent/(kind+'.json')) for kind in ('scene', 'task'))
        scene, task, composition = presentation_inputs(
            scene, task, remove_pads=source['remove_supply_pads_requested'])
        receipt = source['tasks'][task_id]
        if any(receipt.get(key) != value for key, value in composition.items()):
            raise ValueError('Current layout differs from the native-authored presentation inputs')
        scene['workroom_presentation'] = descriptor(asset_root/'worktop.usda')
        for kind in ('objects', 'fixtures'):
            for row in scene[kind]:
                if row['id'] == 'robot_stand':
                    continue
                proof = receipt[kind][row['id']]
                path = asset_root/task_id/(row['id']+'.usda')
                if (proof.get('physical_inputs_unchanged') is not True
                        or proof['runtime_reload'].get('physical_inputs_unchanged') is not True
                        or proof['runtime_reload']['sha256'] != source['files_sha256'][str(path)]):
                    raise ValueError('Missing native composition preservation receipt')
                row['presentation'] = descriptor(path)
        release = output/task_id
        revise_presentation(parent, release, scene, task, receipt)
        releases[task_id] = {'path': str(release), 'manifest_sha256': digest(release/'release_manifest.json')}
    return releases
