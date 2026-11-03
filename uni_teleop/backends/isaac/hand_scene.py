"""Bench task assets for standalone-hand practice; no robot or scoring import."""
import hashlib
from pathlib import Path


def parts_sorting(data):
    from bench.paths import load_json
    from scenes.workrooms.teleop_workbench import apply

    root = Path(data)/'datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/parts_sorting/desktop_v1'
    manifest = load_json(root/'release_manifest.json')
    # Consume the frozen rendering input directly: the standalone hand does
    # not implement the release's Franka reset, evaluator or qualification.
    scene_path = root/'scene.json'
    digest = hashlib.sha256(scene_path.read_bytes()).hexdigest()
    if digest != manifest['files_sha256']['scene.json']:
        raise ValueError('Bench scene hash does not match its release manifest')
    scene = load_json(scene_path)
    definition = dict(family='mechanical_assembly', task_id='parts_sorting',
                      source_release=dict(path=str(root)))
    scene = apply(scene, definition)
    receipt = dict(task='parts_sorting', scene_sha256=digest,
                   release_manifest_sha256=hashlib.sha256((root/'release_manifest.json').read_bytes()).hexdigest(),
                   task_scoring=False, embodiment='standalone_right_sharpa',
                   omitted=['Franka', 'robot_stand', 'desktop_dressing'],
                   physics_profile='standalone_hand_cpu_2ms')
    return scene, receipt


def mount(stage, scene):
    from bench.simulation.isaacsim_backend.workroom import add_workroom, apply_runtime_lighting
    from bench.simulation.isaacsim_backend.scene import add_task_asset

    add_workroom(stage, scene['workroom'], surface_materials=scene['isaacsim'].get('surface_materials'))
    apply_runtime_lighting(stage, scene['workroom'], scene['isaacsim'].get('workroom_lighting'))
    fixtures = [add_task_asset(stage, spec, dynamic=False)
                for spec in scene['fixtures'] if spec['id'] != 'robot_stand']
    objects = [add_task_asset(stage, spec, dynamic=True) for spec in scene['objects']]
    return dict(objects=objects, fixtures=fixtures)
