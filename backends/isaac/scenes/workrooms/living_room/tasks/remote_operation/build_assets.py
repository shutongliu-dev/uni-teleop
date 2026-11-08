"""Build a fresh shared remote asset version without starting the simulator."""
import argparse
import json
from pathlib import Path

from pxr import Usd, UsdGeom
from bench.isaacsim.release import digest, dump
from bench.paths import DATA_ROOT
from scenes.workrooms.living_room.task_scene import physical_inventory
from .assets import author_task


def build(output):
    output = Path(output).resolve()
    if not output.is_relative_to(f'{DATA_ROOT}/assets/task_parts/living_room/remote_operation'):
        raise ValueError('Remote assets belong under the versioned shared task directory')
    output.mkdir(parents=True, exist_ok=False)
    path = output/'initial.usda'
    stage = Usd.Stage.CreateNew(str(path))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, '/World').GetPrim())
    metadata = author_task(stage)
    inventory = physical_inventory(stage)
    stage.GetRootLayer().Save()
    dump(output/'metadata.json', metadata)
    manifest = {'schema': 'living_task_objects_candidate.v1', 'task_id': 'remote_operation',
        'initial_usd': str(path), 'initial_usd_sha256': digest(path),
        'metadata_sha256': digest(output/'metadata.json'), 'inventory': inventory,
        'dependencies': {}, 'native_manipulation_qualified': False,
        'release_status': 'scene_candidate_not_bench_release',
        'author_sources_sha256': {str(Path(__file__).with_name(name)): digest(Path(__file__).with_name(name))
                                 for name in ('assets.py', 'build_assets.py')}}
    dump(output/'manifest.json', manifest)
    return {'path': str(output), 'manifest_sha256': digest(output/'manifest.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    print(json.dumps(build(parser.parse_args().output)))
