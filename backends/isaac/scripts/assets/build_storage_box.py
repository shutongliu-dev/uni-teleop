"""Author a new immutable storage-box object asset from frozen source."""
import argparse
import json
from pathlib import Path
from pxr import Usd, UsdGeom
from bench.isaacsim.release import digest, dump
from bench.paths import DATA_ROOT
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scenes.workrooms.living_room.tasks.storage_box.assets import author_task
from scenes.workrooms.living_room.task_scene import physical_inventory

PROJECT=Path(__file__).resolve().parents[2]


def build(output):
    output=Path(output).resolve()
    if output.exists() or not output.is_relative_to(f'{DATA_ROOT}/assets/task_parts/living_room/storage_box'):
        raise ValueError('Choose a new version under the shared storage_box asset directory')
    source=verify_if_snapshot(PROJECT,required_modules=('scenes.workrooms.living_room.tasks.storage_box.assets',),entrypoint_file=Path(__file__))
    if source is None: raise ValueError('Build from a complete frozen source')
    stage=Usd.Stage.CreateInMemory();UsdGeom.Xform.Define(stage,'/World');stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    metadata=author_task(stage)
    inventory=physical_inventory(stage)
    output.mkdir(parents=True,exist_ok=False)
    stage.Export(str(output/'initial.usda'));dump(output/'metadata.json',metadata)
    manifest={'schema':'living_task_objects_candidate.v1','task_id':'storage_box','source_root':str(PROJECT),
        'source_snapshot':source,'source_snapshot_manifest_sha256':digest(PROJECT/'source_snapshot_manifest.json'),
        'initial_usd':str(output/'initial.usda'),'initial_usd_sha256':digest(output/'initial.usda'),
        'metadata_sha256':digest(output/'metadata.json'),'inventory':inventory,'dependencies':{},
        'license':'Project-authored procedural geometry; no external mesh or texture inputs.',
        'native_manipulation_qualified':False,'release_status':'scene_candidate_not_bench_release'}
    verify_if_snapshot(PROJECT,required_modules=('scenes.workrooms.living_room.tasks.storage_box.assets',),entrypoint_file=Path(__file__))
    dump(output/'manifest.json',manifest)
    return {'asset':str(output),'manifest_sha256':digest(output/'manifest.json'),
            'initial_usd_sha256':manifest['initial_usd_sha256']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(json.dumps(build(parser.parse_args().output)))
