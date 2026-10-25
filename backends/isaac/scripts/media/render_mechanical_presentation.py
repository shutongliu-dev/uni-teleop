#!/usr/bin/env python3
"""Build versioned mechanical presentation releases and photograph native USD."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

from bench.paths import DATA_ROOT, load_json

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from bench.isaacsim.release import digest, dump, validate
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.media.render_workroom import now

TASKS = ('parts_sorting','peg_insertion','shaft_spacer_assembly','fastener_assembly','gear_train_assembly')
PARENTS = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_referee_20261018/rules_v1/releases')


def verify_source():
    return verify_if_snapshot(PROJECT, required_modules=('bench.isaacsim.release',),
                              entrypoint_file=Path(__file__))


def role_hint(task_id, row):
    name = row['id']
    if task_id=='parts_sorting' and name in ('block','cylinder','ring'):
        return {'block':'anodized_blue','cylinder':'anodized_yellow','ring':'anodized_red'}[name]
    if task_id=='shaft_spacer_assembly' and name in ('spacer','wide_ring','top_ring'):
        return {'spacer':'anodized_green','wide_ring':'steel_brushed','top_ring':'anodized_blue'}[name]
    if name in ('gear_workplate','socket_workplate','shaft_fixture') or name.startswith('tray_'):
        return {'*':'preserve','visual_000':'fixture_dark'}
    if name=='bolt':
        return 'steel_polished'
    return 'preserve'


def prepare(args, app, report):
    import omni.usd
    from bench.simulation.isaacsim_backend.scene import build_scene
    from bench.simulation.isaacsim_backend.presentation import apply_presentation
    from scenes.workrooms.mechanical_assembly.robot_layout import robot_asset_reference
    from scenes.workrooms.mechanical_assembly.presentation_release import (
        author_presentation, revise_presentation, parent_releases, presentation_inputs)
    from scenes.workrooms.mechanical_assembly.presentation_materials import required_texture_files

    selected_parents = parent_releases(parents=args.parents, gear_parent=args.gear_parent,
        parent_map=load_json(args.parent_map) if args.parent_map else None)
    args.assets_output.mkdir(parents=True, exist_ok=False)
    room_descriptor = None
    assets_report = {'schema':'mechanical_presentation_assets.v2','created_utc':now(),
        'source_root':str(PROJECT),'original_assets_modified':False,
        'parent_releases': selected_parents, 'remove_supply_pads_requested': args.remove_supply_pads,
        'dependencies_sha256': {str(path): digest(path) for path in required_texture_files()},
        'textures':'Existing versioned ambientCG CC0 scanned normal/roughness maps.', 'tasks':{}}
    pending=[]
    for task_id in TASKS:
        parent = Path(selected_parents[task_id]['path'])
        validate(parent)
        scene = load_json(parent/'scene.json')
        task = load_json(parent/'task.json')
        scene, task, composition_report = presentation_inputs(scene, task, remove_pads=args.remove_supply_pads)
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        build_scene(stage, scene, robot_asset_reference(scene['robot'])['root_usd'])
        task_report = {**composition_report, 'objects':{},'fixtures':{}}
        if room_descriptor is None:
            room_descriptor, room_report = author_presentation(stage, '/World/Room',
                args.assets_output/'worktop.usda', role='worktop_laminate',
                style_root='/World/Room/Geometry/worktop')
            assets_report['worktop'] = room_report
        else:
            apply_presentation(stage,'/World/Room',{'presentation':room_descriptor})
        scene['workroom_presentation'] = room_descriptor
        for kind in ('objects','fixtures'):
            for row in scene[kind]:
                if row['id']=='robot_stand':
                    continue
                root='/World/'+kind.title()+'/'+row['id']
                tool=row['id'] if row['id'] in ('screwdriver','straight_hex_screwdriver','socket_crank','open_end_wrench') else None
                descriptor, receipt = author_presentation(stage, root,
                    args.assets_output/task_id/(row['id']+'.usda'), role=role_hint(task_id,row), tool=tool)
                row['presentation']=descriptor
                task_report[kind][row['id']]=receipt
                print(json.dumps({'event':'styled','task':task_id,'object':row['id']}),flush=True)
        pending.append((task_id,parent,scene,task,task_report))
        assets_report['tasks'][task_id]=task_report
        dump(args.output/'provenance.json',report)
    assets_report['files_sha256']={str(p):digest(p) for p in sorted(args.assets_output.rglob('*.usda'))}
    from bench.isaacsim.release import checked
    for path, expected in assets_report['dependencies_sha256'].items():
        checked(path, expected)
    dump(args.assets_output/'SOURCE.json',assets_report)
    (args.assets_output/'LICENSE.txt').write_text('Original procedural presentation geometry: CC0-1.0.\nReferenced ambientCG maps retain their shared SOURCE.json provenance and CC0 license.\n')
    report['presentation_assets']={'path':str(args.assets_output),'source_sha256':digest(args.assets_output/'SOURCE.json')}
    provenance_files={str(args.assets_output/name):digest(args.assets_output/name) for name in ('SOURCE.json','LICENSE.txt')}
    for task_id,parent,scene,task,task_report in pending:
        for row in scene['objects']+scene['fixtures']:
            if 'presentation' in row:row['presentation']['dependencies_sha256'].update(provenance_files)
        scene['workroom_presentation']['dependencies_sha256'].update(provenance_files)
        release=args.output/'releases'/task_id
        manifest=revise_presentation(parent,release,scene,task,task_report)
        report['tasks'][task_id]={'release':str(release),'manifest_sha256':digest(release/'release_manifest.json'),
            'task_id':task_id,'status':manifest['status'],'catalog_status':'planned' if task_id=='gear_train_assembly' else 'implemented_candidate'}
        dump(args.output/'provenance.json',report)
        print(json.dumps({'event':'prepared','task':task_id,'release':str(release)}),flush=True)


def pose_robot(stage, scene, initial):
    import numpy as np
    from pxr import Gf, Usd, UsdGeom
    from scenes.workrooms.mechanical_assembly.robot_layout import link_world_transforms
    world=link_world_transforms(scene['robot'],initial)
    targets={str(p.GetPath()):np.asarray(world[p.GetName()])
             for p in Usd.PrimRange(stage.GetPrimAtPath('/World/Robot'))
             if p.GetName() in world and p.IsA(UsdGeom.Xform)}
    cache=UsdGeom.XformCache()
    for path in sorted(targets,key=lambda v:(v.count('/'),v)):
        prim=stage.GetPrimAtPath(path)
        parent=cache.GetLocalToWorldTransform(prim.GetParent())
        matrix=Gf.Matrix4d(targets[path].T.tolist())*parent.GetInverse()
        xf=UsdGeom.Xformable(prim);xf.ClearXformOpOrder()
        xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble,'previewPose').Set(matrix)
        cache.Clear()
    for path,expected in targets.items():
        np.testing.assert_allclose(np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))).T,
                                   expected,atol=1.e-7,rtol=0.)
    return len(targets)


def render(args, app, report):
    import omni.usd
    from pxr import UsdPhysics
    from bench.simulation.isaacsim_backend.scene import build_scene, transform
    from scenes.workrooms.mechanical_assembly.robot_layout import robot_asset_reference
    from scenes.workrooms.mechanical_assembly.tasks.gear_train_assembly.task import contact_probe_reset
    from scripts.media.render_living_task import capture
    releases = args.release_root or args.output/'releases'
    for task_id in args.tasks:
        release = releases/task_id
        validate(release)
        scene=load_json(release/'scene.json')
        initial=load_json(release/'initial_state.json')
        omni.usd.get_context().new_stage()
        stage=omni.usd.get_context().get_stage()
        scene_info=build_scene(stage,scene,robot_asset_reference(scene['robot'])['root_usd'])
        posed=pose_robot(stage,scene,initial)
        assert len(scene_info['robot_link_paths'])==43
        assert not any(p.IsA(UsdPhysics.Scene) for p in stage.Traverse())
        cameras={'task':deepcopy(scene['cameras']['task'])}
        if task_id=='fastener_assembly':
            cameras['closeup']=dict(eye=[.73,-.73,1.58],target=[.60,0,.88],focal_length_mm=34.)
            cameras['tool_detail']=dict(eye=[.65,-.35,1.22],target=[.38,0,.96],focal_length_mm=44.)
        elif task_id=='gear_train_assembly':
            cameras['closeup']=dict(eye=[.68,-.73,1.78],target=[.58,0,.81],focal_length_mm=36.)
            cameras['tool_detail']=dict(eye=[.92,-.58,1.30],target=[.68,-.10,.82],focal_length_mm=40.)
        else:
            cameras['closeup']=dict(eye=[.68,-.73,1.78],target=[.58,0,.81],focal_length_mm=36.)
        for camera in cameras.values():
            camera.update(horizontal_aperture_mm=36., vertical_aperture_mm=36.*args.height/args.width)
        out=args.output/'images'/task_id;out.mkdir(parents=True,exist_ok=False)
        record={'task_id':task_id,'release':str(release),'release_manifest_sha256':digest(release/'release_manifest.json'),
            'state':'canonical reset, static native RTX','posed_robot_links':posed,'physics_steps':0,
            'cameras':cameras,'presentation_loading':'Normal build_scene runtime loader; pinned asset overlays.',
            'loaded_presentations':{kind:{k:v.get('presentation') for k,v in scene_info[kind].items()}
                                    for kind in ('objects','fixtures')},
            'scope':'Actual scene materials/geometry at reset; no manipulation qualification.'}
        stage.GetRootLayer().Export(str(out/'preview.usda'))
        settings=SimpleNamespace(task=task_id,cameras=list(cameras),width=args.width,height=args.height,
                                  samples=args.samples,render_updates=160)
        capture(stage,app,cameras,out,settings,record)
        if task_id=='gear_train_assembly':
            assembled=out/'assembled';assembled.mkdir()
            reset=contact_probe_reset()
            for name, row in reset['object_overrides'].items():
                transform(stage.GetPrimAtPath('/World/Objects/'+name),row['position_m'],row['quaternion_wxyz'])
            ac={'assembled':dict(eye=[.71,-.60,1.48],target=[.40,0,.84],focal_length_mm=44.,
                horizontal_aperture_mm=36., vertical_aperture_mm=36.*args.height/args.width)}
            ar={'task_id':task_id,'physics_steps':0,'state':'Explicit privileged assembled presentation pose',
                'scope':'Shows 20/32/20 layout and tool clearance; static display, not a task completion.'}
            capture(stage,app,ac,assembled,SimpleNamespace(task=task_id,cameras=['assembled'],
                width=args.width,height=args.height,samples=args.samples,render_updates=160),ar)
            record['assembled_preview']=ar
        record['status']='rendered';dump(out/'provenance.json',record)
        report['tasks'][task_id]={**report['tasks'].get(task_id,{}),'render':record}
        dump(args.output/'provenance.json',report)
        print(json.dumps({'event':'task_rendered','task':task_id}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['prepare','render','both'],default='both')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--assets-output',type=Path)
    parser.add_argument('--parents',type=Path)
    parser.add_argument('--gear-parent',type=Path)
    parser.add_argument('--parent-map',type=Path)
    parser.add_argument('--remove-supply-pads',action='store_true')
    parser.add_argument('--release-root',type=Path)
    parser.add_argument('--tasks',nargs='+',choices=TASKS,default=list(TASKS))
    parser.add_argument('--width',type=int,default=2560);parser.add_argument('--height',type=int,default=1920)
    parser.add_argument('--samples',type=int,default=96)
    args=parser.parse_args()
    if args.mode!='render' and args.assets_output is None:
        parser.error('Preparation requires --assets-output')
    args.output.mkdir(parents=True,exist_ok=False)
    report={'schema':'mechanical_presentation_preview.v1','created_utc':now(),'status':'starting',
        'source_root':str(PROJECT),'source_snapshot':verify_source(),
        'driver_override':{'verifyDriverVersion_enabled':False,
            'explicit_runtime_argument':'--/rtx/verifyDriverVersion/enabled=false',
            'vendor_supported_driver_configuration_claimed':False},
        'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),'physics_steps':0,'tasks':{}}
    app=None;code=1
    try:
        from isaacsim import SimulationApp
        from bench.simulation.isaacsim_backend.render_sync import startup_arguments, verify_settings
        app=SimulationApp(dict(headless=True,width=args.width,height=args.height,multi_gpu=False,
            active_cuda_gpus=[0],max_gpu_count=1,physics_gpu=0,renderer='PathTracing',fast_shutdown=True,
            extra_args=startup_arguments()+['--/rtx/verifyDriverVersion/enabled=false']),
            experience=str(PROJECT/'bench/apps/workroom.headless.kit'))
        import carb
        report['render_settings']=verify_settings(carb.settings.get_settings())
        if args.mode in ('prepare','both'):prepare(args,app,report)
        if args.mode in ('render','both'):render(args,app,report)
        report.update(status='completed',completed_utc=now(),source_snapshot_after=verify_source());code=0
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        dump(args.output/'provenance.json',report)
        if app is not None:app.close(exit_code=code)
    return code


if __name__=='__main__':
    raise SystemExit(main())
