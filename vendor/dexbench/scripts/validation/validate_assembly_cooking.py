"""Frozen native convex cooking/parameter probe with no post-reset rollout."""
import argparse
import os
from pathlib import Path
import traceback
import numpy as np

from bench.isaacsim.release import dump, digest
from bench.paths import load_json
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.mechanical_convex_geometry import check_convex_geometry
from scripts.validation.validate_assembly_contacts import native_parameters
from scripts.validation.validate_workrooms import stage_inventory, plain
from scripts.validation.validate_scenario import seal_report_before_shutdown


def install_diagnostic_mesh_conditioning(scene_spec, factor):
    """Change only task collider coordinates during initial stage authoring.

    This is an explicit experimental wrapper, not a shared backend default.
    Points and inverse prim scale leave parent-space source vertices unchanged.
    """
    from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
    from bench.simulation.isaacsim_backend import scene as author
    if type(factor) not in (float,int) or not np.isfinite(factor) or factor<1 or factor>1000:
        raise ValueError('Conditioning factor must be finite and in [1,1000]')
    paths=set()
    for group,root in [('objects','Objects'),('fixtures','Fixtures')]:
        for item in scene_spec[group]:
            if item['id']=='robot_stand':continue
            asset=load_task_asset(item['asset_id'],item['asset_root'],verify_hashes=True)
            paths.update(f'/World/{root}/{item["id"]}/collision_{i:03d}' for i in range(len(asset['collision_parts'])))
    records={};installed=[]
    def attach(author):
        mesh,transform=author.mesh,author.transform
        def conditioned_mesh(stage,path,vertices,faces):
            if path in paths:
                source=np.asarray(vertices,dtype=np.float64)
                conditioned=(source*factor).astype(np.float32)
                records[path]={'factor':factor,'parent_space_vertex_max_roundtrip_error_m':float(np.abs(conditioned.astype(float)/factor-source).max())}
                return mesh(stage,path,conditioned,faces)
            return mesh(stage,path,vertices,faces)
        def conditioned_transform(prim,position=(0,0,0),quaternion=(1,0,0,0),scale=None):
            if str(prim.GetPath()) in records:
                if scale is not None:raise ValueError('Expected mesh collider with baked source scale')
                scale=[1./factor]*3
            return transform(prim,position,quaternion,scale)
        author.mesh,author.transform=conditioned_mesh,conditioned_transform
        installed.append((author,mesh,transform))
    attach(author)
    def restore():
        for author,mesh,transform in installed:author.mesh,author.transform=mesh,transform
    return records,restore,paths


def run(args):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2];env=None
    report={'schema':'native_assembly_cooking_probe.v1','passed':False,
        'model_calls':0,'task_success_claimed':False,'post_reset_rollout_steps':0,
        'object_state_writeback_after_reset':False,'external_object_forces':False,
        'scope':'Read native cooked convex surfaces, mass/inertia/COM/materials after task reset. No functional contact or stability qualification.'}
    try:
        imports=verify_if_snapshot(root,required_modules=('scripts.validation.validate_assembly_cooking',),entrypoint_file=Path(__file__))
        if imports is None:raise ValueError('Complete sealed source snapshot required')
        dump(output/'runtime_imports.json',imports)
        if (os.environ.get('ISAAC_BENCH_SINGLE_CUDA_DEVICE')!='1'
                or not os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-')):
            raise ValueError('Select one shared GPU by UUID')
        if digest(args.input)!=args.input_sha256:raise ValueError('Frozen cooking input changed')
        inputs=load_json(args.input);dump(output/'inputs.json',inputs)
        for path,expected in inputs['asset_files_sha256'].items():
            if digest(path)!=expected:raise ValueError('Asset changed: '+path)
        from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
        factor=inputs.get('diagnostic_mesh_coordinate_factor',1.)
        class CookingProbeEnv(FR3DexterousIsaacEnv):
            def _initialize(self):
                records,restore,selected=install_diagnostic_mesh_conditioning(self.scene_spec,factor)
                self.cooking_conditioning_records,self.cooking_selected=records,selected
                try:super()._initialize()
                finally:restore()
        env=CookingProbeEnv(inputs['scene'],device='cuda:0',render_width=640,render_height=480)
        records,selected=env.cooking_conditioning_records,env.cooking_selected
        report['mesh_coordinate_conditioning']={'factor':factor,'colliders':records,
            'all_selected_colliders_conditioned':set(records)==selected,
            'scope':'Diagnostic stage authoring only; source files, mass properties and all parent-space geometry preserved.'}
        env.reset(initial_state=inputs['initial_state'])
        step_before=int(env.world.current_time_step_index)
        report['checks']={'stage_inventory':stage_inventory(env),
            'native_convex_geometry':check_convex_geometry(env,include_vertices=True),
            'native_parameters':native_parameters(env)}
        report['post_reset_rollout_steps']=int(env.world.current_time_step_index)-step_before
        report['passed']=(all(r['passed'] for r in report['checks'].values()) and report['post_reset_rollout_steps']==0
            and bool(selected) and set(records)==selected
            and max(r['parent_space_vertex_max_roundtrip_error_m'] for r in records.values())<1e-7)
        dump(output/'metadata.json',env.metadata())
        dump(output/'runtime_imports_final.json',verify_if_snapshot(root,
            required_modules=('scripts.validation.validate_assembly_cooking',),entrypoint_file=Path(__file__)))
    except BaseException as exc:
        report.update(passed=False,error={'type':type(exc).__name__,'detail':str(exc),'traceback':traceback.format_exc()})
    report['artifacts_sha256']={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}
    return seal_report_before_shutdown(output,plain(report),env)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--input-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(0 if run(args)['passed'] else 1)


if __name__=='__main__':main()
