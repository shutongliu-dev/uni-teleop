"""Native unscored chain timing; no hardware input or physical motor output.

Run through scripts/isaac_python.sh from a frozen source snapshot. Reports
substep coverage, detailed CPU profile, and state arrays for paired comparisons.
"""
import argparse
import cProfile
import json
from pathlib import Path
import pstats
import time

import numpy as np

from scripts.teleop.pico_isaac import load_inputs, parser
from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
from bench.adapters.pico.video import StereoView


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--task',default='nist_cable_routing')
    cli.add_argument('--profile',choices=['cpu_rigid_v1','scene'],default='cpu_rigid_v1')
    cli.add_argument('--steps',type=int,default=80)
    cli.add_argument('--cprofile',action='store_true')
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    options=parser().parse_args(['--family','mechanical_assembly','--task',args.task,
        '--workbench-only','--no-task-scoring','--physics-profile',args.profile,
        '--no-video','--rear-camera','--headless'])
    scene,initial,release=load_inputs(options)
    env=FR3DexterousIsaacEnv(scene,device='cuda:0',headless=True,render_width=640,render_height=480)
    obs=env.reset(seed=0,initial_state=initial)
    view=StereoView(env,'overview',width=640,height=480,robot_poses=obs['robot'],inset_camera='wrist',rear_camera=True)
    target=dict(obs['joint_positions_rad']); rows=[]; states=[]; contact_totals=[]
    profile=cProfile.Profile()
    for i in range(args.steps+20):
        action=dict(target);action['fr3_joint1']+=.02*np.sin(i*.08)
        if i==20 and args.cprofile:profile.enable()
        start=time.perf_counter();obs=env.step({'joint_positions_rad':action,'duration_s':.02});stepped=time.perf_counter()
        view.update(None,obs['robot']);view.render();rendered=time.perf_counter()
        assert obs['contacts']['sampling']['complete']
        assert obs['contacts']['sampling']['sampled_substeps']==40
        assert not obs['terminated']
        if i>=20:
            rows.append({'step_s':stepped-start,'render_s':rendered-stepped})
            states.append(env.capture_state())
            contact_totals.append({'count':len(obs['contacts']['rigid']),'impulse':obs['contacts']['rigid_impulse_magnitude_sum_ns']})
    if args.cprofile:
        profile.disable();profile.dump_stats(str(out/'profile.pstats'))
        with (out/'profile.txt').open('w') as f:pstats.Stats(profile,stream=f).sort_stats('cumulative').print_stats(55)
    np.savez_compressed(out/'states.npz',**{k:np.stack([s[k] for s in states]) for k in states[0]})
    result={'task':args.task,'profile':args.profile,'steps':args.steps,'coverage':40,
        'timings':rows,'contacts':contact_totals,'means':{k:float(np.mean([r[k] for r in rows])) for k in rows[0]},
        'scope':'Synthetic small robot motion and passive objects; not human grasp/task qualification'}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['means']),flush=True)
    env.close()


if __name__=='__main__':main()
