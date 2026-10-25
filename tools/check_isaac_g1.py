"""Synthetic native G1/SONIC interface check; no hardware or trained policy."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from uni_teleop.backends.isaac.g1 import IsaacG1Env
    from uni_teleop.mapping.sonic import SonicController
    from uni_teleop.core.robot import RobotCommand
    from uni_teleop.mapping.sonic.parameters import KP,KD,DEFAULT_ANGLES
    env=IsaacG1Env(output=args.output)
    try:
        state=env.get_robot_state()
        command=RobotCommand(DEFAULT_ANGLES,np.zeros(29),np.zeros(29),KP,KD,
            hand_joint_position=np.ones(40)*.1,neck_joint_position=np.zeros(2))
        env.step(command,steps=20)
        moved=env.get_robot_state()
        controller=SonicController(lambda obs:np.zeros(29))
        mapped=controller.act(moved,np.zeros(64))
        report={'source':'synthetic_zero_policy_no_hardware','joints':len(env.names),'physics_steps':20,
                'body_motion_max_rad':float(np.max(abs(moved.joint_position-state.joint_position))),
                'finite':bool(np.isfinite(moved.joint_position).all()),
                'sonic_observation_dim':len(controller.last_observation),'command_dim':len(mapped.joint_position)}
        report['passed']=report['finite'] and report['joints']==71 and report['body_motion_max_rad']>.001 and report['sonic_observation_dim']==994
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'check.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
        if not report['passed']: raise RuntimeError('Native G1 contract check failed')
    finally:env.close()


if __name__=='__main__':main()
