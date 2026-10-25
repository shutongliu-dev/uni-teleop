"""Two-second, single-axis gravity direction check while the operator supports the arm.

Uses the existing 30/60 mA caps. This does not validate hands-off support.
Always zeros and disables all seven arm motors after an enable attempt.
"""
import argparse
import json
from pathlib import Path
import struct
import time

import numpy as np

from bench.adapters.master_slave.calibration import ArmCalibration
from bench.adapters.master_slave.gravity import LeaderGravity
from bench.adapters.master_slave.haptics import ARM_IDS,CURRENT_MA,FeedbackFrame,GravityCoupling,HapticDriver


def check(driver,calibration,gravity,axis,result,*,clock=time):
    attempted=False
    try:
        models=driver.inspect()
        result['models']=models
        if any(driver.read(i,10,1)!=b'\0' for i in ARM_IDS):
            raise ValueError('Require normal motor drive direction')
        result['trigger_torque_before']=driver.read(8,64,1)[0]
        if result['trigger_torque_before']:
            raise ValueError('Trigger motor must be off')
        coupling=GravityCoupling(models,gravity)
        samples=[]
        for _ in range(30):
            raw,stamp=driver.positions()
            if not 0<=clock.monotonic_ns()-stamp<=50_000_000:
                raise ValueError('Stale leader input before enable')
            samples.append(raw)
            clock.sleep(.01)
        if np.max(np.ptp(samples,axis=0))>.02:
            raise ValueError('Leader must be still before enable')
        origin=np.asarray(raw)
        result['initial_encoder_rad']=origin.tolist()
        attempted=True  # An interrupted enable must still run shutdown.
        driver.enable()
        previous,previous_stamp=driver.positions()
        start=clock.monotonic_ns()
        while clock.monotonic_ns()-start<2_000_000_000:
            clock.sleep(.01)
            raw,stamp=driver.positions()
            dt=(stamp-previous_stamp)/1e9
            if not 0<dt<=.05 or not 0<=clock.monotonic_ns()-stamp<=50_000_000:
                raise ValueError('Leader test input/control interval expired')
            velocity=(np.asarray(raw)-previous)/dt
            if np.max(np.abs(velocity[:7]))>1. or np.max(np.abs(np.asarray(raw)-origin))>.15:
                result['stop_state']={'encoder_rad':list(raw),'velocity_rad_s':velocity.tolist(),
                    'displacement_rad':(np.asarray(raw)-origin).tolist(),'dt_s':dt}
                raise ValueError('Leader test motion bound; output stopped')
            frame=FeedbackFrame(stamp,calibration.leader_reference_rad,
                calibration.robot_reference_rad,calibration.signs,(0.,)*7)
            requested=coupling.command(frame,raw,velocity,sample_ns=stamp,now_ns=clock.monotonic_ns(),dt=dt)
            command=np.zeros(7)
            command[axis-1]=requested[axis-1]
            sent=driver.currents(command)
            actual=struct.unpack('<h',driver.read(axis,126,2))[0]*CURRENT_MA[models[axis]]
            goal=struct.unpack('<h',driver.read(axis,102,2))[0]*CURRENT_MA[models[axis]]
            result['samples'].append({'elapsed_s':(stamp-start)/1e9,'encoder_rad':list(raw),
                'command_ma':sent,'measured_axis_current_ma':actual,'goal_readback_ma':goal,
                'saturated':coupling.saturated})
            if abs(actual)>coupling.cap[axis-1]+20 or abs(goal-sent[axis-1])>1e-6:
                raise ValueError('Current readback outside test envelope')
            previous,previous_stamp=np.asarray(raw),stamp
        result['completed']=True
    except BaseException as error:
        result['error']=str(error)
        raise
    finally:
        if attempted:
            try:
                driver.disable()
                result['torque_enable_final']={i:driver.read(i,64,1)[0] for i in range(1,9)}
                result['goal_current_final']={i:struct.unpack('<h',driver.read(i,102,2))[0] for i in ARM_IDS}
                result['watchdog_final']={i:driver.read(i,98,1)[0] for i in ARM_IDS}
                result['shutdown_verified']=not any(v for name in
                    ('torque_enable_final','goal_current_final','watchdog_final') for v in result[name].values())
                if not result['shutdown_verified']:
                    raise RuntimeError('Final zero/off readback failed')
            except BaseException as error:
                result['shutdown_error']=str(error)
                raise


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--port',type=Path,required=True)
    cli.add_argument('--arm-calibration',type=Path,required=True)
    cli.add_argument('--leader-urdf',type=Path,required=True)
    cli.add_argument('--axis',type=int,choices=ARM_IDS,required=True)
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--execute-supported-test',action='store_true')
    args=cli.parse_args()
    cal=ArmCalibration.load(args.arm_calibration)
    if (not args.execute_supported_test or args.port.parent!=Path('/dev/serial/by-id') or
            cal.status!='measured' or cal.source_kind!='hardware' or cal.leader_id!=args.port.name):
        cli.error('Require explicit supported test and verified calibration for this stable port')
    gravity=LeaderGravity(args.leader_urdf)
    result={'kind':'supported_single_axis_gravity_check','axis':args.axis,'duration_s':2.,
            'calibration_sha256':cal.sha256,'leader_urdf_sha256':gravity.sha256,
            'completed':False,'shutdown_verified':False,'samples':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as receipt:
        driver=None
        try:
            driver=HapticDriver(str(args.port),baud=4000000,timeout=.05)
            check(driver,cal,gravity,args.axis,result)
        except BaseException as error:
            result.setdefault('error',str(error))
            raise
        finally:
            try:
                if driver is not None:driver.close()
            finally:
                receipt.write(json.dumps(result,indent=2)+'\n')
                print(json.dumps({k:v for k,v in result.items() if k!='samples'}),flush=True)


if __name__=='__main__':main()
