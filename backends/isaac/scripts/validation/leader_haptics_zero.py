"""Explicit hardware zero-current enable/watchdog/off check; no motion targets.

Support the leader by hand. Only current-mode, healthy, initially torque-off
XC330-T288/XM430-W210 motors1..7 are accepted; motor8 is untouched.
"""
import argparse
import json
from pathlib import Path
import struct
import time

from uni_teleop.feedback.factr.haptics import ARM_IDS,HapticDriver


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--port',required=True,type=Path)
    cli.add_argument('--output',required=True,type=Path)
    cli.add_argument('--execute-zero-current-check',action='store_true')
    args=cli.parse_args()
    if not args.execute_zero_current_check or args.port.parent!=Path('/dev/serial/by-id'):
        cli.error('Require explicit --execute-zero-current-check and stable serial identity')
    if args.output.exists():cli.error('Use a new output path; preserve previous hardware checks')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    driver=HapticDriver(str(args.port),baud=4000000)
    attempted=False
    result={'kind':'physical_zero_current_check','nonzero_current_requested':False,
            'motor_ids':list(ARM_IDS),'passed':False}
    try:
        result['models']=driver.inspect()
        attempted=True
        driver.enable()
        result['zero_goal_readback']={i:struct.unpack('<h',driver.read(i,102,2))[0] for i in ARM_IDS}
        time.sleep(.16)  # Intentionally no serial traffic; hardware watchdog must trip.
        result['watchdog_after_silence']={i:struct.unpack('<b',driver.read(i,98,1))[0] for i in ARM_IDS}
        if any(result['zero_goal_readback'].values()):raise RuntimeError('Goal current not zero')
        if any(v!=-1 for v in result['watchdog_after_silence'].values()):
            raise RuntimeError('Not all hardware watchdogs tripped at zero current')
        result['passed']=True
    except BaseException as error:
        result['error']=str(error)
        raise
    finally:
        try:
            if attempted:
                driver.disable()
                result['torque_enable_final']={i:driver.read(i,64,1)[0] for i in ARM_IDS}
                result['watchdog_final']={i:struct.unpack('<b',driver.read(i,98,1))[0] for i in ARM_IDS}
                result['passed'] &= not any(result['torque_enable_final'].values()) and not any(result['watchdog_final'].values())
        except BaseException as error:
            result['passed']=False
            result['shutdown_error']=str(error)
            raise
        finally:
            driver.close()
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result),flush=True)
    if not result['passed']:raise RuntimeError('Hardware final torque-off not confirmed')


if __name__=='__main__':main()
