"""The SDK process must not retain USB ownership after abrupt native Kit exit."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


@pytest.mark.skipif(sys.platform!='linux',reason='Linux parent-death signal')
def test_sdk_worker_exits_when_parent_is_killed():
    code='''
import os,subprocess,sys
child=subprocess.Popen([sys.executable,'-B','-c',
    'from uni_teleop.devices.manus.sdk_process import bind_parent; '
    f'bind_parent({os.getpid()}); '
    'print("bound",flush=True); import time; time.sleep(30)'],stdout=subprocess.PIPE,text=True)
print(child.pid,flush=True)
print(child.stdout.readline().strip(),flush=True)
sys.stdin.read(1)
'''
    parent=subprocess.Popen([sys.executable,'-B','-c',code],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    child=None
    try:
        child=int(parent.stdout.readline())
        assert parent.stdout.readline().strip()=='bound'
        parent.kill();parent.wait(timeout=3)
        status=Path(f'/proc/{child}/stat')
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            if not status.exists() or status.read_text().split(') ')[1].startswith('Z'):
                break
            time.sleep(.01)
        else:pytest.fail('Orphan SDK worker survived parent exit')
    finally:
        if parent.poll() is None:parent.kill();parent.wait(timeout=3)
        if child:
            try:os.kill(child,signal.SIGKILL)
            except ProcessLookupError:pass
