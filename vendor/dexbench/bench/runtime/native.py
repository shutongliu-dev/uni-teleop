"""One isolated Isaac application per episode; Kit stays on its main thread."""
from __future__ import annotations

import json
import os
import csv
import platform
import select
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import threading
import time

from .transport import send, receive

PROJECT = Path(__file__).resolve().parents[2]


class NativeStartupError(RuntimeError):
    """A worker explicitly declared that physics cannot start."""


class NativeRuntimeError(RuntimeError):
    """A current worker explicitly declared a fatal GPU failure after startup."""


def execution_environment(device):
    """Static configuration, not evidence of exclusive GPU scheduling."""
    model=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                if line.startswith('model name')),platform.machine())
    value={'cpu_model':model,'cpu_available_cores':len(os.sched_getaffinity(0)),
           'machine_architecture':platform.machine(),'kernel':platform.release(),
           'gpu_count_used':1,'resource_mode':os.environ.get('BENCH_RESOURCE_MODE','shared'),
           'resource_allocation_tag':os.environ.get('BENCH_RESOURCE_ALLOCATION_TAG','development_unreserved'),
           'resource_exclusivity_verified':False}
    try:
        # In single-device mode the CUDA index is relative to CUDA_VISIBLE_DEVICES;
        # query that exact device instead of the same physical index.
        single=os.environ.get('ISAAC_BENCH_SINGLE_CUDA_DEVICE')=='1'
        identity=os.environ.get('CUDA_VISIBLE_DEVICES','') if single else device.split(':')[-1]
        result=subprocess.run(['nvidia-smi','--id='+identity,
            '--query-gpu=name,memory.total,driver_version,uuid','--format=csv,noheader,nounits'],
            capture_output=True,text=True,check=True,timeout=10)
        name,memory,driver,uuid=next(csv.reader(result.stdout.splitlines()))
        value.update(gpu_model=name.strip(),gpu_memory_mb=int(memory.strip()),driver_version=driver.strip(),
                     gpu_uuid=uuid.strip(),gpu_selection='single_cuda_visible_device' if single else 'physical_index',
                     capture_status='recorded')
    except (OSError,ValueError,StopIteration,subprocess.SubprocessError) as exc:
        value.update(capture_status='unavailable',error_type=type(exc).__name__)
    return value


class NativeWorker:
    # Trusted operator subclasses may add read-only engineering instrumentation.
    # This is never selected by an episode request or a public action.
    worker_module = 'bench.runtime.worker'

    def __init__(self, *, output, task_id, hand_id, seed, release, device='cuda:0', scene_variant='original_v1', startup_timeout_s=240):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.temporary = Path(tempfile.mkdtemp(prefix='dex_rpc_'))
        self.address = self.temporary/'worker.sock'
        self.lock = threading.RLock()
        self.connection = None
        self.process = None
        self.closed = False
        self.broken = False
        self._fatal_log_offset = 0
        self._fatal_log_tail = b''
        self._native_failure = None
        self._fatal_exception = None
        config = dict(output=str(self.output),task_id=task_id,hand_id=hand_id,seed=seed,
                      release=str(release),device=device,scene_variant=scene_variant)
        path = self.output/'worker_config.json'
        path.write_text(json.dumps(config,indent=2)+'\n')
        self.log = (self.output/'worker.log').open('w')
        try:
            self.process = subprocess.Popen([str(PROJECT/'scripts/isaac_python.sh'), '-m',
                self.worker_module,'--config',str(path),'--socket',str(self.address)],
                cwd=PROJECT,stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
            deadline = None if startup_timeout_s is None else time.monotonic()+startup_timeout_s
            while not self.address.exists():
                self._check_startup(deadline)
                time.sleep(.05)
            self.connection = socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
            self.connection.settimeout(None if deadline is None else max(1,deadline-time.monotonic()))
            self.connection.connect(str(self.address))
            self.initial = self._checked(receive(self.connection,
                wait_for_data=lambda sock: self._wait_startup_data(sock, deadline)))
            self.execution_environment=execution_environment(device)
            (self.output/'execution_environment.json').write_text(json.dumps(self.execution_environment,indent=2)+'\n')
            self.connection.settimeout(None if startup_timeout_s is None else 120)
        except BaseException as error:
            cleanup_error = None
            try:
                self.close(force=True)
            except Exception as exc:
                cleanup_error = type(exc).__name__
                error.add_note('Startup cleanup failed: '+cleanup_error)
            self._record_failure(error, cleanup_error)
            raise

    def _record_failure(self, error=None, cleanup_error=None):
        if self._native_failure is None:
            return
        diagnostic = dict(self._native_failure)
        diagnostic['worker_returncode_after_close'] = None if not self.closed or self.process is None else self.process.returncode
        diagnostic['cleanup_completed'] = self.closed and self.process is not None and self.process.returncode is not None and cleanup_error is None
        diagnostic['cleanup_error_type'] = cleanup_error
        filename = 'startup_failure.json' if diagnostic['phase']=='startup' else 'runtime_failure.json'
        try:
            (self.output/filename).write_text(json.dumps(diagnostic, indent=2)+'\n')
        except OSError as exc:
            if error is not None:
                error.add_note(('Startup' if diagnostic['phase']=='startup' else 'Runtime')+' diagnostic write failed: '+type(exc).__name__)

    def _check_fatal(self, phase, operation=None, stage=None):
        # Drain exactly the current file-size boundary before accepting a ready
        # RPC reply. A large existing backlog must not hide an already-declared
        # fatal event; later appends are checked on the next fragment/request.
        with (self.output/'worker.log').open('rb') as log:
            end = os.fstat(log.fileno()).st_size
            log.seek(self._fatal_log_offset)
            while log.tell() < end:
                data = log.read(min(65536, end-log.tell()))
                if not data:
                    break
                self._fatal_log_offset = log.tell()
                text = self._fatal_log_tail + data
                self._fatal_log_tail = text[-4096:]
                for line in text.splitlines():
                    physx = (b'[Error] [omni.physx.plugin]' in line and
                             (b'PhysX Internal CUDA error. Simulation cannot continue!' in line or
                              b'PhysX ABORT error: PhysX cannot start GPU simulation' in line))
                    rtx = b'[Error] [omni.rtx] GPU crash is detected.' in line
                    if physx or rtx:
                        diagnostic = {'schema': 'native_failure.v2', 'phase': phase,
                            'stage': stage or ('awaiting_initial_rpc' if phase=='startup' else 'awaiting_rpc_response'),
                            'operation': operation, 'kind': 'physx_cuda_fatal' if physx else 'rtx_gpu_crash',
                            'worker_pid': self.process.pid, 'worker_returncode': self.process.poll(),
                            'log_file': 'worker.log', 'log_read_offset': self._fatal_log_offset,
                            'fatal_line': line.decode('utf-8', errors='replace')[:1000],
                            'worker_initialized': phase=='runtime',
                            'task_started': False if phase=='startup' else None,
                            'retry_attempted': False}
                        self._native_failure = diagnostic
                        error_type = NativeStartupError if phase=='startup' else NativeRuntimeError
                        self._fatal_exception = error_type('Isaac worker declared fatal GPU failure; inspect private '+phase+'_failure.json')
                        raise self._fatal_exception

    def _check_startup(self, deadline):
        self._check_fatal('startup')
        if self.process.poll() is not None:
            raise RuntimeError('Isaac worker exited during startup; inspect private worker.log')
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError('Isaac worker startup timed out')

    def _wait_startup_data(self, sock, deadline):
        # The transport calls this before every header/payload fragment, so a
        # partial initial response cannot bypass fatal-startup propagation.
        while True:
            self._check_startup(deadline)
            wait = .05 if deadline is None else max(0, min(.05, deadline-time.monotonic()))
            if select.select([sock], [], [], wait)[0]:
                return

    def _wait_runtime_data(self, sock, operation):
        # Match existing recv timeouts per fragment. None remains unbounded.
        timeout = sock.gettimeout()
        deadline = None if timeout is None else time.monotonic()+timeout
        while True:
            self._check_fatal('runtime', operation)
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError('Native worker RPC timed out')
            wait = .05 if deadline is None else max(0, min(.05, deadline-time.monotonic()))
            if select.select([sock], [], [], wait)[0]:
                return

    @staticmethod
    def _checked(value):
        if not value.get('ok'):
            raise RuntimeError('Native worker rejected operation: '+value.get('error_type','unknown'))
        return value

    def request(self, operation, **payload):
        with self.lock:
            if self.closed:
                raise RuntimeError('Native worker is closed')
            if self.broken:
                raise RuntimeError('Native worker transport failed')
            try:
                self._check_fatal('runtime', operation, stage='before_rpc_dispatch')
                send(self.connection, {'operation':operation,**payload})
                reply=receive(self.connection,
                    wait_for_data=lambda sock: self._wait_runtime_data(sock, operation))
            except (NativeRuntimeError,OSError,EOFError,ValueError,UnicodeError) as error:
                self.broken=True
                self._record_failure(error)
                raise
            return self._checked(reply)

    def set_deadline(self,remaining_s):
        """Set an RPC deadline; explicit None disables the socket timeout."""
        with self.lock:
            self.connection.settimeout(None if remaining_s is None else min(120,float(remaining_s)))

    def close(self, force=False):
        with self.lock:
            if self.closed:
                return
            try:
                if not force and not self.broken and self.connection is not None and self.process.poll() is None:
                    self.request('finish')
            finally:
                self.closed = True
                if self.connection is not None:
                    self.connection.close()
                if self.process is not None:
                    try:
                        # Closing a broken RPC lets the worker seal already
                        # executed states in its finally block. Killing it after
                        # one second used to lose this evidence during timeout.
                        self.process.wait(timeout=30 if self.broken else (1 if force else 10))
                    except subprocess.TimeoutExpired:
                        self.process.terminate()
                        try:
                            self.process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            self.process.kill()
                            self.process.wait(timeout=10)
                self.log.close()
                shutil.rmtree(self.temporary)
                self._record_failure(self._fatal_exception)
