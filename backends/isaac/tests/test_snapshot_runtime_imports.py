"""New trust-boundary regressions, including actual fresh-process imports."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from bench.paths import load_json


TOOLS = Path(__file__).resolve().parents[1] / 'scripts' / 'maintenance'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


freezer = load('freezer_under_test', TOOLS / 'freeze_project_sources.py')
rebaser = load('rebaser_under_test', TOOLS / 'rebase_runtime_contract.py')


@pytest.fixture
def projects(tmp_path):
    old = tmp_path / 'editable'
    old.mkdir()
    for tree in freezer.SOURCE_TREES:
        (old / tree).mkdir()
    for relative in ('bench/adapters/__init__.py', 'bench/__init__.py', 'bench/simulation/__init__.py',
                     'scripts/__init__.py', 'scripts/helper.py', 'scripts/isaac_python.sh'):
        path = old / relative
        path.parent.mkdir(exist_ok=True)
        path.write_text('# fixture source\n')
    (old / 'scripts/isaac_python.sh').chmod(0o755)
    (old / 'bench/origin.py').write_text("ORIGIN='fixed_source'\n")
    new = tmp_path / 'snapshot'
    freezer.freeze(old, new, ('scripts/isaac_python.sh',))
    return old, new


@pytest.mark.parametrize('change', ['wrong_root', 'wrong_suite_inventory', 'external_manifest'])
def test_manifest_cannot_redirect_launch_or_weaken_suite_inventory(projects, change):
    old, new = projects
    path = new / 'source_snapshot_manifest.json'
    manifest = load_json(path)
    if change == 'wrong_root':
        manifest['snapshot_root'] = str(old)
    elif change == 'wrong_suite_inventory':
        manifest['suite_source_pins'] = {}
    path.write_text(json.dumps(manifest))
    if change == 'external_manifest':
        external = new.parent / 'external_manifest.json'
        path.replace(external)
        path.symlink_to(external)
    with pytest.raises(ValueError):
        freezer.verify(new)


def test_moving_snapshot_requires_new_bound_provenance(projects):
    _, new = projects
    copied = new.parent / 'copied'
    shutil.copytree(new, copied)
    with pytest.raises(ValueError, match='snapshot_root'):
        freezer.verify(copied)
    assert freezer.verify(new)['verified'] is True


@pytest.mark.parametrize('reverse', [False, True])
def test_rebase_refuses_nested_project_boundaries(projects, reverse):
    old, _ = projects
    nested = old / 'nested'
    (nested / 'bench').mkdir(parents=True)
    shutil.copy2(old / 'bench/origin.py', nested / 'bench/origin.py')
    inventory = {'bench/origin.py': freezer.digest(old / 'bench/origin.py')}
    contract = {'runtime_source_inventory': inventory, 'runtime_code_sha256': rebaser.canonical_hash(inventory)}
    with pytest.raises(ValueError, match='disjoint'):
        rebaser.rebase_runtime_contract(contract, *( (nested, old) if reverse else (old, nested)))


PROBE = '''
import importlib.util, json, os, pathlib, sys
old,new,tool,case=sys.argv[1:]
spec=importlib.util.spec_from_file_location('runtime_check',tool)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
if case=='stale_import':
    sys.path.insert(0,old)
    import bench.origin
    sys.path.remove(old)
    os.chdir(new)
elif case=='stale_script':
    sys.path.insert(0,old)
    import scripts.helper
    sys.path.remove(old)
    import bench.origin
elif case=='foreign_namespace':
    import bench.origin
    bench.__path__.append(str(pathlib.Path(old)/'bench'))
elif case!='missing_required':
    import bench.origin
try:
    value=module.verify_runtime_imports(pathlib.Path(new),required_modules=('bench.origin',))
    print(json.dumps({'accepted':True,'evidence':value}))
except ValueError as error:
    print(json.dumps({'accepted':False,'error':str(error),
        'actual_cwd':os.getcwd(),'origin':getattr(sys.modules.get('bench.origin'),'__file__',None)}))
'''


@pytest.mark.parametrize('case', ['clean', 'wrong_cwd', 'stale_import', 'stale_script',
                                 'foreign_namespace', 'missing_required'])
def test_checks_the_actual_process_and_loaded_modules(projects, case):
    old, new = projects
    cwd = old if case in ('wrong_cwd', 'stale_import') else new
    result = subprocess.run([sys.executable, '-B', '-c', PROBE, str(old), str(new),
                             str(TOOLS / 'freeze_project_sources.py'), case],
                            cwd=cwd, env={**os.environ, 'PYTHONPATH': str(new)},
                            text=True, capture_output=True, check=True)
    value = json.loads(result.stdout)
    assert value['accepted'] is (case == 'clean'), value
    if case == 'clean':
        evidence = value['evidence']
        assert evidence['cwd'] == str(new)
        assert evidence['pid'] != os.getpid()
        assert evidence['runtime_imports_verified'] is True
        assert evidence['imported_project_modules']['bench.origin']['path'] == str(new / 'bench/origin.py')
    elif case in ('wrong_cwd', 'stale_import'):
        assert value['origin'] == str(old / 'bench/origin.py')


def test_rebase_keeps_peer_project_boundary_and_frozen_bytes(projects):
    old, new = projects
    inventory = {'bench/origin.py': freezer.digest(old / 'bench/origin.py'),
                 str(old / 'bench/adapters/__init__.py'): freezer.digest(old / 'bench/adapters/__init__.py')}
    contract = {'runtime_source_inventory': inventory, 'runtime_code_sha256': rebaser.canonical_hash(inventory)}
    result = rebaser.rebase_runtime_contract(contract, old, new)
    assert result['path_rebase']['verifier_version'] == 'runtime_contract_path_rebase_verifier_v3'
    assert str(new / 'bench/adapters/__init__.py') in result['runtime_source_inventory']
    assert str(old / 'bench/adapters/__init__.py') not in result['runtime_source_inventory']


@pytest.fixture(scope='module')
def launch_sources(tmp_path_factory):
    """Freeze actual launch code; later probes replace only external effects."""
    base = tmp_path_factory.mktemp('launch_import_guards')
    editable, snapshot = base/'editable', base/'snapshot'
    project = TOOLS.parents[1]
    for tree in (*freezer.SOURCE_TREES, 'scripts'):
        for path in (project/tree).rglob('*.py'):
            if path.is_symlink():
                continue
            target = editable/path.relative_to(project)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    for relative in freezer.DEFAULT_EXTRAS:
        target = editable/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project/relative, target)
    freezer.freeze(editable, snapshot)
    return editable, snapshot


LAUNCH_PROBE = r'''
import argparse, json, pathlib, runpy, socket, sys, types
root, output, role = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
outcome = {'accepted': False}
try:
    if role == 'server':
        # No release/asset reads or native startup. Actual create_app still runs.
        import bench.runtime.controllers as controllers
        from bench.protocol.policies import PolicyRegistry
        controllers.default_registry = lambda release: PolicyRegistry()
        import bench.runtime.server as server
        outcome = server.app.state.source_snapshot_verification
    elif role == 'runner':
        import agent.model_client.runner as runner
        class Client:
            def __init__(self, *args, **kwargs): pass
            def close(self): pass
            def list_tasks(self):
                # This is the first HTTP boundary. Evidence must already exist.
                assert (output/'runtime_imports.json').is_file()
                (output.parent/(output.name+'_request_boundary_reached')).write_text('no network request sent')
                return [{'task_id': 'fixture', 'instruction': 'fixture'}]
            def create(self, request):
                raise runner.BenchHTTPError('fake setup failure; no network request sent')
        config = types.SimpleNamespace(public_metadata=lambda: {'base_url': 'http://invalid'})
        runner.ModelGatewayConfig.from_env = lambda: config
        runner.ModelClient = runner.BenchHTTPClient = Client
        args = argparse.Namespace(sim_budget=1., wall_budget=10., max_output_tokens=100,
            control_mode='joints', visual_input='rgb', max_decisions=1, credentials=None,
            model='fixture', model_timeout=1., reasoning_effort='max',
            bench_url='http://invalid', http_timeout=1., output=output,
            task='fixture', robot='fixture', seed=0)
        runner.run(args)
        outcome = json.loads((output/'runtime_imports.json').read_text())
        assert 'runtime_imports.json' in json.loads((output/'artifact_hashes.json').read_text())
        from agent.model_client import suite
        assert 'runtime_imports.json' in suite.setup_failure_pins(output)
        settings = json.loads((output/'agent_config.json').read_text())
        helper = 'scripts/maintenance/freeze_project_sources.py'
        assert settings['client_sources_sha256']['dependencies/'+helper] == suite.source_pins()[helper]
    elif role == 'worker':
        # Exercise the real -m main-thread worker path without opening IPC or Isaac.
        import bench.isaacsim.release as release
        import bench.simulation.isaacsim_backend.env as native
        import bench.simulation.isaacsim_backend.robot_registry as registry
        import bench.simulation.isaacsim_backend.scene_variants as variants
        release.task_inputs = lambda *args, **kwargs: (None, {}, {}, {})
        registry.scene_with_hand = lambda scene, hand: scene
        variants.apply_scene_variant = lambda scene, variant: scene
        class StopBeforePhysics:
            def __init__(self, *args, **kwargs):
                assert (output/'runtime_imports.json').is_file()
                (output/'env_boundary_reached').write_text('Isaac was not initialized')
                raise RuntimeError('probe stops before physics')
        native.FR3DexterousIsaacEnv = StopBeforePhysics
        class Socket:
            def __init__(self, *args): pass
            def bind(self, *args): pass
            def listen(self, *args): pass
            def accept(self): return self, None
            def close(self): pass
            def sendall(self, *args): pass
        socket.socket = Socket
        config_path = output.parent/(output.name+'_worker_config.json')
        config_path.write_text(json.dumps({'output': str(output), 'release': str(output/'unused'),
            'task_id': 'fixture', 'hand_id': 'fixture', 'device': 'cuda:0', 'seed': 0}))
        sys.argv = ['worker', '--config', str(config_path), '--socket', 'unused']
        try:
            runpy.run_module('bench.runtime.worker', run_name='__main__', alter_sys=True)
        except RuntimeError as error:
            if str(error) != 'probe stops before physics': raise
        outcome = json.loads((output/'runtime_imports.json').read_text())
    outcome['accepted'] = True
except ValueError as error:
    outcome = {'accepted': False, 'error': str(error)}
print(json.dumps(outcome))
'''


@pytest.mark.parametrize('role', ['server', 'runner', 'worker'])
@pytest.mark.parametrize('correct_cwd', [True, False])
def test_actual_launch_sites_check_before_network_or_physics(launch_sources, tmp_path, role, correct_cwd):
    _, snapshot = launch_sources
    output = tmp_path/'private' if role == 'worker' else tmp_path/'client'
    completed = subprocess.run([sys.executable, '-B', '-c', LAUNCH_PROBE,
                                str(snapshot), str(output), role],
        cwd=snapshot if correct_cwd else tmp_path,
        env={**os.environ, 'PYTHONPATH': str(snapshot)}, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    evidence = json.loads(completed.stdout.splitlines()[-1])
    assert evidence['accepted'] is correct_cwd, evidence
    if correct_cwd:
        assert evidence['runtime_imports_verified'] is True
        expected = 'agent/model_client/runner.py' if role == 'runner' else f'bench/runtime/{role}.py'
        assert evidence['entrypoint']['path'] == str(snapshot/expected)
        assert evidence['pid'] != os.getpid()
    else:
        assert not (output.parent/(output.name+'_request_boundary_reached')).exists()
        assert not (output/'env_boundary_reached').exists()
        assert not (output/'runtime_imports.json').exists()


GUARD_PROBE = r'''
import importlib.util, json, pathlib, sys
root, tool = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
spec = importlib.util.spec_from_file_location('guard', tool)
guard = importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
try:
    result = guard.verify_if_snapshot(root, required_modules=('bench.origin',),
        entrypoint_file=root/'bench/origin.py')
    print(json.dumps({'accepted': True, 'evidence': result}))
except (ValueError, FileNotFoundError) as error:
    print(json.dumps({'accepted': False, 'error': str(error)}))
'''


@pytest.mark.parametrize('case', ['editable', 'editable_from_frozen_cwd', 'manifest_symlink',
                                 'dangling_manifest_symlink'])
def test_optional_guard_cannot_skip_frozen_markers(projects, case):
    editable, snapshot = projects
    root, cwd = editable, editable
    if case == 'editable_from_frozen_cwd':
        cwd = snapshot
    elif 'symlink' in case:
        root = cwd = snapshot
        marker = snapshot/'source_snapshot_manifest.json'
        original = snapshot.parent/'external_manifest.json'
        marker.replace(original)
        marker.symlink_to(original)
        if case.startswith('dangling'):
            original.unlink()
    completed = subprocess.run([sys.executable, '-B', '-c', GUARD_PROBE,
        str(root), str(TOOLS/'freeze_project_sources.py')], cwd=cwd,
        env={**os.environ, 'PYTHONPATH': str(root)}, text=True, capture_output=True, check=True)
    result = json.loads(completed.stdout)
    assert result['accepted'] is (case == 'editable')
    if case == 'editable':
        assert result['evidence'] is None
