"""Enforce the bench/simulator dependency direction without importing Isaac."""
from pathlib import Path

import pytest

from scripts.maintenance.check_module_dependencies import check_dependencies, import_references


def test_project_bench_and_simulator_do_not_import_agent_or_interface():
    root = Path(__file__).resolve().parents[1]
    assert check_dependencies(root) == []


@pytest.mark.parametrize('statement', [
    'import agent.model_client as client',
    'from interface import manus',
    'def late_import():\n    from agent.model_client.tools import project_depth_point',
    'if TYPE_CHECKING:\n    from agent import Agent',
    'import importlib as loader\nloader.import_module("agent.model_client")',
    'from importlib import import_module as load\nload(name="interface.manus")',
    '__import__("agent.model_client")',
    'from importlib import import_module\nimport_module(".model_client", package="agent")',
])
@pytest.mark.parametrize('location', ['bench/sdk/query.py', 'bench/runtime/service.py', 'bench/simulation/backend.py'])
def test_boundary_rejects_direct_delayed_and_literal_dynamic_imports(tmp_path, statement, location):
    path = tmp_path / location
    path.parent.mkdir(parents=True)
    path.write_text(statement)
    violations = check_dependencies(tmp_path)
    assert violations
    assert all(value.startswith(location + ':') for value in violations)


def test_runtime_can_assemble_scenes_and_comments_are_not_imports(tmp_path):
    path = tmp_path / 'bench/runtime/service.py'
    path.parent.mkdir(parents=True)
    path.write_text('''
"""The Agent uses this module; import agent.model_client is documentation only."""
from scenes.workrooms.catalog import load_task
from .transport import send
# from interface import manus
''')
    assert check_dependencies(tmp_path) == []
    assert (4, 'bench.runtime.transport') in import_references(path.read_text(), package='bench.runtime')


def test_agent_can_depend_on_bench_without_reversing_the_boundary(tmp_path):
    path = tmp_path / 'agent/model_client/runner.py'
    path.parent.mkdir(parents=True)
    path.write_text('from bench.sdk.depth import decode_metric_depth_npy\nfrom agent import Agent')
    assert check_dependencies(tmp_path) == []
