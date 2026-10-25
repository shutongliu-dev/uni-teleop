"""Real frozen task contracts through /tasks; CPU HTTP, no simulator calls."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from bench.paths import DATA_ROOT, load_json
from bench.runtime.server import create_app

RELEASES=Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/simple_manipulation')
PRIVATE_MARKER='private_goal_test_canary_never_public'


def public_inventory(release):
    manager=SimpleNamespace(release=release,shutdown=lambda:None)
    with TestClient(create_app(service=manager)) as client:
        response=client.get('/tasks')
    assert response.status_code==200,response.text
    return response.json()


@pytest.mark.parametrize('version',['v4_humanlike','v5_curated'])
def test_real_four_tasks_publish_complete_declared_humanlike_constraints(version):
    release=RELEASES/version
    manifest=load_json(release/'release_manifest.json')
    tasks={entry['id']:load_json(release/entry['task']) for entry in manifest['tasks']}
    rows=public_inventory(release)
    assert len(rows)==len(tasks)==4
    for row in rows:
        task=tasks[row['task_id']]
        assert set(row)=={'task_id','instruction','goal'}
        assert row['goal']==task['public_goal']
        constraints=row['goal']['grasp_constraints']
        assert constraints['criteria']==task['public_goal']['grasp_constraints']['criteria']
        assert len(constraints['criteria'])==17
        assert constraints['humanlike_grasp_required']==(row['task_id']!='01_cube_push')
        assert 'evaluation' not in row and 'recommended_budgets' not in row


def test_real_legacy_release_without_grasp_contract_keeps_original_public_goal():
    release=RELEASES/'v3'
    manifest=load_json(release/'release_manifest.json')
    tasks={entry['id']:load_json(release/entry['task']) for entry in manifest['tasks']}
    rows=public_inventory(release)
    assert len(rows)==3
    for row in rows:
        assert row['goal']==tasks[row['task_id']]['public_goal']
        assert 'grasp_constraints' not in row['goal']


@pytest.mark.parametrize('task_id',['01_cube_push','02_cube_lift_hold','03_cube_pick_place','04_cube_grasp_hold'])
def test_tasks_projection_never_exposes_unknown_or_private_fields(tmp_path,monkeypatch,task_id):
    task=load_json(RELEASES/'v5_curated/tasks'/task_id/'task.json')
    expected=deepcopy(task['public_goal'])
    task['private_solver_state']={'objects':PRIVATE_MARKER}
    task['evaluation']['private_live_score']=PRIVATE_MARKER
    task['public_goal']['undeclared_oracle_pose']=PRIVATE_MARKER
    task['public_goal']['grasp_constraints']['unknown_private_field']=PRIVATE_MARKER
    task['public_goal']['grasp_constraints']['criteria']['private_contact_state']=PRIVATE_MARKER
    (tmp_path/'task.json').write_text(json.dumps(task))
    from bench.isaacsim import release
    monkeypatch.setattr(release,'validate',lambda root,assets=False:{'tasks':[{'id':task_id,'task':'task.json'}]})
    rows=public_inventory(tmp_path)
    assert rows==[{'task_id':task_id,'instruction':task['title_zh']+'。'+'；'.join(
        step['instruction_zh'] for step in task['steps']),'goal':expected}]
    assert PRIVATE_MARKER not in json.dumps(rows)
