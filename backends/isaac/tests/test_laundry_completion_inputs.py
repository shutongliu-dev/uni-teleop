"""Independent reset inputs and terminal budget receipts; CPU-only verification."""
from copy import deepcopy
import builtins
import io
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import numpy as np
import pytest

from bench.isaacsim.release import _frozen_evaluator_module, digest
from bench.paths import load_json
from bench.protocol import Budgets
from bench.protocol.task_referee import project_referee_state
from bench.runtime.server import create_app
from scenes.workrooms.catalog import task_definition
from scenes.workrooms.laundry_room import garment_folding_assets as garments
from scenes.workrooms.laundry_room.initial_state_assets import validate_source
from scenes.workrooms.laundry_room.robot_configuration_catalog import RELEASES
from scenes.workrooms.laundry_room.runtime import LaundryTaskRuntime
from test_agent_service import join_action, publish_targets, request
from test_workroom_episode_contracts import workroom_service

PAIRS = [(tid, cid) for tid, row in RELEASES.items() for cid in row['configurations']]
GARMENTS = [pair for pair in PAIRS if pair[0] in ('shirt_folding', 'trousers_folding')]


def frozen(tid, cid):
    option = RELEASES[tid]['configurations'][cid]
    root = Path(option['source_release']['path'])
    assert digest(root/'release_manifest.json') == option['source_release']['manifest_sha256']
    return root, load_json(root/'task.json')


def frame(task, step):
    specs = task['asset_metadata']['cloths']
    return {'schema': 'laundry_native_frame.v1', 'complete': True, 'native_step': step,
        'time_s': step*task['evaluation']['physics_dt_s'], 'control_mode': 'robot_joint_targets',
        'controlled_vertices': {s['id']: [] for s in specs},
        'physics': {'complete': True, 'valid': True, 'validity_scope': 'native_geometry',
            'gravity_enabled': True, 'hidden_constraints': 0, 'self_intersections': 0,
            'intercloth_intersections': 0, 'max_penetration_m': 0.},
        'cloths': {s['id']: {'enabled': True, 'vertices_m': np.asarray(s['initial_vertices_m']),
            'velocities_m_s': np.zeros_like(s['rest_vertices_m']), 'triangles': s['triangles']} for s in specs},
        'robot_interaction': {'schema': 'laundry_robot_interaction.v1', 'complete': True,
            'method': 'native_cooked_convex_geometry', 'hand_near_cloth': False, 'fully_separated': True,
            'hands': {side: {'wrist_position_m': [0., 1., 1.1], 'near_towel': False,
                            'near_any_cloth': False} for side in task['active_sides']}}}


def deadline_score(tid, cid, *, stale_success=False):
    root, task = frozen(tid, cid)
    evaluator = _frozen_evaluator_module(root).SequenceEvaluator(task)
    assert not evaluator.observe(frame(task, 0))['success']
    dt, deadline = evaluator.c['physics_dt_s'], evaluator.limits['max_time_s']
    assert deadline == 120.
    step = round(deadline/dt)
    # Isolate the actual unchanged boundary, without a costly synthetic 120 s rollout.
    evaluator.previous_step, evaluator.previous_time = step-1, deadline-dt
    evaluator.success = stale_success
    score = evaluator.observe(frame(task, step))
    return evaluator, task, score


@pytest.mark.parametrize('tid,cid', PAIRS)
def test_frozen_deadline_is_latched_incomplete_and_never_physical_failure(tid, cid):
    for stale_success in (False, True):
        evaluator, task, score = deadline_score(tid, cid, stale_success=stale_success)
        assert score['terminal'] and score['status'] == 'incomplete'
        assert score['reason'] == 'budget_exhausted' and score['budget_exhausted']
        assert not score['success'] and not score['physical_failure']
        assert score['referee']['state'] == 'ongoing' and not score['referee']['terminal']
        assert score['referee']['reason_codes'] == ['incomplete_budget']
        assert project_referee_state(score, task['public_rules']['referee']) == score['referee']
        assert evaluator.observe({'complete': False}) == score  # No later scoring or success resurrection.


@pytest.mark.parametrize('tid,cid', GARMENTS)
def test_garment_input_loader_needs_no_development_run_files(tid, cid, monkeypatch):
    root, task = frozen(tid, cid)
    scene = load_json(root/'scene.json')
    manifest = Path(scene['laundry_task_provenance']['asset_candidate']['manifest_path'])
    builtins_open, io_open = builtins.open, io.open
    def guard(opener):
        def opened(path, *args, **kwargs):
            if isinstance(path, (str, Path)):
                value = Path(path)
                if '/fr3_sharpa_bench/development/' in str(value) and value.suffix != '.py':
                    raise AssertionError('Runtime attempted to read development artifact: '+str(value))
            return opener(path, *args, **kwargs)
        return opened
    monkeypatch.setattr(builtins, 'open', guard(builtins_open))
    monkeypatch.setattr(io, 'open', guard(io_open))
    descriptor, metadata = garments.load_assets(tid, manifest)
    assert metadata == task['asset_metadata']
    assert metadata['reset_profile'] == 'published_settled_unfolded_v1'
    assert metadata['native_settled_source'] is None
    assert metadata['initial_state_input']['schema'] == 'laundry_settled_input.v1'
    assert not any('report' in Path(p).name or 'qualification' in Path(p).name
                   for p in descriptor['files_sha256'])
    assert not (manifest.parent/'cpu_geometry.json').exists()


@pytest.mark.parametrize('fault', ['mesh', 'material', 'physics', 'velocity', 'extra_outcome', 'stretched', 'nonfinite'])
def test_published_input_identity_and_physical_constraints_cannot_be_skipped(fault):
    _, task = frozen('shirt_folding', 'single_arm')
    m = task['asset_metadata']; cloth, = m['cloths']; source = deepcopy(m['initial_state_input'])
    q = np.asarray(cloth['initial_vertices_m']).copy()
    if fault == 'mesh': source['rest_geometry_sha256'] = '0'*64
    elif fault == 'material': source['material_prior']['mass_kg'] *= 2
    elif fault == 'physics': source['physics_options']['cloth_iterations'] *= 2
    elif fault == 'velocity': source['initial_velocity_m_s'][0] = .1
    elif fault == 'extra_outcome': source['passed'] = True
    elif fault == 'stretched': q[0, 0] += .2
    else: q[0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_source(source, task_id=task['id'], mesh_profile=m['mesh_profile'],
            rest=cloth['rest_vertices_m'], triangles=cloth['triangles'], material_prior=m['material_prior'],
            physics_options=m['physics_options'], centre=m['source_center_xy_m'],
            quarter_turns=m['source_yaw_quarter_turns'], yaw_degrees=m.get('source_yaw_degrees'), vertices=q)


def test_native_runtime_rejects_actions_after_terminal_incomplete():
    runtime = object.__new__(LaundryTaskRuntime)
    runtime.closed = False; runtime.action_id = 'previous'; runtime.action_ids = {'previous'}
    runtime.score = lambda: {'terminal': True, 'referee': {'state': 'ongoing', 'terminal': False}}
    with pytest.raises(RuntimeError, match='terminal'): runtime.begin_action('next')
    with pytest.raises(ValueError, match='active ordinary'): runtime.authorize_step({})


@pytest.mark.parametrize('robot_profile', ('original', 'l_mount'))
@pytest.mark.parametrize('tid,cid', [('shirt_folding', 'single_arm'),
    ('towel_folding', 'dual_same_side'), ('towel_stacking', 'dual_opposed')])
def test_actual_http_maps_task_owned_budget_to_sim_timeout(tmp_path, tid, cid, robot_profile):
    definition = task_definition('laundry_room', tid, cid, robot_profile=robot_profile)
    # The shared fixture is parametrized by (robot profile, catalog definition).
    fixture = workroom_service.__wrapped__(SimpleNamespace(param=(robot_profile, definition)), tmp_path)
    api, workers, _ = next(fixture)
    try:
        with TestClient(create_app(service=api)) as http:
            payload = request(task_id=tid, robot_id=definition['robot_requirements']['hand_id'],
                              seed=0, budgets=Budgets(sim_time_s=None)).model_dump(mode='json')
            created = http.post('/episodes', json=payload)
            assert created.status_code == 201, created.text
            ready = created.json(); session = api.get(ready['episode_id'])
            _, _, score = deadline_score(tid, cid)
            def receipt(worker):
                publish_targets(worker); worker.score = deepcopy(score)
            workers[0].on_step = receipt
            session.submit({'kind': 'direct_joints', 'action_id': 'at_horizon', 'duration_s': .02,
                'observation_sequence': session.sequence,
                'arm_joint_positions_rad': {n: session.raw['joint_positions_rad'][n] for n in session.capabilities.arm_joints},
                'hand_joint_positions_rad': {n: session.raw['joint_positions_rad'][n] for n in session.capabilities.hand_joints}})
            join_action(session)
            result = http.get('/episodes/'+ready['episode_id']+'/result')
            assert result.status_code == 200, result.text
            assert result.json()['outcome'] == 'sim_timeout'
            task = http.get('/episodes/'+ready['episode_id']+'/task').json()
            assert task['episode'] == {'terminal': True, 'outcome': 'sim_timeout'}
            assert task['referee']['state'] == 'ongoing' and not task['referee']['terminal']
    finally:
        try: next(fixture)
        except StopIteration: pass
