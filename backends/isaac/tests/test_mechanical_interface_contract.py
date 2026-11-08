"""CPU checks for the mechanical catalog-to-Agent validation envelope."""

import importlib
import json
from pathlib import Path

import pytest

from agent.model_client.scene_tasks import (TaskSetupError, list_scene_tasks,
                                             published_task, select_task)
from bench.paths import load_json

NIST_TASKS = {'nist_connector_insertion', 'nist_belt_drive_assembly',
              'nist_cable_routing', 'nist_wire_harness'}
PROFILES = ('original', 'l_mount')


def frozen_original(reference):
    """Walk frozen seed-case and L-mount revision wrappers to the original release."""
    reference = {key: reference[key] for key in ('path', 'manifest_sha256')}
    while True:
        manifest = load_json(Path(reference['path'])/'release_manifest.json')
        if manifest['schema'] not in ('workroom_randomized_release.v1', 'workroom_robot_revision_release.v1'):
            return reference
        reference = {key: manifest['base_release'][key] for key in ('path', 'manifest_sha256')}


@pytest.mark.parametrize('robot_profile', PROFILES)
@pytest.mark.parametrize('level', ['L1', 'L2', 'L3', 'L4', 'L5'])
@pytest.mark.parametrize('changes', [
    {'robot_id': 'allegro_v5_right'},
    {'scene_variant': 'enriched_v2'},
])
def test_mechanical_service_rejects_changed_frozen_setup_before_worker(tmp_path, level, changes, robot_profile):
    from bench.protocol import PolicyRegistry
    from bench.runtime.service import AgentService, CreateEpisodeRequest, ServiceError

    selected = select_task(level, family='mechanical_assembly', robot_profile=robot_profile)
    workers = []

    def worker_factory(**kwargs):
        workers.append(kwargs)
        raise AssertionError('A rejected frozen setup must not start a worker')

    api = AgentService(tmp_path / 'episodes', release=Path(selected.source_release['path']),
                       registry=PolicyRegistry(), worker_factory=worker_factory)
    request = CreateEpisodeRequest(task_id=selected.task_id, agent_id='setup_contract',
                                   agent_version='v1', agent_config_sha256='a' * 64, **changes)
    # One admission message covers single-arm and dual-arm frozen installations.
    with pytest.raises(ServiceError, match='frozen robot and original scene') as error:
        api.create(request)
    assert error.value.status == 422
    assert workers == []
    assert api.sessions == {}


@pytest.mark.parametrize('robot_profile', PROFILES)
def test_mechanical_selection_publishes_privileged_validation_identity(robot_profile):
    rows = {row['task_id']: row for row in list_scene_tasks('mechanical_assembly', robot_profile=robot_profile)}
    assert set(rows) == {
        'parts_sorting', 'peg_insertion', 'shaft_spacer_assembly',
        'fastener_assembly', 'gear_train_assembly', 'cooperative_insertion',
    } | NIST_TASKS
    for task_id, row in rows.items():
        validation = row['validation']
        if task_id == 'cooperative_insertion':
            # The dual-arm task keeps its own privileged release validator.
            assert validation['privileged_entrypoint'] == 'scripts.validation.mechanical_cooperative_release:run'
            assert validation['privileged_entrypoint_required'] is True
            assert validation['task_policy_success_claimed'] is False
            assert validation['release_status'] == 'prepared_unvalidated'
            assert validation['release_sha256'] == row['source_release']['manifest_sha256']
            continue
        assert validation['mode'] == 'privileged_native'
        assert validation['validation_scope'] == 'privileged_native_only'
        assert validation['required'] is True
        assert validation['ordinary_agent_validation'] is False
        assert validation['model_calls'] == 0
        assert validation['object_state_writeback'] is False
        assert validation['privileged_entrypoint_required'] is True
        if task_id in NIST_TASKS:
            assert validation['privileged_entrypoint'] == 'scripts.validation.mechanical_nist_release:run'
            assert validation['hand_object_interaction_owner'].startswith('data_collection;')
            assert row['task_number'] in (6, 7, 8, 9)
            if row['status'] == 'planned':
                assert not row['runnable'] and row['source_release'] is None
                assert validation['release_status'] == 'planned'
                assert validation['release_sha256'] is None
                continue
        assert row['status'] == 'implemented'
        assert row['runnable'] is True
        assert validation['release_status'] == 'prepared_unvalidated'
        assert row['source_release']['manifest_sha256'] == validation['release_sha256']
        if task_id == 'gear_train_assembly':
            assert validation['privileged_entrypoint'] == 'scripts.validation.validate_gear_scene:run'
            assert validation['operator_motion_entrypoint'] == 'scripts.validation.validate_gear_driven_tool:run'
            assert validation['tool_contact_entrypoint'] == 'scripts.validation.validate_gear_tool_function:run'
            assert validation['hand_object_interaction_owner'].startswith('data_collection;')
            # Frozen seed cases (and the L-mount revision) wrap the wrench-reach candidate.
            assert frozen_original(row['source_release']) == validation['candidate_release']
            assert validation['candidate_release']['path'].endswith('/gear_train_assembly/wrench_reach_v1')
            selected = select_task('L5', family='mechanical_assembly', robot_profile=robot_profile)
            selected.require_runnable()
            assert selected.pin_release
            assert selected.source_release == row['source_release']
        elif task_id == 'fastener_assembly':
            assert validation['privileged_entrypoint'] is None
            # The loose-bolt candidate precedes the frozen runtime inputs that
            # the seed cases (and the L-mount revision) wrap.
            original = importlib.import_module(
                'scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.definition').TASK
            assert validation['candidate_release'] == original['previous_source_release']
            assert frozen_original(row['source_release']) == original['source_release']
            assert validation['candidate_release']['manifest_sha256'] == (
                'd468bb978a67beb9bb2a9f16aa3e0d780b9ba61a46ddbbb446ece716816d69d8')
            assert validation['release_sha256'] == row['source_release']['manifest_sha256']
            assert validation['candidate_release']['path'].endswith('/loose_bolt_candidate_v10_pgs')
            selected = select_task('L4', family='mechanical_assembly', robot_profile=robot_profile)
            assert selected.pin_release
            assert selected.source_release == row['source_release']
        elif task_id not in NIST_TASKS:
            assert validation['privileged_entrypoint'] is None


def test_nonmechanical_selection_keeps_validation_metadata_unchanged():
    selected = select_task('water_pouring', family='living_room')
    assert selected.validation is None
    assert selected.metadata()['validation'] is None


def test_published_task_rejects_catalog_release_hash_drift():
    selected = select_task('L1', family='mechanical_assembly')
    with pytest.raises(TaskSetupError) as error:
        published_task(selected, [{
            'task_id': 'parts_sorting',
            'task_release_sha256': '0' * 64,
        }])
    assert error.value.code == 'release_mismatch'


def test_created_episode_without_release_hash_fails_closed():
    selected = select_task('L1', family='mechanical_assembly')
    with pytest.raises(TaskSetupError) as error:
        selected.verify_release({})
    assert error.value.code == 'release_mismatch'
