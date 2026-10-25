"""CPU checks for the mechanical catalog-to-Agent validation envelope."""

import pytest

from agent.model_client.scene_tasks import (TaskSetupError, list_scene_tasks,
                                             published_task, select_task)

NIST_TASKS = {'nist_connector_insertion', 'nist_belt_drive_assembly',
              'nist_cable_routing', 'nist_wire_harness'}


@pytest.mark.parametrize('level', ['L1', 'L2', 'L3', 'L4', 'L5'])
@pytest.mark.parametrize('changes', [
    {'robot_id': 'allegro_v5_right'},
    {'scene_variant': 'enriched_v2'},
])
def test_mechanical_service_rejects_changed_frozen_setup_before_worker(tmp_path, level, changes):
    from pathlib import Path
    from bench.protocol import PolicyRegistry
    from bench.runtime.service import AgentService, CreateEpisodeRequest, ServiceError

    selected = select_task(level, family='mechanical_assembly')
    workers = []

    def worker_factory(**kwargs):
        workers.append(kwargs)
        raise AssertionError('A rejected frozen setup must not start a worker')

    api = AgentService(tmp_path / 'episodes', release=Path(selected.source_release['path']),
                       registry=PolicyRegistry(), worker_factory=worker_factory)
    request = CreateEpisodeRequest(task_id=selected.task_id, agent_id='setup_contract',
                                   agent_version='v1', agent_config_sha256='a' * 64, **changes)
    with pytest.raises(ServiceError, match='frozen single-arm robot and original scene') as error:
        api.create(request)
    assert error.value.status == 422
    assert workers == []
    assert api.sessions == {}


def test_mechanical_selection_publishes_privileged_validation_identity():
    rows = {row['task_id']: row for row in list_scene_tasks('mechanical_assembly')}
    assert set(rows) == {
        'parts_sorting', 'peg_insertion', 'shaft_spacer_assembly',
        'fastener_assembly', 'gear_train_assembly',
    } | NIST_TASKS
    for task_id, row in rows.items():
        validation = row['validation']
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
            assert row['source_release'] == validation['candidate_release']
            assert row['source_release']['path'].endswith('/practical_rules_v4/release')
            selected = select_task('L5', family='mechanical_assembly')
            selected.require_runnable()
            assert selected.pin_release
            assert selected.source_release == row['source_release']
        elif task_id == 'fastener_assembly':
            assert validation['privileged_entrypoint'] is None
            assert row['source_release'] == validation['candidate_release']
            assert row['source_release']['manifest_sha256'] == (
                'e3fdb6d0ce747bcc0f4b4786dbe9fe4ad11cfd69ab4266919551820357e40a44')
            assert validation['release_sha256'] == row['source_release']['manifest_sha256']
            assert row['source_release']['path'].endswith('/loose_bolt_candidate_v10_pgs')
            selected = select_task('L4', family='mechanical_assembly')
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
