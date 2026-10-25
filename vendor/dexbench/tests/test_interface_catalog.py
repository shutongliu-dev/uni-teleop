"""Public discovery accurately reports real routes and controller availability."""
from types import SimpleNamespace
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from bench.protocol import PolicyRegistry
from bench.runtime.catalog import interface_catalog
from bench.runtime.server import create_app
from agent.model_client.geometry_actions import parse_geometry_action, GeometryActionLimits
from agent.model_client.actions import parse_model_action, ActionLimits
from agent.model_client.tools import parse_tool_call


def test_empty_registry_does_not_advertise_geometry_as_ready():
    result = interface_catalog()
    assert len(result['interfaces']) == 4
    assert [r['available'] for r in result['interfaces']] == [True, False, False, False]
    assert result['comparison']['mixed_interface_episode_supported'] is False


def test_catalog_is_available_as_public_http_without_starting_simulation():
    service = SimpleNamespace(registry=PolicyRegistry(), shutdown=lambda: None)
    with TestClient(create_app(service=service)) as client:
        reply = client.get('/interfaces')
    assert reply.status_code == 200
    result = reply.json()
    fixed = interface_catalog()
    assert result['interfaces'][:4] == fixed['interfaces']
    assert result['comparison']['mixed_interface_episode_supported'] is False
    free = result['interfaces'][4]
    assert free['id'] == 'free_tool_agent' and free['available'] is False
    assert free['policies'] == []
    assert result['comparison']['free_tool_agent']['independent_track'] is True


def test_catalog_hold_examples_use_current_parsers_and_limits():
    hold = {'action': 'hold', 'duration_s': .4}
    assert parse_model_action(hold)['action'] == 'hold'
    assert parse_geometry_action(hold)['action'] == 'hold'
    assert parse_tool_call({'tool': 'hold', 'arguments': {'duration_s': .4}})['tool'] == 'hold'
    assert ActionLimits().max_hold_duration_s == GeometryActionLimits().max_duration_s == 5.


def _descriptor(bimanual=False):
    from bench.protocol.schemas import DirectControlHandoff, PolicyDescriptor
    return PolicyDescriptor(policy_id='classical_fingertip_bimanual' if bimanual else 'classical_fingertip',
        version='1.2.0' if bimanual else '1.9.0', implementation_sha256='a'*64,
        trained=False, observation_profile='vision_proprio', skills=['move_geometry', 'hold'],
        control_period_s=.02, geometry_inputs=['wrist_pose', 'fingertip_positions', 'hand_posture'],
        direct_control_handoff=None if bimanual else DirectControlHandoff())


def test_bimanual_discovery_separates_native_geometry_from_existing_client_modes():
    from bench.runtime.free_track import advertise_free_track
    dual = _descriptor(bimanual=True)
    result = advertise_free_track(interface_catalog([dual]), [dual])
    rows = {row['id']: row for row in result['interfaces']}
    assert result['geometry_policies'] == [dual.model_dump(mode='json')]
    assert rows['fingertip_action']['available'] is True
    assert rows['fingertip_action']['supported_policy_ids'] == ['classical_fingertip_bimanual']
    assert rows['fingertip_action']['executor'] == 'classical_fingertip_bimanual'
    for name in ('joints_action', 'fingertip_tools', 'fingertip_code'):
        assert rows[name]['backend_available'] is True
        assert rows[name]['available'] is False
        assert rows[name]['unavailable_reason']
    assert rows['free_tool_agent']['available'] is False


def test_single_arm_and_mixed_discovery_keep_supported_policy_ids_explicit():
    from bench.runtime.free_track import advertise_free_track
    single, dual = _descriptor(), _descriptor(bimanual=True)
    for descriptors in ([single], [single, dual]):
        result = advertise_free_track(interface_catalog(descriptors), descriptors)
        rows = {row['id']: row for row in result['interfaces']}
        assert all(row['available'] for row in rows.values())
        for name in ('fingertip_tools', 'fingertip_code'):
            assert rows[name]['supported_policy_ids'] == ['classical_fingertip']
            assert rows[name]['client_robot_scope'] == 'single_arm'
        assert rows['fingertip_action']['supported_policy_ids'] == sorted(d.policy_id for d in descriptors)
        assert [p['policy_id'] for p in rows['free_tool_agent']['policies']] == ['classical_fingertip']


def test_named_instances_do_not_advertise_legacy_runner_automation():
    instances = _descriptor(bimanual=True).model_copy(update={'policy_id': 'classical_fingertip_instances'})
    result = interface_catalog([instances])
    assert result['geometry_policies'] == [instances.model_dump(mode='json')]
    assert all(row['backend_available'] and not row['available'] for row in result['interfaces'])
    assert all(not row['supported_policy_ids'] for row in result['interfaces'])
    assert 'v3 (named instances)' in result['shared_contract']['contact_history_scope']
    mixed = interface_catalog([_descriptor(), instances])
    geometry = next(row for row in mixed['interfaces'] if row['id'] == 'fingertip_action')
    assert geometry['supported_policy_ids'] == ['classical_fingertip']


def test_incomplete_geometry_policy_does_not_advertise_a_callable_geometry_interface():
    incomplete = _descriptor(bimanual=True).model_copy(update={'skills': ['hold']})
    rows = interface_catalog([incomplete])['interfaces']
    assert all(not row['available'] for row in rows)
    assert all(not row['backend_available'] for row in rows if row['id'] != 'joints_action')


@pytest.mark.parametrize('family,task_id', [('card_room', 'task0'),
                                          ('mechanical_assembly', 'fastener_assembly')])
def test_real_ordinary_task_discovery_keeps_legacy_catalog(family, task_id):
    from bench.runtime.controllers import default_registry
    from bench.runtime.free_track import advertise_free_track
    from scenes.workrooms.catalog import task_definition

    release = Path(task_definition(family, task_id)['source_release']['path'])
    registry = default_registry(release)
    service = SimpleNamespace(registry=registry, release=release, shutdown=lambda: None)
    descriptors = registry.descriptors(task_id=task_id)
    expected = advertise_free_track(interface_catalog(descriptors), descriptors)
    with TestClient(create_app(service=service)) as client:
        assert client.get('/interfaces', params={'task_id': task_id}).json() == expected
        assert client.get('/interfaces').json() == expected
        assert client.get('/interfaces', params={'task_id': 'unknown'}).status_code == 422
