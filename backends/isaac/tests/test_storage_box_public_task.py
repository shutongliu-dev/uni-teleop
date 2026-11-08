"""Real discovery projection, without starting Isaac or a robot worker."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from bench.paths import load_json, DATA_ROOT
from bench.protocol.task_referee import project_task_rules,validated_public_contract
from scenes.workrooms.living_room.public_task import project_manipulation_task
from scenes.workrooms.living_room.tasks.storage_box.definition import TASK

@pytest.fixture
def frozen_storage():
    return load_json(Path(TASK['source_release']['path'])/'task.json')

def test_storage_uses_validated_public_contract(frozen_storage):
    public=project_manipulation_task(frozen_storage)
    assert public==project_task_rules(frozen_storage)
    assert public['public_rules']==validated_public_contract(frozen_storage)
    assert public['task_id']=='storage_box' and public['create_endpoint']=='POST /episodes'
    assert public['public_rules']['thresholds']['stable_s']==.3
    assert DATA_ROOT not in json.dumps(public)

def test_private_storage_state_never_enters_discovery(frozen_storage):
    public=project_manipulation_task(frozen_storage);changed=deepcopy(frozen_storage)
    marker='private-storage-position-contact-and-evaluation'
    for key in ('evaluation','asset_metadata','initial_state','public_goal'):
        changed[key]['private_probe']=marker
    changed['task_state']={'private':marker}
    assert project_manipulation_task(changed)==public
    assert marker not in json.dumps(public)
    assert not {'evaluation','asset_metadata','initial_state','task_state'} & public.keys()

@pytest.mark.parametrize('mutation',[lambda t:t['public_rules']['referee']['thresholds'].update(stable_s=.001),
    lambda t:t.update(public_rules_sha256='0'*64),lambda t:t.update(public_referee_contract_sha256='0'*64)])
def test_storage_discovery_rejects_mutated_frozen_contract(frozen_storage,mutation):
    mutation(frozen_storage)
    with pytest.raises(ValueError,match='frozen task identity'):project_manipulation_task(frozen_storage)

def test_storage_cannot_fall_back_to_legacy_water_projection(frozen_storage):
    del frozen_storage['public_rules']['referee']
    with pytest.raises(ValueError,match='validated frozen referee contract'):project_manipulation_task(frozen_storage)

@pytest.mark.parametrize('robot_profile',['original','l_mount'])
def test_get_tasks_lists_storage_without_starting_worker(tmp_path,robot_profile):
    from bench.runtime.service import AgentService
    from bench.runtime.server import create_app
    from bench.simulation.isaacsim_backend.l_mount import public_mount
    from scenes.workrooms.catalog import task_definition
    # Only a current catalog pin advertises the robot menu; serve the release
    # each robot profile publishes, not the historical definition release.
    root=Path(task_definition('living_room','storage_box',robot_profile=robot_profile)['source_release']['path'])
    frozen_storage=load_json(root/'task.json')
    entry,=load_json(root/'release_manifest.json')['tasks']
    mount=public_mount(load_json(root/entry['scene'])['robot'])
    assert (mount is None)==(robot_profile=='original')
    def no_worker(**kwargs):raise AssertionError('Task discovery must not launch a worker')
    manager=AgentService(tmp_path/'api',release=root,worker_factory=no_worker)
    with TestClient(create_app(service=manager)) as client:
        response=client.get('/tasks')
        assert response.status_code==200,response.text
        public, = response.json()
        options = public.pop('robot_configurations')
        assert public.pop('default_configuration_id') == 'single_arm'
        assert public.pop('scene_description_zh')
        assert public.pop('robot_mount',None)==mount
        expected = project_task_rules(frozen_storage)
        if 'reset_case' in frozen_storage:
            from bench.isaacsim.release import digest
            from scenes.workrooms.living_room.randomization.public import project_reset_case
            expected = project_reset_case(frozen_storage, expected)
            assert public.pop('configuration_id') == frozen_storage['reset_case']['configuration_id']
            assert public.pop('task_release_sha256') == digest(root/'release_manifest.json')
        assert public == expected
        assert set(options) == {'single_arm', 'dual_same_side', 'dual_opposed'}
        for configuration_id, option in options.items():
            dual = configuration_id != 'single_arm'
            assert option['configuration_id'] == configuration_id
            assert option['robot_requirements']['articulations'] == (2 if dual else 1)
            assert option['layout']['control_scope'] == ('one_agent_58_dof' if dual else 'one_agent_29_dof')
            placements = option['layout']['placements']
            assert set(placements) == ({'left', 'right'} if dual else {'right'})
            assert all(placement['hand_chirality'] == side for side, placement in placements.items())
        assert DATA_ROOT not in response.text
        for private_key in ('asset_metadata','initial_positions_m','configuration_sha256','robot_actor_prefix'):
            assert private_key not in response.text
