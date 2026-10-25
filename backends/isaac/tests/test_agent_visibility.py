"""Public camera and source-pin regressions using a CPU worker double.

These tests verify HTTP visibility and provenance, not native physics quality.
"""
import hashlib

import pytest
from fastapi.testclient import TestClient

from bench.paths import load_json
from bench.protocol import PolicyRegistry
from bench.runtime.server import create_app
from bench.runtime.service import canonical_hash

# Pytest must discover both fixtures here; importing only runner does not expose
# its module-scoped frozen_inputs dependency in this test module.
from test_agent_service import (Teacher, frozen_inputs, join_action, request,
                                runner, session_for, skill_action)


class CameraTeacher(Teacher):
    def bind_camera_reader(self, reader):
        self.read_camera = reader


@pytest.mark.parametrize('depth',[False,True])
def test_teacher_intermediate_frames_are_not_public_even_when_revision_is_guessed(runner,depth):
    make, _, workers = runner
    teacher = CameraTeacher(block=True)
    registry = PolicyRegistry()
    registry.register(teacher)
    manager = make(registry=registry,depth=depth)
    with TestClient(create_app(service=manager), raise_server_exceptions=False) as client:
        response = client.post('/episodes', json=request(
            track='system0', policy_id='test_teacher', policy_version='v1'
        ).model_dump(mode='json'))
        assert response.status_code == 201
        initial = response.json()
        episode = initial['episode_id']
        session = manager.get(episode)
        published_uris = {camera['rgb_uri'] for camera in initial['observation']['cameras']}
        published_depths = {camera['depth_uri'] for camera in initial['observation']['cameras']} if depth else set()
        submitted = client.post(f'/episodes/{episode}/actions', json=skill_action(session))
        assert submitted.status_code == 202
        assert teacher.started.wait(timeout=1)

        # The policy is blocked outside the session lock with its private teacher
        # observation; HTTP is free to serve only the last decision snapshot.
        policy_cameras = teacher.observations[-1].cameras
        intermediate_revision = [payload['sequence'] for operation, payload in workers[-1].calls
                                 if operation == 'cameras'][-1]
        private_uris = {f'/episodes/{episode}/images/{intermediate_revision}/{camera.camera_id}'
                        for camera in policy_cameras}
        assert private_uris.isdisjoint(published_uris)
        assert {camera.rgb_uri for camera in policy_cameras} == private_uris
        observation = client.get(f'/episodes/{episode}/observation')
        assert observation.status_code == 200
        assert observation.json() == initial['observation']
        for camera in policy_cameras:
            assert client.get(camera.rgb_uri).status_code == 404
            if depth:
                assert camera.depth_uri not in published_depths
                assert client.get(camera.depth_uri).status_code == 404
            # A trusted RL plugin can still consume the exact generated RGB.
            image = teacher.read_camera(camera)
            assert image.startswith(b'\x89PNG\r\n\x1a\n')
            assert hashlib.sha256(image).hexdigest() == camera.rgb_sha256
        for uri in published_uris:
            assert client.get(uri).status_code == 200
        for uri in published_depths:
            assert client.get(uri).status_code == 200

        teacher.release.set()
        join_action(session)
        final = client.get(f'/episodes/{episode}/observation').json()
        assert final['last_action_status'] == 'completed'
        final_uris = {camera['rgb_uri'] for camera in final['cameras']}
        assert final_uris.isdisjoint(private_uris | published_uris)
        for camera in final['cameras']:
            reply = client.get(camera['rgb_uri'])
            assert reply.status_code == 200
            assert hashlib.sha256(reply.content).hexdigest() == camera['rgb_sha256']
            if depth:
                reply = client.get(camera['depth_uri'])
                assert reply.status_code == 200
                assert hashlib.sha256(reply.content).hexdigest() == camera['depth_sha256']
        # Publishing a later frame must not retroactively publish teacher frames.
        for uri in private_uris:
            assert client.get(uri).status_code == 404
        if depth:
            for camera in policy_cameras:
                assert client.get(camera.depth_uri).status_code == 404


def test_policy_camera_reader_rejects_paths_and_unrecorded_frames(runner):
    make, _, _ = runner
    teacher = CameraTeacher()
    registry = PolicyRegistry()
    registry.register(teacher)
    session, _ = session_for(make(registry=registry), track='system0',
                             policy_id='test_teacher', policy_version='v1')
    for reference in (str(session.directory/'private/evaluator_evidence.json'),
                      '/etc/passwd',
                      f'/episodes/{session.spec.episode_id}/images/9999/task'):
        with pytest.raises(ValueError, match='Unknown policy camera frame'):
            teacher.read_camera(reference)


def test_runtime_pin_contains_task_loading_and_geometry_source_closure(runner):
    make, _, _ = runner
    session, _ = session_for(make())
    sources = load_json(session.directory/'source_hashes.json')
    assert canonical_hash(sources) == session.spec.runtime_code_sha256
    for relative in ('bench/isaacsim/release.py', 'bench/simulation/assets/task_parts.py',
                     'bench/geometry.py', 'bench/control/kinematics.py', 'scripts/isaac_python.sh'):
        assert relative in sources, f'Runtime pin omits executable dependency {relative}'
        archived = session.directory/'source_snapshot'/relative
        assert archived.is_file(), f'Runtime source snapshot omits {relative}'
        assert hashlib.sha256(archived.read_bytes()).hexdigest() == sources[relative]
