import pytest

from bench.simulation.isaacsim_backend.env import (
    FR3DexterousIsaacEnv,
    physics_external_forces_every_iteration,
    physics_solver_type,
)


def test_legacy_profile_keeps_tgs_and_explicit_release_can_select_pgs():
    assert physics_solver_type({}) == 'TGS'
    assert physics_solver_type({'solver_type': 'TGS'}) == 'TGS'
    assert physics_solver_type({'solver_type': 'PGS'}) == 'PGS'


@pytest.mark.parametrize('value', [None, True, 0, 'pgs', 'CPU', [], {}])
def test_bad_solver_selection_does_not_silently_change_the_physics_profile(value):
    with pytest.raises(ValueError, match='solver_type'):
        physics_solver_type({'solver_type': value})


@pytest.mark.parametrize('solver', ['TGS', 'PGS'])
def test_legacy_profile_does_not_author_external_force_integration(solver):
    env = FR3DexterousIsaacEnv.__new__(FR3DexterousIsaacEnv)
    env.enable_external_forces_every_iteration = physics_external_forces_every_iteration({'solver_type': solver})
    assert env.enable_external_forces_every_iteration is None
    # An absent option must not even call a schema authoring method.
    env._configure_external_forces_every_iteration(object())


@pytest.mark.parametrize('options, expected', [
    ({'enable_external_forces_every_iteration': True}, True),
    ({'solver_type': 'TGS', 'enable_external_forces_every_iteration': True}, True),
    ({'solver_type': 'TGS', 'enable_external_forces_every_iteration': False}, False),
    ({'solver_type': 'PGS', 'enable_external_forces_every_iteration': False}, False),
])
def test_explicit_external_force_integration_authors_the_requested_bool(options, expected):
    class SceneAPI:
        def __init__(self):
            self.authored = []

        def CreateEnableExternalForcesEveryIterationAttr(self, value):
            self.authored.append(value)

    api = SceneAPI()
    env = FR3DexterousIsaacEnv.__new__(FR3DexterousIsaacEnv)
    env.enable_external_forces_every_iteration = physics_external_forces_every_iteration(options)
    env._configure_external_forces_every_iteration(api)
    assert len(api.authored) == 1
    assert api.authored[0] is expected


@pytest.mark.parametrize('value', [None, 0, 1, 0.0, 1.0, 'true', 'false', [], {}])
def test_external_force_integration_rejects_non_boolean_options(value):
    with pytest.raises(ValueError, match='enable_external_forces_every_iteration must be a bool'):
        physics_external_forces_every_iteration({'enable_external_forces_every_iteration': value})


def test_pgs_cannot_enable_tgs_external_force_integration():
    with pytest.raises(ValueError, match='requires solver_type TGS'):
        physics_external_forces_every_iteration({
            'solver_type': 'PGS', 'enable_external_forces_every_iteration': True})


@pytest.mark.parametrize('options', [
    {'enable_external_forces_every_iteration': None},
    {'solver_type': 'PGS', 'enable_external_forces_every_iteration': True},
])
def test_invalid_external_force_option_fails_before_runtime_initialization(options):
    # No Isaac runtime or robot/physics resources are needed to reject bad options.
    with pytest.raises(ValueError, match='enable_external_forces_every_iteration'):
        FR3DexterousIsaacEnv({'isaacsim': options})
