"""Missing native evidence must not turn the threaded-scene check green."""
from copy import deepcopy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.validation import validate_mechanical_dynamics as validation
from bench.simulation.isaacsim_backend import asset_catalog


ROLES = ('screwdriver', 'bolt', 'threaded_fixture')


@pytest.fixture
def native_env(monkeypatch):
    """Stub only native readback APIs; execute the real parameter validator."""
    def attribute(value):
        return SimpleNamespace(Get=lambda: value)

    usd = SimpleNamespace(
        MassAPI=lambda prim: SimpleNamespace(GetMassAttr=lambda: attribute(prim['mass'])),
        RigidBodyAPI=lambda prim: SimpleNamespace(GetKinematicEnabledAttr=lambda: attribute(prim['kinematic'])),
    )
    monkeypatch.setitem(sys.modules, 'pxr', SimpleNamespace(UsdPhysics=usd))
    monkeypatch.setitem(sys.modules, 'bench.simulation.isaacsim_backend.env', SimpleNamespace(_np=np.asarray))
    materials = {'physics:staticFriction': .6, 'physics:dynamicFriction': .6, 'physics:restitution': 0.}
    env = SimpleNamespace(scene_info={'objects': {}, 'fixtures': {}},
                          scene_spec={'objects': [], 'fixtures': []}, views={}, prims={})
    masses = dict(zip(ROLES, (.3, .17, 1.)))
    for name in ROLES:
        group = 'fixtures' if name == 'threaded_fixture' else 'objects'
        actor = '/World/'+group.title()+'/'+name
        collider = actor+'/collider'
        env.scene_info[group][name] = {
            'path': actor, 'body_mode': 'kinematic_fixture' if group == 'fixtures' else 'dynamic',
            'usd_physics_parameters': {'active_collision_paths': [collider],
                'disabled_collision_paths': [],
                'collider_materials': {collider: {'path': '/World/Materials/source',
                                                'parameters': deepcopy(materials)}}},
        }
        env.scene_spec[group].append({'id': name, 'asset_id': name, 'asset_root': '/assets'})
        env.prims[actor] = {'mass': masses[name], 'kinematic': group == 'fixtures'}
        view = SimpleNamespace(count=1, materials=np.array([[[.6, .6, 0.]]]),
                               masses=np.array([0. if group == 'fixtures' else masses[name]]))
        view.get_material_properties = lambda view=view: view.materials
        view.get_masses = lambda view=view: view.masses
        env.views[actor] = view
    # These diagnostic bodies legitimately have another physics authoring path.
    env.scene_info['objects']['cube_table'] = {'path': '/World/Objects/cube_table'}
    env.scene_info['fixtures']['robot_stand'] = {'path': '/World/Fixtures/robot_stand'}
    env._sim_view = SimpleNamespace(create_rigid_body_view=lambda path: env.views[path])
    env.stage = SimpleNamespace(GetPrimAtPath=lambda path: env.prims[path])
    monkeypatch.setattr(asset_catalog, 'load_task_asset',
        lambda asset_id, root, verify_hashes: {'mass_kg': masses[asset_id]})
    monkeypatch.setattr(validation, 'native_collider_inventory', lambda *args: {'passed': True})
    return env


def info(env, name='bolt'):
    return env.scene_info['fixtures' if name == 'threaded_fixture' else 'objects'][name]


def test_complete_three_body_evidence_passes_and_non_usd_diagnostic_parts_are_allowed(native_env):
    result = validation.native_task_parameters(native_env)
    assert result['passed']
    assert result['required_body_names'] == sorted(ROLES)
    assert set(result['bodies']) == set(ROLES)
    assert result['bodies']['threaded_fixture']['mass_checks']['native_dynamic_mass'] is None


def test_explicit_standalone_tool_still_requires_its_native_evidence(native_env):
    for role in ('bolt', 'threaded_fixture'):
        group = 'fixtures' if role == 'threaded_fixture' else 'objects'
        native_env.scene_info[group].pop(role)
    result = validation.native_task_parameters(native_env, required_roles={'screwdriver': 'objects'})
    assert result['passed'] and result['required_body_names'] == ['screwdriver']
    assert not validation.native_task_parameters(native_env)['passed']
    native_env.scene_info['objects'].pop('screwdriver')
    assert not validation.native_task_parameters(native_env, required_roles={'screwdriver': 'objects'})['passed']


def test_explicit_empty_required_actor_set_is_rejected(native_env):
    with pytest.raises(ValueError, match='nonempty'):
        validation.native_task_parameters(native_env, required_roles={})


@pytest.mark.parametrize('role', ROLES)
@pytest.mark.parametrize('missing', ['actor', 'authored'])
def test_missing_each_required_actor_or_parameter_block_fails(native_env, role, missing):
    if missing == 'actor':
        native_env.scene_info['fixtures' if role == 'threaded_fixture' else 'objects'].pop(role)
    else:
        info(native_env, role).pop('usd_physics_parameters')
    result = validation.native_task_parameters(native_env)
    assert not result['passed']
    assert not result['bodies'][role]['passed']
    assert result['bodies'][role]['error']


@pytest.mark.parametrize('empty', ['actors', 'authored'])
def test_empty_evidence_cannot_pass_via_all_empty(native_env, empty):
    if empty == 'actors':
        native_env.scene_info = {'objects': {}, 'fixtures': {}}
    else:
        for name in ROLES:
            info(native_env, name).pop('usd_physics_parameters')
    result = validation.native_task_parameters(native_env)
    assert not result['passed']
    assert set(result['bodies']) == set(ROLES)
    assert not any(row['passed'] for row in result['bodies'].values())


@pytest.mark.parametrize('fault', ['empty_bindings', 'no_active', 'all_empty', 'missing_disabled',
                                  'active_without_binding', 'duplicate_active', 'nonfinite_material'])
def test_incomplete_collider_evidence_fails_before_native_inventory(native_env, monkeypatch, fault):
    authored = info(native_env)['usd_physics_parameters']
    collider = authored['active_collision_paths'][0]
    if fault == 'empty_bindings':
        authored['collider_materials'] = {}
    elif fault == 'no_active':
        authored['active_collision_paths'] = []
        authored['disabled_collision_paths'] = [collider]
    elif fault == 'all_empty':
        authored.update(active_collision_paths=[], collider_materials={})
        native_env.views[info(native_env)['path']].materials = np.empty((1, 0, 3))
    elif fault == 'missing_disabled':
        authored.pop('disabled_collision_paths')
    elif fault == 'active_without_binding':
        authored['active_collision_paths'].append(collider+'/missing')
    elif fault == 'duplicate_active':
        authored['active_collision_paths'].append(collider)
    else:
        authored['collider_materials'][collider]['parameters']['physics:staticFriction'] = float('inf')
    checked = []
    monkeypatch.setattr(validation, 'native_collider_inventory',
                        lambda env, row, *args: checked.append(row['path']) or {'passed': True})
    result = validation.native_task_parameters(native_env)
    assert not result['passed']
    assert info(native_env)['path'] not in checked


@pytest.mark.parametrize('fault', ['zero_bodies', 'multiple_bodies', 'empty_mass', 'missing_mass',
                                  'nonfinite_mass', 'empty_shapes', 'missing_materials',
                                  'bad_material_shape', 'nonfinite_materials', 'readback_error'])
def test_missing_native_body_or_readbacks_fail(native_env, fault):
    view = native_env.views[info(native_env)['path']]
    if fault in ('zero_bodies', 'multiple_bodies'):
        view.count = 0 if fault == 'zero_bodies' else 2
        # Count must be checked before any data is indexed.
        view.get_material_properties = lambda: pytest.fail('readback before count validation')
    elif fault == 'empty_mass':
        view.masses = np.array([])
    elif fault == 'missing_mass':
        view.masses = None
    elif fault == 'nonfinite_mass':
        view.masses[:] = float('nan')
    elif fault == 'empty_shapes':
        view.materials = np.empty((1, 0, 3))
    elif fault == 'missing_materials':
        view.materials = None
    elif fault == 'bad_material_shape':
        view.materials = np.array([[.6, .6, 0.]])
    elif fault == 'nonfinite_materials':
        view.materials[0, 0, 0] = float('inf')
    else:
        def unavailable():
            raise RuntimeError('Native readback unavailable')
        view.get_masses = unavailable
    result = validation.native_task_parameters(native_env)
    assert not result['passed']
    assert not result['bodies']['bolt']['passed']


def test_valid_disabled_shape_storage_remains_supported(native_env):
    authored = info(native_env)['usd_physics_parameters']
    disabled = info(native_env)['path']+'/disabled_old_head'
    authored['disabled_collision_paths'].append(disabled)
    authored['collider_materials'][disabled] = deepcopy(next(iter(authored['collider_materials'].values())))
    native_env.views[info(native_env)['path']].materials = np.array([[[.6, .6, 0.], [.6, .6, 0.]]])
    assert validation.native_task_parameters(native_env)['passed']


def test_existing_mass_material_and_inventory_mismatch_checks_are_preserved(native_env, monkeypatch):
    view = native_env.views[info(native_env)['path']]
    view.masses[:] = .171
    assert not validation.native_task_parameters(native_env)['passed']
    view.masses[:] = .17
    view.materials[0, 0, 0] = .7
    assert not validation.native_task_parameters(native_env)['passed']
    view.materials[0, 0, 0] = .6
    monkeypatch.setattr(validation, 'native_collider_inventory', lambda *args: {'passed': False})
    assert not validation.native_task_parameters(native_env)['passed']
