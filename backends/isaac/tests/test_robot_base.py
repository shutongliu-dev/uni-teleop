"""CPU topology/immutability checks, not native dynamics qualification."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.paths import DATA_ROOT
from bench.simulation.isaacsim_backend.robot_base import MODE, apply_base_constraint, runtime_base_check


ASSET = Path(f'{DATA_ROOT}/assets/robots/assemblies/fr3_sharpa/isaacsim_6_1_v2/usd/fr3_sharpa/fr3_sharpa.usda')


@pytest.fixture
def scene_stage():
    pytest.importorskip('pxr.Usd')
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateInMemory()
    robot = UsdGeom.Xform.Define(stage, '/World/Robot').GetPrim()
    UsdGeom.Xformable(robot).AddTranslateOp().Set((-.13, 0., .78))
    base = UsdGeom.Xform.Define(stage, '/World/Robot/Geometry/base/fr3_link0').GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(base)
    UsdPhysics.ArticulationRootAPI.Apply(base)
    joint = UsdPhysics.FixedJoint.Define(stage, '/World/Robot/Physics/fr3_base_joint')
    joint.CreateBody0Rel().SetTargets([robot.GetPath()])
    joint.CreateBody1Rel().SetTargets([base.GetPath()])
    scene = {'robot': {'hand_id': 'sharpa_w01_right'}, 'isaacsim': {'robot_base_constraint': MODE}}
    return stage, robot, base, joint, scene


def test_no_opt_in_is_byte_identical(scene_stage):
    stage, robot, _, _, scene = scene_stage
    scene['isaacsim'] = {}
    before = stage.GetRootLayer().ExportToString()
    assert apply_base_constraint(stage, robot, scene) is None
    assert stage.GetRootLayer().ExportToString() == before


def test_joint_to_world_is_root_and_body_pose_is_preserved(scene_stage):
    from pxr import Usd, UsdGeom, UsdPhysics
    stage, robot, base, joint, scene = scene_stage
    pose = UsdGeom.Xformable(base).ComputeLocalToWorldTransform(0)
    record = apply_base_constraint(stage, robot, scene)
    assert not base.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert joint.GetPrim().HasAPI(UsdPhysics.ArticulationRootAPI)
    assert joint.GetBody0Rel().GetTargets() == []
    assert joint.GetBody1Rel().GetTargets() == [base.GetPath()]
    assert UsdGeom.Xformable(base).ComputeLocalToWorldTransform(0) == pose
    assert list(joint.GetLocalPos0Attr().Get()) == pytest.approx([-.13, 0., .78])
    assert record['articulation_path'] == str(joint.GetPath())
    assert record['contact_filtering_changed'] is False
    parsed = UsdPhysics.LoadUsdPhysicsFromRange(stage, ['/World'])
    desc = parsed[UsdPhysics.ObjectType.Articulation][1][0]
    assert list(desc.rootPrims) == [joint.GetPath()]


def test_existing_physx_articulation_settings_move_to_joint(scene_stage):
    PhysxSchema = pytest.importorskip('pxr.PhysxSchema')
    stage, robot, base, joint, scene = scene_stage
    if not base.CanApplyAPI('PhysxArticulationAPI'):
        pytest.skip('PhysxSchema plugin is not registered')
    api = PhysxSchema.PhysxArticulationAPI.Apply(base)
    api.CreateSolverPositionIterationCountAttr(16)
    api.CreateSolverVelocityIterationCountAttr(4)
    api.CreateEnabledSelfCollisionsAttr(True)
    apply_base_constraint(stage, robot, scene)
    assert not base.HasAPI(PhysxSchema.PhysxArticulationAPI)
    assert joint.GetPrim().HasAPI(PhysxSchema.PhysxArticulationAPI)
    moved = PhysxSchema.PhysxArticulationAPI(joint)
    assert moved.GetSolverPositionIterationCountAttr().Get() == 16
    assert moved.GetSolverVelocityIterationCountAttr().Get() == 4
    assert moved.GetEnabledSelfCollisionsAttr().Get() is True


@pytest.mark.parametrize('problem', ['mode', 'hand', 'missing', 'duplicate', 'disabled', 'excluded', 'scale'])
def test_unsupported_import_fails_before_writing(scene_stage, problem):
    from pxr import UsdGeom, UsdPhysics
    stage, robot, base, joint, scene = scene_stage
    if problem == 'mode': scene['isaacsim']['robot_base_constraint'] = 'unknown'
    elif problem == 'hand': scene['robot']['hand_id'] = 'dual_fr3_sharpa_w01'
    elif problem == 'missing': stage.RemovePrim(joint.GetPath())
    elif problem == 'duplicate':
        other = UsdPhysics.FixedJoint.Define(stage, '/World/Robot/Physics/OtherAnchor')
        other.CreateBody1Rel().SetTargets([base.GetPath()])
    elif problem == 'disabled': joint.CreateJointEnabledAttr(False)
    elif problem == 'excluded': joint.CreateExcludeFromArticulationAttr(True)
    elif problem == 'scale': UsdGeom.Xformable(robot).AddScaleOp().Set((2., 1., 1.))
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError): apply_base_constraint(stage, robot, scene)
    assert stage.GetRootLayer().ExportToString() == before


def test_pinned_robot_reference_keeps_29_dofs_meshes_and_all_source_bytes():
    pytest.importorskip('pxr.Usd')
    if not ASSET.is_file(): pytest.skip('Pinned SHARPA asset not mounted')
    from pxr import Usd, UsdGeom, UsdPhysics
    before_files = {p: p.read_bytes() for p in ASSET.parent.rglob('*') if p.is_file()}
    stage = Usd.Stage.CreateInMemory()
    robot = UsdGeom.Xform.Define(stage, '/World/Robot').GetPrim()
    robot.GetReferences().AddReference(str(ASSET))
    UsdGeom.Xformable(robot).AddTranslateOp().Set((-.13, 0., .78))
    def inventory():
        result = {}
        for p in Usd.PrimRange(robot, Usd.TraverseInstanceProxies()):
            if p.IsA(UsdPhysics.RevoluteJoint) or p.HasAPI(UsdPhysics.CollisionAPI) or p.HasAPI(UsdPhysics.RigidBodyAPI):
                # Imported authored values and all physics/geometry values
                # remain unchanged. Removing an inactive alternate root API
                # intentionally removes its unauthored schema-only defaults.
                result[str(p.GetPath())] = {
                    a.GetName(): str(a.Get()) for a in p.GetAttributes()
                    if a.HasAuthoredValueOpinion() or not a.GetName().startswith('newton:')}
        return result
    before = inventory()
    record = apply_base_constraint(stage, robot, {'robot': {'hand_id': 'sharpa_w01_right'},
                                                'isaacsim': {'robot_base_constraint': MODE}})
    assert before == inventory()
    assert len([p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]) == 29
    roots = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    assert roots == [record['articulation_path']]
    parsed = UsdPhysics.LoadUsdPhysicsFromRange(stage, ['/World'])
    desc = parsed[UsdPhysics.ObjectType.Articulation][1][0]
    assert list(map(str, desc.rootPrims)) == roots
    assert all(p.read_bytes() == content for p, content in before_files.items())


def test_imported_alternate_schema_cannot_reintroduce_floating_root():
    pytest.importorskip('pxr.Usd')
    from pxr import Usd, UsdGeom, UsdPhysics
    if not Usd.SchemaRegistry().FindAppliedAPIPrimDefinition('NewtonArticulationRootAPI'):
        pytest.skip('Register the pinned runtime Newton USD schema metadata for this regression test')
    if not ASSET.is_file(): pytest.skip('Pinned SHARPA asset not mounted')
    stage = Usd.Stage.CreateInMemory()
    robot = UsdGeom.Xform.Define(stage, '/World/Robot').GetPrim()
    robot.GetReferences().AddReference(str(ASSET))
    base = stage.GetPrimAtPath('/World/Robot/Geometry/base/fr3_link0')
    # Reproduce the exact failed prototype: removing only the standard API
    # leaves the imported alternate API's built-in root contribution active.
    base.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    assert base.HasAPI(UsdPhysics.ArticulationRootAPI)
    record = apply_base_constraint(stage, robot, {'robot': {'hand_id': 'sharpa_w01_right'},
                                                'isaacsim': {'robot_base_constraint': MODE}})
    assert record['removed_root_contributor_schemas'] == ['NewtonArticulationRootAPI']
    assert not base.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert 'NewtonMassAPI' in base.GetAppliedSchemas()
    assert base.GetAttribute('newton:inertia').Get() is not None
    assert [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)] == [record['articulation_path']]


@pytest.mark.parametrize('actual,requested', [(False, False), (True, False), (True, True), (False, True)])
def test_native_flag_is_read_and_opt_in_fails_closed(actual, requested):
    meta = SimpleNamespace(fixed_base=actual, dof_names=['q'], link_names=['fr3_link0', 'link'])
    view = SimpleNamespace(count=1, get_metatype=lambda index: meta)
    info = {'articulation_path': '/World/Root'}
    if requested: info['robot_base_constraint'] = {'mode': MODE, 'expected_dof_names': ['q'],
                                                 'expected_link_names': ['fr3_link0', 'link']}
    if requested and not actual:
        with pytest.raises(RuntimeError, match='did not create'):
            runtime_base_check(view, info)
    else:
        check = runtime_base_check(view, info)
        assert check['all_fixed_base'] is actual
        assert check['requested_mode'] == (MODE if requested else None)


@pytest.mark.parametrize('actual_links,actual_dofs', [(['fr3_link0'], ['q']),
                                                   (['fr3_link0', 'link'], ['different']),
                                                   (['link', 'fr3_link0'], ['q'])])
def test_native_fixed_flag_alone_does_not_prove_expected_topology(actual_links, actual_dofs):
    meta = SimpleNamespace(fixed_base=True, dof_names=actual_dofs, link_names=actual_links)
    view = SimpleNamespace(count=1, get_metatype=lambda index: meta)
    info = {'articulation_path': '/World/Root', 'robot_base_constraint': {
        'mode': MODE, 'expected_dof_names': ['q'], 'expected_link_names': ['fr3_link0', 'link']}}
    with pytest.raises(RuntimeError, match='inventory or base link changed'):
        runtime_base_check(view, info)


def test_legacy_composite_is_read_only_and_has_per_articulation_flags():
    def view(fixed):
        return SimpleNamespace(count=1, get_metatype=lambda index: SimpleNamespace(
            fixed_base=fixed, dof_names=['q'], link_names=['base']))
    check = runtime_base_check(SimpleNamespace(views=[view(True), view(False)]),
                               {'articulation_path': ['/Left', '/Right']})
    assert [row['fixed_base'] for row in check['articulations']] == [True, False]
