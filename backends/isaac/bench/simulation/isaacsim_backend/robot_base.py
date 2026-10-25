"""Opt-in ideal fixed-base articulations, authored only in the runtime layer.

An ArticulationRootAPI on a rigid body declares a floating articulation. For
an ideal rigid installation, the API belongs on that arm's existing joint to
world. Both dual-FR3 members are preflighted before either is edited. No contact
pair, collider, material, mass, inertia or solver setting is changed.
"""
from __future__ import annotations

MODE = 'native_fixed_base_v1'
REFERENCE = ('https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/'
             'dev_guide/rigid_bodies_articulations/articulations.html#usd-hierarchy-articulation-root')


def _prepare_arm(stage, scope, base_name):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    prims = list(Usd.PrimRange(scope))
    roots = [p for p in prims if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(roots) != 1 or not roots[0].HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError('Expected one imported rigid-body articulation root per arm')
    base = roots[0]
    if base.GetName() != base_name:
        raise ValueError(f'Unexpected FR3 base link; expected {base_name}')
    anchors = []
    for prim in prims:
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        a, b = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        # Each arm must retain its own rigid-body tree. A joint to a rigid
        # body on the peer arm or elsewhere cannot masquerade as an anchor.
        for path in list(a) + list(b):
            body = stage.GetPrimAtPath(path)
            if body and body.HasAPI(UsdPhysics.RigidBodyAPI) and not path.HasPrefix(scope.GetPath()):
                raise ValueError('Robot joint connects bodies across articulation ownership')
        if not prim.IsA(UsdPhysics.FixedJoint):
            continue
        other = a if b == [base.GetPath()] else b if a == [base.GetPath()] else None
        if other is None:
            continue
        if not other:
            anchors.append(UsdPhysics.FixedJoint(prim))
        elif len(other) == 1:
            parent = stage.GetPrimAtPath(other[0])
            if (parent and base.GetPath().HasPrefix(other[0])
                    and not parent.HasAPI(UsdPhysics.RigidBodyAPI)):
                anchors.append(UsdPhysics.FixedJoint(prim))
    if len(anchors) != 1:
        raise ValueError('Expected one existing fixed joint from FR3 base to world')
    joint = anchors[0]
    if not joint.GetJointEnabledAttr().Get() or joint.GetExcludeFromArticulationAttr().Get():
        raise ValueError('Base anchor must be enabled and included in the articulation')
    if joint.GetCollisionEnabledAttr().Get():
        raise ValueError('Unexpected enabled collision on the existing base joint')
    frame = UsdGeom.Xformable(base).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    if any(abs(float(x) - 1.) > 1.e-6 for x in Gf.Transform(frame).GetScale()):
        raise ValueError('Scaled robot base is unsupported')
    record = {'before': {'articulation_path': str(base.GetPath()),
                         'body0': list(map(str, joint.GetBody0Rel().GetTargets())),
                         'body1': list(map(str, joint.GetBody1Rel().GetTargets()))},
              'articulation_path': str(joint.GetPath()), 'root_body_path': str(base.GetPath()),
              'root_body_name': base_name, 'world_joint_path': str(joint.GetPath()),
              'body0': [], 'body1': [str(base.GetPath())],
              'initial_root_world_matrix': [list(row) for row in frame],
              'initial_root_position_m': list(frame.ExtractTranslation()),
              'initial_root_quaternion_wxyz': [frame.ExtractRotationQuat().GetReal(),
                                               *frame.ExtractRotationQuat().GetImaginary()],
              'expected_dof_names': sorted(p.GetName() for p in prims if p.IsA(UsdPhysics.RevoluteJoint)),
              'expected_link_names': sorted(p.GetName() for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI))}
    return base, joint, frame, record


def apply_base_constraint(stage, robot, scene):
    mode = scene.get('isaacsim', {}).get('robot_base_constraint')
    if mode is None:
        return None
    if mode != MODE:
        raise ValueError(f'Unknown robot_base_constraint: {mode!r}')
    hand = scene['robot'].get('hand_id', 'sharpa_w01_right')
    from .robot_instances import ROBOT_ID, checked_instances
    if hand not in ('sharpa_w01_right', 'dual_fr3_sharpa_w01', ROBOT_ID):
        raise ValueError('native_fixed_base_v1 requires single-arm or dual-arm SHARPA')
    from pxr import Gf, Usd, UsdPhysics

    layer = stage.GetEditTarget().GetLayer()
    if layer not in (stage.GetRootLayer(), stage.GetSessionLayer()):
        raise ValueError('Base constraint override requires a local runtime edit layer')
    if hand == ROBOT_ID:
        scopes = [(stage.GetPrimAtPath(robot.GetPath().AppendChild(name)), f'fr3_{name}_link0')
                  for name in checked_instances(scene['robot'])]
        if any(not scope for scope, _ in scopes):
            raise ValueError('Each declared robot instance requires its own reference root')
    elif hand == 'dual_fr3_sharpa_w01':
        scopes = [(stage.GetPrimAtPath(robot.GetPath().AppendChild(side.capitalize())), f'fr3_{side}_link0')
                  for side in ('left', 'right')]
        if any(not scope for scope, _ in scopes):
            raise ValueError('Dual FR3 requires distinct Left and Right reference roots')
    else:
        scopes = [(robot, 'fr3_link0')]
    # Complete all structural validation before editing either articulation.
    prepared = [_prepare_arm(stage, scope, name) for scope, name in scopes]
    original_roots = {str(p.GetPath()) for p in Usd.PrimRange(robot)
                      if p.HasAPI(UsdPhysics.ArticulationRootAPI)}
    if original_roots != {str(base.GetPath()) for base, _, _, _ in prepared}:
        raise ValueError('Unexpected extra articulation root outside the owned robot arms')
    records = []
    for base, joint, frame, record in prepared:
        jp = joint.GetPrim()
        properties = [a for a in base.GetAttributes()
                      if a.GetName().startswith('physxArticulation:') and a.HasAuthoredValueOpinion()]
        # This imported optional USD schema contributes the standard root API
        # even without its simulator. Remove only its root contribution.
        removed = []
        if 'NewtonArticulationRootAPI' in base.GetPrimTypeInfo().GetAppliedAPISchemas():
            if not base.RemoveAppliedSchema('NewtonArticulationRootAPI'):
                raise RuntimeError('Cannot remove imported alternate articulation-root schema')
            removed.append('NewtonArticulationRootAPI')
        base.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        UsdPhysics.ArticulationRootAPI.Apply(jp)
        if 'PhysxArticulationAPI' in base.GetAppliedSchemas():
            base.RemoveAPI('PhysxArticulationAPI')
            if not jp.ApplyAPI('PhysxArticulationAPI'):
                raise RuntimeError('Cannot move the PhysX articulation settings to the world joint')
        for attr in properties:
            target = jp.CreateAttribute(attr.GetName(), attr.GetTypeName(), attr.IsCustom(), attr.GetVariability())
            if attr.Get() is not None:
                target.Set(attr.Get())
            for time in attr.GetTimeSamples():
                target.Set(attr.Get(time), time)
        joint.CreateBody0Rel().SetTargets([])
        joint.CreateBody1Rel().SetTargets([base.GetPath()])
        # Fixed-base parsing ignores these frames; still make the world/base
        # anchors coherent with the independently rotated installation poses.
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*frame.ExtractTranslation()))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(frame.ExtractRotationQuat()))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.))
        record['removed_root_contributor_schemas'] = removed
        records.append(record)
    actual_roots = {str(p.GetPath()) for p in Usd.PrimRange(robot)
                    if p.HasAPI(UsdPhysics.ArticulationRootAPI)}
    if actual_roots != {r['articulation_path'] for r in records}:
        raise RuntimeError(f'Unexpected composed articulation-root contributors: {actual_roots}')
    return {'mode': mode, **(records[0] if len(records) == 1 else {'articulations': records}),
            'edit_layer': layer.identifier, 'reference': REFERENCE,
            'mount_model': 'Ideal rigid fixed installation; no bolt preload or compliance model.',
            'contact_filtering_changed': False, 'geometry_changed': False,
            'solver_settings_changed': False, 'materials_changed': False,
            'native_verification_required': True, 'physics_accepted': False}


def runtime_base_check(robot_view, scene_info):
    """Read actual native per-arm topology and installation pose."""
    import numpy as np

    views = getattr(robot_view, 'views', [robot_view])
    paths = scene_info['articulation_path']
    paths = paths if isinstance(paths, list) else [paths]
    if len(paths) != len(views):
        raise RuntimeError('Native base verification articulation count mismatch')
    requested = scene_info.get('robot_base_constraint')
    records = requested.get('articulations', [requested]) if requested else []
    if requested and len(records) != len(paths):
        raise RuntimeError('Fixed-base request articulation count mismatch')
    by_path = {record.get('articulation_path', paths[0]): record for record in records}
    if requested and set(by_path) != set(paths):
        raise RuntimeError('Fixed-base request articulation paths mismatch')
    rows = []
    for path, view in zip(paths, views):
        if view.count != 1:
            raise RuntimeError('Native base verification needs one articulation per view')
        meta = view.get_metatype(0)
        actual = bool(meta.fixed_base)
        row = {'articulation_path': path, 'fixed_base': actual,
               'dof_count': len(meta.dof_names), 'link_count': len(meta.link_names)}
        record = by_path.get(path)
        if record:
            if not actual:
                raise RuntimeError('native_fixed_base_v1 did not create a native fixed-base articulation')
            if (sorted(meta.dof_names) != record['expected_dof_names']
                    or sorted(meta.link_names) != record['expected_link_names']
                    or meta.link_names[0] != record.get('root_body_name', 'fr3_link0')):
                raise RuntimeError('Native fixed-base DOF/link inventory or base link changed')
            if 'initial_root_position_m' in record:
                pose = view.get_root_transforms()
                if hasattr(pose, 'detach'):
                    pose = pose.detach().cpu().numpy()
                pose = np.asarray(pose, dtype=float)[0]
                expected_q = np.asarray(record['initial_root_quaternion_wxyz'])
                actual_q = pose[[6, 3, 4, 5]]
                position_error = float(np.linalg.norm(pose[:3] - record['initial_root_position_m']))
                rotation_agreement = abs(float(actual_q @ expected_q))
                if position_error > 1.e-5 or abs(rotation_agreement - 1.) > 1.e-5:
                    raise RuntimeError('Native fixed-base installation pose differs from the authored root')
                row.update(root_position_m=pose[:3].tolist(), root_quaternion_wxyz=actual_q.tolist(),
                           root_pose_verified=True, root_position_error_m=position_error)
        rows.append(row)
    return {'source': 'PhysX ArticulationView.get_metatype(0).fixed_base and get_root_transforms()',
            'requested_mode': requested['mode'] if requested else None,
            'articulations': rows, 'all_fixed_base': all(row['fixed_base'] for row in rows),
            'verification_passed': True}
