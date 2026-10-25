"""Robot contact-material options: collider binding and compliant elastomer pads.

Release opt-ins (physics profile ``isaacsim_options``; absent keys keep the legacy
stage byte-for-byte):

* ``robot_material_binding: 'collider_instance_v1'`` binds ``robot_pad`` /
  ``robot_shell`` on each collider's instance prim. The legacy binding
  (``strongerThanDescendants`` on every link prim) resolves every collider to
  ``robot_shell``, because the imported links are nested and the root-most link's
  binding wins (runs/material_binding_01); with pad = shell = 0.5 in every release
  that was harmless, but any pad-only setting was a silent no-op.
* ``robot_pad_compliance`` makes the SHARPA ``*_elastomer`` pads a PhysX compliant
  contact (implicit spring-damper per contact point; PhysX stores it as a negative
  restitution). It requires the collider binding, so it cannot silently vanish.

This module has no pxr import; the stage functions import pxr lazily.
"""
from __future__ import annotations

import math
from collections import Counter

BINDINGS = ('collider_instance_v1',)
COMPLIANCE_MODELS = {
    # PhysX compliant contact: force spring (N/m, N*s/m) or acceleration spring (1/s^2, 1/s).
    'force_spring_v1': {'stiffness': 'N/m per contact point', 'damping': 'N*s/m per contact point', 'acceleration_spring': False},
    'acceleration_spring_v1': {'stiffness': '1/s^2 per contact point', 'damping': '1/s per contact point', 'acceleration_spring': True},
}
PAD_LINK_SUFFIX = '_elastomer'


def material_kind(link_name):
    """Pad material for the SHARPA elastomer links, shell for every other robot link."""
    return 'robot_pad' if link_name.endswith(PAD_LINK_SUFFIX) else 'robot_shell'


def material_options(options):
    """Validate the robot material opt-ins; returns {'binding', 'pad_compliance'} (None = legacy)."""
    options = options or {}
    binding = options.get('robot_material_binding')
    if binding is not None and binding not in BINDINGS:
        raise ValueError(f'Unknown robot_material_binding {binding!r}; expected one of {BINDINGS}')
    materials = options.get('robot_contact_materials')
    if materials is not None:
        if not isinstance(materials, dict) or set(materials) != {'pad_friction', 'shell_friction'}:
            raise ValueError('robot_contact_materials needs exactly pad_friction and shell_friction')
        if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in materials.values()):
            raise ValueError('Robot pad/shell friction must be finite non-negative numbers')
        if binding is None and materials['pad_friction'] != materials['shell_friction']:
            raise ValueError('Different pad/shell friction needs robot_material_binding collider_instance_v1; '
                             'the legacy link binding resolves the pads to robot_shell')
    elif binding is not None:
        raise ValueError('robot_material_binding needs robot_contact_materials')
    compliance = options.get('robot_pad_compliance')
    if compliance is not None:
        if set(compliance) != {'model', 'stiffness', 'damping'}:
            raise ValueError('robot_pad_compliance needs exactly model, stiffness and damping')
        if compliance['model'] not in COMPLIANCE_MODELS:
            raise ValueError(f'Unknown robot_pad_compliance model {compliance["model"]!r}')
        stiffness, damping = compliance['stiffness'], compliance['damping']
        if not all(type(x) in (int, float) and math.isfinite(x) for x in (stiffness, damping)) or stiffness <= 0 or damping < 0:
            raise ValueError('robot_pad_compliance stiffness must be finite and positive, damping finite and non-negative')
        if options.get('robot_contact_materials') is None:
            raise ValueError('robot_pad_compliance needs robot_contact_materials (the pad material it modifies)')
        if binding != 'collider_instance_v1':
            raise ValueError('robot_pad_compliance needs robot_material_binding collider_instance_v1; '
                             'the legacy link binding resolves the pads to robot_shell, so the compliance would not apply')
        compliance = {'model': compliance['model'], 'stiffness': float(stiffness), 'damping': float(damping),
                      'units': {k: v for k, v in COMPLIANCE_MODELS[compliance['model']].items() if k != 'acceleration_spring'}}
    return {'binding': binding, 'pad_compliance': compliance}


def bind_collider_instances(stage, robot_path, link_names, pad, shell):
    """Bind pad/shell on each collider's nearest non-instance-proxy prim; returns {holder path: kind}."""
    from pxr import Sdf, Usd, UsdPhysics, UsdShade
    root = Sdf.Path(robot_path)
    owners = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        link = prim
        while link.GetPath() != root and not link.HasAPI(UsdPhysics.RigidBodyAPI):
            link = link.GetParent()
        if link.GetName() not in link_names:
            raise RuntimeError(f'Robot collider outside a robot link: {prim.GetPath()}')
        holder = prim
        while holder.IsInstanceProxy():
            holder = holder.GetParent()
        kind = material_kind(link.GetName())
        if owners.setdefault(str(holder.GetPath()), kind) != kind:
            raise RuntimeError(f'Instance prim {holder.GetPath()} holds pad and shell colliders')
    for path, kind in sorted(owners.items()):
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(
            pad if kind == 'robot_pad' else shell, UsdShade.Tokens.strongerThanDescendants, 'physics')
    # A holder nested inside another holder (e.g. a collider authored directly on a link
    # prim) would let the outer strongerThanDescendants binding win again: verify every
    # collider resolves to its own kind.
    expected = {'robot_pad': pad.GetPath(), 'robot_shell': shell.GetPath()}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        holder = prim
        while holder.IsInstanceProxy():
            holder = holder.GetParent()
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
        if not material or material.GetPath() != expected[owners[str(holder.GetPath())]]:
            raise RuntimeError(f'Robot collider {prim.GetPath()} does not resolve to {owners[str(holder.GetPath())]}')
    return owners


def resolved_collider_materials(stage, robot_path):
    """USD-resolved physics material per robot collider, counted by material path."""
    from pxr import Usd, UsdPhysics, UsdShade
    counts = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path), Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
            key = str(material.GetPath()) if material else 'none'
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def collider_material_inventory(stage, robot_path):
    """Read enabled collider ownership and USD resolution, including instance proxies.

    USD traversal order is NOT a PhysX shape index. Keep the paths as a per-link
    set; the tensor reader below identifies actual shapes by body path + index.
    """
    from pxr import Sdf, Usd, UsdPhysics, UsdShade
    root = Sdf.Path(robot_path)
    records = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        link = prim
        while link.GetPath() != root and not link.HasAPI(UsdPhysics.RigidBodyAPI):
            link = link.GetParent()
        if not link.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f'Enabled robot collider has no rigid link: {prim.GetPath()}')
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
        records.append({'collider_path': str(prim.GetPath()), 'link_path': str(link.GetPath()),
                        'link_name': link.GetName(), 'kind': material_kind(link.GetName()),
                        'material_path': str(material.GetPath()) if material else None})
    return sorted(records, key=lambda item: item['collider_path'])


def apply_pad_compliance(pad, compliance):
    from pxr import PhysxSchema
    api = PhysxSchema.PhysxMaterialAPI.Apply(pad.GetPrim())
    api.CreateCompliantContactStiffnessAttr(compliance['stiffness'])
    api.CreateCompliantContactDampingAttr(compliance['damping'])
    api.CreateCompliantContactAccelerationSpringAttr(COMPLIANCE_MODELS[compliance['model']]['acceleration_spring'])


def runtime_pad_check(material_rows, robot_materials):
    """Check the runtime PhysX materials (static, dynamic, restitution per robot shape) against the request.

    Returns a record; raises if a requested pad compliance is not live on exactly the pad colliders.
    Without a request it only records the count (historical scenes run unchanged).
    """
    compliance = (robot_materials or {}).get('pad_compliance')
    rows = [tuple(float(x) for x in row) for row in material_rows]
    if any(len(row) != 3 or not all(math.isfinite(x) for x in row) for row in rows):
        raise RuntimeError('Invalid runtime robot material rows')
    compliant = [row for row in rows if row[2] < 0.]
    record = {'robot_shapes': len(rows), 'compliant_shapes': len(compliant)}
    if not compliance:
        # Recorded, not enforced: historical scenes must keep running as they were.
        record['compliance_requested'] = False
        return record
    expected = (robot_materials.get('resolved_collider_materials') or {}).get('/World/PhysicsMaterials/robot_pad', 0)
    if expected <= 0 or len(compliant) != expected:
        raise RuntimeError(f'Pad compliance not live: {len(compliant)} compliant runtime shapes, {expected} pad colliders')
    if any(not math.isclose(-row[2], compliance['stiffness'], rel_tol=1e-6) for row in compliant):
        raise RuntimeError('Runtime pad compliance stiffness differs from the request')
    record.update(pad_colliders=expected, runtime_stiffness=compliance['stiffness'])
    return record


def check_link_materials(link_records, robot_materials, aggregate_rows):
    """Check every live shape on its identified link, not a global pad count.

    Each record comes from a single-body view, so max_shapes is that body's exact
    count, with no padding. This pure checker also guards the native readback's
    completeness against the independent articulation material array.
    """
    values = robot_materials['values']
    compliance = robot_materials.get('pad_compliance')
    if not link_records:
        raise RuntimeError('Missing native robot link material records')
    paths, seen_colliders, flattened = set(), set(), []
    counts = {'robot_pad': 0, 'robot_shell': 0}
    for record in link_records:
        path, name = record['link_path'], record['link_name']
        if path in paths or path.rsplit('/', 1)[-1] != name:
            raise RuntimeError(f'Duplicate or inconsistent native link identity: {path}')
        paths.add(path)
        kind = material_kind(name)
        if record['kind'] != kind:
            raise RuntimeError(f'Native link material kind mismatch: {path}')
        colliders = record['usd_colliders']
        if not colliders or any(row['link_path'] != path or row['link_name'] != name
                               or row['kind'] != kind
                               or row['material_path'] != f'/World/PhysicsMaterials/{kind}' for row in colliders):
            raise RuntimeError(f'USD collider material resolution differs from link intent: {path}')
        for collider in colliders:
            if collider['collider_path'] in seen_colliders:
                raise RuntimeError('Duplicate USD collider in material inventory')
            seen_colliders.add(collider['collider_path'])
        shapes = record['shapes']
        if record['native_view_count'] != 1 or record['native_shape_count'] != len(shapes) or not shapes:
            raise RuntimeError(f'Native single-body shape count mismatch: {path}')
        expected_friction = values['pad_friction' if kind == 'robot_pad' else 'shell_friction']
        expected_restitution = -compliance['stiffness'] if compliance and kind == 'robot_pad' else 0.
        for index, shape in enumerate(shapes):
            if shape['shape_index'] != index:
                raise RuntimeError(f'Noncontiguous native shape indices: {path}')
            row = tuple(float(x) for x in shape['material_properties'])
            expected = (expected_friction, expected_friction, expected_restitution)
            if len(row) != 3 or (row[2] < 0 and expected_restitution >= 0) or any(not math.isfinite(x) or not math.isclose(x, target, rel_tol=1e-6, abs_tol=1e-7)
                                    for x, target in zip(row, expected)):
                raise RuntimeError(f'Native material mismatch on {path} shape {index}: {row} != {expected}')
            flattened.append(row)
            counts[kind] += 1
    aggregate = [tuple(float(x) for x in row) for row in aggregate_rows]
    if Counter(flattened) != Counter(aggregate):
        raise RuntimeError('Per-link native shapes do not cover the articulation material rows')
    expected_pads = (robot_materials.get('resolved_collider_materials') or {}).get('/World/PhysicsMaterials/robot_pad', 0)
    if expected_pads > 0 and counts['robot_pad'] == 0:
        raise RuntimeError('USD pad colliders have no native pad shapes')
    return {'passed': True, 'scope': 'Every enabled robot collider link and every native shape on that link',
            'mapping': 'Exact native body path + body-local shape index; USD collider order is not a shape order',
            'robot_shapes': len(flattened), 'pad_shapes': counts['robot_pad'], 'shell_shapes': counts['robot_shell'],
            'compliant_shapes': sum(row[2] < 0 for row in flattened),
            'compliance_requested': compliance is not None,
            'friction_tolerance': {'relative': 1e-6, 'absolute': 1e-7}, 'links': link_records}


def runtime_material_check(stage, sim_view, robot_view, scene_info):
    """Read-only native verification after view initialization; no material writes.

    Explicit collider binding requires actual per-link values. Legacy equal-value
    releases retain their original stage and count-only diagnostic record.
    """
    def host_list(value):
        if hasattr(value, 'detach'):
            value = value.detach().cpu()
        return value.tolist()

    materials = scene_info.get('robot_contact_materials')
    aggregate = host_list(robot_view.get_material_properties())[0]
    if (materials or {}).get('binding', {}).get('mode') != 'collider_instance_v1':
        if materials and materials.get('values'):
            material_options({'robot_contact_materials': materials['values']})
        return runtime_pad_check(aggregate, materials)
    inventory = collider_material_inventory(stage, scene_info['robot_path'])
    by_link = {}
    for row in inventory:
        by_link.setdefault(row['link_path'], []).append(row)
    members = getattr(robot_view, 'views', [robot_view])
    native_link_paths = []
    for member in members:
        paths = member.link_paths
        if paths and not isinstance(paths[0], str):
            if len(paths) != 1:
                raise RuntimeError('Material check requires one articulation per native member')
            paths = paths[0]
        native_link_paths.extend(paths)
    if len(set(native_link_paths)) != len(native_link_paths):
        raise RuntimeError('Duplicate native articulation link paths')
    if not set(by_link) <= set(native_link_paths):
        raise RuntimeError('USD collider owners missing from native articulation links')
    records = []
    for path, colliders in sorted(by_link.items()):
        # A multi-body view pads to max_shapes. A one-body view has no padding;
        # require its native identity instead of assuming query order or matches.
        view = sim_view.create_rigid_body_view(path)
        if view.count != 1 or list(view.prim_paths) != [path]:
            raise RuntimeError(f'Native material query did not resolve one exact link: {path}')
        rows = host_list(view.get_material_properties())
        if len(rows) != 1 or len(rows[0]) != view.max_shapes:
            raise RuntimeError(f'Unexpected native material tensor dimensions: {path}')
        records.append({'link_path': path, 'link_name': colliders[0]['link_name'], 'kind': colliders[0]['kind'],
                        'native_view_count': view.count, 'native_shape_count': view.max_shapes,
                        'usd_colliders': colliders,
                        'shapes': [{'shape_index': i, 'material_properties': row} for i, row in enumerate(rows[0])]})
    return check_link_materials(records, materials, aggregate)
