"""Optional, hash-pinned visual USD layers for any rigid task asset.

The base asset is authored normally first. A presentation layer may shade its
meshes or add nonphysical detail, but cannot change existing geometry, transforms,
collision, mass or physical material bindings. There are no scene-name branches.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


EXISTING_PROPERTIES = {'visibility', 'normals', 'primvars:st', 'primvars:st:indices',
                       'doubleSided', 'material:binding'}
NEW_TYPES = {'Xform', 'Scope', 'Mesh', 'Material', 'Shader'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def presentation_dependencies(spec):
    value = spec.get('presentation')
    if value is None:
        return {}
    if (not isinstance(value, dict) or value.get('schema') != 'asset_presentation.v1'
            or not isinstance(value.get('dependencies_sha256'), dict)):
        raise ValueError('Invalid asset presentation descriptor')
    files = dict(value['dependencies_sha256'])
    path = value['usd_path']
    if path in files and files[path] != value['sha256']:
        raise ValueError('Conflicting presentation file hashes')
    files[path] = value['sha256']
    for name, expected in files.items():
        if not Path(name).is_absolute() or digest(name) != expected:
            raise ValueError('Presentation dependency changed: '+name)
    return files


def physical_fingerprint(stage, root_path):
    """Compare composed physical inputs, including geometry and resolved friction."""
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    rows = {}
    root = stage.GetPrimAtPath(root_path)
    for prim in Usd.PrimRange(root):
        physics = [s for s in prim.GetAppliedSchemas() if 'Physics' in s or 'Physx' in s]
        if not physics:
            continue
        row = {'schemas': physics, 'attrs': {}, 'ancestors': {}}
        for attr in prim.GetAttributes():
            name = attr.GetName()
            if name.startswith(('physics:', 'physx')) or name in (
                    'points', 'faceVertexCounts', 'faceVertexIndices', 'subdivisionScheme',
                    'size', 'radius', 'height', 'axis', 'extent'):
                row['attrs'][name] = repr(attr.Get())
        ancestor = prim
        while ancestor and ancestor.GetPath().HasPrefix(root.GetPath()):
            if ancestor.IsA(UsdGeom.Xformable):
                row['ancestors'][str(ancestor.GetPath())] = repr(
                    UsdGeom.Xformable(ancestor).GetLocalTransformation())
            ancestor = ancestor.GetParent()
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
            row['material'] = {'path': str(material.GetPath()) if material else None,
                               'attrs': {a.GetName(): repr(a.Get()) for a in material.GetPrim().GetAttributes()
                                         if a.GetName().startswith(('physics:', 'physx'))} if material else {}}
        rows[str(prim.GetPath())] = row
    encoded = json.dumps(rows, sort_keys=True).encode()
    return {'sha256': hashlib.sha256(encoded).hexdigest(), 'physical_prims': len(rows)}


def presentation_stage(stage):
    """A private stronger root with the original session at its original strength."""
    from pxr import Sdf, Usd, UsdGeom
    layer = Sdf.Layer.CreateAnonymous('presentation_preflight.usda')
    layer.subLayerPaths = [stage.GetRootLayer().identifier]
    scratch = Usd.Stage.Open(layer, stage.GetSessionLayer())
    # Stage-wide metrics are resolved from root/session metadata, not inherited
    # from this new root's sublayers. Without these opinions a metre/Z-up stage
    # becomes centimetre/Y-up here, shrinking physical-space UVs by 100 times.
    # Author only the private root; preserve the source and its session bytes.
    UsdGeom.SetStageMetersPerUnit(scratch, UsdGeom.GetStageMetersPerUnit(stage))
    UsdGeom.SetStageUpAxis(scratch, UsdGeom.GetStageUpAxis(stage))
    scratch.SetLoadRules(stage.GetLoadRules())
    scratch.SetPopulationMask(stage.GetPopulationMask())
    if stage.GetMutedLayers():
        scratch.MuteAndUnmuteLayers(stage.GetMutedLayers(), [])
    return scratch


def apply_presentation(stage, root_path, spec):
    from pxr import Sdf, Usd, UsdShade
    files = presentation_dependencies(spec)
    if not files:
        return None
    descriptor = spec['presentation']
    if stage.GetEditTarget().GetLayer() != stage.GetRootLayer():
        raise ValueError('Presentation loading requires the root edit target')
    # Read the bytes we verified, not an unrelated dirty Sdf cache entry.
    source = Sdf.Layer.OpenAsAnonymous(descriptor['usd_path'])
    if source is None or source.subLayerPaths:
        raise ValueError('Presentation must be one self-contained authoring layer')
    source_path = Sdf.Path(descriptor['prim_path'])
    if not source.GetPrimAtPath(source_path):
        raise ValueError('Presentation root is absent')
    # CopySpec remaps all connections and relationship paths within the subtree.
    remapped = Sdf.Layer.CreateAnonymous('remapped_presentation.usda')
    Sdf.CreatePrimInLayer(remapped, str(Sdf.Path(root_path).GetParentPath()))
    Sdf.CopySpec(source, source_path, remapped, root_path)
    overlay = Usd.Stage.Open(remapped)
    authored = list(Usd.PrimRange(overlay.GetPrimAtPath(root_path), Usd.PrimAllPrimsPredicate))
    before = physical_fingerprint(stage, root_path)
    existing = set()
    for prim in authored:
        path = prim.GetPath()
        target = stage.GetPrimAtPath(path)
        layer_spec = remapped.GetPrimAtPath(path)
        if (prim.HasAuthoredReferences() or prim.HasAuthoredPayloads()
                or prim.HasAuthoredInherits() or prim.HasAuthoredSpecializes()
                or prim.GetVariantSets().GetNames() or not prim.IsActive()):
            raise ValueError('Presentation cannot alter USD composition: '+str(path))
        if set(prim.GetAppliedSchemas()) - {'MaterialBindingAPI', 'NodeDefAPI'}:
            raise ValueError('Presentation cannot author physics schemas: '+str(path))
        properties = list(layer_spec.properties)
        if target:
            existing.add(str(path))
            if prim.GetTypeName() and prim.GetTypeName() != target.GetTypeName():
                raise ValueError('Presentation cannot replace prim type: '+str(path))
            if {p.name for p in properties} - EXISTING_PROPERTIES:
                raise ValueError('Presentation changes existing nonvisual properties: '+str(path))
        elif prim.GetTypeName() not in NEW_TYPES and (prim.GetTypeName() or properties):
            raise ValueError('Unsupported presentation detail type: '+str(path))
        for prop in properties:
            if remapped.ListTimeSamplesForPath(prop.path):
                raise ValueError('Presentation layers are static; time samples are unsupported')
            if prop.name.startswith(('physics:', 'physx', 'material:binding:physics')):
                raise ValueError('Presentation contains a physical property')
            for field in ('connectionPaths', 'targetPaths'):
                if prop.HasInfo(field):
                    targets = prop.GetInfo(field).GetAppliedItems()
                    if any(not t.HasPrefix(Sdf.Path(root_path)) for t in targets):
                        raise ValueError('Presentation target escapes the asset')
            if isinstance(prop, Sdf.AttributeSpec):
                paths = ([prop.default] if prop.typeName == Sdf.ValueTypeNames.Asset else
                         list(prop.default or []) if prop.typeName == Sdf.ValueTypeNames.AssetArray else [])
                for asset_path in paths:
                    if asset_path and asset_path.path not in files:
                        raise ValueError('Unpinned presentation texture: '+asset_path.path)

    def author(destination):
        layer = destination.GetEditTarget().GetLayer()
        for prim in authored:
            path = prim.GetPath()
            target = destination.GetPrimAtPath(path)
            if not target:
                target = destination.DefinePrim(path, prim.GetTypeName() or 'Scope')
            else:
                destination.OverridePrim(path)
            Sdf.CreatePrimInLayer(layer, destination.GetEditTarget().MapToSpecPath(path))
            if prim.HasAPI(UsdShade.MaterialBindingAPI):
                UsdShade.MaterialBindingAPI.Apply(target)
            for prop in remapped.GetPrimAtPath(path).properties:
                Sdf.CopySpec(remapped, prop.path, layer, destination.GetEditTarget().MapToSpecPath(prop.path))

    # A separate stronger layer tests the exact composed result without touching
    # the destination. In particular an all-purpose binding can double as the
    # physics material: reject that change before committing any visual property.
    scratch = presentation_stage(stage)
    author(scratch)
    if before != physical_fingerprint(scratch, root_path):
        raise ValueError('Presentation changed composed physical inputs')
    author(stage)
    if before != physical_fingerprint(stage, root_path):
        raise RuntimeError('Composed presentation differs from its preflight')
    return {'usd_path': descriptor['usd_path'], 'sha256': descriptor['sha256'],
            'existing_prims_shaded': len(existing), 'new_visual_prims': len(authored)-len(existing),
            'physical_inputs_unchanged': True, 'physical_fingerprint': before}
