"""Only collider local coordinates change; physical dimensions remain exact."""
from types import SimpleNamespace
from copy import deepcopy
import numpy as np
import pytest
from pxr import Usd,UsdGeom
from scenes.workrooms.kitchen.assets import author_task
from scenes.workrooms.kitchen.utensil_cooking import condition_spoon_convex_coordinates
from scripts.validation.kitchen_asset_identity import _property_graph


def test_conditioned_spoon_has_exact_physical_corners_and_unchanged_body_properties():
    stage=Usd.Stage.CreateInMemory()
    meta=author_task(stage,'utensil_sorting',prop_profile='household_v1',utensil_contact_profile='floor_grid_2mm_fullconvex_v5')
    before=_property_graph(stage,meta['actors'].values());oldmeta=deepcopy(meta)
    targets={p for r in meta['objects'] if r['role']=='spoon' for p in r['collider_paths']}
    original={p:np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath(p)).GetPointsAttr().Get()).copy() for p in targets}
    condition_spoon_convex_coordinates(SimpleNamespace(stage=stage,meta=meta))
    after=_property_graph(stage,meta['actors'].values())
    assert before.keys()==after.keys()
    for p,old in before.items():
        if p not in targets:assert old==after[p];continue
        new=deepcopy(after[p]);attrs=new['attributes'];attrs.pop('xformOp:scale');attrs['xformOpOrder']=old['attributes']['xformOpOrder'];attrs['points']=old['attributes']['points'];attrs['extent']=old['attributes']['extent'];assert new==old
        mesh=UsdGeom.Mesh(stage.GetPrimAtPath(p));matrix=np.asarray(mesh.GetLocalTransformation()).T
        points=np.asarray(mesh.GetPointsAttr().Get())
        np.testing.assert_array_equal(points@matrix[:3,:3].T+matrix[:3,3],original[p])
    for row in meta['objects']:row.pop('collision_coordinate_conditioning',None)
    assert oldmeta==meta
    with pytest.raises(ValueError,match='untransformed'):
        condition_spoon_convex_coordinates(SimpleNamespace(stage=stage,meta=meta))
