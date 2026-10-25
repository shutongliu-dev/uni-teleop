"""Stage-authoring coordinate equivalence; native cooking remains independent."""
import sys
import types
import numpy as np
import pytest

from scripts.validation.validate_assembly_cooking import install_diagnostic_mesh_conditioning
from scripts.validation.mechanical_convex_geometry import convex_surface_error
from bench.simulation.isaacsim_backend import asset_catalog


def test_selected_mesh_coordinates_and_inverse_scale_preserve_parent_geometry(monkeypatch):
    name='bench.simulation.isaacsim_backend.scene'
    module=types.ModuleType(name)
    def mesh(stage,path,vertices,faces):return np.asarray(vertices)
    def transform(prim,position,quaternion,scale):return scale
    module.mesh,module.transform=mesh,transform
    monkeypatch.setitem(sys.modules,name,module)
    monkeypatch.setattr(asset_catalog,'load_task_asset',lambda *a,**k:{'collision_parts':[{}]})
    scene={'objects':[{'id':'peg','asset_id':'peg','asset_root':'unused'}],'fixtures':[]}
    rows,restore,paths=install_diagnostic_mesh_conditioning(scene,1000.)
    path='/World/Objects/peg/collision_000'
    points=np.array([[.01,.02,.03],[-.04,-.02,.08]])
    conditioned=module.mesh(None,path,points,None)
    prim=types.SimpleNamespace(GetPath=lambda:path)
    scale=module.transform(prim,[0.,0.,0.],[1.,0.,0.,0.])
    np.testing.assert_allclose(conditioned*scale,points,atol=1e-8)
    assert paths==set(rows) and rows[path]['parent_space_vertex_max_roundtrip_error_m']<1e-8
    assert module.mesh(None,'/World/unselected',points,None) is points
    with pytest.raises(ValueError,match='baked'):module.transform(prim,scale=[1.,1.,1.])
    restore()
    assert module.mesh is mesh and module.transform is transform


def test_cooked_error_tolerance_uses_parent_metres_not_conditioned_mesh_units():
    original=np.array([[0.,0.,0.],[.03,0.,0.],[0.,.03,0.],[0.,0.,.03]])
    perturbed=original.copy();perturbed[1,0]+=.00006
    linear=np.eye(3)*.001
    assert convex_surface_error(original,perturbed)>20e-6
    assert convex_surface_error(original*1000 @ linear,perturbed*1000 @ linear)==pytest.approx(convex_surface_error(original,perturbed))
