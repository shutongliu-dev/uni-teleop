from types import SimpleNamespace

import numpy as np
from pxr import Usd, UsdGeom

from scenes.workrooms.kitchen import geometry as g
from scenes.workrooms.kitchen.prop_finish import refine_visuals


def test_new_spoon_brushwork_follows_rest_frame_only_once():
    def author(rest):
        stage = Usd.Stage.CreateInMemory()
        row = {'id': 'spoon_0', 'body_path': '/World/Actors/spoon_0'}
        if rest is not None:
            row['natural_rest'] = {'source_to_rest_transform': rest.tolist()}
        material = g.material(stage, '/World/Materials/brass', [.5, .4, .2])
        original = g.box((.009, .0005, .0001), (0, .025, .0081))
        if rest is not None:
            original.apply_transform(rest)
        g.write_mesh(stage, row['body_path']+'/Visual/handle_engraving_0', original, material)
        author = SimpleNamespace(stage=stage, meta={'objects': [row]},
                                 materials={'brass': material})
        refine_visuals(author)
        return stage, original.vertices

    angle = -.015960336596593315
    transform = np.eye(4)
    transform[1:3, 1:3] = [[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]]
    transform[2, 3] = -.0007070128938283597
    old, _ = author(None)
    rested, original = author(transform)
    root = '/World/Actors/spoon_0/Visual/'
    for i in range(7):
        path = root+f'finish_brushed_handle_{i}'
        before = np.array(UsdGeom.Mesh(old.GetPrimAtPath(path)).GetPointsAttr().Get())
        after = np.array(UsdGeom.Mesh(rested.GetPrimAtPath(path)).GetPointsAttr().Get())
        expected = before@transform[:3, :3].T+transform[:3, 3]
        assert np.allclose(after, expected, atol=1e-8, rtol=0)
    unchanged = np.array(UsdGeom.Mesh(rested.GetPrimAtPath(
        root+'handle_engraving_0')).GetPointsAttr().Get())
    assert np.allclose(unchanged, original, atol=1e-8, rtol=0)
