"""Temporary presentation-only USD layer; never edit physics or recorded state."""
from contextlib import contextmanager
import re


class ArmVisibility:
    def __init__(self, env):
        from pxr import Sdf, Usd, UsdGeom
        self.stage = env.stage
        self.layer = Sdf.Layer.CreateAnonymous('operator-arm-hidden.usda')
        links = env.scene_info['robot_link_paths']
        arms = {str(path) for name, path in links.items()
                if re.fullmatch(r'fr3_(?:[a-z]+_)?link[0-8]', name)}
        if not arms:
            raise ValueError('Operator arm hiding requires named Franka arm links')
        others = {str(path) for path in links.values()} - arms
        selected = set()
        for root in arms:
            for prim in Usd.PrimRange(self.stage.GetPrimAtPath(root)):
                path = str(prim.GetPath())
                # Do not hide a hand or any ancestor that would hide a hand.
                if any(path == other or path.startswith(other+'/') or other.startswith(path+'/')
                       for other in others):
                    continue
                if prim.IsInstanceProxy():
                    continue
                if prim.IsA(UsdGeom.Gprim) or prim.IsInstance():
                    selected.add(path)
        if not selected:
            raise ValueError('No independently hideable arm render geometry found')
        self.paths = sorted(selected)
        # Author only visibility overrides into a removable session sublayer.
        with self.hidden(), Usd.EditContext(self.stage, self.layer):
            for path in self.paths:
                UsdGeom.Imageable(self.stage.GetPrimAtPath(path)).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)

    @contextmanager
    def hidden(self):
        session = self.stage.GetSessionLayer()
        identifier = self.layer.identifier
        session.subLayerPaths.insert(0, identifier)
        try:
            yield
        finally:
            session.subLayerPaths.remove(identifier)
