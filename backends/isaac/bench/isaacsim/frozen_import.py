"""Load validated frozen evaluator sources without reading or writing bytecode.

Only explicitly registered private evaluator namespaces are intercepted. Their
relative imports use the same source-only loader, independently of Python's
process-wide -B setting. Ordinary application imports retain their loaders.
"""
from importlib.abc import MetaPathFinder
from importlib.machinery import PathFinder, SourceFileLoader
from pathlib import Path
import sys


class _SourceOnlyLoader(SourceFileLoader):
    def get_code(self, fullname):
        return self.source_to_code(self.get_data(self.path), self.path)


class _FrozenFinder(MetaPathFinder):
    def __init__(self):
        self.roots = {}

    def find_spec(self, fullname, path=None, target=None):
        root = self.roots.get(fullname.partition('.')[0])
        if root is None or path is None:
            return None
        if not all(Path(part).resolve().is_relative_to(root) for part in path):
            raise ImportError('Frozen evaluator import escaped its validated source directory')
        spec = PathFinder.find_spec(fullname, path, target)
        if spec is None:
            return None
        if spec.loader is None and spec.origin is None:
            locations = spec.submodule_search_locations
            if not locations or not all(
                    Path(part).resolve().is_relative_to(root)
                    and Path(part).resolve() != root and Path(part).is_dir()
                    for part in locations):
                raise ImportError('Frozen evaluator namespace escaped its validated source directory')
            return spec
        if (not isinstance(spec.loader, SourceFileLoader) or spec.origin is None
                or not Path(spec.origin).resolve().is_relative_to(root)):
            raise ImportError('Frozen evaluator dependency must be validated Python source')
        spec.loader = _SourceOnlyLoader(fullname, spec.origin)
        return spec


_FINDER = _FrozenFinder()


def register_package(name, directory):
    if not name.startswith('_isaac_frozen_evaluator_') or '.' in name:
        raise ValueError('Expected a private frozen evaluator package')
    root = Path(directory).resolve()
    previous = _FINDER.roots.get(name)
    if previous is not None and previous != root:
        raise ValueError('Frozen evaluator package directory changed')
    _FINDER.roots[name] = root
    if _FINDER not in sys.meta_path:
        sys.meta_path.insert(0, _FINDER)


def source_loader(name, path):
    return _SourceOnlyLoader(name, str(path))
