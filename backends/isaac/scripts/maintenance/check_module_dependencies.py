"""Check Python import boundaries without importing project code or simulators.

Checks ordinary imports (including nested/type-check-only imports) and literal
importlib.import_module / __import__ calls. Computed import names are outside
this static check; this is an architectural regression check, not a sandbox.
"""
from __future__ import annotations

import argparse
import ast
from importlib.util import resolve_name
from pathlib import Path


PROTECTED_ROOTS = ('bench', 'scenes')

FORBIDDEN_ROOTS = frozenset(('agent', 'interface', 'simulation', 'data_tools'))


def import_references(source: str, *, package: str):
    """Return line/module pairs; aliases affect syntax, never module ownership."""
    tree = ast.parse(source)
    importlib_aliases = {'importlib'}
    importer_aliases = {'__import__'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == 'importlib':
                    importlib_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == 'importlib':
            importer_aliases.update(alias.asname or alias.name for alias in node.names
                                    if alias.name == 'import_module')

    references = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            references.update((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            name = '.' * node.level + (node.module or '')
            if node.level:
                name = resolve_name(name, package)
            references.add((node.lineno, name))
            references.update((node.lineno, name + '.' + alias.name)
                              for alias in node.names if alias.name != '*')
        elif isinstance(node, ast.Call):
            function = node.func
            importer = (isinstance(function, ast.Name) and function.id in importer_aliases
                        or isinstance(function, ast.Attribute) and function.attr == 'import_module'
                        and isinstance(function.value, ast.Name) and function.value.id in importlib_aliases)
            name = node.args[0] if node.args else next(
                (item.value for item in node.keywords if item.arg == 'name'), None)
            if importer and isinstance(name, ast.Constant) and isinstance(name.value, str):
                target = name.value
                if target.startswith('.'):
                    specified = node.args[1] if len(node.args) > 1 else next(
                        (item.value for item in node.keywords if item.arg == 'package'), None)
                    if not isinstance(specified, ast.Constant) or not isinstance(specified.value, str):
                        continue  # A dynamic package cannot be resolved statically.
                    target = resolve_name(target, specified.value)
                references.add((node.lineno, target))
    return sorted(references)


def check_dependencies(project_root: Path):
    """Collect forbidden edges; the runtime may compose scenes, never Agents."""
    root = Path(project_root)
    violations = []
    for directory in PROTECTED_ROOTS:
        for path in sorted((root / directory).rglob('*.py')):
            relative = path.relative_to(root)
            package = '.'.join(relative.parent.parts)
            for line, module in import_references(path.read_text(), package=package):
                if module.split('.', 1)[0] in FORBIDDEN_ROOTS:
                    violations.append(f'{relative.as_posix()}:{line}: forbidden dependency {module}')
    return violations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    violations = check_dependencies(args.project_root)
    if violations:
        print('\n'.join(violations))
        return 1
    print('Module dependency boundaries passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
