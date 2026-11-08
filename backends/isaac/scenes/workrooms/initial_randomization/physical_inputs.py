"""Checks on actual scene actor declarations and canonical physical inputs."""
from copy import deepcopy
from pathlib import Path, PurePosixPath

from bench.paths import usd_layer_identity

from .common import read_json


def object_actor_path(spec):
    """Use the frozen catalog's native placement form, without importing USD.

    An anchored USD keeps its source transform below a placement wrapper. Its
    dynamic actor is the declared source body below that anchor, not the wrapper.
    Native validation independently verifies the actual composed actor path.
    """
    wrapper = '/World/Objects/'+spec['id']
    if 'asset_root' not in spec or 'asset_id' not in spec:
        return wrapper
    catalog = read_json(Path(spec['asset_root'])/'manifest.json')
    asset = catalog['assets'][spec['asset_id']]
    if asset.get('placement_form') != 'anchored':
        return wrapper
    root, body = asset.get('usd_default_prim'), asset.get('source_rigid_body_prim')
    for value in (root, body):
        if (not isinstance(value, str) or not value.startswith('/') or value == '/'
                or str(PurePosixPath(value)) != value or '..' in PurePosixPath(value).parts):
            raise ValueError('Anchored USD requires explicit normalized source prim paths')
    if body != root and not body.startswith(root+'/'):
        raise ValueError('Anchored rigid body escapes its declared source root')
    return wrapper+'/'+PurePosixPath(root).name+body[len(root):]


def actor_paths(scene):
    result = {}
    groups = [scene.get('native_task_bundle', {}).get('rigid_bodies', {})]
    groups.extend(row.get('rigid_bodies', {}) for row in scene.get('task_layers', []))
    groups.append({row['id']: object_actor_path(row) for row in scene.get('objects', [])})
    for group in groups:
        if set(group) & result.keys():
            raise ValueError('Scene declares duplicate dynamic actors')
        result.update(group)
    if not result or len(set(result.values())) != len(result):
        raise ValueError('Scene requires unique actual dynamic actors and paths')
    return result


def check_actor_inventory(parent_scene, scene, poses):
    actual = actor_paths(scene)
    if actual != actor_paths(parent_scene):
        raise ValueError('Randomization changed the original actor inventory')
    if actual != {name: row['prim_path'] for name, row in poses.items()}:
        raise ValueError('Private reset actors differ from actual scene declarations')


def check_canonical_physics(parent_scene, scene):
    """Seed zero keeps source USD bytes and plain object/fixture declarations.

    A copied USD may have a new absolute path; its pinned content is identical.
    Observer code and dependency inventories describe the new input wrapper.
    """
    for key in ('objects', 'fixtures'):
        if scene.get(key) != parent_scene.get(key):
            raise ValueError('Seed zero changed canonical '+key)

    def physical(container):
        value = deepcopy(container)
        for key in ('usd_path', 'dependencies_sha256', 'observer', 'reduced_dynamics'):
            value.pop(key, None)
        return value

    if physical(scene.get('native_task_bundle', {})) != physical(parent_scene.get('native_task_bundle', {})):
        raise ValueError('Seed zero changed canonical task USD/physics declarations')
    def layer(row):
        # dexbench: a copied text layer keeps its source bytes up to its layer-relative asset
        # paths (the package relativizes them per location); compare that layer identity.
        value = physical(row)
        path = row.get('usd_path')
        if isinstance(path, str) and isinstance(value.get('sha256'), str) and Path(path).is_file():
            from hashlib import sha256
            if sha256(Path(path).read_bytes()).hexdigest() == value['sha256']:
                value['sha256'] = usd_layer_identity(path)
        return value

    if ([layer(row) for row in scene.get('task_layers', [])]
            != [layer(row) for row in parent_scene.get('task_layers', [])]):
        raise ValueError('Seed zero changed canonical task layer bytes/physics declarations')
