"""Preflight the frozen tray contact identities against physical asset inventory."""
from copy import deepcopy


def validate_tray_contact_mapping(metadata, inventory):
    """Return the exact floor/wall mapping, or reject incomplete tray evidence.

    The asset manifest and metadata must already be hash-verified. This checks
    their declared static collider identities; native filter coverage and loaded
    contact remain separate runtime checks. Historical assets without trays keep
    their empty mapping and do not acquire new configuration requirements.
    """
    configuration = metadata.get('configuration') or {}
    collection = configuration.get('collection_trays', {})
    supply = configuration.get('supply_trays', [])
    mapping = metadata.get('contact_actor_paths', {})
    receipts = metadata.get('trays', [])
    if not collection and not supply:
        if mapping or receipts:
            raise ValueError('Physical tray contacts require configured trays')
        return {}
    if collection and supply:
        raise ValueError('A tower task cannot mix supply and collection tray contracts')
    declared = []
    for tray in collection.values():
        declared.append((tray['actor_id'], list(tray['wall_actor_ids']), tray))
    for tray in supply:
        floor = 'external/' + tray['id']
        declared.append((floor, [actor for actor in tray['contact_actor_ids'] if actor != floor], tray))
        if tray['contact_actor_ids'].count(floor) != 1:
            raise ValueError('Supply tray must declare exactly one physical floor actor')
    ids = [actor for floor, walls, _ in declared for actor in [floor, *walls]]
    if (any(not isinstance(actor, str) or not actor or actor in ('table', 'floor') for actor in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError('Tray floor and wall actors must have distinct nonreserved identities')
    if not isinstance(mapping, dict) or set(mapping) != set(ids):
        raise ValueError('Tray contact mapping must contain every configured floor and wall exactly once')
    if len(receipts) != len(declared):
        raise ValueError('Each configured tray requires one physical asset receipt')
    root = metadata['root_prim_path'] + '/Trays/'
    colliders = set(inventory['colliders'])
    bodies = [row['path'] for row in inventory['bodies']]
    expected = {}
    used_floors = set()
    for receipt in receipts:
        floor = receipt['floor_actor_id']
        match = next((row for row in declared if row[0] == floor), None)
        if match is None or floor in used_floors:
            raise ValueError('Physical tray receipt does not match one configured floor')
        used_floors.add(floor)
        _, walls, tray = match
        if (receipt['interior_bounds_xy_m'] != tray['interior_bounds_xy_m']
                or receipt['floor_z_m'] != tray['floor_z_m']
                or set(receipt['wall_actor_ids']) != set(walls)
                or len(receipt['wall_actor_ids']) != len(walls)):
            raise ValueError('Physical tray surface differs from its configured support')
        components = receipt['components']
        if len(components) != len(walls) + 1:
            raise ValueError('Physical tray components must account for every floor and wall')
        seen = set()
        for component in components:
            actor, path = component['actor_id'], component['prim_path']
            floor_component = actor == floor
            if actor not in [floor, *walls] or actor in seen:
                raise ValueError('Physical tray component identity is missing or duplicated')
            seen.add(actor)
            if (not isinstance(path, str) or not path.startswith(root)
                    or path not in colliders
                    or any(path == body or path.startswith(body + '/') for body in bodies)
                    or component['collision'] is not True
                    or component['dynamic'] is not False or component['kinematic'] is not False):
                raise ValueError('Tray contact actor must bind an enabled static physical collider')
            name = component['name']
            if (component['support_role'] != ('floor' if floor_component else 'wall')
                    or (name != 'Floor' if floor_component else not name.startswith('Walls/'))
                    or path != receipt['prim_path'] + '/' + name):
                raise ValueError('Tray floor and wall contact paths must preserve physical support roles')
            expected[actor] = path
    if (mapping != expected or len(set(mapping.values())) != len(ids)
            or set(metadata.get('static_collision_paths', [])) != set(expected.values())
            or {path for path in colliders if path.startswith(root)} != set(expected.values())):
        raise ValueError('Tray contact mapping differs from the complete physical collider inventory')
    return deepcopy(expected)
