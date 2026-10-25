"""Explicit scene variants preserving task object identities and original files."""
from copy import deepcopy
import hashlib
from pathlib import Path

from bench.paths import DATA_ROOT, load_json

COLLECTION = Path(f'{DATA_ROOT}/assets/task_parts/manipulation_collection/v2')
VARIANTS = {'original_v1': 'Unchanged frozen task scene',
            'enriched_v2': 'Three extra dynamic household objects and two collision-enabled fixtures; separate difficulty stratum',
            'curated_v1': 'Task objects and necessary fixtures only; gray table and robot, without decoration or context props; separate scene stratum'}

# This is an explicit inventory, not a heuristic that deletes every object whose
# name happens to begin with "context". Missing or substituted task assets fail
# closed. The retained specifications, including poses and physical properties,
# are copied byte-for-byte at the JSON-value level.
CURATED_LAYOUTS = {
    'cube_workbench': {
        'title_zh': '方块基础工作台',
        'objects': {'cube': 'collab_cube_small'}, 'fixtures': {}, 'cloth': False,
        'arrangement_zh': '中央方块与公开目标标记；其余桌面留空。',
    },
    'ordered_kitting': {
        'title_zh': '顺序备料装盘',
        'objects': {'gear_large': 'gear_large', 'washer': 'washer', 'spool': 'spool'},
        'fixtures': {'tray_sorting': 'tray_sorting',
                     'gear_large_pickup_nest': 'pickup_pedestal',
                     'washer_pickup_nest': 'pickup_pedestal'}, 'cloth': False,
        'arrangement_zh': '前排三件备料，后排三槽托盘；保留原有取件支座。',
    },
    'loose_gear_assembly': {
        'title_zh': '双轴装配',
        'objects': {'washer': 'washer', 'gear_large': 'gear_large', 'gear_small': 'gear_small'},
        'fixtures': {'dual_shaft_fixture': 'dual_shaft_fixture',
                     'washer_pickup_nest': 'pickup_pedestal',
                     'gear_large_pickup_nest': 'pickup_pedestal',
                     'gear_small_pickup_nest': 'pickup_pedestal'}, 'cloth': False,
        'arrangement_zh': '前排垫圈与两只齿轮，后排双轴夹具；保留原有取件支座。',
    },
    'lined_tray_loading': {
        'title_zh': '衬布托盘',
        'objects': {'gear_large': 'gear_large', 'spool': 'spool'},
        'fixtures': {'tray_lining': 'tray_lining',
                     'gear_large_pickup_nest': 'pickup_pedestal'}, 'cloth': True,
        'arrangement_zh': '左侧自由布料，前右侧两件工件，后右侧空托盘。',
    },
    'cover_and_secure': {
        'title_zh': '覆布固定',
        'objects': {'wrap_cylinder': 'wrap_cylinder', 'near_bar': 'press_bar', 'far_bar': 'press_bar'},
        'fixtures': {'cylinder_cradle': 'cylinder_cradle'}, 'cloth': True,
        'arrangement_zh': '前侧布料与圆柱工件，中央定位座，后排两根压条。',
    },
}


def curate_scene(scene, layout_id=None):
    """Select one of five task inventories without moving any retained asset."""
    result = deepcopy(scene)
    by_kind = {}
    for kind in ('objects', 'fixtures'):
        items = result.get(kind, [])
        by_kind[kind] = {item['id']: item for item in items}
        if len(by_kind[kind]) != len(items):
            raise ValueError('Duplicate scene object/fixture IDs cannot be curated')
    if layout_id is None:
        matches = [name for name, layout in CURATED_LAYOUTS.items()
                   if all(set(layout[kind]).issubset(by_kind[kind])
                          for kind in ('objects', 'fixtures'))
                   and bool(result.get('cloth', {}).get('enabled')) == layout['cloth']]
        if len(matches) != 1:
            raise ValueError(f'Cannot identify exactly one curated task layout: {matches}')
        layout_id = matches[0]
    if layout_id not in CURATED_LAYOUTS:
        raise ValueError('Unknown curated layout: ' + str(layout_id))
    layout = CURATED_LAYOUTS[layout_id]
    if bool(result.get('cloth', {}).get('enabled')) != layout['cloth']:
        raise ValueError('Curated layout must preserve required cloth state')
    removed = {}
    for kind in ('objects', 'fixtures'):
        expected = layout[kind]
        for name, asset_id in expected.items():
            if name not in by_kind[kind] or by_kind[kind][name]['asset_id'] != asset_id:
                raise ValueError(f'Curated layout requires {kind}/{name} asset {asset_id}')
        removed[kind] = [item['id'] for item in result.get(kind, []) if item['id'] not in expected]
        result[kind] = [item for item in result.get(kind, []) if item['id'] in expected]
    result.setdefault('workcell_details', {})['enabled'] = False
    result.setdefault('workbench', {}).update(enabled=True, props=False)
    result['scene_variant'] = {
        'id': 'curated_v1', 'description': VARIANTS['curated_v1'],
        'layout_id': layout_id, 'title_zh': layout['title_zh'],
        'arrangement_zh': layout['arrangement_zh'],
        'ranking_stratum': 'curated_v1', 'frozen_source_mutated': False,
        'removed_instance_ids': removed,
        'workcell_details_enabled': False,
        'retained_asset_specs_unchanged': True,
        'task_initial_state_and_goals_unchanged': True,
        'shared_asset_files_removed': False,
    }
    return result


def apply_scene_variant(scene, variant_id='original_v1'):
    if variant_id not in VARIANTS:
        raise ValueError(f'Unknown scene_variant {variant_id!r}; choose {list(VARIANTS)}')
    result = deepcopy(scene)
    if variant_id == 'original_v1':
        return result
    if variant_id == 'curated_v1':
        return curate_scene(result)
    assets_root = Path(result['robot'].get('assets_root', f'{DATA_ROOT}/assets'))
    root = assets_root / 'task_parts/manipulation_collection/v2'
    path = root / 'manifest.json'
    catalog = load_json(path)['assets']
    names = {x['id'] for x in result['objects'] + result.get('fixtures', [])}
    additions = [('ceramic_mug', [.90,.43]), ('wooden_block', [.85,.65]), ('steel_can', [.70,.65])]
    for asset_id, xy in additions:
        name = 'context_' + asset_id
        if name in names:
            raise ValueError('Scene variant already applied or object ID conflicts: ' + name)
        bottom = catalog[asset_id]['bounds_m'][0][2]
        result['objects'].append({'id': name, 'asset_id': asset_id, 'asset_root': str(root),
                                  'position_m': [*xy, .751-bottom], 'quaternion_wxyz': [1,0,0,0], 'friction': .65})
    for asset_id, xy in [('parts_rack',[.02,.60]),('storage_tray',[.37,.65])]:
        name = 'context_' + asset_id
        if name in names:
            raise ValueError('Scene variant fixture ID conflicts: ' + name)
        result.setdefault('fixtures', []).append({'id':name,'asset_id':asset_id,'asset_root':str(root),
                                                  'position_m':[*xy,.75],'friction':.65})
    result['scene_variant'] = {'id': variant_id, 'description': VARIANTS[variant_id],
                               'asset_catalog': str(path), 'asset_catalog_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                               'frozen_source_mutated': False, 'ranking_stratum': variant_id,
                               'all_added_objects_and_fixtures_have_collision': True}
    return result
