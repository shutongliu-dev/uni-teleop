"""Install native-only reusable layouts using the shared manipulation catalog."""
import argparse
import hashlib
import json
from pathlib import Path

from bench.paths import DATA_ROOT, load_json


def prepare(assets_root):
    assets_root = Path(assets_root)
    layouts = Path(__file__).with_name('manipulation_scene_layouts.json')
    scenes = load_json(layouts)
    output = assets_root / 'scenes/manipulation_collection/v2'
    output.mkdir(parents=True, exist_ok=True)
    catalog = assets_root / 'task_parts/manipulation_collection/v2/manifest.json'
    if not catalog.is_file():
        raise FileNotFoundError(catalog)
    hashes = {}
    for name, scene in scenes.items():
        scene['robot']['assets_root'] = str(assets_root)
        for item in scene['objects'] + scene.get('fixtures', []):
            item['asset_root'] = str(catalog.parent)
        path = output / name
        data = json.dumps(scene, indent=2) + '\n'
        if path.exists() and path.read_text() != data:
            raise ValueError('Existing versioned scene differs: ' + str(path))
        path.write_text(data)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = {
        'source': 'Original reusable native Isaac scene layouts; CC0-1.0',
        'asset_catalog': str(catalog),
        'asset_catalog_sha256': hashlib.sha256(catalog.read_bytes()).hexdigest(),
        'scope': 'Scene examples only; frozen four-task release unchanged',
        'physics': 'All listed objects/fixtures collide; workcell decoration remains visual-only',
        'scene_sha256': hashes,
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'layout_sha256': hashlib.sha256(layouts.read_bytes()).hexdigest(),
    }
    (output / 'PROVENANCE.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return provenance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-root', type=Path, default=Path(f'{DATA_ROOT}/assets'))
    print(json.dumps(prepare(parser.parse_args().assets_root)))
