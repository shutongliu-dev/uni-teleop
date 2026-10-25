"""Export original visual workcell recipes once; no simulation is run.

Run with the existing project interpreter. Isaac Sim consumes the resulting
plain mesh arrays and never imports or runs the old physics engine.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from bench.paths import DATA_ROOT, load_json


def export(output=Path(f'{DATA_ROOT}/assets/scenes/workcell_details/isaacsim_6_1_v1')):
    from bench.simulation.assets import workcell_details as source
    source_path = source.DEFAULT_ASSET_ROOT / 'workcell_details.json'
    config = load_json(source_path)
    arrays, records = {}, []
    palette = config['palette']
    for recipe in config['assemblies']:
        assembly = source._Assembly(recipe, palette)
        source._RECIPES[recipe['kind']](assembly)
        for material, parts in assembly.parts.items():
            i = len(records)
            arrays[f'v_{i}'] = np.concatenate(parts['vertices']).astype(np.float32)
            arrays[f'f_{i}'] = np.concatenate(parts['faces']).astype(np.int32)
            arrays[f'n_{i}'] = np.concatenate(parts['normals']).astype(np.float32)
            records.append({'name': recipe['name'] + '_' + material, 'color': palette[material],
                            'roughness': .27 if material in ('silver', 'steel') else .66,
                            'metallic': .8 if material in ('silver', 'steel') else 0.,
                            'visual_only': True, 'role': recipe['role']})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'meshes.npz', **arrays)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report = {'schema_version': 'isaac-workcell-static-meshes-v1', 'license': 'CC0-1.0',
              'source_recipe': str(source_path), 'source_recipe_sha256': sha(source_path),
              'source_generator': str(Path(source.__file__)), 'source_generator_sha256': sha(source.__file__),
              'source_license': str(source.DEFAULT_ASSET_ROOT / 'SOURCE.json'),
              'mesh_npz_sha256': sha(output / 'meshes.npz'), 'meshes': records,
              'no_simulator_rollout': True, 'usage': 'Static non-colliding decorative geometry only.'}
    (output / 'mesh_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(output / 'mesh_manifest.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(f'{DATA_ROOT}/assets/scenes/workcell_details/isaacsim_6_1_v1'))
    export(parser.parse_args().output)
