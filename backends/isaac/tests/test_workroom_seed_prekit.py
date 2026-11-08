"""The native release consumer must work before Kit exposes USD packages."""
from pathlib import Path
import json
import subprocess
import sys

import pytest

from bench.paths import load_json

PINS = Path(__file__).resolve().parents[1]/'scenes/workrooms/initial_randomization/catalog_pins.json'
RELEASES = [(task, config, row['path']) for task, pin in load_json(PINS).items()
            for config, row in pin['releases'].items()]


PREKIT_RECHECK = r'''
import importlib, importlib.abc, json, sys
from pathlib import Path
class DenyUsd(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pxr' or fullname.startswith('pxr.'):
            raise AssertionError('Case import/composition attempted pre-Kit USD: '+fullname)
sys.meta_path.insert(0, DenyUsd())
for family in ('study', 'kitchen', 'laboratory', 'mechanical_assembly'):
    importlib.import_module('scenes.workrooms.'+family+'.seed_variants')
shared=importlib.import_module('scenes.workrooms.seed_geometry')
assert shared.checked_seed(0)==0
from bench.isaacsim.release import validate
from bench.paths import load_json
from scenes.workrooms.initial_randomization.release import _compose
pins=load_json(Path('scenes/workrooms/initial_randomization/catalog_pins.json'))
all_releases=[Path(ref['path']) for task in pins.values() for ref in task['releases'].values()]
assert len(all_releases)==len(set(all_releases))
releases=[Path(sys.argv[1])]
assert releases[0] in all_releases
count=0
for release in sorted(releases):
    path=release/'release_manifest.json'; manifest=validate(release)
    for seed,row in manifest['cases'].items():
        scene,task,initial,case,sources,dependencies=_compose(Path(manifest['base_release']['path']),
            row['asset_manifest'],manifest['task_id'],int(seed),manifest['configuration_id'])
        for label,value in [('scene',scene),('task',task),('initial_state',initial),('case',case)]:
            expected=load_json(release/row[label])
            assert value==expected,(path,seed,label)
        assert sources=={str(p.relative_to(release/'evaluator')):p.read_text() for p in (release/'evaluator').rglob('*.py')}
        count+=1
assert count==10,count
assert not any(k=='pxr' or k.startswith('pxr.') for k in sys.modules)
assert 'scenes.workrooms.living_room.seed_geometry' not in sys.modules
assert 'isaacsim' not in sys.modules
print(json.dumps({'passed':True,'releases':len(releases),'strict_equal_cases':count,'pxr_loaded':False,'kit_started':False}))
'''


@pytest.mark.parametrize('task_id,configuration,release', RELEASES,
                         ids=[task+'/'+config for task, config, _ in RELEASES])
def test_all_published_cases_compose_without_usd_import(task_id,configuration,release):
    root=Path(__file__).resolve().parents[1]
    # Wall-clock bound of the subprocess only, not a check. A parameterized seed release pins every
    # per-seed asset (gear: 33,475 files); its full validation plus the ten strict recompositions took
    # 386 s alone (2026-11-07), so the bound matches the 1800 s service start-up bound for such releases.
    run=subprocess.run([sys.executable,'-B','-c',PREKIT_RECHECK,release],cwd=root,
                       text=True,capture_output=True,timeout=1800)
    assert run.returncode==0,run.stdout+'\n'+run.stderr
    assert '"strict_equal_cases": 10' in run.stdout
