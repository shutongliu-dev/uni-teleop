"""Hand-written status drifts; the README table and data manifest are generated."""
from pathlib import Path
import re

from bench.paths import DATA_ROOT, load_json
from scenes.workrooms.catalog import list_scenes, list_tasks
from scripts.maintenance.task_status import render

ROOT = Path(__file__).resolve().parents[1]
MACHINE_PATH = re.compile(r'/cpfs\d+/')


def test_readme_task_table_matches_the_catalog():
    text = (ROOT / 'README.md').read_text()
    assert render(text) == text, 'run: python -m scripts.maintenance.task_status --write README.md'


def test_catalog_rooms_and_releases_are_in_the_data_manifest():
    files = load_json(ROOT / 'data/MANIFEST.json')['files']

    def listed(path):
        return str(Path(path).relative_to(DATA_ROOT)) in files

    for scene in list_scenes():
        assert listed(scene['environment']['usd_path']), scene['family']
        assert listed(scene['environment']['dependency_manifest_path']), scene['family']
    for task in list_tasks():
        assert listed(Path(task['source_release']['path']) / 'release_manifest.json'), task['task_id']


def test_sources_name_no_machine_paths():
    offenders = [str(path.relative_to(ROOT)) for top in ('bench', 'scenes', 'agent', 'scripts', 'tests', 'configs')
                 for path in (ROOT / top).rglob('*') if path.suffix in ('.py', '.json', '.sh', '.kit', '.toml')
                 and MACHINE_PATH.search(path.read_text(errors='ignore'))]
    assert offenders == []
