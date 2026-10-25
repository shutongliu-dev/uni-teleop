"""Render the README task-status table from the scene catalog.

The table is generated, never edited by hand: `--write README.md` replaces the
marked block and `--check README.md` fails when it differs from the catalog.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from bench.paths import DATA_ROOT

BEGIN, END = '<!-- task-status:begin -->', '<!-- task-status:end -->'
ROUTES = {'single_agent': '/episodes', 'board_game': '/board-games',
          'two_player_adversarial': '/turn-games'}


def table():
    from agent.model_client.scene_tasks import FAMILIES, list_scene_tasks
    from scenes.workrooms.catalog import task_definition
    lines = ['| 场景 | 编号 | task_id | 名称 | 状态 | 会话 | 任务发布（相对 data/） |',
             '|---|---|---|---|---|---|---|']
    for family in FAMILIES:
        for row in list_scene_tasks(family):
            definition = task_definition(family, row['task_id'])
            number = row['task_number']
            label = ('—' if number is None else
                     f'L{number}' if family == 'mechanical_assembly' else f'task{number}')
            release = definition['source_release']
            published = '—' if release is None else f"`{Path(release['path']).relative_to(DATA_ROOT)}`"
            lines.append(f"| `{family}` | {label} | `{row['task_id']}` | {definition['title']} | "
                         f"{row['status']} | `{ROUTES[row['session_kind']]}` | {published} |")
    return '\n'.join(lines)


def render(text):
    start, end = text.index(BEGIN) + len(BEGIN), text.index(END)
    return f'{text[:start]}\n{table()}\n{text[end:]}'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--write', type=Path)
    action.add_argument('--check', type=Path)
    args = parser.parse_args(argv)
    path = args.write or args.check
    text = path.read_text()
    if args.write:
        path.write_text(render(text))
        return 0
    if render(text) != text:
        print(f'{path}: task table differs from the catalog; run --write {path}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
