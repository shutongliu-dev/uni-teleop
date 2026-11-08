"""Publish a filterable comparison of hash-verified native configuration renders.

No pixels are generated or changed. Reports must describe completed static
native renders, and every release/image hash is checked before linking it.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import html
import json
from pathlib import Path

from bench.isaacsim.release import digest, dump
from bench.paths import load_json

FAMILIES = {'mechanical_assembly': '机械装配室', 'laboratory': '实验室',
            'study': '书房', 'card_room': '棋牌室', 'living_room': '客厅'}
CONFIGURATIONS = {'single_arm': '单臂', 'dual_same_side': '同侧双臂', 'dual_opposed': '对侧双臂'}


def collect(reports, output, preferred_reports=()):
    preferred_reports = {Path(path).resolve() for path in preferred_reports}
    if preferred_reports - {Path(path).resolve() for path in reports}:
        raise ValueError('Preferred report must be part of the input report list')
    records = {}
    owners = {}
    sources = []
    for path in reports:
        report = load_json(path)
        recovered = report.get('schema') == 'catalog_task_png_gallery_recovered.v1'
        if recovered:
            original_path = Path(report['parent_report'])
            original = load_json(original_path)
            if (digest(original_path) != report['parent_report_sha256']
                    or report.get('status') != 'completed_artifacts'
                    or not report.get('terminated_process_gone')
                    or report.get('native_process_exit_code') != 143
                    or report['tasks'] != original['tasks']
                    or any(report.get(key) != original.get(key) for key in
                           ('source_snapshot', 'gpu_uuid', 'samples', 'physics_steps', 'agent_calls'))):
                raise ValueError('Recovered artifacts differ from the original interrupted report')
        if ((not recovered and report.get('status') != 'completed') or report.get('errors')
                or report.get('physics_steps') != 0 or report.get('agent_calls') != 0):
            raise ValueError(f'Report is not a complete static render: {path}')
        sources.append({'path': str(path), 'sha256': digest(path),
                        'source_snapshot': report['source_snapshot'],
                        'gpu_uuid': report['gpu_uuid'],
                        'preferred_presentation': path.resolve() in preferred_reports,
                        'interrupted_process_completed_artifacts_only': recovered,
                        'native_process_exit_code': report.get('native_process_exit_code')})
        for row in report['tasks']:
            key = row['family'], row['task_id'], row['configuration_id']
            previous = records.get(key)
            if previous:
                old_preferred = owners[key] in preferred_reports
                new_preferred = path.resolve() in preferred_reports
                if old_preferred == new_preferred:
                    raise ValueError(f'Duplicate rendered configuration: {key}')
                if any(previous[field] != row[field] for field in
                       ('release', 'release_manifest_sha256')):
                    raise ValueError('Presentation preference cannot replace a different frozen release')
            if row['status'] != 'rendered' or set(row['images']) != {'overview', 'task'}:
                raise ValueError('Expected both native views')
            release = Path(row['release']) / 'release_manifest.json'
            if digest(release) != row['release_manifest_sha256']:
                raise ValueError('Frozen release hash changed')
            for image in row['images'].values():
                file = Path(image['path']).resolve()
                if (not image.get('native_rtx') or image['actual_samples_per_pixel'] < report['samples']
                        or digest(file) != image['sha256']):
                    raise ValueError('Native render evidence does not match image')
                image['relative_path'] = str(file.relative_to(output.parent.resolve()))
            if previous and old_preferred:
                continue
            if previous:
                row['superseded_presentation'] = {'report': str(owners[key]),
                    'images': previous['images'], 'reason': 'Explicit preferred presentation report'}
            records[key] = row
            owners[key] = path.resolve()
    return records, sources


def build(reports, output, preferred_reports=()):
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError('Use a fresh gallery version; preserve prior indexes')
    records, sources = collect(reports, output, preferred_reports)
    groups = defaultdict(dict)
    for (family, task, configuration), row in records.items():
        groups[(family, task)][configuration] = row
    if any(set(group) != set(CONFIGURATIONS) for group in groups.values()):
        raise ValueError('Each gallery task requires all three configurations')
    rows = []
    for (family, task), group in sorted(groups.items(), key=lambda item: (
            list(FAMILIES).index(item[0][0]),
            item[1]['single_arm'].get('difficulty') or 0, item[0][1])):
        first = group['single_arm']
        cards = []
        for configuration, label in CONFIGURATIONS.items():
            record = group[configuration]
            images = record['images']
            overview, detail = (html.escape(images[name]['relative_path'], quote=True)
                                for name in ('overview', 'task'))
            cards.append(f'''<article><h3>{label}</h3><a class="image" href="{overview}" target="_blank"
              data-overview="{overview}" data-task="{detail}"><img src="{overview}" loading="lazy" alt="{html.escape(first['title'])} · {label}"></a>
              <div class="links"><a href="{overview}" target="_blank">机器人全景 ↗</a>
              <a href="{detail}" target="_blank">桌面近景 ↗</a></div></article>''')
        rows.append(f'''<section data-family="{family}" data-search="{html.escape(task+' '+first['title'], quote=True)}">
        <h2>{FAMILIES[family]} · {html.escape(first['title'])}</h2><p class="id">{task}</p>
        <div class="grid">{''.join(cards)}</div></section>''')
    options = ''.join(f'<option value="{key}">{value}</option>' for key, value in FAMILIES.items())
    content = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>任务单臂与双臂构型对照</title><style>
    :root{color-scheme:light;font-family:system-ui,"Noto Sans CJK SC",sans-serif;color:#1e293b;background:#f4f5f7}
    body{margin:0;padding:28px;max-width:1920px;margin:auto}h1{font-size:28px;margin:0 0 12px}h2{font-size:21px;margin:0}
    header{padding:12px 0 22px}header p{color:#536173;line-height:1.6}.controls{position:sticky;top:0;z-index:2;display:flex;gap:12px;flex-wrap:wrap;background:#f4f5f7ef;padding:14px 0;backdrop-filter:blur(8px)}
    select,input,button{font:inherit;padding:9px 12px;border:1px solid #ccd3de;border-radius:7px;background:white}button.active{color:white;background:#2455a1;border-color:#2455a1}input{min-width:240px}
    section{padding:25px 0;border-top:1px solid #d4dae3}.id{color:#677487;font-size:13px;margin:7px 0 15px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}article{background:white;border-radius:9px;overflow:hidden;box-shadow:0 1px 4px #0001}article h3{margin:14px 16px;font-size:17px}img{display:block;width:100%;aspect-ratio:4/3;object-fit:contain;background:#e4e7eb}.links{padding:13px 16px;display:flex;gap:18px;font-size:14px}a{color:#2455a1}footer{padding:30px 0;color:#667085}[hidden]{display:none!important}@media(max-width:800px){body{padding:15px}.grid{grid-template-columns:1fr}input{min-width:150px}}</style>
    <header><h1>任务单臂与双臂构型对照</h1><p>__COUNT__ 个任务 · __VARIANTS__ 个构型 · __IMAGES__ 张原生 RTX 图片。每行依次为单臂、同侧双臂、对侧双臂；点击图片打开原图。</p>
    <p>展示冻结任务的初始状态，使用原机器人、真实左右手及原任务物品。双臂由同一 Agent 控制。静态渲染未启动物理、未执行抓取或任务动作。</p></header>
    <div class="controls"><select id="family"><option value="">全部房间</option>__OPTIONS__</select><input id="search" placeholder="搜索任务名称 / ID"><button class="active" data-view="overview">机器人全景</button><button data-view="task">桌面近景</button></div>
    <main>__ROWS__</main><footer>原生 RTX 原图；不使用生成式修改。<a href="__MANIFEST__">图片与发布来源</a></footer>
    <script>const sections=[...document.querySelectorAll('section')];function filter(){const family=document.querySelector('#family').value,q=document.querySelector('#search').value.toLowerCase();sections.forEach(s=>s.hidden=!!((family&&s.dataset.family!==family)||(q&&!s.dataset.search.toLowerCase().includes(q))))}document.querySelector('#family').onchange=filter;document.querySelector('#search').oninput=filter;document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{document.querySelectorAll('[data-view]').forEach(x=>x.classList.toggle('active',x===b));document.querySelectorAll('a.image').forEach(a=>{a.href=a.dataset[b.dataset.view];a.querySelector('img').src=a.dataset[b.dataset.view]})});</script></html>'''
    for key, value in {'__COUNT__': len(groups), '__VARIANTS__': len(records), '__IMAGES__': 2*len(records),
                       '__OPTIONS__': options, '__ROWS__': ''.join(rows),
                       '__MANIFEST__': output.with_suffix('.json').name}.items():
        content = content.replace(key, str(value))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content)
    dump(output.with_suffix('.json'), {'schema': 'task_configuration_native_gallery.v1',
         'task_count': len(groups), 'configuration_count': len(records), 'image_count': 2*len(records),
         'sources': sources, 'tasks': list(records.values()), 'html_sha256': digest(output),
         'native_rtx': True, 'pixels_modified': False, 'physics_steps': 0, 'agent_calls': 0,
         'task_success_claimed': False})
    return {'gallery': str(output), 'tasks': len(groups), 'configurations': len(records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prefer-report', type=Path, action='append', default=[])
    args = parser.parse_args()
    print(json.dumps(build(args.reports, args.output, args.prefer_report), ensure_ascii=False))


if __name__ == '__main__':
    main()
