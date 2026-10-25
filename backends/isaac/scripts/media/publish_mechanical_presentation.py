#!/usr/bin/env python3
"""Publish verified native mechanical previews into a fresh local demo gallery."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import re

from PIL import Image

from bench.paths import load_json


TASKS = {
    'parts_sorting': ('L1', '零件分类', '方块、圆柱与环形零件，各自归位。'),
    'peg_insertion': ('L2', '插销装配', '独立插销与带导入口的定位孔。'),
    'shaft_spacer_assembly': ('L3', '轴套与隔圈', '沿固定轴依次装入轴套和环形零件。'),
    'fastener_assembly': ('L4', '螺纹紧固', '分件螺丝刀、金属工作头与独立螺栓。'),
    'gear_train_assembly': ('L5', '三齿轮装配', '20 / 32 / 20 齿轮组，搭配三种独立工具。'),
}
SIZE_PX = (2560,1920)
MATERIAL_NAMES = {
    'steel_brushed':'拉丝钢', 'steel_polished':'抛光钢', 'rubber_graphite':'石墨色橡胶',
    'polymer_teal':'青绿色聚合物', 'worktop_laminate':'细纹层压工作台',
    'fixture_dark':'深色定位底座', 'marking_light':'浅色刻度标识',
    'anodized_red':'红色阳极氧化金属', 'anodized_blue':'蓝色阳极氧化金属',
    'anodized_yellow':'黄色阳极氧化金属', 'anodized_green':'绿色阳极氧化金属',
    'polymer_red':'红色聚合物', 'polymer_blue':'蓝色聚合物',
    'polymer_yellow':'黄色聚合物', 'polymer_green':'绿色聚合物',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    value = load_json(Path(path))
    if not isinstance(value,dict):
        raise ValueError(f'Expected a JSON object: {path}')
    return value


def checked_file(root,relative,expected):
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f'Missing or escaping source file: {relative}')
    if not isinstance(expected,str) or not re.fullmatch('[0-9a-f]{64}',expected) or digest(path)!=expected:
        raise ValueError(f'SHA256 mismatch: {path}')
    return path


def material_roles(value):
    roles = set()
    if isinstance(value,dict):
        if value.get('role') in MATERIAL_NAMES: roles.add(value['role'])
        for child in value.values(): roles.update(material_roles(child))
    elif isinstance(value,list):
        for child in value: roles.update(material_roles(child))
    return roles


def release_receipt(task_id,record):
    root = Path(record['release']).resolve()
    manifest_path = checked_file(root,'release_manifest.json',record['release_manifest_sha256'])
    manifest = read_json(manifest_path)
    files = manifest.get('files',{})
    for required in ('scene.json','task.json','presentation_audit.json'):
        if required not in files: raise ValueError(f'Missing presentation release file: {task_id}/{required}')
    for relative,row in files.items(): checked_file(root,relative,row['sha256'])
    scene,task = read_json(root/'scene.json'),read_json(root/'task.json')
    if scene.get('scenario',{}).get('task_id')!=task_id or task.get('task_id')!=task_id:
        raise ValueError(f'Release task identity mismatch: {task_id}')
    distractors = scene.get('task_distractors')
    if (not isinstance(distractors,list) or len(distractors)!=5 or len(set(distractors))!=5
            or set(distractors)!=set(task.get('distractors',{}).get('ids',[]))
            or task.get('distractors',{}).get('scored') is not False
            or set(distractors)&set(task['evaluation']['initial_objects'])
            or not set(distractors).issubset(row['id'] for row in scene['objects'])):
        raise ValueError(f'Expected five explicitly unscored physical distractors: {task_id}')
    from scenes.workrooms.mechanical_assembly.table_distractors import validate_distractors
    validate_distractors(scene, task)
    audit = read_json(root/'presentation_audit.json')
    removed = audit.get('supply_reset', {}).get('removed_fixture_ids', [])
    pads = [row['id'] for row in scene['fixtures'] if row['id'] in removed]
    if pads: raise ValueError(f'Display supply pads remain: {task_id}: {pads}')
    if task_id=='gear_train_assembly' and task.get('physical_geometry',{}).get('gear_teeth')!={
            'input_gear':20,'idler_gear':32,'output_gear':20}:
        raise ValueError('L5 publication requires the actual 20/32/20 gear release')
    return {'path':str(root),'manifest_sha256':digest(manifest_path),
        'files_sha256':{name:row['sha256'] for name,row in files.items()},
        'distractor_ids':distractors,'display_supply_pads':pads,
        'material_roles':sorted(material_roles(audit)),
        'catalog_status':'planned' if task_id=='gear_train_assembly' else 'implemented_candidate'}


def image_receipt(render_root,relative,record):
    path = checked_file(render_root,relative,record.get('sha256'))
    recorded_path = Path(record.get('path',''))
    if not recorded_path.is_absolute(): recorded_path = render_root/recorded_path
    if recorded_path.resolve()!=path:
        raise ValueError(f'Image path differs from provenance: {relative}')
    samples = record.get('actual_samples_per_pixel')
    if (record.get('size_px')!=list(SIZE_PX) or record.get('native_rtx') is not True
            or record.get('pending_stage_loads')!=0 or type(samples) is not int or samples<96):
        raise ValueError(f'Incomplete native RTX image metadata: {relative}')
    with Image.open(path) as image:
        if image.format!='PNG' or image.size!=SIZE_PX or image.mode not in ('RGB','RGBA'):
            raise ValueError(f'Expected a 2560×1920 RGB PNG: {relative}')
        image.verify()
    with Image.open(path) as image: image.load()
    return {'source_path':str(path),'published_path':relative,'sha256':record['sha256'],
            'size_px':list(SIZE_PX),'actual_samples_per_pixel':samples,
            'native_rtx':True,'pending_stage_loads':0,'camera':record.get('camera')}


def validate_render(render_root):
    provenance_path = render_root/'provenance.json'
    provenance = read_json(provenance_path)
    if (provenance.get('schema')!='mechanical_presentation_preview.v1'
            or provenance.get('status')!='completed' or provenance.get('physics_steps')!=0):
        raise ValueError('A completed zero-step mechanical presentation render is required')
    images,releases,pose_counts = {},{},{}
    for task_id in TASKS:
        record = provenance.get('tasks',{}).get(task_id,{}).get('render',{})
        posed = record.get('posed_robot_links')
        if (record.get('status')!='rendered' or record.get('task_id')!=task_id
                or record.get('physics_steps')!=0 or type(posed) is not int or posed<43):
            raise ValueError(f'Incomplete static task render: {task_id}')
        # pose_robot includes all named URDF Xforms, including fixed geometry;
        # scene_info's separate physical robot_link_paths inventory has 43.
        pose_counts[task_id] = posed
        directory = render_root/'images'/task_id
        if read_json(directory/'provenance.json')!=record:
            raise ValueError(f'Task provenance differs from completed render record: {task_id}')
        releases[task_id] = release_receipt(task_id,record)
        views = ['closeup','task'] + (['tool_detail'] if task_id in ('fastener_assembly','gear_train_assembly') else [])
        for view in views:
            relative = f'images/{task_id}/{view}.png'
            images[relative] = image_receipt(render_root,relative,record['images'][view])
        if task_id=='gear_train_assembly':
            assembled = record.get('assembled_preview',{})
            if (assembled.get('physics_steps')!=0
                    or assembled.get('state')!='Explicit privileged assembled presentation pose'
                    or read_json(directory/'assembled'/'provenance.json')!=assembled):
                raise ValueError('L5 assembled view must be an explicitly declared static presentation pose')
            relative = f'images/{task_id}/assembled/assembled.png'
            images[relative] = image_receipt(render_root,relative,assembled['images']['assembled'])
    return {'schema':'mechanical_presentation_gallery_sources.v1',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'render_root':str(render_root),'render_provenance_sha256':digest(provenance_path),
        'render_provenance':provenance,'images':images,'releases':releases,
        'posed_robot_links_by_task':pose_counts,
        'posed_robot_links_scope':'Original renderer counts of URDF-named posed Xforms, including fixed geometry; not the separate physical robot_link_paths inventory.',
        'publication_scope':'Native static RTX previews; zero physics steps. L5 remains planned. The assembled pose is a display, not task completion.',
        'image_processing':'Original PNG bytes copied unchanged; no resize, retouching or generated substitutes.',
        'validation':'Image bytes, dimensions and capture metadata; release files and declared layout. No new native physics/manipulation qualification.'}


def gallery_html(sources):
    cards = []
    for task_id,(level,title,caption) in TASKS.items():
        base = f'images/{task_id}'
        badge = '<span class="planned">planned · 待验证</span>' if level=='L5' else ''
        cards.append(f'''<article class="task" id="{task_id}">
  <a class="photo" href="{base}/closeup.png" target="_blank" rel="noopener"><img src="{base}/closeup.png" alt="{escape(title)}原生场景近景" width="2560" height="1920" loading="lazy"></a>
  <div class="card-body"><div class="title-row"><span class="level">{level}</span><h2>{escape(title)}</h2>{badge}</div>
  <p>{escape(caption)}</p><div class="links"><a href="{base}/closeup.png" download>下载近景 PNG ↗</a><a href="{base}/task.png" target="_blank" rel="noopener">查看全景</a><a href="{base}/task.png" download>下载全景 ↗</a></div></div>
</article>''')
    detail_cards = []
    details = [('fastener_assembly/tool_detail.png','螺丝刀细节','金属接头、分区握柄与尾帽，保留原工作头。'),
               ('gear_train_assembly/tool_detail.png','齿轮工具细节','套筒曲柄与开口扳手，呈现不同的金属和握持表面。')]
    for path,title,caption in details:
        detail_cards.append(f'''<figure class="detail"><a class="photo" href="images/{path}" target="_blank" rel="noopener"><img src="images/{path}" alt="{escape(title)}" width="2560" height="1920" loading="lazy"></a><figcaption><h3>{escape(title)}</h3><p>{escape(caption)}</p><a href="images/{path}" download>下载原图 ↗</a></figcaption></figure>''')
    roles = sorted(set().union(*(set(row['material_roles']) for row in sources['releases'].values())))
    material_text = '、'.join(MATERIAL_NAMES[role] for role in roles)
    return '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>机械装配工作台 · 五任务场景图库</title><meta name="description" content="五个机械装配任务的原生高分辨率静态场景：零件分类、插销、轴套隔圈、螺纹紧固与三齿轮装配。">
<style>
:root{color-scheme:light;--ink:#183331;--muted:#667573;--line:#d9dfd8;--paper:#f3f3ed;--accent:#196b62}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,"PingFang SC","Microsoft YaHei",sans-serif;line-height:1.6}a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}a:focus-visible{outline:3px solid #ba8c42;outline-offset:4px}header,main,footer{width:min(1480px,calc(100% - 64px));margin:auto}header{padding:64px 0 32px}.eyebrow{text-transform:uppercase;letter-spacing:.15em;font-size:12px;color:var(--muted);font-weight:700}h1{font-size:clamp(30px,4vw,52px);line-height:1.15;letter-spacing:-.035em;margin:16px 0}header p{max-width:760px;color:var(--muted);font-size:17px}.features{display:flex;flex-wrap:wrap;gap:9px;margin-top:24px}.features span{border:1px solid var(--line);border-radius:30px;padding:7px 14px;font-size:13px;background:#fafaf6}nav{display:flex;flex-wrap:wrap;gap:22px;padding:24px 0;border-top:1px solid var(--line);margin-top:32px;font-size:14px}.gallery,.details{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:28px}.task,.detail{background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 3px 15px #163c3210;margin:0}.photo{display:block;overflow:hidden;background:#e5e8e2}.photo img{display:block;width:100%;height:auto;aspect-ratio:4/3;object-fit:contain}.card-body,figcaption{padding:22px 26px}.title-row{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.level{font-size:12px;border:1px solid var(--line);border-radius:5px;padding:2px 8px;font-weight:700}h2{font-size:23px;letter-spacing:-.03em;margin:0}h3{font-size:19px;margin:0 0 6px}p{margin:9px 0 16px}.card-body p,figcaption p{font-size:14px;color:var(--muted)}.links{display:flex;flex-wrap:wrap;gap:18px;font-size:13px}.planned{font-size:11px;color:#796747;background:#f4efe4;border-radius:20px;padding:3px 9px}.section-heading{margin:54px 0 20px;font-size:25px}.section-note{color:var(--muted);font-size:14px;max-width:850px}.assembled{margin:26px 0 0;background:white;border-radius:12px;overflow:hidden}.assembled .photo{max-height:none}.assembled figcaption{display:flex;align-items:center;justify-content:space-between;gap:20px}.assembled figcaption p{margin:5px 0 0}.materials{margin:42px 0;padding:24px 26px;border:1px solid var(--line);border-radius:12px}.materials p{color:var(--muted);font-size:14px;margin-bottom:0}footer{padding:24px 0 40px;border-top:1px solid var(--line);font-size:12px;color:var(--muted)}footer p{margin:6px 0}footer a{margin-right:18px}@media(min-width:1100px){.task:last-child{grid-column:1/-1;display:grid;grid-template-columns:1.45fr 1fr}.task:last-child .card-body{display:flex;flex-direction:column;justify-content:center;padding:36px}.task:last-child .title-row{margin-bottom:12px}}@media(max-width:700px){header,main,footer{width:calc(100% - 28px)}header{padding-top:36px}.gallery,.details{grid-template-columns:1fr;gap:22px}.card-body,figcaption{padding:18px}.assembled figcaption{display:block}.assembled figcaption>a{display:inline-block;margin-top:14px}nav{gap:16px}}
</style></head><body>
<header><div class="eyebrow">Mechanical workshop · Native RTX</div><h1>机械装配工作台</h1><p>从零件分类到齿轮传动，五个任务在同一工作台上展开。零件与工具使用各任务的供料布局，保留功能形状并补上分件与表面细节。</p><div class="features"><span>20 / 32 / 20 齿轮组</span><span>保留任务供料支撑</span><span>每桌 5 件五金干扰项</span><span>金属 · 橡胶 · 聚合物</span></div><nav aria-label="任务导航">''' + ''.join(f'<a href="#{task_id}">{level} {escape(title)}</a>' for task_id,(level,title,_) in TASKS.items()) + '''</nav></header>
<main><section class="gallery" aria-label="五任务近景">''' + ''.join(cards) + '''</section>
<h2 class="section-heading">工具与表面细节</h2><p class="section-note">拉丝与抛光金属、防滑握柄、尾帽和金属接头，共同呈现工具的分件关系。</p><section class="details" aria-label="工具细节">''' + ''.join(detail_cards) + '''</section>
<h2 class="section-heading">三齿轮装好后的布局</h2><p class="section-note">20 / 32 / 20 齿轮组与工具空间关系。此图使用单独设置的装配展示姿态。</p><figure class="assembled"><a class="photo" href="images/gear_train_assembly/assembled/assembled.png" target="_blank" rel="noopener"><img src="images/gear_train_assembly/assembled/assembled.png" alt="20、32、20齿轮组的静态装配展示" width="2560" height="1920" loading="lazy"></a><figcaption><div><h3>20 / 32 / 20 · 传动布局</h3><p>保留功能定位底座；桌面的螺母、垫圈与短螺栓为独立可移动物体。</p></div><a href="images/gear_train_assembly/assembled/assembled.png" download>下载装配展示原图 ↗</a></figcaption></figure>
<aside class="materials"><h3>材质与桌面</h3><p>''' + escape(material_text or '拉丝钢、抛光钢、橡胶握柄、聚合物分件与细纹工作台。') + '''</p><p>保留任务需要的供料支撑、孔座、固定轴、安装板和机器人支座。</p></aside></main>
<footer><p>原生 RTX 静态预览 · 2560 × 1920 PNG · 零物理步。L5 保持 planned；装好展示图不代表任务完成或动力学验收。</p><p><a href="sources.json">来源与图像校验记录</a><a href="#">返回顶部 ↑</a></p></footer></body></html>
'''


def publish(render_root,output):
    render_root,output = Path(render_root).resolve(),Path(output).absolute()
    if output.exists() or output.is_symlink(): raise FileExistsError(f'Output must be fresh: {output}')
    if output.parent.parts[-3:]!=('demo','interface','scenes') or not re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_.-]*',output.name):
        raise ValueError('Output must be a new demo/interface/scenes/<name> directory')
    sources = validate_render(render_root)
    html = gallery_html(sources)
    # All source and content checks finish before creating the public folder.
    # Exclusive file creation never overwrites an existing historical output.
    output.parent.mkdir(parents=True,exist_ok=True)
    output.mkdir(exist_ok=False)
    for relative,row in sources['images'].items():
        target = output/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        payload = Path(row['source_path']).read_bytes()
        if hashlib.sha256(payload).hexdigest()!=row['sha256']:
            raise ValueError(f'Source changed after preflight: {row["source_path"]}')
        with target.open('xb') as stream: stream.write(payload)
        if digest(target)!=row['sha256']: raise ValueError(f'Copy verification failed: {target}')
    with (output/'index.html').open('x') as stream: stream.write(html)
    sources['published_html_sha256'] = digest(output/'index.html')
    with (output/'sources.json').open('x') as stream:
        stream.write(json.dumps(sources,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return {'output':str(output),'index':str(output/'index.html'),'images':len(sources['images']),
            'sources_sha256':digest(output/'sources.json'),'html_sha256':digest(output/'index.html')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.render_root,args.output),ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
