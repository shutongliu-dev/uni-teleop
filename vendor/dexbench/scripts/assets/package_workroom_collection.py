#!/usr/bin/env python3
"""Publish verified scene USD assets and an unretouched native-render gallery."""
from pathlib import Path
import argparse,hashlib,json,shutil
from datetime import datetime,timezone
from PIL import Image,ImageDraw,ImageFont

from bench.paths import DATA_ROOT, OUTPUT_ROOT, load_json

NAMES={'mechanical_assembly':'机械装配室','laboratory':'实验室','kitchen':'厨房','study':'书房','card_room':'棋牌室','living_room':'客厅'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--renders',type=Path,required=True);p.add_argument('--version',required=True);args=p.parse_args()
    assets=Path(f'{DATA_ROOT}/assets/scenes/scence_workrooms')/args.version
    media=Path(f'{OUTPUT_ROOT}/media/scenes')/args.version
    if assets.exists() or media.exists():raise FileExistsError('Published versions are immutable; choose a new version')
    records=[]
    for family,name in NAMES.items():
        d=args.renders/family;report=load_json(d/'provenance.json');audit=load_json(d/'scene_audit.json')
        run=args.renders.parent
        exit_path=run/('exit_'+family+'.json')
        if not exit_path.exists():exit_path=run/'logs'/(family+'.exit.json')
        native_exit=load_json(exit_path)
        if native_exit.get('observed_exit_code',native_exit.get('exit_code'))!=0 or native_exit.get('native_error_count') or native_exit.get('native_errors'):
            raise ValueError(family+' native process did not exit cleanly')
        if native_exit.get('capture_checks_passed') is False:raise ValueError(family+' native capture qualification failed')
        if report['status']!='rendered' or report['stability']['status']!='passed':raise ValueError(family+' native build incomplete')
        if audit.get('status')!='passed':raise ValueError(family+' geometric audit incomplete')
        if audit['scene']['sha256']!=sha(d/'scene.usda'):raise ValueError(family+' audit belongs to a different USD')
        if report.get('robot_pose_export',{}).get('body_count')!=43:raise ValueError(family+' native robot pose missing')
        for k in ('room','workcell','tabletop'):
            if sha(d/(k+'.png'))!=report['images'][k]['sha256']:raise ValueError('Image hash mismatch '+family)
            if not report['images'][k].get('sampling_target_reached'):raise ValueError(family+' rendering samples incomplete')
        if sha(d/'scene.usda')!=report['scene_usd']['sha256']:raise ValueError('USD hash mismatch '+family)
        records.append((family,name,d,report,audit,exit_path))
    assets.mkdir(parents=True);media.mkdir(parents=True)
    published=[]
    for family,name,d,report,audit,exit_path in records:
        ad=assets/family;md=media/family;ad.mkdir();md.mkdir()
        for n in ('scene.usda','scene_description.json','provenance.json','stability.json','scene_audit.json','robot_pose_snapshot.json'):shutil.copy2(d/n,ad/n)
        for n in ('room.png','workcell.png','tabletop.png','provenance.json','scene_audit.json'):shutil.copy2(d/n,md/n)
        for folder in (ad,md):shutil.copy2(exit_path,folder/'native_exit.json')
        room=load_json(d/'scene_description.json').get('layout',{}).get('room',{})
        bounds=room.get('bounds')
        area=(bounds[1]-bounds[0])*(bounds[3]-bounds[2]) if bounds else None
        published.append({'family':family,'title_zh':name,'source_run':str(d.resolve()),'scene':str(ad/'scene.usda'),'scene_sha256':sha(ad/'scene.usda'),'native_robot_pose_sha256':sha(ad/'robot_pose_snapshot.json'),'room_theme':room.get('theme'),'room_bounds_m':bounds,'room_area_m2':area,'images':{v:str(md/(v+'.png')) for v in ('room','workcell','tabletop')},'images_sha256':{v:sha(md/(v+'.png')) for v in ('room','workcell','tabletop')}})
    font=f'{DATA_ROOT}/assets/fonts/noto_cjk_sc/v1/NotoSansCJKsc-Regular.otf'
    title=ImageFont.truetype(font,44);label=ImageFont.truetype(font,32)
    for view in ('room','workcell','tabletop'):
        W,H=960,720;gap=22;head=110;lh=56
        sheet=Image.new('RGB',(W*3+gap*4,head+2*(H+lh+gap)+gap),'#f0eee8');dr=ImageDraw.Draw(sheet)
        dr.text((gap,18),'Astra · 1 × 1 m 工作区 · Isaac 原生渲染',font=title,fill='#252a2a')
        dr.text((gap,73),{'room':'房间全景','workcell':'工作台与独立机器人固定台','tabletop':'桌面物品细节'}[view],font=label,fill='#545c5d')
        for i,(family,name,d,_,_,_) in enumerate(records):
            x=gap+(i%3)*(W+gap);y=head+gap+(i//3)*(H+lh+gap)
            area=published[i]['room_area_m2']
            suffix=f' · {area:.1f} m²' if area else ''
            dr.text((x,y+8),f'{i+1:02d}  {name}{suffix}',font=label,fill='#252a2a')
            with Image.open(d/(view+'.png')) as im:
                # All native cameras use 4:3; resize only. No generative edits or retouching.
                if abs(im.width/im.height-W/H)>1.e-6:raise ValueError('Unexpected camera aspect')
                sheet.paste(im.convert('RGB').resize((W,H),Image.Resampling.LANCZOS),(x,y+lh))
        sheet.save(media/(view+'_overview.png'))
    manifest={'schema':'scence.scene_collection.v1','created_utc':datetime.now(timezone.utc).isoformat(),'version':args.version,'scope':'six static furnished rooms with 1x1 m worktable and external anchored FR3/SHARPA pedestal; no task qualification','assets':published,'media':str(media),'image_processing':'native screenshots copied byte-identically; overview resize and external labels only'}
    for folder in (assets,media):(folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    attribution=f'''Scence workrooms — source attribution

Poly Haven furniture models and HDRI: CC0, https://polyhaven.com/license
Furniture originals: {DATA_ROOT}/assets/objects/polyhaven/20261016
Additional furniture and microscope originals: {DATA_ROOT}/assets/objects/polyhaven/20261016_compact
Furniture conversion metadata: {DATA_ROOT}/assets/objects/polyhaven/derived/v1 and v2
HDRI: {DATA_ROOT}/assets/skies/polyhaven/v2/kloofendal_48d_partly_cloudy_puresky

SimReady assets © NVIDIA Corporation, CC BY 4.0, NVIDIA Isaac Sim 6.1 SimReady asset library.
Per-asset source metadata and licenses: {DATA_ROOT}/assets/objects/simready/v1/Industrial
License: https://creativecommons.org/licenses/by/4.0/

ambientCG PBR textures: CC0, https://ambientcg.com/license
Source metadata and original files: {DATA_ROOT}/assets/materials/ambientcg/v1

Franka FR3 and SHARPA source licenses remain in their shared asset directories.
Pinned assembly metadata: {DATA_ROOT}/assets/robots/assemblies/fr3_sharpa/isaacsim_6_1_v2/SOURCE.json

Original architecture, worktable, anchored pedestal and procedural details: scence project.
USD files reference versioned shared assets; keep those paths available when opening the scenes.
Room dressing is static. This collection does not certify grasping, deformation, or task success.
'''
    for folder in (assets,media):(folder/'ATTRIBUTION.txt').write_text(attribution)
    print(json.dumps({'assets':str(assets),'media':str(media),'scenes':6,'images':18},ensure_ascii=False))

if __name__=='__main__':main()
