"""Package actual HTTP-worker RGB frames at their recorded physical timestamps."""
from __future__ import annotations
import argparse
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont
from bench.isaacsim.release import digest, dump
from bench.isaacsim.video import FONT, probe
from bench.paths import load_json

PHASES = {'approach_above':'接近物体上方','descend':'下降','close':'闭合手指',
          'push':'接触推移','pause':'停留','lift':'抬起','hold':'保持抓持','transfer':'搬运',
          'lower':'放低','release':'松开手指','withdraw':'撤离','settle_released':'释放后观察',
          'hold_control':'保持不动对照'}
TITLES = {'01_cube_push': '桌面推移方块', '02_cube_lift_hold': '抓起方块并保持',
          '03_cube_pick_place': '抓起、搬运并放置方块'}


def package(run, output):
    run, output = Path(run), Path(output)
    report = load_json(run/'validation.json')
    episode = Path(report['episode_directory'])
    hashes = load_json(episode/'artifact_hashes.json')
    assert digest(run/'camera_timeline.jsonl') == report['camera_timeline_sha256']
    timeline = [json.loads(line) for line in (run/'camera_timeline.jsonl').read_text().splitlines() if line]
    by_time = {}
    for row in timeline:
        assert all(abs(camera['sim_time_s']-row['sim_time_s']) < 1e-7 for camera in row['cameras'])
        if {c['camera_id'] for c in row['cameras']} == {'overview','task'}:
            by_time[round(row['sim_time_s'],8)] = row
    ordered = [by_time[t] for t in sorted(by_time)]
    if len(ordered) < 2:
        raise ValueError('At least two recorded physical camera timestamps required')
    # Keep at most 25 Hz for the export, including the actual terminal frame.
    selected = [ordered[0]]
    for row in ordered[1:-1]:
        if row['sim_time_s']-selected[-1]['sim_time_s'] >= .04-1e-8:
            selected.append(row)
    selected.append(ordered[-1])
    frames = run/'video_frames'
    frames.mkdir(exist_ok=False)
    exporter = frames/'exporter_source.py'
    exporter.write_bytes(Path(__file__).read_bytes())
    @lru_cache(maxsize=None)
    def font(size):
        return ImageFont.truetype(str(FONT),size) if FONT.exists() else ImageFont.load_default()
    rows = []
    for index,row in enumerate(selected):
        card = Image.new('RGB',(1280,620),'#18232d')
        draw = ImageDraw.Draw(card)
        text = TITLES.get(report['task'], report['task'])
        draw.text((20,12),text+' · FR3 + SHARPA',font=font(26),fill='#eef5fa')
        draw.text((940,17),f"物理时间 {row['sim_time_s']:.2f} s",font=font(22),fill='#83e3cc')
        for camera in row['cameras']:
            revision = int(camera['rgb_uri'].rsplit('/',2)[-2])
            relative = f"private/rgb/{revision:08d}_{camera['camera_id']}.png"
            path = episode/relative
            raw = path.read_bytes()
            assert hashlib.sha256(raw).hexdigest() == camera['rgb_sha256'] == hashes[relative]
            im = Image.open(io.BytesIO(raw)).convert('RGB')
            x = 0 if camera['camera_id']=='overview' else 640
            card.paste(im.resize((640,480)),(x,65))
            draw.rectangle((x+8,74,x+132,105),fill='#18232d')
            draw.text((x+16,76),'整机视角' if x==0 else '操作近景',font=font(20),fill='white')
        phase = row.get('diagnostic',{}).get('phase','')
        label = PHASES.get(phase,phase) or ('保持不动对照' if report['mode']=='idle' else '初始化 / 终局记录')
        draw.text((20,555),label,font=font(24),fill='#eef5fa')
        draw.text((20,590),'脚本 System0 诊断 · 仅关节目标驱动物理 · 1×物理时间播放',font=font(17),fill='#b8cbd8')
        if index == len(selected)-1:
            result = report['result']
            label = '成功' if result['outcome']=='success' and result['physics_valid'] else result['outcome']
            if not result['physics_valid']:
                label += ' / 物理审计未通过'
            draw.text((650,560),'回合结果：'+label,font=font(21),fill='#83e3cc' if label=='成功' else '#f7cb85')
        destination = frames/f'{index:06d}.png'
        card.save(destination, compress_level=1)
        rows.append({'frame':str(destination),'sim_time_s':row['sim_time_s'],
                     'sources':row['cameras'],'phase':phase})
    concat = frames/'timestamps.concat.txt'
    lines = []
    for index,record in enumerate(rows):
        lines.append("file '"+record['frame'].replace("'","'\\''")+"'")
        if index+1 < len(rows):
            duration = rows[index+1]['sim_time_s']-record['sim_time_s']
            assert duration > 0
            lines.append(f'duration {duration:.9f}')
    concat.write_text('\n'.join(lines)+'\n')
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-f','concat','-safe','0','-i',str(concat),
        '-r','25','-c:v','libx264','-preset','medium','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],check=True)
    video_info = probe(output)
    physical_duration = rows[-1]['sim_time_s']-rows[0]['sim_time_s']
    encoded_duration = float(video_info['format']['duration'])
    assert abs(encoded_duration-physical_duration) <= .081, (encoded_duration,physical_duration)
    manifest = {'video':str(output),'sha256':digest(output),'ffprobe':video_info,
        'exporter_source':str(exporter),'exporter_source_sha256':digest(exporter),
        'physical_duration_s':physical_duration,'encoded_duration_s':encoded_duration,
        'playback_duration_verified':True,
        'episode_id':report['episode_id'],'physical_start_s':rows[0]['sim_time_s'],
        'physical_end_s':rows[-1]['sim_time_s'],'playback_speed':1.,
        'source':'Recorded native RTX RGB frames from the actual Agent API episode; no state replay.',
        'controller':report['mode'],'trained_rl':False,'result':report['result'],
        'export_frame_count':len(rows),'maximum_capture_gap_s':max(b['sim_time_s']-a['sim_time_s'] for a,b in zip(rows,rows[1:])),
        'sampling_note':'Up to 25 fps; timestamps preserved. Sparse idle/direct captures are displayed until the next recorded capture.',
        'frames':rows}
    dump(output.with_suffix('.json'),manifest)
    print(json.dumps({'video':str(output),'physical_duration_s':rows[-1]['sim_time_s']-rows[0]['sim_time_s'],
                      'outcome':report['result']['outcome']}),flush=True)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    package(args.run,args.output)


if __name__=='__main__':
    main()
