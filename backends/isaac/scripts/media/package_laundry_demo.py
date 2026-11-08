"""Package five recorded engineering cloth videos without inventing motion.

Single-task videos retain their source frame rate and duration. The compilation
repeats each decoded frame an integer number of times at 24 FPS. Captions occupy
a new border outside the source image. No simulator, state replay, frame
interpolation, transition, speed change or task-success claim is involved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess

from bench.paths import DATA_ROOT, load_json, OUTPUT_ROOT

TASKS = {
    'towel_flattening': '毛巾摊平',
    'towel_folding': '毛巾对折',
    'towel_stacking': '毛巾叠放',
    'shirt_stacking': '已折上衣叠放',
    'trousers_stacking': '已折长裤叠放',
}
DATASETS = Path(OUTPUT_ROOT)  # validation evidence
MEDIA = Path(f'{OUTPUT_ROOT}/media/laundry_tasks_20261021')
FONT = Path(f'{DATA_ROOT}/assets/fonts/noto_cjk_sc/v1/NotoSansCJKsc-Regular.otf')
CAPTION = '局部工程抓持 · 非机器人策略'
HEADER_HEIGHT = 116
COMPILATION_FPS = 24


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            checksum.update(block)
    return checksum.hexdigest()


def read(path):
    return load_json(Path(path))


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')


def pin(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': digest(path)}


def checked_pin(value):
    if not isinstance(value, dict) or set(value) != {'path', 'sha256'}:
        raise ValueError('Each selected source must contain path and sha256')
    path = Path(value['path']).resolve()
    if not path.is_file() or digest(path) != value['sha256']:
        raise ValueError(f'Selected source is missing or changed: {path}')
    return path


def probe(path):
    record = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams',
        '-show_entries', 'stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration',
        '-of', 'json', str(path)], text=True))
    if len(record.get('streams', [])) != 1:
        raise ValueError('Exactly one video stream required')
    result = record['streams'][0]
    fps = Fraction(result['avg_frame_rate'])
    if fps != Fraction(result['r_frame_rate']) or fps <= 0:
        raise ValueError('Source must have a fixed positive frame rate')
    result['frame_count'] = int(result['nb_frames'])
    result['duration_s'] = float(result['duration'])
    if abs(result['duration_s'] - result['frame_count'] / float(fps)) > 1.e-5:
        raise ValueError('Video duration differs from frame count / rate')
    return result


def verify_source(row):
    task_id = row['task_id']
    if task_id not in TASKS:
        raise ValueError('Unknown laundry task')
    paths = {key: checked_pin(row[key]) for key in
             ('video', 'video_manifest', 'frames_manifest', 'native_report')}
    video, frames, native = (read(paths[key]) for key in
                             ('video_manifest', 'frames_manifest', 'native_report'))
    if native.get('task_id') != task_id or native.get('complete') is not True:
        raise ValueError('Native report must be complete and match selected task')
    if native.get('robot_present') is not False:
        raise ValueError('This package explicitly labels robot-free engineering recordings')
    terminal = None
    if 'terminal_review' in row:
        review = read(checked_pin(row['terminal_review']))
        if review.get('task_id') != task_id or review.get('native_report') != row['native_report']:
            raise ValueError('Terminal review belongs to a different native recording')
        terminal = review['terminal_window']
    for key in ('scene_input', 'task_input'):
        checked_pin(native[key])
    if digest(Path(native['release']['path']) / 'release_manifest.json') != native['release']['manifest_sha256']:
        raise ValueError('Native source release manifest changed')
    boundary = native.get('engineering_boundary', {})
    if (not boundary.get('mode') or boundary.get('whole_cloth_animation') is not False
            or boundary.get('post_release_nodal_write_steps') != 0):
        raise ValueError('Declared local engineering operation/release evidence required')
    if native.get('video') != video:
        raise ValueError('Video manifest differs from original native report')
    if (video.get('source') != 'recorded_native_RTX_PNGs'
            or video.get('interpolation') is not False
            or Path(video['path']).resolve() != paths['video']
            or video['sha256'] != row['video']['sha256']):
        raise ValueError('Expected exact original native RTX video, without interpolation')
    if (frames.get('schema') != 'laundry_native_rtx_frames.v1'
            or frames.get('cloth_state_writes') is not False
            or frames.get('physics_steps_during_capture') != 0):
        raise ValueError('Original RTX acquisition receipt is missing')
    selected = []
    for frame in frames['frames']:
        if video['camera_id'] not in frame['images']:
            continue
        if frame.get('native_rtx') is not True or frame.get('capture_state_unchanged') is not True:
            raise ValueError('Image lacks unchanged native-state acquisition evidence')
        image = frame['images'][video['camera_id']]
        if digest(image['path']) != image['sha256']:
            raise ValueError('Original RTX PNG changed')
        if selected and frame['native_time_s'] < selected[-1]['native_time_s']:
            raise ValueError('Native frame times reverse')
        if selected and frame['native_time_s'] == selected[-1]['native_time_s']:
            selected[-1] = frame
        else:
            selected.append(frame)
    times = [frame['native_time_s'] for frame in selected]
    if times != video['source_native_times_s'] or len(times) != video['source_frames']:
        raise ValueError('Native frame timestamps differ from encoded source mapping')
    mapping = video['video_source_indices']
    fps = Fraction(str(video['fps']))
    if (len(mapping) != video['video_frames'] or not times
            or any(type(index) is not int or not 0 <= index < len(times) for index in mapping)
            or mapping != sorted(mapping)):
        raise ValueError('Invalid source-frame mapping')
    # Check that no encoded frame displays a future native state. Existing
    # acquisition quantization and trailing padding are retained explicitly.
    if any(times[index] - times[0] > i / float(fps) + 1.e-8
           for i, index in enumerate(mapping)):
        raise ValueError('Encoded video maps to a future native state')
    measured = probe(paths['video'])
    if (Fraction(measured['avg_frame_rate']) != fps
            or measured['frame_count'] != len(mapping)):
        raise ValueError('Encoded source rate/count differs from acquisition receipt')
    factor = Fraction(COMPILATION_FPS) / fps
    if factor.denominator != 1 or factor < 1:
        raise ValueError('Source FPS must divide 24 for exact integer frame repetition')
    return {**row, 'ffprobe': measured, 'native_video': video, 'terminal_window': terminal,
            'source_frames': selected, 'compilation_repetition': int(factor),
            'robot_present': native.get('robot_present'),
            'engineering_control_mode': boundary['mode'],
            'native_duration_s': native['duration_s'],
            'native_steps': native['native_steps'],
            'release_time_s': boundary.get('release_time_s'),
            'source_release': native['release']}


def inspect_selection(selection, require_five=False):
    value = read(selection)
    if value.get('schema') != 'laundry_engineering_demo_selection.v1':
        raise ValueError('Unsupported selection schema')
    rows = value['tasks']
    ids = [row['task_id'] for row in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Select distinct tasks')
    if require_five and ids != list(TASKS):
        raise ValueError('Packaging requires all five tasks in Task1–Task5 order')
    return [verify_source(row) for row in rows]


def encoder(path, width, height, fps, log):
    return subprocess.Popen([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
        '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
        '-framerate', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
        '-preset', 'medium', '-crf', '18', '-threads', '2', '-pix_fmt', 'yuv420p',
        '-filter_threads', '1', '-movflags', '+faststart', '-video_track_timescale', '24000', str(path)],
        stdin=subprocess.PIPE, stdout=log, stderr=log)


def exact_read(stream, size):
    chunks, remaining = [], size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    data = b''.join(chunks)
    if data and len(data) != size:
        raise RuntimeError('Source decoder ended mid-frame')
    return data


def caption_image(row, number, width, height):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new('RGB', (width, height + HEADER_HEIGHT), '#152332')
    draw = ImageDraw.Draw(image)
    title = f'Task {number}  {TASKS[row["task_id"]]}'
    if row.get('terminal_window') is not None and row['terminal_window'].get('passed') is not True:
        title += '  ·  末窗检查未通过'
    text = [(title, 32, 9, '#ffffff'), (CAPTION, 27, 48, '#ffd38a'),
            (f'1× 正常速度  |  原始采样 {row["native_video"]["fps"]:g} FPS  |  '
             f'物理过程 {row["native_duration_s"]:g} 秒', 21, 84, '#dce8f0')]
    for line, size, y, color in text:
        font = ImageFont.truetype(str(FONT), size)
        if draw.textbbox((20, y), line, font=font)[2] > width - 20:
            raise ValueError('Video is too narrow for readable source captions')
        draw.text((20, y), line, font=font, fill=color)
    return image


def write_html(output, records):
    sections = []
    for record in records:
        name = Path(record['video']['path']).name
        terminal = record.get('terminal_window')
        review_note = ''
        if terminal is not None:
            status = lambda key: '通过' if terminal.get(key) is True else '未通过'
            review_note = (f'<p>末半秒检查：形状{status("geometry_passed")}；'
                f'完整释放{status("complete_release")}；稳定性{status("stable")}。'
                '这是局部末窗检查，未完成本次全流碰撞审计，不授予任务成功。</p>')
        sections.append(f'<section><h2>Task {record["number"]} · {html.escape(TASKS[record["task_id"]])}</h2>'
            f'<video controls preload="metadata" src="{html.escape(name)}"></video>'
            f'<p>原始采样 {record["source_fps"]:g} FPS；物理过程 {record["native_duration_s"]:g} 秒；'
            f'视频 {record["ffprobe"]["duration_s"]:g} 秒（保留原始帧保持）。</p>'
            f'{review_note}<a href="{html.escape(name)}" download>下载单片</a></section>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>洗衣房五任务 · 工程操作录像</title>
<style>body{background:#101d29;color:#e8f1fa;font:17px/1.65 system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 22px}
h1{line-height:1.3}a{color:#93d8ff}video{display:block;width:100%;background:#000;border-radius:12px}
section{margin:42px 0;padding-top:16px;border-top:1px solid #34465a}.note{padding:18px 22px;background:#263345;border-radius:12px}
</style><h1>洗衣房五任务 · 工程操作录像</h1>
<p class="note">局部工程抓持 · 非机器人策略。以下为真实原生柔体仿真录像，保持正常速度；
布料由声明的局部抓区搬运，随后释放。视频不代表机器人抓取或自主策略验收。</p>
<p>三段毛巾录像复用已完成的历史实验；上衣与长裤录像来自本次新实验。
各单片保留原始帧率；合集为 24 FPS，仅重复已有帧，不生成中间动作。画面上方说明条不覆盖原图。</p>
<h2>五项合集</h2><video controls preload="metadata" src="five_tasks.mp4"></video>
<p><a href="five_tasks.mp4" download>下载合集</a> · <a href="manifest.json">来源、时序与哈希</a></p>'''
    page += ''.join(sections) + '</html>\n'
    (output / 'index.html').write_text(page)


def package(selection, output, evidence):
    from PIL import Image
    rows = inspect_selection(selection, require_five=True)
    width, height = rows[0]['ffprobe']['width'], rows[0]['ffprobe']['height']
    if (width % 2 or height % 2 or any((row['ffprobe']['width'], row['ffprobe']['height'])
                                     != (width, height) for row in rows)):
        raise ValueError('Source videos must have the same even pixel dimensions; no image is cropped/rescaled')
    output, evidence = output.resolve(), evidence.resolve()
    if not output.is_relative_to(MEDIA) or output == MEDIA or not evidence.is_relative_to(DATASETS):
        raise ValueError('Videos belong under laundry media and evidence under shared datasets')
    if output.exists() or evidence.exists():
        raise FileExistsError('Choose fresh output and evidence directories')
    output.mkdir(parents=True)
    evidence.mkdir(parents=True)
    script = Path(__file__).resolve()
    source_pin, selection_pin, font_pin = pin(script), pin(selection), pin(FONT)
    shutil.copy2(script, evidence / script.name)
    shutil.copy2(selection, evidence / 'selection.json')
    manifest = {'schema': 'laundry_engineering_demo_package.v1', 'status': 'encoding',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'selection': selection_pin, 'packager': source_pin, 'font': font_pin,
        'playback_speed': 1., 'physics_interpolation': False, 'new_physics_executed': False,
        'new_model_calls': 0, 'original_sources_modified': False,
        'robot_manipulation_claimed': False, 'task_success_claimed': False,
        'caption': CAPTION, 'image_operation': 'Decode RGB, preserve every source pixel position, add 116 px top border, H.264 encode.',
        'records': [], 'compilation_fps': COMPILATION_FPS}
    dump(evidence / 'manifest.json', manifest)
    combined = None
    processes = []
    try:
        with (evidence / 'compilation_ffmpeg.log').open('wb') as log:
            combined = encoder(output / 'five_tasks.mp4', width, height + HEADER_HEIGHT, COMPILATION_FPS, log)
            processes.append(combined)
            combined_count = 0
            for number, row in enumerate(rows, 1):
                video = output / f'task{number}_{row["task_id"]}.mp4'
                fps = Fraction(str(row['native_video']['fps']))
                rate = row['compilation_repetition']
                caption = caption_image(row, number, width, height)
                mapping = []
                with (evidence / f'task{number}_ffmpeg.log').open('wb') as task_log:
                    decoder = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-threads', '2',
                        '-i', row['video']['path'], '-map', '0:v:0', '-vsync', '0', '-f', 'rawvideo',
                        '-pix_fmt', 'rgb24', '-threads', '2', '-filter_threads', '1', 'pipe:1'], stdout=subprocess.PIPE, stderr=task_log)
                    encoded = encoder(video, width, height + HEADER_HEIGHT, fps, task_log)
                    processes.extend((decoder, encoded))
                    for index in range(row['ffprobe']['frame_count']):
                        pixels = exact_read(decoder.stdout, width * height * 3)
                        if not pixels:
                            raise RuntimeError('Source decoder returned too few frames')
                        caption.paste(Image.frombytes('RGB', (width, height), pixels), (0, HEADER_HEIGHT))
                        framed = caption.tobytes()
                        encoded.stdin.write(framed)
                        for _ in range(rate):
                            combined.stdin.write(framed)
                        native_index = row['native_video']['video_source_indices'][index]
                        mapping.append({'source_encoded_frame': index, 'single_video_time_s': index / float(fps),
                            'native_state_time_s': row['native_video']['source_native_times_s'][native_index],
                            'native_source_index': native_index, 'compilation_first_frame': combined_count,
                            'compilation_frame_count': rate})
                        combined_count += rate
                    if decoder.stdout.read(1) or decoder.wait() != 0:
                        raise RuntimeError('Source decoding failed or returned extra frames')
                    encoded.stdin.close()
                    if encoded.wait() != 0:
                        raise RuntimeError('Single-task video encoding failed')
                measured = probe(video)
                if measured['frame_count'] != len(mapping) or Fraction(measured['avg_frame_rate']) != fps:
                    raise RuntimeError('Packaged single-task timing changed')
                record = {'number': number, 'task_id': row['task_id'],
                    'sources': {key: row[key] for key in ('video', 'video_manifest', 'frames_manifest', 'native_report')},
                    'source_release': row['source_release'], 'source_fps': float(fps),
                    'native_duration_s': row['native_duration_s'], 'native_steps': row['native_steps'],
                    'release_time_s': row['release_time_s'], 'robot_present': row['robot_present'],
                    'engineering_control_mode': row['engineering_control_mode'],
                    'terminal_review': row.get('terminal_review'),
                    'terminal_window': row.get('terminal_window'),
                    'native_time_range_s': row['native_video']['native_time_range_s'],
                    'original_final_frame_hold_s': row['native_video']['final_hold_s'],
                    'video': pin(video), 'ffprobe': measured, 'frame_mapping': mapping}
                manifest['records'].append(record)
                dump(evidence / 'manifest.json', manifest)
                print(json.dumps({'task_complete': row['task_id'], 'video': str(video)}), flush=True)
            combined.stdin.close()
            if combined.wait() != 0:
                raise RuntimeError('Compilation encoding failed')
        measured = probe(output / 'five_tasks.mp4')
        if (measured['frame_count'] != combined_count
                or Fraction(measured['avg_frame_rate']) != COMPILATION_FPS):
            raise RuntimeError('Compilation timing differs from integer-repeat mapping')
        manifest['compilation'] = {'video': pin(output / 'five_tasks.mp4'), 'ffprobe': measured,
            'transitions': [], 'added_holds': [], 'frame_operation': 'Exact integer repetition of each decoded source frame in original order'}
        # Bind originals again after encoding; failures retain their logs/artifacts.
        inspect_selection(selection, require_five=True)
        for entry in (source_pin, selection_pin, font_pin):
            checked_pin(entry)
        manifest['status'] = 'complete'
        dump(evidence / 'manifest.json', manifest)
        dump(output / 'manifest.json', manifest)
        write_html(output, manifest['records'])
        dump(evidence / 'published_artifacts.json', {'manifest': pin(output / 'manifest.json'),
             'html': pin(output / 'index.html'), 'videos': [r['video'] for r in manifest['records']],
             'compilation': manifest['compilation']['video']})
    except BaseException as error:
        manifest['status'], manifest['error'] = 'failed', repr(error)
        dump(evidence / 'manifest.json', manifest)
        raise
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--inspect-only', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--evidence', type=Path)
    args = parser.parse_args()
    if args.inspect_only:
        rows = inspect_selection(args.selection)
        print(json.dumps({'verified': True, 'tasks': [{k: row[k] for k in
            ('task_id', 'ffprobe', 'native_duration_s', 'engineering_control_mode', 'robot_present')}
            for row in rows]}, ensure_ascii=False, indent=2))
    else:
        if args.output is None or args.evidence is None:
            parser.error('--output and --evidence are required for packaging')
        package(args.selection, args.output, args.evidence)


if __name__ == '__main__':
    main()
