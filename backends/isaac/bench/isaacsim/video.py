"""Actual RGB recording with explicit physical timestamps and unaltered playback speed."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

from bench.paths import DATA_ROOT

FONT = Path(f'{DATA_ROOT}/assets/fonts/noto_cjk_sc/v1/NotoSansCJKsc-Regular.otf')


def frame_card(rgb, overview, task, score, time_s, phase):
    from PIL import Image, ImageDraw, ImageFont
    overview_is_main = phase == 'nominal' and overview is not None
    main_rgb, inset_rgb = (overview, rgb) if overview_is_main else (rgb, overview)
    main = Image.fromarray(main_rgb[:, :, :3]).convert('RGB')
    image = Image.new('RGB', (1600, 1040), '#152332')
    image.paste(main.resize((1600, 900)), (0, 110))
    drawing = ImageDraw.Draw(image)
    def text(x, y, value, size=24, fill='#f2f6fa'):
        font = ImageFont.truetype(str(FONT), size) if FONT.exists() else ImageFont.load_default()
        drawing.text((x, y), value, font=font, fill=fill)
    text(28, 14, task['title_zh'], 36)
    mode = {'nominal': '任务初态 · 关节控制检查', 'qualification_probe': '重置后的接触检查 · 独立实验',
            'contact_probe_push': '食指接触推移 · 独立实验', 'contact_probe_hover': '悬空运动负对照 · 独立实验'}[phase]
    text(28, 64, 'Isaac Sim  /  FR3 + SHARPA  /  ' + mode, 23)
    text(1130, 27, f"t = {time_s:5.2f} s   |   1× 正常速度", 24, '#8ce4e4')
    text(1130, 66, f"任务步骤 {len(score['completed_steps'])} / {len(task['steps'])}", 23)
    text(24, 125, '整机环境' if overview_is_main else '任务近景', 22, '#ffffff')
    if inset_rgb is not None:
        inset = Image.fromarray(inset_rgb[:, :, :3]).convert('RGB').resize((400, 225))
        image.paste(inset, (1176, 130))
        drawing.rectangle((1176, 130, 1576, 165), fill='#152332')
        text(1188, 133, '任务近景' if overview_is_main else '整机环境', 20)
    drawing.rectangle((0, 910, 1600, 1040), fill='#152332')
    for index, step in enumerate(task['steps']):
        done = step['id'] in score['completed_steps']
        text(28 + (index % 2) * 800, 924 + (index // 2) * 44,
             f"{index + 1}. {step['instruction_zh']}" + ('  ✓' if done else ''),
             24, '#7bebba' if done else '#e0e9f1')
    return image


def encode_frames(frames, output, fps):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-framerate', str(fps),
               '-i', str(Path(frames) / '%06d.png'), '-an', '-c:v', 'libx264', '-preset', 'medium',
               '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
    subprocess.run(command, check=True)
    return probe(output)


def concat_videos(inputs, output):
    """Concatenate equal-format real rollouts; no artificial transition or still holds."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    listing = output.with_suffix('.concat.txt')
    # ffmpeg concat format, not shell interpolation.
    listing.write_text(''.join("file '" + str(Path(p).resolve()).replace("'", "'\\''") + "'\n" for p in inputs))
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0',
                    '-i', str(listing), '-c', 'copy', '-movflags', '+faststart', str(output)], check=True)
    return probe(output)


def probe(path):
    return json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration,size',
        '-of', 'json', str(path)], text=True))
