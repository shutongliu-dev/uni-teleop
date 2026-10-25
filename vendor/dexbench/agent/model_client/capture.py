"""Archive exactly the configured public view used by models and local tools."""
import hashlib
import json

from .depth import depth_model_input
from .transport import png_data_url
from .visibility import observation_view


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def capture(bench, episode_id, observation, folder, output, *, visual_input, tactile_input):
    view = observation_view(observation, visual_input=visual_input, tactile_input=tactile_input)
    if view['episode_id'] != episode_id:
        raise ValueError('Observation belongs to another episode')
    folder.mkdir(parents=True, exist_ok=True)
    save(folder/'observation.json', view)
    rows = bench.download_cameras(episode_id, view)
    if not rows:
        raise ValueError('A visual Agent requires recorded camera input')
    summaries, previews = {}, {}
    if visual_input == 'rgbd':
        for camera in rows:
            if 'depth_m' not in camera or 'depth_npy_bytes' not in camera:
                raise ValueError('RGB-D requested but a current camera has no metric depth')
            name = camera['metadata']['camera_id']
            data = depth_model_input(camera['depth_m'])
            previews[name] = data.pop('preview_png_bytes')
            summaries[name] = data
            (folder/(name+'_depth.npy')).write_bytes(camera['depth_npy_bytes'])
            (folder/(name+'_depth_preview.png')).write_bytes(previews[name])
        save(folder/'depth_summary.json', summaries)
    depth_context = ({'cameras': summaries,
        'interpretation': 'Aligned optical-axis camera z in metres, not ray distance. '
            'Metric grid values are exact pixel samples rounded to four decimals, not region averages. '
            'Use zero-based pixel centers (u+0.5,v+0.5), supplied intrinsics and world_from_camera. '
            'Preview uses fixed 0.1-4.0 m scale; invalid pixels black, near dark, far light. '
            'Raw float32 NPY is archived without preview clipping.'} if visual_input == 'rgbd' else None)
    content, recorded = [], []
    for camera in rows:
        metadata, raw = camera['metadata'], camera['png_bytes']
        name = metadata['camera_id']
        path = folder/(name+'.png')
        path.write_bytes(raw)
        label = {'type': 'text', 'text': 'Camera '+name}
        content.extend([label, {'type':'image_url','image_url':{'url':png_data_url(raw)}}])
        recorded.extend([label, {'type':'recorded_image','path':str(path.relative_to(output)),
            'sha256':hashlib.sha256(raw).hexdigest(),'camera':metadata}])
        if visual_input == 'rgbd':
            preview = previews[name]
            label = {'type':'text','text':'Aligned depth preview for '+name+
                ': fixed 0.1–4.0 m; black invalid, dark near, light far; metric grid is not clipped.'}
            content.extend([label, {'type':'image_url','image_url':{'url':png_data_url(preview)}}])
            recorded.extend([label, {'type':'recorded_image',
                'path':str((folder/(name+'_depth_preview.png')).relative_to(output)),
                'sha256':hashlib.sha256(preview).hexdigest(),'camera':metadata,
                'representation':'metric_depth_fixed_scale_preview',
                'source_path':str((folder/(name+'_depth.npy')).relative_to(output)),
                'source_sha256':hashlib.sha256(camera['depth_npy_bytes']).hexdigest()}])
    return {'observation':view,'camera_rows':rows,'depth_context':depth_context,
            'image_content':content,'recorded_image_content':recorded}
