"""Download a curated six-prop Poly Haven CC0 collection, keeping original bytes."""
from pathlib import Path
import concurrent.futures as cf
import datetime, hashlib, json, urllib.request

from bench.paths import DATA_ROOT

ROOT = Path(f'{DATA_ROOT}/assets/objects/polyhaven/20261016')
SLUGS = ['sofa_02', 'modern_arm_chair_01', 'modern_wooden_cabinet', 'drawer_cabinet', 'electric_stove', 'potted_plant_01']
HEADERS = {'User-Agent': 'ScenceResearch/1.0 (Poly Haven CC0 attribution preserved)'}

def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=120) as r:
        return r.read()

def save_file(item):
    path, info = item
    if path.exists():
        data = path.read_bytes()
    else:
        data = fetch(info['url'])
    if len(data) != info['size'] or hashlib.md5(data).hexdigest() != info['md5']:
        raise RuntimeError(f'Published size/MD5 mismatch: {path}')
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return {'path':str(path), 'url':info['url'], 'bytes':len(data), 'md5':info['md5'],
            'sha256':hashlib.sha256(data).hexdigest()}

def asset(slug):
    folder = ROOT/slug/'gltf_2k'
    folder.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(fetch('https://api.polyhaven.com/info/'+slug))
    api = json.loads(fetch('https://api.polyhaven.com/files/'+slug))
    spec = api['gltf']['2k']['gltf']
    jobs = [(folder/(slug+'_2k.gltf'),spec)]
    jobs += [(folder/key, value) for key,value in spec['include'].items()]
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(save_file,jobs))
    provenance = {'schema':'polyhaven_model_source.v1','asset':slug,'source_url':'https://polyhaven.com/a/'+slug,
                  'api_url':'https://api.polyhaven.com/files/'+slug, 'license':'CC0-1.0',
                  'license_url':'https://polyhaven.com/license','cc0_url':'https://creativecommons.org/publicdomain/zero/1.0/',
                  'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'authors':metadata.get('authors'), 'format':'glTF 2.0', 'up_axis':'Y', 'units':'meters',
                  'source_bytes_unmodified':True,'resolution':'2k','files':records,
                  'material_convention':'baseColor sRGB; normal OpenGL tangent-space raw; ARM raw channels R=AO G=roughness B=metallic'}
    (folder.parent/'SOURCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (folder.parent/'API_FILES.json').write_text(json.dumps(api,indent=2)+'\n')
    (folder.parent/'API_INFO.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (folder.parent/'LICENSE.txt').write_text('Poly Haven asset: '+slug+'\nLicense: CC0 1.0 Universal\nhttps://polyhaven.com/license\nhttps://creativecommons.org/publicdomain/zero/1.0/\nOriginal author(s): '+json.dumps(metadata.get('authors'))+'\n')
    print(json.dumps({'asset':slug,'gltf':str(jobs[0][0]),'bytes':sum(v['bytes'] for v in records),'status':'downloaded_md5_verified'}),flush=True)
    return provenance

if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True)
    with cf.ThreadPoolExecutor(max_workers=6) as pool:
        results=list(pool.map(asset,SLUGS))
    (ROOT/'COLLECTION.json').write_text(json.dumps({'schema':'curated_furniture_collection.v1','source':'Poly Haven','license':'CC0-1.0','assets':results},indent=2)+'\n')
