"""Build immutable NIST-derived mechanical task assets in shared storage."""
import argparse,json
from pathlib import Path
from scenes.workrooms.mechanical_assembly.nist_assets import TASK_IDS, SOURCE_ROOT, ASSET_VERSIONS, build_assets

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=TASK_IDS,action='append')
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--source-root',type=Path,default=SOURCE_ROOT)
    args=parser.parse_args()
    for task in args.task or TASK_IDS:
        path=None if args.output_root is None else args.output_root/task/ASSET_VERSIONS.get(task,'v1')
        manifest=build_assets(task,path,source_root=args.source_root)
        print(json.dumps({'task':task,'usd_path':manifest['usd_path'],'metadata_sha256':manifest['metadata_sha256']}),flush=True)
if __name__=='__main__':main()
