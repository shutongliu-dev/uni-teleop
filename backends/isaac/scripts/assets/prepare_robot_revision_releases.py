"""Publish explicit robot/ready input derivatives from a complete source snapshot.

No native execution, model calls or qualification transfer. An optional exact
installation subset supports a fresh pilot output; catalog selection is separate.
"""
from copy import deepcopy
import argparse
import json
from pathlib import Path

from bench.paths import load_json, OUTPUT_ROOT
from scenes.workrooms.robot_revision.assets import (
    checked_file, file_hash, read_asset_manifest, RUNTIME_SCHEMA)
from scenes.workrooms.robot_revision.inputs import descriptor, canonical_hash
from scenes.workrooms.robot_revision.release import prepare_release

SCHEMA='robot_revision_preparation_request.v1'
ROOT=Path(__file__).resolve().parents[2]


def _json(path):
    return load_json(Path(path))


def _reference(value):
    if not isinstance(value,dict) or set(value)!={'path','sha256'}:
        raise ValueError('Preparation input needs an exact path/digest reference')
    if not isinstance(value['path'],str) or not Path(value['path']).is_absolute():
        raise ValueError('Preparation references must be absolute')
    return checked_file(value['path'],value['sha256'])


def installation_id(row):
    """Internal physical installation ID; public menu identity stays separate."""
    return '/'.join((row['family'],row['task_id'],row['internal_configuration_id'] or 'fixed'))


def read_request(path):
    request=_json(path)
    required={'schema','parent_matrix','ready_report','robot_asset_manifest','output'}
    if (not isinstance(request,dict) or set(request)-required-{'installations'}
            or not required<=set(request) or request['schema']!=SCHEMA):
        raise ValueError('Unknown robot revision preparation request')
    for key in ('parent_matrix','ready_report','robot_asset_manifest'):_reference(request[key])
    output=Path(request['output'])
    if not output.is_absolute() or not output.resolve().is_relative_to(OUTPUT_ROOT):
        raise ValueError('Robot publications require a new shared dataset output')
    selected=request.get('installations')
    if selected is not None and (not isinstance(selected,list) or not selected
            or any(not isinstance(x,str) for x in selected) or len(set(selected))!=len(selected)):
        raise ValueError('Installation subset must be explicit, unique and nonempty')
    return request


def descriptors(request):
    """Bind actual old inputs/ready maps; no mutable or inherited pass receipt."""
    from scenes.workrooms.catalog import task_definition
    parents=_json(_reference(request['parent_matrix']))
    ready=_json(_reference(request['ready_report']))
    asset_path=_reference(request['robot_asset_manifest'])
    asset=read_asset_manifest(asset_path,expected_sha256=request['robot_asset_manifest']['sha256'])
    if asset['schema']!=RUNTIME_SCHEMA:
        raise ValueError('New publication uses only the runtime physical/render dependency closure')
    rows=parents.get('rows',[])
    if len(rows)!=113 or len({installation_id(r) for r in rows})!=113:
        raise ValueError('Parent matrix must contain all 113 distinct original installations')
    if ready.get('parent_matrix')!=request['parent_matrix']:
        raise ValueError('Ready report belongs to another parent matrix')
    if ready.get('new_assembly_version')!=asset['assembly_version']:
        raise ValueError('Ready derivation belongs to another robot asset version')
    key=lambda r:(r['family'],r['task_id'],r['configuration_id'])
    solved=ready.get('rows',[])
    if len(solved)!=len(rows) or {key(r) for r in solved}!={key(r) for r in rows}:
        raise ValueError('Ready map does not exactly cover original installations')
    solved={key(r):r for r in solved}
    selected=request.get('installations') or [installation_id(r) for r in rows]
    if not set(selected)<={installation_id(r) for r in rows}:
        raise ValueError('Requested installation is absent from the actual parent matrix')
    records=[]
    for row in rows:
        if installation_id(row) not in selected:continue
        current=task_definition(row['family'],row['task_id'],row['configuration_id'])
        if current['source_release']!=row['source_release']:
            raise ValueError('Current catalog no longer selects the exact original parent')
        checked_file(Path(row['source_release']['path'])/'release_manifest.json',row['source_release']['manifest_sha256'])
        solution_row=solved[key(row)]
        if (solution_row['parent_release']!=row['source_release']
                or solution_row['parent_input_refs']!=row['parent_input_refs']):
            raise ValueError('Ready coordinates are bound to different original inputs')
        solution=_json(checked_file(solution_row['path'],solution_row['sha256']))
        if any(asset['files_sha256'].get(path)!=digest for side in solution['sides'].values()
               for path,digest in side['new_asset_sha256'].items()):
            raise ValueError('Ready FK used different public robot or hand URDFs')
        if solution.get('release_or_object_mutations') is not False:
            raise ValueError('Ready proposal may not alter task objects or releases')
        values={name:_json(checked_file(ref['path'],ref['sha256'])) for name,ref in row['parent_input_refs'].items()}
        if solution.get('old_initial_state_sha256')!=canonical_hash(values['initial_state.json']):
            raise ValueError('Ready coordinates differ from their complete original initial state')
        revised=descriptor(values['scene.json'],values['initial_state.json'],
            joint_positions_rad=solution['candidate_joint_positions_rad'],
            assembly_version=asset['assembly_version'],mounts_by_side=asset['mounts_by_side'],
            robot_asset_manifest_sha256=request['robot_asset_manifest']['sha256'])
        records.append({'installation_id':installation_id(row),'family':row['family'],'task_id':row['task_id'],
            'configuration_id':row['configuration_id'],'internal_configuration_id':row['internal_configuration_id'],
            'selection_kind':row['selection_kind'],'parent_release':deepcopy(row['source_release']),
            'ready_reference':{'path':solution_row['path'],'sha256':solution_row['sha256']},'revision':revised})
    formal={r['task_id'] for r in records if r['family']=='card_room' and not r['task_id'].endswith('_practice')}
    if any(r['task_id'].removesuffix('_practice') not in formal for r in records if r['selection_kind']=='practice'):
        raise ValueError('Selected practice publication requires its formal parent in the same request')
    return sorted(records,key=lambda r:r['selection_kind']=='practice')


def _dump(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def run(request_path):
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    required=('scripts.assets.prepare_robot_revision_releases','scenes.workrooms.robot_revision.release','scenes.workrooms.robot_revision.assets',
              'scenes.workrooms.robot_revision.inputs')
    start=verify_if_snapshot(ROOT,required_modules=required,entrypoint_file=Path(__file__))
    if not start or start.get('runtime_imports_verified') is not True:
        raise ValueError('Robot revision author must execute from a complete sealed source snapshot')
    request=read_request(request_path);records=descriptors(request);output=Path(request['output']).resolve()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    report={'schema':'robot_revision_preparation.v1','request':{'path':str(Path(request_path).resolve()),'sha256':file_hash(request_path)},
        'source_start':start,'source_final':None,'complete':False,'requested_installations':len(records),
        'rows':[],'physics_accepted':False,'task_policy_success_claimed':False,'inherited_qualification':False,
        'catalog_modified':False,'old_inputs_modified':False,'native_steps':0,'model_calls':0}
    _dump(output/'preparation.json',report)
    published={}
    try:
        for row in records:
            target=output/'releases'/row['installation_id'];formal=None
            if row['selection_kind']=='practice':
                formal=published[('card_room',row['task_id'].removesuffix('_practice'),None)]
            manifest=prepare_release(row['parent_release']['path'],target,
                revision=row['revision'],robot_asset_manifest=request['robot_asset_manifest']['path'],
                configuration_id=row['internal_configuration_id'],formal_parent_revision=formal)
            pin={'path':str(target),'manifest_sha256':file_hash(target/'release_manifest.json')}
            published[(row['family'],row['task_id'],row['configuration_id'])]=pin
            report['rows'].append({k:v for k,v in row.items() if k!='revision'}|
                {'release':pin,'revision_sha256':manifest['robot_revision_sha256'],
                 'selection_kind':manifest['selection_kind'],'status':'prepared_unvalidated'})
            _dump(output/'preparation.json',report)
        report['source_final']=verify_if_snapshot(ROOT,required_modules=required,entrypoint_file=Path(__file__))
        if not report['source_final'] or report['source_final'].get('runtime_imports_verified') is not True:
            raise ValueError('Robot revision source identity did not close')
        report['complete']=True;_dump(output/'preparation.json',report)
    except BaseException as error:
        report['error']={'type':type(error).__name__,'message':str(error)};_dump(output/'preparation.json',report);raise
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True)
    result=run(parser.parse_args().request)
    print(json.dumps({'complete':result['complete'],'prepared_installations':len(result['rows']),
                      'catalog_modified':False,'physics_accepted':False}))


if __name__=='__main__':main()
