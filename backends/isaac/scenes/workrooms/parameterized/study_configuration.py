"""Validate actual parameterized mechanism configurations without a canonical reset substitute."""
from copy import deepcopy
import math

from .spec import canonical_hash, validate_parameters

DRAWER_FIXTURES = ('cabinet_bottom', 'cabinet_top', 'cabinet_left', 'cabinet_right', 'cabinet_rear',
    'slide_left', 'rail_screw_left0', 'rail_screw_left1', 'rail_screw_left2',
    'slide_right', 'rail_screw_right0', 'rail_screw_right1', 'rail_screw_right2', 'drawer_label')


def _layout(parameters, group_id, fixed_axis):
    result=deepcopy(parameters['layout'])
    key=[parameters['stream_sha256']['layout'],parameters['seed'],group_id]
    if not parameters['canonical']:
        angle=int(canonical_hash(key)[:13],16)/(16**13)*2*math.pi
        x,y=result['translation_xy_m'];c,s=math.cos(angle),math.sin(angle)
        result['translation_xy_m']=[c*x-s*y,s*x+c*y]
        if int(canonical_hash([key,'yaw'])[:2],16)%2: result['yaw_rad']*=-1
    if fixed_axis: result['yaw_rad']=0.
    return result


def _close(a,b):
    return len(a)==len(b) and all(abs(x-y)<=1e-9 for x,y in zip(a,b))


def validate_configuration(value, canonical):
    """Derive each actual reset pose; compare untouched fields to the task specification."""
    actual=deepcopy(value)
    if actual.get('configuration_sha256')!=canonical_hash({k:v for k,v in actual.items() if k!='configuration_sha256'}):
        raise ValueError('Parameterized study configuration hash differs')
    record=actual.get('parameterized_physics',{}); parameters=record.get('parameters',{})
    validate_parameters(parameters)
    task_id=actual['task_id']
    if parameters['family']!='study' or parameters['task_id']!=task_id:
        raise ValueError('Parameterized study task identity differs')
    drawer=task_id=='drawer_retrieval'
    if task_id not in ('drawer_retrieval','reading_stand'):
        raise ValueError('Unsupported parameterized study mechanism')
    names=['drawer','notebook'] if drawer else ['book'];group_id='drawer_supply' if drawer else 'book'
    expected=_layout(parameters,group_id,drawer)
    groups=record['layout']
    if len(groups)!=1 or groups[0]['id']!=group_id:
        raise ValueError('Parameterized study group differs')
    group=groups[0]
    if any(group[k]!=v for k,v in expected.items()):
        raise ValueError('Parameterized study layout differs from sampled stream')
    expected_paths=['/World/StudyTask/Objects/'+n for n in names]
    if drawer:expected_paths += ['/World/StudyTask/Fixtures/'+n for n in DRAWER_FIXTURES]
    if group['paths']!=expected_paths:
        raise ValueError('Parameterized study support/actor group differs')
    if not drawer and not _close(group['pivot_m'],canonical['objects']['book']['initial_pose']['position_m']):
        raise ValueError('Reading book must rotate about its own original origin')
    for name in names:
        before=canonical['objects'][name]['initial_pose']; pose=actual['objects'][name]['initial_pose']
        p=before['position_m'];pivot=group['pivot_m'];angle=expected['yaw_rad'];c,s=math.cos(angle),math.sin(angle)
        dx,dy=expected['translation_xy_m'];x,y=p[0]-pivot[0],p[1]-pivot[1]
        position=[c*x-s*y+pivot[0]+dx,s*x+c*y+pivot[1]+dy,p[2]]
        w,x,y,z=before['quaternion_wxyz'];c,s=math.cos(angle/2),math.sin(angle/2)
        q=[c*w-s*z,c*x-s*y,c*y+s*x,c*z+s*w]
        if not _close(pose['position_m'],position) or not (_close(pose['quaternion_wxyz'],q) or _close(pose['quaternion_wxyz'],[-a for a in q])):
            raise ValueError('Actual study pose differs from explicit sampled transform')
        receipt=record['actual_body_poses_m'][name]
        if not _close(receipt['position_m'],pose['position_m']) or not _close(receipt['quaternion_wxyz'],pose['quaternion_wxyz']):
            raise ValueError('Study USD pose readback differs from evaluator configuration')
    protected=deepcopy(actual);protected.pop('parameterized_physics');protected.pop('configuration_sha256')
    expected_protected=deepcopy(canonical);expected_protected.pop('configuration_sha256')
    for name in names:
        # These exact actual fields were checked above; all other object geometry is protected.
        protected['objects'][name].pop('initial_pose');expected_protected['objects'][name].pop('initial_pose')
    if drawer:
        dx,dy=expected['translation_xy_m'];old=canonical['mechanism'];new=actual['mechanism']
        expected_mechanism={'closed_position_m':[old['closed_position_m'][0]+dx,old['closed_position_m'][1]+dy,old['closed_position_m'][2]],
            'cabinet_front_x_m':old['cabinet_front_x_m']+dx,
            'notebook_extraction_max_x_m':old['notebook_extraction_max_x_m']+dx}
        if any(not _close(new[k],v) if isinstance(v,list) else abs(new[k]-v)>1e-9 for k,v in expected_mechanism.items()):
            raise ValueError('Drawer mechanism source references differ')
        for key in expected_mechanism:
            protected['mechanism'].pop(key);expected_protected['mechanism'].pop(key)
    if protected!=expected_protected:
        raise ValueError('Parameterized study changed protected geometry, targets, rules or thresholds')
    return deepcopy(value)
