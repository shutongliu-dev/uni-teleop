from copy import deepcopy
import json

import numpy as np
import pytest

from scripts.validation.laundry_physical_press_probe import (
    command_rows, derive_reset, pin, validate_authoring_source, validate_internal_review,
    validate_source_sample,
)


def fixture():
    scene = {'named_cloth_initialization': 'author-only',
             'cloths': [{'id': 'towel', 'initial_vertices_m': [[0, 0, 1], [1, 0, 1], [0, 1, 1]],
                         'rest_vertices_m': [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                         'mass_kg': .24, 'isaacsim': {'cloth_thickness_m': .0016}}],
             'robot': {'unchanged': True}, 'physics': {'dt_s': .002}}
    initial = {'joint_positions_rad': {'j': .1}, 'objects': {}}
    return scene, initial


def test_checkpoint_authoring_changes_only_shape_and_joint_reset():
    scene, initial = fixture()
    original = deepcopy(scene), deepcopy(initial)
    q = np.asarray(scene['cloths'][0]['initial_vertices_m']) + [0, 0, .025]
    new_scene, new_initial = derive_reset(scene, initial, q, {'j': .2})
    assert (scene, initial) == original
    reverted = deepcopy(new_scene)
    reverted['cloths'][0]['initial_vertices_m'] = original[0]['cloths'][0]['initial_vertices_m']
    assert reverted == scene
    assert new_initial == {'joint_positions_rad': {'j': .2}, 'objects': {},
                           'cloths_vertices_m': {'towel': q.tolist()}}


@pytest.mark.parametrize('q,joints', [([[0, 0, 1]], {'j': .2}),
                                    ([[0, 0, np.nan]] * 3, {'j': .2}),
                                    ([[0, 0, 1]] * 3, {'wrong': .2})])
def test_wrong_checkpoint_identity_rejected(q, joints):
    with pytest.raises(ValueError):
        derive_reset(*fixture(), q, joints)


def plan():
    return {'error': None, 'source_clearance': {'all_clear': True}, 'attachments': False,
            'cloth_state_writes': 0, 'planned_duration_s': .04,
            'samples': [{'time_s': 0., 'joint_positions_rad': {'j': .1}, 'label': 'start'},
                        {'time_s': .02, 'joint_positions_rad': {'j': .2}, 'label': 'first'},
                        {'time_s': .02, 'joint_positions_rad': {'j': .2}, 'label': 'next'},
                        {'time_s': .04, 'joint_positions_rad': {'j': .3}, 'label': 'end'}]}


def test_control_clock_merges_only_identical_boundaries():
    rows = command_rows(plan(), ['j'])
    assert len(rows) == 3 and rows[1]['label'] == 'next'


@pytest.mark.parametrize('case', ['boundary', 'missing', 'nan', 'joint', 'attachment', 'cloth_write'])
def test_no_discontinuity_or_unreviewed_control(case):
    p = plan()
    if case == 'boundary': p['samples'][2]['joint_positions_rad']['j'] += .01
    if case == 'missing': p['samples'] = [p['samples'][0], p['samples'][-1]]
    if case == 'nan': p['samples'][1]['time_s'] = np.nan
    if case == 'joint': p['samples'][1]['joint_positions_rad']['unexpected'] = 0.
    if case == 'attachment': p['attachments'] = True
    if case == 'cloth_write': p['cloth_state_writes'] = 1
    with pytest.raises(ValueError): command_rows(p, ['j'])


def internal_review():
    return {'results': [{'sample_count': 3,
        'rows': [{'sample_index': i} for i in range(3)],
        'intersections_by_pair': {name: {'worst_sample': 0, 'maximum_radius_m': .0008}
            for name in ('left_thumb_MC|left_hand_C_MC', 'right_thumb_MC|right_hand_C_MC')}}]}


@pytest.mark.parametrize('case', ['valid', 'empty', 'missing', 'duplicate', 'count', 'bool'])
def test_internal_cooked_review_covers_every_sample(case):
    review = internal_review()
    if case == 'empty': review['results'] = []
    if case == 'missing': review['results'][0]['rows'].pop()
    if case == 'duplicate': review['results'][0]['rows'][2]['sample_index'] = 1
    if case == 'count': review['results'][0]['sample_count'] = 2
    if case == 'bool': review['results'][0]['rows'][0]['sample_index'] = False
    if case == 'valid': validate_internal_review(review, 3)
    else:
        with pytest.raises(ValueError): validate_internal_review(review, 3)


@pytest.mark.parametrize('case', ['valid', 'unfrozen', 'wrong_source', 'wrong_implementation'])
def test_native_requires_same_frozen_authoring_source(tmp_path, case):
    implementation = tmp_path/'runner.py'
    source = tmp_path/'source_snapshot_manifest.json'
    implementation.write_text('original implementation')
    source.write_text('original frozen source')
    manifest = {'authoring_implementation': pin(implementation),
                'authoring_source_manifest': pin(source)}
    if case == 'unfrozen': manifest['authoring_source_manifest'] = None
    if case == 'wrong_source': source.write_text('different frozen source')
    if case == 'wrong_implementation': implementation.write_text('different implementation')
    if case == 'valid': validate_authoring_source(manifest, implementation, source)
    else:
        with pytest.raises(ValueError): validate_authoring_source(manifest, implementation, source)


def test_failed_prefix_requires_explicit_selection_and_preserves_failed_status():
    native={'complete':False,'states':{'valid_samples':11751},
            'error':{'type':'ValueError','message':'Original next-phase planning failed'}}
    before=deepcopy(native)
    with pytest.raises(ValueError,match='explicit'):
        validate_source_sample(native,11750)
    status=validate_source_sample(native,11750,allow_incomplete=True)
    assert native==before and status['original_native_complete'] is False
    assert status['original_error']==native['error']
    assert status['selected_recorded_sample']==11750
    status['original_error']['message']='independent diagnostic'
    assert native==before
    native['complete']=True
    assert validate_source_sample(native,11750) is None


@pytest.mark.parametrize('sample',[-1,11751,True,1.5])
def test_incomplete_selection_never_accepts_padding_or_an_invalid_sample(sample):
    native={'complete':False,'states':{'valid_samples':11751}}
    with pytest.raises(ValueError):validate_source_sample(native,sample,allow_incomplete=True)


@pytest.mark.parametrize('status,allow',[(None,True),(0,True),(True,1),(False,'yes')])
def test_source_execution_and_optin_are_unambiguous_booleans(status,allow):
    with pytest.raises(ValueError):
        validate_source_sample({'complete':status,'states':{'valid_samples':2}},0,
                               allow_incomplete=allow)


def design_reference(tmp_path):
    def save(name,value):
        path=tmp_path/name;path.write_text(json.dumps(value));return pin(path)
    context={'source_binding':{'native_report':save('native.json',{'original':True}),
        'source_sample':1,'cooked_export':save('cooked.json',{'original':True})},
        'joint_states':{'recorded_native_start':{'joint':.1},'original_release_initial':{'joint':.2}}}
    calculation=tmp_path/'calculation.py';calculation.write_text('# unit fixture, not a physics qualification\n')
    pairs=set(internal_review()['results'][0]['intersections_by_pair'])
    reference={'schema':'laundry_original_design_contact_bounds.v1',
        'source_binding':deepcopy(context['source_binding']),
        'calculation_implementation':pin(calculation),
        'reference_states':{label:{'joint_positions_rad':dict(joints),
            'all_non_design_pairs_clear':True,'intersections_by_pair':{
                p:{'maximum_radius_m':.0006 if label=='recorded_native_start' else .00075}for p in pairs}}
            for label,joints in context['joint_states'].items()}}
    return context,reference,save


def test_explicit_original_park_bound_retains_default_first_sample_guard(tmp_path):
    context,reference,save=design_reference(tmp_path)
    review=internal_review()
    for item in review['results'][0]['intersections_by_pair'].values():
        item.update(worst_sample=2,maximum_radius_m=.00075000005)
    with pytest.raises(ValueError,match='increased'):
        validate_internal_review(review,3)
    review['design_contact_reference']=save('design.json',reference)
    validate_internal_review(review,3,reference_context=context)
    with pytest.raises(ValueError,match='bind'):
        validate_internal_review(review,3)


@pytest.mark.parametrize('failure',['larger','cap','extra_pair','wrong_state','wrong_binding',
                                   'missing_reference','other_collision','bad_sample','nan'])
def test_original_design_bound_cannot_waive_new_geometry_or_wrong_provenance(tmp_path,failure):
    context,reference,save=design_reference(tmp_path)
    review=internal_review();pairs=review['results'][0]['intersections_by_pair']
    for item in pairs.values():item.update(worst_sample=2,maximum_radius_m=.00075)
    first=next(iter(pairs))
    if failure=='larger':pairs[first]['maximum_radius_m']+=2.e-9
    if failure=='cap':
        for record in reference['reference_states'].values():
            for item in record['intersections_by_pair'].values():item['maximum_radius_m']=.001
    if failure=='extra_pair':pairs['left_index_DP|fr3_left_link7']={'worst_sample':1,'maximum_radius_m':.0001}
    if failure=='wrong_state':reference['reference_states']['original_release_initial']['joint_positions_rad']['joint']+=.001
    if failure=='wrong_binding':reference['source_binding']['source_sample']+=1
    if failure=='missing_reference':reference['reference_states'].pop('original_release_initial')
    if failure=='other_collision':reference['reference_states']['recorded_native_start']['all_non_design_pairs_clear']=False
    if failure=='bad_sample':pairs[first]['worst_sample']=3
    if failure=='nan':pairs[first]['maximum_radius_m']=float('nan')
    review['design_contact_reference']=save('design.json',reference)
    with pytest.raises(ValueError):validate_internal_review(review,3,reference_context=context)
