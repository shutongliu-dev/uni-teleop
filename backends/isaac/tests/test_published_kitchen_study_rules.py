"""Adversarial current-release rule checks; synthetic CPU traces, not physics.

Load the exact catalog-pinned evaluator and native observer rather than assuming
that mutable authoring code still matches a published release.
"""
from copy import deepcopy
from functools import lru_cache
import importlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from bench.isaacsim.release import digest, load_evaluator
from bench.paths import load_json
from scenes.workrooms.catalog import list_tasks, task_definition

KITCHEN = tuple(row['task_id'] for row in list_tasks('kitchen'))
# The placement trace has no hinge/slider evidence. Drawer and reading stand
# have dedicated mechanism fixtures and tests. Select by declared rule schema.
STUDY = tuple(row['task_id'] for row in list_tasks('study')
              if row['rules']['contract']['schema'] == 'study_rules_v1')
PLACEMENT = ('produce_transfer', 'utensil_sorting', 'pantry_stocking', 'table_clearing')
HAND = '/World/Robot/hand/right_index_DP'


@lru_cache(None)
def published(family, ident):
    reference = task_definition(family, ident)['source_release']
    root = Path(reference['path'])
    assert digest(root/'release_manifest.json') == reference['manifest_sha256']
    task = load_json(root/'task.json')
    factory = load_evaluator(root)
    return task, factory, importlib.import_module(factory.__module__)


class KitchenTrace:
    def __init__(self, ident):
        self.task, factory, self.module = published('kitchen', ident)
        self.observer = self.module.NativeTaskObserver(self.task)
        self.evaluator = factory(self.task)
        self.meta = self.task['evaluation']['metadata']
        self.dt = self.task['evaluation']['physics_dt_s']
        self.interval = round(self.task['evaluation']['control_period_s']/self.dt)
        self.objects = {row['id']: self.module.fixed_state(row['initial_pose_xyzw'])
                        for row in self.meta['objects']}
        # Synthetic rule fixtures declare a rigidly transported, undeformed
        # elastic body with complete joint/clearance evidence. They are not a
        # native material simulation; all physical qualification is separate.
        self.elastic = self.meta.get('elastic_composites', {}).get('sponge')
        self.elastic_joints = {}
        if self.elastic:
            self.elastic_joints = {link['prim_path'].rsplit('/', 1)[-1]: {
                'kind': 'elastic_d6', 'body0': link['body0'], 'body1': link['body1'],
                'anchor_delta_world_m': [0., 0., 0.], 'anchor_delta_joint0_m': [0., 0., 0.],
                'anchor_separation_m': 0., 'relative_rotation_matrix': np.eye(3).tolist(),
                'force_measured': False} for link in self.elastic['links']}
        self.contacts, self.index = [], 0
        self.send(baseline=True)

    def move_sponge(self, pose):
        """Move the whole synthetic sample, preserving its rest relative poses."""
        self.objects['sponge'] = self.module.fixed_state(pose)
        if not self.elastic:
            return
        specs = self.observer.specs
        initial = specs['sponge']['initial_pose_xyzw']
        delta = Rotation.from_quat(pose[3:]) * Rotation.from_quat(initial[3:]).inv()
        for name in self.elastic['member_ids']:
            rest = specs[name]['initial_pose_xyzw']
            point = delta.apply(np.asarray(rest[:3])-initial[:3])+pose[:3]
            quat = (delta*Rotation.from_quat(rest[3:])).as_quat()
            self.objects[name] = self.module.fixed_state([*point, *quat])

    def pair(self, first, second, impulse=(0., 0., .001)):
        paths = {**self.meta['actors'], **self.meta['fixtures']}
        return {'actor0': self.module.native_room_path(paths.get(first, first)),
                'actor1': self.module.native_room_path(paths.get(second, second)),
                'point_count': 1, 'impulse_magnitude_sum_ns': math.sqrt(sum(v*v for v in impulse)),
                'impulse_vector_ns': list(impulse), 'min_separation_m': 0.}

    def send(self, *, baseline=False):
        if not baseline:
            self.index += 1
        time = self.index*self.dt
        frame = {'baseline': baseline, 'time_s': time, 'native_time_s': time,
            'native_step': self.index, 'physics_dt_s': self.dt, 'objects': deepcopy(self.objects),
            'contacts': [] if baseline else [{**row, 'time_s': time} for row in self.contacts]}
        if self.elastic:
            frame.update(joints=deepcopy(self.elastic_joints), elastic_robot_geometry={
                'complete': True, 'fully_separated': True, 'clearance_m': .005,
                'method': 'native_cooked_rigid_composite_geometry'})
        self.observer.on_frame(frame)
        if self.index % self.interval == 0:
            self.result = self.evaluator.update({'time_s': time, 'task_state': self.observer.on_observe()})
        return self.result

    def run(self, duration=.46):
        for _ in range(round(duration/self.dt)):
            self.send()
        return self.result

    def targets(self):
        goals = self.meta['goals']
        self.objects = {name: self.module.fixed_state(pose) for name, pose in goals['body_poses_xyzw'].items()}
        self.contacts = [self.pair(name, target) for name, target in goals['support_by_actor'].items()]


class StudyTrace:
    def __init__(self, ident):
        self.task, factory, self.module = published('study', ident)
        self.config = self.task['evaluation']['configuration']
        assert self.config['schema'] == 'study_task_configuration_v1'
        self.evaluator = factory(self.task, trusted_normalized_input=True)
        self.objects = {name: {**deepcopy(row['initial_pose']), 'linear_velocity_m_s': [0., 0., 0.],
                        'angular_velocity_rad_s': [0., 0., 0.]} for name, row in self.config['objects'].items()}
        self.contacts, self.index = [], -1
        self.send()

    def send(self, **changes):
        self.index += 1
        frame = {'schema': 'study_native_observation_v1',
            'configuration_sha256': self.config['configuration_sha256'], 'episode_id': 'published_adversarial',
            'step_index': self.index, 'time_s': self.index*.0005, 'physics_dt_s': .0005,
            'sampling_complete': True, 'contacts_complete': True, 'physics_valid': True,
            'robot_only_actions': True, 'object_pose_writes': False, 'budget_exhausted': False,
            'robot_action_count': int(self.index > 0), 'objects': deepcopy(self.objects),
            'contacts': deepcopy(self.contacts)}
        frame.update(changes)
        self.result = self.evaluator.update(frame)
        return self.result

    def run(self, count=603):
        for _ in range(count):
            self.send()
        return self.result

    def targets(self):
        for name, goal in self.config['targets'].items():
            obj = self.config['objects'][name]
            q = ([math.sqrt(.5), math.sqrt(.5), 0., 0.]
                 if obj.get('up_axis_local') == [0, 1, 0]
                 else [1., 0., 0., 0.] if 'up_axis_local' in obj else obj['initial_pose']['quaternion_wxyz'])
            pose = {'position_m': [0., 0., 0.], 'quaternion_wxyz': q,
                    'linear_velocity_m_s': [0., 0., 0.], 'angular_velocity_rad_s': [0., 0., 0.]}
            low, _ = self.module.world_bounds(self.evaluator._state(pose), obj['local_bounds_m'])
            pose['position_m'] = [(goal['bounds_m'][0][axis]+goal['bounds_m'][1][axis])/2 for axis in (0, 1)] + [goal['support_z_m']-low[2]]
            self.objects[name] = pose
        self.contacts = [{'a': name, 'b': goal['support_id'],
            'point_m': [*self.objects[name]['position_m'][:2], goal['support_z_m']],
            'separation_m': 0., 'impulse_on_a_n_s': [0., 0., .001]}
            for name, goal in self.config['targets'].items()]


@pytest.mark.parametrize('ident', KITCHEN)
def test_published_kitchen_reset_and_skipped_native_frame_never_score(ident):
    trace = KitchenTrace(ident)
    assert trace.result['reason'] == 'reset_baseline_not_scored'
    assert not trace.run()['success']
    trace.index += 1
    trace.run(.02)
    assert trace.result['status'] == 'unknown'


@pytest.mark.parametrize('ident', PLACEMENT)
@pytest.mark.parametrize('defect', ['missing_root', 'downward', 'held', 'center_only'])
def test_published_kitchen_whole_shape_support_and_release_reject_then_recover(ident, defect):
    trace = KitchenTrace(ident)
    trace.targets()
    clean_objects, clean_contacts = deepcopy(trace.objects), deepcopy(trace.contacts)
    name = next(iter(trace.objects))
    if defect == 'missing_root':
        # Remove a real root without retaining any prior support witness.
        trace.contacts = []
    elif defect == 'downward':
        trace.contacts = [{**row, 'impulse_vector_ns': [0., 0., -.001]} for row in trace.contacts]
    elif defect == 'held':
        trace.contacts.append(trace.pair(name, HAND, (0., 0., 0.)))
    else:
        spec = trace.observer.specs[name]
        target = (name.split('_')[0]+'_tray' if ident == 'produce_transfer'
                  else spec.get('assigned_fixture', 'cutlery_tray'))
        fixture = trace.observer.fixtures[target]
        bounds = (fixture['channels'][name.split('_')[0]]['interior_bounds_xy_m']
                  if ident == 'utensil_sorting' else fixture['interior_bounds_xy_m'])
        # The center is inside the declared region, but the shape crosses its edge.
        trace.objects[name]['position_m'] = list(trace.objects[name]['position_m'])
        trace.objects[name]['position_m'][0] = fixture['initial_pose_xyzw'][0]+bounds[1][0]-.001
    assert not trace.run()['success']
    trace.objects, trace.contacts = clean_objects, clean_contacts
    assert trace.run()['success'], trace.result


@pytest.mark.parametrize('ident', PLACEMENT)
def test_published_kitchen_shared_window_cannot_join_before_and_after_hand_contact(ident):
    trace = KitchenTrace(ident)
    trace.targets()
    assert not trace.run(.24)['success']
    trace.contacts.append(trace.pair(next(iter(trace.objects)), HAND, (0., 0., 0.)))
    trace.send()
    trace.contacts.pop()
    assert not trace.run(.24)['success']
    assert trace.run(.24)['success']


@pytest.mark.parametrize('ident', STUDY)
@pytest.mark.parametrize('defect', ['missing_root', 'wrong_point', 'held', 'center_only'])
def test_published_study_entire_shape_signed_support_release_and_recovery(ident, defect):
    trace = StudyTrace(ident)
    assert trace.result['reason'] == 'reset_baseline_not_scored'
    trace.targets()
    clean_objects, clean_contacts = deepcopy(trace.objects), deepcopy(trace.contacts)
    name = next(iter(trace.objects))
    if defect == 'missing_root':
        trace.contacts.pop(0)
    elif defect == 'wrong_point':
        trace.contacts[0]['point_m'][0] += 2.
    elif defect == 'held':
        trace.contacts.append({'a': name, 'b': 'robot/finger', 'point_m': trace.objects[name]['position_m'],
                               'separation_m': 0., 'impulse_on_a_n_s': [0., 0., 0.]})
    else:
        trace.objects[name]['position_m'][0] = trace.config['targets'][name]['bounds_m'][1][0]-.001
    assert not trace.run()['success']
    trace.objects, trace.contacts = clean_objects, clean_contacts
    assert trace.run()['success'], trace.result


@pytest.mark.parametrize('ident', STUDY)
def test_published_study_cannot_accumulate_disjoint_shared_windows(ident):
    trace = StudyTrace(ident)
    trace.targets()
    assert not trace.run(400)['success']
    trace.contacts.append({'a': next(iter(trace.objects)), 'b': 'robot/palm', 'point_m': [0., 0., 1.],
                           'separation_m': 0., 'impulse_on_a_n_s': [0., 0., 0.]})
    trace.send()
    trace.contacts.pop()
    assert not trace.run(400)['success']
    assert trace.run(203)['success']


def test_published_breakfast_destination_assembly_cannot_replace_loaded_carry():
    trace = KitchenTrace('breakfast_serving')
    trace.targets()
    assert not trace.run()['success']
    assert not trace.result['detail']['carry_complete']


def breakfast_at(trace, x, z, *, held=False, touch_payload=False):
    trace.targets()
    original = trace.meta['goals']['body_poses_xyzw']['tray']
    for pose in trace.objects.values():
        pose['position_m'] = [pose['position_m'][0]+x-original[0],
                              pose['position_m'][1], pose['position_m'][2]+z-original[2]]
    if held:
        trace.contacts = [row for row in trace.contacts if row['actor0'] != trace.meta['actors']['tray']]
        trace.contacts.append(trace.pair('tray', HAND))
    if touch_payload:
        trace.contacts.append(trace.pair('bread_0', HAND, (0., 0., 0.)))


@pytest.mark.parametrize('touch_payload', [False, True])
def test_published_breakfast_loaded_trip_credit_belongs_to_current_attempt(touch_payload):
    trace = KitchenTrace('breakfast_serving')
    source = trace.meta['goals']['source_tray_pose_xyzw']
    destination = trace.meta['goals']['destination_tray_pose_xyzw']
    breakfast_at(trace, source[0], source[2])
    trace.run(.34)
    assert trace.result['detail']['source_ready']
    for index in range(61):
        x = source[0]+(destination[0]-source[0])*index/60
        breakfast_at(trace, x, source[2]+.04, held=True,
                     touch_payload=touch_payload and index == 30)
        trace.send()
    breakfast_at(trace, destination[0], destination[2])
    result = trace.run(.48)
    assert result['success'] is (not touch_payload)
    if touch_payload:
        assert not result['detail']['carry_complete']
        assert any(event['kind'] == 'carry_retry_required' for event in trace.observer.events)


def test_published_pot_preassembled_pair_cannot_replace_pot_first_stage():
    trace = KitchenTrace('pot_lid_placement')
    trace.targets()
    assert not trace.run(.8)['success']
    assert not trace.result['detail']['pot_ready']
    lid = deepcopy(trace.objects['lid'])
    trace.objects['lid'] = trace.module.fixed_state(trace.observer.specs['lid']['initial_pose_xyzw'])
    trace.contacts.pop()
    trace.run(.36)
    assert trace.result['detail']['pot_ready']
    trace.objects['lid'] = lid
    trace.contacts.append(trace.pair('lid', 'pot'))
    assert trace.run()['success'], trace.result


def test_published_wiping_missing_contact_interval_does_not_bridge_coverage():
    trace = KitchenTrace('surface_wiping')
    model = trace.observer.wiping
    cell = model.cells[0]['center_world_m']
    z = model.goals['wiping_surface_z_m']-model.spec['wiping_face_local_z_m']
    def pose(x):
        return trace.module.fixed_state([x, cell[1], z, 0., 0., 0., 1.])
    # Two loaded endpoints with a contact-free traversal in between must not
    # manufacture the intervening loaded 12 mm wiping stroke.
    model.update(pose(cell[0]-.006), loaded_contact=True)
    model.update(pose(cell[0]+.006), loaded_contact=False)
    result = model.update(pose(cell[0]+.006), loaded_contact=True)
    assert not model.travel.any() and not result['cleaning_ready']
    trace.targets()
    assert not trace.run()['success']  # A pristine return also earns no credit.


def test_published_elastic_wiping_rejects_a_missing_material_connection():
    trace = KitchenTrace('surface_wiping')
    assert trace.elastic and len(trace.elastic_joints) == 63
    trace.elastic_joints.pop(next(iter(trace.elastic_joints)))
    trace.run(.02)
    assert trace.result['status'] == 'unknown'
    assert not trace.result['success']


def test_published_wiping_actual_loaded_motion_then_return_passes():
    trace = KitchenTrace('surface_wiping')
    model = trace.observer.wiping
    surface = trace.meta['goals']['wiping_support_paths'][0]
    height = model.goals['wiping_surface_z_m']-model.spec['wiping_face_local_z_m']+.0002
    for patch in model.goals['soil_patches']:
        cells = [row['center_world_m'] for row in model.cells if row['patch'] == patch['id']]
        x, y = [sum(row[axis] for row in cells)/len(cells) for axis in (0, 1)]
        for offset in (.019, -.019):
            trace.contacts = []
            trace.move_sponge([x-.09, y+offset, height+.03, 0., 0., 0., 1.])
            trace.send()
            trace.contacts = [trace.pair('sponge', surface)]
            # 0.6 m/s, strictly below the frozen 0.8 m/s credit cutoff.
            for index in range(601):
                trace.move_sponge([x-.09+index*.0003, y+offset, height, 0., 0., 0., 1.])
                trace.send()
    detail = trace.observer.on_observe()['detail']
    assert detail['cleaning_ready'] and min(detail['patch_coverage'].values()) >= .6
    trace.targets()
    trace.run(.5)
    # Align the final public observation without skipping native samples.
    while trace.index % trace.interval:
        trace.send()
    assert trace.result['success'], trace.result


@pytest.mark.parametrize('ident', [name for name in STUDY if name != 'stationery_sorting'])
def test_published_books_reversed_spine_is_not_cured_by_supported_rest(ident):
    trace = StudyTrace(ident)
    trace.targets()
    name = next(iter(trace.objects))
    original = deepcopy(trace.objects[name])
    w, x, y, z = original['quaternion_wxyz']
    trace.objects[name]['quaternion_wxyz'] = [-z, -y, x, w]  # World-Z half turn.
    result = trace.run()
    assert not result['success']
    assert 'incorrect_orientation' in result['objects'][name]['reasons']
    trace.objects[name] = original
    assert trace.run()['success']


def test_published_book_retrieval_also_requires_protected_books_restored():
    trace = StudyTrace('book_retrieval')
    trace.targets()
    protected = trace.config['protected_ids'][0]
    original = deepcopy(trace.objects[protected])
    trace.objects[protected]['position_m'][1] += .025
    assert not trace.run()['success']
    trace.objects[protected] = original
    assert trace.run()['success']


def test_published_book_stack_rejects_a_floating_support_cycle():
    trace = StudyTrace('book_stacking')
    trace.targets()
    bottom, _, top = trace.config['stack_order']
    trace.contacts = [row for row in trace.contacts if row['a'] != bottom]
    trace.contacts.append({'a': bottom, 'b': top, 'point_m': trace.objects[bottom]['position_m'],
                           'separation_m': 0., 'impulse_on_a_n_s': [0., 0., .001]})
    result = trace.run()
    assert not result['success']
    assert not any(row['supported'] for row in result['objects'].values())


def test_published_drawer_canonical_order_uses_its_frozen_evaluator():
    from test_study_drawer import Trace
    task, factory, _ = published('study', 'drawer_retrieval')
    trace = Trace()
    trace.eval = factory(task, trusted_normalized_input=True)
    trace.open_and_extract()
    trace.target()
    assert trace.settle()['success']
