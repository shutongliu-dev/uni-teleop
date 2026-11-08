"""Observed local regrasp plans for single-garment engineering folds.

Only declared local patches are controlled. Each phase must be bound to the
current native vertices after all previous attachments have been disabled.
The caller owns real robot motion, attachment authoring and native stepping;
this module never writes cloth positions or velocities.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from scripts.validation.laundry_engineering_motion import minimum_jerk

PROFILE = 'single_garment_hold_and_fold_v1'
TASKS = ('shirt_folding', 'trousers_folding')


def _diameter(points):
    return float(np.linalg.norm(points[:, None]-points[None], axis=-1).max()) if len(points) else 0.


def _material_fragments(ids, triangles):
    selected = set(map(int, ids))
    neighbors = {i: set() for i in selected}
    for tri in triangles:
        included = selected.intersection(map(int, tri))
        for i in included:
            neighbors[i].update(included-{i})
    pending, fragments = selected.copy(), []
    while pending:
        todo, fragment = [pending.pop()], []
        while todo:
            i = todo.pop()
            fragment.append(i)
            new = neighbors[i] & pending
            pending -= new
            todo.extend(new)
        fragments.append(sorted(fragment))
    return fragments


def _select_patch(q, material, triangles, pool, target_xy, *, radius=.026, max_nodes=32):
    """Select a small observed clamp from a declared material feature pool."""
    pool = np.unique(np.asarray(pool, dtype=int))
    if not len(pool):
        raise ValueError('Empty declared garment grasp feature')
    distances = np.linalg.norm(q[pool, :2]-target_xy, axis=1)
    nearest = int(pool[np.argmin(distances)])
    # A folded cuff can expose both fabric sides. Keep those local layers, but
    # never reach through a distant hanging panel or control the whole garment.
    for actual_radius in (radius, radius*.8, radius*.6, radius*.4):
        ids = pool[(np.linalg.norm(q[pool, :2]-q[nearest, :2], axis=1) <= actual_radius)
                   & (np.abs(q[pool, 2]-q[nearest, 2]) <= .020)]
        if len(ids) > max_nodes:
            order = np.argsort(np.linalg.norm(q[ids]-q[nearest], axis=1), kind='stable')
            ids = ids[order[:max_nodes]]
        if len(ids) < 2:
            continue
        ids = sorted(map(int, ids))
        fragments = _material_fragments(ids, triangles)
        # A tightly folded boundary may bring a long connected material arc
        # into one small world region. Trim that arc around its local seed;
        # geometric proximity alone must never weld a whole cuff/hem loop.
        local_ids = []
        for fragment in fragments:
            seed = min(fragment, key=lambda i: float(np.linalg.norm(q[i]-q[nearest])))
            local_ids.extend(i for i in fragment if np.linalg.norm(material[i]-material[seed]) <= .036)
        ids = sorted(set(local_ids))
        if len(ids) < 2:
            continue
        fragments = _material_fragments(ids, triangles)
        material_diameters = [_diameter(material[fragment]) for fragment in fragments]
        world_diameter = _diameter(q[ids])
        if world_diameter <= .075 and max(material_diameters) <= .075:
            return ids, {'world_diameter_m': world_diameter,
                'material_fragments': fragments, 'material_fragment_diameters_m': material_diameters,
                'selection_radius_xy_m': actual_radius, 'selection_z_distance_m': .020,
                'material_radius_m': .036}
    raise ValueError('Observed grasp cannot form a local 75 mm world/material clamp with two samples')


@dataclass
class GarmentFoldingMotion:
    task_id: str
    spec: dict
    phases: list
    duration_s: float
    release_time_s: float
    metadata: dict
    peak_height_m: float
    grip_rotation: str
    withdraw_height_m: float
    bound: dict = field(default_factory=dict)

    def bind_phase(self, index, vertices):
        """Read current q and return two complete attachment declarations."""
        if type(index) is not int or not 0 <= index < len(self.phases) or index in self.bound:
            raise ValueError('Bind each declared folding phase exactly once, in order')
        if index != len(self.bound):
            raise ValueError('Garment phases must be bound in order')
        q = np.asarray(vertices, dtype=float)
        material = np.asarray(self.spec['rest_vertices_m'], dtype=float)
        triangles = np.asarray(self.spec['triangles'], dtype=int)
        if q.shape != material.shape or not np.isfinite(q).all():
            raise ValueError('Complete current finite native garment state required')
        frame = self.metadata.get('fold_coordinate_frame')
        world_q = q
        if frame is not None:
            from scenes.workrooms.laundry_room.garment_folding_assets import transform_fold_frame
            q = transform_fold_frame(q, frame, inverse=True)
        phase = self.phases[index]
        features = self.spec['boundary_features']
        fold_pool = np.unique(np.concatenate([features[name] for name in phase['fold_features']]))
        fold_axis, _, fold_sign, _ = phase['reference_fold']
        if phase['name'] in ('right_sleeve', 'left_sleeve'):
            target = np.median(q[fold_pool, :2], axis=0)
            target[fold_axis] = np.quantile(q[fold_pool, fold_axis], .85 if fold_sign > 0 else .15)
        else:
            target = np.median(q[fold_pool, :2], axis=0)
        dual_fold = phase['anchor_roles'] == ['fold', 'fold']
        if self.task_id == 'trousers_folding' and phase['name'] == 'legs_together':
            # A cuff-only pull leaves the same-side waistband spread out.
            # Carry both ends of the moving leg/body panel across the hinge;
            # the opposite half remains supported by the actual table.
            waist_pool = np.asarray(features['waistband'], dtype=int)
            waist_pool = waist_pool[material[waist_pool, 0] < -.02]
            inner_x = float(phase['fold_patch_material_x_m'])
            def inner_target(pool):
                nearest = pool[np.argsort(abs(material[pool, 0]-inner_x), kind='stable')[:4]]
                return np.median(q[nearest, :2], axis=0)
            waist_target = inner_target(waist_pool)
            target = inner_target(fold_pool)
            folds = [_select_patch(q, material, triangles, waist_pool, waist_target),
                     _select_patch(q, material, triangles, fold_pool, target)]
        elif dual_fold:
            # Two short clamps at opposite ends of the current real hem/cuff
            # span carry the whole free edge across its mirror line.
            targets = [target.copy(), target.copy()]
            tangent_axis = 1-fold_axis
            for value, quantile in zip(targets, (.05, .95)):
                value[tangent_axis] = np.quantile(q[fold_pool, tangent_axis], quantile)
            folds = [_select_patch(q, material, triangles, fold_pool, value) for value in targets]
        else:
            folds = [_select_patch(q, material, triangles, fold_pool, target,
                                   radius=phase.get('fold_patch_radius_m', .026))]

        xy = material[:, :2]
        if self.task_id == 'shirt_folding':
            y_range = (.12, .22) if phase['name'] == 'body_half' else (-.18, -.04)
            hold_pool = np.flatnonzero((abs(xy[:, 0]) < .075)
                & (xy[:, 1] >= y_range[0]) & (xy[:, 1] <= y_range[1]))
        else:
            # The positive-X waist is the stationary material half in both
            # trouser stages; prior left-leg folding cannot move this seed.
            region = phase.get('hold_material_region_xy_m', [.045, .145, .24, .34])
            hold_pool = np.flatnonzero((xy[:, 0] > region[0]) & (xy[:, 0] < region[1])
                                      & (xy[:, 1] > region[2]) & (xy[:, 1] < region[3]))
        hold_target = np.median(q[hold_pool, :2], axis=0)
        hold_ids, hold_receipt = _select_patch(q, material, triangles, hold_pool, hold_target)
        clamps = folds if dual_fold else [(hold_ids, hold_receipt), folds[0]]
        all_ids = set(clamps[0][0]+clamps[1][0])
        if set(clamps[0][0]) & set(clamps[1][0]) or len(all_ids) > .10*len(q):
            raise ValueError('Two distinct local clamps must stay below ten percent of the garment')
        origins = [q[ids].mean(axis=0) for ids, _ in clamps]
        if np.linalg.norm(origins[0]-origins[1]) < .075:
            raise ValueError('Holding and folding patches cannot overlap physically')
        reference = np.asarray(self.spec['unfolded_reference_vertices_m'], dtype=float)
        if frame is not None:
            reference = transform_fold_frame(reference, frame, inverse=True)
        observed_shift = (q[hold_ids, :2]-reference[hold_ids, :2]).mean(axis=0)
        axis, hinge, sign, radius = phase['reference_fold']
        hinge = float(hinge+observed_shift[axis])
        thickness = float(self.spec['physical_thickness_m'])
        anchors = []
        for role, (ids, receipt), origin in zip(phase['anchor_roles'], clamps, origins):
            release = phase['hold_release_time_s'] if role == 'hold' else phase['fold_release_time_s']
            world_origin = (transform_fold_frame(origin[None], frame)[0] if frame is not None else origin)
            anchor = {'role': role, 'cloth_id': 'towel', 'vertex_ids': ids,
                'origin_m': world_origin.tolist(), 'offsets_m': (world_q[ids]-world_origin).tolist(),
                'release_time_s': release, 'locality': receipt}
            if role == 'fold':
                destination = origin.copy()
                destination[axis] = 2*hinge-destination[axis]+sign*2.5*thickness
                support_ids = np.flatnonzero(np.linalg.norm(q[:, :2]-destination[:2], axis=1) < .035)
                support_ids = np.setdiff1d(support_ids, list(all_ids))
                support_z = float(np.quantile(q[support_ids, 2], .90)) if len(support_ids) else .90+thickness
                if support_z > .99:
                    raise ValueError('Observed target is not a table-supported garment region')
                destination[2] = support_z+float(np.ptp(q[ids, 2]))/2+1.25*thickness
                middle = (origin+destination)/2
                middle[2] = max(origin[2], destination[2])+2*phase.get('peak_height_m', self.peak_height_m)
                controls = np.asarray([origin, middle, destination])
                if frame is not None:
                    controls = transform_fold_frame(controls, frame)
                anchor.update(destination_m=controls[2].tolist(),
                    bezier_controls_m=controls.tolist(),
                    observed_target_support_z_m=support_z)
            anchors.append(anchor)
        record = {'phase_index': index, 'phase_name': phase['name'], 'anchors': anchors,
            'observed_hold_shift_xy_m': observed_shift.tolist(), 'observed_hinge_m': hinge,
            'controlled_fraction': len(all_ids)/len(q),
            'initialization': 'Current native q read only; caller must disable previous attachments before binding offsets.',
            'free_cloth_state_written': False, 'success_assumed': False}
        if frame is not None:
            record['fold_coordinate_frame'] = dict(frame)
        self.bound[index] = record
        self.metadata['bound_phases'].append(record)
        return record

    def active(self, phase_index, anchor_index, time_s):
        phase = self.phases[phase_index]
        if anchor_index not in (0, 1):
            raise ValueError('Expected hold or fold anchor')
        release = (phase['hold_release_time_s'] if phase['anchor_roles'][anchor_index] == 'hold'
                   else phase['fold_release_time_s'])
        return phase['grasp_time_s'] <= time_s < release

    def pose(self, phase_index, anchor_index, time_s):
        if phase_index not in self.bound or anchor_index not in (0, 1) or not math.isfinite(time_s):
            raise ValueError('A bound phase, one of two anchors and finite time are required')
        phase, bound = self.phases[phase_index], self.bound[phase_index]
        anchor = bound['anchors'][anchor_index]
        origin = np.asarray(anchor['origin_m'])
        if anchor['role'] == 'hold':
            centre = origin.copy()
        else:
            s = minimum_jerk(time_s, phase['move_begin_s'], phase['move_end_s'])
            a, b, c = np.asarray(anchor['bezier_controls_m'])
            centre = (1-s)**2*a+2*s*(1-s)*b+s*s*c
        # Disabled anchors withdraw; a failure to actually disable would be
        # visible as renewed lifting and cannot silently pass release.
        centre[2] += self.withdraw_height_m*minimum_jerk(time_s, anchor['release_time_s'],
                                     anchor['release_time_s']+.60)
        rotation = np.array([1., 0., 0., 0.])
        if anchor['role'] == 'fold' and self.grip_rotation == 'fold_hinge':
            axis, _, sign, _ = phase['reference_fold']
            progress = minimum_jerk(time_s, phase['move_begin_s'], phase['move_end_s'])
            angle = sign*(2*axis-1)*np.pi*progress
            rotation[0] = np.cos(angle/2)
            rotation[1+(1-axis)] = np.sin(angle/2)
            frame = self.metadata.get('fold_coordinate_frame')
            if frame is not None:
                from scenes.workrooms.laundry_room.garment_folding_assets import _rotate_degrees
                rotation[1:] = _rotate_degrees(rotation[None, 1:], frame['yaw_degrees'], np.zeros(2))[0]
        return centre.tolist(), rotation.tolist()


def build_folding_motion(task, scene, *, settle_s=4., fold_duration_s=5.,
                         interphase_passive_s=1., final_passive_s=5., peak_height_m=.06,
                         hold_release_fraction=.65, grip_rotation='fixed', withdraw_height_m=.06,
                         body_fold_peak_height_m=None, first_leg_grasp_material_x_m=None,
                         second_hold_material_region_xy_m=None):
    task_id = task['id']
    if task_id not in TASKS or len(task['asset_metadata']['cloths']) != 1:
        raise ValueError('Single-garment folding task required')
    spec = task['asset_metadata']['cloths'][0]
    if spec['id'] != 'towel' or spec.get('mesh_type') != 'custom_surface':
        raise ValueError('Expected the released custom sewn garment surface')
    if grip_rotation not in ('fixed', 'fold_hinge'):
        raise ValueError('Explicit fixed or fold_hinge engineering clamp orientation required')
    if first_leg_grasp_material_x_m is not None:
        if (task_id != 'trousers_folding' or type(first_leg_grasp_material_x_m) not in (int, float)
                or not math.isfinite(first_leg_grasp_material_x_m)
                or not -.19 <= first_leg_grasp_material_x_m < -.02):
            raise ValueError('First trouser grasp must select the moving negative-X material half')
    if second_hold_material_region_xy_m is not None:
        region = np.asarray(second_hold_material_region_xy_m, dtype=float)
        if (task_id != 'trousers_folding' or region.shape != (4,) or not np.isfinite(region).all()
                or not 0. <= region[0] < region[1] <= .19 or not 0. < region[2] < region[3] <= .37):
            raise ValueError('Second trouser hold must stay in a finite stationary positive-X/Y material region')
    if (body_fold_peak_height_m is not None and
            (type(body_fold_peak_height_m) not in (int, float) or not math.isfinite(body_fold_peak_height_m)
             or not .02 <= body_fold_peak_height_m <= .15)):
        raise ValueError('Optional transverse body fold peak must be finite and within 20..150 mm')
    values = [settle_s, fold_duration_s, interphase_passive_s, final_passive_s, peak_height_m,
              hold_release_fraction, withdraw_height_m]
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError('Finite positive settling, movement, release and lift parameters required')
    dt = float(scene['physics']['dt_s'])
    stable = float(task['evaluation']['thresholds']['stable_s'])
    if (final_passive_s < stable or not .02 <= peak_height_m <= .15 or fold_duration_s < 1.
            or not .5 <= hold_release_fraction <= .9 or not .03 <= withdraw_height_m <= .12):
        raise ValueError('Explicit profile must retain a complete stable window and bounded slow fold arc')
    sequence = task['asset_metadata']['design']['fold_sequence_axis_hinge_sign_radius_m']
    descriptions = ([('right_sleeve', ['right_cuff']), ('left_sleeve', ['left_cuff']),
                     ('body_half', ['lower_hem'])] if task_id == 'shirt_folding' else
                    [('legs_together', ['left_cuff']), ('waist_half', ['left_cuff', 'right_cuff'])])
    if len(sequence) != len(descriptions):
        raise ValueError('Frozen fold-goal sequence differs from the engineering phases')
    phases, start = [], float(settle_s)
    for index, ((name, features), fold) in enumerate(zip(descriptions, sequence)):
        begin, end = start+.25, start+.25+fold_duration_s
        phase_hold_fraction = .5 if name == 'waist_half' else hold_release_fraction
        phase = {'phase_index': index, 'name': name, 'fold_features': features,
            'peak_height_m': (float(body_fold_peak_height_m) if body_fold_peak_height_m is not None
                and name in ('body_half', 'waist_half') else float(peak_height_m)),
            'anchor_roles': ['fold', 'fold'] if name in ('body_half', 'legs_together') else ['hold', 'fold'],
            'reference_fold': list(fold), 'grasp_time_s': start,
            'move_begin_s': begin, 'move_end_s': end,
            'hold_release_fraction': phase_hold_fraction,
            'hold_release_time_s': begin+phase_hold_fraction*fold_duration_s,
            'fold_release_time_s': end+.5}
        if name == 'legs_together':
            phase['fold_patch_sources'] = [{'feature': 'waistband', 'material_x_max_m': -.02},
                                            {'feature': 'left_cuff'}]
            phase['fold_patch_material_x_m'] = (-.07 if first_leg_grasp_material_x_m is None
                                                else float(first_leg_grasp_material_x_m))
        if name == 'waist_half':
            phase['fold_patch_radius_m'] = .035
            phase['hold_material_region_xy_m'] = ([.020, .065, .26, .34]
                if second_hold_material_region_xy_m is None else list(map(float, second_hold_material_region_xy_m)))
        phases.append(phase)
        start = phase['fold_release_time_s']+interphase_passive_s
    release = phases[-1]['fold_release_time_s']
    duration = release+final_passive_s
    if duration >= task['evaluation']['thresholds']['max_time_s']:
        raise ValueError('Engineering fold duration exceeds the frozen task time budget')
    for time in [settle_s, duration, *[p[key] for p in phases for key in
                 ('grasp_time_s', 'move_begin_s', 'move_end_s', 'hold_release_time_s', 'fold_release_time_s')]]:
        if abs(time/dt-round(time/dt)) > 1.e-7:
            raise ValueError('All phase events must land on exact native physics steps')
    metadata = {'schema': 'laundry_garment_folding_motion.v1', 'profile': PROFILE,
        'task_id': task_id, 'settle_s': settle_s, 'phases': phases, 'bound_phases': [],
        'duration_s': duration, 'release_time_s': release, 'final_passive_s': final_passive_s,
        'hold_release_fraction': hold_release_fraction, 'release_withdraw_delay_s': 0.,
        'withdraw_height_m': withdraw_height_m,
        'peak_height_m': peak_height_m, 'grip_rotation': grip_rotation,
        'orientation': ('Local attachment patch rolls 180 degrees about the fold-hinge tangent; the caller may keep the actual wrist down-facing. This is an engineering edge-rolling clamp, not a frictional grasp claim.'
            if grip_rotation == 'fold_hinge' else 'Constant identity engineering clamp orientation.'),
        'maximum_simultaneously_active_anchors': 2, 'target_shape_written': False,
        'control': 'Sleeves use hold+fold. The first trouser fold carries the moving waistband and cuff together with two local fold clamps. The shirt transverse fold uses both hem ends. Narrow paired trouser cuffs use one fold clamp while the other hand holds the waist. Other cloth nodes remain dynamic.',
        'phase_transition': 'disable all previous attachments before selecting next actual-q patch and offsets',
        'inspiration': 'https://github.com/robosimstudio/robosimstudio/blob/main/examples/08_demos/fold_the_towel/fold_the_towel.py',
        'scope': 'Engineering cloth feasibility; real robot joints and near-hand consistency require separate caller evidence.'}
    frame = task['asset_metadata']['design'].get('fold_coordinate_frame')
    if frame is not None:
        metadata['fold_coordinate_frame'] = dict(frame)
    return GarmentFoldingMotion(task_id, spec, phases, duration, release, metadata, peak_height_m,
                                grip_rotation, withdraw_height_m)
