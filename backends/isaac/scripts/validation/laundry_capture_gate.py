"""Sampled geometric capture prerequisite for explicitly declared local assistance.

This never attaches particles and never qualifies a friction grasp. The caller
limits observations to its closed-hold phase and supplies a measured open-tip
baseline from the same episode. Geometry comes from PinchGeometry, while fixed
material drift uses actual cloth vertices and the native wrist pose.
"""
from __future__ import annotations

import hashlib
import numpy as np

from scripts.validation.laundry_material_topology import face_edge_neighbors, local_material_patch
from scripts.validation.laundry_contact_stability import (
    PROFILE as PRACTICAL_PROFILE, CookedContactMotion, surface_window_metrics)
from scripts.validation.laundry_closure_measurement import (
    validate_closure_measurement, measured_closure_requirement)


def _diameter(points):
    return float(np.linalg.norm(points[:, None]-points[None, :], axis=2).max())


def _local_points(points, pose):
    x, y, z, w = np.asarray(pose[3:7], float)
    norm = x*x+y*y+z*z+w*w
    if abs(norm-1) > .01:
        raise ValueError('Native wrist quaternion must be normalized')
    s = 2/norm
    rotation = np.array([[1-s*(y*y+z*z), s*(x*y-z*w), s*(x*z+y*w)],
                         [s*(x*y+z*w), 1-s*(x*x+z*z), s*(y*z-x*w)],
                         [s*(x*z-y*w), s*(y*z+x*w), 1-s*(x*x+y*y)]])
    return (points-np.asarray(pose[:3])) @ rotation


class CaptureGate:
    """A shared-edge material pair must remain between two actual distal surfaces.

    Call at 20 ms or faster. A gap above 40 ms clears the sampled window. A
    positive result does not establish continuous-time contact between samples,
    normal force, load bearing, friction retention, or successful release.
    """
    def __init__(self, side, triangles, rest_vertices, body_names, joint_names, *, profile='strict_v1',
                 closure_measurement_profile='strict_v1', planned_open_tip_gap_m=None,
                 material_capture_region=None, contact_normal_profile='facet_v1',
                 contact_witness_profile='centroid_v1', contact_motion_reference=None):
        from scripts.validation.laundry_contact_observation import validate_normal_profile
        validate_normal_profile(contact_normal_profile)
        self.contact_normal_profile=contact_normal_profile
        from scripts.validation.laundry_contact_observation import validate_witness_profile
        validate_witness_profile(contact_witness_profile, contact_normal_profile)
        self.contact_witness_profile = contact_witness_profile
        validate_closure_measurement(closure_measurement_profile, planned_open_tip_gap_m)
        self.closure_measurement_profile=closure_measurement_profile
        self.planned_open_tip_gap_m=planned_open_tip_gap_m
        self.measurement_evidence=None
        if side not in ('left', 'right'):
            raise ValueError('Original left/right hand required')
        if profile not in ('strict_v1','local_assist_v2','adjacent_patch_v3','angled_adjacent_patch_v4','loaded_adjacent_patch_v5',PRACTICAL_PROFILE):
            raise ValueError('Unknown explicit capture-assistance tolerance profile')
        if profile == PRACTICAL_PROFILE and (
                contact_normal_profile != 'distance_gradient_v1'
                or contact_witness_profile not in ('facet_regions_v1', 'surface_regions_v2')):
            raise ValueError('Practical hold requires finite-area true distance-gradient witnesses')
        self.profile=profile
        self.side = side
        self.tri = np.asarray(triangles, np.int64).copy()
        self.rest = np.asarray(rest_vertices, float).copy()
        self.body_names, self.joint_names = list(body_names), list(joint_names)
        if (self.tri.ndim != 2 or self.tri.shape[1] != 3 or self.rest.ndim != 2
                or self.rest.shape[1] != 3 or not np.isfinite(self.rest).all()
                or self.tri.min() < 0 or self.tri.max() >= len(self.rest)):
            raise ValueError('Finite matching original cloth topology required')
        from scripts.validation.laundry_capture_region import MaterialCaptureRegion
        self.material_region = (None if material_capture_region is None else
                                MaterialCaptureRegion(material_capture_region,self.tri,self.rest))
        self.wrist = self.body_names.index(side+'_hand_wrist')
        self.thumb = self.body_names.index(side+'_thumb_fingertip')
        self.index = self.body_names.index(side+'_index_fingertip')
        self.fingers = [i for i, n in enumerate(self.joint_names)
                        if n.startswith(side+'_thumb_') or n.startswith(side+'_index_')]
        if not self.fingers:
            raise ValueError('Original thumb/index joint inventory required')
        self.history = []
        self.contact_motion = (CookedContactMotion(contact_motion_reference, side, self.body_names)
                               if profile == PRACTICAL_PROFILE else None)
        if profile != PRACTICAL_PROFILE and contact_motion_reference is not None:
            raise ValueError('Cooked settling reference requires the explicit practical profile')
        self.face_neighbors=face_edge_neighbors(self.tri) if profile in ('adjacent_patch_v3','angled_adjacent_patch_v4','loaded_adjacent_patch_v5',PRACTICAL_PROFILE) else None
        self.thresholds = dict(minimum_duration_s=.1, maximum_sample_gap_s=.04,
            maximum_material_diameter_m=.035, maximum_world_diameter_m=.035,
            maximum_vertices=32, maximum_joint_error_rad=.03,
            maximum_joint_speed_rad_s=.10, maximum_joint_range_rad=.005,
            minimum_tip_closing_m=.008, minimum_tip_closing_fraction=.20,
            maximum_material_hand_drift_m=.001, maximum_opposing_normal_dot=-.5)
        if profile in ('local_assist_v2','adjacent_patch_v3','angled_adjacent_patch_v4','loaded_adjacent_patch_v5',PRACTICAL_PROFILE):
            # This is only a prerequisite for DECLARED local holding aid.
            # Small settling inside a visibly occupied finger slot and a
            # contact-loaded servo error are compatible with that mechanism;
            # neither demonstrates friction retention. Geometry, material
            # identity, dwell, joint motion and deep-penetration gates remain.
            self.thresholds.update(maximum_joint_error_rad=.04,
                                   maximum_material_hand_drift_m=.002)

        if profile=='angled_adjacent_patch_v4':
            self.thresholds['maximum_opposing_normal_dot']=-.25
        if profile=='loaded_adjacent_patch_v5':
            # A stable cloth-filled slot can stop the original fingers short
            # of their unloaded position targets. Native callers must retain
            # the original stopped command; this does not measure grasp force.
            self.thresholds['maximum_joint_error_rad']=.08
        if closure_measurement_profile=='narrow_opening_v1':
            self.thresholds['minimum_tip_closing_m']=.005
        if profile == PRACTICAL_PROFILE:
            # A held target can still produce small contact-loaded convergence.
            # Bound actual original contact surfaces, retaining angular motion
            # as a separate coarse guard and velocity as measured evidence.
            self.thresholds.pop('maximum_joint_speed_rad_s')
            self.thresholds.update(maximum_joint_range_rad=float(np.deg2rad(2.)),
                maximum_commanded_joint_range_rad=0., maximum_contact_surface_hand_span_m=.002)

    def _result(self, reasons, *, ids=(), duration=0., evidence=None):
        result = {'schema': {'strict_v1':'laundry_capture_candidate.v1','local_assist_v2':'laundry_capture_candidate.v2',
                           'adjacent_patch_v3':'laundry_capture_candidate.v3',
                           'angled_adjacent_patch_v4':'laundry_capture_candidate.v4',
                           'loaded_adjacent_patch_v5':'laundry_capture_candidate.v5',
                           PRACTICAL_PROFILE:'laundry_capture_candidate.v6'}[self.profile],
            'capture_profile': self.profile,
            'capture_candidate': not reasons, 'vertex_ids': list(ids),
            'reasons': list(reasons), 'continuous_duration_s': float(duration),
            'continuity_scope': 'Sampled observations only; maximum gap 40ms.',
            'evidence': evidence or {}, 'thresholds': dict(self.thresholds),
            'automatic_attachment_eligible': False, 'friction_grasp_qualified': False,
            'contact_force_measured': False,
            'scope': 'Candidate for explicitly declared local assistance only; no attachment or state mutation.'}
        if self.closure_measurement_profile!='strict_v1':
            result.update(closure_measurement_profile=self.closure_measurement_profile,
                          closure_measurement=dict(self.measurement_evidence or {}))
        if self.profile=='loaded_adjacent_patch_v5':
            result['loaded_servo_error_tolerance']={
                'maximum_joint_target_error_rad':.08,
                'baseline_geometry_and_stability_profile':'adjacent_patch_v3',
                'scope':'Original stopped position targets may differ under contact; no force or friction proof.'}
        if self.profile == PRACTICAL_PROFILE:
            result['practical_surface_stability'] = {
                'cooked_export_sha256': self.contact_motion.export_sha256,
                'motion_reference_sha256': self.contact_motion.reference_sha256,
                'legacy_joint_speed_limit_rad_s': .10, 'legacy_joint_range_limit_rad': .005,
                'scope': 'Original held commands must remain bit-exact for 100 ms; all vertices of '
                         'both witnessed cooked distal surfaces stay within 2 mm in the actual hand frame. '
                         'Actual angular span is bounded by 2 degrees. Joint speed is retained as '
                         'diagnostic evidence of loaded settling, not force or friction retention.'}
        if self.material_region is not None:
            result['material_region_profile']='region_support_v1'
        if self.contact_normal_profile!='facet_v1':result['contact_normal_profile']=self.contact_normal_profile
        if self.contact_witness_profile!='centroid_v1':result['contact_witness_profile']=self.contact_witness_profile
        return result

    def _topology(self, a, b):
        return local_material_patch(self.tri,a,b,face_neighbors=self.face_neighbors)

    def observe(self, time_s, diagnostic, robot_body_q, joint_q, joint_qd,
                commanded_joint_q, initial_open_tip_gap_m, *, cloth_q,
                allowed_material_vertex_ids=None):
        t = float(time_s)
        q = np.asarray(cloth_q, float)
        body = np.asarray(robot_body_q, float)
        actual, speed, target = [np.asarray(a, float) for a in
                                (joint_q, joint_qd, commanded_joint_q)]
        if (q.shape != self.rest.shape or body.shape != (len(self.body_names), 7)
                or any(a.shape != (len(self.joint_names),) for a in (actual, speed, target))
                or not all(np.isfinite(a).all() for a in (q, body, actual, speed, target))
                or not np.isfinite(t) or not np.isfinite(initial_open_tip_gap_m)):
            self.history.clear()
            raise ValueError('Finite actual native states with matching inventories required')
        reasons = []
        identity = {'body_q_float64_sha256': hashlib.sha256(body.astype('<f8').tobytes()).hexdigest(),
                    'cloth_q_float64_sha256': hashlib.sha256(q.astype('<f8').tobytes()).hexdigest()}
        if diagnostic.get('state_identity') != identity:
            reasons.append('diagnostic_does_not_match_actual_native_state')
        if diagnostic.get('contact_normal_profile','facet_v1')!=self.contact_normal_profile:
            reasons.append('contact_normal_profile_mismatch')
        if diagnostic.get('contact_witness_profile','centroid_v1')!=self.contact_witness_profile:
            reasons.append('contact_witness_profile_mismatch')
        if (diagnostic.get('side') != self.side
                or diagnostic.get('distal_group_severe_penetration')
                or diagnostic.get('contact_force_measured') is not False):
            reasons.append('invalid_or_deeply_penetrating_distal_diagnostic')
        expected_profile = self.profile if self.profile=='angled_adjacent_patch_v4' else None
        if diagnostic.get('engineering_geometry_profile') != expected_profile:
            reasons.append('engineering_geometry_profile_mismatch')
        gap = float(np.linalg.norm(body[self.thumb, :3]-body[self.index, :3]))
        closing = float(initial_open_tip_gap_m)-gap
        self.measurement_evidence=measured_closure_requirement(self.closure_measurement_profile,
            initial_open_tip_gap_m,self.planned_open_tip_gap_m)
        if not self.measurement_evidence['open_baseline_valid']:
            reasons.append('narrow_opening_baseline_outside_20_to_30mm')
        if closing < self.measurement_evidence['minimum_required_closure_m']:
            reasons.append('insufficient_measured_tip_closure_from_episode_open_baseline')
        err = float(np.max(np.abs(actual[self.fingers]-target[self.fingers])))
        vmax = float(np.max(np.abs(speed[self.fingers])))
        if err > self.thresholds['maximum_joint_error_rad']: reasons.append('finger_target_error')
        if self.profile != PRACTICAL_PROFILE and vmax > .10: reasons.append('fingers_not_stopped')
        candidates = {}
        for variant in [diagnostic]+diagnostic.get('distal_contact_diagnostics', []):
            pair = variant.get('surface_pair', {})
            if (set(pair) != {'thumb', 'index'}
                    or any(v not in ('elastomer', 'DP') for v in pair.values())
                    or variant.get('severe_penetration_pads')
                    or variant.get('contact_normal_profile','facet_v1')!=self.contact_normal_profile
                    or variant.get('contact_witness_profile','centroid_v1')!=self.contact_witness_profile
                    or variant.get('engineering_geometry_profile') != expected_profile):
                continue
            multi = variant.get('multilayer_diagnostic', {})
            if not multi.get('plausible'): continue
            for witness in multi.get('paired_witnesses', []):
                a, b = int(witness['thumb']['triangle_id']), int(witness['index']['triangle_id'])
                if a == b or min(a, b) < 0 or max(a, b) >= len(self.tri): continue
                topology=self._topology(a,b)
                if topology is None:continue
                region = (None if self.material_region is None else self.material_region.check(
                    witness,topology,q,allowed_material_vertex_ids))
                if region is not None and not region['eligible']:continue
                ids = np.unique(self.tri[topology['support_triangle_ids']])
                if (len(ids) > 32 or _diameter(q[ids]) > .035 or _diameter(self.rest[ids]) > .035
                        or float(witness['opposing_normal_dot']) > self.thresholds['maximum_opposing_normal_dot']): continue
                key = (a, b, pair['thumb'], pair['index'])
                candidates[key] = {'ids': ids, 'witness': witness,'topology':topology,
                    'material_diameter_m': _diameter(self.rest[ids]),
                    'world_diameter_m': _diameter(q[ids]),
                    'local_q': _local_points(q[ids], body[self.wrist])}
                if self.contact_motion is not None:
                    candidates[key]['contact_surface_local'] = self.contact_motion.positions(variant, body)
                if region is not None:candidates[key]['material_region']=region
        if not candidates: reasons.append('no_local_shared_or_single_bridge_two_finger_witness' if self.profile in ('adjacent_patch_v3','angled_adjacent_patch_v4','loaded_adjacent_patch_v5',PRACTICAL_PROFILE)
                                         else 'no_shared_edge_local_two_finger_material_witness')
        if reasons:
            self.history.clear()
            return self._result(reasons, evidence={'tip_gap_m': gap, 'measured_closure_m': closing,
                'maximum_finger_target_error_rad': err, 'maximum_finger_speed_rad_s': vmax})
        if self.history and (t <= self.history[-1]['time'] or t-self.history[-1]['time'] > .040000001):
            self.history.clear()
        # Keep a little more than the required window, allowing a caller's 20ms
        # sampling grid to straddle exactly 100ms without float-rounding loss.
        while len(self.history) > 1 and t-self.history[1]['time'] >= .1-1.e-9:
            self.history.pop(0)
        row = {'time': t, 'candidates': candidates, 'actual': actual[self.fingers].copy(),
            'target': target[self.fingers].copy(), 'error': err, 'speed': vmax, 'gap': gap}
        self.history.append(row)
        shared = set.intersection(*(set(h['candidates']) for h in self.history))
        if not shared:
            self.history = [row]
            return self._result(['material_or_surface_pair_changed'])
        duration = t-self.history[0]['time']
        actual_range = float(np.ptp(np.array([h['actual'] for h in self.history]), axis=0).max())
        target_range = float(np.ptp(np.array([h['target'] for h in self.history]), axis=0).max())
        valid = []
        material_valid = []
        surface_metrics = {}
        for key in sorted(shared):
            initial = self.history[0]['candidates'][key]['local_q']
            drift = max(float(np.linalg.norm(h['candidates'][key]['local_q']-initial, axis=1).max())
                        for h in self.history)
            if drift <= self.thresholds['maximum_material_hand_drift_m']:
                material_valid.append((drift, key))
                if self.contact_motion is not None:
                    metrics = surface_window_metrics(self.history, key)
                    surface_metrics[key] = metrics
                    if metrics['maximum_contact_surface_hand_span_m'] > .002:
                        continue
                valid.append((drift, key))
        if not material_valid: reasons.append('material_sliding_in_actual_hand_frame')
        if self.profile == PRACTICAL_PROFILE:
            if material_valid and not valid: reasons.append('contact_surfaces_moving_in_actual_hand_frame')
            if target_range != 0.: reasons.append('closed_finger_commands_changed')
            if actual_range > self.thresholds['maximum_joint_range_rad']:
                reasons.append('bounded_joint_settling_exceeded')
        elif actual_range > .005 or target_range > .005:
            reasons.append('fingers_or_commands_not_stable')
        if duration < .1-1.e-9: reasons.append('insufficient_sampled_stable_duration')
        key = min(valid)[1] if valid else sorted(shared)[0]
        candidate = candidates[key]
        evidence = {'first_time_s': self.history[0]['time'], 'last_time_s': t,
            'sample_count': len(self.history),
            'maximum_observed_gap_s': max((b['time']-a['time'] for a,b in
                                           zip(self.history,self.history[1:])),default=0.),
            'triangle_ids': list(key[:2]), 'surface_pair': {'thumb': key[2], 'index': key[3]},
            'witnesses': candidate['witness'], 'initial_open_tip_gap_m': float(initial_open_tip_gap_m),
            'actual_tip_gap_m': gap, 'measured_tip_closing_m': closing,
            'maximum_finger_target_error_rad': max(h['error'] for h in self.history),
            'maximum_finger_speed_rad_s': max(h['speed'] for h in self.history),
            'actual_joint_range_rad': actual_range, 'commanded_joint_range_rad': target_range,
            'maximum_fixed_material_hand_drift_m': min(valid)[0] if valid else None,
            'material_diameter_m': candidate['material_diameter_m'],
            'world_diameter_m': candidate['world_diameter_m'],
            'material_topology':candidate['topology'],
            'body_inventory': [self.body_names[self.wrist],self.body_names[self.thumb],self.body_names[self.index]],
            'force_or_load_bearing_proven': False}
        if self.profile == PRACTICAL_PROFILE:
            evidence['original_cooked_surface_motion'] = (surface_metrics[key] if key in surface_metrics
                else surface_window_metrics(self.history, key))
            evidence['closed_finger_commands_held_bitexact'] = target_range == 0.
            evidence['legacy_joint_static_conditions_passed'] = (
                max(h['speed'] for h in self.history) <= .10
                and actual_range <= .005 and target_range <= .005)
        if self.material_region is not None:
            evidence['material_region']=candidate['material_region']
            evidence['material_region_window']=[{'time_s':h['time'],
                'evidence':h['candidates'][key]['material_region']} for h in self.history]
        return self._result(reasons, ids=candidate['ids'].tolist(), duration=duration, evidence=evidence)
