"""Private, read-only towel geometry and release scoring, without a policy.

Every native physics step is required. The caller supplies independently
audited collision/constraint evidence; geometric proximity is not a force
measurement. Synthetic CPU fixtures exercise this contract only.
"""
from copy import deepcopy
from collections import deque
import math
import numpy as np

from .rules import TASK_SCHEMA, public_rules, thresholds, digest, rule_profile, PRACTICAL_PROFILE, LEGACY_PROFILE, control_mode, ENGINEERING_CONTROL, ROBOT_CONTROL, ROBOT_HAND_EXCURSION_M, GARMENT_PROFILE, GARMENT_PROFILES, GARMENT_LEGACY_PROFILE, GARMENT_TASK_IDS, GARMENT_THICKNESS_M, GARMENT_FOLDING_TASK_IDS, GARMENT_FOLDING_AIR_PROFILES, GARMENT_FOLDING_SCALED_PROFILES, PRACTICAL_PROFILES, STACKING_PRACTICAL_PROFILE
from .rules import task_public_rules, FOLDING_PRACTICAL_PROFILE

VERSION = 'laundry_sequence_evaluator.v3'
FRAME_SCHEMA = 'laundry_native_frame.v1'


def configuration(task):
    profile = rule_profile(task)
    if (task.get('schema_version') != TASK_SCHEMA or task.get('id') != task.get('task_id')
            or task.get('public_rules') != task_public_rules(task)
            or task.get('evaluation', {}).get('thresholds') != thresholds(task['id'], profile)):
        raise ValueError('Laundry task identity or frozen rules differ')
    c = deepcopy(task['evaluation'])
    if (c.get('task_id') != task['id'] or c.get('metadata') != task.get('asset_metadata')
            or c.get('public_rules_sha256') != digest(task['public_rules'])
            or c.get('physics_dt_s', 0) <= 0 or c['physics_dt_s'] > .002):
        raise ValueError('Laundry evaluation and physical inputs differ')
    if task.get('public_rules_sha256') != digest(task['public_rules']):
        raise ValueError('Unbound laundry rules')
    return c


def _array(value, shape):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError('Incomplete or nonfinite cloth state')
    return result


def _weighted_rms(delta, weights):
    return float(np.sqrt(np.sum(np.sum(delta*delta, axis=1)*weights)/weights.sum()))


class _ReleasedWindow:
    """Exact rolling drift with cheap chunk bounds for quiet native streams.

    Bounds only prove a pass; an ambiguous bound falls back to the actual
    samples. No downsampling or per-axis approximation changes the decision.
    """
    def __init__(self, duration, dt, drift_limit):
        self.steps = int(math.ceil(duration/dt))
        self.chunk_steps = max(1, int(round(.05/dt)))
        self.limit = drift_limit
        self.samples = deque()
        self.chunks = deque()

    def add(self, step, t, points):
        points = np.concatenate([points[name] for name in sorted(points)]).copy()
        sample = (step, t, points)
        self.samples.append(sample)
        while self.samples[0][0] < step-self.steps:
            self.samples.popleft()
        key = step//self.chunk_steps
        if not self.chunks or self.chunks[-1]['key'] != key:
            self.chunks.append({'key': key, 'lo': points.copy(), 'hi': points.copy(), 'samples': []})
        chunk = self.chunks[-1]
        np.minimum(chunk['lo'], points, out=chunk['lo'])
        np.maximum(chunk['hi'], points, out=chunk['hi'])
        chunk['samples'].append(sample)
        start, start_t, anchor = self.samples[0]
        while self.chunks[0]['samples'][-1][0] < start:
            self.chunks.popleft()
        for chunk in self.chunks:
            bound = np.maximum(np.abs(chunk['lo']-anchor), np.abs(chunk['hi']-anchor))
            suspects = np.flatnonzero(np.sum(bound*bound, axis=1) > self.limit**2)
            if suspects.size:
                for sample_step, _, q in chunk['samples']:
                    if sample_step >= start and np.any(np.linalg.norm(q[suspects]-anchor[suspects], axis=1) > self.limit):
                        return False, 0.
        return True, t-start_t


class SequenceEvaluator:
    def __init__(self, task_spec):
        self.c = configuration(task_spec)
        self.practical = rule_profile(task_spec) in PRACTICAL_PROFILES
        self.single_garment_folding = task_spec['id'] in GARMENT_FOLDING_TASK_IDS
        self.garment = task_spec['id'] in (*GARMENT_TASK_IDS, *GARMENT_FOLDING_TASK_IDS)
        self.profile = rule_profile(task_spec)
        self.practical_stacking = self.profile == STACKING_PRACTICAL_PROFILE
        self.practical_folding = self.profile == FOLDING_PRACTICAL_PROFILE
        if self.practical_stacking:
            ratio = self.c['thresholds']['stability_position_interval_s']/self.c['physics_dt_s']
            if not math.isclose(ratio, round(ratio), abs_tol=1.e-9, rel_tol=0.):
                raise ValueError('Stacking stability requires an integral 20 ms native interval')
            self.stability_lag_steps = int(round(ratio))
        self.scaled_crossings = self.profile in GARMENT_FOLDING_SCALED_PROFILES
        self.control_mode = control_mode(task_spec)
        self.robot_mode = self.control_mode == ROBOT_CONTROL
        self.active_sides = (['right'] if task_spec.get('robot_configuration_id') == 'single_arm'
                             else ['left', 'right'])
        if 'robot_configuration_id' in task_spec and task_spec.get('active_sides') != self.active_sides:
            raise ValueError('Configured robot hand inventory differs from its installation')
        self.task_id = task_spec['id']
        self.meta = self.c['metadata']
        self.limits = self.c['thresholds']
        self.specs = {s['id']: deepcopy(s) for s in self.meta['cloths']}
        expected = {'towel', 'towel_base'} if self.task_id == 'towel_stacking' or self.task_id in GARMENT_TASK_IDS else {'towel'}
        garment_schema = ('laundry_single_garment_folding_asset.v1' if self.single_garment_folding else 'laundry_garment_asset.v1')
        if self.garment and self.meta.get('schema') != garment_schema:
            raise ValueError('Frozen garment asset metadata required')
        if set(self.specs) != expected:
            raise ValueError('Laundry task requires its exact dynamic cloth inventory')
        self.meshes = {}
        self.garment_reference_variants = {}
        for name, spec in self.specs.items():
            rest = _array(spec['rest_vertices_m'], (len(spec['rest_vertices_m']), 3))
            triangles = np.asarray(spec['triangles'], dtype=int)
            thickness = spec['physical_thickness_m']
            expected_thickness = (self.meta.get('physical_thickness_m') if self.single_garment_folding
                                  else GARMENT_THICKNESS_M[self.task_id] if self.garment else .003)
            if not self.garment and self.meta.get('schema') == 'laundry_towel_asset.v3':
                if (self.meta.get('material_profile') != 'towel_4mm_density_v1'
                        or self.meta.get('physical_thickness_m') != .004):
                    raise ValueError('Unknown versioned towel material profile')
                expected_thickness = self.meta['physical_thickness_m']
            if (triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.min() < 0
                    or triangles.max() >= len(rest) or spec['dynamic_model'] != 'native_surface_deformable'
                    or spec['fixed_vertices'] or thickness != expected_thickness
                    or (self.single_garment_folding and (not math.isfinite(thickness) or not .0005 <= thickness <= .002))):
                raise ValueError('Dynamic finite-thickness cloth topology required')
            tri = rest[triangles]
            areas = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)/2
            weights = np.zeros(len(rest))
            np.add.at(weights, triangles.ravel(), np.repeat(areas/3, 3))
            if not math.isclose(float(areas.sum()), spec['rest_area_m2'], abs_tol=1.e-10):
                raise ValueError('Rest-area normalization differs from the frozen mesh')
            edges = np.unique(np.sort(np.concatenate([triangles[:, [0,1]], triangles[:, [1,2]], triangles[:, [2,0]]]), axis=1), axis=0)
            lengths = np.linalg.norm(rest[edges[:, 1]]-rest[edges[:, 0]], axis=1)
            if self.single_garment_folding:
                initial = _array(spec['initial_vertices_m'], rest.shape)
                initial_lengths = np.linalg.norm(initial[edges[:, 1]]-initial[edges[:, 0]], axis=1)
                kind = self.meta.get('initial_state_kind')
                published = kind == 'published_settled_unfolded_shell'
                settled = kind == 'native_settled_unfolded_shell' or published
                if settled:
                    source = self.meta.get('initial_state_input') if published else self.meta.get('native_settled_source')
                    initial_tri = initial[triangles]
                    initial_areas = np.linalg.norm(np.cross(initial_tri[:, 1]-initial_tri[:, 0],
                        initial_tri[:, 2]-initial_tri[:, 0]), axis=1)/2
                    if (self.meta.get('reset_profile') != ('published_settled_unfolded_v1' if published else 'physx_native_settled_unfolded_v1')
                            or not isinstance(source, dict) or not source
                            or (published and source.get('schema') != 'laundry_settled_input.v1')
                            or (not published and (source.get('cloth_position_velocity_writes') != 0
                                or source.get('controlled_cloth_vertices') != 0
                                or source.get('operation_credit') is not False
                                or source.get('rest_geometry_or_topology_modified') is not False))
                            or np.max(initial_lengths/lengths) > self.limits['max_edge_stretch_ratio']
                            or np.min(initial_areas/areas) < self.limits['min_triangle_area_ratio']):
                        raise ValueError('Native settled reset requires frozen provenance and an intact material mesh')
                elif (kind not in ('unfolded_flat', 'unfolded_shell')
                      or not np.allclose(initial_lengths, lengths, atol=1.e-8, rtol=1.e-7)):
                    raise ValueError('Single-garment folding must begin as one unstretched unfolded item')
            if self.garment:
                self.meshes[name] = self._garment_mesh(spec, rest, triangles, weights, areas, edges, lengths)
                if self.single_garment_folding:
                    variants = [('primary', self.meshes[name])]
                    for variant in spec.get('folded_reference_variants', []):
                        variant_id = variant.get('id')
                        if not isinstance(variant_id, str) or not variant_id or variant_id in {key for key, _ in variants}:
                            raise ValueError('Equivalent fold references require unique explicit identifiers')
                        fields = {'folded_reference_vertices_m', 'folded_layer_pairs', 'support_bottom_vertex_indices', 'shell_layer_pairs'}
                        if set(variant) != fields | {'id'}:
                            raise ValueError('Equivalent fold references require complete shape, support and layer evidence')
                        variant_mesh = self._garment_mesh({**spec, **variant}, rest, triangles, weights, areas, edges, lengths)
                        variants.append((variant_id, variant_mesh))
                    self.garment_reference_variants[name] = variants
                continue
            nx, ny = spec['cells_xy']
            if rest.shape != ((nx+1)*(ny+1), 3) or nx % 2:
                raise ValueError('Even rectangular fold correspondence required')
            lower = np.array([j*(nx+1)+i for j in range(ny+1) for i in range(nx//2)])
            upper = np.array([j*(nx+1)+nx-i for j in range(ny+1) for i in range(nx//2)])
            self.meshes[name] = dict(rest=rest, triangles=triangles, weights=weights,
                areas=areas, edges=edges, lengths=lengths, lower=lower, upper=upper)
        if self.single_garment_folding:
            initial = {'towel': np.asarray(self.specs['towel']['initial_vertices_m'], dtype=float)}
            if self._garment_fold(initial)[0]:
                raise ValueError('Single-garment folding reset cannot already satisfy the folded goal')
        self.reset()

    def _garment_mesh(self, spec, rest, triangles, weights, areas, edges, lengths):
        if (spec.get('garment_kind') != self.task_id.removesuffix('_stacking').removesuffix('_folding')
                or np.any(areas <= 1.e-12) or np.any(weights <= 0) or np.any(lengths <= 0)):
            raise ValueError('Finite, nondegenerate garment material topology required')
        reference = _array(spec['folded_reference_vertices_m'], rest.shape)
        bottom = np.asarray(spec['support_bottom_vertex_indices'])
        pairs = np.asarray(spec['folded_layer_pairs'])
        if (bottom.ndim != 1 or not len(bottom) or not np.issubdtype(bottom.dtype, np.integer)
                or len(np.unique(bottom)) != len(bottom) or bottom.min() < 0 or bottom.max() >= len(rest)
                or pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs)
                or not np.issubdtype(pairs.dtype, np.integer) or pairs.min() < 0 or pairs.max() >= len(rest)):
            raise ValueError('Frozen garment support samples and material-layer pairs required')
        pair_delta = reference[pairs[:, 1]]-reference[pairs[:, 0]]
        if np.any(pair_delta[:, 2] < spec['physical_thickness_m']*.5):
            raise ValueError('Garment layer pairs must be ordered bottom to top in the reference')
        shell_pairs = np.empty((0, 2), dtype=int)
        if self.single_garment_folding:
            declared = spec.get('shell_layer_pairs', [])
            if declared:
                shell_pairs = np.asarray(declared)
                if (shell_pairs.ndim != 2 or shell_pairs.shape[1] != 2
                        or not np.issubdtype(shell_pairs.dtype, np.integer)
                        or shell_pairs.min() < 0 or shell_pairs.max() >= len(rest)
                        or np.any(shell_pairs[:, 0] == shell_pairs[:, 1])):
                    raise ValueError('Frozen natural shell layer pairs must name distinct material vertices')
                if np.any(reference[shell_pairs[:, 1], 2]-reference[shell_pairs[:, 0], 2] <= 0):
                    raise ValueError('Natural shell pairs must be ordered bottom to top in each reference')
            elif (self.meta.get('initial_state_kind') in ('unfolded_shell', 'native_settled_unfolded_shell', 'published_settled_unfolded_shell')
                  and self.meta.get('goal_reference_status') in ('authored_fold_reference', 'native_unfolded_reference_fold_target')):
                raise ValueError('Qualified sewn-garment targets require independent natural shell layer pairs')
        ref_tri = reference[triangles]
        projected = np.abs(np.cross(ref_tri[:, 1]-ref_tri[:, 0], ref_tri[:, 2]-ref_tri[:, 0])[:, 2]).sum()/2
        if projected <= 0:
            raise ValueError('Folded garment reference must have finite projected area')
        center = np.average(reference, axis=0, weights=weights)
        return dict(rest=rest, triangles=triangles, weights=weights, areas=areas, edges=edges,
            lengths=lengths, reference=reference-center, reference_center=center,
            bottom=bottom, layer_pairs=pairs, pair_delta=pair_delta, shell_layer_pairs=shell_pairs,
            reference_projected_area=projected)

    def reset(self):
        self.previous_time = self.previous_step = None
        self.initial = None
        self.previous = None
        self.stable_anchor = None
        self.stable_start_time = None
        self.released_window = None
        self.position_history = deque()
        self.stable_s = 0.
        self.near_hand_anchors = {}
        self.near_cloth_anchors = {}
        self.hand_excursion_m = 0.
        self.operation_seen = False
        self.displacement_seen = False
        self.invalid = self.safety = None
        self.success = False
        self.exhausted = False
        self.mask_cache = set()
        self.detail = {}
        self.strict_diagnostics = {'violating_frames': 0, 'first_violation_step': None,
            'max_node_speed_m_s': 0., 'max_pose_speed_m_s': 0.}
        self.last_result = self._result('task_in_progress')

    def _result(self, reason):
        state = ('evaluator_invalid' if self.invalid else 'safety_stop' if self.safety else
                 'success' if self.success and not self.exhausted else 'ongoing')
        result = {'evaluator': VERSION, 'task_id': self.task_id,
            'status': 'unknown' if self.invalid else 'incomplete' if self.exhausted else state,
            'success': state == 'success', 'reason': 'budget_exhausted' if self.exhausted else reason,
            'terminal': self.exhausted or state != 'ongoing', 'physics_accepted': False,
            'robot_manipulation_qualified': False, 'engineering_control_mode': 'engineering_local_grasp',
            'budget_exhausted': self.exhausted, 'physical_failure': False,
            'released_stable_s': self.stable_s, 'detail': deepcopy(self.detail),
            'evidence_status': 'unknown' if self.invalid else 'pending' if self.previous_time is None else 'complete',
            'irrecoverable_reason': self.safety,
            'referee': {'schema': 'physical_task_referee_state.v1', 'state': state,
                'phase': 'settling' if self.stable_s else 'working', 'reason_codes': [reason],
                'terminal': state != 'ongoing', 'recoverable': state == 'ongoing'},
            'scope': 'Released engineering cloth operation only; neither robot grasp success nor calibrated cloth force.'}
        if self.robot_mode:
            result.pop('engineering_control_mode')
            result.update(control_mode=self.control_mode,
                scope='Released robot joint-action task outcome using native geometric interaction evidence; no grasp-force or material calibration claim.')
        if self.practical:
            result.update(rule_profile=self.profile, strict_diagnostics=deepcopy(self.strict_diagnostics))
        if self.invalid:
            result['private_evidence_error'] = self.invalid
        self.last_result = deepcopy(result)
        return result

    def _mesh_metrics(self, name, v):
        m = self.meshes[name]
        triangles = v[m['triangles']]
        cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        stretch = np.linalg.norm(v[m['edges'][:, 1]]-v[m['edges'][:, 0]], axis=1)/m['lengths']
        ratio = np.linalg.norm(cross, axis=1)/(2*m['areas'])
        return {'max_edge_stretch_ratio': float(stretch.max()),
            'min_triangle_area_ratio': float(ratio.min()),
            'projected_area_ratio': float(np.abs(cross[:, 2]).sum()/2/self.specs[name]['rest_area_m2'])}

    def _crossing_gate(self, physics):
        count = physics['self_intersections']
        measure = physics.get('self_crossing_measure')
        if count == 0:
            if measure is not None and (not isinstance(measure, dict)
                    or measure.get('schema') != 'laundry_garment_crossing_measure.v1'
                    or measure.get('complete') is not True or measure.get('pair_count') != 0):
                raise ValueError('Crossing count and size evidence disagree')
            return True, False, None
        if (not isinstance(measure, dict) or measure.get('schema') != 'laundry_garment_crossing_measure.v1'
                or measure.get('complete') is not True or measure.get('pair_count') != count
                or type(measure.get('affected_triangle_count')) is not int
                or not 1 <= measure['affected_triangle_count'] <= len(self.meshes['towel']['triangles'])):
            raise ValueError('Complete crossing size evidence required for single garment folding')
        fields = (('maximum_pair_length_m', 'crossing_pair_length_max_m', 'hard_crossing_pair_length_m'),
                  ('total_intersection_length_m', 'crossing_total_length_max_m', 'hard_crossing_total_length_m'),
                  ('maximum_component_diameter_m', 'crossing_component_diameter_max_m', 'hard_crossing_component_diameter_m'),
                  ('affected_rest_area_fraction', 'crossing_rest_area_fraction_max', 'hard_crossing_rest_area_fraction'),
                  ('coplanar_overlap_area_sum_m2', 'crossing_coplanar_area_max_m2', 'hard_crossing_coplanar_area_m2'))
        for key, _, _ in fields:
            if type(measure.get(key)) not in (int, float) or not math.isfinite(measure[key]) or measure[key] < 0:
                raise ValueError('Finite nonnegative crossing size evidence required')
        if measure['affected_rest_area_fraction'] > 1 or measure['total_intersection_length_m'] < measure['maximum_pair_length_m']:
            raise ValueError('Inconsistent crossing size evidence')
        return (all(measure[key] <= self.limits[soft] for key, soft, _ in fields),
                any(measure[key] > self.limits[hard] for key, _, hard in fields), deepcopy(measure))

    def _flat(self, name, v, metric):
        m = self.meshes[name]
        center = np.asarray(self.meta['target_center_xy_m'])
        errors = [v[:, :2]-(sign*m['rest'][:, :2]+center) for sign in (1, -1)]
        error = min(errors, key=lambda e: _weighted_rms(e, m['weights']))
        rms = _weighted_rms(error, m['weights'])
        maximum = float(np.linalg.norm(error, axis=1).max())
        heights = v[:, 2]-.9
        detail = {**metric, 'target_rms_m': rms, 'target_max_m': maximum,
            'height_range_m': float(np.ptp(heights)), 'min_height_m': float(heights.min()),
            'mean_height_m': float(np.average(heights, weights=m['weights']))}
        good = (rms <= self.limits['flat_target_rms_m'] and maximum <= self.limits['flat_target_max_m']
            and metric['projected_area_ratio'] >= self.limits['flat_projected_area_ratio_min']
            and np.ptp(heights) <= self.limits['flat_height_range_m']
            and heights.min() >= .0015-self.limits['max_penetration_m']
            and detail['mean_height_m'] <= .0015+self.limits['support_gap_max_m'])
        return bool(good), detail

    def _fold(self, name, v, target, *, table_supported):
        m = self.meshes[name]
        lower, upper = v[m['lower']], v[m['upper']]
        delta = upper-lower
        gap = delta[:, 2]
        # Either material half may be above; a single global ordering is needed.
        sign = 1 if np.median(gap) >= 0 else -1
        ordered_gap = gap*sign
        xy = np.linalg.norm(delta[:, :2], axis=1)
        pair_weights = m['weights'][m['lower']]+m['weights'][m['upper']]
        pair_rms = _weighted_rms(delta[:, :2], pair_weights)
        layer_good = (ordered_gap >= self.limits['layer_gap_min_m']) & (ordered_gap <= self.limits['layer_gap_max_m'])
        fraction = float(np.sum(pair_weights*layer_good)/pair_weights.sum())
        lo, hi = v.min(axis=0), v.max(axis=0)
        center = (lo[:2]+hi[:2])/2
        extent = hi[:2]-lo[:2]
        position = float(np.linalg.norm(center-target))
        # An overlapped but crumpled blob cannot pass correspondence alone.
        extent_tolerance = self.limits.get('folded_extent_tolerance_m', .035)
        height = float(hi[2]-lo[2])
        height_ok = height <= self.limits.get('folded_height_range_m', .025)
        height_detail = {}
        if self.practical_stacking:
            order = np.argsort(v[:, 2], kind='stable')
            weights = m['weights'][order]
            rank = min(len(order)-1, int(np.searchsorted(np.cumsum(weights),
                weights.sum()*self.limits['folded_height_area_percentile']/100., side='left')))
            area_height = float(v[order[rank], 2]-lo[2])
            height_ok = (area_height <= self.limits['folded_height_range_m']
                and height <= self.limits['folded_height_absolute_max_m'])
            height_detail = {'rest_area_height_percentile': self.limits['folded_height_area_percentile'],
                'rest_area_height_range_m': area_height, 'absolute_height_range_m': height}
        shape = bool(abs(extent[0]-.3) <= extent_tolerance and abs(extent[1]-.4) <= extent_tolerance
            and height_ok)
        support = bool(lo[2] >= .9015-self.limits['max_penetration_m']
            and lo[2] <= .9015+self.limits['support_gap_max_m']) if table_supported else True
        pair_percentile_ok = True
        pair_detail = {}
        if self.practical_folding:
            order = np.argsort(xy, kind='stable')
            weights = pair_weights[order]
            rank = min(len(order)-1, int(np.searchsorted(np.cumsum(weights),
                weights.sum()*self.limits['folded_pair_area_percentile']/100., side='left')))
            percentile = float(xy[order[rank]])
            pair_percentile_ok = percentile <= self.limits['folded_pair_percentile_m']
            pair_detail = {'pair_rest_area_percentile': self.limits['folded_pair_area_percentile'],
                'pair_percentile_m': percentile}
        good = (pair_rms <= self.limits['folded_pair_rms_m'] and xy.max() <= self.limits['folded_pair_max_m']
            and pair_percentile_ok and fraction >= self.limits['layer_order_fraction_min'] and shape and support
            and position <= self.limits['placement_error_m'])
        return bool(good), {'pair_rms_m': pair_rms, 'pair_max_m': float(xy.max()),
            'ordered_layer_fraction': fraction, 'layer_gap_median_m': float(np.median(ordered_gap)),
            'center_error_m': position, 'extent_xy_m': extent.tolist(),
            'min_z_m': float(lo[2]), 'max_z_m': float(hi[2]), 'table_supported_geometry': support, **height_detail, **pair_detail}

    def _stack(self, cloths):
        target = np.asarray(self.meta['target_center_xy_m'])
        upper, base = cloths['towel'], cloths['towel_base']
        top_good, top_detail = self._fold('towel', upper, target, table_supported=False)
        base_good, base_detail = self._fold('towel_base', base, target, table_supported=True)
        # Match the lower material layer of the moved towel to actual top-layer
        # samples of the base towel. This is geometric support, never a force.
        def layers(name, v):
            m = self.meshes[name]
            a, b = v[m['lower']], v[m['upper']]
            return (a, b) if np.median(a[:, 2]) < np.median(b[:, 2]) else (b, a)
        bottom, _ = layers('towel', upper)
        _, base_top = layers('towel_base', base)
        # Whole-towel 180 degree equivalence is allowed; material identity stays
        # fixed within each cloth so swapping cloth IDs cannot manufacture a pass.
        candidates = (base_top, base_top[::-1])
        counterpart = min(candidates, key=lambda b: float(np.mean(np.linalg.norm(bottom[:, :2]-b[:, :2], axis=1))))
        xy = np.linalg.norm(bottom[:, :2]-counterpart[:, :2], axis=1)
        gap = bottom[:, 2]-counterpart[:, 2]
        supported = (xy <= self.limits['stacking_alignment_m']) & (gap >= self.limits['stacking_gap_min_m']) & (gap <= self.limits['stacking_gap_max_m'])
        weights = self.meshes['towel']['weights'][self.meshes['towel']['lower']]
        fraction = float(np.sum(weights*supported)/weights.sum())
        return bool(top_good and base_good and fraction >= self.limits.get('stacking_support_fraction_min', .95)), {'towel': top_detail,
            'towel_base': base_detail, 'cloth_support_fraction': fraction,
            'intercloth_gap_median_m': float(np.median(gap)), 'support_evidence': 'gravity_release_and_actual_surface_geometry_no_force_measurement'}

    def _garment_shape(self, name, v, target, *, table_supported, reference_mesh=None):
        """Material-correspondence fit permits whole-garment yaw, not unfolding."""
        m = self.meshes[name] if reference_mesh is None else reference_mesh
        weights, reference = m['weights'], m['reference']
        center = np.average(v, axis=0, weights=weights)
        centered = v-center
        covariance = centered[:, :2].T@(reference[:, :2]*weights[:, None])
        u, _, vt = np.linalg.svd(covariance)
        rotation = u@np.diag([1., np.linalg.det(u@vt)])@vt
        aligned = centered.copy()
        aligned[:, :2] = centered[:, :2]@rotation
        error = aligned-reference
        rms = _weighted_rms(error, weights)
        p95 = float(np.percentile(np.linalg.norm(error, axis=1), 95))
        extent_error = float(np.abs(np.ptp(aligned[:, :2], axis=0)-np.ptp(reference[:, :2], axis=0)).max())
        height_excess = float(np.ptp(v[:, 2])-np.ptp(reference[:, 2]))
        # The reference's bounding-box center is the declared placement center.
        reference_box = (reference[:, :2].min(axis=0)+reference[:, :2].max(axis=0))/2
        placement_center = center[:2]+reference_box@rotation.T
        position = float(np.linalg.norm(placement_center-target))
        pairs = m['layer_pairs']
        delta = aligned[pairs[:, 1]]-aligned[pairs[:, 0]]
        pair_xy = np.linalg.norm(delta[:, :2]-m['pair_delta'][:, :2], axis=1)
        gaps = delta[:, 2]
        if self.profile == GARMENT_LEGACY_PROFILE:
            minimum_gap = np.maximum(self.limits['garment_layer_gap_min_m'],
                self.limits['garment_layer_gap_ratio_min']*m['pair_delta'][:, 2])
        else:
            minimum_gap = max(self.limits['garment_layer_gap_min_m'],
                self.limits['garment_layer_thickness_ratio_min']*self.specs[name]['physical_thickness_m'])
        # Reference bend radii contain air. Profiles v3+ permit its natural
        # compression down to the unchanged physical material-thickness floor.
        gap_matches = ((gaps <= m['pair_delta'][:, 2]+self.limits['garment_layer_gap_error_m'])
            if self.profile in GARMENT_FOLDING_AIR_PROFILES else
            (np.abs(gaps-m['pair_delta'][:, 2]) <= self.limits['garment_layer_gap_error_m']))
        layered = ((pair_xy <= self.limits['garment_layer_xy_error_m'])
            & (gaps >= minimum_gap) & gap_matches)
        pair_weights = weights[pairs[:, 0]]+weights[pairs[:, 1]]
        fraction = float(np.average(layered, weights=pair_weights))
        shell_fraction, shell_ok = None, True
        if self.single_garment_folding and len(m['shell_layer_pairs']):
            shell_pairs = m['shell_layer_pairs']
            shell_gap = aligned[shell_pairs[:, 1], 2]-aligned[shell_pairs[:, 0], 2]
            shell_weights = weights[shell_pairs[:, 0]]+weights[shell_pairs[:, 1]]
            shell_fraction = float(np.average(shell_gap >= self.limits['garment_shell_layer_thickness_ratio_min']
                * self.specs[name]['physical_thickness_m'], weights=shell_weights))
            shell_ok = shell_fraction >= self.limits['garment_shell_layer_fraction_min']
        tri = v[m['triangles']]
        projected = float(np.abs(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])[:, 2]).sum()/2)
        projected_ratio = projected/m['reference_projected_area']
        bottom = m['bottom']
        thickness = self.specs[name]['physical_thickness_m']
        clearance = v[bottom, 2]-self.meta['table_top_z_m']-thickness/2
        table_fraction = float(np.average((clearance >= -self.limits['max_penetration_m'])
            & (clearance <= self.limits['support_gap_max_m']), weights=weights[bottom]))
        shape = (rms <= self.limits['garment_shape_rms_m']
            and p95 <= self.limits['garment_shape_p95_m']
            and extent_error <= self.limits['garment_extent_tolerance_m']
            and height_excess <= self.limits['garment_height_excess_m']
            and self.limits['garment_projected_area_ratio_min'] <= projected_ratio <= self.limits['garment_projected_area_ratio_max']
            and fraction >= self.limits['garment_layer_fraction_min'])
        support = not table_supported or table_fraction >= self.limits['garment_table_support_fraction_min']
        detail = {
            'shape_rms_m': rms, 'shape_p95_m': p95, 'extent_error_m': extent_error,
            'height_excess_m': height_excess, 'center_error_m': position,
            'ordered_layer_fraction': fraction, 'projected_area_ratio_to_reference': projected_ratio,
            'table_support_fraction': table_fraction, 'table_supported_geometry': bool(support)}
        if self.single_garment_folding:
            detail.update(natural_shell_layer_fraction=shell_fraction,
                          natural_shell_layer_geometry=bool(shell_ok))
        return bool(shape and shell_ok and support and position <= self.limits['placement_error_m']), detail

    @staticmethod
    def _surface_top_at(points_xy, vertices, triangles):
        """Exact XY triangle containment and interpolated highest surface Z.

        A conservative centroid-radius query only reduces candidate count;
        every retained support sample is checked against the actual triangle.
        """
        from scipy.spatial import cKDTree
        tri = vertices[triangles]
        centers = tri[:, :, :2].mean(axis=1)
        radius = float(np.linalg.norm(tri[:, :, :2]-centers[:, None, :], axis=2).max())
        candidates = cKDTree(centers).query_ball_point(points_xy, radius+1.e-10)
        lengths = np.fromiter((len(row) for row in candidates), dtype=int, count=len(points_xy))
        result = np.full(len(points_xy), -np.inf)
        if not lengths.sum():
            return result
        sample_ids = np.repeat(np.arange(len(points_xy)), lengths)
        triangle_ids = np.concatenate([np.asarray(row, dtype=int) for row in candidates if len(row)])
        selected = tri[triangle_ids]
        a = selected[:, 1, :2]-selected[:, 0, :2]
        b = selected[:, 2, :2]-selected[:, 0, :2]
        p = points_xy[sample_ids]-selected[:, 0, :2]
        denominator = a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]
        valid = np.abs(denominator) > 1.e-12
        denominator = np.where(valid, denominator, 1.)
        u = (p[:, 0]*b[:, 1]-p[:, 1]*b[:, 0])/denominator
        w = (a[:, 0]*p[:, 1]-a[:, 1]*p[:, 0])/denominator
        valid &= (u >= -1.e-8) & (w >= -1.e-8) & (u+w <= 1.+1.e-8)
        z = selected[:, 0, 2]+u*(selected[:, 1, 2]-selected[:, 0, 2])+w*(selected[:, 2, 2]-selected[:, 0, 2])
        np.maximum.at(result, sample_ids[valid], z[valid])
        return result

    def _garment_fold(self, cloths):
        if set(cloths) != {'towel'}:
            raise ValueError('Single-garment fold requires exactly one dynamic item')
        candidates = []
        for variant_id, mesh in self.garment_reference_variants['towel']:
            good, detail = self._garment_shape('towel', cloths['towel'],
                np.asarray(self.meta['target_center_xy_m']), table_supported=True, reference_mesh=mesh)
            candidates.append((good, detail, variant_id))
        good, detail, variant_id = min(candidates, key=lambda result: (not result[0], result[1]['shape_rms_m']))
        status = self.meta.get('goal_reference_status')
        qualified = status in ('authored_fold_reference', 'native_unfolded_reference_fold_target')
        return bool(good and qualified), {'towel': detail, 'operation': 'single_garment_folding',
            'fold_reference_variant': variant_id,
            'equivalent_reference_count': len(candidates),
            'goal_reference_status': status, 'goal_reference_qualified': qualified,
            'support_evidence': 'released_single_garment_on_original_table_geometry_no_force_measurement'}

    def _garment_stack(self, cloths):
        target = np.asarray(self.meta['target_center_xy_m'])
        top_good, top_detail = self._garment_shape('towel', cloths['towel'], target, table_supported=False)
        base_good, base_detail = self._garment_shape('towel_base', cloths['towel_base'], target, table_supported=True)
        m = self.meshes['towel']
        bottom = cloths['towel'][m['bottom']]
        surface = self._surface_top_at(bottom[:, :2], cloths['towel_base'], self.meshes['towel_base']['triangles'])
        gap = bottom[:, 2]-surface
        supported = np.isfinite(surface) & (gap >= self.limits['stacking_gap_min_m']) & (gap <= self.limits['stacking_gap_max_m'])
        fraction = float(np.average(supported, weights=m['weights'][m['bottom']]))
        finite_gap = gap[np.isfinite(gap)]
        return bool(top_good and base_good and fraction >= self.limits['stacking_support_fraction_min']), {
            'towel': top_detail, 'towel_base': base_detail, 'cloth_support_fraction': fraction,
            'intercloth_gap_median_m': float(np.median(finite_gap)) if len(finite_gap) else None,
            'support_evidence': 'gravity_release_and_actual_triangle_surface_geometry_no_force_measurement'}

    def observe(self, frame):
        if self.invalid or self.safety or (self.success and not self.practical) or self.exhausted:
            return deepcopy(self.last_result)
        try:
            if frame.get('schema') != FRAME_SCHEMA or frame.get('complete') is not True:
                raise ValueError('Complete native laundry frame required')
            t, step = frame['time_s'], frame['native_step']
            if type(t) not in (int, float) or not math.isfinite(t) or type(step) is not int or step < 0:
                raise ValueError('Finite native time and integer step required')
            first = self.previous_time is None
            if first:
                if abs(t) > 1.e-9 or step != 0:
                    raise ValueError('Unscored reset frame must start at time/step zero')
            elif step != self.previous_step+1 or abs(t-self.previous_time-self.c['physics_dt_s']) > 1.e-8:
                raise ValueError('Every consecutive native physics step is required')
            if frame.get('control_mode') != self.control_mode:
                raise ValueError('Explicit frozen control provenance required')
            robot_active, robot_released = (self._robot_interaction(frame) if self.robot_mode else (False, True))
            if set(frame['cloths']) != set(self.specs) or set(frame['controlled_vertices']) != set(self.specs):
                raise ValueError('Cloth identity or control-mask inventory differs')
            p = frame['physics']
            if (p.get('complete') is not True or type(p.get('valid')) is not bool
                    or p.get('gravity_enabled') is not True or p.get('hidden_constraints') != 0):
                raise ValueError('Complete gravity/constraint/collision audit required')
            for field in ('self_intersections', 'intercloth_intersections'):
                if type(p.get(field)) is not int or p[field] < 0:
                    raise ValueError('Explicit native collision audit required')
            penetration = p.get('max_penetration_m')
            if type(penetration) not in (int, float) or not math.isfinite(penetration) or penetration < 0:
                raise ValueError('Finite penetration audit required')
            cloths, velocities, metrics = {}, {}, {}
            controlled = False
            for name, spec in self.specs.items():
                row, mesh = frame['cloths'][name], self.meshes[name]
                v = _array(row['vertices_m'], mesh['rest'].shape)
                vel = _array(row['velocities_m_s'], mesh['rest'].shape)
                if row.get('enabled') is not True or not np.array_equal(np.asarray(row['triangles']), mesh['triangles']):
                    raise ValueError('Dynamic mesh topology or enabled flag changed')
                mask = frame['controlled_vertices'][name]
                if (not isinstance(mask, list) or len(mask) != len(set(mask))
                        or any(type(i) is not int or i < 0 or i >= len(v) for i in mask)):
                    raise ValueError('Invalid engineering grasp mask')
                if self.robot_mode and mask:
                    raise ValueError('Robot joint episodes cannot drive cloth vertices')
                if len(mask) > .10*len(v):
                    raise ValueError('Engineering driving must remain local, not whole-cloth state control')
                self._check_local_mask(name, mask)
                controlled |= bool(mask)
                cloths[name], velocities[name] = v, vel
                metrics[name] = self._mesh_metrics(name, v)
            self._check_world_grasp_regions(cloths, frame['controlled_vertices'])
            robot_displaced = self._robot_operation(frame, cloths) if self.robot_mode else False
            self.previous_time, self.previous_step = t, step
            crossing_ok, crossing_hard, crossing_measure = ((not p['self_intersections'], False, None)
                if not self.scaled_crossings else self._crossing_gate(p))
            contact_ok = not (not crossing_ok or p['intercloth_intersections']
                    or penetration > self.limits['max_penetration_m']
                    or any(m['max_edge_stretch_ratio'] > self.limits['max_edge_stretch_ratio']
                           or m['min_triangle_area_ratio'] < self.limits['min_triangle_area_ratio'] for m in metrics.values()))
            invalid_physics = not p['valid']
            if self.practical:
                # The strict geometry aggregate is a diagnostic, not a native
                # engine failure. Only audited callers may declare this scope.
                invalid_physics &= p.get('validity_scope') != 'strict_geometry'
                bad_crossing = ((crossing_hard or p['intercloth_intersections'] > 0) if self.scaled_crossings else
                    p['self_intersections']+p['intercloth_intersections'] > self.limits['hard_intersection_pairs'])
                catastrophic = (penetration > self.limits['hard_penetration_m']
                    or bad_crossing
                    or any(m['max_edge_stretch_ratio'] > self.limits['hard_edge_stretch_ratio'] for m in metrics.values()))
                strict = thresholds('towel_stacking' if self.garment else self.task_id, LEGACY_PROFILE)
                strict_bad = (not p['valid'] or p['self_intersections'] or p['intercloth_intersections']
                    or penetration > strict['max_penetration_m']
                    or any(m['max_edge_stretch_ratio'] > strict['max_edge_stretch_ratio']
                        or m['min_triangle_area_ratio'] < strict['min_triangle_area_ratio'] for m in metrics.values()))
                if strict_bad:
                    self.strict_diagnostics['violating_frames'] += 1
                    if self.strict_diagnostics['first_violation_step'] is None:
                        self.strict_diagnostics['first_violation_step'] = step
            else:
                catastrophic = not contact_ok
            if invalid_physics or catastrophic:
                self.success = False
                self.stable_s = 0.
                self.safety = 'physics_safety_stop'
                return self._result('physics_safety_stop')
            if t >= self.limits['max_time_s']:
                self.exhausted = True
                self.success = False
                self.stable_s = 0.
                return self._result('incomplete_budget')
            interval_speed = None
            if self.practical_stacking:
                self.position_history.append((step, {k: v.copy() for k, v in cloths.items()}))
                while self.position_history[0][0] < step-self.stability_lag_steps:
                    self.position_history.popleft()
                if self.position_history[0][0] == step-self.stability_lag_steps:
                    previous = self.position_history[0][1]
                    interval_speed = max(float(np.percentile(np.linalg.norm(v-previous[k], axis=1), 95)
                        /self.limits['stability_position_interval_s']) for k, v in cloths.items())
            if first:
                self.initial = {k: v.copy() for k, v in cloths.items()}
                self.previous = {k: v.copy() for k, v in cloths.items()}
                return self._result('reset_baseline_not_scored')
            towel_controlled = robot_active if self.robot_mode else bool(frame['controlled_vertices']['towel'])
            self.operation_seen |= towel_controlled
            movement = np.linalg.norm(cloths['towel']-self.initial['towel'], axis=1)
            self.displacement_seen |= (robot_displaced if self.robot_mode else
                bool(towel_controlled and np.max(movement) >= self.limits['operation_displacement_m']))
            if self.task_id == 'towel_flattening':
                good, detail = self._flat('towel', cloths['towel'], metrics['towel'])
            elif self.task_id == 'towel_folding':
                good, detail = self._fold('towel', cloths['towel'], np.asarray(self.meta['target_center_xy_m']), table_supported=True)
            elif self.single_garment_folding:
                good, detail = self._garment_fold(cloths)
            elif self.garment:
                good, detail = self._garment_stack(cloths)
            else:
                good, detail = self._stack(cloths)
            speed = max(float(np.linalg.norm(v, axis=1).max()) for v in velocities.values())
            # Finite-difference motion also prevents a zeroed velocity field from
            # making a moving released cloth look settled.
            pose_speed = max(float(np.linalg.norm(v-self.previous[k], axis=1).max()/self.c['physics_dt_s']) for k, v in cloths.items())
            speed_gate, pose_gate = speed, pose_speed
            if self.practical:
                speed_gate = max(float(np.percentile(np.linalg.norm(v, axis=1), 95)) for v in velocities.values())
                pose_gate = max(float(np.percentile(np.linalg.norm(v-self.previous[k], axis=1), 95)/self.c['physics_dt_s']) for k, v in cloths.items())
                self.strict_diagnostics['max_node_speed_m_s'] = max(speed, self.strict_diagnostics['max_node_speed_m_s'])
                self.strict_diagnostics['max_pose_speed_m_s'] = max(pose_speed, self.strict_diagnostics['max_pose_speed_m_s'])
            quiet = ((interval_speed is not None and interval_speed <= self.limits['max_node_speed_m_s'])
                if self.practical_stacking else max(speed_gate, pose_gate) <= self.limits['max_node_speed_m_s'])
            eligible = bool(good and contact_ok and not controlled and robot_released and self.operation_seen and self.displacement_seen and quiet)
            if self.practical:
                if eligible:
                    if self.released_window is None:
                        self.released_window = _ReleasedWindow(self.limits['stable_s'], self.c['physics_dt_s'],
                                                               self.limits['stable_node_drift_m'])
                    eligible, self.stable_s = self.released_window.add(step, t, cloths)
                if not eligible:
                    self.released_window = None
                    self.stable_s = 0.
            elif eligible:
                if self.stable_anchor is None:
                    self.stable_anchor = {k: v.copy() for k, v in cloths.items()}
                    self.stable_start_time = t
                eligible &= all(np.linalg.norm(v-self.stable_anchor[k], axis=1).max() <= self.limits['stable_node_drift_m'] for k, v in cloths.items())
            if not self.practical:
                self.stable_s = self.stable_s+self.c['physics_dt_s'] if eligible else 0.
            if not eligible:
                self.stable_anchor = None
                self.stable_start_time = None
            self.previous = {k: v.copy() for k, v in cloths.items()}
            self.detail = {'geometry': detail, 'mesh': metrics, 'operation_seen': self.operation_seen,
                'operation_displacement_seen': self.displacement_seen, 'all_control_released': not controlled and robot_released,
                'max_node_speed_m_s': speed, 'pose_speed_m_s': pose_speed}
            if self.scaled_crossings:
                self.detail.update(self_intersection_pairs=p['self_intersections'],
                    self_crossing_measure=crossing_measure, crossing_size_acceptable=bool(crossing_ok))
            if self.robot_mode:
                self.detail.update(hand_near_towel_excursion_m=self.hand_excursion_m,
                    robot_fully_separated=robot_released)
            if self.practical:
                self.detail.update(raw_speed_p95_m_s=speed_gate, pose_speed_p95_m_s=pose_gate,
                    contact_geometry_acceptable=bool(contact_ok))
            if self.practical_stacking:
                self.detail.update(stability_position_interval_s=self.limits['stability_position_interval_s'],
                    control_period_pose_speed_p95_m_s=interval_speed,
                    stability_history_complete=interval_speed is not None)
            self.success = bool(self.stable_s+1.e-9 >= self.limits['stable_s'])
            reason = ('success' if self.success else 'contact_recovery_required' if self.practical and not contact_ok
                else 'waiting_for_release_and_stability' if good else 'task_in_progress')
            return self._result(reason)
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
            self.invalid = str(exc)
            self.success = False
            self.stable_s = 0.
            return self._result('native_evidence_unknown')

    def _robot_operation(self, frame, cloths):
        displaced = False
        for side, hand in frame['robot_interaction']['hands'].items():
            if not hand['near_towel']:
                self.near_cloth_anchors.pop(side, None)
                continue
            anchor = self.near_cloth_anchors.setdefault(side, cloths['towel'].copy())
            hand_excursion = np.linalg.norm(np.asarray(hand['wrist_position_m'])-self.near_hand_anchors[side])
            cloth_excursion = np.linalg.norm(cloths['towel']-anchor, axis=1).max()
            displaced |= bool(hand_excursion >= ROBOT_HAND_EXCURSION_M and
                              cloth_excursion >= self.limits['operation_displacement_m'])
        return displaced

    def _robot_interaction(self, frame):
        evidence = frame['robot_interaction']
        if (not isinstance(evidence, dict)
                or evidence.get('schema') != 'laundry_robot_interaction.v1'
                or evidence.get('complete') is not True
                or evidence.get('method') != 'native_cooked_convex_geometry'
                or type(evidence.get('hand_near_cloth')) is not bool
                or type(evidence.get('fully_separated')) is not bool
                or not isinstance(evidence.get('hands'), dict)
                or set(evidence['hands']) != set(self.active_sides)):
            raise ValueError('Complete native active-hand geometric interaction evidence required')
        near_towel = near_any = active = False
        for side, row in evidence['hands'].items():
            if (not isinstance(row, dict) or type(row.get('near_towel')) is not bool
                    or type(row.get('near_any_cloth')) is not bool
                    or row['near_towel'] and not row['near_any_cloth']):
                raise ValueError('Inconsistent hand/cloth proximity evidence')
            wrist = _array(row['wrist_position_m'], (3,))
            near_towel |= row['near_towel']
            near_any |= row['near_any_cloth']
            if row['near_towel']:
                anchor = self.near_hand_anchors.setdefault(side, wrist.copy())
                excursion = float(np.linalg.norm(wrist-anchor))
                self.hand_excursion_m = max(self.hand_excursion_m, excursion)
                active |= excursion >= ROBOT_HAND_EXCURSION_M
            else:
                self.near_hand_anchors.pop(side, None)
        if (evidence['hand_near_cloth'] != near_towel
                or evidence['fully_separated'] and near_any):
            raise ValueError('Inconsistent robot release evidence')
        return active, evidence['fully_separated']

    def _check_local_mask(self, name, mask):
        key = (name, tuple(sorted(mask)))
        if key in self.mask_cache:
            return
        pending = set(mask)
        neighbors = {i: set() for i in mask}
        for a, b in self.meshes[name]['edges']:
            if a in pending and b in pending:
                neighbors[a].add(int(b)); neighbors[b].add(int(a))
        groups = []
        while pending:
            group, todo = set(), [pending.pop()]
            while todo:
                node = todo.pop()
                group.add(node)
                new = neighbors[node] & pending
                pending -= new
                todo.extend(new)
            groups.append(sorted(group))
        for group in groups:
            rest = self.meshes[name]['rest'][group]
            if np.linalg.norm(rest[:, None, :]-rest[None, :, :], axis=-1).max() > self.limits['max_grasp_patch_diameter_m']:
                raise ValueError('Engineering material grasp fragment is not local in its rest mesh')
        self.mask_cache.add(key)

    def _check_world_grasp_regions(self, cloths, masks):
        # A physical clamp may grasp both layers of a folded towel: those
        # material fragments are disconnected in the rest mesh, yet occupy
        # one small world-space region. Enforce two physical regions across
        # all cloths, independently of how many layers each clamp contains.
        points = [cloths[name][mask] for name, mask in masks.items() if mask]
        if not points:
            return
        points = np.concatenate(points)
        too_far = np.linalg.norm(points[:, None]-points[None, :], axis=-1) > self.limits['max_grasp_patch_diameter_m']
        # Exact two-cluster diameter test: points farther apart than the
        # declared diameter must get different colors. A bipartite conflict
        # graph gives <=2 groups, each with all pair distances <=75 mm.
        # Recompute from current native geometry on every controlled step;
        # a previously valid mask cannot stay cached as the cloth unfolds.
        colors = np.full(len(points), -1, dtype=int)
        for start in range(len(points)):
            if colors[start] != -1:
                continue
            colors[start] = 0
            pending = [start]
            while pending:
                node = pending.pop()
                for neighbor in np.flatnonzero(too_far[node]):
                    if colors[neighbor] == colors[node]:
                        raise ValueError('Controlled vertices exceed two local world-space grasp regions')
                    if colors[neighbor] == -1:
                        colors[neighbor] = 1-colors[node]
                        pending.append(int(neighbor))

    def update(self, observation):
        return self.observe(observation.get('laundry_native_frame', observation))

    def report(self):
        return deepcopy(self.last_result)

    def inspect_conditions(self, observation=None):
        return self.report()
