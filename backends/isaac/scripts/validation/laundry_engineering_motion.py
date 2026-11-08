"""Declared local grasp trajectories for engineering cloth feasibility only.

These targets drive two native attachment anchors. They never prescribe the
remaining cloth state, substitute a folded mesh, or imply a robot policy.
"""
from dataclasses import dataclass
import math

import numpy as np


FOLD_TRAJECTORIES = ('arc', 'bezier')
STACK_TRAJECTORIES = ('lift', 'low-clearance')
STACK_GRIP_LAYOUTS = ('diagonal', 'crease-edge')
GARMENT_STACK_PROFILE = 'folded_garment_leading_patches_v1'
BEZIER_PAPER_URL = 'https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2022.989702/full'


def minimum_jerk(time_s, begin, end):
    u = float(np.clip((time_s-begin)/(end-begin), 0., 1.))
    return u*u*u*(10.+u*(-15.+6.*u))


def ramp(time_s, begin, end):
    u = float(np.clip((time_s-begin)/(end-begin), 0., 1.))
    return u*u*(3.-2.*u)


def low_clearance_stack_profile(task, initial, origins, targets, *, grip_layout='diagonal'):
    """Freeze a table-supported candidate from initial geometry, not live shape targets."""
    bases = [row for row in task['asset_metadata']['cloths'] if row['id'] == 'towel_base']
    if len(bases) != 1:
        raise ValueError('Low-clearance stacking requires one initial towel_base')
    base = np.asarray(bases[0]['initial_vertices_m'], dtype=float)
    if base.ndim != 2 or base.shape[1] != 3 or not len(base) or not np.isfinite(base).all():
        raise ValueError('Low-clearance stacking requires finite initial base geometry')
    displacement = targets-origins
    if (not np.allclose(displacement, [displacement[0, 0], 0., 0.], atol=1.e-12, rtol=0)
            or displacement[0, 0] <= 0):
        raise ValueError('Low-clearance stacking requires a common positive X translation')
    near_edge, front_edge = float(base[:, 0].min()), float(initial[:, 0].max())
    approach = near_edge-front_edge-.030
    total = float(displacement[0, 0])
    if not 0. < approach < total:
        raise ValueError('Low-clearance stacking requires an approach point before the base edge and final target')
    front = int(np.argmax(origins[:, 0]))
    if grip_layout == 'diagonal' and origins[front, 0]-origins[1-front, 0] <= 0:
        raise ValueError('Low-clearance stacking requires distinct front and back grasp regions')
    heights = [.020, .020]
    heights[front] = .030
    if grip_layout == 'crease-edge':
        # Native settled folds can have asymmetric crease corners. The common
        # translation preserves that shape without requiring an ideal U-fold.
        heights = [.030, .030]
    return {'schema': 'laundry_stack_trajectory.v1', 'profile': 'low-clearance',
        'parameterization': 's=10u^3-15u^4+6u^5; u=clip((t-begin)/(end-begin),0,1)',
        'phase_times_s': {'lift': [.6, 1.6], 'approach': [1.6, 4.0],
            'edge_height_transition': [4.0, 4.6], 'cross_base': [4.6, 7.0], 'lower': [7.0, 7.8]},
        'initial_geometry_source': 'frozen task.asset_metadata cloth initial_vertices_m',
        'moving_initial_front_x_m': front_edge, 'base_initial_near_edge_x_m': near_edge,
        'initial_gap_x_m': near_edge-front_edge, 'approach_margin_m': .030,
        'approach_dx_m': approach, 'total_dx_m': total,
        'initial_lift_m': .025, 'crossing_lift_by_anchor_m': heights,
        'front_crease_anchor_index': front,
        'back_open_anchor_index': 1-front if grip_layout == 'diagonal' else None,
        'final_lift_m': .0105, 'release_time_s': 8.5,
        'grip_patches_fixed': True, 'extra_constraints': 0,
        'scope': 'Unvalidated low-clearance candidate. Only two existing local anchors move; free cloth and base remain dynamic.',
        'limitations': 'Initial edge geometry schedules the approach only; anchor height does not guarantee free-edge clearance, support, or absence of base displacement.'}


@dataclass
class Motion:
    task_id: str
    patches: list
    origins: list
    release_time_s: float
    duration_s: float
    metadata: dict
    target_centers: np.ndarray
    fold_x: float
    placement_reference: np.ndarray | None = None

    def observe_placement_reference(self, time_s, vertices):
        """Read the free opposite edge once; never change any cloth state."""
        alignment = self.metadata.get('fold_edge_alignment')
        if alignment is None or time_s < alignment['sample_time_s'] or self.placement_reference is not None:
            return
        if time_s > alignment['sample_time_s'] + alignment['physics_dt_s']:
            raise ValueError('Missed the declared opposite-edge observation time')
        q = np.asarray(vertices, dtype=float)
        if q.shape != (alignment['vertex_count'], 3) or not np.isfinite(q).all():
            raise ValueError('Complete native cloth positions required for edge alignment')
        self.placement_reference = np.asarray([q[ids].mean(axis=0)
            for ids in alignment['observed_vertex_ids']])
        alignment['actual_observation_time_s'] = float(time_s)
        alignment['observed_centers_m'] = self.placement_reference.tolist()

    def pose(self, index, time_s):
        if index not in (0, 1) or not math.isfinite(time_s) or time_s < 0:
            raise ValueError('Expected one of two anchors and nonnegative finite native time')
        start = np.asarray(self.origins[index], dtype=float)
        if self.task_id == 'towel_flattening':
            carry = ramp(time_s, 1.2, 3.5)
            lift = .04*ramp(time_s, .5, 1.2)*(1.-ramp(time_s, 3.5, 4.8))
            center = start + carry*(self.target_centers[index]-start)
            center[2] += lift
            angle = -.18*carry
            rotation = [math.cos(angle/2), 0., 0., math.sin(angle/2)]
        elif self.task_id == 'towel_folding':
            profile = self.metadata.get('fold_trajectory')
            if profile is None:
                # Preserve the historical arc and cubic timing exactly.
                fraction = ramp(time_s, .8, 7.2)
                angle = math.pi*fraction
                radius = start[0]-self.fold_x
                gap = .006*fraction
                center = np.array([self.fold_x+radius*math.cos(angle), start[1],
                    start[2]+radius*math.sin(angle)+gap])
            else:
                fraction = minimum_jerk(time_s, .8, 7.2)
                angle = math.pi*fraction
                p0, p1, p2 = np.asarray(profile['control_points_m'][index])
                center = (1.-fraction)**2*p0+2.*fraction*(1.-fraction)*p1+fraction**2*p2
                center[2] += .006*fraction
            if self.placement_reference is not None:
                # The free lower half can slide during lifting. Place the
                # held corners above its observed corners, rather than assume
                # the opposite edge stayed at its authored coordinates.
                reference = np.array([2*self.fold_x-start[0], start[1], start[2]])
                center += ramp(time_s, 6., 7.2)*(self.placement_reference[index]-reference)
            rotation = [math.cos(angle/2), 0., -math.sin(angle/2), 0.]
        else:
            profile = self.metadata.get('stack_trajectory')
            if profile is None:
                carry = ramp(time_s, 2.0, 5.5)
                center = start + carry*(self.target_centers[index]-start)
                # The final 10.5 mm offset clears the two-layer base. Native
                # gravity/contact must establish actual support after release.
                center[2] = start[2]+.16*ramp(time_s, .6, 2.)-.1495*ramp(time_s, 5.5, 7.5)
            else:
                phases = profile['phase_times_s']
                progress = {key: minimum_jerk(time_s, *times) for key, times in phases.items()}
                center = start.copy()
                center[0] += (profile['approach_dx_m']*progress['approach']
                    +(profile['total_dx_m']-profile['approach_dx_m'])*progress['cross_base'])
                lift, cross = profile['initial_lift_m'], profile['crossing_lift_by_anchor_m'][index]
                center[2] += (lift*progress['lift']+(cross-lift)*progress['edge_height_transition']
                    +(profile['final_lift_m']-cross)*progress['lower'])
            rotation = [1., 0., 0., 0.]
        # A disabled anchor withdraws independently. If disabling were ignored,
        # the resulting cloth lift would fail release/support/stability gates.
        center[2] += .12*ramp(time_s, self.release_time_s+.25, self.release_time_s+1.25)
        return center.tolist(), rotation


def _garment_patches(spec, initial):
    """Two physical clamps, each containing small connected material fragments."""
    lo, hi = initial.min(axis=0), initial.max(axis=0)
    centers = [lo[:2]+[.9*(hi[0]-lo[0]), fraction*(hi[1]-lo[1])]
               for fraction in (.1, .9)]
    patches = [np.flatnonzero(np.linalg.norm(initial[:, :2]-center, axis=1) <= .030).tolist()
               for center in centers]
    selected = set(sum(patches, []))
    if (any(len(ids) < 3 for ids in patches) or len(selected) != sum(map(len, patches))
            or len(selected) > .1*len(initial)):
        raise ValueError('Garment grasp patches must be distinct, nonempty and below ten percent of nodes')
    rest = np.asarray(spec['rest_vertices_m'], dtype=float)
    triangles = np.asarray(spec['triangles'], dtype=int)
    if (rest.shape != initial.shape or not np.isfinite(rest).all() or triangles.ndim != 2
            or triangles.shape[1] != 3 or triangles.min() < 0 or triangles.max() >= len(rest)):
        raise ValueError('Complete garment material topology required')
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
    diameter = lambda points: float(np.linalg.norm(points[:, None]-points[None, :], axis=-1).max())
    world_diameters = [diameter(initial[ids]) for ids in patches]
    material_diameters = [diameter(rest[ids]) for ids in fragments]
    if max(world_diameters+material_diameters) > .075:
        raise ValueError('Garment grasp patch or material fragment exceeds 75 mm')
    return patches, {'selection': 'All initial layers within a 30 mm XY radius at footprint fractions (0.9, 0.1) and (0.9, 0.9)',
        'selection_centers_xy_m': [center.tolist() for center in centers],
        'selection_radius_xy_m': .030, 'controlled_fraction': len(selected)/len(initial),
        'world_patch_diameters_m': world_diameters,
        'material_fragment_vertex_ids': fragments,
        'material_fragment_diameters_m': material_diameters,
        'locality_limit_m': .075, 'base_controlled_nodes': 0}


def _build_garment_motion(task, scene, passive_after_release_s):
    """Candidate transport of the existing fold; no prescribed free-cloth shape."""
    specs = {row['id']: row for row in task['asset_metadata']['cloths']}
    if set(specs) != {'towel', 'towel_base'} or any(row.get('mesh_type') != 'custom_surface' for row in specs.values()):
        raise ValueError('Garment stacking requires its two native custom surfaces')
    spec, base_spec = specs['towel'], specs['towel_base']
    initial, base = (np.asarray(row['initial_vertices_m'], dtype=float) for row in (spec, base_spec))
    if any(q.ndim != 2 or q.shape[1] != 3 or len(q) < 3 or not np.isfinite(q).all() for q in (initial, base)):
        raise ValueError('Finite initial garment and base geometry required')
    patches, patch_receipt = _garment_patches(spec, initial)
    origins = np.asarray([initial[ids].mean(axis=0) for ids in patches])
    target_xy = np.asarray(task['asset_metadata']['target_center_xy_m'], dtype=float)
    if target_xy.shape != (2,) or not np.isfinite(target_xy).all():
        raise ValueError('Finite garment target center required')
    displacement = target_xy-(initial[:, :2].min(axis=0)+initial[:, :2].max(axis=0))/2
    if displacement[0] <= 0 or abs(displacement[1]) > 1.e-10:
        raise ValueError('Garment profile requires a positive X translation to the target')
    targets = origins.copy()
    targets[:, :2] += displacement
    profile = low_clearance_stack_profile(task, initial, origins, targets, grip_layout='crease-edge')
    thickness = [float(row['physical_thickness_m']) for row in (spec, base_spec)]
    if not all(math.isfinite(value) and value > 0 for value in thickness):
        raise ValueError('Positive nominal garment thickness required')
    # This is an initial-geometry clearance, not a live support measurement.
    # Native contact and gravity decide the free-cloth shape and support.
    final_lift = float(base[:, 2].max()-initial[:, 2].min()+sum(thickness)/2)
    if final_lift <= 0:
        raise ValueError('Garment base must require a positive placement height')
    cross_lift = max(.030, final_lift+.015)
    profile.update(profile=GARMENT_STACK_PROFILE, initial_lift_m=max(.025, final_lift+.010),
        crossing_lift_by_anchor_m=[cross_lift, cross_lift], final_lift_m=final_lift,
        front_crease_anchor_index=None, back_open_anchor_index=None,
        final_height_basis='Initial base highest midsurface minus moving lowest midsurface plus both nominal half-thicknesses',
        scope='Unvalidated garment transport candidate with two fixed local multilayer clamps; no robot and no free-cloth shape control.')
    release = float(profile['release_time_s'])
    passive = 6. if passive_after_release_s is None else passive_after_release_s
    limits, dt = task['evaluation']['thresholds'], float(scene['physics']['dt_s'])
    if (type(passive) not in (int, float) or not math.isfinite(passive) or passive < limits['stable_s']
            or release+passive >= limits['max_time_s'] or dt <= 0 or not math.isfinite(dt)):
        raise ValueError('Garment passive duration must cover stability and remain below the frozen task time limit')
    duration, steps = release+passive, (release+passive)/dt
    if abs(steps-round(steps)) > 1.e-8:
        raise ValueError('Total motion duration must contain an integer number of frozen physics steps')
    return Motion(task['id'], patches, origins.tolist(), release, duration,
        {'schema': 'laundry_engineering_motion.v2', 'task_id': task['id'],
         'garment_profile': GARMENT_STACK_PROFILE,
         'control': 'two local native attachment anchors; no robot',
         'patches': patches, 'garment_grasp': patch_receipt,
         'initial_geometry_source': 'frozen task.asset_metadata cloth initial_vertices_m and material triangles',
         'target_center_source': 'frozen task.asset_metadata.target_center_xy_m',
         'target_displacement_xy_m': displacement.tolist(),
         'release_time_s': release, 'passive_after_release_s': passive,
         'total_native_steps': int(round(steps)), 'stack_trajectory': profile,
         'stack_lift_m': cross_lift, 'target_shape_written': False, 'success_assumed': False,
         'post_release_anchor_withdrawal_m': .12,
         'release': 'disable all attachments, flush native changes, continue passive simulation'},
        targets, float((initial[:, 0].min()+initial[:, 0].max())/2))


def build_motion(task, scene, *, fold_align_to_observed_edge=False, fold_grip_span_m=None,
                 fold_trajectory='arc', fold_peak_height_m=None, passive_after_release_s=None,
                 stack_trajectory='lift', stack_grip_span_m=None, stack_grip_layout='diagonal',
                 garment_profile=None):
    task_id = task['id']
    if task_id in ('shirt_stacking', 'trousers_stacking'):
        if garment_profile != GARMENT_STACK_PROFILE:
            raise ValueError('Garment engineering motion requires its explicit garment profile')
        if (fold_align_to_observed_edge or fold_grip_span_m is not None or fold_trajectory != 'arc'
                or fold_peak_height_m is not None or stack_trajectory != 'lift'
                or stack_grip_span_m is not None or stack_grip_layout != 'diagonal'):
            raise ValueError('Towel trajectory options cannot modify the explicit garment profile')
        return _build_garment_motion(task, scene, passive_after_release_s)
    if garment_profile is not None:
        raise ValueError('Garment profile requires a garment stacking task')
    if task_id not in ('towel_flattening', 'towel_folding', 'towel_stacking'):
        raise ValueError('Unknown engineering towel task')
    if stack_trajectory not in STACK_TRAJECTORIES:
        raise ValueError('Unknown stacking trajectory profile')
    if stack_trajectory != 'lift' and task_id != 'towel_stacking':
        raise ValueError('Low-clearance trajectory is only a stacking motion')
    if stack_grip_span_m is not None and task_id != 'towel_stacking':
        raise ValueError('Explicit stack grasp spans are only a stacking option')
    if stack_grip_layout not in STACK_GRIP_LAYOUTS:
        raise ValueError('Unknown stacking grasp layout')
    if stack_grip_layout != 'diagonal' and (task_id != 'towel_stacking' or stack_grip_span_m is None):
        raise ValueError('Crease-edge grasp layout requires stacking with explicit material spans')
    if fold_trajectory not in FOLD_TRAJECTORIES:
        raise ValueError('Unknown folding trajectory profile')
    if fold_peak_height_m is not None and fold_trajectory != 'bezier':
        raise ValueError('A fold peak height is allowed only for the bezier trajectory')
    if fold_trajectory == 'bezier':
        fold_peak_height_m = .16 if fold_peak_height_m is None else fold_peak_height_m
        if (type(fold_peak_height_m) not in (int, float) or not math.isfinite(fold_peak_height_m)
                or fold_peak_height_m <= 0):
            raise ValueError('Bezier peak height must be a finite positive length in meters')
    if (fold_align_to_observed_edge or fold_grip_span_m is not None or fold_trajectory != 'arc') and task_id != 'towel_folding':
        raise ValueError('Opposite-edge alignment is only a folding motion')
    spec = next(row for row in task['asset_metadata']['cloths'] if row['id'] == 'towel')
    nx, ny = spec['cells_xy']
    initial = np.asarray(spec['initial_vertices_m'], dtype=float)
    rest = np.asarray(spec['rest_vertices_m'], dtype=float)
    if task_id == 'towel_flattening':
        # Opposite diagonal corners constrain location and heading while the
        # free interior settles against the existing table.
        patches = [[j*(nx+1)+i for j in (0, 1) for i in (0, 1)],
                   [j*(nx+1)+i for j in (ny-1, ny) for i in (nx-1, nx)]]
        release = 6.0
    elif task_id == 'towel_folding':
        px = py = 1
        if fold_grip_span_m is not None:
            span = np.asarray(fold_grip_span_m, dtype=float)
            pitch = np.asarray(spec['size_m'])/np.asarray([nx, ny])
            if span.shape != (2,) or not np.isfinite(span).all() or np.any(span <= 0) or np.linalg.norm(span) > .075:
                raise ValueError('Fold grasp spans must be finite, positive and fit a 75 mm region')
            cells = np.rint(span/pitch).astype(int)
            if np.any(cells < 1) or not np.allclose(cells*pitch, span, atol=1.e-9, rtol=0):
                raise ValueError('Explicit grasp spans must cover whole native mesh cells')
            px, py = map(int, cells)
        patches = [[j*(nx+1)+i for j in range(py+1) for i in range(nx-px, nx+1)],
                   [j*(nx+1)+i for j in range(ny-py, ny+1) for i in range(nx-px, nx+1)]]
        release = 9.5
    else:
        # Opposite corners of the *folded* rectangle support its diagonal.
        # One clamp holds the two open-edge layers, the other holds the
        # connected crease corner. Both are local in world and rest space;
        # all other nodes remain free to sag, contact and settle.
        if stack_grip_span_m is None:
            patches = [[j*(nx+1)+i for j in (0, 1) for i in (0, 1, nx-1, nx)],
                       [j*(nx+1)+i for j in (ny-1, ny) for i in (nx//2-1, nx//2, nx//2+1)]]
        else:
            span = np.asarray(stack_grip_span_m, dtype=float)
            pitch = np.asarray(spec['size_m'])/np.asarray([nx, ny])
            if span.shape != (2,) or not np.isfinite(span).all() or np.any(span <= 0) or np.linalg.norm(span) > .075:
                raise ValueError('Stack grasp spans must be finite, positive and fit a 75 mm region')
            cells = np.rint(span/pitch).astype(int)
            if np.any(cells < 1) or not np.allclose(cells*pitch, span, atol=1.e-9, rtol=0):
                raise ValueError('Explicit stack grasp spans must cover whole native mesh cells')
            px, py = map(int, cells)
            if nx % 2 or px % 2:
                raise ValueError('Symmetric crease grasp requires even grid and span X cell counts')
            # Open edges are distinct material corners, but one local clamp
            # holds both folded layers. The crease block has total width X,
            # centered on the material midline rather than X on each side.
            open_x = [*range(px+1), *range(nx-px, nx+1)]
            patches = [[j*(nx+1)+i for j in range(py+1) for i in open_x],
                       [j*(nx+1)+i for j in range(ny-py, ny+1)
                        for i in range(nx//2-px//2, nx//2+px//2+1)]]
            if stack_grip_layout == 'crease-edge':
                # Pull both ends of the connected leading fold. The open
                # trailing layers remain free and can be supported by the
                # table instead of suspending the whole folded rectangle.
                patches[0] = [j*(nx+1)+i for j in range(py+1)
                              for i in range(nx//2-px//2, nx//2+px//2+1)]
            if sum(map(len, patches)) > .1*len(initial):
                raise ValueError('Stack grasp patches exceed ten percent of the cloth vertices')
            if any(np.linalg.norm(initial[ids, None]-initial[None, ids], axis=-1).max() > .075
                   for ids in patches):
                raise ValueError('Initial stack grasp geometry exceeds a 75 mm world region')
        release = 8.5
    origins = np.array([initial[ids].mean(axis=0) for ids in patches])
    targets = origins.copy()
    if task_id == 'towel_flattening':
        targets[:, :2] = np.array([rest[ids, :2].mean(axis=0) for ids in patches]) + task['asset_metadata']['target_center_xy_m']
        targets[:, 2] = .903
    elif task_id == 'towel_stacking':
        targets[:, 0] += .65
    fold_x = float((initial[:, 0].min()+initial[:, 0].max())/2)
    passive = 6. if passive_after_release_s is None else passive_after_release_s
    if passive_after_release_s is not None:
        limits = task['evaluation']['thresholds']
        dt = float(task['evaluation']['physics_dt_s'])
        if type(passive) not in (int, float) or not math.isfinite(passive) or passive < limits['stable_s']:
            raise ValueError('Passive duration must be finite and at least the frozen stability interval')
        if release+passive >= limits['max_time_s']:
            raise ValueError('Total motion duration must stay below the frozen task time limit')
        steps = (release+passive)/dt
        if abs(steps-round(steps)) > 1.e-8:
            raise ValueError('Total motion duration must contain an integer number of frozen physics steps')
    duration = release+passive
    result = Motion(task_id, patches, origins.tolist(), release, duration,
        {'schema': 'laundry_engineering_motion.v2', 'task_id': task_id,
         'control': 'two local native attachment anchors; no robot',
         'patches': patches, 'release_time_s': release,
         'passive_after_release_s': duration-release,
         'stack_grasp_layout': 'folded rectangle opposite corners; open-edge two-layer patch and connected crease patch' if task_id == 'towel_stacking' else None,
         'stack_lift_m': .16 if task_id == 'towel_stacking' else None,
         'target_shape_written': False, 'success_assumed': False,
         'post_release_anchor_withdrawal_m': .12,
         'release': 'disable all attachments, flush native changes, continue passive simulation'},
        targets, fold_x)
    if stack_trajectory == 'low-clearance':
        result.metadata['stack_trajectory'] = low_clearance_stack_profile(
            task, initial, origins, targets, grip_layout=stack_grip_layout)
        result.metadata['stack_lift_m'] = .030
    if stack_grip_layout == 'crease-edge':
        result.metadata['stack_grip_layout'] = stack_grip_layout
        result.metadata['stack_grasp_layout'] = 'two end patches of the connected leading crease; open trailing layers remain free'
    if stack_grip_span_m is not None:
        result.metadata['stack_grip_span_m'] = list(map(float, stack_grip_span_m))
        result.metadata['stack_grip_cells_xy'] = [px, py]
        result.metadata['stack_grip_span_definition'] = (
            'Each open-edge layer uses an X by Y material corner block; the crease block '
            'uses total material width X symmetric about nx/2 and an end-corner Y span. '
            'Exactly two fixed local anchors; all controlled vertex IDs are in patches.')
        if stack_grip_layout == 'crease-edge':
            result.metadata['stack_grip_span_definition'] = (
                'Each crease patch uses total material width X symmetric about nx/2 and '
                'one end-corner Y span. Exactly two fixed local anchors, no open-edge '
                'constraints; all controlled vertex IDs are in patches.')
    if passive_after_release_s is not None:
        result.metadata['passive_duration_override'] = {
            'requested_s': float(passive), 'frozen_stable_s': float(limits['stable_s']),
            'frozen_max_time_s': float(limits['max_time_s']), 'physics_dt_s': dt,
            'total_duration_s': float(duration), 'total_native_steps': int(round(steps)),
            'scope': 'Additional real passive simulation after complete release; no cloth state writes or threshold changes.'}
    if fold_trajectory == 'bezier':
        controls = [[start.tolist(), [fold_x, float(start[1]), float(start[2]+2*fold_peak_height_m)],
                     [float(2*fold_x-start[0]), float(start[1]), float(start[2])]] for start in origins]
        result.metadata['fold_trajectory'] = {
            'schema': 'laundry_fold_trajectory.v1', 'profile': 'quadratic_bezier',
            'control_points_m': controls, 'begin_time_s': .8, 'end_time_s': 7.2,
            'peak_height_m': float(fold_peak_height_m),
            'peak_height_basis': 'Nominal rise above the start plane before the separate terminal gap.',
            'timing': 'quintic_minimum_jerk', 'time_parameter': 's=10*u^3-15*u^4+6*u^5; u=clip((t-.8)/6.4,0,1)',
            'terminal_gap_m': .006, 'terminal_gap_schedule': '0.006*s added along world +Z',
            'orientation': 'rotation about world -Y by pi*s', 'lateral_tilt_degrees': 0.,
            'grip_patches_fixed': True, 'extra_constraints': 0,
            'observed_edge_correction': 'Existing smooth 6.0 to 7.2 second correction when explicitly enabled.',
            'paper_url': BEZIER_PAPER_URL, 'paper_section': '3.2',
            'scope': 'Unvalidated trajectory candidate inspired by quadratic Bezier folding; no task or physical success assumed.'}
    if fold_grip_span_m is not None:
        result.metadata['fold_grip_span_m'] = list(map(float, fold_grip_span_m))
    if fold_align_to_observed_edge:
        result.metadata['fold_edge_alignment'] = {
            'sample_time_s': 6., 'blend_end_time_s': 7.2,
            'physics_dt_s': float(scene['physics']['dt_s']),
            'vertex_count': len(initial),
            'observed_vertex_ids': [[(index//(nx+1))*(nx+1)+nx-index%(nx+1)
                                    for index in patch] for patch in patches],
            'observation': 'One read of actual free lower-edge corners before placement; no extra controls or cloth state writes.',
            'terminal_midplane_gap_m': .006}
    return result
