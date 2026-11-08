"""Read-only actual-contact cleaning model; independent of robot strategy.

Only consecutive loaded contact samples can add local tangential travel. A
static pad, position-only projection, unsupported sweep or skipped jump cannot
clean. This model does not simulate absorbency, chemistry or deformable foam.
"""
import math
import numpy as np

from .native_evidence import EPS, angle, rotation, subtract, norm


def validate_wiping_metadata(metadata):
    expected = {'sponge'}
    composites = metadata.get('elastic_composites')
    if composites:
        if set(composites) != {'sponge'} or composites['sponge']['parent_rigid_body'] != 'sponge':
            raise ValueError('Wiping requires one logical elastic sponge')
        expected = set(composites['sponge']['member_ids'])
        if len(expected) != 28 or 'sponge' not in expected:
            raise ValueError('Wiping requires all elastic sponge members')
    if set(metadata['actors']) != expected or set(metadata['fixtures']) != {'sponge_tray'}:
        raise ValueError('Wiping requires one sponge and its real return tray')
    goals = metadata['goals']
    cells, patches = goals['soil_cells'], goals['soil_patches']
    ids = [row['id'] for row in cells]
    if not cells or len(ids) != len(set(ids)) or len(patches) != 3:
        raise ValueError('Three distinct visible wiping patches are required')
    if set(goals['wiping_support_paths']) != set(metadata['work_surface_paths']):
        raise ValueError('Wiping contact must bind the actual task work surface')
    if not math.isfinite(goals['wiping_surface_z_m']):
        raise ValueError('Wiping height must be finite')
    all_members = [name for patch in patches for name in patch['cell_ids']]
    if sorted(ids) != sorted(all_members) or len({p['id'] for p in patches}) != 3:
        raise ValueError('Every wiping cell belongs to exactly one patch')
    lookup = {p['id']: set(p['cell_ids']) for p in patches}
    for row in cells:
        p = row['center_world_m']
        if (len(p) != 3 or not all(math.isfinite(v) for v in p)
                or abs(p[2]-goals['wiping_surface_z_m']) > .001
                or not math.isfinite(row['area_m2']) or row['area_m2'] <= 0
                or row['id'] not in lookup.get(row['patch'], ())):
            raise ValueError('Invalid physical wiping cell geometry')
    rows = metadata['objects']
    obj = rows['sponge'] if isinstance(rows, dict) else next(row for row in rows if row['id'] == 'sponge')
    bounds = np.asarray(obj['wiping_face_bounds_xy_m'], dtype=float)
    if (bounds.shape != (2, 2) or not np.isfinite(bounds).all()
            or not (bounds[0] < bounds[1]).all()
            or not math.isfinite(obj['wiping_face_local_z_m'])):
        raise ValueError('Sponge wiping face must have finite nonempty bounds')


class ContactWiping:
    def __init__(self, metadata, limits, physics_dt_s):
        validate_wiping_metadata(metadata)
        self.limits, self.dt = limits, physics_dt_s
        self.goals = metadata['goals']
        rows = metadata['objects']
        self.spec = rows['sponge'] if isinstance(rows, dict) else next(row for row in rows if row['id'] == 'sponge')
        self.cells = self.goals['soil_cells']
        self.points = np.asarray([c['center_world_m'] for c in self.cells], dtype=float)
        self.points[:, 2] = self.goals['wiping_surface_z_m']
        self.areas = np.asarray([c['area_m2'] for c in self.cells])
        self.groups = {p['id']: np.asarray([c['patch'] == p['id'] for c in self.cells])
                       for p in self.goals['soil_patches']}
        self.reset()

    def reset(self):
        self.travel = np.zeros(len(self.cells))
        self.stroke_anchor = np.zeros((len(self.cells), 2))
        self.anchor_valid = np.zeros(len(self.cells), dtype=bool)
        self.previous_local = None
        self.previous_covered = None
        self.previous_pose = None
        self.valid_contact_steps = 0

    def update(self, pose, *, loaded_contact, baseline=False):
        r = np.asarray(rotation(pose['quaternion_wxyz']))
        local = (self.points-np.asarray(pose['position_m'])) @ r
        lo, hi = np.asarray(self.spec['wiping_face_bounds_xy_m'])
        face = self.spec['wiping_face_local_z_m']
        loaded_contact = bool(loaded_contact and not baseline
            and r[2, 2] >= math.cos(self.limits['wiping_tilt_max_rad']))
        covered = ((local[:, :2] >= lo).all(axis=1) & (local[:, :2] <= hi).all(axis=1)
            & (np.abs(local[:, 2]-face) <= self.limits['wiping_surface_gap_m']))
        if loaded_contact and self.previous_local is not None:
            # Native pose continuity limits are local anti-credit checks. Fast
            # or discontinuous motion earns no cleaning, but remains retryable.
            maximum = self.limits['wiping_max_speed_m_s']*self.dt
            shift = norm(subtract(pose['position_m'], self.previous_pose['position_m']))
            turn = angle(pose['quaternion_wxyz'], self.previous_pose['quaternion_wxyz'])
            if (shift <= maximum+EPS
                    and turn <= self.limits['wiping_max_angular_speed_rad_s']*self.dt+EPS):
                travel = np.linalg.norm(local[:, :2]-self.previous_local[:, :2], axis=1)
                valid = (covered & self.previous_covered
                    & (travel >= self.limits['wiping_min_speed_m_s']*self.dt)
                    & (travel <= maximum+EPS))
                # Solver jitter has substantial summed absolute path length
                # over time. Require a real 3 mm excursion before adding each
                # stroke segment; tiny stationary oscillations never accrue.
                self.anchor_valid &= covered & self.previous_covered
                new = valid & ~self.anchor_valid
                self.stroke_anchor[new] = self.previous_local[new, :2]
                self.anchor_valid |= new
                excursion = np.linalg.norm(local[:, :2]-self.stroke_anchor, axis=1)
                credit = valid & self.anchor_valid & (excursion+EPS >= self.limits['wiping_min_stroke_m'])
                self.travel[credit] += excursion[credit]
                self.stroke_anchor[credit] = local[credit, :2]
                self.valid_contact_steps += int(bool(valid.any()))
            else:
                self.anchor_valid[:] = False
        if not loaded_contact:
            self.anchor_valid[:] = False
        # No contact memory is allowed for cleaning. Reacquisition starts a new
        # segment, so even a brief airborne traversal cannot fill hidden gaps.
        self.previous_local = local if loaded_contact else None
        self.previous_covered = covered if loaded_contact else None
        self.previous_pose = pose if loaded_contact else None
        cleaned = self.travel+EPS >= self.limits['wiping_cell_travel_m']
        coverage = float(self.areas[cleaned].sum()/self.areas.sum())
        groups = {name: float(self.areas[mask & cleaned].sum()/self.areas[mask].sum())
                  for name, mask in self.groups.items()}
        ready = (coverage+EPS >= self.limits['wiping_total_coverage_min']
                 and min(groups.values())+EPS >= self.limits['wiping_patch_coverage_min'])
        return {'cleaned_cells': {c['id']: bool(value) for c, value in zip(self.cells, cleaned)},
            'coverage_fraction': coverage, 'patch_coverage': groups,
            'cleaning_ready': bool(ready), 'loaded_wiping_contact': loaded_contact,
            'valid_contact_motion_steps': self.valid_contact_steps}
