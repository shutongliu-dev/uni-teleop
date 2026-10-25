"""Strict particle inventory and cumulative geometric spill diagnostics.

This consumes synchronized native samples; it does not certify their provenance,
particle inverse mass, robot actions, or task success. Current compartments and
historical spilled IDs are distinct quantities and must never be added together.
"""
from __future__ import annotations

import hashlib
import numpy as np

from .assets import cavity_contains


class LiquidLedger:
    def __init__(self, metadata, *, dt, initial_time):
        self.metadata = metadata
        self.dt, self.previous_time = float(dt), float(initial_time)
        water = metadata['water']
        self.count = int(water['particle_count'])
        self.mass = float(water['particle_mass_kg'])
        self.density = float(water['reference_density_kg_m3'])
        if not np.isfinite([self.dt, self.previous_time, self.mass, self.density]).all() or min(self.dt, self.mass, self.density) <= 0:
            raise ValueError('Invalid time or mass contract')
        self.spilled = np.zeros(self.count, dtype=bool)
        self.ever_left_kettle = np.zeros(self.count, dtype=bool)
        self.first_exit_regions = {'front_rim': 0, 'other_rim': 0, 'below_rim': 0}
        self.tick = 0

    def update(self, *, time_s, ids, points, velocities, world_to_local):
        points, velocities = np.asarray(points, float), np.asarray(velocities, float)
        if points.shape != (self.count, 3) or velocities.shape != points.shape or not np.array_equal(ids, np.arange(self.count)):
            raise ValueError('Particle inventory or ID order changed')
        if not np.isfinite(points).all() or not np.isfinite(velocities).all():
            raise ValueError('Nonfinite particle sample')
        if not np.isfinite(time_s) or abs(time_s-self.previous_time-self.dt) > min(1e-7, self.dt*1e-4):
            raise ValueError('Missing, repeated or out-of-order native step')
        classes, locals_, occupied = {}, {}, np.zeros(self.count, bool)
        for name in ('kettle', 'cup'):
            inverse = np.asarray(world_to_local[name], float)
            if inverse.shape != (4, 4) or not np.isfinite(inverse).all():
                raise ValueError('Invalid synchronized vessel transform')
            local = points @ inverse[:3, :3] + inverse[3, :3]
            mask = cavity_contains(local, self.metadata['containers'][name]['inner_profile_z_radius_spout_m'])
            if np.any(occupied & mask):
                raise ValueError('Vessel cavities overlap')
            classes[name] = mask
            locals_[name] = local
            occupied |= mask
        outside = ~occupied
        tray = self.metadata['tray']
        bounds = np.asarray(tray['bounds_xy_m'])
        over_tray = ((points[:, :2] >= bounds[0]) & (points[:, :2] <= bounds[1])).all(axis=1)
        expanded_tray = ((points[:, :2] >= bounds[0]-.006) & (points[:, :2] <= bounds[1]+.006)).all(axis=1)
        # v1 tray walls are 7 mm thick and rise 17 mm above the interior floor.
        # Include their top/outer surfaces: particles resting on the rim are spills.
        thickness = tray.get('wall_thickness_m', .007)
        rim_top = tray.get('rim_top_z_m', tray['interior_floor_z_m']+.017)
        near_wall = ((points[:, :2] <= bounds[0]+thickness+.006) |
                     (points[:, :2] >= bounds[1]-thickness-.006)).any(axis=1)
        over_table = ((points[:, :2] >= [.15, -.5]) & (points[:, :2] <= [1.15, .5])).all(axis=1)
        # The 6 mm envelope includes the authored 2.5 mm particle rest offset.
        # These are geometric surface interception regions, not contact reports.
        regions = {
            'tray': over_tray & (points[:, 2] <= tray['interior_floor_z_m']+.006),
            'tray_rim': expanded_tray & near_wall & (points[:, 2] <= rim_top+.006),
            'table': over_table & (points[:, 2] <= self.metadata['table_top_z_m']+.006),
            'floor': points[:, 2] <= .006,
            'escaped_domain': ((points < [-2., -2., -.1]) | (points > [3., 2., 2.])).any(axis=1),
        }
        new_hits, hit_counts = np.zeros(self.count, bool), {}
        for name, mask in regions.items():
            hit = mask & outside & ~new_hits
            hit_counts[name] = int(hit.sum())
            new_hits |= hit
        spilled = self.spilled | new_hits
        classes['spilled_outside'] = outside & spilled
        classes['in_flight_or_unresolved'] = outside & ~spilled
        counts = {name: int(mask.sum()) for name, mask in classes.items()}
        if sum(counts.values()) != self.count:
            raise ValueError('Mass compartments failed to close')
        self.spilled = spilled
        first_exit = ~classes['kettle'] & ~self.ever_left_kettle
        q = locals_['kettle']
        high = q[:, 2] >= .075
        front = high & (q[:, 0] > .025) & (np.abs(q[:, 1]) < .04)
        for name, mask in {'front_rim': front, 'other_rim': high & ~front, 'below_rim': ~high}.items():
            self.first_exit_regions[name] += int((first_exit & mask).sum())
        self.ever_left_kettle |= ~classes['kettle']
        self.previous_time, self.tick = float(time_s), self.tick+1
        ml_per_particle = self.mass/self.density*1e6
        return {'tick': self.tick, 'time_s': self.previous_time,
                'counts': counts, 'current_mass_kg': {k: n*self.mass for k, n in counts.items()},
                'current_nominal_volume_ml': {k: n*ml_per_particle for k, n in counts.items()},
                'cumulative_unique_spilled_particles': int(spilled.sum()),
                'cumulative_spill_ml': float(spilled.sum()*ml_per_particle),
                'recaptured_spilled_particles': int((occupied & spilled).sum()),
                'ever_left_kettle_particles': int(self.ever_left_kettle.sum()),
                'first_kettle_exit_regions': dict(self.first_exit_regions),
                'surface_interception_counts': hit_counts,
                'positions_sha256': hashlib.sha256(points.tobytes()).hexdigest(),
                'inventory_closed': True, 'task_success': False,
                'scope': 'Geometric ledger; nominal authored mass; unresolved outside particles cannot count as retained water'}
