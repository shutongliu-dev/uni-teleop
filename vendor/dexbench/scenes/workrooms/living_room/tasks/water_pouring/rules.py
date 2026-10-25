"""Frozen practical pouring rules and native-observer configuration; stdlib only."""
from copy import deepcopy
import math
import hashlib
import json
import struct

RULES_VERSION = 'living_room_water_pouring_rules_v2'
# Provisional task quantities, not a frozen vessel specification or calibration.
DESIGN_QUANTITIES_ML = {
    'initial_water_ml': 250., 'target_water_ml': 100., 'target_tolerance_ml': 10.,
    'max_spill_ml': 5., 'cup_usable_capacity_ml': 200., 'kettle_usable_capacity_ml': 500.,
}
RULES = {
    'schema': RULES_VERSION, 'status': 'prepared_practical_task', 'competitive': False,
    'instruction_zh': '单臂抓起水壶，向杯中倒入公布的目标水量，减少洒漏，再把壶正放回归位区并松手。',
    'stages': ['grasp_handle', 'lift_kettle', 'align_spout', 'tilt_and_pour',
               'stop_flow', 'return_kettle', 'release_and_settle'],
    'objects': {
        'kettle': 'Free dynamic rigid body with an open spout, real inner cavity, handle clearance and a physically represented secured lid if present.',
        'cup': 'Empty free dynamic cup with a real open cavity; never fixed to the table.',
        'water': 'Fixed initial inventory of mass-bearing fluid particles; no spawning, deletion or teleportation after reset.',
        'spill_tray': 'Task-owned physical catch tray; all water landing outside the vessel interiors still counts as spill.',
    },
    'fluid': {
        'candidate': 'PhysX GPU PBD fluid', 'native_adapter': 'living_water_observation_v1',
        'quantitative_water_load_calibration_required_for_scene_use': False,
        'rendering': 'Particle surface reconstruction is visual only; score from unsmoothed simulation state.',
        'forbidden_shortcuts': ['Animated water stream', 'Rigid blue water block', 'Changing a water-level scalar from kettle angle',
                               'Spawning fluid at the spout', 'Deleting spilled particles', 'Kinematic kettle during robot pouring'],
    },
    'measurement': {
        'quantity_basis': 'Sum frozen per-particle physical mass, divide by frozen reference water density; report equivalent mL.',
        'cup_water': 'Current original-water inventory contained inside the actual cup cavity below its rim, excluding exterior/airborne/escaped particles.',
        'spill': 'Cumulative mass of unique original particles entering frozen spill regions on the tray/table/floor/exterior vessel surfaces, or escaping the task domain. Classify from raw particle trajectories and geometry; in-flight water remains pending until resolved.',
        'spill_recovery': 'Recovering spilled water never reduces cumulative spill.',
        'water_difference': 'Initial water minus current kettle and cup water equals current outside water, including in-flight/unresolved water. Subtract in-flight/unresolved water to obtain current spill; settled water difference equals current spill when no water remains unresolved.',
        'conservation': 'Initial water minus current kettle and cup water is the outside-water difference; subtract in-flight/unresolved water to obtain current spill. Every original particle is assigned once. Cumulative spill is a separate history, never added to the current mass partition or reduced by recovery.',
        'missing_inventory': 'Missing particle IDs, lost mass, stale readback, reset changes or incomplete intervals invalidate the episode, never count as low spill.',
        'public_feedback': 'Public task rules declare water targets and spill budget. The saved referee result reports initial, kettle and cup mL, water difference, unresolved mL, current and cumulative spill, and remaining spill budget; raw particle truth stays service-owned.',
    },
    'success': {'all_of': [
        'Actual water transfer from the lifted kettle through its spout is observed; reset and visual appearance alone earn no credit.',
        'Retained cup water lies within the published target interval, and cumulative spill does not exceed its budget.',
        'Cup remains upright and supported; kettle is upright and supported inside its designated return zone.',
        'Hand contacts with both vessels have ended; no vessel is supported by the robot.',
        'Cup volume and released vessel poses meet the published sampled settling thresholds throughout a complete 0.3-second window; at most 1 mL may remain unresolved.',
    ], 'stable_duration_s': .3, 'reset_credit': False},
    'recovery': ['Put down and regrasp the kettle.', 'Adjust spout alignment, pour rate and wrist angle.',
                 'Stop and top up an underfilled cup through actual robot actions.'],
    'failure': ['Cumulative spill exceeds the published limit.', 'The cup tips beyond its published failure limit, or either vessel falls below the table.'],
    'overfill': 'Outside the target interval is not success; draining/deleting water or automatic reset is forbidden.',
    'budget_exhaustion': 'incomplete', 'evidence_fault': 'invalid_no_score',
    'randomization': ['Initial release uses the frozen vessel poses; episode seed and nominal water goal are published.'],
    'asset_freeze_required': ['Vessel mesh/cavity/spout/handle dimensions, dry masses and inertia',
        'Water density, particle masses/spacing, fluid material and solver settings',
        'Collision thickness/resolution, finite leak and mass-accounting tolerances',
        'Cup occupancy, flight/settling, spill-region classification, interval coverage, upright and return-zone thresholds',
        'Readback provenance; known quantitative water-load limitation recorded separately from task scoring'],
    'qualification': 'Practical read-only task scoring. Native initialization and robot-policy success remain separate evidence; quantitative water-weight calibration is a later improvement.',
}


def public_rules(*, quantities=None):
    q = deepcopy(DESIGN_QUANTITIES_ML)
    if quantities is not None:
        if not isinstance(quantities, dict) or set(quantities) - set(q):
            raise ValueError('Unknown water quantity fields')
        q.update(quantities)
    for name, value in q.items():
        if (type(value) not in (int, float) or not math.isfinite(value)
                or value < 0 or (value == 0 and name != 'max_spill_ml')):
            raise ValueError(name + ' must be finite and positive (spill allowance may be zero)')
        q[name] = float(value)
    low = q['target_water_ml'] - q['target_tolerance_ml']
    high = q['target_water_ml'] + q['target_tolerance_ml']
    if low <= 0 or high > min(q['initial_water_ml'], q['cup_usable_capacity_ml']):
        raise ValueError('Target interval must be positive and fit the initial water and usable cup capacity')
    if q['initial_water_ml'] > q['kettle_usable_capacity_ml']:
        raise ValueError('Initial water exceeds usable kettle capacity')
    if q['max_spill_ml'] >= low:
        raise ValueError('Spill allowance must be smaller than the minimum target water')
    result = deepcopy(RULES)
    result['goal'] = {**q, 'target_interval_ml': [low, high],
                      'status': 'published_nominal_quantities'}
    result['success']['thresholds'] = thresholds()
    result['measurement']['unresolved_allowance_ml'] = thresholds()['maximum_unresolved_ml']
    result['measurement']['support_and_release'] = 'Native rigid contact impulses; sleeping support may retain its last observed state only while the vessel pose is unchanged and motion is below the published thresholds. Release means no active robot contact during the sampled window.'
    return result


def thresholds():
    return {'dwell_s': .3, 'maximum_observation_gap_s': .1,
        'maximum_unresolved_ml': 1., 'maximum_cup_volume_range_ml': 2.,
        'maximum_linear_speed_m_s': .03, 'maximum_angular_speed_rad_s': .2,
        'maximum_tilt_rad': math.radians(15.), 'failure_tilt_rad': math.radians(100.),
        'return_xy_radius_m': .06, 'return_z_tolerance_m': .015,
        'minimum_lift_m': .02, 'minimum_spout_exit_ml': 1.,
        'minimum_support_force_n': .01, 'maximum_robot_contact_force_n': .01,
        'maximum_stable_position_range_m': .01, 'maximum_stable_orientation_range_rad': .1,
        'sleep_support_position_tolerance_m': .002, 'sleep_support_orientation_tolerance_rad': .03}


def configuration_hash(configuration):
    value = {k: v for k, v in configuration.items() if k != 'contract_sha256'}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def evaluation_configuration(*, seed=0, physics_dt_s=.0005, asset_metadata=None, quantities=None):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an unsigned 32-bit integer')
    if type(physics_dt_s) not in (int, float) or not math.isfinite(physics_dt_s) or not 0 < physics_dt_s <= .005:
        raise ValueError('physics_dt_s must be positive and at most 5 ms')
    rules = public_rules(quantities=quantities)
    yaw = math.atan2(.12, .28)
    defaults = {
        'containers': {
            'kettle': {'path': '/World/LivingTask/Kettle', 'initial_position_m': [.52, -.055, .7861],
                'initial_yaw_rad': yaw, 'inner_profile_z_radius_spout_m': [[.006,0.,0.],[.006,.045,0.],[.012,.048,0.],[.075,.049,0.],[.096,.042,.020],[.102,.040,.030]]},
            'cup': {'path': '/World/LivingTask/Cup', 'initial_position_m': [.80,.065,.7861],
                'initial_yaw_rad': 0., 'inner_profile_z_radius_spout_m': [[.005,0.,0.],[.005,.032,0.],[.065,.033,0.],[.067,.033,0.]]}},
        'tray': {'path':'/World/LivingTask/CatchTray','bounds_xy_m':[[.385,-.175],[.915,.175]],
            'interior_floor_z_m':.786,'rim_top_z_m':.803,'wall_thickness_m':.007},
        'table_top_z_m':.78,
        'water': {'particle_count':9259,'particle_mass_kg':.25/9259,'total_mass_kg':.25,'reference_density_kg_m3':1000.}}
    meta = deepcopy(defaults if asset_metadata is None else asset_metadata)
    water = meta['water']
    count = water['particle_count']
    if type(count) is not int or not 0 < count <= 1000000:
        raise ValueError('Invalid frozen particle count')
    mass, density = water['particle_mass_kg'], water['reference_density_kg_m3']
    if any(type(v) not in (int,float) or not math.isfinite(v) or v <= 0 for v in (mass,density)):
        raise ValueError('Invalid particle mass/density')
    if abs(count*mass/density*1e6-rules['goal']['initial_water_ml']) > 1.e-4:
        raise ValueError('Frozen water inventory differs from public initial quantity')
    tray = {**deepcopy(defaults['tray']), **deepcopy(meta['tray'])}
    containers = {k: {n:deepcopy(meta['containers'][k][n]) for n in
        ('path','initial_position_m','initial_yaw_rad','inner_profile_z_radius_spout_m')} for k in ('kettle','cup')}
    c = {'schema': 'living_water_evaluation_v1', 'seed': seed, 'physics_dt_s':float(physics_dt_s),
        'thresholds': thresholds(), 'quantities':{k:rules['goal'][k] for k in DESIGN_QUANTITIES_ML},
        'particle_count':count, 'particle_mass_kg':float(mass),'reference_density_kg_m3':float(density),
        'particle_ids_sha256':hashlib.sha256(struct.pack('<'+str(count)+'q',*range(count))).hexdigest(),
        'particle_set_id':'water','vessel_ids':['kettle','cup'], 'containers':containers,
        'vessel_actor_paths':{k:v['path'] for k,v in containers.items()},
        'robot_actor_prefixes':['/World/Robot/','/World/Robots/'],
        'support_actor_prefixes':[meta['tray']['path'], '/World/Fixtures/', '/World/Environment/', '/World/ProbeTable'],
        'tray':tray,'table_top_z_m':float(meta['table_top_z_m']),
        'table_bounds_xy_m':[[.15,-.5],[1.15,.5]], 'task_domain_m':[[-2.,-2.,-.1],[3.,2.,2.]],
        'geometry_spill_envelope_m':.006}
    c['contract_sha256'] = configuration_hash(c)
    return c
