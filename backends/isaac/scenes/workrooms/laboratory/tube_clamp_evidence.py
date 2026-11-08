"""Read-only grip/transport history for the passive tube clamp task."""
from copy import deepcopy
import math
import numpy as np
from .native_evidence import angle, distance, fixed, local, points, rotation, upright


def evaluate(observer, frame, objects, contacts):
    p, t, dt = observer.p, frame['time_s'], observer.c['physics_dt_s']
    if not hasattr(observer, 'clamp_history') or frame['baseline']:
        observer.clamp_history = {'start': None, 'contact_s': 0., 'last_contact_t': None,
                                 'last_relative': None, 'carried': False, 'max_transport_m': 0.}
    history = observer.clamp_history
    tube, tool, jaw = (objects[k] for k in ('tube', 'clamp_frame', 'clamp_jaw'))
    relative = local(tube['position_m'], tool)
    # The tube's origin is its bottom, and the pads occupy local z .039-.083.
    # These bounds reject accidental handle/rail contacts as a jaw grasp.
    geometry = (abs(relative[0]-.060) <= .010 and abs(relative[1]+.003) <= .010
                and -.020 <= relative[2] <= .045
                and abs(local(jaw['position_m'], tool)[1]) < .030
                and float(np.dot(rotation(tube['quaternion_wxyz'])[:,2], rotation(tool['quaternion_wxyz'])[:,2]))
                    >= math.cos(observer.limits['clamp_relative_alignment_rad']))
    bilateral = all(frozenset(('tube', name)) in contacts.pairs for name in ('clamp_frame','clamp_jaw'))
    if bilateral and geometry and 'tube' not in contacts.robot:
        history['last_contact_t'], history['last_relative'] = t, relative.copy()
    matching = (history['last_contact_t'] is not None
        and t-history['last_contact_t'] <= p['grip_contact_memory_s']+1.e-9
        and np.linalg.norm(relative-history['last_relative']) <= .001)
    held = bool(geometry and matching and 'tube' not in contacts.robot and not frame['baseline'])
    if held:
        history['contact_s'] += dt
        if history['start'] is None:
            history['start'] = deepcopy(tube)
    else:
        history['start'], history['contact_s'] = None, 0.
    bottom = points(tube, observer.specs['tube']['local_bounds_m'])[:,2].min()
    externally_supported = any(a == 'tube' and b not in ('clamp_frame','clamp_jaw') for a,b in contacts.upward)
    carried_distance = distance(tube, history['start']) if history['start'] is not None else 0.
    if held and bottom >= p['source_position_m'][2]+p['minimum_lift_m'] and not externally_supported:
        history['max_transport_m'] = max(history['max_transport_m'], carried_distance)
        if (history['contact_s'] >= p['minimum_bilateral_contact_s']
                and carried_distance >= p['minimum_carried_distance_m'] and not history['carried']):
            history['carried'] = True
            observer._event('tube_carried_by_clamp', frame)
    target = np.asarray(p['receiver_position_m'])
    receiver_geometry = (np.linalg.norm(np.asarray(tube['position_m'])[:2]-target[:2]) <= p['receiver_position_tolerance_m']
        and abs(bottom-target[2]) <= .005 and upright(tube, p['tube_upright_tolerance_rad']))
    tube_seated = observer._supported('tube',p['receiver_fixture'],objects,contacts,t,receiver_geometry)
    home = fixed([*p['tool_return_position_m'],0,0,0,1])
    tool_geometry = (distance(tool,home) <= p['tool_return_tolerance_m']
                     and angle(tool,home) <= p['tool_return_angle_rad'])
    tool_home = observer._supported('clamp_frame',p['tool_return_fixture'],objects,contacts,t,tool_geometry)
    detached = not any(frozenset(('tube',name)) in contacts.pairs for name in ('clamp_frame','clamp_jaw'))
    raw_quiet = all(np.linalg.norm(obj['linear_velocity_m_s']) <= observer.limits['maximum_linear_speed_m_s']
        and np.linalg.norm(obj['angular_velocity_rad_s']) <= observer.limits['maximum_angular_speed_rad_s']
        for obj in objects.values())
    goal = bool(history['carried'] and tube_seated and tool_home and detached and raw_quiet)
    return goal, {'bilateral_pad_contact':bool(bilateral and geometry), 'clamp_holding_tube':held,
        'tube_carried':history['carried'], 'maximum_gripped_transport_m':history['max_transport_m'],
        'tube_seated':bool(tube_seated), 'tool_returned':bool(tool_home), 'tool_detached':detached,
        'jaw_supported_by_native_joint':bool(tool_home), 'raw_motion_quiet':bool(raw_quiet)}
