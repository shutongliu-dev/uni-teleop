"""Discover the four implemented public Agent entry points without oracle tools."""
from __future__ import annotations

from copy import deepcopy


def board_game_interface_catalog(catalog):
    """Describe the existing authenticated player transport for a board release.

    The ordinary catalog stays compatible. The current function client already
    uses these routes; this adapter corrects global discovery, not execution.
    """
    result = deepcopy(catalog)
    player = '/board-games/{episode_id}/players/{player_id}'
    call = 'POST ' + player + '/call'
    result['shared_contract'].update({
        'create_endpoint': 'POST /board-games',
        'action_endpoint': call,
        'action_transport': 'bench_functions_v1',
        'action_functions': ['move_geometry', 'move_wrist', 'set_hand_targets', 'move_joints', 'hold'],
        'function_catalog_endpoint': 'GET ' + player + '/functions',
        'function_call_endpoint': call,
        'robot_guide_endpoint': call,
        'robot_guide_function': 'get_robot',
        'ik_query_endpoint': call,
        'ik_query_function': 'query_ik',
        'state_endpoint': 'GET ' + player + '/state',
        'result_endpoint': 'GET ' + player + '/result',
        'intent_endpoint': 'POST ' + player + '/intent',
        'authentication': {
            'scheme': 'bearer', 'header': 'Authorization: Bearer <session_token>',
            'credentials_from': 'POST /board-games response players',
            'scope': 'One episode and one player; use that player session_token for every player route.',
        },
        'motion_ownership': 'Only the active player may move its owned hand; the player function catalog is authoritative.',
        'ordinary_episode_routes': 'Ordinary episode control, functions and robot endpoints are forbidden for board games. '
                                   'Episode images/depth media remain available with a valid player token. '
                                   'There are no player /actions, /robot or /robot/ik routes.',
    })
    for row in result['interfaces']:
        row['available'] = False
        if 'supported_policy_ids' in row:
            row['supported_policy_ids'] = []
        row['unavailable_reason'] = 'The legacy agent.model_client.runner does not support authenticated board-game sessions; use public_functions.'
    policies = [p['policy_id'] for p in result['geometry_policies']
                if p['policy_id'] == 'classical_fingertip_bimanual']
    result['interfaces'].append({
        'id': 'public_functions', 'native_track': 'system0', 'control_mode': 'fingertip',
        'interaction_mode': 'functions', 'available': bool(policies), 'backend_available': bool(policies),
        'supported_policy_ids': sorted(set(policies)), 'client_robot_scope': 'authenticated_board_game_players',
        'unavailable_reason': None if policies else 'No compatible board-game geometry executor is installed.',
        'client_command': 'python -m agent.model_client.function_runner --project-root <source_snapshot> '
                          '--bench-url <bench_url> --task <task_id> --output <new_dataset_directory>',
        'function_catalog_endpoint': 'GET ' + player + '/functions',
        'function_call_endpoint': call,
        'scope': 'Existing public-function client with separate player sessions; availability is not native manipulation qualification.',
    })
    result['availability_scope'] = ('This selected release requires authenticated board-game player routes. '
        'Legacy runner modes are unavailable; public_functions describes the existing function_runner. '
        'Per-player function catalogs and current turn permissions govern each call.')
    return result


def interface_catalog(policy_descriptors=()):
    descriptors = tuple(policy_descriptors)
    geometry = [p.model_dump(mode='json') for p in descriptors
                if p.policy_id in ('classical_fingertip', 'classical_fingertip_bimanual', 'classical_fingertip_instances')
                and {'move_geometry', 'hold'} <= set(p.skills)]
    geometry_ids = sorted({p['policy_id'] for p in geometry})
    client_geometry_ids = [name for name in geometry_ids if name != 'classical_fingertip_instances']
    single_ids = [name for name in geometry_ids if name == 'classical_fingertip']
    # The current runner explicitly restricts dual robots to fingertip/action.
    # Discovery of native robot capabilities must not claim other client modes.
    only_dual = bool(descriptors) and all(p.policy_id in ('classical_fingertip_bimanual', 'classical_fingertip_instances') for p in descriptors)
    common = {
        'observations': ['public_rgbd', 'joint_proprioception', 'robot_fk', 'declared_tactile_proxy',
                         'primitive_feedback', 'static_task_goal'],
        'tactile_scope': ('simulated pad normal-contact proxy; no hardware taxels, shear, cloth forces or object identities. '
                          'Zero pad readings do not rule out non-pad hand collisions; positive readings may include self-contact.'),
        'contact_history_scope': ('When supplied in primitive feedback, robot_rigid_contact_history_v1 (single hand), v2 (left/right bimanual) or v3 (named instances) summarizes the entire acknowledged action: '
                                  'pad/shell/palm/arm contact timing, pad coactivity, and self-contact. '
                                  'Multi-hand v2/v3 add per-hand and cross-side summaries with verified native sampling coverage. '
                                  'No object identities, directions, cloth forces, or grasp-success inference. '
                                  'Missing/invalid history is unknown. It follows the tactile-input ablation.'),
        'action_endpoint': 'POST /episodes/{episode_id}/actions',
        'function_catalog_endpoint': 'GET /episodes/{episode_id}/functions',
        'function_call_endpoint': 'POST /episodes/{episode_id}/call',
        'function_schema_endpoint': 'GET /functions/schema',
        'function_capability_version': '1.0.0',
        'reference': None,
        'robot_guide_endpoint': 'GET /episodes/{episode_id}/robot',
        'ik_query_endpoint': 'POST /episodes/{episode_id}/robot/ik',
        'ik_query_scope': 'Read-only numerical robot IK from a published observation; no object oracle, collision planning or execution guarantee. Availability is reported by the episode robot guide.',
        'hold': {'unit': 'simulation_seconds', 'range_s': [.02, 5.], 'tick_s': .02,
                 'advances_physics': True, 'counts_as_primitive': True},
        'observation_advances_physics': False,
        'completion': 'Action completion is not task success; the independent evaluator scores the physical episode.',
        'limits': 'The negotiated episode capability and budget contracts are authoritative.',
    }
    rows = []
    for name, control, interaction in (
        ('joints_action', 'joints', 'action'), ('fingertip_action', 'fingertip', 'action'),
        ('fingertip_tools', 'fingertip', 'tools'), ('fingertip_code', 'fingertip', 'code')):
        joints = control == 'joints'
        supported = client_geometry_ids if interaction == 'action' and not joints else single_ids
        backend_available = True if joints else bool(geometry)
        available = not only_dual if joints else bool(supported)
        rows.append({
            'id': name, 'control_mode': control, 'interaction_mode': interaction,
            'native_track': 'direct_joints' if joints else 'system0',
            'available': available,
            'backend_available': backend_available,
            'backend_policy_ids': [] if joints else geometry_ids,
            'supported_policy_ids': [] if joints else supported,
            'client_robot_scope': 'single_or_bimanual' if not joints and interaction == 'action' else 'single_arm',
            'unavailable_reason': (None if available else
                'Current agent runner does not support this mode for the selected multi-hand robot.' if backend_available else
                'No compatible geometry executor is installed.'),
            'executor': ('validated_joint_targets' if joints else
                         geometry_ids[0] if len(geometry_ids) == 1 else 'classical_fingertip'),
            'model_actions': ['joint_delta', 'hold', 'stop'] if joints else ['move_geometry', 'hold', 'stop'],
            'tools': [] if interaction == 'action' else ['observe', 'point_at', 'move_geometry', 'hold', 'stop'],
            'model_transport': 'JSON text parsed by the benchmark client; no provider-native function calling required',
            'client_command': f'python -m agent.model_client.runner --control-mode {control} --interaction-mode {interaction}',
            'control_prior': ('Agent chooses advertised active joint deltas; the current client supports single-arm robots.' if joints else
                              'Agent chooses wrist and fingertip geometry; the fixed classical controller solves IK and tracks it.'),
            'program_scope': ('Restricted interpreter; bounded loops and public tool callbacks only; no Python imports, '
                              'filesystem, network or simulator access.' if interaction == 'code' else None),
        })
    return {
        'protocol': 'agent_interface_catalog_v1', 'interfaces': rows, 'shared_contract': common,
        'geometry_policies': geometry,
        'availability_scope': ('backend_available describes the bench execution primitive; available and supported_policy_ids '
                               'describe the existing agent.model_client runner. Select a task before discovering its controller.'),
        'comparison': {
            'condition': 'Fix one interface for an episode. Pin task, initial state, assets, sensors, executor, model and budgets.',
            'recorded_costs': ['model_decisions', 'primitive_attempts', 'tool_calls', 'simulation_time', 'tokens', 'wall_time'],
            'scope': 'Joint and geometric modes have different control priors; results compare complete system interfaces.',
            'mixed_interface_episode_supported': False,
        },
        'unavailable': ['object_pose_oracle', 'automatic_object_id_grasp', 'object_state_writeback',
                        'particle_attachment', 'trained_rl_system0', 'live_manus_hardware'],
    }


def turn_game_interface_catalog(catalog):
    """Advertise the generic player HTTP API, without claiming client support."""
    result = board_game_interface_catalog(catalog)
    def paths(value):
        if isinstance(value, dict):
            return {k: paths(v) for k, v in value.items()}
        if isinstance(value, list):
            return [paths(v) for v in value]
        if isinstance(value, str):
            return value.replace('/board-games', '/turn-games').replace('board-game', 'turn-game').replace('board games', 'turn games')
        return value
    result = paths(result)
    policies = [p['policy_id'] for p in result['geometry_policies']
                if p['policy_id'] == 'classical_fingertip_instances']
    row = result['interfaces'][-1]
    row.update(available=False, backend_available=bool(policies), supported_policy_ids=policies,
               client_robot_scope='authenticated_turn_game_players', client_command=None,
               unavailable_reason='Use the advertised authenticated HTTP functions. The task-aware function_runner supports living_room/tower_extraction; arbitrary turn-game model automation is not advertised.',
               scope='Generic authenticated native execution. Client automation and task success require separate validation.')
    result['availability_scope'] = 'The bench HTTP transport is implemented. Task-aware client support is selected separately; this generic catalog does not advertise arbitrary turn-game model automation.'
    return result
