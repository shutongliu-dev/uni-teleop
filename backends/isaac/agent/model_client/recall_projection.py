"""Explicit compact public history with exact, player-scoped record recall."""
from copy import deepcopy
import json

from .context_projection import STATIC_FIELDS, project_context


VERSION = 'current_with_recall.v1'
GUIDANCE = (
    'The current context, current RGB-D and last_feedback are complete. Older observation '
    'and depth blocks are replaced by explicit decision references; they are not current state. '
    'Use read_decision with your own completed decision_index and sections observation, execution '
    'or depth_summary to recover exact original public records. Assistant decisions, public game '
    'events, execution errors, diagnostics and full public contact/execution summaries remain in '
    'order. A previous read_decision result is represented by its receipt; call read_decision '
    'again if its original values are needed. No record or event is deleted from the archive.')


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def project_recall_history(history, context, archive):
    """Current data is untouched; only documented historical slots are projected."""
    reference = context.get('task', {})
    exact_history, _, exact_receipt = project_context(history, context)
    changes = [row for row in exact_receipt['removed'] if row['path'].startswith('history/')]

    def static(value, path):
        if isinstance(value, dict):
            for key in STATIC_FIELDS:
                if key in value and key in reference and _json(value[key]) == _json(reference[key]):
                    del value[key]
                    changes.append({'path': path+'/'+key, 'same_as': 'current/task/'+key})

    def is_observation(value):
        return (isinstance(value, dict) and value.get('protocol') == 'agent_bench_v1'
                and all(key in value for key in ('episode_id', 'sequence', 'sim_time_s', 'proprioception', 'cameras')))

    def observation_ref(ref, observation, section='observation'):
        identity = {key: deepcopy(ref[key]) for key in ('decision_index', 'player_id', 'action_id')}
        identity.update(observation_sequence=observation.get('sequence'), sim_time_s=observation.get('sim_time_s'),
                        record=deepcopy(ref['records'][section]))
        return {'historical_observation': identity,
                'omitted': f'Original observation block; read_decision sections=[{section}] restores it.'}

    def compact_feedback(value, ref, path):
        result = deepcopy(value)
        if not isinstance(result, dict):
            return result
        payload = result.get('result')
        if result.get('function') == 'read_decision' and isinstance(payload, dict) and payload.get('historical') is True:
            payload = deepcopy(payload)
            for section, record in payload.get('records', {}).items():
                if isinstance(record, dict) and 'value' in record:
                    del record['value']
                    changes.append({'path': path+'/result/records/'+section+'/value',
                                    'read_decision': payload.get('decision_index'), 'section': section})
            payload['omitted'] = 'Earlier recall values; receipt retained. Repeat read_decision to retrieve the same originals.'
            result['result'] = payload
        elif isinstance(payload, dict):
            static(payload, path+'/result')
            # These exact public shapes are observations; unrelated result fields
            # and every error/diagnostic/contact field are left intact.
            if is_observation(payload):
                result['result'] = observation_ref(ref, payload, 'execution')
                changes.append({'path': path+'/result', 'read_decision': ref['decision_index'], 'section': 'execution'})
            elif is_observation(payload.get('observation')):
                payload['observation'] = observation_ref(ref, payload['observation'], 'execution')
                changes.append({'path': path+'/result/observation', 'read_decision': ref['decision_index'], 'section': 'execution'})
        return result

    projected, active_index = [], None
    for index, message in enumerate(exact_history):
        item = deepcopy(message)
        if item.get('role') != 'user' or not isinstance(item.get('content'), str):
            projected.append(item)
            continue
        try:
            value = json.loads(item['content'])
        except ValueError:
            projected.append(item)
            continue
        if not isinstance(value, dict):
            projected.append(item)
            continue
        path = f'history/{index}'
        if type(value.get('decision_index')) is int and isinstance(value.get('state'), dict):
            active_index = value['decision_index']
            ref = archive.reference(active_index)
            state = value['state']
            static(state, path+'/state')
            if is_observation(state.get('observation')):
                state['observation'] = observation_ref(ref, state['observation'])
                changes.append({'path': path+'/state/observation', 'read_decision': active_index, 'section': 'observation'})
            if 'depth_observation' in value:
                value['depth_observation'] = {'decision_index': active_index,
                    'source': ref['records']['depth_summary'],
                    'omitted': 'Original depth grid; read_decision sections=[depth_summary]. Current depth interpretation still applies.'}
                changes.append({'path': path+'/depth_observation', 'read_decision': active_index, 'section': 'depth_summary'})
            # Keep any unique historical feedback, including errors. Its original
            # execution record is normally the previous completed decision.
            if isinstance(value.get('last_feedback'), dict):
                previous = next((r for r in reversed(archive.index()) if r['decision_index'] < active_index), ref)
                value['last_feedback'] = compact_feedback(value['last_feedback'], previous, path+'/last_feedback')
        elif set(value) == {'tool_feedback'} and active_index is not None:
            value['tool_feedback'] = compact_feedback(value['tool_feedback'], archive.reference(active_index), path+'/tool_feedback')
            value['source_decision'] = archive.reference(active_index)
        else:
            # In particular, public_game_events remain byte-for-byte unchanged.
            projected.append(item)
            continue
        item['content'] = json.dumps(value, ensure_ascii=False, allow_nan=False)
        projected.append(item)
    current = deepcopy(context)
    current.setdefault('host_contract', {})['history_projection'] = GUIDANCE
    text_bytes = lambda messages, now: sum(len(m['content'].encode()) for m in messages
        if isinstance(m.get('content'), str)) + len(json.dumps(now, ensure_ascii=False, allow_nan=False).encode())
    return projected, current, {'schema': VERSION, 'changes': changes,
        'original_context_history_text_utf8_bytes': text_bytes(history, context),
        'projected_context_history_text_utf8_bytes': text_bytes(projected, current),
        'scope': 'Current complete context/current images unchanged. All assistant text and public events retained. Original own-player records available through read_decision.'}
