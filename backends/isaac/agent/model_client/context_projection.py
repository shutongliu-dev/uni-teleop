"""Remove exact public-input repetition without shortening dynamic game history."""
from copy import deepcopy
import json


VERSION = 'public_context_exact_dedup.v1'
STATIC_FIELDS = ('public_rules', 'public_rules_sha256', 'rules', 'geometry')
GUIDANCE = (
    'The current task contains the full static task contract. Static fields omitted '
    'from state or a feedback result are exactly equal to that task field. '
    'last_feedback is omitted only when it exactly repeats the immediately preceding '
    'tool_feedback message. All dynamic states, contacts, events, errors and ordered '
    'tool feedback remain available in this conversation.')


def _json(value, *, canonical=False):
    options = {'ensure_ascii': False, 'allow_nan': False}
    if canonical:
        options.update(sort_keys=True, separators=(',', ':'))
    return json.dumps(value, **options)


def _same(left, right):
    # Python equality conflates True/1 and 1/1.0. Here types must agree too.
    return _json(left, canonical=True) == _json(right, canonical=True)


def project_context(history, context):
    """Return new history/context plus a receipt; never mutate the journal inputs.

    Only explicit state and feedback-result slots are eligible. There is no
    recursive field-name deletion, summarization, windowing, numeric rounding,
    assistant rewriting or event coalescing. A different or missing canonical
    task field remains in place. Intervening events break feedback adjacency.
    """
    reference = context.get('task', {})
    reference = reference if isinstance(reference, dict) else {}
    projected_history = deepcopy(history)
    removed = []

    def static_fields(value, path):
        if not isinstance(value, dict):
            return
        for key in STATIC_FIELDS:
            if key in value and key in reference and _same(value[key], reference[key]):
                del value[key]
                removed.append({'path': path + '/' + key, 'same_as': 'current/task/' + key})

    def feedback(value, path):
        if isinstance(value, dict):
            static_fields(value.get('result'), path + '/result')

    def payload(value, path, previous):
        result = deepcopy(value)
        static_fields(result.get('state'), path + '/state')
        if 'last_feedback' in result:
            if previous is not None and _same(result['last_feedback'], previous[1]):
                del result['last_feedback']
                removed.append({'path': path + '/last_feedback', 'same_as': previous[0]})
            else:
                feedback(result['last_feedback'], path + '/last_feedback')
        if 'tool_feedback' in result:
            feedback(result['tool_feedback'], path + '/tool_feedback')
        return result

    previous = None
    for index, message in enumerate(projected_history):
        if message.get('role') != 'user' or not isinstance(message.get('content'), str):
            previous = None
            continue
        try:
            value = json.loads(message['content'])
        except ValueError:
            previous = None
            continue
        if not isinstance(value, dict):
            previous = None
            continue
        path = f'history/{index}'
        result = payload(value, path, previous)
        if not _same(result, value):
            message['content'] = _json(result)
        previous = (path + '/tool_feedback', value['tool_feedback']) if set(value) == {'tool_feedback'} else None

    projected_context = payload(context, 'current', previous)
    if removed:
        projected_context.setdefault('host_contract', {})['input_projection'] = GUIDANCE

    def text_bytes(messages, current):
        return sum(len(m['content'].encode()) for m in messages if isinstance(m.get('content'), str)) + len(_json(current).encode())

    return projected_history, projected_context, {
        'schema': VERSION, 'removed': removed,
        'original_context_history_text_utf8_bytes': text_bytes(history, context),
        'projected_context_history_text_utf8_bytes': text_bytes(projected_history, projected_context),
        'scope': 'Text only; system prompt and current image blocks unchanged. Dynamic history, contacts, events, errors and feedback order retained.'}
