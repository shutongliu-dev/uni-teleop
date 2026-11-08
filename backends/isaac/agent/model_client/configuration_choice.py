"""One model decision over public installation choices, before native startup."""
from copy import deepcopy
import json

from bench.protocol.robot_configurations import checked_public_configurations
from .actions import parse_json_object
from .outcomes import error_record
from .scene_tasks import TaskSetupError
from .transport import ModelGatewayError

PROMPT = '''Choose one robot configuration before this episode starts using only
its published scene description, task requirements and available configurations.
A single arm uses the original right hand; both dual configurations use official
left and right hands controlled by one Agent. Consider the published static base
placements, hand chirality, reference surface, workspace access and the task's
handling needs. This choice fixes the release for the whole episode.
Return exactly {"configuration_id":"<one advertised ID>","reason":"<brief observable reason>"}.
Do not supply paths, object poses, actions or private evaluator state.'''


def choose_configuration(model, published_task, *, max_output_tokens, timeout_s,
                         on_receipt=None, on_event=None):
    try:
        options = checked_public_configurations(published_task.get('robot_configurations'))
    except (KeyError, TypeError, ValueError):
        raise TaskSetupError('configuration_contract_mismatch') from None
    public = {key: deepcopy(published_task[key]) for key in
              ('task_id', 'instruction', 'public_rules', 'public_rules_sha256', 'goal',
               'scoring_contract', 'scene_description_zh') if key in published_task}
    public['robot_configurations'] = options
    public['decision_schema'] = {'configuration_id': {'enum': list(options)},
                                 'reason': {'type': 'string', 'max_length': 1024}}
    messages = [{'role': 'user', 'content': [{'type': 'text',
                 'text': json.dumps(public, ensure_ascii=False, allow_nan=False)}]}]
    receipt = {'system_prompt': PROMPT, 'messages': messages,
               'model_calls': 1, 'selected': None, 'status': 'requesting'}
    def publish():
        if on_receipt is not None:
            on_receipt(deepcopy(receipt))
    publish()
    try:
        kwargs = {'max_output_tokens': max_output_tokens, 'timeout_s': timeout_s}
        if on_event is not None:
            kwargs['on_event'] = on_event
        response = model.generate(PROMPT, messages, **kwargs)
    except (Exception, KeyboardInterrupt) as exc:
        receipt.update(status='request_failed', error=error_record(exc))
        if isinstance(exc, ModelGatewayError):
            receipt['metadata'] = exc.public_metadata()
        publish()
        raise
    receipt.update(response=response.text, metadata=response.public_metadata())
    try:
        decision = parse_json_object(response.text)
        if (set(decision) != {'configuration_id', 'reason'}
                or decision['configuration_id'] not in options
                or not isinstance(decision['reason'], str)
                or not 1 <= len(decision['reason']) <= 1024):
            raise ValueError('Invalid robot configuration decision')
    except (TypeError, ValueError, KeyError):
        receipt['status'] = 'invalid_decision'
        publish()
        return None, receipt
    receipt['selected'] = deepcopy(decision)
    receipt['status'] = 'selected'
    publish()
    return decision['configuration_id'], receipt
