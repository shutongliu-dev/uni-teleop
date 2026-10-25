"""Credential-safe Responses transport and explicitly selected Micu model client.

The caller supplies the provider, model and effort. One HTTP request, no model,
provider or protocol fallback. Micu's documented external-client UA is opt-in.
"""
from __future__ import annotations

import json
import copy
import math
import re
import time
from dataclasses import replace
from urllib.parse import urlsplit

import httpx

from .transport import ModelGatewayConfig, ModelGatewayError, ModelResponse, _usage, _messages, png_data_url

MICU_EXTERNAL_USER_AGENT = 'codex_cli_rs/0.77.0 (external, HexBench)'
MICU_AGENT_USER_AGENT = 'codex_cli_rs/0.77.0 (external, AgentBench)'
MICU_PROTOCOL_REFERENCE = 'https://docs.micuapi.ai/external-compat'
MICU_UA_REFERENCE = 'https://docs.micuapi.ai/external-ua'
MICU_RETRY_POLICY = 'wait_10s_same_endpoint_v2'
MICU_RETRY_DELAY_S = 10.


class ResponsesClient:
    def __init__(self, config: ModelGatewayConfig, *, client=None,
                 user_agent='HexBench/1.0', timeout_s=180.):
        if timeout_s is not None and (type(timeout_s) not in (int, float)
                or not math.isfinite(timeout_s) or not 0 < timeout_s <= 600):
            raise ValueError('timeout_s must be null or positive and at most 600 seconds')
        self.config = config
        self.client = client or httpx.Client(follow_redirects=False, trust_env=False)
        self._owns_client = client is None
        self.user_agent = user_agent
        self.timeout_s = None if timeout_s is None else float(timeout_s)

    def close(self):
        if self._owns_client:
            self.client.close()

    def public_metadata(self):
        return {'base_url': self.config.base_url,
                'endpoint': self.config.base_url + '/responses',
                'protocol': 'responses', 'user_agent': self.user_agent}

    def complete(self, *, model, reasoning_effort, instructions, prompt,
                 images=(), max_output_tokens=2048):
        if not isinstance(model, str) or not model.strip():
            raise ValueError('An explicit model is required')
        if reasoning_effort not in ('low', 'medium', 'high', 'xhigh', 'max', 'ultra'):
            raise ValueError('An explicit supported reasoning effort is required')
        if not instructions or not prompt or type(max_output_tokens) is not int or max_output_tokens < 64:
            raise ValueError('Nonempty input and a valid output bound are required')
        content = [{'type': 'input_text', 'text': prompt}]
        content += [{'type': 'input_image', 'image_url': png_data_url(raw), 'detail': 'high'}
                    for raw in images]
        payload = {'model': model, 'instructions': instructions,
                   'input': [{'role': 'user', 'content': content}],
                   'reasoning': {'effort': reasoning_effort}, 'store': False,
                   'stream': True, 'max_output_tokens': max_output_tokens}
        return self._request(payload, model=model, reasoning_effort=reasoning_effort,
                             max_output_tokens=max_output_tokens, timeout_s=self.timeout_s)

    def generate(self, system_prompt, user_turns, *, model, reasoning_effort,
                 max_output_tokens=2048, timeout_s=None, retry_deadline=None):
        """Use the same validated multimodal history as the ordinary runner."""
        if not isinstance(model, str) or not re.fullmatch(r'[A-Za-z0-9_.:/-]{1,128}', model):
            raise ValueError('An explicit model is required')
        if reasoning_effort not in ('none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultra'):
            raise ValueError('An explicit supported reasoning effort is required')
        if max_output_tokens is not None and (type(max_output_tokens) is not int
                or not 1 <= max_output_tokens <= 32768):
            raise ValueError('max_output_tokens must be null or between 1 and 32768')
        turns = _messages(system_prompt, user_turns)[1:]
        inputs = []
        for turn in turns:
            content = turn['content']
            if isinstance(content, str):
                content = [{'type': 'text', 'text': content}]
            converted = []
            for block in content:
                if block['type'] == 'text':
                    converted.append({'type': 'output_text' if turn['role'] == 'assistant' else 'input_text',
                                      'text': block['text']})
                else:
                    converted.append({'type': 'input_image', 'image_url': block['image_url']['url'],
                                      'detail': block['image_url'].get('detail', 'auto')})
            inputs.append({'role': turn['role'], 'content': converted})
        payload = {'model': model, 'instructions': system_prompt, 'input': inputs,
                   'reasoning': {'effort': reasoning_effort}, 'store': False, 'stream': True}
        if max_output_tokens is not None:
            payload['max_output_tokens'] = max_output_tokens
        budget = self.timeout_s if timeout_s is None else timeout_s
        if budget is not None and (type(budget) not in (int, float) or not math.isfinite(budget)
                or budget <= 0 or (self.timeout_s is not None and budget > self.timeout_s)):
            raise ValueError('per-call timeout must be positive and no larger than client timeout')
        return self._request(payload, model=model, reasoning_effort=reasoning_effort,
                             max_output_tokens=max_output_tokens, timeout_s=budget, retry_deadline=retry_deadline)

    def _request(self, payload, *, model, reasoning_effort, max_output_tokens, timeout_s, retry_deadline=None):
        started = time.monotonic()
        response_value = None
        if retry_deadline is not None:
            remaining = retry_deadline - started
            if remaining <= 0:
                raise ModelGatewayError('Micu recovery deadline expired before sending', attempts=0,
                                        category='request_budget_exhausted')
            timeout_s = remaining if timeout_s is None else min(timeout_s, remaining)

        def consume_event(lines):
            """One parser for blank-line events and the final EOF event."""
            nonlocal response_value
            value = '\n'.join(lines)
            if value == '[DONE]':
                return
            event = json.loads(value)
            if not isinstance(event, dict) or not isinstance(event.get('type'), str):
                raise ModelGatewayError('Responses stream event has invalid structure', attempts=1,
                                        category='invalid_response')
            kind = event['type']
            if kind in ('response.failed', 'error'):
                raise ModelGatewayError('Responses stream reported failure', attempts=1,
                                        category='invalid_response')
            if kind == 'response.incomplete':
                incomplete = event.get('response')
                if (isinstance(incomplete, dict) and isinstance(incomplete.get('incomplete_details'), dict)
                        and incomplete['incomplete_details'].get('reason') == 'max_output_tokens'):
                    raise ModelGatewayError('Responses output token budget exhausted', attempts=1,
                                            category='output_budget_exhausted')
                raise ModelGatewayError('Responses completion was incomplete', attempts=1,
                                        category='invalid_finish_reason')
            if kind == 'response.completed':
                if response_value is not None or not isinstance(event.get('response'), dict):
                    raise ModelGatewayError('Responses completion event has invalid structure', attempts=1,
                                            category='invalid_response')
                response_value = event['response']

        try:
            with self.client.stream(
                    'POST', self.config.base_url + '/responses', json=payload,
                    headers={'Authorization': 'Bearer ' + self.config.api_key,
                             'User-Agent': self.user_agent, 'Accept': 'text/event-stream'},
                    timeout=httpx.Timeout(timeout_s, connect=30. if timeout_s is None else min(30., timeout_s))) as reply:
                if reply.status_code != 200:
                    raise ModelGatewayError('Responses gateway rejected the request',
                                            status_code=reply.status_code, attempts=1, category='http_status')
                # Check raw chunks too: iter_lines/read alone can hide a peer
                # trickling a partial line/body past the total request deadline.
                original_stream = reply.stream
                class TimedStream(httpx.SyncByteStream):
                    def __iter__(self):
                        for chunk in original_stream:
                            if timeout_s is not None and time.monotonic() - started >= timeout_s:
                                raise ModelGatewayError('Responses total time budget exhausted', attempts=1,
                                                        category='transport_timeout')
                            yield chunk
                    def close(self):
                        original_stream.close()
                reply.stream = TimedStream()
                if 'text/event-stream' not in reply.headers.get('content-type', ''):
                    response_value = json.loads(reply.read())
                else:
                    event_data = []
                    for line in reply.iter_lines():
                        if timeout_s is not None and time.monotonic() - started > timeout_s:
                            raise ModelGatewayError('Responses total time budget exhausted', attempts=1,
                                                    category='transport_timeout')
                        if line.startswith('data:'):
                            event_data.append(line[5:].lstrip())
                        if not line and event_data:
                            consume_event(event_data)
                            event_data = []
                    if event_data:
                        consume_event(event_data)
        except httpx.TimeoutException:
            raise ModelGatewayError('Responses request timed out', attempts=1,
                                    category='transport_timeout') from None
        except httpx.HTTPError:
            raise ModelGatewayError('Responses transport failed', attempts=1,
                                    category='transport_error') from None
        except (ValueError, UnicodeError):
            raise ModelGatewayError('Responses gateway returned invalid JSON', attempts=1,
                                    category='invalid_json') from None
        if timeout_s is not None and time.monotonic() - started > timeout_s:
            raise ModelGatewayError('Responses total time budget exhausted', attempts=1,
                                    category='transport_timeout')
        if not isinstance(response_value, dict):
            raise ModelGatewayError('Responses completion event is missing', attempts=1,
                                    category='invalid_response')
        if response_value.get('model') != model:
            raise ModelGatewayError('Responses gateway returned a different model', attempts=1,
                                    category='model_mismatch')
        if response_value.get('status') != 'completed':
            details = response_value.get('incomplete_details')
            if (response_value.get('status') == 'incomplete' and isinstance(details, dict)
                    and details.get('reason') == 'max_output_tokens'):
                raise ModelGatewayError('Responses output token budget exhausted', attempts=1,
                                        category='output_budget_exhausted')
            raise ModelGatewayError('Responses completion was incomplete', attempts=1,
                                    category='invalid_finish_reason')
        output = response_value.get('output')
        if not isinstance(output, list):
            raise ModelGatewayError('Responses output has invalid structure', attempts=1,
                                    category='invalid_response')
        texts = []
        for item in output:
            if not isinstance(item, dict) or not isinstance(item.get('type'), str):
                raise ModelGatewayError('Responses output item has invalid structure', attempts=1,
                                        category='invalid_response')
            if item['type'] != 'message':
                continue
            content = item.get('content')
            if not isinstance(content, list):
                raise ModelGatewayError('Responses message content has invalid structure', attempts=1,
                                        category='invalid_response')
            for block in content:
                if not isinstance(block, dict) or not isinstance(block.get('type'), str):
                    raise ModelGatewayError('Responses content block has invalid structure', attempts=1,
                                            category='invalid_response')
                if block['type'] == 'output_text':
                    if not isinstance(block.get('text'), str):
                        raise ModelGatewayError('Responses text has invalid structure', attempts=1,
                                                category='invalid_response')
                    texts.append(block['text'])
        text = '\n'.join(texts).strip()
        if not text:
            raise ModelGatewayError('Responses text is empty', attempts=1, category='empty_response')
        # A malicious/provider-echoed credential never enters a caller's recording.
        text = text.replace(self.config.api_key, '[redacted]')
        return ModelResponse(text=text, model=model, usage=_usage(response_value.get('usage'), secret=self.config.api_key),
                             request_id=None, duration_s=time.monotonic() - started, attempts=1,
                             finish_reason='completed', requested_model=model,
                             max_output_tokens=max_output_tokens, reasoning_effort=reasoning_effort,
                             protocol='responses')


class MicuModelClient:
    """Runner-compatible Micu client that waits through transient API outages.

    Only model requests are retried here. Physical actions use a separate ledger.
    Successful usage describes the returned response; usage for interrupted
    attempts is unknown and is never manufactured.
    """

    def __init__(self, config, model, *, timeout_s=300., max_retries=None,
                 reasoning_effort='max', transport=None):
        hostname = urlsplit(config.base_url).hostname or ''
        if not (hostname == 'micuapi.ai' or hostname.endswith('.micuapi.ai')):
            raise ValueError('Micu selection requires the explicitly supplied Micu endpoint')
        if not isinstance(model, str) or not re.fullmatch(r'[A-Za-z0-9_.:/-]{1,128}', model):
            raise ValueError('an explicit model name is required')
        if reasoning_effort not in ('none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultra'):
            raise ValueError('unsupported reasoning_effort')
        if max_retries is not None and (type(max_retries) is not int or not 0 <= max_retries <= 3):
            raise ValueError('max_retries must be null or an integer from 0 to 3')
        self.model, self.reasoning_effort = model, reasoning_effort
        self.max_retries, self.timeout_s = max_retries, timeout_s
        self._http = httpx.Client(transport=transport, trust_env=False, follow_redirects=False)
        try:
            self._responses = ResponsesClient(config, client=self._http,
                user_agent=MICU_AGENT_USER_AGENT, timeout_s=timeout_s)
        except Exception:
            self._http.close()
            raise

    def public_metadata(self):
        return {**self._responses.public_metadata(), 'provider': 'micu', 'model': self.model,
                'reasoning_effort': self.reasoning_effort, 'requested_reasoning_effort': self.reasoning_effort,
                'timeout_s': self.timeout_s, 'max_retries': self.max_retries,
                'retry_policy': MICU_RETRY_POLICY, 'retry_delay_s': MICU_RETRY_DELAY_S,
                'timeout_scope': 'per_attempt'}

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def generate(self, system_prompt, user_turns, *, max_output_tokens=2048,
                 timeout_s=None, reasoning_effort=None, retry_deadline=None, on_event=None):
        """Retry the same decision; only the caller's absolute wall deadline ends waiting.

        ``timeout_s`` caps each attempt, not the whole recovery period. With no
        retry count or wall deadline, transient failures wait until recovery or
        cancellation. Interrupts propagate; no physical command is sent here.
        """
        budget = self.timeout_s if timeout_s is None else timeout_s
        if budget is not None and (type(budget) not in (int, float) or not math.isfinite(budget)
                or budget <= 0 or (self.timeout_s is not None and budget > self.timeout_s)):
            raise ValueError('per-call timeout must be positive and no larger than client timeout')
        if retry_deadline is not None and (type(retry_deadline) not in (int, float)
                or not math.isfinite(retry_deadline)):
            raise ValueError('retry_deadline must be null or a finite monotonic timestamp')
        # Keep the decision input fixed even while a host retry observer runs.
        user_turns = copy.deepcopy(user_turns)
        started = time.monotonic()
        attempt = 0

        def emit(event, **fields):
            if on_event is not None:
                on_event({'event': event, 'attempts': attempt,
                          'retry_policy': MICU_RETRY_POLICY,
                          'elapsed_s': time.monotonic() - started, **fields})

        while True:
            remaining = None if retry_deadline is None else retry_deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise ModelGatewayError('Micu recovery wall deadline exhausted', attempts=attempt,
                                        category='request_budget_exhausted')
            attempt += 1
            emit('model_request_started')
            try:
                # The host event writer can itself consume the last wall time.
                remaining = None if retry_deadline is None else retry_deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise ModelGatewayError('Micu recovery deadline expired before sending', attempts=0,
                                            category='request_budget_exhausted')
                attempt_timeout = budget if remaining is None else remaining if budget is None else min(budget, remaining)
                value = self._responses.generate(system_prompt, user_turns, model=self.model,
                    reasoning_effort=self.reasoning_effort if reasoning_effort is None else reasoning_effort,
                    max_output_tokens=max_output_tokens, timeout_s=attempt_timeout, retry_deadline=retry_deadline)
                if retry_deadline is not None and time.monotonic() >= retry_deadline:
                    raise ModelGatewayError('Micu recovery wall deadline exhausted', attempts=attempt,
                                            category='request_budget_exhausted')
                emit('model_request_completed')
                return replace(value, attempts=attempt, duration_s=time.monotonic() - started)
            except ModelGatewayError as exc:
                if exc.attempts == 0:
                    emit('model_request_not_sent')
                    attempt -= 1
                emit('model_request_failed', **{k: v for k, v in exc.public_metadata().items() if k != 'attempts'})
                retryable = (exc.category in ('transport_timeout', 'transport_error') or
                    (exc.category == 'http_status' and type(exc.status_code) is int
                     and (exc.status_code in (408, 429) or 500 <= exc.status_code <= 599)))
                if not retryable or (self.max_retries is not None and attempt > self.max_retries):
                    raise ModelGatewayError('Micu model request did not complete', status_code=exc.status_code,
                                            attempts=attempt, category=exc.category) from None
                remaining = None if retry_deadline is None else retry_deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise ModelGatewayError('Micu recovery wall deadline exhausted', attempts=attempt,
                                            category='request_budget_exhausted') from None
                delay = MICU_RETRY_DELAY_S if remaining is None else min(MICU_RETRY_DELAY_S, remaining)
                emit('model_retry_wait', retry_delay_s=delay,
                     **{k: v for k, v in exc.public_metadata().items() if k != 'attempts'})
                # Recording time also consumes the wall budget. A shortened
                # wait only reaches that deadline; it never starts an early retry.
                if retry_deadline is not None:
                    delay = min(delay, max(0., retry_deadline - time.monotonic()))
                if delay:
                    waiting = time.monotonic()
                    try:
                        time.sleep(delay)
                    finally:
                        emit('model_retry_wait_finished', waited_s=time.monotonic() - waiting)
