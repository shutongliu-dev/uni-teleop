"""Explicit Qwen3.8 Responses binding; one request, no provider fallback.

The documented highest effort is xhigh. Responses does not support a separate
thinking_budget; max_output_tokens includes reasoning and final-answer tokens.
Sources were checked on 2026-10-28. Provider internals are not verified here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from urllib.parse import urlsplit

import httpx

from .responses import ResponsesClient
from .transport import ModelResponse

QWEN_BASE_URL = 'https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1'
QWEN_RETRY_POLICY = 'qwen_single_request_v1'
QWEN_API_REFERENCE = 'https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-responses'
QWEN_MODEL_REFERENCE = 'https://www.qianwenai.com/models/qwen3.8-max-0902'
QWEN_MODELS = ('qwen3.8-max-0902', 'qwen3.8-max')
QWEN_EFFORTS = {'none': 'none', 'minimal': 'low', 'low': 'low',
                'medium': 'medium', 'high': 'xhigh', 'xhigh': 'xhigh', 'max': 'xhigh'}


def is_qwen_endpoint(base_url):
    """Recognize Qianwen credentials so another provider cannot consume them."""
    hostname = urlsplit(base_url).hostname or ''
    return hostname == 'qianwenaiapi.com' or hostname.endswith('.qianwenaiapi.com')


def effective_effort(value):
    if not isinstance(value, str) or value not in QWEN_EFFORTS:
        raise ValueError('Qwen3.8 supports none/minimal/low/medium/high/xhigh/max; highest is xhigh')
    return QWEN_EFFORTS[value]


def request_parameters(effort, max_output_tokens=None):
    result = {'reasoning': {'effort': effective_effort(effort)},
              'store': False, 'stream': True}
    if max_output_tokens is not None:
        result['max_output_tokens'] = max_output_tokens
    return result


@dataclass(frozen=True)
class QwenModelResponse(ModelResponse):
    def public_metadata(self):
        return {**super().public_metadata(), 'provider': 'qwen',
                'effective_reasoning_effort': effective_effort(self.reasoning_effort),
                'request_parameters': request_parameters(self.reasoning_effort, self.max_output_tokens),
                'output_token_scope': 'reasoning_and_final_answer',
                'reasoning_summary_recorded': False}


class QwenModelClient:
    """Runner-compatible client for the explicitly selected Qianwen endpoint.

Only message.output_text enters action/history parsing. Reasoning summaries and
provider IDs are not recorded. The shared transport validates both JSON and SSE
completion, checks the exact requested model and sanitizes credential echoes.
"""

    def __init__(self, config, model, *, timeout_s=300., max_retries=0,
                 reasoning_effort='xhigh', transport=None):
        if config.base_url != QWEN_BASE_URL:
            raise ValueError('Qwen selection requires the explicitly supported Qianwen Responses endpoint')
        if model not in QWEN_MODELS:
            raise ValueError('Qwen binding requires an explicitly supported Qwen3.8-Max model')
        effective_effort(reasoning_effort)
        if type(max_retries) is not int or max_retries != 0:
            raise ValueError('Qwen uses one request; model_retries must be 0')
        self.model, self.reasoning_effort = model, reasoning_effort
        self.timeout_s, self.max_retries = timeout_s, max_retries
        self._http = httpx.Client(transport=transport, trust_env=False, follow_redirects=False)
        try:
            self._responses = ResponsesClient(config, client=self._http,
                user_agent='AgentBench/1.0', timeout_s=timeout_s)
        except Exception:
            self._http.close()
            raise

    def public_metadata(self):
        return {**self._responses.public_metadata(), 'provider': 'qwen', 'model': self.model,
                'reasoning_effort': self.reasoning_effort,
                'requested_reasoning_effort': self.reasoning_effort,
                'effective_reasoning_effort': effective_effort(self.reasoning_effort),
                'reasoning_effort_verified_by_server': False,
                'request_parameters': request_parameters(self.reasoning_effort),
                'request_parameters_scope': 'fixed settings; output bound supplied per generate call',
                'thinking_budget_supported': False,
                'output_token_scope': 'reasoning_and_final_answer',
                'reasoning_summary_recorded': False,
                'timeout_s': self.timeout_s, 'max_retries': 0,
                'retry_policy': QWEN_RETRY_POLICY, 'timeout_scope': 'single_request',
                'api_reference': QWEN_API_REFERENCE, 'model_reference': QWEN_MODEL_REFERENCE}

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def generate(self, system_prompt, user_turns, *, max_output_tokens=None,
                 timeout_s=None, reasoning_effort=None):
        effort = self.reasoning_effort if reasoning_effort is None else reasoning_effort
        canonical = effective_effort(effort)
        if max_output_tokens is not None and (type(max_output_tokens) is not int
                or not 16 <= max_output_tokens <= 32768):
            raise ValueError('Qwen max_output_tokens must be null or between 16 and 32768')
        value = self._responses.generate(system_prompt, user_turns, model=self.model,
            reasoning_effort=canonical, max_output_tokens=max_output_tokens, timeout_s=timeout_s)
        # The request records its user's label as well as the exact wire mapping.
        value = replace(value, reasoning_effort=effort)
        return QwenModelResponse(**vars(value))
