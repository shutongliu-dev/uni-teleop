"""Small synchronous Chat Completions transport; no provider or model fallback.

Credentials stay outside recordings. The caller owns prompt/response recording and
must use ``public_metadata`` instead of serializing client/config objects.
"""
from __future__ import annotations

import base64
import copy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import math
import os
from pathlib import Path
import re
import random
import time
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

import httpx

from bench.paths import load_json


GATEWAY_RETRY_POLICY = 'gateway_transient_backoff_v1'


def gateway_retry_metadata(max_retries=None):
    """Public, immutable policy inputs; null retries means wait until cancellation/deadline."""
    if max_retries is not None and (type(max_retries) is not int or not 0 <= max_retries < 2**31-1):
        raise ValueError('max_retries must be null or a nonnegative bounded integer')
    return {'retry_policy': GATEWAY_RETRY_POLICY, 'max_retries': max_retries,
            'retry_initial_delay_s': 5., 'retry_max_delay_s': 60.,
            'retry_jitter': 'equal_jitter', 'retry_after_max_s': 86400.,
            'retryable_http_statuses': [429, 502, 503, 504],
            'retryable_transport_errors': ['ConnectError', 'ConnectTimeout'],
            'timeout_scope': 'per_attempt_with_absolute_deadline'}


def _retry_after(response):
    """Read only bounded server scheduling advice; never return raw headers."""
    value = response.headers.get('retry-after')
    if value is None:
        return None, 'missing'
    if len(value) > 128:
        return None, 'invalid'
    value = value.strip()
    try:
        if re.fullmatch(r'[0-9]+', value):
            delay, source = float(value), 'delta_seconds'
        else:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            delay, source = max(0., date.timestamp()-time.time()), 'http_date'
    except (ValueError, TypeError, OverflowError):
        return None, 'invalid'
    if not math.isfinite(delay) or delay > 86400.:
        # Do not retry earlier than an otherwise valid, unsupported long wait.
        return None, 'out_of_range'
    return delay, source


def _base_url(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("base_url is required")
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("base_url must be an HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain credentials, query, or fragment")
    if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("remote model credentials require HTTPS")
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        path = path.removesuffix("/chat/completions")
    if not path.endswith("/v1"):
        path += "/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


@dataclass(frozen=True)
class ModelGatewayConfig:
    base_url: str
    api_key: str = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", _base_url(self.base_url))
        if not isinstance(self.api_key, str) or not self.api_key.strip():
            raise ValueError("api_key is required")
        if any(char.isspace() for char in self.api_key):
            raise ValueError("api_key must not contain whitespace")

    @classmethod
    def from_file(cls, path: str | Path) -> "ModelGatewayConfig":
        try:
            config_path = Path(path).expanduser().resolve()
            value = load_json(config_path)
        except (OSError, ValueError, RuntimeError):
            raise ValueError("could not read model credential JSON") from None
        if not isinstance(value, dict):
            raise ValueError("model credential JSON must contain an object")
        if ("api_key" in value) == ("api_key_file" in value):
            raise ValueError("provide exactly one of api_key or api_key_file")
        if "api_key_file" in value:
            reference = value["api_key_file"]
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError("api_key_file must be a non-empty path")
            try:
                key_path = Path(reference).expanduser()
                # Resolve relative references beside the real configuration file,
                # never against the caller's current working directory.
                if not key_path.is_absolute():
                    key_path = config_path.parent / key_path
                if not key_path.is_file():
                    raise ValueError("key is not a regular file")
                key = key_path.read_text().strip()
            except (OSError, ValueError, RuntimeError):
                # Paths and decoding/OS errors can contain secret material.
                raise ValueError("could not read api_key_file") from None
        else:
            key = value["api_key"]
        return cls(base_url=value.get("base_url", ""), api_key=key)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "ModelGatewayConfig":
        env = os.environ if environ is None else environ
        return cls(base_url=env.get("DEX_AGENT_BASE_URL", ""), api_key=env.get("DEX_AGENT_API_KEY", ""))

    @property
    def endpoint(self) -> str:
        return self.base_url + "/chat/completions"

    def public_metadata(self) -> dict[str, str]:
        return {"base_url": self.base_url, "endpoint": self.endpoint, "protocol": "chat/completions"}


def png_data_url(data: bytes) -> str:
    if not isinstance(data, bytes) or not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("image must be PNG bytes")
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _redact(value: Any, secret: str, limit: int = 1024) -> str:
    # Replace the exact key before truncation, then common key/header forms.
    result = str(value).replace(secret, "[redacted]")
    result = re.sub(r"(?i)bearer\s+[^\s\"',}]+", "Bearer [redacted]", result)
    result = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}", "[redacted]", result)
    return result[:limit]


MODEL_ERROR_CATEGORIES = frozenset({
    'unknown', 'request_budget_exhausted', 'transport_timeout', 'transport_error',
    'http_status', 'invalid_json', 'invalid_response', 'empty_response',
    'model_mismatch', 'invalid_finish_reason',
    'output_budget_exhausted',
})


class ModelGatewayError(RuntimeError):
    """Typed failure; archival metadata excludes free text and provider identifiers."""

    def __init__(self, message: str, *, status_code: int | None = None,
                 request_id: str | None = None, attempts: int = 0, category: str = 'unknown',
                 retry_metadata: dict[str, Any] | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id
        self.attempts = attempts
        self.category = category
        self.retry_metadata = copy.deepcopy(retry_metadata)

    def public_metadata(self) -> dict[str, Any]:
        # Never archive str(self), HTTP bodies/headers, exception chains, or request IDs.
        # Only a finite vocabulary and bounded numeric fields cross this boundary.
        category = self.category if isinstance(self.category, str) and self.category in MODEL_ERROR_CATEGORIES else 'unknown'
        status = self.status_code if type(self.status_code) is int and 100 <= self.status_code <= 599 else None
        attempts = self.attempts if type(self.attempts) is int and 0 <= self.attempts <= 2**31 - 1 else None
        return {"error_category": category, "http_status": status, "attempts": attempts,
                **copy.deepcopy(self.retry_metadata or {})}


@dataclass(frozen=True)
class ModelResponse:
    text: str = field(repr=False)
    model: str
    usage: dict[str, Any]
    request_id: str | None
    duration_s: float
    attempts: int
    finish_reason: str | None
    requested_model: str
    max_output_tokens: int | None
    reasoning_effort: str
    protocol: str = "chat/completions"
    retry_metadata: dict[str, Any] | None = None

    def public_metadata(self) -> dict[str, Any]:
        return {"model": self.model, "requested_model": self.requested_model,
                "usage": copy.deepcopy(self.usage), "request_id": self.request_id,
                "duration_s": self.duration_s, "attempts": self.attempts,
                "finish_reason": self.finish_reason, "max_output_tokens": self.max_output_tokens,
                "reasoning_effort": self.reasoning_effort,
                "requested_reasoning_effort": self.reasoning_effort,
                "reasoning_effort_verified_by_server": False,
                "protocol": self.protocol, "http_status": 200,
                **copy.deepcopy(self.retry_metadata or {})}


def _usage(value: Any, *, secret: str | None = None) -> dict[str, Any]:
    # Provider usage extensions are retained only when numeric; no text/headers.
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not re.fullmatch(r"[a-zA-Z0-9_]{1,64}", key):
            continue
        if secret and secret in key:
            continue
        if isinstance(item, dict):
            result[key] = _usage(item, secret=secret)
        elif type(item) in {int, float} and math.isfinite(item) and item >= 0:
            result[key] = item
    return result


def _messages(system_prompt: str, user_turns: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("system_prompt is required")
    if not user_turns or isinstance(user_turns, (str, bytes)):
        raise ValueError("at least one user turn is required")
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for turn in user_turns:
        if not isinstance(turn, dict) or set(turn) != {"role", "content"}:
            raise ValueError("turn must contain exactly role and content")
        role, content = turn["role"], turn["content"]
        if role not in {"user", "assistant"}:
            raise ValueError("only user and assistant turns are accepted")
        if isinstance(content, str):
            if not content.strip():
                raise ValueError("turn content must not be empty")
        elif isinstance(content, list) and content:
            if role != "user":
                raise ValueError("assistant history must be plain text")
            for block in content:
                if not isinstance(block, dict):
                    raise ValueError("invalid multimodal block")
                if block.get("type") == "text":
                    if set(block) != {"type", "text"} or not isinstance(block["text"], str):
                        raise ValueError("invalid text block")
                elif block.get("type") == "image_url":
                    item = block.get("image_url")
                    if set(block) != {"type", "image_url"} or not isinstance(item, dict):
                        raise ValueError("invalid image block")
                    if not set(item) <= {"url", "detail"} or not isinstance(item.get("url"), str):
                        raise ValueError("invalid image URL")
                    # Inline frames prevent external URL access and mutable image references.
                    prefix = "data:image/png;base64,"
                    if not item["url"].startswith(prefix):
                        raise ValueError("images must be inline PNG data URLs")
                    try:
                        decoded = base64.b64decode(item["url"][len(prefix):], validate=True)
                    except ValueError:
                        raise ValueError("invalid PNG base64") from None
                    if not decoded.startswith(b"\x89PNG\r\n\x1a\n"):
                        raise ValueError("image URL does not contain PNG data")
                    if item.get("detail", "auto") not in {"auto", "low", "high"}:
                        raise ValueError("invalid image detail")
                else:
                    raise ValueError("unsupported multimodal block")
        else:
            raise ValueError("turn content must be text or multimodal blocks")
        messages.append(copy.deepcopy(turn))
    if messages[-1]["role"] != "user":
        raise ValueError("last turn must be a user observation")
    return messages


class ModelClient:
    def __init__(self, config: ModelGatewayConfig, model: str, *, timeout_s: float | None = 120.0,
                 max_retries: int | None = 1, reasoning_effort: str = "max",
                 retry_policy: str = 'legacy_gateway_v1',
                 transport: httpx.BaseTransport | None = None):
        if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,128}", model):
            raise ValueError("an explicit model name is required")
        if timeout_s is not None and (type(timeout_s) not in (int, float)
                or not math.isfinite(timeout_s) or not 0 < timeout_s <= 600):
            raise ValueError("timeout_s must be null or positive and at most 600 seconds")
        if retry_policy not in ('legacy_gateway_v1', GATEWAY_RETRY_POLICY):
            raise ValueError('Unknown gateway retry policy')
        if retry_policy == GATEWAY_RETRY_POLICY:
            gateway_retry_metadata(max_retries)
        elif type(max_retries) is not int or not 0 <= max_retries <= 3:
            raise ValueError("max_retries must be an integer from 0 to 3")
        if reasoning_effort not in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("unsupported reasoning_effort")
        self.config = config
        self.model = model
        self.timeout_s = None if timeout_s is None else float(timeout_s)
        self.max_retries = max_retries
        self.retry_policy = retry_policy
        self.reasoning_effort = reasoning_effort
        self._client = httpx.Client(transport=transport, follow_redirects=False, trust_env=False)

    def __repr__(self) -> str:
        return f"ModelClient(model={self.model!r}, endpoint={self.config.endpoint!r})"

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ModelClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def public_metadata(self) -> dict[str, Any]:
        return {**self.config.public_metadata(), "model": self.model,
                "timeout_s": self.timeout_s, "max_retries": self.max_retries,
                "reasoning_effort": self.reasoning_effort,
                "requested_reasoning_effort": self.reasoning_effort,
                **(gateway_retry_metadata(self.max_retries) if self.retry_policy == GATEWAY_RETRY_POLICY else {})}

    def _generate_transient(self, payload, *, budget, max_output_tokens, effort, retry_deadline, on_event):
        if retry_deadline is not None and (type(retry_deadline) not in (int, float)
                or not math.isfinite(retry_deadline)):
            raise ValueError('retry_deadline must be null or a finite monotonic timestamp')
        policy = gateway_retry_metadata(self.max_retries)
        started, attempt = time.monotonic(), 0

        def emit(event, **fields):
            if on_event is not None:
                on_event({'event': event, 'attempts': attempt, 'retry_policy': GATEWAY_RETRY_POLICY,
                          'elapsed_s': time.monotonic()-started,
                          'time_utc': datetime.now(timezone.utc).isoformat(), **fields})

        def expired():
            return retry_deadline is not None and time.monotonic() >= retry_deadline

        def exhausted():
            return ModelGatewayError('Gateway recovery wall deadline exhausted', attempts=attempt,
                category='request_budget_exhausted', retry_metadata=policy)

        while True:
            if expired():
                raise exhausted()
            attempt += 1
            emit('model_request_started')
            # Writing the durable event can consume the remaining wall budget.
            if expired():
                emit('model_request_not_sent')
                attempt -= 1
                raise exhausted()
            remaining = None if retry_deadline is None else retry_deadline-time.monotonic()
            timeout = budget if remaining is None else remaining if budget is None else min(budget, remaining)
            response, retryable, retry_after_s, retry_after_source = None, False, None, 'missing'
            try:
                response = self._client.post(self.config.endpoint, json=payload,
                    headers={'Authorization': 'Bearer '+self.config.api_key}, timeout=timeout)
            except httpx.HTTPError as transport_error:
                retryable = isinstance(transport_error, (httpx.ConnectError, httpx.ConnectTimeout))
                error = ModelGatewayError('Gateway transport did not complete', attempts=attempt,
                    category='transport_timeout' if isinstance(transport_error, httpx.TimeoutException) else 'transport_error',
                    retry_metadata=policy)
            else:
                if response.status_code == 200:
                    try:
                        result = self._decode_response(response, request_id=None, attempt=attempt,
                            start=started, max_output_tokens=max_output_tokens, effort=effort)
                    except ModelGatewayError as error:
                        error.retry_metadata = copy.deepcopy(policy)
                        emit('model_request_failed', **{k: v for k, v in error.public_metadata().items()
                            if k not in ('attempts', 'retry_policy')})
                        raise
                    # Even a response received after the deadline consumed one
                    # request. Do not execute its decision after the wall limit.
                    if expired():
                        error = exhausted()
                        emit('model_request_failed', error_category=error.category, http_status=200)
                        raise error
                    emit('model_request_completed', http_status=200)
                    return replace(result, retry_metadata=policy)
                retryable = response.status_code in policy['retryable_http_statuses']
                retry_after_s, retry_after_source = _retry_after(response)
                error = ModelGatewayError('Gateway returned an unsuccessful HTTP status',
                    status_code=response.status_code, attempts=attempt, category='http_status', retry_metadata=policy)
            emit('model_request_failed', error_category=error.category, http_status=error.status_code,
                 retry_after_s=retry_after_s, retry_after_source=retry_after_source)
            if (not retryable or retry_after_source == 'out_of_range'
                    or (self.max_retries is not None and attempt > self.max_retries)):
                raise error from None
            if expired():
                raise exhausted() from None
            ceiling = min(policy['retry_max_delay_s'], policy['retry_initial_delay_s']*2**min(attempt-1, 4))
            backoff = random.uniform(ceiling/2., ceiling)
            delay = max(backoff, retry_after_s or 0.)
            if retry_deadline is not None:
                delay = min(delay, max(0., retry_deadline-time.monotonic()))
            emit('model_retry_wait', retry_delay_s=delay, backoff_delay_s=backoff,
                 retry_after_s=retry_after_s, retry_after_source=retry_after_source,
                 error_category=error.category, http_status=error.status_code)
            # A shortened wait reaches the deadline; it never sends an early retry.
            if retry_deadline is not None:
                delay = min(delay, max(0., retry_deadline-time.monotonic()))
            waiting = time.monotonic()
            try:
                if delay:
                    time.sleep(delay)
            finally:
                emit('model_retry_wait_finished', waited_s=time.monotonic()-waiting)

    def _decode_response(self, response, *, request_id, attempt, start, max_output_tokens, effort):
        if request_id is None:
            request_id = response.headers.get('x-request-id') or response.headers.get('request-id')
            if request_id is not None:
                request_id = _redact(request_id, self.config.api_key, 256)
        try:
            data = response.json()
        except ValueError:
            raise ModelGatewayError("model gateway returned invalid JSON", status_code=200,
                request_id=request_id, attempts=attempt, category='invalid_json') from None
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"]
            returned_model = data["model"]
            finish_reason = choice.get("finish_reason")
            if not isinstance(text, str) or not isinstance(returned_model, str):
                raise ValueError("invalid response field types")
        except (ValueError, KeyError, IndexError, TypeError):
            raise ModelGatewayError("model gateway returned an invalid response shape",
                status_code=200, request_id=request_id, attempts=attempt, category='invalid_response') from None
        if not text.strip():
            raise ModelGatewayError("model gateway returned an empty assistant response",
                status_code=200, request_id=request_id, attempts=attempt, category='empty_response')
        if returned_model != self.model:
            raise ModelGatewayError("model gateway returned a different model",
                status_code=200, request_id=request_id, attempts=attempt, category='model_mismatch')
        if finish_reason is not None and (not isinstance(finish_reason, str) or finish_reason not in {"stop", "length", "content_filter"}):
            raise ModelGatewayError("model gateway returned an unsupported completion finish reason",
                status_code=200, request_id=request_id, attempts=attempt, category='invalid_finish_reason')
        if request_id is None and isinstance(data.get("id"), str):
            request_id = _redact(data["id"], self.config.api_key, 256)
        return ModelResponse(text=text, model=returned_model, usage=_usage(data.get("usage"), secret=self.config.api_key),
            request_id=request_id, duration_s=time.monotonic()-start, attempts=attempt,
            finish_reason=finish_reason, requested_model=self.model, max_output_tokens=max_output_tokens,
            reasoning_effort=effort)

    def generate(self, system_prompt: str, user_turns: Sequence[dict[str, Any]], *,
                 max_output_tokens: int | None = 2048, timeout_s: float | None = None,
                 reasoning_effort: str | None = None, retry_deadline: float | None = None,
                 on_event=None) -> ModelResponse:
        if max_output_tokens is not None and (type(max_output_tokens) is not int
                or not 1 <= max_output_tokens <= 32768):
            raise ValueError("max_output_tokens must be null or an integer from 1 to 32768")
        budget = self.timeout_s if timeout_s is None else timeout_s
        if budget is not None and (type(budget) not in (int, float) or not math.isfinite(budget)
                or budget <= 0 or (self.timeout_s is not None and budget > self.timeout_s)):
            raise ValueError("per-call timeout must be positive and no larger than client timeout")
        effort = self.reasoning_effort if reasoning_effort is None else reasoning_effort
        if effort not in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("unsupported reasoning_effort")
        payload = {"model": self.model, "messages": _messages(system_prompt, user_turns),
                   "stream": False, "reasoning_effort": effort}
        if max_output_tokens is not None:
            payload["max_completion_tokens"] = max_output_tokens
        if self.retry_policy == GATEWAY_RETRY_POLICY:
            return self._generate_transient(copy.deepcopy(payload), budget=budget,
                max_output_tokens=max_output_tokens, effort=effort,
                retry_deadline=retry_deadline, on_event=on_event)
        if retry_deadline is not None or on_event is not None:
            raise ValueError('Gateway event/deadline recovery requires an explicit retry policy')
        start = time.monotonic()
        for attempt in range(1, self.max_retries + 2):
            remaining = None if budget is None else budget - (time.monotonic() - start)
            if remaining is not None and remaining <= 0:
                raise ModelGatewayError("model request time budget exhausted", attempts=attempt - 1,
                                        category='request_budget_exhausted')
            try:
                response = self._client.post(self.config.endpoint, json=payload,
                    headers={"Authorization": "Bearer " + self.config.api_key}, timeout=remaining)
            except httpx.HTTPError as error:
                # Never chain httpx exceptions: they retain request headers including the key.
                retryable = isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout))
                if (retryable and attempt <= self.max_retries
                        and (budget is None or time.monotonic() - start < budget)):
                    continue
                raise ModelGatewayError(f"model transport failed: {type(error).__name__}",
                                        attempts=attempt, category=('transport_timeout' if isinstance(error, httpx.TimeoutException)
                                                                    else 'transport_error')) from None
            request_id = response.headers.get("x-request-id") or response.headers.get("request-id")
            if request_id is not None:
                request_id = _redact(request_id, self.config.api_key, 256)
            if response.status_code != 200:
                if response.status_code in {429, 502, 503, 504} and attempt <= self.max_retries:
                    delay = 1.0 if budget is None else min(1.0, max(0.0, budget - (time.monotonic() - start)))
                    if delay > 0:
                        time.sleep(delay)
                        continue
                # Preserve useful error detail after redaction; no complete response/headers.
                detail = _redact(response.text, self.config.api_key)
                raise ModelGatewayError(f"model gateway HTTP {response.status_code}: {detail}",
                    status_code=response.status_code, request_id=request_id, attempts=attempt, category='http_status')
            return self._decode_response(response, request_id=request_id, attempt=attempt,
                start=start, max_output_tokens=max_output_tokens, effort=effort)
        raise AssertionError("unreachable retry state")
