"""In-process trusted System0 plugin boundary; no client-controlled imports."""
from __future__ import annotations

from typing import Callable, Literal, Protocol, runtime_checkable

from pydantic import Field, TypeAdapter, model_validator

from .schemas import (ActionDiagnostic, AgentObservation, Contract, EpisodeSpec, PolicyDescriptor,
                      Name, RobotCapabilities, RobotJointCommand, RobotExecutionState, System0Action)


class PolicyStep(Contract):
    command: RobotJointCommand | None = None
    status: Literal["running", "completed", "failed"]
    # Private diagnostic only. The HTTP adapter must publish a fixed status enum,
    # never this text: a privileged teacher can accidentally reveal state here.
    detail: str = Field(default="", max_length=1024)
    diagnostic: ActionDiagnostic | None = None
    robot_execution_state: RobotExecutionState | None = None

    @model_validator(mode="after")
    def running_command(self):
        if self.status == "running" and self.command is None:
            raise ValueError("running policy must emit a joint command")
        if self.status != "running" and self.command is not None:
            raise ValueError("terminal policy status must not emit another command")
        return self


@runtime_checkable
class System0Policy(Protocol):
    """Implement these hooks in the authors' policy adapter.

    A policy receives only projected observations and capabilities. The adapter
    must not be given a native simulator handle. Every output is revalidated by
    the runtime before env.step(). Status completed refers to the current skill,
    never the hidden task evaluator. No RL weights are supplied by this package.

    Stateful adapters may additionally implement commit_pending_step() and
    discard_pending_step(). The service commits only after a successful native
    step reply, and discards every unexecuted candidate (including cancellation
    and the final check-only step). Such adapters must not advance their durable
    reference merely because step() returned a candidate command.
    """
    descriptor: PolicyDescriptor

    def reset(self, spec: EpisodeSpec, capabilities: RobotCapabilities) -> None: ...
    def begin(self, action: System0Action, observation: AgentObservation) -> None: ...
    def step(self, observation: AgentObservation) -> PolicyStep: ...
    def cancel(self, reason: str) -> None: ...


class PolicyRegistry:
    def __init__(self):
        self._policies: dict[tuple[str, str], System0Policy] = {}
        self._descriptors: dict[tuple[str, str], PolicyDescriptor] = {}
        self._task_factories: dict[tuple[str, str], dict[str, tuple[Callable[[], System0Policy], PolicyDescriptor]]] = {}

    def register(self, policy: System0Policy) -> None:
        if not isinstance(policy, System0Policy):
            raise TypeError("policy must implement System0Policy")
        key = (policy.descriptor.policy_id, policy.descriptor.version)
        if key in self._policies or key in self._task_factories:
            raise ValueError("policy ID/version is already registered; replace with a new version")
        self._policies[key] = policy
        self._descriptors[key] = policy.descriptor.model_copy(deep=True)

    def register_task_factory(self, task_id: str, descriptor: PolicyDescriptor,
                              factory: Callable[[], System0Policy]) -> None:
        """Operator-only registration of a zero-argument, task-bound constructor.

        The caller pins the complete descriptor, including implementation/config
        hashes. No client object, scene payload, module name or import path is
        forwarded to the factory. Every constructed policy is checked again.
        """
        task_id = TypeAdapter(Name).validate_python(task_id, strict=True)
        if not callable(factory):
            raise TypeError('policy factory must be an operator-provided callable')
        if not isinstance(descriptor, PolicyDescriptor):
            raise TypeError('policy factory requires a pinned PolicyDescriptor')
        frozen = PolicyDescriptor.model_validate(descriptor.model_dump(mode='json'))
        key = (frozen.policy_id, frozen.version)
        if key in self._policies or task_id in self._task_factories.get(key, {}):
            raise ValueError('policy ID/version is already registered for this task')
        self._task_factories.setdefault(key, {})[task_id] = (factory, frozen)

    def resolve(self, policy_id: str, version: str, *, task_id: str | None = None) -> System0Policy:
        key = (policy_id, version)
        if key in self._task_factories:
            try:
                factory, descriptor = self._task_factories[key][task_id]
            except KeyError as exc:
                raise ValueError('policy ID/version is not registered for this task') from exc
            value = factory()
            if not isinstance(value, System0Policy):
                raise ValueError('registered policy factory returned an invalid policy')
            if value.descriptor != descriptor:
                raise ValueError('registered policy factory descriptor changed; register a new version')
            return value
        try:
            value = self._policies[key]
        except KeyError as exc:
            raise ValueError("policy ID/version is not registered on this server") from exc
        if value.descriptor != self._descriptors[key]:
            raise ValueError("registered policy descriptor changed; register a new version")
        return value

    def descriptors(self, *, task_id: str | None = None) -> list[PolicyDescriptor]:
        values = list(self._descriptors.values())
        for bindings in self._task_factories.values():
            if task_id is not None:
                if task_id in bindings:
                    values.append(bindings[task_id][1])
                continue
            # Legacy discovery remains unambiguous when all task configurations
            # produce the same descriptor. Never select an arbitrary task hash.
            descriptors = [descriptor for _, descriptor in bindings.values()]
            if any(descriptor != descriptors[0] for descriptor in descriptors[1:]):
                raise ValueError('Task ID is required for task-bound controller discovery')
            values.append(descriptors[0])
        return [x.model_copy(deep=True) for x in values]
