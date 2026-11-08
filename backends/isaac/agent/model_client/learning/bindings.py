"""Trusted-host bindings for learner libraries; models never receive host paths."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .journal import canonical_json, read_json
from .skills import SkillLibrary


class _Entry(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    library: str = Field(min_length=1, repr=False)
    phase: Literal['practice', 'validation', 'evaluation']
    candidate_event_id: str | None = None
    bundle_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')

    @model_validator(mode='after')
    def pinned_phase(self):
        if self.phase == 'practice':
            if self.candidate_event_id is not None or self.bundle_sha256 is not None:
                raise ValueError('Practice uses published candidates chosen explicitly by the learner')
        elif not self.candidate_event_id or not self.bundle_sha256:
            raise ValueError('Validation and evaluation require an exact candidate and bundle')
        return self


class _Bindings(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    protocol: Literal['agent_skill_bindings.v1']
    bindings: dict[str, _Entry]

    @model_validator(mode='after')
    def identities(self):
        if set(self.bindings) not in ({'single'}, {'player_0', 'player_1'}):
            raise ValueError('Bind exactly one learner or both game players')
        return self


@dataclass(frozen=True)
class SkillBinding:
    library: SkillLibrary = field(repr=False)
    phase: str
    candidate_event_id: str | None = None
    bundle_sha256: str | None = None

    def __post_init__(self):
        _Entry(library='host-owned', phase=self.phase,
               candidate_event_id=self.candidate_event_id, bundle_sha256=self.bundle_sha256)
        if self.candidate_event_id is not None:
            actual = self.library._candidate(self.candidate_event_id)['payload']['bundle_hash']
            if actual != self.bundle_sha256:
                raise ValueError('Skill binding candidate and bundle differ')
            self.library.verify(actual)

    def check_task(self, selection):
        compatibility = self.library.contract.get('compatibility', {})
        if selection.task_id not in compatibility.get('task_ids', []):
            raise ValueError('Library does not declare compatibility with this task')
        expected = compatibility.get('task_releases', {}).get(selection.task_id)
        actual = (selection.source_release or {}).get('manifest_sha256')
        if not expected or expected != actual:
            raise ValueError('Library task release does not match the selected release')
        if self.phase == 'practice' and not selection.practice_rsi_allowed:
            raise ValueError('This task does not permit practice or skill changes')

    def public_metadata(self):
        return {'protocol': 'agent_skill_binding.v1',
                'learner_id': self.library.experiment['experiment_id'], 'phase': self.phase,
                'candidate_event_id': self.candidate_event_id, 'bundle_sha256': self.bundle_sha256,
                'library_contract_sha256': hashlib.sha256(canonical_json(self.library.contract)).hexdigest(),
                'initial_author_revision': self.library.author_revision() if self.phase == 'practice' else None,
                'skill_mutation_tools_available': self.phase == 'practice',
                'scope': 'Model-authored bounded programs and public robot functions. A pinned bundle is not a claim of validated task success.'}

    def instantiate(self, session):
        from .runtime import SkillRuntime
        return SkillRuntime(self.library, session=session, phase=self.phase,
            journal_path=session.output / 'host' / 'skill_runtime_journal',
            frozen_candidate_event_id=self.candidate_event_id)


def load_skill_bindings(path):
    """Read a host-selected binding once, rejecting unknown fields and links."""
    value = _Bindings.model_validate(read_json(path))
    base = Path(path).absolute().parent
    result = {}
    for key, entry in value.bindings.items():
        library_path = Path(entry.library)
        if not library_path.is_absolute():
            library_path = base / library_path
        result[key] = SkillBinding(SkillLibrary(library_path), entry.phase,
                                   entry.candidate_event_id, entry.bundle_sha256)
    if len({item.library.path for item in result.values()}) != len(result):
        raise ValueError('Different learners require independent libraries')
    return result


def augment_context(context, runtime):
    """Expose only the runtime's public tools; no hidden actions or code injection."""
    tools = runtime.tools()
    names = {item['function'] for item in context['tools'] if 'function' in item}
    if names & {item['function'] for item in tools}:
        raise ValueError('Skill tools conflict with public function names')
    context['tools'].extend(tools)
    context['decision_schema']['one_of'].extend(
        {'function': item['function'], 'arguments': {}} for item in tools)
    context['skill_runtime'] = runtime.public_metadata()
    context['host_contract']['skill_execution'] = (
        'Skills execute a bounded, explicitly documented Python subset through public functions. '
        'Write and publish only in practice. Validation/evaluation use the exact frozen candidate. '
        'A skill completed status is not task success. Inspect actual robot feedback and the referee. '
        'Skill calls share episode action budgets and stop at a game authorization epoch change.')


RSI_PROMPT = '''\nYour own skill library and skill_* tools are available under skill_runtime.
In practice, develop and revise your own reusable skill source using the documented
bounded language. Publish an immutable candidate before executing it. Read your
own execution feedback and revise based on actual observations; task completion
is determined by the Bench, not by a value returned from your skill.
Use your own published source and public learning feedback to carry improvements
across practice episodes; the model conversation and program state start fresh.
Use the available budget to write, publish, execute and revise your own skills.
Your author drafts remain available in practice, but unpublished drafts cannot
enter validation or frozen evaluation and do not change a published candidate.
In validation/evaluation, skill source and reusable parameters are frozen. You
may reason, choose targets, call or compose frozen skills and recover within the
current episode. Do not rewrite skills or carry test feedback into later games.
Independent skill-validation credit requires actually invoking the frozen skill;
direct public function calls alone do not validate a skill bundle.
No operator-authored manipulation solution is supplied by this interface.
'''
