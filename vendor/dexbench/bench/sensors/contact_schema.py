"""Strict public contract for action-scoped, identity-free contact summaries."""
from typing import Annotated, Literal
import math

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from .contact_history import (BIMANUAL_PROTOCOL, BIMANUAL_SEMANTICS, PAIRS, PROTOCOL,
                              REGIONS, SEMANTICS, SIDES, INSTANCE_PROTOCOL, INSTANCE_SEMANTICS)

Nonnegative = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
Positive = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
Count = Annotated[int, Field(strict=True, ge=0)]


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False, validate_assignment=True)


class ContactTiming(StrictRecord):
    contact_substeps: Count
    current_contact_substeps: Count
    longest_contact_substeps: Count

    @model_validator(mode='after')
    def ordered_counts(self):
        if not self.current_contact_substeps <= self.longest_contact_substeps <= self.contact_substeps:
            raise ValueError('Contact duration counters are inconsistent')
        if (self.contact_substeps > 0) != (self.longest_contact_substeps > 0):
            raise ValueError('Positive contact requires a positive contiguous interval')
        return self


class ContactRegion(ContactTiming):
    normal_impulse_sum_ns: Nonnegative

    @model_validator(mode='after')
    def consistent_impulse(self):
        if (self.contact_substeps > 0) != (self.normal_impulse_sum_ns > 0):
            raise ValueError('Positive solved impulse and contact counts must agree')
        return self


class ContactCoverage(StrictRecord):
    all_declared_robot_rigid_links: Annotated[bool, Field(strict=True)]
    cloth: Literal[False]
    contact_directions: Literal[False]
    taxels: Literal[False]
    force_calibrated: Literal[False]
    completeness: Literal['assumed_from_native_step_contract']

    @model_validator(mode='before')
    @classmethod
    def strict_flags(cls, value):
        if isinstance(value, dict):
            for key in ('cloth', 'contact_directions', 'taxels', 'force_calibrated'):
                if key in value and value[key] is not False:
                    raise ValueError('Unavailable measurement flags must be literal false')
        return value


class ContactHistoryObservation(StrictRecord):
    protocol: Literal['robot_rigid_contact_history_v1'] = PROTOCOL
    status: Literal['available', 'not_sampled', 'invalid', 'unavailable']
    reason: Literal['malformed_observation', 'noncontiguous_acknowledged_window',
                    'invalid_rigid_contact_window', 'unmapped_robot_contact',
                    'nonfinite_contact_aggregate'] | None = None
    action_id: Annotated[str, Field(pattern=r'^[A-Za-z0-9_.:-]{1,160}$')] | None = None
    start_sim_time_s: Nonnegative
    sim_time_s: Nonnegative
    duration_s: Nonnegative
    physics_dt_s: Positive | None
    observed_control_windows: Count
    sampled_substeps: Count
    coverage: ContactCoverage
    regions: Annotated[dict[str, ContactRegion], Field(max_length=len(REGIONS))] | None
    pad_coactivity: Annotated[dict[str, ContactTiming], Field(max_length=len(PAIRS))] | None
    self_contact: ContactRegion | None
    semantics: Literal['positive solved normal impulses; coactivity does not imply opposition or a common object; each unique pair counted once per involved region']

    @model_validator(mode='after')
    def consistent_history(self):
        if (self.sim_time_s < self.start_sim_time_s or not math.isclose(
                self.sim_time_s - self.start_sim_time_s, self.duration_s, rel_tol=0., abs_tol=1e-8)):
            raise ValueError('Action contact history clock mismatch')
        if self.coverage.all_declared_robot_rigid_links != (self.status != 'unavailable'):
            raise ValueError('Contact availability and configured coverage disagree')
        if (self.reason is not None) != (self.status == 'invalid'):
            raise ValueError('Invalid history needs a bounded failure reason')
        # Invalid histories can retain a verified prefix, but that prefix must
        # still describe possible native samples. A later gap cannot authorize
        # invented counts or erase the configured sampling interval.
        if self.status != 'unavailable':
            if self.physics_dt_s is None:
                raise ValueError('Configured contact history requires its native timestep')
            per_window = round(.02 / self.physics_dt_s)
            if (not 1 <= per_window <= 10000
                    or not math.isclose(per_window * self.physics_dt_s, .02, rel_tol=0., abs_tol=1e-9)
                    or self.sampled_substeps != self.observed_control_windows * per_window
                    or self.sampled_substeps * self.physics_dt_s > self.duration_s + 1e-8):
                raise ValueError('Observed contact prefix has inconsistent sample counts')
        if self.status != 'available':
            if any(value is not None for value in (self.regions, self.pad_coactivity, self.self_contact)):
                raise ValueError('Missing contact evidence cannot publish measured zeros')
            if self.status in ('not_sampled', 'unavailable') and (self.sampled_substeps or self.observed_control_windows):
                raise ValueError('Unsampled history cannot claim samples')
            if self.status == 'not_sampled' and self.duration_s != 0.:
                raise ValueError('Unsampled action history must remain at its start time')
            return self
        if (self.physics_dt_s is None or self.observed_control_windows <= 0
                or self.sampled_substeps <= 0 or self.duration_s <= 0
                or self.regions is None or set(self.regions) != set(REGIONS)
                or self.pad_coactivity is None or set(self.pad_coactivity) != set(PAIRS)
                or self.self_contact is None):
            raise ValueError('Available contact history requires complete regions and timing')
        if (not math.isclose(self.sampled_substeps * self.physics_dt_s, self.duration_s, rel_tol=0., abs_tol=1e-8)
                or not math.isclose(self.observed_control_windows * .02, self.duration_s, rel_tol=0., abs_tol=1e-8)):
            raise ValueError('Contact sample counts must span the action interval')
        for counter in [*self.regions.values(), *self.pad_coactivity.values(), self.self_contact]:
            if counter.contact_substeps > self.sampled_substeps:
                raise ValueError('Contact counter exceeds sampled substeps')
        for pair, counter in self.pad_coactivity.items():
            for finger in pair.split('+'):
                pad = self.regions[finger + '_pad']
                if (counter.contact_substeps > pad.contact_substeps
                        or counter.current_contact_substeps > pad.current_contact_substeps
                        or counter.longest_contact_substeps > pad.longest_contact_substeps):
                    raise ValueError('Coactivity cannot exceed either pad contact stream')
        return self


class HandContactHistory(StrictRecord):
    regions: Annotated[dict[str, ContactRegion], Field(max_length=len(REGIONS))]
    pad_coactivity: Annotated[dict[str, ContactTiming], Field(max_length=len(PAIRS))]
    self_contact: ContactRegion


class BimanualContactCoverage(ContactCoverage):
    completeness: Literal['verified_native_sampling_clock']


MAX_BIMANUAL_CONTACT_HISTORY_BYTES = 16384


class BimanualContactHistoryObservation(StrictRecord):
    """Two hand summaries share one complete acknowledged action interval.

    Same-side self contact and cross-side robot contact partition the global
    self-contact impulses. Overlapping substep counts must never be summed.
    """
    protocol: Literal['robot_rigid_contact_history_v2'] = BIMANUAL_PROTOCOL
    status: Literal['available', 'not_sampled', 'invalid', 'unavailable']
    reason: Literal['malformed_observation', 'noncontiguous_acknowledged_window',
                    'invalid_rigid_contact_window', 'unmapped_robot_contact',
                    'nonfinite_contact_aggregate'] | None = None
    action_id: Annotated[str, Field(pattern=r'^[A-Za-z0-9_.:-]{1,160}$')] | None = None
    start_sim_time_s: Nonnegative
    sim_time_s: Nonnegative
    duration_s: Nonnegative
    physics_dt_s: Positive | None
    observed_control_windows: Count
    sampled_substeps: Count
    coverage: BimanualContactCoverage
    hands: Annotated[dict[Literal['left', 'right'], HandContactHistory], Field(max_length=2)] | None
    self_contact: ContactRegion | None
    cross_side_contact: ContactRegion | None
    semantics: Literal[BIMANUAL_SEMANTICS]

    def side_names(self):
        return SIDES

    @model_validator(mode='after')
    def consistent_history(self):
        sides = self.side_names()
        if self.status == 'available':
            if (self.hands is None or set(self.hands) != set(sides)
                    or self.self_contact is None or self.cross_side_contact is None):
                raise ValueError('Available bimanual history requires both hands and robot self-contact counters')
        elif self.hands is not None or self.self_contact is not None or self.cross_side_contact is not None:
            raise ValueError('Missing bimanual contact evidence cannot publish measured zeros')
        # Reuse v1's clock, counter, impulse and pad-coactivity consistency rules
        # without changing the v1 serialized contract or accepting v2 as v1.
        common = self.model_dump(exclude={'hands', 'self_contact', 'cross_side_contact', 'instance_names'})
        common.update(protocol=PROTOCOL, semantics=SEMANTICS)
        common['coverage']['completeness'] = 'assumed_from_native_step_contract'
        for side in sides:
            values = self.hands[side].model_dump() if self.hands is not None else {
                'regions': None, 'pad_coactivity': None, 'self_contact': None}
            ContactHistoryObservation.model_validate({**common, **values})
        if self.status == 'available':
            parts = [self.hands[side].self_contact for side in sides] + [self.cross_side_contact]
            for key in ('contact_substeps', 'current_contact_substeps', 'longest_contact_substeps'):
                total = getattr(self.self_contact, key)
                if total > self.sampled_substeps or any(getattr(part, key) > total for part in parts):
                    raise ValueError('Robot self-contact union cannot be smaller than an included stream')
            if self.self_contact.contact_substeps > sum(part.contact_substeps for part in parts):
                raise ValueError('Robot self-contact union cannot exceed its partitioned samples')
            if not math.isclose(self.self_contact.normal_impulse_sum_ns,
                    math.fsum(part.normal_impulse_sum_ns for part in parts), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('Robot self-contact impulses must partition into same-side and cross-side pairs')
            for hand in self.hands.values():
                bound = math.fsum(region.normal_impulse_sum_ns for region in hand.regions.values())
                # With more than two instances the global cross counter also
                # contains contacts between other hands. It is not a per-hand load.
                demand = hand.self_contact.normal_impulse_sum_ns
                if len(sides) == 2:
                    demand += self.cross_side_contact.normal_impulse_sum_ns
                if demand > bound and not math.isclose(demand, bound, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError('Robot self-contact impulses require participating hand regions')
        if len(self.model_dump_json().encode()) > MAX_BIMANUAL_CONTACT_HISTORY_BYTES * max(1, len(sides) / 2):
            raise ValueError('Bimanual contact history exceeds the public payload limit')
        return self


Namespace = Annotated[str, Field(pattern=r'^[a-z][a-z0-9_]{0,31}$')]


class InstanceContactHistoryObservation(BimanualContactHistoryObservation):
    """The same sampling contract for explicitly named robot instances."""
    protocol: Literal['robot_rigid_contact_history_v3'] = INSTANCE_PROTOCOL
    instance_names: list[Namespace] = Field(min_length=2, max_length=8)
    hands: Annotated[dict[Namespace, HandContactHistory], Field(max_length=8)] | None
    semantics: Literal[INSTANCE_SEMANTICS]

    def side_names(self):
        if len(set(self.instance_names)) != len(self.instance_names):
            raise ValueError('Instance namespaces must be unique')
        return tuple(self.instance_names)


ContactHistoryValue = Annotated[
    ContactHistoryObservation | BimanualContactHistoryObservation | InstanceContactHistoryObservation, Field(discriminator='protocol')]
_HISTORY_ADAPTER = TypeAdapter(ContactHistoryValue)


def parse_contact_history(value):
    """Validate either version without interpreting unknown future protocols."""
    return _HISTORY_ADAPTER.validate_python(value)
