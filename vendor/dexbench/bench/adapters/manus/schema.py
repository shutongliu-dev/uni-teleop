"""Project acquisition protocol; never a MANUS native ABI or robot command."""
from __future__ import annotations
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

Finite = Annotated[float, Field(allow_inf_nan=False)]
Vec3 = Annotated[list[Finite], Field(min_length=3, max_length=3)]
Quat = Annotated[list[Finite], Field(min_length=4, max_length=4)]
Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r'^[A-Za-z0-9_.:/-]+$')]
Ns = Annotated[int, Field(ge=0, strict=True)]

class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

class Pose(StrictModel):
    position: Vec3
    quaternion_wxyz: Quat

    @model_validator(mode='after')
    def unit_quaternion(self):
        if abs(sum(v*v for v in self.quaternion_wxyz)-1) > 1e-3:
            raise ValueError('quaternion must be unit length, ordered wxyz')
        return self

class Node(StrictModel):
    node_id: Identifier
    parent_id: Identifier | None
    semantic: Identifier  # vendor chain/finger-joint enum name, not assumed array index
    pose: Pose
    valid: bool = True

class Tracker(StrictModel):
    tracker_id: Identifier
    status: Literal['tracked', 'orientation_only', 'lost', 'unavailable']
    clock_id: Identifier
    capture_time_ns: Ns
    frame_id: Identifier
    tracked_body_frame: Identifier
    pose: Pose | None = None
    source_timestamp: str | None = Field(default=None, max_length=256)

    @model_validator(mode='after')
    def valid_pose(self):
        if self.status == 'tracked' and self.pose is None:
            raise ValueError('tracked requires a complete 6DoF pose')
        return self

class Device(StrictModel):
    device_id: Identifier
    product: str = Field(min_length=1, max_length=128)
    firmware: str = Field(min_length=1, max_length=64)
    sdk_version: str = Field(min_length=1, max_length=64)
    core_version: str = Field(min_length=1, max_length=64)
    hand: Literal['left', 'right']
    source_kind: Literal['hardware', 'synthetic', 'replay']

class AcquisitionPacket(StrictModel):
    protocol: Literal['manus_acquisition_v1'] = 'manus_acquisition_v1'
    stream_id: Identifier
    sequence: Ns
    clock_id: Identifier
    capture_time_ns: Ns  # SDK callback arrival, NOT sensor exposure time
    source_timestamp: str | None = Field(default=None, max_length=256)
    source_timestamp_encoding: str = Field(default='opaque_vendor', max_length=128)
    device: Device
    calibration_id: Identifier
    topology_id: Identifier
    frame_id: Identifier
    axes: Literal['right_handed_z_up_x_forward'] = 'right_handed_z_up_x_forward'
    position_unit: Literal['m', 'cm', 'mm'] = 'm'
    node_space: Literal['world', 'parent_local'] = 'world'
    glove_tracking_valid: bool
    nodes: Annotated[list[Node], Field(max_length=256)]
    ergonomics_degrees: dict[Identifier, Finite] = Field(default_factory=dict, max_length=128)
    ergonomics_capture_time_ns: Ns | None = None
    ergonomics_source_timestamp: str | None = Field(default=None, max_length=256)
    tracker: Tracker | None = None
    sdk_queue_drops_total: Ns = 0

    @model_validator(mode='after')
    def topology(self):
        nodes = {node.node_id: node for node in self.nodes}
        if len(nodes) != len(self.nodes):
            raise ValueError('duplicate node ids')
        if self.glove_tracking_valid and not nodes:
            raise ValueError('valid glove tracking requires nodes')
        for node in self.nodes:
            seen = {node.node_id}
            parent = node.parent_id
            while parent is not None:
                if parent not in nodes:
                    raise ValueError('missing parent node')
                if parent in seen:
                    raise ValueError('cyclic skeleton topology')
                seen.add(parent)
                parent = nodes[parent].parent_id
        return self

class ClockMapping(StrictModel):
    source_clock_id: Identifier
    target_clock_id: Identifier
    offset_ns: int
    uncertainty_ns: Ns
    method: Literal['same_host_monotonic', 'measured_offset', 'ptp_verified']

class CaptureConfig(StrictModel):
    calibration_id: Identifier
    glove_calibrated: bool = False
    world_calibrated: bool = False
    world_frame_id: Identifier = 'bench_world'
    capture_frame_id: Identifier = 'manus_world'
    # T_world_capture; meters, independent of packet position_unit.
    world_from_capture: Pose = Field(default_factory=lambda: Pose(position=[0, 0, 0], quaternion_wxyz=[1, 0, 0, 0]))
    # T_tracker_wrist; required independently of glove calibration.
    tracker_body_frame: Identifier = 'unassigned_tracker'
    tracker_from_wrist: Pose = Field(default_factory=lambda: Pose(position=[0, 0, 0], quaternion_wxyz=[1, 0, 0, 0]))
    wrist_offset_calibrated: bool = False
    target_clock_id: Identifier
    clock_mappings: list[ClockMapping] = Field(default_factory=list)
    max_sample_age_ns: Ns = 150_000_000
    max_tracker_skew_ns: Ns = 30_000_000
    max_clock_uncertainty_ns: Ns = 5_000_000
    operator_id: Identifier = 'anonymous'
    task_id: Identifier = 'unassigned'
    retarget_version: Identifier = 'unassigned'
    robot_profile_id: Identifier = 'unassigned'
    calibration_artifacts_sha256: dict[Identifier, Annotated[str, Field(pattern=r'^[a-f0-9]{64}$')]] = Field(default_factory=dict, max_length=32)
    retarget_implementation_sha256: Annotated[str, Field(pattern=r'^[a-f0-9]{64}$')] | None = None

    @model_validator(mode='after')
    def distinct_clock_maps(self):
        names = [m.source_clock_id for m in self.clock_mappings]
        if len(set(names)) != len(names):
            raise ValueError('duplicate clock mappings')
        if any(m.target_clock_id != self.target_clock_id for m in self.clock_mappings):
            raise ValueError('clock mapping target mismatch')
        return self
