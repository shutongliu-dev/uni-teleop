"""Trusted task runner. Public observations are projected; native state is private."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import secrets
from pathlib import Path
import shutil
import threading
import time
import uuid

import numpy as np
from pydantic import Field, model_validator

from bench.protocol import (ActionLifecycle, BudgetExceeded, Budgets, CameraObservation,
    DirectJointAction, EpisodeClock, EpisodeRecorder, EpisodeResult, EpisodeSpec,
    JointLimit, PolicyRegistry, RobotCapabilities, RobotJointCommand, parse_action,
    project_observation, project_policy_observation, validate_action, validate_joint_command)
from bench.protocol.schemas import (ActionDiagnostic, Contract, Hash, Name, ObservationProfile, Track,
    RobotExecutionState, MAX_ROBOT_EXECUTION_STATE_BYTES, MEASURED_JOINT_LIMIT_SAFETY_BAND_RAD, DurationPreflight)
from bench.protocol.policies import PolicyStep
from bench.protocol.allocations import (MAX_EPISODE_SIM_TIME_S, MAX_EPISODE_WALL_TIME_S,
                                        MAX_EPISODE_ACTIONS)
from bench.isaacsim.release import DEFAULT, digest, task_inputs, validate
from .native import NativeWorker, PROJECT
from .free_track import AcknowledgedDriveLedger, NativeAcknowledgementError, supports_free_track
from bench.evaluation.physics import audit_native_episode, audit_source_paths, physics_profile_sha256
from bench.evaluation.rigid_safety import RigidContactSafetyGuard
from bench.sensors.tactile import tactile_configuration, project_tactile
from bench.sensors.contact_history import ContactHistory, configure_contact_history
from bench.sensors.contact_schema import parse_contact_history
from bench.protocol.task_referee import project_task_rules, project_referee_state
from bench.protocol.task_scoring import project_terminal_scoring

DEPENDENCIES=('bench/paths.py','bench/isaacsim/release.py','bench/simulation/assets/task_parts.py',
              'scenes/manipulation/__init__.py',
              'scenes/manipulation/evaluator.py','scenes/manipulation/release.py',
              'scenes/manipulation/humanlike.py',
              'scenes/manipulation/contracts.py',
              'scenes/board_games/runtime_release.py',
              'bench/geometry.py','bench/control/kinematics.py','scripts/isaac_python.sh',
              'scripts/maintenance/freeze_project_sources.py',
              'bench/sensors/__init__.py','bench/sensors/tactile.py',
              'bench/sensors/contact_history.py','bench/sensors/contact_schema.py',
              'bench/simulation/isaacsim_backend/camera_calibration.py',
              'bench/simulation/isaacsim_backend/robot_instances.py',
              'bench/simulation/isaacsim_backend/instance_asset.py')


def frontend_sources():
    paths=[p for folder in [PROJECT/'bench/runtime',PROJECT/'bench/protocol',PROJECT/'bench/control',PROJECT/'bench/sdk'] for p in folder.glob('*.py')]
    paths.extend((PROJECT/'scenes/board_games/tabletop').glob('*.py'))
    paths.extend((PROJECT/'scenes/workrooms/living_room').rglob('*.py'))
    paths.extend((PROJECT/'scenes/workrooms/study').rglob('*.py'))
    paths.extend((PROJECT/'scenes/workrooms/laboratory').rglob('*.py'))
    paths.extend((PROJECT/'scenes/workrooms/mechanical_assembly').rglob('*.py'))
    paths.append(PROJECT/'scenes/runtime.py')
    paths.extend(PROJECT/p for p in DEPENDENCIES)
    paths.extend(audit_source_paths())
    return {str(p.relative_to(PROJECT)):digest(p) for p in paths}


IMPORTED_FRONTEND_SOURCES=frontend_sources()

# Direct target interpolation has its own velocity-only timing contract. These
# are command-reference caps, not measured speed or acceleration guarantees.
DIRECT_ARM_VELOCITY_RAD_S = .65
DIRECT_HAND_VELOCITY_RAD_S = 1.


def archive_project_sources(sources, directory, *, project=PROJECT):
    """Copy each pinned project file once, including read-only source files.

    Frontend relative keys and native absolute keys can name the same file.
    Validate every binding before copying; never overwrite a conflicting target.
    Project sources and archive paths must not contain filesystem symlinks.
    """
    project=Path(project).absolute()
    directory=Path(directory).absolute()
    if project.resolve(strict=True)!=project:
        raise RuntimeError('Aliased project source root')
    if any(path.is_symlink() for path in (directory,*directory.parents)):
        raise RuntimeError('Aliased source archive directory')
    selected={}
    for name,expected in sources.items():
        original=Path(name)
        if not original.is_absolute():
            original=project/original
        if not original.is_relative_to(project):
            continue
        relative=original.relative_to(project)
        if '..' in relative.parts:
            raise RuntimeError('Invalid source archive path')
        if (not original.is_file() or original.is_symlink()
                or original.resolve(strict=True)!=original):
            raise RuntimeError('Source must be a regular file without aliases')
        if relative in selected and selected[relative][1]!=expected:
            raise RuntimeError('Conflicting source archive bindings')
        if digest(original)!=expected:
            raise RuntimeError('Source changed during episode setup')
        selected[relative]=(original,expected)
    for relative,(original,expected) in selected.items():
        saved=directory/relative
        if any(path.is_symlink() for path in (saved,*saved.parents)):
            raise RuntimeError('Aliased source archive target')
        saved.parent.mkdir(parents=True,exist_ok=True)
        if saved.exists():
            if not saved.is_file() or digest(saved)!=expected:
                raise RuntimeError('Conflicting source archive target')
        else:
            shutil.copy2(original,saved)
        if digest(saved)!=expected:
            raise RuntimeError('Source changed during episode setup')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sensor_configuration(metadata):
    """Pin the worker's actual sensor contract; old RGB workers remain readable."""
    supplied = metadata.get('agent_sensor_config')
    if supplied is None:
        return {'rgb':['overview','task'],'width':640,'height':480,
                'depth':False,'camera_frame':'optical_x_right_y_down_z_forward',
                'capture':'decision_and_policy_steps'}
    expected = {'depth':True,'depth_encoding':'float32_npy',
                'depth_unit':'m','depth_semantics':'camera_z','depth_invalid_value':0.0,
                'depth_rgb_aligned':True,'camera_frame':'optical_x_right_y_down_z_forward',
                'capture':'decision_and_policy_steps'}
    if not isinstance(supplied,dict) or supplied.get('protocol') not in ('agent_rgbd_v1','agent_rgbd_v2'):
        raise RuntimeError('Worker sensor configuration violates the RGB-D contract')
    per_camera = supplied['protocol'] == 'agent_rgbd_v2'
    dimensions = {'cameras'} if per_camera else {'width','height'}
    if set(supplied) != set(expected) | {'protocol','rgb'} | dimensions:
        raise RuntimeError('Worker sensor configuration violates the RGB-D contract')
    if any(supplied[key] != value for key,value in expected.items()):
        raise RuntimeError('Worker sensor configuration violates the RGB-D contract')
    if supplied['depth'] is not True or supplied['depth_rgb_aligned'] is not True:
        raise RuntimeError('Worker sensor availability and alignment must be explicit booleans')
    if not isinstance(supplied['rgb'],list) or not supplied['rgb'] or any(
            not isinstance(name,str) or not name or '/' in name for name in supplied['rgb']):
        raise RuntimeError('Worker camera inventory is invalid')
    if len(set(supplied['rgb'])) != len(supplied['rgb']):
        raise RuntimeError('Worker camera inventory contains duplicates')
    if per_camera:
        cameras = supplied['cameras']
        if not isinstance(cameras,dict) or set(cameras) != set(supplied['rgb']):
            raise RuntimeError('Worker per-camera configuration must match the camera inventory')
        for config in cameras.values():
            if (not isinstance(config,dict)
                    or set(config) != {'width','height','kind','role','mount_link'}
                    or config['kind'] not in ('world','link')
                    or not isinstance(config['role'],str) or not config['role']
                    or (config['kind'] == 'world' and config['mount_link'] is not None)
                    or (config['kind'] == 'link' and
                        (not isinstance(config['mount_link'],str) or not config['mount_link']))):
                raise RuntimeError('Worker per-camera configuration is invalid')
        dimension_configs = cameras.values()
    else:
        dimension_configs = [supplied]
    if any(type(config[key]) is not int or config[key] <= 0
           for config in dimension_configs for key in ('width','height')):
        raise RuntimeError('Worker camera dimensions are invalid')
    return deepcopy(supplied)


class ServiceError(Exception):
    def __init__(self, status, detail, *, code=None):
        self.status=status
        self.code=code or {403:'permission_denied',404:'not_found',409:'episode_busy',
                          422:'invalid_arguments'}.get(status,'infrastructure_error')
        super().__init__(detail)


class PolicyExecutionError(RuntimeError):
    """A policy fault before physics is distinct from invalid physical evidence."""


class CreateEpisodeRequest(Contract):
    task_id: Name
    robot_id: Name = 'sharpa_w01_right'
    scene_variant: Name = 'original_v1'
    agent_id: Name
    agent_version: Name
    agent_config_sha256: Hash
    seed: int = Field(default=0,strict=True,ge=0,le=2**32-1)
    track: Track = 'direct_joints'
    observation_profile: ObservationProfile = 'vision_proprio'
    budgets: Budgets = Field(default_factory=Budgets)
    policy_id: Name | None = None
    policy_version: Name | None = None
    plan_slot_id: Name | None = None

    @model_validator(mode='after')
    def policy_pair(self):
        if (self.track in ('system0','free_tool_agent')) != (self.policy_id is not None and self.policy_version is not None):
            raise ValueError('System0 requires a registered policy ID and version')
        if self.track=='direct_joints' and (self.policy_id is not None or self.policy_version is not None):
            raise ValueError('Direct control has no policy selector')
        if self.track=='free_tool_agent' and self.policy_id!='classical_fingertip':
            raise ValueError('Free tool track requires the registered classical executor')
        return self


class CreateBoardGameRequest(CreateEpisodeRequest):
    robot_id: Name = 'dual_fr3_sharpa_w01'
    track: Track = 'system0'
    policy_id: Name | None = 'classical_fingertip_bimanual'
    policy_version: Name | None = '1.2.0-board1'


class CreateTurnGameRequest(CreateEpisodeRequest):
    robot_id: Name = 'fr3_sharpa_instances'
    track: Track = 'system0'
    policy_id: Name | None = 'classical_fingertip_instances'
    policy_version: Name | None = '1.2.0-board1'


class EpisodeSession:
    def __init__(self, request, directory, *, release, device, registry, worker_factory,
                 allow_privileged=False, expected_spec=None, board_game=False, turn_game=False):
        if request.observation_profile=='privileged_state' and not allow_privileged:
            raise ServiceError(403,'Privileged observations are disabled on this server')
        if ((request.budgets.sim_time_s is not None and request.budgets.sim_time_s>MAX_EPISODE_SIM_TIME_S)
                or (request.budgets.wall_time_s is not None and request.budgets.wall_time_s>MAX_EPISODE_WALL_TIME_S)
                or request.budgets.max_actions>MAX_EPISODE_ACTIONS):
            raise ServiceError(422,'Requested budget exceeds this server allocation')
        self.lock=threading.RLock()
        self.cancelled=threading.Event()
        self.cancellation_requests=set()
        self.done=threading.Event()
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=False)
        self.result=None
        self.requests={}
        self.direct_hold_replays={}
        self.sequence=0
        self.last_action_id=None
        self.thread=None
        self.policy=None
        self.free_handoff=None
        self.player_session=None
        self.board_game=None
        self.turn_game=None
        self.robot_geometry=None
        self.worker=None
        self.clock=None
        self.life=ActionLifecycle()
        self.public_images={}
        self.policy_images={}
        self.public_depths={}
        self.policy_depths={}
        self.image_revision=0
        self._previous_public_q=None
        setup_start=time.monotonic()
        try:
            from bench.simulation.isaacsim_backend.robot_registry import get_robot_spec, scene_with_hand
            from bench.simulation.isaacsim_backend.scene_variants import VARIANTS
            if request.scene_variant not in VARIANTS:
                raise ServiceError(422,'Unknown scene variant')
            manifest=validate(release)
            selected = next((entry for entry in manifest['tasks'] if entry['id'] == request.task_id), None)
            if selected is None:
                raise ServiceError(422,'Unknown task ID')
            kind = selected.get('task_kind', manifest.get('task_kind'))
            if ((kind == 'board_game') != board_game or (kind == 'turn_game') != turn_game):
                raise ServiceError(403, 'Use the authenticated entrypoint advertised for this task')
            if board_game and (request.robot_id != 'dual_fr3_sharpa_w01' or
                               request.scene_variant != 'original_v1' or request.observation_profile != 'vision_proprio'):
                raise ServiceError(422, 'Board games require the frozen dual robot, original scene and public RGB-D profile')
            placement = manifest.get('schema_version') == 'isaacsim-card-room-placement-release-v1'
            manipulation = manifest.get('schema_version') == 'isaacsim-living-room-manipulation-release-v1'
            kitchen = manifest.get('schema_version') == 'isaacsim-kitchen-manipulation-release-v1'
            laboratory = manifest.get('schema_version') == 'isaacsim-laboratory-manipulation-release-v1'
            study = manifest.get('schema') == 'study_task_release.v1'
            living_single = (manifest.get('schema') == 'living_room_task_release.v1'
                             and kind != 'turn_game')
            mechanical = manifest.get('schema_version') in {
                'isaacsim-mechanical-fastener-release-v1',
                'isaacsim-mechanical-assembly-release-v1',
                'isaacsim-mechanical-nist-release-v1',
            }
            mechanical_nist = manifest.get('schema') == 'mechanical_nist_task_release.v1'
            if (placement or manipulation or kitchen or laboratory or study or mechanical or living_single) and (
                    request.robot_id != 'sharpa_w01_right'
                    or request.scene_variant != 'original_v1'):
                raise ServiceError(422,'This task requires its frozen single-arm robot and original scene')
            entry,scene,task,_=task_inputs(release,request.task_id,
                **({'seed':request.seed} if placement or manipulation or kitchen or laboratory or study or mechanical_nist or manifest.get('schema') == 'living_room_task_release.v1' else {}))
            if turn_game and (request.robot_id != scene.get('robot',{}).get('hand_id')
                              or request.scene_variant != 'original_v1'
                              or request.observation_profile != 'vision_proprio'):
                raise ServiceError(422, 'Turn games require the frozen robot, original scene and public RGB-D profile')
            try:
                hand_spec=get_robot_spec(scene_with_hand(scene,request.robot_id)['robot'])
            except (KeyError,ValueError) as exc:
                raise ServiceError(422,'Unknown or incompatible robot configuration') from exc
            self.public_task = None
            self.referee_contract = None
            self.referee_state = None
            from scenes.runtime import terminal_scoring_contract
            referee_task = project_task_rules(task, scoring_contract=terminal_scoring_contract(task))
            if placement:
                from .placement_tasks import project_placement_task
                self.public_task = project_placement_task(task)
            elif manipulation:
                from .manipulation_tasks import project_manipulation_task
                self.public_task = project_manipulation_task(task)
            else:
                self.public_task = referee_task
            if referee_task is not None:
                if self.public_task != referee_task:
                    raise ValueError('Public task projection differs from its frozen referee contract')
                self.referee_contract = referee_task['public_rules']
            if 'scenario' in scene:
                frozen_seed = scene['scenario'].get('seed')
                if type(frozen_seed) is not int or not 0 <= frozen_seed < 2**32:
                    raise ServiceError(422,'Scenario release has no valid frozen initialization seed')
                if request.seed != frozen_seed:
                    raise ServiceError(422,'Episode seed must match the scenario release initialization seed')
                if request.scene_variant != 'original_v1':
                    raise ServiceError(422,'Scenario releases require their frozen original_v1 scene variant')
            if request.track in ('system0','free_tool_agent'):
                try:
                    self.policy=registry.resolve(request.policy_id,request.policy_version,task_id=entry['id'])
                except ValueError as exc:
                    raise ServiceError(422,'Unknown or changed operator-registered policy for this task') from exc
                if abs(self.policy.descriptor.control_period_s-.02)>1e-9:
                    raise ServiceError(422,'This native runner requires a 0.02 second System0 control period')
            if request.track=='free_tool_agent' and not supports_free_track(self.policy):
                raise ServiceError(422,'Registered policy does not implement acknowledged direct handoff')
            if self.policy:
                supported = self.policy.descriptor.supported_robot_ids
                if supported and request.robot_id not in supported:
                    raise ServiceError(422,'Selected controller does not support this robot')
                if hasattr(self.policy, 'robot_configuration'):
                    from .controllers import robot_configuration
                    if self.policy.robot_configuration != robot_configuration(scene):
                        raise ServiceError(422,'Controller robot configuration differs from this task')
            self.worker=worker_factory(output=self.directory/'private',task_id=entry['id'],hand_id=request.robot_id,
                seed=request.seed,release=release,device=device,scene_variant=request.scene_variant,
                **({'startup_timeout_s':None} if request.budgets.wall_time_s is None else {}))
            if request.budgets.wall_time_s is None and hasattr(self.worker,'set_deadline'):
                self.worker.set_deadline(None)
            initial=self.worker.initial
            if self.public_task is not None and initial.get('task') != self.public_task:
                raise ServiceError(503,'Native worker episode targets differ from the public task')
            self.metadata=initial['metadata']
            self.rigid_safety=RigidContactSafetyGuard(required=self.metadata.get('native_isaacsim_app') is True)
            self.sensor_config=sensor_configuration(self.metadata)
            if hand_spec.get('arms'):
                # Bimanual: one five-channel projection per hand, published as hand_tactile.
                sides={side:tactile_configuration(self.metadata,request.robot_id,side) for side in hand_spec['arms']}
                self.tactile_config={'available':all(c['available'] for c in sides.values()),'sides':sides}
            else:
                self.tactile_config=tactile_configuration(self.metadata,request.robot_id)
            self.sensor_config['tactile']=deepcopy(self.tactile_config)
            contact_config=configure_contact_history(self.metadata,request.robot_id)
            self.contact_history=ContactHistory(contact_config)
            self.sensor_config['contact_history']=deepcopy(contact_config)
            self.sensor_config['policy_camera_capture']=(self.policy.descriptor.camera_observation_mode
                if self.policy else 'decision_only')
            self.raw=initial['observation']
            self.score=initial['score']
            self.terminal_score_receipt = self.score
            if (self.public_task or {}).get('scoring_contract') is not None:
                json.dumps(self.score, allow_nan=False)
            if self.referee_contract is not None:
                self.referee_state = project_referee_state(self.score, self.referee_contract)
                if self.referee_state['terminal']:
                    raise ServiceError(503, 'Task referee rejected the initial unscored reset')
            action_spec=self.metadata['action_spec']
            names=action_spec['joint_names']
            arm=action_spec.get('arm_joint_names',[n for n in names if n.startswith('fr3_')])
            hand=action_spec.get('hand_joint_names',[n for n in names if n not in arm])
            from bench.control.fingertip import ClassicalFingertipPolicy
            from bench.control.bimanual import BimanualFingertipPolicy
            if isinstance(self.policy,(ClassicalFingertipPolicy,BimanualFingertipPolicy)):
                self.robot_geometry=self.policy.geometry
            elif self.metadata.get('native_isaacsim_app') is True:
                from .controllers import public_robot_geometry, robot_configuration
                self.robot_geometry=public_robot_geometry(robot_configuration(scene_with_hand(scene,request.robot_id)))
            if self.robot_geometry is not None:
                if set(arm+hand)!=set(self.robot_geometry.names):
                    raise ServiceError(422,'Public robot geometry differs from native joint inventory')
                if self.metadata.get('native_isaacsim_app') is True:
                    from .controllers import robot_configuration
                    native_asset=self.metadata.get('robot_asset',{})
                    # Public FK reads either a native source URDF or the native assembly itself.
                    native_files=dict(native_asset.get('sources',{}))
                    native_files[native_asset.get('urdf_path')]=native_asset.get('urdf_sha256')
                    native_files.update(native_asset.get('assembly_urdfs',{}))
                    if any(native_files.get(path)!=sha for path,sha in self.robot_geometry.asset_sha256.items()):
                        raise ServiceError(422,'Public FK URDF differs from native robot sources')
                    if robot_configuration(self.metadata['scene_spec']) != robot_configuration(scene_with_hand(scene,request.robot_id)):
                        raise ServiceError(422,'Native robot base differs from public FK configuration')
                    mount=native_asset.get('mount',{})
                    expected_mount=self.robot_geometry.embodiment.expected_mount
                    # Bimanual mounts are per side: {side: {parent, child, translation_m, quaternion_wxyz}}.
                    nested='parent' not in expected_mount
                    if any((not isinstance(mount.get(k),dict) or any(mount[k].get(a)!=b for a,b in v.items()))
                           if nested else mount.get(k)!=v for k,v in expected_mount.items()):
                        raise ServiceError(422,'Native hand mount differs from public FK configuration')
                self.sensor_config['robot_geometry']={
                    'source':'measured_joint_URDF_FK',
                    'assets_sha256':self.robot_geometry.asset_sha256,
                    'world_from_robot_base':self.robot_geometry.base.tolist(),
                    'available_in_both_control_tracks':True}
            policy=self.policy.descriptor.model_copy(deep=True) if self.policy else None
            robot_hash=canonical_hash({'root_usd':self.metadata['robot_usd_sha256'],
                'dependencies':self.metadata.get('robot_usd_dependency_sha256',{}),
                'hand_id':request.robot_id})
            self.capabilities=RobotCapabilities(robot_id=request.robot_id,
                robot_asset_sha256=robot_hash,arm_joints=arm,hand_joints=hand,
                joint_limits={n:JointLimit(lower_rad=b[0],upper_rad=b[1],
                    max_velocity_rad_s=self._direct_velocity_limit(n, n in arm))
                    for n,b in action_spec['joint_limits_rad'].items()},
                control_period_s=.02,tracks=['direct_joints']+(['system0'] if policy else [])
                    +(['free_tool_agent'] if supports_free_track(self.policy) else []),
                observation_profiles=['vision_proprio']+(['privileged_state'] if allow_privileged else []),
                camera_ids=self.sensor_config['rgb'],depth_available=self.sensor_config['depth'],system0=policy,
                tactile_available=self.tactile_config['available'],
                tactile_mode='simulated_rigid_contact_proxy' if self.tactile_config['available'] else 'unavailable',
                frame_ids=['world','robot_base']+(getattr(self.robot_geometry,'frame_ids',None)
                    or [self.robot_geometry.frame_id] if self.robot_geometry is not None
                    else ['right_hand_flange'] if request.robot_id=='sharpa_w01_right' else []),
                finger_names=hand_spec.get('finger_names') or list(hand_spec['fingertip_links']),
                direct_joint_execution='min_jerk_velocity_preflight_v2',
                hand_geometry=self._hand_geometry(hand))
            if self._acknowledged_drive_targets() is None:
                if self.metadata.get('native_isaacsim_app') is True:
                    raise RuntimeError('Native direct execution requires acknowledged drive targets')
                # Explicit compatibility for old non-native workers. A measured
                # position is never substituted for an acknowledged drive target.
                self.capabilities.direct_joint_execution='step_target_v1'
            instruction=(entry['title_zh']+'。通过机器人实际抓放棋子，按公开规则轮流对弈；落子由原生接触稳定性裁判确认。'
                         if board_game else task['title_zh']+'。'+'；'.join(s['instruction_zh'] for s in task['steps']))
            identity=manifest.get('runtime_identity', {})
            spec=dict(episode_id=self.directory.name,agent_id=request.agent_id,agent_version=request.agent_version,
                agent_config_sha256=request.agent_config_sha256,task_id=entry['id'],
                task_release_sha256=digest(Path(release)/'release_manifest.json'),
                evaluator_sha256=(identity['evaluator_sha256'] if board_game else
                                  manifest['files']['evaluator/sequence_evaluator.py']['sha256']),
                robot_id=request.robot_id,robot_asset_sha256=robot_hash,
                physics_profile_sha256=canonical_hash({'release_profile':(identity['physics_profile_sha256'] if board_game else manifest['physics_profile_sha256']),
                    'effective_isaacsim':self.metadata.get('scene_spec',{}).get('isaacsim',{}),
                    'isaacsim_version':self.metadata.get('isaacsim_version'),
                    'kit_version':self.metadata.get('kit_version'),
                    'experience':self.metadata.get('experience',{}).get('sha256'),
                    'driver_verification_disabled':self.metadata.get('driver_verification_disabled'),
                    'physics_acceptance_profile':physics_profile_sha256()}),
                scene_variant=request.scene_variant,scene_sha256=canonical_hash(self.metadata.get('scene_spec',scene)),
                seed=request.seed,track=request.track,
                observation_profile=request.observation_profile,instruction=instruction,budgets=request.budgets,
                system0=policy,split=expected_spec.split if expected_spec is not None else 'development')
            sources=frontend_sources()
            if sources!=IMPORTED_FRONTEND_SOURCES:
                raise ServiceError(503,'Runner source changed after import; restart the service before evaluation')
            sources.update(self.metadata.get('sources_sha256',{}))
            if 'runtime_code_sha256' in EpisodeSpec.model_fields:
                spec['runtime_code_sha256']=canonical_hash(sources)
                spec['sensor_config_sha256']=canonical_hash(self.sensor_config)
            environment=getattr(self.worker,'execution_environment',None)
            if environment is not None and 'execution_environment_sha256' in EpisodeSpec.model_fields:
                spec['execution_environment_sha256']=canonical_hash(environment) if environment.get('capture_status')=='recorded' else None
                (self.directory/'execution_environment.json').write_text(json.dumps(environment,indent=2)+'\n')
            self.spec=EpisodeSpec(**spec)
            if expected_spec is not None and self.spec != expected_spec:
                raise ServiceError(409,'Runtime configuration does not match the operator-pinned plan slot')
            (self.directory/'source_hashes.json').write_text(json.dumps(sources,indent=2)+'\n')
            (self.directory/'sensor_config.json').write_text(json.dumps(self.sensor_config,indent=2)+'\n')
            archive_project_sources(sources,self.directory/'source_snapshot')
            self.recorder=EpisodeRecorder(self.directory/'agent',self.spec)
            if board_game:
                from .board_games import BoardGameSession
                self.board_game=BoardGameSession(self,initial['board_game'],scene,task)
                self.player_session=self.board_game
            elif turn_game:
                from .turn_games import TurnGameSession
                self.turn_game=TurnGameSession(self,initial['turn_game'],scene,task)
                self.player_session=self.turn_game
            self.cameras=self._cameras()
            if self.policy:
                if hasattr(self.policy,'bind_camera_reader'):
                    self.policy.bind_camera_reader(self._read_policy_camera)
                self.policy.reset(self.spec.model_copy(deep=True),self.capabilities.model_copy(deep=True))
            if request.track=='free_tool_agent':
                self.free_handoff=AcknowledgedDriveLedger(self.policy,self.capabilities,
                    sequence=self.sequence,sim_time_s=float(self.raw['sim_time_s']))
            self.clock=EpisodeClock(request.budgets)
            self.snapshot=self._projection()
            self.recorder.observation(self.snapshot)
            (self.directory/'setup.json').write_text(json.dumps({'setup_wall_s':time.monotonic()-setup_start,
                'setup_excluded_from_episode_wall_time':True,'execution_mode':'pause_physics_during_agent',
                'model_accounting':'unavailable','native_isaac':self.metadata.get('native_isaacsim_app',False),
                'depth_available':self.capabilities.depth_available,'private_artifacts_not_served':True},indent=2)+'\n')
            self.watchdog=None
            if request.budgets.wall_time_s is not None:
                self.watchdog=threading.Thread(target=self._watch,name='episode-wall-budget',daemon=True)
                self.watchdog.start()
        except BaseException as exc:
            if self.worker is not None:
                self.worker.close(force=True)
            if self.directory.exists():
                (self.directory/'setup_failure.json').write_text(json.dumps({'error_type':type(exc).__name__,
                    'detail':str(exc),'episode_clock_started':self.clock is not None})+'\n')
                failed=self.directory.parent/'setup_failures'
                failed.mkdir(exist_ok=True)
                self.directory.rename(failed/(self.directory.name+'_'+uuid.uuid4().hex[:12]))
            raise

    def _cameras(self):
        revision=self.image_revision
        self.image_revision+=1
        if self.clock is not None and self.clock._stop is None:
            remaining=self.clock.remaining_wall_s
            if remaining is not None and remaining<=0:
                raise BudgetExceeded('wall_timeout')
            if hasattr(self.worker,'set_deadline'):
                self.worker.set_deadline(remaining)
        frames=self.worker.request('cameras',sequence=revision)['frames']
        cameras=[]
        for frame in frames:
            data=dict(frame)
            path=self._camera_path(data.pop('path'),data['rgb_sha256'])
            depth_path=data.pop('depth_path',None)
            # Renderer diagnostics remain private; the public camera contract
            # publishes only its existing intrinsics/extrinsics and image fields.
            data.pop('depth_statistics',None)
            data.pop('depth_source',None)
            data.pop('camera_calibration',None)
            data.pop('render_synchronization',None)
            data.pop('task_visual_state',None)
            data['rgb_uri']=f'/episodes/{self.directory.name}/images/{revision}/{data["camera_id"]}'
            if depth_path is not None:
                if data.get('depth_sha256') is None:
                    raise RuntimeError('Worker depth image requires a digest')
                depth_path=self._camera_path(depth_path,data['depth_sha256'])
                data['depth_uri']=f'/episodes/{self.directory.name}/depth/{revision}/{data["camera_id"]}'
            camera=CameraObservation(**data)
            if (camera.depth_uri is not None) != self.capabilities.depth_available:
                raise RuntimeError('Worker depth frame disagrees with advertised sensor capability')
            if abs(camera.sim_time_s - self.raw['sim_time_s']) > 1e-9:
                raise RuntimeError('Worker camera and observation timestamps must match')
            if 'agent_sensor_config' in self.metadata:
                dimensions = self.sensor_config.get('cameras',{}).get(camera.camera_id,self.sensor_config)
                if any(getattr(camera,key) != dimensions.get(key) for key in ('width','height')):
                    raise RuntimeError('Worker camera dimensions disagree with pinned sensor configuration')
            self.policy_images[camera.rgb_uri] = (path,camera.rgb_sha256)
            if depth_path is not None:
                self.policy_depths[camera.depth_uri] = (depth_path,camera.depth_sha256)
            cameras.append(camera)
        if len(cameras) != len(self.capabilities.camera_ids) or {
                camera.camera_id for camera in cameras} != set(self.capabilities.camera_ids):
            raise RuntimeError('Worker camera inventory disagrees with pinned sensor configuration')
        return cameras

    def _camera_path(self, value, expected):
        path=Path(value).resolve()
        if not path.is_relative_to((self.directory/'private').resolve()):
            raise RuntimeError('Worker image escaped the episode directory')
        if not path.is_file() or digest(path)!=expected:
            raise RuntimeError('Worker camera frame digest does not match recorded bytes')
        return path

    def read_public_depth(self, revision, camera_id):
        """Resolve only published depth frames; never expose filesystem lookup."""
        with self.lock:
            stored=self.public_depths.get((revision,camera_id))
            if stored is None:
                raise ServiceError(404,'Unknown public depth frame')
            path,expected=stored
            path=self._camera_path(path,expected)
            value=path.read_bytes()
            if hashlib.sha256(value).hexdigest()!=expected:
                raise RuntimeError('Recorded depth frame changed')
            return value

    def _read_policy_camera(self, camera):
        """Trusted policy image reader; opaque generated RGB only, no environment handle."""
        with self.lock:
            uri=camera.rgb_uri if isinstance(camera,CameraObservation) else camera
            if uri not in self.policy_images:
                raise ValueError('Unknown policy camera frame')
            path,expected=self.policy_images[uri]
            value=path.read_bytes()
            if hashlib.sha256(value).hexdigest()!=expected:
                raise RuntimeError('Recorded camera frame changed')
            return value

    def _hand_geometry(self, hand_joints):
        """Static robot facts: URDF joint-to-joint offsets and native hand drive gains."""
        if self.robot_geometry is None:
            return None
        value={}
        for chain in self.robot_geometry.finger_joints.values():
            for joint in chain[1:]:
                length=float(np.linalg.norm(joint.origin[:3,3]))
                if length>0:
                    value[f'{joint.name}_from_previous_joint_m']=length
        stiffness=self.metadata.get('native_joint_stiffnesses',{})
        # Publish drive gains only for an adopted actuator model, never the
        # uncalibrated legacy placeholder gains.
        adopted=self.metadata.get('hand_actuator_model') not in (None,'legacy_uniform_gains')
        for name in hand_joints if adopted else ():
            if stiffness.get(name,0)>0:
                value[f'{name}_drive_kp_nm_per_rad']=float(stiffness[name])
        return value or None

    def _raw_proprio(self):
        # Mimic/passive coordinates remain in private physics recordings.
        raw=deepcopy(self.raw)
        for key in ('joint_positions_rad','joint_velocities_rad_s'):
            raw[key]={n:raw[key][n] for n in self.capabilities.joint_limits}
        # Public velocity is the control-period mean, like an encoder difference;
        # raw solver velocities (contact chatter) stay in the private record.
        previous=self._previous_public_q
        if previous is not None and previous[0]<self.raw['sim_time_s']:
            dt=self.raw['sim_time_s']-previous[0]
            raw['joint_velocities_rad_s']={n:(raw['joint_positions_rad'][n]-previous[1][n])/dt
                                           for n in self.capabilities.joint_limits}
        elif previous is None:
            raw['joint_velocities_rad_s']={n:0. for n in self.capabilities.joint_limits}
        # The acknowledged drive targets are robot-only commanded positions.
        targets=raw.get('robot',{}).get('joint_targets_rad')
        raw['commanded_joint_positions_rad']=None if targets is None else {n:targets[n] for n in self.capabilities.joint_limits}
        return raw

    def _projection(self):
        # Only decision-boundary frames enter the public HTTP whitelist. Policy
        # intermediate frames use the injected private image reader.
        for camera in self.cameras:
            pieces=camera.rgb_uri.rsplit('/',2)
            self.public_images[(int(pieces[-2]),pieces[-1])]=self.policy_images[camera.rgb_uri][0]
            if camera.depth_uri is not None:
                pieces=camera.depth_uri.rsplit('/',2)
                self.public_depths[(int(pieces[-2]),pieces[-1])]=self.policy_depths[camera.depth_uri]
        geometry=None
        if self.robot_geometry is not None:
            q=self._raw_proprio()['joint_positions_rad']
            geometry=({'hand_geometries':self.robot_geometry.public_hand_geometries(q)}
                      if hasattr(self.robot_geometry,'public_hand_geometries')
                      else self.robot_geometry.public_finger_geometry(q))
        sides=self.tactile_config.get('sides')
        return project_observation(self._raw_proprio(),self.spec,self.sequence,cameras=self.cameras,
            robot_geometry=geometry,tactile=None if sides else project_tactile(self.raw,self.tactile_config),
            hand_tactile={side:project_tactile(self.raw,config) for side,config in sides.items()} if sides else None,
            last_action_id=self.last_action_id,
            last_action_status=self.life.statuses.get(self.last_action_id,'idle'))

    def _safe_diagnostic(self, value, *, default=None):
        """Do not expose policy detail or an exception's arbitrary text/fields."""
        if value is None:
            return ActionDiagnostic(code=default) if default else None
        try:
            diagnostic=ActionDiagnostic.model_validate(value)
            if diagnostic.joint_name is not None and diagnostic.joint_name not in self.capabilities.joint_limits:
                return ActionDiagnostic(code=diagnostic.code)
            if not set(diagnostic.fingertip_errors_m) <= set(self.capabilities.finger_names):
                return ActionDiagnostic(code=diagnostic.code)
            if not set(diagnostic.limiting_joint_names) <= set(self.capabilities.joint_limits):
                return ActionDiagnostic(code=diagnostic.code)
            return diagnostic.model_copy(deep=True)
        except (ValueError,TypeError):
            return ActionDiagnostic(code=default) if default else None

    def _safe_execution_state(self, value):
        # Robot-only typed data, current acknowledged time, exact capability
        # names and finite per-joint envelopes. Never expose arbitrary detail.
        if value is None:
            return None
        try:
            state=RobotExecutionState.model_validate(value)
            names=(set(state.posture_objective_joint_names) |
                   set(state.recovery_fixed_drive_joint_names) |
                   set(state.maintained_joint_error_rad))
            if not names <= set(self.capabilities.joint_limits):
                return None
            if abs(state.sim_time_s-self.clock.sim_time_s)>1e-9:
                return None
            for name,error in state.maintained_joint_error_rad.items():
                limit=self.capabilities.joint_limits[name]
                # Same measured-state emergency envelope as native execution;
                # this bounds published residuals, not a hardware safety claim.
                if error>limit.upper_rad-limit.lower_rad+MEASURED_JOINT_LIMIT_SAFETY_BAND_RAD+1e-9:
                    return None
            if len(state.model_dump_json().encode())>MAX_ROBOT_EXECUTION_STATE_BYTES:
                return None
            return state.model_copy(deep=True)
        except (ValueError,TypeError):
            return None

    def _event(self, action_id, status, diagnostic=None, *, termination_reason=None,
               robot_execution_state=None):
        if status=='cancelled' and termination_reason is None:
            termination_reason='cancelled'
        event=self.life.transition(action_id,status,sim_time_s=self.clock.sim_time_s,wall_time_s=self.clock.wall_time_s)
        event.diagnostic=self._safe_diagnostic(diagnostic,default='cancelled' if status=='cancelled' else None)
        event.termination_reason=termination_reason
        # Task success may stop a still-running primitive before its next policy
        # check. Do not relabel an older state or manufacture convergence.
        if termination_reason not in ('task_terminated','begin_rejected','episode_terminated','cancelled'):
            event.robot_execution_state=self._safe_execution_state(robot_execution_state)
        if self.contact_history.action_id==action_id:
            event.contact_history=parse_contact_history(self.contact_history.snapshot())
        self.recorder.lifecycle(event)

    def _direct_velocity_limit(self, name, arm):
        """Use the stricter reference cap, pinned URDF and native velocity cap."""
        limits=[DIRECT_ARM_VELOCITY_RAD_S if arm else DIRECT_HAND_VELOCITY_RAD_S]
        if self.robot_geometry is not None:
            limits.append(self.robot_geometry.by_name[name].velocity)
        native=self.metadata.get('native_joint_max_velocities',{})
        if name in native:
            limits.append(native[name])
        if any(isinstance(value,bool) or not isinstance(value,(int,float))
               or not math.isfinite(value) or value<=0 for value in limits):
            raise RuntimeError('Invalid direct joint velocity limit')
        return min(limits)

    def _acknowledged_drive_targets(self):
        """Last acknowledged native drive targets, clamped against float32 roundoff."""
        targets=self.raw.get('robot',{}).get('joint_targets_rad')
        if not isinstance(targets,dict) or not set(self.capabilities.joint_limits)<=set(targets):
            return None
        if any(isinstance(targets[n],bool) or not isinstance(targets[n],(int,float))
               or not math.isfinite(targets[n]) for n in self.capabilities.joint_limits):
            raise RuntimeError('Acknowledged drive targets must be finite numbers')
        for name,limit in self.capabilities.joint_limits.items():
            tolerance=8*np.finfo(np.float32).eps*max(1.,abs(limit.lower_rad),abs(limit.upper_rad))
            if not limit.lower_rad-tolerance<=targets[name]<=limit.upper_rad+tolerance:
                raise RuntimeError('Acknowledged drive target exceeds joint limits beyond float32 roundoff')
        return {n:min(limit.upper_rad,max(limit.lower_rad,float(targets[n])))
                for n,limit in self.capabilities.joint_limits.items()}

    def _direct_trajectory(self, action):
        """Read-only speed eligibility; no handoff, drive or physics mutation.

        Min-jerk scaling bounds velocity inside this requested segment only.
        It does not impose acceleration bounds or smooth a prior moving segment.
        """
        goal=action.arm_joint_positions_rad|action.hand_joint_positions_rad
        start=self._acknowledged_drive_targets()
        period=self.capabilities.control_period_s
        steps=round(action.duration_s/period)
        if start is None:
            if self.capabilities.direct_joint_execution!='step_target_v1':
                raise RuntimeError('Acknowledged direct drive reference is unavailable')
            return goal,goal,1,None
        minimum=max(1,math.ceil(max(1.875*abs(goal[n]-start[n])/
            self.capabilities.joint_limits[n].max_velocity_rad_s/period for n in goal)-1e-9))
        if steps<minimum:
            diagnostic=ActionDiagnostic(code='insufficient_duration',duration_preflight=DurationPreflight(
                skill='direct_joints',requested_duration_s=action.duration_s,
                minimum_duration_s=minimum*period,control_period_s=period,
                minimum_steps=minimum,braking_steps=0,trajectory_steps=minimum,settling_steps=0))
            return start,goal,minimum,diagnostic
        return start,goal,max(math.ceil(.8*steps),minimum),None

    def _objective_in_contact(self, diagnostic):
        """Whether every unreached objective has ongoing contact in its own public regions."""
        history=self.contact_history.snapshot()
        regions=history.get('regions') or {}
        if history.get('status')!='available':
            return False
        touching=lambda names: any(regions.get(name,{}).get('current_contact_substeps',0)>0 for name in names)
        tolerances=diagnostic.geometry_tolerances
        unreached=[finger for finger,error in diagnostic.fingertip_errors_m.items()
                   if tolerances is None or error>tolerances.fingertips_m.get(finger,0.)]
        wrist=(tolerances is not None and tolerances.wrist_position_m is not None
               and ((diagnostic.wrist_position_error_m or 0.)>tolerances.wrist_position_m
                    or (diagnostic.wrist_orientation_error_rad or 0.)>tolerances.wrist_orientation_rad))
        if diagnostic.joint_name is not None and tolerances is not None and tolerances.joint_posture_rad is not None:
            finger=next((f for f in self.capabilities.finger_names
                         if f'_{"pinky" if f=="little" else f}_' in diagnostic.joint_name),None)
            if finger is not None and finger not in unreached:
                unreached.append(finger)
        if not unreached and not wrist:
            return False
        if any(not touching((finger+'_pad',finger+'_shell')) for finger in unreached):
            return False
        return not wrist or touching(tuple(regions))

    def _watch(self):
        while not self.done.wait(.1):
            with self.lock:
                if self.result is not None:
                    return
                if self.clock.wall_expired:
                    self.cancelled.set()
                    self._finish('wall_timeout')
                    return

    def observe(self):
        with self.lock:
            return self.snapshot.model_dump(mode='json',exclude_none=True)

    def task_requirements(self):
        """Return static rules and opt-in coarse status, never raw evidence."""
        with self.lock:
            if self.public_task is None:
                raise ServiceError(404,'This task does not publish structured episode requirements')
            result = deepcopy(self.public_task)
            if getattr(self, 'referee_state', None) is not None:
                result['referee'] = deepcopy(self.referee_state)
                # Runner budgets, cancellation and final audit can end an
                # episode independently of its last valid physical verdict.
                result['episode'] = {'terminal': self.result is not None,
                    'outcome': self.result.outcome if self.result is not None else None}
                if self.result is not None and self.result.scoring is not None:
                    result['scoring'] = self.result.scoring.model_dump(mode='json', by_alias=True)
            return result

    @property
    def functions(self):
        from .functions import EpisodeFunctions
        with self.lock:
            if not hasattr(self, '_function_api'):
                self._function_api = EpisodeFunctions(self)
            return self._function_api

    def robot_guide(self):
        from .robot_support import robot_guide
        with self.lock:
            return robot_guide(self.capabilities, self.robot_geometry)

    def query_ik(self, query):
        from .robot_support import query_ik
        from bench.protocol.robot import IKQueryResult
        with self.lock:
            if self.result is not None:
                raise ServiceError(409, 'IK queries require an active episode', code='episode_finished')
            if self.life.active_action_id is not None or (
                    self.thread is not None and self.thread.is_alive()):
                raise ServiceError(409, 'IK queries require an idle active episode', code='episode_busy')
            if query.observation_sequence != self.snapshot.sequence:
                raise ServiceError(409, 'IK query references a stale observation', code='stale_observation')
            if self.robot_geometry is None:
                raise ServiceError(422, 'Robot geometry is unavailable in this episode', code='unsupported_capability')
            joints = dict(self.snapshot.proprioception.joint_positions_rad)
            geometry = self.robot_geometry
            limits = getattr(self.policy, 'limits', None)
            # Bimanual sides use the same executor limits by construction.
            if limits is None and hasattr(self.policy, 'sides'):
                limits = self.policy.sides['left'].limits
            sample_time = self.snapshot.sim_time_s
        try:
            result = query_ik(geometry, joints, query, limits=limits)
        except ValueError as exc:
            raise ServiceError(422, 'Unsupported or invalid robot geometry query') from exc
        with self.lock:
            if self.result is not None or self.snapshot.sequence != query.observation_sequence or (
                    self.life.active_action_id is not None or (self.thread is not None and self.thread.is_alive())):
                raise ServiceError(409, 'Episode changed while the IK query was evaluated', code='stale_observation')
        return IKQueryResult.model_validate({**result, 'sim_time_s': sample_time}).model_dump(mode='json', exclude_unset=True)

    def status(self, action_id):
        with self.lock:
            if action_id not in self.life.statuses:
                raise ServiceError(404,'Unknown action')
            events=[event for event in self.life.events if event.action_id==action_id]
            last=events[-1]
            elapsed=(last.sim_time_s if last.status in ('completed','failed','cancelled') else self.clock.sim_time_s)-events[0].sim_time_s
            return {'action_id':action_id,'status':self.life.statuses[action_id],
                    'sim_time_s':self.clock.sim_time_s,'episode_finished':self.result is not None,
                    'elapsed_sim_s':max(0.,elapsed),
                    'diagnostic':last.diagnostic.model_dump(mode='json',exclude_none=True) if last.diagnostic else None,
                    'termination_reason':last.termination_reason,
                    'robot_execution_state':last.robot_execution_state.model_dump(mode='json') if last.robot_execution_state else None,
                    'contact_history':last.contact_history.model_dump(mode='json') if last.contact_history else None}

    def submit(self,payload):
        with self.lock:
            if self.player_session is not None:
                self.player_session.admit(payload)
            try:
                # The legacy route and named functions may spell equivalent
                # default fields differently. Identity belongs to the validated
                # action, while the attempt log retains the received payload.
                try:
                    canonical_payload=parse_action(payload).model_dump(mode='json')
                except (ValueError,TypeError):
                    canonical_payload=payload
                fingerprint=canonical_hash(canonical_payload)
            except (ValueError,TypeError):
                fingerprint=None
            action_id=payload.get('action_id') if isinstance(payload,dict) else None
            if isinstance(action_id,str) and action_id in self.requests:
                if fingerprint!=self.requests[action_id]:
                    if self.result is None:
                        try:
                            self._attempt(payload,fingerprint,canonical_action=True)
                        except BudgetExceeded as exc:
                            self._finish(exc.outcome)
                            raise ServiceError(409,exc.outcome,code='budget_exhausted') from exc
                    raise ServiceError(409,'action_id was already used for different content',code='action_conflict')
                return self.status(action_id)
            if self.result is not None:
                raise ServiceError(409,'Episode is finished',code='episode_finished')
            try:
                self._attempt(payload,fingerprint,canonical_action=True)
                if self.life.active_action_id is not None or (self.thread is not None and self.thread.is_alive()):
                    raise ServiceError(409,'An action is already running')
                action=parse_action(payload)
                if action.observation_sequence != self.sequence:
                    raise ServiceError(422,'Action references a stale observation',code='stale_observation')
                validate_action(action,self.capabilities,self.spec,observation_sequence=self.sequence)
            except BudgetExceeded as exc:
                self._finish(exc.outcome)
                raise ServiceError(409,exc.outcome,code='budget_exhausted') from exc
            except (ValueError,TypeError) as exc:
                raise ServiceError(422,'Action violates the advertised contract') from exc
            self.requests[action.action_id]=fingerprint
            if self.player_session is not None:
                self.player_session.accepted(action.action_id)
            if isinstance(action,DirectJointAction):
                # This cache only recognizes cross-API hold retries. Invalid
                # ACK state must follow the executor's normal failure lifecycle,
                # not leave a request admitted without an action event.
                try:
                    acknowledged=self._acknowledged_drive_targets()
                except RuntimeError:
                    acknowledged=None
                if acknowledged is not None and acknowledged == (action.arm_joint_positions_rad | action.hand_joint_positions_rad):
                    self.direct_hold_replays[action.action_id]=action.model_dump(mode='json')
            self.last_action_id=action.action_id
            self.recorder.action(payload)
            self.contact_history.begin(action.action_id,self.clock.sim_time_s)
            accepted_event=self.life.accept(action.action_id,sim_time_s=self.clock.sim_time_s,wall_time_s=self.clock.wall_time_s)
            accepted_event.contact_history=parse_contact_history(self.contact_history.snapshot())
            self.recorder.lifecycle(accepted_event)
            self.cancelled.clear()
            self.thread=threading.Thread(target=self._execute,args=(action,),name='agent-action',daemon=True)
            self.thread.start()
            return self.status(action.action_id)

    def _attempt(self,payload,fingerprint,*,canonical_action=False):
        self.clock.action_attempt()
        try:
            raw_fingerprint=canonical_hash(payload)
        except (ValueError,TypeError):
            raw_fingerprint=None
        value={'attempt':self.clock.action_attempts,'wall_time_s':self.clock.wall_time_s,
               'sim_time_s':self.clock.sim_time_s,'payload_sha256':raw_fingerprint,
               'payload':payload if raw_fingerprint is not None else None}
        if canonical_action:
            value['canonical_action_sha256']=fingerprint
        with (self.directory/'private/action_attempts.jsonl').open('a') as stream:
            stream.write(json.dumps(value,allow_nan=False)+'\n')

    def _execute(self,action):
        try:
            with self.lock:
                if self.result is not None:
                    return
                self._event(action.action_id,'running')
                action_start_sim_s=self.clock.sim_time_s
            if isinstance(action,DirectJointAction):
                with self.lock:
                    if self.result is not None:
                        return
                    start,goal,ramp_steps,diagnostic=self._direct_trajectory(action)
                    if diagnostic is not None:
                        self._event(action.action_id,'failed',diagnostic,termination_reason='begin_rejected')
                        self.snapshot=self._projection()
                        self.recorder.observation(self.snapshot)
                        return
                if self.free_handoff is not None:
                    self.free_handoff.before_direct()
            if not isinstance(action,DirectJointAction):
                try:
                    if self.player_session is not None:
                        actor=self.player_session.actions[action.action_id]['player_id']
                        self.policy.set_player_hand(self.player_session.bindings[actor].hand,
                                                    self.player_session.last_targets)
                    self.policy.begin(action.model_copy(deep=True),project_policy_observation(self._raw_proprio(),self.spec,self.sequence,cameras=self.cameras))
                except Exception as exc:
                    from bench.control.fingertip import GeometryIKError
                    if isinstance(exc,GeometryIKError):
                        with self.lock:
                            if self.result is not None:
                                return
                            with (self.directory/'private/geometry_rejections.jsonl').open('a') as stream:
                                stream.write(json.dumps({'action_id':action.action_id,'reason':str(exc)})+'\n')
                            diagnostic=self._safe_diagnostic(getattr(exc,'public_diagnostic',None),default='ik_solution_unverified')
                            self._event(action.action_id,'failed',diagnostic,termination_reason='begin_rejected')
                            self.snapshot=self._projection()
                            self.recorder.observation(self.snapshot)
                        return
                    raise PolicyExecutionError(str(exc)) from exc
            duration=action.duration_s if isinstance(action,DirectJointAction) else action.timeout_sim_s
            steps=round(duration/self.capabilities.control_period_s)
            pending_terminal=None
            for step_index in range(steps):
                with self.lock:
                    if self.result is not None:
                        return
                    if self.cancelled.is_set():
                        # Physics is paused between requests. Retain the last
                        # submitted drive target; writing measured q here would
                        # introduce an unbounded reference jump on resumption.
                        pending_terminal=('cancelled',None,{})
                        break
                    self.clock.check(next_sim_dt=.02)
                if isinstance(action,DirectJointAction):
                    tau=min(1.,(step_index+1)/ramp_steps)
                    blend=10*tau**3-15*tau**4+6*tau**5
                    command=RobotJointCommand(duration_s=.02,
                        joint_positions_rad={n:goal[n] if tau>=1. else start[n]+blend*(goal[n]-start[n]) for n in goal})
                else:
                    with self.lock:
                        needs_camera=self.policy.descriptor.camera_observation_mode=='every_control_step'
                        if needs_camera:
                            self.cameras=self._cameras()
                        policy_obs=project_policy_observation(self._raw_proprio(),self.spec,self.sequence,
                            cameras=self.cameras if needs_camera else [])
                    try:
                        result=PolicyStep.model_validate(self.policy.step(policy_obs))
                    except Exception as exc:
                        raise PolicyExecutionError(str(exc)) from exc
                    with self.lock:
                        if self.result is not None:
                            return
                        with (self.directory/'private/policy_steps.jsonl').open('a') as stream:
                            stream.write(json.dumps({'sim_time_s':policy_obs.sim_time_s,
                                'result':result.model_dump(mode='json')},allow_nan=False)+'\n')
                    if result.status!='running':
                        if (action.skill=='hold' and result.status=='completed'
                                and self.clock.sim_time_s-action_start_sim_s < action.hold_duration_s-1e-9):
                            raise PolicyExecutionError('Timed hold completed before its requested physics duration')
                        with self.lock:
                            if self.result is not None:
                                return
                            self.clock.check(next_sim_dt=.02)
                            if self.cancelled.is_set():
                                pending_terminal=('cancelled',None,{})
                            else:
                                reason=(('time_hold_completed' if action.skill=='hold' else 'controller_converged')
                                        if result.status=='completed' else 'controller_failed')
                                if (result.status=='failed' and result.diagnostic is not None
                                        and result.diagnostic.code=='tracking_stalled'
                                        and self._objective_in_contact(result.diagnostic)):
                                    # Public contact history shows the stalled objective is pressing on something.
                                    reason='contact_stalled'
                                pending_terminal=('completed' if result.status=='completed' else 'failed',result.diagnostic,
                                    {'termination_reason':reason,'robot_execution_state':result.robot_execution_state})
                        break
                    command=result.command
                with self.lock:
                    if self.result is not None:
                        return
                    if self.cancelled.is_set():
                        pending_terminal=('cancelled',None,{})
                        break
                    self.clock.check(next_sim_dt=.02)
                    try:
                        validate_joint_command(command,self.capabilities)
                    except (ValueError,TypeError) as exc:
                        if self.policy:
                            raise PolicyExecutionError(str(exc)) from exc
                        raise
                    if step_index == 0 and 'action_budget' in self.score:
                        budget = self.score['action_budget']
                        if (not isinstance(budget, dict) or set(budget) != {'maximum', 'admitted'}
                                or type(budget['admitted']) is not int or budget['admitted'] < 0
                                or budget['maximum'] is not None and
                                (type(budget['maximum']) is not int or budget['maximum'] < 1)):
                            raise ValueError('Invalid task action admission budget')
                        if budget['maximum'] is not None and budget['admitted'] >= budget['maximum']:
                            self._finish('action_budget');return
                    remaining=self.clock.remaining_wall_s
                    if hasattr(self.worker,'set_deadline'):
                        self.worker.set_deadline(None if remaining is None else max(.001,remaining))
                    actor=({'actor': self.player_session.command_actor(action.action_id,command)}
                           if self.player_session is not None else {})
                    reply=self.worker.request('step',command=command.model_dump(mode='json'),
                                              action_id=action.action_id,**actor)
                    # Reject a corrupt task receipt before committing either
                    # the controller reference or the public ACK snapshot.
                    referee_state = (project_referee_state(reply['score'], self.referee_contract)
                        if self.referee_contract is not None else None)
                    if (self.public_task or {}).get('scoring_contract') is not None:
                        json.dumps(reply['score'], allow_nan=False)
                    # Only an acknowledged native step may advance a stateful
                    # policy's committed reference. Cancellation and check-only
                    # candidates are discarded in the corresponding exit path.
                    if self.free_handoff is not None:
                        try:
                            try:
                                ack_time=reply['observation']['sim_time_s']
                            except (KeyError,TypeError) as exc:
                                raise NativeAcknowledgementError('Native reply lacks its acknowledged clock') from exc
                            receipt=self.free_handoff.acknowledge(command,kind=action.kind,
                                sequence=self.sequence+1,sim_time_s=ack_time)
                            with (self.directory/'private/control_handoffs.jsonl').open('a') as stream:
                                stream.write(json.dumps({'action_id':action.action_id,**receipt},allow_nan=False)+'\n')
                        except NativeAcknowledgementError:
                            raise
                        except Exception as exc:
                            raise PolicyExecutionError(str(exc)) from exc
                    elif self.policy and hasattr(self.policy,'commit_pending_step'):
                        try:
                            self.policy.commit_pending_step()
                        except Exception as exc:
                            raise PolicyExecutionError(str(exc)) from exc
                    # Validate the task receipt before installing any part of
                    # this ACK. A corrupt referee packet must leave public
                    # robot time/sequence at the last acknowledged snapshot.
                    turn_changed=(self.player_session.acknowledge(reply,command) if self.player_session is not None else False)
                    self._previous_public_q=(self.raw['sim_time_s'],
                        {n:self.raw['joint_positions_rad'][n] for n in self.capabilities.joint_limits})
                    self.raw,self.score=reply['observation'],reply['score']
                    self.referee_state = referee_state
                    self.clock.record_step(float(self.raw['sim_time_s']))
                    # A late physical ACK can update observations, but cannot
                    # earn a numeric record beyond either declared horizon.
                    budgets = self.spec.budgets
                    if ((budgets.sim_time_s is None or self.clock.sim_time_s <= budgets.sim_time_s + 1e-9)
                            and (budgets.wall_time_s is None or self.clock.wall_time_s <= budgets.wall_time_s)):
                        self.terminal_score_receipt = self.score
                    self.contact_history.observe(self.raw)
                    self.recorder.command(command)
                    self.sequence+=1
                    # The complete executed ACK and controller reference above
                    # must be committed even when this window triggers a stop.
                    # Check before task success; never send another worker step.
                    trigger=self.rigid_safety.observe(self.raw)
                    if trigger is not None:
                        (self.directory/'private/physics_safety_stop.json').write_text(json.dumps({
                            **trigger,'action_id':action.action_id,'observation_sequence':self.sequence},
                            sort_keys=True,allow_nan=False)+'\n')
                        self._finish('safety_stop',physics_valid=False);return
                    # Never turn a late completion into success after wall timeout.
                    if self.clock.wall_expired:
                        self._finish('wall_timeout');return
                    if self.player_session is not None:
                        self.player_session.commit_budget_state()
                    if not self.raw['metrics']['state_finite'] or self.raw.get('terminated'):
                        self._finish('safety_stop',physics_valid=False);return
                    band=MEASURED_JOINT_LIMIT_SAFETY_BAND_RAD
                    if any(not limit.lower_rad-band<=self.raw['joint_positions_rad'][n]<=limit.upper_rad+band
                           for n,limit in self.capabilities.joint_limits.items()):
                        self._finish('safety_stop',physics_valid=False);return
                    if self.referee_state is not None and self.referee_state['terminal']:
                        outcome = self.referee_state['state']
                        self._finish(outcome, physics_valid=outcome not in ('evaluator_invalid', 'safety_stop'))
                        return
                    reason=str(self.score.get('reason',''))
                    if self.score.get('status') == 'incomplete' and self.score.get('terminal') is True:
                        # A task's own frozen budget can expire before the host
                        # budget. Preserve its ongoing physical referee while
                        # stopping the episode at this last acknowledged step.
                        if reason not in ('budget_exhausted', 'cancelled'):
                            raise ValueError('Unknown terminal incomplete task reason')
                        self._finish('sim_timeout' if reason == 'budget_exhausted' else 'cancelled')
                        return
                    if self.referee_state is None and reason.startswith('invalid_observation:'):
                        # The frozen evaluator latched invalid evidence, so success
                        # is impossible. Violent object motion is the agent's doing;
                        # any other invalidation is an evidence fault, not a failure.
                        if 'motion speed bound' in reason:
                            self._finish('safety_stop');return
                        (self.directory/'private/evaluator_invalid.json').write_text(json.dumps({
                            'reason':reason,'sim_time_s':self.raw['sim_time_s']},indent=2)+'\n')
                        self._finish('evaluator_invalid',physics_valid=False);return
                    if self.score['success']:
                        self._finish('success');return
                    if (self.score.get('irrecoverable_reason') or
                            (self.turn_game is not None and self.turn_game.state.terminal)):
                        # A latched episode constraint makes success impossible;
                        # spending the remaining budget would change nothing.
                        self._finish('task_failure');return
                    if self.clock.sim_expired:
                        self._finish('sim_timeout');return
                    if turn_changed:
                        reason='turn_changed' if self.turn_game is not None else 'board_turn_changed'
                        if self.policy is not None:
                            self.policy.cancel(reason)
                        pending_terminal=('cancelled',None,{'termination_reason':reason})
                        break
            with self.lock:
                if self.result is not None:
                    return
                if self.life.active_action_id==action.action_id:
                    # Publish a terminal lifecycle event and its current RGB-D
                    # snapshot under one lock; readers must never see a terminal
                    # action with the preceding decision's observation.
                    if pending_terminal is not None:
                        status,diagnostic,fields=pending_terminal
                        self._event(action.action_id,status,diagnostic,**fields)
                    else:
                        completed=isinstance(action,DirectJointAction)
                        if not completed and self.policy.descriptor.camera_observation_mode=='decision_only':
                            # Check the feedback from the final allowed physics step.
                            # A running result must never execute another command.
                            final_obs=project_policy_observation(self._raw_proprio(),self.spec,self.sequence,cameras=[])
                            try:
                                final_step=PolicyStep.model_validate(self.policy.step(final_obs))
                            except Exception as exc:
                                raise PolicyExecutionError(str(exc)) from exc
                            with (self.directory/'private/policy_steps.jsonl').open('a') as stream:
                                stream.write(json.dumps({'sim_time_s':final_obs.sim_time_s,
                                    'terminal_check_only':True,'result':final_step.model_dump(mode='json')})+'\n')
                            completed=final_step.status=='completed'
                            if final_step.status=='running':
                                if hasattr(self.policy,'discard_pending_step'):
                                    self.policy.discard_pending_step()
                                self.policy.cancel('action_timeout')
                        elif not completed:
                            # Policies requiring per-step images do not receive an
                            # extra camera-free final check. Release their active
                            # action through the same cancellation/recovery hook.
                            self.policy.cancel('action_timeout')
                        diagnostic=final_step.diagnostic if not isinstance(action,DirectJointAction) and self.policy.descriptor.camera_observation_mode=='decision_only' else None
                        if not completed and diagnostic is None:
                            diagnostic=ActionDiagnostic(code='convergence_timeout',elapsed_sim_s=duration)
                        self._event(action.action_id,'completed' if completed else 'failed',diagnostic,
                            termination_reason=(('direct_duration_completed' if isinstance(action,DirectJointAction)
                                else 'time_hold_completed' if action.skill=='hold' else 'controller_converged')
                                if completed else 'controller_failed'),
                            robot_execution_state=(final_step.robot_execution_state if not isinstance(action,DirectJointAction)
                                and self.policy.descriptor.camera_observation_mode=='decision_only' else None))
                self.cameras=self._cameras()
                self.snapshot=self._projection()
                self.recorder.observation(self.snapshot)
        except BudgetExceeded as exc:
            with self.lock:
                self._finish(exc.outcome)
        except Exception as exc:
            with self.lock:
                if self.result is not None:
                    return
                (self.directory/'private/action_error.json').write_text(json.dumps({'type':type(exc).__name__,'detail':str(exc)},indent=2)+'\n')
                if isinstance(exc,PolicyExecutionError):
                    self._finish('policy_error')
                else:
                    self._finish('wall_timeout' if isinstance(exc,TimeoutError) and
                        self.clock.wall_expired else 'infrastructure_error',physics_valid=False)
        finally:
            if self.policy and hasattr(self.policy,'discard_pending_step'):
                try:
                    self.policy.discard_pending_step()
                except Exception as exc:
                    with self.lock:
                        (self.directory/'private/policy_discard_error.json').write_text(json.dumps({
                            'type':type(exc).__name__,'detail':str(exc)},indent=2)+'\n')
                        if self.result is None:
                            self._finish('policy_error')
            if self.cancelled.is_set() and self.policy and not isinstance(action,DirectJointAction):
                try:
                    self.policy.cancel('cancelled')
                except Exception:
                    pass

    def cancel(self,action_id):
        with self.lock:
            value=self.status(action_id)
            if value['status'] in ('accepted','running'):
                self.cancelled.set()
                self.cancellation_requests.add(action_id)
            return {**value,'cancellation_requested':action_id in self.cancellation_requests}

    def _finish(self,outcome,physics_valid=True):
        if self.result is not None:
            return self.result
        self.cancelled.set()
        if self.life.active_action_id:
            self._event(self.life.active_action_id,{'success':'completed','cancelled':'cancelled'}.get(outcome,'failed'),
                termination_reason={'success':'task_terminated','cancelled':'cancelled'}.get(outcome,'episode_terminated'))
        self.clock.stop()
        if frontend_sources()!=IMPORTED_FRONTEND_SOURCES:
            outcome='infrastructure_error'
            physics_valid=False
            (self.directory/'private/source_drift.json').write_text(json.dumps({
                'imported':IMPORTED_FRONTEND_SOURCES,'current':frontend_sources()},indent=2)+'\n')
        if self.player_session is not None:
            try:
                self.player_session.end(outcome)
            except Exception as exc:
                outcome='infrastructure_error'
                physics_valid=False
                (self.directory/f'private/{self.player_session.kind}_end_error.json').write_text(json.dumps({
                    'type':type(exc).__name__,'detail':str(exc)})+'\n')
        score_path=self.directory/'private/evaluator_evidence.json'
        score_evidence = deepcopy(self.score)
        if (self.public_task or {}).get('scoring_contract') is not None:
            score_evidence['terminal_scoring_receipt'] = deepcopy(self.terminal_score_receipt)
        score_path.write_text(json.dumps(score_evidence,sort_keys=True,allow_nan=False)+'\n')
        fields=self.clock.snapshot()
        if outcome=='success':
            fields.update(completion_sim_s=fields['elapsed_sim_s'],completion_wall_s=fields['elapsed_wall_s'])
        finalization_errors=[]
        def record_finalize_error(stage,exc):
            finalization_errors.append({'stage':stage,'type':type(exc).__name__,'detail':str(exc)})
        try:
            if getattr(self.worker,'broken',False):
                self.cameras=[]
                record_finalize_error('native_transport',RuntimeError('Native RPC stream failed; last acknowledged state retained'))
                self.worker.close(force=True)
                physics_valid=False
            else:
                if hasattr(self.worker,'set_deadline'):
                    self.worker.set_deadline(None if self.spec.budgets.wall_time_s is None else 30)
                try:
                    self.cameras=self._cameras()
                except Exception as exc:
                    # A failed final image must not skip the worker's state
                    # archive, nor pair an old image with a newer robot state.
                    record_finalize_error('terminal_cameras',exc)
                    self.cameras=[]
                finally:
                    self.worker.close()
            if self.metadata.get('native_isaacsim_app'):
                audit=audit_native_episode(self.directory/'private')
                (self.directory/'private/physics_audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
                physics_valid=physics_valid and audit['physics_valid']
                if not audit['physics_valid']:
                    if audit['status']=='unchecked':
                        outcome='infrastructure_error'
                    elif outcome=='success':
                        outcome='safety_stop'
                    fields.pop('completion_sim_s',None)
                    fields.pop('completion_wall_s',None)
        except Exception as exc:
            record_finalize_error('native_finalize',exc)
            try:
                self.worker.close(force=True)
            except Exception as close_exc:
                record_finalize_error('force_close',close_exc)
        # Native transport can fail after a physical step but before its reply.
        # Keep raw at the last acknowledged observation; a missing final image is
        # explicit, never relabelled with the raw timestamp or reused as current.
        terminal_snapshot=None
        published_images=dict(self.public_images)
        published_depths=dict(self.public_depths)
        try:
            if any(abs(camera.sim_time_s-self.raw['sim_time_s'])>1e-9 for camera in self.cameras):
                record_finalize_error('terminal_observation',RuntimeError('Discarded stale terminal cameras'))
                self.cameras=[]
            terminal_snapshot=self._projection()
        except Exception as exc:
            # Invalid raw joints, FK or tactile data must not prevent sealing an
            # invalid result. Keep the last legal public snapshot exactly as it
            # was; do not append it as a newly captured terminal observation.
            self.public_images=published_images
            self.public_depths=published_depths
            record_finalize_error('terminal_projection',exc)
            (self.directory/'private/terminal_observation_missing.json').write_text(json.dumps({
                'terminal_observation_missing':True,
                'attempted_sequence':self.sequence,
                'acknowledged_sim_time_s':fields['elapsed_sim_s'],
                'last_public_observation_sequence':self.snapshot.sequence,
                'last_public_observation_sim_time_s':self.snapshot.sim_time_s,
                'last_public_observation_preserved':True,
                'terminal_observation_appended':False},indent=2,allow_nan=False)+'\n')
        if finalization_errors:
            outcome='infrastructure_error'
            physics_valid=False
            fields.pop('completion_sim_s',None)
            fields.pop('completion_wall_s',None)
            (self.directory/'private/finalize_error.json').write_text(json.dumps({
                'errors':finalization_errors,'terminal_camera_count':len(self.cameras),
                'acknowledged_sim_time_s':fields['elapsed_sim_s'],
                'terminal_observation_missing':terminal_snapshot is None},indent=2,allow_nan=False)+'\n')
        scoring_contract = (self.public_task or {}).get('scoring_contract')
        latest_scoring = project_terminal_scoring(self.score, scoring_contract,
            outcome=outcome, physics_valid=physics_valid)
        # A late ACK cannot improve the budget-limited record, but its evidence
        # fault must still revoke validity (even when a wall timeout won the
        # existing terminal-outcome race).
        scoring = project_terminal_scoring(self.terminal_score_receipt, scoring_contract,
            outcome=outcome, physics_valid=physics_valid and latest_scoring is not None
            and latest_scoring.score_valid)
        self.result=EpisodeResult(spec=self.spec,outcome=outcome,physics_valid=physics_valid,scoring=scoring,
                                 evaluator_evidence_sha256=digest(score_path),
                                 failure_detail='terminal_observation_missing' if terminal_snapshot is None else '',**fields)
        if terminal_snapshot is not None:
            self.snapshot=terminal_snapshot
            self.recorder.observation(self.snapshot)
        self.recorder.finish(self.result)
        if self.player_session is not None:
            (self.directory/f'agent/{self.player_session.kind}_result.json').write_text(json.dumps(
                self.player_session.result(next(iter(self.player_session.bindings))),indent=2,allow_nan=False)+'\n')
        paths=[p for p in self.directory.rglob('*') if p.is_file()]
        (self.directory/'artifact_hashes.json').write_text(json.dumps({str(p.relative_to(self.directory)):digest(p) for p in paths},indent=2)+'\n')
        self.done.set()
        return self.result

    def finish(self):
        with self.lock:
            return self._finish('cancelled').model_dump(mode='json')


class AgentService:
    def __init__(self,directory,*,release=DEFAULT,device='cuda:0',registry=None,worker_factory=NativeWorker,
                 allow_privileged=False,evaluation_plan=None):
        self.directory=Path(directory)
        self.release=Path(release)
        self.device=device
        if registry is None and worker_factory is NativeWorker:
            from .controllers import default_registry
            registry=default_registry(self.release)
        self.registry=registry or PolicyRegistry()
        self.worker_factory=worker_factory
        self.allow_privileged=allow_privileged
        self.lock=threading.RLock()
        self.sessions={}
        self.evaluation_plan=evaluation_plan.model_copy(deep=True) if evaluation_plan else None
        self.plan_slots={s.episode_id:s for s in self.evaluation_plan.episodes} if self.evaluation_plan else {}

    def create(self,request,*,board_game=False,turn_game=False):
        with self.lock:
            if frontend_sources()!=IMPORTED_FRONTEND_SOURCES:
                raise ServiceError(503,'Runner source changed after import; restart the service before evaluation')
            if any(s.result is None for s in self.sessions.values()):
                raise ServiceError(409,'This server owns one active native episode')
            if any(s.thread is not None and s.thread.is_alive() for s in self.sessions.values()):
                raise ServiceError(409,'The previous policy is still stopping; wait for its worker thread to return')
            expected=None
            if self.evaluation_plan:
                if request.plan_slot_id not in self.plan_slots:
                    raise ServiceError(422,'Select an operator-pinned plan slot')
                expected=self.plan_slots[request.plan_slot_id]
                episode_id=expected.episode_id
                if episode_id in self.sessions:
                    raise ServiceError(409,'Plan slot already ran in this service')
                if not episode_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in episode_id):
                    raise ServiceError(422,'Plan slot IDs must be filesystem-safe opaque identifiers')
            else:
                if request.plan_slot_id is not None:
                    raise ServiceError(422,'This server has no registered evaluation plan')
                episode_id=uuid.uuid4().hex
            if (request.task_id == 'task0' and 'seed' not in request.model_fields_set
                    and validate(self.release,assets=False).get('schema_version') ==
                    'isaacsim-card-room-placement-release-v1'):
                request=request.model_copy(update={'seed':expected.seed if expected is not None
                                                  else secrets.randbelow(2**32)})
            session=EpisodeSession(request,self.directory/episode_id,release=self.release,device=self.device,
                registry=self.registry,worker_factory=self.worker_factory,allow_privileged=self.allow_privileged,
                expected_spec=expected,board_game=board_game,turn_game=turn_game)
            self.sessions[episode_id]=session
            result={'episode_id':episode_id,'spec':session.spec.model_dump(mode='json'),
                    'capabilities':session.capabilities.model_dump(mode='json'),
                    'observation':session.observe()}
            if session.public_task is not None:
                result['task']=session.task_requirements()
            if session.player_session is not None:
                player_session=session.player_session
                result.update(players=player_session.credentials(),
                              **{player_session.kind:player_session.state.model_dump(mode='json')},
                              **deepcopy(player_session.static), reference=None)
            return result

    def _get(self,episode_id):
        try:
            return self.sessions[episode_id]
        except KeyError as exc:
            raise ServiceError(404,'Unknown episode') from exc

    def get(self,episode_id):
        session=self._get(episode_id)
        if session.player_session is not None:
            raise ServiceError(403,'Player games require authenticated player routes')
        return session

    def player(self,episode_id,player_id,token,*,kind=None):
        session=self._get(episode_id)
        if session.player_session is None:
            raise ServiceError(404,'Episode is not a player game')
        if kind is not None and session.player_session.kind != kind:
            raise ServiceError(404,'Incorrect game entrypoint')
        return session.player_session.authenticate(player_id,token)

    def media_session(self,episode_id,token):
        session=self._get(episode_id)
        if session.player_session is not None:
            session.player_session.authenticate_media(token)
            return session
        return self.get(episode_id)

    def shutdown(self):
        for session in self.sessions.values():
            session.finish()
