"""Trusted callable bench operations; HTTP and Python hosts share this dispatch.

This is an episode-bound service facade, not an Agent SDK or task strategy.
All robot actions enter EpisodeSession.submit and its existing budget/ACK ledger.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re

import numpy as np
from bench.geometry import quaternion_matrix
from bench.protocol.functions import (ACTION_FUNCTIONS, CALL_ADAPTER, REPLY_MODELS,
    FUNCTION_INPUTS, FUNCTION_OUTPUTS, BudgetState, CallContext,
    FunctionCapability, FunctionCatalog, FunctionError, FunctionFailure)
from bench.protocol.runtime import BudgetExceeded
from bench.sdk.depth import decode_metric_depth_npy
from bench.sdk.geometry import camera_z_to_world


class FunctionCallError(Exception):
    def __init__(self, code, message, status=422):
        self.error = FunctionError(code=code, message=message, http_status=status)
        super().__init__(message)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                   allow_nan=False).encode()).hexdigest()


def public_failure(*, code, message, status, function=None, request_id=None, context=None):
    if not isinstance(function, str) or function not in FUNCTION_INPUTS:
        function = None
    if not isinstance(request_id, str) or re.fullmatch(r'[A-Za-z0-9_.:/-]{1,160}', request_id) is None:
        request_id = None
    return FunctionFailure(function=function, request_id=request_id, context=context,
        error=FunctionError(code=code, message=message, http_status=status))


class EpisodeFunctions:
    """One session's named operations, immutable action retries and call records."""

    def __init__(self, session):
        self.session = session
        self._prepared_actions = {}
        self._submitted = False

    def catalog(self):
        session = self.session
        with session.lock:
            return FunctionCatalog(robot_capabilities=session.capabilities,
                control_track=session.spec.track,
                functions=[FunctionCapability(name=name, available=(reason is None),
                    unavailable_reason=reason,
                    input_schema=FUNCTION_INPUTS[name].__name__,
                    output_schema=FUNCTION_OUTPUTS[name].__name__,
                    advances_physics=name in ACTION_FUNCTIONS,
                    consumes_action_attempt=name in ACTION_FUNCTIONS,
                    asynchronous_action=name in ACTION_FUNCTIONS)
                    for name in FUNCTION_INPUTS for reason in [self._unavailable(name)]]).model_copy(deep=True)

    def _unavailable(self, name):
        session = self.session
        if name == 'get_task' and getattr(session, 'referee_contract', None) is None:
            return 'This task does not publish a structured physical referee contract.'
        if name == 'query_ik' and session.robot_geometry is None:
            return 'Robot geometry is unavailable in this episode.'
        if name in ('move_geometry', 'move_wrist', 'set_hand_targets'):
            policy = session.capabilities.system0
            if (session.spec.track not in ('system0', 'free_tool_agent') or policy is None
                    or 'move_geometry' not in policy.skills):
                return 'This episode does not enable geometric execution.'
            if name == 'move_wrist' and 'wrist_pose' not in policy.geometry_inputs:
                return 'This executor does not accept wrist targets.'
            if name == 'set_hand_targets' and not {'hand_posture', 'fingertip_positions'} & set(policy.geometry_inputs):
                return 'This executor does not accept hand targets.'
        if name == 'move_joints' and session.spec.track not in ('direct_joints', 'free_tool_agent'):
            return 'This episode does not enable direct joint execution.'
        if name == 'hold' and session.spec.track == 'system0' and 'hold' not in session.capabilities.system0.skills:
            return 'This executor does not expose timed hold.'
        if (name == 'hold' and session.spec.track == 'direct_joints'
                and session.capabilities.direct_joint_execution == 'step_target_v1'):
            return 'This worker has no acknowledged drive reference for timed hold.'
        if name == 'read_camera' and not session.capabilities.camera_ids:
            return 'No public camera is configured.'
        if name == 'project_pixel' and not session.capabilities.depth_available:
            return 'Aligned metric depth is unavailable.'
        if name == 'hand_preset':
            from bench.control.hand_presets import hand_preset_catalog
            if not hand_preset_catalog(session.robot_geometry).available:
                return 'Validated hand preset parameters are unavailable for this robot.'
        return None

    def _context(self, recorded):
        session = self.session
        return CallContext(episode_id=session.spec.episode_id,
            observation_sequence=session.snapshot.sequence,
            observation_sim_time_s=session.snapshot.sim_time_s,
            episode_finished=session.result is not None,
            budget=BudgetState(limits=session.spec.budgets,
                action_attempts=session.clock.action_attempts,
                elapsed_sim_s=session.clock.sim_time_s, elapsed_wall_s=session.clock.wall_time_s,
                remaining_wall_s=session.clock.remaining_wall_s), request_recorded=recorded)

    def _record(self, value):
        recorder = self.session.recorder
        if recorder.closed:
            return False
        recorder.function_call(value)
        return True

    def invoke(self, payload):
        """Return a typed result; never auto-retry, reset or perform a hidden action.

The lock serializes admission and query reads. Asynchronous actions release it
between native control ticks using the existing EpisodeSession execution loop.
        """
        from .service import ServiceError
        session = self.session
        if hasattr(payload, 'model_dump'):
            payload = payload.model_dump(mode='json')
        raw = payload if isinstance(payload, dict) else {}
        name, request_id = raw.get('function'), raw.get('request_id')
        if not isinstance(name, str):
            name = None
        with session.lock:
            self._submitted = False
            recorded = False
            try:
                call = CALL_ADAPTER.validate_python(payload)
                name, request_id = call.function, call.request_id
            except (ValueError, TypeError):
                recorded = self._record({'phase': 'request', 'function': name if name in FUNCTION_INPUTS else None,
                    'request_id': request_id if isinstance(request_id, str) and len(request_id) <= 160 else None,
                    'valid': False})
                error = FunctionCallError('invalid_arguments', 'Request violates the function input schema.')
                if name not in FUNCTION_INPUTS:
                    error = FunctionCallError('unsupported_function', 'Unknown bench function.')
                return self._failed(error, name, request_id, recorded, raw)
            normalized = call.model_dump(mode='json')
            recorded = self._record({'phase': 'request', 'request': normalized, 'valid': True})
            try:
                reason = self._unavailable(name)
                if reason is not None:
                    raise FunctionCallError('unsupported_capability', reason)
                # Action admission owns action/simulation budgets. Public reads
                # remain available after termination, including the final result.
                if session.result is None and name not in ACTION_FUNCTIONS and name not in ('cancel_action', 'finish_episode'):
                    session.clock.check()
                result = getattr(self, '_' + name)(call.arguments)
                if session.result is None and name not in ACTION_FUNCTIONS and name not in ('cancel_action', 'finish_episode'):
                    session.clock.check()
                result = FUNCTION_OUTPUTS[name].model_validate(result)
                response = REPLY_MODELS[name](function=name, request_id=request_id,
                    result=result, context=self._context(recorded))
                self._record({'phase': 'response', 'response': response.model_dump(mode='json', exclude_unset=True)})
                return response.model_copy(deep=True)
            except BudgetExceeded as exc:
                if session.result is None:
                    session._finish(exc.outcome)
                error = FunctionCallError('budget_exhausted', exc.outcome, 409)
            except FunctionCallError as exc:
                error = exc
            except ServiceError as exc:
                error = FunctionCallError(exc.code, str(exc), exc.status)
            except Exception:
                logging.getLogger(__name__).exception('Private function execution failure')
                error = FunctionCallError('infrastructure_error', 'Bench function failed; inspect operator diagnostics.', 503)
            return self._failed(error, name, request_id, recorded, normalized)

    def _failed(self, error, name, request_id, recorded, payload):
        session = self.session
        # Malformed or unsupported motion attempts spend the same action budget
        # as malformed calls to the existing /actions endpoint.
        if name in ACTION_FUNCTIONS and not self._submitted and session.result is None:
            try:
                try:
                    fingerprint = _hash(payload)
                except (ValueError, TypeError):
                    fingerprint = None
                session._attempt(payload, fingerprint)
            except BudgetExceeded as exc:
                session._finish(exc.outcome)
                error = FunctionCallError('budget_exhausted', exc.outcome, 409)
        response = public_failure(code=error.error.code, message=error.error.message,
            status=error.error.http_status, function=name, request_id=request_id,
            context=self._context(recorded))
        self._record({'phase': 'response', 'response': response.model_dump(mode='json')})
        return response.model_copy(deep=True)

    def _snapshot(self, sequence):
        session = self.session
        if sequence != session.snapshot.sequence:
            raise FunctionCallError('stale_observation', 'Query references a stale observation.', 409)
        if session.life.active_action_id is not None or (session.thread is not None and session.thread.is_alive()):
            raise FunctionCallError('episode_busy', 'Query requires a decision-boundary observation.', 409)
        return session.snapshot

    def _get_robot(self, args):
        return self.session.robot_guide()

    def _get_task(self, args):
        return self.session.task_requirements()

    def _observe(self, args):
        return self.session.observe()

    def _read_camera(self, args):
        snapshot = self._snapshot(args.observation_sequence)
        camera = next((camera for camera in snapshot.cameras if camera.camera_id == args.camera_id), None)
        if camera is None:
            raise FunctionCallError('not_found', 'Camera is absent from this observation.', 404)
        return {'observation_sequence': snapshot.sequence, 'camera': camera}

    def _project_pixel(self, args):
        camera = self._read_camera(args)['camera']
        if (camera.depth_uri is None or camera.intrinsics_3x3 is None or camera.world_from_camera is None
                or camera.depth_sim_time_s != camera.sim_time_s):
            raise FunctionCallError('unsupported_capability', 'This frame has no aligned metric depth.')
        if args.u >= camera.width or args.v >= camera.height:
            raise FunctionCallError('invalid_arguments', 'Pixel is outside the camera image.')
        match = re.fullmatch(r'/episodes/' + re.escape(self.session.spec.episode_id)
            + r'/depth/(0|[1-9][0-9]*)/' + re.escape(args.camera_id), camera.depth_uri)
        if match is None:
            raise RuntimeError('Invalid public depth resource binding')
        raw = self.session.read_public_depth(int(match.group(1)), args.camera_id)
        depth = decode_metric_depth_npy(raw, width=camera.width, height=camera.height)
        value = float(depth[args.v, args.u])
        if value <= 0:
            raise FunctionCallError('invalid_depth', 'Selected pixel has no valid depth measurement.')
        point = camera_z_to_world(u=args.u, v=args.v, depth_m=value,
            intrinsics_3x3=camera.intrinsics_3x3, world_from_camera=camera.world_from_camera)
        return {'observation_sequence': args.observation_sequence, 'sim_time_s': camera.sim_time_s,
                'camera_id': args.camera_id, 'pixel_uv': (args.u, args.v), 'depth_m': value,
                'point_m': point.tolist()}

    def _world_from_frame(self, frame, snapshot):
        if frame == 'world':
            return np.eye(4)
        if frame == 'robot_base' and self.session.robot_geometry is not None:
            return np.array(self.session.robot_geometry.base, copy=True)
        poses = {}
        proprio = snapshot.proprioception
        if proprio.geometry is not None:
            poses[proprio.geometry.fingertip_frame_id] = proprio.geometry.wrist
        for geometry in (proprio.hand_geometries or {}).values():
            poses[geometry.fingertip_frame_id] = geometry.wrist
        poses.update({'camera:' + camera.camera_id: camera.world_from_camera
                      for camera in snapshot.cameras if camera.world_from_camera is not None})
        if frame not in poses:
            raise FunctionCallError('unsupported_capability', 'Frame is not available in this observation.')
        pose = poses[frame]
        transform = np.eye(4)
        transform[:3, :3] = quaternion_matrix(pose.quaternion_wxyz)
        transform[:3, 3] = pose.position_m
        return transform

    def _transform_points(self, args):
        snapshot = self._snapshot(args.observation_sequence)
        source = self._world_from_frame(args.source_frame, snapshot)
        target = self._world_from_frame(args.target_frame, snapshot)
        relative = np.linalg.inv(target) @ source
        with np.errstate(over='ignore', invalid='ignore'):
            points = np.asarray(args.points_m) @ relative[:3, :3].T + relative[:3, 3]
        if not np.isfinite(points).all():
            raise FunctionCallError('invalid_arguments', 'Frame conversion produced nonfinite coordinates.')
        return {**args.model_dump(), 'points_m': points.tolist(), 'sim_time_s': snapshot.sim_time_s}

    def _query_ik(self, args):
        return self.session.query_ik(args)

    def _get_hand_presets(self, args):
        from bench.control.hand_presets import hand_preset_catalog
        return hand_preset_catalog(self.session.robot_geometry)

    def _hand_preset(self, args):
        from bench.control.hand_presets import generate_hand_preset
        try:
            return generate_hand_preset(self.session.robot_geometry, args)
        except ValueError as error:
            raise FunctionCallError('invalid_arguments', 'Preset is unsupported for this hand or parameter configuration.') from error

    def _submit(self, name, args, prepare):
        identity = _hash({'function': name, 'arguments': args.model_dump(mode='json')})
        previous = self._prepared_actions.get(args.action_id)
        if previous is not None and previous[0] == identity:
            payload = previous[1]
        else:
            # Function aliases may spell the same physical action differently.
            # The session compares normalized actions across every transport.
            payload = prepare()
        self._submitted = True
        result = self.session.submit(payload)
        # Cache only accepted actions; invalid requests cannot grow this cache.
        if args.action_id in self.session.requests:
            self._prepared_actions[args.action_id] = (identity, payload)
        return result

    def _move_geometry(self, args):
        return self._submit('move_geometry', args, lambda: {'kind': 'system0',
            'skill': 'move_geometry', **args.model_dump(mode='json', exclude_none=True)})

    def _one_hand_action(self, args, objective):
        if f'{args.hand}_hand_flange' not in self.session.capabilities.frame_ids:
            raise FunctionCallError('unsupported_capability', 'This robot does not expose the selected hand.')
        models = getattr(self.session.robot_geometry, 'sides', None)
        geometry = ({'hand_geometries': {args.hand: objective}} if models is not None else {'geometry': objective})
        return {'kind': 'system0', 'skill': 'move_geometry', 'action_id': args.action_id,
            'observation_sequence': args.observation_sequence, 'timeout_sim_s': args.timeout_sim_s, **geometry}

    def _move_wrist(self, args):
        return self._submit('move_wrist', args, lambda: self._one_hand_action(args, {
            'wrist': args.wrist.model_dump(mode='json'), 'fingertip_frame_id': f'{args.hand}_hand_flange',
            'position_tolerance_m': args.position_tolerance_m,
            'orientation_tolerance_rad': args.orientation_tolerance_rad}))

    def _set_hand_targets(self, args):
        return self._submit('set_hand_targets', args, lambda: self._one_hand_action(args, {
            'hand_posture_rad': args.hand_posture_rad,
            'fingertip_positions_m': args.fingertip_positions_m,
            'fingertip_frame_id': f'{args.hand}_hand_flange'}))

    def _move_joints(self, args):
        return self._submit('move_joints', args, lambda: {'kind': 'direct_joints', **args.model_dump(mode='json')})

    def _hold(self, args):
        def prepare():
            if args.mode == 'acknowledged_drive':
                policy = self.session.capabilities.system0
                if (self.session.spec.track == 'direct_joints' or policy is None
                        or args.mode not in policy.hold_modes):
                    raise FunctionCallError('unsupported_capability',
                        'This executor does not verify stationary acknowledged-drive holds.')
            if self.session.spec.track in ('system0', 'free_tool_agent'):
                return {'kind': 'system0', 'skill': 'hold', 'action_id': args.action_id,
                    'observation_sequence': args.observation_sequence,
                    'timeout_sim_s': args.duration_s, 'hold_duration_s': args.duration_s,
                    **({'hold_mode': args.mode} if args.mode != 'reference' else {})}
            previous = self.session.direct_hold_replays.get(args.action_id)
            if previous is not None:
                return {**previous, 'observation_sequence': args.observation_sequence,
                        'duration_s': args.duration_s}
            self._snapshot(args.observation_sequence)
            targets = self.session._acknowledged_drive_targets()
            if targets is None:
                raise FunctionCallError('unsupported_capability', 'Timed direct hold requires acknowledged drive targets.')
            return {'kind': 'direct_joints', 'action_id': args.action_id,
                'observation_sequence': args.observation_sequence, 'duration_s': args.duration_s,
                'arm_joint_positions_rad': {name: targets[name] for name in self.session.capabilities.arm_joints},
                'hand_joint_positions_rad': {name: targets[name] for name in self.session.capabilities.hand_joints}}
        return self._submit('hold', args, prepare)

    def _get_action(self, args):
        return self.session.status(args.action_id)

    def _cancel_action(self, args):
        return self.session.cancel(args.action_id)

    def _get_result(self, args):
        if self.session.result is None:
            raise FunctionCallError('result_pending', 'Episode is still running.', 409)
        return self.session.result

    def _finish_episode(self, args):
        return self.session.finish()
