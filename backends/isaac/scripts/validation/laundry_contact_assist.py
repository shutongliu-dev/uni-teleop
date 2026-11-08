"""Explicit local assistance after measured tabletop contact/capture prerequisites.

The assistance follows the measured wrist at the start of each native step.
It supplies a local holding constraint, not a friction-force measurement. No
cloth state, robot drive, mesh, or collision filter is changed here.
"""
from __future__ import annotations

import json
import re

import numpy as np

from bench.evaluation.cloth.cloth_robot_collision import transform_xyzw
from scripts.validation.laundry_contact_stability import PROFILE as PRACTICAL_PROFILE
from scripts.validation.laundry_contact_engagement import (
    PROFILE as ENGAGED_PROFILE, ContactEngagement, engagement_constraint_result)


def capture_constraint_vertices(result, triangles, mode):
    """Keep the verified patch or its actual shared material edge; never remap."""
    if mode not in ('captured_patch','shared_edge') or not (result.get('capture_candidate')
            or (mode == 'captured_patch' and engagement_constraint_result(result))):
        raise ValueError('An explicit mode and verified captured patch are required')
    captured=list(map(int,result['vertex_ids']))
    if not 3<=len(captured)<=32 or len(set(captured))!=len(captured):
        raise ValueError('Invalid captured patch inventory')
    if mode=='captured_patch':return captured
    tri=np.asarray(triangles,np.int64);pair=result['evidence']['triangle_ids']
    if len(pair)!=2 or pair[0]==pair[1] or min(pair)<0 or max(pair)>=len(tri):
        raise ValueError('The original two witnessed triangles are required')
    ids=np.intersect1d(tri[pair[0]],tri[pair[1]]).tolist()
    if len(ids)!=2 or not set(ids).issubset(captured):
        raise ValueError('The constraint must be the witnessed shared material edge')
    return ids


class CapturedPatchAssist:
    def __init__(self, env, scene, side, collider_export, output, *, name=None,
                 allowed_material_vertex_ids=None, cloth_id=None, capture_profile='strict_v1',
                 attachment_material_mode='captured_patch', contact_closure_stop=False,
                 contact_closure_target_policy='reduce_preload_v1',
                 contact_closure_geometry_profile='shared_edge_v1',
                 closure_measurement_profile='strict_v1', planned_open_tip_gap_m=None,
                 material_capture_region=None,contact_normal_profile='facet_v1',
                 contact_closure_observation_policy='close_only_v1', contact_witness_profile='centroid_v1',
                 pivot_material_reference=None):
        from scripts.validation.laundry_contact_observation import validate_normal_profile,validate_observation_policy
        validate_normal_profile(contact_normal_profile)
        from scripts.validation.laundry_contact_observation import validate_witness_profile
        validate_witness_profile(contact_witness_profile, contact_normal_profile)
        validate_observation_policy(contact_closure_observation_policy,contact_closure_stop,contact_closure_target_policy)
        self.closure_observation_policy=contact_closure_observation_policy
        from scripts.validation.laundry_closure_measurement import validate_closure_measurement
        validate_closure_measurement(closure_measurement_profile, planned_open_tip_gap_m)
        from scripts.validation.laundry_capture_gate import CaptureGate
        from scripts.validation.laundry_pinch_geometry import PinchGeometry
        self.env, self.side = env, side
        if type(contact_closure_stop) is not bool:
            raise ValueError('contact_closure_stop must be an explicit boolean')
        if (contact_closure_target_policy not in ('reduce_preload_v1','hold_command_v2') or
                (not contact_closure_stop and contact_closure_target_policy!='reduce_preload_v1')):
            raise ValueError('Explicit contact target policy requires enabled contact closure stop')
        if attachment_material_mode not in ('captured_patch','shared_edge','captured_pivot_v1'):
            raise ValueError('Unknown local assistance material mode')
        if pivot_material_reference is not None and attachment_material_mode != 'captured_pivot_v1':
            raise ValueError('A material pivot reference requires point-assist mode')
        from scripts.validation.laundry_capture_settings import validate_loaded_contact
        validate_loaded_contact(capture_profile, contact_closure_stop,
                                contact_closure_target_policy, contact_closure_geometry_profile,
                                attachment_material_mode)
        if (contact_closure_geometry_profile not in ('shared_edge_v1','adjacent_patch_v3','angled_adjacent_patch_v4') or
                (contact_closure_geometry_profile in ('adjacent_patch_v3','angled_adjacent_patch_v4') and
                 (not contact_closure_stop or attachment_material_mode not in ('captured_patch','captured_pivot_v1') or
                  capture_profile!=contact_closure_geometry_profile and not (
                      capture_profile in ('loaded_adjacent_patch_v5', PRACTICAL_PROFILE, ENGAGED_PROFILE)
                      and contact_closure_geometry_profile=='adjacent_patch_v3')))):
            raise ValueError('Bridge closure geometry requires enabled stop, adjacent capture and captured-patch material')
        if capture_profile=='angled_adjacent_patch_v4':
            if (attachment_material_mode not in ('captured_patch','captured_pivot_v1') or
                    (contact_closure_stop and contact_closure_geometry_profile!=capture_profile)):
                raise ValueError('Angled capture requires captured-patch material and matching stop geometry')
        self.geometry_options=({'engineering_profile':capture_profile}
                               if capture_profile=='angled_adjacent_patch_v4' else {})
        if contact_normal_profile!='facet_v1':self.geometry_options['contact_normal_profile']=contact_normal_profile
        if contact_witness_profile!='centroid_v1':self.geometry_options['contact_witness_profile']=contact_witness_profile
        self.attachment_material_mode=attachment_material_mode
        if name is not None and not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,63}', name):
            raise ValueError('Assistance name must be one safe USD path component')
        self.name = name
        self.namespace = '/World/MeasuredCaptureAssist'+('/'+name if name else '')
        self.allowed_ids = (None if allowed_material_vertex_ids is None
                            else set(map(int, allowed_material_vertex_ids)))
        self.spec = (scene['cloths'][0] if cloth_id is None else
                     next(c for c in scene['cloths'] if c['id']==cloth_id))
        self.cloth_id = self.spec['id']
        if 'rest_vertices_m' in self.spec and 'triangles' in self.spec:
            rest,triangles=self.spec['rest_vertices_m'],self.spec['triangles']
        else:
            from bench.simulation.isaacsim_backend.geometry import cloth_grid
            rest,triangles=cloth_grid(self.spec)
        self.triangles = np.asarray(triangles, dtype=np.int64)
        self.rest = np.asarray(rest, dtype=float)
        self.set_pivot_material_reference(pivot_material_reference)
        self.thickness = float(self.spec['isaacsim']['cloth_thickness_m'])
        self.geometry = PinchGeometry(collider_export, env.body_names)
        self.engagement = ContactEngagement() if capture_profile == ENGAGED_PROFILE else None
        gate_profile = PRACTICAL_PROFILE if self.engagement is not None else capture_profile
        motion_options = ({'contact_motion_reference': self.geometry.contact_motion_reference(side)}
                          if gate_profile == PRACTICAL_PROFILE else {})
        self.gate = CaptureGate(side, self.triangles, self.rest, env.body_names, env.joint_names,
                                profile=gate_profile,
                                closure_measurement_profile=closure_measurement_profile,
                                planned_open_tip_gap_m=planned_open_tip_gap_m,
                                material_capture_region=material_capture_region,
                                contact_normal_profile=contact_normal_profile,
                                contact_witness_profile=contact_witness_profile, **motion_options)
        self.wrist_index = env.body_names.index(side+'_hand_wrist')
        self.thumb_index = env.body_names.index(side+'_thumb_fingertip')
        self.index_index = env.body_names.index(side+'_index_fingertip')
        self.open_tip_gap_m = None
        self.ids, self.events = [], []
        self.captured_patch_ids = []
        self.last_capture_diagnostic = None
        self.prim = self.translate = self.orient = None
        self.active = False
        self.anchor_pose = np.array([0., 0., 0., 0., 0., 0., 1.])
        self.last_observe_time = -np.inf
        self.closure_stop = None
        self.last_closure_observe_time = -np.inf
        self.closure_diagnostics = self.closure_diagnostics_path = None
        if contact_closure_stop:
            from scripts.validation.laundry_contact_closure_stop import ContactClosureStop
            self.closure_stop = ContactClosureStop(side, self.triangles, self.rest,
                env.body_names, env.joint_names, env.joint_limits,
                actuated_joint_names=env.actuated_joint_names,target_policy=contact_closure_target_policy,
                geometry_profile=contact_closure_geometry_profile,
                closure_measurement_profile=closure_measurement_profile,
                planned_open_tip_gap_m=planned_open_tip_gap_m,
                material_capture_region=material_capture_region,
                contact_normal_profile=contact_normal_profile,
                contact_closure_observation_policy=contact_closure_observation_policy,
                contact_witness_profile=contact_witness_profile)
            self.closure_diagnostics_path = output/('closure_stop_diagnostics'+('_'+name if name else '')+'.jsonl')
            self.closure_diagnostics = self.closure_diagnostics_path.open('x')
        self.diagnostics_path = output/('capture_diagnostics'+('_'+name if name else '')+'.jsonl')
        self.diagnostics = self.diagnostics_path.open('x')

    def observe_closure(self, phase, time_s, state, cloth_q, targets):
        """Propose original position targets from fresh contact; never attach."""
        policy=getattr(self,'closure_observation_policy','close_only_v1')
        phases=('close','closed_hold') if policy=='close_and_hold_v1' else ('close',)
        if (self.closure_stop is None or self.closure_stop.latched or phase not in phases
                or self.active or self.prim is not None or self.open_tip_gap_m is None
                or time_s-self.last_closure_observe_time < .019999):
            return False
        self.last_closure_observe_time = float(time_s)
        diagnostic = self.geometry.diagnose(self.side, state['robot_body_q'], cloth_q,
                                            self.triangles, self.rest, self.thickness, **self.geometry_options)
        result = self.closure_stop.observe(time_s, diagnostic, state['robot_body_q'],
            state['joint_q'], targets, self.open_tip_gap_m, cloth_q=cloth_q,
            allowed_material_vertex_ids=self.allowed_ids,
            **({'observation_phase':phase} if policy!='close_only_v1' else {}))
        if result['latched'] and policy!='close_only_v1':
            # A late stop cannot inherit an earlier unlatched capture window.
            # The same actual sample starts a fresh100ms stability history.
            self.gate.history.clear()
            self.last_observe_time=-np.inf
        self.closure_diagnostics.write(json.dumps({'time_s': float(time_s), 'phase': phase,
            'contact_closure_stop': result, 'pinch_geometry': diagnostic})+'\n')
        self.closure_diagnostics.flush()
        return result['latched']

    def held_joint_targets(self, phase):
        """Only original actuated fingers; keep the first stop through transport."""
        if (self.closure_stop is None or not self.closure_stop.latched
                or phase in ('open', 'release_exit', 'retreat', 'free')):
            return {}
        return dict(self.closure_stop.receipt['finger_joint_targets_rad'])

    def opening_joint_targets(self, postures, start_targets, amount):
        """Open directly from the commands actually held, without re-closing."""
        if self.closure_stop is None or not self.closure_stop.latched:
            return {}
        if not np.isfinite(amount) or not 0. <= amount <= 1.:
            raise ValueError('Opening interpolation must lie in [0, 1]')
        opened = postures[0]['hand_posture_rad']
        if not set(self.closure_stop.receipt['finger_joint_targets_rad']).issubset(opened):
            raise ValueError('Opening posture omits an original stopped finger joint')
        if not all(n in start_targets and np.isfinite([start_targets[n], v]).all()
                   for n, v in opened.items()):
            raise ValueError('Finite original held and open hand targets required')
        return {n: float(start_targets[n]+amount*(v-start_targets[n])) for n, v in opened.items()}

    def opening_postures(self, postures, start_targets):
        """Use the same stopped-to-open path for cooked-hand clearance preview."""
        if self.closure_stop is None or not self.closure_stop.latched:
            return postures
        return [{'hand_posture_rad': self.opening_joint_targets(postures, start_targets, u)}
                for u in np.linspace(0., 1., max(61, len(postures)))]

    def _pose(self, pose):
        from pxr import Gf
        self.anchor_pose = np.asarray(pose, float).copy()
        x, y, z, w = self.anchor_pose[3:]
        self.translate.Set(Gf.Vec3d(*map(float, self.anchor_pose[:3])))
        self.orient.Set(Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z))))

    def update(self, phase, time_s, state, cloth_q, targets):
        """Read actual pre-step state; constrain only a qualified local patch."""
        bodies = state['robot_body_q']
        gap = float(np.linalg.norm(bodies[self.thumb_index, :3]-bodies[self.index_index, :3]))
        if phase == 'descend':
            self.open_tip_gap_m = gap
        if getattr(self, 'engagement', None) is not None:
            try:
                return self._update_engagement(phase, time_s, state, cloth_q, targets, gap)
            except BaseException:
                if self.active:
                    self.engagement.fail('engagement_update_exception')
                    self.release(time_s, 'Engagement update failed; disable before propagating the error.')
                raise
        if self.active:
            if phase in ('open', 'retreat', 'free'):
                self.release(time_s, 'Opening phase releases the local constraint before stepping.')
            else:
                self._pose(bodies[self.wrist_index])
            return
        stopped_close = (phase == 'close' and self.closure_stop is not None and self.closure_stop.latched)
        if (self.prim is not None or (phase != 'closed_hold' and not stopped_close) or self.open_tip_gap_m is None
                or time_s-self.last_observe_time < .019999):
            return
        self.last_observe_time = float(time_s)
        diagnostic = self.geometry.diagnose(self.side, bodies, cloth_q, self.triangles,
                                             self.rest, self.thickness, **self.geometry_options)
        result = self.gate.observe(time_s, diagnostic, bodies, state['joint_q'],
            state['joint_qd'], targets, self.open_tip_gap_m, cloth_q=cloth_q,
            **({'allowed_material_vertex_ids':self.allowed_ids} if self.gate.material_region is not None else {}))
        if (self.gate.profile=='loaded_adjacent_patch_v5' and not self.closure_stop.latched):
            self.gate.history.clear()
            result.update(capture_candidate=False, vertex_ids=[], continuous_duration_s=0.)
            result['reasons'].append('loaded_capture_requires_actual_latched_closure_stop')
        if (self.gate.profile == PRACTICAL_PROFILE and not self.closure_stop.latched):
            self.gate.history.clear()
            result.update(capture_candidate=False, vertex_ids=[], continuous_duration_s=0.)
            result['reasons'].append('practical_capture_requires_actual_latched_closure_stop')
        if (result['capture_candidate'] and self.gate.material_region is None and self.allowed_ids is not None
                and not set(result['vertex_ids']).issubset(self.allowed_ids)):
            result['capture_candidate'] = False
            result['reasons'].append('captured_material_outside_declared_edge_neighbourhood')
        self.diagnostics.write(json.dumps({'time_s': time_s, 'phase': phase,
            'actual_tip_gap_m': gap, 'capture_gate': result, 'pinch_geometry': diagnostic})+'\n')
        self.diagnostics.flush()
        if not result['capture_candidate']:
            return
        if self.attachment_material_mode == 'captured_pivot_v1':
            self.last_capture_diagnostic = diagnostic
        self._enable(time_s, result, bodies, cloth_q)

    @property
    def transport_ready(self):
        protocol = getattr(self, 'engagement', None)
        return self.active and (protocol is None or protocol.confirmed)

    def _engagement_failure(self, time_s, reason):
        self.engagement.fail(reason)
        self.release(time_s, reason)
        self.events.append({'time_s': float(time_s), 'event': 'engagement_failed',
                            'contact_engagement': self.engagement.receipt()})
        raise RuntimeError('Provisional contact assistance failed: '+reason)

    def _update_engagement(self, phase, time_s, state, cloth_q, targets, gap):
        protocol = self.engagement
        bodies = state['robot_body_q']
        if self.active and phase in ('open', 'release_exit', 'retreat', 'free'):
            if not protocol.confirmed:
                self._engagement_failure(time_s, 'opening_before_assisted_confirmation')
            self.release(time_s, 'Opening releases confirmed engineering hold before stepping.')
            return
        if self.active and protocol.confirmed:
            self._pose(bodies[self.wrist_index])
            return
        if self.active and phase not in ('close', 'closed_hold'):
            self._engagement_failure(time_s, 'movement_before_assisted_confirmation')
        if protocol.expired(time_s):
            self._engagement_failure(time_s, 'assisted_confirmation_timeout')
        if self.active:
            if not np.array_equal(np.asarray(targets)[self.gate.fingers], self._engagement_targets):
                self._engagement_failure(time_s, 'issued_finger_targets_changed_after_engagement')
            self._pose(bodies[self.wrist_index])  # Every native step, including between observations.
        stopped = self.closure_stop is not None and self.closure_stop.latched
        if (phase == 'close' and not stopped):
            return
        if (phase not in ('close', 'closed_hold') or self.open_tip_gap_m is None
                or time_s-self.last_observe_time < .019999
                or (self.prim is not None and not self.active)):
            return
        self.last_observe_time = float(time_s)
        diagnostic = self.geometry.diagnose(self.side, bodies, cloth_q, self.triangles,
                                             self.rest, self.thickness, **self.geometry_options)
        result = self.gate.observe(time_s, diagnostic, bodies, state['joint_q'],
            state['joint_qd'], targets, self.open_tip_gap_m, cloth_q=cloth_q,
            **({'allowed_material_vertex_ids': self.allowed_ids}
               if self.gate.material_region is not None else {}))
        if not stopped:
            self.gate.history.clear()
            result.update(capture_candidate=False, vertex_ids=[], continuous_duration_s=0.)
            result['reasons'].append('engagement_requires_actual_latched_closure_stop')
        if (result['vertex_ids'] and self.allowed_ids is not None
                and not set(result['vertex_ids']).issubset(self.allowed_ids)):
            self.gate.history.clear()
            result['capture_candidate'] = False
            result['reasons'].append('captured_material_outside_declared_edge_neighbourhood')
        deep = bool(diagnostic.get('distal_group_severe_penetration')
                    or diagnostic.get('severe_penetration_pads'))
        action, receipt = protocol.observe(time_s, result, latched=stopped, severe_penetration=deep)
        self.diagnostics.write(json.dumps({'time_s': float(time_s), 'phase': phase,
            'actual_tip_gap_m': gap, 'capture_gate': result,
            'contact_engagement': receipt, 'pinch_geometry': diagnostic})+'\n')
        self.diagnostics.flush()
        if action == 'fail':
            self._engagement_failure(time_s, protocol.failure_reason)
        if action == 'engage':
            self._engagement_targets = np.asarray(targets)[self.gate.fingers].copy()
            self._enable(time_s, receipt, bodies, cloth_q)
            self.events[-1]['contact_engagement'] = receipt
            self.gate.history.clear()  # No pre-assistance sample enters confirmation.
        elif action == 'confirm':
            self.events.append({'time_s': float(time_s), 'event': 'engagement_confirmed',
                'contact_engagement': receipt, 'vertex_ids': list(self.ids),
                'confirmation_under_declared_assistance': True})
        if self.active:
            self._pose(bodies[self.wrist_index])

    def _enable(self, time_s, result, bodies, cloth_q):
        pivot = None
        if self.attachment_material_mode == 'captured_pivot_v1':
            from scripts.validation.laundry_capture_pivot import select_captured_pivot
            ids, pivot = select_captured_pivot(result, self.triangles, self.rest, cloth_q, bodies,
                self.last_capture_diagnostic, self.geometry, side=self.side, time_s=time_s,
                **({'material_reference': self.pivot_material_reference}
                   if getattr(self, 'pivot_material_reference', None) is not None else {}))
        else:
            ids = capture_constraint_vertices(result,self.triangles,self.attachment_material_mode)
        from pxr import UsdGeom
        from scripts.validation.laundry_cloth_interactions import attachment
        import omni.physx
        anchor_path = self.namespace+'/anchor'
        xform = UsdGeom.Xform.Define(self.env.stage, anchor_path)
        self.translate = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        self.orient = xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        self._pose(bodies[self.wrist_index])
        matrix = transform_xyzw(self.anchor_pose)
        offsets = (np.asarray(cloth_q)[ids]-matrix[:3, 3]) @ matrix[:3, :3]
        cloth_path = self.env.scene_info['cloths'][self.cloth_id]['path']
        self.prim = attachment(self.env.stage, self.namespace+'/attachment',
                               cloth_path, anchor_path, ids, offsets)
        self.ids, self.active = ids, True
        self.captured_patch_ids = list(result['vertex_ids'])
        omni.physx.get_physx_simulation_interface().flush_changes()
        self.events.append({'time_s': float(time_s), 'event': 'enable', 'vertex_ids': ids,
            'namespace': self.namespace,
            'attachment_material_mode':self.attachment_material_mode,
            'captured_patch_vertex_ids':list(result['vertex_ids']),
            'captured_gate_evidence': result, 'wrist_body': self.side+'_hand_wrist',
            'actual_wrist_pose_xyzw': self.anchor_pose.tolist(),
            'fixed_local_offsets_m': offsets.tolist(),
            'mechanism': 'Declared local vertex constraint follows the actual pre-step wrist frame; no independent rolling.',
            'friction_grasp_proven': False})
        if pivot is not None:
            self.events[-1].update(pivot_selection=pivot,
                trajectory_reference_material_ids=list(self.captured_patch_ids),
                constraint_material_vertex_ids=list(ids))

    def set_pivot_material_reference(self, reference):
        from scripts.validation.laundry_pivot_reference import validate_pivot_reference
        if reference is not None and self.attachment_material_mode != 'captured_pivot_v1':
            raise ValueError('A material pivot reference requires point-assist mode')
        if getattr(self, 'active', False):
            raise ValueError('A captured material pivot reference cannot change while attached')
        validate_pivot_reference(reference, self.rest)
        self.pivot_material_reference = None if reference is None else json.loads(json.dumps(reference))

    def trajectory_reference_ids(self):
        """Pivot only changes its constraint; preserve the full capture-centroid arc."""
        if getattr(self, 'engagement', None) is not None and not self.transport_ready:
            raise RuntimeError('Provisional engagement cannot supply a transport reference')
        return list(self.captured_patch_ids if self.attachment_material_mode == 'captured_pivot_v1'
                    else self.ids)

    def release(self, time_s, reason):
        if not self.active:
            return
        import omni.physx
        self.prim.GetAttribute('omniphysics:attachmentEnabled').Set(False)
        omni.physx.get_physx_simulation_interface().flush_changes()
        self.active = False
        self.events.append({'time_s': float(time_s), 'event': 'disable', 'reason': reason,
            'native_attachment_enabled': bool(self.prim.GetAttribute('omniphysics:attachmentEnabled').Get())})

    def mask(self, count):
        if self.prim is not None:
            enabled = bool(self.prim.GetAttribute('omniphysics:attachmentEnabled').Get())
            if enabled != self.active:
                raise RuntimeError('Native assistance flag disagrees with the recorded control mask')
            actual_ids = list(map(int, self.prim.GetAttribute('omniphysics:vtxIndicesSrc0').Get()))
            if actual_ids != self.ids:
                raise RuntimeError('Native assistance material identity changed')
        value = np.zeros(count, dtype=np.uint8)
        if self.active:
            value[self.ids] = 1
        return value

    def close(self):
        self.diagnostics.close()
        if self.closure_diagnostics:
            self.closure_diagnostics.close()
