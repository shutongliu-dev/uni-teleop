from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from bench.protocol import (ActionLifecycle, AgentObservation, BudgetExceeded,
    Budgets, CameraObservation, DirectJointAction, EpisodeClock, EpisodeRecorder, EpisodeResult,
    EpisodeSpec, EvaluationPlan, JointLimit, PolicyDescriptor, PolicyRegistry, PolicyStep,
    RobotCapabilities, RobotJointCommand, System0Action, aggregate_results,
    negotiate, parse_action, project_observation, project_policy_observation,
    schemas, validate_action, validate_joint_command)

H = "a" * 64


def rgbd_camera(**updates):
    payload = dict(camera_id='task',sim_time_s=1.0,rgb_uri='/images/1/task',rgb_sha256=H,
        width=8,height=6,depth_uri='/depth/1/task',depth_sha256=H,
        intrinsics_3x3=[5.,0.,4.,0.,5.,3.,0.,0.,1.],
        world_from_camera={'frame_id':'world','position_m':[1.,2.,3.],
                           'quaternion_wxyz':[1.,0.,0.,0.]},
        depth_encoding='float32_npy',depth_semantics='camera_z',depth_unit='m',
        depth_invalid_value=0.0,depth_rgb_aligned=True,depth_sim_time_s=1.0)
    return payload | updates


def test_rgbd_camera_is_explicit_and_legacy_rgb_remains_readable():
    camera = CameraObservation(**rgbd_camera())
    assert camera.depth_rgb_aligned and camera.depth_sim_time_s == camera.sim_time_s
    legacy = {key:value for key,value in rgbd_camera().items()
              if not key.startswith('depth_') or key=='depth_unit'}
    assert CameraObservation(**legacy).depth_uri is None


@pytest.mark.parametrize('field', ['camera_calibration', 'render_synchronization'])
def test_public_camera_contract_rejects_private_renderer_diagnostics(field):
    with pytest.raises(ValidationError, match='Extra inputs are not permitted'):
        CameraObservation(**rgbd_camera(**{field: {'private': True}}))


@pytest.mark.parametrize('field', ['depth_uri','depth_sha256','depth_encoding',
    'depth_semantics','depth_invalid_value','depth_sim_time_s','depth_rgb_aligned',
    'intrinsics_3x3','world_from_camera'])
def test_depth_cannot_omit_required_pair_metadata_or_calibration(field):
    payload = rgbd_camera()
    del payload[field]
    with pytest.raises(ValidationError):
        CameraObservation(**payload)


@pytest.mark.parametrize('updates', [dict(depth_sim_time_s=.98),dict(depth_invalid_value=-1.),
    dict(depth_invalid_value=float('nan')),dict(depth_unit='mm'),dict(depth_semantics='ray_distance'),
    dict(depth_encoding='png'),dict(depth_rgb_aligned=False),dict(depth_uri='')])
def test_depth_rejects_ambiguous_units_stale_frames_and_wrong_representation(updates):
    with pytest.raises(ValidationError):
        CameraObservation(**rgbd_camera(**updates))


def test_depth_metadata_without_an_image_is_rejected():
    with pytest.raises(ValidationError,match='metadata requires'):
        CameraObservation(**rgbd_camera(depth_uri=None,depth_sha256=None))


def test_observation_rejects_stale_or_duplicate_camera_frames():
    raw = {'sim_time_s':1.0,'joint_positions_rad':{'arm1':0.,'finger1':0.},
           'joint_velocities_rad_s':{'arm1':0.,'finger1':0.}}
    camera = CameraObservation(**rgbd_camera())
    assert project_observation(raw,spec(),0,cameras=[camera]).cameras[0] == camera
    with pytest.raises(ValidationError,match='timestamps'):
        project_observation(raw | {'sim_time_s':1.02},spec(),0,cameras=[camera])
    with pytest.raises(ValidationError,match='unique'):
        project_observation(raw,spec(),0,cameras=[camera,camera])


def spec(**updates):
    data = dict(episode_id="e0", agent_id="agent", agent_version="v1", agent_config_sha256=H,
        task_id="task", task_release_sha256=H, evaluator_sha256=H, robot_id="fr3_hand",
        robot_asset_sha256=H, physics_profile_sha256=H, runtime_code_sha256=H, sensor_config_sha256=H,
        execution_environment_sha256=H, scene_sha256=H, seed=0,
        track="direct_joints", instruction="Move the spool into the tray.")
    data.update(updates)
    return EpisodeSpec(**data)


def caps(**updates):
    data = dict(robot_id="fr3_hand", robot_asset_sha256=H, arm_joints=["arm1"], hand_joints=["finger1"],
                joint_limits={n: JointLimit(lower_rad=-2., upper_rad=2.) for n in ["arm1", "finger1"]})
    data.update(updates)
    return RobotCapabilities(**data)


def action(**updates):
    data = dict(action_id="a0", observation_sequence=0, duration_s=0.02,
                arm_joint_positions_rad={"arm1": 0.2}, hand_joint_positions_rad={"finger1": 0.3})
    data.update(updates)
    return DirectJointAction(**data)


def policy(**updates):
    data = dict(policy_id="authors_rl", version="v1", implementation_sha256=H,
                trained=False, skills=["GRASP"], geometry_inputs=["wrist_pose", "contacts", "hand_posture", "wrist_trajectory"])
    data.update(updates)
    return PolicyDescriptor(**data)


def result(success=False, **updates):
    data = dict(spec=spec(), outcome="success" if success else "task_failure",
                elapsed_sim_s=1., elapsed_wall_s=10.)
    if success:
        data.update(completion_sim_s=1., completion_wall_s=10., evaluator_evidence_sha256=H)
    data.update(updates)
    return EpisodeResult(**data)


def test_named_dynamic_robot_action():
    validate_action(action(), caps(), spec(), observation_sequence=0)
    # No FR3/29-DoF topology is hard-coded by the protocol.
    c = caps(hand_joints=["finger1", "finger2"], joint_limits={n: JointLimit(lower_rad=-1., upper_rad=1.) for n in ["arm1", "finger1", "finger2"]})
    validate_action(action(hand_joint_positions_rad={"finger1": .1, "finger2": .2}), c, spec())


@pytest.mark.parametrize("bad", [dict(object_pose=[0, 0, 0]), dict(cloth_vertices=[]), dict(python="module.Class"), dict(weights_path="/secret"), dict(duration_s=float("nan")), dict(duration_s=True)])
def test_mutation_and_code_payloads_rejected(bad):
    payload = action().model_dump() | bad
    with pytest.raises(ValidationError):
        parse_action(payload)


@pytest.mark.parametrize("updates", [dict(hand_joint_positions_rad={"wrong": 0.}), dict(arm_joint_positions_rad={"arm1": 9.}), dict(duration_s=.03), dict(duration_s=6.), dict(observation_sequence=1)])
def test_invalid_native_action(updates):
    with pytest.raises(ValueError):
        validate_action(action(**updates), caps(), spec(), observation_sequence=0)


def test_policy_outputs_are_only_complete_single_step_joint_targets():
    validate_joint_command(RobotJointCommand(joint_positions_rad={"arm1": 0., "finger1": 0.}, duration_s=.02), caps())
    with pytest.raises(ValueError):
        validate_joint_command(RobotJointCommand(joint_positions_rad={"arm1": 0.}, duration_s=.02), caps())
    with pytest.raises(ValueError):
        validate_joint_command(RobotJointCommand(joint_positions_rad={"arm1": 0., "finger1": 0.}, duration_s=.04), caps())


def test_dense_semantic_goal_and_capability_checks():
    p = policy()
    s = spec(track="system0", system0=p)
    c = caps(tracks=["direct_joints", "system0"], system0=p, finger_names=["thumb", "index"])
    a = System0Action(action_id="g0", observation_sequence=0, skill="GRASP", timeout_sim_s=2.,
        target={"object": "mug", "part": "handle"}, object_effect="hold",
        geometry={"contacts": [{"finger": "thumb", "role": "opposition", "contact": "required",
                                 "frame_id": "world", "position_m": [.1, .2, .3], "normal_xyz": [1., 0., 0.]}]})
    validate_action(a, c, s)
    a.geometry.contacts[0].frame_id = "oracle_object_frame"
    with pytest.raises(ValueError, match="frame"):
        validate_action(a, c, s)
    a.geometry.contacts[0].frame_id = "world"
    a.geometry.contacts[0].desired_normal_force_n = 1.
    with pytest.raises(ValueError, match="force"):
        validate_action(a, c, s)


def test_system0_advertisement_requires_installed_policy():
    with pytest.raises(ValidationError):
        caps(tracks=["system0"])
    with pytest.raises(ValidationError):
        spec(track="system0")
    with pytest.raises(ValueError, match="pin"):
        negotiate(spec(robot_asset_sha256="b" * 64), caps())
    with pytest.raises(ValidationError, match="checkpoint digest"):
        policy(trained=True)
    assert policy(trained=True, weights_sha256=H).weights_sha256 == H


def test_visibility_whitelist_and_private_teacher_view():
    raw = {"sim_time_s": 0., "joint_positions_rad": {"arm1": 0., "finger1": 0.},
           "joint_velocities_rad_s": {"arm1": 0., "finger1": 0.},
           "objects": {"mug": {"position_m": [1., 2., 3.], "quaternion_wxyz": [1., 0., 0., 0.]}},
           "cloth": {"vertices_m": [[1., 2., 3.]]},
           "metrics": {"success": True}, "goal_pose": [1, 2, 3],
           "contacts": {"privileged": "solver"}, "future_evaluator_secret": "hidden"}
    s = spec(track="system0", system0=policy(observation_profile="privileged_state"))
    public = project_observation(raw, s, 0).model_dump(exclude_none=True)
    assert "objects" not in public and "metrics" not in public and "goal_pose" not in public
    assert "privileged_object_poses" not in public and "contacts" not in public
    private = project_policy_observation(raw, s, 0)
    assert private.privileged_object_poses["mug"].position_m == [1., 2., 3.]
    assert "metrics" not in private.model_dump()
    with pytest.raises(ValidationError):
        AgentObservation.model_validate(public | {"privileged_cloth_vertices_m": [[0., 0., 0.]]})
    raw["joint_velocities_rad_s"] = {"arm1": 0.}
    with pytest.raises(ValidationError, match="inventories"):
        project_observation(raw, s, 1)


def test_wall_clock_includes_model_and_wait_and_timeout():
    now = [100.]
    clock = EpisodeClock(Budgets(wall_time_s=10.), monotonic=lambda: now[0])
    clock.begin_model_call()
    now[0] += 3.
    clock.end_model_call()
    now[0] += 2.
    clock.check(next_sim_dt=.02)
    clock.record_step(.02)
    assert clock.snapshot()["model_latency_s"] == 3.
    assert clock.wall_time_s == 5. and clock.sim_time_s == .02
    now[0] += 5.
    with pytest.raises(BudgetExceeded) as exc:
        clock.check()
    assert exc.value.outcome == "wall_timeout"
    clock.stop()
    now[0] += 9.
    assert clock.wall_time_s == 10.


def test_clock_action_and_model_budget_and_exact_final_step():
    clock = EpisodeClock(Budgets(sim_time_s=.04, max_actions=1, max_model_calls=1), monotonic=lambda: 0.)
    clock.action_attempt()
    with pytest.raises(BudgetExceeded, match="action_budget"):
        clock.action_attempt()
    clock.begin_model_call()
    clock.end_model_call()
    with pytest.raises(BudgetExceeded, match="model_budget"):
        clock.begin_model_call()
    clock.check(next_sim_dt=.04)
    clock.record_step(.04)
    with pytest.raises(BudgetExceeded, match="sim_timeout"):
        clock.check(next_sim_dt=.02)


def test_native_clock_overshoot_is_preserved_as_failure_evidence():
    clock = EpisodeClock(Budgets(sim_time_s=.04), monotonic=lambda: 0.)
    with pytest.raises(BudgetExceeded, match="sim_timeout"):
        clock.record_step(.06)
    assert clock.snapshot()["elapsed_sim_s"] == .06


@pytest.mark.parametrize('field',['wall_time_s','sim_time_s'])
@pytest.mark.parametrize('value',[0.,-1.,float('inf'),float('nan'),False])
def test_disabled_horizon_requires_explicit_null(field,value):
    with pytest.raises(ValidationError):
        Budgets(**{field:value})


def test_null_horizons_keep_real_elapsed_time_and_count_limits():
    now=[0.]
    budget=Budgets(wall_time_s=None,sim_time_s=None,max_actions=1,max_model_calls=1)
    assert Budgets.model_validate_json(budget.model_dump_json())==budget
    clock=EpisodeClock(budget,monotonic=lambda:now[0])
    clock.begin_model_call()
    now[0]=1e9
    clock.end_model_call()
    clock.check(next_sim_dt=1e8)
    clock.record_step(1e8)
    clock.action_attempt()
    assert clock.remaining_wall_s is None
    assert not clock.wall_expired and not clock.sim_expired
    assert clock.snapshot()['elapsed_wall_s']==1e9
    assert clock.snapshot()['elapsed_sim_s']==1e8
    with pytest.raises(BudgetExceeded,match='action_budget'):clock.action_attempt()
    with pytest.raises(BudgetExceeded,match='model_budget'):clock.begin_model_call()


@pytest.mark.parametrize('disabled,limited,outcome',[
    ('wall_time_s','sim_time_s','sim_timeout'),('sim_time_s','wall_time_s','wall_timeout')])
def test_null_horizon_does_not_disable_the_other_clock(disabled,limited,outcome):
    now=[0.]
    clock=EpisodeClock(Budgets(**{disabled:None,limited:.02}),monotonic=lambda:now[0])
    if limited=='sim_time_s':clock.record_step(.02)
    else:now[0]=.02
    with pytest.raises(BudgetExceeded,match=outcome):clock.check()


def test_success_without_horizons_and_metrics_do_not_invent_failure_penalties():
    budget=Budgets(wall_time_s=None,sim_time_s=None)
    win=result(True,spec=spec(budgets=budget),elapsed_wall_s=2000.,completion_wall_s=2000.,
               elapsed_sim_s=200.,completion_sim_s=200.)
    lose=result(spec=spec(episode_id='e1',seed=1,budgets=budget),elapsed_sim_s=3.,elapsed_wall_s=4.)
    report=aggregate_results([win,lose],plan=EvaluationPlan(plan_id='p1',episodes=[win.spec,lose.spec]))
    summary=report['cohorts'][0]['summary']
    assert summary['success_rate']==.5
    for name in ['penalized_sim_time_s','penalized_wall_time_s','success_per_sim_second','success_per_wall_second']:
        assert summary[name] is None
    assert summary['actual_elapsed_sim_s']==203.
    assert summary['actual_elapsed_wall_s']==2004.


def test_lifecycle_cancel_and_duplicate_protection():
    life = ActionLifecycle()
    life.accept("a", sim_time_s=0., wall_time_s=0.)
    with pytest.raises(ValueError):
        life.accept("b", sim_time_s=0., wall_time_s=0.)
    life.transition("a", "running", sim_time_s=0., wall_time_s=1.)
    life.transition("a", "cancelled", sim_time_s=.02, wall_time_s=2.)
    with pytest.raises(ValueError):
        life.transition("a", "completed", sim_time_s=.02, wall_time_s=2.)
    with pytest.raises(ValueError):
        life.accept("a", sim_time_s=.02, wall_time_s=2.)
    assert life.active_action_id is None


def test_metrics_failure_cannot_game_success_per_time():
    win = result(True)
    lose = result(spec=spec(episode_id="e1", seed=1), elapsed_sim_s=0., elapsed_wall_s=.01)
    report = aggregate_results([win, lose], plan=EvaluationPlan(plan_id="p1", episodes=[win.spec, lose.spec]))
    score = report["cohorts"][0]["summary"]
    assert score["success_rate"] == .5
    assert score["success_per_sim_second"] == pytest.approx(1 / 121)
    assert score["success_per_wall_second"] == pytest.approx(1 / 1810)
    assert report["leaderboard_eligible"]
    assert score["mean_completion_sim_s_success_only"] == 1.


def test_metrics_all_failed_missing_and_infrastructure():
    report = aggregate_results([result()], expected_episode_ids=["e0", "e1"])
    score = report["cohorts"][0]["summary"]
    assert score["success_rate"] == 0 and score["mean_completion_sim_s_success_only"] is None
    assert not report["complete"] and not report["leaderboard_eligible"]
    assert not report["cohorts"][0]["per_task"]["task"]["leaderboard_eligible"]
    infra = aggregate_results([result(outcome="infrastructure_error")], expected_episode_ids=["e0"])
    assert infra["cohorts"][0]["summary"]["scored_episodes"] == 0
    assert not infra["leaderboard_eligible"]
    invalid = result(outcome="safety_stop", physics_valid=False)
    invalid_report = aggregate_results([invalid], plan=EvaluationPlan(plan_id="p", episodes=[invalid.spec]))
    assert not invalid_report["leaderboard_eligible"]
    assert invalid_report["cohorts"][0]["summary"]["invalid_physics_episodes"] == 1
    assert invalid_report["cohorts"][0]["summary"]["outcome_counts"]["safety_stop"] == 1


def test_evaluator_invalid_is_accounted_like_infrastructure_not_agent_failure():
    report = aggregate_results([result(outcome="evaluator_invalid", physics_valid=False)], expected_episode_ids=["e0"])
    summary = report["cohorts"][0]["summary"]
    assert summary["scored_episodes"] == 0 and summary["infrastructure_errors"] == 1
    assert summary["success_rate"] is None and not report["leaderboard_eligible"]


def test_metrics_do_not_mix_visibility_policies_or_robots():
    a = result()
    b = result(spec=spec(episode_id="e1", observation_profile="privileged_state"))
    assert len(aggregate_results([a, b])["cohorts"]) == 2
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_results([a, a])
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_results([a, result(spec=spec(episode_id="e1"))])


def test_plan_checks_full_pins_not_just_ids():
    plan = EvaluationPlan(plan_id="p1", episodes=[spec()])
    assert aggregate_results([result()], plan=plan)["plan_pins_verified"]
    with pytest.raises(ValueError, match="plan slot"):
        aggregate_results([result(spec=spec(seed=5))], plan=plan)
    missing_pin = result(spec=spec(runtime_code_sha256=None))
    report = aggregate_results([missing_pin], plan=EvaluationPlan(plan_id="p2", episodes=[missing_pin.spec]))
    assert not report["leaderboard_eligible"]


def test_scene_variants_are_separate_but_per_seed_scene_hashes_can_differ():
    first = result()
    changed_seed = result(spec=spec(episode_id="e1", seed=1, scene_sha256="b" * 64))
    assert len(aggregate_results([first, changed_seed])["cohorts"]) == 1
    changed_variant = result(spec=spec(episode_id="e2", scene_variant="cluttered_v1", scene_sha256="c" * 64))
    assert len(aggregate_results([first, changed_variant])["cohorts"]) == 2
    missing_pin = result(spec=spec(scene_sha256=None))
    assert not aggregate_results([missing_pin], plan=EvaluationPlan(plan_id="p", episodes=[missing_pin.spec]))["leaderboard_eligible"]


def test_execution_environment_pin_separates_hardware_and_allocation_cohorts():
    first = result()
    same_environment = result(spec=spec(episode_id="e1", seed=1))
    assert len(aggregate_results([first, same_environment])["cohorts"]) == 1
    other_environment = result(spec=spec(episode_id="e2", execution_environment_sha256="c" * 64))
    report = aggregate_results([first, other_environment])
    assert len(report["cohorts"]) == 2
    assert {cohort["fixed_configuration"]["execution_environment_sha256"]
            for cohort in report["cohorts"]} == {H, "c" * 64}
    plan = EvaluationPlan(plan_id="fixed_host", episodes=[first.spec])
    changed_slot = result(spec=spec(execution_environment_sha256="c" * 64))
    with pytest.raises(ValueError, match="preregistered plan slot"):
        aggregate_results([changed_slot], plan=plan)


def test_legacy_result_without_execution_environment_is_readable_but_ineligible():
    payload = result().model_dump(mode="json")
    del payload["spec"]["execution_environment_sha256"]
    legacy = EpisodeResult.model_validate(payload)
    assert legacy.spec.execution_environment_sha256 is None
    report = aggregate_results([legacy], plan=EvaluationPlan(plan_id="legacy", episodes=[legacy.spec]))
    assert report["complete"] and report["plan_pins_verified"]
    assert not report["leaderboard_eligible"]
    assert report["cohorts"][0]["summary"]["attempted_episodes"] == 1
    assert not report["cohorts"][0]["per_task"]["task"]["leaderboard_eligible"]


def test_remote_model_latency_is_not_fabricated():
    score = aggregate_results([result()])["cohorts"][0]["summary"]
    assert score["total_model_latency_s"] is None and score["total_model_calls"] is None
    assert not score["model_call_budget_verified"]
    score = aggregate_results([result(model_accounting="server_observed", model_calls=1, model_latency_s=2.)])["cohorts"][0]["summary"]
    assert score["total_model_latency_s"] == 2.
    assert score["model_call_budget_verified"]


@pytest.mark.parametrize("updates", [dict(physics_valid=False), dict(evaluator_evidence_sha256=None), dict(completion_wall_s=1900., elapsed_wall_s=1900.), dict(model_latency_s=11.), dict(action_attempts=6001)])
def test_success_claims_require_valid_evidence_and_budgets(updates):
    with pytest.raises(ValidationError):
        result(True, **updates)


def test_registry_does_not_resolve_python_paths():
    registry = PolicyRegistry()
    with pytest.raises(ValueError, match="not registered"):
        registry.resolve("module.Class", "v1")
    with pytest.raises(TypeError):
        registry.register(object())


def test_registry_freezes_policy_descriptor():
    class Diagnostic:
        descriptor = policy()
        def reset(self, spec, capabilities): pass
        def begin(self, action, observation): pass
        def step(self, observation): return PolicyStep(status="completed")
        def cancel(self, reason): pass
    registry = PolicyRegistry()
    diagnostic = Diagnostic()
    registry.register(diagnostic)
    assert registry.resolve("authors_rl", "v1") is diagnostic
    diagnostic.descriptor.skills.append("TELEPORT")
    assert registry.descriptors()[0].skills == ["GRASP"]
    with pytest.raises(ValueError, match="descriptor changed"):
        registry.resolve("authors_rl", "v1")


def test_terminal_plugin_status_cannot_smuggle_command():
    with pytest.raises(ValidationError):
        PolicyStep(status="completed", command={"joint_positions_rad": {"arm1": 0.}, "duration_s": .02})


def test_typed_record_and_hash_manifest(tmp_path):
    recorder = EpisodeRecorder(tmp_path / "episode", spec())
    recorder.action(action().model_dump())
    recorder.command(RobotJointCommand(joint_positions_rad={"arm1": 0., "finger1": 0.}, duration_s=.02))
    manifest = recorder.finish(result())
    for name, digest in manifest["files_sha256"].items():
        assert hashlib.sha256((tmp_path / "episode" / name).read_bytes()).hexdigest() == digest
    with pytest.raises(RuntimeError):
        recorder.action(action().model_dump())
    with pytest.raises(FileExistsError):
        EpisodeRecorder(tmp_path / "episode", spec())


def test_schema_export_is_json_serializable_and_closed():
    exported = schemas()
    assert exported["EpisodeSpec"]["additionalProperties"] is False
    assert "discriminator" in exported["action"]
    assert "NaN" not in json.dumps(exported, allow_nan=False)
