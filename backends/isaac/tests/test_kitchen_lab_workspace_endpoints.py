"""Synthetic CPU geometry fixtures, never native or manipulation evidence."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from bench.paths import load_json
from scripts.validation import workroom_interaction_endpoints as endpoints
from scripts.validation import workroom_source_interference as interference
from scripts.validation import workroom_randomized_workspace as workspace


def stage_fixture():
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateInMemory()
    body = UsdGeom.Xform.Define(stage, '/World/Task/Tool')
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    cube = UsdGeom.Cube.Define(stage, '/World/Task/Tool/Handle')
    cube.CreateSizeAttr(.04)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return stage


def target(**changes):
    result = {'label': 'handle', 'prim_path': '/World/Task/Tool', 'local_position_m': [0, 0, .02],
        'approach_direction_local': [0, 0, 1], 'approach_distance_m': .06,
        'type': 'actual_surface', 'reference_state': 'initial', 'operation_group': 'tool_grip', 'required': True}
    return {**result, **changes}


def test_real_enabled_surface_not_center_or_visual():
    from pxr import UsdGeom, UsdPhysics
    stage = stage_fixture()
    assert endpoints.surface_witness(stage, target(), [0, 0, .02])['distance_m'] < 1e-10
    with pytest.raises(ValueError, match='not on enabled'):
        endpoints.surface_witness(stage, target(), [0, 0, 0])
    picture = UsdGeom.Cube.Define(stage, '/World/Task/Tool/Picture'); picture.CreateSizeAttr(.10)
    with pytest.raises(ValueError, match='not on enabled'):
        endpoints.surface_witness(stage, target(), [0, 0, .05])
    UsdPhysics.CollisionAPI(stage.GetPrimAtPath('/World/Task/Tool/Handle')).GetCollisionEnabledAttr().Set(False)
    with pytest.raises(ValueError, match='no enabled'):
        endpoints.surface_witness(stage, target(), [0, 0, .02])


def test_surface_world_transform_and_uniform_scale():
    from pxr import Gf, UsdGeom
    stage = stage_fixture()
    body = UsdGeom.Xformable(stage.GetPrimAtPath('/World/Task/Tool'))
    body.AddTranslateOp().Set(Gf.Vec3d(.2, .3, .4)); body.AddRotateYOp().Set(90.)
    body.AddScaleOp().Set(Gf.Vec3f(1.5))
    result = endpoints.surface_witness(stage, target(), [.23, .3, .4])
    assert result['distance_m'] < 1e-8


def test_hollow_mesh_does_not_use_enclosing_bounds_surface():
    from pxr import UsdGeom, UsdPhysics
    stage = stage_fixture(); stage.RemovePrim('/World/Task/Tool/Handle')
    # Two separated plates; the midpoint of their combined envelope is empty.
    points = [[x, y, z] for x in (-.02, .02) for y, z in ((-.02, -.02), (.02, -.02), (.02, .02), (-.02, .02))]
    mesh = UsdGeom.Mesh.Define(stage, '/World/Task/Tool/Shell')
    mesh.CreatePointsAttr(points); mesh.CreateFaceVertexCountsAttr([4, 4]); mesh.CreateFaceVertexIndicesAttr(list(range(8)))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none')
    assert endpoints.surface_witness(stage, target(), [.02, 0, 0])['distance_m'] < 1e-8
    with pytest.raises(ValueError, match='not on enabled'):
        endpoints.surface_witness(stage, target(), [0, 0, .02])


def hand(clear=True):
    return {'converged': True, 'clearance_passed': clear, 'diagnostic': {'retained': 'original'}}


def test_same_hand_must_cover_all_stages_and_blocked_surface_is_not_clear():
    rows = {name: {'hands': {'left': hand(), 'right': hand()}, 'surface': {'blocked': False}}
            for name in ('approach', 'precontact', 'lift')}
    rows['approach']['hands']['left'] = hand(False)
    rows['precontact']['hands']['right'] = hand(False)
    assert endpoints.same_hand_coverage(rows, conditional=False) == []
    rows['precontact']['hands']['right'] = hand()
    assert endpoints.same_hand_coverage(rows, conditional=False) == ['right']
    rows['lift']['surface']['blocked'] = True
    assert endpoints.same_hand_coverage(rows, conditional=False) == []
    assert endpoints.same_hand_coverage(rows, conditional=True) == ['right']


def augment_fixture(monkeypatch, features=None, conditional=False):
    stage = stage_fixture(); features = features or [target()]
    samples = []
    for feature in features:
        samples.append({'feature': feature, 'reference_position_m': [0, 0, .02],
            'free_approach_position_m': [0, 0, .08], 'surface': {'blocked': conditional},
            'hands': {'left': hand(not conditional)}, 'geometric_endpoint_ik': {'left': hand()},
            'requires_prior_mechanism_change': feature['reference_state'] != 'initial'})
    report = {'case': {'task_id': 'pipette_transfer'}, 'samples': samples,
              'checks': {'original_check': False}, 'robot_manipulation_qualified': False}
    calls = []
    def solve(model, joints, point, side, yaw, approach, checker=None, ready=None, **kwargs):
        calls.append({'point': np.asarray(point).copy(), 'checker': checker, 'ready': ready, 'joint_check': kwargs['joint_check']})
        return hand()
    monkeypatch.setattr(endpoints.living, 'solve_target', solve)
    monkeypatch.setattr(endpoints.living, 'free_surface_approach', lambda checker, ref, p, **kw: (p, {'blocked': conditional}))
    monkeypatch.setattr(interference, 'review_source_geometry', lambda *a: {'physics_qualified': False})
    checker = SimpleNamespace(stage=stage)
    kwargs = dict(task={}, checker=checker, paths={'tool': '/World/Task/Tool'}, models={'left': object()},
                  joints={}, yaws={'left': 0}, ready={'left': 'original_boxes'}, arm=SimpleNamespace(check=lambda j: True))
    return report, kwargs, calls


def test_three_endpoints_keep_original_hand_obstacles_and_nonqualification(monkeypatch):
    report, kwargs, calls = augment_fixture(monkeypatch)
    old = deepcopy(report['samples'])
    endpoints.augment(report, **kwargs)
    sequence = report['interaction_endpoint_sequences']['handle']
    assert sequence['covered'] and sequence['usable_sides'] == ['left']
    np.testing.assert_allclose(calls[0]['point'], [0, 0, .03])
    np.testing.assert_allclose(calls[1]['point'], [0, 0, .06])
    assert all(call['checker'] is kwargs['checker'] and call['ready'] is kwargs['ready'] for call in calls)
    assert report['samples'] == old and not report['checks']['original_check']
    assert not report['robot_manipulation_qualified'] and not report['source_geometry_review']['physics_qualified']


def test_conditional_is_robot_only_and_alternatives_do_not_mix_stages(monkeypatch):
    features = [target(), target(label='opened', operation_group='opened_tool', reference_state='opened')]
    report, kwargs, calls = augment_fixture(monkeypatch, features, conditional=True)
    endpoints.augment(report, **kwargs)
    assert not report['interaction_endpoint_sequences']['handle']['covered']
    opened = report['interaction_endpoint_sequences']['opened']
    assert opened['covered'] and not opened['scene_clearance_checked']
    assert all(row['checker'] is None and row['ready'] is None for row in calls[2:])
    assert not report['checks']['initial_required_interaction_sequences_covered']
    assert report['checks']['conditional_required_interaction_sequences_covered']


def test_goal_not_a_grip_and_legacy_hook_is_identity(monkeypatch):
    report, kwargs, calls = augment_fixture(monkeypatch, [target(), target(label='goal', type='goal', operation_group='destination')])
    endpoints.augment(report, **kwargs)
    assert list(report['interaction_endpoint_sequences']) == ['handle'] and len(calls) == 2
    legacy = {'case': {'task_id': 'book_retrieval'}, 'checks': {'preserved': True}}
    old = deepcopy(legacy)
    workspace.augment_interaction_checks(legacy)
    assert legacy == old


def test_source_convex_lp_units_and_sdf_not_convexified():
    from pxr import Gf, UsdGeom, UsdPhysics
    stage = stage_fixture()
    other = UsdGeom.Cube.Define(stage, '/World/Task/Other'); other.CreateSizeAttr(.04)
    UsdPhysics.CollisionAPI.Apply(other.GetPrim())
    move = other.AddTranslateOp(); move.Set(Gf.Vec3d(.03, 0, 0))
    a = interference.source_polytope(stage, '/World/Task/Tool/Handle')
    b = interference.source_polytope(stage, '/World/Task/Other')
    assert interference.intersection_radius(a, b) == pytest.approx(.005)
    move.Set(Gf.Vec3d(.05, 0, 0))
    assert interference.intersection_radius(a, interference.source_polytope(stage, '/World/Task/Other')) == pytest.approx(-.005)
    mesh = UsdGeom.Mesh.Define(stage, '/World/Task/SDF'); UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none')
    assert interference.source_polytope(stage, '/World/Task/SDF') is None


def test_fixed_support_uses_exact_worktops_and_rejects_free_actor():
    from pxr import UsdGeom, UsdPhysics
    stage = stage_fixture()
    for name in ('kitchen_counter_middle', 'kitchen_cooktop', 'worktop', 'floor'):
        prim = UsdGeom.Cube.Define(stage, '/World/Room/Geometry/'+name+'/Collider').GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
    task = {'reset_case': {'family': 'kitchen', 'fixed_support_paths': []}}
    roots = interference.fixed_supports(stage, task, {'tool': '/World/Task/Tool'})
    assert len(roots) == 3 and all('floor' not in root for root in roots)
    task['reset_case']['fixed_support_paths'] = ['/World/Task/Tool']
    task['asset_metadata'] = {'fixtures': {'forged': '/World/Task/Tool'}}
    with pytest.raises(ValueError, match='dynamic actor'):
        interference.fixed_supports(stage, task, {'tool': '/World/Task/Tool'})
    task['reset_case']['fixed_support_paths'] = ['/World/Room/Geometry/floor/Collider']
    with pytest.raises(ValueError, match='outside original fixture'):
        interference.fixed_supports(stage, task, {'tool': '/World/Task/Tool'})


def test_request_capacity_tracks_registered_installations(tmp_path, monkeypatch):
    import json
    families = {f'fixture_{n}': 'laboratory' for n in range(3)}
    monkeypatch.setattr(workspace, 'TASK_FAMILIES', families)
    monkeypatch.setattr(workspace, 'configurations', lambda name: ('a', 'b'))
    monkeypatch.setattr(workspace, 'SEEDS', (0, 1))
    monkeypatch.setattr(workspace, 'selection', lambda *a: None)
    cases = [{'task_id': name, 'configuration_id': cfg, 'seed': seed, 'release': '/fixture'}
             for name in families for cfg in ('a', 'b') for seed in (0, 1)]
    path = tmp_path/'request.json'
    path.write_text(json.dumps({'schema': 'workroom_randomized_validation_request.v1', 'cases': cases}))
    assert len(workspace.read_request(path)['cases']) == 12
    cases.append(cases[0])
    path.write_text(json.dumps({'schema': 'workroom_randomized_validation_request.v1', 'cases': cases}))
    with pytest.raises(ValueError, match='1–12'):
        workspace.read_request(path)


def test_new_task_scope_has_fifteen_original_installations():
    from scenes.workrooms.initial_randomization import SEEDS
    kitchen = {'breakfast_serving', 'pot_lid_placement', 'surface_wiping'}
    lab = {'cap_placement', 'solid_weighing', 'pipette_transfer', 'tube_clamp_transfer'}
    assert endpoints.TASKS == kitchen | lab
    assert (len(kitchen) + 3*len(lab))*len(SEEDS) == 150


def piston_supplement_fixture(tmp_path):
    import json
    from pathlib import Path
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateInMemory()
    actor, tool = '/World/Task/plunger', '/World/Task/pipette'
    for path in (actor, tool):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, path).GetPrim())
    ring = UsdGeom.Cube.Define(stage, actor+'/PullRing'); ring.CreateSizeAttr(.01)
    ring.AddTranslateOp().Set(Gf.Vec3d(0, 0, .1075)); UsdPhysics.CollisionAPI.Apply(ring.GetPrim())
    pad = UsdGeom.Cube.Define(stage, actor+'/ThumbPad'); pad.CreateSizeAttr(.01)
    UsdPhysics.CollisionAPI.Apply(pad.GetPrim())
    joint = UsdPhysics.PrismaticJoint.Define(stage, '/World/Task/Joint')
    joint.CreateBody0Rel().SetTargets([tool]); joint.CreateBody1Rel().SetTargets([actor])
    joint.CreateAxisAttr('Z'); joint.CreateLowerLimitAttr(0.); joint.CreateUpperLimitAttr(.055)
    feature = target(label='plunger_pull_ring', prim_path=actor, local_position_m=[0., 0., .1125],
                     operation_group='plunger_pull', source_collider_path=actor+'/PullRing')
    sample = {'feature': feature, 'reference_position_m': [0., 0., .1125]}
    witness = endpoints.surface_witness(stage, feature, sample['reference_position_m'])
    sequence = {'feature_label': 'plunger_pull_ring', 'operation_group': 'plunger_pull', 'conditional': False,
                'source_surface': witness, 'covered': True, 'endpoints': {'fixture': 'synthetic CPU evidence'}}
    sequences = {'plunger_pull_ring': sequence, 'plunger_press_pad': {'covered': False, 'original_failure': 'retained'}}
    groups = {'plunger_press': {'covered': False, 'required': True, 'conditional': False,
                               'feature_labels': ['plunger_press_pad'], 'covered_feature_labels': []}}
    task = {'id': 'pipette_transfer', 'asset_metadata': {'actors': {'plunger': actor, 'pipette': tool},
        'joints': {'plunger': '/World/Task/Joint'}, 'task_parameters': {'hydraulics': {
            'plunger_id': 'plunger', 'tool_id': 'pipette', 'joint_path': '/World/Task/Joint', 'stroke_m': .055}}}}
    root = Path(endpoints.__file__).resolve().parents[2]
    manifest = {'files_sha256': {}}
    for filename, (relative, expected) in endpoints.PIPETTE_SEMANTIC_SOURCES.items():
        path = tmp_path/filename; path.parent.mkdir(exist_ok=True)
        path.write_bytes((root/relative).read_bytes()); manifest['files_sha256'][filename] = expected
    (tmp_path/'release_manifest.json').write_text(json.dumps(manifest))
    report = {'case': {'release': str(tmp_path)}, 'samples': [sample], 'resolved_inputs_sha256': 'synthetic-cpu-fixture'}
    return report, task, SimpleNamespace(stage=stage), sequences, groups


def test_real_rigid_piston_push_supplement_retains_original_failure(tmp_path):
    report, task, checker, sequences, groups = piston_supplement_fixture(tmp_path)
    original_samples = deepcopy(report['samples']); original_failure = deepcopy(sequences['plunger_press_pad'])
    endpoints.pipette_press_supplement(report, task, checker, sequences, groups)
    assert sequences['plunger_press_pad'] == original_failure and report['samples'] == original_samples
    assert groups['plunger_press']['covered'] and not groups['plunger_press']['original_endpoint_covered_without_supplement']
    supplement = sequences['plunger_press_ring_top_supplement']
    assert supplement['supplemental'] and supplement['covered']
    assert supplement['endpoints'] == sequences['plunger_pull_ring']['endpoints']
    assert supplement['semantic_basis']['source_sequence_sha256'] == endpoints.living.content_digest(sequences['plunger_pull_ring'])


@pytest.mark.parametrize('fault', ['different_body', 'different_axis', 'disabled_joint', 'disabled_ring',
    'changed_source', 'rehashed_source', 'unknown_surface', 'forged_sequence_geometry'])
def test_piston_supplement_rejects_changed_physics_rules_or_surface(tmp_path, fault):
    import json
    from pxr import UsdPhysics
    report, task, checker, sequences, groups = piston_supplement_fixture(tmp_path)
    stage = checker.stage; joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath('/World/Task/Joint'))
    if fault == 'different_body': joint.GetBody1Rel().SetTargets(['/World/Task/pipette'])
    elif fault == 'different_axis': joint.GetAxisAttr().Set('X')
    elif fault == 'disabled_joint': joint.CreateJointEnabledAttr(False)
    elif fault == 'disabled_ring': UsdPhysics.CollisionAPI(stage.GetPrimAtPath('/World/Task/plunger/PullRing')).CreateCollisionEnabledAttr(False)
    elif fault in ('changed_source', 'rehashed_source'):
        path = tmp_path/'evaluator/rules.py'; path.write_text(path.read_text()+'\n# changed semantics fixture\n')
        if fault == 'rehashed_source':
            manifest = load_json(tmp_path/'release_manifest.json')
            manifest['files_sha256']['evaluator/rules.py'] = endpoints.living.digest(path)
            (tmp_path/'release_manifest.json').write_text(json.dumps(manifest))
    elif fault == 'unknown_surface': report['samples'][0]['feature']['local_position_m'][2] = .12
    elif fault == 'forged_sequence_geometry': sequences['plunger_pull_ring']['source_surface']['distance_m'] = .001
    with pytest.raises(ValueError):
        endpoints.pipette_press_supplement(report, task, checker, sequences, groups)
    assert 'plunger_press_ring_top_supplement' not in sequences and not groups['plunger_press']['covered']


def parameterized_piston_fixture(tmp_path):
    """Exercise the actual public-mass authoring adapter, not synthetic pin data."""
    import json
    from scenes.workrooms.parameterized.workroom_rules import laboratory_rules
    fixture = piston_supplement_fixture(tmp_path)
    _, task, _, _, _ = fixture
    rules = (tmp_path/'evaluator/rules.py').read_text()
    namespace = {}; exec(compile(rules, '<original laboratory rules>', 'exec'), namespace)
    task['episode_seed'] = 1
    task['asset_metadata'].update(root_path='/World/LaboratoryTask', objects=[
        {'id': row['id'], 'mass_kg': row['nominal_mass_kg']*1.2}
        for row in namespace['object_properties']('pipette_transfer')])
    task['reset_case'] = {'parameterized_physics': {'parameters': {
        'schema': 'parameterized_physics_v1', 'family': 'laboratory', 'task_id': 'pipette_transfer'}}}
    values = {'task': task, 'evaluator_sources': {'rules.py': rules,
        'sequence_evaluator.py': (tmp_path/'evaluator/sequence_evaluator.py').read_text()}}
    laboratory_rules(values)
    manifest = load_json(tmp_path/'release_manifest.json')
    for name, content in values['evaluator_sources'].items():
        path = tmp_path/'evaluator'/name; path.write_text(content)
        manifest['files_sha256']['evaluator/'+name] = endpoints.living.digest(path)
    (tmp_path/'release_manifest.json').write_text(json.dumps(manifest))
    original = namespace['public_rules']('pipette_transfer', 1)
    for row in original['referee']['object_properties']: row['nominal_mass_kg'] *= 1.2
    assert task['public_rules'] == original
    return fixture


@pytest.mark.parametrize('ring_covered', [True, False])
def test_reviewed_parameterized_piston_keeps_raw_failure_and_exact_sequence(tmp_path, ring_covered):
    report, task, checker, sequences, groups = parameterized_piston_fixture(tmp_path)
    sequences['plunger_pull_ring']['covered'] = ring_covered
    before = deepcopy(sequences); samples = deepcopy(report['samples'])
    endpoints.pipette_press_supplement(report, task, checker, sequences, groups)
    supplement = sequences['plunger_press_ring_top_supplement']
    assert supplement['semantic_basis']['semantic_source_profile'] == 'parameterized_public_dry_mass_v1'
    assert supplement['covered'] == groups['plunger_press']['covered'] == ring_covered
    assert supplement['endpoints'] == before['plunger_pull_ring']['endpoints']
    assert all(sequences[key] == value for key, value in before.items())
    assert report['samples'] == samples


@pytest.mark.parametrize('fault', ['undeclared_parameters', 'wrong_family', 'wrong_task',
    'mixed_profile', 'rehashed_unknown', 'wrong_manifest'])
def test_parameterized_piston_rejects_unknown_or_mixed_semantics(tmp_path, fault):
    import json
    from pathlib import Path
    report, task, checker, sequences, groups = parameterized_piston_fixture(tmp_path)
    manifest = load_json(tmp_path/'release_manifest.json')
    parameters = task['reset_case']['parameterized_physics']['parameters']
    if fault == 'undeclared_parameters': task.pop('reset_case')
    elif fault == 'wrong_family': parameters['family'] = 'study'
    elif fault == 'wrong_task': parameters['task_id'] = 'solid_weighing'
    elif fault == 'mixed_profile':
        path = tmp_path/'evaluator/sequence_evaluator.py'
        path.write_bytes((Path(endpoints.__file__).resolve().parents[2]/'scenes/workrooms/laboratory/evaluation.py').read_bytes())
        manifest['files_sha256']['evaluator/sequence_evaluator.py'] = endpoints.living.digest(path)
    elif fault == 'rehashed_unknown':
        path = tmp_path/'evaluator/rules.py'; path.write_text(path.read_text()+'\n# Unreviewed successor\n')
        manifest['files_sha256']['evaluator/rules.py'] = endpoints.living.digest(path)
    elif fault == 'wrong_manifest': manifest['files_sha256']['evaluator/rules.py'] = '0'*64
    (tmp_path/'release_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        endpoints.pipette_press_supplement(report, task, checker, sequences, groups)
    assert 'plunger_press_ring_top_supplement' not in sequences and not groups['plunger_press']['covered']
