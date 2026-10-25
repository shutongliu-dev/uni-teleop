"""Instrument presentation and public design masses, not native qualification."""
from copy import deepcopy

import pytest
from pxr import Usd, UsdPhysics, UsdShade

from bench.protocol.task_referee import PublicContract, project_task_rules
from bench.simulation.isaacsim_backend.task_visual_state import BooleanMaterialBindings
from scenes.workrooms.laboratory import rules
from scenes.workrooms.laboratory.weight_display import add_weighing_display
from scenes.workrooms.laboratory.weight_readout import readout
from scenes.workrooms.laboratory.tasks.solid_weighing.assets import author_task
from test_laboratory_review_contracts import setup, step


@pytest.mark.parametrize('task_id', rules.TASK_IDS)
def test_public_mass_inventory_matches_frozen_asset_and_rejects_drift(task_id):
    task, meta, _, _ = setup(task_id)
    public = project_task_rules(rules.bind_task_rules(task))
    declared = public['public_rules']['object_properties']
    assert {row['id']: row['nominal_mass_kg'] for row in declared} == {
        row['id']: row['mass_kg'] for row in meta['objects']}
    assert all('body_path' not in row and 'initial_pose_xyzw' not in row for row in declared)
    meta['objects'][0]['mass_kg'] *= 2
    with pytest.raises(ValueError, match='nominal object masses'):
        rules.evaluation_configuration(task_id, meta)


def test_optional_public_properties_preserve_legacy_contract_bytes_and_reject_oracles():
    old = rules.public_rules('sample_sorting')['referee']
    del old['object_properties']
    assert PublicContract.model_validate(old).model_dump(mode='json', by_alias=True) == old
    for mutation in ('zero', 'negative', 'nonfinite', 'bool', 'duplicate', 'live_pose'):
        value = rules.public_rules('sample_sorting')['referee']
        row = value['object_properties'][0]
        if mutation == 'duplicate':
            value['object_properties'].append(deepcopy(row))
        elif mutation == 'live_pose':
            row['position_m'] = [0, 0, 0]
        else:
            row['nominal_mass_kg'] = {'zero': 0, 'negative': -.1,
                'nonfinite': float('nan'), 'bool': True}[mutation]
        with pytest.raises(ValueError):
            PublicContract.model_validate(value)


def balance_fixture(count):
    task, meta, frame, observer = setup('solid_weighing')
    frame['objects']['weighing_cup']['position_m'] = [.71, 2.075, .9255]
    pellets = meta['task_parameters']['pellet_ids'][:count]
    for i, name in enumerate(pellets):
        frame['objects'][name]['position_m'] = [.695+.01*(i//3), 2.075+.01*(i%3-1), .9325]
    pairs = [('weighing_cup', 'balance_platform')]+[(name, 'weighing_cup') for name in pellets]
    # Scoop is deliberately not returned and granules were not scooped. The
    # useful instrument measurement must not manufacture operation credit.
    frame['objects']['scoop']['position_m'][0] += .10
    return task, meta, frame, observer, pairs


@pytest.mark.parametrize('count', [0, 2, 12])
def test_display_reads_stable_tared_mass_before_goal_without_operation_credit(count):
    task, meta, frame, observer, pairs = balance_fixture(count)
    for _ in range(600):
        summary = step(frame, observer, meta, pairs)
    assert not summary['detail']['balance_display']['valid']
    for _ in range(100):
        summary = step(frame, observer, meta, pairs)
    display = summary['detail']['balance_display']
    assert display['value_g'] == pytest.approx(count*.5)
    assert display['text'] == f'{count*.5:04.1f}'
    assert summary['released_stable_s'] == 0
    assert not summary['detail']['scooped_ids']
    assert summary['phase'] == 'working'


@pytest.mark.parametrize('invalidate', ['lift', 'hand_cup', 'hand_grain', 'scoop_grain', 'inventory', 'gap'])
def test_display_clears_immediately_on_unloaded_or_untrusted_evidence(invalidate):
    _, meta, frame, observer, pairs = balance_fixture(12)
    for _ in range(700):
        summary = step(frame, observer, meta, pairs)
    assert summary['detail']['balance_display']['valid']
    if invalidate == 'lift':
        frame['objects']['weighing_cup']['position_m'][2] += .02
    elif invalidate == 'hand_cup':
        pairs += [('weighing_cup', 'robot_hand')]
    elif invalidate == 'hand_grain':
        pairs += [('pellet_00', 'robot_hand')]
    elif invalidate == 'scoop_grain':
        pairs += [('pellet_00', 'scoop')]
    elif invalidate == 'inventory':
        frame['objects']['pellet_00']['position_m'][0] += .1
    else:
        frame.update(native_step=frame['native_step']+2, time_s=frame['time_s']+.001,
                     native_time_s=frame['native_time_s']+.001)
        observer.on_frame(frame)
        summary = observer.on_observe()
    if invalidate != 'gap':
        summary = step(frame, observer, meta, pairs)
    assert not summary['detail']['balance_display']['valid']
    assert summary['detail']['balance_display']['value_g'] is None
    assert summary['detail']['balance_display']['text'] == '---'


def test_display_authors_only_visuals_and_material_updates_preserve_physical_attributes():
    stage = Usd.Stage.CreateInMemory()
    meta = author_task(stage)
    physical = {str(prim.GetPath()): {attr.GetName(): attr.Get() for attr in prim.GetAttributes()}
                for prim in stage.Traverse()}
    if 'balance_display' not in meta:
        add_weighing_display(stage, meta)
    for path, attrs in physical.items():
        assert {attr.GetName(): attr.Get() for attr in stage.GetPrimAtPath(path).GetAttributes()} == attrs
    visual_paths = [row['target_prim_path'] for row in meta['visual_state_bindings']['bindings']]
    assert len(visual_paths) == 46
    for path in visual_paths:
        prim = stage.GetPrimAtPath(path)
        assert not prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.HasAPI(UsdPhysics.RigidBodyAPI)
    bindings = BooleanMaterialBindings(stage, meta['visual_state_bindings'])
    for mass, expected in [(None, '---'), (0., '00.0'), (6., '06.0'), (None, '---')]:
        display = readout(mass)
        assert display['text'] == expected
        receipt = bindings.sync({'detail': {'balance_display': display}}, native_step=7)
        assert receipt['source_native_step'] == 7
        for row in receipt['bindings']:
            material = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(row['target_prim_path'])).ComputeBoundMaterial()[0]
            assert str(material.GetPath()).endswith('/true' if row['value'] else '/false')
        before = stage.GetRootLayer().ExportToString()
        repeat = bindings.sync({'detail': {'balance_display': display}}, native_step=7)
        assert not any(row['binding_changed'] for row in repeat['bindings'])
        assert stage.GetRootLayer().ExportToString() == before
    for path, attrs in physical.items():
        assert {attr.GetName(): attr.Get() for attr in stage.GetPrimAtPath(path).GetAttributes()} == attrs
