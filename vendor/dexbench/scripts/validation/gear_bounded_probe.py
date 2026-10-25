"""Bind the fixed bounded-root experiment to the unchanged failed gear scene.

This is an additional numerical diagnostic. It never replaces the original
geometry verdict, identifies an internal-cap root as exterior, or scores a task.
"""
from pathlib import Path

import numpy as np

from bench.isaacsim.release import digest, dump
from bench.paths import load_json


PROPOSAL_SHA256 = '39499ba423b79a2beca060e8d8b61699ff1957c386e48d971ebe87cd9157f40f'
WITNESSES_SHA256 = '3986c0b5e3f13e3c8368deef3a1dd92839dcea82c870d4c53cf5e2ead07382e3'


def load_bounded_contract(binding):
    if not isinstance(binding, dict) or set(binding) != {'proposal_path', 'proposal_sha256', 'witness_path', 'witness_sha256'}:
        raise ValueError('Declare the exact bounded proposal and fixed witnesses')
    proposal_path, witness_path = (Path(binding[key]) for key in ('proposal_path', 'witness_path'))
    if (binding['proposal_sha256'] != PROPOSAL_SHA256 or digest(proposal_path) != PROPOSAL_SHA256
            or binding['witness_sha256'] != WITNESSES_SHA256 or digest(witness_path) != WITNESSES_SHA256):
        raise ValueError('Fixed bounded-root proposal or witnesses changed')
    proposal = load_json(proposal_path)
    source_run = Path(proposal['source_run'])
    if (digest(source_run/'input.json') != proposal['original_input_sha256']
            or digest(source_run/'result/report.json') != proposal['source_report_sha256']):
        raise ValueError('Original failed native evidence changed')
    for row in proposal['fields']:
        relative = 'original_exterior_projection.npz' if row['field'] == 'original_union' else row['field']+'/piece_projection.npz'
        if digest(source_run/'result/input_gear'/relative) != row['source_projection_sha256']:
            raise ValueError('Original fixed projection evidence changed')
    return proposal


def validate_bounded_binding(binding, inputs):
    proposal = load_bounded_contract(binding)
    original = load_json(Path(proposal['source_run'])/'input.json')
    # Every physical input, reset, load contract and old asset dependency stays
    # exact. Only query-code binding and separately hashed diagnostic data differ.
    mutable_keys = {'asset_files_sha256', 'sdf_geometry_probe', 'bounded_zero_search'}
    if {k:v for k,v in inputs.items() if k not in mutable_keys} != {k:v for k,v in original.items() if k not in mutable_keys}:
        raise ValueError('Bounded query must preserve every original physical input and reset')
    probe_changes = {'contract_root', 'contract_sha256'}
    if ({k:v for k,v in inputs['sdf_geometry_probe'].items() if k not in probe_changes}
            != {k:v for k,v in original['sdf_geometry_probe'].items() if k not in probe_changes}):
        raise ValueError('Only the geometry checker contract binding may differ')
    old_root=Path(original['sdf_geometry_probe']['contract_root'])
    new_root=Path(inputs['sdf_geometry_probe']['contract_root'])
    for root, definition in ((old_root,original),(new_root,inputs)):
        if digest(root/'contract.json') != definition['sdf_geometry_probe']['contract_sha256']:
            raise ValueError('Frozen geometry contract changed')
    old_contract=load_json(old_root/'contract.json')
    new_contract=load_json(new_root/'contract.json')
    allowed={'checker_sha256','bounded_search_provenance'}
    if ({k:v for k,v in new_contract.items() if k not in allowed}
            != {k:v for k,v in old_contract.items() if k not in allowed}):
        raise ValueError('Original geometry witnesses, generator, thresholds and parameters must stay exact')
    if (digest(new_root/'witnesses.npz') != old_contract['witnesses_sha256']
            or digest(old_root/'witnesses.npz') != old_contract['witnesses_sha256']):
        raise ValueError('Original geometry witness bytes must stay exact')
    if new_root != old_root:
        provenance=new_contract.get('bounded_search_provenance',{})
        if (provenance.get('source_contract_path') != str(old_root/'contract.json')
                or provenance.get('source_contract_sha256') != original['sdf_geometry_probe']['contract_sha256']):
            raise ValueError('New checker contract must identify its immutable original')
    inventory = inputs['asset_files_sha256']
    if any(inventory.get(path) != expected for path,expected in original['asset_files_sha256'].items()):
        raise ValueError('Original asset and evidence inventory must remain complete')
    for path_key, hash_key in [('proposal_path', 'proposal_sha256'), ('witness_path', 'witness_sha256')]:
        if inventory.get(binding[path_key]) != binding[hash_key]:
            raise ValueError('Bounded diagnostic dependencies must be in the frozen inventory')
    return proposal


def matching_population(origins, source_indices, actual_origins, projection):
    """A changed native failure population cannot silently select easier points."""
    actual_failed = np.flatnonzero(~np.asarray(projection['qualified'], bool))
    if (not np.array_equal(source_indices, actual_failed)
            or not np.array_equal(origins, np.asarray(actual_origins)[source_indices])):
        raise ValueError('Native failed witnesses differ from the predeclared population')


def run_bounded_probe(binding, asset, queries, layouts, projections, output):
    from scripts.validation.gear_bounded_roots import bounded_zero_search, load_frozen_field

    proposal = load_bounded_contract(binding)
    output = Path(output)
    output.mkdir()
    rows = {}
    with np.load(binding['witness_path']) as witness:
        fields={row['field']:load_frozen_field(witness,row['field']) for row in proposal['fields']}
        for name,fixed in fields.items():
            matching_population(fixed['origins'],fixed['source_indices'],*projections[name])
        for field in proposal['fields']:
            name = field['field']
            fixed = fields[name]
            indices = list(range(8)) if name == 'original_union' else [int(name[-2:])]
            rotations = ([np.asarray(next(s['mesh_to_body_rotation'] for s in asset['sector_pieces'] if s['index'] == i))
                          for i in indices] if name == 'original_union' else None)
            rows[name] = bounded_zero_search(
                [queries[i] for i in indices], **fixed, output=output/name, rotations=rotations,
                distance_columns=[layouts[i]['distance_column'] for i in indices],
                gradient_columns=[layouts[i]['gradient_columns'] for i in indices])
    count = sum(row['native_query_call_count'] for row in rows.values())
    if count > proposal['additional_query_call_upper']:
        raise ValueError('Predeclared native query budget exceeded')
    report = {'schema': 'gear_bounded_zero_probe.v1', 'fields': rows,
        'proposal_sha256': binding['proposal_sha256'], 'witness_sha256': binding['witness_sha256'],
        'native_query_call_count': count, 'maximum_native_query_call_count': proposal['additional_query_call_upper'],
        'numeric_feasibility_passed': all(row['numeric_passed'] for row in rows.values()),
        'geometry_qualified': False, 'task_qualified': False,
        'scope': 'Additional bounded numerical root witnesses; all original geometry failures and independent legacy metric gates remain. Exterior ownership is ambiguous and no new exterior or contact qualification is granted.'}
    dump(output/'report.json', report)
    return report
