"""Explicit repeated transverse regrasp, without changing the public fold goal."""
from __future__ import annotations

SEQUENTIAL_CUFFS = 'longitudinal_then_separate_cuffs_v1'


def reference_phase_indices(options, task_id=None):
    if 'trousers_phase_sequence' not in options:
        return None
    if (options['trousers_phase_sequence'] != SEQUENTIAL_CUFFS
            or task_id not in (None, 'trousers_folding')
            or options.get('transport_mode', 'fold') != 'fold'):
        raise ValueError('Sequential cuffs require the explicit trousers fold profile')
    phases = options.get('phases', {})
    if not isinstance(phases, dict) or set(phases) != {'0', '1', '2'}:
        raise ValueError('Sequential cuffs require exactly three explicit execution phases')
    if any(not isinstance(p, dict) for p in phases.values()):
        raise ValueError('Each execution phase must be an explicit object')
    first_edges = phases['0'].get('edge_indices', [0, 1])
    if (not isinstance(first_edges, list) or len(first_edges) != 2
            or any(type(i) is not int for i in first_edges) or set(first_edges) != {0, 1}):
        raise ValueError('The longitudinal phase must retain both original boundaries')
    for index, feature in ((1, 'left_cuff'), (2, 'right_cuff')):
        phase = phases[str(index)]
        if phase.get('folded_cuff_boundary') not in (feature+'_minimum', feature+'_maximum'):
            raise ValueError('Fold the original left cuff, then freshly capture the original right cuff')
        sides = phase.get('sides')
        if not isinstance(sides, list) or len(sides) != 1 or sides[0] not in ('left', 'right'):
            raise ValueError('Each transverse regrasp requires one explicit original hand')
        if 'capture_order' in phase and phase['capture_order'] != sides:
            raise ValueError('A transverse capture order must contain only its selected hand')
        if 'edge_indices' in phase:
            selected = phase['edge_indices']
            if (not isinstance(selected, list) or len(selected) != 1
                    or type(selected[0]) is not int or selected[0] != 0):
                raise ValueError('Each transverse operation selects its single original cuff')
    return (0, 1, 1)


def reference_phase_index(options, index, task_id=None):
    sequence = reference_phase_indices(options, task_id)
    if sequence is None:
        return index
    if type(index) is not int or not 0 <= index < len(sequence):
        raise ValueError('Unavailable sequential cuff execution phase')
    return sequence[index]
