"""Bounded validation diagnostics without input values or exception text."""
import re


_REASONS = {
    'missing': 'Required field is missing.',
    'extra_forbidden': 'Field is not advertised by this function.',
    'int_type': 'Use an integer.', 'float_type': 'Use a number.',
    'bool_type': 'Use a boolean.', 'string_type': 'Use a string.',
    'list_type': 'Use a list.', 'dict_type': 'Use an object.',
    'literal_error': 'Choose an advertised literal value.',
    'enum': 'Choose an advertised enum value.',
    'greater_than': 'Value must exceed the published lower bound.',
    'greater_than_equal': 'Value must meet the published lower bound.',
    'less_than': 'Value must be below the published upper bound.',
    'less_than_equal': 'Value must meet the published upper bound.',
    'too_short': 'Value has fewer items than the published minimum.',
    'too_long': 'Value has more items than the published maximum.',
    'finite_number': 'Use a finite number.',
    'string_pattern_mismatch': 'Value does not match the published name pattern.',
}
_CUSTOM = {
    'Value error, quaternion must be unit length in wxyz order':
        'Use a unit quaternion in wxyz order: abs(sum(q_i*q_i)-1) must be at most 1e-5.',
    'Value error, Wrist targets use the world frame': 'Wrist targets require frame_id=world.',
    'Value error, A hand joint or fingertip target is required': 'Provide hand joint or fingertip targets.',
}


def controlled_validation_details(error, schema, *, names=()):
    allowed = set(names)

    def visit(value):
        if isinstance(value, dict):
            allowed.update(value.get('properties', {}))
            allowed.update(x for x in value.get('enum', []) if isinstance(x, str)
                           and re.fullmatch(r'[A-Za-z0-9_.:-]{1,80}', x))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(schema)
    errors = error.errors(include_url=False, include_context=False, include_input=False)
    rows = []
    for item in errors[:8]:
        kind = item.get('type')
        reason = _REASONS.get(kind, 'Value does not satisfy a published constraint.')
        if item.get('msg') in _CUSTOM:
            kind, reason = 'value_constraint', _CUSTOM[item['msg']]
        elif kind not in _REASONS:
            kind = 'value_constraint'
        path = [part if (type(part) is int and 0 <= part <= 1000000)
                or (isinstance(part, str) and part in allowed) else '<unrecognized_field>'
                for part in item.get('loc', ())[:12]]
        rows.append({'field_path': path, 'type': kind, 'reason': reason})
    return {'validation_errors': rows, 'validation_error_count': len(errors),
            'additional_validation_errors': max(0, len(errors)-len(rows))}
