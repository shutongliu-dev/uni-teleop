"""A bounded JSON-valued language with Python syntax, not Python execution.

The entire syntax tree is checked before a tool is invoked. The interpreter
never exposes host objects or evaluates/compiles Python code. Only a trusted
callback may perform public bench operations; that callback must enforce any
enabled network and episode deadlines, because synchronous callbacks cannot be
interrupted by the interpreter's fuel or wall-clock checks.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, fields
import hashlib
import json
import math
import re
import time

from bench.protocol.schemas import (ActionDiagnostic, RobotExecutionState, TerminationReason,
    MAX_ROBOT_EXECUTION_STATE_BYTES)
from bench.sensors.contact_schema import ContactHistoryObservation
from typing import get_args

from .actions import ActionValidationError


@dataclass(frozen=True)
class ProgramLimits:
    max_source_bytes: int = 8192
    max_ast_nodes: int = 1024
    max_ast_depth: int = 32
    max_fuel: int = 4096
    max_loop_iterations: int = 16
    max_total_loop_iterations: int = 64
    max_loop_depth: int = 2
    max_tool_calls: int = 32
    max_move_calls: int = 8
    max_observe_calls: int = 16
    max_point_queries: int = 16
    max_variables: int = 64
    max_value_nodes: int = 4096
    max_value_bytes: int = 65536
    max_live_value_bytes: int = 262144
    max_trace_bytes: int = 1048576
    max_container_items: int = 256
    max_value_depth: int = 16
    max_string_length: int = 4096
    max_number_abs: float = 1e9
    max_runtime_s: float | None = 120.0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == 'max_runtime_s' and value is None:
                continue
            if field.name in ('max_number_abs', 'max_runtime_s'):
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ValueError(field.name + ' must be finite and positive')
            elif type(value) is not int or value < 1:
                raise ValueError(field.name + ' must be a positive integer')


class ProgramStop(Exception):
    """The trusted callback may stop a program with an explicitly public result."""
    def __init__(self, reason='stopped', result=None):
        self.reason, self.result = reason, result
        super().__init__('program stopped')


class _Fault(Exception):
    def __init__(self, code, line=None, *, status='runtime_error', public_error=None):
        self.code, self.line, self.status = code, line, status
        self.public_error = public_error

    def metadata(self, line=None):
        value = {'code': self.code, 'line': self.line if line is None else line}
        if self.public_error is not None:
            value['public_error'] = self.public_error
        return value


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


TOOLS = {'observe', 'move_geometry', 'hold', 'point_at', 'stop'}
PURE = {'abs', 'min', 'max', 'len'}
RESERVED = TOOLS | PURE | {'range'}
MOVE_KEYS = {'wrist_delta_m', 'wrist_rotation_delta_rad', 'fingertip_delta_m', 'duration_s', 'reason',
             'position_tolerance_m', 'orientation_tolerance_rad'}


def _bytes(value):
    return len(json.dumps(value, ensure_ascii=True, separators=(',', ':'), allow_nan=False).encode())


def _json_copy(value, limits):
    """Validate exact JSON types, cap cost, then copy without user object hooks."""
    count, active = 0, set()

    def visit(item, depth):
        nonlocal count
        count += 1
        if count > limits.max_value_nodes or depth > limits.max_value_depth:
            raise _Fault('value_structure_limit', status='budget_exceeded')
        kind = type(item)
        if item is None or kind is bool:
            return item
        if kind in (int, float):
            if kind is float and not math.isfinite(item):
                raise _Fault('nonfinite_number')
            if abs(item) > limits.max_number_abs:
                raise _Fault('number_limit', status='budget_exceeded')
            return item
        if kind is str:
            if len(item) > limits.max_string_length:
                raise _Fault('string_limit', status='budget_exceeded')
            return item
        if kind not in (list, dict):
            raise _Fault('non_json_value')
        if id(item) in active:
            raise _Fault('cyclic_value')
        if len(item) > limits.max_container_items:
            raise _Fault('container_limit', status='budget_exceeded')
        active.add(id(item))
        try:
            if kind is list:
                return [visit(child, depth + 1) for child in item]
            result = {}
            for key, child in item.items():
                if type(key) is not str:
                    raise _Fault('invalid_dictionary_key')
                visit(key, depth + 1)
                if '__' in key:
                    raise _Fault('invalid_dictionary_key')
                result[key] = visit(child, depth + 1)
            return result
        finally:
            active.remove(id(item))

    copied = visit(value, 0)
    if _bytes(copied) > limits.max_value_bytes:
        raise _Fault('value_bytes_limit', status='budget_exceeded')
    return copied


def _minimum_primitive_feedback(value):
    """Copy only bounded public execution fields before bulky results are capped.

    Tool callbacks are trusted but may return a malformed or oversized overall
    observation. Its resource fault must not erase an already executed action's
    status. Exact JSON types avoid running hooks on arbitrary callback objects.
    """
    if type(value) is not dict or type(value.get('feedback')) is not dict:
        return None
    raw=value['feedback']
    if type(raw.get('status')) is not str or raw['status'] not in (
            'accepted','running','completed','failed','cancelled'):
        return None
    result={'status':raw['status']}
    identifier=raw.get('action_id')
    if type(identifier) is str and re.fullmatch(r'[A-Za-z0-9_.:/-]{1,160}',identifier):
        result['action_id']=identifier
    for field in ('sim_time_s','elapsed_sim_s'):
        number=raw.get(field)
        if type(number) in (int,float) and 0<=number<=1e100 and math.isfinite(number):
            result[field]=number
    if type(raw.get('episode_finished')) is bool:
        result['episode_finished']=raw['episode_finished']
    reason=raw.get('termination_reason')
    if type(reason) is str and reason in get_args(TerminationReason):
        result['termination_reason']=reason
    elif reason is None and 'termination_reason' in raw:
        result['termination_reason']=None
    state=raw.get('robot_execution_state')
    if state is None and 'robot_execution_state' in raw:
        result['robot_execution_state']=None
    elif type(state) is dict:
        try:
            copied=_json_copy(state,ProgramLimits(max_value_bytes=MAX_ROBOT_EXECUTION_STATE_BYTES))
            result['robot_execution_state']=RobotExecutionState.model_validate(copied).model_dump(mode='json')
        except (_Fault,ValueError,TypeError):
            pass
    history=raw.get('contact_history')
    if history is None and 'contact_history' in raw:
        result['contact_history']=None
    elif type(history) is dict:
        try:
            # The session has already applied the tactile ablation. Preserve
            # every action's bounded public summary before later tools or
            # bulky trace truncation can replace its full callback result.
            copied=_json_copy(history,ProgramLimits(max_value_bytes=16384))
            checked=ContactHistoryObservation.model_validate(copied)
            if checked.action_id != result.get('action_id'):
                raise ValueError('Contact history belongs to another action')
            result['contact_history']=checked.model_dump(mode='json')
        except (_Fault,ValueError,TypeError):
            pass
    diagnostic=raw.get('diagnostic')
    if diagnostic is None:
        result['diagnostic']=None
    elif type(diagnostic) is dict:
        try:
            copied=_json_copy(diagnostic,ProgramLimits(max_value_bytes=8192))
            result['diagnostic']=ActionDiagnostic.model_validate(copied).model_dump(mode='json',exclude_none=True)
        except (_Fault,ValueError,TypeError):
            pass
    return result


def _minimum_point_result(value, arguments):
    """Retain a bounded public pixel sample, never an arbitrary callback object.

    This is historical evidence at its own sequence/camera revision, not a
    current object pose. The trusted depth tool still validates image hashes
    and calibration; this projection validates its result/argument boundary.
    """
    if type(value) is not dict:
        return None
    fields = ('tool', 'episode_id', 'sequence', 'sim_time_s', 'camera_id',
              'camera_revision', 'u', 'v', 'pixel_center_uv', 'depth_m',
              'depth_semantics', 'depth_unit', 'depth_sha256',
              'world_position_m', 'frame_id', 'scope')
    try:
        result = _json_copy({key: value[key] for key in fields}, ProgramLimits(
            max_value_bytes=4096, max_value_nodes=96, max_string_length=256,
            max_container_items=32, max_value_depth=3))
    except (KeyError, _Fault):
        return None
    if (result['tool'] != 'point_at' or result['frame_id'] != 'world'
            or result['depth_semantics'] != 'camera_z' or result['depth_unit'] != 'm'
            or result['scope'] != 'visible RGB-D pixel; no object identity or grasp inference'
            or type(result['episode_id']) is not str
            or re.fullmatch(r'[A-Za-z0-9_.:-]{1,160}', result['episode_id']) is None
            or type(result['camera_id']) is not str
            or type(result['depth_sha256']) is not str
            or re.fullmatch(r'[0-9a-f]{64}', result['depth_sha256']) is None):
        return None
    if any(type(result[key]) is not int or result[key] < 0
           for key in ('sequence', 'camera_revision', 'u', 'v')):
        return None
    if any(result[key] != arguments[key] for key in ('camera_id', 'u', 'v')):
        return None
    if any(type(result[key]) not in (int, float) or result[key] < 0
           for key in ('sim_time_s', 'depth_m')) or result['depth_m'] == 0:
        return None
    if (type(result['pixel_center_uv']) is not list
            or result['pixel_center_uv'] != [result['u'] + .5, result['v'] + .5]
            or type(result['world_position_m']) is not list
            or len(result['world_position_m']) != 3
            or any(type(x) not in (int, float) for x in result['world_position_m'])):
        return None
    return result


def program_feedback(program):
    """The shared fixed/free code response; no full trace or local variables."""
    return {'status': 'program_' + program['status'],
            **{key: program.get(key) for key in
               ('tool_calls', 'move_calls', 'error', 'last_result', 'primitive_results',
                'query_results')}}


def _identifier(name, *, assignment=False):
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}', name) or '__' in name:
        raise _Fault('forbidden_identifier', status='rejected')
    if assignment and name in RESERVED:
        raise _Fault('reserved_identifier', status='rejected')


def _integer_literal(node):
    sign = 1
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        sign = -1 if isinstance(node.op, ast.USub) else 1
        node = node.operand
    if not isinstance(node, ast.Constant) or type(node.value) is not int:
        raise _Fault('range_requires_integer_literals', status='rejected')
    return sign * node.value


def _loop_range(node, limits):
    if (not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name)
            or node.func.id != 'range' or node.keywords or not 1 <= len(node.args) <= 3):
        raise _Fault('for_requires_bounded_range', status='rejected')
    values = [_integer_literal(arg) for arg in node.args]
    if any(abs(value) > limits.max_number_abs for value in values):
        raise _Fault('number_limit', status='budget_exceeded')
    try:
        value = range(*values)
    except ValueError:
        raise _Fault('range_step_zero', status='rejected') from None
    if len(value) > limits.max_loop_iterations:
        raise _Fault('loop_iteration_limit', status='budget_exceeded')
    return value


def _validate(tree, limits):
    stack, count = [(tree, 0)], 0
    while stack:
        node, depth = stack.pop()
        count += 1
        if count > limits.max_ast_nodes or depth > limits.max_ast_depth:
            raise _Fault('ast_structure_limit', getattr(node, 'lineno', None), status='budget_exceeded')
        stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))

    def fail(node, code='unsupported_syntax'):
        raise _Fault(code, getattr(node, 'lineno', None), status='rejected')

    def expression(node, tool=False):
        if isinstance(node, ast.Constant):
            try:
                _json_copy(node.value, limits)
            except _Fault as exc:
                exc.line = node.lineno
                if exc.status == 'runtime_error':
                    exc.status = 'rejected'
                raise
        elif isinstance(node, ast.Name):
            _identifier(node.id)
            if node.id in RESERVED:
                fail(node, 'callable_is_not_a_value')
        elif isinstance(node, ast.List):
            if len(node.elts) > limits.max_container_items:
                raise _Fault('container_limit', node.lineno, status='budget_exceeded')
            for child in node.elts:
                expression(child)
        elif isinstance(node, ast.Dict):
            if len(node.keys) > limits.max_container_items:
                raise _Fault('container_limit', node.lineno, status='budget_exceeded')
            keys = []
            for key, value in zip(node.keys, node.values):
                if not isinstance(key, ast.Constant) or type(key.value) is not str or '__' in key.value:
                    fail(node, 'dictionary_requires_literal_string_keys')
                expression(key); expression(value); keys.append(key.value)
            if len(set(keys)) != len(keys):
                fail(node, 'duplicate_dictionary_key')
        elif isinstance(node, ast.Subscript):
            expression(node.value); expression(node.slice)
            if isinstance(node.slice, ast.Constant) and type(node.slice.value) is str and '__' in node.slice.value:
                fail(node, 'forbidden_dictionary_key')
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.UAdd, ast.USub)):
            expression(node.operand)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)):
            expression(node.left); expression(node.right)
        elif isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            for value in node.values:
                expression(value)
        elif isinstance(node, ast.Compare):
            if any(not isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn)) for op in node.ops):
                fail(node)
            expression(node.left)
            for value in node.comparators:
                expression(value)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in TOOLS | PURE:
                fail(node, 'unadvertised_call')
            name = node.func.id
            if name in TOOLS and not tool:
                fail(node, 'nested_tool_call')
            if any(isinstance(value, ast.Starred) for value in node.args) or any(k.arg is None for k in node.keywords):
                fail(node, 'argument_unpacking_forbidden')
            keywords = [keyword.arg for keyword in node.keywords]
            if len(keywords) != len(set(keywords)):
                fail(node, 'duplicate_keyword')
            if name in PURE and (keywords or (name in ('abs', 'len') and len(node.args) != 1)
                                  or (name in ('min', 'max') and not node.args)):
                fail(node, 'invalid_pure_call_signature')
            if name == 'observe' and (node.args or keywords):
                fail(node, 'observe_takes_no_arguments')
            if name == 'move_geometry' and (node.args or not set(keywords) <= MOVE_KEYS or 'duration_s' not in keywords
                    or not set(keywords) & {'wrist_delta_m', 'wrist_rotation_delta_rad', 'fingertip_delta_m'}):
                fail(node, 'invalid_move_signature')
            if name == 'hold' and (len(node.args)>1 or not set(keywords) <= {'duration_s', 'reason'}
                    or (node.args and 'duration_s' in keywords) or (not node.args and 'duration_s' not in keywords)):
                fail(node, 'invalid_hold_signature')
            if name in ('point_at', 'stop'):
                names = ('camera_id', 'u', 'v') if name == 'point_at' else ('reason',)
                if (len(node.args) > len(names) or not set(keywords) <= set(names)
                        or set(names[:len(node.args)]) & set(keywords)
                        or (name == 'point_at' and set(names[:len(node.args)]) | set(keywords) != set(names))):
                    fail(node, 'invalid_tool_signature')
            for value in node.args:
                expression(value)
            for keyword in node.keywords:
                expression(keyword.value)
        else:
            fail(node)

    def statements(body, loop_depth=0):
        for node in body:
            if isinstance(node, ast.Assign):
                if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                    fail(node, 'assignment_requires_one_name')
                _identifier(node.targets[0].id, assignment=True)
                expression(node.value, tool=True)
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                expression(node.value, tool=True)
            elif isinstance(node, ast.If):
                expression(node.test); statements(node.body, loop_depth); statements(node.orelse, loop_depth)
            elif isinstance(node, ast.For):
                if not isinstance(node.target, ast.Name) or node.orelse:
                    fail(node, 'unsupported_for_syntax')
                _identifier(node.target.id, assignment=True)
                _loop_range(node.iter, limits)
                if loop_depth >= limits.max_loop_depth:
                    raise _Fault('loop_depth_limit', node.lineno, status='budget_exceeded')
                statements(node.body, loop_depth + 1)
            elif isinstance(node, (ast.Break, ast.Continue)) and loop_depth:
                continue
            else:
                fail(node)
    statements(tree.body)


class _Interpreter:
    def __init__(self, invoke, limits):
        self.invoke, self.limits = invoke, limits
        self.variables, self.trace = {}, []
        self.primitive_feedback = {}
        self.query_results = {}
        self.fuel = self.iterations = self.trace_bytes = 0
        self.counts = {'tool_calls': 0, 'move_calls': 0, 'observe_calls': 0, 'point_queries': 0}
        self.last_result = None
        self.started = time.monotonic()
        self.line = None

    def tick(self, node):
        self.line = getattr(node, 'lineno', self.line)
        self.fuel += 1
        if self.fuel > self.limits.max_fuel:
            raise _Fault('fuel_limit', self.line, status='budget_exceeded')
        self.deadline()

    def deadline(self):
        if (self.limits.max_runtime_s is not None
                and time.monotonic() - self.started >= self.limits.max_runtime_s):
            raise _Fault('program_wall_limit', self.line, status='budget_exceeded')

    def copy(self, value):
        return _json_copy(value, self.limits)

    def assign(self, name, value):
        copied = self.copy(value)
        if name not in self.variables and len(self.variables) >= self.limits.max_variables:
            raise _Fault('variable_limit', self.line, status='budget_exceeded')
        size = sum(_bytes(item) for key, item in self.variables.items() if key != name) + _bytes(copied)
        if size > self.limits.max_live_value_bytes:
            raise _Fault('live_value_bytes_limit', self.line, status='budget_exceeded')
        self.variables[name] = copied

    def number(self, value):
        if type(value) not in (int, float):
            raise _Fault('numeric_operand_required', self.line)
        return value

    def evaluate(self, node):
        self.tick(node)
        if isinstance(node, ast.Constant):
            value = node.value
        elif isinstance(node, ast.Name):
            if node.id not in self.variables:
                raise _Fault('undefined_variable', node.lineno)
            value = self.variables[node.id]
        elif isinstance(node, ast.List):
            value = [self.evaluate(child) for child in node.elts]
        elif isinstance(node, ast.Dict):
            value = {key.value: self.evaluate(item) for key, item in zip(node.keys, node.values)}
        elif isinstance(node, ast.Subscript):
            parent, key = self.evaluate(node.value), self.evaluate(node.slice)
            if type(parent) is dict:
                if type(key) is not str or '__' in key or key not in parent:
                    raise _Fault('invalid_dictionary_lookup', node.lineno)
            elif type(parent) in (list, str):
                if type(key) is not int or not -len(parent) <= key < len(parent):
                    raise _Fault('invalid_sequence_lookup', node.lineno)
            else:
                raise _Fault('value_is_not_indexable', node.lineno)
            value = parent[key]
        elif isinstance(node, ast.UnaryOp):
            operand = self.evaluate(node.operand)
            value = not operand if isinstance(node.op, ast.Not) else (
                -self.number(operand) if isinstance(node.op, ast.USub) else self.number(operand))
        elif isinstance(node, ast.BinOp):
            left, right = self.number(self.evaluate(node.left)), self.number(self.evaluate(node.right))
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
                raise _Fault('division_by_zero', node.lineno)
            if isinstance(node.op, ast.Add): value = left + right
            elif isinstance(node.op, ast.Sub): value = left - right
            elif isinstance(node.op, ast.Mult): value = left * right
            elif isinstance(node.op, ast.Div): value = left / right
            elif isinstance(node.op, ast.FloorDiv): value = left // right
            else: value = left % right
        elif isinstance(node, ast.BoolOp):
            for child in node.values:
                value = self.evaluate(child)
                if (isinstance(node.op, ast.And) and not value) or (isinstance(node.op, ast.Or) and value):
                    break
        elif isinstance(node, ast.Compare):
            left, value = self.evaluate(node.left), True
            for operator, other in zip(node.ops, node.comparators):
                right = self.evaluate(other)
                if isinstance(operator, (ast.In, ast.NotIn)):
                    if type(right) not in (str, list, dict) or (type(right) in (str, dict) and type(left) is not str):
                        raise _Fault('invalid_membership_operands', node.lineno)
                    matches = left in right
                    matches = not matches if isinstance(operator, ast.NotIn) else matches
                elif isinstance(operator, (ast.Eq, ast.NotEq)):
                    matches = left == right
                    matches = not matches if isinstance(operator, ast.NotEq) else matches
                else:
                    if not ((type(left) in (int, float) and type(right) in (int, float)) or type(left) is type(right) is str):
                        raise _Fault('invalid_comparison_operands', node.lineno)
                    if isinstance(operator, ast.Lt): matches = left < right
                    elif isinstance(operator, ast.LtE): matches = left <= right
                    elif isinstance(operator, ast.Gt): matches = left > right
                    else: matches = left >= right
                if not matches:
                    value = False; break
                left = right
        elif isinstance(node, ast.Call):
            name = node.func.id
            args = [self.evaluate(arg) for arg in node.args]
            kwargs = {keyword.arg: self.evaluate(keyword.value) for keyword in node.keywords}
            if name in TOOLS:
                return self.tool(name, args, kwargs, node.lineno)
            if name == 'abs': value = abs(self.number(args[0]))
            elif name == 'len':
                if type(args[0]) not in (dict, list, str):
                    raise _Fault('len_requires_container', node.lineno)
                value = len(args[0])
            else:
                values = args[0] if len(args) == 1 and type(args[0]) is list else args
                if not values:
                    raise _Fault('empty_min_max', node.lineno)
                numbers = [self.number(item) for item in values]
                value = min(numbers) if name == 'min' else max(numbers)
        else:
            raise _Fault('unsupported_syntax', self.line)
        return self.copy(value)

    def tool(self, name, args, kwargs, line):
        self.line = line
        if name == 'hold' and args:
            kwargs = {'duration_s':args[0], **kwargs}
        if name in ('point_at', 'stop'):
            formals = ('camera_id', 'u', 'v') if name == 'point_at' else ('reason',)
            kwargs = dict(zip(formals, args)) | kwargs
            if name == 'stop': kwargs.setdefault('reason', 'stopped')
        arguments = self.copy(kwargs)
        if name == 'point_at' and (type(arguments['camera_id']) is not str or
                any(type(arguments[key]) is not int or arguments[key] < 0 for key in ('u', 'v'))):
            raise _Fault('invalid_point_arguments', line)
        if name == 'stop' and type(arguments['reason']) is not str:
            raise _Fault('invalid_stop_reason', line)
        categories = {'observe': ('observe_calls', self.limits.max_observe_calls),
                      'move_geometry': ('move_calls', self.limits.max_move_calls),
                      'hold': ('move_calls', self.limits.max_move_calls),
                      'point_at': ('point_queries', self.limits.max_point_queries)}
        category = categories.get(name)
        if self.counts['tool_calls'] >= self.limits.max_tool_calls or (
                category and self.counts[category[0]] >= category[1]):
            raise _Fault('tool_call_limit', line, status='budget_exceeded')
        row = {'index': len(self.trace), 'line': line, 'tool': name,
               'arguments': arguments, 'status': 'started'}
        # Reserve a bounded diagnostic row before any externally visible effect.
        if self.trace_bytes + _bytes(row) + 192 > self.limits.max_trace_bytes:
            raise _Fault('trace_bytes_limit', line, status='budget_exceeded')
        self.deadline()
        self.trace.append(row)
        self.counts['tool_calls'] += 1
        if category: self.counts[category[0]] += 1
        stopped, should_stop, row_bytes = None, False, 0
        try:
            try:
                value = self.invoke(name, self.copy(arguments))
            except ProgramStop as exc:
                stopped, value, should_stop = exc.reason, exc.result, True
            except ActionValidationError as exc:
                raise _Fault('public_action_validation', line, status='tool_error',
                             public_error=({'code': 'action_validation_error'} if name == 'point_at'
                                           else exc.public_metadata())) from None
            except Exception:
                raise _Fault('tool_callback_failed', line, status='tool_error') from None
            if name in ('move_geometry','hold'):
                # This bounded side channel survives value/trace/deadline
                # faults after the trusted callback has physically acted.
                self.primitive_feedback[row['index']]=_minimum_primitive_feedback(value)
            if name == 'point_at' and not should_stop:
                # Keep every verified public query before an unrelated bulky
                # callback field can fail value/trace/deadline accounting.
                point = _minimum_point_result(value, arguments)
                if point is None:
                    raise _Fault('invalid_public_point_result', line, status='tool_error')
                self.query_results[row['index']] = point
            try:
                value = self.copy(value)
            except _Fault as exc:
                if exc.status != 'budget_exceeded': exc.status = 'tool_error'
                exc.line = line
                raise
            if name == 'stop' and not should_stop:
                stopped, should_stop = arguments['reason'], True
            if should_stop:
                if type(stopped) is not str:
                    raise _Fault('invalid_stop_reason', line, status='tool_error')
                stopped = self.copy(stopped)
            completed = row | {'status': 'stopped' if should_stop else 'completed', 'result': value}
            if self.trace_bytes + _bytes(completed) > self.limits.max_trace_bytes:
                raise _Fault('trace_bytes_limit', line, status='budget_exceeded')
            row.update(completed)
            row_bytes = _bytes(row)
            self.trace_bytes += row_bytes
            self.last_result = self.copy(self.query_results.get(row['index']) if name == 'point_at' else value)
            self.deadline()
            if should_stop: raise ProgramStop(stopped, value)
            return self.copy(value)
        except _Fault as exc:
            self.trace_bytes -= row_bytes
            row.update(status='error', error=exc.metadata(line))
            row.pop('result', None)
            self.trace_bytes += _bytes(row)
            raise

    def run(self, statements):
        for node in statements:
            self.tick(node)
            if isinstance(node, ast.Assign):
                self.assign(node.targets[0].id, self.evaluate(node.value))
            elif isinstance(node, ast.Expr):
                self.evaluate(node.value)
            elif isinstance(node, ast.If):
                self.run(node.body if self.evaluate(node.test) else node.orelse)
            elif isinstance(node, ast.For):
                for value in _loop_range(node.iter, self.limits):
                    self.iterations += 1
                    if self.iterations > self.limits.max_total_loop_iterations:
                        raise _Fault('total_loop_iteration_limit', node.lineno, status='budget_exceeded')
                    self.assign(node.target.id, value)
                    try: self.run(node.body)
                    except _Break: break
                    except _Continue: continue
            elif isinstance(node, ast.Break): raise _Break()
            elif isinstance(node, ast.Continue): raise _Continue()


def execute_program(source: str, invoke, *, limits: ProgramLimits | None = None) -> dict:
    """Interpret one bounded program; return explicit status and public tool trace.

    A completed program is not a successful robot task. Rejections are atomic
    with respect to tool calls; runtime/tool/budget faults can follow completed
    tool effects, which remain in the trace. Nothing is rolled back or retried.
    callback ProgramStop and stop() terminate this program immediately.
    """
    limits = limits or ProgramLimits()
    if not isinstance(limits, ProgramLimits):
        raise TypeError('limits must be ProgramLimits')
    runtime = _Interpreter(invoke, limits)
    result = {'protocol': 'agent_program_v2', 'status': 'rejected', 'source_sha256': None,
              'program_sha256': None, 'trace': runtime.trace, 'last_result': None,
              'error': None, 'stop_reason': None}
    try:
        if type(source) is not str:
            raise _Fault('source_must_be_text', status='rejected')
        # Every UTF-8 code point occupies at least one byte. Check character
        # count first so an oversized source cannot allocate an oversized copy.
        if len(source) > limits.max_source_bytes:
            raise _Fault('source_bytes_limit', status='budget_exceeded')
        try: raw = source.encode('utf-8')
        except UnicodeError: raise _Fault('invalid_source_encoding', status='rejected') from None
        if len(raw) > limits.max_source_bytes:
            raise _Fault('source_bytes_limit', status='budget_exceeded')
        result['source_sha256'] = hashlib.sha256(raw).hexdigest()
        source = source.strip()
        if source.startswith('```'):
            match = re.fullmatch(r'```python\s*\n([\s\S]*?)\n```', source)
            if not match or '```' in match.group(1):
                raise _Fault('invalid_code_fence', status='rejected')
            source = match.group(1)
        if not source or '```' in source:
            raise _Fault('empty_or_ambiguous_source', status='rejected')
        result['program_sha256'] = hashlib.sha256(source.encode('utf-8')).hexdigest()
        try: tree = ast.parse(source, mode='exec')
        except (SyntaxError, ValueError, RecursionError) as exc:
            raise _Fault('invalid_syntax', getattr(exc, 'lineno', None), status='rejected') from None
        if not tree.body:
            raise _Fault('empty_program', status='rejected')
        _validate(tree, limits)
        if not callable(invoke):
            raise _Fault('invalid_tool_callback', status='tool_error')
        runtime.run(tree.body)
        result['status'] = 'completed'
    except ProgramStop as exc:
        result.update(status='stopped', stop_reason=exc.reason)
    except _Fault as exc:
        result.update(status=exc.status, error=exc.metadata(exc.line if exc.line is not None else runtime.line))
    except (OverflowError, ArithmeticError):
        result.update(status='runtime_error', error={'code': 'arithmetic_error', 'line': runtime.line})
    except Exception:
        # Never send exception text, traceback, host path or callback object to
        # the generated program or next model request.
        result.update(status='runtime_error', error={'code': 'interpreter_error', 'line': runtime.line})
    result.update(runtime.counts, fuel_used=runtime.fuel, loop_iterations=runtime.iterations,
                  last_result=runtime.last_result)
    # Per-primitive outcomes survive later successful moves and read-only calls.
    # Its bounded execution feedback survives even if the bulky trace result
    # exceeded a limit after the action. No observations are copied here.
    result['primitive_results'] = []
    for row in runtime.trace:
        if row['tool'] not in ('move_geometry', 'hold'):
            continue
        feedback = runtime.primitive_feedback.get(row['index'])
        result['primitive_results'].append({'tool_index': row['index'], 'tool': row['tool'],
            'call_status': row['status'], 'feedback': feedback, 'error': row.get('error')})
    result['query_results'] = [
        {'tool_index': row['index'], 'tool': 'point_at', 'arguments': row['arguments'],
         'call_status': row['status'], 'result': runtime.query_results.get(row['index']),
         'error': row.get('error')}
        for row in runtime.trace if row['tool'] == 'point_at']
    return result
