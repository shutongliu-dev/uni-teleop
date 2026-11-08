"""JSON-valued skill language with Python syntax; never Python exec/eval.

Only literal ``api.call`` and ``api.intent`` syntax crosses a
trusted callback. Imports, host objects, attribute access and arbitrary Python
are unavailable. Runtime fuel does not interrupt a synchronous callback: the
host bridge must enforce its own request and episode deadlines.
"""
from dataclasses import dataclass, fields
import ast
import hashlib
import json
import math
import re
import time

from ..program import _bytes


LANGUAGE = 'agent_json_skill_ast.v1'
PURE = frozenset(('abs', 'min', 'max', 'len', 'sum', 'sqrt', 'sin', 'cos', 'tan',
                  'asin', 'acos', 'atan2', 'floor', 'ceil', 'get', 'keys'))
RESERVED = PURE | {'api', 'range'}


@dataclass(frozen=True)
class SkillLanguageLimits:
    max_source_bytes: int = 65536
    max_ast_nodes: int = 8192
    max_ast_depth: int = 48
    max_fuel: int = 100000
    max_loop_iterations: int = 1024
    max_total_loop_iterations: int = 4096
    max_loop_depth: int = 4
    max_tool_calls: int = 64
    max_move_calls: int = 32
    max_variables: int = 128
    max_value_nodes: int = 50000
    max_value_bytes: int = 2 * 1024 * 1024
    max_live_value_bytes: int = 4 * 1024 * 1024
    max_container_items: int = 4096
    max_value_depth: int = 24
    max_string_length: int = 65536
    max_number_abs: float = 1e9
    max_runtime_s: float | None = 1800.
    max_function_depth: int = 8
    max_functions: int = 32
    max_function_parameters: int = 32
    max_total_value_copy_bytes: int = 128 * 1024 * 1024

    def __post_init__(self):
        for item in fields(self):
            value = getattr(self,item.name)
            if item.name == 'max_runtime_s' and value is None: continue
            if item.name in ('max_number_abs','max_runtime_s'):
                if type(value) not in (int,float) or not math.isfinite(value) or value <= 0:
                    raise ValueError('Language numeric limits must be finite and positive')
            elif type(value) is not int or value < 1:
                raise ValueError('Language count limits must be positive integers')


class SkillLanguageError(ValueError):
    def __init__(self, code, line=None, *, status='rejected'):
        self.code, self.line, self.status = code, line, status
        super().__init__(code)


class SkillStop(Exception):
    """Host stop after an actual call, with its original public feedback."""
    def __init__(self, reason, feedback=None):
        self.reason, self.feedback = reason, feedback
        super().__init__(reason)


class _Return(Exception):
    def __init__(self, value): self.value = value


class _Break(Exception): pass
class _Continue(Exception): pass


def bounded_json_copy(value, limits):
    """Bound encoded size incrementally, before serializing a whole container."""
    nodes, size, active = 0, 0, set()
    def visit(item, depth):
        nonlocal nodes, size
        nodes += 1
        if nodes > limits.max_value_nodes or depth > limits.max_value_depth:
            raise SkillLanguageError('value_structure_limit',status='budget_exceeded')
        kind = type(item)
        if item is None or kind is bool: copied = item
        elif kind in (int,float):
            if kind is float and not math.isfinite(item): raise SkillLanguageError('nonfinite_number',status='runtime_error')
            if abs(item) > limits.max_number_abs: raise SkillLanguageError('number_limit',status='budget_exceeded')
            copied = item
        elif kind is str:
            if len(item) > limits.max_string_length: raise SkillLanguageError('string_limit',status='budget_exceeded')
            copied = item
        elif kind in (list,dict):
            if id(item) in active: raise SkillLanguageError('cyclic_value',status='runtime_error')
            if len(item) > limits.max_container_items: raise SkillLanguageError('container_limit',status='budget_exceeded')
            size += 2 + max(0,len(item)-1) + (len(item) if kind is dict else 0)
            if size > limits.max_value_bytes: raise SkillLanguageError('value_bytes_limit',status='budget_exceeded')
            active.add(id(item))
            try:
                if kind is list: return [visit(child,depth+1) for child in item]
                result = {}
                for key,child in item.items():
                    if type(key) is not str or '__' in key: raise SkillLanguageError('invalid_dictionary_key',status='runtime_error')
                    result[visit(key,depth+1)] = visit(child,depth+1)
                return result
            finally: active.remove(id(item))
        else: raise SkillLanguageError('non_json_value',status='runtime_error')
        size += len(json.dumps(copied,ensure_ascii=True,allow_nan=False,separators=(',',':')).encode())
        if size > limits.max_value_bytes: raise SkillLanguageError('value_bytes_limit',status='budget_exceeded')
        return copied
    return visit(value,0)


def _name(value, node=None, *, assignment=False):
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}', value) or '__' in value:
        raise SkillLanguageError('forbidden_identifier', getattr(node, 'lineno', None))
    if assignment and value in RESERVED:
        raise SkillLanguageError('reserved_identifier', getattr(node, 'lineno', None))


def validate_source(source, limits=None):
    """Validate the whole module before any public API or library mutation."""
    limits = limits or SkillLanguageLimits()
    if type(source) is not str or len(source) > limits.max_source_bytes:
        raise SkillLanguageError('source_bytes_limit')
    try:
        if len(source.encode()) > limits.max_source_bytes:
            raise SkillLanguageError('source_bytes_limit')
        tree = ast.parse(source)
    except (SyntaxError, UnicodeError, ValueError, RecursionError) as error:
        if isinstance(error, SkillLanguageError): raise
        raise SkillLanguageError('invalid_syntax', getattr(error, 'lineno', None)) from None
    stack, count = [(tree, 0)], 0
    while stack:
        node, depth = stack.pop(); count += 1
        if count > limits.max_ast_nodes or depth > limits.max_ast_depth:
            raise SkillLanguageError('ast_structure_limit', getattr(node, 'lineno', None))
        stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    functions = {}
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and type(node.value.value) is str:
            try: bounded_json_copy(node.value.value, limits)
            except SkillLanguageError as error:
                raise SkillLanguageError(error.code,node.lineno) from None
            continue
        if not isinstance(node, ast.FunctionDef):
            raise SkillLanguageError('module_requires_functions', getattr(node, 'lineno', None))
        _name(node.name, node, assignment=True)
        names = [item.arg for item in node.args.args]
        if (node.name in functions or len(names) > limits.max_function_parameters
                or len(names) != len(set(names)) or node.decorator_list
                or node.returns or node.args.posonlyargs or node.args.kwonlyargs
                or node.args.vararg or node.args.kwarg or node.args.defaults or node.type_comment
                or any(item.annotation for item in node.args.args)):
            raise SkillLanguageError('unsupported_function_signature', node.lineno)
        for name in names: _name(name, node)
        if node.name == 'run':
            if names != ['api', 'args', 'state']:
                raise SkillLanguageError('entrypoint_signature', node.lineno)
        elif any(name in RESERVED for name in names):
            raise SkillLanguageError('reserved_parameter', node.lineno)
        functions[node.name] = node
    if 'run' not in functions or len(functions) > limits.max_functions:
        raise SkillLanguageError('entrypoint_or_function_limit')

    def fail(node, code='unsupported_syntax'):
        raise SkillLanguageError(code, getattr(node, 'lineno', None))

    def expr(node):
        if isinstance(node, ast.Constant):
            try: bounded_json_copy(node.value, limits)
            except SkillLanguageError as error: fail(node, error.code)
        elif isinstance(node, ast.Name):
            _name(node.id, node)
            if node.id in RESERVED | set(functions): fail(node, 'callable_is_not_a_value')
        elif isinstance(node, (ast.List, ast.Tuple)):
            if len(node.elts) > limits.max_container_items: fail(node, 'container_limit')
            for item in node.elts: expr(item)
        elif isinstance(node, ast.Dict):
            if len(node.keys) > limits.max_container_items: fail(node, 'container_limit')
            seen = set()
            for key, value in zip(node.keys, node.values):
                if key is None: fail(node, 'unpacking_forbidden')
                expr(key); expr(value)
                if isinstance(key, ast.Constant) and type(key.value) is str:
                    if key.value in seen: fail(node, 'duplicate_dictionary_key')
                    seen.add(key.value)
        elif isinstance(node, ast.Subscript): expr(node.value); expr(node.slice)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.UAdd, ast.USub)):
            expr(node.operand)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)):
            expr(node.left); expr(node.right)
        elif isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            for item in node.values: expr(item)
        elif isinstance(node, ast.Compare):
            if any(not isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn)) for op in node.ops): fail(node)
            expr(node.left)
            for item in node.comparators: expr(item)
        elif isinstance(node, ast.IfExp): expr(node.test); expr(node.body); expr(node.orelse)
        elif isinstance(node, ast.Call):
            if len(node.args) > limits.max_function_parameters: fail(node,'function_argument_limit')
            if node.keywords or any(isinstance(item, ast.Starred) for item in node.args): fail(node, 'positional_arguments_only')
            if isinstance(node.func, ast.Name) and node.func.id in PURE | set(functions):
                if node.func.id == 'run': fail(node, 'entrypoint_not_callable')
            elif (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == 'api' and node.func.attr in ('call', 'intent')):
                count = {'call': 2, 'intent': 1}[node.func.attr]
                if len(node.args) != count: fail(node, 'api_signature')
            else: fail(node, 'unadvertised_call')
            for item in node.args: expr(item)
        else: fail(node)

    def target(node):
        if isinstance(node, ast.Name): _name(node.id, node, assignment=True)
        elif isinstance(node, ast.Subscript):
            target(node.value); expr(node.slice)
        else: fail(node, 'assignment_target')

    def statements(body, loop=0):
        for node in body:
            if isinstance(node, ast.Assign):
                if len(node.targets) != 1: fail(node, 'single_assignment_target')
                target(node.targets[0]); expr(node.value)
            elif isinstance(node, ast.AugAssign):
                if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)): fail(node)
                target(node.target); expr(node.value)
            elif isinstance(node, ast.Expr): expr(node.value)
            elif isinstance(node, ast.Return):
                if node.value is not None: expr(node.value)
            elif isinstance(node, ast.If):
                expr(node.test); statements(node.body, loop); statements(node.orelse, loop)
            elif isinstance(node, (ast.For, ast.While)):
                if loop >= limits.max_loop_depth or node.orelse: fail(node, 'loop_depth_or_else')
                if isinstance(node, ast.For):
                    if not isinstance(node.target, ast.Name): fail(node, 'loop_requires_name')
                    target(node.target)
                    if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range':
                        if node.iter.keywords or not 1 <= len(node.iter.args) <= 3: fail(node, 'range_signature')
                        for item in node.iter.args: expr(item)
                    else: expr(node.iter)
                else: expr(node.test)
                statements(node.body, loop + 1)
            elif isinstance(node, (ast.Break, ast.Continue)) and loop: pass
            elif isinstance(node, ast.Pass): pass
            else: fail(node)
    for node in functions.values(): statements(node.body)
    return functions


class _Interpreter:
    def __init__(self, functions, invoke, limits):
        self.functions, self.invoke, self.limits = functions, invoke, limits
        self.frames = []
        self.fuel = self.iterations = self.api_calls = 0
        self.copied_bytes = 0
        self.started = time.monotonic()
        self.line = None
        self.last_feedback = None

    def tick(self, node=None):
        self.line = getattr(node, 'lineno', self.line)
        self.fuel += 1
        if self.fuel > self.limits.max_fuel: self.fail('fuel_limit', budget=True)
        if self.limits.max_runtime_s is not None and time.monotonic() - self.started >= self.limits.max_runtime_s:
            self.fail('runtime_wall_limit', budget=True)

    def fail(self, code, *, budget=False):
        raise SkillLanguageError(code, self.line, status='budget_exceeded' if budget else 'runtime_error')

    def copy(self, value):
        try:
            copied = bounded_json_copy(value,self.limits)
            self.copied_bytes += _bytes(copied)
            if self.copied_bytes > self.limits.max_total_value_copy_bytes:
                self.fail('total_value_copy_bytes_limit',budget=True)
            return copied
        except SkillLanguageError as error:
            raise SkillLanguageError(error.code, self.line, status=error.status) from None

    def number(self, value):
        if type(value) not in (int, float): self.fail('numeric_operand_required')
        return value

    def assign(self, name, value):
        frame = self.frames[-1]
        if name not in frame and len(frame) >= self.limits.max_variables: self.fail('variable_limit', budget=True)
        value = self.copy(value)
        live = sum(_bytes(v) for f in self.frames for k,v in f.items() if f is not frame or k != name)
        if live + _bytes(value) > self.limits.max_live_value_bytes: self.fail('live_value_bytes_limit', budget=True)
        frame[name] = value

    def lookup(self, parent, key):
        if type(parent) is dict:
            if type(key) is not str or '__' in key or key not in parent: self.fail('invalid_dictionary_lookup')
        elif type(parent) in (list, str):
            if type(key) is not int or not -len(parent) <= key < len(parent): self.fail('invalid_sequence_lookup')
        else: self.fail('not_indexable')
        return parent[key]

    def put(self, target, value):
        if isinstance(target, ast.Name): self.assign(target.id, value); return
        keys, node = [], target
        while isinstance(node, ast.Subscript):
            keys.append(self.eval(node.slice)); node = node.value
        if node.id not in self.frames[-1]: self.fail('undefined_variable')
        root = self.copy(self.frames[-1][node.id]); current = root
        for key in reversed(keys[1:]): current = self.lookup(current, key)
        key = keys[0]
        if type(current) is dict:
            if type(key) is not str or '__' in key: self.fail('invalid_dictionary_key')
        elif type(current) is list: self.lookup(current, key)
        else: self.fail('assignment_target_not_container')
        current[key] = self.copy(value)
        self.assign(node.id, root)

    def binary(self, op, left, right):
        if isinstance(op, ast.Add) and type(left) is type(right) and type(left) in (list, str):
            if len(left) + len(right) > (self.limits.max_container_items if type(left) is list else self.limits.max_string_length):
                self.fail('concatenation_limit', budget=True)
            return left + right
        left, right = self.number(left), self.number(right)
        if isinstance(op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0: self.fail('division_by_zero')
        if isinstance(op, ast.Add): return left + right
        if isinstance(op, ast.Sub): return left - right
        if isinstance(op, ast.Mult): return left * right
        if isinstance(op, ast.Div): return left / right
        if isinstance(op, ast.FloorDiv): return left // right
        return left % right

    def pure(self, name, args):
        if name == 'get':
            if len(args) not in (2,3) or type(args[0]) is not dict or type(args[1]) is not str or '__' in args[1]: self.fail('invalid_get_arguments')
            return args[0].get(args[1], args[2] if len(args) == 3 else None)
        if name in ('keys','len'):
            if len(args) != 1 or type(args[0]) not in ((dict,) if name == 'keys' else (dict,list,str)): self.fail('invalid_container_argument')
            return list(args[0]) if name == 'keys' else len(args[0])
        if name in ('min','max','sum'):
            values = args[0] if len(args) == 1 and type(args[0]) is list else args
            if (not values and name != 'sum') or (name == 'sum' and (len(args) != 1 or type(args[0]) is not list)): self.fail('invalid_numeric_sequence')
            values = [self.number(v) for v in values]
            return {'min': min, 'max': max, 'sum': sum}[name](values)
        if len(args) != (2 if name == 'atan2' else 1): self.fail('pure_function_signature')
        values = [self.number(v) for v in args]
        return abs(values[0]) if name == 'abs' else getattr(math, name)(*values)

    def eval(self, node):
        self.tick(node)
        if isinstance(node, ast.Constant): value = node.value
        elif isinstance(node, ast.Name):
            if node.id not in self.frames[-1]: self.fail('undefined_variable')
            value = self.frames[-1][node.id]
        elif isinstance(node, (ast.List, ast.Tuple)): value = [self.eval(item) for item in node.elts]
        elif isinstance(node, ast.Dict):
            value = {}
            for k,v in zip(node.keys,node.values):
                key = self.eval(k)
                if type(key) is not str or '__' in key or key in value: self.fail('invalid_or_duplicate_dictionary_key')
                value[key] = self.eval(v)
        elif isinstance(node, ast.Subscript): value = self.lookup(self.eval(node.value), self.eval(node.slice))
        elif isinstance(node, ast.UnaryOp):
            arg = self.eval(node.operand)
            value = not arg if isinstance(node.op, ast.Not) else (-self.number(arg) if isinstance(node.op, ast.USub) else self.number(arg))
        elif isinstance(node, ast.BinOp): value = self.binary(node.op, self.eval(node.left), self.eval(node.right))
        elif isinstance(node, ast.BoolOp):
            for item in node.values:
                value = self.eval(item)
                if isinstance(node.op, ast.And) and not value or isinstance(node.op, ast.Or) and value: break
        elif isinstance(node, ast.Compare):
            left, value = self.eval(node.left), True
            for op, child in zip(node.ops,node.comparators):
                right = self.eval(child)
                if isinstance(op, (ast.In, ast.NotIn)):
                    if type(right) not in (str,list,dict) or (type(right) in (str,dict) and type(left) is not str): self.fail('invalid_membership')
                    match = left in right
                    if isinstance(op, ast.NotIn): match = not match
                elif isinstance(op, (ast.Eq,ast.NotEq)):
                    match = left == right
                    if isinstance(op, ast.NotEq): match = not match
                else:
                    if not (type(left) in (int,float) and type(right) in (int,float) or type(left) is type(right) is str): self.fail('invalid_comparison')
                    if isinstance(op, ast.Lt): match = left < right
                    elif isinstance(op, ast.LtE): match = left <= right
                    elif isinstance(op, ast.Gt): match = left > right
                    else: match = left >= right
                if not match: value = False; break
                left = right
        elif isinstance(node, ast.IfExp): value = self.eval(node.body if self.eval(node.test) else node.orelse)
        elif isinstance(node, ast.Call):
            args = [self.eval(item) for item in node.args]
            if isinstance(node.func, ast.Name):
                value = self.pure(node.func.id,args) if node.func.id in PURE else self.function(node.func.id,args)
            else:
                if self.api_calls >= self.limits.max_tool_calls: self.fail('api_call_limit',budget=True)
                self.api_calls += 1
                value = self.invoke(node.func.attr,self.copy(args))
                self.last_feedback = self.copy(value)
                self.tick(node)
        else: self.fail('unsupported_syntax')
        return self.copy(value)

    def function(self, name, args):
        node = self.functions[name]
        names = [item.arg for item in node.args.args]
        if name == 'run': names = names[1:]
        if len(args) != len(names): self.fail('function_argument_count')
        if len(self.frames) >= self.limits.max_function_depth: self.fail('function_depth_limit',budget=True)
        self.frames.append({})
        try:
            for key,value in zip(names,args): self.assign(key,value)
            try: self.block(node.body)
            except _Return as result: return result.value
            return None
        finally: self.frames.pop()

    def loop_tick(self, count):
        self.iterations += 1
        if count > self.limits.max_loop_iterations or self.iterations > self.limits.max_total_loop_iterations:
            self.fail('loop_iteration_limit',budget=True)

    def block(self, body):
        for node in body:
            self.tick(node)
            if isinstance(node, ast.Assign): self.put(node.targets[0],self.eval(node.value))
            elif isinstance(node, ast.AugAssign): self.put(node.target,self.binary(node.op,self.eval(node.target),self.eval(node.value)))
            elif isinstance(node, ast.Expr): self.eval(node.value)
            elif isinstance(node, ast.Return): raise _Return(self.eval(node.value) if node.value is not None else None)
            elif isinstance(node, ast.If): self.block(node.body if self.eval(node.test) else node.orelse)
            elif isinstance(node, (ast.For,ast.While)):
                count = 0
                if isinstance(node,ast.For):
                    if isinstance(node.iter,ast.Call) and isinstance(node.iter.func,ast.Name) and node.iter.func.id == 'range':
                        args = [self.eval(item) for item in node.iter.args]
                        if any(type(v) is not int for v in args): self.fail('range_requires_integers')
                        values = range(*args)
                    else:
                        values = self.eval(node.iter)
                        if type(values) not in (list,dict): self.fail('loop_requires_list_or_keys')
                    if len(values) > self.limits.max_loop_iterations: self.fail('loop_iteration_limit',budget=True)
                    iterator = iter(values)
                while True:
                    self.tick(node)
                    if isinstance(node,ast.For):
                        try: item = next(iterator)
                        except StopIteration: break
                        self.assign(node.target.id,item)
                    elif not self.eval(node.test): break
                    count += 1; self.loop_tick(count)
                    try: self.block(node.body)
                    except _Break: break
                    except _Continue: continue
            elif isinstance(node, ast.Break): raise _Break()
            elif isinstance(node, ast.Continue): raise _Continue()


def execute_source(source, args, state, invoke, *, limits=None):
    limits = limits or SkillLanguageLimits()
    result = {'language': LANGUAGE, 'status': 'rejected', 'source_sha256': None,
              'return_value': None, 'last_feedback': None, 'error': None, 'stop_reason': None}
    interpreter = None
    try:
        functions = validate_source(source,limits)
        result['source_sha256'] = hashlib.sha256(source.encode()).hexdigest()
        interpreter = _Interpreter(functions,invoke,limits)
        args, state = interpreter.copy(args),interpreter.copy(state)
        if type(args) is not dict or type(state) is not dict:
            raise SkillLanguageError('args_and_state_must_be_objects')
        result['return_value'] = interpreter.function('run',[args,state])
        result['status'] = 'completed'
    except SkillStop as stop:
        result.update(status='stopped',stop_reason=stop.reason,last_feedback=stop.feedback)
    except SkillLanguageError as error:
        result.update(status=error.status,error={'code':error.code,'line':error.line})
    except (ArithmeticError,ValueError,TypeError,OverflowError):
        result.update(status='runtime_error',error={'code':'invalid_calculation','line':interpreter.line if interpreter else None})
    except Exception:
        result.update(status='runtime_error',error={'code':'runtime_internal_error','line':interpreter.line if interpreter else None})
    result.update(fuel_used=interpreter.fuel if interpreter else 0,
                  loop_iterations=interpreter.iterations if interpreter else 0,
                  value_copy_bytes=interpreter.copied_bytes if interpreter else 0,
                  api_calls=interpreter.api_calls if interpreter else 0)
    if interpreter and result['last_feedback'] is None: result['last_feedback'] = interpreter.last_feedback
    return result
