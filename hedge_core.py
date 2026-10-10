"""Numerical benchmark and restricted arithmetic for the level-5 hedge simulator.

Black–Scholes is a reference premium/delta model, not a claim about market prices.
Formula evaluation walks an allowlisted AST; it never executes Python source.
"""
import ast
import operator
import re

import numpy as np
from scipy.special import ndtr as N


def bs_call(S, K, T, r, sigma):
    S = np.asarray(S, dtype=float)
    if K <= 0 or T <= 0 or sigma <= 0 or np.any(S <= 0):
        raise ValueError('S, K, T and sigma must be positive')
    st = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / st
    return S * N(d1) - K * np.exp(-r * T) * N(d1 - st)


class BadExpression(ValueError):
    pass


FUNCS = dict(exp=np.exp, log=np.log, sqrt=np.sqrt, abs=np.abs,
             maximum=np.maximum, minimum=np.minimum)
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: np.divide, ast.Pow: np.power}


def last_line(text, pattern):
    text = re.sub(r'<think>.*?</think>', '', text or '', flags=re.S)
    matches = re.findall(r'^\s*' + pattern + r'\s*$', text, re.M)
    return matches[-1].strip().strip('`') if matches else None


def compile_expr(expression, variables):
    if not isinstance(expression, str) or not expression or len(expression) > 600:
        raise BadExpression('expression must contain 1–600 characters')
    try:
        tree = ast.parse(expression, mode='eval')
    except (SyntaxError, RecursionError) as exc:
        raise BadExpression('invalid expression') from exc
    if sum(1 for _ in ast.walk(tree)) > 150:
        raise BadExpression('expression is too complex')
    names = set(variables)

    def validate(node):
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float):
                raise BadExpression('only numeric constants are allowed')
            try:
                finite = np.isfinite(float(node.value))
            except (OverflowError, ValueError):
                finite = False
            if not finite:
                raise BadExpression('constants must be finite')
        elif isinstance(node, ast.Name):
            if node.id not in names | {'pi'}:
                raise BadExpression(f'unknown variable: {node.id}')
        elif isinstance(node, ast.BinOp) and type(node.op) in OPS:
            validate(node.left)
            validate(node.right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            validate(node.operand)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCS:
            arity = 2 if node.func.id in ('minimum', 'maximum') else 1
            if node.keywords or len(node.args) != arity:
                raise BadExpression(f'{node.func.id} requires {arity} positional argument(s)')
            for arg in node.args:
                validate(arg)
        else:
            raise BadExpression('only arithmetic and the documented functions are allowed')

    validate(tree.body)

    def evaluate(**env):
        def walk(node):
            if isinstance(node, ast.Constant):
                return float(node.value)
            if isinstance(node, ast.Name):
                return np.pi if node.id == 'pi' else env[node.id]
            if isinstance(node, ast.BinOp):
                return OPS[type(node.op)](walk(node.left), walk(node.right))
            if isinstance(node, ast.UnaryOp):
                value = walk(node.operand)
                return -value if isinstance(node.op, ast.USub) else value
            return FUNCS[node.func.id](*(walk(arg) for arg in node.args))
        with np.errstate(all='ignore'):
            return walk(tree.body)
    return evaluate
