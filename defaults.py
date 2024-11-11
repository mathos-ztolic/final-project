import math
import cmath
import statistics
import functools
import random
from typing import Callable
import itertools
from sets import (
    UNIVERSAL_SET, INTEGERS, NATURALS, NATURALS_WITH_ZERO, REALS, LISTS,
    FUNCTIONS, COMPLEX, EMPTY_SET, SETS, QuestionMark, Star, Plus,
    AtLeast, AtMost, Between, Exactly, Pack, Unpack, SetOf,
    Unsliceable
)
from custom_types import (
    CallableWrapper, GeneralSlice, BLANK_SLICE, Vector,
    classmethod as _classmethod, method
)

def one(args):
    seen_true = False
    for arg in args:
        if arg and seen_true:
            return False
        elif arg:
            seen_true = True
    return seen_true

def trymath(mathfun: Callable, cmathfun: Callable) -> Callable:
    def fun(*args):
        try:
            return mathfun(*args)
        except (ValueError, TypeError):
            return cmathfun(*args)
    return fun


def proxy_or_value(k, v):
    if hasattr(v, '__name__'):
        proxy = lambda *args: v(*args)
        proxy.__name__ = k
        proxy.__qualname__ = k
        return proxy
    return v
    

method_info = {
    str.upper: ('string.upper', 'uppercases the string')
}


def remap(v, i, I, o=None, O=None):
    if o is O is None:
        o, O = 0, 1
    elif o is None or O is None:
        raise ValueError("Function either takes 5 or 3 arguments.")
    return o + (O - o) * ((v - i) / (I - i))


function_signatures = {
    'sqrt': "[0,∞⟩ -> ℝ or ℂ -> ℂ",
    'cbrt': "ℝ -> ℝ or ℂ -> ℂ",
    'root_': "(ℝ\\{0})×ℝ -> ℝ",
    'argument': "ℂ -> [-π,π]",
    'rect': "ℝ×ℝ -> ℝ",
    'sin': "ℝ -> ℝ or ℂ -> ℂ",
    'cos': "ℝ -> ℝ or ℂ -> ℂ",
    'tan': "ℝ\\{π/2 + nπ | n∈ℤ} -> ℝ or ℂ\\{π/2 + nπ | n∈ℤ} -> ℂ",
    'arcsin': "[-1,1] -> ℝ or ℂ -> ℂ",
    'arccos': "[-1,1] -> ℝ or ℂ -> ℂ",
    'arctan': "ℝ -> ℝ or ℂ -> ℂ",
    'sinh': "ℝ -> ℝ or ℂ -> ℂ",
    'cosh': "ℝ -> ℝ or ℂ -> ℂ",
    'tanh': "ℝ -> ℝ or ℂ -> ℂ",
    'arcsinh': "ℝ -> ℝ or ℂ -> ℂ",
    'arccosh': "[1,∞⟩ -> ℝ or ℂ -> ℂ",
    'arctanh': "⟨-1,1⟩ -> ℝ or ℂ -> ℂ\\{z∈ℂ | z² = 1} -> ℂ",
    'atan2': "ℝ×ℝ -> ℝ",
    'min': "Ord a => Λ[1,,a]×[null]? -> a or Ord b => Λ[1,,a]×Φ[a->b] -> a",
    'max': "Ord a => Λ[1,,a]×[null]? -> a or Ord b => Λ[1,,a]×Φ[a->b] -> a",
    'abs': "ℝ -> ℝ or ℂ -> ℝ",
    'lcm': "ℤ* -> ℤ",
    'gcd': 'ℤ* -> ℤ',
}
defaults = {
    'sqrt': trymath(math.sqrt, cmath.sqrt),
    'cbrt': trymath(math.cbrt, lambda arg: arg**(1/3)),
    'root_': lambda n, arg: arg**1/n,
    'argument': cmath.phase,
    'rect': cmath.rect,
    'sin': trymath(math.sin, cmath.sin),
    'cos': trymath(math.cos, cmath.cos),
    'tan': trymath(math.tan, cmath.tan),
    'arcsin': trymath(math.asin, cmath.asin),
    'arccos': trymath(math.acos, cmath.acos),
    'arctan': trymath(math.atan, cmath.atan),
    'sinh': trymath(math.sinh, cmath.sinh),
    'cosh': trymath(math.cosh, cmath.cosh),
    'tanh': trymath(math.tanh, cmath.tanh),
    'arcsinh': trymath(math.asinh, cmath.asinh),
    'arccosh': trymath(math.acosh, cmath.acosh),
    'arctanh': trymath(math.atanh, cmath.atanh),
    'atan2': math.atan2,
    'min': (lambda args, key=None: min(args, key=key)),
    'max': (lambda args, key=None: max(args, key=key)),
    'abs': abs,
    'gcd': lambda *args: math.gcd(
        *map(lambda arg: Unsliceable(INTEGERS)[arg], args)
    ),
    'lcm': lambda *args: math.lcm(
        *map(lambda arg: Unsliceable(INTEGERS)[arg], args)
    ),
    'printreturn': lambda arg: (print('got value:', arg), arg)[1],
    'slice': lambda *args: (
        BLANK_SLICE if not args else
        GeneralSlice(*args, None) if len(args) == 1 else
        GeneralSlice(*args)
    ),
    'floor': trymath(
        math.floor, lambda z: math.floor(z.real) + math.floor(z.imag)*1j
    ),
    'ceil': math.ceil,
    'round': round, 'trunc': math.trunc,
    'frac': (lambda x: x - int(x)),
    'exp': trymath(math.exp, cmath.exp),
    'ln': trymath(lambda x: math.log(x), lambda x: cmath.log(x)),
    'lg': trymath(math.log2, lambda arg: cmath.log(arg, 2)),
    'log': trymath(math.log10, cmath.log10),
    'log_': trymath(lambda b, x: math.log(x, b), lambda b, x: cmath.log(x, b)),
    'nPr': math.perm, 'nCr': math.comb,
    'sgn': (lambda x: -1 if x < 0 else 1 if x > 0 else 0),
    'dist': math.dist, 'deg': math.degrees, 'rad': math.radians,
    'random': random.random, 'randint': random.randint,
    'choice': random.choice,
    'mean': statistics.fmean,
    'arithmetic_mean': statistics.fmean,
    'geometric_mean': statistics.geometric_mean,
    'harmonic_mean': statistics.harmonic_mean,
    'median': statistics.median,
    'low_median': statistics.median_low,
    'high_median': statistics.median_high,
    'mode': statistics.mode,
    'all': all,
    'any': any,
    'one': one,
    'distinct': lambda args: len(set(float(arg) for arg in args)) == len(args),
    'same': lambda args: len(set(float(arg) for arg in args)) <= 1,
    'lerp': lambda a, b, t: (1-t)*a + t*b,
    'clamp': lambda v, m, M: m if v < m else M if v > m else v,
    'remap': remap,
    'erf': math.erf, 'erfc': math.erfc,
    'map': lambda *args: list(map(*args)),
    'filter': lambda *args: list(filter(*args)),
    'sum': sum,
    'foldl': CallableWrapper(
        lambda f, a, xs: functools.reduce(f, xs, a),
        information='Φ[a×b->a] × a × Λ[,,b] -> a'
    ),
    'foldr': CallableWrapper(
        lambda f, a, xs: functools.reduce(
            lambda x, y: f(y,x), reversed(xs), a
        ),
        information='Φ[b×a->a] × a × Λ[,,b] -> a'
    ),
    'apply': CallableWrapper(
        lambda f, *args: f(*args),
        information='Φ[... -> a] -> a'
    ),
    'sort': CallableWrapper(
        sorted,
        information='Λ[,,a] -> Λ[,,a]'
    ),
    'zip': lambda *args: [list(x) for x in zip(*args)],
    'zip_longest': lambda default, *args: [
        list(x) for x in itertools.zip_longest(*args, fillvalue=default)
    ],
    'reverse': lambda l: list(reversed(l)),
    'identity': lambda x: x,
    'vector': Vector,
    'dict': dict,
    'type': type,
    'classmethod': _classmethod,
    'method': method,

    'pi': math.pi,
    'e': math.e,
    'tau': math.tau,
    'euler_mascheroni': 0.57721566490153286060,
    'omega': 0.56714329040978387299,
    'phi': 0.5 + 0.5*math.sqrt(5),
    'inf': math.inf, 'nan': math.nan,
    'infi': cmath.infj, 'nani': cmath.nanj,
    'i': 1j,
    'Z': INTEGERS,
    'R': REALS,
    'L': LISTS,
    'F': FUNCTIONS,
    'N': NATURALS,
    'N0': NATURALS_WITH_ZERO,
    'C': COMPLEX,
    'E': EMPTY_SET,
    'U': UNIVERSAL_SET,
    'S': SETS,
    'Star': Star,
    'Plus': Plus,
    'QuestionMark': QuestionMark,
    'AtLeast': AtLeast,
    'AtMost': AtMost,
    'Between': Between,
    'Exactly': Exactly,
    'Pack': Pack,
    'Unpack': Unpack,
    'true': True,
    'false': False,
    'null': None,
    'string': str,
    'integer': int,
    'float': float,
    'complex': complex,
    'list': list,
    'boolean': bool,
    'dir': lambda x: [
        e for e in dir(x) if not e.startswith('__') or not e.endswith('__')
    ],
    'SetOf': SetOf,
}