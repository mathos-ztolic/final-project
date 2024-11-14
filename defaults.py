import math
import cmath
import statistics
import functools
import random
from typing import Callable
import itertools
from sets import (
    UNIVERSAL_SET, INTEGERS, NATURALS, NATURALS_WITH_ZERO, REALS,
    POSITIVE_REALS, NEGATIVE_REALS, NONNEGATIVE_REALS, NONPOSITIVE_REALS,
    FUNCTIONS, COMPLEX, EMPTY_SET, SETS, QuestionMark, Star, Plus,
    AtLeast, AtMost, Between, Exactly, Pack, Unpack, SetOf, SetDifference
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



ARCTANH_REAL_DOMAIN = REALS[
    math.nextafter(-1, math.inf):math.nextafter(1, -math.inf)
]


defaults = {
    'sqrt': lambda x, /: math.sqrt(NONNEGATIVE_REALS(x)),
    'csqrt': lambda x, /: cmath.sqrt(COMPLEX(x)),
    'cbrt': lambda x, /: math.cbrt(REALS(x)),
    'ccbrt': lambda x, /: COMPLEX(x) ** 1/3,
    'root_': (
        lambda n, arg, /: REALS(arg) ** 1/SetDifference(REALS, {0})(n)
    ),
    'croot_': (
        lambda n, arg, /: COMPLEX(arg) ** 1/SetDifference(COMPLEX, {0})(n)
    ),
    'argument': lambda x, /: cmath.phase(COMPLEX(x)),
    'rect': lambda r, phi, /: cmath.rect(REALS(r), REALS(phi)),
    'sin': lambda x, /: math.sin(REALS(x)),
    'cos': lambda x, /: math.cos(REALS(x)),
    'tan': lambda x, /: math.tan(REALS(x)),
    'csin': lambda x, /: cmath.sin(COMPLEX(x)),
    'ccos': lambda x, /: cmath.cos(COMPLEX(x)),
    'ctan': lambda x, /: cmath.tan(COMPLEX(x)),
    'sinh': lambda x, /: math.sinh(REALS(x)),
    'cosh': lambda x, /: math.cosh(REALS(x)),
    'tanh': lambda x, /: math.tanh(REALS(x)),
    'csinh': lambda x, /: cmath.sinh(COMPLEX(x)),
    'ccosh': lambda x, /: cmath.cosh(COMPLEX(x)),
    'ctanh': lambda x, /: cmath.tanh(COMPLEX(x)),
    'arcsin': lambda x, /: math.asin(REALS[-1:1](x)),
    'arccos': lambda x, /: math.acos(REALS[-1:1](x)),
    'arctan': lambda x, /: math.atan(REALS(x)),
    'carcsin': lambda x, /: cmath.asin(COMPLEX(x)),
    'carccos': lambda x, /: cmath.acos(COMPLEX(x)),
    'carctan': lambda x, /: cmath.atan(COMPLEX(x)),
    'arcsinh': lambda x, /: math.asinh(REALS(x)),
    'arccosh': lambda x, /: math.acosh(REALS[1:](x)),
    'arctanh': lambda x, /: math.atanh(ARCTANH_REAL_DOMAIN(x)),
    'carcsinh': lambda x, /: cmath.asinh(COMPLEX(x)),
    'carccosh': lambda x, /: cmath.acosh(COMPLEX(x)),
    'carctanh': lambda x, /: cmath.atanh(SetDifference(COMPLEX, {1, -1})(x)),
    'atan2': lambda y, x, /: math.atan2(REALS(y), REALS(x)),
    'min': (lambda args, key=None, /: min(args, key=key)),
    'max': (lambda args, key=None, /: max(args, key=key)),
    'abs': abs,
    'gcd': lambda *args: math.gcd(*(INTEGERS(arg) for arg in args)),
    'lcm': lambda *args: math.lcm(*(INTEGERS(arg) for arg in args)),
    'slice': lambda *args: (
        BLANK_SLICE if not args else
        GeneralSlice(*args, None) if len(args) == 1 else
        GeneralSlice(*args)
    ),
    'floor': lambda x, /: math.floor(x),
    'ceil': lambda x, /: math.ceil(x),
    'trunc': lambda x, /: math.trunc(x),
    'round': lambda x, nd=None, /: round(x, nd),
    'frac': (lambda x: x - math.trunc(x)),
}

function_signatures = {
    'sqrt': "[0,∞⟩ -> ℝ",
    'csqrt': "ℂ -> ℂ",
    'cbrt': "ℝ -> ℝ",
    'ccbrt': "ℂ -> ℂ",
    'root_': "(ℝ\\{0})×ℝ -> ℝ",
    'croot_': "(ℂ\\{0})×ℂ -> ℂ",
    'argument': "ℂ -> [-π,π]",
    'rect': "ℝ×ℝ -> ℝ",
    'sin':  "ℝ -> ℝ",
    'cos':  "ℝ -> ℝ",
    'tan':  "ℝ -> ℝ",
    'csin': "ℂ -> ℂ",
    'ccos': "ℂ -> ℂ",
    'ctan': "ℂ -> ℂ",
    'sinh':  "ℝ -> ℝ",
    'cosh':  "ℝ -> ℝ",
    'tanh':  "ℝ -> ℝ",
    'csinh': "ℂ -> ℂ",
    'ccosh': "ℂ -> ℂ",
    'ctanh': "ℂ -> ℂ",
    'arcsin':  "[-1,1] -> ℝ",
    'arccos':  "[-1,1] -> ℝ",
    'arctan':  "ℝ -> ℝ",
    'carcsin': "ℂ -> ℂ",
    'carccos': "ℂ -> ℂ",
    'carctan': "ℂ -> ℂ",
    'arcsinh':  "ℝ -> ℝ",
    'arccosh':  "[1,∞⟩ -> ℝ",
    'arctanh':  "⟨-1,1⟩ -> ℝ",
    'carcsinh': "ℂ -> ℂ",
    'carccosh': "ℂ -> ℂ",
    'carctanh': "ℂ\\{1, -1} -> ℂ",
    'atan2': "ℝ×ℝ -> ℝ",
    'min': (
        "(Ordered a => (a+)×{null}? -> a) or (Ordered b => (a+)×Φ[a->b] -> a)",
        "(ℝ+)×{null}? -> ℝ or (a+)×Φ[a->ℝ] -> a"
    ),
    'max': (
        "(Ordered a => (a+)×{null}? -> a) or (Ordered b => (a+)×Φ[a->b] -> a)",
        "(ℝ+)×{null}? -> ℝ or (a+)×Φ[a->ℝ] -> a"
    ),
    'abs': ("HasAbs a r => a -> r", "ℝ -> ℝ", "ℂ -> ℝ"),
    'lcm': "ℤ* -> ℤ",
    'gcd': "ℤ* -> ℤ",
    'slice': "...ts -> Slice[...ts]"
    'floor': ("HasFloor a r => a -> r", "ℝ -> ℤ", "ℂ -> ℤ[i]"),
    'ceil': ("HasCeil a r => a -> r", "ℝ -> ℤ", "ℂ -> ℤ[i]"),
    'trunc': ("HasTrunc a r => a -> r", "ℝ -> ℤ", "ℂ -> ℤ[i]"),
    'round': (
        "HasRound a r1 r2 => (a×{null}? -> r1 or a×ℤ -> r2)",
        "ℝ×{null}? -> ℤ or ℝ×ℤ -> ℝ", "ℂ×{null}? -> ℤ[i] or ℂ×ℤ -> ℂ"
    ),
    'frac': (
        # where (HasTrunc a b) has functional dependency a -> b
        # where (HasSub a b c) has functional dependency a b -> c
        # note: figuring out how to get this to work in Haskell took
        # me longer than i would like to admit
        "(HasTrunc a r1, HasSub a r1 r2) => a -> r2",
        "ℝ -> ℝ", "ℂ -> ℂ"
    ),

}


defaults = {
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
