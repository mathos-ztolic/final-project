from __future__ import annotations
from dataclasses import dataclass
from collections import ChainMap
from collections.abc import Mapping
from typing import cast

BRACKETS = {
    '(': ')',
    '[': ']',
    '[<': '>]',
    '{': '}', 
    '?[': ']',
    '??[': ']'
}

BRACKET_START_CHARS = set(b[0] for b in BRACKETS)
BRACKET_END_CHARS = set(b[0] for b in BRACKETS.values())
NUMBER_LITERAL_START_CHARS = '0123456789.'
NUMBER_LITERAL_TYPES = {
    '0b': (2, 'binary', '01'),
    '0t': (3, 'ternary', '012'),
    '0q': (4, 'quaternary', '0123'),
    '0s': (6, 'seximal', '012345'),
    '0o': (8, 'octal', '01234567'),
    '0d': (12, 'dozenal', '0123456789abAB'),
    '0x': (16, 'hexadecimal', '0123456789abcdefABCDEF'),
}

@dataclass(frozen=True)
class Operator:
    symbol: str
    precedence: int
class UnaryOperator(Operator): pass
class PrefixUnaryOperator(UnaryOperator): pass
class PostfixUnaryOperator(UnaryOperator): pass
class BinaryOperator(Operator): pass
class LeftAssociativeBinaryOperator(BinaryOperator): pass
class RightAssociativeBinaryOperator(BinaryOperator): pass
@dataclass(frozen=True)
class TernaryOperator(Operator):
    parts: tuple[str, str]
class LeftAssociativeTernaryOperator(TernaryOperator): pass
class RightAssociativeTernaryOperator(TernaryOperator): pass

LeftAssociative = (
    PostfixUnaryOperator | LeftAssociativeBinaryOperator |
    LeftAssociativeTernaryOperator
)
RightAssociative = (
    PrefixUnaryOperator | RightAssociativeBinaryOperator |
    RightAssociativeTernaryOperator
)

OPERATORS: list[Operator] = [
    ## prefix unary ##
    # arithmetic
    PrefixUnaryOperator('+', 110),  # positive
    PrefixUnaryOperator('-', 110),  # negative
    PrefixUnaryOperator('++', 110),  # increment then return
    PrefixUnaryOperator('--', 110),  # decrement then return
    # logical
    PrefixUnaryOperator('!', 110),  # not
    # bitwise
    PrefixUnaryOperator('~', 110),  # not
    # functions
    PrefixUnaryOperator('`', 140),   # prepend placeholder argument
    PrefixUnaryOperator('^`', 140),  # prepend placeholder argument right
    
    ## postfix unary ##
    #arithmetic
    PostfixUnaryOperator('!', 135),  # factorial
    PostfixUnaryOperator('++', 136),  # return then increment
    PostfixUnaryOperator('--', 136),  # return then decrement
    # functions
    PostfixUnaryOperator("'", 140),   # append placeholder argument
    PostfixUnaryOperator("^'", 140),  # append placeholder argument left
    
    ## binary ##
    # conditional
    RightAssociativeBinaryOperator('||', 20),  # left true:left, else:right
    RightAssociativeBinaryOperator('&&', 30),  # left true:right, else:left
    RightAssociativeBinaryOperator('??', 15),  # left null:right, else:left
    RightAssociativeBinaryOperator('!!', 16),  # left null:left, else:right
    # relational
    LeftAssociativeBinaryOperator('>', 40),   # greater than
    LeftAssociativeBinaryOperator('>=', 40),  # greater than or equal to
    LeftAssociativeBinaryOperator('<', 40),   # less than
    LeftAssociativeBinaryOperator('<=', 40),  # less than or equal to
    LeftAssociativeBinaryOperator('==', 40),  # equal to
    LeftAssociativeBinaryOperator('!=', 40),  # not equal to
    # bitwise
    LeftAssociativeBinaryOperator('|', 50),   # or
    LeftAssociativeBinaryOperator('#', 60),   # xor
    LeftAssociativeBinaryOperator('&', 70),   # and
    LeftAssociativeBinaryOperator('<<', 80),  # shift left
    LeftAssociativeBinaryOperator('>>', 80),  # shift right
    # arithmetic
    LeftAssociativeBinaryOperator('+', 90),    # addition
    LeftAssociativeBinaryOperator('-', 90),    # subtraction
    LeftAssociativeBinaryOperator('*', 100),   # multiplication
    LeftAssociativeBinaryOperator('**', 100),  # matrix multiplication
    LeftAssociativeBinaryOperator('/', 100),   # division
    LeftAssociativeBinaryOperator('//', 100),  # floor division
    LeftAssociativeBinaryOperator('%', 100),   # modulo
    RightAssociativeBinaryOperator('^', 120),  # exponentiation
    # functions
    RightAssociativeBinaryOperator('@', 130),   # compose functions
    RightAssociativeBinaryOperator('`', 140),   # prepend argument
    RightAssociativeBinaryOperator('^`', 140),  # prepend argument right
    LeftAssociativeBinaryOperator("'", 140),    # append argument
    LeftAssociativeBinaryOperator("^'", 140),   # append argument left
    # membership
    LeftAssociativeBinaryOperator('in', 33),   # in
    LeftAssociativeBinaryOperator('!in', 33),  # not in
    # set operators
    LeftAssociativeBinaryOperator('$-', 34),  # difference
    LeftAssociativeBinaryOperator('$|', 35),  # union
    LeftAssociativeBinaryOperator('$#', 36),  # symmetric difference
    LeftAssociativeBinaryOperator('$&', 37),  # intersection
    LeftAssociativeBinaryOperator('$*', 38),  # cartesian product
    # ranges
    LeftAssociativeBinaryOperator('..', 150),    # [a, b]
    LeftAssociativeBinaryOperator('..!', 150),   # [a, b)
    LeftAssociativeBinaryOperator('!..', 150),   # (a, b]
    LeftAssociativeBinaryOperator('!..!', 150),  # (a, b)
    # assignment
    RightAssociativeBinaryOperator(':=', 1),  # assign and return
    # chaining
    LeftAssociativeBinaryOperator(';', 0),
    # classes
    LeftAssociativeBinaryOperator('==>', 200),  # create class
    LeftAssociativeBinaryOperator('<==', 201),  # inherit from

    ## ternary ##
    # conditional
    # if left true, mid, else right
    RightAssociativeTernaryOperator('?:', 10, ('?', ':')),
    # if left not null, mid, else right
    RightAssociativeTernaryOperator(':!:?', 9, (':!', ':?'))
    
]

VALID_OPERATOR_SYMBOLS = [
    op.symbol
    for op in OPERATORS
    if not isinstance(op, TernaryOperator)
] + [
    part
    for op in OPERATORS if isinstance(op, TernaryOperator)
    for part in op.parts
]

OPERATOR_START_CHARS = set(op[0] for op in VALID_OPERATOR_SYMBOLS)

PREFIX_UNARY_OPERATORS = {
    op.symbol: op for op in OPERATORS
    if isinstance(op, PrefixUnaryOperator)
}
POSTFIX_UNARY_OPERATORS = {
    op.symbol: op for op in OPERATORS
    if isinstance(op, PostfixUnaryOperator)
}
LEFT_ASSOCIATIVE_BINARY_OPERATORS = {
    op.symbol: op for op in OPERATORS
    if isinstance(op, LeftAssociativeBinaryOperator)
}
RIGHT_ASSOCIATIVE_BINARY_OPERATORS = {
    op.symbol: op for op in OPERATORS
    if isinstance(op, RightAssociativeBinaryOperator)
}
LEFT_ASSOCIATIVE_TERNARY_OPERATORS = {
    part: op
    for op in OPERATORS if isinstance(op, LeftAssociativeTernaryOperator)
    for part in op.parts
}
RIGHT_ASSOCIATIVE_TERNARY_OPERATORS = {
    part: op
    for op in OPERATORS if isinstance(op, RightAssociativeTernaryOperator)
    for part in op.parts
}

class ReadOnlyChainMap[K, V](Mapping[K, V]):
    _maps: tuple[Mapping[K, V], ...]
    def __init__(self, *maps: Mapping[K, V]):
        self._maps = maps

    def __getitem__(self, key: K) -> V:
        for mapping in self._maps:
            try:
                return mapping[key]
            except KeyError:
                pass
        raise KeyError(key)
    
    def __iter__(self):
        for mapping in self._maps:
            yield from mapping
    
    def __len__(self):
        return sum(len(mapping) for mapping in self._maps)

BINARY_OPERATORS = ReadOnlyChainMap(
    LEFT_ASSOCIATIVE_BINARY_OPERATORS,
    RIGHT_ASSOCIATIVE_BINARY_OPERATORS,
)
TERNARY_OPERATORS = ReadOnlyChainMap(
    LEFT_ASSOCIATIVE_TERNARY_OPERATORS,
    RIGHT_ASSOCIATIVE_TERNARY_OPERATORS,
)

LU = POSTFIX_UNARY_OPERATORS
RU = PREFIX_UNARY_OPERATORS
LB = LEFT_ASSOCIATIVE_BINARY_OPERATORS
RB = RIGHT_ASSOCIATIVE_BINARY_OPERATORS
LT = LEFT_ASSOCIATIVE_TERNARY_OPERATORS
RT = RIGHT_ASSOCIATIVE_TERNARY_OPERATORS
# example:
#     say & is left associative, and | is right associative
#     they have equal precedence
#     a & b & c | d | e = (a & b) & c | (d | e)
#     which operator gets to claim c first?
#     if the clash dictionary says &:  ((a & b) & c) | (d | e)
#     if the clash dictionary says |:  (a & b) & (c | (d | e))
CLASH_RESOLUTION: dict[
    tuple[LeftAssociative, RightAssociative],
    LeftAssociative | RightAssociative
] = {
    (LU["'"], RU['`']): RU['`'],
    (LU["^'"], RU['`']): RU['`'],
    (LU["'"], RU['^`']): RU['^`'],
    (LU["^'"], RU['^`']): RU['^`'],
    
    (LB["'"], RB['`']): RB['`'],
    (LB["^'"], RB['`']): RB['`'],
    (LB["'"], RB['^`']): RB['^`'],
    (LB["^'"], RB['^`']): RB['^`'],
    
    (LU["'"], RB['`']): RB['`'],
    (LU["^'"], RB['`']): RB['`'],
    (LU["'"], RB['^`']): RB['^`'],
    (LU["^'"], RB['^`']): RB['^`'],
    
    (LB["'"], RU['`']): RU['`'],
    (LB["^'"], RU['`']): RU['`'],
    (LB["'"], RU['^`']): RU['^`'],
    (LB["^'"], RU['^`']): RU['^`'],
}

del LU, RU, LB, RB, LT, RT
