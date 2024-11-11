from dataclasses import dataclass, field
from constants import (
    Operator, PREFIX_UNARY_OPERATORS, POSTFIX_UNARY_OPERATORS,
    BINARY_OPERATORS, TERNARY_OPERATORS
)
from typing import Optional, Self
from enum import Enum
import abc
from typing import Any

@dataclass(frozen=True, eq=True)
class Token:
    value: str

@dataclass
class TokenGroup:
    bracket: Optional[str] = None
    values: list[Token | Self] = field(default_factory=list)

class NumberToken(Token): pass
class StringToken(Token): pass
class IdentifierToken(Token): pass
class UnknownOperatorToken(Token): pass
class OpenBracketToken(Token): pass
class ClosedBracketToken(Token): pass
class SeparatorToken(Token): pass
class ArrowToken(Token): pass
class UnpackToken(Token): pass
@dataclass(frozen=True, eq=True)
class OperatorToken(Token):
    operator: Operator

class MatchFailed(Enum):
    FAIL = 0

class PatternMeta(abc.ABCMeta):

    @staticmethod
    def match(inst: Any) -> Any | MatchFailed:
        raise NotImplementedError

    def __instancecheck__(cls, inst: object) -> bool:
        return cls.match(inst) is not MatchFailed.FAIL

class Pattern(metaclass=PatternMeta):
    pass

# operator function token patterns
type OperatorPrefillType = (
    TokenGroup | IdentifierToken | NumberToken | StringToken
)
type OperatorPlaceholderType = TokenGroup

class OperatorFunctionPrefill(Pattern):
    @staticmethod
    def match(obj: Any) -> OperatorPrefillType | MatchFailed:
        match obj:
            case (
                TokenGroup('[') | TokenGroup('('|'{', [_, *_]) |
                IdentifierToken() | NumberToken() | StringToken()
            ):
                return obj
        return MatchFailed.FAIL

class OperatorFunctionPlaceholder(Pattern):
    @staticmethod
    def match(obj: Any) -> OperatorPlaceholderType | MatchFailed:
        match obj:
            case TokenGroup('{', []):
                return obj
        return MatchFailed.FAIL

class PrefixUnaryOperatorFunctionPattern(Pattern):
    @staticmethod
    def match(obj: Any) -> tuple[OperatorToken, TokenGroup] | MatchFailed:
        match obj:
            case [
                UnknownOperatorToken(op),
                OperatorFunctionPlaceholder()
            ] | [
                UnknownOperatorToken(op)
            ] if op in PREFIX_UNARY_OPERATORS:
                opt = OperatorToken(op, PREFIX_UNARY_OPERATORS[op])
                return (opt, TokenGroup('{', []))
        return MatchFailed.FAIL

class PostfixUnaryOperatorFunctionPattern(Pattern):
    @staticmethod
    def match(obj: Any) -> tuple[TokenGroup, OperatorToken] | MatchFailed:
        match obj:
            case [
                OperatorFunctionPlaceholder(),
                UnknownOperatorToken(op)
            ] | [
                UnknownOperatorToken(op)
            ] if op in POSTFIX_UNARY_OPERATORS:
                opt = OperatorToken(op, POSTFIX_UNARY_OPERATORS[op])
                return (TokenGroup('{', []), opt)
        return MatchFailed.FAIL

class BinaryOperatorFunctionPattern(Pattern):
    @staticmethod
    def match(obj: Any) -> tuple[
        TokenGroup, OperatorToken, TokenGroup
    ] | MatchFailed:
        match obj:
            case [
                OperatorFunctionPlaceholder(),
                UnknownOperatorToken(op),
                OperatorFunctionPlaceholder(),
            ] | [
                UnknownOperatorToken(op)
            ] if op in BINARY_OPERATORS:
                opt = OperatorToken(op, BINARY_OPERATORS[op])
                return (TokenGroup('{', []), opt, TokenGroup('{', []))
            case [
                OperatorFunctionPrefill() as lhs,
                UnknownOperatorToken(op),
                OperatorFunctionPlaceholder()
            ] | [
                OperatorFunctionPrefill() as lhs,
                UnknownOperatorToken(op)
            ] if op in BINARY_OPERATORS:
                assert isinstance(lhs, Token | TokenGroup)
                opt = OperatorToken(op, BINARY_OPERATORS[op])
                if not isinstance(lhs, TokenGroup):
                    lhs = TokenGroup(None, [lhs])
                return (lhs, opt, TokenGroup('{', []))
            case [
                OperatorFunctionPlaceholder(),
                UnknownOperatorToken(op),
                OperatorFunctionPrefill() as rhs
            ] | [
                UnknownOperatorToken(op),
                OperatorFunctionPrefill() as rhs
            ] if op in BINARY_OPERATORS:
                assert isinstance(rhs, Token | TokenGroup)
                opt = OperatorToken(op, BINARY_OPERATORS[op])
                if not isinstance(rhs, TokenGroup):
                    rhs = TokenGroup(None, [rhs])
                return (TokenGroup('{', []), opt, rhs)
        return MatchFailed.FAIL

class TernaryOperatorFunctionPattern(Pattern):
    @staticmethod
    def match(
        obj: Any
    ) -> tuple[
        TokenGroup, OperatorToken, TokenGroup, OperatorToken, TokenGroup
    ] | MatchFailed:
        match obj:
            case [
                UnknownOperatorToken(op1),
                UnknownOperatorToken(op2),
            ] | [
                OperatorFunctionPlaceholder(),
                UnknownOperatorToken(op1),
                OperatorFunctionPlaceholder(),
                UnknownOperatorToken(op2),
                OperatorFunctionPlaceholder()
            ] if (
                op1 in TERNARY_OPERATORS and
                (op1, op2) == TERNARY_OPERATORS[op1].parts
            ):
                opt1 = OperatorToken(op1, TERNARY_OPERATORS[op1])
                opt2 = OperatorToken(op2, TERNARY_OPERATORS[op2])
                return (
                    TokenGroup('{', []), opt1,
                    TokenGroup('{', []), opt2,
                    TokenGroup('{', [])
                )
            case [
                OperatorFunctionPrefill() as lhs,
                UnknownOperatorToken(op1),
                UnknownOperatorToken(op2),
            ] | [
                OperatorFunctionPrefill() as lhs,
                UnknownOperatorToken(op1),
                OperatorFunctionPlaceholder(),
                UnknownOperatorToken(op2),
                OperatorFunctionPlaceholder()
            ] if (
                op1 in TERNARY_OPERATORS and
                (op1, op2) == TERNARY_OPERATORS[op1].parts
            ):
                assert isinstance(lhs, Token | TokenGroup)
                opt1 = OperatorToken(op1, TERNARY_OPERATORS[op1])
                opt2 = OperatorToken(op2, TERNARY_OPERATORS[op2])
                if not isinstance(lhs, TokenGroup):
                    lhs = TokenGroup(None, [lhs])
                return (
                    lhs, opt1,
                    TokenGroup('{', []), opt2,
                    TokenGroup('{', [])
                )
            case [
                UnknownOperatorToken(op1),
                OperatorFunctionPrefill() as mhs,
                UnknownOperatorToken(op2)
            ] | [
                OperatorFunctionPlaceholder(),
                UnknownOperatorToken(op1),
                OperatorFunctionPrefill() as mhs,
                UnknownOperatorToken(op2),
                OperatorFunctionPlaceholder()
            ] if (
                op1 in TERNARY_OPERATORS and
                (op1, op2) == TERNARY_OPERATORS[op1].parts
            ):
                assert isinstance(mhs, Token | TokenGroup)
                opt1 = OperatorToken(op1, TERNARY_OPERATORS[op1])
                opt2 = OperatorToken(op2, TERNARY_OPERATORS[op2])
                if not isinstance(mhs, TokenGroup):
                    mhs = TokenGroup(None, [mhs])
                return (
                    TokenGroup('{', []), opt1,
                    mhs, opt2,
                    TokenGroup('{', [])
                )
            case [
                UnknownOperatorToken(op1),
                UnknownOperatorToken(op2),
                OperatorFunctionPrefill() as rhs
            ] | [
                OperatorFunctionPlaceholder(),
                UnknownOperatorToken(op1),
                OperatorFunctionPlaceholder(),
                UnknownOperatorToken(op2),
                OperatorFunctionPrefill() as rhs
            ] if (
                op1 in TERNARY_OPERATORS and
                (op1, op2) == TERNARY_OPERATORS[op1].parts
            ):
                assert isinstance(rhs, Token | TokenGroup)
                opt1 = OperatorToken(op1, TERNARY_OPERATORS[op1])
                opt2 = OperatorToken(op2, TERNARY_OPERATORS[op2])
                if not isinstance(rhs, TokenGroup):
                    rhs = TokenGroup(None, [rhs])
                return (
                    TokenGroup('{', []), opt1,
                    TokenGroup('{', []), opt2,
                    rhs
                )
            case [
                OperatorFunctionPrefill() as lhs,
                UnknownOperatorToken(op1),
                OperatorFunctionPrefill() as mhs,
                UnknownOperatorToken(op2)
            ] | [
                OperatorFunctionPrefill() as lhs,
                UnknownOperatorToken(op1),
                OperatorFunctionPrefill() as mhs,
                UnknownOperatorToken(op2),
                OperatorFunctionPlaceholder()
            ] if (
                op1 in TERNARY_OPERATORS and
                (op1, op2) == TERNARY_OPERATORS[op1].parts
            ):
                assert isinstance(lhs, Token | TokenGroup)
                assert isinstance(mhs, Token | TokenGroup)
                opt1 = OperatorToken(op1, TERNARY_OPERATORS[op1])
                opt2 = OperatorToken(op2, TERNARY_OPERATORS[op2])
                if not isinstance(lhs, TokenGroup):
                    lhs = TokenGroup(None, [lhs])
                if not isinstance(mhs, TokenGroup):
                    mhs = TokenGroup(None, [mhs])
                return (
                    lhs, opt1,
                    mhs, opt2,
                    TokenGroup('{', [])
                )
            case [
                OperatorFunctionPrefill() as lhs,
                UnknownOperatorToken(op1),
                UnknownOperatorToken(op2),
                OperatorFunctionPrefill() as rhs
            ] | [
                OperatorFunctionPrefill() as lhs,
                UnknownOperatorToken(op1),
                OperatorFunctionPlaceholder(),
                UnknownOperatorToken(op2),
                OperatorFunctionPrefill() as rhs
            ] if (
                op1 in TERNARY_OPERATORS and
                (op1, op2) == TERNARY_OPERATORS[op1].parts
            ):
                assert isinstance(lhs, Token | TokenGroup)
                assert isinstance(rhs, Token | TokenGroup)
                opt1 = OperatorToken(op1, TERNARY_OPERATORS[op1])
                opt2 = OperatorToken(op2, TERNARY_OPERATORS[op2])
                if not isinstance(lhs, TokenGroup):
                    lhs = TokenGroup(None, [lhs])
                if not isinstance(rhs, TokenGroup):
                    rhs = TokenGroup(None, [rhs])
                return (
                    lhs, opt1,
                    TokenGroup('{', []), opt2,
                    rhs
                )
            case [
                UnknownOperatorToken(op1),
                OperatorFunctionPrefill() as mhs,
                UnknownOperatorToken(op2),
                OperatorFunctionPrefill() as rhs
            ] | [
                OperatorFunctionPlaceholder(),
                UnknownOperatorToken(op1),
                OperatorFunctionPrefill() as mhs,
                UnknownOperatorToken(op2),
                OperatorFunctionPrefill() as rhs
            ] if (
                op1 in TERNARY_OPERATORS and
                (op1, op2) == TERNARY_OPERATORS[op2].parts
            ):
                assert isinstance(mhs, Token | TokenGroup)
                assert isinstance(rhs, Token | TokenGroup)
                opt1 = OperatorToken(op1, TERNARY_OPERATORS[op1])
                opt2 = OperatorToken(op2, TERNARY_OPERATORS[op2])
                if not isinstance(mhs, TokenGroup):
                    mhs = TokenGroup(None, [mhs])
                if not isinstance(rhs, TokenGroup):
                    rhs = TokenGroup(None, [rhs])
                return (
                    TokenGroup('{', []), opt1,
                    mhs, opt2,
                    rhs
                )
        return MatchFailed.FAIL
