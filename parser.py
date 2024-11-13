from __future__ import annotations
from typing import Self, Optional, cast
from dataclasses import dataclass, field
from enum import Enum
from collections import ChainMap, deque
import copy
import statistics
import random
import operator
import math
import cmath
import functools
import itertools
import types
import traceback
import enum
from types import EllipsisType
from tokens import (
    Token, TokenGroup, IdentifierToken, StringToken, NumberToken,
    OperatorToken, ArrowToken, UnpackToken, SeparatorToken
)
from errors import (
    InvalidExpressionList, InvalidLambdaExpression, ParserError
)
from constants import (
    NUMBER_LITERAL_TYPES, PrefixUnaryOperator, PostfixUnaryOperator,
    BinaryOperator, TernaryOperator, LeftAssociative, RightAssociative,
    CLASH_RESOLUTION
)

class _Unpacked(list): pass
class _KWUnpacked(list): pass

class Uncomputed(Enum):
    UNCOMPUTED = 0

@dataclass
class Node: pass
@dataclass
class ExpressionNode(Node):
    result: object = field(
        default=Uncomputed.UNCOMPUTED, compare=False, kw_only=True
    )
    null_conditional_failed: bool = field(init=False, default=False)
@dataclass
class IdentifierNode(ExpressionNode):
    value: str
@dataclass
class ConstantNode(ExpressionNode):
    value: bool | int | float | complex | str
@dataclass
class ListNode(ExpressionNode):
    exprs: list[ExpressionNode]
@dataclass
class TupleNode(ExpressionNode):
    exprs: list[ExpressionNode]
@dataclass
class DictionaryNode(ExpressionNode):
    keys: list[ExpressionNode]
    values: list[Optional[ExpressionNode]]
@dataclass
class GroupNode(ExpressionNode):
    expr: ExpressionNode
class OperationNode(ExpressionNode): pass
@dataclass
class UnaryOperationNode(OperationNode):
    operator: str
    expr: ExpressionNode
class PrefixUnaryOperationNode(UnaryOperationNode): pass
class PostfixUnaryOperationNode(UnaryOperationNode): pass
@dataclass
class BinaryOperationNode(OperationNode):
    operator: str
    left_expr: ExpressionNode
    right_expr: ExpressionNode
    
@dataclass
class TernaryOperationNode(OperationNode):
    operator: str
    left_expr: ExpressionNode
    mid_expr: ExpressionNode
    right_expr: ExpressionNode
@dataclass
class OperatorFunctionNode(ExpressionNode):
    operator: str
class UnaryOperatorFunctionNode(OperatorFunctionNode): pass
class PrefixUnaryOperatorFunctionNode(UnaryOperatorFunctionNode): pass
class PostfixUnaryOperatorFunctionNode(UnaryOperatorFunctionNode): pass
type BinaryArgsType = (
    tuple[EllipsisType, EllipsisType] | 
    tuple[EllipsisType, ExpressionNode] |
    tuple[ExpressionNode, EllipsisType]
)
type TernaryArgsType = (
    tuple[EllipsisType, EllipsisType, EllipsisType] |
    tuple[EllipsisType, EllipsisType, ExpressionNode] |
    tuple[EllipsisType, ExpressionNode, EllipsisType] |
    tuple[ExpressionNode, EllipsisType, EllipsisType] |
    tuple[ExpressionNode, ExpressionNode, EllipsisType] |
    tuple[ExpressionNode, EllipsisType, ExpressionNode] |
    tuple[EllipsisType, ExpressionNode, ExpressionNode]
)
@dataclass
class BinaryOperatorFunctionNode(OperatorFunctionNode):
    args: BinaryArgsType
@dataclass
class TernaryOperatorFunctionNode(OperatorFunctionNode):
    args: TernaryArgsType
@dataclass
class LambdaFunctionNode(ExpressionNode):
    positional_parameters: list[str]
    keyword_parameters: list[str]
    expr: ExpressionNode
@dataclass
class FunctionApplicationNode(ExpressionNode):
    function: (
        GroupNode | OperatorFunctionNode |
        IdentifierNode | LambdaFunctionNode |
        IndexingNode | AttributeAccessNode |
        ConstantNode | ListNode | Self
    )
    args: list[ExpressionNode]
    kwargs: list[tuple[Optional[str], ExpressionNode]]
@dataclass
class IndexingNode(ExpressionNode):
    indexed: (
        GroupNode | OperatorFunctionNode |
        IdentifierNode | LambdaFunctionNode |
        FunctionApplicationNode | AttributeAccessNode | ConstantNode |
        ListNode | Self
    )
    index: list[Optional[ExpressionNode]]
    null_conditional_level: int
@dataclass
class UnpackNode(ExpressionNode):
    expr: ExpressionNode
@dataclass
class AttributeAccessNode(ExpressionNode):
    accessed: (
        GroupNode | OperatorFunctionNode |
        IdentifierNode | LambdaFunctionNode |
        FunctionApplicationNode | IndexingNode | ConstantNode |
        ListNode | Self
    )
    attribute: str
    null_conditional_level: int
    

def split_expression_list(
    tokenized: list[Token | TokenGroup],
    allow_blank_parts=False
) -> list[list[Token | TokenGroup]]:
    result: list[list[Token | TokenGroup]] = []
    if not tokenized:
        return result
    if isinstance(tokenized[0], SeparatorToken) and not allow_blank_parts:
        raise InvalidExpressionList(
            "First element of expression list is a separator."
        )
    if isinstance(tokenized[0], UnpackToken) and tokenized[0].value == '...':
        result.append(_Unpacked())
    elif isinstance(tokenized[0], UnpackToken):
        result.append(_KWUnpacked())
    elif isinstance(tokenized[0], SeparatorToken):
        result.append([])
        result.append([])
    else:
        result.append([tokenized[0]])
    for token in tokenized[1:]:
        if isinstance(token, SeparatorToken):
            if not result[-1] and isinstance(result[-1], _Unpacked):
                raise InvalidExpressionList("... is not a valid argument.")
            if not result[-1] and isinstance(result[-1], _KWUnpacked):
                raise InvalidExpressionList(".... is not a valid argument.")
            elif not result[-1] and not allow_blank_parts:
                raise InvalidExpressionList("Multiple separators in a row.")
            result.append([])
            continue
        elif isinstance(token, UnpackToken) and not result[-1]:
            if token.value == '...':
                result[-1] = _Unpacked()
            else:
                result[-1] = _KWUnpacked()
            continue
        elif isinstance(token, UnpackToken) and result[-1]:
            raise InvalidExpressionList("Unpack in the middle of argument.")
        result[-1].append(token)
    
    if not result[-1] and isinstance(result[-1], _Unpacked):
        raise InvalidExpressionList("... is not a valid argument.")
    elif not result[-1] and isinstance(result[-1], _KWUnpacked):
        raise InvalidExpressionList(".... is not a valid argument.")
    elif not result[-1] and not allow_blank_parts:
        raise InvalidExpressionList("Trailing separator.")
    return result



def convert_number(number: str) ->  int | float | complex:
    is_imaginary = number.endswith('i')
    number = number.rstrip('i')
    base = NUMBER_LITERAL_TYPES.get(number[:2], (10,))[0]
    if ('.' in number or 'e' in number) and base == 10:
        return complex(0, float(number)) if is_imaginary else float(number)
    return (
        complex(0, int(number[2*(base!=10):], base=base))
        if is_imaginary else
        int(number[2*(base!=10):], base=base)
    )
    
def parse_expression(
    tokenized: list[Token | TokenGroup] | TokenGroup | Token
) -> ExpressionNode:
    if isinstance(tokenized, (TokenGroup, Token)):
        tokenized = [tokenized]
    match tokenized:
        case []:
            raise ParserError("Blank expression encountered.")
        case [TokenGroup(bracket='{', values=[])]:
            raise ParserError("Blank expression encountered.")
        ### number ###
        case [NumberToken() as tok]:
            return ConstantNode(convert_number(tok.value))
        ### string ###
        case [StringToken() as tok]:
            return ConstantNode(tok.value)
        ### identifier ###
        case [IdentifierToken() as tok]:
            return IdentifierNode(tok.value)
        ### group ###
        case [TokenGroup(bracket='(') as tokg]:
            return GroupNode(expr=parse_expression(tokg.values))
        ### fake group ###
        case [TokenGroup(bracket=None) as tokg]:
            return parse_expression(tokg.values)
        ### tuple ###
        case [TokenGroup(bracket='[<') as tokg]:
            exprs = []
            for expr in split_expression_list(tokg.values):
                if isinstance(expr, _KWUnpacked):
                    raise ParserError(
                        "Keyword unpacking not allowed in tuple."
                    )
                elif isinstance(expr, _Unpacked):
                    exprs.append(UnpackNode(parse_expression(expr)))
                else:
                    exprs.append(parse_expression(expr))
            return TupleNode(exprs)
        ### list ###
        case [TokenGroup(bracket='[') as tokg]:
            exprs = []
            for expr in split_expression_list(tokg.values):
                if isinstance(expr, _KWUnpacked):
                    raise ParserError(
                        "Keyword unpacking not allowed in list."
                    )
                elif isinstance(expr, _Unpacked):
                    exprs.append(UnpackNode(parse_expression(expr)))
                else:
                    exprs.append(parse_expression(expr))
            return ListNode(exprs)
        ### dictionary ###
        case [TokenGroup('{', [ArrowToken('='), *dict_expr])]:
            keys = []
            values = []
            current = keys
            for x in dict_expr:
                if isinstance(x, ArrowToken):
                    current = values
                elif isinstance(x, SeparatorToken):
                    current = keys
                elif isinstance(x, UnpackToken):
                    if x.value == '...':
                        raise ParserError(
                            "Iterable unpacking not allowed in dictionary."
                        )
                    values.append(None)
                else:
                    current.append(parse_expression(x))
            return DictionaryNode(keys, values)
        ### lambda expression ###
        case [
            TokenGroup('{', [
                TokenGroup(None, parameters),
                ArrowToken('=>'),
                TokenGroup(None, expression)
            ])
        ]:
            positional_parameters = []
            keyword_parameters = []
            current_parameters = positional_parameters
            next_variable = ''
            for parameter in parameters:
                if isinstance(parameter, UnpackToken):
                    next_variable = parameter.value
                    if next_variable == '....':
                        current_parameters = keyword_parameters
                    continue
                if isinstance(parameter, SeparatorToken):
                    current_parameters = keyword_parameters
                    continue
                assert isinstance(parameter, IdentifierToken)
                if next_variable:
                    current_parameters.append(next_variable+parameter.value)
                    next_variable = ''
                else:
                    current_parameters.append(parameter.value)
            return LambdaFunctionNode(
                positional_parameters,
                keyword_parameters,
                parse_expression(expression)
            )
        ### prefix unary operator function ###
        case [TokenGroup('{', [OperatorToken(op), TokenGroup('{', [])])]:
            return PrefixUnaryOperatorFunctionNode(op)
        ### postfix unary operator function ###
        case [TokenGroup('{', [TokenGroup('{', []), OperatorToken(op)])]:
            return PostfixUnaryOperatorFunctionNode(op)
        ### binary operator function ###
        case [
            TokenGroup(
                '{',
                [TokenGroup() as lhs, OperatorToken(op), TokenGroup() as rhs]
            )
        ]:
            lhsa = (
                ... if lhs.bracket == '{' and not lhs.values else
                parse_expression(lhs)
            )
            rhsa = (
                ... if rhs.bracket == '{' and not rhs.values else
                parse_expression(rhs)
            )
            return BinaryOperatorFunctionNode(
                op, cast(BinaryArgsType, (lhsa, rhsa))
            )
        ### ternary operator function ###
        case [
            TokenGroup(
                '{',
                [
                    TokenGroup() as lhs, OperatorToken(_, op),
                    TokenGroup() as mhs, OperatorToken(),
                    TokenGroup() as rhs
                ]
            )
        ]:
            lhsa = (
                ... if lhs.bracket == '{' and not lhs.values else
                parse_expression(lhs)
            )
            mhsa = (
                ... if mhs.bracket == '{' and not mhs.values else
                parse_expression(mhs)
            )
            rhsa = (
                ... if rhs.bracket == '{' and not rhs.values else
                parse_expression(rhs)
            )
            return TernaryOperatorFunctionNode(
                op.symbol, cast(TernaryArgsType, (lhsa, mhsa, rhsa))
            )
        ### function application, list indexing, attribute access ###
        case [
            (
                TokenGroup(bracket=('(' | '{' | '[' | '[<' | None)) |
                IdentifierToken() | StringToken() | NumberToken()
            )
            as first,
            *rest
        ] if (
            all(isinstance(x, (
                TokenGroup, ArrowToken, IdentifierToken
            )) for x in rest) and
            all(isinstance(x, (ArrowToken, IdentifierToken)) or (
                isinstance(x, TokenGroup) and
                x.bracket in ('(', '[', '?[', '??[')
            ) for x in rest) and rest
        ):
            node = cast(
                IdentifierNode | ConstantNode | GroupNode | ListNode |
                OperatorFunctionNode | LambdaFunctionNode |
                FunctionApplicationNode | AttributeAccessNode | IndexingNode,
                parse_expression(first)
            )
            previous_arrow = None
            for args in rest:
                if isinstance(args, ArrowToken) and previous_arrow:
                    raise ParserError("Multiple arrows in a row.")
                if isinstance(args, ArrowToken):
                    previous_arrow = args.value
                    continue
                if previous_arrow and not isinstance(args, IdentifierToken):
                    raise ParserError("Expected identifier after arrow.")
                if previous_arrow:
                    assert isinstance(args, IdentifierToken)
                    node = AttributeAccessNode(
                        node, args.value, (
                            2*(previous_arrow == '??.') +
                            (previous_arrow == '?.')
                        )
                    )
                    previous_arrow = None
                elif isinstance(args, TokenGroup) and args.bracket == '(':
                    fn_args = []
                    fn_kwargs = []
                    for expr in split_expression_list(args.values):
                        if isinstance(expr, _Unpacked):
                            fn_args.append(UnpackNode(parse_expression(expr)))
                        elif isinstance(expr, _KWUnpacked):
                            fn_kwargs.append(
                                (None, UnpackNode(parse_expression(expr)))
                            )
                        else:
                            match expr:
                                # pep 736 because why not
                                case [
                                    IdentifierToken(ident),
                                    ArrowToken('=')
                                ]:
                                    fn_kwargs.append(
                                        (ident, IdentifierNode(ident))
                                    )
                                case [
                                    IdentifierToken(ident),
                                    ArrowToken('='),
                                    *rest
                                ]:
                                    fn_kwargs.append(
                                        (ident, parse_expression(rest))
                                    )
                                case _:
                                    fn_args.append(parse_expression(expr))

                    node = FunctionApplicationNode(node, fn_args, fn_kwargs)
                else:
                    assert isinstance(args, TokenGroup)
                    null_conditional_level = (
                        1 if args.bracket == '?[' else
                        2 if args.bracket == '??[' else
                        0
                    )

                    exprs = []
                    for expr in split_expression_list(tokg.values, True):
                        if isinstance(expr, _KWUnpacked):
                            raise ParserError(
                                "Keyword unpacking not allowed in index."
                            )
                        elif isinstance(expr, _Unpacked):
                            exprs.append(UnpackNode(parse_expression(expr)))
                        elif expr:
                            exprs.append(parse_expression(expr))
                        else:
                            exprs.append(None)
                    node = IndexingNode(node, exprs, null_conditional_level)
            if previous_arrow:
                raise ParserError("Trailing arrow.")
            return node
        case (
            values
        ) if UnpackToken('...') in values or UnpackToken('....') in values:
            raise ParserError("Unpacking is not allowed here.")
        ### complex expression ###
        case _:
            return parse_complex_expression(tokenized)

def parse_complex_expression(
    tokenized: list[Token | TokenGroup]
) -> ExpressionNode:
    match tokenized:
        case [
            OperatorToken(uop, PrefixUnaryOperator()),
            (
                TokenGroup() | NumberToken() |
                IdentifierToken() | StringToken()
            ) as operand
        ]:
            return PrefixUnaryOperationNode(
                uop, parse_expression(operand)
            )
        case [
            (
                TokenGroup() | NumberToken() |
                IdentifierToken() | StringToken()
            ) as operand,
            OperatorToken(uop, PostfixUnaryOperator())
        ]:
            return PostfixUnaryOperationNode(
                uop, parse_expression(operand)
            )
        case [
            (
                TokenGroup() | NumberToken() |
                IdentifierToken() | StringToken()
            ) as operand1,
            OperatorToken(bop, BinaryOperator()),
            (
                TokenGroup() | NumberToken() |
                IdentifierToken() | StringToken()
            ) as operand2,
        ]:
            return BinaryOperationNode(
                bop,
                parse_expression(operand1),
                parse_expression(operand2)
            )
        case [
            (
                TokenGroup() | NumberToken() |
                IdentifierToken() | StringToken()
            ) as operand1,
            OperatorToken(_, TernaryOperator(top)),
            (
                TokenGroup() | NumberToken() |
                IdentifierToken() | StringToken()
            ) as operand2,
            OperatorToken(_, TernaryOperator()),
            (
                TokenGroup() | NumberToken() |
                IdentifierToken() | StringToken()
            ) as operand3,
        ]:
            return TernaryOperationNode(
                top,
                parse_expression(operand1),
                parse_expression(operand2),
                parse_expression(operand3)
            )
    
    current_operator = None
    current_operator_index = None
    for i, tok in enumerate(tokenized):
        if not isinstance(tok, OperatorToken):
            continue
        operator = tok.operator
        if current_operator is None:
            current_operator = operator
            current_operator_index = i
            continue
        # only take open ternaries, the closed one is i+2 anyway,
        # because they have been grouped
        if (
            isinstance(operator, TernaryOperator) and 
            tok.value == operator.parts[1]
        ):
            continue
        if operator.precedence < current_operator.precedence:
            current_operator = operator
            current_operator_index = i
        elif operator.precedence > current_operator.precedence:
            continue
        # if left associative, split on rightmost first
        elif (
            isinstance(operator, LeftAssociative) and
            isinstance(current_operator, LeftAssociative)
        ):
            current_operator = operator
            current_operator_index = i
        elif (
            isinstance(operator, RightAssociative) and
            isinstance(current_operator, RightAssociative)
        ):
            continue
        elif isinstance(operator, LeftAssociative):
            assert isinstance(current_operator, RightAssociative)
            if (operator, current_operator) not in CLASH_RESOLUTION:
                raise ParserError(
                    "Unaccounted for clash between operators."
                )
            # current has higher priority, split on other one
            if CLASH_RESOLUTION[(operator, current_operator)] != operator:
                current_operator = operator
                current_operator_index = i
        elif isinstance(operator, RightAssociative):
            assert isinstance(current_operator, LeftAssociative)
            if (current_operator, operator) not in CLASH_RESOLUTION:
                raise ParserError(
                    "Unaccounted for clash between operators."
                )
            # current has higher priority, split on other one
            if CLASH_RESOLUTION[(current_operator, operator)] != operator:
                current_operator = operator
                current_operator_index = i
    
    if isinstance(current_operator, BinaryOperator):
        assert isinstance(current_operator_index, int)
        return BinaryOperationNode(
            current_operator.symbol,
            parse_expression(tokenized[:current_operator_index]),
            parse_expression(tokenized[current_operator_index+1:])
        )
    elif isinstance(current_operator, TernaryOperator):
        assert isinstance(current_operator_index, int)
        return TernaryOperationNode(
            current_operator.symbol,
            parse_expression(tokenized[:current_operator_index]),
            parse_expression(tokenized[current_operator_index+1]),
            parse_expression(tokenized[current_operator_index+3:])
        )
    else:
        assert False, tokenized
