from __future__ import annotations
from dataclasses import dataclass, field
from collections.abc import Generator
from typing import Self, Optional, cast
import abc
from errors import (
    TokenizerError, UnbalancedBracketsError, UnclosedBracketsError,
    MismatchedBracketsError, ParserError, InvalidNumberLiteral,
    InvalidLambdaExpression, UnknownCharacterError
)
from tokens import (
    Token, TokenGroup, IdentifierToken, StringToken, NumberToken,
    OperatorToken, ArrowToken, UnpackToken, SeparatorToken,
    OpenBracketToken, ClosedBracketToken, UnknownOperatorToken,
    PrefixUnaryOperatorFunctionPattern, PostfixUnaryOperatorFunctionPattern,
    BinaryOperatorFunctionPattern, TernaryOperatorFunctionPattern, MatchFailed
)
from constants import (
    NUMBER_LITERAL_TYPES, PrefixUnaryOperator, PostfixUnaryOperator,
    BinaryOperator, TernaryOperator, LeftAssociative, RightAssociative,
    CLASH_RESOLUTION, OPERATORS, NUMBER_LITERAL_START_CHARS,
    BRACKET_START_CHARS, BRACKET_END_CHARS, OPERATOR_START_CHARS,
    PREFIX_UNARY_OPERATORS, POSTFIX_UNARY_OPERATORS, BINARY_OPERATORS,
    TERNARY_OPERATORS, BRACKETS, VALID_OPERATOR_SYMBOLS, UnaryOperator
)
from utils import prev_curr_next_iter

def _tokenize_string_literal(expression: str, cursor: int) -> tuple[int, str]:
    escape_sequences = {
        '"': '"',
        'n': '\n',
        't': '\t',
        '\\': '\\'
    }
    escaping = False
    cursor += 1
    final_string = ""
    while cursor < len(expression) and (expression[cursor] != '"' or escaping):
        if expression[cursor] == '\\' and escaping:
            final_string += '\\'
            escaping = False
            cursor += 1
        elif expression[cursor] == '\\' and not escaping:
            escaping = True
            cursor += 1
        elif escaping and expression[cursor] in escape_sequences:
            final_string += escape_sequences[expression[cursor]]
            escaping = False
            cursor += 1
        elif escaping and expression[cursor] not in escape_sequences:
            raise TokenizerError(
                f"Unknown escape sequence '\\{expression[cursor]}'"
            )
        else:
            final_string += expression[cursor]
            cursor += 1
    
    if cursor == len(expression):
        raise TokenizerError("Expected closing quotation.")
    
    cursor += 1
    return (cursor, final_string)
    

def _tokenize_decimal_literal(expression: str, cursor: int) -> str:
    dot_encountered = expression[cursor] == '.'
    e_encountered = False
    new_cursor = cursor+1
    while (
        new_cursor < len(expression) and
        expression[new_cursor] in '0123456789e.' and
        not (dot_encountered and expression[new_cursor] == '.') and
        not (e_encountered and expression[new_cursor] == 'e')
    ):
        if expression[new_cursor] == 'e':
            if new_cursor + 1 == len(expression):
                raise InvalidNumberLiteral(
                     "Invalid decimal literal "
                    f"'{expression[cursor:new_cursor+1]}'."
                )
            # negative exponents allowed, skip the first minus
            if expression[new_cursor+1] == '-':
                new_cursor += 1
            e_encountered = True
            dot_encountered = True
        if expression[new_cursor] == '.':
            dot_encountered = True
        if expression[new_cursor:new_cursor+2] == '..':
            break
        new_cursor += 1
    if new_cursor == len(expression):
        return expression[cursor:new_cursor]
    # imaginary literal
    if expression[new_cursor] == 'i':
        new_cursor += 1
    if new_cursor == len(expression):
        return expression[cursor:new_cursor]
    if expression[new_cursor].isalpha():
        raise InvalidNumberLiteral(
            f"Invalid decimal literal '{expression[cursor:new_cursor+1]}'."
        )
    return expression[cursor:new_cursor]

def _tokenize_nondecimal_literal(
    expression: str, cursor: int, name: str, chars: str
) -> str:
    new_cursor = cursor+2
    while (
        new_cursor < len(expression) and 
        expression[new_cursor] in chars
    ):
        new_cursor += 1
    if new_cursor == len(expression):
        return expression[cursor:new_cursor]
    # imaginary literal
    if expression[new_cursor] == 'i':
        new_cursor += 1
    if new_cursor == len(expression):
        return expression[cursor:new_cursor]
    if expression[new_cursor].isalnum():
        raise InvalidNumberLiteral(
            f"Invalid {name} literal '{expression[cursor:new_cursor+1]}'."
        )
    return expression[cursor:new_cursor]

def _tokenize_number(expression: str, cursor: int) -> str:
    literal_type = expression[cursor:cursor+2]
    if literal_type in NUMBER_LITERAL_TYPES:
        return _tokenize_nondecimal_literal(
            expression, cursor, *NUMBER_LITERAL_TYPES[literal_type][1:]
        )
    return _tokenize_decimal_literal(expression, cursor)

def _tokenize_identifier(expression: str, cursor: int) -> str:
    new_cursor = cursor+1
    while (
        new_cursor < len(expression) and 
        (expression[new_cursor].isalnum() or expression[new_cursor] == '_')
    ):
        new_cursor += 1
    if new_cursor == len(expression):
        return expression[cursor:new_cursor]
    return expression[cursor:new_cursor]

def tokenize_statement(statement: str):
    pass

def tokenize_expression(expression: str) -> Generator[Token, int]:
    cursor = 0
    max_operator_length = len(max(VALID_OPERATOR_SYMBOLS, key=len))
    min_operator_length = len(min(VALID_OPERATOR_SYMBOLS, key=len))
    max_bracket_start_length = len(max(BRACKETS, key=len))
    min_bracket_start_length = len(min(BRACKETS, key=len))
    max_bracket_end_length = len(max(BRACKETS.values(), key=len))
    min_bracket_end_length = len(min(BRACKETS.values(), key=len))
    
    bracket_stack: list[str] = []
    string_stack: list[str] = []
    num_failed = False
    op_failed = False
    bracket_start_failed = False
    bracket_end_failed = False
    while cursor < len(expression):
        if expression[cursor] == ';' and not bracket_stack:
            cursor += 1
            break
        if expression[cursor] == '"':
            cursor, string = _tokenize_string_literal(expression, cursor)
            string_stack.append(string)
            continue
        elif expression[cursor].strip() and string_stack:
            yield StringToken(''.join(string_stack))
            string_stack.clear()
        if expression[cursor] in NUMBER_LITERAL_START_CHARS and not num_failed:
            if expression[cursor] == '.' and (
                cursor + 1 == len(expression) or
                expression[cursor+1] not in '0123456789'
            ):
                num_failed = True
                continue
            number = _tokenize_number(expression, cursor)
            yield NumberToken(number)
            cursor += len(number)
        elif expression[cursor:cursor+3] == '...':
            yield UnpackToken('...')
            cursor += 3
        elif expression[cursor:cursor+4] == '??.':
            yield ArrowToken(expression[cursor:cursor+3])
            cursor += 4
        elif expression[cursor:cursor+3] == '?.':
            yield ArrowToken(expression[cursor:cursor+2])
            cursor += 3
        elif expression[cursor:cursor+2] == '=>':
            yield ArrowToken(expression[cursor:cursor+2])
            cursor += 2
        elif (
            not bracket_start_failed and
            expression[cursor] in BRACKET_START_CHARS
        ):
            for i in range(
                max_bracket_start_length, min_bracket_start_length-1, -1
            ):
                if cursor + i > len(expression):
                    continue
                if expression[cursor:cursor+i] in BRACKETS:
                    bracket = expression[cursor:cursor+i]
                    bracket_stack.append(bracket)
                    yield OpenBracketToken(bracket)
                    cursor += i
                    break
            else:
                bracket_start_failed = True
                continue
        elif (
            not bracket_end_failed and
            expression[cursor] in BRACKET_END_CHARS
        ):
            
            for i in range(
                max_bracket_end_length, min_bracket_end_length-1, -1
            ):
                if cursor + i > len(expression):
                    continue
                if expression[cursor:cursor+i] in BRACKETS.values():
                    bracket = expression[cursor:cursor+i]
                    if not bracket_stack:
                        raise UnbalancedBracketsError(
                            f"Closing {bracket} without an opening match."
                        )
                    if BRACKETS[bracket_stack[-1]] != bracket:
                        raise MismatchedBracketsError(
                            f"Mismatched opening {bracket_stack[-1]} "
                            f"with closing {bracket}."
                        )
                    bracket_stack.pop()
                    yield ClosedBracketToken(bracket)
                    cursor += i
                    break
            else:
                bracket_end_failed = True
                continue
        elif not op_failed and expression[cursor] in OPERATOR_START_CHARS:
            for i in range(max_operator_length, min_operator_length-1, -1):
                if cursor + i > len(expression):
                    continue
                if expression[cursor:cursor+i] in VALID_OPERATOR_SYMBOLS:
                    # if the operator ends with an identifier character
                    # but there is no whitespace after, then it's not
                    # valid
                    lastchar = expression[cursor+i-1]
                    nextchar = expression[cursor+i:cursor+i+1]
                    if (
                        (lastchar.isalnum() or lastchar == '_') and
                        (nextchar.isalnum() or nextchar == '_')
                    ):
                        continue
                    yield UnknownOperatorToken(expression[cursor:cursor+i])
                    cursor += i
                    break
            else:
                op_failed = True
                continue
        elif expression[cursor] == '.':
            yield ArrowToken(expression[cursor])
            cursor += 1
        elif expression[cursor] == ',':
            yield SeparatorToken(expression[cursor])
            cursor += 1
        elif expression[cursor].isalpha() or expression[cursor] == '_':
            identifier = _tokenize_identifier(expression, cursor)
            yield IdentifierToken(identifier)
            cursor += len(identifier)
        elif not expression[cursor].strip():
            cursor += 1
        else:
            raise UnknownCharacterError(
                f"Unknown character '{expression[cursor]}'."
            )
        op_failed = False
        num_failed = False
        bracket_end_failed = False
        bracket_start_failed = False
    if string_stack:
        yield StringToken(''.join(string_stack))
    if bracket_stack:
        raise UnclosedBracketsError(
            f"Brackets are not closed: {' '.join(bracket_stack)}."
        )
    return cursor


def get_lambda_divider_index(tokenized: list[Token | TokenGroup]) -> int:
    match tokenized:
        case [UnpackToken(), *_]: pass
        case [IdentifierToken(), *_]: pass
        case [ArrowToken(value="=>"), *_]: pass
        case _:
            raise InvalidLambdaExpression(
                "Lambda expression must start with identifiers, a variable "
                "argument identifier, or =>."
            )
    divider_index = None
    variable_identifier_encountered = False
    next_is_variable = False
    for i, tok in enumerate(tokenized):
        if ( 
            isinstance(tok, ArrowToken) and
            tok.value == '=>' and
            divider_index is not None
        ):
            raise InvalidLambdaExpression("Multiple lambda dividers.")
        if ( 
            isinstance(tok, ArrowToken) and
            tok.value == '=>' and
            divider_index is None
        ):
            divider_index = i
            if next_is_variable:
                raise InvalidLambdaExpression(
                    "Expected an identifier after variable argument marker."
                )
            continue
        if (
            variable_identifier_encountered and
            isinstance(tok, UnpackToken) and
            divider_index is None
        ):
            raise InvalidLambdaExpression(
                "Multiple variable argument identifiers."
            )
        elif (
            not variable_identifier_encountered and
            isinstance(tok, UnpackToken)
        ):
            next_is_variable = True
            variable_identifier_encountered = True
            continue
        if next_is_variable:
            next_is_variable = False
        if (
            not isinstance(tok, IdentifierToken) and
            divider_index is None
        ):
            raise InvalidLambdaExpression(
                "Only identifiers or a variable argument identifier allowed "
                "before lambda divider."
            )
    assert divider_index is not None
    return divider_index



def _specify_operator_type(
    tokenized: list[Token | TokenGroup],
    is_curly: bool = False
) -> list[Token | TokenGroup]:
    if not tokenized:
        return tokenized
    if is_curly:
        match tokenized:
            case [*_] if ArrowToken('=>') in tokenized:
                i = get_lambda_divider_index(tokenized)
                parameters = TokenGroup(None, tokenized[:i])
                expression = TokenGroup(None, tokenized[i+1:])
                _specify_operator_type(expression.values)
                tokenized[:] = [parameters, ArrowToken('=>'), expression]
            case TernaryOperatorFunctionPattern():
                matched = TernaryOperatorFunctionPattern.match(tokenized)
                assert matched is not MatchFailed.FAIL
                if matched[0].values:
                    _specify_operator_type(
                        matched[0].values, matched[0].bracket == '{'
                    )
                if matched[2].values:
                    _specify_operator_type(
                        matched[2].values, matched[2].bracket == '{'
                    )
                if matched[4].values:
                    _specify_operator_type(
                        matched[4].values, matched[4].bracket == '{'
                    )
                tokenized[:] = matched
            case BinaryOperatorFunctionPattern():
                matched = BinaryOperatorFunctionPattern.match(tokenized)
                assert matched is not MatchFailed.FAIL
                if matched[0].values:
                    _specify_operator_type(
                        matched[0].values, matched[0].bracket == '{'
                    )
                if matched[2].values:
                    _specify_operator_type(
                        matched[2].values, matched[2].bracket == '{'
                    )
                tokenized[:] = matched
            case PrefixUnaryOperatorFunctionPattern():
                matched = PrefixUnaryOperatorFunctionPattern.match(tokenized)
                assert matched is not MatchFailed.FAIL
                tokenized[:] = matched
            case PostfixUnaryOperatorFunctionPattern():
                matched = PostfixUnaryOperatorFunctionPattern.match(tokenized)
                assert matched is not MatchFailed.FAIL
                tokenized[:] = matched
            case _:
                raise ParserError("Invalid curly brace expression.")
        return tokenized

    for ci, (p, c, n) in enumerate(prev_curr_next_iter(tokenized)):
        if isinstance(c, TokenGroup):
            _specify_operator_type(c.values, c.bracket == '{')
            continue
        if not isinstance(c, UnknownOperatorToken):
            continue
        match (p, c, n):
            case (
                None | ArrowToken() |
                SeparatorToken() | UnpackToken(),
                UnknownOperatorToken(), None
            ):
                raise ParserError("An operator alone is not an expression.")
            # mark an operator as a prefix unary operator
            case (
                (
                    None | ArrowToken() | SeparatorToken() | UnpackToken() |
                    OperatorToken(
                        operator=(
                            PrefixUnaryOperator() | 
                            BinaryOperator() | TernaryOperator()
                        )
                    )
                ),
                UnknownOperatorToken(value=op),
                _
            ):
                if op not in PREFIX_UNARY_OPERATORS:
                    raise ParserError('Expected prefix operator at start.')
                if n is None:
                    raise ParserError(
                        "Expression ends with a prefix unary operator."
                    )
                tokenized[ci] = OperatorToken(op, PREFIX_UNARY_OPERATORS[op])
            # mark an operator as a postfix unary operator if it is
            # at the end of an expression, or as another appropriate
            # operator, ternary > binary > postfix unary
            case (
                (
                    TokenGroup() | NumberToken() | IdentifierToken() |
                    OperatorToken(operator=PostfixUnaryOperator()) |
                    StringToken()
                ),
                UnknownOperatorToken(value=op),
                _
            ):
                if (
                    n is None or isinstance(n, (SeparatorToken, ArrowToken)) or
                    isinstance(n, UnknownOperatorToken) and (
                        n.value in BINARY_OPERATORS or
                        n.value in TERNARY_OPERATORS
                    )
                ) and op in POSTFIX_UNARY_OPERATORS:
                    tokenized[ci] = OperatorToken(
                        op, POSTFIX_UNARY_OPERATORS[op]
                    )
                elif n is None:
                    raise ParserError(
                        "Unexpected operator at the end of expression."
                    )
                elif op in TERNARY_OPERATORS:
                    tokenized[ci] = OperatorToken(op, TERNARY_OPERATORS[op])
                elif op in BINARY_OPERATORS:
                    tokenized[ci] = OperatorToken(op, BINARY_OPERATORS[op])
                elif op in POSTFIX_UNARY_OPERATORS:
                    tokenized[ci] = OperatorToken(
                        op, POSTFIX_UNARY_OPERATORS[op]
                    )
                else:
                    raise ParserError("Unexpected prefix unary operator.")
    return tokenized

def group_brackets(tokenized: list[Token]) -> list[Token | TokenGroup]:
    def _group(
        start_index: int,
        final: list[Token | TokenGroup],
        until = None
    ) -> int:
        i = start_index
        while i < len(tokenized):
            match tokenized[i]:
                case OpenBracketToken() as tok:
                    result: list[Token | TokenGroup] = []
                    i = _group(i+1, result, BRACKETS[tok.value])
                    final.append(TokenGroup(tok.value, result))
                case ClosedBracketToken() as tok:
                    if tok.value == until:
                        return i+1
                    i += 1
                case tok:
                    final.append(tok)
                    i += 1
        return len(tokenized)
    result: list[Token | TokenGroup] = []
    _group(0, result)
    return result

def group_ternary(
    tokenized: list[Token | TokenGroup]
) -> list[Token | TokenGroup]:
    def _group(
        start_index: int,
        final: list[Token | TokenGroup],
        until = None
    ) -> int:
        i = start_index
        while i < len(tokenized):
            match tokenized[i]:
                case OperatorToken(
                    symbol, operator=TernaryOperator() as operator
                ) if symbol == operator.parts[0]:
                    final.append(tokenized[i])
                    result: list[Token | TokenGroup] = []
                    i = _group(i+1, result, operator.parts[1])
                    if i == len(tokenized):
                        raise ParserError('Unbalanced ternary operation.')
                    if len(result) != 1:
                        final.append(TokenGroup(None, result))
                    else:
                        final.append(result[0])
                    final.append(tokenized[i])
                    i += 1
                case OperatorToken(
                    symbol, operator=TernaryOperator() as operator
                ) if symbol == operator.parts[1]:
                    if symbol == until:
                        return i
                    raise ParserError('Unbalanced ternary operation.')
                case TokenGroup() as tokg:
                    tokg.values = group_ternary(tokg.values)
                    final.append(tokg)
                    i += 1
                case tok:
                    final.append(tok)
                    i += 1
        return len(tokenized)
    result: list[Token | TokenGroup] = []
    _group(0, result)
    return result


def get_highest_precedence_unary_operator_locations(
    tokenized: list[Token | TokenGroup]
) -> tuple[Optional[int], Optional[int]]:

    current_prefix_operator = None
    current_postfix_operator = None
    current_prefix_operator_index = None
    current_postfix_operator_index = None
    current_precedence = None
    
    for i, tok in enumerate(tokenized):
        if not isinstance(tok, OperatorToken):
            continue
        if not isinstance(tok.operator, UnaryOperator):
            continue
        if (
            current_prefix_operator is None and
            current_postfix_operator is None
        ):
            if isinstance(tok.operator, PrefixUnaryOperator):
                current_prefix_operator = tok
                current_prefix_operator_index = i
            else:
                current_postfix_operator = tok
                current_postfix_operator_index = i
            current_precedence = tok.operator.precedence
        elif (
            current_precedence is not None and
            (tok.operator.precedence > current_precedence)
        ):
            if isinstance(tok.operator, PrefixUnaryOperator):
                current_prefix_operator = tok
                current_prefix_operator_index = i
                current_postfix_operator = None
                current_postfix_operator_index = None
            else:
                current_postfix_operator = tok
                current_postfix_operator_index = i
                current_prefix_operator = None
                current_prefix_operator_index = None
        elif (
            current_precedence is not None and
            (tok.operator.precedence < current_precedence)
        ):
            continue
        # precedence is equal to the current precedence here
        # going left to right, choose right
        elif isinstance(tok.operator, PrefixUnaryOperator):
            current_prefix_operator = tok
            current_prefix_operator_index = i
        # going left to right, both postfix, choose left
        elif isinstance(tok.operator, PostfixUnaryOperator):
            # keep the operator if it exists, postfix is left-associative
            if current_postfix_operator is None:
                current_postfix_operator = tok
                current_postfix_operator_index = i
    return current_prefix_operator_index, current_postfix_operator_index

def get_grouped_by_prefix(
    tokenized: list[Token | TokenGroup],
    operator_index: int
) -> int:
    operator = cast(
        PrefixUnaryOperator,
        cast(OperatorToken, tokenized[operator_index]).operator
    )
    for i in range(operator_index+1, len(tokenized)):
        tok = tokenized[i]
        if (
            isinstance(tok, (SeparatorToken, UnpackToken)) or
            isinstance(tok, ArrowToken) and
            tok.value == '=>'
        ):
            break
        # only care about non-prefix operators
        if not isinstance(tok, OperatorToken):
            continue
        if isinstance(tok.operator, PrefixUnaryOperator):
            continue
        if tok.operator.precedence < operator.precedence:
            break
        # if the other operator is right-associative, it gets applied first
        # so don't break yet, otherwise resolve the clash
        elif (
            tok.operator.precedence == operator.precedence and
            isinstance(tok.operator, LeftAssociative)
        ):
            if (tok.operator, operator) not in CLASH_RESOLUTION:
                raise ParserError(
                    "Unaccounted for clash between operators."
                )
            if CLASH_RESOLUTION[(tok.operator, operator)] == operator:
                break
            
    else:
        i = len(tokenized)
    return i

def get_grouped_by_postfix(
    tokenized: list[Token | TokenGroup],
    operator_index: int
):
    operator = cast(
        PostfixUnaryOperator,
        cast(OperatorToken, tokenized[operator_index]).operator
    )
    i = cast(int, None)
    for i in range(operator_index-1, -1, -1):
        tok = tokenized[i]
        if (
            isinstance(tok, (SeparatorToken, UnpackToken)) or
            isinstance(tok, ArrowToken) and
            tok.value == '=>'
        ):
            break
        # only care about non postfix operators
        if not isinstance(tok, OperatorToken):
            continue
        if isinstance(tok.operator, PostfixUnaryOperator):
            continue
        if tok.operator.precedence < operator.precedence:
            break
        # if the other operator is left-associative, it gets applied first
        # so don't break yet, otherwise resolve the clash
        elif (
            tok.operator.precedence == operator.precedence and
            isinstance(tok.operator, RightAssociative)
        ):
            if (operator, tok.operator) not in CLASH_RESOLUTION:
                raise ParserError(
                    "Unaccounted for clash between operators."
                )
            if CLASH_RESOLUTION[(operator, tok.operator)] == operator:
                break
    else:
        # if unbroken, go beyond 0, as that should be included too
        i -= 1
    # exclude the operator that caused the breaking
    # this should be a start index, so inclusive at the start
    i += 1
    return i

def group_unary(
    tokenized: list[Token | TokenGroup]
) -> list[Token | TokenGroup]:
    # already in the correct format
    match tokenized:
        case [OperatorToken()]:
            return tokenized
        case [OperatorToken() as op, TokenGroup() as tokg]:
            return [op, TokenGroup(
                bracket=tokg.bracket,
                values=group_unary(tokg.values)
            )]
        case [TokenGroup() as tokg, OperatorToken() as op]:
            return [TokenGroup(
                bracket=tokg.bracket,
                values=group_unary(tokg.values)
            ), op]

    tokenized_copy = tokenized[:]
    for i, tok in enumerate(tokenized_copy):
        if isinstance(tok, TokenGroup):
            tokenized_copy[i] = TokenGroup(
                tok.bracket, group_unary(tok.values)
            )
    while True:
        prefix_index, postfix_index = (
            get_highest_precedence_unary_operator_locations(tokenized_copy)
        )
        if prefix_index is postfix_index is None:
            return tokenized_copy
        prefix = (
            tokenized_copy[prefix_index]
            if prefix_index is not None else None
        )
        postfix = (
            tokenized_copy[postfix_index]
            if postfix_index is not None else None
        )
        if prefix_index is None:
            picked = 'postfix'
        elif postfix_index is None:
            picked = 'prefix'
        # if they don't overlap it doesn't matter which one you pick
        # if they aren't in CLASH_RESOLUTION and overlap, an error
        # will be raised when trying to group them (get_grouped_by_[type])
        # if they are in CLASH_RESOLUTION, either they won't overlap
        # in which case it doesn't matter again, or they will overlap
        # and you choose the one that won the clash
        else:
            assert isinstance(postfix, OperatorToken)
            assert isinstance(prefix, OperatorToken)
            assert isinstance(prefix.operator, PrefixUnaryOperator)
            assert isinstance(postfix.operator, PostfixUnaryOperator)
            if CLASH_RESOLUTION.get(
                (postfix.operator, prefix.operator), prefix.operator
            ) == prefix.operator:
                picked = 'prefix'
            else:
                picked = 'postfix'

        result = []
        # grouped by prefix: [prefix_index, prefix_end_index)
        if picked == 'prefix':
            assert prefix_index is not None
            assert prefix is not None
            prefix_end_index = get_grouped_by_prefix(
                tokenized_copy, prefix_index
            )
            # everything before the operator
            result.extend(tokenized_copy[:prefix_index])
            # -expr should be grouped as (-(expr))
            inner = TokenGroup(bracket=None, values=[])
            outer = TokenGroup(bracket=None, values=[prefix, inner])
            result.append(outer)
            
            # put everything that the operator envelops into the group
            inner.values.extend(
                group_unary(tokenized_copy[prefix_index+1:prefix_end_index])
            )
            # everything after that the operator doesn't envelop
            result.extend(tokenized_copy[prefix_end_index:])
        # grouped by postfix: [postfix_start_index, postfix_index]
        else:
            assert postfix is not None
            assert postfix_index is not None
            postfix_start_index = get_grouped_by_postfix(
                tokenized_copy, postfix_index
            )
            # everything before that the operator doesn't envelop
            result.extend(tokenized_copy[:postfix_start_index])
            # expr- should be grouped as ((expr)-)
            inner = TokenGroup(bracket=None)
            outer = TokenGroup(bracket=None, values=[inner, postfix])
            result.append(outer)
            # everything that the operator envelops
            inner.values.extend(
                group_unary(tokenized_copy[postfix_start_index:postfix_index])
            )
            # everything after the operator
            result.extend(tokenized_copy[postfix_index+1:])
        tokenized_copy = result

def full_tokenize(expression: str):
    tokenized = list(tokenize_expression(expression))
    tokenized = group_brackets(tokenized)
    tokenized = _specify_operator_type(tokenized)
    tokenized = group_ternary(tokenized)
    tokenized = group_unary(tokenized)
    return tokenized