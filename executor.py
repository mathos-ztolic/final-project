from __future__ import annotations
from collections import ChainMap
from typing import Any, Callable, Optional
import types
from parser import (
    ExpressionNode, IdentifierNode, ConstantNode, ListNode,
    PrefixUnaryOperationNode, PostfixUnaryOperationNode,
    BinaryOperationNode, TernaryOperationNode, UnpackNode,
    GroupNode, LambdaFunctionNode, FunctionApplicationNode,
    IndexingNode, PrefixUnaryOperatorFunctionNode,
    PostfixUnaryOperatorFunctionNode, BinaryOperatorFunctionNode,
    TernaryOperatorFunctionNode, AttributeAccessNode, TupleNode,
    Uncomputed, DictionaryNode
)
import copy
from custom_types import (
    coerce, CallableWrapper, CallableObject, GeneralSlice,
    BLANK_SLICE, BlankSlice, CallableType
)
from defaults import defaults
from sets import (
    SetCartesianProduct, SetUnion, SetIntersection,
    SetDifference, SetSymmetricDifference
)
import operator
import math
from collections.abc import Iterable
from enum import Enum

class Scope[K, V](ChainMap[K, V]):
    def __getitem__(self, key):
        try:
            return self.maps[0][key]
        except KeyError:
            pass
        for mapping in self.maps:
            try:
                try:
                    if mapping['*']:  # marks a class scope
                        continue
                except KeyError:
                    pass
                return mapping[key]
            except KeyError:
                pass
        return self.__missing__(key)

global_scope = Scope[str, Any]({})


PREFIX_UNARY_OPERATOR_FUNCTIONS: dict[str, Optional[Callable[[Any], Any]]] = {
    '-': operator.neg,
    '+': operator.pos,
    '~': operator.invert,
    '!': operator.not_,
    '`': lambda f: (
        f.prepend(...)
        if isinstance(f, CallableObject) else
        CallableWrapper(f, (...,), ())
    ),
    '^`': lambda f: (
        f.prepend_right(...)
        if isinstance(f, CallableObject) else
        CallableWrapper(f, (...,), ())
    )
}
POSTFIX_UNARY_OPERATOR_FUNCTIONS: dict[str, Optional[Callable[[Any], Any]]] = {
    '!': lambda x: math.gamma(x+1),
    '\'': lambda f: (
        f.append(...)
        if isinstance(f, CallableObject) else
        CallableWrapper(f, (), (...,))
    ),
    '^\'': lambda f: (
        f.append_left(...)
        if isinstance(f, CallableObject) else
        CallableWrapper(f, (), (...,))
    ),
}
BINARY_OPERATOR_FUNCTIONS: dict[str, Optional[Callable[[Any, Any], Any]]] = {
    '`': lambda x, f: (
        f.prepend(x)
        if isinstance(f, CallableObject) else
        CallableWrapper(f, (x,), ())
    ),
    '^`': lambda x, f: (
        f.prepend_right(x)
        if isinstance(f, CallableWrapper) else
        CallableWrapper(f, (x,), ())
    ),
    '\'': lambda f, x: (
        f.append(x)
        if isinstance(f, CallableObject) else
        CallableWrapper(f, (), (x,))
    ),
    '^\'': lambda f, x: (
        f.append_left(x)
        if isinstance(f, CallableObject) else
        CallableWrapper(f, (), (x,))
    ),
    '||': (lambda a, b: a or b),
    '&&': (lambda a, b: a and b),
    '&': operator.and_, '|': operator.or_, '#': operator.xor,
    '<<': operator.lshift, '>>': operator.rshift,
    '+': operator.add, '-': operator.sub, '*': operator.mul,
    '/': operator.truediv, '//': operator.floordiv,
    '%': operator.mod, '^': operator.pow,
    '>': operator.gt, '<': operator.lt,
    '>=': operator.ge, '<=': operator.le,
    '==': operator.eq, '!=': operator.ne,
    '@': lambda f, g: CallableWrapper(lambda *args: f(g(*args))),
    'in': lambda e, t: e in t,
    '!in': lambda e, t: e not in t,
    '$|': lambda a, b: SetUnion(a, b),
    '$&': lambda a, b: SetIntersection(a, b),
    '$-': lambda a, b: SetDifference(a, b),
    '$#': lambda a, b: SetSymmetricDifference(a, b),
    '$*': lambda a, b: SetCartesianProduct(a, b),
    '??': lambda a, b: a if a is not None else b,
    '!!': lambda a, b: a if a is None else b,
    '..!': lambda a, b: (
        list(range(a, b))
        if b >= a else
        list(range(a, b, -1))
    ),
    '..': lambda a, b: (
        list(range(a, b+1))
        if b >= a else
        list(range(a, b-1, -1))
    ),
    '!..': lambda a, b: (
        list(range(a+1, b+1))
        if b >= a else
        list(range(a-1, b-1, -1))
    ),
    '!..!': lambda a, b: (
        list(range(a+1, b))
        if b >= a else
        list(range(a-1, b, -1))
    ),
    
    ':=': None,
    '==>': None,
    '<==': None,
    ';': lambda a, b: b,
}
TERNARY_OPERATOR_FUNCTIONS: dict[
    str, Optional[Callable[[Any, Any, Any], Any]]
] = {
    '?:': (lambda test, true, false: true if test else false),
    ':!:?': (lambda test, nonnull, null: nonnull if test is not None else null)
}


class Missing(Enum):
    MISSING = 0

def prepare_index(node, scope) -> Any | GeneralSlice | BlankSlice:
    index = []
    for i in node.index:
        if isinstance(i, UnpackNode):
            for i2 in execute_tree(i.expr, scope):
                index.append(i2)
        else:
            index.append(
                execute_tree(i, scope) if i is not None else None
            )
    
    final_index: Any | GeneralSlice | BlankSlice
    if len(index) == 0:
        final_index = BLANK_SLICE
    elif len(index) == 1:
        final_index = index[0]
    else:
        final_index = GeneralSlice(*index)
    
    return final_index

def execute_tree(node: ExpressionNode, scope = global_scope):
    if node.result is not Uncomputed.UNCOMPUTED:
        return node.result
    match node:
        case IdentifierNode():
            try:
                node.result = coerce(scope[node.value])
            except KeyError:
                node.result = coerce(defaults[node.value])
            return node.result
        case ConstantNode():
            node.result = coerce(node.value)
            return node.result
        case ListNode():
            exprs = []
            for expr in node.exprs:
                if isinstance(expr, UnpackNode):
                    for expr2 in execute_tree(expr.expr, scope):
                        exprs.append(expr2)
                else:
                    exprs.append(execute_tree(expr, scope))
            node.result = coerce(exprs)
            return node.result
        case TupleNode():
            exprs = []
            for expr in node.exprs:
                if isinstance(expr, UnpackNode):
                    for expr2 in execute_tree(expr.expr, scope):
                        exprs.append(expr2)
                else:
                    exprs.append(execute_tree(expr, scope))
            node.result = coerce(tuple(exprs))
            return node.result
        case DictionaryNode():
            dct = {}
            for key, value in zip(node.keys, node.values):
                # unpack
                if value is None:
                    dct.update(execute_tree(key, scope))
                else:
                    dct[execute_tree(key, scope)] = execute_tree(value, scope)
            node.result = coerce(dct)
            return node.result
        case GroupNode():
            node.result = execute_tree(node.expr, scope)
            return node.result
        case PrefixUnaryOperationNode():
            if node.operator in ('++', '--'):
                what = 'increment' if node.operator == '++' else 'decrement'
                if isinstance(node.expr, IdentifierNode):
                    if what == 'increment':
                        scope.maps[0][node.expr.value] += 1
                    else:
                        scope.maps[0][node.expr.value] -= 1
                    node.result = scope.maps[0][node.expr.value]
                elif isinstance(node.expr, IndexingNode):
                    if node.expr.null_conditional_level != 0:
                        raise IndexError(
                            "Null-conditional indexing "
                            "not allowed in assignment."
                        )
                    indexed = execute_tree(node.expr.indexed, scope)
                    index = prepare_index(node.expr, scope)
                    if what == 'increment':
                        indexed[index] += 1
                    else:
                        indexed[index] -= 1
                    node.result = indexed[index]
                elif isinstance(node.expr, AttributeAccessNode):
                    if node.expr.null_conditional_level != 0:
                        raise IndexError(
                            "Null-conditional attribute access "
                            "not allowed in assignment."
                        )
                    if (
                        node.expr.attribute.startswith('__') and
                        node.expr.attribute.endswith('__')
                    ):
                        raise AttributeError(
                            "Cannot access dunder attributes."
                        )
                    accessed = execute_tree(node.expr.accessed, scope)
                    atttibute = node.expr.attribute
                    
                    if what == 'increment':
                        setattr(
                            accessed, attribute, getattr(accessed, attribute)+1
                        )
                    else:
                        setattr(
                            accessed, attribute, getattr(accessed, attribute)-1
                        )
                    node.result = getattr(accessed, attribute)
                else:
                    raise ValueError("Invalid assignment target.")
                return node.result
            uop = PREFIX_UNARY_OPERATOR_FUNCTIONS[node.operator]
            assert uop is not None
            node.result = coerce(uop(execute_tree(node.expr, scope)))
            return node.result
        case PostfixUnaryOperationNode():
            if node.operator in ('++', '--'):
                what = 'increment' if node.operator == '++' else 'decrement'
                if isinstance(node.expr, IdentifierNode):
                    node.result = scope.maps[0][node.expr.value]
                    if what == 'increment':
                        scope.maps[0][node.expr.value] += 1
                    else:
                        scope.maps[0][node.expr.value] -= 1
                elif isinstance(node.expr, IndexingNode):
                    if node.expr.null_conditional_level != 0:
                        raise IndexError(
                            "Null-conditional indexing "
                            "not allowed in assignment."
                        )
                    indexed = execute_tree(node.expr.indexed, scope)
                    index = prepare_index(node.expr, scope)
                    node.result = indexed[index]
                    if what == 'increment':
                        indexed[index] += 1
                    else:
                        indexed[index] -= 1
                elif isinstance(node.expr, AttributeAccessNode):
                    if node.expr.null_conditional_level != 0:
                        raise IndexError(
                            "Null-conditional attribute access "
                            "not allowed in assignment."
                        )
                    if (
                        node.expr.attribute.startswith('__') and
                        node.expr.attribute.endswith('__')
                    ):
                        raise AttributeError(
                            "Cannot access dunder attributes."
                        )
                    accessed = execute_tree(node.expr.accessed, scope)
                    atttibute = node.expr.attribute
                    node.result = getattr(accessed, attribute)
                    if what == 'increment':
                        setattr(
                            accessed, attribute, getattr(accessed, attribute)+1
                        )
                    else:
                        setattr(
                            accessed, attribute, getattr(accessed, attribute)-1
                        )
                else:
                    raise ValueError("Invalid assignment target.")
                return node.result
            uop = POSTFIX_UNARY_OPERATOR_FUNCTIONS[node.operator]
            assert uop is not None
            node.result = coerce(uop(execute_tree(node.expr, scope)))
            return node.result
        case BinaryOperationNode():
            bop = BINARY_OPERATOR_FUNCTIONS[node.operator]
            # special case: class creation
            if node.operator == '==>':
                if (
                    isinstance(node.left_expr, BinaryOperationNode) and
                    node.left_expr.operator == '<=='
                ):
                    if (
                        isinstance(node.left_expr.left_expr, ConstantNode) and
                        isinstance(node.left_expr.left_expr.value, str) and
                        node.left_expr.left_expr.value.isidentifier()
                    ):
                        assign_to_scope = False
                    elif isinstance(node.left_expr.left_expr, IdentifierNode):
                        assign_to_scope = True
                    else:
                        raise ValueError("Class name has to be identifier.")
                    name = node.left_expr.left_expr.value
                    result = execute_tree(node.left_expr.right_expr, scope)
                    if isinstance(result, Iterable):
                        result = tuple(result)
                    if (
                        isinstance(result, tuple) and
                        all(isinstance(x, type) for x in result)
                    ):
                        bases = result
                    elif isinstance(result, type):
                        bases = (result,)
                    else:
                        raise ValueError("Invalid base class(es).")
                    
                elif isinstance(node.left_expr, IdentifierNode):
                    bases = ()
                    name = node.left_expr.value
                    assign_to_scope = True
                elif (
                    isinstance(node.left_expr, ConstantNode) and
                    isinstance(node.left_expr.value, str) and
                    node.left_expr.value.isidentifier()
                ):
                    bases = ()
                    name = node.left_expr.value
                    assign_to_scope = False
                else:
                    raise ValueError("Invalid class spec.")
                
                new_scope = scope.new_child({'*': True})
                execute_tree(node.right_expr, new_scope)
                dictionary = new_scope.maps[0]
                # docs say the dictionary 'may' be copied or wrapped
                # doesn't seem like a guarantee, so copy it manually,
                # as it gets cleared later
                node.result = type(name, bases, dictionary)
                if assign_to_scope:
                    scope[name] = node.result
                # this prevents the functions defined inside the scope
                # from accessing other functions without attribute
                # access
                #dictionary.clear()
                return node.result
            # special case: assignment
            elif node.operator == ':=':
                if isinstance(node.left_expr, IdentifierNode):
                    scope[node.left_expr.value] = node.result = (
                        execute_tree(node.right_expr, scope)
                    )
                elif isinstance(node.left_expr, IndexingNode):
                    if node.left_expr.null_conditional_level != 0:
                        raise IndexError(
                            "Null-conditional indexing "
                            "not allowed in assignment."
                        )
                    indexed = execute_tree(node.left_expr.indexed, scope)
                    index = prepare_index(node.left_expr, scope)
                    indexed[index] = node.result = execute_tree(
                        node.right_expr, scope
                    )
                elif isinstance(node.left_expr, AttributeAccessNode):
                    if node.left_expr.null_conditional_level != 0:
                        raise IndexError(
                            "Null-conditional attribute access "
                            "not allowed in assignment."
                        )
                    if (
                        node.left_expr.attribute.startswith('__') and
                        node.left_expr.attribute.endswith('__')
                    ):
                        raise AttributeError(
                            "Cannot access dunder attributes."
                        )
                    accessed = execute_tree(node.left_expr.accessed, scope)
                    attribute = node.left_expr.attribute
                    setattr(
                        accessed, attribute,
                        execute_tree(node.right_expr, scope)
                    )
                    node.result = getattr(accessed, attribute)
                else:
                    raise ValueError("Invalid assignment target.")
                return node.result
            # special case: null-coalescing operator
            elif node.operator == '??':
                node.result = coerce(
                    node.left_expr.result
                    if execute_tree(node.left_expr, scope) is not None else
                    execute_tree(node.right_expr, scope)
                )
            # special case: opposite of null-coalescing
            elif node.operator == '!!':
                node.result = coerce(
                    node.left_expr.result
                    if execute_tree(node.left_expr, scope) is None else
                    execute_tree(node.right_expr, scope)
                )
            # special case: short circuit and
            elif node.operator == '&&':
                node.result = coerce(
                    execute_tree(node.left_expr, scope) and
                    execute_tree(node.right_expr, scope)
                )
            # special case: short circuit or
            elif node.operator == '||':
                node.result = coerce(
                    execute_tree(node.left_expr, scope) or
                    execute_tree(node.right_expr, scope)
                )
            # special cases: chained relationals
            elif (
                node.operator in ('>', '<', '>=', '<=', '==', '!=') and
                isinstance(node.left_expr, BinaryOperationNode) and
                node.left_expr.operator in ('>', '<', '>=', '<=', '==', '!=')
            ):
                assert bop is not None
                node.result = coerce(
                    execute_tree(node.left_expr, scope) and
                    coerce(bop(
                        node.left_expr.right_expr.result,
                        execute_tree(node.right_expr, scope)
                    ))
                )
            # special case: n-ary cartesian product
            elif node.operator == '$*' and (
                isinstance(node.left_expr, BinaryOperationNode) and
                node.left_expr.operator == '$*'
            ):
                node.result = coerce(SetCartesianProduct(
                    *execute_tree(node.left_expr, scope).sets,
                    execute_tree(node.right_expr, scope)
                ))
            else:
                assert bop is not None
                node.result = coerce(bop(
                    execute_tree(node.left_expr, scope),
                    execute_tree(node.right_expr, scope)
                ))
            return node.result
        case TernaryOperationNode():
            # special case: conditional ternary short circuit
            if node.operator == '?:':
                node.result = coerce(
                    execute_tree(node.mid_expr, scope) if
                    execute_tree(node.left_expr, scope) else
                    execute_tree(node.right_expr, scope)
                )
                return node.result
            # special case: null-conditional ternary short circuit
            if node.operator == ':!:?':
                node.result = coerce(
                    execute_tree(node.mid_expr, scope) if
                    execute_tree(node.left_expr, scope) is not None else
                    execute_tree(node.right_expr, scope)
                )
                return node.result
            top = TERNARY_OPERATOR_FUNCTIONS[node.operator]
            assert top is not None
            node.result = coerce(top(
                execute_tree(node.left_expr, scope),
                execute_tree(node.mid_expr, scope),
                execute_tree(node.right_expr, scope)
            ))
            return node.result
        case LambdaFunctionNode():
            variable_arg_index = next((
                i for i,x in enumerate(node.positional_parameters)
                if x.startswith('...')
            ), None)
            def caller(
                *args,
                **kwargs
            ):
                # these are defaults for function calls
                # the CallableWrapper's descriptor will
                # modify them if invoked
                # invalid identifier to reduce the chance of being set
                # as a kwarg
                special_kwargs = kwargs.pop('*', {})
                this = special_kwargs.pop('this', Missing.MISSING)
                cls = special_kwargs.pop('cls', Missing.MISSING)
                callable_type = special_kwargs.pop(
                    'callable_type', CallableType.NON_METHOD
                )
                pparameters = node.positional_parameters[:]
                kparameters = node.keyword_parameters[:]
                args = list(args)
                if callable_type is CallableType.UNBOUND_METHOD:
                    pparameters.insert(0, 'this')
                elif callable_type is CallableType.UNBOUND_CLASS_METHOD:
                    pparameters.insert(0, 'cls')
                    
                if variable_arg_index is not None:
                    if len(args) < len(pparameters) - 1:
                        raise ValueError("Too few arguments.")
                elif len(args) < len(pparameters):
                    raise ValueError("Too few arguments.")
                elif len(args) > len(pparameters):
                    raise ValueError("Too many arguments.")

                new_kwargs = {}
                if kparameters:
                    if kparameters[-1].startswith('....'):
                        new_kwargs[kparameters[-1][4:]] = {}
                        for k in kwargs:
                            if k not in kparameters:
                                if k in pparameters:
                                    raise ValueError(
                                        f"{k} is a positional argument."
                                    )
                                new_kwargs[kparameters[-1][4:]][k] = kwargs[k]
                            else:
                                new_kwargs[k] = kwargs[k]
                        for k in kparameters[:-1]:
                            if k not in new_kwargs:
                                raise ValueError(
                                    f"Expected keyword argument {k}."
                                )
                    else:
                        for k in kwargs:
                            if k not in kparameters:
                                raise ValueError(
                                    f"Unknown keyword argument {k}."
                                )
                            else:
                                new_kwargs[k] = kwargs[k]
                        for k in kparameters:
                            if k not in new_kwargs:
                                raise ValueError(
                                    f"Expected keyword argument {k}."
                                )
                elif kwargs:
                    raise ValueError(
                        "Function does not take keyword arguments."
                    )
                
                if variable_arg_index is not None:
                    before_va = len(pparameters[:variable_arg_index])
                    after_va = len(pparameters[variable_arg_index+1:])
                    new_scope = scope.new_child({
                        **{
                            pparameters[i]: args[i]
                            for i in range(before_va)
                        },
                        pparameters[variable_arg_index][3:]: coerce(tuple(
                            args[i] for i in range(
                                before_va, len(args)-after_va
                            )
                        )),
                        **{
                            pparameters[variable_arg_index+1+i]: args[
                                len(args)-after_va+i
                            ]
                            for i in range(after_va)
                        },
                        **new_kwargs
                    })
                else:
                    new_scope = scope.new_child(
                        {
                            pparameters[i]: args[i]
                            for i in range(len(args))
                        },
                        **new_kwargs
                    )
                if callable_type is CallableType.UNBOUND_METHOD:
                    new_scope['this'] = args[0]
                    new_scope['cls'] = type(args[0])
                elif callable_type is CallableType.UNBOUND_CLASS_METHOD:
                    new_scope['cls'] = args[0]
                elif callable_type is CallableType.BOUND_METHOD:
                    new_scope['this'] = this
                    new_scope['cls'] = type(this)
                elif callable_type is CallableType.BOUND_CLASS_METHOD:
                    new_scope['cls'] = cls
                expr = copy.deepcopy(node.expr)
                return execute_tree(expr, new_scope)
            node.result = CallableWrapper(
                caller,
                callable_type='lambda function',
                name='<lambda>'
            )
            return node.result
        case PrefixUnaryOperatorFunctionNode():
            def caller(*args):
                if len(args) == 0:
                    raise ValueError("Too few arguments.")
                elif len(args) > 1:
                    raise ValueError("Too many arguments.")
                arg = ExpressionNode(result=args[0])
                opnode = PrefixUnaryOperationNode(node.operator, arg)
                return execute_tree(opnode, scope.new_child({}))
            node.result = CallableWrapper(
                caller,
                name=f'{{{node.operator}}}',
                callable_type='prefix unary operator function'
            )
            return node.result
        case PostfixUnaryOperatorFunctionNode():
            def caller(*args):
                if len(args) == 0:
                    raise ValueError("Too few arguments.")
                elif len(args) > 1:
                    raise ValueError("Too many arguments.")
                arg = ExpressionNode(result=args[0])
                opnode = PostfixUnaryOperationNode(node.operator, arg)
                return execute_tree(opnode, scope.new_child({}))
            node.result = CallableWrapper(
                caller,
                name=f'{{{node.operator}}}',
                callable_type='postfix unary operator function'
            )
            return node.result
        case BinaryOperatorFunctionNode():
            def caller(*args):
                argcount = 2
                if node.args[0] is not ...:
                    argcount -= 1
                if node.args[1] is not ...:
                    argcount -= 1
                if len(args) < argcount:
                    raise ValueError("Too few arguments.")
                elif len(args) > argcount:
                    raise ValueError("Too many arguments.")
                args = iter(args)
                left = (
                    ExpressionNode(result=next(args))
                    if node.args[0] is ... else copy.deepcopy(node.args[0])
                )
                right = (
                    ExpressionNode(result=next(args))
                    if node.args[1] is ... else copy.deepcopy(node.args[1])
                )
                opnode = BinaryOperationNode(
                    node.operator, left, right
                )
                return execute_tree(opnode, scope.new_child({}))
            node.result = CallableWrapper(
                caller,
                callable_type='binary operator function',
                name=f'{{{node.operator}}}',
            )
            return node.result
        case TernaryOperatorFunctionNode():
            def caller(*args):
                argcount = 3
                if node.args[0] is not ...:
                    argcount -= 1
                if node.args[1] is not ...:
                    argcount -= 1
                if node.args[2] is not ...:
                    argcount -= 1
                if len(args) < argcount:
                    raise ValueError("Too few arguments.")
                elif len(args) > argcount:
                    raise ValueError("Too many arguments.")
                args = iter(args)
                left = (
                    ExpressionNode(result=next(args))
                    if node.args[0] is ... else copy.deepcopy(node.args[0])
                )
                mid = (
                    ExpressionNode(result=next(args))
                    if node.args[1] is ... else copy.deepcopy(node.args[1])
                )
                right = (
                    ExpressionNode(result=next(args))
                    if node.args[2] is ... else copy.deepcopy(node.args[2])
                )
                opnode = TernaryOperationNode(
                    node.operator, left, mid, right
                )
                return execute_tree(opnode, scope.new_child({}))
            node.result = CallableWrapper(
                caller,
                callable_type='ternary operator function',
                name=f'{{{node.operator}}}',
            )
            return node.result
        case FunctionApplicationNode():
            function = execute_tree(node.function, scope)
            if node.function.null_conditional_failed:
                node.null_conditional_failed = True
                return
            args = []
            kwargs = {}
            for arg in node.args:
                if isinstance(arg, UnpackNode):
                    args.extend(execute_tree(arg.expr, scope))
                else:
                    args.append(execute_tree(arg, scope))
            for key, value in node.kwargs:
                if key is None:
                    kwargs.update(execute_tree(value.expr, scope))
                else:
                    kwargs[coerce(key)] = execute_tree(value, scope)
            if '*' in kwargs:
                raise ValueError("The keyword argument '*' is reserved.")
            node.result = coerce(function(*args, **kwargs))
            return node.result
        case IndexingNode():
            indexed = execute_tree(node.indexed, scope)
            if (
                node.indexed.null_conditional_failed or
                indexed is None and node.null_conditional_level
            ):
                node.null_conditional_failed = True
                return
            
            try:
                node.result = coerce(indexed[prepare_index(node, scope)])
            except IndexError:
                if node.null_conditional_level == 2:
                    node.null_conditional_failed = True
                    return
                raise
            return node.result
        case AttributeAccessNode():
            if (
                node.attribute.startswith('__') and
                node.attribute.endswith('__')
            ):
                raise AttributeError("Cannot access dunder attributes.")
            accessed = execute_tree(node.accessed, scope)
            if (
                node.accessed.null_conditional_failed or
                accessed is None and node.null_conditional_level
            ):
                node.null_conditional_failed = True
                return
            try:
                node.result = getattr(accessed, node.attribute)
            except AttributeError as e:
                if node.null_conditional_level == 2:
                    node.null_conditional_failed = True
                    return
                e.name = None
                raise
            if (
                callable(node.result) and
                not isinstance(node.result, CallableObject) and
                isinstance(node.result, (
                    types.MethodType, types.BuiltinMethodType,
                    types.WrapperDescriptorType, types.MethodWrapperType,
                    types.MethodDescriptorType, types.ClassMethodDescriptorType
                ))
            ):
                node.result = CallableWrapper(
                    node.result, name=node.result.__name__,
                    callable_type='method'
                )
            elif (
                callable(node.result) and
                not isinstance(node.result, CallableObject)
            ):
                assert hasattr(node.result, '__name__')
                node.result = CallableWrapper(
                    node.result, name=node.result.__name__,
                    callable_type='function'
                )
            node.result = coerce(node.result)
            return node.result
        case _:
            assert False, node
