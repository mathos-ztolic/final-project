from __future__ import annotations
from typing import overload, Optional
import copy
import types
from functools import partial, reduce
from itertools import (
    starmap, islice, cycle, filterfalse, takewhile, dropwhile, product, chain
)
from collections import deque
from typing import Self, Any, Optional, cast
from collections.abc import Iterable, Iterator, Callable, Sequence
from operator import mul
import math
from reprlib import recursive_repr
from enum import Enum
import operator

class Pipeline:
    
    orig: Sequence[Any]
    __transformations__: tuple[Callable[[Iterator[Any]], Iterable[Any]], ...]
    
    def __init__(self, lst: Sequence[Any]):
        self.orig = lst
        self.__transformations__ = ()
    
    def __new_pipeline__(
        self, function: Callable[[Iterator[Any]], Iterable[Any]]
    ) -> Self:
        new = self.__class__(self.orig)
        new.__transformations__ = self.__transformations__ + (function,)
        return new
    
    # transformations
    def transform(self, function: Callable):
        return self.__new_pipeline__(function)
    
    def map(self, function: Callable) -> Self:
        return self.__new_pipeline__(partial(map, function))
    
    def umap(self, function: Callable) -> Self:
        return self.__new_pipeline__(partial(starmap, function))
    
    def peek(self, function: Callable) -> Self:
        return self.__new_pipeline__(
            lambda g: ((x, function(x))[0] for x in g)
        )
    
    def upeek(self, function: Callable) -> Self:
        return self.__new_pipeline__(
            lambda g: ((x, function(*x))[0] for x in g)
        )
    
    def filter(self, predicate: Callable = bool) -> Self:
        return self.__new_pipeline__(partial(filter, predicate))
    
    def ufilter(self, predicate: Callable = bool) -> Self:
        return self.__new_pipeline__(partial(filter, lambda e: predicate(*e)))
    
    def filterfalse(self, predicate: Callable = bool) -> Self:
        return self.__new_pipeline__(partial(filterfalse, predicate))
    
    def ufilterfalse(self, predicate: Callable = bool) -> Self:
        return self.__new_pipeline__(
            partial(filterfalse, lambda e: predicate(*e))
        )
    
    def takewhile(self, predicate: Callable) -> Self:
        return self.__new_pipeline__(partial(takewhile, predicate))
    
    def utakewhile(self, predicate: Callable) -> Self:
        return self.__new_pipeline__(
            partial(takewhile, lambda e: predicate(*e))
        )
    
    def dropwhile(self, predicate: Callable) -> Self:
        return self.__new_pipeline__(partial(dropwhile, predicate))
    
    def udropwhile(self, predicate: Callable) -> Self:
        return self.__new_pipeline__(
            partial(dropwhile, lambda e: predicate(*e))
        )
    
    def flatmap(self, function: Callable) -> Self:
        return self.__new_pipeline__(
            lambda g: (j for i in map(function, g) for j in i)
        )
    
    def uflatmap(self, function: Callable) -> Self:
        return self.__new_pipeline__(
            lambda g: (j for i in starmap(function, g) for j in i)
        )
    
    def sorted(self, function: Optional[Callable[[Any], Any]] = None) -> Self:
        # branch is useless at runtime, but
        # stops pyright from yelling at me
        if function is None:
            return self.__new_pipeline__(sorted)
        else:
            return self.__new_pipeline__(partial(sorted, key=function))
    
    def usorted(self, function: Optional[Callable] = None) -> Self:
        key = (lambda e: function(*e)) if function else function
        if key is None:
            return self.__new_pipeline__(sorted)
        else:
            return self.__new_pipeline__(partial(sorted, key=key))
    
    def rsorted(self, function: Optional[Callable] = None) -> Self:
        if function is None:
            return self.__new_pipeline__(partial(sorted, reverse=True))
        else:
            return self.__new_pipeline__(
                partial(sorted, key=function, reverse=True)
            )
    
    def ursorted(self, function: Optional[Callable] = None) -> Self:
        key = (lambda e: function(*e)) if function else function
        if key is None:
            return self.__new_pipeline__(partial(sorted, reverse=True))
        else:
            return self.__new_pipeline__(
                partial(sorted, key=key, reverse=True)
            )
    
    def reversed(self) -> Self:
        return self.__new_pipeline__(lambda g: reversed(list(g)))
    
    def flatten(self) -> Self:
        return self.__new_pipeline__(lambda g: (j for i in g for j in i))
        
    
    def take(self, n: int) -> Self:
        return self.__new_pipeline__(lambda g: islice(g, n))
        
    
    def drop(self, n: int) -> Self:
        return self.__new_pipeline__(lambda g: islice(g, n, None))
        
    
    def roundrobin(self) -> Self:
        def roundrobin_(*iterables):
            iterators = map(iter, iterables)
            for num_active in range(len(iterables), 0, -1):
                iterators = cycle(islice(iterators, num_active))
                yield from map(next, iterators)
        return self.__new_pipeline__(lambda g: roundrobin_(*g))
        
    
    def slide(self, n: int, skip: int = 1) -> Self:
        if skip <= 0:
            raise ValueError("Skip has to be at least 1.")
        def sliding_window(iterable):
            curr_skip = skip
            iterator = iter(iterable)
            window = deque(islice(iterator, n), maxlen=n)
            if len(window) != n:
                return
            yield list(window)
            for x in iterator:
                window.append(x)
                curr_skip -= 1
                if not curr_skip:
                    yield list(window)
                    curr_skip = skip
        return self.__new_pipeline__(sliding_window)
        
    
    def enumerate(self, start: int = 0) -> Self:
        return self.__new_pipeline__(lambda g: enumerate(g, start))
        
    
    def batch(self, n: int) -> Self:
        if n < 1:
            raise ValueError('n must be at least one')
        def batched(iterable):
            iterator = iter(iterable)
            while batch := tuple(islice(iterator, n)):
                yield batch
        return self.__new_pipeline__(batched)
    
    def batch_strict(self, n: int) -> Self:
        if n < 1:
            raise ValueError('n must be at least one')
        def batched(iterable):
            iterator = iter(iterable)
            while batch := list(islice(iterator, n)):
                if len(batch) != n:
                    raise ValueError('incomplete batch')
                yield batch
        return self.__new_pipeline__(batched)
        
    
    def append(self, elem: Any) -> Self:
        return self.__new_pipeline__(lambda g: chain(g, [elem]))
        
    
    def prepend(self, elem: Any) -> Self:
        return self.__new_pipeline__(lambda g: chain([elem], g))
        
    
    def extendleft(self, elem: Any) -> Self:
        return self.__new_pipeline__(lambda g: chain(elem, g))
        
    
    def extend(self, elem: Any) -> Self:
        return self.__new_pipeline__(lambda g: chain(g, elem))
        
    
    def cartesianproduct(self, *iters: Iterable | Self) -> Self:
        iterables = [
            i.__gen__() if isinstance(i, Pipeline) else i
            for i in iters
        ]
        return self.__new_pipeline__(
            lambda g: map(list, product(g, *iterables))
        )
    
    def ucartesianproduct(self, *iters: Iterable | Self) -> Self:
        iterables = [
            i.__gen__() if isinstance(i, Pipeline) else i
            for i in iters
        ]
        
        return self.__new_pipeline__(
            lambda g: map(
                lambda p: [*p[0], *p[1:]], product(g, *iterables)
            )
        )
    
    def rcartesianproduct(self, r: int, *iters: Iterable | Self) -> Self:
        iterables = [
            i.__gen__() if isinstance(i, Pipeline) else i
            for i in iters
        ]
        return self.__new_pipeline__(
            lambda g: map(list, product(g, *iterables, repeat=r))
        )
        
    
    def urcartesianproduct(self, r: int, *iters: Iterable | Self) -> Self:
        iterables = [
            i.__gen__() if isinstance(i, Pipeline) else i
            for i in iters
        ]
        return self.__new_pipeline__(
            lambda g: map(lambda p: [*p[0], *p[1:]], product(
                g, *iterables, repeat=r)
            )
        )
    
    cartprod = cartesianproduct
    ucartprod = ucartesianproduct
    rcartprod = rcartesianproduct
    urcartprod = urcartesianproduct

    # actions
    def min(self, function: Optional[Callable] = None) -> Any:
        return min(self.__gen__(), key=function)
    
    def umin(self, function: Optional[Callable] = None) -> Any:
        key = (lambda e: function(*e)) if function else function
        return min(self.__gen__(), key=key)

    def max(self, function: Optional[Callable] = None) -> Any:
        return max(self.__gen__(), key=function)
    
    def umax(self, function: Optional[Callable] = None) -> Any:
        key = (lambda e: function(*e)) if function else function
        return max(self.__gen__(), key=key)

    def sum(self, initial: Any = 0) -> Any:
        return sum(self.__gen__(), initial)

    def product(self, initial: Any = 1) -> Any:
        return self.foldl(mul, initial)
    
    def all(self) -> bool:
        return all(self.__gen__())
    
    def any(self) -> bool:
        return any(self.__gen__())
    
    def count(self, predicate: Callable = bool):
        return sum(1 for x in filter(predicate, self.__gen__()))
        
    def foreach(self, function: Callable) -> None:
        for x in self.__gen__():
            function(x)
        
    def uforeach(self, function: Callable) -> None:
        for x in self.__gen__():
            function(*x)
    
    def foldl(self, function: Callable, initial: Any) -> Any:
        return reduce(function, self.__gen__(), initial)
    
    def ufoldl(self, function: Callable, initial: Any) -> Any:
        function = lambda a, b: function(a, *b)
        return reduce(function, self.__gen__(), initial)
    
    def foldr(self, function: Callable, initial: Any) -> Any:
        function = lambda a, b: function(b, a)
        return reduce(function, reversed(self.compute()), initial)
    
    def ufoldr(self, function: Callable, initial: Any) -> Any:
        function = lambda b, a: function(*b, a)
        return reduce(function, reversed(self.compute()), initial)
    
    def exec(self, function: Callable) -> Any:
        return function(self.compute())
    
    def uexec(self, function: Callable) -> Any:
        return function(*self.__gen__())
    
    def __gen__(self) -> Iterator[Any]:
        result = iter(self.orig)
        for transformation in self.__transformations__:
            result = iter(transformation(result))
        return result
    
    def compute(self) -> list[Any]:
        return list(self.__gen__())
    
    @recursive_repr()
    def __repr__(self) -> str:
        transformations = len(self.__transformations__)
        s = 's' if transformations != 1 else ''
        return (
            f"<pipeline for {self.orig} with "
            f"{transformations} transformation{s}>"
        )

class ListExtension(list):

    def get(self, index, default=None):
        try:
            return self[index]
        except IndexError:
            return default
    
    def pipe(self):
        return Pipeline(self[:])
    
    def __getitem__(self, item):
        if isinstance(item, GeneralSlice):
            if len(item.elements) > 3:
                raise IndexError("Lists take a maximum of 3 slice elements.")
            return super().__getitem__(slice(*item.elements))
        elif item is BLANK_SLICE:
            return super().__getitem__(slice(None))
        return super().__getitem__(item)
    
    def __setitem__(self, item, value):
        if isinstance(item, GeneralSlice):
            if len(item.elements) > 3:
                raise IndexError("Lists take a maximum of 3 slice elements.")
            return super().__setitem__(slice(*item.elements), value)
        elif item is BLANK_SLICE:
            return super().__setitem__(slice(None), value)
        return super().__setitem__(item, value)
    
class TupleExtension(tuple):
    
    def get(self, index, default=None):
        try:
            return self[index]
        except IndexError:
            return default
    
    def __getitem__(self, item):
        if isinstance(item, GeneralSlice):
            if len(item.elements) > 3:
                raise IndexError("Tuples take a maximum of 3 slice elements.")
            return super().__getitem__(slice(*item.elements))
        elif item is BLANK_SLICE:
            return super().__getitem__(slice(None))
        return super().__getitem__(item)
    
    def __repr__(self):
        orig = super().__repr__()[1:-2 if len(self) == 1 else -1]
        return f'[<{orig}>]'

class DictExtension(dict):
    
    def __repr__(self):
        return '{->}' if not self else '{'+', '.join(
                f"{key} -> {value}" for key, value in self.items()
        ) + '}'

class ComplexExtension(complex):
    def __str__(self):
        rsgn = '' if (math.copysign(1, self.real)+1)//2  else '-'
        isgn = '+' if (math.copysign(1, self.imag)+1)//2 else '-'
        real, imag = self.real, self.imag
        real = abs(int(real)) if real.is_integer() else abs(real)
        imag = int(abs(imag)) if imag.is_integer() else abs(imag)
        return f"({rsgn}{real} {isgn} {imag}i)"

class Slice:
    pass

class Vector:
    
    def __init__(self, *args):
        self.dimension = len(args)
        self.__components__ = list(args)
    
    @property
    def components(self):
        return tuple(self.__components__)
    
    def __getitem__(self, item):
        if item is BLANK_SLICE:
            return copy.deepcopy(self)
        if isinstance(item, GeneralSlice):
            elements = list(item.elements)
            # support vec1 with vec[n,] instead of vec[n]
            if elements[-1] is None:
                del elements[-1]
            return self.__class__(
                *(self.__components__[int(i)] for i in elements)
            )
        return self.__components__[item]
    
    def __setitem__(self, item, value):
        if item is BLANK_SLICE:
            self.__components__[:] = value
            return
        if isinstance(item, GeneralSlice):
            elements = list(item.elements)
            # support vec1 with vec[n,] instead of vec[n]
            if elements[-1] is None:
                del elements[-1]
            if not isinstance(value, Sequence):
                raise IndexError("Multiple assignment has to have a sequence.")
            if len(elements) != len(value):
                raise IndexError("Sequence lengths do not match.")
            for e, v in zip(elements, value):
                self.__components__[int(e)] = v
            return
        self.__components__[item] = value
    
    def __getattr__(self, attr):
        charmap = {'x': 0, 'y': 1, 'z': 2, 'w': 3}
        if self.dimension == 1 and attr == 'x':
            return self[0]
        elif (
            self.dimension == 2 and all(c in 'xy' for c in attr) or
            self.dimension == 3 and all(c in 'xyz' for c in attr) or
            self.dimension == 4 and all(c in 'xyzw' for c in attr)
        ):
            if len(attr) == 1:
                return self.__components__[charmap[attr]]
            return self.__class__(*(
                self.__components__[charmap[c]]
                for c in attr
            ))
        raise AttributeError
    
    @recursive_repr()
    def __repr__(self):
        return (
            f"<Vec{self.dimension} with {self.components}>"
        )
    
    
    def _perform_binop(function, reverse=False):
        op = (lambda a, b: function(b, a)) if reverse else function
        def actual(self, other):
            if not isinstance(other, Vector):
                return self.__class__(
                    *(op(x, other) for x in self.__components__)
                )
            if self.dimension != other.dimension:
                raise ValueError("Incompatible dimensions.")
            return self.__class__(
                *(op(self[i], other[i]) for i in range(self.dimension))
            )
        return actual
    
    _perform_rbinop = partial(_perform_binop, reverse=True)
    
    __add__ = _perform_binop(operator.add)
    __radd__ = _perform_rbinop(operator.add)
    __sub__ = _perform_binop(operator.sub)
    __rsub__ = _perform_rbinop(operator.sub)
    __mul__ = _perform_binop(operator.mul)
    __rmul__ = _perform_rbinop(operator.mul)
    __floordiv__ = _perform_binop(operator.floordiv)
    __rfloordiv__ = _perform_rbinop(operator.floordiv)
    __truediv__ = _perform_binop(operator.truediv)
    __rtruediv__ = _perform_rbinop(operator.truediv)
    __pow__ = _perform_binop(operator.pow)
    __rpow__ = _perform_rbinop(operator.pow)
    
    
    def __neg__(self):
        return self.__class__(*(-x for x in self.__components__))
    def __pos__(self):
        return self.__class__(*(+x for x in self.__components__))
            

# slice that can take more than 3 elements, object[a,b,c,d,...]
class GeneralSlice(Slice):
    def __init__(self, element1, element2, *elements):
        self.elements = coerce((element1, element2, *elements))
    
    @recursive_repr()
    def __repr__(self) -> str:
        return f"<slice <{', '.join(map(str, self.elements))}>>"

# slice that has 0 elements, object[]
# for built in types, these act the same as
# object[,], but other objects may override
# this behavior
class BlankSlice(Slice):
    def __repr__(self) -> str:
        return f"<blank slice>"

BLANK_SLICE = BlankSlice()

# coerces a python object into the type used by the expression interpreter

@overload
def coerce(object: list) -> ListExtension: ...
@overload
def coerce(object: tuple) -> TupleExtension: ...
@overload
def coerce(object: dict) -> DictExtension: ...
@overload
def coerce(object: complex) -> ComplexExtension: ...
@overload
def coerce(object: slice) -> GeneralSlice: ...
@overload
def coerce[T](object: T) -> T: ...

def coerce(object):
    # not a bug, explicitly checking for exact type, no isinstance
    if type(object) == list:
        return ListExtension(object)
    elif type(object) == tuple:
        return TupleExtension(object)
    elif type(object) == dict:
        return DictExtension(object)
    elif type(object) == complex:
        return ComplexExtension(object)
    elif type(object) == slice:
        return GeneralSlice(object.start, object.stop, object.step)
    else:
        return object

# note: these functions do not need to define any of these as
#       the first argument like in python, the argument is implicit
#       'this' and 'cls' will probably become keywords in the furure
#       so you will not be able to name any arguments 'this' or 'cls'
class CallableType(Enum):
    # class call:    do not insert this and cls
    # instance call: do not insert this and cls
    # normal call:   do not insert this and cls
    NON_METHOD = 0
    BOUND_METHOD = 1
    BOUND_CLASS_METHOD = 2
    # normal call:   expect extra first parameter, insert it as this
    # class call:    insert cls, expect this as first parameter
    # instance call: insert this and cls
    UNBOUND_METHOD = 3
    # normal call:   expect extra first parameter, insert it as cls
    # class call:    insert cls
    # instance call: insert cls
    UNBOUND_CLASS_METHOD = 4

def classmethod(obj: CallableWrapper):
    if obj.__wrapped_callable_type__  != 'lambda function':
        raise ValueError("Can only convert a function.")
    return CallableWrapper(
        obj,
        name=obj.name,
        information=obj.information,
        method_type=CallableType.UNBOUND_CLASS_METHOD,
        callable_type='unbound lambda class method'
    )

def method(obj: CallableWrapper):
    if obj.__wrapped_callable_type__ != 'lambda function':
        raise ValueError("Can only convert a function.")
    return CallableWrapper(
        obj,
        name=obj.name,
        information=obj.information,
        method_type=CallableType.UNBOUND_METHOD,
        callable_type='unbound lambda method'
    )

class CallableObject:
    prefills: tuple
    postfills: tuple
    information: Optional[str]
    def append(self, arg) -> CallableObject:
        new = copy.copy(self)
        new.postfills += (arg,)
        return new
    def append_left(self, arg) -> CallableObject:
        new = copy.copy(self)
        new.postfills = (arg,) + new.postfills
        return new
    def prepend(self, arg) -> CallableObject:
        new = copy.copy(self)
        new.prefills = (arg,) + new.prefills
        return new
    def prepend_right(self, arg) -> CallableObject:
        new = copy.copy(self)
        new.prefills += (arg,)
        return new
    
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

class CallableWrapper(CallableObject):
    def __init__(
        self, call, prefills = (), postfills = (), *,
        name=None, information=None, callable_type='function',
        method_type=CallableType.NON_METHOD
    ):
        if not callable(call):
            raise TypeError(
                f"{type(call).__name__!r} object is not callable."
            )
        self.__wrapped_callable__ = call
        self.prefills = prefills
        self.postfills = postfills
        self.information = information
        self.name = name
        self.__wrapped_callable_type__ = callable_type
        self.__method_type__ = method_type

    def __call__(self, *args, **kwargs):
        args_iter = iter(args)
        try:
            args_pre = [
                arg if arg != ... else next(args_iter) for arg in self.prefills
            ]
        except StopIteration:
            raise ValueError("Too few arguments for prefills.")
        args_iter = reversed((*args_iter,))
        try:
            args_post = [
                arg if arg != ... else next(args_iter)
                for arg in reversed(self.postfills)
            ][::-1]
        except StopIteration:
            raise ValueError("Too few arguments for postfills.")
        args_mid = (*args_iter,)[::-1]
        
        wrapped_callable = self.__wrapped_callable__
        if self.__method_type__ in (
            CallableType.UNBOUND_METHOD, CallableType.UNBOUND_CLASS_METHOD
        ):
            # the wrapped callable is another CallableWrapper,
            # the underlying function, add a placeholder prefill to it
            # so that the first argument to the unbound method
            # gets put as the first argument to the actual callable
            wrapped_callable = cast(
                CallableWrapper, copy.copy(wrapped_callable)
            )
            wrapped_callable.prefills = (..., *wrapped_callable.prefills)
            # nothing to add, this will also prevent a weird edge case
            # of the wrapper having one too many prefills to its actual
            # callable, causing this to 'work' without being intended to
            if not (args_pre or args_mid or args_post):
                raise ValueError("Too few arguments.")
        # a non method is wrapped by a bound/unbound method, use the
        # a non method is wrapped by a bound/unbound method, use the
        # state of the wrapper, not the wrapped
        if self.__method_type__ is not CallableType.NON_METHOD:
            kwargs.update(callable_type=self.__method_type__)
        return wrapped_callable(
            *args_pre, *args_mid, *args_post, **kwargs
        )
            
    
    @recursive_repr()
    def __repr__(self):
        fn = (
            self.name or
            getattr(self.__wrapped_callable__, '__name__', 'unknown')
        )
        t = self.__wrapped_callable_type__
        
        prefills = ['{}' if p is ... else str(p) for p in self.prefills]
        postfills = ['{}' if p is ... else str(p) for p in self.postfills]
        args = (*prefills, '<arguments>', *postfills)
        if not self.information:
            return f"<{t} {fn} with call {fn}({', '.join(args)})>"
        else:
            return (
                f"<{t} {fn} with call {fn}({', '.join(args)}), "
                f"{self.information}>"
            )
    
    def __get__(self, obj, objtype=None):
        if obj is not None:
            objtype = type(obj)
        if (
            self.__method_type__ in (
                CallableType.NON_METHOD,
                CallableType.BOUND_METHOD,
                CallableType.BOUND_CLASS_METHOD
            ) or
            self.__method_type__ is CallableType.UNBOUND_METHOD and obj is None
        ):
            return self
        elif self.__method_type__ is CallableType.UNBOUND_METHOD:
            new = copy.copy(self)
            new.__wrapped_callable__ = partial(
                new.__wrapped_callable__, this=obj, cls=objtype
            )
            new.__method_type__ = CallableType.BOUND_METHOD
            new.__wrapped_callable_type__ = "bound lambda method"
            return new
        elif self.__method_type__ is CallableType.UNBOUND_CLASS_METHOD:
            new = copy.copy(self)
            new.__wrapped_callable__ = partial(
                new.__wrapped_callable__, cls=objtype
            )
            new.__wrapped_callable_type__ = "bound lambda class method"
            new.__method_type__ = CallableType.BOUND_CLASS_METHOD
            return new
