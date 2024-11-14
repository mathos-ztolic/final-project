from __future__ import annotations
from collections.abc import Container, Sequence
from typing import Optional, overload, Self, Any, NoReturn
import itertools
from custom_types import coerce, GeneralSlice, BLANK_SLICE, BlankSlice
import math

# set of all objects that pass a test
class SetOf:
    def __init__(self, function, description=None):
        self.function = function
        self.description = (
            description or getattr(function, '__name__', 'unkown contents')
        )
    def __contains__(self, elem):
        return bool(self.function(elem))
    def __repr__(self):
        return f"Set({self.description})"

def Nullable(s):
    return SetUnion([None], s)

class EmptySet:
    def __contains__(self, elem):
        return False
    def __repr__(self):
        return '∅'
    def __call__(self, item) -> NoReturn:
        raise ValueError(f"{item} ∉ {self}")

class UniversalSet:
    def __contains__(self, elem):
        return True
    def __repr__(self):
        return 'ξ'
    def __call__(self, item) -> Any:
        return item


class Sets:
    def __contains__(self, elem):
        return isinstance(elem, Container)
    def __repr__(self):
        return 'Σ'

class Integers:
    
    def __contains__(self, elem):
        if not isinstance(elem, (bool, int, float, complex)):
            return False
        return elem.imag == 0 and float(elem.real).is_integer()
    
    @overload
    def __getitem__(self, item: BlankSlice) -> Self: ...
    @overload
    def __getitem__(self, item: slice | GeneralSlice) -> SlicedIntegers: ...
    @overload
    def __getitem__(self, item: Any) -> NoReturn: ...
    
    def __getitem__(self, item):
        if item is BLANK_SLICE:
            return self
        if isinstance(item, slice):
            return self.__getitem__(
                GeneralSlice(item.start, item.stop, item.step)
            )
        if isinstance(item, GeneralSlice):
            return SlicedIntegers(item)

        raise IndexError(f"Only slicing is supported.")
    
    def __call__(self, item: Any) -> int:
        if item not in self:
            raise ValueError(f"{item} ∉ {self}")
        return int(item.real)

    def __repr__(self):
        return 'ℤ'


class SlicedIntegers:
    def __init__(self, slice):
        start = slice.elements.get(0)
        stop = slice.elements.get(1)
        divisible_by = slice.elements.get(2)
        not_divisible_by = slice.elements.get(3)
        if len(slice.elements) > 4:
            raise ValueError("Integer slices take a maximum of 4 elements.")
        if (
            start not in Nullable(INTEGERS) or
            stop not in Nullable(INTEGERS) or
            divisible_by not in Nullable(
                SetUnion(Star(INTEGERS), INTEGERS)
            ) or
            not_divisible_by not in Nullable(
                SetUnion(Star(INTEGERS), INTEGERS)
            )
        ):
            raise ValueError(
                "Bounds have to be integers/blank. (Not) "
                "divisible by has to be an integer/list of integers/blank. "
            )
        self.start = start
        self.stop = stop
        self.divisible_by = set(
            [] if divisible_by is None else
            divisible_by if isinstance(divisible_by, Sequence) else
            [divisible_by]
        )
        self.not_divisible_by = set(
            [] if not_divisible_by is None else
            not_divisible_by if isinstance(not_divisible_by, Sequence) else
            [not_divisible_by]
        )

    def __contains__(self, e):
        if e not in INTEGERS:
            return False
        start = self.start if self.start is not None else -float('inf')
        stop = self.stop if self.stop is not None else float('inf')
        if e < start or e > stop:
            return False
        return (
            all(n != 0 and e % n == 0 for n in self.divisible_by) and
            all(n == 0 or e % n != 0 for n in self.not_divisible_by)
        )
        
    @overload
    def __getitem__(self, item: BlankSlice) -> Self: ...
    @overload
    def __getitem__(self, item: slice | GeneralSlice) -> Self: ...
    @overload
    def __getitem__(self, item: Any) -> NoReturn: ...
    
    def __getitem__(self, item):
        if item is BLANK_SLICE:
            return self
        if isinstance(item, slice):
            return self.__getitem__(
                GeneralSlice(item.start, item.stop, item.step)
            )
        if isinstance(item, GeneralSlice):
            start = (
                item.elements.get(0) if self.start is None else
                self.start if item.elements.get(0) is None else
                max(self.start, item.elements[0])
            )
            stop = (
                item.elements.get(1) if self.stop is None else
                self.stop if item.elements.get(1) is None else
                min(self.stop, item.elements[1])
            )
            divisible_by = list(self.divisible_by.union(
                [] if item.elements.get(2) is None else
                item.elements[2] if isinstance(item.elements[2], Sequence) else
                [item.elements[2]]
            ))
            not_divisible_by = list(self.not_divisible_by.union(
                [] if item.elements.get(3) is None else
                item.elements[3] if isinstance(item.elements[3], Sequence) else
                [item.elements[3]]
            ))
            return SlicedIntegers(
                GeneralSlice(start, stop, divisible_by, not_divisible_by)
            )
        raise IndexError(f"Only slicing is supported.")
    
    def __call__(self, item: Any) -> int:
        if item not in self:
            raise ValueError(f"{item} ∉ {self}")
        return int(item.real)
    
    def __repr__(self):
        divisible_by = [str(x) for x  in sorted(self.divisible_by)]
        not_divisible_by = [str(x) for x in sorted(self.not_divisible_by)]
        greater_than = f"{self.start}≤" if self.start is not None else ''
        less_than = f"≤{self.stop}" if self.stop is not None else ''
        conditions = []
        if less_than or greater_than:
            conditions.append(f"{greater_than}n{less_than}")
        if len(divisible_by) == 1:
            conditions.append(f"n divides {divisible_by[0]}")
        elif divisible_by:
            conditions.append(
                f"n divides k ∀k∈{{{', '.join(divisible_by)}}}"
            )
        if len(not_divisible_by) == 1:
            conditions.append(f"n does not divide {not_divisible_by[0]}")
        elif not_divisible_by:
            conditions.append(
                f"k does not divide n ∀k∈{{{', '.join(not_divisible_by)}}}"
            )
        if not conditions:
            return 'ℤ'
        if len(conditions) > 1:
            conditions = [f"({c})" for c in conditions]
        return f"{{n∈ℤ | {' ∧ '.join(conditions)}}}"


class Reals:
    def __contains__(self, elem):
        if not isinstance(elem, (bool, int, float, complex)):
            return False
        return elem.imag == 0

    @overload
    def __getitem__(self, item: BlankSlice) -> Self: ...
    @overload
    def __getitem__(self, item: slice | GeneralSlice) -> SlicedReals: ...
    @overload
    def __getitem__(self, item: Any) -> NoReturn: ...

    def __getitem__(self, item):
        if item is BLANK_SLICE:
            return self
        if isinstance(item, slice):
            if item.step is not None:
                return self.__getitem__(
                    GeneralSlice(item.start, item.stop, item.step)
                )
            return self.__getitem__(GeneralSlice(item.start, item.stop))
        if isinstance(item, GeneralSlice):
            return SlicedReals(item)
        raise IndexError(f"Only slicing is supported.")
    
    def __call__(self, item: Any) -> float:
        if item not in self:
            raise ValueError(f"{item} ∉ {self}")
        return float(item.real)

    def __repr__(self):
        return 'ℝ'


class SlicedReals:
    start: Optional[int]
    stop: Optional[int]
    def __init__(self, slice):
        if len(slice.elements) > 2:
            raise ValueError(
                "Real slices have a maximum of 2 elements."
            )
        start = slice.elements.get(0)
        stop = slice.elements.get(1)
        if start not in Nullable(REALS) or stop not in Nullable(REALS):
            raise ValueError("Bounds are not real numbers.")
        self.start = start
        self.stop = stop

    def __contains__(self, e):
        if e not in REALS:
            return False
        start = self.start if self.start is not None else -float('inf')
        stop = self.stop if self.stop is not None else float('inf')
        return start <= e <= stop
    
    @overload
    def __getitem__(self, item: BlankSlice) -> Self: ...
    @overload
    def __getitem__(self, item: slice | GeneralSlice) -> Self: ...
    @overload
    def __getitem__(self, item: Any) -> NoReturn: ...
    
    def __getitem__(self, item):
        if item is BLANK_SLICE:
            return self
        if isinstance(item, slice):
            if item.step is not None:
                return self.__getitem__(
                    GeneralSlice(item.start, item.stop, item.step)
                )
            return self.__getitem__(GeneralSlice(item.start, item.stop))
        if isinstance(item, GeneralSlice):
            start = (
                item.elements.get(0) if self.start is None else
                self.start if item.elements.get(0) is None else
                max(self.start, item.elements[0])
            )
            stop = (
                item.elements.get(1) if self.stop is None else
                self.stop if item.elements.get(1) is None else
                min(self.stop, item.elements[1])
            )
            return SlicedReals(GeneralSlice(start, stop))
        raise IndexError(f"Only slicing is supported.")
    
    def __call__(self, item: Any) -> float:
        if item not in self:
            raise ValueError(f"{item} ∉ {self}")
        return float(item.real)
    
    def __repr__(self):
        greater_than = f"{self.start}≤" if self.start is not None else ''
        less_than = f"≤{self.stop}" if self.stop is not None else ''
        if less_than or greater_than:
            return f"{{x∈ℝ | {greater_than}x{less_than}}}"
        return 'ℝ'
    

class Complex:
    def __contains__(self, e):
        return isinstance(e, (bool, int, float, complex))
    def __repr__(self):
        return 'ℂ'

    @overload
    def __getitem__(self, item: BlankSlice) -> Self: ...
    @overload
    def __getitem__(self, item: slice | GeneralSlice) -> NoReturn: ...
    @overload
    def __getitem__(self, item: Any) -> NoReturn: ...
    
    def __call__(self, item: Any) -> complex:
        if item not in self:
            raise ValueError(f"{item} ∉ {self}")
        return complex(item)

    def __getitem__(self, item):
        if item is BLANK_SLICE:
            return self
        raise ValueError("Only blank slices are supported for now.")
        
class Functions:
    def __contains__(self, e):
        return callable(e)
    def __repr__(self):
        return 'Φ'
    def __call__[T](self, item: T) -> T:
        if not callable(item):
            raise ValueError(f"{item} ∉ {self}")
        return item

class SetUnion:
    
    sets: tuple[Container, ...]
    
    def __init__(self, set1: Container, set2: Container):
        if (
            isinstance(set1, SetUnion) and
            isinstance(set2, SetUnion)
        ):
            self.sets = (*set1.sets, *set2.sets)
        elif isinstance(set1, SetUnion):
            self.sets = (*set1.sets, set2)
        elif isinstance(set2, SetUnion):
            self.sets = (set1, *set2.sets)
        else:
            self.sets = (set1, set2)

    def __contains__(self, e):
        for s in self.sets:
            if e in s:
                return True
        return False

    def __call__[T](self, item: T) -> T:
        for s in self.sets:
            if item in s:
                return s(item)
        raise ValueError(f"{item} ∉ {self}")

    def __repr__(self):
        return f"({' ∪ '.join(str(s) for s in self.sets)})"

class SetIntersection:
    
    sets: tuple[Container, ...]
    
    def __init__(self, set1: Container, set2: Container):
        if (
            isinstance(set1, SetIntersection) and
            isinstance(set2, SetIntersection)
        ):
            self.sets = (*set1.sets, *set2.sets)
        elif isinstance(set1, SetIntersection):
            self.sets = (*set1.sets, set2)
        elif isinstance(set2, SetIntersection):
            self.sets = (set1, *set2.sets)
        else:
            self.sets = (set1, set2)
    def __contains__(self, e):
        for s in self.sets:
            if e not in s:
                return False
        return True
    
    def __call__[T](self, item: T) -> T:
        if item not in self:
            raise ValueError(f"{item} ∉ {self}")
        return self.sets[0](item)

    def __repr__(self):
        return f"({' ∩ '.join(str(s) for s in self.sets)})"

class SetDifference:
    
    sets: tuple[Container, ...]
    
    def __init__(self, set1: Container, set2: Container):
        if isinstance(set1, SetDifference):
            self.sets = (*set1.sets, set2)
        else:
            self.sets = (set1, set2)
    def __contains__(self, e):
        if e not in self.sets[0]:
            return False
        for s in self.sets[1:]:
            if e in s:
                return False
        return True
    
    def __call__[T](self, item: T) -> T:
        if item not in self:
            raise ValueError(f"{item} ∉ {self}")
        return self.sets[0](item)

    def __repr__(self):
        return '(' + ' \\ '.join(str(s) for s in self.sets) + ')'

class SetSymmetricDifference:
    
    sets: tuple[Container, ...]
    
    def __init__(self, set1: Container, set2: Container):
        if (
            isinstance(set1, SetSymmetricDifference) and
            isinstance(set2, SetSymmetricDifference)
        ):
            self.sets = (*set1.sets, *set2.sets)
        elif isinstance(set1, SetSymmetricDifference):
            self.sets = (*set1.sets, set2)
        elif isinstance(set2, SetSymmetricDifference):
            self.sets = (set1, *set2.sets)
        else:
            self.sets = (set1, set2)

    def __contains__(self, e):
        e_count = 0
        for s in self.sets:
            if e in s:
                e_count += 1
        return e_count % 2 == 1
    
    def __call__[T](self, item: T) -> T:
        first = None
        e_count = 0
        for s in self.sets:
            if e in s:
                e_count += 1
                if first is None:
                    first = s
        if e_count % 2 == 1:
            assert first is not None
            return first(item)
        raise ValueError(f"{item} ∉ {self}")

    def __repr__(self):
        return f"({' ⊕ '.join(str(s) for s in self.sets)})"
    
    
class Pack:
    
    unpacked: Exactly | _VariableCartesianProduct
    
    def __init__(self, arg):
        self.unpacked = arg
        if not isinstance(arg, (Exactly, _VariableCartesianProduct)):
            raise ValueError(
                "Can only pack Exactly or variable Cartesian products."
            )
    def __contains__(self, e):
        return e in self.unpacked
    def __repr__(self):
        if isinstance(self.unpacked, Exactly):
            return repr(self.unpacked)
        else:
            return f"({self.unpacked!r})"

class UnpackCartesianProduct:
    
    arg: SetCartesianProduct
    
    def __init__(self, arg):
        if not isinstance(arg, SetCartesianProduct):
            raise ValueError(
                "Can only unpack normal Cartesian products/packs."
            )
        self.arg = arg

def Unpack(arg: Pack | SetCartesianProduct):
    if isinstance(arg, Pack):
        return arg.unpacked
    return UnpackCartesianProduct(arg)

class SetCartesianProduct:
    
    sets: tuple[Container, ...]
    var_index: Optional[int]
    
    def __init__(self, *product_sets: Container):
        sets: list[Container] = []
        self.var_index = None
        for s in product_sets:
            if (
                isinstance(s, _VariableCartesianProduct) and
                self.var_index is None
            ):
                self.var_index = len(sets)
                sets.append(s)
            elif isinstance(s, _VariableCartesianProduct):
                raise ValueError("Multiple variable Cartesian products.")
            elif isinstance(s, Exactly):
                for _, s2 in zip(range(s.n), itertools.cycle(s.sets)):
                    sets.append(s2)
            elif isinstance(s, UnpackCartesianProduct):
                if not isinstance(s.arg, SetCartesianProduct):
                    raise ValueError(
                        "Can only unpack normal Cartesian products."
                    )
                for s2 in s.arg.sets:
                    if (
                        isinstance(s2, _VariableCartesianProduct) and
                        self.var_index is None
                    ):
                        self.var_index = len(sets)
                        sets.append(s2)
                    elif isinstance(s2, _VariableCartesianProduct):
                        raise ValueError(
                            "Multiple variable Cartesian products."
                        )
                    elif isinstance(s2, Exactly):
                        for _, s3 in zip(
                            range(s2.n), itertools.cycle(s2.sets)
                        ):
                            sets.append(s3)
                    else:
                        sets.append(s2)
            else:
                sets.append(s)
        self.sets = tuple(sets)

    def __contains__(self, e):
        if not isinstance(e, list):
            return False
        if self.var_index is None:
            if len(e) != len(self.sets):
                return False
            for x, s in zip(e, self.sets):
                if x not in s:
                    return False
            return True
        else:
            if len(e) < len(self.sets) - 1:
                return False
            va_count = len(e) - len(self.sets) + 1
            for x, s in zip(e[:self.var_index], self.sets):
                if x not in s:
                    return False
            if (
                e[self.var_index:self.var_index+va_count]
                not in
                self.sets[self.var_index]
            ):
                return False
            for x, s in zip(
                e[self.var_index+va_count:], self.sets[self.var_index+1:]
            ):
                if x not in s:
                    return False
            return True
    
    def __repr__(self):
        return f"({' × '.join(str(x) for x in self.sets)})"

class _VariableCartesianProduct:
    _at_least: float
    _at_most: float
    sets: tuple[Container, ...]
    def __contains__(self, e):
        if not isinstance(e, list):
            return False
        if len(e) > self._at_most or len(e) < self._at_least:
            return False
        for x, s in zip(e, itertools.cycle(self.sets)):
            if x not in s:
                return False
        return True

class Exactly:
    n: int
    sets: tuple[Container, ...]
    def __init__(self, n, *sets: Container):
        self.n = n
        self.sets = sets
    def __contains__(self, e):
        if not isinstance(e, list):
            return False
        if len(e) != self.n:
            return False
        for x, s in zip(e, itertools.cycle(self.sets)):
            if x not in s:
                return False
        return True
    def __repr__(self):
        result = ' × '.join(str(x) for _, x in zip(
            range(self.n), itertools.cycle(self.sets)
        ))
        return f"({result})"

class Star(_VariableCartesianProduct):
    def __init__(self, *sets: Container):
        self.sets = sets
        self._at_least = 0
        self._at_most = float('inf')
    def __repr__(self):
        if len(self.sets) == 1:
            return f"{' × '.join(str(x) for x in self.sets)}*"
        else:
            return f"({' × '.join(str(x) for x in self.sets)})*"

class Plus(_VariableCartesianProduct):
    _at_least = 1
    _at_most = float('inf')
    def __init__(self, *sets: Container):
        self.sets = sets
    def __repr__(self):
        if len(self.sets) == 1:
            return f"{' × '.join(str(x) for x in self.sets)}+"
        else:
            return f"({' × '.join(str(x) for x in self.sets)})+"

class QuestionMark(_VariableCartesianProduct):
    _at_least = 0
    _at_most = 1
    def __init__(self, set: Container):
        self.sets = (set,)
    def __repr__(self):
        return f"{self.sets[0]}?"

class AtLeast(_VariableCartesianProduct):
    _at_most = float('inf')
    def __init__(self, at_least: int, *sets: Container):
        self.sets = sets
        self._at_least = at_least
    def __repr__(self):
        result = ' × '.join(str(x) for x in self.sets)
        if len(self.sets) == 1:
            return f"{result}{{{self._at_least},}}"
        else:
            return f"({result}){{{self._at_least},}}"

class AtMost(_VariableCartesianProduct):
    _at_least = 0
    def __init__(self, at_most: int, *sets: Container):
        self.sets = sets
        self._at_most = at_most
    def __repr__(self):
        result = ' × '.join(str(x) for x in self.sets)
        if len(self.sets) == 1:
            return f"{result}{{,{self._at_most}}}"
        else:
            return f"({result}){{,{self._at_most}}}"

class Between(_VariableCartesianProduct):
    def __init__(self, at_least: int, at_most: int, *sets: Container):
        self.sets = sets
        self._at_least = at_least
        self._at_most = at_most
    def __repr__(self):
        result = ' × '.join(str(x) for x in self.sets)
        if len(self.sets) == 1:
            return f"{result}{{{self._at_least},{self._at_most}}}"
        else:
            return f"({result}){{{self._at_least},{self._at_most}}}"



COMPLEX = Complex()
FUNCTIONS = Functions()
REALS = Reals()
POSITIVE_REALS = REALS[math.nextafter(0, math.inf):]
NEGATIVE_REALS = REALS[:math.nextafter(0, -math.inf)]
NONNEGATIVE_REALS = REALS[0:]
NONPOSITIVE_REALS = REALS[:0]
INTEGERS = Integers()
NATURALS = INTEGERS[1:]
NATURALS_WITH_ZERO = INTEGERS[0:]
UNIVERSAL_SET = UniversalSet()
EMPTY_SET = EmptySet()
SETS = Sets()

