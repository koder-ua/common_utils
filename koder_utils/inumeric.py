from __future__ import annotations

import array
import operator
from typing import Generic, Union, TypeVar, List, Callable

try:
    import numpy
except ImportError:
    numpy = None


Number = Union[int, float]


if numpy:
    NumVector = Union[numpy.ndarray, array.array, List[int], List[float]]
    NumVector1d = NumVector
    NumVector2d = Union[numpy.ndarray, array.array]
    Numpy1d = numpy.ndarray
    Numpy2d = numpy.ndarray
else:
    NumVector = Union[array.array, List[int], List[float]]
    NumVector1d = NumVector
    NumVector2d = array.array


T = TypeVar('T')


class Array(Generic[T]):
    def __getitem__(self, item) -> Union[Array[T], T]:
        pass

    def sum(self) -> T:
        pass

    def __sub__(self, right: Array[T]) -> Array[T]:
        pass

    def __add__(self, right: Array[T]) -> Array[T]:
        pass

    def max(self) -> T:
        pass

    def min(self) -> T:
        pass


V = TypeVar('V', float, int)


class ArithmeticMixin(Generic[V]):
    def _combine_binop(self: T, other: T, op: Callable[[V, V], V]) -> T:
        res = {}
        for name, vl in self.__dict__.items():
            other_vl = getattr(other, name)
            res[name] = None if vl is None or other_vl is None else op(vl, other_vl)
        return self.__class__(**res)

    def _combine_with_self_binop(self: T, other: T, op: Callable[[V, V], V]) -> T:
        for name, vl in self.__dict__.items():
            other_vl = getattr(other, name)
            self.__dict__[name] = None if vl is None or other_vl is None else op(vl, other_vl)
        return self

    def _combine_uninop(self: T, coef: Number, op: Callable[[V, Number], V]) -> T:
        res = {name: (None if vl is None else op(vl, coef)) for name, vl in self.__dict__.items()}
        return self.__class__(**res)

    def _combine_with_self_uniop(self: T, coef: Number, op: Callable[[V, Number], V]) -> T:
        for name, vl in self.__dict__.items():
            if vl is not None:
                self.__dict__[name] = op(vl, coef)
        return self

    def __add__(self: T, other: T) -> T:
        return self._combine_binop(other, operator.add)

    def __sub__(self: T, other: T) -> T:
        return self._combine_binop(other, operator.sub)

    def __iadd__(self: T, other: T) -> T:
        return self._combine_with_self_binop(other, operator.add)

    def __isub__(self: T, other: T) -> T:
        return self._combine_with_self_binop(other, operator.sub)

    def __div__(self: T, divider: float) -> T:
        return self._combine_with_self_uniop(divider, operator.truediv)

    def __idiv__(self: T, divider: float) -> T:
        return self._combine_with_self_uniop(divider, operator.truediv)


class IntArithmeticMixin(ArithmeticMixin[int]):
    def __div__(self: T, divider: float) -> T:
        return self._combine_with_self_uniop(divider, lambda x, y: int(x / y))

    def __idiv__(self: T, divider: float) -> T:
        return self._combine_with_self_uniop(divider, lambda x, y: int(x / y))


class FloatArithmeticMixin(ArithmeticMixin[float]):
    pass