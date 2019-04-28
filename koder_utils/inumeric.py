from __future__ import annotations

import array
from typing import Generic, Union, TypeVar, List

T = TypeVar('T')

try:
    import numpy
    NumVector = Union[numpy.ndarray, array.array, List[int], List[float]]
except ImportError:
    numpy = None
    NumVector = Union[array.array, List[int], List[float]]


class Array(Generic[T]):
    def __getitem__(self, item) -> Union[Array[T], T]:
        pass

    def sum(self) -> T:
        pass

    def __sub__(self, right: Array[T]) -> Array[T]:
        pass

    def __add__(self, right: Array[T]) -> Array[T]:
        pass
