from typing import Generic, Union, TypeVar

T = TypeVar('T')


class Array(Generic[T]):
    def __getitem__(self, item) -> Union['Array', T]:
        pass

    def sum(self) -> T:
        pass

    def __sub__(self, right: 'Array[T]') -> 'Array[T]':
        pass

    def __add__(self, right: 'Array[T]') -> 'Array[T]':
        pass
