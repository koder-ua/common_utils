from __future__ import annotations

import abc
from typing import IO, List, Dict, Iterator, Union, Type, Iterable, TypeVar, Tuple, Any


from . import NumVector


class IStorable(metaclass=abc.ABCMeta):
    """Interface for type, which can be stored"""

    @abc.abstractmethod
    def raw(self) -> Dict[str, Any]:
        pass

    @classmethod
    def fromraw(cls, data: Dict[str, Any]) -> IStorable:
        pass


class Storable(IStorable):
    """Default implementation"""

    __ignore_fields__: List[str] = []

    def raw(self) -> Dict[str, Any]:
        return {name: val
                for name, val in self.__dict__.items()
                if not name.startswith("_") and name not in self.__ignore_fields__}

    @classmethod
    def fromraw(cls, data: Dict[str, Any]) -> IStorable:
        obj = cls.__new__(cls)
        if cls.__ignore_fields__:
            data = data.copy()
            data.update(dict.fromkeys(cls.__ignore_fields__))
        obj.__dict__.update(data)
        return obj


Basic = Union[int, str, bytes, bool, None]
StorableType = Union[IStorable, Dict[str, Any], List[Any], int, str, bytes, bool, None]


class ISerializer(metaclass=abc.ABCMeta):
    """Interface for serialization class"""
    @abc.abstractmethod
    def pack(self, value: StorableType) -> bytes:
        pass

    @abc.abstractmethod
    def unpack(self, data: bytes) -> Any:
        pass


class _Raise:
    pass


class ISimpleStorage(metaclass=abc.ABCMeta):
    """interface for low-level storage, which doesn't support serialization
    and can operate only on bytes"""

    @abc.abstractmethod
    def put(self, value: bytes, path: str) -> None:
        pass

    @abc.abstractmethod
    def get(self, path: str) -> bytes:
        pass

    @abc.abstractmethod
    def rm(self, path: str) -> None:
        pass

    @abc.abstractmethod
    def sync(self) -> None:
        pass

    @abc.abstractmethod
    def __contains__(self, path: str) -> bool:
        pass

    @abc.abstractmethod
    def get_fd(self, path: str, mode: str = "rb+") -> IO:
        pass

    @abc.abstractmethod
    def get_fname(self, path: str) -> str:
        pass

    @abc.abstractmethod
    def sub_storage(self, path: str) -> ISimpleStorage:
        pass

    @abc.abstractmethod
    def list(self, path: str) -> Iterator[Tuple[bool, str]]:
        pass


ObjClass = TypeVar('ObjClass', bound=IStorable)


class IStorage(metaclass=abc.ABCMeta):
    sstorage: ISimpleStorage
    serializer: ISerializer
    other_caches: Dict[str, Dict] = None

    @abc.abstractmethod
    def sub_storage(self, *path: str) -> IStorage:
        pass

    @abc.abstractmethod
    def put(self, value: Any, *path: str) -> None:
        pass

    @abc.abstractmethod
    def get(self, path: str, default: Any = _Raise) -> Any:
        pass

    @abc.abstractmethod
    def rm(self, *path: str) -> None:
        pass

    @abc.abstractmethod
    def load(self, obj_class: Type[ObjClass], *path: str) -> ObjClass:
        pass

    # ---------------  List of values ----------------------------------------------------------------------------------

    @abc.abstractmethod
    def put_list(self, value: Iterable[IStorable], *path: str) -> None:
        pass

    @abc.abstractmethod
    def load_list(self, obj_class: Type[ObjClass], *path: str) -> List[ObjClass]:
        pass

    @abc.abstractmethod
    def __contains__(self, path: str) -> bool:
        pass

    @abc.abstractmethod
    def put_raw(self, val: bytes, *path: str) -> str:
        pass

    @abc.abstractmethod
    def get_fname(self, fpath: str) -> str:
        pass

    @abc.abstractmethod
    def get_raw(self, *path: str) -> bytes:
        pass

    @abc.abstractmethod
    def append_raw(self, value: bytes, *path: str) -> None:
        pass

    @abc.abstractmethod
    def get_fd(self, path: str, mode: str = "r") -> IO:
        pass

    @abc.abstractmethod
    def sync(self) -> None:
        pass

    @abc.abstractmethod
    def flush(self) -> None:
        pass

    @abc.abstractmethod
    def __enter__(self) -> IStorage:
        pass

    @abc.abstractmethod
    def __exit__(self, x: Any, y: Any, z: Any) -> None:
        pass

    @abc.abstractmethod
    def list(self, *path: str) -> Iterator[Tuple[bool, str]]:
        pass

    @abc.abstractmethod
    def iter_paths(self, root: str, path_parts: List[str],
                   already_found_groups: Dict[str, str]) -> Iterator[Tuple[bool, str, Dict[str, str]]]:
        pass

    # --------------  Arrays -------------------------------------------------------------------------------------------

    @abc.abstractmethod
    def put_array(self, path: str,
                  data: NumVector,
                  header: List[str],
                  header2: NumVector = None,
                  append_on_exists: bool = False) -> None:
        pass

