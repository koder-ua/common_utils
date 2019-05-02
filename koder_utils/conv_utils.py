from __future__ import annotations

import copy
import inspect
import json
import threading
from dataclasses import Field, field as dataclass_field
from enum import Enum, IntEnum
from pathlib import Path
from typing import Callable, Type, Dict, Any, TypeVar, Union, List, Tuple, Optional, Set, get_type_hints


T = TypeVar('T')


class DeserializationError(ValueError):
    def __init__(self, cls: type, field_name: str, message: str, jsname: str) -> None:
        self.structs_stack: List[Tuple[str, str, Optional[str]]] = [(cls.__name__, field_name, jsname)]
        self.message = message

    def push(self, cls: Type, field_name: str, jsname: str) -> None:
        self.structs_stack.append((cls.__name__, field_name, jsname))

    def __str__(self) -> str:
        stack = []
        for cls_name, field_name, jsfield in self.structs_stack:
            stack.append(f"{cls_name}::{field_name}" + (f"(json name '{jsfield}')" if jsfield != field_name else ""))

        if len(stack) > 1:
            str_stack = "\n    ".join(stack)
            return f"During deserialization of\n    {str_stack}\n{self.message}"
        else:
            return f"During deserialization of {stack[0]} : {self.message}"


def strict_converter(t: Type[T]) -> Callable[[T], T]:
    """
    Only check that incoming value has type t and return value as is
    """
    assert t in {int, str, bool, float}

    def closure(v: T) -> T:
        # isinstance have unacceptable behavior for bool
        if type(v) is not t:
            raise ValueError(f"Expected value of type {t.__name__}, get {v!r} with type {type(v).__name__}")
        return v

    return closure


# Trying to convert incoming value into target type, using python default conversion
_CONVERTERS: Dict[Type, Callable[[Any], Any]] = {
    int: int,
    str: str,
    float: float,
    bool: strict_converter(bool),
    Any: lambda x: x,
}


# Only check that incoming value has type t and return value as is
_STRICT_CONVERTERS: Dict[Type, Callable[[Any], Any]] = {
    int: strict_converter(int),
    str: strict_converter(str),
    float: strict_converter(float),
    bool: strict_converter(bool),
}


def converter_from_dict(t: Type[T]) -> Callable[[Any], T]:
    """
    Decorator to register converters for class T
    """
    def closure(func: Callable[[Any], T]) -> Callable[[Any], T]:
        assert t not in _CONVERTERS
        _CONVERTERS[t] = func
        return func
    return closure


def check_types(v: Any, *types: Any):
    if not isinstance(v, types):
        raise TypeError(f"Expected one of: {','.join(t.__name__ for t in types)}, get {type(v).__name__}")


# containers converters factories

def convert_dict(t, strict: bool) -> Callable[[Dict], Any]:
    key_tp, val_tp = t.__args__
    k_conv = get_converter(key_tp, strict)
    v_conv = get_converter(val_tp, strict)

    def convert(v: Any) -> T:
        check_types(v, dict)
        return {k_conv(k): v_conv(v) for k, v in v.items()}

    return convert


def convert_list(t, strict: bool) -> Callable[[List], Any]:
    it_conv = get_converter(t.__args__[0], strict)

    def convert(v: List) -> T:
        check_types(v, list)
        return [it_conv(item) for item in v]

    return convert


def convert_tuple(t, strict: bool) -> Callable[[List], Any]:
    raise NotImplemented()


def convert_set(t, strict: bool) -> Callable[[Any], Any]:
    it_conv = get_converter(t.__args__[0], strict)

    def convert(v: Any) -> T:
        check_types(v, list, set, tuple, frozenset)
        return {it_conv(item) for item in v}

    return convert


def convert_union(t, strict: bool) -> Callable[[Any], Any]:
    if len(t.__args__) == 2 and type(None) in t.__args__:
        other = list(t.__args__)
        other.remove(type(None))
        conv = get_converter(other[0], strict)

        def convert(v: Any) -> T:
            return None if v is None else conv(v)
    else:
        convs = [get_converter(tp, strict) for tp in t.__args__]

        def convert(v: Any) -> T:
            val = _NotAllowed
            for conv in convs:
                val2 = _NotAllowed
                try:
                    val2 = conv(v)
                except (TypeError, ValueError, AssertionError):
                    pass

                if val2 is not _NotAllowed:
                    if val is not _NotAllowed:
                        raise ValueError("Can't reliable decode union field - two or more inner types match")
                    val = val2

            if val is _NotAllowed:
                raise ValueError(f"Can't reliable decode union field - none of types match")

            return val

    return convert


_CONVERTERS_FACTORYS: Dict[Any, Callable[[Any, bool], Callable[[Any], Any]]] = {
    dict: convert_dict,
    list: convert_list,
    set: convert_set,
    Union: convert_union,
    tuple: convert_tuple
}


def get_converter_no_cache(t: Type[T], strict: bool) -> Callable[[Any], T]:
    """
    Return converter from type T
    """
    if hasattr(t, 'from_dict'):
        return t.from_dict

    cmap = _STRICT_CONVERTERS if strict else _CONVERTERS

    if t in cmap:
        return cmap[t]

    if hasattr(t, '__origin__') and t.__origin__ in _CONVERTERS_FACTORYS:
        return _CONVERTERS_FACTORYS[t.__origin__](t, strict)

    if inspect.isclass(t):
        if issubclass(t, Enum):
            return t.__getitem__
        elif issubclass(t, IntEnum):
            return t

    raise TypeError(f"Can't find converter for type {t.__name__}")


_CACHE: Dict[Tuple[Any, bool], Callable[[Any], Any]] = {}


def get_converter(t: Type[T], strict: bool) -> Callable[[Any], T]:
    if (t, strict) not in _CACHE:
        _CACHE[(t, strict)] = get_converter_no_cache(t, strict)
    return _CACHE[(t, strict)]


@converter_from_dict(Path)
def path_from_json(v: Any) -> bool:
    assert isinstance(v, str)
    return Path(v)


noauto_key = 'conv::noauto'
converter_key = 'conv::converter'
name_key = 'conv::key'
default_key = 'conv::default'
default_factory_key = 'conv::default_factory_attr'
inline_key = 'conv::inline'
strict_key = 'conv::strict'


class _NotAllowed:
    pass


def field(*, noauto: bool = False,
          converter: Callable[[Any], Any] = None,
          key: str = None,
          default: Any = _NotAllowed,
          allow_shared_default: bool = False,
          default_factory: Callable[[], Any] = _NotAllowed,
          inline: bool = False,
          strict: bool = False,
          **params) -> Field:

    metadata = params.pop("metadata", {})
    if noauto:
        assert noauto_key not in metadata
        metadata[noauto_key] = None
        if default is _NotAllowed and default_factory is _NotAllowed:
            params['default'] = None

        if converter or key or inline or default is not _NotAllowed or strict:
            raise ValueError("Can't combine 'noauto' with any other parameter")

    if converter:
        assert converter_key not in metadata
        metadata[converter_key] = converter

    if key:
        assert name_key not in metadata
        metadata[name_key] = key

    if inline:
        assert inline_key not in metadata
        metadata[inline_key] = None

    if default is not _NotAllowed:
        assert default_factory is _NotAllowed, "Can't have default and default_factory_attr set and the same time"
        assert default_key not in metadata
        if not isinstance(default, (str, bool, int, float, bytes, type(None), tuple, frozenset)):
            if isinstance(default, (list, set, dict)):
                default = copy.copy(default)
            else:
                if not allow_shared_default:
                    raise ValueError(f"Can't set default value to instance of mutable type {type(default).__name__}" +
                                     " pass allow_shared_default=True to disable this check")

        metadata[default_key] = default

    if default_factory is not _NotAllowed:
        assert default_factory_key not in metadata
        metadata[default_factory_key] = default

    if strict:
        assert strict_key not in metadata
        metadata[strict_key] = None

    return dataclass_field(**params, metadata=metadata)


class ConvBase:
    # this fields would be set after __init_subclass__ call
    __conv_inited__: bool
    __conv_init_lock__: threading.Lock

    # this fields will be set during first call to from_dict/from_json
    __conv_converters__: Dict[str, Callable[[Any], Any]]
    __conv_mapping__: Dict[str, str]
    __conv_default__: Dict[str, Any]
    __conv_default_factory__: Dict[str, Callable[[], Any]]
    __conv_inline__: Set[set]
    __conv_clear__: bool

    def __init__(self, **params: Any) -> None:
        self.__dict__.update(params)

    @classmethod
    def __init_subclass__(cls: T) -> T:
        return convertable(cls)

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        return cls(**dict_from_json(cls, data))

    @classmethod
    def from_json(cls: Type[T], data: str) -> T:
        return cls.from_dict(json.loads(data))


def dict_from_json(cls: Type[T], data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Actually convert dict required format
    """
    if not cls.__conv_inited__:
        do_init_class(cls)

    # fast path
    if cls.__conv_clear__:
        try:
            return {name: conv(data[name]) for name, conv in cls.__conv_converters__.items()}
        except (ValueError, TypeError, AssertionError, DeserializationError, KeyError):
            # go to long path to get correct exception
            pass

    res: Dict[str, Any] = {}

    for pcls in cls.__mro__[1:-1:-1]:
        if hasattr(pcls, '__conv_mapping__'):
            res.update(dict_from_json(pcls, data))

    mp = cls.__conv_mapping__
    default = cls.__conv_default__
    default_factory = cls.__conv_default_factory__
    inline = cls.__conv_inline__

    localy_raised = False
    for name, conv in cls.__conv_converters__.items():
        jsname = mp[name]
        try:
            if name in inline:
                res[name] = conv(data)
            else:
                v = data.get(jsname, _NotAllowed)
                if v is _NotAllowed:
                    if name in default:
                        res[name] = default[name]
                    elif name in default_factory:
                        res[name] = default_factory[name]()
                    else:
                        msg = f"Input js dict has no key '{jsname}'. Only fields {','.join(data)} present."
                        localy_raised = True
                        raise DeserializationError(cls, name, msg, jsname)
                else:
                    res[name] = conv(v)
        except DeserializationError as exc:
            if not localy_raised:
                exc.push(cls, name, jsname)
            raise
        except (ValueError, TypeError, AssertionError) as exc:
            raise DeserializationError(cls, name, str(exc), jsname) from exc
    return res


def do_init_class(cls):
    with cls.__conv_init_lock__:
        if cls.__conv_inited__:
            return

        converters: Dict[str, Callable[[Any], Any]] = {}
        mapping: Dict[str, str] = {}
        default: Dict[str, Any] = {}
        default_factory: Dict[str, Callable[[], Any]] = {}
        inline: Set[str] = set()

        # if dataclass decorator already applied
        if hasattr(cls, "__dataclass_fields__"):
            fields_dct = cls.__dataclass_fields__
        else:
            fields_dct = cls.__dict__

        annotation = get_type_hints(cls, localns=None, globalns=None)
        for name, tp in annotation.items():
            if name.startswith("_"):
                continue
            converter = None
            v = fields_dct.get(name)
            strict = False
            if isinstance(v, Field):
                if noauto_key in v.metadata:
                    continue

                if converter_key in v.metadata:
                    converter = v.metadata[converter_key]

                if default_key in v.metadata:
                    default[name] = v.metadata[default_key]

                if default_factory_key in v.metadata:
                    default_factory[name] = v.metadata[default_factory_key]

                if inline_key in v.metadata:
                    inline.add(name)

                if strict_key in v.metadata:
                    strict = True

                mapping[name] = v.metadata.get(name_key, name)

            if not converter:
                conv_attr = f"__convert_{name}__"
                converter = getattr(cls, conv_attr) if hasattr(cls, conv_attr) else get_converter(tp, strict)

            converters[name] = converter
            if name not in mapping:
                mapping[name] = name

        cls.__conv_converters__ = converters
        cls.__conv_mapping__ = mapping
        cls.__conv_default__ = default
        cls.__conv_default_factory__ = default_factory
        cls.__conv_inline__ = inline

        cls.__conv_clear__ = not default and not inline and all(k == v for k, v in mapping.items()) and \
            all(getattr(pcls, "__conv_clear__", True) for pcls in cls.__mro__[1:])

        cls.__conv_inited__ = True


def from_json(cls: Type[T], data: str) -> T:
    return from_dict(cls, json.loads(data))


def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    return cls(**dict_from_json(cls, data))


def init_func(self, **params: Any) -> None:
    self.__dict__.update(params)


def convertable(cls: T) -> T:
    """
    Class decorator
    """
    cls.__conv_inited__ = False
    cls.__conv_init_lock__ = threading.Lock()

    if not hasattr(cls, 'from_dict'):
        cls.from_dict = classmethod(from_dict)

    if not hasattr(cls, 'from_json'):
        cls.from_json = classmethod(from_json)

    if cls.__init__ is object.__init__:
        cls.__init__ = init_func

    return cls

