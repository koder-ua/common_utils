import inspect
import json
from dataclasses import Field, field
from enum import Enum, IntEnum
from pathlib import Path
from typing import Callable, Type, Dict, Any, TypeVar, Union, cast, List, Tuple, Optional

T = TypeVar('T')


_FROM_JSON_MAP: Dict[Type, Callable[[Any], Any]] = {
    int: int,
    str: str,
    float: float,
    Any: lambda x: x,
}


def register_from_json(t: Type[T]) -> Callable[[Any], T]:
    def closure(func: Callable[[Any], T]) -> Callable[[Any], T]:
        assert t not in _FROM_JSON_MAP
        _FROM_JSON_MAP[t] = func
        return func
    return closure


def json_converted(t: Type[T]) -> Callable[[Any], T]:
    def closure(v: Any) -> T:
        return _FROM_JSON_MAP[t](v)
    return closure


@register_from_json(bool)
def bool_from_json(v: Any) -> bool:
    assert isinstance(v, bool)
    return cast(bool, v)


@register_from_json(Path)
def path_from_json(v: Any) -> bool:
    assert isinstance(v, str)
    return Path(v)


no_auto_js_key = 'json::noauto'
js_converter_key = 'json::converter'
js_key = 'json::key'


def js(*, noauto: bool = False, converter: Callable[[Any], Any] = None, key: str = None, **params) -> Field:
    metadata = params.pop("metadata", {})
    if noauto:
        assert no_auto_js_key not in metadata
        metadata[no_auto_js_key] = None
        if 'default' not in params and 'default_factory' not in params:
            params['default'] = None

    if converter:
        assert js_converter_key not in metadata
        metadata[js_converter_key] = converter

    if key:
        assert js_key not in metadata
        metadata[js_key] = key

    return field(**params, metadata=metadata)


def get_converter(t: Type[T], _cache: Dict[Any, Callable[[Any], Any]] = {}) -> Callable[[Any], T]:
    if t not in _cache:
        convert = None

        if hasattr(t, "from_json"):
            convert = t.from_json

        elif hasattr(t, '__origin__'):
            if t.__origin__ is dict:
                key_tp, val_tp = t.__args__
                k_conv = get_converter(key_tp)
                v_conv = get_converter(val_tp)

                def convert(v: Any) -> T:
                    if not isinstance(v, dict):
                        raise TypeError(f"Expected dict, get {type(v).__name__}")
                    return {k_conv(k): v_conv(v) for k, v in v.items()}

            elif t.__origin__ is list:
                it_conv = get_converter(t.__args__[0])

                def convert(v: Any) -> T:
                    if not isinstance(v, list):
                        raise TypeError(f"Expected list, get {type(v).__name__}")
                    return [it_conv(item) for item in v]

            elif t.__origin__ is set:
                it_conv = get_converter(t.__args__[0])

                def convert(v: Any) -> T:
                    if not isinstance(v, list):
                        raise TypeError(f"Expected list, get {type(v).__name__}")
                    return {it_conv(item) for item in v}

            elif t.__origin__ is Union:
                if len(t.__args__) == 2 and type(None) in t.__args__:
                    other = list(t.__args__)
                    other.remove(type(None))
                    conv = get_converter(other[0])

                    def convert(v: Any) -> T:
                        return None if v is None else conv(v)

        if convert is None and inspect.isclass(t):
            if issubclass(t, Enum):
                convert = t.__getitem__
            elif issubclass(t, IntEnum):
                convert = t

        if convert is None:
            if t not in _FROM_JSON_MAP:
                raise TypeError(f"Can't find converter for type {t.__name__}")
            else:
                convert = _FROM_JSON_MAP[t]

        _cache[t] = convert

    return _cache[t]


class JSONDeserializationError(ValueError):
    def __init__(self, cls: type, field: str, message: str, jsname: str) -> None:
        self.structs_stack: List[Tuple[str, str, Optional[str]]] = [(cls.__name__, field, jsname)]
        self.message = message

    def push(self, cls: Type, field: str, jsname: str) -> None:
        self.structs_stack.append((cls.__name__, field, jsname))

    def __str__(self) -> str:
        stack = []
        for cls_name, field, jsfield in self.structs_stack:
            stack.append(f"{cls_name}::{field}" + (f"(json name '{jsfield}')" if jsfield != field else ""))

        if len(stack) > 1:
            str_stack = "\n    ".join(stack)
            return f"During deserialization of\n    {str_stack}\n{self.message}"
        else:
            return f"During deserialization of {stack[0]} : {self.message}"


def dict_from_json(cls: Type[T], data: Dict[str, Any]) -> Dict[str, Any]:
    mp = cls.__js_mapping__
    # this is fast-path
    try:
        return {name: conv(data[mp[name]]) for name, conv in cls.__js_converters__.items()}
    except (ValueError, TypeError, AssertionError, KeyError):
        pass

    # this part is to make useful exception message
    for name, conv in cls.__js_converters__.items():
        jsname = mp[name]
        try:
            conv(data[jsname])
        except JSONDeserializationError as exc:
            exc.push(cls, name, jsname)
            raise
        except (ValueError, TypeError, AssertionError) as exc:
            raise JSONDeserializationError(cls, name, str(exc), jsname) from exc
        except KeyError as exc:
            if jsname not in data:
                msg = f"Input js dict has no key '{jsname}'. Only fields {','.join(data)} present.\n{exc}"
                raise JSONDeserializationError(cls, name, msg, jsname) from exc
            raise


def from_json(cls: Type[T], data: Dict[str, Any]) -> T:
    return cls(**dict_from_json(cls, data))


def jsonable(cls: T) -> T:
    converters: Dict[str, Callable[[Any], Any]] = {}
    mapping: Dict[str, str] = {}

    # if dataclass decorator already applied
    if hasattr(cls, "__dataclass_fields__"):
        fields_dct = cls.__dataclass_fields__
    else:
        fields_dct = cls.__dict__

    for base in cls.__mro__[1:][::-1]:
        if base is JsonBase:
            break
        if issubclass(base, JsonBase):
            converters.update(cls.__js_converters__)

    annotation = getattr(cls, "__annotations__", {})
    for name, tp in annotation.items():
        converter = None
        v = fields_dct.get(name)

        if isinstance(v, Field):
            if no_auto_js_key in v.metadata:
                continue

            if js_converter_key in v.metadata:
                converter = v.metadata[js_converter_key]

            mapping[name] = v.metadata.get(js_key, name)

        if not converter:
            conv_attr = f"__convert_{name}__"
            converter = getattr(cls, conv_attr) if hasattr(cls, conv_attr) else get_converter(tp)

        converters[name] = converter
        if name not in mapping:
            mapping[name] = name

    cls.__js_converters__ = converters
    cls.__js_mapping__ = mapping

    if not hasattr(cls, 'from_json'):
        cls.from_json = classmethod(from_json)

    return cls


class JsonBase:
    __js_converters__: Dict[str, Callable[[Any], Any]] = {}
    __js_mapping__: Dict[str, str] = {}

    @classmethod
    def __init_subclass__(cls: T) -> T:
        return jsonable(cls)

    @classmethod
    def from_json(cls: Type[T], data: Dict[str, Any]) -> T:
        return from_json(cls, data)
