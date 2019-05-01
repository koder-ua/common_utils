import copy
import inspect
from dataclasses import Field, field
from enum import Enum, IntEnum
from pathlib import Path
from typing import Callable, Type, Dict, Any, TypeVar, Union, List, Tuple, Optional, Set, get_type_hints


T = TypeVar('T')


def strict_converter(t: Type[T]) -> Callable[[T], T]:
    assert t in {int, str, bool, float}

    def closure(v: T) -> T:
        # isinstance have unacceptable behavior for bool
        if type(v) is not t:
            raise ValueError(f"Expected value of type {t.__name__}, get {v!r} with type {type(v).__name__}")
        return v

    return closure


_FROM_JSON_MAP: Dict[Type, Callable[[Any], Any]] = {
    int: int,
    str: str,
    float: float,
    bool: strict_converter(bool),
    Any: lambda x: x,
}


_FROM_JSON_MAP_STRICT: Dict[Type, Callable[[Any], Any]] = {
    int: strict_converter(int),
    str: strict_converter(str),
    float: strict_converter(float),
    bool: strict_converter(bool),
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


@register_from_json(Path)
def path_from_json(v: Any) -> bool:
    assert isinstance(v, str)
    return Path(v)


js_noauto_key = 'json::noauto'
js_converter_key = 'json::converter'
js_key = 'json::key'
js_default = 'json::default'
js_default_factory = 'json::default_factory'
js_inline = 'json::inline'
js_strict = 'json::strict'


class _NotAllowed:
    pass


def js(*, noauto: bool = False,
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
        assert js_noauto_key not in metadata
        metadata[js_noauto_key] = None
        if 'default' not in params and 'default_factory' not in params:
            params['default'] = None

        if converter or key or inline or default is not _NotAllowed or strict:
            raise ValueError("Can't combine 'noauto' with any other parameter")

    if converter:
        assert js_converter_key not in metadata
        metadata[js_converter_key] = converter

    if key:
        assert js_key not in metadata
        metadata[js_key] = key

    if inline:
        assert js_inline not in metadata
        metadata[js_inline] = None

    if default is not _NotAllowed:
        assert default_factory is _NotAllowed, "Can't have default and default_factory set and the same time"
        assert js_default not in metadata
        if not isinstance(default, (str, bool, int, float, bytes, type(None), tuple, frozenset)):
            if isinstance(default, (list, set, dict)):
                default = copy.copy(default)
            else:
                if not allow_shared_default:
                    raise ValueError(f"Can't set default value to instance of mutable type {type(default).__name__}" +
                                     " pass allow_shared_default=True to disable this check")

        metadata[js_default] = default

    if default_factory is not _NotAllowed:
        assert js_default_factory not in metadata
        metadata[js_default_factory] = default

    if strict:
        assert js_strict not in metadata
        metadata[js_strict] = None

    return field(**params, metadata=metadata)


def get_converter_no_cache(t: Type[T], strict: bool) -> Callable[[Any], T]:
    if hasattr(t, 'from_json'):
        return t.from_json

    cmap = _FROM_JSON_MAP_STRICT if strict else _FROM_JSON_MAP

    if t in cmap:
        return cmap[t]

    if hasattr(t, '__origin__'):
        convert = None
        if t.__origin__ is dict:
            key_tp, val_tp = t.__args__
            k_conv = get_converter(key_tp, strict)
            v_conv = get_converter(val_tp, strict)

            def convert(v: Any) -> T:
                if not isinstance(v, dict):
                    raise TypeError(f"Expected dict, get {type(v).__name__}")
                return {k_conv(k): v_conv(v) for k, v in v.items()}

        elif t.__origin__ is list:
            it_conv = get_converter(t.__args__[0], strict)

            def convert(v: Any) -> T:
                if not isinstance(v, list):
                    raise TypeError(f"Expected list, get {type(v).__name__}")
                return [it_conv(item) for item in v]

        elif t.__origin__ is set:
            it_conv = get_converter(t.__args__[0], strict)

            def convert(v: Any) -> T:
                if not isinstance(v, list):
                    raise TypeError(f"Expected list, get {type(v).__name__}")
                return {it_conv(item) for item in v}

        elif t.__origin__ is Union:
            if len(t.__args__) == 2 and type(None) in t.__args__:
                other = list(t.__args__)
                other.remove(type(None))
                conv = get_converter(other[0], strict)

                def convert(v: Any) -> T:
                    return None if v is None else conv(v)

        if convert is not None:
            return convert

    if inspect.isclass(t):
        if issubclass(t, Enum):
            return t.__getitem__
        elif issubclass(t, IntEnum):
            return t

    raise TypeError(f"Can't find converter for type {t.__name__}")


_CACHE: Dict[Any, Callable[[Tuple[bool, Any]], Any]] = {}


def get_converter(t: Type[T], strict: bool) -> Callable[[Any], T]:
    if (t, strict) not in _CACHE:
        _CACHE[(t, strict)] = get_converter_no_cache(t, strict)
    return _CACHE[(t, strict)]


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

    # fast path
    if cls.__js_clear__:
        try:
            return {name: conv(data[name]) for name, conv in cls.__js_converters__.items()}
        except (ValueError, TypeError, AssertionError, JSONDeserializationError):
            # go to long path to get correct exception
            pass

    res: Dict[str, Any] = {}

    for pcls in cls.__mro__[1:-1:-1]:
        if hasattr(pcls, '__js_mapping__'):
            res.update(dict_from_json(pcls, data))

    mp = cls.__js_mapping__
    default = cls.__js_default__
    default_factory = cls.__js_default_factory__
    inline = cls.__js_inline__

    localy_raised = False
    for name, conv in cls.__js_converters__.items():
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
                        raise JSONDeserializationError(cls, name, msg, jsname)
                else:
                    res[name] = conv(v)
        except JSONDeserializationError as exc:
            if not localy_raised:
                exc.push(cls, name, jsname)
            raise
        except (ValueError, TypeError, AssertionError) as exc:
            raise JSONDeserializationError(cls, name, str(exc), jsname) from exc
    return res


def from_json(cls: Type[T], data: Dict[str, Any]) -> T:
    dct = dict_from_json(cls, data)

    if hasattr(cls, "__dataclass_fields__"):
        return cls(**dct)

    if hasattr(cls, "from_dict"):
        return cls.from_dict(**dct)

    obj = cls()
    obj.__dict__.update(dct)
    return obj


def jsonable(cls: T) -> T:
    converters: Dict[str, Callable[[Any], Any]] = {}
    mapping: Dict[str, str] = {}
    default: Dict[str, Any] = {}
    default_factory: Dict[str, Any] = {}
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
            if js_noauto_key in v.metadata:
                continue

            if js_converter_key in v.metadata:
                converter = v.metadata[js_converter_key]

            if js_default in v.metadata:
                default[name] = v.metadata[js_default]

            if js_default_factory in v.metadata:
                default_factory[name] = v.metadata[js_default_factory]

            if js_inline in v.metadata:
                inline.add(name)

            if js_strict in v.metadata:
                strict = True

            mapping[name] = v.metadata.get(js_key, name)

        if not converter:
            conv_attr = f"__convert_{name}__"
            converter = getattr(cls, conv_attr) if hasattr(cls, conv_attr) else get_converter(tp, strict)

        converters[name] = converter
        if name not in mapping:
            mapping[name] = name

    cls.__js_converters__ = converters
    cls.__js_mapping__ = mapping
    cls.__js_default__ = default
    cls.__js_default_factory__ = default_factory
    cls.__js_inline__ = inline

    cls.__js_clear__ = not default and not inline and all(k == v for k, v in mapping.items()) and \
        all(not hasattr(pcls, '__js_mapping__') for pcls in cls.__mro__[1:])

    if not hasattr(cls, 'from_json'):
        cls.from_json = classmethod(from_json)

    return cls


class JsonBase:
    __js_converters__: Dict[str, Callable[[Any], Any]] = {}
    __js_mapping__: Dict[str, str] = {}
    __js_default__: Dict[str, Any] = {}
    __js_inline__: Set[set] = set()

    @classmethod
    def __init_subclass__(cls: T) -> T:
        return jsonable(cls)

    @classmethod
    def from_json(cls: Type[T], data: Dict[str, Any]) -> T:
        return from_json(cls, data)

