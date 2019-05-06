from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
from enum import Enum, IntEnum
from dataclasses import Field, field as dataclass_field, asdict, is_dataclass, dataclass
from typing import Callable, Type, Dict, Any, TypeVar, Union, List, Tuple, Optional, Set, get_type_hints


#   Conversion find order for field x of type T in class Cls
#       x.converter
#       Cls.__convert_$x.name$__
#       T.convert
#       _CONVERTERS[T] => convert each field in dict item
#


T = TypeVar('T')


class ConversionError(ValueError):
    def __init__(self, cls: type, field_name: str, message: str, src_name: str) -> None:
        self.structs_stack: List[Tuple[str, str, Optional[str]]] = [(cls.__name__, field_name, src_name)]
        self.message = message

    def push(self, cls: Type, field_name: str, jsname: str) -> None:
        self.structs_stack.append((cls.__name__, field_name, jsname))

    def __str__(self) -> str:
        stack = []
        for cls_name, field_name, src_name in self.structs_stack:
            source_name = f"(source name '{src_name}')" if src_name != field_name else ""
            stack.append(f"{cls_name}::{field_name} {source_name}")

        if len(stack) > 1:
            str_stack = "\n    ".join(stack)
            return f"During (de)conversion of\n    {str_stack}\n{self.message}"
        else:
            return f"During (de)conversion of {stack[0]} : {self.message}"


def base_converter(t: Type[T]) -> Callable[[T], T]:
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


def base_converter2(t: Type[T], *other_types: Any) -> Callable[[T], T]:
    """
    Only check that incoming value has type t and return value as is
    """
    assert t in {int, str, bool, float}

    all_allowed = t, *other_types

    def closure(v: T) -> T:
        # isinstance have unacceptable behavior for bool
        if type(v) not in all_allowed:
            raise ValueError(f"Expected value of type {t.__name__}, get {v!r} with type {type(v).__name__}")
        return t(v)

    return closure


class ToInt(int):
    def __new__(cls, vl: Union[str, int]) -> int:
        if not isinstance(vl, (int, str)):
            raise TypeError(f"{cls.__name__} expected int or string as input parameter, get " +
                            f"{vl!r} of type {vl.__class__.__name__}")
        return int(vl)


class ToStr(int):
    def __new__(cls, vl: Any) -> str:
        return str(vl)


# Only check that incoming value has type t and return value as is
_CONVERTERS: Dict[Type, Callable[[Any], Any]] = {
    int: base_converter(int),
    str: base_converter(str),
    float: base_converter2(float, int),
    bool: base_converter(bool),
    Any: lambda x: x,
    ToInt: ToInt,
    ToStr: ToStr
}


def register_converter(t: Type[T]) -> Callable[[Any], T]:
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

def get_dict_converter(t) -> Callable[[Dict], Dict]:
    key_tp, val_tp = t.__args__
    k_conv = get_converter(key_tp)
    v_conv = get_converter(val_tp)

    def convert(v: Any) -> T:
        check_types(v, dict)
        return {k_conv(k): v_conv(v) for k, v in v.items()}

    return convert


def get_list_converter(t) -> Callable[[List], List]:
    it_conv = get_converter(t.__args__[0])

    def convert(v: List) -> List:
        check_types(v, list)
        return [it_conv(item) for item in v]

    return convert


def get_tuple_converter(t) -> Callable[[List], Tuple]:
    converters = list(map(get_converter, t.__args__))

    def convert(v: List) -> Tuple:
        check_types(v, list, tuple)
        assert len(v) == len(converters)
        return tuple(it_conv(item) for it_conv, item in zip(converters, v))

    return convert


def get_set_converter(t) -> Callable[[Any], Set]:
    it_conv = get_converter(t.__args__[0])

    def convert(v: Any) -> Set:
        check_types(v, list, set, tuple, frozenset)
        return {it_conv(item) for item in v}

    return convert


def get_union_converter(t) -> Callable[[Any], Any]:
    if len(t.__args__) == 2 and type(None) in t.__args__:
        other = list(t.__args__)
        other.remove(type(None))
        conv = get_converter(other[0])

        def convert(v: Any) -> T:
            return None if v is None else conv(v)
    else:
        convs = [get_converter(tp) for tp in t.__args__]

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
                        raise ValueError("Can't reliable convert to union - two or more types match")
                    val = val2

            if val is _NotAllowed:
                raise ValueError(f"Can't convert to union field - none of types match")

            return val

    return convert


_CONVERTERS_FACTORIES: Dict[Any, Callable[[Any], Callable[[Any], Any]]] = {
    dict: get_dict_converter,
    list: get_list_converter,
    set: get_set_converter,
    Union: get_union_converter,
    tuple: get_tuple_converter
}


def get_converter(t: Type[T], *, _cache: Dict[Any, Callable[[Any], Any]] = {}) -> Callable[[Any], T]:
    if t not in _cache:
        if hasattr(t, 'convert'):
            _cache[t] = t.convert
        elif t in _CONVERTERS:
            _cache[t] = _CONVERTERS[t]
        elif hasattr(t, '__origin__') and t.__origin__ in _CONVERTERS_FACTORIES:
            _cache[t] = _CONVERTERS_FACTORIES[t.__origin__](t)
        elif inspect.isclass(t):
            if issubclass(t, Enum):
                _cache[t] = t.__getitem__
            elif issubclass(t, IntEnum):
                _cache[t] = t
        else:
            raise TypeError(f"Can't find converter for type {getattr(t, '__name__', t)}")
    return _cache[t]


@register_converter(Path)
def to_path(v: str) -> Path:
    assert isinstance(v, str)
    return Path(v)


_METADATA_KEY = "conv::metadata"


class _NotAllowed:
    pass


def field(*, noauto: bool = False,
          converter: Callable[[Any], Any] = None,
          key: str = None,
          default: Any = _NotAllowed,
          allow_shared_default: bool = False,
          default_factory: Callable[[], Any] = None,
          inline: bool = False,
          **params) -> Field:

    have_default = default is not _NotAllowed
    metadata = params.pop("metadata", {})
    assert _METADATA_KEY not in metadata

    if inline:
        if noauto or converter or key or have_default or default_factory or allow_shared_default:
            raise ValueError("Can't combine 'inline' with any other parameter")

    if have_default:
        assert not default_factory, "Can't have default and default_factory_attr set and the same time"
        if not isinstance(default, (str, bool, int, float, bytes, type(None), tuple, frozenset)):
            if isinstance(default, (list, set, dict)):
                default = copy.copy(default)
            else:
                if not allow_shared_default:
                    raise ValueError(f"Can't set default value to instance of mutable type " +
                                     f"{type(default).__name__} pass allow_shared_default=True " +
                                     "to disable this check")

        def default_factory() -> Any:
            return default

    if noauto:
        if converter or key:
            raise ValueError("Can't combine 'noauto' with 'converter' or 'key' parameters")

        if default:
            params['default'] = default
        elif default_factory:
            params['default_factory'] = default_factory
        else:
            params['default'] = None

    metadata[_METADATA_KEY] = (noauto, converter, key, default_factory, inline)
    return dataclass_field(**params, metadata=metadata)


CT = TypeVar('CT', bound='ConvBase')


@dataclass
class FieldData:
    name: str
    src_name: str
    converter: Callable[[Any], Any]
    default_factory: Callable[[Any], Any]


class _NotValueMarker:
    pass


class ConvBase:
    # this fields will be set during first call to convert/from_json
    __conv_converters__: List[FieldData]
    __conv_inline__: Set[str]

    def __init__(self, **params: Any) -> None:
        self.__dict__.update(params)

    @classmethod
    def __init_subclass__(cls: Type[CT]) -> None:
        converters: List[FieldData] = []
        inline: Set[str] = set()

        # if dataclass decorator already applied
        fields_dct = asdict(cls) if is_dataclass(cls) else cls.__dict__

        annotation = get_type_hints(cls, localns=cls.__dict__, globalns=None)
        for name, tp in annotation.items():
            if name.startswith("_"):
                continue

            converter = None
            v = fields_dct.get(name)

            if isinstance(v, Field) and _METADATA_KEY in v.metadata:
                noauto, converter, src_name, default_fct, is_inline = v.metadata[_METADATA_KEY]

                if src_name is None:
                    src_name = name

                if noauto:
                    continue

                if is_inline:
                    inline.add(name)
            else:
                src_name = name
                default_fct = None
                is_inline = False

            if not converter:
                conv_attr = f"__convert_{name}__"
                converter = getattr(cls, conv_attr) if hasattr(cls, conv_attr) else get_converter(tp)

            if not is_inline:
                converters.append(FieldData(name, src_name, converter, default_fct))

        cls.__conv_converters__ = converters
        cls.__conv_inline__ = inline

    @classmethod
    def convert(cls: Type[CT], data: Any) -> CT:
        return cls(**cls.convert_dict(data))

    @classmethod
    def from_json(cls: Type[CT], data: str) -> CT:
        return cls.convert(json.loads(data))

    @classmethod
    def convert_dict(cls: Type[CT], data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Actually validate/convert dict fields
        """
        # TODO: add fastline conversion
        assert isinstance(data, dict)
        res: Dict[str, Any] = {}
        inline = cls.__conv_inline__
        localy_raised = False
        for fld in cls.__conv_converters__:
            try:
                if fld.name in inline:
                    res[fld.name] = fld.converter(data)
                else:
                    vl = data.get(fld.src_name, _NotValueMarker)
                    if vl is _NotValueMarker:
                        if fld.default_factory:
                            res[fld.name] = fld.default_factory()
                        else:
                            msg = f"Input dict has no key '{fld.src_name}'. Only fields {','.join(data)} present."
                            localy_raised = True
                            raise ConversionError(cls, fld.name, msg, fld.src_name)
                    else:
                        res[fld.name] = fld.converter(vl)
            except ConversionError as exc:
                if not localy_raised:
                    exc.push(cls, fld.name, fld.src_name)
                raise
            except (ValueError, TypeError, AssertionError) as exc:
                raise ConversionError(cls, fld.name, str(exc), fld.src_name) from exc
        return res
