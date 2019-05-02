from dataclasses import dataclass
from typing import List, Dict, Optional, Union

import pytest

from koder_utils import ConvBase, field, DeserializationError, convertable


def test_simple():
    class D(ConvBase):
        x: int
        y: float
        z: str
        a: bool

    data = {'x': 1, 'y': 2.1, 'z': "3", 'a': False}

    d = D.from_dict(data)

    assert d.x == data['x']
    assert d.y == data['y']
    assert d.z == data['z']
    assert d.a == data['a']


def test_dataclass():
    @dataclass
    class D(ConvBase):
        x: int
        y: float
        z: str
        a: bool

    data = {'x': 1, 'y': 2.1, 'z': "3", 'a': False}

    d = D.from_dict(data)

    assert d.x == data['x']
    assert d.y == data['y']
    assert d.z == data['z']
    assert d.a == data['a']


def test_embedded():
    class D(ConvBase):
        class E(ConvBase):
            t: bool
            f: float
        x: int
        y: int
        z: str
        e: E

    data = {'x': 1, 'y': 2, 'z': "3", 'e': {'t': True, 'f': 2.1}}

    d = D.from_dict(data)

    assert d.x == data['x']
    assert d.y == data['y']
    assert d.z == data['z']
    assert d.e.t == data['e']['t']
    assert d.e.f == data['e']['f']


def test_containers():
    class D(ConvBase):
        class E(ConvBase):
            t: bool
            f: int
        x: List[int]
        y: Dict[str, int]
        z: str
        e: List[E]

    data = {'x': [1, 2, 3], 'y': {"d": 2}, 'z': "3", 'e': [{'t': True, 'f': 2}, {'t': False, 'f': -5}]}

    d = D.from_dict(data)

    assert d.x == data['x']
    assert d.y == data['y']
    assert d.z == data['z']
    assert len(d.e) == len(data['e'])
    for idx in (0, 1):
        assert d.e[idx].t == data['e'][idx]['t']
        assert d.e[idx].f == data['e'][idx]['f']


def test_strict():
    class D(ConvBase):
        x: int

    class DStrict(ConvBase):
        x: int = field(strict=True)

    assert D.from_dict({'x': 1.1}).x == 1
    assert D.from_dict({'x': "1"}).x == 1

    with pytest.raises(ValueError):
        DStrict.from_dict({'x': 1.1})

    with pytest.raises(ValueError):
        DStrict.from_dict({'x': "1"})

    assert D.from_dict({'x': 1}).x == 1
    assert DStrict.from_dict({'x': 1}).x == 1


def test_no_auto():
    class D(ConvBase):
        x: int
        y: str = field(noauto=True)

    d = D.from_dict({'x': 1, 'y': 'u'})
    assert d.x == 1
    assert d.y is D.y

    with pytest.raises(ValueError):
        class D2(ConvBase):
            x: int
            y: str = field(noauto=True, default="t")


def test_default():
    class D(ConvBase):
        x: int
        y: str = field(default="111")

    with pytest.raises(DeserializationError):
        D.from_dict({'y': "2"})

    d = D.from_dict({'x': 1, 'y': "2"})
    assert d.x == 1
    assert d.y == "2"

    d = D.from_dict({'x': 1})
    assert d.x == 1
    assert d.y == "111"


def test_from_dict():
    class D(ConvBase):
        x: int
        y: str = field(default="111")

    with pytest.raises(DeserializationError):
        D.from_dict({'y': "2"})

    d = D.from_dict({'x': 1, 'y': "2"})
    assert d.x == 1
    assert d.y == "2"

    d = D.from_dict({'x': 1})
    assert d.x == 1
    assert d.y == "111"


def test_from_json():
    class D(ConvBase):
        x: int
        y: str = field(default="111")

    with pytest.raises(DeserializationError):
        D.from_json('{"y": "2"}')

    d = D.from_json('{"x": 1, "y": "2"}')
    assert d.x == 1
    assert d.y == "2"

    d = D.from_json('{"x": 1}')
    assert d.x == 1
    assert d.y == "111"


def test_optional():
    class D(ConvBase):
        x: Optional[int]
        y: str

    with pytest.raises(DeserializationError):
        D.from_dict({'y': "a", 'x': 'fff'})

    with pytest.raises(DeserializationError):
        D.from_dict({'y': "a"})

    d = D.from_dict({'y': "a", 'x': 1})
    assert d.x == 1
    assert d.y == "a"

    d = D.from_dict({'y': "a", 'x': None})
    assert d.x is None
    assert d.y == "a"


def test_union():
    class D(ConvBase):
        x: Union[int, str]
        y: str

    with pytest.raises(DeserializationError):
        D.from_dict({'y': "a", 'x': '1'})

    with pytest.raises(DeserializationError):
        D.from_dict({'y': "a", 'x': 1})

    class D(ConvBase):
        x: Union[int, str] = field(strict=True)
        y: str

    d = D.from_dict({'y': "a", 'x': 1})
    assert d.x == 1
    assert d.y == "a"

    d = D.from_dict({'y': "a", 'x': "1"})
    assert d.x == "1"
    assert d.y == "a"

    with pytest.raises(DeserializationError):
        D.from_dict({'y': "a", 'x': [1]})


def test_decorator():
    @convertable
    class D:
        x: int
        y: str = field(default="111")

    with pytest.raises(DeserializationError):
        D.from_json('{"y": "2"}')

    d = D.from_json('{"x": 1, "y": "2"}')
    assert d.x == 1
    assert d.y == "2"

    d = D.from_json('{"x": 1}')
    assert d.x == 1
    assert d.y == "111"