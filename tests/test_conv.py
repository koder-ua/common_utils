from dataclasses import dataclass
from typing import List, Dict, Optional, Union

import pytest

from koder_utils import ConvBase, field, ConversionError, ToInt, ToStr


def test_simple():
    class D(ConvBase):
        x: int
        y: float
        z: str
        a: bool

    data = {'x': 1, 'y': 2.1, 'z': "3", 'a': False}

    d = D.convert(data)

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

    d = D.convert(data)

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

    d = D.convert(data)

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

    d = D.convert(data)

    assert d.x == data['x']
    assert d.y == data['y']
    assert d.z == data['z']
    assert len(d.e) == len(data['e'])
    for idx in (0, 1):
        assert d.e[idx].t == data['e'][idx]['t']
        assert d.e[idx].f == data['e'][idx]['f']


def test_strict():
    class D(ConvBase):
        x: ToInt

    class DStrict(ConvBase):
        x: int

    assert D.convert({'x': 1.1}).x == 1
    assert D.convert({'x': "1"}).x == 1

    with pytest.raises(ValueError):
        DStrict.convert({'x': 1.1})

    with pytest.raises(ValueError):
        DStrict.convert({'x': "1"})

    assert D.convert({'x': 1}).x == 1
    assert DStrict.convert({'x': 1}).x == 1


def test_no_auto():
    class D(ConvBase):
        x: int
        y: str = field(noauto=True)

    d = D.convert({'x': 1, 'y': 'u'})
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

    with pytest.raises(ConversionError):
        D.convert({'y': "2"})

    d = D.convert({'x': 1, 'y': "2"})
    assert d.x == 1
    assert d.y == "2"

    d = D.convert({'x': 1})
    assert d.x == 1
    assert d.y == "111"

    class D2(ConvBase):
        x: int
        y: str = field(noauto=True, default="222")

    d2 = D2.convert({'x': 1})
    assert d2.x == 1
    assert d2.y == "222"

    class D3(ConvBase):
        x: int
        z: str
        y: str = field(noauto=True, default_factory=lambda: "333")

    d3 = D3.convert({'x': 3, 'z': 'rrr'})
    assert d3.x == 3
    assert d3.z == 'rrr'
    assert d3.y == "333"

    with pytest.raises(AssertionError):
        class D4(ConvBase):
            y: str = field(default="2", default_factory=lambda: "333")


def test_from_dict():
    class D(ConvBase):
        x: int
        y: str = field(default="111")

    with pytest.raises(ConversionError):
        D.convert({'y': "2"})

    d = D.convert({'x': 1, 'y': "2"})
    assert d.x == 1
    assert d.y == "2"

    d = D.convert({'x': 1})
    assert d.x == 1
    assert d.y == "111"


def test_from_json():
    class D(ConvBase):
        x: int
        y: str = field(default="111")

    with pytest.raises(ConversionError):
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

    with pytest.raises(ConversionError):
        D.convert({'y': "a", 'x': 'fff'})

    with pytest.raises(ConversionError):
        D.convert({'y': "a"})

    d = D.convert({'y': "a", 'x': 1})
    assert d.x == 1
    assert d.y == "a"

    d = D.convert({'y': "a", 'x': None})
    assert d.x is None
    assert d.y == "a"


def test_union():
    class D(ConvBase):
        x: Union[ToInt, ToStr]
        y: str

    with pytest.raises(ConversionError):
        D.convert({'y': "a", 'x': '1'})

    with pytest.raises(ConversionError):
        D.convert({'y': "a", 'x': 1})

    class D(ConvBase):
        x: Union[int, str]
        y: str

    d = D.convert({'y': "a", 'x': 1})
    assert d.x == 1
    assert d.y == "a"

    d = D.convert({'y': "a", 'x': "1"})
    assert d.x == "1"
    assert d.y == "a"

    with pytest.raises(ConversionError):
        D.convert({'y': "a", 'x': [1]})

