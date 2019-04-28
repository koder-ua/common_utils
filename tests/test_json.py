from dataclasses import dataclass
from typing import List, Dict

import pytest

from koder_utils import JsonBase, register_from_json, js, dict_from_json, JSONDeserializationError


def test_simple():
    class D(JsonBase):
        x: int
        y: float
        z: str
        a: bool

    data = {'x': 1, 'y': 2.1, 'z': "3", 'a': False}

    d = D.from_json(data)

    assert d.x == data['x']
    assert d.y == data['y']
    assert d.z == data['z']
    assert d.a == data['a']


def test_dataclass():
    @dataclass
    class D(JsonBase):
        x: int
        y: float
        z: str
        a: bool

    data = {'x': 1, 'y': 2.1, 'z': "3", 'a': False}

    d = D.from_json(data)

    assert d.x == data['x']
    assert d.y == data['y']
    assert d.z == data['z']
    assert d.a == data['a']


def test_embedded():
    class D(JsonBase):
        class E(JsonBase):
            t: bool
            f: float
        x: int
        y: int
        z: str
        e: E

    data = {'x': 1, 'y': 2, 'z': "3", 'e': {'t': True, 'f': 2.1}}

    d = D.from_json(data)

    assert d.x == data['x']
    assert d.y == data['y']
    assert d.z == data['z']
    assert d.e.t == data['e']['t']
    assert d.e.f == data['e']['f']


def test_containers():
    class D(JsonBase):
        class E(JsonBase):
            t: bool
            f: int
        x: List[int]
        y: Dict[str, int]
        z: str
        e: List[E]

    data = {'x': [1, 2, 3], 'y': {"d": 2}, 'z': "3", 'e': [{'t': True, 'f': 2}, {'t': False, 'f': -5}]}

    d = D.from_json(data)

    assert d.x == data['x']
    assert d.y == data['y']
    assert d.z == data['z']
    assert len(d.e) == len(data['e'])
    for idx in (0, 1):
        assert d.e[idx].t == data['e'][idx]['t']
        assert d.e[idx].f == data['e'][idx]['f']


def test_strict():
    class D(JsonBase):
        x: int

    class DStrict(JsonBase):
        x: int = js(strict=True)

    assert D.from_json({'x': 1.1}).x == 1
    assert D.from_json({'x': "1"}).x == 1

    with pytest.raises(ValueError):
        DStrict.from_json({'x': 1.1})

    with pytest.raises(ValueError):
        DStrict.from_json({'x': "1"})

    assert D.from_json({'x': 1}).x == 1
    assert DStrict.from_json({'x': 1}).x == 1


def test_no_auto():
    class D(JsonBase):
        x: int
        y: str = js(noauto=True)

    d = D.from_json({'x': 1, 'y': 'u'})
    assert d.x == 1
    assert d.y is D.y

    with pytest.raises(ValueError):
        class D2(JsonBase):
            x: int
            y: str = js(noauto=True, default="t")


def test_default():
    class D(JsonBase):
        x: int
        y: str = js(default="111")

    with pytest.raises(JSONDeserializationError):
        D.from_json({'y': "2"})

    d = D.from_json({'x': 1, 'y': "2"})
    assert d.x == 1
    assert d.y == "2"

    d = D.from_json({'x': 1})
    assert d.x == 1
    assert d.y == "111"
