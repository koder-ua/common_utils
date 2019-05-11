import json
from dataclasses import dataclass

from koder_utils import (pack_structs, unpack_structs, int8, int16, int32, int64, uint8, uint16, uint32, uint64,
                         categorical)


def test_simple():
    class X:
        i8 = int8
        i16 = int16
        i32 = int32
        i64 = int64
        u8 = uint8
        u16 = uint16
        u32 = uint32
        u64 = uint64
        c = categorical

    @dataclass
    class D:
        u8: int
        u16: int
        u32: int
        u64: int
        i8: int
        i16: int
        i32: int
        i64: int
        c: str

    ds = [D(1, 2, 3, 4, -1, -2, -3, -4, "x"),
          D(1, 20, 3, 4, -10, -2, -3, -4, "x"),
          D(1, 2, 3, 4, -1, -200, -3, -4, "x"),
          D(1, 20, 3, 4, -1, -2, -3000, -4, "y"),
          D(1, 2, 300, 4, -1, -2, -3, -4, "y"),
          D(0, 77, 3, 4000000, -1, -2, -3, -40000, "z")]

    data = pack_structs(X, ds)
    assert isinstance(data, bytes)
    uds, offset = unpack_structs(X, data)
    assert len(uds) == len(ds)
    assert len(data) == offset

    for orig, unpacked in zip(ds, uds):
        for name in 'i8 i16 i32 i64 u8 u16 u32 u64 c'.split():
            assert getattr(orig, name) == unpacked[name]
