import struct
from dataclasses import dataclass
from typing import Optional, Any, List, Dict, Tuple, Union, Iterator, Sequence, Iterable


@dataclass
class Int:
    bit_size: int
    pack_format: Optional[str] = None


@dataclass
class UInt:
    bit_size: int
    pack_format: Optional[str] = None


class Categorical:
    pass


categorical = Categorical()

uint8 = UInt(8, 'B')
uint16 = UInt(16, 'H')
uint32 = UInt(32, 'L')
uint64 = UInt(64, 'Q')

int8 = Int(8, 'b')
int16 = Int(16, 'h')
int32 = Int(32, 'l')
int64 = Int(64, 'q')


def pack_bytes_list(data: Iterable[bytes], sz_format: str = 'H', count_format: str = 'H') -> bytes:
    res = b""
    idx = 0
    for idx, chunk in enumerate(data):
        res += chunk

    if not res:
        return b""

    idx += 1
    return struct.pack(count_format, idx) + struct.pack(f"!{idx}{sz_format}", *map(len, data)) + res


def unpack_bytes_list(data: bytes, offset: int, sz_format: str = 'H', count_format: str = 'H') \
        -> Tuple[int, Iterator[bytes]]:

    cnt_sz = struct.calcsize(count_format)
    sz_sz = struct.calcsize(sz_format)

    count, = struct.unpack(count_format, data[offset: offset + cnt_sz])
    offset += cnt_sz
    sizes = struct.unpack(f"!{count}{sz_format}", data[offset: offset + sz_sz * count])
    offset += sz_sz * count

    def unpack() -> Iterator[bytes]:
        ioffset = offset
        for itm_sz in sizes:
            yield data[ioffset: ioffset + itm_sz]
            ioffset += itm_sz

    return offset + sum(sizes), unpack()


def get_fields(format) -> Iterator[Tuple[str, Union[Int, UInt, Categorical]]]:
    for name, val in format.__dict__.items():
        if not name.startswith('_'):
            if isinstance(val, (UInt, Int, Categorical)):
                yield name, val


def pack_cat_list(vals: List[str]) -> bytes:
    assert len(vals) < 2 ** 16
    uniq = list(set(vals))
    assert all(isinstance(vl, str) for vl in uniq)

    mapping = {val: vid for vid, val in enumerate(uniq)}
    mapping_b = pack_bytes_list([vl.encode() for vl in uniq], sz_format='B')
    mapped_vals = [mapping[vl] for vl in vals]
    return mapping_b + struct.pack(f"!{len(vals)}{'B' if len(uniq) < 256 else 'H'}", *mapped_vals)


def unpack_cat_list(data: bytes, offset: int, count: int) -> Tuple[List[str], int]:
    offset, itr = unpack_bytes_list(data, offset, sz_format='B')
    rmapping = {idx: name.decode() for idx, name in enumerate(itr)}

    fmt = f"!{count}{'B' if len(rmapping) < 256 else 'H'}"
    data_sz = struct.calcsize(fmt)
    data_b = data[offset: offset + data_sz]
    return [rmapping[vl] for vl in struct.unpack(fmt, data_b)], offset + data_sz


def pack_structs(format, data: Sequence[Any]) -> bytes:
    res = struct.pack('!Q', len(data))
    is_dct = isinstance(data[0], dict)
    for name, tp in sorted(get_fields(format)):
        vals = [(obj[name] if is_dct else getattr(obj, name)) for obj in data]
        if isinstance(tp, (UInt, Int)):
            assert tp.pack_format
            res += struct.pack(f"!{len(vals)}{tp.pack_format}", *vals)
        elif tp is categorical:
            res += pack_cat_list(vals)
        else:
            raise ValueError(f"Can't serialize field {tp}")
    return res


def unpack_structs(format, data: bytes, offset: int = 0) -> Tuple[List[Dict[str, Union[int, str]]], int]:
    sz = struct.calcsize('!Q')
    count, = struct.unpack('!Q', data[offset: offset + sz])
    offset += sz
    res: List[Dict[str, Union[int, str]]] = [{} for _ in range(count)]

    for name, tp in sorted(get_fields(format)):
        if isinstance(tp, (UInt, Int)):
            assert tp.pack_format
            fmt = f"!{len(res)}{tp.pack_format}"
            sz = struct.calcsize(fmt)
            vals = struct.unpack(fmt, data[offset: offset + sz])
            offset += sz
        elif tp is categorical:
            vals, offset = unpack_cat_list(data, offset, count)
        else:
            raise ValueError(f"Can't serialize field {tp}")

        for dct, val in zip(res, vals):
            dct[name] = val

    return res, offset
