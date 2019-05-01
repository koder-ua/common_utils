from __future__ import annotations

import abc
import weakref
from enum import Enum
from typing import Any, Union, Optional, Callable, Iterable, Tuple, List, Dict, Type, Set
from dataclasses import dataclass, field

from . import b2ssize, b2ssize_10, seconds_to_str, RawContent


class Align(Enum):
    default = 0
    left = 1
    center = 2
    right = 3


@dataclass
class Cell:
    data: Any
    colspan: int = 1
    align: Align = Align.default
    color: str = 'default'
    attrs: Dict[str, Any] = field(default_factory=dict)


class Separator:
    pass


@dataclass
class Field:
    tp: Any
    header: Optional[str] = None
    converter: Callable[[Any], Any] = str
    custom_sort: Callable[[Any], str] = str
    align: Align = Align.default
    allow_none: bool = True
    skip_if_no_data: bool = True
    null_sort_key: str = ''
    null_value: str = ''
    dont_sort: bool = False
    help: Optional[str] = None
    attrs: Dict[str, Any] = field(default_factory=dict)
    attr_name: Optional[str] = None
    chars_per_line: Optional[int] = None

    def __post_init__(self):
        assert self.tp is not tuple


@dataclass
class Ok:
    value: Any = 'ok'


@dataclass
class Fail:
    value: Any = 'fail'


class Column:
    @staticmethod
    def i(header: str = None, **extra) -> Field:
        return Field(None, header, lambda x: x, null_sort_key='-1', **extra)

    @staticmethod
    def d(header: str = None, **extra) -> Field:
        return Field(int, header, b2ssize_10, null_sort_key='-1', **extra)

    @staticmethod
    def ed(header: str = None, **extra) -> Field:
        return Field(int, header, null_sort_key='-1', **extra)

    @staticmethod
    def sz(header: str = None, **extra) -> Field:
        return Field(int, header, b2ssize, null_sort_key='-1', **extra)

    @staticmethod
    def s(header: str = None, **extra) -> Field:
        return Field(str, header, **extra)

    @staticmethod
    def to_str(header: str = None, **extra) -> Field:
        return Field(None, header, converter=lambda x: x if isinstance(x, RawContent) else str(x), **extra)

    @staticmethod
    def f2(header: str = None, **extra) -> Field:
        return Field((int, float), header, converter=lambda x: f"{x:.2f}", **extra)

    @staticmethod
    def seconds(header: str = None, **extra) -> Field:
        return Field((int, float), header, converter=seconds_to_str, **extra)

    @staticmethod
    def list(header: str = None, delim: str = ' ', **extra) -> Field:

        def joiner(x: Iterable[Any]) -> str:
            return delim.join(map(str, x))

        return Field(list, header, converter=joiner, dont_sort=True, **extra)

    @staticmethod
    def ok_or_fail(header: str = None, **kwargs) -> Field:
        return Field(bool, header, converter=lambda val: Ok() if val else Fail(), **kwargs)

    @staticmethod
    def yes_or_no(header: str = None, true_fine: bool = True, **kwargs) -> Field:
        def converter(val: bool) -> Any:
            if true_fine:
                return Ok('yes') if val else Fail('no')
            else:
                return Fail('yes') if val else Ok('no')
        return Field(bool, header, converter=converter, **kwargs)


class WithAttrs:
    def __init__(self, val: Any, **attrs: str) -> None:
        self.val = val
        self.attrs = attrs


class _NotUsed:
    pass


class Table:
    RowTp = Union[Type[Separator], List[Union[Cell, Type[_NotUsed]]]]
    RowReadyTp = Union[Type[Separator], List[Cell]]

    def __init__(self) -> None:
        self.fields = [fld for _, fld in self.all_fields()]
        self.fields_map = {fld.attr_name: fld for fld in self.fields}
        self.fields_names = [fld.attr_name for fld in self.fields]
        self.columns_count = len(self.fields)
        self.rows: List[Table.RowTp] = []

    @classmethod
    def add_column(cls, name: str, fld: Field) -> None:
        setattr(cls, name, fld)

    @classmethod
    def all_fields(cls) -> Iterable[Tuple[str, Field]]:
        return ((attr_name, fld)
                for attr_name, fld in cls.__dict__.items()
                if isinstance(fld, Field))

    @classmethod
    def __init_subclass__(cls, **kwargs):
        for attr, fld in cls.all_fields():
            if fld.header is None:
                fld.header = attr.replace("_", " ").capitalize()
            fld.attr_name = attr

    def prepare_field(self, val: Any, *, fld: Field = None, name: str = None) -> Cell:
        if fld is None:
            assert name is not None
            fld = self.fields_map[name]

        if isinstance(val, tuple):
            val, sort_by = val
            attrs = {'sort_by': str(sort_by)}
        else:
            attrs = {}

        if fld.tp:
            assert isinstance(val, fld.tp) or (fld.allow_none and val is None), \
                f"Field {fld.attr_name} requires type {fld.tp}{' or None' if fld.allow_none else ''} " + \
                f"but get {val!r} of type {type(val)}"

        return Cell(fld.converter(val), attrs=attrs, align=fld.align)

    def next_row(self) -> Row:
        self.rows.append([_NotUsed] * self.columns_count)
        return Row(self, self.rows[-1])

    def get_used_columns(self) -> List[Field]:
        # find all used keys
        all_idx: Set[int] = set()
        for row in self.rows:
            if row is not Separator:
                all_idx.update(idx for idx, v in enumerate(row) if v is not _NotUsed)
        return [self.fields[idx] for idx in sorted(all_idx)]

    def headers(self, hide_unused: bool = False) -> List[Field]:
        return self.get_used_columns() if hide_unused else self.fields

    def add_row(self, *vals) -> None:
        self.rows.append([self.prepare_field(val, fld=fld) for val, fld in zip(vals, self.fields)])

    def add_named_row(self, **vals: Any):
        expected = {fld.attr_name for fld in self.fields}
        unexpected = set(vals) - expected
        assert not unexpected, f"Get unexpected fields {','.join(unexpected)}"
        self.rows.append([self.prepare_field(v, name=name) for name, v in vals.items()])

    def add_separator(self):
        self.rows.append(Separator)

    def content(self, hide_unused: bool = False) -> List[RowReadyTp]:
        active_fields = self.get_used_columns() if hide_unused else self.fields
        active_attrs = {fld.attr_name for fld in active_fields}

        res: List[Table.RowReadyTp] = []
        for row in self.rows:
            if row is Separator:
                res.append(row)
            else:
                res.append([(Cell("") if cell is _NotUsed else cell)
                            for fld, cell in zip(self.fields, row)
                            if fld.attr_name in active_attrs])
        return res


class Row:
    _target__: List
    _table__: Callable[[], Optional[Table]]

    def __init__(self, table: Table, target: List) -> None:
        self.__dict__['_target__'] = target
        self.__dict__['_table__'] = weakref.ref(table)

    def __setitem__(self, name: str, val: Any) -> None:
        setattr(self, name, val)

    def __setattr__(self, name: str, val: Any) -> None:
        table = self._table__()
        assert table
        self._target__[table.fields_names.index(name)] = table.prepare_field(val, name=name)


class SimpleTable:
    def __init__(self, *headers: str) -> None:
        self.hdrs = [Field(tp=str, header=header, attr_name=header) for header in headers]
        self.data: List[List[Cell]] = []

    def last_line_size(self) -> int:
        if not self.data:
            return 0
        return sum(cell.colspan for cell in self.data[-1])

    def add_row(self, *cells: Any) -> None:
        assert not self.data or self.last_line_size() in (len(self.hdrs), 0), \
            f"self.last_line_size()={self.last_line_size()} != len(self.headers)={len(self.hdrs)}"
        assert len(cells) == len(self.hdrs), f"len(cells)={len(cells)} != len(self.headers)={len(self.hdrs)}"
        if self.data and len(self.data[-1]) == 0:
            self.data[-1] = [Cell(vl) for vl in cells]
        else:
            self.data.append([Cell(vl) for vl in cells])
        self.next_row()

    def next_row(self) -> None:
        assert self.data and self.last_line_size() == len(self.hdrs)
        self.data.append([])

    def headers(self, hide_unused: bool = False) -> List[Field]:
        return self.hdrs

    def content(self, hide_unused: bool = False) -> List[Table.RowReadyTp]:
        assert not self.data or self.last_line_size() <= len(self.hdrs)
        return self.data

    def add_cell(self, val: Any, colspan: int = 1, **attrs) -> None:
        if not self.data:
            self.data.append([])
        self.data[-1].append(Cell(val, colspan=colspan, attrs=attrs))
        assert self.last_line_size() <= len(self.hdrs)


class Style(metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def sep(cls, col_width: List[int]) -> str:
        ...

    @classmethod
    @abc.abstractmethod
    def header_sep(cls, col_width: List[int]) -> str:
        ...

    @classmethod
    @abc.abstractmethod
    def top_line(cls, col_width: List[int]) -> str:
        ...

    @classmethod
    @abc.abstractmethod
    def bottom_line(cls, col_width: List[int]) -> str:
        ...

    @classmethod
    @abc.abstractmethod
    def format_line(cls, cells: List[str]) -> str:
        ...


class TableStyleDefault(Style):
    h_line = chr(0x2500)
    v_line = chr(0x2502)
    d_h_line = chr(0x2550)
    d_right_end = chr(0x2561)
    d_left_end = chr(0x255E)
    d_cross = chr(0x256a)
    cross = chr(0x253c)
    right_end = chr(0x2524)
    left_end = chr(0x251c)
    upper_end = chr(0x252c)
    down_end = chr(0x2534)
    top_right_corner = chr(0x256e)
    top_left_corner = chr(0x256d)
    bottom_right_corner = chr(0x256f)
    bottom_left_corner = chr(0x2570)
    padding = 1

    @classmethod
    def sep(cls, col_width: List[int]) -> str:
        return cls.d_left_end + \
               cls.d_cross.join(cls.d_h_line * (w + 2 * cls.padding) for w in col_width) + cls.d_right_end

    @classmethod
    def header_sep(cls, col_width: List[int]) -> str:
        return cls.left_end + cls.cross.join(cls.h_line * (w + 2 * cls.padding) for w in col_width) + cls.right_end

    @classmethod
    def top_line(cls, col_width: List[int]) -> str:
        return cls.top_left_corner + cls.upper_end.join(cls.h_line * (w + 2 * cls.padding) for w in col_width) + \
               cls.top_right_corner

    @classmethod
    def bottom_line(cls, col_width: List[int]) -> str:
        return cls.bottom_left_corner + cls.down_end.join(cls.h_line * (w + 2 * cls.padding) for w in col_width) \
               + cls.bottom_right_corner

    @classmethod
    def format_line(cls, cells: List[str]) -> str:
        pd = " " * cls.padding
        return f"{cls.v_line}{pd}{(pd + cls.v_line + pd).join(cells)}{pd}{cls.v_line}"


class TableStyleNoLines(TableStyleDefault):
    h_line = ""
    v_line = ""
    d_h_line = ""
    d_right_end = ""
    d_left_end = ""
    d_cross = ""
    cross = ""
    right_end = ""
    left_end = ""
    upper_end = ""
    down_end = ""
    top_right_corner = ""
    top_left_corner = ""
    bottom_right_corner = ""
    bottom_left_corner = ""


class TableStyleNoBorders(TableStyleDefault):
    @classmethod
    def sep(cls, col_width: List[int]) -> str:
        return cls.d_cross.join(cls.d_h_line * (w + 2 * cls.padding) for w in col_width)

    @classmethod
    def header_sep(cls, col_width: List[int]) -> str:
        return cls.cross.join(cls.h_line * (w + 2 * cls.padding) for w in col_width)

    @classmethod
    def top_line(cls, col_width: List[int]) -> str:
        return ""

    @classmethod
    def bottom_line(cls, col_width: List[int]) -> str:
        return ""

    @classmethod
    def format_line(cls, cells: List[str]) -> str:
        pd = " " * cls.padding
        return f"{pd}{(pd + cls.v_line + pd).join(cells)}{pd}"


def renter_to_text(tbl: Union[Table, SimpleTable],
                   hide_unused: bool = False, style: Type[Style] = TableStyleDefault) -> str:

    data = tbl.content(hide_unused=hide_unused)
    if not data:
        return ""

    headers = [header.header for header in tbl.headers(hide_unused=hide_unused)]
    for row in data:
        assert row is Separator or len(row) == len(headers)

    col_widths: List[Set[int]] = [{len(header)} for header in headers]

    # calculate columns with
    for row in data:
        if row is not Separator:
            assert len(row) == len(col_widths)
            for cell, cw in zip(row, col_widths):
                assert cell.colspan == 1
                assert isinstance(cell.data, str)
                cw.add(len(cell.data))

    col_width = [max(w) for w in col_widths]

    formatted_data = [style.top_line(col_width)]

    sep = style.sep(col_width)
    for idx, row in enumerate([[Cell(hdr, align=Align.center) for hdr in headers]] + data):
        if row is Separator:
            formatted_data.append(sep)
        else:
            cells = []
            assert len(row) == len(col_width)
            for cell, cw in zip(row, col_width):
                # align data
                if cell.align in (Align.center, Align.default):
                    cont = cell.data.center(cw)
                elif cell.align is Align.left:
                    cont = cell.data.ljust(cw)
                else:
                    assert cell.align is Align.right
                    cont = cell.data.rjust(cw)

                cells.append(cont)

            formatted_data.append(style.format_line(cells))

        if idx == 0:
            formatted_data.append(style.header_sep(col_width))

    formatted_data.append(style.bottom_line(col_width))
    return "\n".join(ln for ln in formatted_data if ln)
