import sys
from typing import TypeVar, Iterator, Tuple, Union

from koder_utils.table import Separator
from . import Align, XMLDocument, Table, XMLNode, SimpleTable


assert sys.version_info >= (3, 6), "This python module must run on 3.6, it requires buitin dict ordering"

T = TypeVar('T')


def do_to_html(node: XMLNode, level: int) -> Iterator[Tuple[int, str]]:
    childs = list(node)
    tg = node._tag
    attrs = node._attrs

    if tg == 'br':
        assert attrs == {}
        assert not childs
        yield level, "<br>"
    else:
        attrs = " ".join(f'{name if name != "_class" else "class"}="{val}"' for name, val in attrs.items())
        if len(childs) == 0:
            if attrs:
                attrs += " "
            yield f"<{tg} {attrs}/>", level
        else:
            yield f"<{tg} {attrs}>"
            for child in childs:
                if isinstance(child, str):
                    yield level + 1, child
                else:
                    assert isinstance(child, XMLNode)
                    yield from do_to_html(child, level + 1)


def doc_to_html(doc: XMLDocument, step: str = "") -> str:
    res = ""
    root, = list(doc)
    for level, data in do_to_html(root, 0):
        res += step * level + data
    return res


def ok(text: str) -> XMLNode:
    return XMLNode("font", color="green")(text)


def fail(text: str) -> XMLNode:
    return XMLNode("font", color="red")(text)


def unknown() -> XMLNode:
    return XMLNode("font", color="orange")("???")


def href(text: str, link: str) -> XMLNode:
    return XMLNode("a", href=link)(text)


HTML_ALIGN_MAPPING = {
    Align.center: 'center',
    Align.left: 'left',
    Align.right: 'right',
    Align.default: 'right',
}


def table_to_html(t: Union[Table, SimpleTable], hide_unused: bool = False) -> XMLDocument:
    # doc = XMLBuilder("table", **{"class": "table table-bordered table-striped table-condensed table-hover",
    #                              "style": "width: auto;"})

    content = t.content(hide_unused=hide_unused)
    headers = t.headers(hide_unused=hide_unused)

    doc = XMLDocument('table')

    with doc.thead:
        with doc.tr:
            for header in headers:
                doc.th(header)

    with doc.tbody:
        for row in content:
            if row is Separator:
                continue
            with doc.tr:
                for cell in row:
                    doc.td(cell.data, align=HTML_ALIGN_MAPPING[cell.align], colspan=cell.colspan)

    return doc
