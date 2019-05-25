from __future__ import annotations

import sys
from typing import TypeVar, Iterator, Tuple, Iterable, List, Union

from . import Align, XMLBuilder, XMLNode, RawContent, AnyXML, root_xml_node
from .table import Separator, ITable, Ok, Fail


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
        attrs = " ".join(f'{name}="{val}"' for name, val in attrs.items())
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


def embed_raw(content: AnyXML, tag: str, **attrs: str) -> RawContent:
    node = XMLNode(tag, **attrs)
    node << content
    return RawContent(node)


def ok(text: AnyXML) -> RawContent:
    return embed_raw(text, 'font', color="green")


def fail(text: AnyXML) -> RawContent:
    return embed_raw(text, 'font', color="red")


def unknown(text: AnyXML) -> RawContent:
    return embed_raw(text, 'font', color="orange")


def href(text: AnyXML, link: str) -> RawContent:
    return embed_raw(text, "a", href=link)


HTML_ALIGN_MAPPING = {
    Align.center: 'center',
    Align.left: 'left',
    Align.right: 'right',
}


def table_to_html(t: ITable, hide_unused: bool = False,
                  classes: Iterable[str] = None) -> XMLNode:
    # doc = XMLBuilder("table", **{"class": "table table-bordered table-striped table-condensed table-hover",
    #                              "style": "width: auto;"})

    content = t.content(hide_unused=hide_unused)
    headers = t.headers(hide_unused=hide_unused)
    column_width = t.columns_width(hide_unused=hide_unused)
    classes = [] if not classes else list(classes)
    for name in dir(t):
        if name.startswith("__html_") and name.endswith("__"):
            if name == '__html_classes__':
                classes.extend(t.__html_classes__.split(" "))
            else:
                assert False, f"Unknown html meta attribute {name!r}"

    doc = XMLBuilder()
    with doc.table:
        if classes:
            doc(class_=" ".join(classes))

        with doc.thead:
            with doc.tr:
                for header, width in zip(headers, column_width):
                    attrs = {}

                    if header.align is not Align.default:
                        attrs['align'] = HTML_ALIGN_MAPPING[header.align]

                    if width:
                        attrs['width'] = f"{width}%"

                    doc.th(RawContent(header.header), **attrs)

        with doc.tbody:
            for row in content:
                if row is Separator:
                    continue
                with doc.tr:
                    for cell in row:
                        args = {}
                        if cell.colspan != 1:
                            args["colspan"] = str(cell.colspan)

                        if cell.align is not Align.default:
                            args['align'] = HTML_ALIGN_MAPPING[cell.align]

                        if isinstance(cell.data, Ok):
                            v = ok(cell.data.value)
                        elif isinstance(cell.data, Fail):
                            v = fail(cell.data.value)
                        else:
                            v = RawContent(cell.data)

                        doc.td(v, **args)

    return root_xml_node(doc)


