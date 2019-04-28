#!/usr/bin/env python
from __future__ import annotations

import weakref
from typing import List, Any, Union, TypeVar, Callable, Dict, Iterator, Optional
from xml.etree.ElementTree import TreeBuilder, tostring, ElementTree
import xml.dom.minidom

__doc__ = """
XMLBuilder is tiny library build on top of ElementTree.TreeBuilder to
make xml files creation more pythonic. `XMLBuilder` use `with`
statement and attribute access to define xml document structure.

from __future__ import with_statement # only for python 2.5
from koder_utils import XMLDocument, to_etree, to_string

x = XMLDocument('root')
x.some_tag
x.some_tag_with_data('text', a='12')

with x.some_tree(a='1'):
    with x.data:
        x.mmm
        for i in range(10):
            x.node(val=str(i))

etree_node = to_etree(x) # <= return xml.etree.ElementTree object
print(to_string(x)) # <= string object

will result:

<?xml version="1.0" encoding="utf-8" ?>
<root>
    <some_tag />
    <some_tag_with_data a="12">text</some_tag_with_data>
    <some_tree a="1">
        <data>
            <mmm />
            <node val="0" />
            <node val="1" />
            <node val="2" />
            <node val="3" />
            <node val="4" />
            <node val="5" />
            <node val="6" />
            <node val="7" />
            <node val="8" />
            <node val="9" />
        </data>
    </some_tree>
</root>

Happy xml'ing.
"""


T = TypeVar('T')


class XMLDocument:
    def __init__(self, name: str, text: str = None, **attrs: str) -> None:
        self._root_tag = XMLNode(name, doc=weakref.ref(self))
        self._root_tag(text, **attrs)
        self._stack = [self._root_tag]

    def __call__(self, text: str = None, **attrs: str) -> None:
        assert self._stack, "Can't add text to empty document, open tag first"
        self._stack[-1](text, **attrs)

    def __getattr__(self, name: str) -> XMLNode:
        return getattr(self._stack[-1], name)

    def __iter__(self) -> Iterator[XMLNode]:
        yield self._root_tag

    def __lshift__(self: T, other: Union[XMLDocument, T, XMLNode]) -> T:
        self._stack[-1] << other
        return self


class XMLNode:
    def __init__(self, tag: str, doc: Optional[Callable[[], XMLDocument]] = None, text: str = None,
                 **attrs: str) -> None:

        self._doc = doc
        self._tag = tag
        self._attrs = attrs
        self._childs: List[Union[XMLNode, str]] = []

        if text:
            self._childs.append(text)

    def __call__(self: T, text: str = None, **attrs: str) -> T:
        if text:
            self._childs.append(text)
        self._attrs.update(attrs)
        return self

    def __getattr__(self: T, name: str) -> T:
        node = self.__class__(name, doc=self._doc)
        self._childs.append(node)
        return node

    def __enter__(self) -> None:
        assert self._doc
        return self._doc()._stack.append(self)

    def __exit__(self, x, y, z) -> bool:
        assert self._doc
        assert self is self._doc()._stack.pop()
        return False

    def __iter__(self) -> Iterator[Union[XMLNode, str]]:
        return iter(self._childs)

    def __lshift__(self: T, other: Union[XMLDocument, T, XMLNode]) -> T:
        if isinstance(other, XMLDocument):
            other = other._root_tag
        self._childs.append(other)
        return self


def put_to_builder(node: XMLNode, builder: Any):
    builder.start(node._tag, node._attrs)

    for child in node:
        if isinstance(child, str):
            builder.data(child)
        else:
            put_to_builder(child, builder)

    builder.end(node._tag)


def doc_to_etree(doc: XMLDocument, builder_cls: Any = TreeBuilder) -> ElementTree:
    builder = builder_cls()
    root_tag, = list(doc)
    put_to_builder(root_tag, builder)
    return builder.close()


def doc_to_bytes(doc: XMLDocument, builder_cls: Any = TreeBuilder,
                 encoding: str = "utf8", pretty: bool = False) -> bytes:
    res = tostring(doc_to_etree(doc, builder_cls), encoding=encoding)
    if pretty:
        doc2 = xml.dom.minidom.parseString(res.decode(encoding))
        res = doc2.toprettyxml().encode(encoding)
    return res


def doc_to_string(doc: XMLDocument, builder_cls: Any = TreeBuilder, pretty: bool = False) -> str:
    return doc_to_bytes(doc, builder_cls, "utf8", pretty).decode("utf8")
