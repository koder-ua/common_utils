#!/usr/bin/env python
from __future__ import annotations

import abc
import weakref
from typing import List, Any, Union, TypeVar, Callable, Iterator, Optional, NewType, Mapping, Dict
from xml.sax.saxutils import quoteattr, escape


T = TypeVar('T')


class IXMLBuilder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, text: Union[RawContent, str] = None, **attrs: str) -> None:
        ...

    @abc.abstractmethod
    def __getattr__(self, name: str) -> XMLNode:
        ...

    @abc.abstractmethod
    def __iter__(self) -> Iterator[XMLNode]:
        ...

    @abc.abstractmethod
    def __lshift__(self: T, other: AnyXML) -> T:
        ...

    def __ilshift__(self, other: AnyXML) -> None:
        self << other

    @abc.abstractmethod
    def _to_str(self, pretty: bool = False, step: str = "    ", level: int = 0) -> str:
        ...


class RawContent:
    def __init__(self, content: AnyXML) -> None:
        self.content = str(content)

    def __add__(self, other: AnyXML) -> RawContent:
        return RawContent(self.content + str(other))

    def __str__(self) -> str:
        return self.content


def prepare_childs(obj: AnyXML) -> Iterator[XMLDocPart]:
    if isinstance(obj, XMLBuilder):
        yield from obj._roots
    elif isinstance(obj, str):
        yield escape(obj)
    else:
        assert isinstance(obj, (XMLNode, RawContent)), f"Can't add child of type {type(obj).__name__} to xml doc"
        yield obj


def check_and_fix_tag_params(name: str, text: Optional[Union[RawContent, str]], attrs: Dict[str, str]) -> None:
    if not isinstance(name, str):
        raise ValueError(f"Tag name must be string, not {type(name).__name__}")

    if text is not None and not isinstance(text, (str, RawContent)):
        raise ValueError(f"Text must be string, not {type(text).__name__}")

    for attr_name, attr_val in attrs.items():
        if not isinstance(attr_name, str):
            raise ValueError(f"Attr name must be string, not {type(attr_name).__name__}")
        if not isinstance(attr_val, str):
            raise ValueError(f"Attr value must be string, not {type(attr_val).__name__}")

    if "class_" in attrs:
        attrs['class'] = attrs.pop('class_')


class XMLBuilder(IXMLBuilder):
    """
    Document builder and pointer to current build site
    """

    def __init__(self) -> None:
        self._stack: List[XMLNode] = []
        self._roots: List[XMLDocPart] = []

    def __call__(self, text: Union[RawContent, str] = None, **attrs: str) -> None:
        check_and_fix_tag_params("", text, attrs)
        assert self._stack, "Can't set attrs to empty document, open tag first"
        self._stack[-1](text, **attrs)

    def __getattr__(self, name: str) -> XMLNode:
        if self._stack:
            return getattr(self._stack[-1], name)
        node = XMLNode(name, doc_ref=weakref.ref(self))
        self._roots.append(node)
        return node

    def __iter__(self) -> Iterator[XMLDocPart]:
        return iter(self._roots)

    def __lshift__(self, other: AnyXML) -> XMLBuilder:
        if self._stack:
            self._stack[-1] << other
        else:
            self._roots.extend(prepare_childs(other))
        return self

    def __str__(self) -> str:
        return "".join(map(str, self._roots))

    def _to_str(self, pretty: bool = False, step: str = "    ", level: int = 0) -> str:
        prefix = level * step if pretty else ""
        return "\n".join(ch._to_str(pretty, step, level) if hasattr(ch, "_to_str") else prefix + str(ch)
                         for ch in self._roots)


class XMLNode(IXMLBuilder):
    __slots__ = ("_doc_ref", "_tag", "_attrs", "_childs")

    def __init__(self, tag: str, doc_ref: Optional[Callable[[], XMLBuilder]] = None, text: str = None,
                 **attrs: str) -> None:

        check_and_fix_tag_params(tag, text, attrs)

        self._doc_ref = doc_ref
        self._tag = tag
        self._attrs = attrs
        self._childs: List[XMLDocPart] = [text] if text else []

    def __call__(self, text: Union[RawContent, str] = None, **attrs: str) -> XMLNode:
        check_and_fix_tag_params("", text, attrs)

        if isinstance(text, str):
            self._childs.append(escape(text))
        else:
            if text is not None:
                assert isinstance(text, RawContent), type(text)
                self._childs.append(text)
        self._attrs.update(attrs)
        return self

    def __getattr__(self, name: str) -> XMLNode:
        node = self.__class__(name, doc_ref=self._doc_ref)
        self._childs.append(node)
        return node

    def __enter__(self) -> None:
        assert self._doc_ref
        return self._doc_ref()._stack.append(self)

    def __exit__(self, x, y, z) -> bool:
        assert self._doc_ref
        assert self is self._doc_ref()._stack.pop()
        return False

    def __iter__(self) -> Iterator[XMLDocPart]:
        return iter(self._childs)

    def __lshift__(self, other: AnyXML) -> XMLNode:
        self._childs.extend(prepare_childs(other))
        return self

    def __str__(self) -> str:
        return to_string(self)

    def _to_str(self, pretty: bool = False, step: str = "    ", level: int = 0) -> str:
        prefix = step * level if pretty else ""
        ch_params = dict(pretty=pretty, step=step, level=level + 1)

        if self._attrs:
            attrs = " " + " ".join(f'{name}={quoteattr(value)}' for name, value in self._attrs.items())
        else:
            attrs = ""

        open = f"{prefix}<{self._tag}{attrs}"

        if self._childs:
            close = f"</{self._tag}>"
            childs = []
            for ch in self._childs:
                if hasattr(ch, '_to_str'):
                    childs.append(ch._to_str(**ch_params))
                else:
                    childs.append(prefix + str(ch))

            return f"{open}>\n{childs}\n{prefix}{close}"

        return f"{open} />"


XMLDocPart = Union[str, XMLNode, RawContent]
AnyXML = Union[XMLDocPart, XMLBuilder]


def to_string(obj: AnyXML, pretty: bool = False, step: str = "    ", level: int = 0) -> str:
    if isinstance(obj, (str, RawContent)):
        return str(obj)

    assert hasattr(obj, '_to_str'), f"Can't convert object of type {type(obj).__name__}"
    return obj._to_str(pretty, step, level)


class SimpleBuilder:
    def __init__(self, *, tag: XMLNode = None) -> None:
        self._root_tag = tag
        self._curr_tag = tag

    def __getattr__(self, name: str) -> SimpleBuilder:
        if self._curr_tag:
            self._curr_tag = getattr(self._curr_tag, name)
            return self
        else:
            return self.__class__(tag=XMLNode(name))

    def __call__(self, text: Union[RawContent, str] = None, **attrs: str) -> SimpleBuilder:
        self._curr_tag(text, **attrs)
        return self

    def __str__(self) -> str:
        return str(self._root_tag)

    def __invert__(self) -> RawContent:
        return RawContent(self)


def root_xml_node(doc: XMLBuilder) -> XMLNode:
    assert len(doc._roots) == 1
    assert isinstance(doc._roots[0], XMLNode)
    return doc._roots[0]


htag = SimpleBuilder()
