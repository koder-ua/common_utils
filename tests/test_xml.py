import pytest

from koder_utils import XMLBuilder, XMLNode, RawContent, htag


def test_simple():
    doc = XMLBuilder()
    doc.root()
    assert str(doc) == "<root />"


def test_simple2():
    doc = XMLBuilder()
    with doc.root:
        doc.a()
        doc.b()
    assert str(doc) == "<root><a /><b /></root>"


def test_simple3():
    doc = XMLBuilder()
    with doc.root:
        doc.a("abc")
        doc.b("cde", x="1")
    assert str(doc) == '<root><a>abc</a><b x="1">cde</b></root>'


def test_with():
    doc = XMLBuilder()
    with doc.root.a("abc"):
        with doc.b:
            doc("cde", x="1")

    assert str(doc) == '<root><a>abc<b x="1">cde</b></a></root>'


def test_with_nested_attrs():
    doc = XMLBuilder()
    with doc.root.a("abc"):
        with doc.b.c:
            doc("cde", x="1")

    assert str(doc) == '<root><a>abc<b><c x="1">cde</c></b></a></root>'


def test_text():
    doc = XMLBuilder()
    with doc.root.a:
        doc("111")
        doc("222")
        doc.b()
        doc("333")
        doc.c
        doc("444")
    assert str(doc) == "<root><a>111222<b />333<c />444</a></root>"


def test_lshift():
    doc1 = XMLBuilder()
    with doc1.inner_root:
        doc1("222")
        with doc1.b:
            doc1("333")

    doc2 = XMLBuilder()
    with doc2.root.a:
        doc2("111")
        doc2 << doc1

    assert str(doc2) == "<root><a>111<inner_root>222<b>333</b></inner_root></a></root>"


def test_lshift_to_tag():
    doc1 = XMLBuilder()
    with doc1.inner_root:
        doc1("222")
        with doc1.b:
            doc1("333")

    doc2 = XMLBuilder()
    with doc2.root.a:
        doc2("111")
        doc2.test << doc1

    assert str(doc2) == "<root><a>111<test><inner_root>222<b>333</b></inner_root></test></a></root>"


def test_lshift_tag():
    doc = XMLBuilder()
    with doc.root.a:
        doc("111")
        doc.test << XMLNode('inner_root', a="12")

    assert str(doc) == '<root><a>111<test><inner_root a="12" /></test></a></root>'


def test_raw_data():
    doc = XMLBuilder()
    with doc.root.a:
        doc << RawContent('<d />')

    assert str(doc) == '<root><a><d /></a></root>'


def test_node_builder():
    doc = XMLBuilder()
    with doc.root.a:
        doc << RawContent('<d />')

    assert str(doc) == '<root><a><d /></a></root>'


def test_multy_root():
    doc = XMLBuilder()
    doc.root1
    doc.root2(a="12")
    doc.root2.a
    with doc.root3:
        doc.t
    assert str(doc) == '<root1 /><root2 a="12" /><root2><a /></root2><root3><t /></root3>'


def test_incorrect_tar_attr():
    doc = XMLBuilder()

    with pytest.raises(ValueError):
        doc.root(12)

    doc.root("")

    with pytest.raises(ValueError):
        doc.root("", a=3)


def test_htag():
    assert str(htag.test("12").a.b) == '<test>12<a><b /></a></test>'
