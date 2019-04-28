from koder_utils import XMLDocument, doc_to_string
from koder_utils.xmlbuilder import XMLNode


def no_header(doc: XMLDocument) -> str:
    return doc_to_string(doc).split("\n", 1)[1]


def test_simple():
    doc = XMLDocument('root')
    assert no_header(doc) == "<root />"


def test_simple2():
    doc = XMLDocument('root')
    doc.a
    doc.b
    assert no_header(doc) == "<root><a /><b /></root>"


def test_simple3():
    doc = XMLDocument('root')
    doc.a("abc")
    doc.b("cde", x="1")
    assert no_header(doc) == '<root><a>abc</a><b x="1">cde</b></root>'


def test_with():
    doc = XMLDocument('root')
    with doc.a("abc"):
        with doc.b:
            doc("cde", x="1")

    assert no_header(doc) == '<root><a>abc<b x="1">cde</b></a></root>'


def test_with_nested_attrs():
    doc = XMLDocument('root')
    with doc.a("abc"):
        with doc.b.c:
            doc("cde", x="1")

    assert no_header(doc) == '<root><a>abc<b><c x="1">cde</c></b></a></root>'


def test_text():
    doc = XMLDocument('root')
    with doc.a:
        doc("111")
        doc("222")
        doc.b
        doc("333")
        doc.c
        doc("444")
    assert no_header(doc) == "<root><a>111222<b />333<c />444</a></root>"


def test_lshift():
    doc2 = XMLDocument('inner_root')
    doc2("222")
    with doc2.b:
        doc2("333")

    doc1 = XMLDocument('root')
    with doc1.a:
        doc1("111")
        doc1 << doc2

    assert no_header(doc1) == "<root><a>111<inner_root>222<b>333</b></inner_root></a></root>"


def test_lshift_to_tag():
    doc2 = XMLDocument('inner_root')
    doc2("222")
    with doc2.b:
        doc2("333")

    doc1 = XMLDocument('root')
    with doc1.a:
        doc1("111")
        doc1.test << doc2

    assert no_header(doc1) == "<root><a>111<test><inner_root>222<b>333</b></inner_root></test></a></root>"


def test_lshift_tag():
    doc1 = XMLDocument('root')
    with doc1.a:
        doc1("111")
        doc1.test << XMLNode('inner_root', a="12")

    assert no_header(doc1) == '<root><a>111<test><inner_root a="12" /></test></a></root>'
