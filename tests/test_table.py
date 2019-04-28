from koder_utils import Column, Table, renter_to_text, Align
from koder_utils.table import TableStyleNoLines, TableStyleNoBorders

test_table_simple_res = """
╭──────┬───╮
│  X   │ Y │
├──────┼───┤
│    1 │ 1 │
│ 2344 │ 2 │
╰──────┴───╯
"""


def test_table_simple():
    class Test(Table):
        x = Column.ei(align=Align.right)
        y = Column.s()

    t = Test()
    r = t.next_row()
    r.x = 1
    r.y = '1'

    r = t.next_row()
    r.x = 2344
    r.y = '2'

    assert renter_to_text(t).strip() == test_table_simple_res.strip()


test_separator_res = """
╭──────┬───────╮
│  X   │   tt  │
├──────┼───────┤
│    1 │   1   │
│ 2344 │   2   │
╞══════╪═══════╡
│    2 │ 25235 │
╰──────┴───────╯
"""


def test_separator():
    class Test(Table):
        x = Column.ei(align=Align.right)
        y = Column.s(header='tt')

    t = Test()
    r = t.next_row()
    r.x = 1
    r.y = '1'

    r = t.next_row()
    r.x = 2344
    r.y = '2'

    t.add_separator()

    r = t.next_row()
    r.x = 2
    r.y = '25235'

    print(renter_to_text(t, style=TableStyleNoLines))
    assert renter_to_text(t).strip() == test_separator_res.strip()



test_separator()