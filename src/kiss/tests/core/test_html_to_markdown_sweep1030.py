# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tables written with omitted ``</td>``/``</th>``/``</tr>`` end tags.

HTML lets a writer leave these end tags out and browsers imply them.
The converter used to treat a ``<td>``/``<tr>`` that started while a cell
was open as text nested in that cell, and ``</table>`` popped the table
without flushing the open cell and row, so a table lost rows or
collapsed to flat text, and the output depended on whether anything
followed the table.  Task summaries reach this code on their way into
squash-merge commit messages and the memory corpus.
"""

from __future__ import annotations

from kiss.core.html_to_markdown import html_to_markdown


def test_closing_the_table_keeps_the_open_last_row() -> None:
    assert html_to_markdown("<table><tr><td>a</td><tr><td>b</td></table>") == (
        "| a |\n| --- |\n| b |"
    )


def test_a_table_without_any_cell_or_row_end_tags_is_still_a_table() -> None:
    html = "<p>x</p><table><tr><th>H1<th>H2<tr><td>1<td>2</table>"
    assert html_to_markdown(html) == "x\n\n| H1 | H2 |\n| --- | --- |\n| 1 | 2 |"


def test_the_same_table_renders_alike_whatever_follows_it() -> None:
    table = "<table><tr><th>H1<th>H2<tr><td>1<td>2"
    expected = "| H1 | H2 |\n| --- | --- |\n| 1 | 2 |"
    assert html_to_markdown(table) == expected
    assert html_to_markdown(table + "</table>") == expected
    assert html_to_markdown(table + "</table><p>after</p>") == expected + "\n\nafter"


def test_a_row_end_tag_closes_the_open_cell() -> None:
    assert html_to_markdown("<table><tr><td>a</tr><tr><td>b</tr></table>") == (
        "| a |\n| --- |\n| b |"
    )


def test_implied_end_tags_inside_tbody() -> None:
    assert html_to_markdown("<table><tbody><tr><td>a<td>b</tbody></table>x") == (
        "| a | b |\n| --- | --- |\n\nx"
    )


def test_nested_table_text_stays_space_separated_in_the_outer_cell() -> None:
    html = "<table><tr><td>a<table><tr><td>in</td></tr></table>b</td></tr></table>"
    assert html_to_markdown(html) == "| a in b |\n| --- |"
    html = "<table><tr><td>a<table><tr><td>in<td>in2</table>b<td>c</table>"
    assert html_to_markdown(html) == "| a in in2 b | c |\n| --- | --- |"
