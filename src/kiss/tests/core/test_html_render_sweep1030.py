# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The console HTML renderer keeps the spacing the HTML has around inline tags.

``html_to_rich`` used to strip every text node and then put one space in
front of it whenever the text so far did not end in a space, so
punctuation after ``<code>``/``<b>``/``<a>`` was pushed away
(``Run pytest , then commit .``), ``foo<b>bar</b>baz`` gained spaces,
and the space before a link was lost.  It renders every ``finish``
summary in the CLI Result panel.
"""

from __future__ import annotations

import io

import yaml

from kiss.core.html_render import html_to_rich
from kiss.core.print_to_console import ConsolePrinter


def _result_panel(summary: str) -> str:
    """Render *summary* in a real console Result panel and return the output."""
    buf = io.StringIO()
    printer = ConsolePrinter(file=buf)
    payload = {"success": True, "is_continue": False, "summary": summary}
    printer.print(
        yaml.dump(payload, sort_keys=False),
        type="result",
        step_count=1,
        total_tokens=1,
        cost="$0.0100",
    )
    return buf.getvalue()


def test_result_panel_keeps_punctuation_next_to_inline_tags() -> None:
    out = _result_panel("<p>Run <code>pytest</code>, then <b>commit</b>.</p>")
    assert "Run pytest, then commit." in out


def test_space_before_a_link_survives_and_none_is_invented_after_it() -> None:
    assert html_to_rich('<p>see <a href="http://x">docs</a>.</p>').plain == (
        "see [http://x] docs."
    )


def test_adjacent_inline_runs_are_not_split_by_invented_spaces() -> None:
    assert html_to_rich("<p>foo<b>bar</b>baz</p>").plain == "foobarbaz"


def test_styled_span_covers_exactly_the_styled_word() -> None:
    text = html_to_rich("<p>a <b>bold</b> tail</p>")
    assert text.plain == "a bold tail"
    bold = [(span.start, span.end) for span in text.spans if "bold" in str(span.style)]
    assert bold == [(2, 6)]


def test_whitespace_at_line_and_cell_ends_is_dropped() -> None:
    assert html_to_rich("<p>x <br> y</p>").plain == "x\ny"
    assert html_to_rich("<p>hello <b>world</b> </p><p>next</p>").plain == (
        "hello world\n\nnext"
    )
    pretty = "<h3>Title</h3>\n<p>text\n</p>\n<ul>\n  <li>\n    item one\n  </li>\n</ul>"
    assert html_to_rich(pretty).plain == "Title\n\ntext\n\n• item one"


def test_pretty_printed_cells_keep_a_two_space_gap_and_empty_cells_keep_theirs() -> None:
    html = (
        "<table>\n<tr>\n<td>\n a \n</td>\n<td> b </td>\n</tr>\n"
        "<tr><td>c</td><td></td><td>d</td></tr>\n</table>"
    )
    assert html_to_rich(html).plain == "a  b\nc    d"


def test_preformatted_spaces_survive_line_breaks_and_the_closing_tag() -> None:
    """Only spaces left by whitespace collapsing are cropped, never ``<pre>`` text."""
    assert html_to_rich("<pre>a  <br>b</pre><p>after</p>").plain == "a  \nb\n\nafter"
    assert html_to_rich("<pre>a  </pre><p>after</p>").plain == "a  \n\nafter"
    assert html_to_rich("<pre>x</pre>y <br>z").plain == "x\ny\nz"
