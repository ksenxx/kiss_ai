# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Branch-coverage tests for :mod:`kiss.core.html_render`.

The console printer renders finish()'s HTML summaries with
``html_to_rich``; these tests exercise every element type the renderer
supports against real HTML inputs.
"""

from __future__ import annotations

from kiss.core.html_render import html_to_rich


def html_to_text(html: str) -> str:
    """Plain-text rendering of *html* via ``html_to_rich`` (test helper)."""
    return html_to_rich(html).plain


class TestBlocksAndHeadings:
    def test_headings_are_bold_lines(self) -> None:
        text = html_to_rich("<h1>Big</h1><p>body</p>")
        assert "Big" in text.plain
        assert "body" in text.plain
        assert text.plain.index("Big") < text.plain.index("body")
        styles = [str(span.style) for span in text.spans]
        assert any("bold" in s for s in styles)

    def test_paragraphs_separated_by_blank_line(self) -> None:
        out = html_to_text("<p>one</p><p>two</p>")
        assert "one\n\ntwo" in out

    def test_br_and_hr(self) -> None:
        out = html_to_text("<p>a<br>b</p><hr><p>c</p>")
        assert "a\nb" in out
        assert "─" in out
        assert "c" in out

    def test_blockquote_and_div(self) -> None:
        out = html_to_text("<div>x</div><blockquote>quoted</blockquote>")
        assert "x" in out
        assert "quoted" in out


class TestInlineStyles:
    def test_bold_italic_underline_code(self) -> None:
        text = html_to_rich(
            "<p><b>bb</b> <strong>ss</strong> <i>ii</i> <em>ee</em> "
            "<u>uu</u> <code>cc</code></p>"
        )
        plain = text.plain
        for token in ("bb", "ss", "ii", "ee", "uu", "cc"):
            assert token in plain
        styles = {str(span.style) for span in text.spans}
        assert any("bold" in s for s in styles)
        assert any("italic" in s for s in styles)
        assert any("underline" in s for s in styles)

    def test_link_shows_target(self) -> None:
        out = html_to_text('<p><a href="https://x.example">site</a></p>')
        assert "[https://x.example]" in out
        assert "site" in out

    def test_link_without_href(self) -> None:
        out = html_to_text("<p><a>anchor</a></p>")
        assert "anchor" in out
        assert "[" not in out

    def test_link_with_non_href_attributes(self) -> None:
        out = html_to_text('<p><a title="t" href="u">v</a></p>')
        assert "[u]" in out
        out2 = html_to_text('<p><a title="t">w</a></p>')
        assert "w" in out2
        assert "[" not in out2


class TestLists:
    def test_unordered_list_bullets(self) -> None:
        out = html_to_text("<ul><li>alpha</li><li>beta</li></ul>")
        assert "• alpha" in out
        assert "• beta" in out

    def test_ordered_list_numbering(self) -> None:
        out = html_to_text("<ol><li>first</li><li>second</li></ol>")
        assert "1. first" in out
        assert "2. second" in out

    def test_nested_list_indentation(self) -> None:
        out = html_to_text(
            "<ul><li>outer<ul><li>inner</li></ul></li></ul>"
        )
        assert "• outer" in out
        assert "  • inner" in out

    def test_li_outside_list(self) -> None:
        out = html_to_text("<li>stray</li>")
        assert "• stray" in out


class TestPreAndCode:
    def test_pre_preserves_whitespace(self) -> None:
        out = html_to_text("<pre><code>def f():\n    return 1\n</code></pre>")
        assert "def f():\n    return 1" in out

    def test_whitespace_collapsed_outside_pre(self) -> None:
        out = html_to_text("<p>a    b\n\n   c</p>")
        assert "a b c" in out


class TestTables:
    def test_table_cells_spaced(self) -> None:
        out = html_to_text(
            "<table><tr><th>H1</th><th>H2</th></tr>"
            "<tr><td>a</td><td>b</td></tr></table>"
        )
        assert "H1  H2" in out
        assert "a  b" in out


class TestSkippedContent:
    def test_script_and_style_content_dropped(self) -> None:
        out = html_to_text(
            "<p>keep</p><script>alert(1)</script><style>p{}</style>"
        )
        assert "keep" in out
        assert "alert" not in out
        assert "p{}" not in out

    def test_unbalanced_end_tags_are_tolerated(self) -> None:
        out = html_to_text("</script></ul></b><p>ok</p>")
        assert "ok" in out


class TestPlainAndEmptyInput:
    def test_plain_text_passthrough(self) -> None:
        assert html_to_text("just words") == "just words"

    def test_empty_input(self) -> None:
        assert html_to_text("") == ""

    def test_entities_unescaped(self) -> None:
        out = html_to_text("<p>a &lt; b &amp;&amp; c &gt; d</p>")
        assert "a < b && c > d" in out

    def test_details_summary(self) -> None:
        out = html_to_text(
            "<details><summary>More</summary><p>hidden</p></details>"
        )
        assert "More" in out
        assert "hidden" in out


class TestNestedInlineStyles:
    def test_nested_styles_are_composed(self) -> None:
        """Text inside <em> nested in <strong> must be bold AND italic."""
        text = html_to_rich("<p><strong>bold <em>both</em> bold</strong></p>")
        plain = text.plain
        start = plain.index("both")
        styles = {
            str(span.style)
            for span in text.spans
            if span.start <= start < span.end
        }
        assert any("bold" in s and "italic" in s for s in styles), styles

    def test_duplicate_style_tokens_deduped(self) -> None:
        """<b> inside <b> must not produce a 'bold bold' style string."""
        text = html_to_rich("<p><b>outer <b>inner</b></b></p>")
        for span in text.spans:
            assert str(span.style).split().count("bold") <= 1
