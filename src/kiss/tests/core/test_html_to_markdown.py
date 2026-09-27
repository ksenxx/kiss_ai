# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for :func:`kiss.core.html_to_markdown.html_to_markdown`.

The converter turns a ``finish()`` HTML result into Markdown for git
commit messages.  Each test feeds real HTML and checks the exact
Markdown so that every element type and escaping rule is exercised.
"""

from kiss.core.html_to_markdown import html_to_markdown


class TestBlocks:
    def test_headings_and_paragraphs(self) -> None:
        html = "<h1>One</h1><h2>Two</h2><h3>Three</h3><p>first</p><p>second\n  line</p>"
        assert html_to_markdown(html) == (
            "# One\n\n## Two\n\n### Three\n\nfirst\n\nsecond line"
        )

    def test_deep_headings_div_section_and_hr(self) -> None:
        html = "<h4>a</h4><h5>b</h5><h6>c</h6><div>d</div><hr><section>e</section>"
        assert html_to_markdown(html) == (
            "#### a\n\n##### b\n\n###### c\n\nd\n\n---\n\ne"
        )

    def test_empty_heading_is_dropped(self) -> None:
        assert html_to_markdown("<h3></h3><p>after</p>") == "after"

    def test_br_breaks_line_without_paragraph(self) -> None:
        assert html_to_markdown("<p>a<br>b<br/>c</p>") == "a\nb\nc"

    def test_pre_becomes_fenced_block_and_keeps_whitespace(self) -> None:
        html = "<p>run</p><pre><code>\nx = 1\n    y = `2`\n</code></pre><p>done</p>"
        assert html_to_markdown(html) == (
            "run\n\n```\nx = 1\n    y = `2`\n```\n\ndone"
        )

    def test_pre_without_trailing_newline_is_closed(self) -> None:
        assert html_to_markdown("<pre>x</pre>") == "```\nx\n```"

    def test_stray_pre_end_tag_is_ignored(self) -> None:
        assert html_to_markdown("a</pre>b") == "ab"

    def test_unclosed_pre_is_flushed_at_end(self) -> None:
        assert html_to_markdown("<p>a</p><pre>code") == "a\n\n```\ncode\n```"

    def test_empty_pre_and_br_inside_pre(self) -> None:
        assert html_to_markdown("<pre></pre><pre>a<br>b<b>c</b></pre>") == (
            "```\n```\n\n```\na\nbc\n```"
        )

    def test_fence_grows_past_backtick_runs_in_code(self) -> None:
        html = "<pre><code>```python\nx = 1\n```</code></pre><p>after</p>"
        assert html_to_markdown(html) == "````\n```python\nx = 1\n```\n````\n\nafter"

    def test_void_meta_and_link_tags_do_not_swallow_the_body(self) -> None:
        html = "<head><meta charset='utf-8'><link rel='x'></head><meta charset='utf-8'><p>done</p>"
        assert html_to_markdown(html) == "done"

    def test_blockquote_prefixes_every_line(self) -> None:
        html = "<blockquote><p>quoted</p><p>again</p></blockquote><p>after</p>"
        assert html_to_markdown(html) == "> quoted\n>\n> again\n\nafter"

    def test_stray_blockquote_end_tag_is_ignored(self) -> None:
        assert html_to_markdown("a</blockquote>") == "a"

    def test_details_summary(self) -> None:
        html = "<details><summary>More</summary><p>hidden</p></details>"
        assert html_to_markdown(html) == "**More**\n\nhidden"

    def test_script_style_and_head_are_skipped(self) -> None:
        html = (
            "<head><title>t</title><style>p{}</style></head>"
            "<p>x<script>alert(1)</script>y</p>"
        )
        assert html_to_markdown(html) == "xy"

    def test_stray_skip_end_tag_does_not_underflow(self) -> None:
        assert html_to_markdown("</script><p>ok</p>") == "ok"

    def test_tags_nested_inside_skipped_element_are_ignored(self) -> None:
        assert html_to_markdown("<head><b>x</b><br></head>ok") == "ok"

    def test_pre_keeps_text_verbatim_and_ignores_inline_markup(self) -> None:
        assert html_to_markdown("<pre>\n\na<b>b</b>\nc</pre>") == "```\n\nab\nc\n```"


class TestLists:
    def test_unordered_and_nested_lists(self) -> None:
        html = "<ul><li>one<ul><li>nested</li></ul></li><li>two</li></ul>"
        assert html_to_markdown(html) == "- one\n  - nested\n- two"

    def test_ordered_list_with_paragraphs_and_code_block(self) -> None:
        html = (
            "<ol><li>a</li><li><p>b</p><p>more</p></li>"
            "<li>c<pre>x = 1\n</pre></li></ol>"
        )
        assert html_to_markdown(html) == (
            "1. a\n2. b\n\n   more\n\n3. c\n\n   ```\n   x = 1\n   ```"
        )

    def test_heading_blockquote_and_pre_stay_inside_the_list_item(self) -> None:
        html = (
            "<ol><li><p>first</p><h3>second</h3><blockquote><p>q1</p><p>q2</p>"
            "</blockquote><pre>a\n\nb\n</pre>third</li><li>fourth</li></ol>"
        )
        assert html_to_markdown(html) == (
            "1. first\n\n   ### second\n\n   > q1\n   >\n   > q2\n\n"
            "   ```\n   a\n\n   b\n   ```\n\n   third\n2. fourth"
        )

    def test_blockquote_as_first_child_continues_the_marker_line(self) -> None:
        html = "<ul><li><blockquote><p>quoted</p></blockquote>after</li><li>next</li></ul>"
        assert html_to_markdown(html) == "- > quoted\n\n  after\n- next"

    def test_syntax_right_after_a_list_marker_is_escaped(self) -> None:
        html = "<ul><li># not a heading</li><li>1. nor a list</li></ul>"
        assert html_to_markdown(html) == "- \\# not a heading\n- 1\\. nor a list"

    def test_list_item_with_line_break_indents_continuation(self) -> None:
        assert html_to_markdown("<ul><li>a<br>b</li></ul>") == "- a\n  b"

    def test_bare_li_gets_an_implicit_list(self) -> None:
        assert html_to_markdown("<li>bare</li>") == "- bare"

    def test_empty_items_are_dropped(self) -> None:
        assert html_to_markdown("<ul><li></li><li>x</li><li></li></ul>") == "- x"

    def test_trailing_empty_item_is_dropped(self) -> None:
        assert html_to_markdown("<li>x</li><li>") == "- x"

    def test_stray_list_end_tag_is_ignored(self) -> None:
        assert html_to_markdown("a</ul>b") == "a\n\nb"

    def test_list_after_paragraph_and_text_after_list(self) -> None:
        html = "<p>intro</p><ul><li>x</li></ul><p>outro</p>"
        assert html_to_markdown(html) == "intro\n\n- x\n\noutro"


class TestInline:
    def test_bold_italic_and_whitespace_moved_outside_markers(self) -> None:
        html = "<p>a<b> bold </b>b <i>it</i> <strong>s</strong> <em>e</em></p>"
        assert html_to_markdown(html) == "a **bold** b *it* **s** *e*"

    def test_empty_inline_element_emits_nothing(self) -> None:
        assert html_to_markdown("<p>a<b> </b>b<em></em></p>") == "a b"

    def test_unmatched_inline_end_tag_is_ignored(self) -> None:
        assert html_to_markdown("<p>a</b>b</p>") == "ab"

    def test_unclosed_inner_marker_is_dropped_when_outer_closes(self) -> None:
        assert html_to_markdown("<p><b>x<i>y</b>z</p>") == "**xy**z"

    def test_code_span_and_backtick_fencing(self) -> None:
        html = "<p><code>foo_bar</code> <code> spaced `tick` </code><code> </code></p>"
        assert html_to_markdown(html) == "`foo_bar` `` spaced `tick` ``"

    def test_code_inside_pre_is_not_a_span(self) -> None:
        assert html_to_markdown("<pre><code>a `b`</code></pre>") == "```\na `b`\n```"

    def test_links(self) -> None:
        html = (
            "<p>See <a href='http://x.y/z'>the docs</a>, "
            "<a href='http://a.b'>http://a.b</a>, <a href='http://c'></a> "
            "and <a>no href</a>.</p>"
        )
        assert html_to_markdown(html) == (
            "See [the docs](http://x.y/z), http://a.b, http://c and no href."
        )

    def test_link_destinations_and_labels_with_special_characters(self) -> None:
        html = (
            "<p><a href='https://e.org/foo)bar'>x</a> <a href='./my report.html'>r</a> "
            "<a href='http://q'>a]b</a> <a href='http://a<b>'>t</a></p>"
        )
        assert html_to_markdown(html) == (
            "[x](<https://e.org/foo)bar>) [r](<./my report.html>) [a\\]b](http://q) "
            "[t](<http://a\\<b\\>>)"
        )

    def test_emphasis_and_links_inside_code_are_plain(self) -> None:
        html = "<p><code>a<b>b</b>c</code> <code>x<a href='u'>l</a></code></p>"
        assert html_to_markdown(html) == "`abc` `xl`"

    def test_adjacent_emphasis_runs_are_merged(self) -> None:
        html = "<p><b>foo</b><b>bar</b> <b>x</b><i>y</i> <i>p</i><b>q</b> <b>a</b> <b>b</b></p>"
        assert html_to_markdown(html) == "**foobar** **x***y* *p***q** **a** **b**"

    def test_escaped_literal_star_is_not_merged_with_emphasis(self) -> None:
        html = "<p>literal *<i>x</i> **<b>y</b></p>"
        assert html_to_markdown(html) == "literal \\**x* \\*\\***y**"

    def test_unbalanced_code_inside_bold_keeps_escaping_afterwards(self) -> None:
        html = "<p><b><code>text</b></code></p><p>*literal* &lt;h3&gt;</p>"
        assert html_to_markdown(html) == "**text**\n\n\\*literal\\* \\<h3>"

    def test_image_alt_text(self) -> None:
        assert html_to_markdown("<p><img src='a.png' alt='pic'><img src='b.png'></p>") == "pic"


class TestTables:
    def test_table_with_header_separator_and_escaped_pipes(self) -> None:
        html = (
            "<table><thead><tr><th>A</th><th>B|C</th></tr></thead>"
            "<tbody><tr><td>1</td><td><b>2</b></td></tr></tbody></table><p>x</p>"
        )
        assert html_to_markdown(html) == (
            "| A | B\\|C |\n| --- | --- |\n| 1 | **2** |\n\nx"
        )

    def test_row_outside_table_and_cell_outside_row(self) -> None:
        assert html_to_markdown("<tr><td>a</td></tr>") == "a"
        assert html_to_markdown("<td>a</td>") == "a"
        assert html_to_markdown("<table>x</td></tr></table>") == "x"
        assert html_to_markdown("a</table>b") == "ab"
        assert html_to_markdown("<table></table><p>z</p>") == "z"
        assert html_to_markdown("<table><td>a</td></table>") == "a"
        assert html_to_markdown("<table><tr></tr><tr><td>x</td><tr>y</tr></tr></table>") == (
            "y\n\n| x |\n| --- |"
        )

    def test_nested_table_flows_into_the_outer_cell(self) -> None:
        html = (
            "<table><tr><td>OUTER</td><td><table><tr><td>INNER</td><td>X</td></tr>"
            "<tr><td>IN2</td></tr></table></td></tr></table>"
        )
        assert html_to_markdown(html) == "| OUTER | INNER X IN2 |\n| --- | --- |"

    def test_unclosed_tables_are_flushed_at_end_of_input(self) -> None:
        assert html_to_markdown("<table><tr><td>Tests</td><td>84 passed</td></tr>") == (
            "| Tests | 84 passed |\n| --- | --- |"
        )
        assert html_to_markdown("<table><tr><td>O<table><tr><td>I</td>") == "| O I |\n| --- |"

    def test_table_as_first_child_of_a_list_item_stays_in_the_item(self) -> None:
        html = (
            "<ul><li><table><tr><td>Tests</td><td><code>84</code></td></tr></table>"
            "tail</li><li>Next</li></ul>"
        )
        assert html_to_markdown(html) == (
            "- | Tests | `84` |\n  | --- | --- |\n\n  tail\n- Next"
        )

    def test_pre_inside_a_cell_becomes_code_spans_per_line(self) -> None:
        html = "<table><tr><td><pre>uv run pytest\n\necho done</pre></td></tr></table>"
        assert html_to_markdown(html) == "| `uv run pytest` \u23ce `echo done` |\n| --- |"

    def test_short_rows_are_padded_to_the_widest_row(self) -> None:
        html = (
            "<table><tr><th colspan=2>Results</th></tr>"
            "<tr><td>Tests</td><td>84 passed</td></tr></table>"
        )
        assert html_to_markdown(html) == (
            "| Results |  |\n| --- | --- |\n| Tests | 84 passed |"
        )


class TestEscaping:
    def test_inline_markdown_syntax_is_escaped(self) -> None:
        html = "<p>*star* `tick` back\\slash __init__ snake_case &lt;h3&gt; a &lt; b</p>"
        assert html_to_markdown(html) == (
            "\\*star\\* \\`tick\\` back\\\\slash \\_\\_init\\_\\_ snake_case \\<h3> a < b"
        )

    def test_line_start_syntax_is_escaped(self) -> None:
        html = "<p>1. not a list<br># not heading<br>- not bullet<br>&gt; not quote<br>---</p>"
        assert html_to_markdown(html) == (
            "1\\. not a list\n\\# not heading\n\\- not bullet\n\\> not quote\n\\---"
        )

    def test_literal_links_images_and_entities_stay_literal(self) -> None:
        html = "<p>[gone](https://example.org) ![img](x.png) &amp;copy; &amp;#169; a &amp; b</p>"
        assert html_to_markdown(html) == (
            "\\[gone\\](https://example.org) !\\[img\\](x.png) \\&copy; \\&#169; a & b"
        )

    def test_plain_text_passes_through(self) -> None:
        assert html_to_markdown("plain text only") == "plain text only"
        assert html_to_markdown("") == ""

    def test_whitespace_between_blocks_is_dropped(self) -> None:
        assert html_to_markdown("<p>a</p>\n   \n<p>b</p>  ") == "a\n\nb"

    def test_entities_are_decoded(self) -> None:
        assert html_to_markdown("<p>Tom &amp; Jerry &#8594; done</p>") == "Tom & Jerry → done"
