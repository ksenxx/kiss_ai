# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for :func:`kiss.core.html_to_markdown.html_to_markdown`.

The converter turns a ``finish()`` HTML result into Markdown for git
commit messages.  Each test feeds real HTML and checks the exact
Markdown so that every element type and escaping rule is exercised.
"""

from collections.abc import Iterator
from html import escape
from html.parser import HTMLParser
from time import perf_counter

import pytest
from markdown_it import MarkdownIt

from kiss.core.html_to_markdown import html_to_markdown


@pytest.mark.parametrize(
    "html,rendered",
    [
        ("<p><span>~</span><span>~</span><span>~</span> x</p><p>end</p>",
         "<p>~~~ x</p>\n<p>end</p>\n"),
        ("<p>a <span>~</span><span>~</span>b<span>~</span><span>~</span> c</p>",
         "<p>a ~~b~~ c</p>\n"),
        ('<a href="">+</a>', "<p>+</p>\n"),
        ('<a href="1.">1.</a> x', "<p>1. x</p>\n"),
        ('<a href="+"></a>', "<p>+</p>\n"),
        ('<p>old<br><a href="1.">1.</a> x</p>', "<p>old\n1. x</p>\n"),
        ('<ul><li><a href="1.">1.</a> x</li></ul>',
         "<ul>\n<li>1. x</li>\n</ul>\n"),
        ('<p><img alt="~"><img alt="~"><img alt="~"> x</p>', "<p>~~~ x</p>\n"),
        ("<b>a*</b>b", "<p><strong>a*</strong>b</p>\n"),
        ("<b>a:</b><i>b</i>c", "<p><strong>a:</strong><em>b</em>c</p>\n"),
        ("<i>a:</i><b>b</b>c", "<p><em>a:</em><strong>b</strong>c</p>\n"),
        ("<table><tr><th><b>a:</b></th><th><code>12345678</code></th></tr></table>",
         "<table>\n<thead>\n<tr>\n<th><strong>a:</strong></th>\n"
         "<th><code>12345678</code></th>\n</tr>\n</thead>\n</table>\n"),
        ("<b><code>a</code></b><b><code>b</code></b>",
         "<p><strong><code>a</code></strong><strong><code>b</code></strong></p>\n"),
        ("<b><code>a</code></b><b><code>b</code></b>c",
         "<p><strong><code>a</code></strong><strong><code>b</code></strong>c</p>\n"),
        ("a<b>(b)</b>", "<p>a<strong>(b)</strong></p>\n"),
        ("<b>a:</b>b", "<p><strong>a:</strong>b</p>\n"),
        ("a<i>(b)</i>c", "<p>a<em>(b)</em>c</p>\n"),
        ("a<b><code>b</code></b>c", "<p>a<strong><code>b</code></strong>c</p>\n"),
        ('<b>a:</b><a href="b">b</a>', "<p><strong>a:</strong>b</p>\n"),
        ('a<a href="/x"><b>(b)</b></a>',
         '<p>a<a href="/x"><strong>(b)</strong></a></p>\n'),
        ("a<b><i>(b)</i></b>", "<p>a<strong><em>(b)</em></strong></p>\n"),
        ("<b><i>a:</i></b>b", "<p><strong><em>a:</em></strong>b</p>\n"),
        ('<p><a href="~"></a><a href="~"></a><a href="~"></a> x</p><p>end</p>',
         "<p>~~~ x</p>\n<p>end</p>\n"),
        ('<ul><li>old<br><a href="1.">1.</a> x</li></ul>',
         "<ul>\n<li>old\n1. x</li>\n</ul>\n"),
        ('<ul><li><p>old</p><p><a href="1.">1.</a> x</p></li></ul>',
         "<ul>\n<li>\n<p>old</p>\n<p>1. x</p>\n</li>\n</ul>\n"),
        # An emphasis that starts where another one ended: a merge takes the closer
        # between them away, and the two markers are one run otherwise.
        ("<b>a:</b><b><i>b</i></b>q", "<p><strong>a:<em>b</em></strong>q</p>\n"),
        ("<i>a:</i><i><b>b</b></i>q", "<p><em>a:<strong>b</strong></em>q</p>\n"),
        ("<b>:</b><i><b>a</b></i>b", "<p><strong>:</strong><em><strong>a</strong></em>b</p>\n"),
        ("<b>a</b><i>:</i>", "<p><strong>a</strong><em>:</em></p>\n"),
        ("<b>(</b><i>)</i>a", "<p><strong>(</strong><em>)</em>a</p>\n"),
        ("<b>a</b><i><code>c</code></i>", "<p><strong>a</strong><em><code>c</code></em></p>\n"),
        ('<i>a</i><b><a href="/x">a</a></b>',
         '<p><em>a</em><strong><a href="/x">a</a></strong></p>\n'),
        ("<b>a</b><b><code>c</code></b><code>d</code>",
         "<p><strong>a<code>c</code></strong><code>d</code></p>\n"),
        ("<b>a</b><b><i>:</i></b>", "<p><strong>a</strong><strong><em>:</em></strong></p>\n"),
        ('<b>!</b><b><a href="/x">a</a></b>', '<p><strong>!<a href="/x">a</a></strong></p>\n'),
        # Three asterisks that open leave one or two behind their first closer:
        # an opener inside a word would close those.
        ("<b>a</b><i>b<b>c</b></i>", "<p><strong>a</strong><em>b<strong>c</strong></em></p>\n"),
        ("<b>x</b><i>y</i><i>z<b>w</b></i>",
         "<p><strong>x</strong><em>y</em><em>z<strong>w</strong></em></p>\n"),
        ("<b><i>a</i>b<i>c</i></b>d", "<p><strong><em>a</em>b<em>c</em></strong>d</p>\n"),
        ("<b>a</b><i><b>b</b>c</i>d", "<p><strong>a</strong><em><strong>b</strong>c</em>d</p>\n"),
        ("<b>a<i>b</i></b><i>c</i>d", "<p><strong>a<em>b</em></strong><em>c</em>d</p>\n"),
        ("<b>a<i>b</i></b><b>c</b>d", "<p><strong>a<em>b</em></strong><strong>c</strong>d</p>\n"),
        ("a<i><b>b</b>c<b>d</b></i>", "<p>a<em><strong>b</strong>c<strong>d</strong></em></p>\n"),
        ("<i><b>a</b></i><b><i>b</i>c<i>d</i></b>",
         "<p><em><strong>a</strong></em><strong><em>b</em>c<em>d</em></strong></p>\n"),
        ("<i><b>a</b></i><b><i>b</i>:<i>:c</i></b>",
         "<p><em><strong>a</strong></em><strong><em>b</em>:<em>:c</em></strong></p>\n"),
        ('<b>x<a href="/x">y<i>z</i></a><b>p<i>q</i></b></b>',
         '<p><strong>x<a href="/x">y<em>z</em></a>p<em>q</em></strong></p>\n'),
        # A character reference next to an underscore that was inside a word.
        ("<b><i>:</i>a_b</b>", "<p><strong><em>:</em>a_b</strong></p>\n"),
        ("x_a<i>:</i> _b<i>:</i>", "<p>x_a<em>:</em> _b<em>:</em></p>\n"),
        ("<i><b>1</b></i><i>a<b>_</b></i>",
         "<p><em><strong>1</strong></em><em>a<strong>_</strong></em></p>\n"),
        # Two text runs that are a tag or an entity only together.
        ("&lt;<i></i>a&gt; &amp;<i></i>lt;", "<p>&lt;a&gt; &amp;lt;</p>\n"),
        # Adjacent runs of one emphasis are not joined where their texts would be one.
        ("<b>&lt;</b><b>a&gt;</b>", "<p><strong>&lt;</strong><strong>a&gt;</strong></p>\n"),
        ("<i>&amp;</i><i>lt;</i>", "<p><em>&amp;</em><em>lt;</em></p>\n"),
        ("<b>&lt;1</b><b>@x.y&gt;</b>", "<p><strong>&lt;1</strong><strong>@x.y&gt;</strong></p>\n"),
        ("<b>~</b><b>~a~</b><b>~</b>",
         "<p><strong>~</strong><strong>~a~</strong><strong>~</strong></p>\n"),
        # A closer right behind the underscore closer of such a run needs a boundary too.
        ("<b><i>&lt;</i><i>b</i></b>1", "<p><strong><em>&lt;</em><em>b</em></strong>1</p>\n"),
        # The line a <br> starts in a list item is a line start too: its text is no block.
        ("<ul><li>Run it<br># a comment</li></ul>", "<ul>\n<li>Run it\n# a comment</li>\n</ul>\n"),
        ("<ol><li>Steps:<br>1. first<br>- second<br>&gt; third<br>===</li></ol>",
         "<ol>\n<li>Steps:\n1. first\n- second\n&gt; third\n===</li>\n</ol>\n"),
        # The text of an image is text: no block at a line start, verbatim in code.
        ('<p><img alt="# x"></p><ul><li><img alt="1. x"> y</li></ul>',
         "<p># x</p>\n<ul>\n<li>1. x y</li>\n</ul>\n"),
        ('<code><img alt="a*b"></code>', "<p><code>a*b</code></p>\n"),
        # A heading is one line, and no line break follows a marker.
        ("<h3>Title<br>Subtitle</h3><h3><br>Next</h3>", "<h3>Title Subtitle</h3>\n<h3>Next</h3>\n"),
        ("<ul><li><br><br>text</li></ul>", "<ul>\n<li>text</li>\n</ul>\n"),
        # A blank line would end the inline element it is in.
        ("<p><b>a<br><br>b</b></p>", "<p><strong>a\nb</strong></p>\n"),
        # Two text runs that are a list marker only together.
        ("<p>1<span></span>. x</p><p>1<i></i>)</p><ul><li>1<i></i>1.</li></ul>",
         "<p>1. x</p>\n<p>1)</p>\n<ul>\n<li>11.</li>\n</ul>\n"),
        # An exclamation mark that is escaped already stays so before a link.
        ('&lt;<i></i>!<a href="/u">a</a>', '<p>&lt;!<a href="/u">a</a></p>\n'),
        ('<b>&lt;<i></i>!</b><b><a href="/u">a</a></b>',
         '<p><strong>&lt;!<a href="/u">a</a></strong></p>\n'),
        # What follows a nested list in its item is not the end of that list's last item.
        ("<ul><li>x<ul><li>y</li></ul>tail <b>z</b></li><li>w<ul><li>v</li></ul><b>u</b></li></ul>",
         "<ul>\n<li>\n<p>x</p>\n<ul>\n<li>y</li>\n</ul>\n<p>tail <strong>z</strong></p>\n</li>\n"
         "<li>\n<p>w</p>\n<ul>\n<li>v</li>\n</ul>\n<p><strong>u</strong></p>\n</li>\n</ul>\n"),
        # An empty heading goes alone: the marker of the item it is in stays.
        ("<ol><li><h3></h3>text</li><li>b</li></ol>", "<ol>\n<li>text</li>\n<li>b</li>\n</ol>\n"),
        # An ordered list that does not count from 1 cannot start right below text.
        ('<ul><li>x<ol start="5"><li>five</li></ol></li>'
         "<li>y<ol><li></li><li>two</li></ol></li></ul>",
         '<ul>\n<li>\n<p>x</p>\n<ol start="5">\n<li>five</li>\n</ol>\n</li>\n'
         '<li>\n<p>y</p>\n<ol start="2">\n<li>two</li>\n</ol>\n</li>\n</ul>\n'),
        # An item that starts with a nested list keeps its own marker: the list stays nested.
        ("<ul><li><ul><li>a</li><li>b</li></ul></li><li>c</li></ul>",
         "<ul>\n<li>\n<ul>\n<li>a</li>\n<li>b</li>\n</ul>\n</li>\n<li>c</li>\n</ul>\n"),
        ("<ol><li><ul><li>a</li></ul></li><li>b</li></ol>",
         "<ol>\n<li>\n<ul>\n<li>a</li>\n</ul>\n</li>\n<li>b</li>\n</ol>\n"),
        # ... and when that list turns out to be empty.
        ("<ul><li><ul></ul>text</li><li><ol><li></li></ol>more</li></ul>",
         "<ul>\n<li>text</li>\n<li>more</li>\n</ul>\n"),
        # The marker of an item that an empty blockquote starts is not taken for an empty one.
        ("<ul><li><blockquote></blockquote></li><li>a</li></ul>",
         "<ul>\n<li>\n<blockquote></blockquote>\n</li>\n<li>\n<p>a</p>\n</li>\n</ul>\n"),
        # The end of a list ends the empty heading left open in its last item, and a heading
        # that is never closed ends with its line: a <br> below it is a line break again.
        ("<ul><li>a</li><li><h3></ul><p>text<br>next</p>",
         "<ul>\n<li>a</li>\n</ul>\n<p>text\nnext</p>\n"),
        ("<h3>Title<p>text<br>next</p>", "<h3>Title</h3>\n<p>text\nnext</p>\n"),
        # The heading is the line its marker is on, whatever was written and taken back since:
        # a table cell that held a heading.
        ("<table><tr><td><h3>x</td>text<br>next</tr></table>",
         "<p>text\nnext</p>\n<table>\n<thead>\n<tr>\n<th>### x</th>\n</tr>\n</thead>\n</table>\n"),
        # A code span is one line: a <br> in it is the space that a renderer makes of a line
        # ending there, so that no code text starts a line as a heading, an item, a rule or a quote.
        ("<code># x<br></code><br>y", "<p><code># x</code>\ny</p>\n"),
        ("<code>a<br>- b</code> <code>x<br>---</code> <code>y<br>&gt; z</code>",
         "<p><code>a - b</code> <code>x ---</code> <code>y &gt; z</code></p>\n"),
        # A list counts below zero too.
        ('<ol start="-1"><li><h3>x<br><br>y</h3></li></ol>', "<p>-1. ### x y</p>\n"),
        # A heading holds one line of inline content: a block that starts in an empty one ends it.
        ("<h3><h4></h4>x<br>y</h3>", "<p>x\ny</p>\n"),
        ("<h3><br><hr><h3><pre>x</pre><h3><h4>y</h4><h3><blockquote>q</blockquote>"
         "<h3><table><tr><td>c</td></tr></table>",
         "<hr />\n<pre><code>x\n</code></pre>\n<h4>y</h4>\n<blockquote>\n<p>q</p>\n</blockquote>\n"
         "<table>\n<thead>\n<tr>\n<th>c</th>\n</tr>\n</thead>\n</table>\n"),
        # ... while text right inside a table is the heading's: a browser moves it before the table.
        ("<h3><table>text<tr><td>c</td></tr></table></h3>",
         "<h3>text</h3>\n<table>\n<thead>\n<tr>\n<th>c</th>\n</tr>\n</thead>\n</table>\n"),
        # A code span behind a "<" that its text makes an address with: an autolink would
        # take precedence over the span, whatever the address holds before its "@".
        ("&lt;<code>a@b.c&gt;</code> &lt;1<code>a@b.c&gt;</code>",
         "<p>&lt;<code>a@b.c&gt;</code> &lt;1<code>a@b.c&gt;</code></p>\n"),
        ("&lt;<b>x</b><code>a@b.c&gt;</code> &lt;<code>a</code><b>!</b><code>@b.c&gt;</code>",
         "<p>&lt;<strong>x</strong><code>a@b.c&gt;</code> "
         "&lt;<code>a</code><strong>!</strong><code>@b.c&gt;</code></p>\n"),
        # A link that starts a paragraph is no link reference definition, whose label would
        # end at the first "]" of a code span: in a list item, in a quote, over two lines.
        ('<p><a href="/u"><code>]: x</code></a></p>'
         '<ul><li><a href="/u"><code>a]</code><code>: x</code></a></li></ul>',
         '<p> <a href="/u"><code>]: x</code></a></p>\n'
         '<ul>\n<li> <a href="/u"><code>a]: x</code></a></li>\n</ul>\n'),
        ('<blockquote><a href="/u"><code>]: x<br>y</code></a></blockquote>',
         '<blockquote>\n<p> <a href="/u"><code>]: x y</code></a></p>\n</blockquote>\n'),
        # ... but a link behind text or a marker, in a heading or in a table cell stays as it is.
        ('<p>t<br><a href="/u"><code>]: x</code></a></p><h3><a href="/u"><code>]: y</code></a></h3>'
         '<p><b><a href="/u"><code>]: z</code></a></b></p>',
         '<p>t\n<a href="/u"><code>]: x</code></a></p>\n'
         '<h3><a href="/u"><code>]: y</code></a></h3>\n'
         '<p><strong><a href="/u"><code>]: z</code></a></strong></p>\n'),
        # An inline element that is never closed ends with its block or with the input:
        # the marker that the text behind its start tag was escaped for is written.
        ("<p>a</p><b>2)", "<p>a</p>\n<p><strong>2)</strong></p>\n"),
        ("<p><i>+</p><ul><li><b>1.</li></ul><details><summary>+</details>",
         "<p><em>+</em></p>\n<ul>\n<li><strong>1.</strong></li>\n</ul>\n<p><strong>+</strong></p>\n"),
        ("&lt;<b>div&gt; &amp;<i>lt; 2<b>) x",
         "<p>&lt;<strong>div&gt; &amp;<em>lt; 2) x</em></strong></p>\n"),
        ('!<b><a href="/u">x</a>', '<p>!<strong><a href="/u">x</a></strong></p>\n'),
        ("<code>a</code><b><code>b</code>",
         "<p><code>a</code><strong><code>b</code></strong></p>\n"),
        ("<b>a:</b><i>b", "<p><strong>a:</strong><em>b</em></p>\n"),
        ("a<i><b>(b)</b>", "<p>a<em><strong>(b)</strong></em></p>\n"),
        ("<p>Fixed <b>the bug</p><p>More</p>",
         "<p>Fixed <strong>the bug</strong></p>\n<p>More</p>\n"),
        # ... and one still open where the element around it closes ends with that element.
        ("<i>a &lt;<b>div&gt;</i>", "<p><em>a &lt;<strong>div&gt;</strong></em></p>\n"),
        ("<b>x<code>*y*</b> <i>2<b>) z</i>",
         "<p><strong>x<code>*y*</code></strong> <em>2<strong>) z</strong></em></p>\n"),
        # No emphasis, link or code span reaches past its block: what it holds of the
        # first one is marked, and one without content starts in the block that follows.
        ("<b>x<p>y</p></b><i><p>a</p><p>b</p></i>",
         "<p><strong>x</strong></p>\n<p>y</p>\n<p><em>a</em></p>\n<p>b</p>\n"),
        ('<a href="/u"><h3>Title</h3><p>text</p></a><a href="/v"><div>Card</div></a>',
         '<h3><a href="/u">Title</a></h3>\n<p>text</p>\n<p><a href="/v">Card</a></p>\n'),
        ("<b><ul><li>a</li><li>b</li></ul></b><code><p>x</p></code>",
         "<ul>\n<li><strong>a</strong></li>\n<li>b</li>\n</ul>\n<p><code>x</code></p>\n"),
        ("<b><table><tr><td>c</td><td>d</td></tr></table></b><blockquote>q<b></blockquote>x</b>",
         "<table>\n<thead>\n<tr>\n<th><strong>c</strong></th>\n<th>d</th>\n</tr>\n</thead>\n"
         "</table>\n<blockquote>\n<p>q</p>\n</blockquote>\n<p><strong>x</strong></p>\n"),
        ("<i>a<hr>b</i><b><summary>y</summary>z</b>",
         "<p><em>a</em></p>\n<hr />\n<p>b</p>\n<p><strong>y</strong></p>\n<p>z</p>\n"),
        # A link that got no text is written as its address where it ends: below the
        # block it holds, and not at all if it never ends.
        ('<h3>x<a href="/a"><p></p></a><br>y</h3>', "<h3>x</h3>\n<p>/a\ny</p>\n"),
        ('<a href="/u"><pre>x</pre></a><b><p></p></i></b><a href="/v"><p>',
         "<pre><code>x\n</code></pre>\n<p>/u</p>\n"),
        # A link that is written as its address alone has no bracket before it, nor has
        # one that is written from its address.
        ('&lt;<a href="div&gt;">div&gt;</a> &amp;<a href="lt;">lt;</a>',
         "<p>&lt;div&gt; &amp;lt;</p>\n"),
        ('2<a href=") x">) x</a>', "<p>2) x</p>\n"),
        ('2<a href=") x"></a>', "<p>2) x</p>\n"),
        # A line of dashes, colons and pipes below a line with a pipe is text, not the
        # delimiter row that makes a table of the two.
        ("<p>a|b<br>-|-</p><ol><li>|<br>-:</li></ol>",
         "<p>a|b\n-|-</p>\n<ol>\n<li>|\n-:</li>\n</ol>\n"),
        ("<blockquote>x|<br><a>:-</a><br>|-|</blockquote><p><code>a|b</code><br>:<i>-</i></p>",
         "<blockquote>\n<p>x|\n:-\n|-|</p>\n</blockquote>\n<p><code>a|b</code>\n:<em>-</em></p>\n"),
        # A rule that starts a bullet item is no line of dashes, which is a rule outside the list.
        ("<ul><li><hr></li><li><ul><li><hr></li><li>x</li></ul></li></ul><ol><li><hr></li></ol>",
         "<ul>\n<li>\n<hr />\n</li>\n<li>\n<ul>\n<li>\n<hr />\n</li>\n<li>\n<p>x</p>\n</li>\n"
         "</ul>\n</li>\n</ul>\n<ol>\n<li>\n<hr />\n</li>\n</ol>\n"),
        # A link holds no link: the start tag of a second one ends the first, as in a
        # browser, with the code element left open in it.
        ('<a href="/u">a<a href="/v">b</a>c</a>', '<p><a href="/u">a</a><a href="/v">b</a>c</p>\n'),
        ('<a href="/u">x<code>w<a href="/v">y</a>z</a>',
         '<p><a href="/u">x<code>w</code></a><a href="/v">y</a>z</p>\n'),
    ],
)
def test_inline_boundaries_render_without_extra_characters(html: str, rendered: str) -> None:
    """HTML tag boundaries must not create fences, lists or literal emphasis markers."""
    render = MarkdownIt("commonmark").enable(["strikethrough", "table"]).render
    assert render(html_to_markdown(html)) == rendered


class _StyledText(HTMLParser):
    """The words of an HTML fragment and each character's bold / italic / code styles."""

    styles = {"b": "bold", "strong": "bold", "i": "italic", "em": "italic", "code": "code"}

    def __init__(self, html: str) -> None:
        super().__init__(convert_charrefs=True)
        self.open: list[str] = []
        self.chars: list[tuple[str, frozenset[str]]] = []
        self.feed(html)
        self.close()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.styles:
            self.open.append(self.styles[tag])

    def handle_endtag(self, tag: str) -> None:
        if tag in self.styles:
            self.open.pop()

    def handle_data(self, data: str) -> None:
        self.chars.extend((char, frozenset(self.open)) for char in data)

    def styled(self) -> tuple[list[str], list[tuple[str, frozenset[str]]]]:
        """Return the words and every non-space character with its styles."""
        words = "".join(char for char, _ in self.chars).split()
        return words, [(char, styles) for char, styles in self.chars if not char.isspace()]


def _fragments(tokens: int) -> Iterator[str]:
    """Yield every nesting of ``<b>`` and ``<i>`` around a letter, a colon, a space
    and a code span that takes exactly *tokens* tags and text runs."""
    if not tokens:
        yield ""
        return
    for text in ("a", ":", " ", "<code>c</code>"):
        for rest in _fragments(tokens - 1):
            yield text + rest
    for tag in "bi":
        for inside in range(tokens - 1):
            for body in _fragments(inside):
                for rest in _fragments(tokens - 2 - inside):
                    yield f"<{tag}>{body}</{tag}>{rest}"


def test_emphasis_renders_as_written() -> None:
    """Whether a marker opens or closes depends on the characters around it: a
    letter, punctuation, whitespace or another marker.  Every small nesting
    must render with the words and the styles of its HTML."""
    render = MarkdownIt("commonmark").render
    for tokens in range(1, 7):
        for html in _fragments(tokens):
            rendered = render(html_to_markdown(html))
            assert _StyledText(rendered).styled() == _StyledText(html).styled(), html


def test_many_paragraphs_scale_linearly() -> None:
    """Completed paragraphs must not be recopied for every subsequent text run."""
    paragraph = "<p>" + "short paragraph text " * 8 + "</p>"
    times = []
    for count in (1000, 8000):
        start = perf_counter()
        result = html_to_markdown(paragraph * count)
        times.append(perf_counter() - start)
        assert result.split("\n\n") == [("short paragraph text " * 8).strip()] * count
    # Eight times the input permits 16x time plus scheduler noise, not the
    # previous quadratic 64x. No machine-specific absolute runtime threshold.
    assert times[1] < times[0] * 16 + 0.2, times


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

    def test_ordered_list_keeps_its_start_number(self) -> None:
        """``<ol start="5">`` (what ``ensure_html`` makes of ``5. ...``) counts from 5.

        Every list counted from 1, so steps numbered 5 and 6 in a
        summary came back as 1 and 2 in the commit message and the
        memory page; a nested list's start leaves its parent's count
        alone, and a start that is no number means 1.
        """
        html = (
            '<ol start="5"><li>five</li><li>six<ol start="3"><li>three</li></ol></li>'
            '<li>seven</li></ol><ol start="x"><li>one</li></ol>'
        )
        # Below the text of its item a list counts from 1 or follows a blank line.
        assert html_to_markdown(html) == "5. five\n6. six\n\n   3. three\n7. seven\n\n1. one"

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

    def test_an_end_tag_closes_the_elements_still_open_in_its_element(self) -> None:
        assert html_to_markdown("<p><b>x<i>y</b>z</p>") == "**x*y***z"
        assert html_to_markdown("<em><code>*</em>a") == "*`*`*&#97;"

    def test_a_code_span_is_padded_only_where_it_would_end_an_autolink(self) -> None:
        """``<`a@b.c>` `` is an email autolink to a renderer, backticks and all: the
        span that ends the address is padded, and no other span next to a ``<``."""
        assert html_to_markdown("&lt;<code>a@b.c&gt;</code>") == "<` a@b.c> `"
        assert html_to_markdown("\\&lt;<code>a@b.c&gt;</code>") == "\\\\<` a@b.c> `"
        # The span's closing backtick is no part of a domain, an escaped "<" begins nothing,
        # and an address inside an earlier span is code.
        assert html_to_markdown("Ann &lt;<code>a@b.c</code>&gt;") == "Ann <`a@b.c`>"
        assert html_to_markdown("&lt;a<code>x@b.c&gt;</code>") == "\\<a`x@b.c>`"
        assert html_to_markdown("<code>&lt;a@b.c&gt;</code><b>x</b><code>y</code>") == (
            "`<a@b.c>`**x**`y`"
        )

    def test_an_unclosed_code_element_ends_with_its_block(self) -> None:
        """A summary cut inside ``<code>``, or a ``<!`` in code text (it swallows
        the end tag), leaves the element open; all that followed came out raw,
        with no escapes, code spans or emphasis."""
        render = MarkdownIt("commonmark").render
        html = (
            "<ul><li>see <code>datagen/event_types</li></ul>\n"
            "<h3>Previous <b>Session</b> 2</h3><p>a * <code>x</code> &lt;tool&gt;</p>"
            "<p>lookbehind <code>(?<!a)</code></p><p>*y* <code>z"
        )
        markdown = html_to_markdown(html)
        assert markdown == (
            "- see `datagen/event_types`\n\n### Previous **Session** 2\n\n"
            "a \\* `x` \\<tool>\n\nlookbehind `(?`\n\n\\*y\\* `z`"
        )
        assert render(markdown) == (
            "<ul>\n<li>see <code>datagen/event_types</code></li>\n</ul>\n"
            "<h3>Previous <strong>Session</strong> 2</h3>\n"
            "<p>a * <code>x</code> &lt;tool&gt;</p>\n"
            "<p>lookbehind <code>(?</code></p>\n<p>*y* <code>z</code></p>\n"
        )

    def test_code_span_and_backtick_fencing(self) -> None:
        html = "<p><code>foo_bar</code> <code> spaced `tick` </code><code> </code></p>"
        assert html_to_markdown(html) == "`foo_bar` `` spaced `tick` ``"

    def test_adjacent_code_spans_are_merged_into_one(self) -> None:
        html = (
            "<p><code>foo</code><code>bar</code><code>`baz`</code> "
            "<b><code>x</code></b><code>y</code></p>"
        )
        assert html_to_markdown(html) == "`` foobar`baz` `` **`x`**`y`"
        assert html_to_markdown("<p><code>a</code><b><code>b</code></b></p>") == "`a`**`b`**"
        html = "<table><tr><td><code>a</code></td><td>yyy<code>b</code></td></tr></table>"
        assert html_to_markdown(html) == "| `a` | yyy`b` |\n| --- | --- |"

    def test_code_span_keeps_word_separating_spaces(self) -> None:
        html = "<p>Run<code> git status </code>now, one<code> </code>two.</p>"
        assert html_to_markdown(html) == "Run `git status` now, one two."

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

    def test_an_exclamation_mark_before_a_link_does_not_make_it_an_image(self) -> None:
        """``![text](url)`` is an image: the link and its text would be gone."""
        render = MarkdownIt("commonmark").render
        html = (
            '<p>Done!<a href="https://e.x">details</a> and !<b><a href="u">y</a></b>, '
            '! <a href="u">z</a></p>'
        )
        markdown = html_to_markdown(html)
        assert markdown == "Done\\![details](https://e.x) and !**[y](u)**, ! [z](u)"
        assert render(markdown) == (
            '<p>Done!<a href="https://e.x">details</a> and !<strong><a href="u">y</a></strong>, '
            '! <a href="u">z</a></p>\n'
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

    def test_link_destinations_keep_entity_text_and_backslashes(self) -> None:
        """Markdown reads ``&copy;`` in a destination as an entity and a backslash before
        punctuation as an escape: the link went to ``©``, or lost its backslash."""
        render = MarkdownIt("commonmark").render
        for href, destination, rendered in (
            ("http://x/?q=&copy;&y", "http://x/?q=\\&copy;&y", "http://x/?q=&amp;copy;&amp;y"),
            ("http://x/a)?q=&#169;", "<http://x/a)?q=\\&#169;>", "http://x/a)?q=&amp;#169;"),
            ("http://x/?q=&#xA9;", "http://x/?q=\\&#xA9;", "http://x/?q=&amp;#xA9;"),
            ("a\\&copy;\\*", "a\\\\\\&copy;\\\\*", "a%5C&amp;copy;%5C*"),
        ):
            markdown = html_to_markdown(f'<a href="{escape(href)}">t</a>')
            assert markdown == f"[t]({destination})"
            assert render(markdown) == f'<p><a href="{rendered}">t</a></p>\n'

    def test_a_link_destination_drops_tabs_and_line_endings(self) -> None:
        """No destination can hold a line ending: the link was lost (a tab became ``%09``)."""
        markdown = html_to_markdown('<a href="http://x/a\nb\rc\td e">t</a>')
        assert markdown == "[t](<http://x/abcd e>)"  # the address a browser follows
        rendered = MarkdownIt("commonmark").render(markdown)
        assert rendered == '<p><a href="http://x/abcd%20e">t</a></p>\n'

    def test_emphasis_and_links_inside_code_are_plain(self) -> None:
        html = "<p><code>a<b>b</b>c</code> <code>x<a href='u'>l</a></code></p>"
        assert html_to_markdown(html) == "`abc` `xl`"
        # Nor is a nested <code> a span: its backticks ended up inside the outer fence.
        render = MarkdownIt("commonmark").render
        for nested, text in (
            ("<code>a<code>b</code>c</code>", "abc"),
            ("<code>a<code>`</code>c</code>", "a`c"),
            ("<code>x</code><code>a<code>b</code>c</code>", "xabc"),
        ):
            assert render(html_to_markdown(f"<p>{nested}</p>")) == f"<p><code>{text}</code></p>\n"

    def test_adjacent_emphasis_runs_are_merged(self) -> None:
        html = "<p><b>foo</b><b>bar</b> <b>x</b><i>y</i> <i>p</i><b>q</b> <b>a</b> <b>b</b></p>"
        assert html_to_markdown(html) == "**foobar** **x***y* *p***q** **a** **b**"
        html = "<p><b>y</b><a href='u'><b>z</b></a></p>"
        assert html_to_markdown(html) == "**y**[**z**](u)"  # not merged across the link
        # The italic inside leaves the boundary to the bold that merges.
        assert html_to_markdown("<b>a:</b><b><i>b</i></b>q") == "**a:*b***q"

    def test_nested_same_emphasis_is_marked_once(self) -> None:
        html = "<p><b>a<strong>b</strong>c</b> <i>d<em>e</em>f</i> <b>g<i>h<b>i</b></i></b></p>"
        assert html_to_markdown(html) == "**abc** *def* **g*hi***"
        html = "<p><i><b>a</b><i><b>b</b></i></i></p>"  # the redundant <i> does not split the bold
        assert html_to_markdown(html) == "_**ab**_"

    def test_emphasis_starting_with_the_other_emphasis_keeps_its_runs_apart(self) -> None:
        """``***a** b**c***`` lets the intraword ``**`` of ``b**c`` close the italic
        early (CommonMark's rule of 3); the outer emphasis is written with
        underscores, which only parse next to whitespace or punctuation, so inside
        a word the asterisk form stays."""
        render = MarkdownIt("commonmark").render
        html = "<p><i><b><i>a</i></b> b<b>c</b></i></p>"
        assert html_to_markdown(html) == "_**a** b**c**_"
        assert render(html_to_markdown(html)) == (
            "<p><em><strong>a</strong> b<strong>c</strong></em></p>\n"
        )
        html = "<p><b><i>a</i> b<i>c</i></b> <i><b>x</b></i>, (<b><i>y</i></b>)<b>z</b></p>"
        assert html_to_markdown(html) == "__*a* b*c*__ _**x**_, (__*y*__)**z**"
        assert render(html_to_markdown(html)) == (
            "<p><strong><em>a</em> b<em>c</em></strong> <em><strong>x</strong></em>, "
            "(<strong><em>y</em></strong>)<strong>z</strong></p>\n"
        )
        html = "<p><b>x <i>y</i></b><i>z</i>, <i>p<b>q</b></i><b>r</b>.</p>"  # after a run of three
        assert html_to_markdown(html) == "**x *y***_z_, *p**q***__r__."
        assert render(html_to_markdown(html)) == (
            "<p><strong>x <em>y</em></strong><em>z</em>, "
            "<em>p<strong>q</strong></em><strong>r</strong>.</p>\n"
        )
        html = "<ul><li>w<i><b>a</b></i> <i><b>b</b> c</i>w <b>v</b><i><b>d</b></i></li></ul>"
        assert html_to_markdown(html) == "- w***a*** ***b** c*w **v**_**d**_"
        assert render(html_to_markdown(html)) == (
            "<ul>\n<li>w<em><strong>a</strong></em> <em><strong>b</strong> c</em>w "
            "<strong>v</strong><em><strong>d</strong></em></li>\n</ul>\n"
        )
        html = "<table><tr><td><i><b>a</b></i></td><td><i><b>b</b></i>x</td></tr></table>"
        assert html_to_markdown(html) == "| _**a**_ | ***b***x |\n| --- | --- |"
        assert html_to_markdown("<p><i><b>a</b> </i>b</p>") == "_**a**_ b"
        # Whitespace, punctuation and symbols are the only boundaries: a combining
        # mark is none, and an anchor that writes no brackets (no href, a label
        # that is its href, an empty label) puts raw text after the closer.
        for html, markdown, rendered in (
            ("a\u0301<i><b>b</b></i>", "a\u0301***b***", "a\u0301<em><strong>b</strong></em>"),
            ("<i><b>a</b></i>\u0301b", "***a***\u0301b", "<em><strong>a</strong></em>\u0301b"),
            ("<i><b>a</b></i><a id='t'>x</a>", "***a***x", "<em><strong>a</strong></em>x"),
            (
                "<i><b>a</b></i><a href='http://e'>http://e</a>",
                "***a***http://e",
                "<em><strong>a</strong></em>http://e",
            ),
            (
                "<i><b>a</b></i><a href='u'></a>, <i><b>c</b></i><a>x<b>y</b></a> "
                "<b>d</b><a id='t'><b>e</b></a>f",
                "***a***u, ***c***x**y** **de**f",
                "<em><strong>a</strong></em>u, <em><strong>c</strong></em>x<strong>y</strong> "
                "<strong>de</strong>f",
            ),
            (
                "<a href='u'><i><b>a</b></i></a>x <i><b>c</b></i><b>d</b>e "
                "\u2192<i><b>f</b></i>\u2192",
                "[_**a**_](u)x _**c**_**d**e \u2192_**f**_\u2192",
                '<a href="u"><em><strong>a</strong></em></a>x <em><strong>c</strong></em>'
                "<strong>d</strong>e \u2192<em><strong>f</strong></em>\u2192",
            ),
            # Mixed runs next to a symbol: the asterisk form would fuse them.
            (
                "<i><b>a</b> b<b>c</b></i>\u2192 \u2192<i><b>a</b> b<b>c</b></i>",
                "_**a** b**c**_\u2192 \u2192_**a** b**c**_",
                "<em><strong>a</strong> b<strong>c</strong></em>\u2192 "
                "\u2192<em><strong>a</strong> b<strong>c</strong></em>",
            ),
        ):
            assert html_to_markdown(f"<p>{html}</p>") == markdown
            assert render(markdown) == f"<p>{rendered}</p>\n"
        html = "<blockquote><i><b>a</b>x</i></blockquote><p>y</p>"
        assert html_to_markdown(html) == "> _**a**x_\n\ny"

    def test_escaped_literal_star_is_not_merged_with_emphasis(self) -> None:
        html = "<p>literal *<i>x</i> **<b>y</b></p>"
        assert html_to_markdown(html) == "literal \\**x* \\*\\***y**"

    def test_unbalanced_code_inside_bold_keeps_escaping_afterwards(self) -> None:
        html = "<p><b><code>text</b></code></p><p>*literal* &lt;h3&gt;</p>"
        assert html_to_markdown(html) == "**`text`**\n\n\\*literal\\* \\<h3>"

    def test_image_alt_text(self) -> None:
        assert html_to_markdown("<p><img src='a.png' alt='pic'><img src='b.png'></p>") == "pic"
        assert html_to_markdown("<pre>a<img alt='*pic*'>b</pre>") == "```\na*pic*b\n```"


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

    def test_a_line_that_is_only_a_block_marker_stays_text(self) -> None:
        """A list marker is one with nothing after it too (an empty item, whose
        text is lost), and ``- --`` is a thematic break.  A marker that cannot
        start a list -- in a table cell, behind an emphasis marker -- stays
        as it is."""
        render = MarkdownIt("commonmark").enable("table").render
        html = (
            "<p>+</p><p>1)</p><p><span>7.</span> x</p><p><b>8.</b> y</p>"
            "<ul><li>+</li><li>-- </li></ul>"
            "<table><tr><th>n</th></tr><tr><td>1.</td></tr><tr><td>+</td></tr></table>"
        )
        markdown = html_to_markdown(html)
        assert markdown == (
            "\\+\n\n1\\)\n\n7\\. x\n\n**8.** y\n\n- \\+\n- \\--\n\n"
            "| n |\n| --- |\n| 1. |\n| + |"
        )
        assert render(markdown) == (
            "<p>+</p>\n<p>1)</p>\n<p>7. x</p>\n<p><strong>8.</strong> y</p>\n"
            "<ul>\n<li>+</li>\n<li>--</li>\n</ul>\n"
            "<table>\n<thead>\n<tr>\n<th>n</th>\n</tr>\n</thead>\n<tbody>\n"
            "<tr>\n<td>1.</td>\n</tr>\n<tr>\n<td>+</td>\n</tr>\n</tbody>\n</table>\n"
        )

    def test_tilde_runs_stay_text(self) -> None:
        """Two tildes strike text through where that is supported, and three at
        the start of a line open a code fence that takes the rest."""
        render = MarkdownIt("commonmark").enable("strikethrough").render
        markdown = html_to_markdown("<p>a ~~b~~ c, ~/d</p><p>~~~ x</p><p>end</p>")
        assert markdown == "a \\~\\~b\\~\\~ c, ~/d\n\n\\~\\~\\~ x\n\nend"
        assert render(markdown) == "<p>a ~~b~~ c, ~/d</p>\n<p>~~~ x</p>\n<p>end</p>\n"

    def test_a_heading_keeps_its_trailing_hashes(self) -> None:
        """``## Issue #`` ends in an ATX closing sequence, which a renderer drops."""
        render = MarkdownIt("commonmark").render
        markdown = html_to_markdown("<h2>Issue #</h2><h3>C# and F ##</h3><h4>A <b>#</b> #</h4>")
        assert markdown == "## Issue \\#\n\n### C# and F \\##\n\n#### A **#** \\#"
        assert render(markdown) == (
            "<h2>Issue #</h2>\n<h3>C# and F ##</h3>\n<h4>A <strong>#</strong> #</h4>\n"
        )
        # Also where no end tag ends its line: a block starts in it, its item or its quote
        # ends, or the input does.
        markdown = html_to_markdown(
            "<h2>Issue #<p>text</p></h2><ul><li><h3>C #</li></ul>"
            "<blockquote><h3>q #</blockquote><h4>A #"
        )
        assert render(markdown) == (
            "<h2>Issue #</h2>\n<p>text</p>\n<ul>\n<li>\n<h3>C #</h3>\n</li>\n</ul>\n"
            "<blockquote>\n<h3>q #</h3>\n</blockquote>\n<h4>A #</h4>\n"
        )

    def test_literal_links_images_and_entities_stay_literal(self) -> None:
        html = "<p>[gone](https://example.org) ![img](x.png) &amp;copy; &amp;#169; a &amp; b</p>"
        assert html_to_markdown(html) == (
            "\\[gone\\](https://example.org) !\\[img\\](x.png) \\&copy; \\&#169; a & b"
        )

    def test_an_address_in_angle_brackets_is_no_autolink(self) -> None:
        """``<1@x.y>`` is an email autolink, which shows the address without
        its brackets; a local part may start with a digit or a symbol."""
        render = MarkdownIt("commonmark").render
        html = "<p>To: A B &lt;112301042@smail.example.in&gt;, &lt;+x@y.z&gt;; 1 &lt; 2 @ 3</p>"
        markdown = html_to_markdown(html)
        assert markdown == "To: A B \\<112301042@smail.example.in>, \\<+x@y.z>; 1 < 2 @ 3"
        assert render(markdown) == (
            "<p>To: A B &lt;112301042@smail.example.in&gt;, &lt;+x@y.z&gt;; 1 &lt; 2 @ 3</p>\n"
        )

    def test_text_runs_that_meet_are_no_tag_or_entity(self) -> None:
        """Text runs are escaped one by one: a tag, an autolink or an entity
        that the runs before and behind an element only make together stays
        text as well, whether the element writes nothing or a marker that an
        address may hold.  A raw ``<`` in the HTML is a text run too: what
        follows it needs no escape."""
        render = MarkdownIt("commonmark").render
        html = (
            "&lt;<span>div</span>&gt; &lt;<span>/b</span>&gt; &lt;<span>+x@y.z</span>&gt; "
            "&lt;1<span>@y.z</span>&gt; &amp;<i></i>lt; &amp;<span>#38;</span> "
            "&amp;am<span>p;</span> &lt;<b>b</b>&gt; &lt;<b>x</b>@y.z&gt; "
            "&lt;1<i>x@y.z&gt;</i> p < 0.05 (<0.025) R&D"
        )
        markdown = html_to_markdown(html)
        assert markdown == (
            "<&#100;iv> <\\/b> <+x\\@y.z> <1\\@y.z> &lt\\; &#38\\; &amp\\; "
            "<**b**> <**x**\\@y.z> <1*x\\@y.z>* p < 0.05 (<0.025) R&D"
        )
        assert render(markdown) == (
            "<p>&lt;div&gt; &lt;/b&gt; &lt;+x@y.z&gt; &lt;1@y.z&gt; &amp;lt; &amp;#38; &amp;amp; "
            "&lt;<strong>b</strong>&gt; &lt;<strong>x</strong>@y.z&gt; &lt;1<em>x@y.z&gt;</em> "
            "p &lt; 0.05 (&lt;0.025) R&amp;D</p>\n"
        )

    def test_plain_text_passes_through(self) -> None:
        assert html_to_markdown("plain text only") == "plain text only"
        assert html_to_markdown("") == ""

    def test_whitespace_between_blocks_is_dropped(self) -> None:
        assert html_to_markdown("<p>a</p>\n   \n<p>b</p>  ") == "a\n\nb"

    def test_entities_are_decoded(self) -> None:
        assert html_to_markdown("<p>Tom &amp; Jerry &#8594; done</p>") == "Tom & Jerry → done"
