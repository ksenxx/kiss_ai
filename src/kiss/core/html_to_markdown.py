# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Convert an HTML fragment to Markdown.

The ``finish`` tool guarantees that a task's result summary is HTML
(:func:`kiss.core.utils.ensure_html`).  Git commit messages are plain
text, and the viewers that show them (``git log``, GitHub, the VS Code
Git hovers) either print tags literally or, in VS Code's case, render
the message as Markdown with raw HTML disabled so that every HTML block
vanishes.  This module turns the HTML summary into Markdown that reads
well in all of those places, using only the stdlib
:class:`html.parser.HTMLParser`.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

_SKIP_TAGS = {"script", "style", "head", "title"}
_PARAGRAPH_TAGS = {"p", "div", "section", "article", "details", "figure"}
_HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
_INLINE_MARKERS = {"b": "**", "strong": "**", "i": "*", "em": "*", "summary": "**"}
_TABLE_TAGS = {"table", "tr", "td", "th"}

_WHITESPACE_RE = re.compile(r"\s+")
_BOUNDARY_UNDERSCORE_RE = re.compile(r"(?<![A-Za-z0-9])_|_(?![A-Za-z0-9])")
_TAG_START_RE = re.compile(r"<(?=[A-Za-z/!?])")
_ENTITY_START_RE = re.compile(r"&(?=[A-Za-z][A-Za-z0-9]*;|#[0-9]+;|#[xX][0-9a-fA-F]+;)")
_LINE_START_SYNTAX_RE = re.compile(r"^(?:#{1,6}(?:\s|$)|[-+]\s|>|=+$|-+$)")
_LINE_START_NUMBER_RE = re.compile(r"^(\d+)([.)])(\s)")
_LINK_DESTINATION_UNSAFE_RE = re.compile(r"[\s()<>]")


def _escape_inline(text: str) -> str:
    """Backslash-escape characters that would otherwise be Markdown syntax."""
    for char in "\\`*[]":
        text = text.replace(char, "\\" + char)
    text = _BOUNDARY_UNDERSCORE_RE.sub(r"\\_", text)
    text = _ENTITY_START_RE.sub(r"\\&", text)
    return _TAG_START_RE.sub(r"\\<", text)


def _escape_line_start(text: str) -> str:
    """Escape a leading heading / list / quote marker so a plain line stays plain."""
    if _LINE_START_SYNTAX_RE.match(text):
        return "\\" + text
    return _LINE_START_NUMBER_RE.sub(r"\1\\\2\3", text, count=1)


def _backtick_fence(text: str, minimum: int) -> str:
    """Return a run of backticks longer than any backtick run inside *text*."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    return "`" * max(longest + 1, minimum)


def _code_span(text: str) -> str:
    """Wrap *text* in a code span whose fence is longer than any inner backtick run."""
    fence = _backtick_fence(text, 1)
    if len(fence) > 1 or text.startswith(" ") or text.endswith(" "):
        return f"{fence} {text} {fence}"
    return f"{fence}{text}{fence}"


def _link_destination(href: str) -> str:
    """Return *href* in a form that survives inside ``[label](...)``."""
    if _LINK_DESTINATION_UNSAFE_RE.search(href):
        return "<" + href.replace("\\", "\\\\").replace("<", "\\<").replace(">", "\\>") + ">"
    return href


class _ListState:
    """Rendering state of one open ``<ul>``/``<ol>``."""

    def __init__(self, ordered: bool, indent: str) -> None:
        self.counter = 1 if ordered else 0
        self.indent = indent
        self.child_indent = indent


class _HtmlToMarkdownParser(HTMLParser):
    """Stream HTML into a Markdown string held in :attr:`text`."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text = ""
        self._skip_depth = 0
        self._code_depth = 0
        self._pre: list[str] | None = None
        self._lists: list[_ListState] = []
        self._inline: list[tuple[str, int, str | None]] = []
        self._quotes: list[int] = []
        self._tables: list[list[list[str]]] = []
        self._row: list[str] | None = None
        self._cell_start = -1
        self._cell_marker_end = -1
        self._marker_end = -1

    def _indent(self) -> str:
        return self._lists[-1].child_indent if self._lists else ""

    def _at_marker(self) -> bool:
        """True right after a heading or list marker, before any content."""
        return len(self.text) == self._marker_end

    def _at_line_start(self) -> bool:
        return not self.text or self.text.endswith("\n")

    def _newline(self, count: int) -> None:
        """End the current line and ensure *count* newlines precede the next text."""
        if not self.text or self._at_marker():
            return
        stripped = self.text.rstrip(" ")
        trailing = len(stripped) - len(stripped.rstrip("\n"))
        self.text = stripped + "\n" * max(count - trailing, 0)

    def _append(self, text: str) -> None:
        """Append *text*, indenting continuation lines inside list items."""
        indent = self._indent()
        if indent and text and self.text.endswith("\n"):
            text = indent + text
        if indent:
            text = text.replace("\n", "\n" + indent).replace(indent + "\n", "\n")
        self.text += text
        self._marker_end = -1

    def _emit_marker(self, marker: str) -> None:
        """Append a heading / list marker that must not be followed by a line break."""
        self.text += marker
        self._marker_end = len(self.text)

    def finish(self) -> None:
        """Flush constructs left open at the end of the input."""
        if self._pre is not None:
            self._end_pre()
        while self._tables:
            if len(self._tables) == 1:
                self._end_table_tag("td")
                self._end_table_tag("tr")
            self._end_table_tag("table")
        self.drop_empty_marker()

    def drop_empty_marker(self) -> None:
        """Remove a heading / list marker whose element turned out to be empty."""
        if self._at_marker():
            self.text = self.text[: self.text.rfind("\n") + 1]
            self._marker_end = -1

    def _wrap_since(self, pos: int, before: str, after: str) -> None:
        """Replace everything appended since *pos* with ``before + inner + after``.

        Surrounding whitespace is moved outside the markers so that
        ``<b> bold </b>`` becomes `` **bold** `` rather than the
        invalid ``** bold **``; an empty inner text drops the markers.
        Two adjacent runs of the same emphasis (``<b>a</b><b>b</b>``)
        are merged into one so no ``****`` delimiter run is produced.
        """
        inner = self.text[pos:]
        core = inner.strip()
        if not core:
            return
        head = self.text[:pos]
        lead = inner[: len(inner) - len(inner.lstrip())]
        trail = inner[len(inner.rstrip()):]
        if (
            not lead and before == after and head.endswith(after)
            and not head[: -len(after)].endswith(("*", "\\"))
        ):
            head = head[: -len(after)]
            before = ""
        self.text = f"{head}{lead}{before}{core}{after}{trail}"

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Open block/inline constructs and emit list markers."""
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if self._pre is not None:
            if tag == "br":
                self._pre.append("\n")
            return
        if tag == "br":
            self._append("\n")
        elif tag == "hr":
            self._newline(2)
            self._append("---")
            self._newline(2)
        elif tag == "img":
            alt = dict(attrs).get("alt")
            if alt:
                self._append(_escape_inline(alt))
        elif tag in _HEADING_TAGS:
            self._newline(2)
            indent = self._indent() if self.text.endswith("\n") else ""
            self._emit_marker(indent + "#" * _HEADING_TAGS[tag] + " ")
        elif tag in _PARAGRAPH_TAGS:
            self._newline(2)
        elif tag == "blockquote":
            self._newline(2)
            self._quotes.append(len(self.text))
        elif tag == "pre":
            self._newline(2)
            self._pre = []
        elif tag == "code":
            self._code_depth += 1
            self._inline.append((tag, len(self.text), None))
        elif tag in ("ul", "ol"):
            self._newline(1 if self._lists else 2)
            self._lists.append(_ListState(tag == "ol", self._indent()))
        elif tag == "li":
            self._start_list_item()
        elif tag in _TABLE_TAGS:
            self._start_table_tag(tag)
        elif tag == "a":
            self._inline.append((tag, len(self.text), dict(attrs).get("href")))
        elif tag in _INLINE_MARKERS:
            if tag == "summary":
                self._newline(2)
            self._inline.append((tag, len(self.text), None))

    def _start_list_item(self) -> None:
        """Emit the ``- `` / ``N. `` marker for a new list item."""
        if not self._lists:
            self._lists.append(_ListState(False, ""))
        state = self._lists[-1]
        marker = f"{state.counter}. " if state.counter else "- "
        if state.counter:
            state.counter += 1
        state.child_indent = state.indent + " " * len(marker)
        self.drop_empty_marker()
        self._newline(1)
        self._emit_marker(state.indent + marker)

    def _start_table_tag(self, tag: str) -> None:
        """Open a table, row or cell; nested tables flow into the enclosing cell."""
        if tag == "table":
            if not self._tables:
                self._newline(2)
            self._tables.append([])
        elif len(self._tables) != 1 or self._cell_start >= 0:
            if self.text and not self.text.endswith((" ", "\n")):
                self._append(" ")
        elif tag == "tr":
            if self._row is not None:
                self._tables[0].append(self._row)
            self._row = []
        elif self._row is not None:
            self._cell_start = len(self.text)
            self._cell_marker_end = self._marker_end

    def handle_endtag(self, tag: str) -> None:
        """Close block/inline constructs."""
        if tag in _SKIP_TAGS:
            self._skip_depth = max(self._skip_depth - 1, 0)
            return
        if self._skip_depth:
            return
        if self._pre is not None:
            if tag == "pre":
                self._end_pre()
            return
        if tag in _HEADING_TAGS:
            self.drop_empty_marker()
            self._newline(2)
        elif tag in _PARAGRAPH_TAGS:
            self._newline(2)
        elif tag == "blockquote":
            self._end_blockquote()
        elif tag in ("ul", "ol"):
            self.drop_empty_marker()
            if self._lists:
                self._lists.pop()
            self._newline(1 if self._lists else 2)
        elif tag == "li":
            self._newline(1)
        elif tag in _TABLE_TAGS:
            self._end_table_tag(tag)
        elif tag in _INLINE_MARKERS or tag in ("a", "code"):
            self._end_inline(tag)

    def _end_pre(self) -> None:
        """Emit the buffered ``<pre>`` text as a fenced code block."""
        body = "".join(self._pre or []).removeprefix("\n").rstrip("\n")
        self._pre = None
        if self._cell_start >= 0:
            # A pipe-table cell is a single line: keep the code as one
            # code span per line, joined by a visible return symbol.
            lines = [_code_span(line) for line in body.split("\n") if line.strip()]
            self._append(" \u23ce ".join(lines))
            return
        fence = _backtick_fence(body, 3)
        self._append(f"{fence}\n{body}\n{fence}" if body else f"{fence}\n{fence}")
        self._newline(2)

    def _end_inline(self, tag: str) -> None:
        """Close the innermost open *tag*, dropping unclosed inner markers."""
        for index in range(len(self._inline) - 1, -1, -1):
            if self._inline[index][0] != tag:
                continue
            _, pos, href = self._inline[index]
            dropped = self._inline[index:]
            del self._inline[index:]
            self._code_depth -= sum(1 for entry in dropped if entry[0] == "code")
            if tag == "code":
                core = self.text[pos:].strip()
                self.text = self.text[:pos] + (_code_span(core) if core else "")
            elif self._code_depth:
                return  # emphasis and links have no meaning inside code
            elif tag == "a":
                inner = self.text[pos:].strip()
                if href and not inner:
                    self.text = self.text[:pos] + _escape_inline(href)
                elif href and inner != href:
                    self._wrap_since(pos, "[", f"]({_link_destination(href)})")
            else:
                marker = _INLINE_MARKERS[tag]
                self._wrap_since(pos, marker, marker)
                if tag == "summary":
                    self._newline(2)
            return

    def _end_blockquote(self) -> None:
        """Prefix everything since the matching ``<blockquote>`` with ``> ``."""
        if not self._quotes:
            return
        pos = self._quotes.pop()
        indent = self._indent()
        lines = self.text[pos:].strip("\n").split("\n")
        quoted = [f"{indent}> {line.removeprefix(indent)}".rstrip() for line in lines]
        if pos and not self.text[:pos].endswith("\n"):
            quoted[0] = quoted[0].removeprefix(indent)  # continues a list marker line
        self.text = self.text[:pos] + "\n".join(quoted)
        self._newline(2)

    def _end_table_tag(self, tag: str) -> None:
        """Close a cell, row or table; the table is emitted when it closes."""
        if not self._tables:
            return
        if tag == "table":
            rows = self._tables.pop()
            if not self._tables:
                self._emit_table(rows)
        elif len(self._tables) != 1:
            return
        elif tag == "tr":
            if self._row is not None:
                self._tables[0].append(self._row)
                self._row = None
        elif self._row is not None and self._cell_start >= 0:
            cell = " ".join(self.text[self._cell_start:].split())
            self.text = self.text[: self._cell_start]
            self._marker_end = self._cell_marker_end
            self._row.append(cell.replace("|", "\\|"))
            self._cell_start = -1

    def _emit_table(self, rows: list[list[str]]) -> None:
        """Emit *rows* as a pipe table whose first row is the header."""
        self._row = None
        self._cell_start = -1
        rows = [row for row in rows if row]
        if not rows:
            return
        width = max(len(row) for row in rows)
        lines = ["| " + " | ".join(row + [""] * (width - len(row))) + " |" for row in rows]
        lines.insert(1, "| " + " | ".join(["---"] * width) + " |")
        self._newline(2)
        self._append("\n".join(lines))
        self._newline(2)

    def handle_data(self, data: str) -> None:
        """Append text, escaping Markdown syntax and collapsing whitespace."""
        if self._skip_depth or not data:
            return
        if self._pre is not None:
            self._pre.append(data)
            return
        collapsed = _WHITESPACE_RE.sub(" ", data)
        if self._at_line_start() or self.text.endswith(" ") or self._at_marker():
            collapsed = collapsed.lstrip()
        if not collapsed:
            return
        if not self._code_depth:
            collapsed = _escape_inline(collapsed)
            if self._at_line_start() or self._at_marker():
                collapsed = _escape_line_start(collapsed)
        self._append(collapsed)


def html_to_markdown(html: str) -> str:
    """Convert an HTML fragment to Markdown.

    Headings become ``#`` headings, ``<p>`` blocks become paragraphs,
    ``<ul>``/``<ol>`` become (nested) ``-``/``1.`` lists, ``<pre>``
    becomes a fenced code block, ``<code>`` a code span, ``<b>``/``<i>``
    emphasis, ``<a>`` a ``[text](href)`` link, ``<table>`` a pipe table
    and ``<blockquote>`` a ``>`` quote.  Text that would accidentally be
    parsed as Markdown syntax is backslash-escaped.

    Args:
        html: The HTML string to convert.

    Returns:
        The Markdown text, stripped of leading/trailing blank lines and
        trailing whitespace on each line.
    """
    parser = _HtmlToMarkdownParser()
    parser.feed(html)
    parser.close()
    parser.finish()
    lines = [line.rstrip() for line in parser.text.split("\n")]
    return "\n".join(lines).strip("\n")
