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
import string
import unicodedata
from html.parser import HTMLParser

# Subtrees whose tags and text are dropped whole.  Only container tags:
# void elements (meta, link) get no end-tag event, so counting them here
# would leave the parser skipping for the rest of the document.  A <head>
# parent already hides them.
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


def _is_boundary(char: str) -> bool:
    """Whether an underscore emphasis may open after or close before *char*.

    CommonMark's flanking rules let an underscore run open only after, and
    close only before, whitespace or punctuation (``""`` stands for the ends
    of the text); next to a letter, digit or combining mark it is literal.
    Symbols (``→``, ``€``) are punctuation since CommonMark 0.31, which
    markdown-it, the renderer of the webapp and of VS Code, follows: next
    to them the underscore form is the one that keeps mixed emphasis
    (``_**a** b**c**_→``) apart, the asterisk form fuses its runs.
    """
    if not char or char in "\t\n\f\r" or char in string.punctuation:
        return True
    category = unicodedata.category(char)
    return category == "Zs" or category.startswith(("P", "S"))


class _ListState:
    """Rendering state of one open ``<ul>``/``<ol>``."""

    def __init__(self, ordered: bool, indent: str) -> None:
        self.counter = 1 if ordered else 0
        self.indent = indent
        self.child_indent = indent


class SkippingHTMLParser(HTMLParser):
    """An :class:`HTMLParser` that drops ``<script>``, ``<style>``, ``<head>``
    and ``<title>`` subtrees whole: tags and text alike.

    A subclass starts ``handle_starttag`` / ``handle_endtag`` with
    ``if self.skipped(tag, end): return`` and ``handle_data`` with
    ``if self.skipping: return``.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0

    @property
    def skipping(self) -> bool:
        """Whether the parser is inside a dropped subtree."""
        return self._skip_depth > 0

    def skipped(self, tag: str, end: bool) -> bool:
        """Record *tag* (an end tag when *end*) and say whether it is dropped.

        Args:
            tag: The lower-cased tag name of the event.
            end: ``True`` for an end tag, ``False`` for a start tag.

        Returns:
            ``True`` when the tag opens, closes or sits inside a dropped
            subtree, so the subclass must ignore the event.
        """
        if tag in _SKIP_TAGS:
            self._skip_depth = max(self._skip_depth - 1, 0) if end else self._skip_depth + 1
            return True
        return self._skip_depth > 0


class _HtmlToMarkdownParser(SkippingHTMLParser):
    """Stream HTML into a Markdown string held in :attr:`text`."""

    def __init__(self) -> None:
        super().__init__()
        self.text = ""
        self._code_depth = 0
        self._pre: list[str] | None = None
        self._lists: list[_ListState] = []
        self._inline: list[tuple[str, int, str | None]] = []
        self._last_code: tuple[int, int, str] | None = None  # (start, end, text) of last span
        self._underscores: list[tuple[int, int, int]] = []  # (open, close, width), undecided
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
        self._settle_underscores()

    def _settle_underscores(self) -> None:
        """Turn an underscore emphasis :meth:`_wrap_since` wrote back to asterisks
        when the text continues its word: there, underscores are literal.

        The character after the closing run decides (:func:`_is_boundary`)
        once it is final: an inline element that opens right there may still
        write its own markers first (``[``, a backtick, ``**``) or nothing at
        all (``<a>`` without an href), so the run waits until it has closed.
        """
        pending = []
        for run in self._underscores:
            opening, closing, width = run
            end = closing + width
            if len(self.text) <= end or self._opens_at(end):
                pending.append(run)
            elif not _is_boundary(self.text[end]):
                stars = "*" * width
                self.text = (
                    f"{self.text[:opening]}{stars}{self.text[opening + width : closing]}"
                    f"{stars}{self.text[end:]}"
                )
        self._underscores = pending

    def _emit_marker(self, marker: str) -> None:
        """Append a heading / list marker that must not be followed by a line break."""
        self.text += marker
        self._marker_end = len(self.text)

    def finish(self) -> None:
        """Flush constructs left open at the end of the input."""
        if self._pre is not None:
            self._end_pre()
        while self._tables:
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
        are merged into one so no ``****`` delimiter run is produced,
        unless an open ancestor (``<b>a</b><a><b>b</b></a>``) starts between
        them: its markers must not land inside the merged run.
        An emphasis whose text starts with the other emphasis
        (``<i><b>a</b> b<b>c</b></i>``) would open with a ``***`` run, which
        CommonMark lets an intraword run inside close early
        (``***a** b**c***``), and one opening right after such a run of three
        (``<b>x <i>y</i></b><i>z</i>``) would make it a run of four, which the
        rule of 3 keeps from closing; both are written with underscores,
        ``_**a** b**c**_``, where they parse: after whitespace or punctuation,
        and before it, which the text that follows decides
        (:meth:`_settle_underscores`).
        """
        inner = self.text[pos:]
        core = inner.strip()
        if not core:
            return
        head = self.text[:pos]
        lead = inner[: len(inner) - len(inner.lstrip())]
        trail = inner[len(inner.rstrip()):]
        # A run this wrap encloses or directly precedes is decided by the wrap.
        self._underscores = [run for run in self._underscores if pos > run[1] + run[2]]
        if (
            not lead and before == after and head.endswith(after)
            and not head[: -len(after)].endswith(("*", "\\"))
            and not self._opens_at(pos)
        ):
            head = head[: -len(after)]
            before = ""
        elif (
            before == after
            and (core.startswith("*") or (head + lead).endswith("***"))
            and _is_boundary((head + lead)[-1:])
            and not (head + lead).endswith("_")
        ):
            before = after = "_" * len(after)
            if not trail:
                opening = len(head) + len(lead)
                self._underscores.append((opening, opening + len(before) + len(core), len(before)))
        self.text = f"{head}{lead}{before}{core}{after}{trail}"

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Open block/inline constructs and emit list markers."""
        if self.skipped(tag, False):
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

    def _separate(self) -> None:
        """Append a space unless the text is empty or already ends in whitespace."""
        if self.text and not self.text.endswith((" ", "\n")):
            self._append(" ")

    def _start_table_tag(self, tag: str) -> None:
        """Open a table, row or cell; nested tables flow into the enclosing cell.

        As in a browser, a new row closes the open cell and row, and a new
        cell closes the open cell, so omitted ``</td>``/``</tr>`` lose nothing.
        """
        if tag == "table":
            if not self._tables:
                self._newline(2)
            self._tables.append([])
        elif len(self._tables) != 1:
            self._separate()
        elif tag == "tr":
            self._end_table_tag("tr")
            self._row = []
        else:
            self._end_table_tag(tag)
            if self._row is not None:
                self._cell_start = len(self.text)
                self._cell_marker_end = self._marker_end

    def handle_endtag(self, tag: str) -> None:
        """Close block/inline constructs."""
        if self.skipped(tag, True):
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
            redundant = self._redundant(index)
            dropped = self._inline[index:]
            del self._inline[index:]
            self._code_depth -= sum(1 for entry in dropped if entry[0] == "code")
            if tag == "code":
                self._end_code(pos)
            elif self._code_depth:
                return  # emphasis and links have no meaning inside code
            elif tag == "a":
                inner = self.text[pos:].strip()
                if href and not inner:
                    self.text = self.text[:pos] + _escape_inline(href)
                elif href and inner != href:
                    self._wrap_since(pos, "[", f"]({_link_destination(href)})")
                self._settle_underscores()
            else:
                marker = _INLINE_MARKERS[tag]
                if not redundant:
                    self._wrap_since(pos, marker, marker)
                if tag == "summary":
                    self._newline(2)
            return

    def _redundant(self, index: int) -> bool:
        """Whether an entry below *index* on the inline stack already supplies its emphasis.

        Such an entry writes no markers of its own: ``<b>a<b>b</b>c</b>`` is
        ``**abc**``, not ``**a**b**c**``, which would un-bold ``b``.
        """
        marker = _INLINE_MARKERS.get(self._inline[index][0])
        return marker is not None and any(
            _INLINE_MARKERS.get(entry[0]) == marker for entry in self._inline[:index]
        )

    def _opens_at(self, pos: int) -> bool:
        """Whether an open inline element that writes markers of its own starts at *pos*.

        Adjacent code spans or emphasis runs do not merge across it: the merge
        would move its start inside the merged text (``<b>y</b><a><b>z</b></a>``).
        An anchor without an href writes nothing, and a redundant emphasis
        writes nothing either (:meth:`_redundant`).
        """
        return any(
            entry[1] == pos and entry != ("a", pos, None) and not self._redundant(index)
            for index, entry in enumerate(self._inline)
        )

    def _end_code(self, pos: int) -> None:
        """Replace the text appended since *pos* with a code span.

        Surrounding whitespace stays outside the span, as in :meth:`_wrap_since`,
        and an all-whitespace element keeps separating the words around it.  A
        span that directly follows the previous one (``<code>a</code><code>b</code>``)
        is merged into it, since ``` `a``b` ``` would render as one span ``a``b``.
        """
        inner = self.text[pos:]
        core = inner.strip()
        if not core:
            return
        lead = inner[: len(inner) - len(inner.lstrip())]
        trail = inner[len(inner.rstrip()):]
        last = self._last_code
        if last and last[1] == pos and not lead and not self._opens_at(pos):
            pos, core = last[0], last[2] + core
        span = _code_span(core)
        self.text = f"{self.text[:pos]}{lead}{span}{trail}"
        self._last_code = (pos + len(lead), pos + len(lead) + len(span), core)

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
        self._underscores = []  # a newline follows every pending run (those inside moved)
        self._newline(2)

    def _end_table_tag(self, tag: str) -> None:
        """Close a cell, row or table; the table is emitted when it closes.

        Closing a row or the table first closes its open cell, and closing
        the table its open row, as a browser does for omitted end tags.
        The text of a nested table's cells stays space-separated.
        """
        if not self._tables:
            return
        if len(self._tables) != 1:
            self._separate()
            if tag == "table":
                self._tables.pop()
            return
        if self._row is not None and self._cell_start >= 0:
            cell = " ".join(self.text[self._cell_start:].split())
            self.text = self.text[: self._cell_start]
            self._last_code = None
            self._underscores = []
            self._marker_end = self._cell_marker_end
            self._row.append(cell.replace("|", "\\|"))
            self._cell_start = -1
        if tag in ("tr", "table") and self._row is not None:
            self._tables[0].append(self._row)
            self._row = None
        if tag == "table":
            self._emit_table(self._tables.pop())

    def _emit_table(self, rows: list[list[str]]) -> None:
        """Emit *rows* as a pipe table whose first row is the header."""
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
        if self.skipping or not data:
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
