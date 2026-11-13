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

A list in a table cell is flattened onto the cell's line. Completed
blocks are accumulated separately, so appending a paragraph does not
copy all preceding paragraphs.
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
# Where a line ends: an inline element does not reach past one.
_BLOCK_TAGS = {
    *_PARAGRAPH_TAGS, *_HEADING_TAGS, *_TABLE_TAGS, "blockquote", "pre", "ul", "ol", "li", "hr"
}
# What a heading, one line of inline content, cannot hold.  Nor can it hold a
# list item, which drops the markers before its own (``_start_list_item``), or a
# table, which is written where it ends (``_emit_table``).
_NO_HEADING_CONTENT_TAGS = {*_HEADING_TAGS, "blockquote", "pre", "hr"}

_WHITESPACE_RE = re.compile(r"\s+")
_BOUNDARY_UNDERSCORE_RE = re.compile(r"(?<![A-Za-z0-9])_|_(?![A-Za-z0-9])")
# What opens a tag or an autolink: ``<1@x.y>`` is one too, an email address.
# Group 1 is what makes it one, up to the character that completes it.
_TAG_START_RE = re.compile(r"<(?=([A-Za-z/!?]|[^\s<>]+@))")
_ENTITY_START_RE = re.compile(r"&(?=([A-Za-z][A-Za-z0-9]*;|#[0-9]+;|#[xX][0-9a-fA-F]+;))")
# Two tildes strike text through where that is supported; three open a code fence.
_TILDE_RUN_RE = re.compile(r"~(?=~)|(?<=~)~")
# Hyphens and spaces alone are a thematic break or, like equals signs, a
# setext underline.
_LINE_START_SYNTAX_RE = re.compile(r"^(?:#{1,6}(?:\s|$)|[-+]\s|>|=+ ?$|-[- ]*$)")
_LINE_START_NUMBER_RE = re.compile(r"^(\d+)([.)])(\s)")
# A list marker that is a whole text run: an empty item if its line ends
# there, an item if the element after it starts with a space.
_LONE_LIST_MARKER_RE = re.compile(r"\+|\d+[.)]")
# A line that is a number so far, and the text that makes a list marker of it:
# the two can be separate text runs (``1<span></span>. x``).
_NUMBER_LINE_RE = re.compile(r"(?: *(?:[-+]|\d+[.)]) )* *\d+")
# The line of a heading: its marker behind the indentation and the markers of the
# list items it starts (an ordered list counts from any integer).
_HEADING_LINE_RE = re.compile(r"(?: *(?:-|-?\d+\.) )* *#{1,6} ")
_MARKER_END_RE = re.compile(r"^(\d*)([.)](?:\s|$))")
_CLOSING_HASHES_RE = re.compile(r"(?<= )#+ *$")  # what ends an ATX heading: dropped when rendered
# A text run that may start the delimiter row of a table, below a line with a pipe.
_DELIMITER_ROW_RE = re.compile(r"[-:|][-:| ]*")
_BULLETS_RE = re.compile(r"[- ]*-[- ]*")  # a line that is the markers of bullet items so far
# The text of a link that ends a link label before a colon: its first bracket
# that no backslash escapes is a closing one.
_DEFINITION_LABEL_RE = re.compile(r"(?:\\.|[^\\\[\]])*\]:", re.S)
_ITEM_MARKERS_RE = re.compile(r"(?: *(?:-|\d+\.) )* *")  # what starts the line of a list item
_EMAIL_AUTOLINK_RE = re.compile(  # as CommonMark defines one
    r"<[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
    r"(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*>"
)
_LINK_DESTINATION_UNSAFE_RE = re.compile(r"[\s()<>]")
_URL_REMOVED_RE = re.compile(r"[\t\n\r]")  # what a URL parser removes from an address


def _escape_inline(text: str) -> str:
    """Backslash-escape characters that would otherwise be Markdown syntax."""
    for char in "\\`*[]":
        text = text.replace(char, "\\" + char)
    text = _BOUNDARY_UNDERSCORE_RE.sub(r"\\_", text)
    text = _TILDE_RUN_RE.sub(r"\\~", text)
    text = _ENTITY_START_RE.sub(r"\\&", text)
    return _TAG_START_RE.sub(r"\\<", text)


def _escape_line_start(text: str) -> str:
    """Escape a leading heading / list / quote marker so a plain line stays plain."""
    if _LINE_START_SYNTAX_RE.match(text):
        return "\\" + text
    return _LINE_START_NUMBER_RE.sub(r"\1\\\2\3", text, count=1)


def _line_above(text: str) -> str:
    """Return the line above the last line of *text*, ``""`` if there is none."""
    end = max(text.rfind("\n"), 0)
    return text[text.rfind("\n", 0, end) + 1 : end]


def _backtick_fence(text: str, minimum: int) -> str:
    """Return a run of backticks longer than any backtick run inside *text*."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    return "`" * max(longest + 1, minimum)


def _code_span(text: str, padded: bool = False) -> str:
    """Wrap *text* in a code span whose fence is longer than any inner backtick run.

    A space on each side, which a renderer strips, keeps the text off the fence
    where it must not touch it and, if *padded*, in any case.
    """
    fence = _backtick_fence(text, 1)
    if padded or len(fence) > 1 or text.startswith(" ") or text.endswith(" "):
        return f"{fence} {text} {fence}"
    return f"{fence}{text}{fence}"


def _link_destination(href: str) -> str:
    """Return *href* in a form that survives inside ``[label](...)``.

    Markdown reads backslash escapes and entities in a destination, bare
    or in angle brackets: both are escaped first.  A tab or line ending
    goes, as in the address a browser follows: neither form can hold a
    line ending.
    """
    href = _URL_REMOVED_RE.sub("", href).replace("\\", "\\\\")
    href = _ENTITY_START_RE.sub(r"\\&", href)
    if _LINK_DESTINATION_UNSAFE_RE.search(href):
        return "<" + href.replace("<", "\\<").replace(">", "\\>") + ">"
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


def _entity_at(text: str, index: int) -> str:
    """Return *text* with the character at *index* as a numeric character reference.

    To the delimiter parser the reference is punctuation, the boundary an
    emphasis marker next to the character may need, and it renders as the
    character itself.  An underscore next to it was literal inside a word
    and is escaped: it is at the edge of one now.
    """
    left, right = text[:index], text[index + 1 :]
    if left.endswith("_") and not left.endswith("\\_"):
        left = left[:-1] + "\\_"
    if right.startswith("_"):
        right = "\\" + right
    return f"{left}&#{ord(text[index])};{right}"


def _apart(before: str, text: str, marker: str = "") -> str:
    """Return *text* escaped so that it completes nothing that *before* began.

    Text runs are escaped one by one, and neither run of
    ``&lt;<span>a</span>&gt;``, ``&amp;am<span>p;</span>`` or
    ``&lt;1<span>@x.y</span>&gt;`` holds a tag, an entity or an autolink: the
    two together do.  The character that completes one is escaped, a letter
    as a character reference, and so is a tilde that would lengthen a run.
    *marker* stands for what an element that opens between the two writes
    there: it keeps a tag or an entity apart, an address may hold it.
    """
    if text.startswith("~") and before.endswith("~"):
        text = "\\" + text
    word = before[before.rfind(" ") + 1 :]
    word = word[word.rfind("\n") + 1 :] + marker
    for opener, pattern in (("<", _TAG_START_RE), ("&", _ENTITY_START_RE)):
        start = word.rfind(opener)
        match = pattern.match(word + text, start) if start >= 0 else None
        if match and match.end(1) > len(word):
            index = match.end(1) - len(word) - 1
            if text[index].isalpha():
                text = _entity_at(text, index)
            else:
                text = f"{text[:index]}\\{text[index:]}"
    return text


def _ends_autolink(before: str, span: str) -> bool:
    """Whether a ``<`` of *before* and the code *span* behind it are an email autolink.

    ``<`a@b.c>` `` is one to a renderer, for which it takes precedence over
    the span: an address may hold backticks before its ``@``.  It starts at
    the last ``<``, if no backslash escapes that, and ends in the span.
    """
    start = before.rfind("<")
    if start < 0 or (start - len(before[:start].rstrip("\\"))) % 2:
        return False
    address = _EMAIL_AUTOLINK_RE.match(before + span, start)
    return address is not None and address.end() > len(before)


def _closing_stars(text: str) -> int:
    """Return the length of the asterisk delimiter run *text* ends with.

    What is written ends in asterisks only where emphasis closes, or in a
    literal one, which follows an odd number of backslashes.
    """
    stripped = text.rstrip("*")
    escaped = (len(stripped) - len(stripped.rstrip("\\"))) % 2
    return len(text) - len(stripped) - escaped


def _bare_bang(text: str, end: int) -> bool:
    """Whether *text* has, right before *end*, an exclamation mark that no backslash escapes.

    Before a link it would make an image of it, and takes a backslash.  One that
    has a backslash already (``<\\!``, see :func:`_apart`) would lose it to a second.
    """
    start = end - 1
    while start > 0 and text[start - 1] == "\\":
        start -= 1
    return text[end - 1 : end] == "!" and (end - start) % 2 == 1


def list_start(attrs: list[tuple[str, str | None]]) -> int:
    """Return the number an ``<ol>`` with *attrs* counts from: its ``start``, else 1.

    ``ensure_html`` turns a Markdown list beginning at ``5.`` into
    ``<ol start="5">``; a renderer that counted every list from 1
    renumbered the steps a summary referred to.  A ``start`` that is
    no integer counts as absent, as browsers treat it.

    Args:
        attrs: The tag's attributes as :class:`~html.parser.HTMLParser`
            hands them to ``handle_starttag``.
    """
    try:
        return int(dict(attrs).get("start") or 1)
    except ValueError:
        return 1


class _ListState:
    """Rendering state of one open ``<ul>``/``<ol>``."""

    def __init__(self, ordered: bool, indent: str, start: int = 1) -> None:
        self.ordered = ordered
        self.counter = start
        self.indent = indent
        self.child_indent = indent
        self.items = 0  # its item markers in the text: the marker of an empty item is dropped


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
    """Stream HTML into a Markdown string held in :attr:`text`.

    Text is escaped as it is written, and what an inline element starts with
    is escaped for the marker that the element writes before it where it
    ends: at its end tag, with the element it is in, or with its block.  Only
    a link whose text is its address writes none after all: the number of its
    start tag is added to *bare_links*, for a second run that is told of it
    (:func:`html_to_markdown`).
    """

    def __init__(self, bare_links: set[int]) -> None:
        """Start a run that opens the links numbered in *bare_links* without their address."""
        super().__init__()
        self.text = ""
        self._bare_links = bare_links
        self._links = 0  # start tags of links so far: at most one link is open, the last
        self._chunks: list[str] = []
        self._bounded_closes: list[int] = []  # ends of closers that need a boundary next
        self._code_depth = 0
        self._pre: list[str] | None = None
        self._lists: list[_ListState] = []
        self._inline: list[tuple[str, int, str | None]] = []  # (tag, start, href), innermost last
        # Those that a block boundary found without content, to open where some starts.
        self._waiting: list[tuple[str, int, str | None]] = []
        self._last_code: tuple[int, int, str] | None = None  # (start, end, text) of last span
        self._underscores: list[tuple[int, int, int]] = []  # (open, close, width), undecided
        self._ambiguous: list[int] = []  # asterisk openers that could close as well
        self._protected = -1  # where a top-level emphasis ends whose opener is a run of its own
        self._quotes: list[int] = []
        self._tables: list[list[list[str]]] = []
        self._row: list[str] | None = None
        self._cell_start = -1
        # The markers that have no content yet, innermost last: where each starts
        # and the list it is an item of (None for a heading).
        self._markers: list[tuple[int, _ListState | None]] = []
        self._marker_end = -1  # the length of the text right after the last of them
        self._cell_markers = (self._markers, self._marker_end)  # as they were where the cell starts
        self._list_end = -1  # the length of the text right after a nested list

    def _indent(self) -> str:
        return self._lists[-1].child_indent if self._lists else ""

    def _at_marker(self) -> bool:
        """True right after a heading or list marker, before any content."""
        return len(self.text) == self._marker_end

    def _at_line_start(self) -> bool:
        """True where a line starts: behind a line break and, in a list item,
        the indentation that :meth:`_append` writes with the break."""
        return not self.text or self.text.endswith(("\n", "\n" + self._indent()))

    def _break_line(self) -> None:
        """Write the line break of a ``<br>`` where the Markdown can hold one.

        No line break follows a marker.  A blank line would end the inline
        element it is in: a second break in a row is dropped there.  A heading
        is one line, as is a code span, whose text is not escaped and in which
        a renderer makes a space of a line ending: a space stands for the break.
        """
        if self._at_marker() or (self._inline and self._at_line_start()):
            return
        if self._code_depth or self._on_heading_line():
            self._separate()
        else:
            self._append("\n")

    def _on_heading_line(self) -> bool:
        """True on the line of a heading's marker.

        Read from the text: a flag would have to follow every rewrite of it.
        Text never starts a line like a marker (``_escape_line_start``), and
        verbatim code text is in its span by now: no code element is open here.
        """
        return _HEADING_LINE_RE.match(self.text, self.text.rfind("\n") + 1) is not None

    def _keep_closing_hashes(self) -> None:
        """Escape the hashes that end a heading's line, where that line ends.

        A renderer drops them as the closing sequence of the heading, whatever
        ends the line: the heading's end tag, a block that starts in it, the
        end of the input.
        """
        hashes = _CLOSING_HASHES_RE.search(self.text, self.text.rfind("\n") + 1)
        if hashes and self._on_heading_line():
            self.text = f"{self.text[: hashes.start()]}\\{self.text[hashes.start() :]}"

    def _part_from_list(self) -> None:
        """Put a blank line before inline content that follows a nested list:
        right below the list it would continue the last item of that list."""
        if len(self.text) == self._list_end:
            self._newline(2)

    def _newline(self, count: int) -> None:
        """End the current line and ensure *count* newlines precede the next text."""
        if not self.text or self._at_marker():
            return
        self._keep_closing_hashes()
        stripped = self.text.rstrip(" ")
        trailing = len(stripped) - len(stripped.rstrip("\n"))
        self.text = stripped + "\n" * max(count - trailing, 0)
        if not (self._inline or self._quotes or self._tables):
            # Keep the trailing newlines in the active block: list indentation
            # and subsequent separators still need to see them.
            end = len(self.text.rstrip("\n"))
            if end:
                self._chunks.append(self.text[:end])
                self.text = self.text[end:]
                self._forget_positions()

    def _append(self, text: str) -> None:
        """Append *text*, indenting continuation lines inside list items."""
        indent = self._indent()
        if not self._code_depth:
            text = _apart(self.text, text, "*" if self._opens_at(len(self.text)) else "")
        if indent and text and self.text.endswith("\n"):
            text = indent + text
        if indent:
            text = text.replace("\n", "\n" + indent).replace(indent + "\n", "\n")
        self.text += text
        self._marker_end = -1
        self._list_end = -1
        self._settle_underscores()

    def _settle_underscores(self) -> None:
        """Resolve pending emphasis once the following character is known.

        A closer behind punctuation, and underscores that asterisks cannot
        replace, parse only before whitespace or punctuation: a letter that
        follows becomes a character reference, a boundary that displays as
        that letter. The other underscore runs turn back to asterisks when
        text continues their word.

        The character after the closing run decides (:func:`_is_boundary`)
        once it is final: an inline element that opens right there may still
        write its own markers first (``[``, a backtick, ``**``) or nothing at
        all (``<a>`` without an href), so the run waits until it has closed.
        """
        pending_closes = []
        for end in self._bounded_closes:
            if len(self.text) <= end or self._opens_at(end):
                pending_closes.append(end)
            elif not _is_boundary(self.text[end]):
                self.text = self.text[:end] + _entity_at(self.text[end:], 0)
        self._bounded_closes = pending_closes
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

    def _forget_positions(self) -> None:
        """Drop the positions remembered in text that is complete or has moved."""
        self._last_code = None
        self._underscores = []
        self._bounded_closes = []
        self._ambiguous = []
        self._protected = -1
        self._marker_end = self._list_end = -1

    def _emit_marker(self, marker: str, owner: _ListState | None = None) -> None:
        """Append a heading marker, or the marker of an item of the list *owner*.

        No line break follows a marker, and it is dropped if no content does.
        One written right behind another (an item that starts with a list or a
        heading) joins the markers that wait for content.
        """
        if not self._at_marker():
            self._markers = []
        self._markers.append((len(self.text), owner))
        self.text += marker
        self._marker_end = len(self.text)
        self._list_end = -1
        if owner:
            owner.items += 1

    def finish(self) -> None:
        """Flush constructs left open at the end of the input."""
        self._end_open_inline()
        if self._pre is not None:
            self._end_pre()
        while self._tables:
            self._end_table_tag("table")
        while self._at_marker():
            self.drop_empty_marker()
        self._keep_closing_hashes()
        self.text = "".join(self._chunks) + self.text
        self._chunks = []

    def drop_empty_marker(self) -> None:
        """Remove the last heading / list marker when its element turned out to be empty.

        The marker written before it, if any, has no content again.
        """
        if self._at_marker():
            start, owner = self._markers.pop()
            self.text = self.text[:start]
            if owner:
                owner.items -= 1
            self._marker_end = start if self._markers else -1

    def _end_empty_heading(self) -> None:
        """Drop the marker of a heading that has no content, where a block starts in it."""
        if self._at_marker() and self._markers[-1][1] is None:
            self.drop_empty_marker()

    def _wrap_since(self, pos: int, before: str, after: str) -> None:
        """Replace everything appended since *pos* with ``before + inner + after``.

        Surrounding whitespace is moved outside the markers so that
        ``<b> bold </b>`` becomes `` **bold** `` rather than the
        invalid ``** bold **``; an empty inner text drops the markers.

        An emphasis is written so that CommonMark pairs its two markers,
        which depends on the characters around each (a letter, punctuation,
        whitespace, another marker) and, for a run that can both open and
        close, on the lengths of the runs (the rule of 3):

        * It joins a run of the same emphasis that ends where it starts
          (``<b>a</b><b>b</b>``), so that no ``****`` is produced, unless code
          spans would meet there (their fences would fuse), the two texts
          would be a tag or an entity together (:func:`_apart`), or an emphasis
          it starts with would land where it cannot open, or could close.
        * Behind a run of the other emphasis its opener makes one run with
          that closer, ``**x***y*``, which parses between two letters or two
          punctuation characters, as a run of three.
        * Starting with the other emphasis (``<i><b>a</b> b</i>``) it opens
          with a run of three, ``***a** b*``.
        * What a run of three leaves behind its first closer pairs with an
          opener inside that can close too (``***a** b**c***``,
          ``**x***y**z***``): the rule of 3 guards such an opener only
          against a run of one or two.

        Where asterisks do not parse it takes underscores, which never share
        a run with them: ``_**a** b**c**_``, ``**x**_y**z**_``, ``**a:**_b_``.
        They need whitespace or punctuation on both sides: a letter before or
        after them becomes a character reference (:func:`_entity_at`), unless
        asterisks parse as well, which the text that follows decides
        (:meth:`_settle_underscores`).  An opener between a letter and
        punctuation (``a<b>(b)</b>``) gets the same reference before it.

        An element still open at *pos* writes its marker before this one: an
        emphasis decides all of this itself, a link is a ``[`` to it.
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
        # An element still open at pos is what will follow a closer that ends there:
        # the boundary is its to decide, it may merge that closer away.
        enclosing = self._opens_at(pos)
        self._bounded_closes = [
            end for end in self._bounded_closes if end < pos or (enclosing and end == pos)
        ]
        # An asterisk opener inside that could close too: it would pair with what
        # is left of a longer run of asterisks that opens this element.
        fragile = any(opener > pos for opener in self._ambiguous)
        self._ambiguous = [opener for opener in self._ambiguous if opener < pos]
        self._last_code = None  # a marker follows the last span: no span can join it
        emphasis = before == after
        bounded = False  # whether the closer parses only before whitespace or punctuation
        owned = False  # whether the opener is a run of its own: the rule of 3 guards it
        # An emphasis that opens here writes its marker before this one, and decides.
        if emphasis and enclosing in ("", "a"):
            first = core.lstrip("*_")[:1]  # none in the text of an unclosed <code>*
            nested = core.startswith("*")  # the other emphasis opens here too: a run of three
            stars = 0 if lead or enclosing else _closing_stars(head)
            behind = head[-stars - 1] if stars else ""  # what the closer before follows
            # An opener of the other emphasis that a merge puts behind that character:
            # it must open there, and close nothing unless the rule of 3 guards it.
            opens = not nested or _is_boundary(behind) or not _is_boundary(first)
            plain = not fragile and (
                not nested or (_is_boundary(behind) and not _is_boundary(first))
            )
            if (
                stars == len(after)
                and not (behind == "`" and core.startswith("`"))
                and (plain or (opens and self._protected == pos))
                and _apart(head[:-stars], core) == core
            ):
                owned = self._protected == pos
                head = head[:-stars]
                if _bare_bang(head, len(head)) and core.startswith("["):  # "![": an image
                    head = head[:-1] + "\\!"
                before = ""
            elif stars:
                # The closer before and this opener are one run, which does both
                # only between two letters or two punctuation characters, as a
                # run of three, and leaves an opener that the rule of 3 does not guard.
                if (
                    stars + len(after) != 3 or nested or fragile
                    or _is_boundary(behind) != _is_boundary(first)
                ):
                    before = after = "_" * len(after)
                    bounded = True
            else:
                if (
                    not lead and not enclosing and head and not _is_boundary(head[-1])
                    and (_is_boundary(first) or (fragile and nested))
                ):
                    head = self._bound(head)
                previous = (("[" if enclosing else head) + lead)[-1:]
                ambiguous = bool(previous.strip()) and _is_boundary(previous) == _is_boundary(first)
                if nested and _is_boundary(previous) and previous != "_":
                    before = after = "_" * len(after)
                    bounded = fragile
                    if not trail and not fragile:
                        opening = len(head) + len(lead)
                        self._underscores.append(
                            (opening, opening + len(before) + len(core), len(before))
                        )
                elif ambiguous and self._exposed():
                    if _is_boundary(previous):
                        before = after = "_" * len(after)
                        bounded = True
                    else:
                        head = self._bound(head)
                else:
                    if ambiguous and self._inline:  # for the element around it to see
                        self._ambiguous.append(len(head) + len(lead))
                    owned = not nested and not any(
                        entry[0] in _INLINE_MARKERS for entry in self._inline
                    )
        end = len(head) + len(lead) + len(before) + len(core) + len(after)
        if owned:
            self._protected = end
        # Behind punctuation, an underscore closer inside included, the closer needs one next.
        if emphasis and not trail and (bounded or _is_boundary(core.rstrip("*")[-1:])):
            self._bounded_closes.append(end)
        self.text = f"{head}{lead}{before}{core}{after}{trail}"

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Open block/inline constructs and emit list markers."""
        if self.skipped(tag, False):
            return
        if tag == "img":
            self.handle_data(dict(attrs).get("alt") or "")  # its text, like any other
            return
        if self._pre is not None:
            if tag == "br":
                self._pre.append("\n")
            return
        if tag in _BLOCK_TAGS:
            self._end_open_inline()
            if tag in _NO_HEADING_CONTENT_TAGS:
                self._end_empty_heading()
        elif tag in _INLINE_MARKERS or tag in ("a", "code"):
            self._part_from_list()
        if tag == "br":
            self._break_line()
        elif tag == "hr":
            self._newline(2)
            # Behind the markers of bullet items alone, dashes would make a rule of the whole line.
            in_bullets = _BULLETS_RE.fullmatch(self.text, self.text.rfind("\n") + 1)
            self._append("***" if in_bullets else "---")
            self._newline(2)
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
            self._open(tag)
        elif tag in ("ul", "ol"):
            self._newline(1 if self._lists else 2)
            self._lists.append(_ListState(tag == "ol", self._indent(), list_start(attrs)))
        elif tag == "li":
            self._start_list_item()
        elif tag in _TABLE_TAGS:
            self._start_table_tag(tag)
        elif tag == "a":
            href = dict(attrs).get("href")
            if href:
                self._end_link()  # a link holds no link: as in a browser, this one ends it
                self._links += 1
                if self._links in self._bare_links:
                    href = None
            self._open(tag, href)
        elif tag in _INLINE_MARKERS:
            if tag == "summary":  # a block of its own
                self._end_open_inline()
                self._newline(2)
            self._open(tag)

    def _open(self, tag: str, href: str | None = None) -> None:
        """Open an inline element, inside those that waited for content."""
        self._waiting.append((tag, 0, href))
        self._resume()

    def _resume(self) -> None:
        """Open the inline elements that waited for content, where it starts."""
        self._inline += [(tag, len(self.text), href) for tag, _, href in self._waiting]
        self._code_depth += sum(1 for entry in self._waiting if entry[0] == "code")
        self._waiting = []

    def _start_list_item(self) -> None:
        """Emit the ``- `` / ``N. `` marker for a new list item."""
        if not self._lists:
            self._lists.append(_ListState(False, ""))
        state = self._lists[-1]
        marker = f"{state.counter}. " if state.ordered else "- "
        if state.ordered:
            state.counter += 1
        state.child_indent = state.indent + " " * len(marker)
        # The marker of the empty item before this one goes, like an empty heading's.
        # The marker of the outer item that this list starts stays: this one follows it.
        while self._at_marker() and self._markers[-1][1] in (state, None):
            self.drop_empty_marker()
        indent = "" if self._at_marker() else state.indent
        # Below other text an ordered list starts at 1 only: any other first
        # number takes a blank line.
        self._newline(1 if state.items or marker in ("- ", "1. ") else 2)
        self._emit_marker(indent + marker, state)

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
                self._cell_markers = (list(self._markers), self._marker_end)

    def handle_endtag(self, tag: str) -> None:
        """Close block/inline constructs."""
        if self.skipped(tag, True):
            return
        if self._pre is not None:
            if tag == "pre":
                self._end_pre()
            return
        if tag in _BLOCK_TAGS:
            self._end_open_inline()
        if tag in _HEADING_TAGS:
            self.drop_empty_marker()
            self._newline(2)
        elif tag in _PARAGRAPH_TAGS:
            self._newline(2)
        elif tag == "blockquote":
            self._end_blockquote()
        elif tag in ("ul", "ol"):
            closed = self._lists.pop() if self._lists else None
            # Its last item was empty, as was a heading left open in it.
            while self._at_marker() and self._markers[-1][1] in (closed, None):
                self.drop_empty_marker()
            self._newline(1 if self._lists else 2)
            self._list_end = len(self.text)
        elif tag == "li":
            self._newline(1)
        elif tag in _TABLE_TAGS:
            self._end_table_tag(tag)
        elif tag in _INLINE_MARKERS or tag in ("a", "code"):
            if self._end_inline(tag) and tag == "summary":  # its block ends with it
                self._end_open_inline()
                self._newline(2)

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

    def _end_open_inline(self) -> None:
        """End the inline elements that a block boundary or the end of the input finds open.

        Markdown has no emphasis, link or code span that reaches past its block,
        and an element left open would hold all that follows: a summary cut
        inside it leaves one, as does a ``<!`` in code text, which takes the end
        tag into a comment.  An element with no content yet waits for some
        instead: it may hold the block that starts here (``<a href="/u"><h3>``).
        """
        started = sum(1 for _, pos, _ in self._inline if self.text[pos:].strip())
        waiting = self._inline[started:]  # an element in one without content has none either
        del self._inline[started:]
        self._code_depth -= sum(1 for entry in waiting if entry[0] == "code")
        self._waiting += waiting
        while self._inline:
            self._close()

    def _end_inline(self, tag: str) -> bool:
        """Close the innermost open *tag* and the elements still open in it.

        A browser reopens those behind the end tag (``<b>x<i>y</b>z``: an italic
        ``z``); here they end with it, as Markdown markers nest.  One that waits
        for content ends here without any, like one that is empty.

        Returns:
            Whether one was open.
        """
        open_tags = [element[0] for element in self._inline + self._waiting]
        if tag not in open_tags:
            return False
        self._close_from(len(open_tags) - 1 - open_tags[::-1].index(tag))
        return True

    def _end_link(self) -> None:
        """Close the open link, if there is one, and the elements still open in it."""
        for index, (tag, _, href) in enumerate(self._inline + self._waiting):
            if tag == "a" and href:
                self._close_from(index)
                return

    def _close_from(self, index: int) -> None:
        """Close the open inline elements from the one at *index* on, innermost first."""
        self._resume()
        while len(self._inline) > index:
            self._close()

    def _close(self) -> None:
        """Close the innermost open inline element: write its markers around its content."""
        redundant = self._redundant(len(self._inline) - 1)
        tag, pos, href = self._inline.pop()
        self._code_depth -= tag == "code"
        if self._code_depth:
            return  # emphasis, links and inner spans have no meaning inside code
        if tag == "code":
            self._end_code(pos)
        elif tag == "a":
            inner = self.text[pos:].strip()
            if href and not inner:
                self.text = self.text[:pos]
                self.handle_data(href)  # its text, like any other
            elif href and inner != href:
                if (
                    _bare_bang(self.text, pos)
                    and not self.text[pos].isspace()
                    and not self._opens_at(pos)
                ):  # "![" would start an image
                    self.text = f"{self.text[: pos - 1]}\\{self.text[pos - 1 :]}"
                    pos += 1
                opener = "&#32;[" if self._defines(pos, inner) else "["
                self._wrap_since(pos, opener, f"]({_link_destination(href)})")
            elif href:
                self._bare_links.add(self._links)  # written as its address alone
        elif not redundant:
            self._wrap_since(pos, _INLINE_MARKERS[tag], _INLINE_MARKERS[tag])
        self._settle_underscores()

    def _defines(self, pos: int, label: str) -> bool:
        """Whether a link written at *pos* with the text *label* would define a link reference.

        A paragraph that starts with ``[label]:`` defines a reference and shows
        nothing, and a label ends at the first ``]`` that no backslash escapes:
        inside a code span, whose text cannot be escaped.  The reference of a
        space before such a link makes text of its line; where a paragraph
        starts, a renderer does not show that space.

        A paragraph starts behind the markers of its list items, or behind
        their indentation below a blank line, and not in a table cell.  An
        element still open at *pos* writes its own marker before the link.
        """
        line = self.text.rfind("\n", 0, pos) + 1
        markers = self.text[line:pos]
        return (
            _DEFINITION_LABEL_RE.match(label) is not None
            and self._cell_start < 0
            and not self._opens_at(pos)
            and _ITEM_MARKERS_RE.fullmatch(markers) is not None
            and (bool(markers.strip()) or line == 0 or self.text.endswith("\n\n", 0, line))
        )

    def _redundant(self, index: int) -> bool:
        """Whether an entry below *index* on the inline stack already supplies its emphasis.

        Such an entry writes no markers of its own: ``<b>a<b>b</b>c</b>`` is
        ``**abc**``, not ``**a**b**c**``, which would un-bold ``b``.
        """
        marker = _INLINE_MARKERS.get(self._inline[index][0])
        return marker is not None and any(
            _INLINE_MARKERS.get(entry[0]) == marker for entry in self._inline[:index]
        )

    def _bound(self, head: str) -> str:
        """Return *head* with the letter it ends in as a character reference.

        What precedes a letter that starts an open element is no part of its
        word: the marker of that element will come between them.
        """
        word = len(head) - 1 if self._opens_at(len(head) - 1) else 0
        return head[:word] + _entity_at(head[word:], len(head) - 1 - word)

    def _exposed(self) -> bool:
        """Whether the emphasis around the one that closes must open with three asterisks.

        It starts with the other emphasis, right behind an underscore, where
        underscores of its own would not parse.  What a first closer leaves of
        such a run pairs with any opener inside that could close as well.
        """
        for index in range(len(self._inline) - 1, -1, -1):
            tag, start, href = self._inline[index]
            if tag == "a" and href:
                return False  # the text of a link is parsed on its own
            if tag in _INLINE_MARKERS and not self._redundant(index):
                return self.text.endswith("_", 0, start) and self.text.startswith("*", start)
        return False

    def _opens_at(self, pos: int) -> str:
        """Return the tag of the innermost open inline element that starts at *pos*
        and writes markers of its own, ``""`` if there is none.

        Adjacent code spans or emphasis runs do not merge across it: the merge
        would move its start inside the merged text (``<b>y</b><a><b>z</b></a>``).
        An anchor without an href writes nothing, and a redundant emphasis
        writes nothing either (:meth:`_redundant`).
        """
        for index in range(len(self._inline) - 1, -1, -1):
            tag, start, href = self._inline[index]
            if start == pos and (tag != "a" or href) and not self._redundant(index):
                return tag
        return ""

    def _end_code(self, pos: int) -> None:
        """Replace the text appended since *pos* with a code span.

        Surrounding whitespace stays outside the span, as in :meth:`_wrap_since`,
        and an all-whitespace element keeps separating the words around it.  A
        span that directly follows the previous one (``<code>a</code><code>b</code>``)
        is merged into it, since ``` `a``b` ``` would render as one span ``a``b``.

        Code text cannot be escaped: a span that would end an autolink is padded
        (:func:`_ends_autolink`), a space being no part of an address.
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
        span = _code_span(core, _ends_autolink(self.text[:pos] + lead, _code_span(core)))
        self.text = f"{self.text[:pos]}{lead}{span}{trail}"
        self._last_code = (pos + len(lead), pos + len(lead) + len(span), core)

    def _end_blockquote(self) -> None:
        """Prefix everything since the matching ``<blockquote>`` with ``> ``."""
        if not self._quotes:
            return
        self._keep_closing_hashes()  # the line ends here, before it is prefixed
        pos = self._quotes.pop()
        indent = self._indent()
        lines = self.text[pos:].strip("\n").split("\n")
        quoted = [f"{indent}> {line.removeprefix(indent)}".rstrip() for line in lines]
        if pos and not self.text[:pos].endswith("\n"):
            quoted[0] = quoted[0].removeprefix(indent)  # continues a list marker line
        self.text = self.text[:pos] + "\n".join(quoted)
        self._forget_positions()  # a newline follows every pending run (those inside moved)
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
            self._forget_positions()
            self._markers, self._marker_end = self._cell_markers
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
        self._end_empty_heading()
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
        self._part_from_list()
        self._resume()
        if not self._code_depth:
            collapsed = _escape_inline(collapsed)
            if self._at_line_start() or self._at_marker():
                collapsed = _escape_line_start(collapsed)
                if _DELIMITER_ROW_RE.fullmatch(collapsed) and "|" in _line_above(self.text):
                    collapsed = "\\" + collapsed  # no table of the two lines
                # Not in a table cell, which holds no blocks, nor where an inline
                # element opens, whose own marker then comes first on the line.
                if (
                    _LONE_LIST_MARKER_RE.fullmatch(collapsed)
                    and self._cell_start < 0
                    and not self._opens_at(len(self.text))
                ):
                    collapsed = f"{collapsed[:-1]}\\{collapsed[-1]}"
            elif (
                self.text[-1:].isdigit()
                and self._cell_start < 0
                and not self._opens_at(len(self.text))
                and _NUMBER_LINE_RE.fullmatch(self.text, self.text.rfind("\n") + 1)
            ):  # a number so far: this text may end the list marker it starts
                collapsed = _MARKER_END_RE.sub(r"\1\\\2", collapsed)
        self._append(collapsed)


def html_to_markdown(html: str) -> str:
    """Convert an HTML fragment to Markdown.

    Headings become ``#`` headings, ``<p>`` blocks become paragraphs,
    ``<ul>``/``<ol>`` become (nested) ``-``/``1.`` lists, ``<pre>``
    becomes a fenced code block, ``<code>`` a code span, ``<b>``/``<i>``
    emphasis, ``<a>`` a ``[text](href)`` link, ``<table>`` a pipe table
    and ``<blockquote>`` a ``>`` quote.  Text that would accidentally be
    parsed as Markdown syntax is backslash-escaped.

    A link whose text is its address is written as that address alone,
    which is known where the link ends, behind text that was escaped for
    its bracket: the input is converted again, with such a link opened
    without its address, until a run finds no further one.

    Args:
        html: The HTML string to convert.

    Returns:
        The Markdown text, stripped of leading/trailing blank lines and
        trailing whitespace on each line.
    """
    bare_links: set[int] = set()
    while True:
        known = len(bare_links)
        parser = _HtmlToMarkdownParser(bare_links)
        parser.feed(html)
        parser.close()
        parser.finish()
        if len(bare_links) == known:
            break
    lines = [line.rstrip() for line in parser.text.split("\n")]
    return "\n".join(lines).strip("\n")
