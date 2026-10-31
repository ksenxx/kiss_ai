# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Render HTML summaries as styled terminal text.

The ``finish`` tool guarantees the result summary is HTML
(:func:`kiss.core.utils.ensure_html`), so console output needs an
HTML renderer.  This module converts an HTML fragment to a
:class:`rich.text.Text` using the stdlib :class:`html.parser.HTMLParser`:
headings/bold render bold, emphasis renders italic, code renders in a
highlighted style, list items get bullets, and links show their targets.
"""

from __future__ import annotations

import re

from rich.text import Text

from kiss.core.html_to_markdown import SkippingHTMLParser

_BLOCK_TAGS = {
    "p",
    "div",
    "section",
    "article",
    "blockquote",
    "ul",
    "ol",
    "li",
    "table",
    "tr",
    "pre",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "details",
    "summary",
}

_STYLE_TAGS = {
    "b": "bold",
    "strong": "bold",
    "i": "italic",
    "em": "italic",
    "u": "underline",
    "h1": "bold underline",
    "h2": "bold underline",
    "h3": "bold",
    "h4": "bold",
    "h5": "bold",
    "h6": "bold",
    "code": "bold cyan",
    "pre": "cyan",
    "a": "underline blue",
    "th": "bold",
    "summary": "bold",
}


class _HtmlToRichParser(SkippingHTMLParser):
    """Stream HTML into a styled :class:`rich.text.Text`."""

    def __init__(self) -> None:
        super().__init__()
        self.text = Text()
        self._style_stack: list[str] = []
        self._list_stack: list[int | None] = []
        self._in_pre = 0
        self._pre_end = 0
        self._cell_start = 0

    def _crop_spaces(self, floor: int) -> None:
        """Remove the trailing spaces that were appended after text offset *floor*.

        Spaces of ``<pre>`` text are content: nothing up to the end of
        the last preformatted text (``_pre_end``) is removed.
        """
        plain = self.text.plain
        excess = len(plain) - max(len(plain.rstrip(" ")), floor, self._pre_end)
        if excess > 0:
            self.text.right_crop(excess)

    def _newline(self, count: int = 1) -> None:
        """End the line without its trailing spaces and append newlines,
        collapsing runs beyond *count* blank lines."""
        self._crop_spaces(0)
        plain = self.text.plain
        if not plain:
            return
        trailing = len(plain) - len(plain.rstrip("\n"))
        for _ in range(count - trailing):
            self.text.append("\n")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Open styles (composed with enclosing styles), start blocks, and
        emit list bullets / numbering."""
        if self.skipped(tag, False):
            return
        if tag == "br":
            self._crop_spaces(0)
            self.text.append("\n")
            return
        if tag == "hr":
            self._newline()
            self.text.append("─" * 40, style="dim")
            self.text.append("\n")
            return
        if tag in _BLOCK_TAGS:
            self._newline(2 if tag == "p" or tag.startswith("h") else 1)
        if tag == "pre":
            self._in_pre += 1
        if tag in ("td", "th"):
            self._cell_start = len(self.text)
        if tag == "ul":
            self._list_stack.append(None)
        elif tag == "ol":
            self._list_stack.append(1)
        elif tag == "li":
            indent = "  " * max(len(self._list_stack) - 1, 0)
            marker = self._list_stack[-1] if self._list_stack else None
            if marker is None:
                self.text.append(f"{indent}• ")
            else:
                self.text.append(f"{indent}{marker}. ")
                self._list_stack[-1] = marker + 1
        style = _STYLE_TAGS.get(tag)
        if style is not None:
            parent = self._style_stack[-1] if self._style_stack else ""
            merged = f"{parent} {style}".split()
            self._style_stack.append(" ".join(dict.fromkeys(merged)))
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value:
                    self.text.append(f"[{value}] ", style="dim")
                    break

    def handle_endtag(self, tag: str) -> None:
        """Close styles and terminate blocks with newlines."""
        if self.skipped(tag, True):
            return
        if tag in _STYLE_TAGS and self._style_stack:
            self._style_stack.pop()
        if tag == "pre":
            self._in_pre = max(self._in_pre - 1, 0)
        if tag in ("ul", "ol") and self._list_stack:
            self._list_stack.pop()
        if tag in _BLOCK_TAGS:
            self._newline(2 if tag == "p" or tag.startswith("h") else 1)
        if tag in ("td", "th"):
            self._crop_spaces(self._cell_start)
            self.text.append("  ")

    def handle_data(self, data: str) -> None:
        """Append text content, collapsing whitespace outside ``<pre>``.

        Outside ``<pre>`` each whitespace run becomes one space, dropped at
        the start of a line or after a space, so text next to an inline tag
        keeps exactly the spacing the HTML gives it.
        """
        if self.skipping:
            return
        if not self._in_pre:
            data = re.sub(r"\s+", " ", data)
            plain = self.text.plain
            if not plain or plain.endswith(("\n", " ")):
                data = data.lstrip(" ")
            if not data:
                return
        style = self._style_stack[-1] if self._style_stack else None
        self.text.append(data, style=style)
        if self._in_pre:
            self._pre_end = len(self.text)


def html_to_rich(html: str) -> Text:
    """Convert an HTML fragment to a styled :class:`rich.text.Text`.

    Args:
        html: The HTML string to render (plain text is returned as-is).

    Returns:
        A :class:`rich.text.Text` with the rendered, styled content.
    """
    parser = _HtmlToRichParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # pragma: no cover — stdlib parser is very tolerant
        return Text(html)
    rendered = parser.text
    rendered.rstrip()
    return rendered
