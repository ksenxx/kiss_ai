# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Render HTML summaries as styled terminal text for the CLI.

The ``finish`` tool guarantees the result summary is HTML
(:func:`kiss.core.utils.ensure_html`), so terminal interfaces need an
HTML renderer.  This module converts an HTML fragment to a
:class:`rich.text.Text` using the stdlib :class:`html.parser.HTMLParser`:
headings/bold render bold, emphasis renders italic, code renders in a
highlighted style, list items get bullets, and links show their targets.
"""

from __future__ import annotations

from html.parser import HTMLParser

from rich.text import Text

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

_SKIP_TAGS = {"script", "style", "head", "title", "meta", "link"}


class _HtmlToRichParser(HTMLParser):
    """Stream HTML into a styled :class:`rich.text.Text`."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text = Text()
        self._style_stack: list[str] = []
        self._list_stack: list[int | None] = []
        self._skip_depth = 0
        self._in_pre = 0

    def _newline(self, count: int = 1) -> None:
        """Append newlines, collapsing runs beyond *count* blank lines."""
        plain = self.text.plain
        if not plain:
            return
        trailing = len(plain) - len(plain.rstrip("\n"))
        for _ in range(count - trailing):
            self.text.append("\n")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Open styles (composed with enclosing styles), start blocks, and
        emit list bullets / numbering."""
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "br":
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
        if tag in _SKIP_TAGS:
            self._skip_depth = max(self._skip_depth - 1, 0)
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
            self.text.append("  ")

    def handle_data(self, data: str) -> None:
        """Append text content, collapsing whitespace outside ``<pre>``."""
        if self._skip_depth:
            return
        if not self._in_pre:
            if not data.strip():
                return
            data = " ".join(data.split())
            plain = self.text.plain
            if plain and not plain.endswith(("\n", " ", "• ", ". ", "] ")):
                data = " " + data
        style = self._style_stack[-1] if self._style_stack else None
        self.text.append(data, style=style)


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


def html_to_text(html: str) -> str:
    """Convert an HTML fragment to plain terminal text (no styling).

    Args:
        html: The HTML string to render.

    Returns:
        The plain-text rendering of *html*.
    """
    return html_to_rich(html).plain
